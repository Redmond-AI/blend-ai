"""High-level, spatially aware relighting tools.

The Blender add-on owns scene inspection and mutation.  This module validates
MCP inputs, forwards bounded command envelopes, and converts a live Cycles
viewport capture into native MCP image content.
"""

from __future__ import annotations

import atexit
import asyncio
import base64
import binascii
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.server.fastmcp.utilities.types import Image as MCPImage
from mcp.types import CallToolResult, TextContent
from PIL import Image as PILImage
from PIL import UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from blend_ai.server import get_connection, mcp
from blend_ai.validators import (
    ValidationError,
    validate_color,
    validate_enum,
    validate_numeric_range,
    validate_object_name,
    validate_vector,
)

logger = logging.getLogger(__name__)

BUILT_IN_SEMANTIC_TERMS = [
    "skylight",
    "roof",
    "window",
    "glass",
    "opening",
    "ceiling",
    "wall",
    "floor",
    "column",
    "beam",
    "fixture",
    "aisle",
]

MAX_INSTANCES = 5_000
MAX_SURFACE_TRIANGLES = 1_000_000
MAX_RAYS = 512
MAX_RAY_HITS = 32
MAX_PATTERN_COUNT = 32
MAX_LIGHTS = 128
MAX_IMAGE_PIXELS = 50_000_000
MAX_INPUT_IMAGE_BYTES = 32 * 1024 * 1024
MAX_OUTPUT_IMAGE_BYTES = 16 * 1024 * 1024

_MACOS_CAPTURE_HELPER_LOCK = threading.Lock()
_MACOS_CAPTURE_HELPER: Path | None = None
_MACOS_CAPTURE_SOURCE = (
    Path(__file__).resolve().parents[1] / "resources" / "macos_capture_window.swift"
)

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")

Vec3 = tuple[float, float, float]
RaycastMode = Literal["CAMERA", "SCENE", "COLLECTIONS"]
LightingDetail = Literal["BOUNDS", "CANDIDATES"]
CacheMode = Literal["USE", "REFRESH"]
LightType = Literal["POINT", "SUN", "SPOT", "AREA"]


def _validate_identifier(value: str, *, name: str, max_length: int = 128) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    if len(value) > max_length:
        raise ValueError(f"{name} must be at most {max_length} characters")
    if not _SAFE_ID_RE.fullmatch(value):
        raise ValueError(
            f"{name} may contain only letters, numbers, underscores, hyphens, dots, and colons"
        )
    return value


def _finite_vec3(value: Any, *, name: str) -> Vec3:
    try:
        validated = validate_vector(value, size=3, name=name)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc
    return tuple(float(component) for component in validated)  # type: ignore[return-value]


def _finite_number(
    value: Any,
    *,
    name: str,
    minimum: float | int | None = None,
    maximum: float | int | None = None,
) -> float:
    try:
        validate_numeric_range(value, min_val=minimum, max_val=maximum, name=name)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc
    return float(value)


def _validate_safe_strings(
    values: list[str],
    *,
    name: str,
    max_count: int,
    max_length: int = 128,
) -> list[str]:
    if not isinstance(values, list):
        raise ValidationError(f"{name} must be a list")
    if len(values) > max_count:
        raise ValidationError(f"{name} may contain at most {max_count} entries")
    normalized: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise ValidationError(f"{name}[{index}] must be a non-empty string")
        value = value.strip()
        if len(value) > max_length:
            raise ValidationError(f"{name}[{index}] must be at most {max_length} characters")
        if any(ord(character) < 32 for character in value):
            raise ValidationError(f"{name}[{index}] contains control characters")
        normalized.append(value)
    return normalized


def _send_relighting_command(command: str, params: dict[str, Any]) -> Any:
    """Send one allowlisted add-on command and normalize error handling."""
    response = get_connection().send_command(command, params)
    if not isinstance(response, dict):
        raise RuntimeError(f"Blender returned an invalid response for '{command}'")
    if response.get("status") == "error":
        raise RuntimeError(f"Blender error during '{command}': {response.get('result')}")
    if response.get("status") != "ok":
        raise RuntimeError(
            f"Blender returned unexpected status {response.get('status')!r} for '{command}'"
        )
    return response.get("result")


async def _send_relighting_command_async(
    command: str,
    params: dict[str, Any],
    *,
    on_cancel_result: Any | None = None,
) -> Any:
    return await _to_thread_cancellation_safe(
        _send_relighting_command,
        command,
        params,
        on_cancel_result=on_cancel_result,
    )


async def _to_thread_cancellation_safe(
    function: Any,
    *args: Any,
    on_cancel_result: Any | None = None,
) -> Any:
    """Finish a started blocking phase before propagating cancellation.

    Cancelling ``asyncio.to_thread`` only abandons the await; it cannot stop the
    worker thread.  Socket and ScreenCaptureKit phases mutate or depend on
    Blender UI state, so cleanup must wait until the in-flight operation has
    actually finished.  A prepare callback can retain the returned session ID
    for the outer restoration ``finally`` block.
    """
    worker = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        try:
            result = await asyncio.shield(worker)
        except Exception:
            logger.exception("Blocking relighting phase failed while cancellation was pending")
        else:
            if on_cancel_result is not None:
                try:
                    on_cancel_result(result)
                except Exception:
                    logger.exception("Could not retain cancelled relighting phase result")
        raise


class RaySpec(BaseModel):
    """A bounded scene ray expressed as an endpoint or direction and distance."""

    model_config = ConfigDict(extra="forbid")

    id: str
    origin: Vec3
    target: Vec3 | None = None
    direction: Vec3 | None = None
    max_distance: float | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _validate_identifier(value, name="ray id")

    @field_validator("origin", "target", "direction", mode="before")
    @classmethod
    def validate_vec3(cls, value: Any, info: Any) -> Any:
        if value is None:
            return None
        return _finite_vec3(value, name=info.field_name)

    @field_validator("max_distance", mode="before")
    @classmethod
    def validate_max_distance(cls, value: Any) -> Any:
        if value is None:
            return None
        return _finite_number(
            value,
            name="max_distance",
            minimum=1e-6,
            maximum=1_000_000_000.0,
        )

    @model_validator(mode="after")
    def validate_ray_form(self) -> RaySpec:
        has_target = self.target is not None
        has_direction = self.direction is not None
        if has_target == has_direction:
            raise ValueError("provide exactly one of target or direction")

        if has_target:
            if self.max_distance is not None:
                raise ValueError("max_distance is only valid with direction")
            assert self.target is not None
            if math.dist(self.origin, self.target) <= 1e-9:
                raise ValueError("target must differ from origin")
        else:
            if self.max_distance is None:
                raise ValueError("direction rays require max_distance")
            assert self.direction is not None
            if math.sqrt(sum(component * component for component in self.direction)) <= 1e-12:
                raise ValueError("direction must be non-zero")
        return self


class LightSpec(BaseModel):
    """One managed light in an atomic relighting plan."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str | None = None
    type: LightType
    location_world: Vec3
    target_point: Vec3 | None = None
    target_object: str | None = None
    energy: float = 1000.0
    color_rgb: Vec3 = (1.0, 1.0, 1.0)
    use_shadow: bool = True
    radius: float | None = None
    sun_angle_degrees: float | None = None
    area_shape: Literal["SQUARE", "RECTANGLE", "DISK", "ELLIPSE"] | None = None
    size: float | None = None
    size_y: float | None = None
    spot_angle_degrees: float | None = None
    spot_blend: float | None = None
    diffuse_factor: float = 1.0
    specular_factor: float = 1.0
    volume_factor: float = 1.0

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _validate_identifier(value, name="light id")

    @field_validator("name", "target_object")
    @classmethod
    def validate_optional_object_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            return validate_object_name(value)
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc

    @field_validator("location_world", "target_point", mode="before")
    @classmethod
    def validate_position(cls, value: Any, info: Any) -> Any:
        if value is None:
            return None
        return _finite_vec3(value, name=info.field_name)

    @field_validator("color_rgb", mode="before")
    @classmethod
    def validate_rgb(cls, value: Any) -> Vec3:
        try:
            color = validate_color(value)
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc
        if len(color) != 3:
            raise ValueError("color_rgb must contain exactly three components")
        return tuple(float(component) for component in color)  # type: ignore[return-value]

    @field_validator("energy", mode="before")
    @classmethod
    def validate_energy(cls, value: Any) -> float:
        return _finite_number(value, name="energy", minimum=0.0, maximum=1_000_000_000.0)

    @field_validator("use_shadow", mode="before")
    @classmethod
    def validate_use_shadow(cls, value: Any) -> bool:
        if not isinstance(value, bool):
            raise ValueError("use_shadow must be a boolean")
        return value

    @field_validator("radius", mode="before")
    @classmethod
    def validate_radius(cls, value: Any) -> Any:
        if value is None:
            return None
        return _finite_number(value, name="radius", minimum=0.0, maximum=1_000_000.0)

    @field_validator("sun_angle_degrees", mode="before")
    @classmethod
    def validate_sun_angle(cls, value: Any) -> Any:
        if value is None:
            return None
        return _finite_number(
            value,
            name="sun_angle_degrees",
            minimum=0.0,
            maximum=180.0,
        )

    @field_validator("size", "size_y", mode="before")
    @classmethod
    def validate_size(cls, value: Any, info: Any) -> Any:
        if value is None:
            return None
        return _finite_number(
            value,
            name=info.field_name,
            minimum=1e-6,
            maximum=1_000_000.0,
        )

    @field_validator("spot_angle_degrees", mode="before")
    @classmethod
    def validate_spot_angle(cls, value: Any) -> Any:
        if value is None:
            return None
        return _finite_number(
            value,
            name="spot_angle_degrees",
            minimum=1.0,
            maximum=180.0,
        )

    @field_validator(
        "spot_blend", "diffuse_factor", "specular_factor", "volume_factor", mode="before"
    )
    @classmethod
    def validate_factor(cls, value: Any, info: Any) -> float | None:
        if value is None:
            return None
        return _finite_number(value, name=info.field_name, minimum=0.0, maximum=1.0)

    @model_validator(mode="after")
    def validate_type_specific_fields(self) -> LightSpec:
        if self.target_point is not None and self.target_object is not None:
            raise ValueError("target_point and target_object are mutually exclusive")
        if self.type == "POINT" and (
            self.target_point is not None or self.target_object is not None
        ):
            raise ValueError("POINT lights do not accept targets")

        if self.type != "SUN" and self.sun_angle_degrees is not None:
            raise ValueError("sun_angle_degrees is only valid for SUN lights")
        if self.type != "AREA" and any(
            value is not None for value in (self.area_shape, self.size, self.size_y)
        ):
            raise ValueError("area_shape, size, and size_y are only valid for AREA lights")
        if self.type != "SPOT" and any(
            value is not None for value in (self.spot_angle_degrees, self.spot_blend)
        ):
            raise ValueError("spot_angle_degrees and spot_blend are only valid for SPOT lights")
        if self.radius is not None and self.type not in {"POINT", "SPOT"}:
            raise ValueError("radius is only valid for POINT and SPOT lights")
        if self.size_y is not None and self.area_shape not in {"RECTANGLE", "ELLIPSE"}:
            raise ValueError("size_y requires RECTANGLE or ELLIPSE area_shape")
        return self


class WorldOverride(BaseModel):
    """Optional managed solid-color world override."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["KEEP", "MANAGED_SOLID"] = "KEEP"
    color_rgb: Vec3 = (0.0, 0.0, 0.0)
    strength: float = 0.0

    @field_validator("color_rgb", mode="before")
    @classmethod
    def validate_rgb(cls, value: Any) -> Vec3:
        try:
            color = validate_color(value)
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc
        if len(color) != 3:
            raise ValueError("color_rgb must contain exactly three components")
        return tuple(float(component) for component in color)  # type: ignore[return-value]

    @field_validator("strength", mode="before")
    @classmethod
    def validate_strength(cls, value: Any) -> float:
        return _finite_number(value, name="strength", minimum=0.0, maximum=1000.0)


class SceneOverrides(BaseModel):
    """Non-light scene changes applied transactionally with a light plan."""

    model_config = ConfigDict(extra="forbid")

    existing_light_policy: Literal["KEEP", "MUTE_NON_MANAGED"] = "KEEP"
    world: WorldOverride | None = None
    exposure: float | None = None

    @field_validator("exposure", mode="before")
    @classmethod
    def validate_exposure(cls, value: Any) -> Any:
        if value is None:
            return None
        return _finite_number(value, name="exposure", minimum=-32.0, maximum=32.0)


class CaptureMetadata(BaseModel):
    """Structured metadata paired with a native MCP viewport image."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["CAPTURE", "RESTORE"]
    session_id: str | None = None
    workspace_name: str | None = None
    area_index: int | None = None
    camera_name: str | None = None
    engine: str | None = None
    capture_backend: (
        Literal[
            "BLENDER_SCREENSHOT_AREA",
            "MACOS_SCREENCAPTUREKIT",
            "QUICK_CYCLES_RENDER",
        ]
        | None
    ) = None
    requested_preview_samples: int | None = None
    denoise: bool | None = None
    device: str | None = None
    settle_basis: Literal["elapsed", "not_applicable"] | None = None
    settle_seconds: float | None = None
    source_width: int | None = None
    source_height: int | None = None
    output_width: int | None = None
    output_height: int | None = None
    format: Literal["JPEG", "PNG"] | None = None
    jpeg_quality: int | None = None
    byte_count: int | None = None
    staging_path: str | None = None
    staging_sha256: str | None = None
    keep_session: bool | None = None
    restored: bool = False
    warnings: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)


@mcp.tool()
def get_lighting_context(
    camera_name: str | None = None,
    scope: RaycastMode = "CAMERA",
    collection_names: list[str] = [],
    detail: LightingDetail = "CANDIDATES",
    semantic_terms: list[str] = BUILT_IN_SEMANTIC_TERMS,
    include_hidden: bool = False,
    max_instances: int = 500,
    cursor: str | None = None,
    max_surface_triangles: int = 200_000,
    cache_mode: CacheMode = "USE",
) -> dict[str, Any]:
    """Return bounded, camera-aware scene geometry and relighting candidates.

    Results include revision tokens that should be passed to ``batch_raycast``
    and ``apply_light_plan`` to reject stale spatial decisions.
    """
    if camera_name is not None:
        camera_name = validate_object_name(camera_name)
    validate_enum(scope, {"CAMERA", "SCENE", "COLLECTIONS"}, name="scope")
    validate_enum(detail, {"BOUNDS", "CANDIDATES"}, name="detail")
    validate_enum(cache_mode, {"USE", "REFRESH"}, name="cache_mode")
    if not isinstance(include_hidden, bool):
        raise ValidationError("include_hidden must be a boolean")

    collection_names = _validate_safe_strings(
        collection_names,
        name="collection_names",
        max_count=64,
        max_length=63,
    )
    for collection_name in collection_names:
        validate_object_name(collection_name)
    if scope == "COLLECTIONS" and not collection_names:
        raise ValidationError("collection_names is required when scope='COLLECTIONS'")

    semantic_terms = _validate_safe_strings(
        semantic_terms,
        name="semantic_terms",
        max_count=64,
        max_length=64,
    )
    validate_numeric_range(max_instances, min_val=1, max_val=MAX_INSTANCES, name="max_instances")
    validate_numeric_range(
        max_surface_triangles,
        min_val=0,
        max_val=MAX_SURFACE_TRIANGLES,
        name="max_surface_triangles",
    )
    if cursor is not None:
        if not isinstance(cursor, str) or len(cursor) > 1024:
            raise ValidationError("cursor must be an opaque string of at most 1024 characters")
        if any(ord(character) < 32 for character in cursor):
            raise ValidationError("cursor contains control characters")

    params = {
        "camera_name": camera_name,
        "scope": scope,
        "collection_names": collection_names,
        "detail": detail,
        "semantic_terms": semantic_terms,
        "include_hidden": include_hidden,
        "max_instances": max_instances,
        "cursor": cursor,
        "max_surface_triangles": max_surface_triangles,
        "cache_mode": cache_mode,
    }
    result = _send_relighting_command("get_lighting_context", params)
    if not isinstance(result, dict):
        raise RuntimeError("Blender returned invalid lighting context data")
    return result


@mcp.tool()
def batch_raycast(
    rays: list[RaySpec],
    max_hits: int = 8,
    ignore_object_patterns: list[str] = [],
    ignore_material_patterns: list[str] = [],
    include_ignored_hits: bool = True,
    expected_geometry_revision: int | None = None,
    time_budget_ms: int = 2000,
) -> dict[str, Any]:
    """Trace many bounded rays against one evaluated dependency graph snapshot."""
    if not rays:
        raise ValidationError("rays must contain at least one ray")
    if len(rays) > MAX_RAYS:
        raise ValidationError(f"rays may contain at most {MAX_RAYS} entries")
    ray_models = [ray if isinstance(ray, RaySpec) else RaySpec.model_validate(ray) for ray in rays]
    ray_ids = [ray.id for ray in ray_models]
    if len(ray_ids) != len(set(ray_ids)):
        raise ValidationError("ray ids must be unique")

    validate_numeric_range(max_hits, min_val=1, max_val=MAX_RAY_HITS, name="max_hits")
    validate_numeric_range(time_budget_ms, min_val=10, max_val=5000, name="time_budget_ms")
    if not isinstance(include_ignored_hits, bool):
        raise ValidationError("include_ignored_hits must be a boolean")
    if expected_geometry_revision is not None:
        validate_numeric_range(
            expected_geometry_revision,
            min_val=0,
            max_val=2**63 - 1,
            name="expected_geometry_revision",
        )

    ignore_object_patterns = _validate_safe_strings(
        ignore_object_patterns,
        name="ignore_object_patterns",
        max_count=MAX_PATTERN_COUNT,
    )
    ignore_material_patterns = _validate_safe_strings(
        ignore_material_patterns,
        name="ignore_material_patterns",
        max_count=MAX_PATTERN_COUNT,
    )

    params = {
        "rays": [ray.model_dump(mode="json", exclude_none=True) for ray in ray_models],
        "max_hits": max_hits,
        "ignore_object_patterns": ignore_object_patterns,
        "ignore_material_patterns": ignore_material_patterns,
        "include_ignored_hits": include_ignored_hits,
        "expected_geometry_revision": expected_geometry_revision,
        "time_budget_ms": time_budget_ms,
    }
    result = _send_relighting_command("batch_raycast", params)
    if not isinstance(result, dict):
        raise RuntimeError("Blender returned invalid raycast data")
    return result


@mcp.tool()
def apply_light_plan(
    action: Literal["VALIDATE", "APPLY", "ROLLBACK"],
    plan_id: str | None = None,
    mode: Literal["REPLACE_MANAGED", "PATCH_MANAGED"] = "PATCH_MANAGED",
    expected_geometry_revision: int | None = None,
    collection_name: str = "AI_RELIGHT",
    lights: list[LightSpec] = [],
    remove_ids: list[str] = [],
    scene_overrides: SceneOverrides | None = None,
    transaction_id: str | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Validate, atomically apply, or explicitly roll back a managed light plan."""
    validate_enum(action, {"VALIDATE", "APPLY", "ROLLBACK"}, name="action")
    validate_enum(mode, {"REPLACE_MANAGED", "PATCH_MANAGED"}, name="mode")
    collection_name = validate_object_name(collection_name)
    if not isinstance(strict, bool):
        raise ValidationError("strict must be a boolean")

    if expected_geometry_revision is not None:
        validate_numeric_range(
            expected_geometry_revision,
            min_val=0,
            max_val=2**63 - 1,
            name="expected_geometry_revision",
        )
    if plan_id is not None:
        try:
            plan_id = _validate_identifier(plan_id, name="plan_id")
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
    if transaction_id is not None:
        try:
            transaction_id = _validate_identifier(transaction_id, name="transaction_id")
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc

    if len(lights) > MAX_LIGHTS:
        raise ValidationError(f"lights may contain at most {MAX_LIGHTS} entries")
    light_models = [
        light if isinstance(light, LightSpec) else LightSpec.model_validate(light)
        for light in lights
    ]
    light_ids = [light.id for light in light_models]
    if len(light_ids) != len(set(light_ids)):
        raise ValidationError("light ids must be unique")

    if len(remove_ids) > MAX_LIGHTS:
        raise ValidationError(f"remove_ids may contain at most {MAX_LIGHTS} entries")
    try:
        remove_ids = [_validate_identifier(remove_id, name="remove id") for remove_id in remove_ids]
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    if len(remove_ids) != len(set(remove_ids)):
        raise ValidationError("remove_ids must be unique")
    if set(light_ids).intersection(remove_ids):
        raise ValidationError("a light id cannot appear in both lights and remove_ids")

    if scene_overrides is not None and not isinstance(scene_overrides, SceneOverrides):
        scene_overrides = SceneOverrides.model_validate(scene_overrides)

    if action == "ROLLBACK":
        if transaction_id is None and plan_id is None:
            raise ValidationError("ROLLBACK requires transaction_id or plan_id")
        if light_models or remove_ids or scene_overrides is not None:
            raise ValidationError("ROLLBACK does not accept lights, remove_ids, or scene_overrides")
    else:
        if plan_id is None:
            raise ValidationError(f"plan_id is required for {action}")
        if transaction_id is not None:
            raise ValidationError("transaction_id is only valid for ROLLBACK")

    params = {
        "action": action,
        "plan_id": plan_id,
        "mode": mode,
        "expected_geometry_revision": expected_geometry_revision,
        "collection_name": collection_name,
        "lights": [light.model_dump(mode="json", exclude_none=True) for light in light_models],
        "remove_ids": remove_ids,
        "scene_overrides": (
            scene_overrides.model_dump(mode="json", exclude_none=True)
            if scene_overrides is not None
            else None
        ),
        "transaction_id": transaction_id,
        "strict": strict,
    }
    result = _send_relighting_command("apply_light_plan", params)
    if not isinstance(result, dict):
        raise RuntimeError("Blender returned invalid light plan data")
    return result


def _remove_macos_capture_helper() -> None:
    """Remove the process-local compiled ScreenCaptureKit helper."""
    global _MACOS_CAPTURE_HELPER
    helper = _MACOS_CAPTURE_HELPER
    _MACOS_CAPTURE_HELPER = None
    if helper is not None:
        try:
            helper.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove macOS capture helper %s", helper)


def _ensure_macos_capture_helper() -> Path:
    """Compile the bundled Swift helper once for warm viewport captures."""
    global _MACOS_CAPTURE_HELPER
    with _MACOS_CAPTURE_HELPER_LOCK:
        if _MACOS_CAPTURE_HELPER is not None and _MACOS_CAPTURE_HELPER.is_file():
            return _MACOS_CAPTURE_HELPER
        if not _MACOS_CAPTURE_SOURCE.is_file():
            raise RuntimeError(f"Bundled macOS capture source is missing: {_MACOS_CAPTURE_SOURCE}")
        swiftc = shutil.which("swiftc")
        if swiftc is None:
            raise RuntimeError(
                "macOS ScreenCaptureKit capture requires the Xcode Command Line Tools "
                "('swiftc' was not found)"
            )
        source_digest = hashlib.sha256(_MACOS_CAPTURE_SOURCE.read_bytes()).hexdigest()[:16]
        helper = Path(tempfile.gettempdir()) / (
            f"blend-ai-macos-capture-{os.getpid()}-{source_digest}"
        )
        helper.unlink(missing_ok=True)
        try:
            completed = subprocess.run(
                [swiftc, "-O", str(_MACOS_CAPTURE_SOURCE), "-o", str(helper)],
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            helper.unlink(missing_ok=True)
            raise RuntimeError(f"Could not compile macOS capture helper: {exc}") from exc
        if completed.returncode != 0 or not helper.is_file():
            helper.unlink(missing_ok=True)
            detail = (completed.stderr or completed.stdout or "unknown swiftc error").strip()
            raise RuntimeError("Could not compile macOS capture helper: " + detail[:2000])
        _MACOS_CAPTURE_HELPER = helper
        atexit.register(_remove_macos_capture_helper)
        return helper


def _capture_macos_window(
    target: Any,
) -> tuple[bytes, int, int, dict[str, Any], dict[str, Any]]:
    """Capture one visible Blender window with ScreenCaptureKit.

    Blender 5.2's screenshot operators return only the gray compositor backing
    surface for a Metal-rendered Cycles viewport on the verified Mac.  This
    helper captures that same visible native window without invoking a render.
    """
    if not isinstance(target, dict):
        raise RuntimeError("Blender did not return a macOS capture target")

    def target_int(name: str, minimum: int, maximum: int) -> int:
        value = target.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError(f"macOS capture target {name} must be an integer")
        if value < minimum or value > maximum:
            raise RuntimeError(
                f"macOS capture target {name} must be between {minimum} and {maximum}"
            )
        return value

    process_id = target_int("process_id", 1, 2_147_483_647)
    content_width = target_int("window_content_width", 1, 32_768)
    content_height = target_int("window_content_height", 1, 32_768)
    window_x = target_int("window_x", -131_072, 131_072)
    window_y = target_int("window_y", -131_072, 131_072)
    content_region = target.get("region_rect")
    # Validate the content-relative rectangle before spawning any process.
    content_crop = _crop_box(content_region, content_width, content_height)
    if content_crop is None:
        raise RuntimeError("macOS capture target is missing region_rect")

    helper = _ensure_macos_capture_helper()
    with tempfile.TemporaryDirectory(prefix="blend_ai_macos_capture_") as temp_dir:
        output_path = Path(temp_dir) / "blender-window.png"
        try:
            completed = subprocess.run(
                [
                    str(helper),
                    str(process_id),
                    str(content_width),
                    str(content_height),
                    str(window_x),
                    str(window_y),
                    str(output_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"macOS ScreenCaptureKit capture failed: {exc}") from exc
        if completed.returncode != 0 or not output_path.is_file():
            detail = (completed.stderr or completed.stdout or "unknown capture error").strip()
            raise RuntimeError("macOS ScreenCaptureKit capture failed: " + detail[:2000])
        try:
            helper_metadata = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("macOS ScreenCaptureKit helper returned invalid metadata") from exc
        try:
            byte_count = output_path.stat().st_size
        except OSError as exc:
            raise RuntimeError("macOS ScreenCaptureKit output disappeared") from exc
        if byte_count <= 0 or byte_count > MAX_INPUT_IMAGE_BYTES:
            raise RuntimeError("macOS ScreenCaptureKit image exceeds the byte limit")
        source_bytes = output_path.read_bytes()

    try:
        with PILImage.open(BytesIO(source_bytes)) as source:
            source_width, source_height = source.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise RuntimeError("macOS ScreenCaptureKit returned an invalid image") from exc
    if source_width * source_height > MAX_IMAGE_PIXELS:
        raise RuntimeError("macOS ScreenCaptureKit image exceeds the pixel limit")

    # ScreenCaptureKit includes native title-bar pixels above Blender's RNA
    # content area.  The horizontal ratio provides the Retina scale; the
    # residual vertical pixels are deterministic window chrome.
    scale = source_width / content_width
    if not math.isfinite(scale) or scale <= 0.0 or scale > 4.0:
        raise RuntimeError("macOS ScreenCaptureKit returned an invalid Retina scale")
    scaled_content_height = round(content_height * scale)
    top_chrome = source_height - scaled_content_height
    if top_chrome < 0 or top_chrome > round(512 * scale):
        raise RuntimeError("macOS window chrome dimensions are inconsistent")
    left, top, right, bottom = content_crop
    window_region = {
        "x": round(left * scale),
        "y": top_chrome + round(top * scale),
        "width": round((right - left) * scale),
        "height": round((bottom - top) * scale),
        "origin": "TOP_LEFT",
    }
    # Validate the translated crop against the actual native PNG.
    _crop_box(window_region, source_width, source_height)
    if not isinstance(helper_metadata, dict):
        raise RuntimeError("macOS ScreenCaptureKit metadata must be an object")
    bounds = helper_metadata.get("bounds")
    if not isinstance(bounds, dict):
        raise RuntimeError("macOS ScreenCaptureKit metadata lacks window bounds")

    def metadata_int(container: dict[str, Any], name: str) -> int:
        value = container.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError(f"macOS ScreenCaptureKit metadata {name} must be an integer")
        return value

    if metadata_int(helper_metadata, "ownerPID") != process_id:
        raise RuntimeError("macOS ScreenCaptureKit captured a different process")
    if (
        metadata_int(helper_metadata, "width") != source_width
        or metadata_int(helper_metadata, "height") != source_height
    ):
        raise RuntimeError("macOS ScreenCaptureKit metadata does not match its PNG")
    bounds_x = metadata_int(bounds, "x")
    bounds_y = metadata_int(bounds, "y")
    bounds_width = metadata_int(bounds, "width")
    bounds_height = metadata_int(bounds, "height")
    if (
        abs(bounds_width - content_width) > 4
        or bounds_height < content_height
        or bounds_height - content_height > 128
        or abs(bounds_x - window_x) > 8
        or abs(bounds_y - window_y) > 128
    ):
        raise RuntimeError(
            "macOS ScreenCaptureKit selected a Blender window outside the "
            "requested position/dimension tolerances"
        )
    return source_bytes, source_width, source_height, window_region, helper_metadata


def _decode_capture_image(result: dict[str, Any]) -> bytes:
    encoded = result.get("image_base64")
    if not isinstance(encoded, str) or not encoded:
        raise RuntimeError("Blender capture did not return image_base64")
    if len(encoded) > ((MAX_INPUT_IMAGE_BYTES + 2) // 3) * 4 + 4:
        raise RuntimeError("Blender capture image exceeds the input size limit")
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError("Blender capture returned invalid base64 image data") from exc
    if not image_bytes or len(image_bytes) > MAX_INPUT_IMAGE_BYTES:
        raise RuntimeError("Blender capture image exceeds the input size limit")
    return image_bytes


def _crop_box(region_rect: Any, width: int, height: int) -> tuple[int, int, int, int] | None:
    if region_rect is None:
        return None
    if not isinstance(region_rect, dict):
        raise RuntimeError("region_rect must be an object")
    try:
        x = int(region_rect["x"])
        y = int(region_rect["y"])
        region_width = int(region_rect["width"])
        region_height = int(region_rect["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("region_rect must contain integer x, y, width, and height") from exc
    if region_width <= 0 or region_height <= 0:
        raise RuntimeError("region_rect width and height must be positive")
    origin = region_rect.get("origin", "TOP_LEFT")
    if origin == "BOTTOM_LEFT":
        y = height - y - region_height
    elif origin != "TOP_LEFT":
        raise RuntimeError("region_rect origin must be TOP_LEFT or BOTTOM_LEFT")
    box = (x, y, x + region_width, y + region_height)
    if x < 0 or y < 0 or box[2] > width or box[3] > height:
        raise RuntimeError("region_rect lies outside the captured image")
    return box


def _encode_capture_image(
    source_bytes: bytes,
    *,
    source_width: int | None,
    source_height: int | None,
    region_rect: Any,
    max_size: int,
    output_format: Literal["JPEG", "PNG"],
    jpeg_quality: int,
) -> tuple[bytes, int, int, int, int]:
    try:
        with PILImage.open(BytesIO(source_bytes)) as source:
            actual_source_width, actual_source_height = source.size
            if actual_source_width * actual_source_height > MAX_IMAGE_PIXELS:
                raise RuntimeError("Blender capture exceeds the image pixel limit")
            if source_width is not None and source_width != actual_source_width:
                raise RuntimeError("reported source_width does not match the captured image")
            if source_height is not None and source_height != actual_source_height:
                raise RuntimeError("reported source_height does not match the captured image")
            source.load()
            image = source.copy()
    except RuntimeError:
        raise
    except (PILImage.DecompressionBombError, UnidentifiedImageError, OSError, ValueError) as exc:
        raise RuntimeError("Blender capture did not contain a supported image") from exc

    crop_box = _crop_box(region_rect, actual_source_width, actual_source_height)
    if crop_box is not None:
        image = image.crop(crop_box)

    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), resample=PILImage.Resampling.LANCZOS)

    output = BytesIO()
    if output_format == "JPEG":
        if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
            rgba = image.convert("RGBA")
            background = PILImage.new("RGB", rgba.size, (0, 0, 0))
            background.paste(rgba, mask=rgba.getchannel("A"))
            image = background
        else:
            image = image.convert("RGB")
        image.save(
            output,
            format="JPEG",
            quality=jpeg_quality,
            optimize=True,
            progressive=True,
        )
    else:
        image.save(output, format="PNG", optimize=True)

    encoded = output.getvalue()
    if not encoded or len(encoded) > MAX_OUTPUT_IMAGE_BYTES:
        raise RuntimeError("encoded viewport image exceeds the output size limit")
    return (
        encoded,
        actual_source_width,
        actual_source_height,
        image.width,
        image.height,
    )


def _validate_capture_staging_path(
    value: str,
    *,
    output_format: Literal["JPEG", "PNG"],
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError("staging_path must be a non-empty absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise ValidationError("staging_path must be an absolute path")
    expected_suffixes = {".jpg", ".jpeg"} if output_format == "JPEG" else {".png"}
    if path.suffix.lower() not in expected_suffixes:
        expected = ".jpg or .jpeg" if output_format == "JPEG" else ".png"
        raise ValidationError(f"staging_path must end in {expected} for {output_format}")
    if not path.parent.is_dir():
        raise ValidationError("staging_path parent directory must already exist")
    if path.exists() or path.is_symlink():
        raise ValidationError("staging_path already exists; capture staging never overwrites")
    return path


def _stage_capture_bytes(
    image_bytes: bytes,
    destination: Path,
    output_format: Literal["JPEG", "PNG"],
    output_width: int,
    output_height: int,
) -> str:
    """Atomically create one validated exact-byte capture artifact."""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(image_bytes)
            handle.flush()
            os.fsync(handle.fileno())

        try:
            with PILImage.open(temporary_path) as image:
                actual_format = str(image.format or "").upper()
                actual_size = image.size
                image.verify()
        except (
            PILImage.DecompressionBombError,
            UnidentifiedImageError,
            OSError,
            ValueError,
        ) as exc:
            raise RuntimeError("staged viewport capture failed image validation") from exc
        if actual_format != output_format:
            raise RuntimeError(
                f"staged viewport capture format mismatch: {actual_format} != {output_format}"
            )
        if actual_size != (output_width, output_height):
            raise RuntimeError("staged viewport capture dimensions do not match the encoded result")
        digest = hashlib.sha256(image_bytes).hexdigest()
        if temporary_path.stat().st_size != len(image_bytes):
            raise RuntimeError("staged viewport capture byte count does not match")

        # Linking a fully flushed same-directory temporary file publishes the
        # artifact atomically and fails if another process claimed the literal
        # ledger path.  Unlike os.replace(), this can never overwrite evidence.
        try:
            os.link(temporary_path, destination)
        except FileExistsError as exc:
            raise RuntimeError(
                "staging_path already exists; capture staging never overwrites"
            ) from exc
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # The image file itself is already flushed and atomically visible;
            # some filesystems do not permit opening directories for fsync.
            pass
        return digest
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _capture_result(
    metadata: CaptureMetadata,
    image_bytes: bytes | None = None,
) -> CallToolResult:
    content: list[Any] = []
    if image_bytes is not None:
        assert metadata.format is not None
        content.append(
            MCPImage(data=image_bytes, format=metadata.format.lower()).to_image_content()
        )
    # The MCP specification recommends duplicating structured output as text
    # for clients that predate structuredContent.
    content.append(TextContent(type="text", text=metadata.model_dump_json()))
    return CallToolResult(
        content=content,
        structuredContent=metadata.model_dump(mode="json"),
    )


@mcp.tool()
async def capture_cycles_viewport(
    action: Literal["CAPTURE", "RESTORE"] = "CAPTURE",
    session_id: str | None = None,
    workspace_name: str = "AI Preview",
    area_index: int | None = None,
    camera_name: str | None = None,
    preview_samples: int = 16,
    settle_seconds: float = 2.0,
    denoise: bool = True,
    capture_mode: Literal["VIEWPORT", "QUICK_RENDER"] = "VIEWPORT",
    device: Literal["KEEP", "GPU", "CPU"] = "GPU",
    max_size: int = 1024,
    format: Literal["JPEG", "PNG"] = "JPEG",
    jpeg_quality: int = 85,
    keep_session: bool = True,
    staging_path: str | None = None,
) -> Annotated[CallToolResult, CaptureMetadata]:
    """Capture pixels from a live Cycles Rendered viewport as native MCP image content.

    ``capture_mode='VIEWPORT'`` prepares the visible viewport, returns control
    to Blender while Cycles accumulates, then captures the editor pixels.
    ``capture_mode='QUICK_RENDER'`` instead writes a temporary low-sample,
    denoised camera PNG and is an explicit reliability alternative.
    With ``keep_session=True``, pass the returned session ID to later captures
    to reuse the configured preview, then use ``action='RESTORE'`` when done.
    """
    started = time.perf_counter()
    validate_enum(action, {"CAPTURE", "RESTORE"}, name="action")
    if session_id is not None:
        try:
            session_id = _validate_identifier(session_id, name="session_id")
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc

    if action == "RESTORE":
        if staging_path is not None:
            raise ValidationError("staging_path is only valid for action='CAPTURE'")
        restore_started = time.perf_counter()
        result = await _send_relighting_command_async(
            "restore_cycles_viewport",
            {"session_id": session_id},
        )
        if not isinstance(result, dict):
            raise RuntimeError("Blender returned invalid viewport restoration data")
        resolved_session_id = result.get("session_id", session_id)
        metadata = CaptureMetadata(
            action="RESTORE",
            session_id=resolved_session_id,
            workspace_name=result.get("workspace_name"),
            area_index=result.get("area_index"),
            camera_name=result.get("camera_name"),
            restored=bool(result.get("restored", True)),
            timings_ms={
                "restore": (time.perf_counter() - restore_started) * 1000.0,
                "total": (time.perf_counter() - started) * 1000.0,
            },
        )
        return _capture_result(metadata)

    workspace_name = validate_object_name(workspace_name)
    if camera_name is not None:
        camera_name = validate_object_name(camera_name)
    if area_index is not None:
        validate_numeric_range(area_index, min_val=0, max_val=128, name="area_index")
    validate_numeric_range(
        preview_samples,
        min_val=1,
        max_val=4096,
        name="preview_samples",
    )
    validate_numeric_range(
        settle_seconds,
        min_val=0.0,
        max_val=30.0,
        name="settle_seconds",
    )
    validate_numeric_range(max_size, min_val=64, max_val=4096, name="max_size")
    validate_numeric_range(jpeg_quality, min_val=1, max_val=100, name="jpeg_quality")
    validate_enum(capture_mode, {"VIEWPORT", "QUICK_RENDER"}, name="capture_mode")
    validate_enum(device, {"KEEP", "GPU", "CPU"}, name="device")
    validate_enum(format, {"JPEG", "PNG"}, name="format")
    if not isinstance(denoise, bool):
        raise ValidationError("denoise must be a boolean")
    if not isinstance(keep_session, bool):
        raise ValidationError("keep_session must be a boolean")
    resolved_staging_path = (
        _validate_capture_staging_path(staging_path, output_format=format)
        if staging_path is not None
        else None
    )

    prepare_params = {
        "session_id": session_id,
        "workspace_name": workspace_name,
        "area_index": area_index,
        "camera_name": camera_name,
        "preview_samples": preview_samples,
        "denoise": denoise,
        "device": device,
        "keep_session": keep_session,
    }

    prepared_session_id: str | None = None
    prepare_attempted = False
    capture_succeeded = False
    timings_ms: dict[str, float] = {}

    def remember_cancelled_prepare(result: Any) -> None:
        nonlocal prepared_session_id
        if isinstance(result, dict):
            candidate = result.get("session_id")
            if isinstance(candidate, str) and candidate:
                prepared_session_id = candidate

    try:
        prepare_started = time.perf_counter()
        prepare_attempted = True
        prepared = await _send_relighting_command_async(
            "prepare_cycles_viewport",
            prepare_params,
            on_cancel_result=remember_cancelled_prepare,
        )
        timings_ms["prepare"] = (time.perf_counter() - prepare_started) * 1000.0
        if not isinstance(prepared, dict):
            raise RuntimeError("Blender returned invalid viewport preparation data")
        prepared_session_id = prepared.get("session_id")
        if not isinstance(prepared_session_id, str) or not prepared_session_id:
            raise RuntimeError("Blender viewport preparation did not return a session_id")
        try:
            prepared_session_id = _validate_identifier(
                prepared_session_id,
                name="prepared session_id",
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

        if capture_mode == "VIEWPORT":
            settle_started = time.perf_counter()
            await asyncio.sleep(float(settle_seconds))
            timings_ms["settle"] = (time.perf_counter() - settle_started) * 1000.0
        else:
            timings_ms["settle"] = 0.0

        capture_started = time.perf_counter()
        requested_backend = prepared.get(
            "capture_backend_required",
            "BLENDER_SCREENSHOT_AREA",
        )
        use_quick_render = capture_mode == "QUICK_RENDER"
        use_macos_capture = not use_quick_render and requested_backend == "MACOS_SCREENCAPTUREKIT"
        if use_macos_capture and sys.platform != "darwin":
            raise RuntimeError(
                "Blender requested macOS ScreenCaptureKit from a non-macOS MCP process"
            )
        if use_quick_render:
            captured = await _send_relighting_command_async(
                "capture_cycles_quick_render",
                {
                    "session_id": prepared_session_id,
                    "max_size": max_size,
                    "denoise": denoise,
                },
            )
        else:
            capture_params: dict[str, Any] = {
                "session_id": prepared_session_id,
                # ScreenCaptureKit must read the native window before Blender
                # restores it, so the private metadata phase always retains it.
                "keep_session": True if use_macos_capture else keep_session,
            }
            if use_macos_capture:
                capture_params["metadata_only"] = True
            captured = await _send_relighting_command_async(
                "capture_cycles_viewport_area",
                capture_params,
            )
        if not isinstance(captured, dict):
            raise RuntimeError("Blender returned invalid viewport capture data")
        if captured.get("session_id") != prepared_session_id:
            raise RuntimeError("Blender viewport capture returned a mismatched session_id")

        helper_metadata: dict[str, Any] = {}
        if use_macos_capture:
            (
                source_bytes,
                source_width,
                source_height,
                region_rect,
                helper_metadata,
            ) = await _to_thread_cancellation_safe(
                _capture_macos_window,
                captured.get("capture_target"),
            )
            captured["capture_backend"] = "MACOS_SCREENCAPTUREKIT"
            if not keep_session:
                restored = await _send_relighting_command_async(
                    "restore_cycles_viewport",
                    {"session_id": prepared_session_id},
                )
                if not isinstance(restored, dict) or not restored.get("restored"):
                    raise RuntimeError("Blender did not restore the captured viewport session")
                captured["restored"] = True
        else:
            source_bytes = _decode_capture_image(captured)
            source_width = captured.get("source_width")
            source_height = captured.get("source_height")
            if source_width is not None:
                source_width = int(source_width)
            if source_height is not None:
                source_height = int(source_height)
            region_rect = captured.get("region_rect")
            captured["capture_backend"] = (
                "QUICK_CYCLES_RENDER" if use_quick_render else "BLENDER_SCREENSHOT_AREA"
            )
            if use_quick_render and not keep_session:
                restored = await _send_relighting_command_async(
                    "restore_cycles_viewport",
                    {"session_id": prepared_session_id},
                )
                if not isinstance(restored, dict) or not restored.get("restored"):
                    raise RuntimeError("Blender did not restore the captured viewport session")
                captured["restored"] = True
        timings_ms["capture"] = (time.perf_counter() - capture_started) * 1000.0

        encoding_started = time.perf_counter()
        (
            image_bytes,
            actual_source_width,
            actual_source_height,
            output_width,
            output_height,
        ) = _encode_capture_image(
            source_bytes,
            source_width=source_width,
            source_height=source_height,
            region_rect=region_rect,
            max_size=max_size,
            output_format=format,
            jpeg_quality=jpeg_quality,
        )
        timings_ms["encode"] = (time.perf_counter() - encoding_started) * 1000.0
        staging_sha256: str | None = None
        if resolved_staging_path is not None:
            staging_started = time.perf_counter()
            staging_sha256 = await _to_thread_cancellation_safe(
                _stage_capture_bytes,
                image_bytes,
                resolved_staging_path,
                format,
                output_width,
                output_height,
            )
            timings_ms["stage"] = (time.perf_counter() - staging_started) * 1000.0
        timings_ms["total"] = (time.perf_counter() - started) * 1000.0

        metadata = CaptureMetadata(
            action="CAPTURE",
            session_id=prepared_session_id,
            workspace_name=captured.get(
                "workspace_name",
                prepared.get("workspace_name", workspace_name),
            ),
            area_index=captured.get("area_index", prepared.get("area_index", area_index)),
            camera_name=captured.get(
                "camera_name",
                prepared.get("camera_name", camera_name),
            ),
            engine=captured.get("engine", prepared.get("engine", "CYCLES")),
            capture_backend=captured.get("capture_backend"),
            requested_preview_samples=preview_samples,
            denoise=captured.get("denoise", prepared.get("denoise", denoise)),
            device=captured.get("device", prepared.get("device", device)),
            settle_basis=("elapsed" if capture_mode == "VIEWPORT" else "not_applicable"),
            settle_seconds=(float(settle_seconds) if capture_mode == "VIEWPORT" else 0.0),
            source_width=actual_source_width,
            source_height=actual_source_height,
            output_width=output_width,
            output_height=output_height,
            format=format,
            jpeg_quality=jpeg_quality if format == "JPEG" else None,
            byte_count=len(image_bytes),
            staging_path=(
                str(resolved_staging_path) if resolved_staging_path is not None else None
            ),
            staging_sha256=staging_sha256,
            keep_session=keep_session,
            restored=bool(captured.get("restored", not keep_session)),
            warnings=(
                list(prepared.get("warnings", []))
                + list(captured.get("warnings", []))
                + (
                    [
                        "Captured visible Blender window "
                        f"{helper_metadata.get('windowID')} via ScreenCaptureKit."
                    ]
                    if helper_metadata
                    else []
                )
            ),
            timings_ms=timings_ms,
        )
        tool_result = _capture_result(metadata, image_bytes)
        capture_succeeded = True
        return tool_result
    finally:
        if not capture_succeeded and prepare_attempted:
            primary_error = sys.exc_info()[1]
            cleanup_error: BaseException | None = None
            try:
                restored = await _send_relighting_command_async(
                    "restore_cycles_viewport",
                    {"session_id": prepared_session_id},
                )
                if not isinstance(restored, dict):
                    raise RuntimeError("Blender returned invalid viewport restoration data")
                if prepared_session_id is not None and not restored.get("restored"):
                    raise RuntimeError(
                        f"Blender did not restore viewport session {prepared_session_id}"
                    )
            except BaseException as exc:
                cleanup_error = exc
            if cleanup_error is not None:
                raise RuntimeError(
                    "Cycles capture did not complete and preview restoration also "
                    f"failed: {cleanup_error}"
                ) from primary_error
