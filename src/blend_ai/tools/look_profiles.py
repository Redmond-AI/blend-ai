"""Strict MCP boundary for managed Blender look profiles.

Phase 1 deliberately implements only client-side schemas, bounded request
validation, command forwarding, and Composite-result image packaging.  Blender
scene mutation remains the responsibility of the add-on handlers so this
module cannot bypass their ownership, revision, transaction, or rollback
checks.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import math
import re
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.server.fastmcp.utilities.types import Image as MCPImage
from mcp.types import CallToolResult, TextContent
from PIL import Image as PILImage
from PIL import ImageDraw, ImageOps
from PIL import UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from blend_ai.server import get_connection, mcp
from blend_ai.tools.relighting import LightSpec


MAX_PROFILES = 128
MAX_TARGET_SCENES = 64
MAX_CAMERAS = 32
MAX_ATMOSPHERE_COMPONENTS = 32
MAX_TAGS = 32
MAX_RENDER_FRAMES = 10_000
MAX_RENDER_ITEMS = 10_000
MAX_PROXY_INPUT_BYTES = 32 * 1024 * 1024
MAX_PROXY_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_PROXY_PIXELS = 50_000_000

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_. -]{1,63}$")
_SAFE_TEMPLATE_RE = re.compile(r"^[A-Za-z0-9_.{}:-]{1,128}$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_FINGERPRINT_RE = re.compile(r"^sha256:[a-f0-9]{64}$")

Vec3 = tuple[float, float, float]
Rgb = tuple[float, float, float]


def _identifier(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID_RE.fullmatch(value):
        raise ValueError(
            f"{name} must start with a letter or number and contain only "
            "letters, numbers, underscores, hyphens, dots, and colons"
        )
    return value


def _blender_name(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if not value.strip() or not _SAFE_NAME_RE.fullmatch(value):
        raise ValueError(
            f"{name} must be 1-63 characters using letters, numbers, spaces, "
            "underscores, hyphens, or dots"
        )
    return value


def _bounded_text(value: str, *, name: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    value = value.strip()
    if not value or len(value) > maximum:
        raise ValueError(f"{name} must be 1-{maximum} characters")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} contains control characters")
    return value


def _finite_vec3(value: Any, *, name: str) -> Vec3:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three numbers")
    result: list[float] = []
    for index, component in enumerate(value):
        if (
            isinstance(component, bool)
            or not isinstance(component, (int, float))
            or not math.isfinite(component)
        ):
            raise ValueError(f"{name}[{index}] must be a finite number")
        result.append(float(component))
    return tuple(result)  # type: ignore[return-value]


def _rgb(value: Any, *, name: str) -> Rgb:
    color = _finite_vec3(value, name=name)
    if any(component < 0.0 or component > 1.0 for component in color):
        raise ValueError(f"{name} components must be between 0 and 1")
    return color


def _unique_identifiers(values: list[str], *, name: str, maximum: int) -> list[str]:
    if not values:
        raise ValueError(f"{name} must contain at least one entry")
    if len(values) > maximum:
        raise ValueError(f"{name} may contain at most {maximum} entries")
    normalized = [_identifier(value, name=f"{name}[{index}]") for index, value in enumerate(values)]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must not contain duplicates")
    return normalized


def _unique_scene_names(values: list[str]) -> list[str]:
    if not values:
        raise ValueError("target_scenes must contain at least one scene")
    if len(values) > MAX_TARGET_SCENES:
        raise ValueError(f"target_scenes may contain at most {MAX_TARGET_SCENES} entries")
    normalized = [
        _blender_name(value, name=f"target_scenes[{index}]")
        for index, value in enumerate(values)
    ]
    if len(normalized) != len(set(normalized)):
        raise ValueError("target_scenes must not contain duplicates")
    return normalized


def _revision(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 2**63 - 1:
        raise ValueError(f"{name} must be an integer between 0 and {2**63 - 1}")
    return value


def _send_look_profile_command(command: str, params: dict[str, Any]) -> Any:
    """Send one allowlisted look-profile command with strict response handling."""
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


class LightingProfileSpec(BaseModel):
    """Profile-owned lights; spatial placement still uses the relighting workflow."""

    model_config = ConfigDict(extra="forbid")

    existing_light_policy: Literal["KEEP", "MUTE_NON_MANAGED"] = "KEEP"
    lights: list[LightSpec] = Field(default_factory=list, max_length=128)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> LightingProfileSpec:
        ids = [light.id for light in self.lights]
        if len(ids) != len(set(ids)):
            raise ValueError("lighting.lights contains duplicate ids")
        names = [light.name for light in self.lights if light.name is not None]
        if len(names) != len(set(names)):
            raise ValueError("lighting.lights contains duplicate display names")
        return self


class WorldProfileSpec(BaseModel):
    """Typed managed World templates; arbitrary artist node graphs are excluded."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["KEEP", "MANAGED_SOLID", "MANAGED_SKY", "MANAGED_HDRI"] = "KEEP"
    color_rgb: Rgb = (0.0, 0.0, 0.0)
    strength: float = Field(default=0.0, ge=0.0, le=1000.0)
    hdri_path: str | None = Field(default=None, max_length=4096)
    rotation_degrees: float = Field(default=0.0, ge=-360000.0, le=360000.0)
    sun_elevation_degrees: float = Field(default=35.0, ge=-90.0, le=90.0)
    sun_rotation_degrees: float = Field(default=0.0, ge=-360000.0, le=360000.0)
    altitude_m: float = Field(default=0.0, ge=-1000.0, le=100000.0)
    air_density: float = Field(default=1.0, ge=0.0, le=10.0)
    dust_density: float = Field(default=1.0, ge=0.0, le=10.0)
    ozone_density: float = Field(default=1.0, ge=0.0, le=10.0)

    @field_validator("color_rgb", mode="before")
    @classmethod
    def validate_color(cls, value: Any) -> Rgb:
        return _rgb(value, name="world.color_rgb")

    @field_validator("hdri_path")
    @classmethod
    def validate_hdri_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if "\x00" in value or any(ord(character) < 32 for character in value):
            raise ValueError("world.hdri_path contains control characters")
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ValueError("world.hdri_path must be absolute")
        if path.suffix.lower() not in {".hdr", ".exr"}:
            raise ValueError("world.hdri_path must use .hdr or .exr")
        return str(path.resolve())

    @model_validator(mode="after")
    def validate_mode_fields(self) -> WorldProfileSpec:
        if self.mode == "MANAGED_HDRI" and self.hdri_path is None:
            raise ValueError("world.hdri_path is required for MANAGED_HDRI")
        if self.mode != "MANAGED_HDRI" and self.hdri_path is not None:
            raise ValueError("world.hdri_path is only valid for MANAGED_HDRI")
        return self


class AtmosphereComponentSpec(BaseModel):
    """One bounded, profile-owned fog or rain component."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: Literal["FOG_VOLUME", "RAIN_RIG"]
    enabled: bool = True
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    location_world: Vec3 = (0.0, 0.0, 0.0)
    size_xyz: Vec3 = (10.0, 10.0, 10.0)
    color_rgb: Rgb = (1.0, 1.0, 1.0)
    density: float | None = Field(default=None, ge=0.0, le=1000.0)
    anisotropy: float | None = Field(default=None, ge=-1.0, le=1.0)
    rain_rate: float | None = Field(default=None, ge=0.0, le=5_000.0)
    drop_size_m: float | None = Field(default=None, ge=0.00001, le=1.0)
    fall_speed_mps: float | None = Field(default=None, ge=0.0, le=1000.0)
    wind_vector: Vec3 | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _identifier(value, name="atmosphere component id")

    @field_validator("location_world", "size_xyz", "wind_vector", mode="before")
    @classmethod
    def validate_vec3(cls, value: Any, info: Any) -> Any:
        if value is None:
            return None
        result = _finite_vec3(value, name=f"atmosphere.{info.field_name}")
        if info.field_name == "size_xyz" and any(component <= 0.0 for component in result):
            raise ValueError("atmosphere.size_xyz components must be positive")
        return result

    @field_validator("color_rgb", mode="before")
    @classmethod
    def validate_color(cls, value: Any) -> Rgb:
        return _rgb(value, name="atmosphere.color_rgb")

    @model_validator(mode="after")
    def validate_kind_fields(self) -> AtmosphereComponentSpec:
        fog_fields = (self.density, self.anisotropy)
        rain_fields = (self.rain_rate, self.drop_size_m, self.fall_speed_mps, self.wind_vector)
        if self.kind == "FOG_VOLUME":
            if self.density is None:
                raise ValueError("FOG_VOLUME requires density")
            if any(value is not None for value in rain_fields):
                raise ValueError("rain fields are only valid for RAIN_RIG")
        else:
            if self.rain_rate is None or self.drop_size_m is None or self.fall_speed_mps is None:
                raise ValueError(
                    "RAIN_RIG requires rain_rate, drop_size_m, and fall_speed_mps"
                )
            if any(value is not None for value in fog_fields):
                raise ValueError("density and anisotropy are only valid for FOG_VOLUME")
        return self


class BloomSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    threshold: float = Field(default=1.0, ge=0.0, le=1000.0)
    strength: float = Field(default=0.0, ge=0.0, le=100.0)
    radius: float = Field(default=0.5, ge=0.0, le=1.0)


class GrainSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    strength: float = Field(default=0.0, ge=0.0, le=1.0)
    scale: float = Field(default=1.0, ge=0.01, le=1000.0)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)


class VignetteSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    strength: float = Field(default=0.0, ge=0.0, le=1.0)
    feather: float = Field(default=0.5, ge=0.0, le=1.0)


class PostProfileSpec(BaseModel):
    """Parameters for a marked managed compositor stack only."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["KEEP", "MANAGED_STACK"] = "KEEP"
    bloom: BloomSpec = Field(default_factory=BloomSpec)
    grain: GrainSpec = Field(default_factory=GrainSpec)
    vignette: VignetteSpec = Field(default_factory=VignetteSpec)

    @model_validator(mode="after")
    def reject_effects_in_keep_mode(self) -> PostProfileSpec:
        if self.mode == "KEEP" and (
            self.bloom.enabled or self.grain.enabled or self.vignette.enabled
        ):
            raise ValueError("post effects require post.mode='MANAGED_STACK'")
        return self


class ColorManagementSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["KEEP", "MANAGED"] = "KEEP"
    view_transform: str | None = Field(default=None, max_length=128)
    look: str | None = Field(default=None, max_length=128)
    exposure: float | None = Field(default=None, ge=-32.0, le=32.0)
    gamma: float | None = Field(default=None, ge=0.01, le=10.0)

    @field_validator("view_transform", "look")
    @classmethod
    def validate_enum_label(cls, value: str | None, info: Any) -> str | None:
        if value is None:
            return None
        return _bounded_text(value, name=f"color_management.{info.field_name}", maximum=128)

    @model_validator(mode="after")
    def reject_overrides_in_keep_mode(self) -> ColorManagementSpec:
        if self.mode == "KEEP" and any(
            value is not None
            for value in (self.view_transform, self.look, self.exposure, self.gamma)
        ):
            raise ValueError("color-management overrides require mode='MANAGED'")
        return self


class CameraProfileSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["KEEP", "MANAGED_CLONE"] = "KEEP"
    source_camera: str | None = None
    location_world: Vec3 | None = None
    target_point: Vec3 | None = None
    use_dof: bool | None = None
    focus_object: str | None = None
    focus_distance_m: float | None = Field(default=None, ge=0.0001, le=1_000_000.0)
    aperture_fstop: float | None = Field(default=None, ge=0.1, le=1024.0)

    @field_validator("source_camera", "focus_object")
    @classmethod
    def validate_names(cls, value: str | None, info: Any) -> str | None:
        if value is None:
            return None
        return _blender_name(value, name=f"camera.{info.field_name}")

    @field_validator("location_world", "target_point", mode="before")
    @classmethod
    def validate_vec3(cls, value: Any, info: Any) -> Any:
        if value is None:
            return None
        return _finite_vec3(value, name=f"camera.{info.field_name}")

    @model_validator(mode="after")
    def validate_mode_fields(self) -> CameraProfileSpec:
        overrides = (
            self.source_camera,
            self.location_world,
            self.target_point,
            self.use_dof,
            self.focus_object,
            self.focus_distance_m,
            self.aperture_fstop,
        )
        if self.mode == "KEEP" and any(value is not None for value in overrides):
            raise ValueError("camera overrides require camera.mode='MANAGED_CLONE'")
        if self.mode == "MANAGED_CLONE" and self.source_camera is None:
            raise ValueError("MANAGED_CLONE requires source_camera")
        if self.focus_object is not None and self.focus_distance_m is not None:
            raise ValueError("camera accepts focus_object or focus_distance_m, not both")
        if self.location_world is not None and self.target_point is not None:
            delta = tuple(
                target - origin
                for target, origin in zip(self.target_point, self.location_world, strict=True)
            )
            if sum(component * component for component in delta) <= 1e-18:
                raise ValueError("camera.target_point must differ from camera.location_world")
        return self


class RenderPresetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine: Literal[
        "KEEP",
        "CYCLES",
        "BLENDER_EEVEE",
        "BLENDER_EEVEE_NEXT",
    ] = "KEEP"
    samples: int | None = Field(default=None, ge=1, le=100_000)
    denoise: bool | None = None
    resolution_x: int | None = Field(default=None, ge=1, le=8192)
    resolution_y: int | None = Field(default=None, ge=1, le=8192)
    resolution_percentage: int | None = Field(default=None, ge=1, le=100)
    film_transparent: bool | None = None
    use_motion_blur: bool | None = None

    @model_validator(mode="after")
    def reject_overrides_in_keep_mode(self) -> RenderPresetSpec:
        if self.engine == "KEEP" and any(
            value is not None
            for value in (
                self.samples,
                self.denoise,
                self.resolution_x,
                self.resolution_y,
                self.resolution_percentage,
                self.film_transparent,
                self.use_motion_blur,
            )
        ):
            raise ValueError("render overrides require an explicit render engine")
        return self


class ReviewIntentSpec(BaseModel):
    """Persisted artistic intent and explicit renderer expectations."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=2000)
    expects_shadows: bool = True
    expects_reflections: bool = True
    expects_volume: bool = False
    expects_compositing: bool = True

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        return _bounded_text(value, name="review_intent.summary", maximum=2000)


class LookProfileSpec(BaseModel):
    """One deterministic, reusable look manifest compiled by the Blender add-on."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    profile_id: str
    display_name: str
    description: str | None = Field(default=None, max_length=2000)
    status: Literal["DRAFT", "ACCEPTED"] = "DRAFT"
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    generator_version: str = "manual-v1"
    tags: list[str] = Field(default_factory=list, max_length=MAX_TAGS)
    lighting: LightingProfileSpec = Field(default_factory=LightingProfileSpec)
    world: WorldProfileSpec = Field(default_factory=WorldProfileSpec)
    atmosphere: list[AtmosphereComponentSpec] = Field(
        default_factory=list,
        max_length=MAX_ATMOSPHERE_COMPONENTS,
    )
    post: PostProfileSpec = Field(default_factory=PostProfileSpec)
    color_management: ColorManagementSpec = Field(default_factory=ColorManagementSpec)
    camera: CameraProfileSpec = Field(default_factory=CameraProfileSpec)
    render: RenderPresetSpec = Field(default_factory=RenderPresetSpec)
    review_intent: ReviewIntentSpec | None = None

    @field_validator("profile_id")
    @classmethod
    def validate_profile_id(cls, value: str) -> str:
        return _identifier(value, name="profile_id")

    @field_validator("display_name")
    @classmethod
    def validate_display_name(cls, value: str) -> str:
        return _bounded_text(value, name="display_name", maximum=128)

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_text(value, name="description", maximum=2000)

    @field_validator("generator_version")
    @classmethod
    def validate_generator_version(cls, value: str) -> str:
        return _identifier(value, name="generator_version")

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, values: list[str]) -> list[str]:
        normalized = [
            _bounded_text(value, name=f"tags[{index}]", maximum=64)
            for index, value in enumerate(values)
        ]
        if len(normalized) != len(set(normalized)):
            raise ValueError("tags must not contain duplicates")
        return normalized

    @model_validator(mode="after")
    def validate_component_ids(self) -> LookProfileSpec:
        ids = [component.id for component in self.atmosphere]
        if len(ids) != len(set(ids)):
            raise ValueError("atmosphere contains duplicate component ids")
        return self


class FrameSelection(BaseModel):
    """An explicit frame list or inclusive range."""

    model_config = ConfigDict(extra="forbid")

    frames: list[int] | None = Field(default=None, max_length=MAX_RENDER_FRAMES)
    start: int | None = Field(default=None, ge=-1_000_000, le=1_000_000)
    end: int | None = Field(default=None, ge=-1_000_000, le=1_000_000)
    step: int | None = Field(default=None, ge=1, le=1_000_000)

    @field_validator("frames")
    @classmethod
    def validate_frames(cls, values: list[int] | None) -> list[int] | None:
        if values is None:
            return None
        if not values:
            raise ValueError("frames must not be empty")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < -1_000_000
            or value > 1_000_000
            for value in values
        ):
            raise ValueError("frames must contain bounded integers")
        if len(values) != len(set(values)):
            raise ValueError("frames must not contain duplicates")
        return values

    @model_validator(mode="after")
    def validate_selection(self) -> FrameSelection:
        has_frames = self.frames is not None
        has_range = any(value is not None for value in (self.start, self.end, self.step))
        if has_frames == has_range:
            raise ValueError("provide exactly one of frames or start/end/step")
        if has_range:
            if self.start is None or self.end is None or self.step is None:
                raise ValueError("frame ranges require start, end, and step")
            if self.end < self.start:
                raise ValueError("frame range end must be greater than or equal to start")
            if self.count > MAX_RENDER_FRAMES:
                raise ValueError(f"frame range may contain at most {MAX_RENDER_FRAMES} frames")
        return self

    @property
    def count(self) -> int:
        if self.frames is not None:
            return len(self.frames)
        assert self.start is not None and self.end is not None and self.step is not None
        return ((self.end - self.start) // self.step) + 1


class BatchRenderSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    samples: int = Field(default=64, ge=1, le=100_000)
    denoise: bool = True
    resolution_percentage: int = Field(default=100, ge=1, le=100)
    file_format: Literal["PNG", "OPEN_EXR", "JPEG"] = "PNG"
    color_depth: Literal["8", "16", "32"] = "16"
    existing_file_policy: Literal["ERROR", "SKIP"] = "ERROR"

    @model_validator(mode="after")
    def validate_format_depth(self) -> BatchRenderSettings:
        if self.file_format == "JPEG" and self.color_depth != "8":
            raise ValueError("JPEG render batches require color_depth='8'")
        if self.file_format == "PNG" and self.color_depth == "32":
            raise ValueError("PNG render batches support color_depth='8' or '16'")
        if self.file_format == "OPEN_EXR" and self.color_depth == "8":
            raise ValueError("OPEN_EXR render batches require 16- or 32-bit depth")
        return self


class LookRenderBatchSpec(BaseModel):
    """One bounded, auditable batch over explicit profiles and scenes."""

    model_config = ConfigDict(extra="forbid")

    profile_ids: list[str]
    target_scenes: list[str]
    pairing: Literal["CROSS_PRODUCT", "PAIRWISE"] = "PAIRWISE"
    frames: FrameSelection
    output_root: str = Field(max_length=4096)
    filename_template: str = "{scene}-{profile}-{frame}"
    render_settings: BatchRenderSettings = Field(default_factory=BatchRenderSettings)
    continue_on_error: bool = False
    include_review_packet: bool = False
    camera_names: list[str] | None = Field(default=None, max_length=MAX_CAMERAS)

    @field_validator("profile_ids")
    @classmethod
    def validate_profile_ids(cls, values: list[str]) -> list[str]:
        return _unique_identifiers(values, name="profile_ids", maximum=MAX_PROFILES)

    @field_validator("target_scenes")
    @classmethod
    def validate_target_scenes(cls, values: list[str]) -> list[str]:
        return _unique_scene_names(values)

    @field_validator("camera_names")
    @classmethod
    def validate_camera_names(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        if not values:
            raise ValueError("camera_names must be omitted or contain at least one camera")
        normalized = [
            _blender_name(value, name=f"camera_names[{index}]")
            for index, value in enumerate(values)
        ]
        if len(normalized) != len(set(normalized)):
            raise ValueError("camera_names must not contain duplicates")
        return normalized

    @field_validator("output_root")
    @classmethod
    def validate_output_root(cls, value: str) -> str:
        if "\x00" in value or any(ord(character) < 32 for character in value):
            raise ValueError("output_root contains control characters")
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ValueError("output_root must be absolute")
        return str(path.resolve())

    @field_validator("filename_template")
    @classmethod
    def validate_filename_template(cls, value: str) -> str:
        if not _SAFE_TEMPLATE_RE.fullmatch(value):
            raise ValueError(
                "filename_template may use safe filename characters and "
                "{scene}, {profile}, and {frame} placeholders"
            )
        placeholders = {match.group(1) for match in re.finditer(r"\{([^{}]+)\}", value)}
        if not placeholders or not placeholders <= {"scene", "profile", "camera", "frame"}:
            raise ValueError(
                "filename_template must use only {scene}, {profile}, {camera}, and "
                "{frame} placeholders"
            )
        return value

    @model_validator(mode="after")
    def validate_batch_size(self) -> LookRenderBatchSpec:
        if self.pairing == "PAIRWISE" and len(self.profile_ids) != len(self.target_scenes):
            raise ValueError("PAIRWISE batches require equal profile_ids and target_scenes lengths")
        if self.pairing == "CROSS_PRODUCT" and len(self.profile_ids) != 1:
            raise ValueError(
                "CROSS_PRODUCT supports exactly one profile_id because each compiled Scene "
                "owns one profile; use PAIRWISE for multiple profiles"
            )
        combinations = (
            len(self.profile_ids) * len(self.target_scenes)
            if self.pairing == "CROSS_PRODUCT"
            else len(self.profile_ids)
        )
        camera_count = len(self.camera_names) if self.camera_names is not None else 1
        if camera_count > 1 and "{camera}" not in self.filename_template:
            raise ValueError("multi-camera batches require {camera} in filename_template")
        item_count = combinations * camera_count * self.frames.count
        if item_count > MAX_RENDER_ITEMS:
            raise ValueError(
                f"render batch expands to {item_count} items; maximum is {MAX_RENDER_ITEMS}"
            )
        return self


class ReviewTileMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slug: Literal[
        "combined",
        "diffuse_direct",
        "glossy",
        "emission",
        "depth",
        "cryptomatte_object",
    ]
    label: str = Field(min_length=1, max_length=128)
    available: bool
    artifact_sha256: str
    artifact_byte_count: int = Field(ge=1, le=2**63 - 1)
    width: int = Field(ge=1, le=65536)
    height: int = Field(ge=1, le=65536)
    mean_energy: float = Field(ge=0.0)
    nonzero_coverage: float = Field(ge=0.0, le=1.0)
    magenta_coverage: float = Field(ge=0.0, le=1.0)

    @field_validator("artifact_sha256")
    @classmethod
    def validate_artifact_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("artifact_sha256 must be a lowercase SHA-256 digest")
        return value


class ReviewPacketMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    image_source: Literal["SAME_RENDER_RESULT_PASSES"]
    tile_order: list[str] = Field(min_length=6, max_length=6)
    tiles: list[ReviewTileMetadata] = Field(min_length=6, max_length=6)
    technical_check: dict[str, Any]
    review_intent: ReviewIntentSpec
    editable_controls: dict[str, Any]
    beauty_magenta_coverage: float = Field(ge=0.0, le=1.0)
    contact_sheet_width: int | None = Field(default=None, ge=1, le=4096)
    contact_sheet_height: int | None = Field(default=None, ge=1, le=4096)
    contact_sheet_byte_count: int | None = Field(
        default=None, ge=1, le=MAX_PROXY_OUTPUT_BYTES
    )
    contact_sheet_sha256: str | None = None

    @field_validator("contact_sheet_sha256")
    @classmethod
    def validate_contact_hash(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256_RE.fullmatch(value):
            raise ValueError("contact_sheet_sha256 must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def validate_tiles(self) -> ReviewPacketMetadata:
        expected = [
            "combined",
            "diffuse_direct",
            "glossy",
            "emission",
            "depth",
            "cryptomatte_object",
        ]
        if self.tile_order != expected or [tile.slug for tile in self.tiles] != expected:
            raise ValueError("review packet tiles must use the canonical six-tile order")
        return self


class LookRenderResultMetadata(BaseModel):
    """Metadata paired with an optional bounded Composite proxy image."""

    model_config = ConfigDict(extra="forbid")

    batch_id: str
    result_id: str
    status: Literal["QUEUED", "RUNNING", "SUCCEEDED", "SKIPPED", "FAILED", "CANCELLED"]
    profile_id: str
    target_scene: str
    frame: int = Field(ge=-1_000_000, le=1_000_000)
    output_pass: Literal["COMPOSITE"] = "COMPOSITE"
    output_path: str | None = Field(default=None, max_length=4096)
    error: str | None = Field(default=None, max_length=4000)
    image_source: Literal["COMPOSITE_OUTPUT"] | None = None
    artifact_sha256: str | None = None
    artifact_byte_count: int | None = Field(default=None, ge=1, le=2**63 - 1)
    source_width: int | None = Field(default=None, ge=1, le=65536)
    source_height: int | None = Field(default=None, ge=1, le=65536)
    proxy_width: int | None = Field(default=None, ge=1, le=4096)
    proxy_height: int | None = Field(default=None, ge=1, le=4096)
    proxy_format: Literal["JPEG", "PNG"] | None = None
    proxy_byte_count: int | None = Field(default=None, ge=1, le=MAX_PROXY_OUTPUT_BYTES)
    content_sha256: str | None = None
    scene: str | None = None
    view_layer: str | None = None
    camera: str | None = None
    engine: str | None = Field(default=None, max_length=128)
    samples: int | None = Field(default=None, ge=1, le=100_000)
    denoise: bool | None = None
    color_management: dict[str, Any] | None = None
    compositor_hash: str | None = None
    compositor_adapter: Literal[
        "SCENE_COMPOSITING_NODE_GROUP",
        "SCENE_EMBEDDED_NODE_TREE",
    ] | None = None
    source_scene: str | None = None
    profile_version: int | None = Field(default=None, ge=1, le=1_000_000)
    profile_revision: int | None = Field(default=None, ge=1, le=1_000_000)
    profile_status: Literal["DRAFT", "ACCEPTED"] | None = None
    definition_hash: str | None = None
    base_revision: int | None = Field(default=None, ge=0, le=2**63 - 1)
    geometry_revision: int | None = Field(default=None, ge=0, le=2**63 - 1)
    base_fingerprint: str | None = None
    geometry_fingerprint: str | None = None
    manifest_entry_hash: str | None = None
    submitted_at: float | None = Field(default=None, ge=0.0)
    started_at: float | None = Field(default=None, ge=0.0)
    completed_at: float | None = Field(default=None, ge=0.0)
    warnings: list[str] = Field(default_factory=list, max_length=64)
    review_packet: ReviewPacketMetadata | None = None

    @field_validator("batch_id", "result_id", "profile_id")
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _identifier(value, name=info.field_name)

    @field_validator("target_scene", "scene", "view_layer", "camera", "source_scene")
    @classmethod
    def validate_blender_names(cls, value: str | None, info: Any) -> str | None:
        if value is None:
            return None
        return _blender_name(value, name=info.field_name)

    @field_validator(
        "content_sha256",
        "artifact_sha256",
        "compositor_hash",
        "definition_hash",
        "manifest_entry_hash",
    )
    @classmethod
    def validate_hash(cls, value: str | None, info: Any) -> str | None:
        if value is not None and not _SHA256_RE.fullmatch(value):
            raise ValueError(f"{info.field_name} must be a lowercase SHA-256 digest")
        return value

    @field_validator("base_fingerprint", "geometry_fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str | None, info: Any) -> str | None:
        if value is not None and not _FINGERPRINT_RE.fullmatch(value):
            raise ValueError(f"{info.field_name} must be a sha256: fingerprint")
        return value

    @field_validator("output_path")
    @classmethod
    def validate_output_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if "\x00" in value or any(ord(character) < 32 for character in value):
            raise ValueError("output_path contains control characters")
        if not Path(value).expanduser().is_absolute():
            raise ValueError("output_path must be absolute")
        return value

    @field_validator("color_management")
    @classmethod
    def validate_color_management(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        allowed = {"view_transform", "look", "exposure", "gamma"}
        if set(value) - allowed:
            raise ValueError("color_management contains unsupported fields")
        if not {"view_transform", "look", "exposure", "gamma"} <= set(value):
            raise ValueError("color_management is incomplete")
        for name in ("view_transform", "look"):
            _bounded_text(value[name], name=f"color_management.{name}", maximum=128)
        for name in ("exposure", "gamma"):
            component = value[name]
            if (
                isinstance(component, bool)
                or not isinstance(component, (int, float))
                or not math.isfinite(component)
            ):
                raise ValueError(f"color_management.{name} must be finite")
        return value

    @field_validator("warnings")
    @classmethod
    def validate_warnings(cls, values: list[str]) -> list[str]:
        return [
            _bounded_text(value, name=f"warnings[{index}]", maximum=1000)
            for index, value in enumerate(values)
        ]

    @model_validator(mode="after")
    def validate_success_provenance(self) -> LookRenderResultMetadata:
        if self.status == "SKIPPED":
            forbidden = {
                "image_source": self.image_source,
                "artifact_sha256": self.artifact_sha256,
                "artifact_byte_count": self.artifact_byte_count,
                "proxy_width": self.proxy_width,
                "proxy_height": self.proxy_height,
                "proxy_format": self.proxy_format,
                "proxy_byte_count": self.proxy_byte_count,
                "content_sha256": self.content_sha256,
            }
            present = [name for name, value in forbidden.items() if value is not None]
            if present:
                raise ValueError(
                    "Skipped render result must not carry acceptance image evidence: "
                    + ", ".join(present)
                )
            return self
        if self.status != "SUCCEEDED":
            return self
        required = {
            "output_path": self.output_path,
            "image_source": self.image_source,
            "artifact_sha256": self.artifact_sha256,
            "artifact_byte_count": self.artifact_byte_count,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "proxy_width": self.proxy_width,
            "proxy_height": self.proxy_height,
            "proxy_format": self.proxy_format,
            "proxy_byte_count": self.proxy_byte_count,
            "scene": self.scene,
            "view_layer": self.view_layer,
            "camera": self.camera,
            "engine": self.engine,
            "samples": self.samples,
            "color_management": self.color_management,
            "compositor_hash": self.compositor_hash,
            "compositor_adapter": self.compositor_adapter,
            "source_scene": self.source_scene,
            "profile_version": self.profile_version,
            "profile_revision": self.profile_revision,
            "profile_status": self.profile_status,
            "definition_hash": self.definition_hash,
            "base_revision": self.base_revision,
            "geometry_revision": self.geometry_revision,
            "base_fingerprint": self.base_fingerprint,
            "geometry_fingerprint": self.geometry_fingerprint,
            "manifest_entry_hash": self.manifest_entry_hash,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "Successful Composite result lacks provenance fields: "
                + ", ".join(missing)
            )
        if self.error is not None:
            raise ValueError("Successful Composite result must not include error")
        return self


def _validated_result(command: str, params: dict[str, Any]) -> dict[str, Any]:
    result = _send_look_profile_command(command, params)
    if not isinstance(result, dict):
        raise RuntimeError(f"Blender returned invalid {command} data")
    return result


@mcp.tool()
def get_look_profile_context(
    source_scene: str,
    profile_id: str | None = None,
    detail: Literal["SUMMARY", "CAPABILITIES", "OWNERSHIP", "FULL"] = "SUMMARY",
) -> dict[str, Any]:
    """Inspect profile summaries, capabilities, or ownership for a source scene."""
    source_scene = _blender_name(source_scene, name="source_scene")
    if profile_id is not None:
        profile_id = _identifier(profile_id, name="profile_id")
    if detail not in {"SUMMARY", "CAPABILITIES", "OWNERSHIP", "FULL"}:
        raise ValueError("detail must be SUMMARY, CAPABILITIES, OWNERSHIP, or FULL")
    return _validated_result(
        "get_look_profile_context",
        {
            "source_scene": source_scene,
            "profile_id": profile_id,
            "detail": detail,
        },
    )


@mcp.tool()
def inspect_look_review_state(
    profile_id: str,
    target_scene: str,
    camera: str,
    expected_profile_revision: int | None = None,
    expected_geometry_revision: int | None = None,
) -> dict[str, Any]:
    """Audit deterministic cinematic-review failures without mutating Blender."""
    profile_id = _identifier(profile_id, name="profile_id")
    target_scene = _blender_name(target_scene, name="target_scene")
    camera = _blender_name(camera, name="camera")
    if expected_profile_revision is not None:
        expected_profile_revision = _revision(
            expected_profile_revision, name="expected_profile_revision"
        )
    if expected_geometry_revision is not None:
        expected_geometry_revision = _revision(
            expected_geometry_revision, name="expected_geometry_revision"
        )
    return _validated_result(
        "inspect_look_review_state",
        {
            "profile_id": profile_id,
            "target_scene": target_scene,
            "camera": camera,
            "expected_profile_revision": expected_profile_revision,
            "expected_geometry_revision": expected_geometry_revision,
        },
    )


@mcp.tool()
def upsert_look_profile(
    action: Literal["VALIDATE", "COMPILE"],
    source_scene: str,
    profile: LookProfileSpec,
    update_mode: Literal["CREATE_VERSION", "REPLACE_DRAFT"],
    expected_base_revision: int | None = None,
    expected_profile_revision: int | None = None,
    expected_geometry_revision: int | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Validate or compile one dormant, managed look profile without activating it."""
    if action not in {"VALIDATE", "COMPILE"}:
        raise ValueError("action must be VALIDATE or COMPILE")
    source_scene = _blender_name(source_scene, name="source_scene")
    if not isinstance(profile, LookProfileSpec):
        profile = LookProfileSpec.model_validate(profile)
    if update_mode not in {"CREATE_VERSION", "REPLACE_DRAFT"}:
        raise ValueError("update_mode must be CREATE_VERSION or REPLACE_DRAFT")
    if expected_base_revision is not None:
        expected_base_revision = _revision(
            expected_base_revision,
            name="expected_base_revision",
        )
    if expected_profile_revision is not None:
        expected_profile_revision = _revision(
            expected_profile_revision,
            name="expected_profile_revision",
        )
    if expected_geometry_revision is not None:
        expected_geometry_revision = _revision(
            expected_geometry_revision,
            name="expected_geometry_revision",
        )
    if not isinstance(strict, bool):
        raise ValueError("strict must be a boolean")
    return _validated_result(
        "upsert_look_profile",
        {
            "action": action,
            "source_scene": source_scene,
            "profile": profile.model_dump(mode="json", exclude_none=True),
            "update_mode": update_mode,
            "expected_base_revision": expected_base_revision,
            "expected_profile_revision": expected_profile_revision,
            "expected_geometry_revision": expected_geometry_revision,
            "strict": strict,
        },
    )


@mcp.tool()
def activate_look_profile(
    profile_id: str,
    target_scene: str,
    window_index: int | None = None,
    expected_profile_revision: int | None = None,
    expected_geometry_revision: int | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Activate one compiled profile in a target Scene/window context."""
    profile_id = _identifier(profile_id, name="profile_id")
    target_scene = _blender_name(target_scene, name="target_scene")
    if window_index is not None and (
        isinstance(window_index, bool)
        or not isinstance(window_index, int)
        or not 0 <= window_index <= 128
    ):
        raise ValueError("window_index must be an integer between 0 and 128")
    if expected_profile_revision is not None:
        expected_profile_revision = _revision(
            expected_profile_revision,
            name="expected_profile_revision",
        )
    if expected_geometry_revision is not None:
        expected_geometry_revision = _revision(
            expected_geometry_revision,
            name="expected_geometry_revision",
        )
    if not isinstance(strict, bool):
        raise ValueError("strict must be a boolean")
    return _validated_result(
        "activate_look_profile",
        {
            "profile_id": profile_id,
            "target_scene": target_scene,
            "window_index": window_index,
            "expected_profile_revision": expected_profile_revision,
            "expected_geometry_revision": expected_geometry_revision,
            "strict": strict,
        },
    )


@mcp.tool()
def accept_look_profile(
    profile_id: str,
    target_scene: str,
    batch_id: str,
    result_id: str,
    artifact_sha256: str,
    review_acknowledged: bool,
    expected_profile_revision: int | None = None,
    expected_geometry_revision: int | None = None,
) -> dict[str, Any]:
    """Accept a DRAFT from one reviewed, full-preset, current Composite result."""
    profile_id = _identifier(profile_id, name="profile_id")
    target_scene = _blender_name(target_scene, name="target_scene")
    batch_id = _identifier(batch_id, name="batch_id")
    result_id = _identifier(result_id, name="result_id")
    if not isinstance(artifact_sha256, str) or not _SHA256_RE.fullmatch(
        artifact_sha256
    ):
        raise ValueError("artifact_sha256 must be a lowercase SHA-256 digest")
    if review_acknowledged is not True:
        raise ValueError("review_acknowledged must be true after visual Composite review")
    if expected_profile_revision is not None:
        expected_profile_revision = _revision(
            expected_profile_revision,
            name="expected_profile_revision",
        )
    if expected_geometry_revision is not None:
        expected_geometry_revision = _revision(
            expected_geometry_revision,
            name="expected_geometry_revision",
        )
    return _validated_result(
        "accept_look_profile",
        {
            "profile_id": profile_id,
            "target_scene": target_scene,
            "batch_id": batch_id,
            "result_id": result_id,
            "artifact_sha256": artifact_sha256,
            "review_acknowledged": True,
            "expected_profile_revision": expected_profile_revision,
            "expected_geometry_revision": expected_geometry_revision,
        },
    )


@mcp.tool()
def submit_look_render_batch(
    batch: LookRenderBatchSpec,
    expected_profile_revision: int | None = None,
    expected_profile_revisions: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Submit a bounded render batch over explicit profile and scene targets."""
    if not isinstance(batch, LookRenderBatchSpec):
        batch = LookRenderBatchSpec.model_validate(batch)
    if expected_profile_revision is not None:
        expected_profile_revision = _revision(
            expected_profile_revision,
            name="expected_profile_revision",
        )
        pair_count = (
            len(batch.profile_ids) * len(batch.target_scenes)
            if batch.pairing == "CROSS_PRODUCT"
            else len(batch.profile_ids)
        )
        if pair_count != 1:
            raise ValueError(
                "Scalar expected_profile_revision is valid only for one profile/Scene "
                "pair; use expected_profile_revisions keyed by target Scene"
            )
    if expected_profile_revisions is not None:
        if expected_profile_revision is not None:
            raise ValueError(
                "Provide expected_profile_revision or expected_profile_revisions, not both"
            )
        normalized_revisions: dict[str, int] = {}
        for scene, revision in expected_profile_revisions.items():
            scene_name = _blender_name(scene, name="expected_profile_revisions key")
            normalized_revisions[scene_name] = _revision(
                revision,
                name=f"expected_profile_revisions[{scene_name!r}]",
            )
        if set(normalized_revisions) != set(batch.target_scenes):
            raise ValueError(
                "expected_profile_revisions must contain exactly one entry for every "
                "target Scene"
            )
        expected_profile_revisions = normalized_revisions
    return _validated_result(
        "submit_look_render_batch",
        {
            "batch": batch.model_dump(mode="json", exclude_none=True),
            "expected_profile_revision": expected_profile_revision,
            "expected_profile_revisions": expected_profile_revisions,
            "output_pass": "COMPOSITE",
            "restore_scene_state": True,
            "save_blend": False,
        },
    )


@mcp.tool()
def get_look_render_batch(
    batch_id: str,
    include_results: bool = False,
    max_results: int = 100,
    cursor: str | None = None,
) -> dict[str, Any]:
    """Return bounded status and optionally paginated results for a render batch."""
    batch_id = _identifier(batch_id, name="batch_id")
    if not isinstance(include_results, bool):
        raise ValueError("include_results must be a boolean")
    if isinstance(max_results, bool) or not isinstance(max_results, int) or not 1 <= max_results <= 1000:
        raise ValueError("max_results must be an integer between 1 and 1000")
    if cursor is not None:
        if not isinstance(cursor, str) or len(cursor) > 1024 or any(
            ord(character) < 32 for character in cursor
        ):
            raise ValueError("cursor must be an opaque string of at most 1024 characters")
    return _validated_result(
        "get_look_render_batch",
        {
            "batch_id": batch_id,
            "include_results": include_results,
            "max_results": max_results,
            "cursor": cursor,
        },
    )


@mcp.tool()
def cancel_look_render_batch(batch_id: str) -> dict[str, Any]:
    """Request cancellation of a render batch; handlers must still restore scene state."""
    return _validated_result(
        "cancel_look_render_batch",
        {"batch_id": _identifier(batch_id, name="batch_id")},
    )


def _decode_composite_proxy(value: Any) -> bytes:
    if not isinstance(value, str) or not value:
        raise RuntimeError("Successful Composite result did not return image_base64")
    if len(value) > ((MAX_PROXY_INPUT_BYTES + 2) // 3) * 4 + 4:
        raise RuntimeError("Composite proxy exceeds the input byte limit")
    try:
        image_bytes = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError("Composite proxy contains invalid base64") from exc
    if not image_bytes or len(image_bytes) > MAX_PROXY_INPUT_BYTES:
        raise RuntimeError("Composite proxy exceeds the input byte limit")
    return image_bytes


def _encode_composite_proxy(
    image_bytes: bytes,
    *,
    max_size: int,
    output_format: Literal["JPEG", "PNG"],
    jpeg_quality: int,
) -> tuple[bytes, int, int, int, int]:
    try:
        with PILImage.open(BytesIO(image_bytes)) as source:
            source_width, source_height = source.size
            if source_width * source_height > MAX_PROXY_PIXELS:
                raise RuntimeError("Composite proxy exceeds the pixel limit")
            source.load()
            image = source.copy()
    except RuntimeError:
        raise
    except (PILImage.DecompressionBombError, UnidentifiedImageError, OSError, ValueError) as exc:
        raise RuntimeError("Composite result did not contain a supported image") from exc

    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), resample=PILImage.Resampling.LANCZOS)
    output = BytesIO()
    if output_format == "JPEG":
        if image.mode in {"RGBA", "LA"} or (
            image.mode == "P" and "transparency" in image.info
        ):
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
    proxy = output.getvalue()
    if not proxy or len(proxy) > MAX_PROXY_OUTPUT_BYTES:
        raise RuntimeError("Encoded Composite proxy exceeds the output byte limit")
    return proxy, source_width, source_height, image.width, image.height


def _decode_review_tiles(value: Any, metadata: ReviewPacketMetadata) -> list[bytes]:
    if not isinstance(value, list) or len(value) != 6:
        raise RuntimeError("Successful review result did not return six review tiles")
    result: list[bytes] = []
    for index, expected in enumerate(metadata.tiles):
        item = value[index]
        if not isinstance(item, dict) or item.get("slug") != expected.slug:
            raise RuntimeError("Review tiles are not in the canonical order")
        encoded = item.get("image_base64")
        if not isinstance(encoded, str) or not encoded:
            raise RuntimeError(f"Review tile '{expected.slug}' lacks image_base64")
        if len(encoded) > ((MAX_PROXY_INPUT_BYTES + 2) // 3) * 4 + 4:
            raise RuntimeError(f"Review tile '{expected.slug}' exceeds the input limit")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise RuntimeError(f"Review tile '{expected.slug}' has invalid base64") from exc
        if not data or len(data) > MAX_PROXY_INPUT_BYTES:
            raise RuntimeError(f"Review tile '{expected.slug}' exceeds the input limit")
        if len(data) != expected.artifact_byte_count:
            raise RuntimeError(f"Review tile '{expected.slug}' byte count does not match")
        if hashlib.sha256(data).hexdigest() != expected.artifact_sha256:
            raise RuntimeError(f"Review tile '{expected.slug}' checksum does not match")
        result.append(data)
    return result


def _review_contact_sheet(
    tiles: list[bytes], metadata: ReviewPacketMetadata
) -> tuple[bytes, int, int]:
    cell_width = 512
    image_height = 288
    label_height = 32
    cell_height = image_height + label_height
    sheet = PILImage.new("RGB", (cell_width * 3, cell_height * 2), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for index, (data, tile_metadata) in enumerate(zip(tiles, metadata.tiles, strict=True)):
        try:
            with PILImage.open(BytesIO(data)) as source:
                source.load()
                tile = ImageOps.fit(
                    source.convert("RGB"),
                    (cell_width, image_height),
                    method=PILImage.Resampling.LANCZOS,
                )
        except (PILImage.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(
                f"Review tile '{tile_metadata.slug}' is not a supported image"
            ) from exc
        column = index % 3
        row = index // 3
        x = column * cell_width
        y = row * cell_height
        sheet.paste(tile, (x, y + label_height))
        draw.rectangle((x, y, x + cell_width - 1, y + label_height - 1), fill=(20, 20, 20))
        label = tile_metadata.label
        if not tile_metadata.available:
            label += " — UNAVAILABLE"
        draw.text((x + 10, y + 9), label, fill=(255, 255, 255))
        draw.rectangle(
            (x, y, x + cell_width - 1, y + cell_height - 1),
            outline=(90, 90, 90),
            width=1,
        )
    output = BytesIO()
    sheet.save(output, format="PNG", optimize=True)
    encoded = output.getvalue()
    if not encoded or len(encoded) > MAX_PROXY_OUTPUT_BYTES:
        raise RuntimeError("Diagnostic contact sheet exceeds the output byte limit")
    return encoded, sheet.width, sheet.height


@mcp.tool()
def get_look_render_result(
    batch_id: str,
    result_id: str,
    max_size: int = 1024,
    format: Literal["JPEG", "PNG"] = "JPEG",
    jpeg_quality: int = 85,
) -> Annotated[CallToolResult, LookRenderResultMetadata]:
    """Return one final-Composite result as metadata and a bounded native MCP proxy."""
    batch_id = _identifier(batch_id, name="batch_id")
    result_id = _identifier(result_id, name="result_id")
    if isinstance(max_size, bool) or not isinstance(max_size, int) or not 64 <= max_size <= 4096:
        raise ValueError("max_size must be an integer between 64 and 4096")
    if format not in {"JPEG", "PNG"}:
        raise ValueError("format must be JPEG or PNG")
    if (
        isinstance(jpeg_quality, bool)
        or not isinstance(jpeg_quality, int)
        or not 1 <= jpeg_quality <= 100
    ):
        raise ValueError("jpeg_quality must be an integer between 1 and 100")

    result = _validated_result(
        "get_look_render_result",
        {
            "batch_id": batch_id,
            "result_id": result_id,
            "output_pass": "COMPOSITE",
            "proxy_max_size": max_size,
        },
    )
    raw_metadata = result.get("metadata")
    if not isinstance(raw_metadata, dict):
        raise RuntimeError("Blender render result lacks metadata")
    metadata = LookRenderResultMetadata.model_validate(raw_metadata)
    if metadata.batch_id != batch_id or metadata.result_id != result_id:
        raise RuntimeError("Blender returned mismatched render result identifiers")

    content: list[Any] = []
    encoded: bytes | None = None
    if metadata.status == "SUCCEEDED":
        source = _decode_composite_proxy(result.get("image_base64"))
        encoded, input_width, input_height, proxy_width, proxy_height = (
            _encode_composite_proxy(
                source,
                max_size=max_size,
                output_format=format,
                jpeg_quality=jpeg_quality,
            )
        )
        if metadata.proxy_width is not None and metadata.proxy_width != input_width:
            raise RuntimeError("Composite proxy width does not match returned image bytes")
        if metadata.proxy_height is not None and metadata.proxy_height != input_height:
            raise RuntimeError("Composite proxy height does not match returned image bytes")
        if metadata.proxy_byte_count is not None and metadata.proxy_byte_count != len(source):
            raise RuntimeError("Composite proxy byte count does not match returned image bytes")
        metadata = metadata.model_copy(
            update={
                "proxy_width": proxy_width,
                "proxy_height": proxy_height,
                "proxy_format": format,
                "proxy_byte_count": len(encoded),
                "content_sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
        content.append(MCPImage(data=encoded, format=format.lower()).to_image_content())
        if metadata.review_packet is not None:
            tiles = _decode_review_tiles(result.get("review_tiles"), metadata.review_packet)
            contact_sheet, sheet_width, sheet_height = _review_contact_sheet(
                tiles, metadata.review_packet
            )
            packet = metadata.review_packet.model_copy(
                update={
                    "contact_sheet_width": sheet_width,
                    "contact_sheet_height": sheet_height,
                    "contact_sheet_byte_count": len(contact_sheet),
                    "contact_sheet_sha256": hashlib.sha256(contact_sheet).hexdigest(),
                }
            )
            metadata = metadata.model_copy(update={"review_packet": packet})
            content.append(MCPImage(data=contact_sheet, format="png").to_image_content())
        elif result.get("review_tiles") is not None:
            raise RuntimeError("Blender returned review tiles without review metadata")
    elif result.get("image_base64") is not None:
        raise RuntimeError("Non-successful render results must not include a Composite image")
    elif result.get("review_tiles") is not None:
        raise RuntimeError("Non-successful render results must not include review tiles")

    content.append(TextContent(type="text", text=metadata.model_dump_json()))
    return CallToolResult(
        content=content,
        structuredContent=metadata.model_dump(mode="json"),
    )
