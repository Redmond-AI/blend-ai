"""Persistent, scene-isolated look-profile compilation.

This module is the Blender-side core for reusable lighting/rendering looks.  It
Three allowlisted commands form the add-on boundary:

``get_look_profile_context``
    Read scene/profile state without creating datablocks or a manifest.

``upsert_look_profile``
    Validate a deterministic LINK_COPY compilation plan, or compile that plan
    atomically into a new Scene with a unique profile Collection and World.

``activate_look_profile``
    Select one exact manifest-owned compiled Scene in one Blender window,
    without saving the file or changing any other window.

Profile definitions are persisted as JSON in a marked Blender Text datablock.
Compiled revisions are immutable.  ``REPLACE_DRAFT`` therefore creates a new
revision of the same profile version and marks the previous draft manifest
entry as superseded instead of destructively editing its Blender datablocks.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from typing import Any

import bpy

from .. import dispatcher, look_compositor, look_profile_assets, spatial_cache


MANIFEST_TEXT = "AI_LOOK_PROFILES"
MANIFEST_PROP = "blend_ai_look_profile_manifest"
MANIFEST_SCHEMA_VERSION = 1
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_PROFILE_ENTRIES = 512

MANAGED_PROP = "blend_ai_look_managed"
PROFILE_ID_PROP = "blend_ai_look_profile_id"
PROFILE_VERSION_PROP = "blend_ai_look_profile_version"
PROFILE_REVISION_PROP = "blend_ai_look_profile_revision"
SOURCE_SCENE_PROP = "blend_ai_look_source_scene"
ROLE_PROP = "blend_ai_look_role"
SCHEMA_PROP = "blend_ai_look_schema_version"
COMPILE_MODE_PROP = "blend_ai_look_compile_mode"
DEFINITION_HASH_PROP = "blend_ai_look_definition_hash"
BASE_FINGERPRINT_PROP = "blend_ai_look_base_fingerprint"
GEOMETRY_FINGERPRINT_PROP = "blend_ai_look_geometry_fingerprint"

COMPILE_MODE = "LINK_COPY"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SAFE_NAME = re.compile(r"^[^\x00-\x1f\x7f]{1,128}$")
_ALLOWED_ENGINES = {
    "CYCLES",
    "BLENDER_EEVEE",
    "BLENDER_EEVEE_NEXT",
    "BLENDER_WORKBENCH",
}
_PROFILE_FIELDS = {
    "schema_version",
    "profile_id",
    "display_name",
    "description",
    "status",
    "seed",
    "generator_version",
    "tags",
    "lighting",
    "world",
    "atmosphere",
    "post",
    "color_management",
    "camera",
    "render",
    "review_intent",
}


def _iter_values(collection: Any) -> list[Any]:
    if collection is None:
        return []
    if isinstance(collection, dict):
        return list(collection.values())
    try:
        return list(collection)
    except (TypeError, ReferenceError):
        return []


def _lookup(collection: Any, name: str) -> Any | None:
    getter = getattr(collection, "get", None)
    if callable(getter):
        try:
            return getter(name)
        except (TypeError, ReferenceError):
            pass
    for value in _iter_values(collection):
        if str(getattr(value, "name", "")) == name:
            return value
    return None


def _custom_get(value: Any, key: str, default: Any = None) -> Any:
    getter = getattr(value, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except (TypeError, ReferenceError):
            return default
    return default


def _custom_set(value: Any, key: str, item: Any) -> None:
    try:
        value[key] = item
    except Exception as exc:
        raise RuntimeError(f"Unable to set look-profile marker '{key}': {exc}") from exc


def _plain(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False, separators=(",", ":")))


def _reject_unknown(value: dict[str, Any], allowed: set[str], field: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{field} contains unsupported fields: {unknown}")


def _safe_name(value: Any, field: str, *, maximum: int = 128) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    value = value.strip()
    if not value or len(value) > maximum or not _SAFE_NAME.fullmatch(value):
        raise ValueError(f"{field} must be a safe non-empty name of at most {maximum} characters")
    return value


def _identifier(value: Any, field: str = "profile_id") -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError(
            f"{field} must start with a letter or number and contain at most 128 "
            "letters, numbers, underscores, hyphens, dots, or colons"
        )
    return value


def _integer(value: Any, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{field} must be between {minimum} and {maximum}")
    return value


def _number(value: Any, field: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if result < minimum or result > maximum:
        raise ValueError(f"{field} must be between {minimum} and {maximum}")
    return result


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _color(value: Any, field: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field} must contain exactly three RGB components")
    return [
        _number(component, f"{field}[{index}]", 0.0, 1.0) for index, component in enumerate(value)
    ]


def _text_as_string(text: Any) -> str | None:
    method = getattr(text, "as_string", None)
    if not callable(method):
        return None
    try:
        result = method()
    except Exception:
        return None
    return result if isinstance(result, str) else None


def _empty_manifest() -> dict[str, Any]:
    return {"schema_version": MANIFEST_SCHEMA_VERSION, "profiles": []}


def _manifest_text() -> Any | None:
    texts = getattr(getattr(bpy, "data", None), "texts", None)
    text = _lookup(texts, MANIFEST_TEXT)
    if text is not None and not bool(_custom_get(text, MANIFEST_PROP, False)):
        raise ValueError(
            f"Text datablock '{MANIFEST_TEXT}' exists but is not marked as the "
            "blend-ai look-profile manifest"
        )
    return text


def _validate_manifest(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Look-profile manifest must contain a JSON object")
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported look-profile manifest schema version: {payload.get('schema_version')!r}"
        )
    if set(payload) != {"schema_version", "profiles"}:
        raise ValueError("Look-profile manifest contains unsupported top-level fields")
    profiles = payload.get("profiles")
    if not isinstance(profiles, list):
        raise ValueError("Look-profile manifest profiles must be an array")
    if len(profiles) > MAX_PROFILE_ENTRIES:
        raise ValueError(f"Look-profile manifest contains more than {MAX_PROFILE_ENTRIES} entries")
    required_fields = {
        "profile_id",
        "display_name",
        "source_scene",
        "version",
        "revision",
        "status",
        "compile_mode",
        "scene_name",
        "collection_name",
        "world_name",
        "compositor_name",
        "compositor_hash",
        "compositor_adapter",
        "definition_hash",
        "base_revision",
        "geometry_revision",
        "base_fingerprint",
        "geometry_fingerprint",
        "definition",
        "managed_inventory",
    }
    allowed_fields = required_fields | {"acceptance"}
    identity_claims: set[tuple[str, str, int, int]] = set()
    artifact_claims: dict[str, set[str]] = {
        "scene_name": set(),
        "collection_name": set(),
        "world_name": set(),
        "compositor_name": set(),
    }
    raw_hash = re.compile(r"^[0-9a-f]{64}$")
    prefixed_hash = re.compile(r"^sha256:[0-9a-f]{64}$")
    for index, entry in enumerate(profiles):
        if not isinstance(entry, dict):
            raise ValueError(f"Look-profile manifest profiles[{index}] must be an object")
        missing = sorted(required_fields - set(entry))
        if missing:
            raise ValueError(f"Look-profile manifest profiles[{index}] is missing fields {missing}")
        unknown = sorted(set(entry) - allowed_fields)
        if unknown:
            raise ValueError(
                f"Look-profile manifest profiles[{index}] has unsupported fields {unknown}"
            )
        profile_id = _identifier(entry["profile_id"], f"profiles[{index}].profile_id")
        display_name = _bounded_text(entry["display_name"], f"profiles[{index}].display_name", 128)
        source_scene = _safe_name(
            entry["source_scene"], f"profiles[{index}].source_scene", maximum=63
        )
        version = _integer(entry["version"], f"profiles[{index}].version", 1, 2**63 - 1)
        revision = _integer(entry["revision"], f"profiles[{index}].revision", 1, 2**63 - 1)
        if entry["status"] not in {"DRAFT", "ACCEPTED", "SUPERSEDED"}:
            raise ValueError(f"Look-profile manifest profiles[{index}] has invalid status")
        if entry["compile_mode"] != COMPILE_MODE:
            raise ValueError(f"Look-profile manifest profiles[{index}] has invalid compile_mode")
        for field in artifact_claims:
            name = _safe_name(entry[field], f"profiles[{index}].{field}", maximum=63)
            if name in artifact_claims[field]:
                raise ValueError(f"Look-profile manifest has duplicate {field} claim '{name}'")
            artifact_claims[field].add(name)
        if entry["compositor_adapter"] not in {
            "SCENE_COMPOSITING_NODE_GROUP",
            "SCENE_EMBEDDED_NODE_TREE",
        }:
            raise ValueError(
                f"Look-profile manifest profiles[{index}] has invalid compositor_adapter"
            )
        for field in ("definition_hash", "compositor_hash"):
            if not isinstance(entry[field], str) or not raw_hash.fullmatch(entry[field]):
                raise ValueError(f"Look-profile manifest profiles[{index}].{field} is invalid")
        for field in ("base_fingerprint", "geometry_fingerprint"):
            if not isinstance(entry[field], str) or not prefixed_hash.fullmatch(entry[field]):
                raise ValueError(f"Look-profile manifest profiles[{index}].{field} is invalid")
        _integer(entry["base_revision"], f"profiles[{index}].base_revision", 0, 2**63 - 1)
        _integer(
            entry["geometry_revision"],
            f"profiles[{index}].geometry_revision",
            0,
            2**63 - 1,
        )
        if not isinstance(entry["definition"], dict) or not isinstance(
            entry["managed_inventory"], dict
        ):
            raise ValueError(
                f"Look-profile manifest profiles[{index}] definition/inventory must be objects"
            )
        if _definition_hash(entry["definition"]) != entry["definition_hash"]:
            raise ValueError(
                f"Look-profile manifest profiles[{index}] definition hash is inconsistent"
            )
        identity = (source_scene, profile_id, version, revision)
        if identity in identity_claims:
            raise ValueError("Look-profile manifest contains a duplicate profile revision")
        identity_claims.add(identity)
        if entry["status"] == "ACCEPTED" and not isinstance(entry.get("acceptance"), dict):
            raise ValueError(
                f"Look-profile manifest profiles[{index}] ACCEPTED entry lacks evidence"
            )
        if entry["status"] != "ACCEPTED" and "acceptance" in entry:
            raise ValueError(
                f"Look-profile manifest profiles[{index}] non-ACCEPTED entry has evidence"
            )
        if entry["status"] == "ACCEPTED":
            acceptance = entry["acceptance"]
            acceptance_fields = {
                "schema_version",
                "batch_id",
                "result_id",
                "frame",
                "artifact_path",
                "artifact_sha256",
                "artifact_byte_count",
                "source_width",
                "source_height",
                "engine",
                "samples",
                "denoise",
                "completed_at",
                "accepted_at",
                "review_acknowledged",
            }
            if set(acceptance) != acceptance_fields or acceptance.get("schema_version") != 1:
                raise ValueError(
                    f"Look-profile manifest profiles[{index}] acceptance schema is invalid"
                )
            _identifier(acceptance["batch_id"], f"profiles[{index}].acceptance.batch_id")
            _identifier(acceptance["result_id"], f"profiles[{index}].acceptance.result_id")
            _integer(
                acceptance["frame"],
                f"profiles[{index}].acceptance.frame",
                -1_000_000,
                1_000_000,
            )
            artifact_path = acceptance["artifact_path"]
            if (
                not isinstance(artifact_path, str)
                or not artifact_path.startswith("/")
                or "\x00" in artifact_path
                or any(ord(character) < 32 for character in artifact_path)
            ):
                raise ValueError(
                    f"Look-profile manifest profiles[{index}] acceptance path is invalid"
                )
            if not isinstance(acceptance["artifact_sha256"], str) or not raw_hash.fullmatch(
                acceptance["artifact_sha256"]
            ):
                raise ValueError(
                    f"Look-profile manifest profiles[{index}] acceptance hash is invalid"
                )
            for field, maximum in (
                ("artifact_byte_count", 2**63 - 1),
                ("source_width", 65536),
                ("source_height", 65536),
                ("samples", 100_000),
            ):
                _integer(
                    acceptance[field],
                    f"profiles[{index}].acceptance.{field}",
                    1,
                    maximum,
                )
            _bounded_text(acceptance["engine"], f"profiles[{index}].acceptance.engine", 128)
            _boolean(acceptance["denoise"], f"profiles[{index}].acceptance.denoise")
            for field in ("completed_at", "accepted_at"):
                _number(
                    acceptance[field],
                    f"profiles[{index}].acceptance.{field}",
                    0.0,
                    1.0e20,
                )
            if acceptance["review_acknowledged"] is not True:
                raise ValueError(f"Look-profile manifest profiles[{index}] acceptance lacks review")
        # Force detached, normalized primitive values through validation.
        _ = display_name
    return _plain(payload)


def _load_manifest() -> dict[str, Any]:
    text = _manifest_text()
    if text is None:
        return _empty_manifest()
    raw = _text_as_string(text)
    if not raw:
        raise ValueError(f"Look-profile manifest '{MANIFEST_TEXT}' is empty")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Look-profile manifest is not valid JSON: {exc}") from exc
    return _validate_manifest(payload)


def _persist_manifest(payload: dict[str, Any]) -> None:
    payload = _validate_manifest(payload)
    raw = json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
    if len(raw.encode("utf-8")) > MAX_MANIFEST_BYTES:
        raise ValueError(f"Look-profile manifest exceeds the {MAX_MANIFEST_BYTES}-byte limit")

    texts = getattr(getattr(bpy, "data", None), "texts", None)
    if texts is None:
        raise RuntimeError("Blender Text datablocks are unavailable")
    text = _manifest_text()
    created = False
    previous = _text_as_string(text) if text is not None else None
    try:
        if text is None:
            creator = getattr(texts, "new", None)
            if not callable(creator):
                raise RuntimeError("Blender Text datablocks cannot be created")
            text = creator(MANIFEST_TEXT)
            created = True
            if str(getattr(text, "name", "")) != MANIFEST_TEXT:
                raise RuntimeError(f"Unable to create dedicated Text datablock '{MANIFEST_TEXT}'")
            _custom_set(text, MANIFEST_PROP, True)
        text.clear()
        text.write(raw)
    except Exception as exc:
        if text is not None:
            if created:
                remover = getattr(texts, "remove", None)
                if callable(remover):
                    try:
                        remover(text)
                    except Exception:
                        pass
            elif previous is not None:
                try:
                    text.clear()
                    text.write(previous)
                except Exception:
                    pass
        raise RuntimeError(f"Unable to persist look-profile manifest: {exc}") from exc


def _scene(
    value: Any,
    field: str = "source_scene",
    *,
    require_writable: bool = True,
) -> Any:
    name = _safe_name(value, field, maximum=63)
    scene = _lookup(getattr(getattr(bpy, "data", None), "scenes", None), name)
    if scene is None:
        raise ValueError(f"Scene '{name}' was not found")
    if require_writable and getattr(scene, "library", None) is not None:
        raise ValueError(f"Scene '{name}' is linked/read-only and cannot be compiled")
    return scene


def _value(target: Any, attribute: str, default: Any) -> Any:
    result = getattr(target, attribute, default)
    return default if result is None else result


def _source_render_settings(scene: Any) -> dict[str, Any]:
    render = getattr(scene, "render", None)
    cycles = getattr(scene, "cycles", None)
    eevee = getattr(scene, "eevee", None)
    engine = str(_value(render, "engine", "CYCLES"))
    samples = int(_value(cycles, "samples", 64))
    if engine.startswith("BLENDER_EEVEE") and eevee is not None:
        samples = int(_value(eevee, "taa_render_samples", samples))
    return {
        "engine": engine,
        "resolution_x": int(_value(render, "resolution_x", 1920)),
        "resolution_y": int(_value(render, "resolution_y", 1080)),
        "resolution_percentage": int(_value(render, "resolution_percentage", 100)),
        "samples": samples,
        "use_denoising": bool(_value(cycles, "use_denoising", True)),
        "film_transparent": bool(_value(render, "film_transparent", False)),
        "use_motion_blur": bool(_value(render, "use_motion_blur", False)),
    }


def _blender_version() -> dict[str, Any]:
    app = getattr(bpy, "app", None)
    raw = getattr(app, "version", ())
    try:
        version = [int(value) for value in tuple(raw)[:3]]
    except (TypeError, ValueError):
        version = []
    while len(version) < 3:
        version.append(0)
    version_string = str(getattr(app, "version_string", ""))
    if not version_string and any(version):
        version_string = ".".join(str(value) for value in version)
    return {"version": version, "version_string": version_string or "unknown"}


def _resolve_engine(engine: str) -> str:
    """Resolve Blender's 4.2/5.2 Eevee identifier seam deterministically."""
    if engine not in {"BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"}:
        return engine
    major = _blender_version()["version"][0]
    if major >= 5:
        return "BLENDER_EEVEE"
    if major == 4:
        return "BLENDER_EEVEE_NEXT"
    # Tests and older add-on hosts may not expose bpy.app.  Preserve the
    # source/request spelling in that case rather than guessing.
    return engine


def _normalize_render(value: Any, scene: Any) -> dict[str, Any]:
    result = _source_render_settings(scene)
    if value is None:
        return result
    if not isinstance(value, dict):
        raise ValueError("render_settings must be an object or null")
    _reject_unknown(
        value,
        {
            "engine",
            "resolution_x",
            "resolution_y",
            "resolution_percentage",
            "samples",
            "use_denoising",
            "film_transparent",
            "use_motion_blur",
        },
        "render_settings",
    )
    if "engine" in value:
        engine = str(value["engine"]).upper()
        if engine not in _ALLOWED_ENGINES:
            raise ValueError(f"render_settings.engine must be one of {sorted(_ALLOWED_ENGINES)}")
        result["engine"] = _resolve_engine(engine)
    if "resolution_x" in value:
        result["resolution_x"] = _integer(value["resolution_x"], "resolution_x", 1, 8192)
    if "resolution_y" in value:
        result["resolution_y"] = _integer(value["resolution_y"], "resolution_y", 1, 8192)
    if "resolution_percentage" in value:
        result["resolution_percentage"] = _integer(
            value["resolution_percentage"], "resolution_percentage", 1, 100
        )
    if "samples" in value:
        result["samples"] = _integer(value["samples"], "samples", 1, 10_000)
    if "use_denoising" in value:
        result["use_denoising"] = _boolean(value["use_denoising"], "use_denoising")
    if "film_transparent" in value:
        result["film_transparent"] = _boolean(value["film_transparent"], "film_transparent")
    if "use_motion_blur" in value:
        result["use_motion_blur"] = _boolean(value["use_motion_blur"], "use_motion_blur")
    return result


def _source_color_settings(scene: Any) -> dict[str, Any]:
    settings = getattr(scene, "view_settings", None)
    return {
        "view_transform": str(_value(settings, "view_transform", "AgX")),
        "look": str(_value(settings, "look", "Medium High Contrast")),
        "exposure": float(_value(settings, "exposure", 0.0)),
        "gamma": float(_value(settings, "gamma", 1.0)),
    }


def _normalize_color(value: Any, scene: Any) -> dict[str, Any]:
    result = _source_color_settings(scene)
    if value is None:
        return result
    if not isinstance(value, dict):
        raise ValueError("color_settings must be an object or null")
    _reject_unknown(value, {"view_transform", "look", "exposure", "gamma"}, "color_settings")
    if "view_transform" in value:
        result["view_transform"] = _safe_name(value["view_transform"], "view_transform", maximum=64)
    if "look" in value:
        result["look"] = _safe_name(value["look"], "look", maximum=64)
    if "exposure" in value:
        result["exposure"] = _number(value["exposure"], "exposure", -32.0, 32.0)
    if "gamma" in value:
        result["gamma"] = _number(value["gamma"], "gamma", 0.01, 5.0)
    return result


def _normalize_world(value: Any) -> dict[str, Any]:
    if value is None:
        return {"mode": "COPY_SOURCE"}
    if not isinstance(value, dict):
        raise ValueError("world_settings must be an object or null")
    _reject_unknown(value, {"mode", "color_rgb", "strength"}, "world_settings")
    mode = str(value.get("mode", "COPY_SOURCE")).upper()
    if mode not in {"COPY_SOURCE", "SOLID"}:
        raise ValueError("world_settings.mode must be COPY_SOURCE or SOLID")
    if mode == "COPY_SOURCE":
        if "color_rgb" in value or "strength" in value:
            raise ValueError("COPY_SOURCE world_settings does not accept color_rgb or strength")
        return {"mode": mode}
    return {
        "mode": mode,
        "color_rgb": _color(value.get("color_rgb", [0.05, 0.05, 0.05]), "color_rgb"),
        "strength": _number(value.get("strength", 1.0), "strength", 0.0, 1000.0),
    }


def _mapping(value: Any, field: str, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    _reject_unknown(value, allowed, field)
    return value


def _optional_text(value: Any, field: str, maximum: int) -> str | None:
    if value is None:
        return None
    return _safe_name(value, field, maximum=maximum)


def _optional_blender_name(value: Any, field: str, maximum: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{field} must be a non-empty Blender datablock name")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{field} contains control characters")
    if not _SAFE_NAME.fullmatch(value):
        raise ValueError(f"{field} contains unsupported characters")
    return value


def _bounded_text(value: Any, field: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    value = value.strip()
    if (
        not value
        or len(value) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(
            f"{field} must be non-empty, contain no control characters, and use at most "
            f"{maximum} characters"
        )
    return value


def _vec3(
    value: Any,
    field: str,
    *,
    minimum: float = -1_000_000_000.0,
    maximum: float = 1_000_000_000.0,
) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field} must contain exactly three numbers")
    return [
        _number(component, f"{field}[{index}]", minimum, maximum)
        for index, component in enumerate(value)
    ]


def _normalize_lighting(value: Any) -> dict[str, Any]:
    value = _mapping(
        value if value is not None else {},
        "profile.lighting",
        {"existing_light_policy", "lights"},
    )
    policy = str(value.get("existing_light_policy", "KEEP")).upper()
    if policy not in {"KEEP", "MUTE_NON_MANAGED"}:
        raise ValueError("profile.lighting.existing_light_policy is invalid")
    lights = value.get("lights", [])
    if not isinstance(lights, list) or len(lights) > 128:
        raise ValueError("profile.lighting.lights must be an array of at most 128 entries")
    allowed = {
        "id",
        "name",
        "type",
        "location_world",
        "target_point",
        "target_object",
        "energy",
        "color_rgb",
        "use_shadow",
        "radius",
        "sun_angle_degrees",
        "area_shape",
        "size",
        "size_y",
        "spot_angle_degrees",
        "spot_blend",
        "diffuse_factor",
        "specular_factor",
        "volume_factor",
    }
    normalized: list[dict[str, Any]] = []
    ids: set[str] = set()
    names: set[str] = set()
    for index, item in enumerate(lights):
        item = _mapping(item, f"profile.lighting.lights[{index}]", allowed)
        light_id = _identifier(item.get("id"), f"profile.lighting.lights[{index}].id")
        if light_id in ids:
            raise ValueError("profile.lighting.lights contains duplicate ids")
        ids.add(light_id)
        name = _optional_text(item.get("name"), f"profile.lighting.lights[{index}].name", 63)
        if name is not None:
            if name in names:
                raise ValueError("profile.lighting.lights contains duplicate names")
            names.add(name)
        light_type = str(item.get("type", "")).upper()
        if light_type not in {"AREA", "POINT", "SPOT", "SUN"}:
            raise ValueError(f"profile.lighting.lights[{index}].type is invalid")
        location = _vec3(
            item.get("location_world"),
            f"profile.lighting.lights[{index}].location_world",
        )
        target_point = (
            None
            if item.get("target_point") is None
            else _vec3(
                item["target_point"],
                f"profile.lighting.lights[{index}].target_point",
            )
        )
        target_object = _optional_text(
            item.get("target_object"),
            f"profile.lighting.lights[{index}].target_object",
            63,
        )
        if target_point is not None and target_object is not None:
            raise ValueError("Light target_point and target_object are mutually exclusive")
        if light_type == "POINT" and (target_point is not None or target_object is not None):
            raise ValueError("POINT lights do not accept targets")
        radius = (
            None
            if item.get("radius") is None
            else _number(
                item["radius"], f"profile.lighting.lights[{index}].radius", 0.0, 1_000_000.0
            )
        )
        sun_angle = (
            None
            if item.get("sun_angle_degrees") is None
            else _number(
                item["sun_angle_degrees"],
                f"profile.lighting.lights[{index}].sun_angle_degrees",
                0.0,
                180.0,
            )
        )
        area_shape = item.get("area_shape")
        if area_shape is not None:
            area_shape = str(area_shape).upper()
            if area_shape not in {"SQUARE", "RECTANGLE", "DISK", "ELLIPSE"}:
                raise ValueError(f"profile.lighting.lights[{index}].area_shape is invalid")
        size = (
            None
            if item.get("size") is None
            else _number(
                item["size"], f"profile.lighting.lights[{index}].size", 0.000001, 1_000_000.0
            )
        )
        size_y = (
            None
            if item.get("size_y") is None
            else _number(
                item["size_y"],
                f"profile.lighting.lights[{index}].size_y",
                0.000001,
                1_000_000.0,
            )
        )
        spot_angle = (
            None
            if item.get("spot_angle_degrees") is None
            else _number(
                item["spot_angle_degrees"],
                f"profile.lighting.lights[{index}].spot_angle_degrees",
                1.0,
                180.0,
            )
        )
        spot_blend = (
            None
            if item.get("spot_blend") is None
            else _number(
                item["spot_blend"],
                f"profile.lighting.lights[{index}].spot_blend",
                0.0,
                1.0,
            )
        )
        if light_type != "SUN" and sun_angle is not None:
            raise ValueError("sun_angle_degrees is only valid for SUN lights")
        if light_type != "AREA" and any(value is not None for value in (area_shape, size, size_y)):
            raise ValueError("area_shape, size, and size_y are only valid for AREA lights")
        if light_type != "SPOT" and any(value is not None for value in (spot_angle, spot_blend)):
            raise ValueError("spot fields are only valid for SPOT lights")
        if radius is not None and light_type not in {"POINT", "SPOT"}:
            raise ValueError("radius is only valid for POINT and SPOT lights")
        if size_y is not None and area_shape not in {"RECTANGLE", "ELLIPSE"}:
            raise ValueError("size_y requires RECTANGLE or ELLIPSE area_shape")
        result = {
            "id": light_id,
            "name": name,
            "type": light_type,
            "location_world": location,
            "target_point": target_point,
            "target_object": target_object,
            "energy": _number(
                item.get("energy", 1000.0),
                f"profile.lighting.lights[{index}].energy",
                0.0,
                1_000_000_000.0,
            ),
            "color_rgb": _color(
                item.get("color_rgb", [1.0, 1.0, 1.0]),
                f"profile.lighting.lights[{index}].color_rgb",
            ),
            "use_shadow": _boolean(
                item.get("use_shadow", True),
                f"profile.lighting.lights[{index}].use_shadow",
            ),
            "radius": radius,
            "sun_angle_degrees": sun_angle,
            "area_shape": area_shape,
            "size": size,
            "size_y": size_y,
            "spot_angle_degrees": spot_angle,
            "spot_blend": spot_blend,
            "diffuse_factor": _number(
                item.get("diffuse_factor", 1.0),
                f"profile.lighting.lights[{index}].diffuse_factor",
                0.0,
                1.0,
            ),
            "specular_factor": _number(
                item.get("specular_factor", 1.0),
                f"profile.lighting.lights[{index}].specular_factor",
                0.0,
                1.0,
            ),
            "volume_factor": _number(
                item.get("volume_factor", 1.0),
                f"profile.lighting.lights[{index}].volume_factor",
                0.0,
                1.0,
            ),
        }
        normalized.append(result)
    return {"existing_light_policy": policy, "lights": normalized}


def _normalize_world_profile(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    value = _mapping(
        value if value is not None else {},
        "profile.world",
        {
            "mode",
            "color_rgb",
            "strength",
            "hdri_path",
            "rotation_degrees",
            "sun_elevation_degrees",
            "sun_rotation_degrees",
            "altitude_m",
            "air_density",
            "dust_density",
            "ozone_density",
        },
    )
    mode = str(value.get("mode", "KEEP")).upper()
    if mode not in {"KEEP", "MANAGED_SOLID", "MANAGED_SKY", "MANAGED_HDRI"}:
        raise ValueError("profile.world.mode is invalid")
    hdri_path = value.get("hdri_path")
    if hdri_path is not None:
        if not isinstance(hdri_path, str) or not hdri_path.startswith("/"):
            raise ValueError("profile.world.hdri_path must be absolute")
        if not hdri_path.lower().endswith((".hdr", ".exr")):
            raise ValueError("profile.world.hdri_path must use .hdr or .exr")
    if mode == "MANAGED_HDRI" and hdri_path is None:
        raise ValueError("profile.world.hdri_path is required for MANAGED_HDRI")
    if mode != "MANAGED_HDRI" and hdri_path is not None:
        raise ValueError("profile.world.hdri_path is only valid for MANAGED_HDRI")
    requested = {
        "mode": mode,
        "color_rgb": _color(value.get("color_rgb", [0.0, 0.0, 0.0]), "profile.world.color_rgb"),
        "strength": _number(value.get("strength", 0.0), "profile.world.strength", 0.0, 1000.0),
        "hdri_path": hdri_path,
        "rotation_degrees": _number(
            value.get("rotation_degrees", 0.0),
            "profile.world.rotation_degrees",
            -360000.0,
            360000.0,
        ),
        "sun_elevation_degrees": _number(
            value.get("sun_elevation_degrees", 35.0),
            "profile.world.sun_elevation_degrees",
            -90.0,
            90.0,
        ),
        "sun_rotation_degrees": _number(
            value.get("sun_rotation_degrees", 0.0),
            "profile.world.sun_rotation_degrees",
            -360000.0,
            360000.0,
        ),
        "altitude_m": _number(
            value.get("altitude_m", 0.0), "profile.world.altitude_m", -1000.0, 100000.0
        ),
        "air_density": _number(
            value.get("air_density", 1.0), "profile.world.air_density", 0.0, 10.0
        ),
        "dust_density": _number(
            value.get("dust_density", 1.0), "profile.world.dust_density", 0.0, 10.0
        ),
        "ozone_density": _number(
            value.get("ozone_density", 1.0), "profile.world.ozone_density", 0.0, 10.0
        ),
    }
    if mode == "KEEP":
        resolved = {"mode": "COPY_SOURCE"}
    elif mode == "MANAGED_SOLID":
        resolved = {
            "mode": "SOLID",
            "color_rgb": requested["color_rgb"],
            "strength": requested["strength"],
        }
    else:
        resolved = _plain(requested)
    return requested, resolved


def _normalize_color_profile(value: Any, scene: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    value = _mapping(
        value if value is not None else {},
        "profile.color_management",
        {"mode", "view_transform", "look", "exposure", "gamma"},
    )
    mode = str(value.get("mode", "KEEP")).upper()
    if mode not in {"KEEP", "MANAGED"}:
        raise ValueError("profile.color_management.mode is invalid")
    requested = {
        "mode": mode,
        "view_transform": _optional_text(
            value.get("view_transform"), "profile.color_management.view_transform", 128
        ),
        "look": _optional_text(value.get("look"), "profile.color_management.look", 128),
        "exposure": (
            None
            if value.get("exposure") is None
            else _number(value["exposure"], "profile.color_management.exposure", -32.0, 32.0)
        ),
        "gamma": (
            None
            if value.get("gamma") is None
            else _number(value["gamma"], "profile.color_management.gamma", 0.01, 10.0)
        ),
    }
    if mode == "KEEP" and any(
        requested[field] is not None for field in ("view_transform", "look", "exposure", "gamma")
    ):
        raise ValueError("profile.color_management overrides require mode MANAGED")
    source = _source_color_settings(scene)
    resolved = {
        field: requested[field] if requested[field] is not None else source[field]
        for field in ("view_transform", "look", "exposure", "gamma")
    }
    view = getattr(scene, "view_settings", None)
    resolved["view_transform"] = _canonical_color_enum(
        view,
        "view_transform",
        str(resolved["view_transform"]),
        scene=scene,
    )
    resolved["look"] = _canonical_color_enum(
        view,
        "look",
        str(resolved["look"]),
        view_transform=str(resolved["view_transform"]),
        scene=scene,
    )
    return requested, resolved


def _canonical_color_enum(
    owner: Any,
    attribute: str,
    requested: str,
    *,
    view_transform: str | None = None,
    scene: Any | None = None,
) -> str:
    """Resolve Blender-version color enum labels without mutating the source Scene."""
    properties = getattr(getattr(owner, "bl_rna", None), "properties", None)
    prop = _lookup(properties, attribute)
    items = _iter_values(getattr(prop, "enum_items", None))
    if not items:
        return _canonical_ocio_color_enum(
            scene,
            attribute,
            requested,
            view_transform=view_transform,
        )

    candidates: list[tuple[str, set[str]]] = []
    for item in items:
        identifier = str(getattr(item, "identifier", "") or "")
        if not identifier:
            continue
        labels = {
            identifier,
            str(getattr(item, "name", "") or ""),
            str(getattr(item, "description", "") or ""),
        }
        candidates.append((identifier, {label for label in labels if label}))
    if not candidates or (len(candidates) == 1 and candidates[0][0] == "NONE"):
        return _canonical_ocio_color_enum(
            scene,
            attribute,
            requested,
            view_transform=view_transform,
        )
    for identifier, labels in candidates:
        if requested in labels:
            return identifier

    def normalized(label: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", label.casefold())

    requested_key = normalized(requested)
    transform_key = normalized(view_transform or "")
    matches: list[str] = []
    for identifier, labels in candidates:
        alias_keys: set[str] = set()
        for label in labels:
            alias_keys.add(normalized(label))
            if " - " in label:
                prefix, suffix = label.split(" - ", 1)
                if transform_key and normalized(prefix) == transform_key:
                    alias_keys.add(normalized(suffix))
        if requested_key in alias_keys:
            matches.append(identifier)
    matches = list(dict.fromkeys(matches))
    if len(matches) == 1:
        return matches[0]
    available = [identifier for identifier, _labels in candidates]
    raise ValueError(
        f"profile.color_management.{attribute} '{requested}' is unavailable; "
        f"Blender offers {available}"
    )


def _canonical_ocio_color_enum(
    scene: Any,
    attribute: str,
    requested: str,
    *,
    view_transform: str | None,
) -> str:
    """Resolve Blender's dynamic OCIO-backed enums when RNA exposes only NONE."""
    try:
        import PyOpenColorIO as ocio

        config = ocio.GetCurrentConfig()
        display = str(getattr(getattr(scene, "display_settings", None), "display_device", "sRGB"))
        available = (
            list(config.getViews(display))
            if attribute == "view_transform"
            else ["None", *list(config.getLookNames())]
        )
    except (AttributeError, ImportError, RuntimeError, TypeError):
        return requested

    if requested in available:
        if attribute != "look" or requested == "None":
            return requested
        prefix = f"{view_transform} - " if view_transform else ""
        if not prefix or requested.startswith(prefix):
            return requested
        prefixed = f"{prefix}{requested}"
        if prefixed in available:
            return prefixed
        # Unprefixed contrast looks belong to the legacy Filmic view.
        if str(view_transform) == "Filmic":
            return requested
    elif attribute == "look" and view_transform:
        prefixed = f"{view_transform} - {requested}"
        if prefixed in available:
            return prefixed

    normalized = re.sub(r"[^a-z0-9]+", "", requested.casefold())
    matches = [
        value for value in available if re.sub(r"[^a-z0-9]+", "", value.casefold()) == normalized
    ]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        f"profile.color_management.{attribute} '{requested}' is unavailable for "
        f"view '{view_transform}'; Blender/OCIO offers {available}"
    )


def _normalize_render_profile(value: Any, scene: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    value = _mapping(
        value if value is not None else {},
        "profile.render",
        {
            "engine",
            "samples",
            "denoise",
            "resolution_x",
            "resolution_y",
            "resolution_percentage",
            "film_transparent",
            "use_motion_blur",
        },
    )
    engine = str(value.get("engine", "KEEP")).upper()
    if engine not in {
        "KEEP",
        "CYCLES",
        "BLENDER_EEVEE",
        "BLENDER_EEVEE_NEXT",
    }:
        raise ValueError("profile.render.engine is invalid")
    requested = {
        "engine": engine,
        "samples": (
            None
            if value.get("samples") is None
            else _integer(value["samples"], "profile.render.samples", 1, 100_000)
        ),
        "denoise": (
            None
            if value.get("denoise") is None
            else _boolean(value["denoise"], "profile.render.denoise")
        ),
        "resolution_x": (
            None
            if value.get("resolution_x") is None
            else _integer(value["resolution_x"], "profile.render.resolution_x", 1, 8192)
        ),
        "resolution_y": (
            None
            if value.get("resolution_y") is None
            else _integer(value["resolution_y"], "profile.render.resolution_y", 1, 8192)
        ),
        "resolution_percentage": (
            None
            if value.get("resolution_percentage") is None
            else _integer(
                value["resolution_percentage"],
                "profile.render.resolution_percentage",
                1,
                100,
            )
        ),
        "film_transparent": (
            None
            if value.get("film_transparent") is None
            else _boolean(value["film_transparent"], "profile.render.film_transparent")
        ),
        "use_motion_blur": (
            None
            if value.get("use_motion_blur") is None
            else _boolean(value["use_motion_blur"], "profile.render.use_motion_blur")
        ),
    }
    if engine == "KEEP" and any(
        requested[field] is not None
        for field in (
            "samples",
            "denoise",
            "resolution_x",
            "resolution_y",
            "resolution_percentage",
            "film_transparent",
            "use_motion_blur",
        )
    ):
        raise ValueError("profile.render overrides require an explicit render engine")
    source = _source_render_settings(scene)
    resolved = dict(source)
    if engine != "KEEP":
        resolved["engine"] = _resolve_engine(engine)
        for requested_name, resolved_name in (
            ("samples", "samples"),
            ("denoise", "use_denoising"),
            ("resolution_x", "resolution_x"),
            ("resolution_y", "resolution_y"),
            ("resolution_percentage", "resolution_percentage"),
            ("film_transparent", "film_transparent"),
            ("use_motion_blur", "use_motion_blur"),
        ):
            if requested[requested_name] is not None:
                resolved[resolved_name] = requested[requested_name]
    return requested, resolved


def _normalize_post_profile(value: Any) -> dict[str, Any]:
    value = _mapping(
        value if value is not None else {},
        "profile.post",
        {"mode", "bloom", "grain", "vignette"},
    )
    mode = str(value.get("mode", "KEEP")).upper()
    if mode not in {"KEEP", "MANAGED_STACK"}:
        raise ValueError("profile.post.mode is invalid")
    bloom = _mapping(
        value.get("bloom", {}),
        "profile.post.bloom",
        {"enabled", "threshold", "strength", "radius"},
    )
    grain = _mapping(
        value.get("grain", {}),
        "profile.post.grain",
        {"enabled", "strength", "scale", "seed"},
    )
    vignette = _mapping(
        value.get("vignette", {}),
        "profile.post.vignette",
        {"enabled", "strength", "feather"},
    )
    result = {
        "mode": mode,
        "bloom": {
            "enabled": _boolean(bloom.get("enabled", False), "profile.post.bloom.enabled"),
            "threshold": _number(
                bloom.get("threshold", 1.0), "profile.post.bloom.threshold", 0.0, 1000.0
            ),
            "strength": _number(
                bloom.get("strength", 0.0), "profile.post.bloom.strength", 0.0, 100.0
            ),
            "radius": _number(bloom.get("radius", 0.5), "profile.post.bloom.radius", 0.0, 1.0),
        },
        "grain": {
            "enabled": _boolean(grain.get("enabled", False), "profile.post.grain.enabled"),
            "strength": _number(
                grain.get("strength", 0.0), "profile.post.grain.strength", 0.0, 1.0
            ),
            "scale": _number(grain.get("scale", 1.0), "profile.post.grain.scale", 0.01, 1000.0),
            "seed": _integer(grain.get("seed", 0), "profile.post.grain.seed", 0, 2**31 - 1),
        },
        "vignette": {
            "enabled": _boolean(vignette.get("enabled", False), "profile.post.vignette.enabled"),
            "strength": _number(
                vignette.get("strength", 0.0),
                "profile.post.vignette.strength",
                0.0,
                1.0,
            ),
            "feather": _number(
                vignette.get("feather", 0.5),
                "profile.post.vignette.feather",
                0.0,
                1.0,
            ),
        },
    }
    if mode == "KEEP" and any(
        result[effect]["enabled"] for effect in ("bloom", "grain", "vignette")
    ):
        raise ValueError("profile.post effects require mode MANAGED_STACK")
    return result


def _normalize_camera_profile(value: Any) -> dict[str, Any]:
    value = _mapping(
        value if value is not None else {},
        "profile.camera",
        {
            "mode",
            "source_camera",
            "location_world",
            "target_point",
            "use_dof",
            "focus_object",
            "focus_distance_m",
            "aperture_fstop",
        },
    )
    mode = str(value.get("mode", "KEEP")).upper()
    if mode not in {"KEEP", "MANAGED_CLONE"}:
        raise ValueError("profile.camera.mode is invalid")
    result = {
        "mode": mode,
        "source_camera": _optional_blender_name(
            value.get("source_camera"), "profile.camera.source_camera", 63
        ),
        "location_world": (
            None
            if value.get("location_world") is None
            else _vec3(value["location_world"], "profile.camera.location_world")
        ),
        "target_point": (
            None
            if value.get("target_point") is None
            else _vec3(value["target_point"], "profile.camera.target_point")
        ),
        "use_dof": (
            None
            if value.get("use_dof") is None
            else _boolean(value["use_dof"], "profile.camera.use_dof")
        ),
        "focus_object": _optional_blender_name(
            value.get("focus_object"), "profile.camera.focus_object", 63
        ),
        "focus_distance_m": (
            None
            if value.get("focus_distance_m") is None
            else _number(
                value["focus_distance_m"],
                "profile.camera.focus_distance_m",
                0.0001,
                1_000_000.0,
            )
        ),
        "aperture_fstop": (
            None
            if value.get("aperture_fstop") is None
            else _number(
                value["aperture_fstop"],
                "profile.camera.aperture_fstop",
                0.1,
                1024.0,
            )
        ),
    }
    overrides = [result[field] for field in result if field != "mode"]
    if mode == "KEEP" and any(item is not None for item in overrides):
        raise ValueError("profile.camera overrides require mode MANAGED_CLONE")
    if mode == "MANAGED_CLONE" and result["source_camera"] is None:
        raise ValueError("profile.camera MANAGED_CLONE requires source_camera")
    if result["focus_object"] is not None and result["focus_distance_m"] is not None:
        raise ValueError("profile.camera accepts focus_object or focus_distance_m, not both")
    if result["location_world"] is not None and result["target_point"] is not None:
        delta = [
            target - origin
            for target, origin in zip(result["target_point"], result["location_world"])
        ]
        if sum(component * component for component in delta) <= 1e-18:
            raise ValueError(
                "profile.camera.target_point must differ from profile.camera.location_world"
            )
    return result


def _normalize_profile_spec(
    value: Any,
    source: Any,
    *,
    legacy_render: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value = _mapping(value, "profile", _PROFILE_FIELDS)
    schema_version = _integer(value.get("schema_version", 1), "profile.schema_version", 1, 1)
    profile_id = _identifier(value.get("profile_id"))
    display_name = _safe_name(
        value.get("display_name", profile_id), "profile.display_name", maximum=128
    )
    status = str(value.get("status", "DRAFT")).upper()
    if status != "DRAFT":
        raise ValueError(
            "upsert_look_profile only creates DRAFT revisions; ACCEPTED requires "
            "a fresh full composited-render acceptance transition"
        )
    description = value.get("description")
    if description is not None:
        description = _bounded_text(description, "profile.description", 2000)
    generator_version = _identifier(
        value.get("generator_version", "manual-v1"), "profile.generator_version"
    )
    tags = value.get("tags", [])
    if not isinstance(tags, list) or len(tags) > 32:
        raise ValueError("profile.tags must be an array of at most 32 entries")
    normalized_tags = [
        _safe_name(tag, f"profile.tags[{index}]", maximum=64) for index, tag in enumerate(tags)
    ]
    if len(normalized_tags) != len(set(normalized_tags)):
        raise ValueError("profile.tags must not contain duplicates")

    lighting = _normalize_lighting(value.get("lighting"))
    requested_world, resolved_world = _normalize_world_profile(value.get("world"))
    requested_color, resolved_color = _normalize_color_profile(
        value.get("color_management"), source
    )
    requested_render, resolved_render = _normalize_render_profile(value.get("render"), source)
    if legacy_render is not None:
        resolved_render = _normalize_render(legacy_render, source)

    atmosphere = value.get("atmosphere", [])
    if not isinstance(atmosphere, list) or len(atmosphere) > 32:
        raise ValueError("profile.atmosphere must be an array of at most 32 entries")
    atmosphere_ids: set[str] = set()
    normalized_atmosphere: list[dict[str, Any]] = []
    atmosphere_allowed = {
        "id",
        "kind",
        "enabled",
        "seed",
        "location_world",
        "size_xyz",
        "color_rgb",
        "density",
        "anisotropy",
        "rain_rate",
        "drop_size_m",
        "fall_speed_mps",
        "wind_vector",
    }
    for index, component in enumerate(atmosphere):
        component = _mapping(component, f"profile.atmosphere[{index}]", atmosphere_allowed)
        component_id = _identifier(component.get("id"), f"profile.atmosphere[{index}].id")
        if component_id in atmosphere_ids:
            raise ValueError("profile.atmosphere contains duplicate component ids")
        atmosphere_ids.add(component_id)
        kind = str(component.get("kind", "")).upper()
        if kind not in {"FOG_VOLUME", "RAIN_RIG"}:
            raise ValueError(f"profile.atmosphere[{index}].kind is invalid")
        size_xyz = _vec3(
            component.get("size_xyz", [10.0, 10.0, 10.0]),
            f"profile.atmosphere[{index}].size_xyz",
        )
        if any(value <= 0.0 for value in size_xyz):
            raise ValueError(f"profile.atmosphere[{index}].size_xyz must be positive")
        density = (
            None
            if component.get("density") is None
            else _number(
                component["density"],
                f"profile.atmosphere[{index}].density",
                0.0,
                1000.0,
            )
        )
        anisotropy = (
            None
            if component.get("anisotropy") is None
            else _number(
                component["anisotropy"],
                f"profile.atmosphere[{index}].anisotropy",
                -1.0,
                1.0,
            )
        )
        rain_rate = (
            None
            if component.get("rain_rate") is None
            else _number(
                component["rain_rate"],
                f"profile.atmosphere[{index}].rain_rate",
                0.0,
                5_000.0,
            )
        )
        drop_size = (
            None
            if component.get("drop_size_m") is None
            else _number(
                component["drop_size_m"],
                f"profile.atmosphere[{index}].drop_size_m",
                0.00001,
                1.0,
            )
        )
        fall_speed = (
            None
            if component.get("fall_speed_mps") is None
            else _number(
                component["fall_speed_mps"],
                f"profile.atmosphere[{index}].fall_speed_mps",
                0.0,
                1000.0,
            )
        )
        wind = (
            None
            if component.get("wind_vector") is None
            else _vec3(
                component["wind_vector"],
                f"profile.atmosphere[{index}].wind_vector",
            )
        )
        if kind == "FOG_VOLUME":
            if density is None:
                raise ValueError(f"profile.atmosphere[{index}] FOG_VOLUME requires density")
            if any(item is not None for item in (rain_rate, drop_size, fall_speed, wind)):
                raise ValueError("rain fields are only valid for RAIN_RIG")
        else:
            if any(item is None for item in (rain_rate, drop_size, fall_speed)):
                raise ValueError(
                    f"profile.atmosphere[{index}] RAIN_RIG requires rain_rate, "
                    "drop_size_m, and fall_speed_mps"
                )
            if any(item is not None for item in (density, anisotropy)):
                raise ValueError("density and anisotropy are only valid for FOG_VOLUME")
        normalized_component = {
            "id": component_id,
            "kind": kind,
            "enabled": _boolean(
                component.get("enabled", True), f"profile.atmosphere[{index}].enabled"
            ),
            "seed": _integer(
                component.get("seed", 0),
                f"profile.atmosphere[{index}].seed",
                0,
                2**31 - 1,
            ),
            "location_world": _vec3(
                component.get("location_world", [0.0, 0.0, 0.0]),
                f"profile.atmosphere[{index}].location_world",
            ),
            "size_xyz": size_xyz,
            "color_rgb": _color(
                component.get("color_rgb", [1.0, 1.0, 1.0]),
                f"profile.atmosphere[{index}].color_rgb",
            ),
            "density": density,
            "anisotropy": anisotropy,
            "rain_rate": rain_rate,
            "drop_size_m": drop_size,
            "fall_speed_mps": fall_speed,
            "wind_vector": wind,
        }
        normalized_atmosphere.append(normalized_component)

    post = _normalize_post_profile(value.get("post"))
    camera = _normalize_camera_profile(value.get("camera"))
    review_intent = value.get("review_intent")
    if review_intent is not None:
        review_intent = _mapping(
            review_intent,
            "profile.review_intent",
            {
                "summary",
                "expects_shadows",
                "expects_reflections",
                "expects_volume",
                "expects_compositing",
            },
        )
        review_intent = {
            "summary": _bounded_text(
                review_intent.get("summary"),
                "profile.review_intent.summary",
                2000,
            ),
            "expects_shadows": _boolean(
                review_intent.get("expects_shadows", True),
                "profile.review_intent.expects_shadows",
            ),
            "expects_reflections": _boolean(
                review_intent.get("expects_reflections", True),
                "profile.review_intent.expects_reflections",
            ),
            "expects_volume": _boolean(
                review_intent.get("expects_volume", False),
                "profile.review_intent.expects_volume",
            ),
            "expects_compositing": _boolean(
                review_intent.get("expects_compositing", True),
                "profile.review_intent.expects_compositing",
            ),
        }
    requested = {
        "schema_version": schema_version,
        "profile_id": profile_id,
        "display_name": display_name,
        "description": description,
        "status": "DRAFT",
        "seed": _integer(value.get("seed", 0), "profile.seed", 0, 2**31 - 1),
        "generator_version": generator_version,
        "tags": normalized_tags,
        "lighting": lighting,
        "world": requested_world,
        "atmosphere": normalized_atmosphere,
        "post": post,
        "color_management": requested_color,
        "camera": camera,
        "render": requested_render,
    }
    if review_intent is not None:
        requested["review_intent"] = review_intent
    return {
        **requested,
        "source_scene": str(getattr(source, "name", "")),
        "resolved": {
            "world": resolved_world,
            "render": resolved_render,
            "color_management": resolved_color,
            "camera_name": (
                str(getattr(getattr(source, "camera", None), "name", ""))
                if getattr(source, "camera", None) is not None
                else None
            ),
        },
    }


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return result or "profile"


def _artifact_name(
    role: str,
    *,
    source_scene: str,
    profile_id: str,
    version: int,
    revision: int,
) -> str:
    identity = f"{source_scene}\0{profile_id}\0{version}\0{revision}\0{role}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
    readable = _slug(profile_id)[:24]
    name = f"AI_LOOK_{readable}_v{version:03d}_r{revision:03d}_{role}_{digest}"
    return name[:63]


def _profile_entries(
    manifest: dict[str, Any], source_scene: str, profile_id: str
) -> list[dict[str, Any]]:
    return [
        entry
        for entry in manifest["profiles"]
        if entry.get("source_scene") == source_scene and entry.get("profile_id") == profile_id
    ]


def _version_revision(
    manifest: dict[str, Any],
    *,
    source_scene: str,
    profile_id: str,
    update_mode: str,
) -> tuple[int, int, dict[str, Any] | None]:
    entries = _profile_entries(manifest, source_scene, profile_id)
    if update_mode == "CREATE_VERSION":
        version = max((int(entry.get("version", 0)) for entry in entries), default=0) + 1
        revision = max((int(entry.get("revision", 0)) for entry in entries), default=0) + 1
        return version, revision, None

    drafts = [entry for entry in entries if entry.get("status") == "DRAFT"]
    if not drafts:
        raise ValueError(
            f"REPLACE_DRAFT requires an existing DRAFT for profile '{profile_id}' "
            f"and source scene '{source_scene}'"
        )
    previous = max(
        drafts,
        key=lambda entry: (int(entry.get("version", 0)), int(entry.get("revision", 0))),
    )
    return int(previous["version"]), int(previous["revision"]) + 1, previous


def _definition_hash(definition: dict[str, Any]) -> str:
    raw = json.dumps(definition, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _hash_payload(value: Any) -> str:
    raw = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _matrix_snapshot(value: Any) -> list[list[float]] | None:
    if value is None:
        return None
    try:
        rows = [[round(float(component), 9) for component in row] for row in value]
    except (TypeError, ValueError):
        return None
    return rows or None


def _collection_snapshot(collection: Any, visited: set[int] | None = None) -> dict[str, Any]:
    if collection is None:
        return {"name": None, "objects": [], "children": []}
    if visited is None:
        visited = set()
    identity = _rna_identity(collection)
    if identity in visited:
        return {"name": str(getattr(collection, "name", "")), "cycle": True}
    visited.add(identity)
    return {
        "name": str(getattr(collection, "name", "")),
        "objects": sorted(
            str(getattr(item, "name", ""))
            for item in _iter_values(getattr(collection, "objects", None))
        ),
        "children": sorted(
            (
                _collection_snapshot(child, visited)
                for child in _iter_values(getattr(collection, "children", None))
            ),
            key=lambda item: str(item.get("name", "")),
        ),
    }


def _object_snapshot(value: Any) -> dict[str, Any]:
    data = getattr(value, "data", None)
    parent = getattr(value, "parent", None)
    return {
        "name": str(getattr(value, "name", "")),
        "type": str(getattr(value, "type", "")),
        "data_name": str(getattr(data, "name", "")) if data is not None else None,
        "parent": str(getattr(parent, "name", "")) if parent is not None else None,
        "matrix_world": _matrix_snapshot(getattr(value, "matrix_world", None)),
        "hide_render": bool(getattr(value, "hide_render", False)),
        "hide_viewport": bool(getattr(value, "hide_viewport", False)),
    }


def _geometry_fingerprint(scene: Any) -> str:
    payload = {
        "objects": sorted(
            (_object_snapshot(value) for value in _iter_values(getattr(scene, "objects", None))),
            key=lambda item: item["name"],
        ),
        "collection_tree": _collection_snapshot(getattr(scene, "collection", None)),
    }
    return "sha256:" + _hash_payload(payload)


def _compositor_tree(scene: Any) -> tuple[str, Any | None]:
    if hasattr(scene, "compositing_node_group"):
        return "NODE_GROUP", getattr(scene, "compositing_node_group", None)
    if hasattr(scene, "node_tree"):
        return "LEGACY_NODE_TREE", getattr(scene, "node_tree", None)
    return "UNSUPPORTED", None


def _stable_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return round(value, 9) if math.isfinite(value) else str(value)
    if isinstance(value, (list, tuple)):
        return [_stable_value(item) for item in list(value)[:64]]
    try:
        return [_stable_value(item) for item in list(value)[:64]]
    except (TypeError, ReferenceError):
        return str(value)


def _node_tree_payload(tree: Any) -> dict[str, Any] | None:
    if tree is None:
        return None
    nodes = []
    for node in _iter_values(getattr(tree, "nodes", None)):
        inputs = []
        for socket in _iter_values(getattr(node, "inputs", None)):
            inputs.append(
                {
                    "name": str(getattr(socket, "name", "")),
                    "default_value": _stable_value(getattr(socket, "default_value", None)),
                }
            )
        nodes.append(
            {
                "name": str(getattr(node, "name", "")),
                "type": str(getattr(node, "bl_idname", getattr(node, "type", ""))),
                "mute": bool(getattr(node, "mute", False)),
                "operation": str(getattr(node, "operation", "")),
                "blend_type": str(getattr(node, "blend_type", "")),
                "inputs": inputs,
            }
        )
    links = []
    for link in _iter_values(getattr(tree, "links", None)):
        links.append(
            {
                "from_node": str(getattr(getattr(link, "from_node", None), "name", "")),
                "from_socket": str(getattr(getattr(link, "from_socket", None), "name", "")),
                "to_node": str(getattr(getattr(link, "to_node", None), "name", "")),
                "to_socket": str(getattr(getattr(link, "to_socket", None), "name", "")),
            }
        )
    return {
        "name": str(getattr(tree, "name", "")),
        "nodes": sorted(nodes, key=lambda item: (item["name"], item["type"])),
        "links": sorted(
            links,
            key=lambda item: (
                item["from_node"],
                item["from_socket"],
                item["to_node"],
                item["to_socket"],
            ),
        ),
    }


def _compositor_fingerprint(scene: Any) -> str | None:
    adapter, tree = _compositor_tree(scene)
    if tree is None:
        return None
    payload = {"adapter": adapter, "tree": _node_tree_payload(tree)}
    return "sha256:" + _hash_payload(payload)


def _base_fingerprint(scene: Any) -> str:
    world = getattr(scene, "world", None)
    payload = {
        "scene": str(getattr(scene, "name", "")),
        "geometry_fingerprint": _geometry_fingerprint(scene),
        "world": (
            {
                "name": str(getattr(world, "name", "")),
                "color": _stable_value(getattr(world, "color", None)),
                "use_nodes": bool(getattr(world, "use_nodes", False)),
                "node_tree": _node_tree_payload(getattr(world, "node_tree", None)),
            }
            if world is not None
            else None
        ),
        "camera": (
            str(getattr(getattr(scene, "camera", None), "name", ""))
            if getattr(scene, "camera", None) is not None
            else None
        ),
        "render": _source_render_settings(scene),
        "color_management": _source_color_settings(scene),
        "compositor_fingerprint": _compositor_fingerprint(scene),
    }
    return "sha256:" + _hash_payload(payload)


def _base_revision(fingerprint: str) -> int:
    digest = fingerprint.removeprefix("sha256:")
    return int(digest[:15], 16)


def _spatial_revisions() -> dict[str, int]:
    getter = getattr(spatial_cache, "get_revisions", None)
    if not callable(getter):
        return {"geometry_revision": 0, "lighting_revision": 0}
    try:
        result = getter()
    except Exception:
        return {"geometry_revision": 0, "lighting_revision": 0}
    return {
        "geometry_revision": int(result.get("geometry_revision", 0)),
        "lighting_revision": int(result.get("lighting_revision", 0)),
    }


def _manifest_fingerprint(manifest: dict[str, Any]) -> str:
    return "sha256:" + _hash_payload(manifest)


def _current_profile_revision(manifest: dict[str, Any], source_scene: str, profile_id: str) -> int:
    return max(
        (
            int(entry.get("revision", 0))
            for entry in _profile_entries(manifest, source_scene, profile_id)
        ),
        default=0,
    )


def _check_expected_revision(
    value: Any,
    *,
    field: str,
    current: int,
) -> None:
    if value is None:
        return
    expected = _integer(value, field, 0, 2**63 - 1)
    if expected != current:
        raise ValueError(f"Stale {field}: expected {expected}, current {current}")


def _preflight_artifact_names(plan: dict[str, Any]) -> None:
    data = getattr(bpy, "data", None)
    for role, collection_name, name in (
        ("SCENE", "scenes", plan["scene_name"]),
        ("COLLECTION", "collections", plan["collection_name"]),
        ("WORLD", "worlds", plan["world_name"]),
        ("COMPOSITOR", "node_groups", plan["compositor_name"]),
    ):
        if _lookup(getattr(data, collection_name, None), name) is not None:
            raise ValueError(f"Planned {role} datablock name '{name}' already exists")


def _normalize_upsert(params: Any) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    nested = "profile" in params
    envelope_fields = {
        "action",
        "update_mode",
        "source_scene",
        "profile",
        "expected_base_revision",
        "expected_profile_revision",
        "expected_geometry_revision",
        "strict",
    }
    legacy_fields = {
        "action",
        "update_mode",
        "source_scene",
        "profile_id",
        "display_name",
        "random_seed",
        "world_settings",
        "render_settings",
        "color_settings",
        "expected_base_revision",
        "expected_profile_revision",
        "expected_geometry_revision",
        "strict",
    }
    _reject_unknown(params, envelope_fields if nested else legacy_fields, "params")
    action = str(params.get("action", "")).upper()
    if action not in {"VALIDATE", "COMPILE"}:
        raise ValueError("action must be VALIDATE or COMPILE")
    update_mode = str(params.get("update_mode", "")).upper()
    if update_mode not in {"CREATE_VERSION", "REPLACE_DRAFT"}:
        raise ValueError("update_mode must be CREATE_VERSION or REPLACE_DRAFT")
    source = _scene(params.get("source_scene"))
    source_name = str(getattr(source, "name", ""))
    strict = params.get("strict", True)
    if not isinstance(strict, bool):
        raise ValueError("strict must be a boolean")
    if nested:
        definition = _normalize_profile_spec(params.get("profile"), source)
    else:
        legacy_world = _normalize_world(params.get("world_settings"))
        legacy_render = _normalize_render(params.get("render_settings"), source)
        legacy_profile_render = (
            {"engine": "KEEP"}
            if legacy_render["engine"] == "BLENDER_WORKBENCH"
            else {
                "engine": legacy_render["engine"],
                "samples": legacy_render["samples"],
                "denoise": legacy_render["use_denoising"],
                "resolution_percentage": legacy_render["resolution_percentage"],
                "film_transparent": legacy_render["film_transparent"],
                "use_motion_blur": legacy_render["use_motion_blur"],
            }
        )
        world = (
            {"mode": "KEEP"}
            if legacy_world["mode"] == "COPY_SOURCE"
            else {
                "mode": "MANAGED_SOLID",
                "color_rgb": legacy_world["color_rgb"],
                "strength": legacy_world["strength"],
            }
        )
        legacy_color = _normalize_color(params.get("color_settings"), source)
        definition = _normalize_profile_spec(
            {
                "profile_id": params.get("profile_id"),
                "display_name": params.get("display_name", params.get("profile_id")),
                "seed": params.get("random_seed", 0),
                "world": world,
                "render": legacy_profile_render,
                "color_management": {"mode": "MANAGED", **legacy_color},
            },
            source,
            legacy_render=legacy_render,
        )
    profile_id = definition["profile_id"]
    display_name = definition["display_name"]
    # VALIDATE must surface fail-closed shared-Collection conflicts before a
    # caller enters the compiler transaction.  This preflight performs the
    # same recursive View Layer/Collection analysis as asset creation without
    # changing any exclusion flags.
    look_profile_assets.validate_profile_light_isolation(
        source,
        str(definition["lighting"]["existing_light_policy"]),
    )
    manifest = _load_manifest()
    source_base_fingerprint = _base_fingerprint(source)
    source_geometry_fingerprint = _geometry_fingerprint(source)
    revisions = _spatial_revisions()
    current_profile_revision = _current_profile_revision(manifest, source_name, profile_id)
    _check_expected_revision(
        params.get("expected_base_revision"),
        field="expected_base_revision",
        current=_base_revision(source_base_fingerprint),
    )
    _check_expected_revision(
        params.get("expected_profile_revision"),
        field="expected_profile_revision",
        current=current_profile_revision,
    )
    _check_expected_revision(
        params.get("expected_geometry_revision"),
        field="expected_geometry_revision",
        current=revisions["geometry_revision"],
    )
    version, revision, previous = _version_revision(
        manifest,
        source_scene=source_name,
        profile_id=profile_id,
        update_mode=update_mode,
    )
    plan = {
        "compile_mode": COMPILE_MODE,
        "update_mode": update_mode,
        "profile_id": profile_id,
        "display_name": display_name,
        "source_scene": source_name,
        "version": version,
        "revision": revision,
        "previous_profile_revision": current_profile_revision,
        "base_revision": _base_revision(source_base_fingerprint),
        "geometry_revision": revisions["geometry_revision"],
        "lighting_revision": revisions["lighting_revision"],
        "base_fingerprint": source_base_fingerprint,
        "geometry_fingerprint": source_geometry_fingerprint,
        "manifest_fingerprint": _manifest_fingerprint(manifest),
        "strict": strict,
        "supersedes_scene_name": previous.get("scene_name") if previous else None,
        "scene_name": _artifact_name(
            "SCENE",
            source_scene=source_name,
            profile_id=profile_id,
            version=version,
            revision=revision,
        ),
        "collection_name": _artifact_name(
            "COLLECTION",
            source_scene=source_name,
            profile_id=profile_id,
            version=version,
            revision=revision,
        ),
        "world_name": _artifact_name(
            "WORLD",
            source_scene=source_name,
            profile_id=profile_id,
            version=version,
            revision=revision,
        ),
        "compositor_name": _artifact_name(
            "COMPOSITOR",
            source_scene=source_name,
            profile_id=profile_id,
            version=version,
            revision=revision,
        ),
        "definition_hash": _definition_hash(definition),
        "definition": definition,
    }
    _preflight_artifact_names(plan)
    return source, manifest, plan


def _mark_artifact(value: Any, plan: dict[str, Any], role: str) -> None:
    for key, item in (
        (MANAGED_PROP, True),
        (PROFILE_ID_PROP, plan["profile_id"]),
        (PROFILE_VERSION_PROP, plan["version"]),
        (PROFILE_REVISION_PROP, plan["revision"]),
        (SOURCE_SCENE_PROP, plan["source_scene"]),
        (ROLE_PROP, role),
        (SCHEMA_PROP, MANIFEST_SCHEMA_VERSION),
        (COMPILE_MODE_PROP, COMPILE_MODE),
        (DEFINITION_HASH_PROP, plan["definition_hash"]),
        (BASE_FINGERPRINT_PROP, plan["base_fingerprint"]),
        (GEOMETRY_FINGERPRINT_PROP, plan["geometry_fingerprint"]),
    ):
        _custom_set(value, key, item)


def _rna_identity(value: Any) -> int:
    pointer = getattr(value, "as_pointer", None)
    if callable(pointer):
        try:
            return int(pointer())
        except Exception:
            pass
    return id(value)


def _verify_link_copy(source: Any, compiled: Any) -> None:
    source_objects = {_rna_identity(obj) for obj in _iter_values(getattr(source, "objects", None))}
    compiled_objects = {
        _rna_identity(obj) for obj in _iter_values(getattr(compiled, "objects", None))
    }
    if source_objects != compiled_objects:
        raise RuntimeError(
            "Scene.copy() did not preserve linked object identity; refusing to label "
            "the compiled scene as LINK_COPY"
        )


def _copy_scene(source: Any, name: str) -> Any:
    copier = getattr(source, "copy", None)
    if not callable(copier):
        raise RuntimeError(f"Source scene '{getattr(source, 'name', '')}' cannot be copied")
    compiled = copier()
    if compiled is None:
        raise RuntimeError("Blender did not return the linked Scene copy")
    try:
        compiled.name = name
        if str(getattr(compiled, "name", "")) != name:
            raise RuntimeError(f"Blender could not assign deterministic Scene name '{name}'")
        _verify_link_copy(source, compiled)
    except Exception:
        _remove_if_present(getattr(getattr(bpy, "data", None), "scenes", None), compiled)
        raise
    return compiled


def _new_collection(name: str) -> Any:
    collections = getattr(getattr(bpy, "data", None), "collections", None)
    creator = getattr(collections, "new", None)
    if not callable(creator):
        raise RuntimeError("Blender Collections cannot be created")
    collection = creator(name)
    if str(getattr(collection, "name", "")) != name:
        _remove_if_present(collections, collection)
        raise RuntimeError(f"Blender could not assign deterministic Collection name '{name}'")
    return collection


def _new_world(name: str) -> Any:
    worlds = getattr(getattr(bpy, "data", None), "worlds", None)
    creator = getattr(worlds, "new", None)
    if not callable(creator):
        raise RuntimeError("Blender Worlds cannot be created")
    world = creator(name)
    if str(getattr(world, "name", "")) != name:
        _remove_if_present(worlds, world)
        raise RuntimeError(f"Blender could not assign deterministic World name '{name}'")
    return world


def _copy_world(source_world: Any, name: str) -> Any:
    if source_world is None:
        return _new_world(name)
    copier = getattr(source_world, "copy", None)
    if not callable(copier):
        raise RuntimeError("Source World cannot be copied")
    world = copier()
    if world is None:
        raise RuntimeError("Blender did not return the copied World")
    world.name = name
    if str(getattr(world, "name", "")) != name:
        _remove_if_present(getattr(getattr(bpy, "data", None), "worlds", None), world)
        raise RuntimeError(f"Blender could not assign deterministic World name '{name}'")
    return world


def _set_solid_world(world: Any, settings: dict[str, Any]) -> None:
    color = settings["color_rgb"]
    strength = settings["strength"]
    if hasattr(world, "color"):
        world.color = tuple(color)
    world.use_nodes = True
    tree = getattr(world, "node_tree", None)
    nodes = getattr(tree, "nodes", None)
    links = getattr(tree, "links", None)
    if nodes is None or links is None:
        raise RuntimeError("Blender did not create a World node tree")
    nodes.clear()
    background = nodes.new(type="ShaderNodeBackground")
    background.inputs["Color"].default_value = (*color, 1.0)
    background.inputs["Strength"].default_value = strength
    output = nodes.new(type="ShaderNodeOutputWorld")
    links.new(background.outputs["Background"], output.inputs["Surface"])


def _create_profile_world(source: Any, plan: dict[str, Any]) -> Any:
    settings = plan["definition"]["resolved"]["world"]
    if settings["mode"] == "COPY_SOURCE":
        return _copy_world(getattr(source, "world", None), plan["world_name"])
    return _new_world(plan["world_name"])


def _apply_render_settings(scene: Any, settings: dict[str, Any]) -> None:
    render = getattr(scene, "render", None)
    if render is None:
        raise RuntimeError("Compiled Scene has no render settings")
    render.engine = settings["engine"]
    render.resolution_x = settings["resolution_x"]
    render.resolution_y = settings["resolution_y"]
    render.resolution_percentage = settings["resolution_percentage"]
    if hasattr(render, "film_transparent"):
        render.film_transparent = settings["film_transparent"]
    if hasattr(render, "use_motion_blur"):
        render.use_motion_blur = settings["use_motion_blur"]

    if settings["engine"] == "CYCLES":
        cycles = getattr(scene, "cycles", None)
        if cycles is None:
            raise RuntimeError("Compiled Cycles Scene has no Cycles settings")
        cycles.samples = settings["samples"]
        if hasattr(cycles, "use_denoising"):
            cycles.use_denoising = settings["use_denoising"]
    elif settings["engine"].startswith("BLENDER_EEVEE"):
        eevee = getattr(scene, "eevee", None)
        if eevee is not None and hasattr(eevee, "taa_render_samples"):
            eevee.taa_render_samples = settings["samples"]


def _apply_color_settings(scene: Any, settings: dict[str, Any]) -> None:
    view = getattr(scene, "view_settings", None)
    if view is None:
        raise RuntimeError("Compiled Scene has no color-management settings")
    for attribute in ("view_transform", "look", "exposure", "gamma"):
        if not hasattr(view, attribute):
            raise RuntimeError(f"Compiled Scene does not support color setting '{attribute}'")
        setattr(view, attribute, settings[attribute])


def _remove_if_present(collection: Any, value: Any) -> None:
    if collection is None or value is None:
        return
    remover = getattr(collection, "remove", None)
    if not callable(remover):
        return
    try:
        remover(value, do_unlink=True)
    except TypeError:
        remover(value)


def _cleanup_created(scene: Any, collection: Any, world: Any) -> list[str]:
    errors: list[str] = []
    data = getattr(bpy, "data", None)
    # Removing the Scene first releases its references to the profile Collection
    # and World.  The source Scene is never passed to this helper.
    for label, owner, value in (
        ("Scene", getattr(data, "scenes", None), scene),
        ("Collection", getattr(data, "collections", None), collection),
        ("World", getattr(data, "worlds", None), world),
    ):
        if value is None:
            continue
        try:
            _remove_if_present(owner, value)
        except Exception as exc:
            errors.append(f"{label}: {exc}")
    return errors


def _manifest_with_entry(
    manifest: dict[str, Any], plan: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = _plain(manifest)
    if len(result["profiles"]) >= MAX_PROFILE_ENTRIES:
        raise ValueError(f"Look-profile manifest is limited to {MAX_PROFILE_ENTRIES} entries")
    if plan["update_mode"] == "REPLACE_DRAFT":
        previous_name = plan["supersedes_scene_name"]
        matches = [
            entry
            for entry in result["profiles"]
            if entry.get("scene_name") == previous_name and entry.get("status") == "DRAFT"
        ]
        if len(matches) != 1:
            raise RuntimeError("Draft selected during validation changed before compilation")
        matches[0]["status"] = "SUPERSEDED"

    entry = {
        "profile_id": plan["profile_id"],
        "display_name": plan["display_name"],
        "source_scene": plan["source_scene"],
        "version": plan["version"],
        "revision": plan["revision"],
        "status": "DRAFT",
        "compile_mode": COMPILE_MODE,
        "scene_name": plan["scene_name"],
        "collection_name": plan["collection_name"],
        "world_name": plan["world_name"],
        "compositor_name": plan.get("compositor_name"),
        "compositor_hash": plan.get("compositor_hash"),
        "compositor_adapter": plan.get("compositor_adapter"),
        "definition_hash": plan["definition_hash"],
        "base_revision": plan["base_revision"],
        "geometry_revision": plan["geometry_revision"],
        "base_fingerprint": plan["base_fingerprint"],
        "geometry_fingerprint": plan["geometry_fingerprint"],
        "definition": plan["definition"],
        "managed_inventory": plan.get("managed_inventory", {}),
    }
    result["profiles"].append(entry)
    return result, entry


def _flush_dependency_graph_updates() -> None:
    """Flush compiler-created datablock tags inside the managed-edit scope."""
    view_layer = getattr(getattr(bpy, "context", None), "view_layer", None)
    update = getattr(view_layer, "update", None)
    if callable(update):
        try:
            update()
        except (AttributeError, ReferenceError, RuntimeError):
            pass


def _compile(source: Any, manifest: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    # Repeat collision checks immediately before mutation so a VALIDATE result
    # cannot be used after another actor creates one of the planned datablocks.
    if _manifest_fingerprint(_load_manifest()) != plan["manifest_fingerprint"]:
        raise ValueError("Look-profile manifest changed after preflight")
    if _base_fingerprint(source) != plan["base_fingerprint"]:
        raise ValueError("Source Scene base fingerprint changed after preflight")
    if _geometry_fingerprint(source) != plan["geometry_fingerprint"]:
        raise ValueError("Source Scene geometry fingerprint changed after preflight")
    _preflight_artifact_names(plan)
    compiled_scene = None
    profile_collection = None
    profile_world = None
    profile_compositor = None
    profile_compositor_is_node_group = False
    created_profile_assets: list[tuple[str, Any]] = []
    images_before = {
        str(getattr(image, "name", ""))
        for image in _iter_values(getattr(getattr(bpy, "data", None), "images", None))
    }
    begin_managed = getattr(spatial_cache, "begin_managed_edit", None)
    end_managed = getattr(spatial_cache, "end_managed_edit", None)
    begin_evaluation = getattr(spatial_cache, "begin_render_evaluation", None)
    end_evaluation = getattr(spatial_cache, "end_render_evaluation", None)
    if callable(begin_managed):
        begin_managed()
    # Scene.copy() can transiently tag the shared source Camera even though its
    # transform/data remain byte-for-byte unchanged.  Reuse the narrow
    # Scene/Camera evaluation suppressor while source fingerprints guard actual
    # mutations.
    if callable(begin_evaluation):
        begin_evaluation()
    try:
        compiled_scene = _copy_scene(source, plan["scene_name"])
        _mark_artifact(compiled_scene, plan, "SCENE")

        profile_collection = _new_collection(plan["collection_name"])
        _mark_artifact(profile_collection, plan, "COLLECTION")
        children = getattr(getattr(compiled_scene, "collection", None), "children", None)
        linker = getattr(children, "link", None)
        if not callable(linker):
            raise RuntimeError("Compiled Scene root Collection cannot link the profile Collection")
        linker(profile_collection)

        profile_world = _create_profile_world(source, plan)
        world_settings = plan["definition"]["resolved"]["world"]
        if world_settings["mode"] == "SOLID":
            _set_solid_world(profile_world, world_settings)
        _mark_artifact(profile_world, plan, "WORLD")
        compiled_scene.world = profile_world
        _apply_render_settings(compiled_scene, plan["definition"]["resolved"]["render"])
        _apply_color_settings(compiled_scene, plan["definition"]["resolved"]["color_management"])

        source_compositor = look_compositor.get_compositor_tree(source)
        post_spec = plan["definition"]["post"]
        keep_source_post = post_spec["mode"] == "KEEP" and source_compositor is not None
        profile_compositor = look_compositor.ensure_unique_managed_compositor(
            compiled_scene,
            name=plan["compositor_name"],
            source_tree=source_compositor,
            copy_source=keep_source_post,
        )
        look_compositor.rebind_render_layers_to_scene(profile_compositor, compiled_scene)
        profile_compositor_is_node_group = hasattr(compiled_scene, "compositing_node_group")
        nested_compositors = look_compositor.owned_nested_compositor_trees(profile_compositor)
        created_profile_assets.extend(("NODE_GROUP", tree) for tree in nested_compositors)
        _mark_artifact(profile_compositor, plan, "COMPOSITOR")
        for tree in nested_compositors:
            _mark_artifact(tree, plan, "COMPOSITOR_NESTED")
        compositor_api = look_compositor.detect_compositor_api(compiled_scene)
        if post_spec["mode"] == "MANAGED_STACK" or source_compositor is None:
            managed_post = (
                post_spec
                if post_spec["mode"] == "MANAGED_STACK"
                else {
                    "mode": "MANAGED_STACK",
                    "bloom": {"enabled": False},
                    "grain": {"enabled": False},
                    "vignette": {"enabled": False},
                }
            )
            compositor_result = look_compositor.build_managed_post_stack(
                compiled_scene,
                managed_post,
                tree=profile_compositor,
            )
        else:
            output = look_compositor.validate_final_image_linked(
                compiled_scene,
                tree=profile_compositor,
            )
            compositor_result = {
                "api": compositor_api,
                "tree_name": str(getattr(profile_compositor, "name", "")),
                "node_names": sorted(
                    str(getattr(node, "name", ""))
                    for node in _iter_values(getattr(profile_compositor, "nodes", None))
                ),
                "authoritative_output": output,
                "compositor_hash": look_compositor.compositor_hash(
                    compiled_scene,
                    tree=profile_compositor,
                ),
            }
        compiled_scene.render.use_compositing = True
        adapter = (
            "SCENE_COMPOSITING_NODE_GROUP"
            if compositor_api == look_compositor.API_NODE_GROUP
            else "SCENE_EMBEDDED_NODE_TREE"
        )
        plan["compositor_hash"] = compositor_result["compositor_hash"]
        plan["compositor_adapter"] = adapter
        asset_definition = _plain(plan["definition"])
        resolved_render = plan["definition"]["resolved"]["render"]
        asset_definition["render"] = {
            "engine": resolved_render["engine"],
            "samples": resolved_render["samples"],
            "denoise": resolved_render["use_denoising"],
            "resolution_x": resolved_render["resolution_x"],
            "resolution_y": resolved_render["resolution_y"],
            "resolution_percentage": resolved_render["resolution_percentage"],
            "film_transparent": resolved_render["film_transparent"],
            "use_motion_blur": resolved_render["use_motion_blur"],
        }
        asset_definition["color_management"] = {
            "mode": "MANAGED",
            **plan["definition"]["resolved"]["color_management"],
        }
        asset_result = look_profile_assets.build_profile_assets(
            compiled_scene,
            profile_collection,
            profile_world,
            asset_definition,
            name_namespace=plan["scene_name"],
        )
        created_profile_assets.extend(asset_result.get("created", []))
        plan["managed_inventory"] = _plain(asset_result.get("inventory", {}))
        plan["managed_inventory"]["compositor"] = _plain(
            {
                **compositor_result,
                "adapter": adapter,
                "use_compositing": bool(compiled_scene.render.use_compositing),
                "owned_nested_tree_names": [
                    str(getattr(tree, "name", "")) for tree in nested_compositors
                ],
            }
        )

        next_manifest, entry = _manifest_with_entry(manifest, plan)
        if _base_fingerprint(source) != plan["base_fingerprint"]:
            raise RuntimeError("LINK_COPY compilation changed the source Scene base fingerprint")
        if _geometry_fingerprint(source) != plan["geometry_fingerprint"]:
            raise RuntimeError(
                "LINK_COPY compilation changed the source Scene geometry fingerprint"
            )
        response = _plain(
            {
                "action": "COMPILE",
                "compiled": True,
                "compile_mode": COMPILE_MODE,
                "profile": entry,
                "base_scene_preserved": True,
                "warnings": (
                    [
                        "REPLACE_DRAFT retained the superseded compiled datablocks; "
                        "the manifest now points to the new immutable draft revision"
                    ]
                    if plan["update_mode"] == "REPLACE_DRAFT"
                    else []
                ),
            }
        )
        # Manifest persistence is the commit point.  It restores its previous
        # contents on write failure; this outer transaction then removes every
        # newly created Blender datablock.
        _persist_manifest(next_manifest)
        return response
    except Exception as compile_error:
        cleanup_errors = _cleanup_created(compiled_scene, None, None)
        if profile_compositor_is_node_group and profile_compositor is not None:
            cleanup_errors.extend(
                look_profile_assets.cleanup_created([("NODE_GROUP", profile_compositor)])
            )
        new_grain_images = [
            image
            for image in _iter_values(getattr(getattr(bpy, "data", None), "images", None))
            if str(getattr(image, "name", "")) not in images_before
            and bool(_custom_get(image, "blend_ai_look_grain_image", False))
        ]
        if new_grain_images:
            cleanup_errors.extend(
                look_profile_assets.cleanup_created(
                    [("IMAGE", image) for image in new_grain_images]
                )
            )
        cleanup_errors.extend(look_profile_assets.cleanup_created(created_profile_assets))
        cleanup_errors.extend(_cleanup_created(None, profile_collection, profile_world))
        if cleanup_errors:
            raise RuntimeError(
                f"Look-profile compilation failed ({compile_error}); cleanup also failed: "
                + "; ".join(cleanup_errors)
            ) from compile_error
        raise RuntimeError(
            f"Look-profile compilation failed and created artifacts were cleaned up: "
            f"{compile_error}"
        ) from compile_error
    finally:
        _flush_dependency_graph_updates()
        if callable(end_evaluation):
            end_evaluation()
        if callable(end_managed):
            end_managed()


def _scene_context(scene: Any) -> dict[str, Any]:
    world = getattr(scene, "world", None)
    return {
        "name": str(getattr(scene, "name", "")),
        "object_count": len(_iter_values(getattr(scene, "objects", None))),
        "collection_count": len(
            _iter_values(getattr(getattr(scene, "collection", None), "children", None))
        ),
        "world_name": str(getattr(world, "name", "")) if world is not None else None,
        "camera_name": (
            str(getattr(getattr(scene, "camera", None), "name", ""))
            if getattr(scene, "camera", None) is not None
            else None
        ),
        "render_settings": _source_render_settings(scene),
        "color_settings": _source_color_settings(scene),
    }


def _compositor_capabilities(scene: Any) -> dict[str, Any]:
    adapter, tree = _compositor_tree(scene)
    render = getattr(scene, "render", None)
    return {
        "adapter_mode": adapter,
        "supported": adapter != "UNSUPPORTED",
        "assigned": tree is not None,
        "node_tree_name": str(getattr(tree, "name", "")) if tree is not None else None,
        "fingerprint": _compositor_fingerprint(scene),
        "use_compositing": bool(getattr(render, "use_compositing", False)),
        "authoritative_output": (
            "NODE_GROUP_OUTPUT"
            if adapter == "NODE_GROUP"
            else "COMPOSITE_NODE"
            if adapter == "LEGACY_NODE_TREE"
            else None
        ),
    }


def _profile_summary(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        field: entry.get(field)
        for field in (
            "profile_id",
            "display_name",
            "source_scene",
            "version",
            "revision",
            "status",
            "scene_name",
            "definition_hash",
            "base_revision",
            "geometry_revision",
            "compositor_hash",
            "compositor_adapter",
        )
    }


def _artifact_ownership(
    value: Any,
    *,
    role: str,
    entry: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    name = str(getattr(value, "name", "")) if value is not None else None
    checks = {
        "managed": bool(_custom_get(value, MANAGED_PROP, False)) if value is not None else False,
        "role": _custom_get(value, ROLE_PROP) == role if value is not None else False,
        "profile_id": (
            _custom_get(value, PROFILE_ID_PROP) == entry.get("profile_id")
            if value is not None
            else False
        ),
        "source_scene": (
            _custom_get(value, SOURCE_SCENE_PROP) == entry.get("source_scene")
            if value is not None
            else False
        ),
        "version": (
            _custom_get(value, PROFILE_VERSION_PROP) == entry.get("version")
            if value is not None
            else False
        ),
        "revision": (
            _custom_get(value, PROFILE_REVISION_PROP) == entry.get("revision")
            if value is not None
            else False
        ),
        "definition_hash": (
            _custom_get(value, DEFINITION_HASH_PROP) == entry.get("definition_hash")
            if value is not None
            else False
        ),
        "schema_version": (
            _custom_get(value, SCHEMA_PROP) == MANIFEST_SCHEMA_VERSION
            if value is not None
            else False
        ),
        "compile_mode": (
            _custom_get(value, COMPILE_MODE_PROP) == COMPILE_MODE if value is not None else False
        ),
        "base_fingerprint": (
            _custom_get(value, BASE_FINGERPRINT_PROP) == entry.get("base_fingerprint")
            if value is not None
            else False
        ),
        "geometry_fingerprint": (
            _custom_get(value, GEOMETRY_FINGERPRINT_PROP) == entry.get("geometry_fingerprint")
            if value is not None
            else False
        ),
    }
    conflicts = []
    if value is None:
        conflicts.append(f"Missing {role} artifact")
    else:
        for field, matches in checks.items():
            if not matches:
                conflicts.append(f"{role} '{name}' has mismatched {field} ownership")
    return {"role": role, "name": name, "exists": value is not None, "checks": checks}, conflicts


def _entry_ownership(entry: dict[str, Any]) -> dict[str, Any]:
    data = getattr(bpy, "data", None)
    scene = _lookup(getattr(data, "scenes", None), str(entry.get("scene_name", "")))
    collection = _lookup(getattr(data, "collections", None), str(entry.get("collection_name", "")))
    world = _lookup(getattr(data, "worlds", None), str(entry.get("world_name", "")))
    artifacts = []
    conflicts: list[str] = []
    for value, role in ((scene, "SCENE"), (collection, "COLLECTION"), (world, "WORLD")):
        artifact, artifact_conflicts = _artifact_ownership(value, role=role, entry=entry)
        artifacts.append(artifact)
        conflicts.extend(artifact_conflicts)
    if scene is not None and collection is not None:
        children = _iter_values(getattr(getattr(scene, "collection", None), "children", None))
        if all(_rna_identity(item) != _rna_identity(collection) for item in children):
            conflicts.append(
                f"COLLECTION '{getattr(collection, 'name', '')}' is not linked to managed Scene"
            )
    if scene is not None and world is not None:
        if _rna_identity(getattr(scene, "world", None)) != _rna_identity(world):
            conflicts.append(
                f"WORLD '{getattr(world, 'name', '')}' is not assigned to managed Scene"
            )
    compositor: dict[str, Any] = {
        "assigned": False,
        "managed": False,
        "hash": None,
        "adapter": None,
        "valid": False,
    }
    if scene is not None:
        try:
            tree = look_compositor.get_compositor_tree(scene)
            if tree is None:
                raise RuntimeError("managed Scene has no compositor")
            api = look_compositor.detect_compositor_api(scene)
            adapter = (
                "SCENE_COMPOSITING_NODE_GROUP"
                if api == look_compositor.API_NODE_GROUP
                else "SCENE_EMBEDDED_NODE_TREE"
            )
            look_compositor.validate_final_image_linked(scene, tree=tree)
            current_hash = look_compositor.compositor_hash(scene, tree=tree)
            managed = bool(_custom_get(tree, look_compositor.MANAGED_TREE_PROP, False))
            compositor = {
                "assigned": True,
                "managed": managed,
                "name": str(getattr(tree, "name", "")),
                "hash": current_hash,
                "adapter": adapter,
                "valid": (
                    managed
                    and str(getattr(tree, "name", "")) == entry.get("compositor_name")
                    and _custom_get(tree, PROFILE_ID_PROP) == entry.get("profile_id")
                    and _custom_get(tree, PROFILE_VERSION_PROP) == entry.get("version")
                    and _custom_get(tree, PROFILE_REVISION_PROP) == entry.get("revision")
                    and current_hash == entry.get("compositor_hash")
                    and adapter == entry.get("compositor_adapter")
                ),
            }
            if not managed:
                conflicts.append("Profile compositor is not marked as managed")
            if str(getattr(tree, "name", "")) != entry.get("compositor_name"):
                conflicts.append("Profile compositor name differs from the manifest")
            for marker, expected in (
                (PROFILE_ID_PROP, entry.get("profile_id")),
                (PROFILE_VERSION_PROP, entry.get("version")),
                (PROFILE_REVISION_PROP, entry.get("revision")),
            ):
                if _custom_get(tree, marker) != expected:
                    conflicts.append(
                        f"Profile compositor has mismatched ownership marker {marker!r}"
                    )
            if current_hash != entry.get("compositor_hash"):
                conflicts.append("Profile compositor hash differs from the manifest")
            if adapter != entry.get("compositor_adapter"):
                conflicts.append("Profile compositor adapter differs from the manifest")
        except Exception as exc:
            conflicts.append(f"Profile compositor ownership is invalid: {exc}")
    return {
        "profile_id": entry.get("profile_id"),
        "scene_name": entry.get("scene_name"),
        "version": entry.get("version"),
        "revision": entry.get("revision"),
        "valid": not conflicts,
        "artifacts": artifacts,
        "compositor": compositor,
        "conflicts": conflicts,
    }


def _ownership_inventory(
    manifest: dict[str, Any], profiles: list[dict[str, Any]]
) -> dict[str, Any]:
    entries = [_entry_ownership(entry) for entry in profiles]
    claimed = {
        (role, str(entry.get(field, "")))
        for entry in manifest["profiles"]
        for role, field in (
            ("SCENE", "scene_name"),
            ("COLLECTION", "collection_name"),
            ("WORLD", "world_name"),
        )
    }
    orphaned: list[dict[str, Any]] = []
    data = getattr(bpy, "data", None)
    for role, owner_name in (
        ("SCENE", "scenes"),
        ("COLLECTION", "collections"),
        ("WORLD", "worlds"),
    ):
        for value in _iter_values(getattr(data, owner_name, None)):
            if not bool(_custom_get(value, MANAGED_PROP, False)):
                continue
            name = str(getattr(value, "name", ""))
            if (role, name) not in claimed:
                orphaned.append(
                    {
                        "role": role,
                        "name": name,
                        "profile_id": _custom_get(value, PROFILE_ID_PROP),
                    }
                )
    claimed_compositors = {str(entry.get("compositor_name", "")) for entry in manifest["profiles"]}
    for tree in _iter_values(getattr(data, "node_groups", None)):
        if not bool(_custom_get(tree, look_compositor.MANAGED_TREE_PROP, False)):
            continue
        name = str(getattr(tree, "name", ""))
        if name not in claimed_compositors:
            orphaned.append(
                {
                    "role": "COMPOSITOR",
                    "name": name,
                    "profile_id": _custom_get(tree, PROFILE_ID_PROP),
                }
            )
    conflicts = [conflict for entry in entries for conflict in entry["conflicts"]]
    return {
        "valid": not conflicts and not orphaned,
        "profiles": entries,
        "managed_profile_count": len(entries),
        "orphaned_managed_artifacts": orphaned,
        "conflicts": conflicts,
    }


def handle_get_look_profile_context(params: dict[str, Any]) -> dict[str, Any]:
    """Return source-scene and manifest state without creating any datablock."""
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    _reject_unknown(params, {"source_scene", "profile_id", "detail"}, "params")
    source = _scene(params.get("source_scene"), require_writable=False)
    detail = str(params.get("detail", "SUMMARY")).upper()
    if detail not in {"SUMMARY", "CAPABILITIES", "OWNERSHIP", "FULL"}:
        raise ValueError("detail must be SUMMARY, CAPABILITIES, OWNERSHIP, or FULL")
    profile_filter = params.get("profile_id")
    if profile_filter is not None:
        profile_filter = _identifier(profile_filter)
    manifest = _load_manifest()
    source_name = str(getattr(source, "name", ""))
    profiles = [
        entry
        for entry in manifest["profiles"]
        if entry.get("source_scene") == source_name
        and (profile_filter is None or entry.get("profile_id") == profile_filter)
    ]
    base_fingerprint = _base_fingerprint(source)
    geometry_fingerprint = _geometry_fingerprint(source)
    revisions = _spatial_revisions()
    version = _blender_version()
    major = version["version"][0]
    eevee_engine = "BLENDER_EEVEE" if major >= 5 else "BLENDER_EEVEE_NEXT" if major == 4 else None
    profile_payload = (
        profiles if detail == "FULL" else [_profile_summary(entry) for entry in profiles]
    )
    return _plain(
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "detail": detail,
            "manifest_name": MANIFEST_TEXT,
            "manifest_exists": _manifest_text() is not None,
            "manifest_fingerprint": _manifest_fingerprint(manifest),
            "compile_mode": COMPILE_MODE,
            "source_scene": _scene_context(source),
            "base_fingerprint": base_fingerprint,
            "geometry_fingerprint": geometry_fingerprint,
            "base_revision": _base_revision(base_fingerprint),
            "geometry_revision": revisions["geometry_revision"],
            "lighting_revision": revisions["lighting_revision"],
            "profile_revision": (
                _current_profile_revision(manifest, source_name, profile_filter)
                if profile_filter is not None
                else max(
                    (
                        int(entry.get("revision", 0))
                        for entry in manifest["profiles"]
                        if entry.get("source_scene") == source_name
                    ),
                    default=0,
                )
            ),
            "profiles": profile_payload,
            "profile_count": len(profiles),
            "capabilities": {
                "actions": ["VALIDATE", "COMPILE"],
                "update_modes": ["CREATE_VERSION", "REPLACE_DRAFT"],
                "world_modes": ["KEEP", "MANAGED_SOLID", "MANAGED_SKY", "MANAGED_HDRI"],
                "render_engines": [
                    "CYCLES",
                    *([eevee_engine] if eevee_engine is not None else []),
                ],
                "activation": True,
                "long_render_jobs": True,
                "locks": {
                    "mutation_blocked": bool(_profile_mutation_conflicts()),
                    "conflicts": _profile_mutation_conflicts(),
                },
                "blender": version,
                "compositor": _compositor_capabilities(source),
            },
            "ownership": _ownership_inventory(manifest, profiles),
        }
    )


def handle_upsert_look_profile(params: dict[str, Any]) -> dict[str, Any]:
    """Validate or atomically compile one persistent look-profile revision."""
    source, manifest, plan = _normalize_upsert(params)
    action = str(params["action"]).upper()
    if action == "VALIDATE":
        return _plain(
            {
                "action": "VALIDATE",
                "valid": True,
                "plan": plan,
                "mutation_count": 0,
                "warnings": [],
            }
        )
    _require_profile_mutation_idle("compile a look profile")
    return _compile(source, manifest, plan)


def _profile_mutation_conflicts() -> list[str]:
    """Return active resources that make profile compilation or switching unsafe."""
    conflicts: list[str] = []
    try:
        from . import cycles_viewport

        active_sessions = getattr(cycles_viewport, "active_session_ids", lambda: [])()
    except (AttributeError, ImportError):
        active_sessions = []
    if active_sessions:
        conflicts.append(f"retained Cycles viewport sessions {list(active_sessions)}")
    try:
        from . import look_profile_rendering

        has_render = getattr(
            look_profile_rendering,
            "has_pending_or_active_render",
            lambda: False,
        )()
    except (AttributeError, ImportError):
        has_render = False
    if has_render:
        conflicts.append("a queued or active look render")
    return conflicts


def _require_profile_mutation_idle(action: str) -> None:
    conflicts = _profile_mutation_conflicts()
    if conflicts:
        raise RuntimeError(f"Cannot {action} while " + " and ".join(conflicts))


def handle_activate_look_profile(params: dict[str, Any]) -> dict[str, Any]:
    """Activate exactly one manifest-owned profile Scene in one Blender window."""
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    _reject_unknown(
        params,
        {
            "profile_id",
            "target_scene",
            "window_index",
            "expected_profile_revision",
            "expected_geometry_revision",
            "strict",
        },
        "params",
    )
    profile_id = _identifier(params.get("profile_id"))
    target_name = _safe_name(params.get("target_scene"), "target_scene", maximum=63)
    strict = params.get("strict", True)
    if not isinstance(strict, bool):
        raise ValueError("strict must be a boolean")
    _require_profile_mutation_idle("activate a look profile")
    manifest = _load_manifest()
    matches = [
        entry
        for entry in manifest["profiles"]
        if entry.get("profile_id") == profile_id and entry.get("scene_name") == target_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"target_scene '{target_name}' is not exactly one manifest-owned revision "
            f"of profile '{profile_id}'"
        )
    entry = matches[0]
    if entry.get("status") not in {"DRAFT", "ACCEPTED"}:
        raise ValueError(
            f"Profile Scene '{target_name}' has non-activatable status {entry.get('status')!r}"
        )
    _check_expected_revision(
        params.get("expected_profile_revision"),
        field="expected_profile_revision",
        current=int(entry.get("revision", 0)),
    )
    revisions = _spatial_revisions()
    _check_expected_revision(
        params.get("expected_geometry_revision"),
        field="expected_geometry_revision",
        current=revisions["geometry_revision"],
    )
    ownership = _entry_ownership(entry)
    if not ownership["valid"]:
        raise ValueError(
            "Profile ownership validation failed: " + "; ".join(ownership["conflicts"])
        )
    scene = _scene(target_name, "target_scene", require_writable=False)
    if getattr(scene, "library", None) is not None:
        raise ValueError("Managed profile Scene must be local and writable")
    source = _scene(entry.get("source_scene"), require_writable=False)
    warnings: list[str] = []
    current_base = _base_fingerprint(source)
    current_geometry = _geometry_fingerprint(source)
    for label, current, recorded in (
        ("base", current_base, entry.get("base_fingerprint")),
        ("geometry", current_geometry, entry.get("geometry_fingerprint")),
    ):
        if recorded is not None and current != recorded:
            message = f"Source Scene {label} fingerprint is stale for profile '{profile_id}'"
            if strict:
                raise ValueError(message)
            warnings.append(message)

    context = getattr(bpy, "context", None)
    window_manager = getattr(context, "window_manager", None)
    windows = _iter_values(getattr(window_manager, "windows", None))
    requested_index = params.get("window_index")
    if requested_index is not None:
        requested_index = _integer(requested_index, "window_index", 0, 128)
        if requested_index >= len(windows):
            raise ValueError(
                f"window_index {requested_index} is unavailable; Blender has {len(windows)} windows"
            )
        window = windows[requested_index]
        selected_index = requested_index
    else:
        window = getattr(context, "window", None)
        if window is None and windows:
            window = windows[0]
        if window is None:
            raise RuntimeError("Blender has no window in which to activate the profile")
        selected_index = next(
            (
                index
                for index, candidate in enumerate(windows)
                if _rna_identity(candidate) == _rna_identity(window)
            ),
            0,
        )
    previous = getattr(window, "scene", None)
    window.scene = scene
    if _rna_identity(getattr(window, "scene", None)) != _rna_identity(scene):
        raise RuntimeError(f"Blender refused to activate Scene '{target_name}'")
    return _plain(
        {
            "activated": True,
            "profile_id": profile_id,
            "target_scene": target_name,
            "source_scene": entry.get("source_scene"),
            "version": entry.get("version"),
            "revision": entry.get("revision"),
            "status": entry.get("status"),
            "window_index": selected_index,
            "previous_scene": (
                str(getattr(previous, "name", "")) if previous is not None else None
            ),
            "geometry_revision": revisions["geometry_revision"],
            "ownership": ownership,
            "saved_blend": False,
            "warnings": warnings,
        }
    )


def _get_acceptance_evidence(batch_id: str, result_id: str) -> dict[str, Any]:
    """Resolve renderer-owned evidence lazily without exposing its in-memory jobs."""
    try:
        from . import look_profile_rendering
    except (AttributeError, ImportError) as exc:
        raise RuntimeError("The managed look renderer is unavailable") from exc
    getter = getattr(look_profile_rendering, "get_acceptance_evidence", None)
    if not callable(getter):
        raise RuntimeError("The managed look renderer cannot provide acceptance evidence")
    return getter(batch_id, result_id)


def handle_accept_look_profile(params: dict[str, Any]) -> dict[str, Any]:
    """Promote one DRAFT only from a reviewed, current, full-quality Composite."""
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    _reject_unknown(
        params,
        {
            "profile_id",
            "target_scene",
            "batch_id",
            "result_id",
            "artifact_sha256",
            "review_acknowledged",
            "expected_profile_revision",
            "expected_geometry_revision",
        },
        "params",
    )
    profile_id = _identifier(params.get("profile_id"))
    target_scene = _safe_name(params.get("target_scene"), "target_scene", maximum=63)
    batch_id = _identifier(params.get("batch_id"), "batch_id")
    result_id = _identifier(params.get("result_id"), "result_id")
    artifact_sha256 = params.get("artifact_sha256")
    if not isinstance(artifact_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", artifact_sha256):
        raise ValueError("artifact_sha256 must be a lowercase SHA-256 digest")
    if params.get("review_acknowledged") is not True:
        raise ValueError("review_acknowledged must be true after visual Composite review")
    _require_profile_mutation_idle("accept a look profile")

    manifest = _load_manifest()
    matches = [
        entry
        for entry in manifest["profiles"]
        if entry.get("profile_id") == profile_id and entry.get("scene_name") == target_scene
    ]
    if len(matches) != 1:
        raise ValueError("Acceptance target is not exactly one manifest-owned profile revision")
    entry = matches[0]
    if entry.get("status") != "DRAFT":
        raise ValueError(f"Only a DRAFT can be accepted; profile is {entry.get('status')!r}")
    _check_expected_revision(
        params.get("expected_profile_revision"),
        field="expected_profile_revision",
        current=int(entry["revision"]),
    )
    revisions = _spatial_revisions()
    _check_expected_revision(
        params.get("expected_geometry_revision"),
        field="expected_geometry_revision",
        current=revisions["geometry_revision"],
    )
    ownership = _entry_ownership(entry)
    if not ownership["valid"]:
        raise ValueError(
            "Profile ownership validation failed: " + "; ".join(ownership["conflicts"])
        )
    evidence = _get_acceptance_evidence(batch_id, result_id)
    expected_evidence = {
        "profile_id": profile_id,
        "target_scene": target_scene,
        "profile_version": entry["version"],
        "profile_revision": entry["revision"],
        "profile_status": "DRAFT",
        "definition_hash": entry["definition_hash"],
        "base_revision": entry["base_revision"],
        "geometry_revision": entry["geometry_revision"],
        "base_fingerprint": entry["base_fingerprint"],
        "geometry_fingerprint": entry["geometry_fingerprint"],
        "compositor_hash": entry["compositor_hash"],
        "compositor_adapter": entry["compositor_adapter"],
        "image_source": "COMPOSITE_OUTPUT",
        "output_pass": "COMPOSITE",
        "artifact_sha256": artifact_sha256,
    }
    mismatched = [
        field for field, expected in expected_evidence.items() if evidence.get(field) != expected
    ]
    if mismatched:
        raise ValueError(f"Acceptance evidence differs for fields: {mismatched}")

    scene = _scene(target_scene, "target_scene", require_writable=False)
    resolved_render = entry["definition"]["resolved"]["render"]
    expected_width = max(
        1,
        round(
            int(getattr(scene.render, "resolution_x", 0))
            * int(resolved_render["resolution_percentage"])
            / 100
        ),
    )
    expected_height = max(
        1,
        round(
            int(getattr(scene.render, "resolution_y", 0))
            * int(resolved_render["resolution_percentage"])
            / 100
        ),
    )
    quality_mismatches: list[str] = []
    if (evidence.get("source_width"), evidence.get("source_height")) != (
        expected_width,
        expected_height,
    ):
        quality_mismatches.append(f"dimensions must be {expected_width}x{expected_height}")
    if evidence.get("engine") != resolved_render["engine"]:
        quality_mismatches.append(f"engine must be {resolved_render['engine']}")
    if int(evidence.get("samples", 0)) < int(resolved_render["samples"]):
        quality_mismatches.append(f"samples must be at least {resolved_render['samples']}")
    if evidence.get("denoise") != bool(resolved_render["use_denoising"]):
        quality_mismatches.append(f"denoise must be {bool(resolved_render['use_denoising'])}")
    if quality_mismatches:
        raise ValueError(
            "Acceptance requires the compiled profile's full render preset: "
            + "; ".join(quality_mismatches)
        )

    next_manifest = _plain(manifest)
    accepted_entry = next(
        item for item in next_manifest["profiles"] if item["scene_name"] == target_scene
    )
    accepted_at = time.time()
    acceptance = {
        "schema_version": 1,
        "batch_id": batch_id,
        "result_id": result_id,
        "frame": int(evidence["frame"]),
        "artifact_path": str(evidence["artifact_path"]),
        "artifact_sha256": artifact_sha256,
        "artifact_byte_count": int(evidence["artifact_byte_count"]),
        "source_width": int(evidence["source_width"]),
        "source_height": int(evidence["source_height"]),
        "engine": str(evidence["engine"]),
        "samples": int(evidence["samples"]),
        "denoise": bool(evidence["denoise"]),
        "completed_at": float(evidence["completed_at"]),
        "accepted_at": accepted_at,
        "review_acknowledged": True,
    }
    accepted_entry["status"] = "ACCEPTED"
    accepted_entry["acceptance"] = acceptance
    _persist_manifest(next_manifest)
    return _plain(
        {
            "accepted": True,
            "profile_id": profile_id,
            "target_scene": target_scene,
            "version": entry["version"],
            "revision": entry["revision"],
            "status": "ACCEPTED",
            "acceptance": acceptance,
            "ownership": ownership,
            "saved_blend": False,
        }
    )


def register() -> None:
    """Register look-profile commands with the Blender dispatcher."""
    dispatcher.register_handler("get_look_profile_context", handle_get_look_profile_context)
    dispatcher.register_handler("upsert_look_profile", handle_upsert_look_profile)
    dispatcher.register_handler("activate_look_profile", handle_activate_look_profile)
    dispatcher.register_handler("accept_look_profile", handle_accept_look_profile)


def unregister() -> None:
    """No process-local state is retained by this handler."""
