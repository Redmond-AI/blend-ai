"""Asynchronous, profile-scene-only Composite render batches.

The socket bridge has a short request timeout, while a production Cycles render
may take minutes.  This module therefore records bounded in-memory jobs and
starts one render at a time from a Blender timer with ``INVOKE_DEFAULT``.
Blender's render callbacks only record completion.  A main-loop timer finalizes
the item, restores every temporary setting, and schedules the next item.  This
keeps datablock mutations out of render callbacks while Blender may still own
render/dependency-graph threads.  No operation in this module saves the
``.blend``.

Only explicitly named Scenes produced by the look-profile compiler are
accepted.  The marker checks intentionally do not infer a profile from the
current window or from a collection name.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import struct
import threading
import time
import uuid
from array import array
from pathlib import Path
from typing import Any

import bpy

try:
    from bpy.app.handlers import persistent as _persistent_handler
except (AttributeError, ImportError):
    # Focused unit tests provide a minimal non-package bpy stub.
    def _persistent_handler(callback: Any) -> Any:
        return callback


from .. import dispatcher, look_compositor
from . import look_profiles as profile_compiler


MANAGED_PROP = profile_compiler.MANAGED_PROP
PROFILE_ID_PROP = profile_compiler.PROFILE_ID_PROP
PROFILE_VERSION_PROP = profile_compiler.PROFILE_VERSION_PROP
PROFILE_REVISION_PROP = profile_compiler.PROFILE_REVISION_PROP
SOURCE_SCENE_PROP = profile_compiler.SOURCE_SCENE_PROP
ROLE_PROP = profile_compiler.ROLE_PROP
COMPILE_MODE_PROP = profile_compiler.COMPILE_MODE_PROP

MAX_RENDER_ITEMS = 10_000
MAX_RENDER_FRAMES = 10_000
MAX_BATCHES = 128
MAX_PROXY_BYTES = 16 * 1024 * 1024
MAX_REVIEW_TILE_BYTES = 16 * 1024 * 1024
MAX_PROXY_DIMENSION = 2048
REVIEW_TILE_ORDER = (
    ("combined", "Combined (pre-compositor)"),
    ("diffuse_direct", "Diffuse Direct"),
    ("glossy", "Glossy Direct + Indirect"),
    ("emission", "Emission"),
    ("depth", "Camera Depth (near=white)"),
    ("cryptomatte_object", "Object Cryptomatte"),
)
TERMINAL_ITEM_STATUSES = frozenset({"SUCCEEDED", "SKIPPED", "FAILED", "CANCELLED"})
TERMINAL_BATCH_STATUSES = frozenset({"SUCCEEDED", "SKIPPED", "FAILED", "CANCELLED"})

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SAFE_SCENE_NAME = re.compile(r"^[A-Za-z0-9_. -]{1,63}$")
_SAFE_TEMPLATE = re.compile(r"^[A-Za-z0-9_.{}:-]{1,128}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_FINGERPRINT = re.compile(r"^sha256:[a-f0-9]{64}$")
_FORMAT_EXTENSIONS = {"PNG": ".png", "OPEN_EXR": ".exr", "JPEG": ".jpg"}
_ENTRY_PROVENANCE_FIELDS = (
    "profile_id",
    "source_scene",
    "version",
    "revision",
    "status",
    "definition_hash",
    "base_revision",
    "geometry_revision",
    "base_fingerprint",
    "geometry_fingerprint",
    "compositor_hash",
    "compositor_adapter",
)

# Ordered insertion semantics on normal dicts are guaranteed by supported
# Python versions and make scheduling deterministic without another queue.
_batches: dict[str, dict[str, Any]] = {}
_active_render: dict[str, Any] | None = None
_timer_armed = False
_completion_request: dict[str, Any] | None = None
_completion_timer_armed = False


def has_pending_or_active_render() -> bool:
    """Return whether a managed profile render is queued or running."""
    if _active_render is not None:
        return True
    return any(
        item.get("status") in {"QUEUED", "RUNNING"}
        for batch in _batches.values()
        for item in batch.get("items", [])
    )


def _active_cycles_viewport_session_ids() -> list[str]:
    """Read the viewport coordinator lazily to keep module imports acyclic."""
    try:
        from . import cycles_viewport
    except (AttributeError, ImportError):
        return []
    active = getattr(cycles_viewport, "active_session_ids", None)
    return list(active()) if callable(active) else []


def _utc_now() -> float:
    return time.time()


def _reject_unknown(value: dict[str, Any], allowed: set[str], field: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{field} contains unsupported fields: {unknown}")


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{field} is not a safe identifier")
    return value


def _scene_name(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SAFE_SCENE_NAME.fullmatch(value):
        raise ValueError(f"{field} is not a safe Blender Scene name")
    return value


def _integer(value: Any, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{field} must be between {minimum} and {maximum}")
    return value


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _collection_values(collection: Any) -> list[Any]:
    if collection is None:
        return []
    if isinstance(collection, dict):
        return list(collection.values())
    try:
        return list(collection)
    except (ReferenceError, TypeError):
        return []


def _lookup(collection: Any, name: str) -> Any | None:
    getter = getattr(collection, "get", None)
    if callable(getter):
        try:
            return getter(name)
        except (ReferenceError, TypeError):
            pass
    return next(
        (value for value in _collection_values(collection) if getattr(value, "name", None) == name),
        None,
    )


def _custom_get(value: Any, key: str, default: Any = None) -> Any:
    getter = getattr(value, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except (ReferenceError, TypeError):
            return default
    try:
        return value[key]
    except (KeyError, ReferenceError, TypeError):
        return default


def _entry_digest(entry: dict[str, Any]) -> str:
    raw = json.dumps(entry, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _compositor_adapter(scene: Any) -> str:
    api = look_compositor.detect_compositor_api(scene)
    return (
        "SCENE_COMPOSITING_NODE_GROUP"
        if api == look_compositor.API_NODE_GROUP
        else "SCENE_EMBEDDED_NODE_TREE"
    )


def _entry_provenance(scene: Any, entry: dict[str, Any]) -> dict[str, Any]:
    source_name = entry.get("source_scene")
    source = _lookup(
        getattr(getattr(bpy, "data", None), "scenes", None),
        str(source_name or ""),
    )
    if source is None:
        raise ValueError(
            f"Manifest-owned source Scene '{source_name}' for '{scene.name}' was not found"
        )
    current_base_fingerprint = profile_compiler._base_fingerprint(source)
    current_geometry_fingerprint = profile_compiler._geometry_fingerprint(source)
    if current_base_fingerprint != entry.get("base_fingerprint"):
        raise ValueError(
            f"Source Scene '{source_name}' base fingerprint changed after profile compilation"
        )
    if current_geometry_fingerprint != entry.get("geometry_fingerprint"):
        raise ValueError(
            f"Source Scene '{source_name}' geometry fingerprint changed after profile compilation"
        )
    tree = look_compositor.get_compositor_tree(scene)
    if tree is None:
        raise ValueError(f"Compiled profile Scene '{scene.name}' has no managed compositor")
    look_compositor.validate_final_image_linked(scene, tree=tree)
    current_compositor_hash = look_compositor.compositor_hash(scene, tree=tree)
    current_adapter = _compositor_adapter(scene)
    if current_compositor_hash != entry.get("compositor_hash"):
        raise ValueError(
            f"Compiled profile Scene '{scene.name}' compositor hash differs from its manifest entry"
        )
    if current_adapter != entry.get("compositor_adapter"):
        raise ValueError(
            f"Compiled profile Scene '{scene.name}' compositor adapter differs from its manifest entry"
        )
    provenance = {field: entry.get(field) for field in _ENTRY_PROVENANCE_FIELDS}
    required = [field for field, value in provenance.items() if value is None]
    if required:
        raise ValueError(
            f"Manifest entry for Scene '{scene.name}' lacks render provenance: {required}"
        )
    version = provenance.pop("version")
    revision = provenance.pop("revision")
    for field, value, minimum in (
        ("profile version", version, 1),
        ("profile revision", revision, 1),
        ("base revision", provenance["base_revision"], 0),
        ("geometry revision", provenance["geometry_revision"], 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"Manifest {field} is invalid for Scene '{scene.name}'")
    for field in ("definition_hash", "compositor_hash"):
        if not isinstance(provenance[field], str) or not _SHA256.fullmatch(provenance[field]):
            raise ValueError(f"Manifest {field} is invalid for Scene '{scene.name}'")
    for field in ("base_fingerprint", "geometry_fingerprint"):
        if not isinstance(provenance[field], str) or not _FINGERPRINT.fullmatch(provenance[field]):
            raise ValueError(f"Manifest {field} is invalid for Scene '{scene.name}'")
    provenance.update(
        {
            "profile_version": version,
            "profile_revision": revision,
            "profile_status": str(provenance.pop("status")),
            "manifest_entry_hash": _entry_digest(entry),
        }
    )
    if provenance["profile_status"] not in {"DRAFT", "ACCEPTED"}:
        raise ValueError(
            f"Manifest entry for Scene '{scene.name}' is {provenance['profile_status']}, "
            "not a renderable DRAFT or ACCEPTED revision"
        )
    return provenance


def _profile_scene(
    name: str,
    profile_id: str,
    expected_revision: int | None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Resolve one exact manifest entry and verify all compiler-owned artifacts."""
    scene = _lookup(getattr(getattr(bpy, "data", None), "scenes", None), name)
    if scene is None:
        raise ValueError(f"Compiled profile Scene '{name}' was not found")
    if getattr(scene, "library", None) is not None:
        raise ValueError(f"Compiled profile Scene '{name}' is linked/read-only")
    expected_markers = {
        MANAGED_PROP: True,
        PROFILE_ID_PROP: profile_id,
        ROLE_PROP: "SCENE",
        COMPILE_MODE_PROP: "LINK_COPY",
    }
    for key, expected in expected_markers.items():
        actual = _custom_get(scene, key)
        if actual != expected:
            raise ValueError(
                f"Scene '{name}' is not the compiled LINK_COPY Scene for profile "
                f"'{profile_id}': marker {key!r} is {actual!r}"
            )
    version = _custom_get(scene, PROFILE_VERSION_PROP)
    revision = _custom_get(scene, PROFILE_REVISION_PROP)
    source_scene = _custom_get(scene, SOURCE_SCENE_PROP)
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version < 1
        or isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 1
        or not isinstance(source_scene, str)
        or not source_scene
    ):
        raise ValueError(f"Scene '{name}' has incomplete compiled-profile provenance markers")
    if expected_revision is not None and revision != expected_revision:
        raise ValueError(
            f"Scene '{name}' profile revision {revision} does not match expected "
            f"revision {expected_revision}"
        )
    manifest = profile_compiler._load_manifest()
    matches = [
        entry
        for entry in manifest.get("profiles", [])
        if entry.get("scene_name") == name
        and entry.get("profile_id") == profile_id
        and entry.get("source_scene") == source_scene
        and entry.get("version") == version
        and entry.get("revision") == revision
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Scene '{name}' does not resolve to exactly one manifest-owned revision "
            f"for profile '{profile_id}'"
        )
    entry = matches[0]
    ownership = profile_compiler._entry_ownership(entry)
    if not ownership.get("valid", False):
        conflicts = ownership.get("conflicts", [])
        raise ValueError(
            f"Scene '{name}' failed managed ownership validation: " + "; ".join(conflicts)
        )
    return scene, entry, _entry_provenance(scene, entry)


def _assert_item_provenance(
    item: dict[str, Any],
    *,
    phase: str,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    scene, entry, current = _profile_scene(
        item["target_scene"],
        item["profile_id"],
        item["expected_profile_revision"],
    )
    expected = item["expected_provenance"]
    changed = [
        field
        for field in sorted(set(expected) | set(current))
        if expected.get(field) != current.get(field)
    ]
    if changed:
        raise RuntimeError(f"Render item provenance changed between enqueue and {phase}: {changed}")
    return scene, entry, current


def _frames(value: Any) -> list[int]:
    if not isinstance(value, dict):
        raise ValueError("batch.frames must be an object")
    _reject_unknown(value, {"frames", "start", "end", "step"}, "batch.frames")
    explicit = value.get("frames")
    range_present = any(value.get(key) is not None for key in ("start", "end", "step"))
    if (explicit is not None) == range_present:
        raise ValueError("batch.frames must provide exactly one list or range")
    if explicit is not None:
        if not isinstance(explicit, list) or not explicit:
            raise ValueError("batch.frames.frames must be a non-empty list")
        if len(explicit) > MAX_RENDER_FRAMES:
            raise ValueError(f"batch.frames is limited to {MAX_RENDER_FRAMES} frames")
        result = [
            _integer(frame, f"batch.frames.frames[{index}]", -1_000_000, 1_000_000)
            for index, frame in enumerate(explicit)
        ]
        if len(result) != len(set(result)):
            raise ValueError("batch.frames.frames contains duplicates")
        return result
    start = _integer(value.get("start"), "batch.frames.start", -1_000_000, 1_000_000)
    end = _integer(value.get("end"), "batch.frames.end", -1_000_000, 1_000_000)
    step = _integer(value.get("step"), "batch.frames.step", 1, 1_000_000)
    if end < start:
        raise ValueError("batch.frames.end must be greater than or equal to start")
    count = ((end - start) // step) + 1
    if count > MAX_RENDER_FRAMES:
        raise ValueError(f"batch.frames is limited to {MAX_RENDER_FRAMES} frames")
    return list(range(start, end + 1, step))


def _render_settings(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("batch.render_settings must be an object")
    _reject_unknown(
        value,
        {
            "samples",
            "denoise",
            "resolution_percentage",
            "file_format",
            "color_depth",
            "existing_file_policy",
        },
        "batch.render_settings",
    )
    result = {
        "samples": _integer(value.get("samples", 64), "samples", 1, 100_000),
        "denoise": _boolean(value.get("denoise", True), "denoise"),
        "resolution_percentage": _integer(
            value.get("resolution_percentage", 100),
            "resolution_percentage",
            1,
            100,
        ),
        "file_format": str(value.get("file_format", "PNG")),
        "color_depth": str(value.get("color_depth", "16")),
        "existing_file_policy": str(value.get("existing_file_policy", "ERROR")),
    }
    if result["file_format"] not in _FORMAT_EXTENSIONS:
        raise ValueError("file_format must be PNG, OPEN_EXR, or JPEG")
    if result["color_depth"] not in {"8", "16", "32"}:
        raise ValueError("color_depth must be 8, 16, or 32")
    if result["existing_file_policy"] not in {"ERROR", "SKIP"}:
        raise ValueError("existing_file_policy must be ERROR or SKIP")
    if result["file_format"] == "JPEG" and result["color_depth"] != "8":
        raise ValueError("JPEG render batches require color_depth='8'")
    if result["file_format"] == "PNG" and result["color_depth"] == "32":
        raise ValueError("PNG render batches require 8- or 16-bit depth")
    if result["file_format"] == "OPEN_EXR" and result["color_depth"] == "8":
        raise ValueError("OPEN_EXR render batches require 16- or 32-bit depth")
    return result


def _resolved_output_root(value: Any) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("batch.output_root must be an absolute path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("batch.output_root must be absolute")
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise ValueError("batch.output_root is not a directory")
    return path.resolve(strict=True)


def _output_path(
    root: Path,
    template: str,
    *,
    scene: str,
    profile: str,
    camera: str,
    frame: int,
    file_format: str,
) -> Path:
    stem = template.format(scene=scene, profile=profile, camera=camera, frame=frame)
    if not stem or stem in {".", ".."} or Path(stem).name != stem:
        raise ValueError("filename_template must resolve to one safe filename component")
    candidate = (root / f"{stem}{_FORMAT_EXTENSIONS[file_format]}").resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Resolved render output escapes output_root") from exc
    return candidate


def _normalized_batch(params: Any) -> dict[str, Any]:
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    _reject_unknown(
        params,
        {
            "batch",
            "expected_profile_revision",
            "expected_profile_revisions",
            "output_pass",
            "restore_scene_state",
            "save_blend",
        },
        "params",
    )
    if params.get("output_pass") != "COMPOSITE":
        raise ValueError("output_pass must be COMPOSITE")
    if params.get("restore_scene_state") is not True:
        raise ValueError("restore_scene_state must be true")
    if params.get("save_blend") is not False:
        raise ValueError("save_blend must be false")
    expected_revision = params.get("expected_profile_revision")
    if expected_revision is not None:
        expected_revision = _integer(expected_revision, "expected_profile_revision", 0, 2**63 - 1)
    expected_revisions = params.get("expected_profile_revisions")
    if expected_revisions is not None and not isinstance(expected_revisions, dict):
        raise ValueError("expected_profile_revisions must be an object keyed by target Scene")
    if expected_revision is not None and expected_revisions is not None:
        raise ValueError(
            "Provide expected_profile_revision or expected_profile_revisions, not both"
        )
    batch = params.get("batch")
    if not isinstance(batch, dict):
        raise ValueError("batch must be an object")
    _reject_unknown(
        batch,
        {
            "profile_ids",
            "target_scenes",
            "pairing",
            "frames",
            "output_root",
            "filename_template",
            "render_settings",
            "continue_on_error",
            "include_review_packet",
            "camera_names",
        },
        "batch",
    )
    profile_values = batch.get("profile_ids")
    scene_values = batch.get("target_scenes")
    if not isinstance(profile_values, list) or not profile_values:
        raise ValueError("batch.profile_ids must be a non-empty list")
    if not isinstance(scene_values, list) or not scene_values:
        raise ValueError("batch.target_scenes must be a non-empty list")
    if len(profile_values) > 128 or len(scene_values) > 64:
        raise ValueError("batch exceeds the profile or target Scene limit")
    profiles = [
        _identifier(value, f"batch.profile_ids[{index}]")
        for index, value in enumerate(profile_values)
    ]
    scenes = [
        _scene_name(value, f"batch.target_scenes[{index}]")
        for index, value in enumerate(scene_values)
    ]
    if len(profiles) != len(set(profiles)) or len(scenes) != len(set(scenes)):
        raise ValueError("batch profile_ids and target_scenes must not contain duplicates")
    pairing = str(batch.get("pairing", "PAIRWISE"))
    if pairing not in {"CROSS_PRODUCT", "PAIRWISE"}:
        raise ValueError("batch.pairing must be CROSS_PRODUCT or PAIRWISE")
    if pairing == "PAIRWISE" and len(profiles) != len(scenes):
        raise ValueError("PAIRWISE requires equal profile_ids and target_scenes lengths")
    if pairing == "CROSS_PRODUCT" and len(profiles) != 1:
        raise ValueError(
            "CROSS_PRODUCT supports exactly one profile_id because each compiled Scene "
            "owns one profile; use PAIRWISE for multiple profiles"
        )
    selected_frames = _frames(batch.get("frames"))
    settings = _render_settings(batch.get("render_settings", {}))
    root = _resolved_output_root(batch.get("output_root"))
    template = batch.get("filename_template", "{scene}-{profile}-{frame}")
    if not isinstance(template, str) or not _SAFE_TEMPLATE.fullmatch(template):
        raise ValueError("batch.filename_template contains unsupported characters")
    fields = set(re.findall(r"\{([^{}]+)\}", template))
    if not fields or not fields <= {"scene", "profile", "camera", "frame"}:
        raise ValueError("filename_template must use only scene, profile, camera, and frame fields")
    include_review_packet = _boolean(
        batch.get("include_review_packet", False), "batch.include_review_packet"
    )
    raw_camera_names = batch.get("camera_names")
    if raw_camera_names is None:
        requested_camera_names = None
    else:
        if not isinstance(raw_camera_names, list) or not raw_camera_names:
            raise ValueError("batch.camera_names must be omitted or a non-empty list")
        if len(raw_camera_names) > 32:
            raise ValueError("batch.camera_names may contain at most 32 cameras")
        requested_camera_names = [
            _scene_name(value, f"batch.camera_names[{index}]")
            for index, value in enumerate(raw_camera_names)
        ]
        if len(requested_camera_names) != len(set(requested_camera_names)):
            raise ValueError("batch.camera_names must not contain duplicates")
        if len(requested_camera_names) > 1 and "camera" not in fields:
            raise ValueError("multi-camera batches require {camera} in filename_template")
    continue_on_error = _boolean(batch.get("continue_on_error", False), "batch.continue_on_error")
    pairs = (
        list(zip(profiles, scenes, strict=True))
        if pairing == "PAIRWISE"
        else [(profile, scene) for profile in profiles for scene in scenes]
    )
    if expected_revision is not None and len(pairs) != 1:
        raise ValueError(
            "Scalar expected_profile_revision is valid only for one profile/Scene pair; "
            "use expected_profile_revisions keyed by target Scene"
        )
    normalized_expected_revisions: dict[str, int] | None = None
    if expected_revisions is not None:
        normalized_expected_revisions = {}
        for raw_name, raw_revision in expected_revisions.items():
            scene_key = _scene_name(raw_name, "expected_profile_revisions key")
            normalized_expected_revisions[scene_key] = _integer(
                raw_revision,
                f"expected_profile_revisions[{scene_key!r}]",
                0,
                2**63 - 1,
            )
        if set(normalized_expected_revisions) != set(scenes):
            raise ValueError(
                "expected_profile_revisions must contain exactly one entry for every target Scene"
            )
    camera_count = len(requested_camera_names) if requested_camera_names is not None else 1
    count = len(pairs) * camera_count * len(selected_frames)
    if count > MAX_RENDER_ITEMS:
        raise ValueError(f"batch expands to {count} items; maximum is {MAX_RENDER_ITEMS}")

    items: list[dict[str, Any]] = []
    output_paths: set[str] = set()
    for profile_id, target_scene in pairs:
        item_expected_revision = (
            normalized_expected_revisions[target_scene]
            if normalized_expected_revisions is not None
            else expected_revision
        )
        scene, _entry, provenance = _profile_scene(
            target_scene,
            profile_id,
            item_expected_revision,
        )
        _view_layer_name(scene)
        active_camera = getattr(scene, "camera", None)
        if active_camera is None:
            raise ValueError(f"Compiled profile Scene '{target_scene}' has no active camera")
        scene_objects = getattr(scene, "objects", None)
        camera_names = requested_camera_names or [str(getattr(active_camera, "name", ""))]
        cameras: list[Any] = []
        for camera_name in camera_names:
            camera = (
                active_camera
                if getattr(active_camera, "name", None) == camera_name
                else _lookup(scene_objects, camera_name)
            )
            if camera is None or str(getattr(camera, "type", "CAMERA")) != "CAMERA":
                raise ValueError(
                    f"Camera '{camera_name}' is not a camera object in Scene '{target_scene}'"
                )
            cameras.append(camera)
        for camera in cameras:
            camera_name = str(getattr(camera, "name", ""))
            technical_check = None
            if include_review_packet:
                technical_check = handle_inspect_look_review_state(
                    {
                        "profile_id": profile_id,
                        "target_scene": target_scene,
                        "camera": camera_name,
                        "expected_profile_revision": provenance["profile_revision"],
                        "expected_geometry_revision": provenance["geometry_revision"],
                    }
                )
                if technical_check["status"] == "FAIL":
                    codes = [item["code"] for item in technical_check["failures"]]
                    raise ValueError(
                        f"Review-state audit failed for Scene '{target_scene}', camera "
                        f"'{camera_name}': {codes}"
                    )
            for frame in selected_frames:
                output = _output_path(
                    root,
                    template,
                    scene=target_scene,
                    profile=profile_id,
                    camera=camera_name,
                    frame=frame,
                    file_format=settings["file_format"],
                )
                output_key = os.path.normcase(str(output))
                if output_key in output_paths:
                    raise ValueError(f"batch resolves multiple items to '{output}'")
                output_paths.add(output_key)
                if output.exists() and settings["existing_file_policy"] == "ERROR":
                    raise FileExistsError(f"Render output already exists: '{output}'")
                status = "QUEUED"
                warnings: list[str] = []
                completed_at = None
                if output.exists() and settings["existing_file_policy"] == "SKIP":
                    status = "SKIPPED"
                    warnings.append(
                        "Existing output preserved but is unproven and not acceptable "
                        "as a managed Composite result"
                    )
                    completed_at = _utc_now()
                items.append(
                    {
                        "profile_id": profile_id,
                        "target_scene": target_scene,
                        "expected_profile_revision": provenance["profile_revision"],
                        "expected_provenance": provenance,
                        "camera": camera_name,
                        "frame": frame,
                        "output_path": str(output),
                        "status": status,
                        "error": None,
                        "warnings": warnings,
                        "metadata": None,
                        "proxy_path": None,
                        "review_packet_path": None,
                        "technical_check": technical_check,
                        "started_at": None,
                        "completed_at": completed_at,
                    }
                )
    return {
        "items": items,
        "output_root": str(root),
        "render_settings": settings,
        "continue_on_error": continue_on_error,
        "include_review_packet": include_review_packet,
    }


def _stable_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return round(value, 12) if math.isfinite(value) else str(value)
    if isinstance(value, (list, tuple)):
        return [_stable_value(item) for item in value]
    to_list = getattr(value, "to_list", None)
    if callable(to_list):
        try:
            return [_stable_value(item) for item in to_list()]
        except (ReferenceError, TypeError):
            pass
    name = getattr(value, "name", None)
    return str(name) if isinstance(name, str) else type(value).__name__


def _sample_owner(scene: Any) -> tuple[Any | None, str | None]:
    engine = str(getattr(getattr(scene, "render", None), "engine", ""))
    if engine == "CYCLES":
        owner = getattr(scene, "cycles", None)
        return (
            (owner, "samples") if owner is not None and hasattr(owner, "samples") else (None, None)
        )
    owner = getattr(scene, "eevee", None)
    if owner is not None:
        for attribute in ("taa_render_samples", "render_samples", "taa_samples"):
            if hasattr(owner, attribute):
                return owner, attribute
    return None, None


def _denoise_owner(scene: Any) -> tuple[Any | None, str | None]:
    owner = getattr(scene, "cycles", None)
    if owner is not None and hasattr(owner, "use_denoising"):
        return owner, "use_denoising"
    return None, None


def _set_recorded(changes: list[tuple[Any, str, Any]], owner: Any, name: str, value: Any) -> None:
    changes.append((owner, name, getattr(owner, name)))
    setattr(owner, name, value)


def _redirect_file_outputs(tree: Any, root: Path, result_id: str) -> list[dict[str, Any]]:
    """Redirect version-specific File Output nodes using the shared adapter."""
    sidecar = (root / ".blend_ai_file_outputs" / result_id).resolve(strict=False)
    sidecar.relative_to(root)
    sidecar.mkdir(parents=True, exist_ok=True)
    return look_compositor.redirect_file_outputs(tree, str(sidecar))


def _review_sidecar(root: Path, result_id: str) -> Path:
    path = (root / ".blend_ai_review" / result_id).resolve(strict=False)
    path.relative_to(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _configure_review_file_output(
    tree: Any,
    source_socket: Any,
    sidecar: Path,
    slug: str,
) -> Any:
    node = tree.nodes.new("CompositorNodeOutputFile")
    node.name = f"BLEND_AI_REVIEW_{slug}"
    node.label = f"Transient blend-ai review: {slug}"
    if hasattr(node, "directory") and hasattr(node, "file_output_items"):
        node.directory = str(sidecar)
        node.file_name = f"{slug}_"
        node.file_output_items.clear()
        item = node.file_output_items.new(socket_type="RGBA", name="Review")
        item.override_node_format = True
        item.format.file_format = "PNG"
        item.format.color_mode = "RGBA"
        item.format.color_depth = "8"
        item.save_as_render = True
        socket = node.inputs.get("Review")
    else:
        node.base_path = str(sidecar)
        node.format.file_format = "PNG"
        node.format.color_mode = "RGBA"
        node.format.color_depth = "8"
        node.file_slots[0].path = f"{slug}_"
        socket = node.inputs[0]
    if socket is None:
        tree.nodes.remove(node)
        raise RuntimeError(f"Review File Output for '{slug}' has no image input")
    tree.links.new(source_socket, socket)
    return node


def _bind_review_render_layers_scene(render_layers: Any, scene: Any) -> None:
    """Keep diagnostic passes on the already-rendered compiled profile Scene."""
    if not hasattr(render_layers, "scene"):
        raise RuntimeError("Review Render Layers node has no explicit Scene binding")
    try:
        render_layers.scene = scene
    except Exception as exc:
        raise RuntimeError(
            "Unable to bind the review Render Layers node to the compiled profile Scene"
        ) from exc
    if getattr(render_layers, "scene", None) is not scene:
        raise RuntimeError(
            "Review Render Layers node did not retain the compiled profile Scene binding"
        )


def _prepare_review_nodes(
    scene: Any,
    tree: Any,
    root: Path,
    result_id: str,
    changes: list[tuple[Any, str, Any]],
) -> dict[str, Any]:
    view_layers = [
        layer
        for layer in _collection_values(getattr(scene, "view_layers", None))
        if bool(getattr(layer, "use", True))
    ]
    if len(view_layers) != 1:
        raise RuntimeError("Review packet requires exactly one enabled View Layer")
    view_layer = view_layers[0]
    warnings: list[str] = []
    for attribute in (
        "use_pass_z",
        "use_pass_diffuse_direct",
        "use_pass_glossy_direct",
        "use_pass_glossy_indirect",
        "use_pass_emit",
        "use_pass_cryptomatte_object",
        "use_pass_cryptomatte_accurate",
    ):
        if hasattr(view_layer, attribute):
            _set_recorded(changes, view_layer, attribute, True)
        else:
            warnings.append(f"View Layer does not support {attribute}")

    created: list[Any] = []
    sidecar = _review_sidecar(root, result_id)
    render_layers = tree.nodes.new("CompositorNodeRLayers")
    render_layers.name = f"BLEND_AI_REVIEW_LAYERS_{result_id[-12:]}"
    render_layers.label = "Transient blend-ai diagnostic passes"
    _bind_review_render_layers_scene(render_layers, scene)
    if hasattr(render_layers, "layer"):
        render_layers.layer = str(getattr(view_layer, "name", ""))
    created.append(render_layers)

    def output(name: str) -> Any | None:
        return render_layers.outputs.get(name)

    def color_mix(name: str, blend_type: str) -> tuple[Any, Any, Any, Any]:
        try:
            node = tree.nodes.new("ShaderNodeMix")
            node.data_type = "RGBA"
            node.blend_type = blend_type
            node.inputs[0].default_value = 1.0
            return node, node.inputs[6], node.inputs[7], node.outputs[2]
        except Exception:
            node = tree.nodes.new("CompositorNodeMixRGB")
            node.blend_type = blend_type
            node.inputs[0].default_value = 1.0
            return node, node.inputs[1], node.inputs[2], node.outputs[0]

    sockets: dict[str, Any | None] = {
        "combined": output("Image"),
        "diffuse_direct": output("Diffuse Direct"),
        "emission": output("Emission"),
    }
    glossy_direct = output("Glossy Direct")
    glossy_indirect = output("Glossy Indirect")
    if glossy_direct is not None and glossy_indirect is not None:
        mix, mix_a, mix_b, mix_output = color_mix("glossy", "ADD")
        mix.name = f"BLEND_AI_REVIEW_GLOSSY_{result_id[-12:]}"
        tree.links.new(glossy_direct, mix_a)
        tree.links.new(glossy_indirect, mix_b)
        created.append(mix)
        sockets["glossy"] = mix_output
    else:
        sockets["glossy"] = glossy_direct
        if glossy_direct is not None:
            warnings.append("Glossy Indirect is unavailable; the glossy tile is direct-only")

    depth = output("Depth")
    if depth is not None:
        camera = getattr(scene, "camera", None)
        camera_data = getattr(camera, "data", None)
        dof = getattr(camera_data, "dof", None)
        focus_distance = float(getattr(dof, "focus_distance", 0.0) or 0.0)
        clip_start = float(getattr(camera_data, "clip_start", 0.1) or 0.1)
        if not math.isfinite(clip_start) or clip_start <= 0.0:
            clip_start = 0.1
        clip_end = float(getattr(camera_data, "clip_end", 1000.0) or 1000.0)
        if not math.isfinite(clip_end) or clip_end <= clip_start:
            clip_end = max(1000.0, clip_start * 1000.0)
        reference_distance = (
            focus_distance
            if math.isfinite(focus_distance) and focus_distance > clip_start
            else math.sqrt(clip_start * clip_end)
        )
        depth_scale = max(clip_start, reference_distance * 0.25)
        try:
            add = tree.nodes.new("ShaderNodeMath")
            add.name = f"BLEND_AI_REVIEW_DEPTH_ADD_{result_id[-12:]}"
            add.operation = "ADD"
            add.inputs[1].default_value = depth_scale
            tree.links.new(depth, add.inputs[0])
            created.append(add)

            divide = tree.nodes.new("ShaderNodeMath")
            divide.name = f"BLEND_AI_REVIEW_DEPTH_DIVIDE_{result_id[-12:]}"
            divide.operation = "DIVIDE"
            divide.inputs[0].default_value = depth_scale
            tree.links.new(add.outputs[0], divide.inputs[1])
            created.append(divide)

            ramp = tree.nodes.new("ShaderNodeValToRGB")
            ramp.name = f"BLEND_AI_REVIEW_DEPTH_RAMP_{result_id[-12:]}"
            ramp.color_ramp.interpolation = "LINEAR"
            ramp.color_ramp.elements[0].position = 0.0
            ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
            ramp.color_ramp.elements[1].position = 1.0
            ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
            tree.links.new(divide.outputs[0], ramp.inputs[0])
            created.append(ramp)
            sockets["depth"] = ramp.outputs.get("Color")
        except Exception as exc:
            sockets["depth"] = None
            warnings.append(f"Camera Depth normalization is unavailable: {exc}")
    else:
        sockets["depth"] = None
        warnings.append("Render Layers Depth output is unavailable")

    unavailable_node, unavailable_a, unavailable_b, unavailable_output = color_mix(
        "unavailable", "MULTIPLY"
    )
    unavailable_node.name = f"BLEND_AI_REVIEW_UNAVAILABLE_{result_id[-12:]}"
    unavailable_b.default_value = (0.0, 0.0, 0.0, 1.0)
    combined = sockets.get("combined")
    if combined is None:
        raise RuntimeError("Render Layers node has no Combined/Image output")
    tree.links.new(combined, unavailable_a)
    created.append(unavailable_node)
    unavailable: list[str] = []

    crypto = None
    try:
        crypto = tree.nodes.new("CompositorNodeCryptomatteV2")
        crypto.name = f"BLEND_AI_REVIEW_CRYPTO_{result_id[-12:]}"
        crypto.source = "RENDER"
        layer_name = f"{getattr(view_layer, 'name', '')}.CryptoObject"
        if hasattr(crypto, "layer_name"):
            crypto.layer_name = layer_name
        created.append(crypto)
        sockets["cryptomatte_object"] = crypto.outputs.get("Pick")
    except Exception as exc:
        if crypto is not None and crypto not in created:
            created.append(crypto)
        sockets["cryptomatte_object"] = None
        warnings.append(f"Object Cryptomatte Pick output is unavailable: {exc}")
        unavailable.append("cryptomatte_object")

    prefixes: dict[str, str] = {}
    for slug, _label in REVIEW_TILE_ORDER:
        socket = sockets.get(slug)
        if socket is None:
            unavailable.append(slug)
            socket = unavailable_output
            warnings.append(f"Diagnostic pass '{slug}' is unavailable")
        node = _configure_review_file_output(tree, socket, sidecar, slug)
        created.append(node)
        prefixes[slug] = f"{slug}_"
    return {
        "nodes": created,
        "sidecar": str(sidecar),
        "prefixes": prefixes,
        "warnings": warnings,
        "unavailable": sorted(set(unavailable)),
    }


def _review_tile_paths(prepared: dict[str, Any]) -> dict[str, Path | None]:
    review = prepared.get("review")
    if not isinstance(review, dict):
        return {}
    sidecar = Path(review["sidecar"])
    result: dict[str, Path | None] = {}
    for slug, prefix in review["prefixes"].items():
        candidates = sorted(
            path for path in sidecar.rglob("*") if path.is_file() and path.name.startswith(prefix)
        )
        if len(candidates) > 1:
            raise RuntimeError(
                f"Review pass '{slug}' produced {len(candidates)} files: {candidates}"
            )
        result[slug] = candidates[0] if candidates else None
    return result


def _view_layer_name(scene: Any) -> str:
    layers = _collection_values(getattr(scene, "view_layers", None))
    enabled = [layer for layer in layers if bool(getattr(layer, "use", True))]
    if len(enabled) != 1:
        names = [str(getattr(layer, "name", "")) for layer in enabled]
        raise ValueError(
            "Compiled profile Scene must have exactly one enabled View Layer for "
            f"unambiguous Composite provenance; found {names}"
        )
    return str(getattr(enabled[0], "name", ""))


def _item_camera(scene: Any, name: str) -> Any:
    active = getattr(scene, "camera", None)
    camera = (
        active
        if getattr(active, "name", None) == name
        else _lookup(getattr(scene, "objects", None), name)
    )
    if camera is None or str(getattr(camera, "type", "CAMERA")) != "CAMERA":
        raise ValueError(f"Camera '{name}' is not a camera object in Scene '{scene.name}'")
    return camera


def _color_settings(scene: Any) -> dict[str, Any]:
    view = getattr(scene, "view_settings", None)
    return {
        name: _stable_value(getattr(view, name, None))
        for name in ("view_transform", "look", "exposure", "gamma")
    }


def _find_layer_collection(layer_collection: Any, collection_name: str) -> Any | None:
    collection = getattr(layer_collection, "collection", None)
    if str(getattr(collection, "name", "")) == collection_name:
        return layer_collection
    for child in _collection_values(getattr(layer_collection, "children", None)):
        found = _find_layer_collection(child, collection_name)
        if found is not None:
            return found
    return None


def _review_intent(entry: dict[str, Any]) -> dict[str, Any] | None:
    definition = entry.get("definition")
    value = definition.get("review_intent") if isinstance(definition, dict) else None
    return dict(value) if isinstance(value, dict) else None


def _editable_review_controls(entry: dict[str, Any]) -> dict[str, Any]:
    definition = entry.get("definition")
    if not isinstance(definition, dict):
        return {}
    controls: dict[str, Any] = {}
    lighting = definition.get("lighting")
    if isinstance(lighting, dict):
        for light in lighting.get("lights", []):
            if not isinstance(light, dict) or not isinstance(light.get("id"), str):
                continue
            prefix = f"lighting.lights.{light['id']}"
            for field in (
                "energy",
                "color_rgb",
                "target_point",
                "size",
                "size_y",
                "radius",
                "sun_angle_degrees",
            ):
                if light.get(field) is not None:
                    controls[f"{prefix}.{field}"] = _stable_value(light[field])
    world = definition.get("world")
    if isinstance(world, dict) and world.get("strength") is not None:
        controls["world.strength"] = _stable_value(world["strength"])
    for component in definition.get("atmosphere", []):
        if (
            isinstance(component, dict)
            and isinstance(component.get("id"), str)
            and component.get("density") is not None
        ):
            controls[f"atmosphere.{component['id']}.density"] = _stable_value(component["density"])
    color = definition.get("color_management")
    if isinstance(color, dict) and color.get("exposure") is not None:
        controls["color_management.exposure"] = _stable_value(color["exposure"])
    post = definition.get("post")
    if isinstance(post, dict):
        for group, fields in (
            ("bloom", ("threshold", "strength", "radius")),
            ("grain", ("strength", "scale")),
            ("vignette", ("strength", "feather")),
        ):
            values = post.get(group)
            if not isinstance(values, dict):
                continue
            for field in fields:
                if values.get(field) is not None:
                    controls[f"post.{group}.{field}"] = _stable_value(values[field])
    return dict(sorted(controls.items()))


def handle_inspect_look_review_state(params: dict[str, Any]) -> dict[str, Any]:
    """Return hard renderer failures and non-mutating artist-content warnings."""
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    _reject_unknown(
        params,
        {
            "profile_id",
            "target_scene",
            "camera",
            "expected_profile_revision",
            "expected_geometry_revision",
        },
        "params",
    )
    profile_id = _identifier(params.get("profile_id"), "profile_id")
    target_scene = _scene_name(params.get("target_scene"), "target_scene")
    camera_name = _scene_name(params.get("camera"), "camera")
    expected_revision = params.get("expected_profile_revision")
    if expected_revision is not None:
        expected_revision = _integer(expected_revision, "expected_profile_revision", 0, 2**63 - 1)
    expected_geometry = params.get("expected_geometry_revision")
    if expected_geometry is not None:
        expected_geometry = _integer(expected_geometry, "expected_geometry_revision", 0, 2**63 - 1)
    scene, entry, provenance = _profile_scene(target_scene, profile_id, expected_revision)
    if expected_geometry is not None and provenance["geometry_revision"] != expected_geometry:
        raise ValueError(
            f"Scene '{target_scene}' geometry revision "
            f"{provenance['geometry_revision']} does not match expected revision "
            f"{expected_geometry}"
        )
    _item_camera(scene, camera_name)
    intent = _review_intent(entry)
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    def issue(target: list[dict[str, Any]], code: str, message: str, **evidence: Any) -> None:
        target.append({"code": code, "message": message, "evidence": evidence})

    if intent is None:
        issue(
            failures,
            "MISSING_REVIEW_INTENT",
            "Profile has no persisted review_intent",
        )
        intent = {
            "expects_shadows": False,
            "expects_reflections": False,
            "expects_volume": False,
            "expects_compositing": False,
        }

    render = getattr(scene, "render", None)
    if intent.get("expects_compositing", True) and not bool(
        getattr(render, "use_compositing", False)
    ):
        issue(
            failures,
            "COMPOSITING_DISABLED",
            "The profile expects compositing but Scene.render.use_compositing is false",
        )

    inventory = entry.get("managed_inventory")
    inventory = inventory if isinstance(inventory, dict) else {}
    for light in inventory.get("lights", []):
        if not isinstance(light, dict):
            continue
        object_name = str(light.get("object_name", ""))
        obj = _lookup(getattr(scene, "objects", None), object_name)
        data = getattr(obj, "data", None) if obj is not None else None
        if data is None:
            issue(
                failures,
                "MANAGED_LIGHT_MISSING",
                f"Managed light '{object_name}' is missing from the compiled Scene",
                light=object_name,
            )
        elif intent.get("expects_shadows", True) and not bool(getattr(data, "use_shadow", True)):
            issue(
                failures,
                "MANAGED_LIGHT_SHADOWS_DISABLED",
                f"Managed light '{object_name}' has use_shadow=false",
                light=object_name,
            )

    renderable_types = {"MESH", "CURVE", "SURFACE", "META", "FONT", "VOLUME"}
    renderable = [
        obj
        for obj in _collection_values(getattr(scene, "objects", None))
        if str(getattr(obj, "type", "")) in renderable_types
        and not bool(getattr(obj, "hide_render", False))
    ]
    shadow_disabled = [
        str(getattr(obj, "name", ""))
        for obj in renderable
        if hasattr(obj, "visible_shadow") and not bool(getattr(obj, "visible_shadow"))
    ]
    if intent.get("expects_shadows", True) and renderable:
        if len(shadow_disabled) == len(renderable):
            issue(
                failures,
                "ALL_CASTERS_DISABLED",
                "Every renderable object has visible_shadow=false",
                objects=shadow_disabled,
            )
        elif shadow_disabled:
            issue(
                warnings,
                "OBJECT_SHADOW_VISIBILITY_DISABLED",
                "Some artist-owned objects have visible_shadow=false",
                objects=shadow_disabled,
            )

    glossy_disabled = [
        str(getattr(obj, "name", ""))
        for obj in renderable
        if hasattr(obj, "visible_glossy") and not bool(getattr(obj, "visible_glossy"))
    ]
    engine = str(getattr(render, "engine", ""))
    if intent.get("expects_reflections", True):
        cycles = getattr(scene, "cycles", None)
        if (
            engine == "CYCLES"
            and hasattr(cycles, "glossy_bounces")
            and int(getattr(cycles, "glossy_bounces")) <= 0
        ):
            issue(
                failures,
                "GLOSSY_TRANSPORT_DISABLED",
                "Cycles glossy_bounces is zero",
            )
        eevee = getattr(scene, "eevee", None)
        if (
            engine.startswith("BLENDER_EEVEE")
            and hasattr(eevee, "light_path_glossy_intensity")
            and float(getattr(eevee, "light_path_glossy_intensity")) <= 0.0
        ):
            issue(
                failures,
                "GLOSSY_TRANSPORT_DISABLED",
                "Eevee glossy light-path intensity is zero",
            )
        if renderable and len(glossy_disabled) == len(renderable):
            issue(
                failures,
                "ALL_REFLECTION_VISIBILITY_DISABLED",
                "Every renderable object has visible_glossy=false",
                objects=glossy_disabled,
            )
        elif glossy_disabled:
            issue(
                warnings,
                "OBJECT_REFLECTION_VISIBILITY_DISABLED",
                "Some artist-owned objects have visible_glossy=false",
                objects=glossy_disabled,
            )

    layers = [
        layer
        for layer in _collection_values(getattr(scene, "view_layers", None))
        if bool(getattr(layer, "use", True))
    ]
    collection_name = str(entry.get("collection_name", ""))
    if len(layers) == 1:
        layer_collection = _find_layer_collection(
            getattr(layers[0], "layer_collection", None), collection_name
        )
        if layer_collection is None:
            issue(
                failures,
                "PROFILE_COLLECTION_NOT_IN_VIEW_LAYER",
                "The managed profile Collection is absent from the active View Layer",
                collection=collection_name,
            )
        elif bool(getattr(layer_collection, "exclude", False)):
            issue(
                failures,
                "PROFILE_COLLECTION_EXCLUDED",
                "The managed profile Collection is excluded from the active View Layer",
                collection=collection_name,
            )

    return {
        "status": "FAIL" if failures else "WARN" if warnings else "PASS",
        "profile_id": profile_id,
        "target_scene": target_scene,
        "camera": camera_name,
        "review_intent": intent,
        "profile_revision": provenance["profile_revision"],
        "geometry_revision": provenance["geometry_revision"],
        "compositor_hash": provenance["compositor_hash"],
        "failures": failures,
        "warnings": warnings,
    }


def _prepare_render(batch: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    root = Path(batch["output_root"]).resolve(strict=True)
    output = Path(item["output_path"]).resolve(strict=False)
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("Render output no longer resolves beneath output_root") from exc
    scene, _entry, manifest_provenance = _assert_item_provenance(
        item,
        phase="render start",
    )
    tree = look_compositor.get_compositor_tree(scene)
    if tree is None:
        raise RuntimeError(f"Scene '{scene.name}' has no managed compositor")
    render = getattr(scene, "render", None)
    image_settings = getattr(render, "image_settings", None)
    if render is None or image_settings is None:
        raise RuntimeError(f"Scene '{scene.name}' has no render image settings")
    if not hasattr(render, "use_compositing"):
        raise RuntimeError(f"Scene '{scene.name}' cannot enable compositor rendering")
    settings = batch["render_settings"]
    sample_owner, sample_attribute = _sample_owner(scene)
    if sample_owner is None or sample_attribute is None:
        raise RuntimeError(f"Scene '{scene.name}' has no supported final-render sample setting")
    denoise_owner, denoise_attribute = _denoise_owner(scene)
    changes: list[tuple[Any, str, Any]] = []
    file_output_snapshots: list[dict[str, Any]] | None = None
    review: dict[str, Any] | None = None
    frame = (int(getattr(scene, "frame_current", 1)), float(getattr(scene, "frame_subframe", 0.0)))
    try:
        _set_recorded(changes, render, "filepath", item["output_path"])
        _set_recorded(changes, image_settings, "file_format", settings["file_format"])
        _set_recorded(changes, image_settings, "color_depth", settings["color_depth"])
        _set_recorded(changes, render, "resolution_percentage", settings["resolution_percentage"])
        _set_recorded(changes, render, "use_compositing", True)
        if hasattr(render, "use_lock_interface"):
            _set_recorded(changes, render, "use_lock_interface", True)
        _set_recorded(changes, sample_owner, sample_attribute, settings["samples"])
        camera = _item_camera(scene, item["camera"])
        if getattr(scene, "camera", None) is not camera:
            _set_recorded(changes, scene, "camera", camera)
        if denoise_owner is not None and denoise_attribute is not None:
            _set_recorded(changes, denoise_owner, denoise_attribute, settings["denoise"])
        if frame != (item["frame"], 0.0):
            frame_set = getattr(scene, "frame_set", None)
            if callable(frame_set):
                frame_set(item["frame"], subframe=0.0)
            else:
                scene.frame_current = item["frame"]
        file_output_snapshots = _redirect_file_outputs(tree, root, item["result_id"])
        if batch.get("include_review_packet", False):
            review = _prepare_review_nodes(
                scene,
                tree,
                root,
                item["result_id"],
                changes,
            )
            item["warnings"].extend(review["warnings"])
        effective_samples = int(getattr(sample_owner, sample_attribute))
        return {
            "scene": scene,
            "tree": tree,
            "changes": changes,
            "file_output_snapshots": file_output_snapshots,
            "frame": frame,
            "review": review,
            "provenance": {
                **manifest_provenance,
                "scene": str(getattr(scene, "name", "")),
                "view_layer": _view_layer_name(scene),
                "camera": str(getattr(camera, "name", "")),
                "frame": item["frame"],
                "engine": str(getattr(render, "engine", "")),
                "samples": effective_samples,
                "denoise": (
                    bool(getattr(denoise_owner, denoise_attribute))
                    if denoise_owner is not None and denoise_attribute is not None
                    else None
                ),
                "color_management": _color_settings(scene),
                "image_source": "COMPOSITE_OUTPUT",
            },
        }
    except Exception:
        _restore_prepared(
            {
                "scene": scene,
                "tree": tree,
                "changes": changes,
                "file_output_snapshots": file_output_snapshots,
                "frame": frame,
                "review": review,
            }
        )
        raise


def _restore_prepared(prepared: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    review = prepared.get("review")
    tree = prepared.get("tree")
    if isinstance(review, dict) and tree is not None:
        for node in reversed(review.get("nodes", [])):
            try:
                tree.nodes.remove(node)
            except (ReferenceError, RuntimeError, TypeError, ValueError) as exc:
                errors.append(f"review node: {exc}")
    snapshots = prepared.get("file_output_snapshots")
    if snapshots is not None and tree is not None:
        try:
            look_compositor.restore_file_outputs(tree, snapshots)
        except Exception as exc:
            errors.append(f"File Output nodes: {exc}")
    for owner, attribute, value in reversed(prepared.get("changes", [])):
        try:
            setattr(owner, attribute, value)
        except Exception as exc:
            errors.append(f"{attribute}: {exc}")
    scene = prepared.get("scene")
    old_frame = prepared.get("frame")
    if scene is not None and old_frame is not None:
        try:
            current_frame = (
                int(getattr(scene, "frame_current", old_frame[0])),
                float(getattr(scene, "frame_subframe", old_frame[1])),
            )
            if current_frame != old_frame:
                frame_set = getattr(scene, "frame_set", None)
                if callable(frame_set):
                    frame_set(old_frame[0], subframe=old_frame[1])
                else:
                    scene.frame_current = old_frame[0]
        except Exception as exc:
            errors.append(f"frame: {exc}")
    return errors


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _png_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        width, height = struct.unpack(">II", data[16:24])
        return (width, height) if width > 0 and height > 0 else None
    return None


def _image_dimensions(path: Path, scene: Any) -> tuple[int, int]:
    try:
        with path.open("rb") as handle:
            header = handle.read(32)
        dimensions = _png_dimensions(header)
        if dimensions is not None:
            return dimensions
    except OSError:
        pass
    images = getattr(getattr(bpy, "data", None), "images", None)
    loader = getattr(images, "load", None)
    image = None
    if callable(loader):
        try:
            image = loader(str(path), check_existing=False)
            size = tuple(int(value) for value in getattr(image, "size", (0, 0)))
            if len(size) >= 2 and size[0] > 0 and size[1] > 0:
                return size[0], size[1]
        except Exception:
            pass
        finally:
            remover = getattr(images, "remove", None)
            if image is not None and callable(remover):
                try:
                    remover(image)
                except Exception:
                    pass
    render = getattr(scene, "render", None)
    width = int(getattr(render, "resolution_x", 0))
    height = int(getattr(render, "resolution_y", 0))
    percentage = int(getattr(render, "resolution_percentage", 100))
    width = max(1, round(width * percentage / 100))
    height = max(1, round(height * percentage / 100))
    return width, height


def _image_pixel_metrics(path: Path) -> dict[str, Any]:
    images = getattr(getattr(bpy, "data", None), "images", None)
    loader = getattr(images, "load", None)
    remover = getattr(images, "remove", None)
    if not callable(loader):
        raise RuntimeError("Blender image loading is unavailable for review metrics")
    image = None
    try:
        image = loader(str(path), check_existing=False)
        width, height = (int(value) for value in getattr(image, "size", (0, 0)))
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Review image '{path}' has invalid dimensions")
        pixels = getattr(image, "pixels", None)
        if pixels is None:
            raise RuntimeError(f"Review image '{path}' has no pixel buffer")
        values = array("f", [0.0]) * (width * height * 4)
        foreach_get = getattr(pixels, "foreach_get", None)
        if callable(foreach_get):
            foreach_get(values)
        else:
            values = array("f", pixels[:])
        pixel_count = width * height
        stride = max(1, pixel_count // 65_536)
        luminance_sum = 0.0
        nonzero = 0
        magenta = 0
        sampled = 0
        for pixel in range(0, pixel_count, stride):
            index = pixel * 4
            red = max(0.0, float(values[index]))
            green = max(0.0, float(values[index + 1]))
            blue = max(0.0, float(values[index + 2]))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            luminance_sum += luminance
            nonzero += int(luminance > 1e-5)
            magenta += int(red > 0.7 and blue > 0.7 and green < 0.35)
            sampled += 1
        return {
            "width": width,
            "height": height,
            "mean_energy": round(luminance_sum / max(1, sampled), 8),
            "nonzero_coverage": round(nonzero / max(1, sampled), 8),
            "magenta_coverage": round(magenta / max(1, sampled), 8),
        }
    finally:
        if image is not None and callable(remover):
            try:
                remover(image)
            except Exception:
                pass


def _convert_review_tile(source: Path, destination: Path, slug: str) -> None:
    """Convert Blender 5.2 File Output EXR into a bounded display PNG."""
    try:
        import OpenImageIO as oiio
    except ImportError as exc:
        raise RuntimeError("Blender's OpenImageIO module is required for review tiles") from exc
    image = oiio.ImageBuf(str(source))
    spec = image.spec()
    if image.has_error or spec.width <= 0 or spec.height <= 0:
        raise RuntimeError(f"Review pass '{slug}' is not a readable EXR: {image.geterror()}")
    working = image
    if slug == "depth":
        cleaned = oiio.ImageBuf()
        if not oiio.ImageBufAlgo.median_filter(cleaned, image, 3, 3):
            raise RuntimeError(f"Review pass '{slug}' median filter failed: {cleaned.geterror()}")
        working = cleaned
    elif slug != "cryptomatte_object":
        stats = oiio.ImageBufAlgo.computePixelStats(image)
        maximum = max(float(value) for value in stats.max[: min(3, len(stats.max))])
        scale = 1.0 / max(1e-6, maximum)
        normalized = oiio.ImageBuf()
        factors = tuple([scale] * min(3, spec.nchannels) + [1.0] * max(0, spec.nchannels - 3))
        if not oiio.ImageBufAlgo.mul(normalized, image, factors):
            raise RuntimeError(
                f"Review pass '{slug}' normalization failed: {normalized.geterror()}"
            )
        working = normalized
    display = oiio.ImageBuf()
    if not oiio.ImageBufAlgo.colorconvert(display, working, "linear", "sRGB"):
        raise RuntimeError(f"Review pass '{slug}' color conversion failed: {display.geterror()}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not display.write(str(destination), oiio.UINT8):
        raise RuntimeError(f"Review pass '{slug}' PNG write failed: {display.geterror()}")
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise RuntimeError(f"Review pass '{slug}' PNG was not written")
    if destination.stat().st_size > MAX_REVIEW_TILE_BYTES:
        raise RuntimeError(f"Review pass '{slug}' PNG exceeds the byte limit")


def _write_unavailable_review_tile(destination: Path, width: int, height: int) -> None:
    """Write an explicit black placeholder for a pass the engine did not emit."""
    try:
        import OpenImageIO as oiio
    except ImportError as exc:
        raise RuntimeError("Blender's OpenImageIO module is required for review tiles") from exc
    if width <= 0 or height <= 0 or width * height > 100_000_000:
        raise RuntimeError("Unavailable review tile dimensions are invalid")
    image = oiio.ImageBuf(oiio.ImageSpec(width, height, 4, oiio.UINT8))
    if not oiio.ImageBufAlgo.zero(image):
        raise RuntimeError(f"Unavailable review tile initialization failed: {image.geterror()}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not image.write(str(destination), oiio.UINT8):
        raise RuntimeError(f"Unavailable review tile write failed: {image.geterror()}")
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise RuntimeError("Unavailable review tile PNG was not written")


def _proxy_path(batch: dict[str, Any], item: dict[str, Any]) -> Path:
    root = Path(batch["output_root"])
    directory = (root / ".blend_ai_proxies").resolve(strict=False)
    directory.relative_to(root)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{item['result_id']}.png"


def _create_proxy(_scene: Any, source: Path, destination: Path) -> tuple[int, int]:
    """Create an 8-bit display-referred PNG without another scene view transform.

    The rendered PNG/JPEG artifact already contains the profile Scene's view
    transform, look, exposure, gamma, and display encoding. Loading that file
    into ``bpy.data.images`` and calling ``Image.save_render(scene=...)`` applies
    those scene settings a second time. OpenImageIO reads, resizes, and quantizes
    the encoded display pixels directly, so the reviewer sees the same Composite
    as the artist while the source Scene remains untouched.
    """
    try:
        import OpenImageIO as oiio
    except ImportError as exc:
        raise RuntimeError(
            "Blender's OpenImageIO module is required for Composite proxy creation"
        ) from exc

    image = oiio.ImageBuf(str(source))
    spec = image.spec()
    width = int(spec.width)
    height = int(spec.height)
    channels = int(spec.nchannels)
    if image.has_error or width <= 0 or height <= 0 or channels <= 0:
        raise RuntimeError(f"Rendered Composite artifact is not readable: {image.geterror()}")

    scale = min(1.0, MAX_PROXY_DIMENSION / max(width, height))
    proxy_width = max(1, round(width * scale))
    proxy_height = max(1, round(height * scale))
    output = image
    if (proxy_width, proxy_height) != (width, height):
        output = oiio.ImageBuf()
        roi = oiio.ROI(0, proxy_width, 0, proxy_height, 0, 1, 0, channels)
        if not oiio.ImageBufAlgo.resize(
            output,
            image,
            "lanczos3",
            0.0,
            roi,
        ):
            raise RuntimeError(f"Composite proxy resize failed: {output.geterror()}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if not output.write(str(destination), oiio.UINT8):
        raise RuntimeError(f"Composite proxy PNG write failed: {output.geterror()}")
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise RuntimeError("Composite proxy was not written")
    if destination.stat().st_size > MAX_PROXY_BYTES:
        raise RuntimeError("Composite proxy exceeds the model-visible byte limit")
    return proxy_width, proxy_height


def _base_metadata(batch: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "batch_id": batch["batch_id"],
        "result_id": item["result_id"],
        "status": item["status"],
        "profile_id": item["profile_id"],
        "target_scene": item["target_scene"],
        "frame": item["frame"],
        "output_pass": "COMPOSITE",
        "output_path": item["output_path"],
        "warnings": list(item["warnings"]),
        "submitted_at": batch["submitted_at"],
        "started_at": item["started_at"],
        "completed_at": item["completed_at"],
    }
    if item.get("error"):
        metadata["error"] = item["error"]
    if item.get("metadata"):
        metadata.update(item["metadata"])
    return metadata


def _batch_counts(batch: dict[str, Any]) -> dict[str, int]:
    return {
        status.lower(): sum(item["status"] == status for item in batch["items"])
        for status in ("QUEUED", "RUNNING", "SUCCEEDED", "SKIPPED", "FAILED", "CANCELLED")
    }


def _update_batch_status(batch: dict[str, Any]) -> None:
    statuses = [item["status"] for item in batch["items"]]
    if any(status == "RUNNING" for status in statuses):
        batch["status"] = "RUNNING"
    elif any(status == "QUEUED" for status in statuses):
        batch["status"] = "CANCELLING" if batch["cancel_requested"] else "QUEUED"
    elif any(status == "FAILED" for status in statuses):
        batch["status"] = "FAILED"
    elif any(status == "CANCELLED" for status in statuses):
        batch["status"] = "CANCELLED"
    elif any(status == "SKIPPED" for status in statuses):
        batch["status"] = "SKIPPED"
    else:
        batch["status"] = "SUCCEEDED"


def _cancel_queued(batch: dict[str, Any], reason: str) -> None:
    for item in batch["items"]:
        if item["status"] == "QUEUED":
            item["warnings"].append(reason)
            item["completed_at"] = _utc_now()
            # Publish terminal status only after all observable fields exist;
            # render-safe socket reads may run concurrently with this update.
            item["status"] = "CANCELLED"
    _update_batch_status(batch)


def _arm_timer() -> None:
    global _timer_armed
    if _timer_armed or _active_render is not None or _next_item() is None:
        return
    timers = getattr(getattr(bpy, "app", None), "timers", None)
    register_timer = getattr(timers, "register", None)
    if not callable(register_timer):
        raise RuntimeError("Blender application timers are unavailable")
    _timer_armed = True
    register_timer(_job_timer, first_interval=0.0)


def _next_item() -> tuple[dict[str, Any], dict[str, Any]] | None:
    for batch in _batches.values():
        if batch["status"] in TERMINAL_BATCH_STATUSES or batch["cancel_requested"]:
            continue
        for item in batch["items"]:
            if item["status"] == "QUEUED":
                return batch, item
    return None


def _mark_item_failed(batch: dict[str, Any], item: dict[str, Any], error: Exception | str) -> None:
    item["error"] = str(error)
    item["completed_at"] = _utc_now()
    item["status"] = "FAILED"
    if not batch["continue_on_error"]:
        _cancel_queued(batch, "Skipped after an earlier render item failed")
    _update_batch_status(batch)


def _job_timer() -> None:
    """Start at most one render and return immediately to Blender's event loop."""
    global _active_render, _timer_armed
    _timer_armed = False
    if _active_render is not None:
        return None
    selected = _next_item()
    if selected is None:
        return None
    batch, item = selected
    if Path(item["output_path"]).exists():
        policy = batch["render_settings"]["existing_file_policy"]
        if policy == "SKIP":
            item["warnings"].append(
                "Existing output preserved but is unproven and not acceptable "
                "as a managed Composite result"
            )
            item["completed_at"] = _utc_now()
            item["status"] = "SKIPPED"
            _update_batch_status(batch)
        else:
            _mark_item_failed(
                batch,
                item,
                f"Render output appeared after submission and was not overwritten: "
                f"'{item['output_path']}'",
            )
        _arm_timer()
        return None
    prepared = None
    try:
        prepared = _prepare_render(batch, item)
        item["started_at"] = _utc_now()
        item["status"] = "RUNNING"
        batch["status"] = "RUNNING"
        active = {"batch": batch, "item": item, "prepared": prepared}
        _active_render = active
        result = bpy.ops.render.render(
            "INVOKE_DEFAULT",
            write_still=True,
            scene=item["target_scene"],
        )
        # A mocked/background operator may finish synchronously without invoking
        # handlers.  Real asynchronous renders report RUNNING_MODAL here.
        if _active_render is active and "FINISHED" in set(result or ()):
            _finish_active(success=True)
        elif _active_render is active and "CANCELLED" in set(result or ()):
            _finish_active(success=False, cancelled=True, error="Blender cancelled render")
    except Exception as exc:
        if _active_render is not None and _active_render.get("item") is item:
            _finish_active(success=False, error=str(exc))
        else:
            if prepared is not None:
                restoration_errors = _restore_prepared(prepared)
                item["warnings"].extend(
                    f"State restoration warning: {error}" for error in restoration_errors
                )
            _mark_item_failed(batch, item, exc)
            _arm_timer()
    return None


def _finish_active(
    *,
    success: bool,
    cancelled: bool = False,
    error: str | None = None,
) -> None:
    global _active_render, _completion_request, _completion_timer_armed
    active = _active_render
    if active is None:
        return
    # A mocked/background render may report FINISHED synchronously after also
    # invoking a completion callback.  Consume any deferred duplicate so the
    # registered timer becomes a harmless no-op.
    _completion_request = None
    _completion_timer_armed = False
    batch = active["batch"]
    item = active["item"]
    prepared = active["prepared"]
    restoration_errors = _restore_prepared(prepared)
    item["warnings"].extend(
        f"State restoration warning: {restore_error}" for restore_error in restoration_errors
    )
    try:
        if restoration_errors:
            raise RuntimeError(
                "Failed to restore exact pre-render state: " + "; ".join(restoration_errors)
            )
        if success:
            completion_scene, _entry, completion_provenance = _assert_item_provenance(
                item,
                phase="render completion",
            )
            if completion_scene is not prepared["scene"]:
                raise RuntimeError("Compiled profile Scene identity changed during render")
            if any(
                prepared["provenance"].get(key) != value
                for key, value in completion_provenance.items()
            ):
                raise RuntimeError("Prepared render provenance changed during completion")
            artifact = Path(item["output_path"])
            if not artifact.is_file() or artifact.stat().st_size <= 0:
                raise RuntimeError("Blender completed without writing the Composite artifact")
            width, height = _image_dimensions(artifact, prepared["scene"])
            review_packet = None
            if batch.get("include_review_packet", False):
                raw_tile_paths = _review_tile_paths(prepared)
                tile_paths: dict[str, Path] = {}
                unavailable = set((prepared.get("review") or {}).get("unavailable", []))
                for slug, raw_path in raw_tile_paths.items():
                    if raw_path is None:
                        destination = (
                            artifact.parent
                            / ".blend_ai_review"
                            / item["result_id"]
                            / f"{slug}_unavailable.png"
                        )
                        _write_unavailable_review_tile(destination, width, height)
                        tile_paths[slug] = destination
                        unavailable.add(slug)
                        item["warnings"].append(
                            f"Diagnostic pass '{slug}' was not emitted by the render engine; "
                            "the tile is explicitly UNAVAILABLE"
                        )
                        continue
                    if raw_path.suffix.lower() == ".png":
                        tile_paths[slug] = raw_path
                        continue
                    destination = raw_path.with_name(f"{slug}_proxy.png")
                    _convert_review_tile(raw_path, destination, slug)
                    tile_paths[slug] = destination
                item["review_tile_paths"] = {slug: str(path) for slug, path in tile_paths.items()}
                tiles = []
                for slug, label in REVIEW_TILE_ORDER:
                    path = tile_paths[slug]
                    metrics = _image_pixel_metrics(path)
                    tiles.append(
                        {
                            "slug": slug,
                            "label": label,
                            "available": slug not in unavailable,
                            "artifact_sha256": _sha256_file(path),
                            "artifact_byte_count": path.stat().st_size,
                            **metrics,
                        }
                    )
                beauty_metrics = _image_pixel_metrics(artifact)
                if beauty_metrics["magenta_coverage"] >= 0.02:
                    item["warnings"].append(
                        "Possible missing-texture magenta: "
                        f"{beauty_metrics['magenta_coverage']:.2%} sampled pixels"
                    )
                review_packet = {
                    "schema_version": 1,
                    "image_source": "SAME_RENDER_RESULT_PASSES",
                    "tile_order": [slug for slug, _label in REVIEW_TILE_ORDER],
                    "tiles": tiles,
                    "technical_check": item.get("technical_check"),
                    "review_intent": _review_intent(
                        _assert_item_provenance(item, phase="review packet")[1]
                    ),
                    "editable_controls": _editable_review_controls(
                        _assert_item_provenance(item, phase="review controls")[1]
                    ),
                    "beauty_magenta_coverage": beauty_metrics["magenta_coverage"],
                }
            proxy = _proxy_path(batch, item)
            try:
                proxy_width, proxy_height = _create_proxy(prepared["scene"], artifact, proxy)
                item["proxy_path"] = str(proxy)
            except Exception:
                if artifact.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    raise
                if artifact.stat().st_size > MAX_PROXY_BYTES:
                    raise
                item["proxy_path"] = str(artifact)
                proxy_width, proxy_height = width, height
            item["completed_at"] = _utc_now()
            proxy_artifact = Path(item["proxy_path"])
            item["metadata"] = {
                **prepared["provenance"],
                "artifact_sha256": _sha256_file(artifact),
                "artifact_byte_count": artifact.stat().st_size,
                "source_width": width,
                "source_height": height,
                "proxy_width": proxy_width,
                "proxy_height": proxy_height,
                "proxy_format": (
                    "JPEG" if proxy_artifact.suffix.lower() in {".jpg", ".jpeg"} else "PNG"
                ),
                "proxy_byte_count": proxy_artifact.stat().st_size,
                "review_packet": review_packet,
            }
            item["status"] = "SUCCEEDED"
        elif cancelled or batch["cancel_requested"]:
            item["completed_at"] = _utc_now()
            item["warnings"].append(error or "Render cancelled cooperatively")
            item["status"] = "CANCELLED"
        else:
            _mark_item_failed(batch, item, error or "Blender render failed")
    except Exception as exc:
        _mark_item_failed(batch, item, exc)
    finally:
        _active_render = None
        if batch["cancel_requested"]:
            _cancel_queued(batch, "Cancelled by request")
        _update_batch_status(batch)
        _arm_timer()


def _completion_timer() -> None:
    """Finalize one render from Blender's main loop, outside render callbacks."""
    global _completion_request, _completion_timer_armed
    request = _completion_request
    _completion_request = None
    _completion_timer_armed = False
    if request is not None:
        _finish_active(**request)
    return None


def _defer_active_completion(
    *,
    success: bool,
    cancelled: bool = False,
    error: str | None = None,
) -> None:
    """Record callback state and defer all datablock work to a main-loop timer."""
    global _completion_request, _completion_timer_armed
    if _active_render is None or _completion_request is not None:
        return
    timers = getattr(getattr(bpy, "app", None), "timers", None)
    register_timer = getattr(timers, "register", None)
    if not callable(register_timer):
        # Focused test/background runtimes may not expose timers.  Do not leave
        # an active render permanently wedged if Blender cannot defer cleanup.
        _finish_active(success=success, cancelled=cancelled, error=error)
        return
    _completion_request = {
        "success": success,
        "cancelled": cancelled,
        "error": error,
    }
    _completion_timer_armed = True
    register_timer(_completion_timer, first_interval=0.0)


@_persistent_handler
def _on_render_complete(scene: Any, *_args: Any) -> None:
    active = _active_render
    if active is None:
        return
    active_scene = active["prepared"]["scene"]
    if scene is not None and getattr(scene, "name", None) != getattr(active_scene, "name", None):
        return
    _defer_active_completion(success=True)


@_persistent_handler
def _on_render_cancel(scene: Any, *_args: Any) -> None:
    active = _active_render
    if active is None:
        return
    active_scene = active["prepared"]["scene"]
    if scene is not None and getattr(scene, "name", None) != getattr(active_scene, "name", None):
        return
    _defer_active_completion(
        success=False,
        cancelled=True,
        error="Blender render was cancelled",
    )


def _request_render_cancel() -> None:
    cancel = getattr(getattr(getattr(bpy, "ops", None), "render", None), "cancel", None)
    if callable(cancel):
        try:
            cancel()
        except Exception:
            # Cooperative cancellation remains set; the active render will be
            # retained if it completes and all queued work is cancelled.
            pass


def _trim_terminal_batches() -> None:
    if len(_batches) < MAX_BATCHES:
        return
    removable = [
        batch_id
        for batch_id, batch in _batches.items()
        if batch["status"] in TERMINAL_BATCH_STATUSES
    ]
    if not removable:
        raise RuntimeError(f"At most {MAX_BATCHES} in-memory render batches are retained")
    del _batches[removable[0]]


def handle_submit_look_render_batch(params: dict[str, Any]) -> dict[str, Any]:
    """Validate and enqueue a bounded batch without waiting for any render."""
    viewport_sessions = _active_cycles_viewport_session_ids()
    if viewport_sessions:
        raise RuntimeError(
            "Cannot submit a look render while a retained Cycles viewport session is "
            f"active: {viewport_sessions}. Restore it first."
        )
    _trim_terminal_batches()
    normalized = _normalized_batch(params)
    batch_id = f"lookbatch-{uuid.uuid4().hex}"
    batch = {
        "batch_id": batch_id,
        "status": "QUEUED",
        "cancel_requested": False,
        "created_monotonic": time.monotonic(),
        "submitted_at": _utc_now(),
        **normalized,
    }
    for index, item in enumerate(batch["items"]):
        item["result_id"] = f"lookresult-{index + 1:06d}-{uuid.uuid4().hex[:12]}"
    _update_batch_status(batch)
    _batches[batch_id] = batch
    try:
        _arm_timer()
    except Exception:
        del _batches[batch_id]
        raise
    return {
        "batch_id": batch_id,
        "status": batch["status"],
        "item_count": len(batch["items"]),
        "result_ids": [item["result_id"] for item in batch["items"]],
        "output_root": batch["output_root"],
        "output_pass": "COMPOSITE",
        "save_blend": False,
    }


def _batch(batch_id: Any) -> dict[str, Any]:
    batch_id = _identifier(batch_id, "batch_id")
    batch = _batches.get(batch_id)
    if batch is None:
        raise ValueError(f"Render batch '{batch_id}' was not found")
    return batch


def handle_get_look_render_batch(params: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    _reject_unknown(params, {"batch_id", "include_results", "max_results", "cursor"}, "params")
    batch = _batch(params.get("batch_id"))
    include_results = _boolean(params.get("include_results", False), "include_results")
    maximum = _integer(params.get("max_results", 100), "max_results", 1, 1000)
    cursor = params.get("cursor")
    if cursor is None:
        offset = 0
    else:
        if not isinstance(cursor, str) or not cursor.isdigit():
            raise ValueError("cursor is not a valid render-result cursor")
        offset = int(cursor)
        if offset < 0 or offset > len(batch["items"]):
            raise ValueError("cursor is outside the render batch")
    _update_batch_status(batch)
    response = {
        "batch_id": batch["batch_id"],
        "status": batch["status"],
        "item_count": len(batch["items"]),
        "counts": _batch_counts(batch),
        "cancel_requested": batch["cancel_requested"],
        "output_root": batch["output_root"],
    }
    if include_results:
        selected = batch["items"][offset : offset + maximum]
        response["results"] = [_base_metadata(batch, item) for item in selected]
        next_offset = offset + len(selected)
        response["next_cursor"] = str(next_offset) if next_offset < len(batch["items"]) else None
    return response


def handle_cancel_look_render_batch(params: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    _reject_unknown(params, {"batch_id"}, "params")
    batch = _batch(params.get("batch_id"))
    if batch["status"] in TERMINAL_BATCH_STATUSES:
        return {
            "batch_id": batch["batch_id"],
            "status": batch["status"],
            "cancel_requested": False,
            "counts": _batch_counts(batch),
        }
    batch["cancel_requested"] = True
    _cancel_queued(batch, "Cancelled by request")
    active = _active_render
    # During a render the socket server may invoke this bounded state-only
    # handler off the main thread so cancellation can be acknowledged even
    # while Blender is compiling shaders.  Never call bpy.ops from that thread;
    # queued work is already terminal and the active item may finish before its
    # callback can service cancellation.  Main-thread calls still request
    # immediate cancellation.
    if (
        active is not None
        and active["batch"] is batch
        and threading.current_thread() is threading.main_thread()
    ):
        _request_render_cancel()
    _update_batch_status(batch)
    return {
        "batch_id": batch["batch_id"],
        "status": batch["status"],
        "cancel_requested": True,
        "counts": _batch_counts(batch),
    }


def handle_get_look_render_result(params: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    _reject_unknown(
        params,
        {"batch_id", "result_id", "output_pass", "proxy_max_size"},
        "params",
    )
    if params.get("output_pass") != "COMPOSITE":
        raise ValueError("output_pass must be COMPOSITE")
    _integer(params.get("proxy_max_size", 1024), "proxy_max_size", 64, 4096)
    batch = _batch(params.get("batch_id"))
    result_id = _identifier(params.get("result_id"), "result_id")
    matches = [item for item in batch["items"] if item["result_id"] == result_id]
    if len(matches) != 1:
        raise ValueError(
            f"Render result '{result_id}' was not found in batch '{batch['batch_id']}'"
        )
    item = matches[0]
    metadata = _base_metadata(batch, item)
    response: dict[str, Any] = {"metadata": metadata}
    if item["status"] == "SUCCEEDED":
        proxy_path = Path(item["proxy_path"])
        root = Path(batch["output_root"])
        try:
            proxy_path.resolve(strict=True).relative_to(root)
        except (FileNotFoundError, ValueError) as exc:
            raise RuntimeError("Composite proxy is missing or outside output_root") from exc
        data = proxy_path.read_bytes()
        if not data or len(data) > MAX_PROXY_BYTES:
            raise RuntimeError("Composite proxy is empty or exceeds the byte limit")
        response["image_base64"] = base64.b64encode(data).decode("ascii")
        review_packet = metadata.get("review_packet")
        if isinstance(review_packet, dict):
            tiles: list[dict[str, str]] = []
            root = Path(batch["output_root"])
            paths = item.get("review_tile_paths") or {}
            for slug, _label in REVIEW_TILE_ORDER:
                tile_path = Path(paths.get(slug, ""))
                try:
                    tile_path.resolve(strict=True).relative_to(root)
                except (FileNotFoundError, ValueError) as exc:
                    raise RuntimeError(
                        f"Review tile '{slug}' is missing or outside output_root"
                    ) from exc
                tile_data = tile_path.read_bytes()
                if not tile_data or len(tile_data) > MAX_REVIEW_TILE_BYTES:
                    raise RuntimeError(f"Review tile '{slug}' is empty or exceeds the byte limit")
                tiles.append(
                    {
                        "slug": slug,
                        "image_base64": base64.b64encode(tile_data).decode("ascii"),
                    }
                )
            response["review_tiles"] = tiles
    return response


def get_acceptance_evidence(batch_id: str, result_id: str) -> dict[str, Any]:
    """Return checksum-bound evidence for one still-current successful result.

    This helper is intentionally not a socket command.  The profile handler can
    call it immediately before an acceptance transaction without trusting
    client-supplied paths, hashes, or a result whose manifest ownership drifted.
    """
    batch = _batch(_identifier(batch_id, "batch_id"))
    result_id = _identifier(result_id, "result_id")
    matches = [item for item in batch["items"] if item["result_id"] == result_id]
    if len(matches) != 1:
        raise ValueError(f"Render result '{result_id}' was not found in batch '{batch_id}'")
    item = matches[0]
    if item["status"] != "SUCCEEDED":
        raise ValueError(
            f"Render result '{result_id}' is {item['status']}, not a successful "
            "managed Composite result"
        )
    metadata = item.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("image_source") != "COMPOSITE_OUTPUT":
        raise RuntimeError("Successful render result lacks authoritative Composite provenance")
    _scene, _entry, current = _assert_item_provenance(
        item,
        phase="acceptance lookup",
    )
    for field, value in current.items():
        if metadata.get(field) != value:
            raise RuntimeError(f"Render acceptance provenance differs for '{field}'")

    root = Path(batch["output_root"]).resolve(strict=True)
    artifact = Path(item["output_path"]).resolve(strict=True)
    try:
        artifact.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("Composite artifact is outside its declared output root") from exc
    if not artifact.is_file() or artifact.stat().st_size <= 0:
        raise RuntimeError("Composite artifact is missing or empty")
    artifact_hash = _sha256_file(artifact)
    if artifact_hash != metadata.get("artifact_sha256"):
        raise RuntimeError("Composite artifact checksum changed after render completion")
    if artifact.stat().st_size != metadata.get("artifact_byte_count"):
        raise RuntimeError("Composite artifact byte count changed after render completion")
    quality_fields = {
        field: metadata.get(field)
        for field in (
            "source_width",
            "source_height",
            "samples",
            "denoise",
            "engine",
            "color_management",
            "compositor_hash",
            "compositor_adapter",
        )
    }
    missing_quality = [field for field, value in quality_fields.items() if value is None]
    if missing_quality:
        raise RuntimeError(
            "Successful render result lacks acceptance quality evidence: "
            + ", ".join(missing_quality)
        )
    return {
        "batch_id": batch["batch_id"],
        "result_id": result_id,
        "profile_id": item["profile_id"],
        "target_scene": item["target_scene"],
        "frame": item["frame"],
        "output_pass": "COMPOSITE",
        "image_source": "COMPOSITE_OUTPUT",
        "artifact_path": str(artifact),
        "artifact_sha256": artifact_hash,
        "artifact_byte_count": artifact.stat().st_size,
        "completed_at": item["completed_at"],
        **current,
        **quality_fields,
    }


def register() -> None:
    dispatcher.register_handler("inspect_look_review_state", handle_inspect_look_review_state)
    dispatcher.register_handler("submit_look_render_batch", handle_submit_look_render_batch)
    dispatcher.register_handler("get_look_render_batch", handle_get_look_render_batch)
    dispatcher.register_handler("cancel_look_render_batch", handle_cancel_look_render_batch)
    dispatcher.register_handler("get_look_render_result", handle_get_look_render_result)
    handlers = getattr(getattr(bpy, "app", None), "handlers", None)
    for collection_name, callback in (
        ("render_complete", _on_render_complete),
        ("render_cancel", _on_render_cancel),
    ):
        collection = getattr(handlers, collection_name, None)
        if collection is not None and callback not in collection:
            collection.append(callback)


def unregister() -> None:
    global _completion_request, _completion_timer_armed, _timer_armed
    if _active_render is not None:
        _request_render_cancel()
        raise RuntimeError(
            "Cannot unregister the look-profile renderer while an asynchronous render "
            "is active. Cancellation was requested; wait for Blender's render_cancel "
            "or render_complete callback, then unregister again."
        )
    for batch in _batches.values():
        if batch["status"] not in TERMINAL_BATCH_STATUSES:
            batch["cancel_requested"] = True
            _cancel_queued(batch, "Cancelled because the look-profile renderer was unregistered")
    timers = getattr(getattr(bpy, "app", None), "timers", None)
    is_registered = getattr(timers, "is_registered", None)
    unregister_timer = getattr(timers, "unregister", None)
    if callable(is_registered) and callable(unregister_timer):
        try:
            if is_registered(_job_timer):
                unregister_timer(_job_timer)
            if is_registered(_completion_timer):
                unregister_timer(_completion_timer)
        except Exception:
            pass
    _timer_armed = False
    _completion_request = None
    _completion_timer_armed = False
    handlers = getattr(getattr(bpy, "app", None), "handlers", None)
    for collection_name, callback in (
        ("render_complete", _on_render_complete),
        ("render_cancel", _on_render_cancel),
    ):
        collection = getattr(handlers, collection_name, None)
        if collection is not None and callback in collection:
            collection.remove(callback)
