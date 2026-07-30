"""Revision tracking and JSON-only caches for spatial relighting tools.

Blender RNA values are deliberately never retained here.  Evaluated objects are
only valid for the depsgraph evaluation in which they were obtained; keeping
them in a cache is both unsafe and a common source of crashes after scene edits.
"""

from __future__ import annotations

from collections import OrderedDict
import copy
import json
import time
from typing import Any

import bpy


MAX_CACHE_ENTRIES = 8
MAX_RECENT_UPDATE_BATCHES = 32

_geometry_revision = 0
_lighting_revision = 0
_cache: "OrderedDict[str, Any]" = OrderedDict()
_cache_timestamps: dict[str, float] = {}
_handlers_registered = False
_managed_edit_depth = 0
_render_evaluation_depth = 0
_recent_update_batches: list[dict[str, Any]] = []
MANAGED_PROP = "blend_ai_relight_managed"
PROFILE_MANAGED_PROPS = ("blend_ai_managed", "blend_ai_look_managed")

try:
    _persistent = bpy.app.handlers.persistent
except (AttributeError, TypeError):
    def _persistent(callback: Any) -> Any:
        return callback


def _json_clone(value: Any) -> Any:
    """Return a detached JSON value, raising for RNA or other unsafe values."""
    return json.loads(json.dumps(value, allow_nan=False, separators=(",", ":")))


def get_revisions() -> dict[str, int]:
    """Return monotonically increasing geometry and lighting revisions."""
    return {
        "geometry_revision": _geometry_revision,
        "lighting_revision": _lighting_revision,
    }


def make_scene_revision(scene: Any | None = None) -> str:
    """Return a human-readable composite revision for diagnostics."""
    if scene is None:
        scene = getattr(getattr(bpy, "context", None), "scene", None)
    frame = int(getattr(scene, "frame_current", 0) or 0)
    return f"g{_geometry_revision}:l{_lighting_revision}:f{frame}"


def get_cached(key: str) -> Any | None:
    """Get a detached cached value and update its LRU position."""
    if key not in _cache:
        return None
    value = _cache.pop(key)
    _cache[key] = value
    return copy.deepcopy(value)


def get_cache_age_ms(key: str) -> float | None:
    """Return monotonic cache age without changing LRU order."""
    created = _cache_timestamps.get(key)
    if created is None or key not in _cache:
        return None
    return max(0.0, (time.monotonic() - created) * 1000.0)


def put_cached(key: str, value: Any) -> None:
    """Store only detached JSON data in the bounded LRU cache."""
    safe = _json_clone(value)
    _cache.pop(key, None)
    _cache[key] = safe
    _cache_timestamps[key] = time.monotonic()
    while len(_cache) > MAX_CACHE_ENTRIES:
        evicted, _value = _cache.popitem(last=False)
        _cache_timestamps.pop(evicted, None)


def make_cache_key(namespace: str, payload: dict[str, Any]) -> str:
    """Create a deterministic key from JSON-compatible request state."""
    safe_payload = _json_clone(payload)
    return namespace + ":" + json.dumps(safe_payload, sort_keys=True, separators=(",", ":"))


def clear_cache() -> None:
    _cache.clear()
    _cache_timestamps.clear()


def get_recent_updates() -> list[dict[str, Any]]:
    """Return detached dependency-graph diagnostics for live verification.

    The bounded records contain only strings, booleans and numbers.  They are
    intentionally useful through ``execute_blender_code`` without retaining
    Blender RNA values or exposing a fifth public MCP tool.
    """
    return _json_clone(_recent_update_batches)


def clear_recent_updates() -> None:
    """Forget bounded dependency-graph diagnostics."""
    _recent_update_batches.clear()


def _clear_grace_state() -> None:
    """Reset transient classifier scopes after file/add-on lifecycle changes."""
    global _managed_edit_depth, _render_evaluation_depth
    _managed_edit_depth = 0
    _render_evaluation_depth = 0


def mark_geometry_dirty(*, clear: bool = True) -> int:
    """Advance the geometry revision after transforms/topology/visibility edits."""
    global _geometry_revision
    _geometry_revision += 1
    if clear:
        clear_cache()
    return _geometry_revision


def mark_lighting_dirty(*, clear: bool = False) -> int:
    """Advance the lighting revision after light/world/shading edits."""
    global _lighting_revision
    _lighting_revision += 1
    if clear:
        clear_cache()
    return _lighting_revision


def invalidate_all() -> dict[str, int]:
    """Invalidate both revision domains and all cached payloads."""
    global _geometry_revision, _lighting_revision
    _geometry_revision += 1
    _lighting_revision += 1
    clear_cache()
    return get_revisions()


def begin_managed_edit() -> None:
    """Begin classifying collection/view-layer tags from a managed transaction.

    Callers must flush dependency-graph updates before ending the scope.  A
    depth counter is deliberately used instead of a wall-clock grace period:
    time-based suppression can hide a real artist camera, collection or view
    layer edit made immediately after a relighting operation returns.
    """
    global _managed_edit_depth
    _managed_edit_depth += 1


def end_managed_edit() -> None:
    """End one managed-edit classifier scope."""
    global _managed_edit_depth
    _managed_edit_depth = max(0, _managed_edit_depth - 1)


def begin_render_evaluation() -> None:
    """Begin suppressing transient scene/camera tags from a synchronous render.

    Blender 5.2 tags the active Camera object and Scene as geometry-updated
    while preparing and tearing down a Cycles render even when neither is
    mutated. Quick-render handlers bracket the blocking render operation with
    this and :func:`end_render_evaluation`. The ID allowlist avoids masking
    ordinary mesh/object edits while Blender's UI thread is occupied.
    """
    global _render_evaluation_depth
    _render_evaluation_depth += 1


def end_render_evaluation() -> None:
    """End one render-evaluation classifier scope.

    Callers flush dependency-graph updates while the scope remains active. No
    post-render grace interval is retained, so subsequent artist edits are
    always visible to revision tracking.
    """
    global _render_evaluation_depth
    _render_evaluation_depth = max(0, _render_evaluation_depth - 1)


def _rna_identifier(value: Any) -> str:
    """Best-effort Blender RNA type name without retaining the value."""
    identifier = getattr(getattr(value, "bl_rna", None), "identifier", None)
    if isinstance(identifier, str):
        return identifier
    return type(value).__name__


def _is_managed(value: Any) -> bool:
    getter = getattr(value, "get", None)
    if not callable(getter):
        return False
    try:
        return bool(getter(MANAGED_PROP, False)) or any(
            bool(getter(key, False)) for key in PROFILE_MANAGED_PROPS
        )
    except Exception:
        return False


def _classify_update(update: Any) -> tuple[bool, bool]:
    """Return ``(geometry_changed, lighting_changed)`` for a depsgraph update."""
    value = getattr(update, "id", None)
    kind = _rna_identifier(value).lower()

    if "material" in kind:
        # Material names participate in semantic opening candidates.
        return True, True
    if "nodetree" in kind:
        tree_kind = str(
            getattr(value, "bl_idname", "") or getattr(value, "type", "")
        ).lower()
        if "geometry" in tree_kind:
            return True, False
        return False, True
    if "world" in kind or "image" in kind:
        return False, True
    if "light" in kind:
        return False, True
    if "collection" in kind:
        return (False, True) if _is_managed(value) else (True, False)
    if "camera" in kind or "mesh" in kind or "curve" in kind:
        geometry = bool(
            getattr(update, "is_updated_geometry", False)
            or getattr(update, "is_updated_transform", False)
        )
        shading = bool(getattr(update, "is_updated_shading", False))
        # Entering Rendered viewport shading tags evaluated scene IDs for
        # shading without changing their transforms or topology.  Treating
        # those notifications as geometry invalidates spatial plans after
        # every visual-feedback capture.  A flag-less update remains a
        # conservative geometry invalidation for Blender/API compatibility.
        if geometry or shading:
            return geometry, shading
        return True, False
    if "scene" in kind:
        # Assigning the separate managed World or exposure produces a Scene
        # update even though evaluated geometry is unchanged. Collection and
        # non-light Object updates still independently invalidate geometry.
        if _is_managed(getattr(value, "world", None)):
            return False, True
        return True, True
    if "viewlayer" in kind:
        return True, True
    if "text" in kind:
        # The transaction ledger is stored in a Text datablock and has no
        # evaluated-scene effect.
        return False, False
    if "object" in kind:
        object_type = str(getattr(value, "type", "")).upper()
        if object_type == "LIGHT":
            return False, True
        geometry = bool(
            getattr(update, "is_updated_geometry", False)
            or getattr(update, "is_updated_transform", False)
        )
        shading = bool(getattr(update, "is_updated_shading", False))
        # Object transforms, visibility, parenting and instance changes affect
        # the spatial model.  Shading-only redraw notifications do not; actual
        # material edits independently report Material/NodeTree updates.
        if geometry or shading:
            return geometry, shading
        return True, False

    # Unknown ID types are rare and correctness is more important than a cache
    # hit.  Conservative invalidation also covers add-on/custom property edits.
    return True, True


@_persistent
def depsgraph_update_post(_scene: Any, depsgraph: Any) -> None:
    """Blender handler that advances revisions from dependency graph updates."""
    geometry_changed = False
    lighting_changed = False
    try:
        updates = list(getattr(depsgraph, "updates", ()))
    except Exception:
        updates = ()

    managed_grace = _managed_edit_depth > 0
    render_grace = _render_evaluation_depth > 0
    update_records: list[dict[str, Any]] = []
    for update in updates:
        value = getattr(update, "id", None)
        kind = _rna_identifier(value)
        geometry, lighting = _classify_update(update)
        classified_geometry = geometry
        classified_lighting = lighting
        suppressed_by_managed = False
        suppressed_by_render = False
        normalized_kind = kind.lower()
        object_type = str(getattr(value, "type", "") or "").upper()
        if render_grace and geometry:
            if (
                "scene" in normalized_kind
                or "viewlayer" in normalized_kind
                or "collection" in normalized_kind
                or "camera" in normalized_kind
                or ("object" in normalized_kind and object_type == "CAMERA")
            ):
                geometry, lighting = False, True
                suppressed_by_render = True
        if managed_grace and geometry:
            if (
                _is_managed(value)
                or "collection" in normalized_kind
                or "viewlayer" in normalized_kind
                or "scene" in normalized_kind
            ):
                geometry, lighting = False, True
                suppressed_by_managed = True
        update_records.append(
            {
                "kind": kind,
                "name": str(
                    getattr(value, "name_full", None)
                    or getattr(value, "name", None)
                    or ""
                ),
                "object_type": str(getattr(value, "type", "") or ""),
                "managed": _is_managed(value),
                "is_updated_geometry": bool(
                    getattr(update, "is_updated_geometry", False)
                ),
                "is_updated_transform": bool(
                    getattr(update, "is_updated_transform", False)
                ),
                "is_updated_shading": bool(
                    getattr(update, "is_updated_shading", False)
                ),
                "classified_geometry": classified_geometry,
                "classified_lighting": classified_lighting,
                "geometry": geometry,
                "lighting": lighting,
                "suppressed_by_managed_grace": suppressed_by_managed,
                "suppressed_by_render_grace": suppressed_by_render,
            }
        )
        geometry_changed = geometry_changed or geometry
        lighting_changed = lighting_changed or lighting

    if geometry_changed:
        mark_geometry_dirty(clear=False)
    if lighting_changed:
        mark_lighting_dirty(clear=False)
    if geometry_changed:
        clear_cache()

    if update_records:
        _recent_update_batches.append(
            {
                "managed_grace": managed_grace,
                "managed_grace_remaining_ms": 0,
                "render_grace": render_grace,
                "render_grace_remaining_ms": 0,
                "geometry_changed": geometry_changed,
                "lighting_changed": lighting_changed,
                "geometry_revision": _geometry_revision,
                "lighting_revision": _lighting_revision,
                "updates": update_records,
            }
        )
        del _recent_update_batches[:-MAX_RECENT_UPDATE_BATCHES]


@_persistent
def load_post(_unused: Any) -> None:
    """Blender handler used after loading a file."""
    clear_recent_updates()
    _clear_grace_state()
    invalidate_all()


def _handler_list(name: str) -> Any | None:
    handlers = getattr(getattr(bpy, "app", None), "handlers", None)
    return getattr(handlers, name, None) if handlers is not None else None


def _append_once(sequence: Any, callback: Any) -> None:
    if sequence is None:
        return
    try:
        if callback not in sequence:
            sequence.append(callback)
    except (AttributeError, TypeError):
        # Some test doubles and minimal Blender builds do not expose handlers.
        return


def _remove_if_present(sequence: Any, callback: Any) -> None:
    if sequence is None:
        return
    try:
        while callback in sequence:
            sequence.remove(callback)
    except (AttributeError, TypeError, ValueError):
        return


def register_handlers() -> None:
    """Install revision handlers idempotently."""
    global _handlers_registered
    _append_once(_handler_list("depsgraph_update_post"), depsgraph_update_post)
    _append_once(_handler_list("load_post"), load_post)
    _handlers_registered = True


def unregister_handlers() -> None:
    """Remove revision handlers if installed."""
    global _handlers_registered
    _remove_if_present(_handler_list("depsgraph_update_post"), depsgraph_update_post)
    _remove_if_present(_handler_list("load_post"), load_post)
    clear_cache()
    clear_recent_updates()
    _clear_grace_state()
    _handlers_registered = False
    clear_cache()


def _reset_for_tests() -> None:
    """Reset module state.  Intended only for isolated unit tests."""
    global _geometry_revision, _lighting_revision, _handlers_registered
    _geometry_revision = 0
    _lighting_revision = 0
    _handlers_registered = False
    _clear_grace_state()
    clear_cache()
