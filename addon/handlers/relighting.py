"""Spatial inspection, ray queries, and transactional managed relighting.

The handlers in this module are intentionally conservative:

* evaluated Blender RNA is converted to plain JSON data before returning or
  caching it;
* expensive scene queries have explicit scan and response bounds;
* light-plan validation completes before the first mutation; and
* only objects carrying this add-on's stable managed-ID property are replaced
  or rolled back.
"""

from __future__ import annotations

import base64
import fnmatch
import hashlib
import heapq
import json
import math
import re
import time
import uuid
from collections import OrderedDict
from typing import Any, Iterable

import bpy

from .. import dispatcher, spatial_cache


MANAGED_COLLECTION = "AI_RELIGHT"
MANAGED_WORLD = "AI_RELIGHT_WORLD"
MANAGED_PROP = "blend_ai_relight_managed"
MANAGED_ID_PROP = "blend_ai_relight_light_id"
MANAGED_PLAN_PROP = "blend_ai_relight_plan_id"
MANAGED_WORLD_PROP = MANAGED_PROP
LEDGER_TEXT = "AI_RELIGHT_TRANSACTIONS"
LEDGER_PROP = "blend_ai_relight_transaction_ledger"
LEDGER_SCHEMA_VERSION = 1
MAX_LEDGER_ENTRIES = 20
MAX_LEDGER_BYTES = 2 * 1024 * 1024

MAX_INSTANCES = 5_000
DEFAULT_MAX_INSTANCES = 2_000
MAX_SURFACE_TRIANGLES = 1_000_000
DEFAULT_MAX_SURFACE_TRIANGLES = 200_000
MAX_SURFACE_RESULTS = 64
DEFAULT_PAYLOAD_TARGET_BYTES = 500 * 1024
MAX_PAYLOAD_BYTES = 2 * 1024 * 1024
MAX_RAYS = 512
MAX_RAY_HITS = 32
MAX_IGNORE_PATTERNS = 32
MAX_PATTERN_LENGTH = 128
MAX_MANAGED_LIGHTS = 128
OPENING_SEMANTIC_WORDS = ("skylight", "window", "opening", "glass", "rooflight")

_SAFE_NAME = re.compile(r"^[^\x00-\x1f\x7f]{1,128}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_ledger: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_ledger_loaded = False
_ledger_source_token: tuple[int, str] | None = None
_raycast_index_key: tuple[int, str, str, int] | None = None
_raycast_index_records: list[dict[str, Any]] = []
_raycast_accel_key: tuple[int, str, str, int] | None = None
_raycast_accel: dict[str, Any] | None = None

try:
    _persistent = bpy.app.handlers.persistent
except (AttributeError, TypeError):
    def _persistent(callback: Any) -> Any:
        return callback


def _load_post_handlers() -> Any | None:
    return getattr(
        getattr(getattr(bpy, "app", None), "handlers", None),
        "load_post",
        None,
    )


def _reset_ledger_state() -> None:
    """Forget process-local transactions before reading another blend file."""
    global _ledger_loaded, _ledger_source_token
    _ledger.clear()
    _ledger_loaded = False
    _ledger_source_token = None


def _reset_raycast_index() -> None:
    global _raycast_index_key, _raycast_accel_key, _raycast_accel
    _raycast_index_key = None
    _raycast_index_records.clear()
    _raycast_accel_key = None
    _raycast_accel = None


@_persistent
def _clear_ledger_on_load(_unused: Any, *_handler_args: Any) -> None:
    """Prevent transaction snapshots from crossing Blender file boundaries."""
    _reset_ledger_state()
    _reset_raycast_index()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _number(
    value: Any,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if not _is_number(value):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{field} must be <= {maximum}")
    return result


def _integer(
    value: Any,
    field: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field} must be <= {maximum}")
    return value


def _vec3(value: Any, field: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field} must be a three-element array")
    return [_number(component, f"{field}[{index}]") for index, component in enumerate(value)]


def _color(value: Any, field: str = "color_rgb") -> list[float]:
    color = _vec3(value, field)
    if any(component < 0.0 or component > 1.0 for component in color):
        raise ValueError(f"{field} components must be between 0 and 1")
    return color


def _bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _plain(value: Any) -> Any:
    """Detach and validate a public payload as strict JSON."""
    return json.loads(json.dumps(value, allow_nan=False, separators=(",", ":")))


def _iter_values(collection: Any) -> list[Any]:
    if collection is None:
        return []
    if isinstance(collection, dict):
        return list(collection.values())
    try:
        return list(collection)
    except (TypeError, AttributeError):
        return []


def _lookup(collection: Any, name: str) -> Any | None:
    getter = getattr(collection, "get", None)
    if callable(getter):
        try:
            return getter(name)
        except Exception:
            return None
    for value in _iter_values(collection):
        if getattr(value, "name", None) == name:
            return value
    return None


def _custom_get(value: Any, key: str, default: Any = None) -> Any:
    getter = getattr(value, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            try:
                result = getter(key)
                return default if result is None else result
            except Exception:
                pass
    try:
        return value[key]
    except Exception:
        return default


def _custom_set(value: Any, key: str, item: Any) -> None:
    try:
        value[key] = item
    except Exception as exc:
        raise RuntimeError(f"Unable to set managed property '{key}': {exc}") from exc


def _sequence(value: Any, length: int | None = None) -> list[float]:
    try:
        result = [float(component) for component in value]
    except (TypeError, ValueError):
        return []
    return result if length is None or len(result) == length else []


def _matrix_rows(matrix: Any) -> list[list[float]]:
    try:
        rows = [[float(component) for component in row] for row in matrix]
        if len(rows) == 4 and all(len(row) == 4 for row in rows):
            return rows
    except (TypeError, ValueError):
        pass
    return []


def _transform_point(matrix: Any, point: Any) -> list[float]:
    source = _sequence(point, 3)
    if len(source) != 3:
        raise ValueError("Invalid point in evaluated geometry")
    try:
        from mathutils import Vector

        return _sequence(matrix @ Vector(source), 3)
    except Exception:
        rows = _matrix_rows(matrix)
        if not rows:
            return source
        vector = source + [1.0]
        return [sum(rows[row][column] * vector[column] for column in range(4)) for row in range(3)]


def _world_location(obj: Any) -> list[float]:
    matrix = getattr(obj, "matrix_world", None)
    translation = getattr(matrix, "translation", None)
    result = _sequence(translation, 3)
    if result:
        return result
    rows = _matrix_rows(matrix)
    if rows:
        return [rows[0][3], rows[1][3], rows[2][3]]
    result = _sequence(getattr(obj, "location", (0.0, 0.0, 0.0)), 3)
    return result or [0.0, 0.0, 0.0]


def _bounds(obj: Any, matrix: Any) -> dict[str, list[float]] | None:
    corners = getattr(obj, "bound_box", None)
    try:
        points = [_transform_point(matrix, corner) for corner in corners]
    except (TypeError, ValueError):
        return None
    if not points:
        return None
    minimum = [min(point[axis] for point in points) for axis in range(3)]
    maximum = [max(point[axis] for point in points) for axis in range(3)]
    return {
        "min": minimum,
        "max": maximum,
        "center": [(minimum[axis] + maximum[axis]) * 0.5 for axis in range(3)],
        "size": [maximum[axis] - minimum[axis] for axis in range(3)],
        # Internal only: camera projection must use the transformed evaluated
        # box, not the inflated corners of its axis-aligned envelope.
        "evaluated_corners": points,
    }


def _merge_bounds(current: dict[str, list[float]] | None, item: Any) -> dict[str, list[float]] | None:
    if item is None:
        return current
    if current is None:
        return {
            key: _plain(item[key])
            for key in ("min", "max", "center", "size")
        }
    minimum = [min(current["min"][axis], item["min"][axis]) for axis in range(3)]
    maximum = [max(current["max"][axis], item["max"][axis]) for axis in range(3)]
    return {
        "min": minimum,
        "max": maximum,
        "center": [(minimum[axis] + maximum[axis]) * 0.5 for axis in range(3)],
        "size": [maximum[axis] - minimum[axis] for axis in range(3)],
    }


def _normalize(vector: list[float], field: str) -> tuple[list[float], float]:
    length = math.sqrt(sum(component * component for component in vector))
    if not math.isfinite(length) or length <= 1e-12:
        raise ValueError(f"{field} must not be a zero-length vector")
    return [component / length for component in vector], length


def _sub(a: Iterable[float], b: Iterable[float]) -> list[float]:
    return [float(left) - float(right) for left, right in zip(a, b)]


def _add_scaled(a: Iterable[float], direction: Iterable[float], distance: float) -> list[float]:
    return [float(left) + float(axis) * distance for left, axis in zip(a, direction)]


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _safe_name(value: Any, field: str, *, identifier: bool = False) -> str:
    if not isinstance(value, str) or not (_SAFE_ID if identifier else _SAFE_NAME).fullmatch(value):
        kind = "identifier" if identifier else "name"
        raise ValueError(f"{field} must be a safe non-empty {kind} of at most 128 characters")
    return value


def _reject_unknown(params: dict[str, Any], allowed: set[str], field: str = "params") -> None:
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(f"{field} contains unsupported fields: {unknown}")


# ---------------------------------------------------------------------------
# Spatial context


def _collection_paths(scene: Any) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}

    def visit(collection: Any, parents: list[str]) -> None:
        name = str(getattr(collection, "name", ""))
        path = parents + ([name] if name else [])
        if name:
            result[name] = path
        for child in _iter_values(getattr(collection, "children", None)):
            visit(child, path)

    root = getattr(scene, "collection", None)
    if root is not None:
        visit(root, [])
    return result


def _object_collections(obj: Any, paths: dict[str, list[str]]) -> tuple[list[str], list[list[str]]]:
    names = sorted(
        {
            str(getattr(collection, "name", ""))
            for collection in _iter_values(getattr(obj, "users_collection", None))
            if getattr(collection, "name", None)
        }
    )
    return names, [paths[name] for name in names if name in paths]


def _scene_collection_membership(
    scene: Any,
    paths: dict[str, list[str]],
) -> dict[str, set[str]]:
    """Index scene-linked object names to collection ancestry in one pass.

    Looking up ``users_collection`` through evaluated instance parents is
    disproportionately expensive in Blender when a scene contains tens of
    thousands of collection instances.  Walking each scene collection once
    gives the same active-scene ancestry without repeating that RNA lookup for
    every evaluated duplicate.
    """
    membership: dict[str, set[str]] = {}
    visited: set[str] = set()

    def visit(collection: Any) -> None:
        collection_name = str(getattr(collection, "name", ""))
        if not collection_name or collection_name in visited:
            return
        visited.add(collection_name)
        ancestry = set(paths.get(collection_name, [collection_name]))
        for obj in _iter_values(getattr(collection, "objects", None)):
            for name in _object_match_names(obj):
                membership.setdefault(name, set()).update(ancestry)
        for child in _iter_values(getattr(collection, "children", None)):
            visit(child)

    root = getattr(scene, "collection", None)
    if root is not None:
        visit(root)
    return membership


def _instance_collections(
    instance: Any,
    source: Any,
    paths: dict[str, list[str]],
    scene_membership: dict[str, set[str]] | None = None,
    direct_cache: dict[str, tuple[str, ...]] | None = None,
) -> tuple[list[str], list[list[str]]]:
    """Return active-scene collection ancestry for an evaluated instance.

    Instanced source objects commonly live in a library/asset collection that
    is not linked directly into the active scene.  In that case the meaningful
    scene membership comes from the evaluated instance's parent (the collection
    instancer or Geometry Nodes owner), not from ``source.users_collection``.
    """
    direct_names: set[str] = set()
    candidates = [source]
    parent = getattr(instance, "parent", None)
    seen: set[str] = set()
    while parent is not None:
        parent_names = _object_match_names(parent)
        token = min(parent_names) if parent_names else f"pointer:{id(parent)}"
        if token in seen:
            break
        seen.add(token)
        candidates.append(parent)
        original = getattr(parent, "original", None)
        if original is not None:
            candidates.append(original)
        parent = getattr(parent, "parent", None)

    for candidate in candidates:
        candidate_names = _object_match_names(candidate)
        scene_names: set[str] = set()
        if scene_membership is not None:
            for candidate_name in candidate_names:
                scene_names.update(scene_membership.get(candidate_name, set()))
        if scene_names:
            direct_names.update(scene_names)
            continue

        cache_key = min(candidate_names) if candidate_names else f"pointer:{id(candidate)}"
        cached = direct_cache.get(cache_key) if direct_cache is not None else None
        if cached is None:
            cached = tuple(
                sorted(
                    {
                        str(getattr(collection, "name", ""))
                        for collection in _iter_values(
                            getattr(candidate, "users_collection", None)
                        )
                        if getattr(collection, "name", None)
                    }
                )
            )
            # An evaluated wrapper can have no direct users while its original
            # object (often visited immediately afterward under the same name)
            # carries the real collection link.  Do not let that empty wrapper
            # poison the command-local cache.
            if direct_cache is not None and cached:
                direct_cache[cache_key] = cached
        direct_names.update(cached)

    names: set[str] = set(direct_names)
    for name in direct_names:
        names.update(paths.get(name, []))
    ordered_names = sorted(names)
    return ordered_names, [paths[name] for name in ordered_names if name in paths]


def _is_visible(obj: Any, view_layer: Any) -> tuple[bool, bool]:
    viewport = not bool(getattr(obj, "hide_viewport", False))
    visible_get = getattr(obj, "visible_get", None)
    if callable(visible_get):
        try:
            viewport = viewport and bool(visible_get(view_layer=view_layer))
        except TypeError:
            try:
                viewport = viewport and bool(visible_get())
            except Exception:
                pass
        except Exception:
            pass
    render = not bool(getattr(obj, "hide_render", False))
    return viewport, render


def _camera_object(scene: Any, camera_name: Any) -> Any | None:
    if camera_name is not None:
        _safe_name(camera_name, "camera_name")
        camera = _lookup(getattr(bpy.data, "objects", None), camera_name)
    else:
        camera = getattr(scene, "camera", None)
    if camera is not None and str(getattr(camera, "type", "")).upper() != "CAMERA":
        raise ValueError(f"Object '{getattr(camera, 'name', camera_name)}' is not a camera")
    return camera


def _camera_context(scene: Any, camera: Any, depsgraph: Any) -> dict[str, Any] | None:
    if camera is None:
        return None
    data = getattr(camera, "data", None)
    render = getattr(scene, "render", None)
    scale = float(getattr(render, "resolution_percentage", 100) or 100) / 100.0
    width = max(1, int(float(getattr(render, "resolution_x", 1920) or 1920) * scale))
    height = max(1, int(float(getattr(render, "resolution_y", 1080) or 1080) * scale))
    pixel_aspect_x = float(getattr(render, "pixel_aspect_x", 1.0) or 1.0)
    pixel_aspect_y = float(getattr(render, "pixel_aspect_y", 1.0) or 1.0)

    projection_matrix = None
    calc = getattr(camera, "calc_matrix_camera", None)
    if callable(calc):
        try:
            projection_matrix = _matrix_rows(
                calc(
                    depsgraph,
                    x=width,
                    y=height,
                    scale_x=pixel_aspect_x,
                    scale_y=pixel_aspect_y,
                )
            ) or None
        except Exception:
            projection_matrix = None

    result = {
        "name": str(getattr(camera, "name", "")),
        "type": str(getattr(data, "type", "PERSP")),
        "location_world": _world_location(camera),
        "matrix_world": _matrix_rows(getattr(camera, "matrix_world", None)) or None,
        "lens_mm": float(getattr(data, "lens", 50.0) or 50.0),
        "sensor_width_mm": float(getattr(data, "sensor_width", 36.0) or 36.0),
        "sensor_height_mm": float(getattr(data, "sensor_height", 24.0) or 24.0),
        "sensor_fit": str(getattr(data, "sensor_fit", "AUTO")),
        "clip_start": float(getattr(data, "clip_start", 0.1) or 0.1),
        "clip_end": float(getattr(data, "clip_end", 1000.0) or 1000.0),
        "ortho_scale": float(getattr(data, "ortho_scale", 6.0) or 6.0),
        "shift_x": float(getattr(data, "shift_x", 0.0) or 0.0),
        "shift_y": float(getattr(data, "shift_y", 0.0) or 0.0),
        "resolution": [width, height],
        "pixel_aspect": [pixel_aspect_x, pixel_aspect_y],
        "projection_matrix": projection_matrix,
    }
    result["lens"] = result["lens_mm"]
    result["clipping"] = {
        "start": result["clip_start"],
        "end": result["clip_end"],
    }
    for source, destination in (("angle", "field_of_view"), ("angle_x", "field_of_view_x"), ("angle_y", "field_of_view_y")):
        value = getattr(data, source, None)
        if _is_number(value):
            result[destination] = float(value)
    if result["type"] == "PANO":
        result["frustum_test_supported"] = False
    else:
        result["frustum_test_supported"] = True
    return result


def _project_bounds(scene: Any, camera: Any, item_bounds: Any) -> dict[str, Any]:
    if camera is None or item_bounds is None:
        return {
            "projection_supported": False,
            "in_frustum": None,
            "screen_rect": None,
            "depth_range": None,
            "projected_area": None,
        }
    if str(getattr(getattr(camera, "data", None), "type", "PERSP")) == "PANO":
        return {
            "projection_supported": False,
            "in_frustum": None,
            "screen_rect": None,
            "depth_range": None,
            "projected_area": None,
        }
    try:
        from bpy_extras.object_utils import world_to_camera_view
        from mathutils import Vector

        corners = item_bounds.get("evaluated_corners")
        if not isinstance(corners, list) or len(corners) != 8:
            minimum = item_bounds["min"]
            maximum = item_bounds["max"]
            corners = [
                [x, y, z]
                for x in (minimum[0], maximum[0])
                for y in (minimum[1], maximum[1])
                for z in (minimum[2], maximum[2])
            ]
        projected = [world_to_camera_view(scene, camera, Vector(point)) for point in corners]
        in_front = [point for point in projected if float(point.z) > 0.0]
        if not in_front:
            return {
                "projection_supported": True,
                "in_frustum": False,
                "screen_rect": None,
                "depth_range": None,
                "projected_area": 0.0,
            }
        x_values = [float(point.x) for point in in_front]
        y_values = [float(point.y) for point in in_front]
        z_values = [float(point.z) for point in in_front]
        screen_rect = [min(x_values), min(y_values), max(x_values), max(y_values)]
        clipped_width = max(0.0, min(1.0, screen_rect[2]) - max(0.0, screen_rect[0]))
        clipped_height = max(0.0, min(1.0, screen_rect[3]) - max(0.0, screen_rect[1]))
        projected_area = clipped_width * clipped_height
        return {
            "projection_supported": True,
            "in_frustum": projected_area > 0.0,
            "screen_rect": screen_rect,
            "depth_range": [min(z_values), max(z_values)],
            "projected_area": projected_area,
        }
    except Exception:
        # This helper may be unavailable in background/minimal builds.  Falling
        # back to view-layer visibility is safer than silently dropping objects.
        return {
            "projection_supported": False,
            "in_frustum": None,
            "screen_rect": None,
            "depth_range": None,
            "projected_area": None,
        }


def _in_camera_frustum(scene: Any, camera: Any, item_bounds: Any) -> bool:
    """Compatibility helper used by tests and older callers."""
    projection = _project_bounds(scene, camera, item_bounds)
    return projection["in_frustum"] is not False


def _instance_source(instance: Any) -> tuple[Any, Any]:
    evaluated = getattr(instance, "object", None)
    source = getattr(evaluated, "original", None) or evaluated
    return evaluated, source


def _instance_identifier(instance: Any, source: Any) -> str:
    source_name = str(getattr(source, "name_full", None) or getattr(source, "name", "Object"))
    parent = getattr(instance, "parent", None)
    parent_name = str(getattr(parent, "name", ""))
    try:
        persistent = [int(value) for value in getattr(instance, "persistent_id", ())]
    except Exception:
        persistent = []
    while persistent and persistent[-1] == 0:
        persistent.pop()
    raw = json.dumps([source_name, parent_name, persistent], separators=(",", ":"))
    suffix = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{source_name}:{suffix}"


def _object_match_names(obj: Any) -> set[str]:
    """Names Blender may expose for the same original/evaluated object."""
    result: set[str] = set()
    for candidate in (obj, getattr(obj, "original", None)):
        if candidate is None:
            continue
        for attribute in ("name_full", "name"):
            value = getattr(candidate, attribute, None)
            if value:
                result.add(str(value))
    return result


def _object_pointer_tokens(obj: Any) -> set[int]:
    """Best-effort RNA identity tokens without relying on wrapper identity."""
    result: set[int] = set()
    for candidate in (obj, getattr(obj, "original", None)):
        if candidate is None:
            continue
        pointer = getattr(candidate, "as_pointer", None)
        if callable(pointer):
            try:
                value = int(pointer())
                if value:
                    result.add(value)
            except Exception:
                pass
        result.add(id(candidate))
    return result


def _raycast_index_revision_key(geometry_revision: int) -> tuple[int, str, str, int]:
    scene = getattr(getattr(bpy, "context", None), "scene", None)
    view_layer = getattr(getattr(bpy, "context", None), "view_layer", None)
    return (
        geometry_revision,
        str(getattr(scene, "name", "")),
        str(getattr(view_layer, "name", "")),
        int(getattr(scene, "frame_current", 0) or 0),
    )


def _store_raycast_index(
    geometry_revision: int,
    records: list[dict[str, Any]],
) -> None:
    global _raycast_index_key, _raycast_index_records
    # The index contains only strings, integers, and 4x4 float matrices.  It is
    # explicitly revision-bound and never retains evaluated RNA wrappers.
    _raycast_index_key = _raycast_index_revision_key(geometry_revision)
    _raycast_index_records = records


def _raycast_instance_records(
    depsgraph: Any,
    geometry_revision: int,
) -> list[dict[str, Any]]:
    """Build a command-local map from ray-cast object/matrix to context IDs."""
    if _raycast_index_key == _raycast_index_revision_key(geometry_revision):
        return _raycast_index_records
    records: list[dict[str, Any]] = []
    # DepsgraphObjectInstance wrappers are iterator-scoped in Blender.  Turning
    # the collection into a list invalidates early wrappers before they are
    # read, producing ``StructRNA ... has been removed`` on large scenes.
    instances = getattr(depsgraph, "object_instances", ())
    for instance in instances:
        evaluated, source = _instance_source(instance)
        if evaluated is None or source is None:
            continue
        matrix = getattr(instance, "matrix_world", None) or getattr(
            evaluated, "matrix_world", None
        )
        records.append(
            {
                "id": _instance_identifier(instance, source),
                "names": _object_match_names(evaluated) | _object_match_names(source),
                "pointers": _object_pointer_tokens(evaluated)
                | _object_pointer_tokens(source),
                "matrix": _matrix_rows(matrix),
            }
        )
    _store_raycast_index(geometry_revision, records)
    return records


class _RaycastProxy:
    """Minimal non-RNA hit identity returned by the revision-bound BVH path."""

    def __init__(self, name: str, object_id: str, material_name: str | None):
        self.name = name
        self.name_full = name
        self._blend_ai_object_id = object_id
        self._blend_ai_material_name = material_name
        self.data = None
        self.material_slots = ()


def _source_accel_key(evaluated: Any, source: Any) -> str:
    data = getattr(evaluated, "data", None)
    pointer = getattr(data, "as_pointer", None)
    try:
        data_token = int(pointer()) if callable(pointer) else id(data)
    except Exception:
        data_token = id(data)
    return f"{getattr(source, 'name_full', getattr(source, 'name', 'Mesh'))}:{data_token}"


def _mesh_bvh_record(evaluated: Any) -> dict[str, Any] | None:
    try:
        from mathutils.bvhtree import BVHTree
    except Exception:
        return None
    to_mesh = getattr(evaluated, "to_mesh", None)
    if not callable(to_mesh):
        return None
    mesh = None
    try:
        try:
            mesh = to_mesh(preserve_all_data_layers=False)
        except TypeError:
            mesh = to_mesh()
        calculate = getattr(mesh, "calc_loop_triangles", None)
        if callable(calculate):
            calculate()
        vertices = [tuple(float(value) for value in vertex.co) for vertex in mesh.vertices]
        triangles = [tuple(int(index) for index in item.vertices) for item in mesh.loop_triangles]
        if not vertices or not triangles:
            return None
        polygon_indices = [int(item.polygon_index) for item in mesh.loop_triangles]
        polygon_materials: list[str | None] = []
        materials = getattr(mesh, "materials", ())
        polygons = getattr(mesh, "polygons", ())
        for polygon_index in polygon_indices:
            material_name = None
            try:
                material_index = int(polygons[polygon_index].material_index)
                material = materials[material_index]
                material_name = str(getattr(material, "name", "")) or None
            except Exception:
                pass
            polygon_materials.append(material_name)
        return {
            "bvh": BVHTree.FromPolygons(vertices, triangles, all_triangles=True),
            "polygon_indices": polygon_indices,
            "material_names": polygon_materials,
        }
    finally:
        clear = getattr(evaluated, "to_mesh_clear", None)
        if mesh is not None and callable(clear):
            try:
                clear()
            except Exception:
                pass


def _combined_aabb(records: list[dict[str, Any]], indices: list[int]) -> tuple[list[float], list[float]]:
    minimum = [math.inf, math.inf, math.inf]
    maximum = [-math.inf, -math.inf, -math.inf]
    for index in indices:
        bounds = records[index]["bounds"]
        for axis in range(3):
            minimum[axis] = min(minimum[axis], float(bounds["min"][axis]))
            maximum[axis] = max(maximum[axis], float(bounds["max"][axis]))
    return minimum, maximum


def _build_aabb_tree(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int | None]:
    nodes: list[dict[str, Any]] = []

    def build(indices: list[int]) -> int:
        minimum, maximum = _combined_aabb(records, indices)
        node_index = len(nodes)
        nodes.append({})
        if len(indices) <= 8:
            nodes[node_index] = {
                "min": minimum,
                "max": maximum,
                "indices": indices,
                "left": None,
                "right": None,
            }
            return node_index
        extents = [maximum[axis] - minimum[axis] for axis in range(3)]
        axis = max(range(3), key=extents.__getitem__)
        indices.sort(key=lambda index: float(records[index]["bounds"]["center"][axis]))
        midpoint = len(indices) // 2
        left = build(indices[:midpoint])
        right = build(indices[midpoint:])
        nodes[node_index] = {
            "min": minimum,
            "max": maximum,
            "indices": None,
            "left": left,
            "right": right,
        }
        return node_index

    if not records:
        return nodes, None
    return nodes, build(list(range(len(records))))


def _ensure_raycast_accel(depsgraph: Any, geometry_revision: int) -> dict[str, Any] | None:
    """Build a large-scene instance BVH without retaining evaluated RNA."""
    global _raycast_accel_key, _raycast_accel
    key = _raycast_index_revision_key(geometry_revision)
    if _raycast_accel_key == key:
        return _raycast_accel
    try:
        from mathutils import Matrix
    except Exception:
        return None

    source_records: dict[str, dict[str, Any]] = {}
    instances: list[dict[str, Any]] = []
    for instance in getattr(depsgraph, "object_instances", ()):
        evaluated, source = _instance_source(instance)
        if (
            evaluated is None
            or source is None
            or str(getattr(source, "type", "")) != "MESH"
        ):
            continue
        matrix = getattr(instance, "matrix_world", None) or getattr(
            evaluated, "matrix_world", None
        )
        bounds = _bounds(evaluated, matrix)
        if bounds is None:
            continue
        source_key = _source_accel_key(evaluated, source)
        if source_key not in source_records:
            source_record = _mesh_bvh_record(evaluated)
            if source_record is None:
                continue
            source_records[source_key] = source_record
        matrix_value = Matrix(matrix)
        try:
            inverse = matrix_value.inverted()
            normal = matrix_value.to_3x3().inverted().transposed()
        except Exception:
            continue
        instances.append(
            {
                "id": _instance_identifier(instance, source),
                "name": str(getattr(source, "name", "")),
                "source_key": source_key,
                "matrix": _matrix_rows(matrix_value),
                "inverse": _matrix_rows(inverse),
                "normal": [
                    [float(normal[row][column]) for column in range(3)]
                    for row in range(3)
                ],
                "bounds": {
                    field: bounds[field] for field in ("min", "max", "center", "size")
                },
            }
        )
    nodes, root = _build_aabb_tree(instances)
    _raycast_accel = {
        "instances": instances,
        "sources": source_records,
        "nodes": nodes,
        "root": root,
    }
    _raycast_accel_key = key
    return _raycast_accel


def _ray_aabb_entry(
    origin: list[float],
    direction: list[float],
    distance: float,
    minimum: list[float],
    maximum: list[float],
) -> float | None:
    lower = 0.0
    upper = distance
    for axis in range(3):
        component = float(direction[axis])
        if abs(component) <= 1e-15:
            if origin[axis] < minimum[axis] or origin[axis] > maximum[axis]:
                return None
            continue
        first = (minimum[axis] - origin[axis]) / component
        second = (maximum[axis] - origin[axis]) / component
        if first > second:
            first, second = second, first
        lower = max(lower, first)
        upper = min(upper, second)
        if lower > upper:
            return None
    return lower


def _accel_candidates(
    accel: dict[str, Any],
    origin: list[float],
    direction: list[float],
    distance: float,
) -> list[tuple[float, int]]:
    root = accel.get("root")
    if root is None:
        return []
    nodes = accel["nodes"]
    records = accel["instances"]
    pending = [int(root)]
    candidates: list[tuple[float, int]] = []
    while pending:
        node = nodes[pending.pop()]
        entry = _ray_aabb_entry(
            origin, direction, distance, node["min"], node["max"]
        )
        if entry is None:
            continue
        indices = node.get("indices")
        if indices is not None:
            for index in indices:
                record = records[index]
                record_entry = _ray_aabb_entry(
                    origin,
                    direction,
                    distance,
                    record["bounds"]["min"],
                    record["bounds"]["max"],
                )
                if record_entry is not None:
                    candidates.append((record_entry, index))
        else:
            pending.append(int(node["left"]))
            pending.append(int(node["right"]))
    candidates.sort(key=lambda item: item[0])
    return candidates


def _accel_ray_cast(
    accel: dict[str, Any],
    origin: list[float],
    direction: list[float],
    distance: float,
) -> Any:
    try:
        from mathutils import Matrix, Vector
    except Exception:
        return None
    nearest: tuple[float, Any] | None = None
    for entry, index in _accel_candidates(accel, origin, direction, distance):
        if nearest is not None and entry > nearest[0]:
            break
        instance = accel["instances"][index]
        source = accel["sources"][instance["source_key"]]
        inverse = Matrix(instance["inverse"])
        local_origin = inverse @ Vector(origin)
        local_delta = inverse.to_3x3() @ Vector(direction)
        scale = float(local_delta.length)
        if not math.isfinite(scale) or scale <= 1e-15:
            continue
        local_direction = local_delta / scale
        hit = source["bvh"].ray_cast(
            local_origin, local_direction, distance * scale
        )
        location, normal, triangle_index, _local_distance = hit
        if location is None or normal is None or triangle_index is None:
            continue
        matrix = Matrix(instance["matrix"])
        world_location = matrix @ location
        world_distance = float((world_location - Vector(origin)).dot(Vector(direction)))
        if world_distance < -1e-7 or world_distance > distance + 1e-7:
            continue
        normal_matrix = Matrix(instance["normal"])
        world_normal = normal_matrix @ normal
        if world_normal.length > 0.0:
            world_normal.normalize()
        triangle = int(triangle_index)
        try:
            polygon_index = int(source["polygon_indices"][triangle])
            material_name = source["material_names"][triangle]
        except Exception:
            polygon_index = triangle
            material_name = None
        proxy = _RaycastProxy(instance["name"], instance["id"], material_name)
        result = (
            True,
            world_location,
            world_normal,
            polygon_index,
            proxy,
            matrix,
        )
        if nearest is None or world_distance < nearest[0]:
            nearest = (world_distance, result)
    if nearest is None:
        return (False, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), -1, None, None)
    return nearest[1]


def _matrix_match_distance(left: Any, right: Any) -> tuple[float, float] | None:
    left_rows = _matrix_rows(left)
    right_rows = _matrix_rows(right)
    if not left_rows or not right_rows:
        return None
    differences = [
        abs(left_rows[row][column] - right_rows[row][column])
        for row in range(4)
        for column in range(4)
    ]
    magnitude = max(
        1.0,
        *(abs(component) for row in left_rows for component in row),
        *(abs(component) for row in right_rows for component in row),
    )
    return max(differences), magnitude


def _raycast_object_id(
    hit_object: Any,
    hit_matrix: Any,
    records: list[dict[str, Any]],
    geometry_revision: int,
) -> str | None:
    """Resolve a hit to the exact instance ID emitted by lighting context."""
    if hit_object is None:
        return None
    accelerated_id = getattr(hit_object, "_blend_ai_object_id", None)
    if isinstance(accelerated_id, str):
        return accelerated_id
    hit_names = _object_match_names(hit_object)
    hit_pointers = _object_pointer_tokens(hit_object)
    candidates = [
        record
        for record in records
        if hit_pointers & set(record["pointers"])
        or hit_names & set(record["names"])
    ]
    if candidates:
        matrix_candidates: list[tuple[float, float, dict[str, Any]]] = []
        for record in candidates:
            distance = _matrix_match_distance(hit_matrix, record["matrix"])
            if distance is not None:
                matrix_candidates.append((*distance, record))
        if matrix_candidates:
            difference, magnitude, record = min(
                matrix_candidates,
                key=lambda item: (item[0] / item[1], item[2]["id"]),
            )
            if difference <= max(1e-6, magnitude * 1e-6):
                return str(record["id"])
        if len(candidates) == 1:
            return str(candidates[0]["id"])

    # A result outside depsgraph.object_instances is unexpected, but retaining
    # the prior explicit fallback is more useful than dropping object identity.
    object_name = str(getattr(hit_object, "name", ""))
    return f"{object_name}:g{geometry_revision}" if object_name else None


def _material_name(obj: Any, face_index: int) -> str | None:
    if isinstance(obj, _RaycastProxy):
        return obj._blend_ai_material_name
    data = getattr(obj, "data", None)
    material_index = 0
    polygons = getattr(data, "polygons", None)
    try:
        if face_index >= 0 and polygons is not None and face_index < len(polygons):
            material_index = int(polygons[face_index].material_index)
    except Exception:
        material_index = 0
    slots = getattr(obj, "material_slots", None)
    try:
        if slots is not None and material_index < len(slots):
            material = getattr(slots[material_index], "material", None)
            name = getattr(material, "name", None)
            return str(name) if name else None
    except Exception:
        pass
    materials = getattr(data, "materials", None)
    try:
        material = materials[material_index]
        name = getattr(material, "name", None)
        return str(name) if name else None
    except Exception:
        return None


def _material_names(obj: Any) -> list[str]:
    """Return unique material-slot names without retaining Blender RNA."""
    names: set[str] = set()
    for slot in _iter_values(getattr(obj, "material_slots", None)):
        material = getattr(slot, "material", None)
        name = getattr(material, "name", None)
        if name:
            names.add(str(name))
    for material in _iter_values(getattr(getattr(obj, "data", None), "materials", None)):
        name = getattr(material, "name", None)
        if name:
            names.add(str(name))
    return sorted(names)


def _instance_record(
    instance: Any,
    evaluated: Any,
    source: Any,
    matrix: Any,
    item_bounds: Any,
    paths: dict[str, list[str]],
    view_layer: Any,
    semantic_terms: list[str],
    projection: dict[str, Any] | None = None,
    collection_info: tuple[list[str], list[list[str]]] | None = None,
) -> dict[str, Any]:
    collection_names, collection_paths = collection_info or _instance_collections(
        instance, source, paths
    )
    visible_viewport, visible_render = _is_visible(source, view_layer)
    lower_name = str(getattr(source, "name", "")).lower()
    semantic_matches = [term for term in semantic_terms if term.lower() in lower_name]
    parent = getattr(source, "parent", None)
    instance_id = _instance_identifier(instance, source)
    persistent_id = [int(value) for value in getattr(instance, "persistent_id", ())]
    while persistent_id and persistent_id[-1] == 0:
        persistent_id.pop()
    instance_source = getattr(instance, "instance_object", None) or getattr(
        instance, "parent", None
    )
    result = {
        "revision_scoped_id": instance_id,
        "instance_id": instance_id,
        "name": str(getattr(evaluated, "name", getattr(source, "name", ""))),
        "source_name": str(getattr(source, "name", "")),
        "instance_source": str(getattr(instance_source, "name", "")) or None,
        "persistent_id": persistent_id,
        "type": str(getattr(source, "type", getattr(evaluated, "type", "UNKNOWN"))),
        "is_instance": bool(getattr(instance, "is_instance", False)),
        "matrix_world": _matrix_rows(matrix) or None,
        "world_aabb": item_bounds,
        "bounds": item_bounds,
        "collection_ids": collection_names,
        "collections": collection_names,
        "collection_paths": collection_paths,
        "parent_id": str(getattr(parent, "name", "")) if parent is not None else None,
        "parent": str(getattr(parent, "name", "")) if parent is not None else None,
        "viewport_visible": visible_viewport,
        "render_visible": visible_render,
        "visible_viewport": visible_viewport,
        "visible_render": visible_render,
        "material_names": _material_names(evaluated),
        "semantic_tags": semantic_matches,
        "semantic_matches": semantic_matches,
    }
    if projection is not None:
        result["camera"] = {
            "in_front": (
                projection.get("depth_range") is not None
                and projection["depth_range"][1] > 0.0
            ),
            "in_frustum": projection.get("in_frustum"),
            "screen_rect": projection.get("screen_rect"),
            "depth_range": projection.get("depth_range"),
            "projected_area": projection.get("projected_area"),
        }
        result.update(projection)
    return result


def _triangle_candidates(
    evaluated: Any,
    source: Any,
    matrix: Any,
    instance_id: str,
    triangle_budget: int,
    semantic_terms: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, list[str]]:
    """Scan evaluated triangles and return bounded large-surface/opening heaps."""
    if triangle_budget <= 0 or str(getattr(source, "type", "")) != "MESH":
        return [], [], 0, []
    to_mesh = getattr(evaluated, "to_mesh", None)
    if not callable(to_mesh):
        return [], [], 0, []

    warnings: list[str] = []
    mesh = None
    surfaces: list[tuple[float, int, dict[str, Any]]] = []
    openings: list[tuple[float, int, dict[str, Any]]] = []
    scanned = 0
    sequence = 0
    try:
        try:
            mesh = to_mesh(preserve_all_data_layers=False)
        except TypeError:
            mesh = to_mesh()
        calc = getattr(mesh, "calc_loop_triangles", None)
        if callable(calc):
            calc()
        vertices = getattr(mesh, "vertices", ())
        polygons = getattr(mesh, "polygons", ())
        source_name = str(getattr(source, "name", ""))
        object_lower = source_name.lower()
        for triangle in getattr(mesh, "loop_triangles", ()):
            if scanned >= triangle_budget:
                break
            scanned += 1
            indices = list(getattr(triangle, "vertices", ()))
            if len(indices) != 3:
                continue
            try:
                points = [_transform_point(matrix, vertices[index].co) for index in indices]
            except Exception:
                continue
            edge_a = _sub(points[1], points[0])
            edge_b = _sub(points[2], points[0])
            cross = _cross(edge_a, edge_b)
            try:
                normal, doubled_area = _normalize(cross, "triangle normal")
            except ValueError:
                continue
            area = doubled_area * 0.5
            if area <= 1e-10:
                continue
            polygon_index = int(getattr(triangle, "polygon_index", -1))
            material_name = None
            try:
                material_index = int(polygons[polygon_index].material_index)
                materials = getattr(mesh, "materials", ())
                if material_index < len(materials):
                    material_name = str(getattr(materials[material_index], "name", "")) or None
            except Exception:
                pass
            center = [sum(point[axis] for point in points) / 3.0 for axis in range(3)]
            combined = " ".join(filter(None, (object_lower, (material_name or "").lower())))
            semantic_matches = [term for term in semantic_terms if term.lower() in combined]
            opening_words = [word for word in OPENING_SEMANTIC_WORDS if word in combined]
            candidate = {
                "object_id": instance_id,
                "instance_id": instance_id,
                "object_name": source_name,
                "polygon_index": polygon_index,
                "center": center,
                "normal": normal,
                "world_area": area,
                "area": area,
                "dimensions": [
                    max(point[axis] for point in points) - min(point[axis] for point in points)
                    for axis in range(3)
                ],
                "material_name": material_name,
                "semantic_matches": semantic_matches,
                "score": area * (1.25 if semantic_matches else 1.0),
                "reasons": semantic_matches or ["large evaluated surface"],
            }
            sequence += 1
            heapq.heappush(surfaces, (area, sequence, candidate))
            if len(surfaces) > MAX_SURFACE_RESULTS:
                heapq.heappop(surfaces)

            # Openings cannot be proven from triangle geometry alone.  These
            # candidates are explicitly labelled as name/normal heuristics.
            if opening_words:
                opening = dict(candidate)
                opening["heuristic"] = True
                opening["heuristic_reasons"] = opening_words
                opening["upward_facing"] = normal[2] > 0.5
                opening["reasons"] = opening["heuristic_reasons"]
                opening["confidence"] = min(
                    0.95,
                    0.35
                    + 0.1 * len(opening["heuristic_reasons"])
                    + (0.15 if opening["upward_facing"] else 0.0),
                )
                sequence += 1
                heapq.heappush(openings, (area, sequence, opening))
                if len(openings) > MAX_SURFACE_RESULTS:
                    heapq.heappop(openings)
    except ValueError:
        # Degenerate triangles are expected in imperfect production meshes.
        pass
    except Exception as exc:
        warnings.append(f"Surface scan skipped for '{getattr(source, 'name', '')}': {exc}")
    finally:
        clear = getattr(evaluated, "to_mesh_clear", None)
        if mesh is not None and callable(clear):
            try:
                clear()
            except Exception:
                pass

    surface_result = [entry[2] for entry in sorted(surfaces, key=lambda item: item[0], reverse=True)]
    opening_result = [entry[2] for entry in sorted(openings, key=lambda item: item[0], reverse=True)]
    return surface_result, opening_result, scanned, warnings


def _light_context(scene: Any) -> list[dict[str, Any]]:
    result = []
    for obj in _iter_values(getattr(scene, "objects", None)):
        if str(getattr(obj, "type", "")) != "LIGHT":
            continue
        data = getattr(obj, "data", None)
        direction = None
        try:
            from mathutils import Vector

            direction = _sequence(
                getattr(obj, "matrix_world").to_quaternion() @ Vector((0.0, 0.0, -1.0)),
                3,
            )
        except Exception:
            pass
        item = {
            "name": str(getattr(obj, "name", "")),
            "managed_plan_id": _custom_get(obj, MANAGED_PLAN_PROP)
            if bool(_custom_get(obj, MANAGED_PROP, False))
            else None,
            "managed_light_id": _custom_get(obj, MANAGED_ID_PROP)
            if bool(_custom_get(obj, MANAGED_PROP, False))
            else None,
            "managed_id": _custom_get(obj, MANAGED_ID_PROP)
            if bool(_custom_get(obj, MANAGED_PROP, False))
            else None,
            "type": str(getattr(data, "type", "")),
            "matrix_world": _matrix_rows(getattr(obj, "matrix_world", None)) or None,
            "direction": direction,
            "location_world": _world_location(obj),
            "energy": float(getattr(data, "energy", 0.0) or 0.0),
            "color_rgb": _sequence(getattr(data, "color", (1.0, 1.0, 1.0)), 3),
            "use_shadow": bool(getattr(data, "use_shadow", True)),
            "visible_render": not bool(getattr(obj, "hide_render", False)),
            "visible_viewport": not bool(getattr(obj, "hide_viewport", False)),
        }
        light_type = item["type"]
        if light_type in {"POINT", "SPOT"}:
            item["radius"] = float(getattr(data, "shadow_soft_size", 0.0) or 0.0)
        if light_type == "SUN":
            item["sun_angle_degrees"] = math.degrees(float(getattr(data, "angle", 0.0)))
        if light_type == "AREA":
            item["area_shape"] = str(getattr(data, "shape", "SQUARE"))
            item["size"] = float(getattr(data, "size", 1.0) or 1.0)
            item["size_y"] = float(getattr(data, "size_y", item["size"]) or item["size"])
        if light_type == "SPOT":
            item["spot_angle_degrees"] = math.degrees(
                float(getattr(data, "spot_size", math.radians(45.0)))
            )
            item["spot_blend"] = float(getattr(data, "spot_blend", 0.15))
        result.append(item)
    return result


def _world_context(scene: Any) -> dict[str, Any] | None:
    world = getattr(scene, "world", None)
    if world is None:
        return None
    result = {
        "name": str(getattr(world, "name", "")),
        "managed": bool(_custom_get(world, MANAGED_WORLD_PROP, False)),
        "use_nodes": bool(getattr(world, "use_nodes", False)),
        "color_rgb": _sequence(getattr(world, "color", (0.0, 0.0, 0.0)), 3),
    }
    tree = getattr(world, "node_tree", None)
    for node in _iter_values(getattr(tree, "nodes", None)):
        if str(getattr(node, "type", "")) == "BACKGROUND":
            try:
                result["background_color_rgb"] = _sequence(node.inputs["Color"].default_value, 4)[:3]
                result["strength"] = float(node.inputs["Strength"].default_value)
            except Exception:
                pass
            break
    return result


def _cursor_signature(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _encode_cursor(offset: int, geometry_revision: int, signature: str) -> str:
    raw = json.dumps({"o": offset, "g": geometry_revision, "s": signature}, separators=(",", ":"))
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii").rstrip("=")


def _decode_cursor(value: Any, geometry_revision: int, signature: str) -> int:
    if value is None:
        return 0
    if not isinstance(value, str) or len(value) > 1024:
        raise ValueError("cursor must be an opaque string or null")
    try:
        padded = value + "=" * (-len(value) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
        offset = _integer(payload["o"], "cursor offset", minimum=0)
        if payload["g"] != geometry_revision or payload["s"] != signature:
            raise ValueError("cursor is stale for the current geometry or query")
        return offset
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("cursor is invalid") from exc


def _payload_size(value: Any) -> int:
    return len(json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8"))


def _fit_context_payload(
    result: dict[str, Any],
    *,
    offset: int,
    total: int,
    geometry_revision: int,
    signature: str,
) -> None:
    """Deterministically fit variable records below response safety ceilings."""
    trimmed = False
    while _payload_size(result) > DEFAULT_PAYLOAD_TARGET_BYTES and len(result["instances"]) > 1:
        result["instances"].pop()
        trimmed = True
    candidates = result["candidates"]
    while _payload_size(result) > DEFAULT_PAYLOAD_TARGET_BYTES and (
        candidates["surfaces"] or candidates["openings"]
    ):
        key = "surfaces" if len(candidates["surfaces"]) >= len(candidates["openings"]) else "openings"
        candidates[key].pop()
        trimmed = True
    while _payload_size(result) > MAX_PAYLOAD_BYTES and len(result["instances"]) > 1:
        result["instances"].pop()
        trimmed = True
    if _payload_size(result) > MAX_PAYLOAD_BYTES:
        raise RuntimeError("Lighting context metadata exceeds the absolute 2 MiB response limit")

    next_offset = offset + len(result["instances"])
    more_instances = next_offset < total
    result["next_cursor"] = (
        _encode_cursor(next_offset, geometry_revision, signature) if more_instances else None
    )
    result["truncated"] = bool(more_instances or trimmed)
    result["page"] = {
        "offset": offset,
        "returned": len(result["instances"]),
        "total": total,
        "complete": not more_instances,
    }
    if trimmed:
        result["warnings"].append(
            "Response was deterministically trimmed to stay below the payload safety target"
        )


def handle_get_lighting_context(params: dict[str, Any]) -> dict[str, Any]:
    """Return a bounded, revisioned spatial description of the evaluated scene."""
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    _reject_unknown(
        params,
        {
            "camera_name",
            "scope",
            "collection_names",
            "detail",
            "semantic_terms",
            "include_hidden",
            "max_instances",
            "cursor",
            "max_surface_triangles",
            "cache_mode",
        },
    )
    started = time.perf_counter()
    scene = bpy.context.scene
    view_layer = getattr(bpy.context, "view_layer", None)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    revisions = spatial_cache.get_revisions()

    scope = str(params.get("scope", "CAMERA")).upper()
    if scope not in {"CAMERA", "SCENE", "COLLECTIONS"}:
        raise ValueError("scope must be CAMERA, SCENE, or COLLECTIONS")
    detail = str(params.get("detail", "CANDIDATES")).upper()
    if detail not in {"BOUNDS", "CANDIDATES"}:
        raise ValueError("detail must be BOUNDS or CANDIDATES")
    include_hidden = params.get("include_hidden", False)
    _bool(include_hidden, "include_hidden")
    max_instances = _integer(
        params.get("max_instances", DEFAULT_MAX_INSTANCES),
        "max_instances",
        minimum=1,
        maximum=MAX_INSTANCES,
    )
    max_surface_triangles = _integer(
        params.get("max_surface_triangles", DEFAULT_MAX_SURFACE_TRIANGLES),
        "max_surface_triangles",
        minimum=0,
        maximum=MAX_SURFACE_TRIANGLES,
    )
    cache_mode = str(params.get("cache_mode", "USE")).upper()
    if cache_mode not in {"USE", "REFRESH"}:
        raise ValueError("cache_mode must be USE or REFRESH")

    collection_names = params.get("collection_names", [])
    if collection_names is None:
        collection_names = []
    if not isinstance(collection_names, list) or len(collection_names) > 64:
        raise ValueError("collection_names must be an array of at most 64 names")
    collection_names = [_safe_name(value, f"collection_names[{index}]") for index, value in enumerate(collection_names)]
    if any(len(value) > 63 for value in collection_names):
        raise ValueError("collection_names entries must be at most 63 characters")
    if scope == "COLLECTIONS" and not collection_names:
        raise ValueError("collection_names is required when scope is COLLECTIONS")
    semantic_terms = params.get("semantic_terms", []) or []
    if not isinstance(semantic_terms, list) or len(semantic_terms) > 64:
        raise ValueError("semantic_terms must be an array of at most 64 strings")
    semantic_terms = [
        _safe_name(value, f"semantic_terms[{index}]")
        for index, value in enumerate(semantic_terms)
    ]
    if any(len(value) > 64 for value in semantic_terms):
        raise ValueError("semantic_terms entries must be at most 64 characters")
    camera = _camera_object(scene, params.get("camera_name"))
    if scope == "CAMERA" and camera is None:
        raise ValueError("No active camera is available for CAMERA scope")

    signature_payload = {
        "scene": str(getattr(scene, "name", "")),
        "view_layer": str(getattr(view_layer, "name", "")),
        "frame": int(getattr(scene, "frame_current", 0) or 0),
        "camera": str(getattr(camera, "name", "")) if camera else None,
        "scope": scope,
        "collections": sorted(collection_names),
        "detail": detail,
        "semantic_terms": semantic_terms,
        "include_hidden": include_hidden,
        "max_instances": max_instances,
        "max_surface_triangles": max_surface_triangles,
    }
    signature = _cursor_signature(signature_payload)
    offset = _decode_cursor(params.get("cursor"), revisions["geometry_revision"], signature)
    cache_payload = dict(signature_payload)
    # Geometry pages are intentionally reusable across managed-light edits.
    # Light/world/exposure fields are refreshed on a cache hit below.
    cache_payload["geometry_revision"] = revisions["geometry_revision"]
    cache_payload["offset"] = offset
    cache_key = spatial_cache.make_cache_key("lighting_context", cache_payload)
    if cache_mode == "USE":
        cached = spatial_cache.get_cached(cache_key)
        if cached is not None:
            current_scene_revision = spatial_cache.make_scene_revision(scene)
            cached["geometry_revision"] = revisions["geometry_revision"]
            cached["lighting_revision"] = revisions["lighting_revision"]
            cached["scene_revision"] = current_scene_revision
            cached["scene"]["geometry_revision"] = revisions["geometry_revision"]
            cached["scene"]["lighting_revision"] = revisions["lighting_revision"]
            cached["scene"]["scene_revision"] = current_scene_revision
            cached["scene"]["exposure"] = float(
                getattr(getattr(scene, "view_settings", None), "exposure", 0.0)
            )
            render = getattr(scene, "render", None)
            cycles = getattr(scene, "cycles", None)
            cached["scene"]["render_engine"] = str(
                getattr(render, "engine", "")
            )
            cached["scene"]["cycles_device"] = (
                str(getattr(cycles, "device", "")) or None
            )
            cached["scene"]["preview_samples"] = int(
                getattr(cycles, "preview_samples", 0) or 0
            )
            world_summary = _world_context(scene)
            cached["scene"]["world_summary"] = world_summary
            cached["scene"]["world"] = world_summary
            cached["lights"] = _light_context(scene)
            cached["cache"] = {
                "hit": True,
                "age_ms": (
                    spatial_cache.get_cache_age_ms(cache_key)
                    if hasattr(spatial_cache, "get_cache_age_ms")
                    else None
                ),
                "build_ms": cached.get("cache", {}).get("build_ms"),
            }
            return _plain(cached)

    paths = _collection_paths(scene)
    scene_membership = _scene_collection_membership(scene, paths)
    direct_collection_cache: dict[str, tuple[str, ...]] = {}
    requested_collections = set(collection_names)
    ranked_records: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    scene_bounds = None
    collection_bounds: dict[str, dict[str, list[float]] | None] = {
        name: None for name in paths if scope != "COLLECTIONS" or name in requested_collections
    }
    collection_counts = {name: 0 for name in collection_bounds}
    warnings: list[str] = []
    eligible_count = 0
    surface_heaps: list[tuple[float, int, dict[str, Any]]] = []
    opening_heaps: list[tuple[float, int, dict[str, Any]]] = []
    candidate_scan_queue: list[
        tuple[tuple[Any, ...], Any, Any, Any, str]
    ] = []
    surface_scanned = 0
    mesh_instances_scanned = 0
    sequence = 0
    raycast_index_records: list[dict[str, Any]] = []

    try:
        instances = getattr(depsgraph, "object_instances", ())
    except Exception as exc:
        raise RuntimeError(f"Unable to enumerate evaluated scene instances: {exc}") from exc

    for instance in instances:
        evaluated, source = _instance_source(instance)
        if evaluated is None or source is None:
            continue
        matrix = getattr(instance, "matrix_world", None) or getattr(evaluated, "matrix_world", None)
        instance_id = _instance_identifier(instance, source)
        raycast_index_records.append(
            {
                "id": instance_id,
                "names": sorted(
                    _object_match_names(evaluated) | _object_match_names(source)
                ),
                "pointers": [],
                "matrix": _matrix_rows(matrix),
            }
        )
        item_bounds = _bounds(evaluated, matrix)
        collection_names_for_instance, _collection_paths_for_instance = (
            _instance_collections(
                instance,
                source,
                paths,
                scene_membership,
                direct_collection_cache,
            )
        )
        collection_set = set(collection_names_for_instance)
        visible_viewport, visible_render = _is_visible(source, view_layer)
        if not include_hidden and not (visible_viewport or visible_render):
            continue
        if scope == "COLLECTIONS" and not (collection_set & requested_collections):
            continue
        eligible_count += 1
        scene_bounds = _merge_bounds(scene_bounds, item_bounds)
        for collection_name in collection_set:
            if collection_name in collection_bounds:
                collection_bounds[collection_name] = _merge_bounds(
                    collection_bounds[collection_name], item_bounds
                )
                collection_counts[collection_name] += 1
        projection = _project_bounds(scene, camera, item_bounds) if scope == "CAMERA" else None
        public_bounds = (
            {
                key: item_bounds[key]
                for key in ("min", "max", "center", "size")
            }
            if item_bounds is not None
            else None
        )
        record = _instance_record(
            instance,
            evaluated,
            source,
            matrix,
            public_bounds,
            paths,
            view_layer,
            semantic_terms,
            projection,
            (
                collection_names_for_instance,
                _collection_paths_for_instance,
            ),
        )
        rank = (
            0 if projection and projection.get("in_frustum") else 1,
            0 if record["semantic_matches"] else 1,
            -float(projection.get("projected_area") or 0.0) if projection else 0.0,
            instance_id,
        )
        ranked_records.append((rank, record))

        if detail == "CANDIDATES" and str(getattr(source, "type", "")) == "MESH":
            semantic_text = " ".join(
                [str(getattr(source, "name", "")), *record["material_names"]]
            ).lower()
            opening_semantic = any(
                word in semantic_text for word in OPENING_SEMANTIC_WORDS
            )
            in_frustum = bool(projection and projection.get("in_frustum"))
            projected_area = (
                float(projection.get("projected_area") or 0.0)
                if projection
                else 0.0
            )
            size = item_bounds.get("size", [0.0, 0.0, 0.0]) if item_bounds else []
            bounds_size = sum(abs(float(value)) for value in size)
            scan_rank = (
                0 if opening_semantic else 1,
                0 if record["semantic_matches"] else 1,
                0 if in_frustum else 1,
                -projected_area,
                -bounds_size,
                instance_id,
            )
            candidate_scan_queue.append(
                (scan_rank, evaluated, source, matrix, instance_id)
            )

    _store_raycast_index(revisions["geometry_revision"], raycast_index_records)
    acceleration_build_ms = 0.0
    if len(raycast_index_records) > 5_000:
        acceleration_started = time.perf_counter()
        _ensure_raycast_accel(depsgraph, revisions["geometry_revision"])
        acceleration_build_ms = (time.perf_counter() - acceleration_started) * 1000.0

    # Mesh-level work is deliberately second-stage.  Sorting the lightweight
    # instance records first prevents an unrelated high-poly mesh encountered
    # early in depsgraph order from exhausting the triangle budget before a
    # later skylight/window candidate can be inspected.
    candidate_scan_queue.sort(key=lambda item: item[0])
    for _rank, evaluated, source, matrix, instance_id in candidate_scan_queue:
        if surface_scanned >= max_surface_triangles:
            break
        remaining = max_surface_triangles - surface_scanned
        surfaces, openings, scanned, scan_warnings = _triangle_candidates(
            evaluated, source, matrix, instance_id, remaining, semantic_terms
        )
        if scanned:
            mesh_instances_scanned += 1
        surface_scanned += scanned
        warnings.extend(scan_warnings)
        for candidate in surfaces:
            sequence += 1
            heapq.heappush(surface_heaps, (candidate["area"], sequence, candidate))
            if len(surface_heaps) > MAX_SURFACE_RESULTS:
                heapq.heappop(surface_heaps)
        for candidate in openings:
            sequence += 1
            heapq.heappush(opening_heaps, (candidate["area"], sequence, candidate))
            if len(opening_heaps) > MAX_SURFACE_RESULTS:
                heapq.heappop(opening_heaps)

    ranked_records.sort(key=lambda item: item[0])
    records = [item[1] for item in ranked_records[offset : offset + max_instances]]
    next_offset = offset + len(records)
    complete = next_offset >= eligible_count
    next_cursor = None if complete else _encode_cursor(
        next_offset, revisions["geometry_revision"], signature
    )
    if surface_scanned >= max_surface_triangles and max_surface_triangles:
        warnings.append("Surface scan reached max_surface_triangles; candidates are partial")
    if scope == "CAMERA" and camera and str(getattr(getattr(camera, "data", None), "type", "")) == "PANO":
        warnings.append("Panoramic projection is unsupported; view-layer objects were retained")

    camera_payload = _camera_context(scene, camera, depsgraph)
    if camera_payload is not None:
        camera_payload["projection_supported"] = camera_payload.pop(
            "frustum_test_supported", True
        )
    surfaces_result = [
        entry[2] for entry in sorted(surface_heaps, key=lambda item: item[0], reverse=True)
    ]
    openings_result = [
        entry[2] for entry in sorted(opening_heaps, key=lambda item: item[0], reverse=True)
    ]
    collection_result = [
        {
            "id": name,
            "name": name,
            "parent_id": (
                paths[name][-2] if len(paths.get(name, [])) > 1 else None
            ),
            "path": paths.get(name, [name]),
            "bounds": collection_bounds.get(name),
            "object_count": collection_counts.get(name, 0),
        }
        for name in sorted(collection_bounds)
    ]

    scene_revision = spatial_cache.make_scene_revision(scene)
    world_payload = _world_context(scene)
    render = getattr(scene, "render", None)
    cycles = getattr(scene, "cycles", None)
    units = getattr(scene, "unit_settings", None)
    result = {
        "schema_version": 1,
        **revisions,
        "scene": {
            "name": str(getattr(scene, "name", "")),
            "view_layer": str(getattr(view_layer, "name", "")),
            "frame": int(getattr(scene, "frame_current", 0) or 0),
            "unit_system": str(getattr(units, "system", "NONE")),
            "unit_scale": float(getattr(units, "scale_length", 1.0) or 1.0),
            "render_engine": str(getattr(render, "engine", "")),
            "cycles_device": str(getattr(cycles, "device", "")) or None,
            "preview_samples": int(getattr(cycles, "preview_samples", 0) or 0),
            "exposure": float(
                getattr(getattr(scene, "view_settings", None), "exposure", 0.0)
            ),
            **revisions,
            "scene_revision": scene_revision,
            "scope": scope,
            "detail": detail,
            "bounds": scene_bounds,
            "camera": camera_payload,
            "world_summary": world_payload,
            "world": world_payload,
        },
        "camera": camera_payload,
        "collections": collection_result,
        "instances": records,
        "lights": _light_context(scene),
        "candidates": {"surfaces": surfaces_result, "openings": openings_result},
        "next_cursor": next_cursor,
        "truncated": not complete,
        "warnings": warnings,
        "timings_ms": {},
        "cache": {"hit": False, "age_ms": 0.0, "build_ms": None},
        # Scan accounting is small and important for judging partial candidate
        # coverage in a complicated production scene.
        "surface_scan": {
            "triangles_scanned": surface_scanned,
            "triangle_budget": max_surface_triangles,
            "mesh_instances_scanned": mesh_instances_scanned,
            "candidates_are_ranked_and_bounded": True,
            "opening_candidates_are_heuristic": True,
        },
    }
    result["timings_ms"]["total"] = (time.perf_counter() - started) * 1000.0
    result["timings_ms"]["raycast_acceleration_build"] = acceleration_build_ms
    result["cache"]["build_ms"] = result["timings_ms"]["total"]
    _fit_context_payload(
        result,
        offset=offset,
        total=eligible_count,
        geometry_revision=revisions["geometry_revision"],
        signature=signature,
    )
    result = _plain(result)
    spatial_cache.put_cached(cache_key, result)
    return result


# ---------------------------------------------------------------------------
# Batched, bounded ray casting


def _patterns(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > MAX_IGNORE_PATTERNS:
        raise ValueError(f"{field} must be an array of at most {MAX_IGNORE_PATTERNS} patterns")
    result = []
    for index, pattern in enumerate(value):
        if not isinstance(pattern, str) or not pattern or len(pattern) > MAX_PATTERN_LENGTH:
            raise ValueError(
                f"{field}[{index}] must be a non-empty string of at most {MAX_PATTERN_LENGTH} characters"
            )
        result.append(pattern.casefold())
    return result


def _matches_patterns(name: str | None, patterns: list[str]) -> bool:
    value = (name or "").casefold()
    return any(fnmatch.fnmatchcase(value, pattern) for pattern in patterns)


def _blender_vector(value: list[float]) -> Any:
    try:
        from mathutils import Vector

        return Vector(value)
    except Exception:
        return tuple(value)


def _ray_definition(value: Any, index: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"rays[{index}] must be an object")
    _reject_unknown(value, {"id", "origin", "target", "direction", "max_distance"}, f"rays[{index}]")
    ray_id = _safe_name(value.get("id"), f"rays[{index}].id", identifier=True)
    origin = _vec3(value.get("origin"), f"rays[{index}].origin")
    has_target = value.get("target") is not None
    has_direction = value.get("direction") is not None
    if has_target == has_direction:
        raise ValueError(f"rays[{index}] must provide exactly one of target or direction")
    if has_target:
        target = _vec3(value["target"], f"rays[{index}].target")
        direction, distance = _normalize(_sub(target, origin), f"rays[{index}].target")
        endpoint_tolerance = min(distance * 0.25, max(1e-5, distance * 1e-6))
        return {
            "id": ray_id,
            "origin": origin,
            "direction": direction,
            "distance": distance,
            "cast_distance": max(0.0, distance - endpoint_tolerance),
            "endpoint_tolerance": endpoint_tolerance,
            "target": target,
            "targeted": True,
        }
    direction, _ = _normalize(_vec3(value["direction"], f"rays[{index}].direction"), f"rays[{index}].direction")
    max_distance = _number(
        value.get("max_distance", 1_000_000.0),
        f"rays[{index}].max_distance",
        minimum=1e-6,
        maximum=1_000_000_000.0,
    )
    return {
        "id": ray_id,
        "origin": origin,
        "direction": direction,
        "distance": max_distance,
        "cast_distance": max_distance,
        "endpoint_tolerance": 0.0,
        "target": None,
        "targeted": False,
    }


def _call_ray_cast(
    scene: Any,
    depsgraph: Any,
    origin: list[float],
    direction: list[float],
    distance: float,
    accel: dict[str, Any] | None = None,
) -> Any:
    """Call Blender's current ray-cast signature, with a compatibility fallback."""
    if accel is not None:
        accelerated = _accel_ray_cast(accel, origin, direction, distance)
        if accelerated is not None:
            return accelerated
    try:
        return scene.ray_cast(
            depsgraph,
            _blender_vector(origin),
            _blender_vector(direction),
            distance=distance,
        )
    except TypeError:
        return scene.ray_cast(depsgraph, _blender_vector(origin), _blender_vector(direction), distance)


def _ray_advance_epsilon(
    unit_scale: float,
    *coordinates: Iterable[float],
) -> float:
    """Return an advance large enough to change Blender float coordinates.

    A fixed millimetre-scale offset is ineffective when a scene lives hundreds
    of kilometres from the origin: adding it can round back to exactly the same
    float and repeatedly hit the ignored face.  Scale the offset by the largest
    coordinate encountered while retaining a small unit-aware floor near zero.
    """
    coordinate_magnitude = max(
        (abs(float(component)) for value in coordinates for component in value),
        default=0.0,
    )
    return max(
        1e-7,
        1e-4 * max(abs(float(unit_scale)), 1e-3),
        coordinate_magnitude * 2e-7,
    )


def handle_batch_raycast(params: dict[str, Any]) -> dict[str, Any]:
    """Cast multiple multi-hit rays without exposing Blender RNA values."""
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    _reject_unknown(
        params,
        {
            "rays",
            "max_hits",
            "ignore_object_patterns",
            "ignore_material_patterns",
            "include_ignored_hits",
            "expected_geometry_revision",
            "time_budget_ms",
        },
    )
    revisions = spatial_cache.get_revisions()
    expected = params.get("expected_geometry_revision")
    if expected is not None:
        expected = _integer(
            expected, "expected_geometry_revision", minimum=0, maximum=2**63 - 1
        )
        if expected != revisions["geometry_revision"]:
            raise ValueError(
                f"Stale geometry revision: expected {expected}, current {revisions['geometry_revision']}"
            )
    rays_value = params.get("rays")
    if not isinstance(rays_value, list) or not rays_value:
        raise ValueError("rays must be a non-empty array")
    if len(rays_value) > MAX_RAYS:
        raise ValueError(f"rays must contain at most {MAX_RAYS} entries")
    rays = [_ray_definition(value, index) for index, value in enumerate(rays_value)]
    identifiers = [str(ray["id"]) for ray in rays]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("ray ids must be unique")
    max_hits = _integer(params.get("max_hits", 8), "max_hits", minimum=1, maximum=MAX_RAY_HITS)
    include_ignored = params.get("include_ignored_hits", True)
    _bool(include_ignored, "include_ignored_hits")
    object_patterns = _patterns(params.get("ignore_object_patterns"), "ignore_object_patterns")
    material_patterns = _patterns(params.get("ignore_material_patterns"), "ignore_material_patterns")
    time_budget_ms = _integer(
        params.get("time_budget_ms", 2000), "time_budget_ms", minimum=10, maximum=5000
    )
    scene = bpy.context.scene
    unit_scale = float(
        getattr(getattr(scene, "unit_settings", None), "scale_length", 1.0) or 1.0
    )
    depsgraph = bpy.context.evaluated_depsgraph_get()
    instance_records = _raycast_instance_records(
        depsgraph, revisions["geometry_revision"]
    )
    acceleration = (
        _raycast_accel
        if _raycast_accel_key
        == _raycast_index_revision_key(revisions["geometry_revision"])
        else None
    )
    if acceleration is None and len(instance_records) > 5_000:
        acceleration = _ensure_raycast_accel(
            depsgraph, revisions["geometry_revision"]
        )
    started = time.perf_counter()
    deadline = started + time_budget_ms / 1000.0
    output: list[dict[str, Any]] = []
    all_complete = True

    for ray in rays:
        hits: list[dict[str, Any]] = []
        origin = list(ray["origin"])
        direction = ray["direction"]
        remaining = float(ray["cast_distance"])
        travelled = 0.0
        accepted_hits = 0
        encounters = 0
        ignored_encounters = 0
        termination = "DISTANCE_EXHAUSTED"
        complete = True

        # The coordinate-scaled epsilon is an advancement distance, not a
        # reason to skip the initial cast when a short ray starts far away.
        while remaining > 1e-12:
            if time.perf_counter() >= deadline:
                termination = "TIME_BUDGET"
                complete = False
                all_complete = False
                break
            if len(hits) >= max_hits:
                termination = "MAX_HITS"
                complete = False
                all_complete = False
                break
            if encounters >= 512:
                termination = "ENCOUNTER_LIMIT"
                complete = False
                all_complete = False
                break

            cast = _call_ray_cast(
                scene,
                depsgraph,
                origin,
                direction,
                remaining,
                acceleration,
            )
            try:
                hit, location, normal, face_index, hit_object, hit_matrix = cast
            except Exception as exc:
                raise RuntimeError(f"Blender returned an invalid ray_cast result: {exc}") from exc
            if not hit:
                travelled += remaining
                remaining = 0.0
                termination = "TARGET_REACHED" if ray["targeted"] else "MISS"
                break

            location_value = _sequence(location, 3)
            normal_value = _sequence(normal, 3)
            if not location_value or not normal_value:
                raise RuntimeError("Blender ray_cast returned non-finite hit coordinates")
            step = math.sqrt(sum(component * component for component in _sub(location_value, origin)))
            if not math.isfinite(step):
                raise RuntimeError("Blender ray_cast returned a non-finite distance")
            cumulative = travelled + step
            object_name = str(getattr(hit_object, "name", "")) or None
            face = int(face_index)
            material_name = _material_name(hit_object, face) if hit_object is not None else None
            ignored_by: list[str] = []
            if _matches_patterns(object_name, object_patterns):
                ignored_by.append("OBJECT_PATTERN")
            if _matches_patterns(material_name, material_patterns):
                ignored_by.append("MATERIAL_PATTERN")
            ignored = bool(ignored_by)
            record = {
                "distance": cumulative,
                "location": location_value,
                "normal": normal_value,
                "face_index": face,
                "object_id": _raycast_object_id(
                    hit_object,
                    hit_matrix,
                    instance_records,
                    revisions["geometry_revision"],
                ),
                "object_name": object_name,
                "material_name": material_name,
                "ignored": ignored,
                "ignored_by": ignored_by,
                "ignore_reason": ignored_by[0] if ignored_by else None,
            }
            if not ignored or include_ignored:
                hits.append(record)
            if ignored:
                ignored_encounters += 1
            else:
                accepted_hits += 1
            encounters += 1

            if not ignored:
                termination = "BLOCKED"
                break

            # Adaptive advancement avoids repeatedly hitting the same face at
            # large scene scales while preserving closely layered geometry.
            advance = _ray_advance_epsilon(
                unit_scale,
                origin,
                location_value,
            )
            if step + advance >= remaining:
                travelled += min(remaining, step + advance)
                remaining = 0.0
                origin = _add_scaled(location_value, direction, advance)
                termination = "TARGET_REACHED" if ray["targeted"] else "DISTANCE_EXHAUSTED"
                break
            travelled += step + advance
            remaining -= step + advance
            origin = _add_scaled(location_value, direction, advance)

        output.append(
            {
                "id": ray["id"],
                "complete": complete,
                "clear_to_target": (
                    (
                        termination == "TARGET_REACHED"
                        if complete
                        else None
                    )
                    if ray["targeted"]
                    else None
                ),
                "termination": termination,
                "hits": hits,
                "accepted_hits": accepted_hits,
                "ignored_encounters": ignored_encounters,
                "requested_distance": ray["distance"],
                "distance_requested": ray["distance"],
                "endpoint_tolerance": ray["endpoint_tolerance"],
                "distance_travelled": min(travelled, ray["distance"]),
            }
        )
        if not complete and termination == "TIME_BUDGET":
            # Preserve one result per requested ray so callers can retry only
            # unfinished IDs without guessing which inputs were skipped.
            remaining_index = len(output)
            for skipped in rays[remaining_index:]:
                output.append(
                    {
                        "id": skipped["id"],
                        "complete": False,
                        "clear_to_target": None,
                        "termination": "TIME_BUDGET",
                        "hits": [],
                        "accepted_hits": 0,
                        "ignored_encounters": 0,
                        "requested_distance": skipped["distance"],
                        "distance_requested": skipped["distance"],
                        "endpoint_tolerance": skipped["endpoint_tolerance"],
                        "distance_travelled": 0.0,
                    }
                )
            break

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return _plain(
        {
            **revisions,
            "scene_revision": spatial_cache.make_scene_revision(scene),
            "rays": output,
            "complete": all_complete,
            "elapsed_ms": elapsed_ms,
            "warnings": [] if all_complete else ["Ray query returned partial results; retry unfinished IDs"],
        }
    )


# ---------------------------------------------------------------------------
# Transactional managed light plans


_LIGHT_FIELDS = {
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

# Writable Blender 5.2 Light properties that are not part of the public
# LightSpec but must round-trip through an atomic rollback.  Each is guarded by
# ``hasattr`` so the extension remains usable on supported earlier Blender
# versions where an EEVEE/Cycles property may not exist.
_ROLLBACK_LIGHT_DATA_FIELDS = (
    "use_fake_user",
    "use_extra_user",
    "use_temperature",
    "temperature",
    "transmission_factor",
    "use_custom_distance",
    "cutoff_distance",
    "exposure",
    "normalize",
    "use_soft_falloff",
    "shadow_buffer_clip_start",
    "shadow_filter_radius",
    "shadow_maximum_resolution",
    "use_shadow_jitter",
    "shadow_jitter_overblur",
    "use_absolute_resolution",
    "shadow_cascade_max_distance",
    "shadow_cascade_count",
    "shadow_cascade_exponent",
    "shadow_cascade_fade",
    "use_square",
    "show_cone",
    "spread",
)
_ROLLBACK_CYCLES_LIGHT_FIELDS = (
    "max_bounces",
    "use_multiple_importance_sampling",
    "is_portal",
    "is_caustics_light",
)


def _warn_or_raise(strict: bool, warnings: list[str], message: str) -> None:
    if strict:
        raise ValueError(message)
    warnings.append(message)


def _resolve_target(value: str) -> Any:
    target = _lookup(getattr(bpy.data, "objects", None), value)
    if target is None:
        raise ValueError(f"target_object '{value}' was not found")
    return target


def _validate_light_spec(value: Any, index: int, strict: bool, warnings: list[str]) -> dict[str, Any]:
    field = f"lights[{index}]"
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    unknown = sorted(set(value) - _LIGHT_FIELDS)
    if unknown:
        _warn_or_raise(strict, warnings, f"{field} contains unsupported fields: {unknown}")
    light_id = _safe_name(value.get("id"), f"{field}.id", identifier=True)
    name = _safe_name(value.get("name", light_id), f"{field}.name")
    light_type = str(value.get("type", "")).upper()
    if light_type not in {"POINT", "SUN", "SPOT", "AREA"}:
        raise ValueError(f"{field}.type must be POINT, SUN, SPOT, or AREA")
    location = _vec3(value.get("location_world"), f"{field}.location_world")
    target_point = value.get("target_point")
    target_object = value.get("target_object")
    if target_point is not None and target_object is not None:
        raise ValueError(f"{field} must not specify both target_point and target_object")
    if target_point is not None:
        target_point = _vec3(target_point, f"{field}.target_point")
    if target_object is not None:
        target_object = _safe_name(target_object, f"{field}.target_object")
        target_point_resolved = _world_location(_resolve_target(target_object))
    else:
        target_point_resolved = target_point
    if target_point_resolved is not None:
        _normalize(_sub(target_point_resolved, location), f"{field} target direction")
    if light_type == "POINT" and target_point_resolved is not None:
        _warn_or_raise(strict, warnings, f"{field}: POINT lights do not support aiming; target was ignored")
        target_point = None
        target_object = None

    normalized: dict[str, Any] = {
        "id": light_id,
        "name": name,
        "type": light_type,
        "location_world": location,
        "energy": _number(
            value.get("energy", 1000.0),
            f"{field}.energy",
            minimum=0.0,
            maximum=1_000_000_000.0,
        ),
        "color_rgb": _color(value.get("color_rgb", [1.0, 1.0, 1.0]), f"{field}.color_rgb"),
        "use_shadow": _bool(value.get("use_shadow", True), f"{field}.use_shadow"),
        "diffuse_factor": _number(
            value.get("diffuse_factor", 1.0), f"{field}.diffuse_factor", minimum=0.0, maximum=1.0
        ),
        "specular_factor": _number(
            value.get("specular_factor", 1.0), f"{field}.specular_factor", minimum=0.0, maximum=1.0
        ),
        "volume_factor": _number(
            value.get("volume_factor", 1.0), f"{field}.volume_factor", minimum=0.0, maximum=1.0
        ),
        "target_point": target_point,
        "target_object": target_object,
    }
    if value.get("radius") is not None:
        if light_type not in {"POINT", "SPOT"}:
            _warn_or_raise(strict, warnings, f"{field}.radius only applies to POINT or SPOT lights")
        else:
            normalized["radius"] = _number(
                value["radius"], f"{field}.radius", minimum=0.0, maximum=1_000_000.0
            )
    if value.get("sun_angle_degrees") is not None:
        if light_type != "SUN":
            _warn_or_raise(strict, warnings, f"{field}.sun_angle_degrees only applies to SUN lights")
        else:
            normalized["sun_angle_degrees"] = _number(
                value["sun_angle_degrees"],
                f"{field}.sun_angle_degrees",
                minimum=0.0,
                maximum=180.0,
            )
    if light_type == "AREA":
        shape = str(value.get("area_shape", "SQUARE")).upper()
        if shape not in {"SQUARE", "RECTANGLE", "DISK", "ELLIPSE"}:
            raise ValueError(f"{field}.area_shape must be SQUARE, RECTANGLE, DISK, or ELLIPSE")
        normalized["area_shape"] = shape
        normalized["size"] = _number(value.get("size", 1.0), f"{field}.size", minimum=1e-6, maximum=1_000_000.0)
        if shape in {"RECTANGLE", "ELLIPSE"}:
            normalized["size_y"] = _number(
                value.get("size_y", normalized["size"]),
                f"{field}.size_y",
                minimum=1e-6,
                maximum=1_000_000.0,
            )
    else:
        for extra in ("area_shape", "size", "size_y"):
            if value.get(extra) is not None:
                _warn_or_raise(strict, warnings, f"{field}.{extra} only applies to AREA lights")
    if light_type == "SPOT":
        normalized["spot_angle_degrees"] = _number(
            value.get("spot_angle_degrees", 45.0),
            f"{field}.spot_angle_degrees",
            minimum=1.0,
            maximum=180.0,
        )
        normalized["spot_blend"] = _number(
            value.get("spot_blend", 0.15), f"{field}.spot_blend", minimum=0.0, maximum=1.0
        )
    else:
        for extra in ("spot_angle_degrees", "spot_blend"):
            if value.get(extra) is not None:
                _warn_or_raise(strict, warnings, f"{field}.{extra} only applies to SPOT lights")
    return normalized


def _validate_scene_overrides(value: Any) -> dict[str, Any]:
    if value is None:
        return {
            "existing_light_policy": "KEEP",
            "world": {"mode": "KEEP", "color_rgb": [0.0, 0.0, 0.0], "strength": 0.0},
            "exposure": None,
        }
    if not isinstance(value, dict):
        raise ValueError("scene_overrides must be an object or null")
    unknown = sorted(set(value) - {"existing_light_policy", "world", "exposure"})
    if unknown:
        raise ValueError(f"scene_overrides contains unsupported fields: {unknown}")
    policy = str(value.get("existing_light_policy", "KEEP")).upper()
    if policy not in {"KEEP", "MUTE_NON_MANAGED"}:
        raise ValueError("scene_overrides.existing_light_policy must be KEEP or MUTE_NON_MANAGED")
    world_value = value.get("world")
    if world_value is None:
        world = {"mode": "KEEP", "color_rgb": [0.0, 0.0, 0.0], "strength": 0.0}
    else:
        if not isinstance(world_value, dict):
            raise ValueError("scene_overrides.world must be an object or null")
        _reject_unknown(
            world_value,
            {"mode", "color_rgb", "strength"},
            "scene_overrides.world",
        )
        mode = str(world_value.get("mode", "KEEP")).upper()
        if mode not in {"KEEP", "MANAGED_SOLID"}:
            raise ValueError("scene_overrides.world.mode must be KEEP or MANAGED_SOLID")
        world = {
            "mode": mode,
            "color_rgb": _color(world_value.get("color_rgb", [0.0, 0.0, 0.0]), "scene_overrides.world.color_rgb"),
            "strength": _number(
                world_value.get("strength", 0.0),
                "scene_overrides.world.strength",
                minimum=0.0,
                maximum=1000.0,
            ),
        }
    exposure = value.get("exposure")
    if exposure is not None:
        exposure = _number(exposure, "scene_overrides.exposure", minimum=-32.0, maximum=32.0)
    return {"existing_light_policy": policy, "world": world, "exposure": exposure}


def _has_drivers(value: Any) -> bool:
    animation_data = getattr(value, "animation_data", None)
    return bool(_iter_values(getattr(animation_data, "drivers", None)))


def _has_animation_state(value: Any) -> bool:
    animation_data = getattr(value, "animation_data", None)
    if animation_data is None:
        return False
    return bool(
        getattr(animation_data, "action", None) is not None
        or _iter_values(getattr(animation_data, "drivers", None))
        or _iter_values(getattr(animation_data, "nla_tracks", None))
    )


def _has_non_default_delta_transforms(obj: Any) -> bool:
    defaults = (
        ("delta_location", (0.0, 0.0, 0.0)),
        ("delta_rotation_euler", (0.0, 0.0, 0.0)),
        ("delta_rotation_quaternion", (1.0, 0.0, 0.0, 0.0)),
        ("delta_scale", (1.0, 1.0, 1.0)),
    )
    for attribute, expected in defaults:
        actual = _sequence(getattr(obj, attribute, expected), len(expected))
        if len(actual) != len(expected) or any(
            not math.isclose(value, default, rel_tol=0.0, abs_tol=1e-9)
            for value, default in zip(actual, expected)
        ):
            return True
    return False


def _custom_property_names(value: Any) -> set[str]:
    keys = getattr(value, "keys", None)
    if not callable(keys):
        return set()
    try:
        return {str(key) for key in keys()}
    except Exception:
        return set()


def _has_unsupported_light_nodes(data: Any) -> bool:
    use_nodes = bool(getattr(data, "use_nodes", False))
    tree = getattr(data, "node_tree", None)
    nodes = _iter_values(getattr(tree, "nodes", None))
    if not use_nodes:
        return bool(nodes)
    # Blender 5.2 creates lights with the canonical Emission -> Light Output
    # node pair enabled.  That stock graph is part of supported Light state;
    # arbitrary additions or rewiring remain outside the transaction contract.
    if sorted(str(getattr(node, "type", "")) for node in nodes) != [
        "EMISSION",
        "OUTPUT_LIGHT",
    ]:
        return True
    links = _iter_values(getattr(tree, "links", None))
    return len(links) != 1


def _constraint_references_targets(constraint: Any, target_ids: set[int]) -> bool:
    for attribute in (
        "target",
        "pole_target",
        "space_object",
        "camera",
        "depth_object",
    ):
        value = getattr(constraint, attribute, None)
        if value is not None and _rna_identity(value) in target_ids:
            return True
    return False


def _drivers_reference_targets(owner: Any, target_ids: set[int]) -> bool:
    animation_data = getattr(owner, "animation_data", None)
    for fcurve in _iter_values(getattr(animation_data, "drivers", None)):
        driver = getattr(fcurve, "driver", None)
        for variable in _iter_values(getattr(driver, "variables", None)):
            for target in _iter_values(getattr(variable, "targets", None)):
                target_id = getattr(target, "id", None)
                if target_id is not None and _rna_identity(target_id) in target_ids:
                    return True
    return False


def _preflight_managed_state(
    scene: Any,
    collection_name: str,
    scene_overrides: dict[str, Any],
    strict: bool,
) -> None:
    """Reject collisions and unsupported state before any plan mutation."""
    collection = _lookup(getattr(bpy.data, "collections", None), collection_name)
    if collection is not None and not bool(_custom_get(collection, MANAGED_PROP, False)):
        raise ValueError(
            f"Collection '{collection_name}' exists but is not marked as blend-ai managed"
        )
    collection = _scene_collection(scene, collection_name)
    managed_targets: set[int] = set()
    if collection is not None:
        if getattr(collection, "library", None) is not None:
            raise ValueError(f"Managed collection '{collection_name}' is linked/read-only")
        for obj in _iter_values(getattr(collection, "objects", None)):
            if not _is_managed_light(obj, collection):
                raise ValueError(
                    f"Managed collection '{collection_name}' contains unmarked object "
                    f"'{getattr(obj, 'name', '')}'"
                )
            data = getattr(obj, "data", None)
            managed_targets.update({_rna_identity(obj), _rna_identity(data)})
            if getattr(obj, "library", None) is not None or getattr(data, "library", None) is not None:
                raise ValueError(f"Managed light '{getattr(obj, 'name', '')}' is linked/read-only")
            if strict and (
                _iter_values(getattr(obj, "constraints", None))
                or _has_animation_state(obj)
                or _has_animation_state(data)
            ):
                raise ValueError(
                    f"Managed light '{getattr(obj, 'name', '')}' has unsupported constraints or animation"
                )
            if strict and len(_iter_values(getattr(obj, "users_collection", None))) > 1:
                raise ValueError(
                    f"Managed light '{getattr(obj, 'name', '')}' belongs to multiple collections"
                )
            if strict and int(getattr(data, "users", 0) or 0) > 1:
                raise ValueError(
                    f"Managed light '{getattr(obj, 'name', '')}' shares its Light datablock "
                    "with another object"
                )
            if strict and getattr(obj, "parent", None) is not None:
                raise ValueError(
                    f"Managed light '{getattr(obj, 'name', '')}' has unsupported parenting"
                )
            if strict and _iter_values(getattr(obj, "children", None)):
                raise ValueError(
                    f"Managed light '{getattr(obj, 'name', '')}' has child objects that "
                    "would not survive removal or type replacement"
                )
            light_linking = getattr(obj, "light_linking", None)
            if strict and light_linking is not None and (
                getattr(light_linking, "receiver_collection", None) is not None
                or getattr(light_linking, "blocker_collection", None) is not None
            ):
                raise ValueError(
                    f"Managed light '{getattr(obj, 'name', '')}' has unsupported light linking"
                )
            scale = _sequence(getattr(obj, "scale", (1.0, 1.0, 1.0)), 3)
            if strict and (
                len(scale) != 3
                or any(
                    not math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-9)
                    for value in scale
                )
            ):
                raise ValueError(
                    f"Managed light '{getattr(obj, 'name', '')}' has unsupported non-default scale"
                )
            if strict and _has_non_default_delta_transforms(obj):
                raise ValueError(
                    f"Managed light '{getattr(obj, 'name', '')}' has unsupported delta transforms"
                )
            if strict and str(getattr(obj, "rotation_mode", "XYZ")) == "AXIS_ANGLE":
                raise ValueError(
                    f"Managed light '{getattr(obj, 'name', '')}' has unsupported axis-angle rotation"
                )
            allowed_properties = {
                MANAGED_PROP,
                MANAGED_ID_PROP,
                MANAGED_PLAN_PROP,
            }
            object_properties = _custom_property_names(obj) - allowed_properties
            data_properties = _custom_property_names(data) - allowed_properties
            if strict and (object_properties or data_properties):
                unsupported = sorted(object_properties | data_properties)
                raise ValueError(
                    f"Managed light '{getattr(obj, 'name', '')}' has unsupported custom properties: "
                    f"{unsupported}"
                )
            if strict and _has_unsupported_light_nodes(data):
                raise ValueError(
                    f"Managed light '{getattr(obj, 'name', '')}' has unsupported node state"
                )

    if strict and managed_targets:
        all_objects = _iter_values(
            getattr(getattr(bpy, "data", None), "objects", None)
        ) or _iter_values(getattr(scene, "objects", None))
        for owner in all_objects:
            if _rna_identity(owner) in managed_targets:
                continue
            for constraint in _iter_values(getattr(owner, "constraints", None)):
                if _constraint_references_targets(constraint, managed_targets):
                    raise ValueError(
                        f"Object '{getattr(owner, 'name', '')}' has a constraint that "
                        "references a managed light"
                    )
            for id_owner in (owner, getattr(owner, "data", None)):
                if id_owner is not None and _drivers_reference_targets(
                    id_owner, managed_targets
                ):
                    raise ValueError(
                        f"Object '{getattr(owner, 'name', '')}' has a driver that "
                        "references a managed light"
                    )
        for collection_name in ("scenes", "worlds", "materials", "cameras"):
            for owner in _iter_values(
                getattr(getattr(bpy, "data", None), collection_name, None)
            ):
                if _rna_identity(owner) in managed_targets:
                    continue
                if _drivers_reference_targets(owner, managed_targets):
                    raise ValueError(
                        f"Datablock '{getattr(owner, 'name', '')}' has a driver that "
                        "references a managed light"
                    )

    if strict and scene_overrides["existing_light_policy"] == "MUTE_NON_MANAGED":
        for obj in _iter_values(getattr(scene, "objects", None)):
            if str(getattr(obj, "type", "")) != "LIGHT" or _is_managed_light(obj):
                continue
            data = getattr(obj, "data", None)
            other_scene_owners = [
                owner
                for owner in _object_scene_owners(obj)
                if _rna_identity(owner) != _rna_identity(scene)
            ]
            if (
                getattr(obj, "library", None) is not None
                or _has_animation_state(obj)
                or _has_animation_state(data)
                or other_scene_owners
            ):
                raise ValueError(
                    f"Non-managed light '{getattr(obj, 'name', '')}' cannot be safely muted"
                )

    ledger_text = _lookup(getattr(getattr(bpy, "data", None), "texts", None), LEDGER_TEXT)
    if ledger_text is not None and not bool(_custom_get(ledger_text, LEDGER_PROP, False)):
        raise ValueError(
            f"Text datablock '{LEDGER_TEXT}' exists but is not marked as the "
            "blend-ai relighting ledger"
        )


def _validate_plan(params: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    strict = params.get("strict", True)
    _bool(strict, "strict")
    warnings: list[str] = []
    mode = str(params.get("mode", "PATCH_MANAGED")).upper()
    if mode not in {"REPLACE_MANAGED", "PATCH_MANAGED"}:
        raise ValueError("mode must be REPLACE_MANAGED or PATCH_MANAGED")
    collection_name = _safe_name(params.get("collection_name", MANAGED_COLLECTION), "collection_name")
    plan_id = params.get("plan_id")
    if plan_id is not None:
        plan_id = _safe_name(plan_id, "plan_id", identifier=True)
    raw_lights = params.get("lights", [])
    if not isinstance(raw_lights, list) or len(raw_lights) > MAX_MANAGED_LIGHTS:
        raise ValueError(
            f"lights must be an array of at most {MAX_MANAGED_LIGHTS} LightSpec objects"
        )
    lights = [_validate_light_spec(value, index, strict, warnings) for index, value in enumerate(raw_lights)]
    ids = [value["id"] for value in lights]
    if len(set(ids)) != len(ids):
        raise ValueError("lights contains duplicate ids")
    display_names = [value["name"] for value in lights]
    if len(set(display_names)) != len(display_names):
        raise ValueError("lights contains duplicate display names")
    raw_remove = params.get("remove_ids", [])
    if not isinstance(raw_remove, list) or len(raw_remove) > MAX_MANAGED_LIGHTS:
        raise ValueError(f"remove_ids must be an array of at most {MAX_MANAGED_LIGHTS} ids")
    remove_ids = [_safe_name(value, f"remove_ids[{index}]", identifier=True) for index, value in enumerate(raw_remove)]
    if len(set(remove_ids)) != len(remove_ids):
        raise ValueError("remove_ids contains duplicates")
    overlap = set(ids) & set(remove_ids)
    if overlap:
        raise ValueError(f"ids cannot appear in both lights and remove_ids: {sorted(overlap)}")
    expected = params.get("expected_geometry_revision")
    if expected is not None:
        expected = _integer(
            expected, "expected_geometry_revision", minimum=0, maximum=2**63 - 1
        )
    normalized = {
            "plan_id": plan_id,
            "mode": mode,
            "expected_geometry_revision": expected,
            "collection_name": collection_name,
            "lights": lights,
            "remove_ids": remove_ids,
            "scene_overrides": _validate_scene_overrides(params.get("scene_overrides")),
            "strict": strict,
        }
    _preflight_managed_state(
        bpy.context.scene,
        collection_name,
        normalized["scene_overrides"],
        strict,
    )
    return normalized, warnings


def _is_managed_light(obj: Any, collection: Any | None = None) -> bool:
    if str(getattr(obj, "type", "")) != "LIGHT":
        return False
    data = getattr(obj, "data", None)
    light_id = _custom_get(obj, MANAGED_ID_PROP)
    if not bool(_custom_get(obj, MANAGED_PROP, False)) or not isinstance(light_id, str):
        return False
    if not bool(_custom_get(data, MANAGED_PROP, False)):
        return False
    if _custom_get(data, MANAGED_ID_PROP) != light_id:
        return False
    if str(_custom_get(data, MANAGED_PLAN_PROP, "")) != str(
        _custom_get(obj, MANAGED_PLAN_PROP, "")
    ):
        return False
    if collection is None:
        return True
    try:
        if obj in collection.objects:
            return True
    except Exception:
        pass
    return any(item is collection for item in _iter_values(getattr(obj, "users_collection", None)))


def _managed_map(collection: Any | None) -> dict[str, Any]:
    if collection is None:
        return {}
    result: dict[str, Any] = {}
    for obj in _iter_values(getattr(collection, "objects", None)):
        if not _is_managed_light(obj, collection):
            continue
        light_id = str(_custom_get(obj, MANAGED_ID_PROP))
        if light_id in result:
            raise RuntimeError(f"Managed collection contains duplicate light id '{light_id}'")
        result[light_id] = obj
    return result


def _rna_identity(value: Any) -> int:
    pointer = getattr(value, "as_pointer", None)
    if callable(pointer):
        try:
            result = int(pointer())
            if result:
                return result
        except Exception:
            pass
    return id(value)


def _collection_in_scene(scene: Any, target: Any) -> bool:
    root = getattr(scene, "collection", None)
    if root is None or target is None:
        return False
    pending = [root]
    seen: set[int] = set()
    target_identity = _rna_identity(target)
    while pending:
        collection = pending.pop()
        identity = _rna_identity(collection)
        if identity == target_identity:
            return True
        if identity in seen:
            continue
        seen.add(identity)
        pending.extend(_iter_values(getattr(collection, "children", None)))
    return False


def _collection_scene_owners(collection: Any) -> list[Any]:
    return [
        scene
        for scene in _iter_values(getattr(getattr(bpy, "data", None), "scenes", None))
        if _collection_in_scene(scene, collection)
    ]


def _object_scene_owners(obj: Any) -> list[Any]:
    target_identity = _rna_identity(obj)
    return [
        scene
        for scene in _iter_values(getattr(getattr(bpy, "data", None), "scenes", None))
        if any(
            _rna_identity(candidate) == target_identity
            for candidate in _iter_values(getattr(scene, "objects", None))
        )
    ]


def _scene_collection(scene: Any, name: str) -> Any | None:
    """Resolve a collection only when it belongs exclusively to this scene."""
    collection = _lookup(getattr(bpy.data, "collections", None), name)
    if collection is None:
        return None
    if not _collection_in_scene(scene, collection):
        raise ValueError(
            f"Collection '{name}' exists but is linked only to another scene"
        )
    other_owners = [
        owner
        for owner in _collection_scene_owners(collection)
        if owner is not scene
        and str(getattr(owner, "name", ""))
        != str(getattr(scene, "name", ""))
    ]
    if other_owners:
        owner_names = sorted(str(getattr(owner, "name", "")) for owner in other_owners)
        raise ValueError(
            f"Collection '{name}' is shared with other scenes: {owner_names}"
        )
    return collection


def _ensure_collection(scene: Any, name: str) -> Any:
    collection = _scene_collection(scene, name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        try:
            _custom_set(collection, MANAGED_PROP, True)
            scene.collection.children.link(collection)
        except Exception:
            try:
                bpy.data.collections.remove(collection)
            except Exception:
                pass
            raise
    elif not bool(_custom_get(collection, MANAGED_PROP, False)):
        raise ValueError(
            f"Collection '{name}' exists but is not marked as blend-ai managed"
        )
    return collection


def _remove_light_object(obj: Any) -> None:
    data = getattr(obj, "data", None)
    bpy.data.objects.remove(obj, do_unlink=True)
    if data is not None and int(getattr(data, "users", 0) or 0) == 0:
        bpy.data.lights.remove(data)


def _set_managed_markers(obj: Any, data: Any, light_id: str, plan_id: str | None) -> None:
    for value in (obj, data):
        _custom_set(value, MANAGED_PROP, True)
        _custom_set(value, MANAGED_ID_PROP, light_id)
        _custom_set(value, MANAGED_PLAN_PROP, plan_id or "")


def _new_light_object(collection: Any, spec: dict[str, Any], plan_id: str | None) -> Any:
    data = bpy.data.lights.new(name=spec["name"], type=spec["type"])
    obj = None
    try:
        obj = bpy.data.objects.new(name=spec["name"], object_data=data)
        collection.objects.link(obj)
        _set_managed_markers(obj, data, spec["id"], plan_id)
        return obj
    except Exception:
        if obj is not None:
            try:
                bpy.data.objects.remove(obj, do_unlink=True)
            except Exception:
                pass
        if int(getattr(data, "users", 0) or 0) == 0:
            try:
                bpy.data.lights.remove(data)
            except Exception:
                pass
        raise


def _replace_light_data(obj: Any, spec: dict[str, Any], plan_id: str | None) -> Any:
    old = getattr(obj, "data", None)
    data = bpy.data.lights.new(name=spec["name"], type=spec["type"])
    try:
        # Mark the detached replacement first.  If marker assignment or the
        # object swap fails, the existing managed object/data pair remains
        # discoverable by automatic rollback.
        _custom_set(data, MANAGED_PROP, True)
        _custom_set(data, MANAGED_ID_PROP, spec["id"])
        _custom_set(data, MANAGED_PLAN_PROP, plan_id or "")
        obj.data = data
    except Exception:
        if getattr(obj, "data", None) is data and old is not None:
            try:
                obj.data = old
            except Exception:
                pass
        if int(getattr(data, "users", 0) or 0) == 0:
            try:
                bpy.data.lights.remove(data)
            except Exception:
                pass
        raise
    if old is not None and int(getattr(old, "users", 0) or 0) == 0:
        bpy.data.lights.remove(old)
    return data


def _aim_object(obj: Any, target: list[float]) -> None:
    direction = _sub(target, _world_location(obj))
    _normalize(direction, "light target direction")
    try:
        from mathutils import Vector

        quaternion = Vector(direction).to_track_quat("-Z", "Y")
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = quaternion
    except Exception as exc:
        raise RuntimeError(f"Unable to aim managed light '{getattr(obj, 'name', '')}': {exc}") from exc


def _apply_light_spec(obj: Any, spec: dict[str, Any], plan_id: str | None) -> None:
    data = getattr(obj, "data", None)
    if str(getattr(data, "type", "")) != spec["type"]:
        data = _replace_light_data(obj, spec, plan_id)
    _set_managed_markers(obj, data, spec["id"], plan_id)
    obj.name = spec["name"]
    data.name = spec["name"]
    try:
        obj.parent = None
    except Exception:
        pass
    obj.location = tuple(spec["location_world"])
    data.energy = spec["energy"]
    data.color = tuple(spec["color_rgb"])
    data.use_shadow = spec["use_shadow"]
    data.diffuse_factor = spec["diffuse_factor"]
    data.specular_factor = spec["specular_factor"]
    data.volume_factor = spec["volume_factor"]
    if "radius" in spec:
        data.shadow_soft_size = spec["radius"]
    if spec["type"] == "SUN" and "sun_angle_degrees" in spec:
        data.angle = math.radians(spec["sun_angle_degrees"])
    if spec["type"] == "AREA":
        data.shape = spec["area_shape"]
        data.size = spec["size"]
        if "size_y" in spec:
            data.size_y = spec["size_y"]
    if spec["type"] == "SPOT":
        data.spot_size = math.radians(spec["spot_angle_degrees"])
        data.spot_blend = spec["spot_blend"]
    target = spec.get("target_point")
    if spec.get("target_object") is not None:
        target = _world_location(_resolve_target(spec["target_object"]))
    if target is not None and spec["type"] != "POINT":
        _aim_object(obj, target)


def _snapshot_extra_light_data(data: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for attribute in _ROLLBACK_LIGHT_DATA_FIELDS:
        if not hasattr(data, attribute):
            continue
        value = getattr(data, attribute)
        if isinstance(value, bool):
            result[attribute] = value
        elif isinstance(value, int) and not isinstance(value, bool):
            result[attribute] = int(value)
        elif _is_number(value):
            result[attribute] = float(value)
        elif isinstance(value, str):
            result[attribute] = value
    return result


def _restore_extra_light_data(data: Any, state: Any) -> None:
    if not isinstance(state, dict):
        return
    for attribute, value in state.items():
        if attribute in _ROLLBACK_LIGHT_DATA_FIELDS and hasattr(data, attribute):
            setattr(data, attribute, value)


def _snapshot_cycles_light_data(data: Any) -> dict[str, Any]:
    cycles = getattr(data, "cycles", None)
    result: dict[str, Any] = {}
    for attribute in _ROLLBACK_CYCLES_LIGHT_FIELDS:
        if cycles is None or not hasattr(cycles, attribute):
            continue
        value = getattr(cycles, attribute)
        if isinstance(value, bool):
            result[attribute] = value
        elif isinstance(value, int) and not isinstance(value, bool):
            result[attribute] = int(value)
        elif _is_number(value):
            result[attribute] = float(value)
    return result


def _restore_cycles_light_data(data: Any, state: Any) -> None:
    cycles = getattr(data, "cycles", None)
    if cycles is None or not isinstance(state, dict):
        return
    for attribute, value in state.items():
        if attribute in _ROLLBACK_CYCLES_LIGHT_FIELDS and hasattr(cycles, attribute):
            setattr(cycles, attribute, value)


def _serialize_managed_light(obj: Any) -> dict[str, Any]:
    data = obj.data
    result = {
        "id": str(_custom_get(obj, MANAGED_ID_PROP)),
        "plan_id": str(_custom_get(obj, MANAGED_PLAN_PROP, "")) or None,
        "name": str(getattr(obj, "name", "")),
        "data_name": str(getattr(data, "name", "")),
        "type": str(getattr(data, "type", "")),
        "location_world": _world_location(obj),
        "rotation_mode": str(getattr(obj, "rotation_mode", "XYZ")),
        "rotation_quaternion": _sequence(getattr(obj, "rotation_quaternion", (1.0, 0.0, 0.0, 0.0)), 4),
        "rotation_euler": _sequence(getattr(obj, "rotation_euler", (0.0, 0.0, 0.0)), 3),
        "energy": float(getattr(data, "energy", 0.0)),
        "color_rgb": _sequence(getattr(data, "color", (1.0, 1.0, 1.0)), 3),
        "use_shadow": bool(getattr(data, "use_shadow", True)),
        "diffuse_factor": float(getattr(data, "diffuse_factor", 1.0)),
        "specular_factor": float(getattr(data, "specular_factor", 1.0)),
        "volume_factor": float(getattr(data, "volume_factor", 1.0)),
        "radius": float(getattr(data, "shadow_soft_size", 0.0)),
        "hide_render": bool(getattr(obj, "hide_render", False)),
        "hide_viewport": bool(getattr(obj, "hide_viewport", False)),
        "extra_light_data": _snapshot_extra_light_data(data),
        "cycles_light_data": _snapshot_cycles_light_data(data),
    }
    if result["type"] == "SUN":
        result["sun_angle_degrees"] = math.degrees(float(getattr(data, "angle", 0.0)))
    elif result["type"] == "AREA":
        result["area_shape"] = str(getattr(data, "shape", "SQUARE"))
        result["size"] = float(getattr(data, "size", 1.0))
        result["size_y"] = float(getattr(data, "size_y", result["size"]))
    elif result["type"] == "SPOT":
        result["spot_angle_degrees"] = math.degrees(float(getattr(data, "spot_size", math.radians(45.0))))
        result["spot_blend"] = float(getattr(data, "spot_blend", 0.15))
    return result


def _world_snapshot(scene: Any) -> dict[str, Any] | None:
    world = getattr(scene, "world", None)
    if world is None:
        return None
    result = {
        "name": str(getattr(world, "name", "")),
        "managed": bool(_custom_get(world, MANAGED_PROP, False)),
        "plan_id": str(_custom_get(world, MANAGED_PLAN_PROP, "")) or None,
        "use_fake_user": bool(getattr(world, "use_fake_user", False)),
        "color_rgb": _sequence(getattr(world, "color", (0.0, 0.0, 0.0)), 3),
        "background_color_rgb": None,
        "strength": None,
    }
    for node in _iter_values(getattr(getattr(world, "node_tree", None), "nodes", None)):
        if str(getattr(node, "type", "")) == "BACKGROUND":
            try:
                result["background_color_rgb"] = _sequence(node.inputs["Color"].default_value, 4)[:3]
                result["strength"] = float(node.inputs["Strength"].default_value)
            except Exception:
                pass
            break
    return result


def _snapshot_scene(
    scene: Any,
    collection_name: str,
    plan_id: str | None,
    *,
    snapshot_non_managed_visibility: bool,
) -> dict[str, Any]:
    collection = _scene_collection(scene, collection_name)
    managed = [_serialize_managed_light(obj) for obj in _managed_map(collection).values()]
    visibility = []
    if snapshot_non_managed_visibility:
        for obj in _iter_values(getattr(scene, "objects", None)):
            if str(getattr(obj, "type", "")) == "LIGHT" and not _is_managed_light(obj):
                visibility.append(
                    {
                        "name": str(getattr(obj, "name", "")),
                        "hide_render": bool(getattr(obj, "hide_render", False)),
                        "hide_viewport": bool(getattr(obj, "hide_viewport", False)),
                    }
                )
    return _plain(
        {
            "scene_name": str(getattr(scene, "name", "")),
            "collection_name": collection_name,
            "collection_existed": collection is not None,
            "plan_id": plan_id,
            "managed_lights": managed,
            "world": _world_snapshot(scene),
            "exposure": float(getattr(getattr(scene, "view_settings", None), "exposure", 0.0)),
            "non_managed_visibility": visibility,
        }
    )


def _restore_light(collection: Any, item: dict[str, Any], obj: Any | None = None) -> Any:
    spec = {
        "id": item["id"],
        "name": item["name"],
        "type": item["type"],
        "location_world": item["location_world"],
        "energy": item["energy"],
        "color_rgb": item["color_rgb"],
        "use_shadow": item["use_shadow"],
        "diffuse_factor": item["diffuse_factor"],
        "specular_factor": item["specular_factor"],
        "volume_factor": item["volume_factor"],
        "radius": item["radius"],
        "target_point": None,
        "target_object": None,
    }
    for key in (
        "sun_angle_degrees",
        "area_shape",
        "size",
        "size_y",
        "spot_angle_degrees",
        "spot_blend",
    ):
        if key in item:
            spec[key] = item[key]
    if obj is None:
        obj = _new_light_object(collection, spec, item.get("plan_id"))
    _apply_light_spec(obj, spec, item.get("plan_id"))
    _restore_extra_light_data(getattr(obj, "data", None), item.get("extra_light_data"))
    _restore_cycles_light_data(
        getattr(obj, "data", None), item.get("cycles_light_data")
    )
    obj.data.name = item.get("data_name", item["name"])
    obj.rotation_mode = item.get("rotation_mode", "XYZ")
    quaternion = item.get("rotation_quaternion")
    euler = item.get("rotation_euler")
    if quaternion:
        obj.rotation_quaternion = tuple(quaternion)
    if euler:
        obj.rotation_euler = tuple(euler)
    obj.hide_render = item.get("hide_render", False)
    obj.hide_viewport = item.get("hide_viewport", False)
    return obj


def _set_managed_world(world: Any, color: list[float], strength: float, plan_id: str | None) -> None:
    _custom_set(world, MANAGED_PROP, True)
    _custom_set(world, MANAGED_PLAN_PROP, plan_id or "")
    world.color = tuple(color)
    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()
    background = tree.nodes.new(type="ShaderNodeBackground")
    background.inputs["Color"].default_value = (*color, 1.0)
    background.inputs["Strength"].default_value = strength
    output = tree.nodes.new(type="ShaderNodeOutputWorld")
    tree.links.new(background.outputs["Background"], output.inputs["Surface"])


def _apply_world_override(scene: Any, value: dict[str, Any], plan_id: str | None) -> None:
    if value["mode"] == "KEEP":
        return
    # Always use a transaction-local datablock. Reusing a same-named or
    # otherwise managed World could mutate a different scene, and rebuilding
    # its nodes would violate the promise to leave the original world intact.
    world = bpy.data.worlds.new(MANAGED_WORLD)
    try:
        _set_managed_world(world, value["color_rgb"], value["strength"], plan_id)
        scene.world = world
    except Exception:
        if int(getattr(world, "users", 0) or 0) == 0:
            try:
                bpy.data.worlds.remove(world)
            except Exception:
                pass
        raise


def _retain_snapshot_world(scene: Any, snapshot: dict[str, Any]) -> None:
    """Keep the previous local World alive across save/reopen until rollback.

    Assigning a transaction World can leave the prior local datablock with zero
    users.  Blender may purge that datablock when the file is saved and opened,
    so the ledger's name alone is insufficient.  The original fake-user bit is
    snapshotted and restored when the transaction is rolled back.
    """
    world_snapshot = snapshot.get("world")
    world = getattr(scene, "world", None)
    if not isinstance(world_snapshot, dict) or world is None:
        return
    if str(getattr(world, "name", "")) != world_snapshot.get("name"):
        raise RuntimeError("Active World changed after the rollback snapshot was created")
    if getattr(world, "library", None) is not None:
        # Linked datablocks are retained by their library relationship and are
        # not writable locally.
        return
    world.use_fake_user = True


def _restore_world(scene: Any, snapshot: dict[str, Any] | None) -> None:
    current_world = getattr(scene, "world", None)
    if snapshot is None:
        scene.world = None
        restored_world = None
    else:
        restored_world = _lookup(getattr(bpy.data, "worlds", None), snapshot["name"])
        if restored_world is None:
            raise RuntimeError(
                f"World '{snapshot['name']}' needed for rollback no longer exists"
            )
        # Transaction worlds never mutate the previous world in place, so
        # rollback restores only the pointer. Rebuilding the old node tree here
        # could alter another scene that shares it.
        scene.world = restored_world
        if "use_fake_user" in snapshot and getattr(restored_world, "library", None) is None:
            restored_world.use_fake_user = bool(snapshot["use_fake_user"])

    if (
        current_world is not None
        and current_world is not restored_world
        and bool(_custom_get(current_world, MANAGED_PROP, False))
        and not bool(getattr(current_world, "use_fake_user", False))
        and int(getattr(current_world, "users", 0) or 0) == 0
    ):
        try:
            bpy.data.worlds.remove(current_world)
        except Exception:
            pass


def _release_snapshot_world(snapshot: dict[str, Any]) -> None:
    """Restore a retained World's fake-user bit and remove an orphaned transaction World."""
    world_snapshot = snapshot.get("world")
    if not isinstance(world_snapshot, dict):
        return
    world = _lookup(getattr(bpy.data, "worlds", None), world_snapshot.get("name"))
    if world is None or getattr(world, "library", None) is not None:
        return
    if "use_fake_user" in world_snapshot:
        world.use_fake_user = bool(world_snapshot["use_fake_user"])
    if (
        bool(_custom_get(world, MANAGED_PROP, False))
        and not bool(getattr(world, "use_fake_user", False))
        and int(getattr(world, "users", 0) or 0) == 0
    ):
        try:
            bpy.data.worlds.remove(world)
        except Exception:
            pass


def _restore_scene(scene: Any, snapshot: dict[str, Any]) -> None:
    collection = _scene_collection(scene, snapshot["collection_name"])
    if collection is not None and not bool(_custom_get(collection, MANAGED_PROP, False)):
        raise RuntimeError(
            f"Collection '{snapshot['collection_name']}' is no longer marked as blend-ai managed"
        )
    current = _managed_map(collection)
    desired_items = {
        str(item["id"]): item for item in snapshot.get("managed_lights", [])
    }
    if len(desired_items) != len(snapshot.get("managed_lights", [])):
        raise RuntimeError("Rollback snapshot contains duplicate managed light IDs")
    for light_id in sorted(set(current) - set(desired_items)):
        _remove_light_object(current.pop(light_id))
    if desired_items:
        collection = collection or _ensure_collection(scene, snapshot["collection_name"])
        for light_id, item in desired_items.items():
            _restore_light(collection, item, current.get(light_id))
    elif snapshot.get("collection_existed", False) and collection is None:
        collection = _ensure_collection(scene, snapshot["collection_name"])
    elif not snapshot.get("collection_existed", True) and collection is not None:
        if _iter_values(getattr(collection, "objects", None)) or _iter_values(
            getattr(collection, "children", None)
        ):
            raise RuntimeError(
                f"New managed collection '{snapshot['collection_name']}' is not empty during rollback"
            )
        if bool(_custom_get(collection, MANAGED_PROP, False)):
            bpy.data.collections.remove(collection)
    _restore_world(scene, snapshot.get("world"))
    view_settings = getattr(scene, "view_settings", None)
    if view_settings is not None:
        view_settings.exposure = snapshot["exposure"]
    for item in snapshot.get("non_managed_visibility", []):
        obj = _lookup(getattr(bpy.data, "objects", None), item["name"])
        if obj is not None and str(getattr(obj, "type", "")) == "LIGHT" and not _is_managed_light(obj):
            obj.hide_render = item["hide_render"]
            obj.hide_viewport = item["hide_viewport"]


def _text_as_string(text: Any) -> str | None:
    method = getattr(text, "as_string", None)
    if callable(method):
        try:
            value = method()
            return value if isinstance(value, str) else None
        except Exception:
            return None
    return None


def _load_ledger() -> None:
    global _ledger_loaded, _ledger_source_token
    texts = getattr(bpy.data, "texts", None)
    source_token = (
        id(texts),
        str(getattr(getattr(bpy, "data", None), "filepath", "") or ""),
    )
    if _ledger_loaded and _ledger_source_token == source_token:
        return
    _ledger.clear()
    _ledger_loaded = True
    _ledger_source_token = source_token
    text = _lookup(texts, LEDGER_TEXT)
    if text is not None and not bool(_custom_get(text, LEDGER_PROP, False)):
        # Never adopt a same-named artist Text datablock.  APPLY preflight
        # reports the collision; this guard also keeps read-only lookups safe.
        return
    raw = _text_as_string(text) if text is not None else None
    if not raw:
        return
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            if payload.get("schema_version") != LEDGER_SCHEMA_VERSION:
                return
            entries = payload.get("transactions")
        else:
            # Read the pre-schema development format once so existing local
            # test files can migrate when the next transaction is persisted.
            entries = payload
        if not isinstance(entries, list):
            return
        for entry in entries[-MAX_LEDGER_ENTRIES:]:
            if isinstance(entry, dict) and isinstance(entry.get("transaction_id"), str):
                _ledger[entry["transaction_id"]] = _plain(entry)
    except Exception:
        return


def _ledger_payload() -> str:
    return json.dumps(
        {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "transactions": list(_ledger.values()),
        },
        allow_nan=False,
        separators=(",", ":"),
    )


def _persist_ledger() -> None:
    texts = getattr(bpy.data, "texts", None)
    if texts is None:
        raise RuntimeError("Blender Text datablocks are unavailable for the rollback ledger")
    created = False
    text = _lookup(texts, LEDGER_TEXT)
    if text is not None and not bool(_custom_get(text, LEDGER_PROP, False)):
        raise RuntimeError(
            f"Text datablock '{LEDGER_TEXT}' is not marked as the blend-ai relighting ledger"
        )
    previous = _text_as_string(text) if text is not None else None
    try:
        if text is None:
            text = texts.new(LEDGER_TEXT)
            created = True
            if str(getattr(text, "name", "")) != LEDGER_TEXT:
                raise RuntimeError(
                    f"Unable to create the dedicated Text datablock '{LEDGER_TEXT}'"
                )
            _custom_set(text, LEDGER_PROP, True)
        text.clear()
        text.write(_ledger_payload())
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
        raise RuntimeError(f"Unable to persist rollback ledger: {exc}") from exc


def _store_transaction(entry: dict[str, Any]) -> None:
    _load_ledger()
    safe = _plain(entry)
    single_size = len(
        json.dumps(
            {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "transactions": [safe],
            },
            separators=(",", ":"),
        ).encode("utf-8")
    )
    if single_size > MAX_LEDGER_BYTES:
        raise ValueError(
            f"Rollback snapshot is {single_size} bytes, exceeding the {MAX_LEDGER_BYTES}-byte limit"
        )
    previous_entries = list(_ledger.items())
    pruned_entries: list[dict[str, Any]] = []
    try:
        transaction_id = safe["transaction_id"]
        _ledger.pop(transaction_id, None)
        _ledger[transaction_id] = safe
        while len(_ledger) > MAX_LEDGER_ENTRIES:
            _key, removed_entry = _ledger.popitem(last=False)
            pruned_entries.append(removed_entry)
        while len(_ledger_payload().encode("utf-8")) > MAX_LEDGER_BYTES and len(_ledger) > 1:
            _key, removed_entry = _ledger.popitem(last=False)
            pruned_entries.append(removed_entry)
        _persist_ledger()
    except Exception:
        _ledger.clear()
        _ledger.update(previous_entries)
        raise
    retained_world_names = {
        entry.get("snapshot", {}).get("world", {}).get("name")
        for entry in _ledger.values()
        if isinstance(entry.get("snapshot", {}).get("world"), dict)
    }
    for removed_entry in pruned_entries:
        world_snapshot = removed_entry.get("snapshot", {}).get("world")
        if (
            isinstance(world_snapshot, dict)
            and world_snapshot.get("name") not in retained_world_names
        ):
            _release_snapshot_world({"world": world_snapshot})


def _find_transaction(transaction_id: str | None, plan_id: str | None) -> tuple[str, dict[str, Any]]:
    _load_ledger()
    if not _ledger:
        raise ValueError("No rollback transactions are available")
    latest_key = next(reversed(_ledger))
    if transaction_id is not None:
        transaction_id = _safe_name(transaction_id, "transaction_id", identifier=True)
        entry = _ledger.get(transaction_id)
        if entry is None:
            raise ValueError(f"Rollback transaction '{transaction_id}' was not found")
        if transaction_id != latest_key:
            raise ValueError(
                f"Rollback transaction '{transaction_id}' is out of order; "
                f"roll back latest transaction '{latest_key}' first"
            )
        return transaction_id, entry
    if plan_id is None:
        raise ValueError("ROLLBACK requires transaction_id or plan_id")
    plan_id = _safe_name(plan_id, "plan_id", identifier=True)
    for key, entry in reversed(_ledger.items()):
        if entry.get("plan_id") == plan_id:
            if key != latest_key:
                raise ValueError(
                    f"Latest transaction belongs to plan_id "
                    f"'{_ledger[latest_key].get('plan_id')}'; roll it back first"
                )
            return key, entry
    raise ValueError(f"No rollback transaction was found for plan_id '{plan_id}'")


def _rollback(params: dict[str, Any]) -> dict[str, Any]:
    transaction_id, entry = _find_transaction(params.get("transaction_id"), params.get("plan_id"))
    scene = bpy.context.scene
    strict = params.get("strict", True)
    _bool(strict, "strict")
    expected_scene = entry.get("scene_name") or entry.get("snapshot", {}).get(
        "scene_name"
    )
    current_scene = str(getattr(scene, "name", ""))
    if not isinstance(expected_scene, str) or not expected_scene:
        raise ValueError(
            f"Rollback transaction '{transaction_id}' lacks a scene identity"
        )
    if current_scene != expected_scene:
        raise ValueError(
            f"Rollback transaction '{transaction_id}' belongs to scene "
            f"'{expected_scene}', not active scene '{current_scene}'"
        )
    target_snapshot = entry["snapshot"]
    expected_revision = params.get("expected_geometry_revision")
    if expected_revision is not None:
        expected_revision = _integer(
            expected_revision,
            "expected_geometry_revision",
            minimum=0,
            maximum=2**63 - 1,
        )
        current_revision = spatial_cache.get_revisions()["geometry_revision"]
        if expected_revision != current_revision:
            raise ValueError(
                f"Stale geometry revision: expected {expected_revision}, current "
                f"{current_revision}"
            )
    _preflight_managed_state(
        scene,
        target_snapshot["collection_name"],
        _validate_scene_overrides(None),
        strict,
    )
    safety_snapshot = _snapshot_scene(
        scene,
        target_snapshot["collection_name"],
        entry.get("plan_id"),
        snapshot_non_managed_visibility=bool(
            target_snapshot.get("non_managed_visibility")
        ),
    )
    previous_ledger = list(_ledger.items())
    spatial_cache.begin_managed_edit()
    try:
        try:
            _retain_snapshot_world(scene, safety_snapshot)
            _restore_scene(scene, target_snapshot)
            _ledger.pop(transaction_id, None)
            _persist_ledger()
        except Exception as rollback_error:
            _ledger.clear()
            _ledger.update(previous_ledger)
            try:
                _restore_scene(scene, safety_snapshot)
                _release_snapshot_world(safety_snapshot)
            except Exception as safety_error:
                raise RuntimeError(
                    f"Rollback transaction '{transaction_id}' failed ({rollback_error}); "
                    f"restoring the pre-rollback state also failed ({safety_error})"
                ) from rollback_error
            raise RuntimeError(
                f"Rollback transaction '{transaction_id}' failed and its pre-rollback "
                f"state was restored: {rollback_error}"
            ) from rollback_error
    finally:
        _flush_dependency_graph_updates()
        spatial_cache.end_managed_edit()
    _release_snapshot_world(safety_snapshot)
    spatial_cache.mark_lighting_dirty()
    revisions = spatial_cache.get_revisions()
    return _plain(
        {
            "action": "ROLLBACK",
            "rolled_back": True,
            "transaction_id": transaction_id,
            "plan_id": entry.get("plan_id"),
            **revisions,
            "scene_revision": spatial_cache.make_scene_revision(scene),
            "warnings": [],
        }
    )


def _apply_overrides(scene: Any, overrides: dict[str, Any], plan_id: str | None) -> None:
    if overrides["existing_light_policy"] == "MUTE_NON_MANAGED":
        for obj in _iter_values(getattr(scene, "objects", None)):
            if str(getattr(obj, "type", "")) == "LIGHT" and not _is_managed_light(obj):
                obj.hide_render = True
                obj.hide_viewport = True
    _apply_world_override(scene, overrides["world"], plan_id)
    if overrides["exposure"] is not None:
        scene.view_settings.exposure = overrides["exposure"]


def _flush_dependency_graph_updates() -> None:
    """Flush deferred managed-light tags before ending revision suppression."""
    view_layer = getattr(getattr(bpy, "context", None), "view_layer", None)
    update = getattr(view_layer, "update", None)
    if callable(update):
        try:
            update()
        except (AttributeError, ReferenceError, RuntimeError):
            # Once the scope ends, any deferred update is classified normally
            # and may conservatively invalidate the geometry cache.
            pass


def _managed_plan_map(collection: Any | None, plan_id: str) -> dict[str, Any]:
    """Return managed lights owned by exactly one model-supplied plan ID."""
    return {
        light_id: obj
        for light_id, obj in _managed_map(collection).items()
        if str(_custom_get(obj, MANAGED_PLAN_PROP, "")) == plan_id
    }


def _assert_display_name_available(spec: dict[str, Any], existing: Any | None) -> None:
    """Reject Blender's implicit ``.001`` renaming for managed light plans."""
    name = spec["name"]
    object_collision = _lookup(getattr(bpy.data, "objects", None), name)
    if object_collision is not None and object_collision is not existing:
        raise ValueError(
            f"Light display name '{name}' collides with existing object "
            f"'{getattr(object_collision, 'name', name)}'"
        )
    data_collision = _lookup(getattr(bpy.data, "lights", None), name)
    existing_data = getattr(existing, "data", None) if existing is not None else None
    if data_collision is not None and data_collision is not existing_data:
        raise ValueError(
            f"Light display name '{name}' collides with existing light data "
            f"'{getattr(data_collision, 'name', name)}'"
        )


def _state_values_equal(left: Any, right: Any) -> bool:
    if _is_number(left) and _is_number(right):
        return math.isclose(float(left), float(right), rel_tol=1e-7, abs_tol=1e-8)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _state_values_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _light_spec_would_change(obj: Any, spec: dict[str, Any], plan_id: str) -> bool:
    """Predict the supported fields APPLY writes without mutating Blender."""
    data = getattr(obj, "data", None)
    if data is None:
        return True
    if (
        str(_custom_get(obj, MANAGED_ID_PROP, "")) != spec["id"]
        or str(_custom_get(data, MANAGED_ID_PROP, "")) != spec["id"]
        or str(_custom_get(obj, MANAGED_PLAN_PROP, "")) != plan_id
        or str(_custom_get(data, MANAGED_PLAN_PROP, "")) != plan_id
        or str(getattr(obj, "name", "")) != spec["name"]
        or str(getattr(data, "name", "")) != spec["name"]
        or str(getattr(data, "type", "")) != spec["type"]
    ):
        return True

    comparisons = (
        (_world_location(obj), spec["location_world"]),
        (getattr(data, "energy", None), spec["energy"]),
        (_sequence(getattr(data, "color", ()), 3), spec["color_rgb"]),
        (getattr(data, "use_shadow", None), spec["use_shadow"]),
        (getattr(data, "diffuse_factor", None), spec["diffuse_factor"]),
        (getattr(data, "specular_factor", None), spec["specular_factor"]),
        (getattr(data, "volume_factor", None), spec["volume_factor"]),
    )
    if any(not _state_values_equal(current, desired) for current, desired in comparisons):
        return True
    if "radius" in spec and not _state_values_equal(
        getattr(data, "shadow_soft_size", None), spec["radius"]
    ):
        return True
    if "sun_angle_degrees" in spec and not _state_values_equal(
        math.degrees(float(getattr(data, "angle", 0.0))),
        spec["sun_angle_degrees"],
    ):
        return True
    if spec["type"] == "AREA":
        if str(getattr(data, "shape", "")) != spec["area_shape"]:
            return True
        if not _state_values_equal(getattr(data, "size", None), spec["size"]):
            return True
        if "size_y" in spec and not _state_values_equal(
            getattr(data, "size_y", None), spec["size_y"]
        ):
            return True
    if spec["type"] == "SPOT" and (
        not _state_values_equal(
            math.degrees(float(getattr(data, "spot_size", 0.0))),
            spec["spot_angle_degrees"],
        )
        or not _state_values_equal(
            getattr(data, "spot_blend", None), spec["spot_blend"]
        )
    ):
        return True

    target = spec.get("target_point")
    if spec.get("target_object") is not None:
        target = _world_location(_resolve_target(spec["target_object"]))
    if target is not None and spec["type"] != "POINT":
        try:
            from mathutils import Vector

            expected = Vector(_sub(target, spec["location_world"])).to_track_quat(
                "-Z", "Y"
            )
            actual = getattr(obj, "rotation_quaternion", None)
            if str(getattr(obj, "rotation_mode", "")) != "QUATERNION" or actual is None:
                return True
            dot = abs(sum(float(actual[index]) * float(expected[index]) for index in range(4)))
            if not math.isclose(dot, 1.0, rel_tol=1e-7, abs_tol=1e-7):
                return True
        except Exception:
            return True
    return False


def _propose_plan_changes(scene: Any, normalized: dict[str, Any]) -> dict[str, Any]:
    """Return the exact bounded operations VALIDATE would permit APPLY to run."""
    plan_id = normalized["plan_id"]
    assert isinstance(plan_id, str)
    collection = _scene_collection(scene, normalized["collection_name"])
    all_managed = _managed_map(collection)
    existing = {
        light_id: obj
        for light_id, obj in all_managed.items()
        if str(_custom_get(obj, MANAGED_PLAN_PROP, "")) == plan_id
    }
    desired = {spec["id"] for spec in normalized["lights"]}

    for spec in normalized["lights"]:
        light_id = spec["id"]
        collision = all_managed.get(light_id)
        if collision is not None and light_id not in existing:
            owner = str(_custom_get(collision, MANAGED_PLAN_PROP, "")) or "<unknown>"
            raise ValueError(
                f"Managed light id '{light_id}' belongs to plan '{owner}', not "
                f"'{plan_id}'"
            )
        _assert_display_name_available(spec, existing.get(light_id))

    for light_id in normalized["remove_ids"]:
        collision = all_managed.get(light_id)
        if collision is not None and light_id not in existing:
            owner = str(_custom_get(collision, MANAGED_PLAN_PROP, "")) or "<unknown>"
            raise ValueError(
                f"remove_ids contains managed light id '{light_id}' owned by plan "
                f"'{owner}', not '{plan_id}'"
            )

    remove = set(normalized["remove_ids"])
    if normalized["mode"] == "REPLACE_MANAGED":
        remove.update(set(existing) - desired)
    final_ids = (set(existing) - remove) | desired
    if len(final_ids) > MAX_MANAGED_LIGHTS:
        raise ValueError(
            f"Plan '{plan_id}' would contain {len(final_ids)} lights; maximum is "
            f"{MAX_MANAGED_LIGHTS}"
        )

    muted_names: list[str] = []
    already_muted_names: list[str] = []
    if normalized["scene_overrides"]["existing_light_policy"] == "MUTE_NON_MANAGED":
        for obj in _iter_values(getattr(scene, "objects", None)):
            if str(getattr(obj, "type", "")) != "LIGHT" or _is_managed_light(obj):
                continue
            name = str(getattr(obj, "name", ""))
            if bool(getattr(obj, "hide_render", False)) and bool(
                getattr(obj, "hide_viewport", False)
            ):
                already_muted_names.append(name)
            else:
                muted_names.append(name)
        muted_names.sort()
        already_muted_names.sort()

    create_ids = [
        spec["id"] for spec in normalized["lights"] if spec["id"] not in existing
    ]
    update_ids = [
        spec["id"]
        for spec in normalized["lights"]
        if spec["id"] in existing
        and _light_spec_would_change(existing[spec["id"]], spec, plan_id)
    ]
    unchanged_ids = [
        spec["id"]
        for spec in normalized["lights"]
        if spec["id"] in existing and spec["id"] not in update_ids
    ]
    exposure_override = normalized["scene_overrides"]["exposure"]
    current_exposure = float(
        getattr(getattr(scene, "view_settings", None), "exposure", 0.0)
    )

    return {
        "plan_id": plan_id,
        "mode": normalized["mode"],
        "create_ids": create_ids,
        "update_ids": update_ids,
        "unchanged_ids": unchanged_ids,
        "remove_ids": sorted(light_id for light_id in remove if light_id in existing),
        "missing_remove_ids": sorted(
            light_id for light_id in normalized["remove_ids"] if light_id not in all_managed
        ),
        "final_plan_light_ids": sorted(final_ids),
        "final_plan_light_count": len(final_ids),
        "mute_non_managed_light_names": muted_names,
        "already_muted_non_managed_light_names": already_muted_names,
        "world_override": normalized["scene_overrides"]["world"],
        "world_will_change": normalized["scene_overrides"]["world"]["mode"]
        == "MANAGED_SOLID",
        "exposure_override": exposure_override,
        "exposure_will_change": exposure_override is not None
        and not _state_values_equal(current_exposure, exposure_override),
    }


def handle_apply_light_plan(params: dict[str, Any]) -> dict[str, Any]:
    """Validate, apply, or roll back an atomic managed-light transaction."""
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    _reject_unknown(
        params,
        {
            "action",
            "plan_id",
            "mode",
            "expected_geometry_revision",
            "collection_name",
            "lights",
            "remove_ids",
            "scene_overrides",
            "transaction_id",
            "strict",
        },
    )
    action = str(params.get("action", "VALIDATE")).upper()
    if action not in {"VALIDATE", "APPLY", "ROLLBACK"}:
        raise ValueError("action must be VALIDATE, APPLY, or ROLLBACK")
    if action == "ROLLBACK":
        if params.get("lights") or params.get("remove_ids") or params.get("scene_overrides") is not None:
            raise ValueError("ROLLBACK does not accept lights, remove_ids, or scene_overrides")
        return _rollback(params)

    if params.get("plan_id") is None:
        raise ValueError(f"plan_id is required for {action}")
    if params.get("transaction_id") is not None:
        raise ValueError("transaction_id is only valid for ROLLBACK")

    normalized, warnings = _validate_plan(params)
    scene = bpy.context.scene
    before_revisions = spatial_cache.get_revisions()
    expected = normalized["expected_geometry_revision"]
    if expected is not None and expected != before_revisions["geometry_revision"]:
        raise ValueError(
            f"Stale geometry revision: expected {expected}, current {before_revisions['geometry_revision']}"
        )
    proposed_changes = _propose_plan_changes(scene, normalized)
    if action == "VALIDATE":
        return _plain(
            {
                "action": "VALIDATE",
                "valid": True,
                "plan": normalized,
                "proposed_changes": proposed_changes,
                **before_revisions,
                "scene_revision": spatial_cache.make_scene_revision(scene),
                "warnings": warnings,
            }
        )

    # Freeze object targets to world-space before the first mutation.  A plan
    # may legitimately remove or rename another managed light that served as
    # an aiming anchor; resolving it later would make APPLY diverge from its
    # successful validation.
    apply_specs: list[dict[str, Any]] = []
    for source_spec in normalized["lights"]:
        spec = dict(source_spec)
        if spec.get("target_object") is not None:
            spec["target_point"] = _world_location(
                _resolve_target(spec["target_object"])
            )
            spec["target_object"] = None
        apply_specs.append(spec)

    snapshot = _snapshot_scene(
        scene,
        normalized["collection_name"],
        normalized["plan_id"],
        snapshot_non_managed_visibility=(
            normalized["scene_overrides"]["existing_light_policy"]
            == "MUTE_NON_MANAGED"
        ),
    )
    transaction_id = uuid.uuid4().hex
    entry = {
        "transaction_id": transaction_id,
        "plan_id": normalized["plan_id"],
        "scene_name": str(getattr(scene, "name", "")),
        "created_at_unix": time.time(),
        "snapshot": snapshot,
    }
    # Refuse before mutation if the complete transaction entry cannot retain
    # its manual rollback snapshot within the documented ledger ceiling.
    entry_bytes = _payload_size(
        {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "transactions": [entry],
        }
    )
    if entry_bytes > MAX_LEDGER_BYTES:
        raise ValueError(
            f"Rollback snapshot is {entry_bytes} bytes, exceeding the {MAX_LEDGER_BYTES}-byte limit"
        )
    created: list[str] = []
    updated: list[str] = []
    unchanged: list[str] = []
    removed: list[str] = []
    spatial_cache.begin_managed_edit()
    try:
        try:
            if normalized["scene_overrides"]["world"]["mode"] == "MANAGED_SOLID":
                _retain_snapshot_world(scene, snapshot)
            collection = _ensure_collection(scene, normalized["collection_name"])
            existing = _managed_plan_map(collection, normalized["plan_id"])
            for light_id in proposed_changes["remove_ids"]:
                obj = existing.pop(light_id, None)
                if obj is not None:
                    _remove_light_object(obj)
                    removed.append(light_id)
            for spec in apply_specs:
                obj = existing.get(spec["id"])
                if obj is None:
                    obj = _new_light_object(collection, spec, normalized["plan_id"])
                    _apply_light_spec(obj, spec, normalized["plan_id"])
                    created.append(spec["id"])
                else:
                    before = _plain(_serialize_managed_light(obj))
                    _apply_light_spec(obj, spec, normalized["plan_id"])
                    after = _plain(_serialize_managed_light(obj))
                    if before == after:
                        unchanged.append(spec["id"])
                    else:
                        updated.append(spec["id"])
            _apply_overrides(
                scene,
                normalized["scene_overrides"],
                normalized["plan_id"],
            )
            _store_transaction(entry)
        except Exception as apply_error:
            _ledger.pop(transaction_id, None)
            try:
                _restore_scene(scene, snapshot)
            except Exception as rollback_error:
                raise RuntimeError(
                    f"Light plan failed ({apply_error}) and automatic rollback also failed ({rollback_error})"
                ) from apply_error
            raise RuntimeError(
                f"Light plan failed and was rolled back: {apply_error}"
            ) from apply_error
    finally:
        _flush_dependency_graph_updates()
        spatial_cache.end_managed_edit()

    spatial_cache.mark_lighting_dirty()
    after_revisions = spatial_cache.get_revisions()
    return _plain(
        {
            "action": "APPLY",
            "applied": True,
            "plan_id": normalized["plan_id"],
            "transaction_id": transaction_id,
            "mode": normalized["mode"],
            "collection_name": normalized["collection_name"],
            "created": created,
            "updated": updated,
            "unchanged": unchanged,
            "removed": removed,
            "proposed_changes": proposed_changes,
            "rollback_available": True,
            "geometry_revision_before": before_revisions["geometry_revision"],
            "lighting_revision_before": before_revisions["lighting_revision"],
            **after_revisions,
            "scene_revision": spatial_cache.make_scene_revision(scene),
            "warnings": warnings,
        }
    )


def register() -> None:
    """Register spatial relighting commands and revision handlers."""
    spatial_cache.register_handlers()
    dispatcher.register_handler("get_lighting_context", handle_get_lighting_context)
    dispatcher.register_handler("batch_raycast", handle_batch_raycast)
    dispatcher.register_handler("apply_light_plan", handle_apply_light_plan)
    handlers = _load_post_handlers()
    if handlers is not None and _clear_ledger_on_load not in handlers:
        handlers.append(_clear_ledger_on_load)


def unregister() -> None:
    """Remove only the revision hooks; dispatcher owns command registry lifetime."""
    handlers = _load_post_handlers()
    if handlers is not None:
        while _clear_ledger_on_load in handlers:
            handlers.remove(_clear_ledger_on_load)
    _reset_ledger_state()
    _reset_raycast_index()
    spatial_cache.unregister_handlers()
