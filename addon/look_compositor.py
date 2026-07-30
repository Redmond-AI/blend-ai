"""Managed look-profile compositor compatibility for Blender 4.2 and 5.x.

Blender 5.0 replaced the Scene-owned ``node_tree`` compositor with an
assignable ``compositing_node_group``.  The output contract changed with it:
5.x renders the Image interface socket linked to the active
``NodeGroupOutput``, while 4.2 renders a ``CompositorNodeComposite``.  Keeping
that distinction in one module prevents profile compilation and render-batch
code from guessing which pixels are authoritative.

Every mutating function in this module requires a compositor tree marked by
``ensure_unique_managed_compositor``.  That guard is intentional: artist node
trees must never be silently cleared or rewired.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from typing import Any

import bpy


API_NODE_GROUP = "NODE_GROUP_5X"
API_LEGACY = "LEGACY_4X"

MANAGED_TREE_PROP = "blend_ai_look_compositor_managed"
MANAGED_NESTED_TREE_PROP = "blend_ai_look_compositor_nested_managed"
MANAGED_PARENT_PROP = "blend_ai_look_compositor_parent"
MANAGED_NODE_PROP = "blend_ai_look_compositor_node"
MANAGED_IMAGE_PROP = "blend_ai_look_grain_image"
ROLE_PROP = "blend_ai_look_role"
SETTINGS_PROP = "blend_ai_look_settings"
SCHEMA_PROP = "blend_ai_look_compositor_schema"
API_PROP = "blend_ai_look_compositor_api"
MANAGED_NAME_PROP = "blend_ai_look_compositor_name"
SCHEMA_VERSION = 1

_SAFE_NAME = re.compile(r"^[^\x00-\x1f\x7f]{1,128}$")
_SEMANTIC_NODE_ATTRIBUTES = (
    "blend_type",
    "glare_type",
    "quality",
    "threshold",
    "size",
    "mix",
    "x",
    "y",
    "width",
    "height",
    "rotation",
    "filter_type",
    "use_relative",
    "factor_x",
    "factor_y",
    "size_x",
    "size_y",
    "space",
    "frame_method",
    "use_alpha",
    "is_active_output",
    "save_as_render",
    "use_file_extension",
    "scene",
    "image",
)
_FORMAT_ATTRIBUTES = (
    "file_format",
    "color_mode",
    "color_depth",
    "compression",
    "quality",
    "exr_codec",
    "tiff_codec",
    "color_management",
)
_TREE_MARKER_KEYS = (
    MANAGED_TREE_PROP,
    MANAGED_NESTED_TREE_PROP,
    MANAGED_PARENT_PROP,
    SCHEMA_PROP,
    API_PROP,
    MANAGED_NAME_PROP,
)
_ITEM_STATE_ATTRIBUTES = (
    "name",
    "path",
    "file_name",
    "socket_type",
    "vector_socket_dimensions",
    "override_node_format",
    "use_node_format",
    "save_as_render",
)
_MISSING = object()


class CompositorCompatibilityError(RuntimeError):
    """Raised when Blender cannot honor the managed compositor contract."""


def _custom_get(value: Any, key: str, default: Any = None) -> Any:
    getter = getattr(value, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except (TypeError, ReferenceError):
            return default
    props = getattr(value, "_custom_properties", None)
    if isinstance(props, dict):
        return props.get(key, default)
    return default


def _custom_set(value: Any, key: str, item: Any) -> None:
    try:
        value[key] = item
        return
    except Exception:
        props = getattr(value, "_custom_properties", None)
        if isinstance(props, dict):
            props[key] = item
            return
    raise CompositorCompatibilityError(f"Unable to set compositor marker '{key}'")


def _custom_delete(value: Any, key: str) -> None:
    try:
        del value[key]
        return
    except (KeyError, TypeError, AttributeError, ReferenceError):
        pass
    except Exception as exc:
        raise CompositorCompatibilityError(
            f"Unable to remove compositor marker '{key}': {exc}"
        ) from exc
    props = getattr(value, "_custom_properties", None)
    if isinstance(props, dict):
        props.pop(key, None)
        return
    raise CompositorCompatibilityError(f"Unable to remove compositor marker '{key}'")


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
            result = getter(name)
            if result is not None:
                return result
        except (TypeError, ReferenceError):
            pass
    for value in _iter_values(collection):
        if str(getattr(value, "name", "")) == name:
            return value
    return None


def _node_type(node: Any) -> str:
    return str(getattr(node, "bl_idname", getattr(node, "type", "")))


def _safe_name(value: str, field: str = "name") -> str:
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value):
        raise ValueError(f"{field} must be a safe non-empty name of at most 128 characters")
    return value


def _safe_path_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return (token or "file_output")[:64]


def _safe_output_pattern(value: Any, fallback: str) -> str:
    """Return one path-free File Output pattern while retaining frame hashes."""

    raw = "" if value is None else str(value)
    basename = raw.replace("\\", "/").rsplit("/", 1)[-1]
    token = re.sub(r"[^A-Za-z0-9_.#-]+", "_", basename).strip("._")
    if not token or token in {".", ".."}:
        token = _safe_path_token(fallback)
    return token[:128]


def _identity(value: Any) -> tuple[str, int]:
    pointer = getattr(value, "as_pointer", None)
    if callable(pointer):
        try:
            resolved = int(pointer())
            if resolved:
                return ("RNA", resolved)
        except (ReferenceError, TypeError, ValueError):
            pass
    return ("PY", id(value))


def _group_child(node: Any) -> Any | None:
    child = getattr(node, "node_tree", None)
    if child is None or not hasattr(child, "nodes"):
        return None
    return child


def _sorted_nodes(tree: Any) -> list[Any]:
    return sorted(
        _iter_values(getattr(tree, "nodes", None)),
        key=lambda node: (str(getattr(node, "name", "")), _node_type(node)),
    )


def _walk_compositor_trees(root: Any) -> list[tuple[Any, tuple[str, ...]]]:
    """Walk every unique nested node group and reject recursive group cycles."""

    result: list[tuple[Any, tuple[str, ...]]] = []
    visited: set[tuple[str, int]] = set()
    visiting: set[tuple[str, int]] = set()

    def visit(tree: Any, path: tuple[str, ...]) -> None:
        identity = _identity(tree)
        if identity in visiting:
            location = " -> ".join(path) or "<root>"
            raise CompositorCompatibilityError(
                f"Recursive compositor node-group cycle detected at {location}"
            )
        if identity in visited:
            return
        visiting.add(identity)
        visited.add(identity)
        result.append((tree, path))
        for node in _sorted_nodes(tree):
            child = _group_child(node)
            if child is not None:
                visit(child, (*path, str(getattr(node, "name", ""))))
        visiting.remove(identity)

    visit(root, ())
    return result


def _resolve_tree_path(root: Any, path: list[str] | tuple[str, ...]) -> Any:
    tree = root
    for node_name in path:
        node = _lookup(getattr(tree, "nodes", None), str(node_name))
        child = _group_child(node) if node is not None else None
        if child is None:
            raise CompositorCompatibilityError(
                f"Nested compositor group path {'/'.join(path)!r} is no longer resolvable"
            )
        tree = child
    return tree


def _owned_group_name(root_name: str, path: tuple[str, ...]) -> str:
    identity = "/".join(path)
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
    label = _safe_path_token("__".join(path))
    return f"{root_name}__{label}_{digest}"[:128]


def _validate_group_graph_acyclic(root: Any) -> None:
    _walk_compositor_trees(root)


def _copy_tree(tree: Any) -> Any:
    copier = getattr(tree, "copy", None)
    if not callable(copier):
        raise CompositorCompatibilityError(
            f"Compositor '{getattr(tree, 'name', '')}' cannot be copied"
        )
    try:
        copied = copier()
    except Exception as exc:
        raise CompositorCompatibilityError(
            f"Unable to copy compositor '{getattr(tree, 'name', '')}': {exc}"
        ) from exc
    if copied is None or copied is tree:
        raise CompositorCompatibilityError(
            "Blender did not return a unique compositor node-group copy"
        )
    return copied


def _clone_nested_group_graph(
    source_root: Any,
    owned_root: Any,
    *,
    root_name: str,
    created: list[Any],
) -> None:
    """Deep-copy every nested group, preserving shared DAG references."""

    memo: dict[tuple[str, int], Any] = {_identity(source_root): owned_root}

    def clone_children(
        source_tree: Any,
        owned_tree: Any,
        path: tuple[str, ...],
    ) -> None:
        owned_nodes = getattr(owned_tree, "nodes", None)
        for source_node in _sorted_nodes(source_tree):
            source_child = _group_child(source_node)
            if source_child is None:
                continue
            node_name = str(getattr(source_node, "name", ""))
            owned_node = _lookup(owned_nodes, node_name)
            if owned_node is None:
                raise CompositorCompatibilityError(
                    f"Copied compositor lost nested group node '{node_name}'"
                )
            child_identity = _identity(source_child)
            owned_child = memo.get(child_identity)
            if owned_child is None:
                owned_child = _copy_tree(source_child)
                created.append(owned_child)
                child_path = (*path, node_name)
                try:
                    owned_child.name = _owned_group_name(root_name, child_path)
                except Exception as exc:
                    raise CompositorCompatibilityError(
                        f"Unable to name owned nested compositor group '{node_name}': {exc}"
                    ) from exc
                memo[child_identity] = owned_child
                clone_children(source_child, owned_child, child_path)
            try:
                owned_node.node_tree = owned_child
            except Exception as exc:
                raise CompositorCompatibilityError(
                    f"Unable to remap nested compositor group '{node_name}': {exc}"
                ) from exc

    clone_children(source_root, owned_root, ())


def _mark_managed_tree(
    tree: Any,
    api: str,
    name: str,
    *,
    nested: bool = False,
    parent_name: str | None = None,
) -> None:
    if nested:
        if _custom_get(tree, MANAGED_TREE_PROP, _MISSING) is not _MISSING:
            _custom_delete(tree, MANAGED_TREE_PROP)
        _custom_set(tree, MANAGED_NESTED_TREE_PROP, True)
        _custom_set(tree, MANAGED_PARENT_PROP, parent_name or "")
    else:
        for key in (MANAGED_NESTED_TREE_PROP, MANAGED_PARENT_PROP):
            if _custom_get(tree, key, _MISSING) is not _MISSING:
                _custom_delete(tree, key)
        _custom_set(tree, MANAGED_TREE_PROP, True)
    _custom_set(tree, SCHEMA_PROP, SCHEMA_VERSION)
    _custom_set(tree, API_PROP, api)
    _custom_set(tree, MANAGED_NAME_PROP, name)


def owned_nested_compositor_trees(tree: Any) -> list[Any]:
    """Return the distinct deep-copy groups owned below one managed root."""

    graph = _walk_compositor_trees(tree)
    return [nested_tree for nested_tree, _path in graph[1:]]


def _remove_node_groups(created: list[Any]) -> list[str]:
    errors: list[str] = []
    node_groups = getattr(getattr(bpy, "data", None), "node_groups", None)
    remover = getattr(node_groups, "remove", None)
    for tree in created:
        if not callable(remover):
            errors.append(f"cannot remove temporary node group '{getattr(tree, 'name', '')}'")
            continue
        try:
            try:
                remover(tree, do_unlink=True)
            except TypeError:
                remover(tree)
        except Exception as exc:
            errors.append(f"node group '{getattr(tree, 'name', '')}': {exc}")
    return errors


def _restore_custom_markers(tree: Any, snapshot: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key, value in snapshot.items():
        try:
            if value is _MISSING:
                _custom_delete(tree, key)
            else:
                _custom_set(tree, key, value)
        except Exception as exc:
            errors.append(f"marker '{key}': {exc}")
    return errors


def detect_compositor_api(scene: Any) -> str:
    """Return the compositor API exposed by this concrete Scene instance."""

    if scene is None:
        raise CompositorCompatibilityError("A Scene instance is required")
    if hasattr(scene, "compositing_node_group"):
        return API_NODE_GROUP
    if hasattr(scene, "node_tree") and hasattr(scene, "use_nodes"):
        return API_LEGACY
    raise CompositorCompatibilityError(
        "Scene exposes neither Blender 5.x compositing_node_group nor Blender 4.2 node_tree"
    )


def get_compositor_tree(scene: Any) -> Any | None:
    """Return the Scene's root compositor without creating any datablocks."""

    api = detect_compositor_api(scene)
    attribute = "compositing_node_group" if api == API_NODE_GROUP else "node_tree"
    return getattr(scene, attribute, None)


def _new_node_group(name: str) -> Any:
    node_groups = getattr(getattr(bpy, "data", None), "node_groups", None)
    creator = getattr(node_groups, "new", None)
    if not callable(creator):
        raise CompositorCompatibilityError("Blender cannot create compositor node groups")
    try:
        return creator(name, "CompositorNodeTree")
    except TypeError:
        return creator(name=name, type="CompositorNodeTree")
    except Exception as exc:
        raise CompositorCompatibilityError(
            f"Unable to create compositor node group '{name}': {exc}"
        ) from exc


def ensure_unique_managed_compositor(
    scene: Any,
    *,
    name: str,
    source_tree: Any | None = None,
    copy_source: bool = True,
) -> Any:
    """Copy or create the unique compositor owned by a compiled profile Scene.

    On Blender 5.x the compositing node group is explicitly copied and assigned.
    Blender 4.2 owns its tree through ``Scene.use_nodes``; callers compiling a
    copied Scene should pass the original Scene's tree as ``source_tree`` so an
    accidental shared tree is detected before any rewrite.
    """

    name = _safe_name(name, "compositor name")
    api = detect_compositor_api(scene)
    current = get_compositor_tree(scene)
    created: list[Any] = []

    if api == API_NODE_GROUP:
        previous_tree = current
        had_use_compositing = hasattr(scene, "use_compositing")
        previous_use_compositing = getattr(scene, "use_compositing", None)
        try:
            template = source_tree if source_tree is not None else current
            if copy_source and template is not None:
                _validate_group_graph_acyclic(template)
                tree = _copy_tree(template)
                created.append(tree)
                _clone_nested_group_graph(
                    template,
                    tree,
                    root_name=name,
                    created=created,
                )
            else:
                tree = _new_node_group(name)
                created.append(tree)
            try:
                tree.name = name
            except Exception as exc:
                raise CompositorCompatibilityError(
                    f"Unable to name managed compositor '{name}': {exc}"
                ) from exc
            for owned_tree in created:
                _mark_managed_tree(
                    owned_tree,
                    api,
                    str(getattr(owned_tree, "name", name)),
                    nested=owned_tree is not tree,
                    parent_name=name,
                )
            scene.compositing_node_group = tree
            if had_use_compositing:
                scene.use_compositing = True
            return tree
        except Exception as exc:
            rollback_errors: list[str] = []
            try:
                scene.compositing_node_group = previous_tree
            except Exception as rollback_exc:
                rollback_errors.append(f"Scene compositor assignment: {rollback_exc}")
            if had_use_compositing:
                try:
                    scene.use_compositing = previous_use_compositing
                except Exception as rollback_exc:
                    rollback_errors.append(f"Scene use_compositing: {rollback_exc}")
            rollback_errors.extend(_remove_node_groups(created))
            if rollback_errors:
                raise CompositorCompatibilityError(
                    f"Managed compositor creation failed ({exc}); rollback also failed: "
                    + "; ".join(rollback_errors)
                ) from exc
            if isinstance(exc, CompositorCompatibilityError):
                raise
            raise CompositorCompatibilityError(
                f"Unable to create managed compositor '{name}': {exc}"
            ) from exc

    previous_use_nodes = bool(getattr(scene, "use_nodes", False))
    tree = None
    original_name: Any = _MISSING
    marker_snapshot: dict[str, Any] = {}
    original_group_references: list[tuple[Any, Any]] = []
    try:
        scene.use_nodes = True
        tree = getattr(scene, "node_tree", None)
        if tree is None:
            raise CompositorCompatibilityError(
                "Blender 4.2 did not create a Scene compositor after use_nodes=True"
            )
        if source_tree is tree:
            raise CompositorCompatibilityError(
                "Compiled Blender 4.2 Scene still shares its source compositor; "
                "refusing to rewrite the artist tree"
            )
        _validate_group_graph_acyclic(tree)
        original_name = getattr(tree, "name", _MISSING)
        marker_snapshot = {
            key: _custom_get(tree, key, _MISSING) for key in _TREE_MARKER_KEYS
        }
        if copy_source:
            original_group_references = [
                (node, child)
                for node in _sorted_nodes(tree)
                if (child := _group_child(node)) is not None
            ]
            _clone_nested_group_graph(
                tree,
                tree,
                root_name=name,
                created=created,
            )
        try:
            tree.name = name
        except (AttributeError, TypeError):
            # Blender 4.2 can expose the embedded Scene node tree with a
            # read-only ID name.  The explicit marker remains deterministic.
            pass
        _mark_managed_tree(tree, api, name)
        for owned_tree in created:
            _mark_managed_tree(
                owned_tree,
                api,
                str(getattr(owned_tree, "name", name)),
                nested=True,
                parent_name=name,
            )
        return tree
    except Exception as exc:
        rollback_errors = []
        if tree is not None:
            for node, original_child in original_group_references:
                try:
                    node.node_tree = original_child
                except Exception as rollback_exc:
                    rollback_errors.append(
                        f"nested group '{getattr(node, 'name', '')}': {rollback_exc}"
                    )
            if original_name is not _MISSING:
                try:
                    tree.name = original_name
                except Exception as rollback_exc:
                    rollback_errors.append(f"embedded compositor name: {rollback_exc}")
            rollback_errors.extend(_restore_custom_markers(tree, marker_snapshot))
        try:
            scene.use_nodes = previous_use_nodes
        except Exception as rollback_exc:
            rollback_errors.append(f"Scene use_nodes: {rollback_exc}")
        rollback_errors.extend(_remove_node_groups(created))
        if rollback_errors:
            raise CompositorCompatibilityError(
                f"Managed legacy compositor setup failed ({exc}); rollback also failed: "
                + "; ".join(rollback_errors)
            ) from exc
        if isinstance(exc, CompositorCompatibilityError):
            raise
        raise CompositorCompatibilityError(
            f"Unable to prepare managed legacy compositor '{name}': {exc}"
        ) from exc


def rebind_render_layers_to_scene(tree: Any, scene: Any) -> int:
    """Bind every Render Layers node in a managed compositor graph to its profile Scene."""

    if tree is None or not bool(_custom_get(tree, MANAGED_TREE_PROP, False)):
        raise CompositorCompatibilityError(
            "Refusing to rebind Render Layers nodes in an unmarked compositor"
        )
    count = 0
    for current, _path in _walk_compositor_trees(tree):
        for node in _sorted_nodes(current):
            if str(getattr(node, "bl_idname", "")) != "CompositorNodeRLayers":
                continue
            if not hasattr(node, "scene"):
                raise CompositorCompatibilityError(
                    f"Render Layers node '{getattr(node, 'name', '')}' has no Scene binding"
                )
            node.scene = scene
            count += 1
    return count


def _require_managed_tree(scene: Any, tree: Any | None = None) -> tuple[str, Any]:
    api = detect_compositor_api(scene)
    tree = tree if tree is not None else get_compositor_tree(scene)
    if tree is None:
        raise CompositorCompatibilityError("Scene has no compositor tree")
    if tree is not get_compositor_tree(scene):
        raise CompositorCompatibilityError("Supplied compositor is not assigned to the Scene")
    if not bool(_custom_get(tree, MANAGED_TREE_PROP, False)):
        raise CompositorCompatibilityError(
            "Refusing to rewrite an unmarked compositor; compile a unique profile tree first"
        )
    return api, tree


def _number(value: Any, field: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < minimum or result > maximum:
        raise ValueError(f"{field} must be between {minimum} and {maximum}")
    return result


def _integer(value: Any, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{field} must be between {minimum} and {maximum}")
    return value


def _normalize_effect(value: Any, defaults: dict[str, Any], field: str) -> dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError(f"post.{field} must be an object")
    unknown = sorted(set(value) - set(defaults))
    if unknown:
        raise ValueError(f"post.{field} contains unsupported fields: {unknown}")
    return {**defaults, **value}


def _normalize_post_spec(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("post must be an object")
    unknown = sorted(set(value) - {"mode", "bloom", "grain", "vignette"})
    if unknown:
        raise ValueError(f"post contains unsupported fields: {unknown}")
    mode = str(value.get("mode", "KEEP")).upper()
    if mode != "MANAGED_STACK":
        raise ValueError("Managed compositor construction requires post.mode='MANAGED_STACK'")

    bloom = _normalize_effect(
        value.get("bloom"),
        {"enabled": False, "threshold": 1.0, "strength": 0.0, "radius": 0.5},
        "bloom",
    )
    grain = _normalize_effect(
        value.get("grain"),
        {"enabled": False, "strength": 0.0, "scale": 1.0, "seed": 0},
        "grain",
    )
    vignette = _normalize_effect(
        value.get("vignette"),
        {"enabled": False, "strength": 0.0, "feather": 0.5},
        "vignette",
    )
    for field, effect in (("bloom", bloom), ("grain", grain), ("vignette", vignette)):
        if not isinstance(effect["enabled"], bool):
            raise ValueError(f"post.{field}.enabled must be a boolean")
    bloom["threshold"] = _number(bloom["threshold"], "post.bloom.threshold", 0.0, 1000.0)
    bloom["strength"] = _number(bloom["strength"], "post.bloom.strength", 0.0, 100.0)
    bloom["radius"] = _number(bloom["radius"], "post.bloom.radius", 0.0, 1.0)
    grain["strength"] = _number(grain["strength"], "post.grain.strength", 0.0, 1.0)
    grain["scale"] = _number(grain["scale"], "post.grain.scale", 0.01, 1000.0)
    grain["seed"] = _integer(grain["seed"], "post.grain.seed", 0, 2**31 - 1)
    vignette["strength"] = _number(
        vignette["strength"], "post.vignette.strength", 0.0, 1.0
    )
    vignette["feather"] = _number(
        vignette["feather"], "post.vignette.feather", 0.0, 1.0
    )
    return {"mode": mode, "bloom": bloom, "grain": grain, "vignette": vignette}


def _clear_tree(tree: Any, api: str) -> None:
    nodes = getattr(tree, "nodes", None)
    remover = getattr(nodes, "remove", None)
    if not callable(remover):
        raise CompositorCompatibilityError("Compositor nodes cannot be removed")
    for node in _iter_values(nodes):
        try:
            remover(node)
        except Exception as exc:
            raise CompositorCompatibilityError(
                f"Unable to clear managed compositor node '{getattr(node, 'name', '')}': {exc}"
            ) from exc
    if api == API_NODE_GROUP:
        interface = getattr(tree, "interface", None)
        clearer = getattr(interface, "clear", None)
        if not callable(clearer):
            raise CompositorCompatibilityError(
                "Blender 5.x compositor has no mutable node-group interface"
            )
        try:
            clearer()
            interface.new_socket(
                name="Image", in_out="OUTPUT", socket_type="NodeSocketColor"
            )
        except Exception as exc:
            raise CompositorCompatibilityError(
                f"Unable to create Blender 5.x final Image interface: {exc}"
            ) from exc


def _new_node(tree: Any, bl_idname: str, name: str, role: str) -> Any:
    creator = getattr(getattr(tree, "nodes", None), "new", None)
    if not callable(creator):
        raise CompositorCompatibilityError("Compositor nodes cannot be created")
    try:
        node = creator(bl_idname)
    except Exception as exc:
        raise CompositorCompatibilityError(
            f"Required compositor node '{bl_idname}' is unsupported: {exc}"
        ) from exc
    node.name = name
    node.label = name
    _custom_set(node, MANAGED_NODE_PROP, True)
    _custom_set(node, ROLE_PROP, role)
    return node


def _socket(sockets: Any, name: str, *, index: int | None = None) -> Any:
    values = _iter_values(sockets)
    matches = [socket for socket in values if str(getattr(socket, "name", "")) == name]
    if len(matches) == 1:
        result = matches[0]
    elif index is not None and 0 <= index < len(values):
        result = values[index]
    else:
        result = matches[0] if matches else None
    if result is None:
        raise CompositorCompatibilityError(f"Required compositor socket '{name}' is unavailable")
    return result


def _set_socket(sockets: Any, name: str, value: Any, *, index: int | None = None) -> None:
    socket = _socket(sockets, name, index=index)
    try:
        socket.default_value = value
    except Exception as exc:
        raise CompositorCompatibilityError(
            f"Unable to set compositor socket '{name}' to {value!r}: {exc}"
        ) from exc


def _link(tree: Any, output: Any, input_socket: Any) -> None:
    creator = getattr(getattr(tree, "links", None), "new", None)
    if not callable(creator):
        raise CompositorCompatibilityError("Compositor links cannot be created")
    try:
        creator(output, input_socket)
    except Exception as exc:
        raise CompositorCompatibilityError(f"Unable to link compositor nodes: {exc}") from exc


def _mark_settings(node: Any, settings: dict[str, Any]) -> None:
    raw = json.dumps(settings, allow_nan=False, sort_keys=True, separators=(",", ":"))
    _custom_set(node, SETTINGS_PROP, raw)


def _build_bloom(tree: Any, api: str, current: Any, spec: dict[str, Any]) -> Any:
    glare = _new_node(tree, "CompositorNodeGlare", "AI_LOOK_BLOOM", "BLOOM")
    _mark_settings(glare, spec)
    _link(tree, current, _socket(glare.inputs, "Image", index=0))
    if api == API_NODE_GROUP:
        _set_socket(glare.inputs, "Type", "Fog Glow")
        _set_socket(glare.inputs, "Quality", "High")
        _set_socket(glare.inputs, "Threshold", spec["threshold"])
        _set_socket(glare.inputs, "Strength", spec["strength"])
        _set_socket(glare.inputs, "Size", spec["radius"])
        return _socket(glare.outputs, "Image", index=0)

    try:
        glare.glare_type = "FOG_GLOW"
        glare.quality = "HIGH"
        glare.threshold = spec["threshold"]
        glare.size = max(6, min(9, int(round(6 + spec["radius"] * 3))))
        glare.mix = 0.0
    except Exception as exc:
        raise CompositorCompatibilityError(f"Unable to configure legacy Fog Glow: {exc}") from exc
    if spec["strength"] == 1.0:
        return _socket(glare.outputs, "Image", index=0)
    blend = _new_node(tree, "CompositorNodeMixRGB", "AI_LOOK_BLOOM_BLEND", "BLOOM_BLEND")
    _mark_settings(blend, {"strength": spec["strength"]})
    try:
        blend.blend_type = "MIX"
    except Exception as exc:
        raise CompositorCompatibilityError(f"Unable to configure bloom blend: {exc}") from exc
    amount = spec["strength"] / (1.0 + spec["strength"])
    _set_socket(blend.inputs, "Fac", amount, index=0)
    _link(tree, current, _socket(blend.inputs, "Image", index=1))
    _link(tree, _socket(glare.outputs, "Image", index=0), _socket(blend.inputs, "Image", index=2))
    return _socket(blend.outputs, "Image", index=0)


def _grain_image(tree: Any, spec: dict[str, Any]) -> Any:
    identity = json.dumps(
        {"seed": spec["seed"], "scale": spec["scale"]},
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    name = f"AI_LOOK_GRAIN_{digest[:16]}"
    images = getattr(getattr(bpy, "data", None), "images", None)
    image = _lookup(images, name)
    if image is not None:
        if not bool(_custom_get(image, MANAGED_IMAGE_PROP, False)):
            raise CompositorCompatibilityError(
                f"Image datablock '{name}' exists but is not a managed grain image"
            )
        return image
    creator = getattr(images, "new", None)
    if not callable(creator):
        raise CompositorCompatibilityError(
            "Blender cannot create the deterministic grain image required by this profile"
        )
    side = max(32, min(512, int(round(256.0 / math.sqrt(spec["scale"])))))
    try:
        image = creator(name=name, width=side, height=side, alpha=True, float_buffer=False)
    except TypeError:
        image = creator(name, side, side, alpha=True, float_buffer=False)
    except Exception as exc:
        raise CompositorCompatibilityError(
            f"Unable to create deterministic grain image '{name}': {exc}"
        ) from exc
    _custom_set(image, MANAGED_IMAGE_PROP, True)
    _custom_set(image, SETTINGS_PROP, identity)
    state = (spec["seed"] ^ int(round(spec["scale"] * 1_000_000)) ^ 0x9E3779B9) & 0xFFFFFFFF
    pixels: list[float] = []
    for _index in range(side * side):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        value = ((state >> 8) & 0xFFFFFF) / 16777215.0
        pixels.extend((value, value, value, 1.0))
    target = getattr(image, "pixels", None)
    setter = getattr(target, "foreach_set", None)
    try:
        if callable(setter):
            setter(pixels)
        else:
            target[:] = pixels
        updater = getattr(image, "update", None)
        if callable(updater):
            updater()
    except Exception as exc:
        raise CompositorCompatibilityError(
            f"Unable to populate deterministic grain image '{name}': {exc}"
        ) from exc
    _custom_set(tree, "blend_ai_look_grain_image_name", name)
    return image


def _build_grain(tree: Any, api: str, current: Any, spec: dict[str, Any]) -> Any:
    image = _grain_image(tree, spec)
    source = _new_node(tree, "CompositorNodeImage", "AI_LOOK_GRAIN_IMAGE", "GRAIN_IMAGE")
    source.image = image
    _mark_settings(source, spec)
    scale = _new_node(tree, "CompositorNodeScale", "AI_LOOK_GRAIN_SCALE", "GRAIN_SCALE")
    _link(tree, _socket(source.outputs, "Image", index=0), _socket(scale.inputs, "Image", index=0))
    if api == API_NODE_GROUP:
        _set_socket(scale.inputs, "Type", "Render Size")
        _set_socket(scale.inputs, "Frame Type", "Stretch")
        _set_socket(scale.inputs, "Interpolation", "Nearest")
        blend = _new_node(tree, "CompositorNodeAlphaOver", "AI_LOOK_GRAIN_BLEND", "GRAIN_BLEND")
        _set_socket(blend.inputs, "Factor", spec["strength"])
        _link(tree, current, _socket(blend.inputs, "Background", index=0))
        _link(
            tree,
            _socket(scale.outputs, "Image", index=0),
            _socket(blend.inputs, "Foreground", index=1),
        )
    else:
        try:
            scale.space = "RENDER_SIZE"
            scale.frame_method = "STRETCH"
        except Exception as exc:
            raise CompositorCompatibilityError(
                f"Unable to scale legacy grain image to render size: {exc}"
            ) from exc
        blend = _new_node(tree, "CompositorNodeMixRGB", "AI_LOOK_GRAIN_BLEND", "GRAIN_BLEND")
        try:
            blend.blend_type = "OVERLAY"
        except Exception as exc:
            raise CompositorCompatibilityError(
                f"Unable to configure deterministic grain blend: {exc}"
            ) from exc
        _set_socket(blend.inputs, "Fac", spec["strength"], index=0)
        _link(tree, current, _socket(blend.inputs, "Image", index=1))
        _link(
            tree,
            _socket(scale.outputs, "Image", index=0),
            _socket(blend.inputs, "Image", index=2),
        )
    _mark_settings(blend, spec)
    return _socket(blend.outputs, "Image", index=0)


def _build_vignette(
    tree: Any,
    api: str,
    current: Any,
    spec: dict[str, Any],
    scene: Any,
) -> Any:
    mask = _new_node(tree, "CompositorNodeEllipseMask", "AI_LOOK_VIGNETTE_MASK", "VIGNETTE_MASK")
    blur = _new_node(tree, "CompositorNodeBlur", "AI_LOOK_VIGNETTE_BLUR", "VIGNETTE_BLUR")
    invert = _new_node(tree, "CompositorNodeInvert", "AI_LOOK_VIGNETTE_INVERT", "VIGNETTE_INVERT")
    black = _new_node(tree, "CompositorNodeRGB", "AI_LOOK_VIGNETTE_COLOR", "VIGNETTE_COLOR")
    alpha = _new_node(tree, "CompositorNodeSetAlpha", "AI_LOOK_VIGNETTE_ALPHA", "VIGNETTE_ALPHA")
    blend = _new_node(tree, "CompositorNodeAlphaOver", "AI_LOOK_VIGNETTE_BLEND", "VIGNETTE_BLEND")
    _mark_settings(blend, spec)

    if api == API_NODE_GROUP:
        _set_socket(mask.inputs, "Operation", "Add")
        _set_socket(mask.inputs, "Value", 1.0)
        _set_socket(mask.inputs, "Position", (0.5, 0.5))
        _set_socket(mask.inputs, "Size", (0.82, 0.82))
        render = getattr(scene, "render", None)
        width = int(getattr(render, "resolution_x", 1920) or 1920)
        height = int(getattr(render, "resolution_y", 1080) or 1080)
        feather_pixels = max(1, int(round(min(width, height) * spec["feather"] * 0.25)))
        _set_socket(blur.inputs, "Size", (feather_pixels, feather_pixels))
        _set_socket(blur.inputs, "Type", "Gaussian")
        _set_socket(invert.inputs, "Factor", 1.0)
        _set_socket(blend.inputs, "Factor", spec["strength"])
        background = _socket(blend.inputs, "Background", index=0)
        foreground = _socket(blend.inputs, "Foreground", index=1)
    else:
        try:
            mask.x = 0.5
            mask.y = 0.5
            mask.width = 0.82
            mask.height = 0.82
            blur.filter_type = "GAUSS"
            blur.use_relative = True
            blur.factor_x = spec["feather"] * 0.25
            blur.factor_y = spec["feather"] * 0.25
        except Exception as exc:
            raise CompositorCompatibilityError(
                f"Unable to configure legacy vignette mask: {exc}"
            ) from exc
        _set_socket(invert.inputs, "Fac", 1.0, index=0)
        _set_socket(blend.inputs, "Fac", spec["strength"], index=0)
        background = _socket(blend.inputs, "Image", index=1)
        foreground = _socket(blend.inputs, "Image", index=2)

    try:
        _socket(black.outputs, "Color", index=0).default_value = (0.0, 0.0, 0.0, 1.0)
    except Exception as exc:
        raise CompositorCompatibilityError(f"Unable to set vignette color: {exc}") from exc
    _link(tree, _socket(mask.outputs, "Mask", index=0), _socket(blur.inputs, "Image", index=0))
    _link(tree, _socket(blur.outputs, "Image", index=0), _socket(invert.inputs, "Color", index=1))
    _link(tree, _socket(black.outputs, "Color", index=0), _socket(alpha.inputs, "Image", index=0))
    _link(tree, _socket(invert.outputs, "Color", index=0), _socket(alpha.inputs, "Alpha", index=1))
    _link(tree, current, background)
    _link(tree, _socket(alpha.outputs, "Image", index=0), foreground)
    return _socket(blend.outputs, "Image", index=0)


def build_managed_post_stack(
    scene: Any,
    post_spec: dict[str, Any],
    *,
    tree: Any | None = None,
) -> dict[str, Any]:
    """Replace a unique managed tree with a deterministic full post stack."""

    api, tree = _require_managed_tree(scene, tree)
    spec = _normalize_post_spec(post_spec)
    _clear_tree(tree, api)

    render_layers = _new_node(
        tree, "CompositorNodeRLayers", "AI_LOOK_RENDER_LAYERS", "RENDER_LAYERS"
    )
    if hasattr(render_layers, "scene"):
        render_layers.scene = scene
    current = _socket(render_layers.outputs, "Image", index=0)

    if spec["bloom"]["enabled"]:
        current = _build_bloom(tree, api, current, spec["bloom"])
    if spec["grain"]["enabled"]:
        current = _build_grain(tree, api, current, spec["grain"])
    if spec["vignette"]["enabled"]:
        current = _build_vignette(tree, api, current, spec["vignette"], scene)

    if api == API_NODE_GROUP:
        output = _new_node(tree, "NodeGroupOutput", "AI_LOOK_OUTPUT", "FINAL_OUTPUT")
        if hasattr(output, "is_active_output"):
            output.is_active_output = True
        final_input = _socket(output.inputs, "Image", index=0)
    else:
        output = _new_node(
            tree, "CompositorNodeComposite", "AI_LOOK_OUTPUT", "FINAL_OUTPUT"
        )
        if hasattr(output, "is_active_output"):
            output.is_active_output = True
        final_input = _socket(output.inputs, "Image", index=0)
    _link(tree, current, final_input)
    _custom_set(
        tree,
        SETTINGS_PROP,
        json.dumps(spec, allow_nan=False, sort_keys=True, separators=(",", ":")),
    )
    validate_final_image_linked(scene, tree=tree)
    return {
        "api": api,
        "tree_name": str(getattr(tree, "name", "")),
        "node_names": sorted(str(getattr(node, "name", "")) for node in _iter_values(tree.nodes)),
        "compositor_hash": compositor_hash(scene, tree=tree),
    }


def _interface_output_items(tree: Any) -> list[Any]:
    interface = getattr(tree, "interface", None)
    return [
        item
        for item in _iter_values(getattr(interface, "items_tree", None))
        if str(getattr(item, "item_type", "SOCKET")) == "SOCKET"
        and str(getattr(item, "in_out", "")) == "OUTPUT"
        and str(getattr(item, "name", "")) == "Image"
        and str(getattr(item, "socket_type", "NodeSocketColor")) == "NodeSocketColor"
    ]


def authoritative_final_output(
    scene: Any, *, tree: Any | None = None
) -> tuple[Any, Any]:
    """Return the one node/socket whose Image is the post-composited result."""

    api = detect_compositor_api(scene)
    tree = tree if tree is not None else get_compositor_tree(scene)
    if tree is None:
        raise CompositorCompatibilityError("Scene has no compositor tree")
    nodes = _iter_values(getattr(tree, "nodes", None))
    if api == API_NODE_GROUP:
        interface_outputs = _interface_output_items(tree)
        if len(interface_outputs) != 1:
            raise CompositorCompatibilityError(
                "Blender 5.x compositor must expose exactly one Image output interface socket"
            )
        candidates = [node for node in nodes if _node_type(node) == "NodeGroupOutput"]
        active = [node for node in candidates if bool(getattr(node, "is_active_output", False))]
        if len(active) != 1:
            raise CompositorCompatibilityError(
                "Blender 5.x compositor must have exactly one active NodeGroupOutput"
            )
        node = active[0]
    else:
        candidates = [
            node
            for node in nodes
            if _node_type(node) == "CompositorNodeComposite"
            and not bool(getattr(node, "mute", False))
        ]
        explicit_active = [
            node for node in candidates if bool(getattr(node, "is_active_output", False))
        ]
        if explicit_active:
            candidates = explicit_active
        if len(candidates) != 1:
            raise CompositorCompatibilityError(
                "Blender 4.2 compositor must have exactly one active CompositorNodeComposite"
            )
        node = candidates[0]
    return node, _socket(node.inputs, "Image", index=0)


def validate_final_image_linked(scene: Any, *, tree: Any | None = None) -> dict[str, str]:
    """Validate that the authoritative final Image socket has an incoming link."""

    tree = tree if tree is not None else get_compositor_tree(scene)
    node, socket = authoritative_final_output(scene, tree=tree)
    linked = bool(getattr(socket, "is_linked", False))
    if not linked:
        linked = any(link.to_socket is socket for link in _iter_values(getattr(tree, "links", None)))
    if not linked:
        raise CompositorCompatibilityError(
            f"Authoritative compositor output '{getattr(node, 'name', '')}.Image' is unlinked"
        )
    return {
        "api": detect_compositor_api(scene),
        "node_name": str(getattr(node, "name", "")),
        "socket_name": str(getattr(socket, "name", "Image")),
    }


def _stable_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return format(value, ".17g") if math.isfinite(value) else str(value)
    if isinstance(value, (list, tuple)):
        return [_stable_value(item) for item in value]
    if hasattr(value, "name"):
        return str(getattr(value, "name", ""))
    try:
        return [_stable_value(item) for item in value]
    except (TypeError, ReferenceError):
        return str(value)


def _socket_index(sockets: Any, target: Any) -> int:
    for index, socket in enumerate(_iter_values(sockets)):
        if socket is target:
            return index
    return -1


def _node_payload(node: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": _node_type(node),
        "name": str(getattr(node, "name", "")),
        "label": str(getattr(node, "label", "")),
        "mute": bool(getattr(node, "mute", False)),
        "inputs": [],
        "properties": {},
    }
    for index, socket in enumerate(_iter_values(getattr(node, "inputs", None))):
        payload["inputs"].append(
            {
                "index": index,
                "name": str(getattr(socket, "name", "")),
                "default": _stable_value(getattr(socket, "default_value", None)),
            }
        )
    for attribute in _SEMANTIC_NODE_ATTRIBUTES:
        if hasattr(node, attribute):
            payload["properties"][attribute] = _stable_value(getattr(node, attribute))
    for key in (MANAGED_NODE_PROP, ROLE_PROP, SETTINGS_PROP):
        value = _custom_get(node, key, None)
        if value is not None:
            payload["properties"][key] = _stable_value(value)
    return payload


def _interface_payload(tree: Any) -> list[dict[str, str]]:
    interface = getattr(tree, "interface", None)
    result: list[dict[str, str]] = []
    for item in _iter_values(getattr(interface, "items_tree", None)):
        if str(getattr(item, "item_type", "SOCKET")) != "SOCKET":
            continue
        result.append(
            {
                "name": str(getattr(item, "name", "")),
                "in_out": str(getattr(item, "in_out", "")),
                "socket_type": str(getattr(item, "socket_type", "")),
            }
        )
    return sorted(
        result,
        key=lambda item: (item["in_out"], item["name"], item["socket_type"]),
    )


def _tree_links_payload(tree: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for link in _iter_values(getattr(tree, "links", None)):
        from_socket = getattr(link, "from_socket", None)
        to_socket = getattr(link, "to_socket", None)
        from_node = getattr(link, "from_node", getattr(from_socket, "node", None))
        to_node = getattr(link, "to_node", getattr(to_socket, "node", None))
        result.append(
            {
                "from_node": str(getattr(from_node, "name", "")),
                "from_socket": str(getattr(from_socket, "name", "")),
                "from_index": _socket_index(
                    getattr(from_node, "outputs", None),
                    from_socket,
                ),
                "to_node": str(getattr(to_node, "name", "")),
                "to_socket": str(getattr(to_socket, "name", "")),
                "to_index": _socket_index(
                    getattr(to_node, "inputs", None),
                    to_socket,
                ),
            }
        )
    result.sort(key=lambda item: tuple(str(value) for value in item.values()))
    return result


def compositor_hash(scene: Any, *, tree: Any | None = None) -> str:
    """Hash visual compositor topology and values, excluding output destinations."""

    api = detect_compositor_api(scene)
    tree = tree if tree is not None else get_compositor_tree(scene)
    if tree is None:
        raise CompositorCompatibilityError("Scene has no compositor tree to hash")
    graph = _walk_compositor_trees(tree)
    tree_ids = {
        _identity(item): f"tree_{index:04d}"
        for index, (item, _path) in enumerate(graph)
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "api": api,
        "root_tree": tree_ids[_identity(tree)],
        "trees": [],
    }
    for nested_tree, path in graph:
        nodes = []
        for node in _sorted_nodes(nested_tree):
            node_payload = _node_payload(node)
            child = _group_child(node)
            if child is not None:
                node_payload["group_tree"] = tree_ids[_identity(child)]
            nodes.append(node_payload)
        payload["trees"].append(
            {
                "tree_id": tree_ids[_identity(nested_tree)],
                "discovery_path": list(path),
                "tree_markers": {
                    key: _stable_value(_custom_get(nested_tree, key, None))
                    for key in (
                        MANAGED_TREE_PROP,
                        MANAGED_NESTED_TREE_PROP,
                        MANAGED_PARENT_PROP,
                        MANAGED_NAME_PROP,
                        SCHEMA_PROP,
                        API_PROP,
                        SETTINGS_PROP,
                    )
                },
                "interface": _interface_payload(nested_tree),
                "nodes": nodes,
                "links": _tree_links_payload(nested_tree),
            }
        )
    raw = json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _snapshot_format(settings: Any) -> dict[str, Any]:
    if settings is None:
        return {}
    return {
        attribute: getattr(settings, attribute)
        for attribute in _FORMAT_ATTRIBUTES
        if hasattr(settings, attribute)
    }


def _restore_format(settings: Any, snapshot: dict[str, Any]) -> None:
    if settings is None:
        if snapshot:
            raise CompositorCompatibilityError("File Output format settings disappeared")
        return
    for attribute, value in snapshot.items():
        if not hasattr(settings, attribute):
            raise CompositorCompatibilityError(
                f"File Output format setting '{attribute}' disappeared"
            )
        if getattr(settings, attribute) != value:
            setattr(settings, attribute, value)


def _snapshot_item(item: Any) -> dict[str, Any]:
    return {
        "attributes": {
            attribute: getattr(item, attribute)
            for attribute in _ITEM_STATE_ATTRIBUTES
            if hasattr(item, attribute)
        },
        "format": _snapshot_format(getattr(item, "format", None)),
    }


def _restore_attributes(value: Any, snapshot: dict[str, Any]) -> None:
    for attribute, original in snapshot.items():
        if not hasattr(value, attribute):
            raise CompositorCompatibilityError(
                f"File Output setting '{attribute}' disappeared"
            )
        if getattr(value, attribute) != original:
            setattr(value, attribute, original)


def snapshot_file_outputs(tree: Any) -> list[dict[str, Any]]:
    """Capture every path-bearing File Output setting for later restoration."""

    snapshots: list[dict[str, Any]] = []
    for nested_tree, path in _walk_compositor_trees(tree):
        for node in _sorted_nodes(nested_tree):
            if _node_type(node) != "CompositorNodeOutputFile":
                continue
            variant = "NODE_GROUP_5X" if hasattr(node, "directory") else "LEGACY_4X"
            node_attributes = {
                attribute: getattr(node, attribute)
                for attribute in ("directory", "file_name", "base_path")
                if hasattr(node, attribute)
            }
            items = (
                getattr(node, "file_output_items", None)
                if variant == "NODE_GROUP_5X"
                else getattr(node, "file_slots", None)
            )
            snapshots.append(
                {
                    "tree_path": list(path),
                    "node_name": str(getattr(node, "name", "")),
                    "variant": variant,
                    "attributes": node_attributes,
                    # Retain the original top-level fields for callers that
                    # inspect snapshots created by older adapter versions.
                    "directory": node_attributes.get("directory"),
                    "file_name": node_attributes.get("file_name"),
                    "base_path": node_attributes.get("base_path"),
                    "format": _snapshot_format(getattr(node, "format", None)),
                    "items": [_snapshot_item(item) for item in _iter_values(items)],
                }
            )
    return snapshots


def _snapshot_node(tree: Any, snapshot: dict[str, Any]) -> Any:
    nested_tree = _resolve_tree_path(tree, snapshot.get("tree_path", []))
    node = _lookup(getattr(nested_tree, "nodes", None), snapshot.get("node_name", ""))
    if node is None or _node_type(node) != "CompositorNodeOutputFile":
        location = "/".join(snapshot.get("tree_path", [])) or "<root>"
        raise CompositorCompatibilityError(
            f"File Output node '{snapshot.get('node_name', '')}' is missing at {location}"
        )
    return node


def _redirect_target(root: str, snapshot: dict[str, Any]) -> str:
    parts = [*snapshot.get("tree_path", []), str(snapshot.get("node_name", ""))]
    identity = "/".join(parts)
    label = _safe_path_token("__".join(parts))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
    target = os.path.normpath(os.path.join(root, f"{label}_{digest}"))
    if os.path.commonpath((root, target)) != root:
        raise CompositorCompatibilityError(
            f"Redirected File Output path escaped declared root '{root}'"
        )
    return target


def _sanitize_item_paths(items: list[Any], node_token: str) -> None:
    for index, item in enumerate(items):
        for attribute in ("path", "file_name"):
            if not hasattr(item, attribute):
                continue
            current = getattr(item, attribute)
            safe = _safe_output_pattern(current, f"{node_token}_{index:03d}_")
            setattr(item, attribute, f"{index:03d}_{safe}"[:128])


def redirect_file_outputs(tree: Any, directory: str) -> list[dict[str, Any]]:
    """Redirect all File Output writes beneath one declared absolute directory."""

    if not isinstance(directory, str) or not directory or not os.path.isabs(directory):
        raise ValueError("File Output redirect directory must be an absolute path")
    root = os.path.normpath(directory)
    snapshots = snapshot_file_outputs(tree)
    for snapshot in snapshots:
        node = _snapshot_node(tree, snapshot)
        target = _redirect_target(root, snapshot)
        node_token = _safe_path_token(str(snapshot["node_name"]))
        try:
            if snapshot["variant"] == "NODE_GROUP_5X":
                if not hasattr(node, "file_name") or not hasattr(node, "file_output_items"):
                    raise AttributeError("missing file_name/file_output_items")
                node.directory = target
                node.file_name = _safe_output_pattern(
                    node.file_name,
                    f"{node_token}_",
                )
                _sanitize_item_paths(
                    _iter_values(node.file_output_items),
                    node_token,
                )
            else:
                if not hasattr(node, "file_slots"):
                    raise AttributeError("missing file_slots")
                node.base_path = target
                _sanitize_item_paths(_iter_values(node.file_slots), node_token)
        except Exception as exc:
            try:
                restore_file_outputs(tree, snapshots)
            except Exception:
                pass
            raise CompositorCompatibilityError(
                f"Unable to redirect File Output node '{snapshot['node_name']}': {exc}"
            ) from exc
    return snapshots


def restore_file_outputs(tree: Any, snapshots: list[dict[str, Any]]) -> None:
    """Restore a snapshot returned by :func:`redirect_file_outputs`."""

    if not isinstance(snapshots, list):
        raise ValueError("File Output snapshot must be a list")
    errors: list[str] = []
    for snapshot in snapshots:
        node_name = str(snapshot.get("node_name", ""))
        try:
            node = _snapshot_node(tree, snapshot)
            if snapshot.get("variant") == "NODE_GROUP_5X":
                items = _iter_values(getattr(node, "file_output_items", None))
            else:
                items = _iter_values(getattr(node, "file_slots", None))
            attributes = snapshot.get("attributes")
            if not isinstance(attributes, dict):
                attributes = {
                    attribute: snapshot.get(attribute)
                    for attribute in ("directory", "file_name", "base_path")
                    if attribute in snapshot and snapshot.get(attribute) is not None
                }
            _restore_attributes(node, attributes)
            _restore_format(getattr(node, "format", None), snapshot.get("format", {}))
            expected_items = snapshot.get("items", [])
            if len(items) != len(expected_items):
                raise CompositorCompatibilityError(
                    f"item count changed from {len(expected_items)} to {len(items)}"
                )
            for item, item_snapshot in zip(items, expected_items):
                item_attributes = item_snapshot.get("attributes")
                if not isinstance(item_attributes, dict):
                    item_attributes = {
                        attribute: item_snapshot.get(attribute)
                        for attribute in _ITEM_STATE_ATTRIBUTES
                        if attribute in item_snapshot and item_snapshot.get(attribute) is not None
                    }
                _restore_attributes(item, item_attributes)
                _restore_format(getattr(item, "format", None), item_snapshot.get("format", {}))
        except Exception as exc:
            errors.append(f"File Output node '{node_name}': {exc}")
    if errors:
        raise CompositorCompatibilityError("; ".join(errors))


__all__ = [
    "API_LEGACY",
    "API_NODE_GROUP",
    "CompositorCompatibilityError",
    "MANAGED_NESTED_TREE_PROP",
    "MANAGED_PARENT_PROP",
    "authoritative_final_output",
    "build_managed_post_stack",
    "compositor_hash",
    "detect_compositor_api",
    "ensure_unique_managed_compositor",
    "get_compositor_tree",
    "owned_nested_compositor_trees",
    "redirect_file_outputs",
    "restore_file_outputs",
    "snapshot_file_outputs",
    "validate_final_image_linked",
]
