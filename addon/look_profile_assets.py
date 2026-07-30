"""Managed, profile-local Blender assets for compiled look Scenes.

The look-profile compiler owns transaction boundaries.  This module owns the
datablocks placed inside that transaction: lights, optional camera clones,
bounded fog, deterministic rain geometry, and typed World/render/color state.
It never mutates a shared mesh, material, light, camera-data, or Collection.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import bpy


SCHEMA_VERSION = 1
MANAGED_PROP = "blend_ai_managed"
PROFILE_ID_PROP = "blend_ai_profile_id"
COMPONENT_ID_PROP = "blend_ai_component_id"
SCHEMA_PROP = "blend_ai_schema_version"
CONTENT_HASH_PROP = "blend_ai_content_hash"
ROLE_PROP = "blend_ai_role"

MAX_RAIN_DROPS = 5000


def _values(collection: Any) -> list[Any]:
    try:
        return list(collection or [])
    except (TypeError, ReferenceError):
        return []


def _get(collection: Any, name: str) -> Any | None:
    getter = getattr(collection, "get", None)
    if callable(getter):
        try:
            return getter(name)
        except (TypeError, ReferenceError):
            pass
    for value in _values(collection):
        if str(getattr(value, "name", "")) == name:
            return value
    return None


def _custom_get(value: Any, key: str, default: Any = None) -> Any:
    getter = getattr(value, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except (TypeError, ReferenceError):
            pass
    return default


def _custom_set(value: Any, key: str, item: Any) -> None:
    try:
        value[key] = item
    except Exception as exc:
        raise RuntimeError(f"Unable to mark managed look datablock: {exc}") from exc


def _content_hash(value: Any) -> str:
    raw = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def mark_managed(
    value: Any,
    *,
    profile_id: str,
    component_id: str,
    role: str,
    definition: Any,
) -> None:
    """Apply the portable managed-resource marker contract to one ID."""
    for key, item in (
        (MANAGED_PROP, True),
        (PROFILE_ID_PROP, profile_id),
        (COMPONENT_ID_PROP, component_id),
        (SCHEMA_PROP, SCHEMA_VERSION),
        (CONTENT_HASH_PROP, _content_hash(definition)),
        (ROLE_PROP, role),
        # Keep the compiler's original marker readable during migration.
        ("blend_ai_look_managed", True),
        ("blend_ai_look_profile_id", profile_id),
    ):
        _custom_set(value, key, item)


def _safe_datablock_name(
    profile_id: str,
    component_id: str,
    role: str,
    *,
    name_namespace: str | None = None,
) -> str:
    """Return a deterministic managed-ID name, optionally scoped to one asset set.

    Omitting ``name_namespace`` preserves the original naming algorithm for
    callers that build only one asset set per profile.  Immutable compilers can
    provide a version/revision-specific namespace so otherwise identical
    component IDs coexist without changing their portable managed markers.
    """
    readable_profile = "".join(
        character if character.isalnum() else "_" for character in profile_id
    ).strip("_")[:18]
    readable_component = "".join(
        character if character.isalnum() else "_" for character in component_id
    ).strip("_")[:18]
    identity = f"{profile_id}\0{component_id}\0{role}"
    if name_namespace is not None:
        identity = f"{identity}\0{name_namespace}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8]
    return f"AI_{role}_{readable_profile}_{readable_component}_{digest}"[:63]


def _remove(owner: Any, value: Any) -> None:
    if owner is None or value is None:
        return
    remover = getattr(owner, "remove", None)
    if not callable(remover):
        return
    try:
        remover(value, do_unlink=True)
    except TypeError:
        remover(value)


def _id_token(value: Any) -> tuple[str, int]:
    """Return a stable-enough identity token for one live Blender ID wrapper."""
    as_pointer = getattr(value, "as_pointer", None)
    if callable(as_pointer):
        try:
            pointer = int(as_pointer())
            if pointer:
                return ("POINTER", pointer)
        except (ReferenceError, TypeError, ValueError):
            pass
    return ("PYTHON", id(value))


def cleanup_created(created: list[tuple[str, Any]]) -> list[str]:
    """Remove transaction-local IDs in reverse dependency order."""
    data = getattr(bpy, "data", None)
    owners = {
        "OBJECT": getattr(data, "objects", None),
        "LIGHT": getattr(data, "lights", None),
        "CAMERA": getattr(data, "cameras", None),
        "CURVE": getattr(data, "curves", None),
        "MESH": getattr(data, "meshes", None),
        "MATERIAL": getattr(data, "materials", None),
        "NODE_GROUP": getattr(data, "node_groups", None),
        "IMAGE": getattr(data, "images", None),
        "COLLECTION": getattr(data, "collections", None),
    }
    errors: list[str] = []
    for kind, value in reversed(created):
        try:
            _remove(owners.get(kind), value)
        except Exception as exc:
            errors.append(f"{kind} {getattr(value, 'name', '<unknown>')}: {exc}")
    return errors


def _node_input(node: Any, *names: str) -> Any | None:
    inputs = getattr(node, "inputs", None)
    for name in names:
        socket = _get(inputs, name)
        if socket is not None:
            return socket
    return None


def configure_world(
    world: Any,
    spec: dict[str, Any],
    *,
    profile_id: str,
    created_sink: list[tuple[str, Any]] | None = None,
) -> dict[str, Any]:
    """Configure a compiler-owned World from one typed profile specification."""
    mode = str(spec.get("mode", "KEEP")).upper()
    if mode in {"KEEP", "COPY_SOURCE"}:
        mark_managed(
            world,
            profile_id=profile_id,
            component_id="world",
            role="WORLD",
            definition=spec,
        )
        return {"mode": "KEEP", "world_name": str(getattr(world, "name", ""))}

    if mode not in {"MANAGED_SOLID", "SOLID", "MANAGED_SKY", "MANAGED_HDRI"}:
        raise ValueError(f"Unsupported managed World mode '{mode}'")
    hdri_path = spec.get("hdri_path")
    if mode == "MANAGED_HDRI" and (not isinstance(hdri_path, str) or not hdri_path):
        raise ValueError("MANAGED_HDRI requires hdri_path")
    world.use_nodes = True
    tree = getattr(world, "node_tree", None)
    if tree is None:
        raise RuntimeError("Managed World has no node tree")
    helper_created: list[tuple[str, Any]] = []
    try:
        tree.nodes.clear()

        background = tree.nodes.new(type="ShaderNodeBackground")
        output = tree.nodes.new(type="ShaderNodeOutputWorld")
        strength = float(spec.get("strength", 0.0))
        background.inputs["Strength"].default_value = strength
        tree.links.new(background.outputs["Background"], output.inputs["Surface"])

        if mode in {"MANAGED_SOLID", "SOLID"}:
            color = tuple(float(value) for value in spec.get("color_rgb", (0.0, 0.0, 0.0)))
            background.inputs["Color"].default_value = (*color, 1.0)
            if hasattr(world, "color"):
                world.color = color
        elif mode == "MANAGED_SKY":
            sky = tree.nodes.new(type="ShaderNodeTexSky")
            for attribute, value in (
                ("sun_elevation", math.radians(float(spec.get("sun_elevation_degrees", 35.0)))),
                ("sun_rotation", math.radians(float(spec.get("sun_rotation_degrees", 0.0)))),
                ("altitude", float(spec.get("altitude_m", 0.0))),
                ("air_density", float(spec.get("air_density", 1.0))),
                ("dust_density", float(spec.get("dust_density", 1.0))),
                ("ozone_density", float(spec.get("ozone_density", 1.0))),
            ):
                if hasattr(sky, attribute):
                    setattr(sky, attribute, value)
            tree.links.new(sky.outputs["Color"], background.inputs["Color"])
        else:
            images = getattr(getattr(bpy, "data", None), "images", None)
            before_images = {_id_token(image) for image in _values(images)}
            environment = tree.nodes.new(type="ShaderNodeTexEnvironment")
            loaded_image = images.load(hdri_path, check_existing=True)
            if _id_token(loaded_image) not in before_images:
                helper_created.append(("IMAGE", loaded_image))
            environment.image = loaded_image
            texcoord = tree.nodes.new(type="ShaderNodeTexCoord")
            mapping = tree.nodes.new(type="ShaderNodeMapping")
            rotation = _node_input(mapping, "Rotation")
            if rotation is not None:
                rotation.default_value[2] = math.radians(float(spec.get("rotation_degrees", 0.0)))
            vector_output = _get(getattr(texcoord, "outputs", None), "Generated") or _get(
                getattr(texcoord, "outputs", None), "Normal"
            )
            if vector_output is None:
                raise RuntimeError("World Texture Coordinate node has no usable vector output")
            tree.links.new(vector_output, mapping.inputs["Vector"])
            tree.links.new(mapping.outputs["Vector"], environment.inputs["Vector"])
            tree.links.new(environment.outputs["Color"], background.inputs["Color"])

        mark_managed(
            world,
            profile_id=profile_id,
            component_id="world",
            role="WORLD",
            definition=spec,
        )
    except Exception as exc:
        errors = cleanup_created(helper_created)
        if errors:
            raise RuntimeError(
                f"Managed World configuration failed ({exc}); cleanup also failed: "
                + "; ".join(errors)
            ) from exc
        raise
    if created_sink is not None:
        created_sink.extend(helper_created)
    return {"mode": mode, "world_name": str(getattr(world, "name", ""))}


def _walk_layer_collections(layer_collection: Any):
    if layer_collection is None:
        return
    yield layer_collection
    for child in _values(getattr(layer_collection, "children", None)):
        yield from _walk_layer_collections(child)


def _collection_light_only(
    collection: Any,
    *,
    visited: set[int] | None = None,
) -> tuple[bool, str | None]:
    """Return whether excluding a Collection can hide only artist lights."""
    if visited is None:
        visited = set()
    token = id(collection)
    if token in visited:
        return True, None
    visited.add(token)
    for member in _values(getattr(collection, "objects", None)):
        name = str(getattr(member, "name", ""))
        if str(getattr(member, "type", "")) != "LIGHT":
            return False, f"contains non-light object '{name}'"
        if bool(_custom_get(member, MANAGED_PROP, False)):
            return False, f"contains managed light '{name}'"
    for child in _values(getattr(collection, "children", None)):
        safe, reason = _collection_light_only(child, visited=visited)
        if not safe:
            return False, f"child Collection '{getattr(child, 'name', '')}' {reason}"
    return True, None


def _restore_layer_exclusions(changes: list[tuple[Any, bool]]) -> list[str]:
    errors: list[str] = []
    for layer_collection, previous in reversed(changes):
        try:
            layer_collection.exclude = previous
        except Exception as exc:
            errors.append(
                f"{getattr(getattr(layer_collection, 'collection', None), 'name', '')}: {exc}"
            )
    return errors


def _plan_existing_light_isolation(
    scene: Any,
    policy: str,
) -> tuple[dict[int, Any], dict[int, Any]]:
    """Plan safe light-only LayerCollection exclusions without mutating Blender."""
    if policy == "KEEP":
        return {}, {}
    if policy != "MUTE_NON_MANAGED":
        raise ValueError(f"Unsupported existing-light policy '{policy}'")

    targets = [
        obj
        for obj in _values(getattr(scene, "objects", None))
        if str(getattr(obj, "type", "")) == "LIGHT"
        and not bool(_custom_get(obj, MANAGED_PROP, False))
        and not bool(getattr(obj, "hide_render", False))
    ]
    if not targets:
        return {}, {}

    view_layers = _values(getattr(scene, "view_layers", None))
    if not view_layers:
        raise RuntimeError("MUTE_NON_MANAGED requires at least one View Layer")

    target_tokens = {_id_token(obj): obj for obj in targets}
    planned_nodes: dict[int, Any] = {}
    planned_collections: dict[int, Any] = {}
    found_paths: set[tuple[str, int]] = set()
    for view_index, view_layer in enumerate(view_layers):
        root = getattr(view_layer, "layer_collection", None)
        if root is None:
            raise RuntimeError(
                f"View Layer '{getattr(view_layer, 'name', view_index)}' has no LayerCollection"
            )
        for layer_collection in _walk_layer_collections(root):
            collection = getattr(layer_collection, "collection", None)
            if collection is None:
                continue
            direct_target_tokens = {
                _id_token(member)
                for member in _values(getattr(collection, "objects", None))
                if _id_token(member) in target_tokens
            }
            if not direct_target_tokens:
                continue
            safe, reason = _collection_light_only(collection)
            if not safe:
                light_names = ", ".join(
                    sorted(
                        str(getattr(target_tokens[token], "name", ""))
                        for token in direct_target_tokens
                    )
                )
                raise ValueError(
                    f"Shared light path for {light_names} uses Collection "
                    f"'{getattr(collection, 'name', '')}', which {reason}; "
                    "MUTE_NON_MANAGED would hide non-profile content"
                )
            for target_token in direct_target_tokens:
                found_paths.add((target_token[0], target_token[1]))
            planned_nodes[id(layer_collection)] = layer_collection
            planned_collections[id(collection)] = collection

    missing = [
        str(getattr(obj, "name", ""))
        for token, obj in target_tokens.items()
        if token not in found_paths
    ]
    if missing:
        raise RuntimeError(
            "No safely excludable View Layer membership was found for shared light(s): "
            + ", ".join(sorted(missing))
        )
    return planned_nodes, planned_collections


def validate_existing_light_isolation(scene: Any, policy: str) -> list[str]:
    """Validate MUTE_NON_MANAGED and return its Collection plan without mutation."""
    _planned_nodes, planned_collections = _plan_existing_light_isolation(scene, policy)
    return sorted(
        str(getattr(collection, "name", "")) for collection in planned_collections.values()
    )


def _plan_profile_light_isolation(
    scene: Any,
    policy: str,
) -> tuple[
    dict[int, Any],
    dict[int, Any],
    dict[int, Any],
    dict[int, Any],
    dict[int, tuple[Any, Any]],
]:
    """Plan profile-local isolation, cloning mixed Collections when necessary.

    A compiled look Scene may safely exclude a shared mixed Collection from its
    own View Layers when every non-light member is relinked into a new,
    compiler-owned Collection.  This keeps shared Objects, meshes, materials,
    and the source Collection untouched while avoiding global visibility edits.
    """
    if policy == "KEEP":
        return {}, {}, {}, {}, {}
    if policy != "MUTE_NON_MANAGED":
        raise ValueError(f"Unsupported existing-light policy '{policy}'")

    targets = [
        obj
        for obj in _values(getattr(scene, "objects", None))
        if str(getattr(obj, "type", "")) == "LIGHT"
        and not bool(_custom_get(obj, MANAGED_PROP, False))
        and not bool(getattr(obj, "hide_render", False))
    ]
    if not targets:
        return {}, {}, {}, {}, {}

    target_tokens = {_id_token(obj): obj for obj in targets}
    safe_nodes: dict[int, Any] = {}
    safe_collections: dict[int, Any] = {}
    clone_nodes: dict[int, Any] = {}
    clone_collections: dict[int, Any] = {}
    root_unlinks: dict[int, tuple[Any, Any]] = {}
    found_paths: set[tuple[str, int]] = set()
    for view_index, view_layer in enumerate(_values(getattr(scene, "view_layers", None))):
        root = getattr(view_layer, "layer_collection", None)
        if root is None:
            raise RuntimeError(
                f"View Layer '{getattr(view_layer, 'name', view_index)}' has no LayerCollection"
            )
        for layer_collection in _walk_layer_collections(root):
            collection = getattr(layer_collection, "collection", None)
            if collection is None:
                continue
            direct_target_tokens = {
                _id_token(member)
                for member in _values(getattr(collection, "objects", None))
                if _id_token(member) in target_tokens
            }
            if not direct_target_tokens:
                continue
            safe, _reason = _collection_light_only(collection)
            if safe:
                safe_nodes[id(layer_collection)] = layer_collection
                safe_collections[id(collection)] = collection
            else:
                if layer_collection is root:
                    # A copied Scene owns a distinct master Collection even
                    # though its linked Objects are shared with the source.
                    # Unlinking a direct artist light from that copied root is
                    # therefore profile-local and does not hide or move any
                    # non-light content in the source Scene.
                    for target_token in direct_target_tokens:
                        root_unlinks[id(target_tokens[target_token])] = (
                            collection,
                            target_tokens[target_token],
                        )
                else:
                    clone_nodes[id(layer_collection)] = layer_collection
                    clone_collections[id(collection)] = collection
            found_paths.update(direct_target_tokens)

    missing = [
        str(getattr(obj, "name", ""))
        for token, obj in target_tokens.items()
        if token not in found_paths
    ]
    if missing:
        raise RuntimeError(
            "No profile-local View Layer membership was found for shared light(s): "
            + ", ".join(sorted(missing))
        )
    return safe_nodes, safe_collections, clone_nodes, clone_collections, root_unlinks


def validate_profile_light_isolation(scene: Any, policy: str) -> dict[str, list[str]]:
    """Validate Scene-local light isolation without changing Blender state."""
    _safe_nodes, safe_collections, _clone_nodes, clone_collections, root_unlinks = (
        _plan_profile_light_isolation(scene, policy)
    )
    result = {
        "excluded": sorted(str(getattr(value, "name", "")) for value in safe_collections.values()),
        "cloned": sorted(str(getattr(value, "name", "")) for value in clone_collections.values()),
    }
    if root_unlinks:
        result["unlinked"] = sorted(
            str(getattr(obj, "name", "")) for _collection, obj in root_unlinks.values()
        )
    return result


def _clone_collection_without_artist_lights(
    source: Any,
    parent: Any,
    *,
    profile_id: str,
    name_namespace: str,
    created: list[tuple[str, Any]],
    component_path: str,
) -> Any:
    component_id = f"source-isolation-{component_path}"
    name = _safe_datablock_name(
        profile_id,
        component_id,
        "COLLECTION",
        name_namespace=name_namespace,
    )
    clone = bpy.data.collections.new(name=name)
    created.append(("COLLECTION", clone))
    parent.children.link(clone)
    mark_managed(
        clone,
        profile_id=profile_id,
        component_id=component_id,
        role="SOURCE_ISOLATION_COLLECTION",
        definition={"source_collection": str(getattr(source, "name", ""))},
    )
    for obj in _values(getattr(source, "objects", None)):
        if str(getattr(obj, "type", "")) == "LIGHT" and not bool(
            _custom_get(obj, MANAGED_PROP, False)
        ):
            continue
        clone.objects.link(obj)
    for index, child in enumerate(_values(getattr(source, "children", None))):
        _clone_collection_without_artist_lights(
            child,
            clone,
            profile_id=profile_id,
            name_namespace=name_namespace,
            created=created,
            component_path=f"{component_path}-{index}",
        )
    return clone


def isolate_profile_lights(
    scene: Any,
    policy: str,
    destination: Any,
    *,
    profile_id: str,
    name_namespace: str,
    mutation_sink: list[tuple[Any, bool]] | None = None,
    created_sink: list[tuple[str, Any]] | None = None,
) -> dict[str, list[str]]:
    """Apply profile-local Collection exclusion plus filtered linked clones."""
    safe_nodes, safe_collections, clone_nodes, clone_collections, root_unlinks = (
        _plan_profile_light_isolation(scene, policy)
    )
    changes: list[tuple[Any, bool]] = []
    created: list[tuple[str, Any]] = []
    unlinked: list[tuple[Any, Any]] = []
    try:
        for index, collection in enumerate(clone_collections.values()):
            _clone_collection_without_artist_lights(
                collection,
                destination,
                profile_id=profile_id,
                name_namespace=name_namespace,
                created=created,
                component_path=str(index),
            )
        for collection, obj in root_unlinks.values():
            unlinker = getattr(getattr(collection, "objects", None), "unlink", None)
            if not callable(unlinker):
                raise RuntimeError(
                    f"Compiled Scene root Collection cannot unlink light '{getattr(obj, 'name', '')}'"
                )
            unlinker(obj)
            unlinked.append((collection, obj))
        for layer_collection in [*safe_nodes.values(), *clone_nodes.values()]:
            previous = bool(getattr(layer_collection, "exclude", False))
            changes.append((layer_collection, previous))
            layer_collection.exclude = True
    except Exception as exc:
        relink_errors: list[str] = []
        for collection, obj in reversed(unlinked):
            linker = getattr(getattr(collection, "objects", None), "link", None)
            try:
                if not callable(linker):
                    raise RuntimeError("link is unavailable")
                linker(obj)
            except Exception as relink_error:
                relink_errors.append(f"{getattr(obj, 'name', '')}: {relink_error}")
        errors = cleanup_created(created)
        errors.extend(_restore_layer_exclusions(changes))
        errors.extend(relink_errors)
        suffix = "; rollback also failed: " + "; ".join(errors) if errors else ""
        raise RuntimeError(f"Unable to apply profile-local light isolation: {exc}{suffix}") from exc
    if mutation_sink is not None:
        mutation_sink.extend(changes)
    if created_sink is not None:
        created_sink.extend(created)
    result = {
        "excluded": sorted(str(getattr(value, "name", "")) for value in safe_collections.values()),
        "cloned": sorted(str(getattr(value, "name", "")) for value in clone_collections.values()),
    }
    if root_unlinks:
        result["unlinked"] = sorted(
            str(getattr(obj, "name", "")) for _collection, obj in root_unlinks.values()
        )
    return result


def isolate_existing_lights(
    scene: Any,
    policy: str,
    *,
    mutation_sink: list[tuple[Any, bool]] | None = None,
) -> list[str]:
    """Exclude light-only shared Collections per View Layer without leaking globally."""
    planned_nodes, planned_collections = _plan_existing_light_isolation(scene, policy)

    changes: list[tuple[Any, bool]] = []
    try:
        for layer_collection in planned_nodes.values():
            previous = bool(getattr(layer_collection, "exclude", False))
            changes.append((layer_collection, previous))
            layer_collection.exclude = True
    except Exception as exc:
        restore_errors = _restore_layer_exclusions(changes)
        suffix = ""
        if restore_errors:
            suffix = "; rollback also failed: " + "; ".join(restore_errors)
        raise RuntimeError(f"Unable to exclude shared-light Collections: {exc}{suffix}") from exc
    if mutation_sink is not None:
        mutation_sink.extend(changes)
    return sorted(
        str(getattr(collection, "name", "")) for collection in planned_collections.values()
    )


def _aim_light(obj: Any, target: Any) -> None:
    from mathutils import Vector

    origin = Vector(tuple(float(value) for value in obj.location))
    direction = Vector(tuple(float(value) for value in target)) - origin
    if direction.length_squared <= 1e-18:
        raise ValueError("Light target must differ from its location")
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = direction.to_track_quat("-Z", "Y")


def create_lights(
    scene: Any,
    collection: Any,
    lighting: dict[str, Any],
    *,
    profile_id: str,
    name_namespace: str | None = None,
    exclusion_sink: list[tuple[Any, bool]] | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[str, Any]], dict[str, list[str]]]:
    """Create unique managed light object/data pairs in the profile Collection."""
    created: list[tuple[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    exclusion_changes: list[tuple[Any, bool]] = []
    try:
        isolation = isolate_profile_lights(
            scene,
            str(lighting.get("existing_light_policy", "KEEP")),
            collection,
            profile_id=profile_id,
            name_namespace=name_namespace or profile_id,
            mutation_sink=exclusion_changes,
            created_sink=created,
        )
        for spec in lighting.get("lights", []):
            component_id = str(spec["id"])
            name = _safe_datablock_name(
                profile_id,
                component_id,
                "LIGHT",
                name_namespace=name_namespace,
            )
            if _get(bpy.data.objects, name) is not None or _get(bpy.data.lights, name) is not None:
                raise ValueError(f"Managed light datablock '{name}' already exists")
            light_data = bpy.data.lights.new(name=name, type=str(spec["type"]))
            created.append(("LIGHT", light_data))
            obj = bpy.data.objects.new(name=name, object_data=light_data)
            created.append(("OBJECT", obj))
            collection.objects.link(obj)
            obj.location = tuple(float(value) for value in spec["location_world"])
            light_data.energy = float(spec.get("energy", 1000.0))
            light_data.color = tuple(float(value) for value in spec.get("color_rgb", (1, 1, 1)))
            for attribute, value in (
                ("use_shadow", spec.get("use_shadow", True)),
                ("diffuse_factor", spec.get("diffuse_factor", 1.0)),
                ("specular_factor", spec.get("specular_factor", 1.0)),
                ("volume_factor", spec.get("volume_factor", 1.0)),
            ):
                if hasattr(light_data, attribute):
                    setattr(light_data, attribute, value)
            if spec.get("radius") is not None and hasattr(light_data, "shadow_soft_size"):
                light_data.shadow_soft_size = float(spec["radius"])
            if spec.get("sun_angle_degrees") is not None and hasattr(light_data, "angle"):
                light_data.angle = math.radians(float(spec["sun_angle_degrees"]))
            if str(spec["type"]) == "AREA":
                if spec.get("area_shape") is not None:
                    light_data.shape = str(spec["area_shape"])
                if spec.get("size") is not None:
                    light_data.size = float(spec["size"])
                if spec.get("size_y") is not None:
                    light_data.size_y = float(spec["size_y"])
            if str(spec["type"]) == "SPOT":
                if spec.get("spot_angle_degrees") is not None:
                    light_data.spot_size = math.radians(float(spec["spot_angle_degrees"]))
                if spec.get("spot_blend") is not None:
                    light_data.spot_blend = float(spec["spot_blend"])

            target = spec.get("target_point")
            if spec.get("target_object") is not None:
                target_object = _get(bpy.data.objects, str(spec["target_object"]))
                if target_object is None:
                    raise ValueError(f"Light target object '{spec['target_object']}' was not found")
                target = tuple(float(value) for value in target_object.matrix_world.translation)
            if target is not None:
                _aim_light(obj, target)

            for value, role in ((obj, "LIGHT_OBJECT"), (light_data, "LIGHT_DATA")):
                mark_managed(
                    value,
                    profile_id=profile_id,
                    component_id=component_id,
                    role=role,
                    definition=spec,
                )
            inventory.append(
                {
                    "component_id": component_id,
                    "object_name": obj.name,
                    "data_name": light_data.name,
                    "type": str(spec["type"]),
                }
            )
        if exclusion_sink is not None:
            exclusion_sink.extend(exclusion_changes)
        return inventory, created, isolation
    except Exception as exc:
        cleanup_errors = cleanup_created(created)
        cleanup_errors.extend(_restore_layer_exclusions(exclusion_changes))
        if cleanup_errors:
            raise RuntimeError(
                f"Managed light creation failed ({exc}); cleanup also failed: "
                + "; ".join(cleanup_errors)
            ) from exc
        raise


def _new_material(
    name: str,
    *,
    profile_id: str,
    component_id: str,
    definition: dict[str, Any],
) -> tuple[Any, tuple[str, Any]]:
    material = bpy.data.materials.new(name=name)
    try:
        material.use_nodes = True
        mark_managed(
            material,
            profile_id=profile_id,
            component_id=component_id,
            role="MATERIAL",
            definition=definition,
        )
    except Exception as exc:
        try:
            _remove(bpy.data.materials, material)
        except Exception as cleanup_exc:
            raise RuntimeError(
                f"Managed material creation failed ({exc}); cleanup also failed: {cleanup_exc}"
            ) from exc
        raise
    return material, ("MATERIAL", material)


def _create_fog(
    collection: Any,
    spec: dict[str, Any],
    *,
    profile_id: str,
    name_namespace: str | None = None,
    created: list[tuple[str, Any]],
) -> dict[str, Any]:
    component_id = str(spec["id"])
    name = _safe_datablock_name(
        profile_id,
        component_id,
        "FOG",
        name_namespace=name_namespace,
    )
    mesh = bpy.data.meshes.new(name=f"{name}_MESH"[:63])
    created.append(("MESH", mesh))
    vertices = [
        (-1, -1, -1),
        (1, -1, -1),
        (1, 1, -1),
        (-1, 1, -1),
        (-1, -1, 1),
        (1, -1, 1),
        (1, 1, 1),
        (-1, 1, 1),
    ]
    faces = [
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (4, 0, 3, 7),
    ]
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name=name, object_data=mesh)
    created.append(("OBJECT", obj))
    collection.objects.link(obj)
    obj.location = tuple(float(value) for value in spec.get("location_world", (0, 0, 0)))
    size = tuple(float(value) for value in spec.get("size_xyz", (10, 10, 10)))
    obj.scale = tuple(value / 2.0 for value in size)
    if hasattr(obj, "display_type"):
        obj.display_type = "BOUNDS"

    material, material_record = _new_material(
        f"{name}_MAT"[:63],
        profile_id=profile_id,
        component_id=component_id,
        definition=spec,
    )
    created.append(material_record)
    nodes = material.node_tree.nodes
    nodes.clear()
    volume = nodes.new(type="ShaderNodeVolumePrincipled")
    output = nodes.new(type="ShaderNodeOutputMaterial")
    density = _node_input(volume, "Density")
    color = _node_input(volume, "Color")
    anisotropy = _node_input(volume, "Anisotropy")
    if density is None or color is None:
        raise RuntimeError("Principled Volume node lacks required Density/Color inputs")
    density.default_value = float(spec.get("density", 0.0))
    color.default_value = (*tuple(float(value) for value in spec.get("color_rgb", (1, 1, 1))), 1.0)
    if anisotropy is not None:
        anisotropy.default_value = float(spec.get("anisotropy", 0.0) or 0.0)
    material.node_tree.links.new(volume.outputs["Volume"], output.inputs["Volume"])
    obj.data.materials.append(material)
    for value, role in ((obj, "FOG_OBJECT"), (mesh, "FOG_MESH")):
        mark_managed(
            value,
            profile_id=profile_id,
            component_id=component_id,
            role=role,
            definition=spec,
        )
    return {
        "component_id": component_id,
        "kind": "FOG_VOLUME",
        "object_name": obj.name,
        "density": float(spec.get("density", 0.0)),
    }


def _stable_unit(seed: int, index: int, axis: int) -> float:
    digest = hashlib.sha256(f"{seed}:{index}:{axis}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def _resolve_rain_drop_count(value: Any) -> tuple[float, int]:
    requested_rate = float(value)
    if not math.isfinite(requested_rate) or requested_rate < 0.0:
        raise ValueError("RAIN_RIG rain_rate must be a finite value greater than or equal to 0")
    if requested_rate > MAX_RAIN_DROPS:
        raise ValueError(
            f"RAIN_RIG rain_rate {requested_rate:g} exceeds the supported maximum "
            f"of {MAX_RAIN_DROPS} drops"
        )
    return requested_rate, int(round(requested_rate))


def _create_rain(
    collection: Any,
    spec: dict[str, Any],
    *,
    profile_id: str,
    name_namespace: str | None = None,
    created: list[tuple[str, Any]],
) -> dict[str, Any]:
    component_id = str(spec["id"])
    name = _safe_datablock_name(
        profile_id,
        component_id,
        "RAIN",
        name_namespace=name_namespace,
    )
    requested_rate, count = _resolve_rain_drop_count(spec.get("rain_rate", 1.0))
    curve = bpy.data.curves.new(name=f"{name}_CURVE"[:63], type="CURVE")
    created.append(("CURVE", curve))
    curve.dimensions = "3D"
    curve.resolution_u = 1
    curve.bevel_resolution = 0
    curve.bevel_depth = float(spec.get("drop_size_m", 0.002)) / 2.0
    size = tuple(float(value) for value in spec.get("size_xyz", (10, 10, 10)))
    wind = tuple(float(value) for value in spec.get("wind_vector", (0, 0, 0)) or (0, 0, 0))
    fall_speed = float(spec.get("fall_speed_mps", 10.0))
    length = max(float(spec.get("drop_size_m", 0.002)) * 8.0, fall_speed / 48.0)
    seed = int(spec.get("seed", 0))
    for index in range(count):
        x = (_stable_unit(seed, index, 0) - 0.5) * size[0]
        y = (_stable_unit(seed, index, 1) - 0.5) * size[1]
        z = (_stable_unit(seed, index, 2) - 0.5) * size[2]
        spline = curve.splines.new("POLY")
        spline.points.add(1)
        spline.points[0].co = (x, y, z, 1.0)
        spline.points[1].co = (
            x + wind[0] / 48.0,
            y + wind[1] / 48.0,
            z - length + wind[2] / 48.0,
            1.0,
        )
    obj = bpy.data.objects.new(name=name, object_data=curve)
    created.append(("OBJECT", obj))
    collection.objects.link(obj)
    obj.location = tuple(float(value) for value in spec.get("location_world", (0, 0, 0)))

    material, material_record = _new_material(
        f"{name}_MAT"[:63],
        profile_id=profile_id,
        component_id=component_id,
        definition=spec,
    )
    created.append(material_record)
    nodes = material.node_tree.nodes
    principled = _get(nodes, "Principled BSDF")
    if principled is not None:
        base_color = _node_input(principled, "Base Color")
        roughness = _node_input(principled, "Roughness")
        transmission = _node_input(principled, "Transmission Weight", "Transmission")
        ior = _node_input(principled, "IOR")
        if base_color is not None:
            base_color.default_value = (
                *tuple(float(value) for value in spec.get("color_rgb", (0.8, 0.9, 1.0))),
                1.0,
            )
        if roughness is not None:
            roughness.default_value = 0.08
        if transmission is not None:
            transmission.default_value = 0.7
        if ior is not None:
            ior.default_value = 1.333
    curve.materials.append(material)
    for value, role in ((obj, "RAIN_OBJECT"), (curve, "RAIN_CURVE")):
        mark_managed(
            value,
            profile_id=profile_id,
            component_id=component_id,
            role=role,
            definition=spec,
        )
    return {
        "component_id": component_id,
        "kind": "RAIN_RIG",
        "object_name": obj.name,
        "seed": seed,
        "requested_rain_rate": requested_rate,
        "drop_count": count,
    }


def create_atmosphere(
    collection: Any,
    components: list[dict[str, Any]],
    *,
    profile_id: str,
    name_namespace: str | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[str, Any]]]:
    """Create all enabled atmosphere/weather components transactionally."""
    inventory: list[dict[str, Any]] = []
    created: list[tuple[str, Any]] = []
    try:
        for spec in components:
            if not bool(spec.get("enabled", True)):
                continue
            kind = str(spec.get("kind", ""))
            if kind == "FOG_VOLUME":
                item = _create_fog(
                    collection,
                    spec,
                    profile_id=profile_id,
                    name_namespace=name_namespace,
                    created=created,
                )
            elif kind == "RAIN_RIG":
                item = _create_rain(
                    collection,
                    spec,
                    profile_id=profile_id,
                    name_namespace=name_namespace,
                    created=created,
                )
            else:
                raise ValueError(f"Unsupported atmosphere component kind '{kind}'")
            inventory.append(item)
        return inventory, created
    except Exception as exc:
        errors = cleanup_created(created)
        if errors:
            raise RuntimeError(
                f"Managed atmosphere creation failed ({exc}); cleanup also failed: "
                + "; ".join(errors)
            ) from exc
        raise


def clone_camera(
    scene: Any,
    collection: Any,
    spec: dict[str, Any],
    *,
    profile_id: str,
    name_namespace: str | None = None,
    created_sink: list[tuple[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, list[tuple[str, Any]]]:
    """Clone camera object/data only when the profile explicitly changes optics."""
    mode = str(spec.get("mode", "KEEP"))
    if mode == "KEEP":
        camera = getattr(scene, "camera", None)
        return (
            {"mode": mode, "camera_name": getattr(camera, "name", None)},
            [],
        )
    if mode != "MANAGED_CLONE":
        raise ValueError(f"Unsupported camera mode '{mode}'")
    source = _get(getattr(bpy.data, "objects", None), str(spec.get("source_camera", "")))
    if source is None or str(getattr(source, "type", "")) != "CAMERA":
        raise ValueError(f"Camera '{spec.get('source_camera')}' was not found")
    if source not in _values(getattr(scene, "objects", None)):
        raise ValueError("Camera clone source is not linked to the compiled profile Scene")

    created: list[tuple[str, Any]] = []
    original_camera = getattr(scene, "camera", None)
    try:
        name = _safe_datablock_name(
            profile_id,
            "camera",
            "CAMERA",
            name_namespace=name_namespace,
        )
        data_copy = source.data.copy()
        created.append(("CAMERA", data_copy))
        obj_copy = bpy.data.objects.new(name=name, object_data=data_copy)
        created.append(("OBJECT", obj_copy))
        data_copy.name = f"{name}_DATA"[:63]
        collection.objects.link(obj_copy)
        obj_copy.matrix_world = source.matrix_world.copy()
        if spec.get("location_world") is not None or spec.get("target_point") is not None:
            from mathutils import Matrix, Vector

            source_matrix = source.matrix_world.copy()
            origin = (
                Vector(tuple(float(value) for value in spec["location_world"]))
                if spec.get("location_world") is not None
                else source_matrix.translation.copy()
            )
            rotation = source_matrix.to_quaternion()
            if spec.get("target_point") is not None:
                direction = Vector(tuple(float(value) for value in spec["target_point"])) - origin
                if direction.length_squared <= 1e-18:
                    raise ValueError("Camera target must differ from its location")
                rotation = direction.to_track_quat("-Z", "Y")
            obj_copy.parent = None
            obj_copy.matrix_parent_inverse = Matrix.Identity(4)
            obj_copy.matrix_world = Matrix.LocRotScale(origin, rotation, source_matrix.to_scale())
        dof = getattr(data_copy, "dof", None)
        if dof is not None:
            if spec.get("use_dof") is not None:
                dof.use_dof = bool(spec["use_dof"])
            if spec.get("focus_object") is not None:
                focus = _get(bpy.data.objects, str(spec["focus_object"]))
                if focus is None:
                    raise ValueError(f"Focus object '{spec['focus_object']}' was not found")
                dof.focus_object = focus
            if spec.get("focus_distance_m") is not None:
                dof.focus_distance = float(spec["focus_distance_m"])
            if spec.get("aperture_fstop") is not None:
                dof.aperture_fstop = float(spec["aperture_fstop"])
        scene.camera = obj_copy
        for value, role in ((obj_copy, "CAMERA_OBJECT"), (data_copy, "CAMERA_DATA")):
            mark_managed(
                value,
                profile_id=profile_id,
                component_id="camera",
                role=role,
                definition=spec,
            )
    except Exception as exc:
        errors: list[str] = []
        try:
            scene.camera = original_camera
        except Exception as restore_exc:
            errors.append(f"Scene camera restore: {restore_exc}")
        errors.extend(cleanup_created(created))
        if errors:
            raise RuntimeError(
                f"Managed camera creation failed ({exc}); cleanup also failed: " + "; ".join(errors)
            ) from exc
        raise
    if created_sink is not None:
        created_sink.extend(created)
        return {
            "mode": mode,
            "camera_name": obj_copy.name,
            "location_world": [float(value) for value in obj_copy.location],
        }, []
    return {
        "mode": mode,
        "camera_name": obj_copy.name,
        "location_world": [float(value) for value in obj_copy.location],
    }, created


def configure_render(scene: Any, spec: dict[str, Any]) -> dict[str, Any]:
    """Apply typed render-preset overrides without touching output paths."""
    render = scene.render
    engine = str(spec.get("engine", "KEEP"))
    if engine != "KEEP":
        requested = engine
        attempts = [requested]
        if requested == "BLENDER_EEVEE":
            attempts.append("BLENDER_EEVEE_NEXT")
        elif requested == "BLENDER_EEVEE_NEXT":
            attempts.append("BLENDER_EEVEE")
        failures: list[str] = []
        for candidate in attempts:
            try:
                render.engine = candidate
                break
            except (TypeError, ValueError) as exc:
                failures.append(f"{candidate}: {exc}")
        else:
            raise RuntimeError("No compatible render engine: " + "; ".join(failures))
    if spec.get("resolution_x") is not None:
        render.resolution_x = int(spec["resolution_x"])
    if spec.get("resolution_y") is not None:
        render.resolution_y = int(spec["resolution_y"])
    if spec.get("resolution_percentage") is not None:
        render.resolution_percentage = int(spec["resolution_percentage"])
    if spec.get("film_transparent") is not None:
        render.film_transparent = bool(spec["film_transparent"])
    if spec.get("use_motion_blur") is not None:
        for owner in (render, getattr(scene, "eevee", None)):
            if owner is not None and hasattr(owner, "use_motion_blur"):
                owner.use_motion_blur = bool(spec["use_motion_blur"])
                break
    samples = spec.get("samples")
    denoise = spec.get("denoise")
    if str(render.engine) == "CYCLES":
        if samples is not None:
            scene.cycles.samples = int(samples)
        if denoise is not None and hasattr(scene.cycles, "use_denoising"):
            scene.cycles.use_denoising = bool(denoise)
    elif str(render.engine).startswith("BLENDER_EEVEE"):
        eevee = getattr(scene, "eevee", None)
        if samples is not None and eevee is not None and hasattr(eevee, "taa_render_samples"):
            eevee.taa_render_samples = int(samples)
    return {
        "engine": str(render.engine),
        "resolution_x": int(getattr(render, "resolution_x", 0)),
        "resolution_y": int(getattr(render, "resolution_y", 0)),
        "resolution_percentage": int(render.resolution_percentage),
        "samples": int(
            getattr(
                scene.cycles if str(render.engine) == "CYCLES" else getattr(scene, "eevee", None),
                "samples" if str(render.engine) == "CYCLES" else "taa_render_samples",
                0,
            )
        ),
    }


def configure_color(scene: Any, spec: dict[str, Any]) -> dict[str, Any]:
    """Apply explicit color-management overrides."""
    if str(spec.get("mode", "KEEP")) == "MANAGED":
        for key in ("view_transform", "look", "exposure", "gamma"):
            if spec.get(key) is not None:
                setattr(scene.view_settings, key, spec[key])
    return {
        key: getattr(scene.view_settings, key, None)
        for key in ("view_transform", "look", "exposure", "gamma")
    }


def build_profile_assets(
    scene: Any,
    collection: Any,
    world: Any,
    profile: dict[str, Any],
    *,
    name_namespace: str | None = None,
) -> dict[str, Any]:
    """Build the non-compositor payload and return cleanup/inventory data.

    Callers must invoke :func:`cleanup_created` if a later compiler phase fails.
    This function cleans its own partial work if any phase here raises.
    """
    profile_id = str(profile["profile_id"])
    created: list[tuple[str, Any]] = []
    exclusion_changes: list[tuple[Any, bool]] = []
    original_camera = getattr(scene, "camera", None)
    try:
        world_result = configure_world(
            world,
            profile.get("world", {}),
            profile_id=profile_id,
            created_sink=created,
        )
        lights, light_records, isolation = create_lights(
            scene,
            collection,
            profile.get("lighting", {}),
            profile_id=profile_id,
            name_namespace=name_namespace,
            exclusion_sink=exclusion_changes,
        )
        created.extend(light_records)
        atmosphere, atmosphere_records = create_atmosphere(
            collection,
            profile.get("atmosphere", []),
            profile_id=profile_id,
            name_namespace=name_namespace,
        )
        created.extend(atmosphere_records)
        camera, camera_records = clone_camera(
            scene,
            collection,
            profile.get("camera", {}),
            profile_id=profile_id,
            name_namespace=name_namespace,
            created_sink=created,
        )
        created.extend(camera_records)
        render = configure_render(scene, profile.get("render", {}))
        color = configure_color(scene, profile.get("color_management", {}))
        return {
            "created": created,
            "inventory": {
                "world": world_result,
                "lights": lights,
                "excluded_light_collections": isolation["excluded"],
                "cloned_light_collections": isolation["cloned"],
                "atmosphere": atmosphere,
                "camera": camera,
                "render": render,
                "color_management": color,
            },
        }
    except Exception as exc:
        errors: list[str] = []
        try:
            scene.camera = original_camera
        except Exception as restore_exc:
            errors.append(f"Scene camera restore: {restore_exc}")
        errors.extend(cleanup_created(created))
        errors.extend(_restore_layer_exclusions(exclusion_changes))
        if errors:
            raise RuntimeError(
                f"Managed profile asset build failed ({exc}); cleanup also failed: "
                + "; ".join(errors)
            ) from exc
        raise
