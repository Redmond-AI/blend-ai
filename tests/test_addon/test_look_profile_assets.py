"""Focused tests for profile-local asset creation and isolation rules."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


MODULE_PATH = Path(__file__).parents[2] / "addon" / "look_profile_assets.py"


def _load_module():
    bpy_module = ModuleType("bpy")
    previous = sys.modules.get("bpy")
    try:
        sys.modules["bpy"] = bpy_module
        spec = importlib.util.spec_from_file_location("addon.look_profile_assets", MODULE_PATH)
        assert spec is not None and spec.loader is not None
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        return loaded
    finally:
        if previous is None:
            sys.modules.pop("bpy", None)
        else:
            sys.modules["bpy"] = previous


module = _load_module()


class FakeID(dict):
    def __init__(self, **attributes):
        super().__init__()
        for key, value in attributes.items():
            setattr(self, key, value)


class Named(list):
    def get(self, name, default=None):
        return next((item for item in self if getattr(item, "name", None) == name), default)


def test_managed_markers_include_profile_component_schema_and_hash():
    value = FakeID()
    definition = {"strength": 0.25, "color": [0.1, 0.2, 0.3]}

    module.mark_managed(
        value,
        profile_id="night",
        component_id="moon",
        role="LIGHT_DATA",
        definition=definition,
    )

    assert value[module.MANAGED_PROP] is True
    assert value[module.PROFILE_ID_PROP] == "night"
    assert value[module.COMPONENT_ID_PROP] == "moon"
    assert value[module.SCHEMA_PROP] == 1
    assert len(value[module.CONTENT_HASH_PROP]) == 64


def test_legacy_datablock_name_is_unchanged_without_namespace():
    expected = "AI_LIGHT_night_moon_f3b752fc"

    assert module._safe_datablock_name("night", "moon", "LIGHT") == expected
    assert (
        module._safe_datablock_name(
            "night",
            "moon",
            "LIGHT",
            name_namespace=None,
        )
        == expected
    )


def test_namespaced_light_asset_sets_with_identical_component_ids_coexist(monkeypatch):
    class LightValues(Named):
        def new(self, *, name, type):
            value = FakeID(name=name, type=type)
            self.append(value)
            return value

    class ObjectValues(Named):
        def new(self, *, name, object_data):
            value = FakeID(name=name, data=object_data, type="LIGHT")
            self.append(value)
            return value

    class LinkedObjects(Named):
        def link(self, value):
            self.append(value)

    lights = LightValues()
    objects = ObjectValues()
    monkeypatch.setattr(
        module,
        "bpy",
        SimpleNamespace(data=SimpleNamespace(lights=lights, objects=objects)),
    )
    scene = FakeID(objects=[], view_layers=[])
    lighting = {
        "existing_light_policy": "KEEP",
        "lights": [
            {
                "id": "moon",
                "type": "POINT",
                "location_world": [1.0, 2.0, 3.0],
                "energy": 250.0,
            }
        ],
    }

    first_inventory, _first_created, _first_excluded = module.create_lights(
        scene,
        FakeID(objects=LinkedObjects()),
        lighting,
        profile_id="night",
        name_namespace="AI_LOOK_night_v001_r001_SCENE",
    )
    second_inventory, _second_created, _second_excluded = module.create_lights(
        scene,
        FakeID(objects=LinkedObjects()),
        lighting,
        profile_id="night",
        name_namespace="AI_LOOK_night_v001_r002_SCENE",
    )

    first_name = first_inventory[0]["object_name"]
    second_name = second_inventory[0]["object_name"]
    assert first_name != second_name
    assert len(objects) == len(lights) == 2
    assert {value.name for value in objects} == {first_name, second_name}
    assert all(value[module.PROFILE_ID_PROP] == "night" for value in objects)
    assert all(value[module.COMPONENT_ID_PROP] == "moon" for value in objects)


def test_profile_asset_builder_propagates_namespace_to_every_named_asset_kind(monkeypatch):
    namespaces = {}

    monkeypatch.setattr(
        module,
        "configure_world",
        lambda _world, _spec, **_kwargs: {"mode": "KEEP"},
    )

    def create_lights(_scene, _collection, _spec, *, name_namespace=None, **_kwargs):
        namespaces["lights"] = name_namespace
        return [], [], {"excluded": [], "cloned": []}

    def create_atmosphere(_collection, _spec, *, name_namespace=None, **_kwargs):
        namespaces["atmosphere"] = name_namespace
        return [], []

    def clone_camera(_scene, _collection, _spec, *, name_namespace=None, **_kwargs):
        namespaces["camera"] = name_namespace
        return {"mode": "KEEP", "camera_name": None}, []

    monkeypatch.setattr(module, "create_lights", create_lights)
    monkeypatch.setattr(module, "create_atmosphere", create_atmosphere)
    monkeypatch.setattr(module, "clone_camera", clone_camera)
    monkeypatch.setattr(module, "configure_render", lambda _scene, _spec: {})
    monkeypatch.setattr(module, "configure_color", lambda _scene, _spec: {})

    module.build_profile_assets(
        FakeID(camera=None),
        FakeID(),
        FakeID(),
        {"profile_id": "night"},
        name_namespace="AI_LOOK_night_v002_r004_SCENE",
    )

    assert namespaces == {
        "lights": "AI_LOOK_night_v002_r004_SCENE",
        "atmosphere": "AI_LOOK_night_v002_r004_SCENE",
        "camera": "AI_LOOK_night_v002_r004_SCENE",
    }


def test_stable_rain_units_are_deterministic_and_component_separated():
    first = [module._stable_unit(42, 7, axis) for axis in range(3)]
    second = [module._stable_unit(42, 7, axis) for axis in range(3)]

    assert first == second
    assert len(set(first)) == 3
    assert all(0.0 <= value <= 1.0 for value in first)


def _layer_tree(collections):
    children = []
    for collection in collections:
        children.append(FakeID(collection=collection, children=[], exclude=False))
    return (
        FakeID(
            collection=FakeID(name="Root", objects=[], children=[]),
            children=children,
            exclude=False,
        ),
        children,
    )


def _nested_layer_tree(collection, children):
    layer_children = [_nested_layer_tree(child, child.children) for child in children]
    return FakeID(collection=collection, children=layer_children, exclude=False)


def test_nonmanaged_lights_are_excluded_per_view_layer_not_globally():
    light = FakeID(name="Artist Sun", type="LIGHT", hide_render=False)
    light_collection = FakeID(name="Artist Lights", objects=[light])
    light.users_collection = [light_collection]
    root, children = _layer_tree([light_collection])
    scene = FakeID(
        objects=[light],
        view_layers=[FakeID(layer_collection=root)],
    )

    excluded = module.isolate_existing_lights(scene, "MUTE_NON_MANAGED")

    assert excluded == ["Artist Lights"]
    assert children[0].exclude is True
    assert light.hide_render is False


def test_light_isolation_preflight_reports_plan_without_mutating_layer_flags():
    light = FakeID(name="Artist Sun", type="LIGHT", hide_render=False)
    light_collection = FakeID(name="Artist Lights", objects=[light])
    light.users_collection = [light_collection]
    root, children = _layer_tree([light_collection])
    scene = FakeID(
        objects=[light],
        view_layers=[FakeID(layer_collection=root)],
    )

    planned = module.validate_existing_light_isolation(
        scene,
        "MUTE_NON_MANAGED",
    )

    assert planned == ["Artist Lights"]
    assert children[0].exclude is False
    assert light.hide_render is False


def test_mixed_collection_refuses_leaky_light_muting():
    light = FakeID(name="Artist Sun", type="LIGHT", hide_render=False)
    mesh = FakeID(name="Ground", type="MESH")
    mixed = FakeID(name="Mixed", objects=[light, mesh])
    light.users_collection = [mixed]
    root, _children = _layer_tree([mixed])
    scene = FakeID(objects=[light, mesh], view_layers=[FakeID(layer_collection=root)])

    with pytest.raises(ValueError, match="would hide non-profile content"):
        module.isolate_existing_lights(scene, "MUTE_NON_MANAGED")

    assert light.hide_render is False


def test_profile_preflight_allows_scene_local_clone_for_mixed_collection():
    light = FakeID(name="Artist Sun", type="LIGHT", hide_render=False)
    mesh = FakeID(name="Ground", type="MESH")
    mixed = FakeID(name="Mixed", objects=[light, mesh], children=[])
    light.users_collection = [mixed]
    root, children = _layer_tree([mixed])
    scene = FakeID(
        objects=[light, mesh],
        view_layers=[FakeID(layer_collection=root)],
    )

    plan = module.validate_profile_light_isolation(scene, "MUTE_NON_MANAGED")

    assert plan == {"excluded": [], "cloned": ["Mixed"]}
    assert children[0].exclude is False
    assert mixed.objects == [light, mesh]


def test_profile_isolation_clones_nonlights_and_excludes_original(monkeypatch):
    class LinkedValues(Named):
        def link(self, value):
            self.append(value)

    class CollectionValues(Named):
        def new(self, *, name):
            value = FakeID(name=name, objects=LinkedValues(), children=LinkedValues())
            self.append(value)
            return value

    light = FakeID(name="Artist Sun", type="LIGHT", hide_render=False)
    mesh = FakeID(name="Ground", type="MESH")
    mixed = FakeID(name="Mixed", objects=[light, mesh], children=[])
    light.users_collection = [mixed]
    root, children = _layer_tree([mixed])
    scene = FakeID(
        objects=[light, mesh],
        view_layers=[FakeID(layer_collection=root)],
    )
    destination = FakeID(children=LinkedValues())
    collections = CollectionValues()
    monkeypatch.setattr(
        module,
        "bpy",
        SimpleNamespace(data=SimpleNamespace(collections=collections)),
    )
    created = []
    changes = []

    result = module.isolate_profile_lights(
        scene,
        "MUTE_NON_MANAGED",
        destination,
        profile_id="night",
        name_namespace="AI_LOOK_night_v001_r001_SCENE",
        mutation_sink=changes,
        created_sink=created,
    )

    assert result == {"excluded": [], "cloned": ["Mixed"]}
    assert len(destination.children) == 1
    clone = destination.children[0]
    assert list(clone.objects) == [mesh]
    assert all(value is not light for value in clone.objects)
    assert children[0].exclude is True
    assert changes == [(children[0], False)]
    assert created == [("COLLECTION", clone)]
    assert mixed.objects == [light, mesh]


def test_profile_isolation_unlinks_direct_root_light_without_touching_geometry():
    class LinkedValues(Named):
        def link(self, value):
            self.append(value)

        def unlink(self, value):
            self.remove(value)

    light = FakeID(name="Artist Sun", type="LIGHT", hide_render=False)
    mesh = FakeID(name="Ground", type="MESH")
    root_collection = FakeID(
        name="Compiled Scene Root",
        objects=LinkedValues([light, mesh]),
        children=[],
    )
    root_layer = FakeID(collection=root_collection, children=[], exclude=False)
    scene = FakeID(
        objects=[light, mesh],
        view_layers=[FakeID(layer_collection=root_layer)],
    )

    plan = module.validate_profile_light_isolation(scene, "MUTE_NON_MANAGED")
    result = module.isolate_profile_lights(
        scene,
        "MUTE_NON_MANAGED",
        FakeID(children=[]),
        profile_id="night",
        name_namespace="AI_LOOK_night_v001_r001_SCENE",
    )

    expected = {"excluded": [], "cloned": [], "unlinked": ["Artist Sun"]}
    assert plan == expected
    assert result == expected
    assert root_collection.objects == [mesh]
    assert light.hide_render is False


def test_nested_child_geometry_refuses_parent_collection_exclusion():
    light = FakeID(name="Artist Sun", type="LIGHT", hide_render=False)
    mesh = FakeID(name="Nested Ground", type="MESH")
    geometry_child = FakeID(name="Nested Geometry", objects=[mesh], children=[])
    parent = FakeID(name="Artist Lights", objects=[light], children=[geometry_child])
    light.users_collection = [parent]
    root = FakeID(
        collection=FakeID(name="Root", objects=[], children=[parent]),
        children=[_nested_layer_tree(parent, parent.children)],
        exclude=False,
    )
    parent_layer = root.children[0]
    scene = FakeID(
        objects=[light, mesh],
        view_layers=[FakeID(layer_collection=root)],
    )

    with pytest.raises(ValueError, match="Nested Ground"):
        module.isolate_existing_lights(scene, "MUTE_NON_MANAGED")

    assert parent_layer.exclude is False
    assert parent_layer.children[0].exclude is False


def test_nested_light_only_collection_can_be_excluded_without_global_hides():
    parent_light = FakeID(name="Artist Sun", type="LIGHT", hide_render=False)
    child_light = FakeID(name="Artist Fill", type="LIGHT", hide_render=False)
    child = FakeID(name="Nested Lights", objects=[child_light], children=[])
    parent = FakeID(name="Artist Lights", objects=[parent_light], children=[child])
    parent_light.users_collection = [parent]
    child_light.users_collection = [child]
    root = FakeID(
        collection=FakeID(name="Root", objects=[], children=[parent]),
        children=[_nested_layer_tree(parent, parent.children)],
        exclude=False,
    )
    scene = FakeID(
        objects=[parent_light, child_light],
        view_layers=[FakeID(layer_collection=root)],
    )

    excluded = module.isolate_existing_lights(scene, "MUTE_NON_MANAGED")

    assert excluded == ["Artist Lights", "Nested Lights"]
    assert root.children[0].exclude is True
    assert root.children[0].children[0].exclude is True
    assert parent_light.hide_render is False
    assert child_light.hide_render is False


def test_multilinked_light_requires_every_collection_path_to_be_safe():
    light = FakeID(name="Artist Sun", type="LIGHT", hide_render=False)
    mesh = FakeID(name="Ground", type="MESH")
    safe = FakeID(name="Artist Lights", objects=[light], children=[])
    unsafe = FakeID(name="Mixed", objects=[light, mesh], children=[])
    light.users_collection = [safe, unsafe]
    root, children = _layer_tree([safe, unsafe])
    scene = FakeID(
        objects=[light, mesh],
        view_layers=[FakeID(layer_collection=root)],
    )

    with pytest.raises(ValueError, match="Mixed"):
        module.isolate_existing_lights(scene, "MUTE_NON_MANAGED")

    assert [child.exclude for child in children] == [False, False]


def test_multilinked_light_excludes_every_safe_membership_and_view_layer_path():
    light = FakeID(name="Artist Sun", type="LIGHT", hide_render=False)
    first = FakeID(name="Key Lights", objects=[light], children=[])
    second = FakeID(name="Exterior Lights", objects=[light], children=[])
    light.users_collection = [first, second]
    root_a, children_a = _layer_tree([first, second])
    root_b, children_b = _layer_tree([first, second])
    scene = FakeID(
        objects=[light],
        view_layers=[
            FakeID(name="Beauty", layer_collection=root_a),
            FakeID(name="Masks", layer_collection=root_b),
        ],
    )

    excluded = module.isolate_existing_lights(scene, "MUTE_NON_MANAGED")

    assert excluded == ["Exterior Lights", "Key Lights"]
    assert all(child.exclude for child in children_a + children_b)


def test_exclusion_changes_can_be_rolled_back_after_a_later_build_failure():
    light = FakeID(name="Artist Sun", type="LIGHT", hide_render=False)
    collection = FakeID(name="Artist Lights", objects=[light], children=[])
    light.users_collection = [collection]
    root, children = _layer_tree([collection])
    scene = FakeID(objects=[light], view_layers=[FakeID(layer_collection=root)])
    changes = []

    module.isolate_existing_lights(
        scene,
        "MUTE_NON_MANAGED",
        mutation_sink=changes,
    )
    errors = module._restore_layer_exclusions(changes)

    assert errors == []
    assert children[0].exclude is False


def test_exclusion_application_rolls_back_if_a_later_layer_node_rejects_mutation():
    class RejectingLayerCollection:
        def __init__(self, collection):
            self.collection = collection
            self.children = []
            self._exclude = False

        @property
        def exclude(self):
            return self._exclude

        @exclude.setter
        def exclude(self, value):
            if value:
                raise RuntimeError("read-only layer")
            self._exclude = value

    light = FakeID(name="Artist Sun", type="LIGHT", hide_render=False)
    first = FakeID(name="First Lights", objects=[light], children=[])
    second = FakeID(name="Read-only Lights", objects=[light], children=[])
    light.users_collection = [first, second]
    first_layer = FakeID(collection=first, children=[], exclude=False)
    second_layer = RejectingLayerCollection(second)
    root = FakeID(
        collection=FakeID(name="Root", objects=[], children=[]),
        children=[first_layer, second_layer],
        exclude=False,
    )
    scene = FakeID(objects=[light], view_layers=[FakeID(layer_collection=root)])

    with pytest.raises(RuntimeError, match="Unable to exclude"):
        module.isolate_existing_lights(scene, "MUTE_NON_MANAGED")

    assert first_layer.exclude is False
    assert second_layer.exclude is False


def test_rain_zero_is_empty_and_values_above_supported_max_are_rejected():
    assert module._resolve_rain_drop_count(0) == (0.0, 0)
    assert module._resolve_rain_drop_count(17.6) == (17.6, 18)

    with pytest.raises(ValueError, match="supported maximum"):
        module._resolve_rain_drop_count(module.MAX_RAIN_DROPS + 0.01)

    for invalid in (-0.01, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite value"):
            module._resolve_rain_drop_count(invalid)


def test_render_engine_falls_back_between_blender_42_and_52_identifiers(monkeypatch):
    class Render:
        resolution_percentage = 100
        film_transparent = False

        def __init__(self):
            self._engine = "CYCLES"

        @property
        def engine(self):
            return self._engine

        @engine.setter
        def engine(self, value):
            if value == "BLENDER_EEVEE_NEXT":
                raise TypeError("unsupported")
            self._engine = value

    scene = FakeID(
        render=Render(),
        cycles=FakeID(samples=64, use_denoising=True),
        eevee=FakeID(taa_render_samples=16),
    )

    result = module.configure_render(
        scene,
        {"engine": "BLENDER_EEVEE_NEXT", "samples": 32, "denoise": True},
    )

    assert result["engine"] == "BLENDER_EEVEE"
    assert scene.eevee.taa_render_samples == 32


def test_cleanup_runs_in_reverse_dependency_order(monkeypatch):
    calls = []

    def owner(kind):
        return SimpleNamespace(remove=lambda value, **kwargs: calls.append((kind, value.name)))

    module.bpy = SimpleNamespace(
        data=SimpleNamespace(
            objects=owner("OBJECT"),
            lights=owner("LIGHT"),
            cameras=owner("CAMERA"),
            curves=owner("CURVE"),
            meshes=owner("MESH"),
            materials=owner("MATERIAL"),
            node_groups=owner("NODE_GROUP"),
        )
    )
    light = FakeID(name="Light")
    obj = FakeID(name="Object")

    errors = module.cleanup_created([("LIGHT", light), ("OBJECT", obj)])

    assert errors == []
    assert calls == [("OBJECT", "Object"), ("LIGHT", "Light")]
