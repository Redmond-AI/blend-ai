"""Targeted unit tests for the spatial relighting handler contract."""

import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest


MODULE_PATH = Path(__file__).parents[2] / "addon" / "handlers" / "relighting.py"


class FakeID(dict):
    """Small Blender-ID-like object supporting attributes and custom props."""

    def __init__(self, **values):
        super().__init__()
        self.__dict__.update(values)

    def __bool__(self):
        return True


class FakeNamedValues(list):
    def get(self, name):
        return next((value for value in self if value.name == name), None)


class FakeText(FakeID):
    def __init__(self, name):
        super().__init__(name=name)
        self._body = ""

    def as_string(self):
        return self._body

    def clear(self):
        self._body = ""

    def write(self, value):
        self._body += value


class FakeTexts(FakeNamedValues):
    def new(self, name):
        text = FakeText(name)
        self.append(text)
        return text

    def remove(self, text):
        list.remove(self, text)


class FakeCollectionObjects(FakeNamedValues):
    def __init__(self, scene):
        super().__init__()
        self.scene = scene

    def link(self, obj):
        if obj not in self:
            self.append(obj)
        if obj not in self.scene.objects:
            self.scene.objects.append(obj)
        if self.owner not in obj.users_collection:
            obj.users_collection.append(self.owner)


def _managed_runtime(module):
    scene = FakeID(
        name="Scene",
        frame_current=1,
        objects=FakeNamedValues(),
        world=None,
        view_settings=FakeID(exposure=0.0),
    )
    collections = FakeNamedValues()
    lights = FakeNamedValues()
    objects = FakeNamedValues()

    def new_collection(name):
        collection = FakeID(name=name)
        collection.objects = FakeCollectionObjects(scene)
        collection.objects.owner = collection
        collection.children = FakeNamedValues()
        collections.append(collection)
        return collection

    root = new_collection("Scene Collection")
    root.children.link = lambda collection: root.children.append(collection)
    scene.collection = root

    def new_light(name, type):
        data = FakeID(
            name=name,
            type=type,
            users=0,
            energy=0.0,
            color=(1.0, 1.0, 1.0),
            use_shadow=True,
            diffuse_factor=1.0,
            specular_factor=1.0,
            volume_factor=1.0,
            shadow_soft_size=0.0,
        )
        lights.append(data)
        return data

    def new_object(name, object_data):
        obj = FakeID(
            name=name,
            type="LIGHT",
            data=object_data,
            users_collection=[],
            location=(0.0, 0.0, 0.0),
            matrix_world=None,
            parent=None,
            rotation_mode="XYZ",
            rotation_quaternion=(1.0, 0.0, 0.0, 0.0),
            rotation_euler=(0.0, 0.0, 0.0),
            hide_render=False,
            hide_viewport=False,
        )
        object_data.users += 1
        objects.append(obj)
        return obj

    def remove_object(obj, do_unlink=True):
        if obj in objects:
            list.remove(objects, obj)
        if obj in scene.objects:
            scene.objects.remove(obj)
        for collection in list(obj.users_collection):
            if obj in collection.objects:
                collection.objects.remove(obj)
        obj.data.users = max(0, obj.data.users - 1)

    def remove_light(data):
        if data in lights:
            list.remove(lights, data)

    collections.new = new_collection
    lights.new = new_light
    lights.remove = remove_light
    objects.new = new_object
    objects.remove = remove_object
    fake_bpy = FakeID(
        context=FakeID(scene=scene, view_layer=FakeID(update=MagicMock())),
        data=FakeID(
            collections=collections,
            lights=lights,
            objects=objects,
            worlds=FakeNamedValues(),
            texts=FakeTexts(),
            scenes=[scene],
        ),
    )
    module.bpy = fake_bpy
    return fake_bpy, scene


def _load(monkeypatch, *, scene=None):
    if scene is None:
        scene = FakeID(name="Scene", frame_current=1, objects=[])
    if not hasattr(scene, "view_settings"):
        scene.view_settings = FakeID(exposure=0.0)
    depsgraph = FakeID(object_instances=[], updates=[])
    context = FakeID(
        scene=scene,
        view_layer=FakeID(name="ViewLayer"),
        evaluated_depsgraph_get=lambda: depsgraph,
        _depsgraph=depsgraph,
    )
    lights = FakeID(get=lambda _name: None, new=MagicMock())
    fake_bpy = FakeID(
        context=context,
        data=FakeID(
            objects=FakeID(get=lambda _name: None),
            collections=FakeID(get=lambda _name: None),
            lights=lights,
            worlds=FakeID(get=lambda _name: None),
            texts=FakeID(get=lambda _name: None),
        ),
    )
    dispatcher = MagicMock()
    spatial = SimpleNamespace(
        get_revisions=lambda: {"geometry_revision": 3, "lighting_revision": 4},
        make_scene_revision=lambda _scene=None: "g3:l4:f1",
        make_cache_key=lambda namespace, payload: f"{namespace}:{payload}",
        get_cached=lambda _key: None,
        put_cached=lambda _key, _value: None,
        mark_lighting_dirty=lambda: 5,
        begin_managed_edit=MagicMock(),
        end_managed_edit=MagicMock(),
        register_handlers=MagicMock(),
        unregister_handlers=MagicMock(),
    )
    addon = ModuleType("addon")
    addon.__path__ = []
    addon.dispatcher = dispatcher
    addon.spatial_cache = spatial
    handlers = ModuleType("addon.handlers")
    handlers.__path__ = []
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    monkeypatch.setitem(sys.modules, "addon", addon)
    monkeypatch.setitem(sys.modules, "addon.dispatcher", dispatcher)
    monkeypatch.setitem(sys.modules, "addon.spatial_cache", spatial)
    monkeypatch.setitem(sys.modules, "addon.handlers", handlers)
    spec = importlib.util.spec_from_file_location("addon.handlers.relighting", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "addon.handlers.relighting", module)
    spec.loader.exec_module(module)
    return module, fake_bpy, spatial, dispatcher


def _light_spec(**overrides):
    value = {
        "id": "red_key",
        "type": "AREA",
        "location_world": [1.0, 2.0, 3.0],
        "energy": 750.0,
        "color_rgb": [1.0, 0.0, 0.0],
        "area_shape": "RECTANGLE",
        "size": 2.0,
        "size_y": 1.0,
    }
    value.update(overrides)
    return value


def _point_spec(light_id: str, **overrides):
    value = {
        "id": light_id,
        "name": light_id,
        "type": "POINT",
        "location_world": [1.0, 2.0, 3.0],
        "energy": 250.0,
        "color_rgb": [1.0, 0.1, 0.05],
        "radius": 0.1,
    }
    value.update(overrides)
    return value


def _triangle_candidate_fixture(*, object_name, material_name):
    mesh = FakeID(
        vertices=[
            FakeID(co=(0.0, 0.0, 0.0)),
            FakeID(co=(2.0, 0.0, 0.0)),
            FakeID(co=(0.0, 2.0, 0.0)),
        ],
        polygons=[FakeID(material_index=0)],
        materials=[FakeID(name=material_name)],
        loop_triangles=[FakeID(vertices=(0, 1, 2), polygon_index=0)],
        calc_loop_triangles=lambda: None,
    )
    evaluated = FakeID(
        to_mesh=lambda **_kwargs: mesh,
        to_mesh_clear=MagicMock(),
    )
    source = FakeID(name=object_name, type="MESH")
    matrix = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return evaluated, source, matrix


@pytest.mark.parametrize(
    ("object_name", "material_name"),
    [
        ("Warehouse_Floor", "Concrete Floor"),
        ("North_Wall", "Painted Wall"),
        ("Roof_Panel", "Roof Membrane"),
        ("Ceiling_Beam", "Steel"),
    ],
)
def test_triangle_candidates_keep_generic_semantics_out_of_openings(
    monkeypatch, object_name, material_name
):
    module, _bpy, _spatial, _dispatcher = _load(monkeypatch)
    evaluated, source, matrix = _triangle_candidate_fixture(
        object_name=object_name,
        material_name=material_name,
    )

    surfaces, openings, scanned, warnings = module._triangle_candidates(
        evaluated,
        source,
        matrix,
        "instance-1",
        10,
        ["skylight", "roof", "window", "glass", "opening", "ceiling", "wall", "floor", "beam"],
    )

    assert scanned == 1
    assert warnings == []
    assert len(surfaces) == 1
    assert surfaces[0]["semantic_matches"]
    assert openings == []


def test_triangle_candidates_mark_only_aperture_semantics_as_heuristic_openings(monkeypatch):
    module, _bpy, _spatial, _dispatcher = _load(monkeypatch)
    evaluated, source, matrix = _triangle_candidate_fixture(
        object_name="North_Skylight",
        material_name="Roof Glass",
    )

    surfaces, openings, scanned, warnings = module._triangle_candidates(
        evaluated,
        source,
        matrix,
        "instance-2",
        10,
        ["skylight", "roof", "glass"],
    )

    assert scanned == 1
    assert warnings == []
    assert len(surfaces) == 1
    assert surfaces[0]["semantic_matches"] == ["skylight", "roof", "glass"]
    assert len(openings) == 1
    opening = openings[0]
    assert opening["heuristic"] is True
    assert opening["heuristic_reasons"] == ["skylight", "glass"]
    assert opening["reasons"] == ["skylight", "glass"]
    assert 0.0 < opening["confidence"] <= 0.95


def test_context_prioritizes_late_skylight_before_irrelevant_triangle_budget(
    monkeypatch,
):
    module, fake_bpy, _spatial, _dispatcher = _load(monkeypatch)
    irrelevant = FakeID(name="Irrelevant_Highpoly", type="MESH")
    skylight = FakeID(name="North_Skylight", type="MESH")
    fake_bpy.context._depsgraph.object_instances = [
        FakeID(object=FakeID(original=irrelevant), persistent_id=(1,)),
        FakeID(object=FakeID(original=skylight), persistent_id=(2,)),
    ]
    bounds = {
        "min": [0.0, 0.0, 0.0],
        "max": [1.0, 1.0, 0.1],
        "center": [0.5, 0.5, 0.05],
        "size": [1.0, 1.0, 0.1],
    }
    monkeypatch.setattr(module, "_bounds", lambda *_args: bounds)
    monkeypatch.setattr(module, "_is_visible", lambda *_args: (True, True))
    monkeypatch.setattr(
        module,
        "_instance_record",
        lambda _instance, _evaluated, source, *_args, **_kwargs: {
            "revision_scoped_id": source.name,
            "material_names": [],
            "semantic_matches": (
                ["skylight"] if "Skylight" in source.name else []
            ),
        },
    )
    scan_order = []

    def scan(_evaluated, source, _matrix, instance_id, budget, _terms):
        scan_order.append(source.name)
        candidate = {
            "object_id": instance_id,
            "area": 1.0,
            "heuristic": True,
            "confidence": 0.9,
            "reasons": ["skylight"],
        }
        if source is skylight:
            return [candidate], [candidate], 1, []
        return [], [], budget, []

    monkeypatch.setattr(module, "_triangle_candidates", scan)

    result = module.handle_get_lighting_context(
        {
            "scope": "SCENE",
            "detail": "CANDIDATES",
            "semantic_terms": ["skylight"],
            "max_surface_triangles": 1,
            "cache_mode": "REFRESH",
        }
    )

    assert scan_order == ["North_Skylight"]
    assert result["candidates"]["openings"][0]["object_id"].startswith(
        "North_Skylight"
    )


def test_camera_projection_uses_exact_transformed_evaluated_corners(monkeypatch):
    module, _bpy, _spatial, _dispatcher = _load(monkeypatch)
    local_corners = [
        (x, y, z)
        for x in (-5.0, 5.0)
        for y in (-0.1, 0.1)
        for z in (-0.1, 0.1)
    ]
    angle = math.radians(45.0)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    matrix = [
        [cosine, 0.0, sine, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-sine, 0.0, cosine, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    bounds = module._bounds(FakeID(bound_box=local_corners), matrix)
    assert bounds is not None

    def project(_scene, _camera, point):
        depth = 20.0 - point[2]
        return SimpleNamespace(
            x=0.5 + point[0] / depth,
            y=0.5 + point[1] / depth,
            z=depth,
        )

    object_utils = ModuleType("bpy_extras.object_utils")
    object_utils.world_to_camera_view = project
    bpy_extras = ModuleType("bpy_extras")
    bpy_extras.object_utils = object_utils
    mathutils = ModuleType("mathutils")
    mathutils.Vector = lambda value: value
    monkeypatch.setitem(sys.modules, "bpy_extras", bpy_extras)
    monkeypatch.setitem(sys.modules, "bpy_extras.object_utils", object_utils)
    monkeypatch.setitem(sys.modules, "mathutils", mathutils)

    projection = module._project_bounds(
        FakeID(),
        FakeID(data=FakeID(type="PERSP")),
        bounds,
    )
    independently_projected = [
        project(None, None, point) for point in bounds["evaluated_corners"]
    ]
    expected = [
        min(point.x for point in independently_projected),
        min(point.y for point in independently_projected),
        max(point.x for point in independently_projected),
        max(point.y for point in independently_projected),
    ]

    assert projection["screen_rect"] == pytest.approx(expected)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True])
def test_light_plan_rejects_non_finite_or_boolean_energy(monkeypatch, bad):
    module, _bpy, _spatial, _dispatcher = _load(monkeypatch)

    with pytest.raises(ValueError, match="finite number"):
        module.handle_apply_light_plan(
            {"action": "VALIDATE", "plan_id": "test", "lights": [_light_spec(energy=bad)]}
        )


def test_validation_rejects_duplicate_ids_without_mutating_blender(monkeypatch):
    module, fake_bpy, _spatial, _dispatcher = _load(monkeypatch)

    with pytest.raises(ValueError, match="duplicate ids"):
        module.handle_apply_light_plan(
            {
                "action": "VALIDATE",
                "plan_id": "test",
                "lights": [_light_spec(), _light_spec()],
            }
        )

    fake_bpy.data.lights.new.assert_not_called()


def test_validate_returns_normalized_plan_without_creating_data(monkeypatch):
    module, fake_bpy, _spatial, _dispatcher = _load(monkeypatch)

    result = module.handle_apply_light_plan(
        {
            "action": "VALIDATE",
            "plan_id": "night_warehouse",
            "lights": [_light_spec()],
            "expected_geometry_revision": 3,
        }
    )

    assert result["valid"] is True
    assert result["plan"]["lights"][0]["location_world"] == [1.0, 2.0, 3.0]
    fake_bpy.data.lights.new.assert_not_called()


def test_validate_rejects_unmarked_managed_collection_collision(monkeypatch):
    module, fake_bpy, _spatial, _dispatcher = _load(monkeypatch)
    collision = FakeID(name="AI_RELIGHT", objects=[])
    fake_bpy.data.collections = FakeID(get=lambda _name: collision)

    with pytest.raises(ValueError, match="not marked"):
        module.handle_apply_light_plan(
            {
                "action": "VALIDATE",
                "plan_id": "night",
                "lights": [_light_spec()],
            }
        )


def test_validate_rejects_managed_collection_linked_only_to_another_scene(
    monkeypatch,
):
    managed = FakeID(name="AI_RELIGHT", objects=[], children=[])
    managed["blend_ai_relight_managed"] = True
    active_root = FakeID(name="Active Root", children=[])
    foreign_root = FakeID(name="Foreign Root", children=[managed])
    active_scene = FakeID(
        name="Active",
        frame_current=1,
        objects=[],
        collection=active_root,
    )
    foreign_scene = FakeID(
        name="Foreign",
        frame_current=1,
        objects=[],
        collection=foreign_root,
    )
    module, fake_bpy, _spatial, _dispatcher = _load(
        monkeypatch,
        scene=active_scene,
    )
    fake_bpy.data.collections = FakeNamedValues([managed])
    fake_bpy.data.scenes = [active_scene, foreign_scene]

    with pytest.raises(ValueError, match="linked only to another scene"):
        module.handle_apply_light_plan(
            {"action": "VALIDATE", "plan_id": "night", "lights": []}
        )


def test_validate_rejects_managed_collection_shared_between_scenes(monkeypatch):
    managed = FakeID(name="AI_RELIGHT", objects=[], children=[])
    managed["blend_ai_relight_managed"] = True
    active_root = FakeID(name="Active Root", children=[managed])
    foreign_root = FakeID(name="Foreign Root", children=[managed])
    active_scene = FakeID(
        name="Active",
        frame_current=1,
        objects=[],
        collection=active_root,
    )
    foreign_scene = FakeID(
        name="Foreign",
        frame_current=1,
        objects=[],
        collection=foreign_root,
    )
    module, fake_bpy, _spatial, _dispatcher = _load(
        monkeypatch,
        scene=active_scene,
    )
    fake_bpy.data.collections = FakeNamedValues([managed])
    fake_bpy.data.scenes = [active_scene, foreign_scene]

    with pytest.raises(ValueError, match="shared with other scenes"):
        module.handle_apply_light_plan(
            {"action": "VALIDATE", "plan_id": "night", "lights": []}
        )


def test_plan_hard_limits_managed_lights(monkeypatch):
    module, _bpy, _spatial, _dispatcher = _load(monkeypatch)
    lights = [_light_spec(id=f"red_{index}") for index in range(129)]

    with pytest.raises(ValueError, match="at most 128"):
        module.handle_apply_light_plan(
            {"action": "VALIDATE", "plan_id": "night", "lights": lights}
        )


def test_managed_light_requires_exact_markers_on_object_and_data(monkeypatch):
    module, _bpy, _spatial, _dispatcher = _load(monkeypatch)
    data = FakeID(type="AREA")
    obj = FakeID(type="LIGHT", data=data)
    obj[module.MANAGED_PROP] = True
    obj[module.MANAGED_ID_PROP] = "key"

    assert module._is_managed_light(obj) is False
    data[module.MANAGED_PROP] = True
    data[module.MANAGED_ID_PROP] = "different"
    assert module._is_managed_light(obj) is False
    data[module.MANAGED_ID_PROP] = "key"
    assert module._is_managed_light(obj) is True


@pytest.mark.parametrize(
    ("unsupported_state", "message"),
    [
        ("parent", "unsupported parenting"),
        ("object_custom", "unsupported custom properties"),
        ("data_custom", "unsupported custom properties"),
        ("scale", "unsupported non-default scale"),
        ("object_action", "unsupported constraints or animation"),
        ("data_action", "unsupported constraints or animation"),
        ("delta_location", "unsupported delta transforms"),
        ("delta_rotation", "unsupported delta transforms"),
        ("delta_scale", "unsupported delta transforms"),
        ("axis_angle", "unsupported axis-angle rotation"),
        ("nodes", "unsupported node state"),
    ],
)
def test_strict_preflight_rejects_managed_state_rollback_cannot_snapshot(
    monkeypatch,
    unsupported_state,
    message,
):
    module, _fake_bpy, _spatial, _dispatcher = _load(monkeypatch)
    _runtime, scene = _managed_runtime(module)
    module._ledger.clear()
    module._ledger_loaded = True
    module.handle_apply_light_plan(
        {
            "action": "APPLY",
            "plan_id": "warehouse_night",
            "lights": [
                {
                    "id": "red_point",
                    "type": "POINT",
                    "location_world": [1.0, 2.0, 3.0],
                    "energy": 500.0,
                    "color_rgb": [1.0, 0.0, 0.0],
                }
            ],
        }
    )
    obj = scene.objects[0]
    if unsupported_state == "parent":
        obj.parent = FakeID(name="Artist_Rig")
    elif unsupported_state == "object_custom":
        obj["artist_note"] = "preserve me"
    elif unsupported_state == "data_custom":
        obj.data["third_party_light_profile"] = "preserve me"
    elif unsupported_state == "scale":
        obj.scale = (2.0, 1.0, 1.0)
    elif unsupported_state == "object_action":
        obj.animation_data = FakeID(action=FakeID(name="Object_Action"))
    elif unsupported_state == "data_action":
        obj.data.animation_data = FakeID(action=FakeID(name="Light_Action"))
    elif unsupported_state == "delta_location":
        obj.delta_location = (0.25, 0.0, 0.0)
    elif unsupported_state == "delta_rotation":
        obj.delta_rotation_euler = (0.0, 0.1, 0.0)
    elif unsupported_state == "delta_scale":
        obj.delta_scale = (1.0, 0.5, 1.0)
    elif unsupported_state == "axis_angle":
        obj.rotation_mode = "AXIS_ANGLE"
    else:
        obj.data.use_nodes = True

    with pytest.raises(ValueError, match=message):
        module.handle_apply_light_plan(
            {
                "action": "VALIDATE",
                "plan_id": "warehouse_night",
                "lights": [],
                "strict": True,
            }
        )


def test_world_override_never_reuses_arbitrary_managed_world(monkeypatch):
    scene = FakeID(
        name="Scene",
        frame_current=1,
        objects=[],
        world=FakeID(name="Original_World"),
    )
    module, fake_bpy, _spatial, _dispatcher = _load(monkeypatch, scene=scene)
    arbitrary = FakeID(name=module.MANAGED_WORLD, users=1)
    arbitrary[module.MANAGED_PROP] = True
    fresh = FakeID(name=f"{module.MANAGED_WORLD}.001", users=0)
    worlds = FakeNamedValues([scene.world, arbitrary])

    def new_world(_name):
        worlds.append(fresh)
        return fresh

    worlds.new = MagicMock(side_effect=new_world)
    fake_bpy.data.worlds = worlds
    set_world = MagicMock()
    monkeypatch.setattr(module, "_set_managed_world", set_world)

    module._apply_world_override(
        scene,
        {"mode": "MANAGED_SOLID", "color_rgb": [0.0, 0.0, 0.02], "strength": 0.1},
        "night",
    )

    assert scene.world is fresh
    worlds.new.assert_called_once_with(module.MANAGED_WORLD)
    assert set_world.call_args.args[0] is fresh


def test_world_rollback_restores_pointer_without_rewriting_shared_world(monkeypatch):
    original = FakeID(name="Earlier_Managed_World", users=1)
    original["blend_ai_relight_managed"] = True
    current = FakeID(name="Current_Transaction_World", users=0)
    current["blend_ai_relight_managed"] = True
    scene = FakeID(name="Scene", frame_current=1, objects=[], world=current)
    module, fake_bpy, _spatial, _dispatcher = _load(monkeypatch, scene=scene)
    worlds = FakeNamedValues([original, current])
    worlds.remove = MagicMock()
    fake_bpy.data.worlds = worlds
    set_world = MagicMock()
    monkeypatch.setattr(module, "_set_managed_world", set_world)

    module._restore_world(
        scene,
        {
            "name": "Earlier_Managed_World",
            "managed": True,
            "plan_id": "older",
            "color_rgb": [0.1, 0.1, 0.1],
            "background_color_rgb": [0.2, 0.2, 0.2],
            "strength": 0.5,
        },
    )

    assert scene.world is original
    set_world.assert_not_called()
    worlds.remove.assert_called_once_with(current)


def test_batch_raycast_advances_through_ignored_layers(monkeypatch):
    glass = FakeID(name="GlassPane", type="MESH", data=FakeID(polygons=[]), material_slots=[])
    wall = FakeID(name="BackWall", type="MESH", data=FakeID(polygons=[]), material_slots=[])
    casts = iter(
        [
            (True, (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), -1, glass, None),
            (True, (2.0, 0.0, 0.0), (-1.0, 0.0, 0.0), -1, wall, None),
            (False, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), -1, None, None),
        ]
    )
    scene = FakeID(
        name="Scene",
        frame_current=1,
        objects=[glass, wall],
        ray_cast=lambda *_args, **_kwargs: next(casts),
    )
    module, _bpy, _spatial, _dispatcher = _load(monkeypatch, scene=scene)

    result = module.handle_batch_raycast(
        {
            "rays": [
                {"id": "clearance", "origin": [0, 0, 0], "direction": [1, 0, 0], "max_distance": 10}
            ],
            "ignore_object_patterns": ["glass*"],
        }
    )

    ray = result["rays"][0]
    assert ray["termination"] == "BLOCKED"
    assert ray["ignored_encounters"] == 1
    assert [hit["object_name"] for hit in ray["hits"]] == ["GlassPane", "BackWall"]
    assert ray["hits"][0]["ignored"] is True
    assert ray["hits"][1]["ignored"] is False


def test_batch_raycast_advances_past_ignored_geometry_at_large_coordinates(
    monkeypatch,
):
    glass = FakeID(
        name="GlassPane",
        type="MESH",
        data=FakeID(polygons=[]),
        material_slots=[],
    )
    wall = FakeID(
        name="BackWall",
        type="MESH",
        data=FakeID(polygons=[]),
        material_slots=[],
    )
    origins = []

    def ray_cast(_depsgraph, origin, _direction, distance):
        origins.append(tuple(origin))
        if len(origins) == 1:
            return (
                True,
                (100_000_010.0, 0.0, 0.0),
                (-1.0, 0.0, 0.0),
                -1,
                glass,
                None,
            )
        assert distance < 1000.0
        return (
            True,
            (100_000_100.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            -1,
            wall,
            None,
        )

    scene = FakeID(
        name="Scene",
        frame_current=1,
        objects=[glass, wall],
        unit_settings=FakeID(scale_length=1.0),
        ray_cast=ray_cast,
    )
    module, _bpy, _spatial, _dispatcher = _load(monkeypatch, scene=scene)

    result = module.handle_batch_raycast(
        {
            "rays": [
                {
                    "id": "large_world",
                    "origin": [100_000_000.0, 0.0, 0.0],
                    "direction": [1.0, 0.0, 0.0],
                    "max_distance": 1000.0,
                }
            ],
            "ignore_object_patterns": ["glass*"],
        }
    )

    assert origins[1][0] - 100_000_010.0 >= 10.0
    assert result["rays"][0]["termination"] == "BLOCKED"
    assert [hit["object_name"] for hit in result["rays"][0]["hits"]] == [
        "GlassPane",
        "BackWall",
    ]


def test_large_coordinate_epsilon_does_not_skip_initial_short_ray(monkeypatch):
    wall = FakeID(
        name="NearbyWall",
        type="MESH",
        data=FakeID(polygons=[]),
        material_slots=[],
    )
    ray_cast = MagicMock(
        return_value=(
            True,
            (1_000_000_000.5, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            -1,
            wall,
            None,
        )
    )
    scene = FakeID(
        name="Scene",
        frame_current=1,
        objects=[wall],
        unit_settings=FakeID(scale_length=1.0),
        ray_cast=ray_cast,
    )
    module, _bpy, _spatial, _dispatcher = _load(monkeypatch, scene=scene)

    result = module.handle_batch_raycast(
        {
            "rays": [
                {
                    "id": "short_large_world",
                    "origin": [1_000_000_000.0, 0.0, 0.0],
                    "direction": [1.0, 0.0, 0.0],
                    "max_distance": 1.0,
                }
            ]
        }
    )

    ray_cast.assert_called_once()
    assert result["rays"][0]["termination"] == "BLOCKED"


def test_batch_raycast_max_hits_is_explicitly_incomplete(monkeypatch):
    glass = FakeID(
        name="GlassPane",
        type="MESH",
        data=FakeID(polygons=[]),
        material_slots=[],
    )
    wall = FakeID(
        name="OpaqueWall",
        type="MESH",
        data=FakeID(polygons=[]),
        material_slots=[],
    )
    casts = iter(
        [
            (True, (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), -1, glass, None),
            (True, (2.0, 0.0, 0.0), (-1.0, 0.0, 0.0), -1, glass, None),
            (True, (3.0, 0.0, 0.0), (-1.0, 0.0, 0.0), -1, wall, None),
        ]
    )
    scene = FakeID(
        name="Scene",
        frame_current=1,
        objects=[glass, wall],
        ray_cast=lambda *_args, **_kwargs: next(casts),
    )
    module, _bpy, _spatial, _dispatcher = _load(monkeypatch, scene=scene)

    result = module.handle_batch_raycast(
        {
            "rays": [
                {
                    "id": "layered",
                    "origin": [0, 0, 0],
                    "target": [10, 0, 0],
                }
            ],
            "ignore_object_patterns": ["glass*"],
            "max_hits": 2,
        }
    )

    ray = result["rays"][0]
    assert result["complete"] is False
    assert ray["complete"] is False
    assert ray["termination"] == "MAX_HITS"
    assert ray["clear_to_target"] is None


def test_batch_raycast_time_budget_returns_all_ids_as_unknown_partial(monkeypatch):
    ray_cast = MagicMock()
    scene = FakeID(
        name="Scene",
        frame_current=1,
        objects=[],
        ray_cast=ray_cast,
    )
    module, _bpy, _spatial, _dispatcher = _load(monkeypatch, scene=scene)
    monkeypatch.setattr(
        module.time,
        "perf_counter",
        MagicMock(side_effect=[0.0, 1.0, 1.1]),
    )

    result = module.handle_batch_raycast(
        {
            "rays": [
                {"id": "first", "origin": [0, 0, 0], "target": [1, 0, 0]},
                {"id": "second", "origin": [0, 0, 0], "target": [2, 0, 0]},
            ],
            "time_budget_ms": 10,
        }
    )

    assert result["complete"] is False
    assert [ray["id"] for ray in result["rays"]] == ["first", "second"]
    assert all(ray["complete"] is False for ray in result["rays"])
    assert all(ray["termination"] == "TIME_BUDGET" for ray in result["rays"])
    assert all(ray["clear_to_target"] is None for ray in result["rays"])
    ray_cast.assert_not_called()


def test_target_raycast_uses_endpoint_tolerance(monkeypatch):
    distances = []

    def ray_cast(*_args, distance):
        distances.append(distance)
        return (False, (0, 0, 0), (0, 0, 0), -1, None, None)

    scene = FakeID(name="Scene", frame_current=1, objects=[], ray_cast=ray_cast)
    module, _bpy, _spatial, _dispatcher = _load(monkeypatch, scene=scene)

    result = module.handle_batch_raycast(
        {"rays": [{"id": "endpoint", "origin": [0, 0, 0], "target": [1, 0, 0]}]}
    )

    ray = result["rays"][0]
    assert distances[0] < 1.0
    assert ray["clear_to_target"] is True
    assert ray["termination"] == "TARGET_REACHED"
    assert ray["endpoint_tolerance"] > 0.0


def test_rollback_can_resolve_latest_transaction_by_plan(monkeypatch):
    module, fake_bpy, _spatial, _dispatcher = _load(monkeypatch)
    module._ledger_loaded = True
    module._ledger_source_token = (id(fake_bpy.data.texts), "")
    module._ledger.clear()
    module._ledger["old"] = {"transaction_id": "old", "plan_id": "night"}
    module._ledger["new"] = {"transaction_id": "new", "plan_id": "night"}

    key, entry = module._find_transaction(None, "night")

    assert key == "new"
    assert entry["transaction_id"] == "new"


def test_rollback_enforces_global_reverse_chronological_order(monkeypatch):
    module, fake_bpy, _spatial, _dispatcher = _load(monkeypatch)
    module._ledger_loaded = True
    module._ledger_source_token = (id(fake_bpy.data.texts), "")
    module._ledger.clear()
    module._ledger["old"] = {
        "transaction_id": "old",
        "plan_id": "night",
    }
    module._ledger["new"] = {
        "transaction_id": "new",
        "plan_id": "accent",
    }

    with pytest.raises(ValueError, match="out of order"):
        module._find_transaction("old", None)
    with pytest.raises(ValueError, match="roll it back first"):
        module._find_transaction(None, "night")


def test_ledger_reloads_only_transactions_from_new_file(monkeypatch):
    module, fake_bpy, _spatial, _dispatcher = _load(monkeypatch)

    def texts_for(transaction_id):
        payload = json.dumps(
            {
                "schema_version": module.LEDGER_SCHEMA_VERSION,
                "transactions": [
                    {
                        "transaction_id": transaction_id,
                        "plan_id": "night",
                    }
                ],
            }
        )
        text = FakeID(name=module.LEDGER_TEXT, as_string=lambda: payload)
        text[module.LEDGER_PROP] = True
        return FakeNamedValues([text])

    fake_bpy.data.filepath = "/tmp/file-a.blend"
    fake_bpy.data.texts = texts_for("file-a")
    key, _entry = module._find_transaction("file-a", None)
    assert key == "file-a"

    fake_bpy.data.filepath = "/tmp/file-b.blend"
    fake_bpy.data.texts = texts_for("file-b")
    key, _entry = module._find_transaction("file-b", None)
    assert key == "file-b"
    assert list(module._ledger) == ["file-b"]


def test_register_exposes_exact_commands(monkeypatch):
    module, _bpy, spatial, dispatcher = _load(monkeypatch)

    module.register()

    commands = [call.args[0] for call in dispatcher.register_handler.call_args_list]
    assert commands == ["get_lighting_context", "batch_raycast", "apply_light_plan"]
    spatial.register_handlers.assert_called_once_with()


def test_file_load_and_unregister_clear_process_local_ledger(monkeypatch):
    module, fake_bpy, spatial, _dispatcher = _load(monkeypatch)
    load_post = []
    fake_bpy.app = FakeID(handlers=FakeID(load_post=load_post))
    module._ledger["old-file"] = {
        "transaction_id": "old-file",
        "plan_id": "night",
    }
    module._ledger_loaded = True

    module.register()
    assert module._clear_ledger_on_load in load_post

    module._clear_ledger_on_load(None, object())
    assert module._ledger == {}
    assert module._ledger_loaded is False

    module._ledger["current-file"] = {
        "transaction_id": "current-file",
        "plan_id": "night",
    }
    module._ledger_loaded = True
    module.unregister()
    assert module._clear_ledger_on_load not in load_post
    assert module._ledger == {}
    assert module._ledger_loaded is False
    spatial.unregister_handlers.assert_called_once_with()


def test_payload_fitter_keeps_response_below_absolute_limit(monkeypatch):
    module, _bpy, _spatial, _dispatcher = _load(monkeypatch)
    result = {
        "scene": {},
        "collections": [],
        "instances": [{"id": str(index), "blob": "x" * 20_000} for index in range(100)],
        "lights": [],
        "candidates": {"surfaces": [], "openings": []},
        "next_cursor": None,
        "truncated": False,
        "warnings": [],
        "timings_ms": {},
    }

    module._fit_context_payload(
        result,
        offset=0,
        total=100,
        geometry_revision=3,
        signature="query",
    )

    assert module._payload_size(result) < module.MAX_PAYLOAD_BYTES
    assert result["truncated"] is True
    assert result["next_cursor"] is not None


def test_context_cache_hit_refreshes_render_summary_and_age(monkeypatch):
    scene = FakeID(
        name="Scene",
        frame_current=1,
        objects=[],
        camera=None,
        world=None,
        render=FakeID(engine="BLENDER_EEVEE_NEXT"),
        cycles=FakeID(device="CPU", preview_samples=8),
        unit_settings=FakeID(system="METRIC", scale_length=1.0),
        view_settings=FakeID(exposure=0.0),
    )
    module, _fake_bpy, spatial, _dispatcher = _load(monkeypatch, scene=scene)
    cache = {}
    spatial.get_cached = lambda key: cache.get(key)
    spatial.put_cached = lambda key, value: cache.__setitem__(key, value)
    spatial.get_cache_age_ms = lambda _key: 12.5

    cold = module.handle_get_lighting_context(
        {"scope": "SCENE", "detail": "BOUNDS", "cache_mode": "REFRESH"}
    )
    assert cold["scene"]["render_engine"] == "BLENDER_EEVEE_NEXT"

    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.preview_samples = 32
    warm = module.handle_get_lighting_context(
        {"scope": "SCENE", "detail": "BOUNDS", "cache_mode": "USE"}
    )

    assert warm["cache"]["hit"] is True
    assert warm["cache"]["age_ms"] == 12.5
    assert warm["scene"]["render_engine"] == "CYCLES"
    assert warm["scene"]["cycles_device"] == "GPU"
    assert warm["scene"]["preview_samples"] == 32


def test_context_returns_canonical_shape_and_complete_collection_bounds(monkeypatch):
    warehouse = FakeID(name="Warehouse", children=[])
    root_collection = FakeID(name="Scene Collection", children=[warehouse])
    source = FakeID(
        name="Floor",
        name_full="Floor",
        type="MESH",
        users_collection=[warehouse],
        hide_viewport=False,
        hide_render=False,
        parent=None,
    )
    matrix = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    evaluated = FakeID(
        name="Floor",
        type="MESH",
        original=source,
        matrix_world=matrix,
        bound_box=[
            (-2, -3, 0),
            (2, -3, 0),
            (-2, 3, 0),
            (2, 3, 0),
            (-2, -3, 0.2),
            (2, -3, 0.2),
            (-2, 3, 0.2),
            (2, 3, 0.2),
        ],
    )
    instance = FakeID(
        object=evaluated,
        matrix_world=matrix,
        is_instance=False,
        persistent_id=(0,) * 8,
        parent=None,
    )
    scene = FakeID(
        name="Scene",
        frame_current=1,
        objects=[source],
        collection=root_collection,
        camera=None,
        world=None,
    )
    module, fake_bpy, _spatial, _dispatcher = _load(monkeypatch, scene=scene)
    fake_bpy.context._depsgraph.object_instances = [instance]

    result = module.handle_get_lighting_context(
        {"scope": "SCENE", "detail": "BOUNDS", "cache_mode": "REFRESH"}
    )

    assert set(
        [
            "scene",
            "collections",
            "instances",
            "lights",
            "candidates",
            "next_cursor",
            "truncated",
            "warnings",
            "timings_ms",
        ]
    ).issubset(result)
    assert result["geometry_revision"] == 3
    assert result["lighting_revision"] == 4
    assert result["schema_version"] == 1
    bounds_by_name = {item["name"]: item["bounds"] for item in result["collections"]}
    assert bounds_by_name["Warehouse"]["size"] == [4.0, 6.0, 0.2]
    assert bounds_by_name["Scene Collection"]["size"] == [4.0, 6.0, 0.2]


def test_collection_scope_includes_instancer_parent_collection_ancestry(monkeypatch):
    host_collection = FakeID(name="Warehouse_Host", children=[])
    root_collection = FakeID(name="Scene Collection", children=[host_collection])
    asset_collection = FakeID(name="Unlinked_Asset_Source", children=[])
    source = FakeID(
        name="Repeated_Fixture",
        name_full="Repeated_Fixture",
        type="MESH",
        users_collection=[asset_collection],
        hide_viewport=False,
        hide_render=False,
        parent=None,
    )
    instancer_source = FakeID(
        name="Warehouse_GN_Instancer",
        users_collection=[host_collection],
        parent=None,
    )
    instancer_evaluated = FakeID(
        name="Warehouse_GN_Instancer",
        original=instancer_source,
        users_collection=[],
        parent=None,
    )
    matrix = [
        [1.0, 0.0, 0.0, 4.0],
        [0.0, 1.0, 0.0, 5.0],
        [0.0, 0.0, 1.0, 6.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    evaluated = FakeID(
        name="Repeated_Fixture",
        type="MESH",
        original=source,
        matrix_world=matrix,
        bound_box=[
            (x, y, z)
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for z in (-1.0, 1.0)
        ],
    )
    instance = FakeID(
        object=evaluated,
        matrix_world=matrix,
        is_instance=True,
        persistent_id=(7, 2, 0, 0),
        parent=instancer_evaluated,
    )
    scene = FakeID(
        name="Scene",
        frame_current=1,
        objects=[instancer_source],
        collection=root_collection,
        camera=None,
        world=None,
    )
    module, fake_bpy, _spatial, _dispatcher = _load(monkeypatch, scene=scene)
    fake_bpy.context._depsgraph.object_instances = [instance]

    result = module.handle_get_lighting_context(
        {
            "scope": "COLLECTIONS",
            "collection_names": ["Warehouse_Host"],
            "detail": "BOUNDS",
            "cache_mode": "REFRESH",
        }
    )

    assert len(result["instances"]) == 1
    record = result["instances"][0]
    assert "Warehouse_Host" in record["collection_ids"]
    assert "Scene Collection" in record["collection_ids"]
    assert "Unlinked_Asset_Source" in record["collection_ids"]
    assert result["collections"][0]["bounds"]["center"] == [4.0, 5.0, 6.0]


def test_raycast_matrix_resolves_same_distinct_context_id_for_repeated_instances(
    monkeypatch,
):
    source = FakeID(
        name="RepeatedPanel",
        name_full="RepeatedPanel",
        type="MESH",
        data=FakeID(polygons=[]),
        material_slots=[],
    )
    evaluated = FakeID(
        name="RepeatedPanel",
        name_full="RepeatedPanel",
        original=source,
        type="MESH",
        data=FakeID(polygons=[]),
        material_slots=[],
    )
    parent_a = FakeID(name="Instancer_A")
    parent_b = FakeID(name="Instancer_B")
    matrix_a = [
        [1.0, 0.0, 0.0, 10.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    matrix_b = [
        [1.0, 0.0, 0.0, 20.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    instance_a = FakeID(
        object=evaluated,
        matrix_world=matrix_a,
        parent=parent_a,
        persistent_id=(1, 0, 0),
    )
    instance_b = FakeID(
        object=evaluated,
        matrix_world=matrix_b,
        parent=parent_b,
        persistent_id=(2, 0, 0),
    )
    scene = FakeID(
        name="Scene",
        frame_current=1,
        objects=[source],
        ray_cast=lambda *_args, **_kwargs: (
            True,
            (20.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            -1,
            evaluated,
            matrix_b,
        ),
    )
    module, fake_bpy, _spatial, _dispatcher = _load(monkeypatch, scene=scene)
    fake_bpy.context._depsgraph.object_instances = [instance_a, instance_b]
    monkeypatch.setattr(
        module,
        "_iter_values",
        MagicMock(
            side_effect=AssertionError(
                "DepsgraphObjectInstance wrappers must not be materialized as a list"
            )
        ),
    )

    result = module.handle_batch_raycast(
        {
            "rays": [
                {
                    "id": "instance_b",
                    "origin": [0.0, 0.0, 0.0],
                    "direction": [1.0, 0.0, 0.0],
                    "max_distance": 100.0,
                }
            ]
        }
    )

    expected_b = module._instance_identifier(instance_b, source)
    assert expected_b != module._instance_identifier(instance_a, source)
    assert result["rays"][0]["hits"][0]["object_id"] == expected_b


def test_validate_reports_exact_plan_scoped_proposal_and_replace_preserves_other_plan(
    monkeypatch,
):
    module, _fake_bpy, _spatial, _dispatcher = _load(monkeypatch)
    _runtime, scene = _managed_runtime(module)
    module._ledger.clear()
    module._ledger_loaded = True

    module.handle_apply_light_plan(
        {
            "action": "APPLY",
            "plan_id": "alpha",
            "mode": "REPLACE_MANAGED",
            "lights": [_point_spec("alpha_old")],
        }
    )
    module.handle_apply_light_plan(
        {
            "action": "APPLY",
            "plan_id": "beta",
            "mode": "REPLACE_MANAGED",
            "lights": [_point_spec("beta_keep")],
        }
    )

    request = {
        "action": "VALIDATE",
        "plan_id": "alpha",
        "mode": "REPLACE_MANAGED",
        "lights": [_point_spec("alpha_new")],
    }
    validated = module.handle_apply_light_plan(request)
    assert validated["proposed_changes"] == {
        "plan_id": "alpha",
        "mode": "REPLACE_MANAGED",
        "create_ids": ["alpha_new"],
        "update_ids": [],
        "unchanged_ids": [],
        "remove_ids": ["alpha_old"],
        "missing_remove_ids": [],
        "final_plan_light_ids": ["alpha_new"],
        "final_plan_light_count": 1,
        "mute_non_managed_light_names": [],
        "already_muted_non_managed_light_names": [],
        "world_override": {
            "mode": "KEEP",
            "color_rgb": [0.0, 0.0, 0.0],
            "strength": 0.0,
        },
        "world_will_change": False,
        "exposure_override": None,
        "exposure_will_change": False,
    }

    applied = module.handle_apply_light_plan({**request, "action": "APPLY"})
    assert applied["created"] == ["alpha_new"]
    assert applied["removed"] == ["alpha_old"]
    owners = {
        obj[module.MANAGED_ID_PROP]: obj[module.MANAGED_PLAN_PROP]
        for obj in scene.objects
        if module._is_managed_light(obj)
    }
    assert owners == {"alpha_new": "alpha", "beta_keep": "beta"}


def test_patch_cannot_grow_one_plan_above_128_lights(monkeypatch):
    module, _fake_bpy, _spatial, _dispatcher = _load(monkeypatch)
    _runtime, _scene = _managed_runtime(module)
    module._ledger.clear()
    module._ledger_loaded = True
    module.handle_apply_light_plan(
        {
            "action": "APPLY",
            "plan_id": "full_plan",
            "mode": "REPLACE_MANAGED",
            "lights": [_point_spec(f"bulk_{index:03d}") for index in range(128)],
        }
    )

    with pytest.raises(ValueError, match="would contain 129 lights"):
        module.handle_apply_light_plan(
            {
                "action": "VALIDATE",
                "plan_id": "full_plan",
                "mode": "PATCH_MANAGED",
                "lights": [_point_spec("one_too_many")],
            }
        )


def test_validate_rejects_light_id_owned_by_another_plan(monkeypatch):
    module, _fake_bpy, _spatial, _dispatcher = _load(monkeypatch)
    _runtime, _scene = _managed_runtime(module)
    module._ledger.clear()
    module._ledger_loaded = True
    module.handle_apply_light_plan(
        {
            "action": "APPLY",
            "plan_id": "alpha",
            "lights": [_point_spec("shared_id")],
        }
    )

    with pytest.raises(ValueError, match="belongs to plan 'alpha'"):
        module.handle_apply_light_plan(
            {
                "action": "VALIDATE",
                "plan_id": "beta",
                "lights": [_point_spec("shared_id")],
            }
        )


def test_validate_rejects_unmanaged_display_name_collision(monkeypatch):
    module, _fake_bpy, _spatial, _dispatcher = _load(monkeypatch)
    runtime, scene = _managed_runtime(module)
    unmanaged_data = runtime.data.lights.new("artist_light", type="POINT")
    unmanaged = runtime.data.objects.new("artist_light", unmanaged_data)
    scene.collection.objects.link(unmanaged)

    with pytest.raises(ValueError, match="display name 'artist_light' collides"):
        module.handle_apply_light_plan(
            {
                "action": "VALIDATE",
                "plan_id": "night",
                "lights": [
                    _point_spec("managed_artist_light", name="artist_light")
                ],
            }
        )

    assert unmanaged in scene.objects


def test_apply_marks_both_ids_and_explicit_rollback_restores_snapshot(monkeypatch):
    module, _fake_bpy, spatial, _dispatcher = _load(monkeypatch)
    runtime, scene = _managed_runtime(module)
    module._ledger.clear()
    module._ledger_loaded = True

    applied = module.handle_apply_light_plan(
        {
            "action": "APPLY",
            "plan_id": "warehouse_night",
            "lights": [
                {
                    "id": "red_point",
                    "type": "POINT",
                    "location_world": [1.0, 2.0, 3.0],
                    "energy": 500.0,
                    "color_rgb": [1.0, 0.0, 0.0],
                }
            ],
        }
    )

    assert applied["created"] == ["red_point"]
    obj = scene.objects[0]
    assert obj[module.MANAGED_PROP] is True
    assert obj[module.MANAGED_ID_PROP] == "red_point"
    assert obj.data[module.MANAGED_PROP] is True
    assert obj.data[module.MANAGED_ID_PROP] == "red_point"
    assert runtime.data.collections.get(module.MANAGED_COLLECTION)[module.MANAGED_PROP] is True

    rolled_back = module.handle_apply_light_plan(
        {"action": "ROLLBACK", "transaction_id": applied["transaction_id"]}
    )

    assert rolled_back["rolled_back"] is True
    assert scene.objects == []
    assert runtime.data.collections.get(module.MANAGED_COLLECTION) is None
    assert spatial.begin_managed_edit.call_count == 2
    assert spatial.end_managed_edit.call_count == 2
    assert runtime.context.view_layer.update.call_count == 2


def test_rollback_rejects_transaction_from_another_scene(monkeypatch):
    module, _fake_bpy, _spatial, _dispatcher = _load(monkeypatch)
    _runtime, scene = _managed_runtime(module)
    module._ledger.clear()
    module._ledger_loaded = True

    applied = module.handle_apply_light_plan(
        {
            "action": "APPLY",
            "plan_id": "warehouse_night",
            "lights": [
                {
                    "id": "red_point",
                    "type": "POINT",
                    "location_world": [1.0, 2.0, 3.0],
                    "energy": 500.0,
                    "color_rgb": [1.0, 0.0, 0.0],
                }
            ],
        }
    )
    scene.name = "Different Scene"

    with pytest.raises(ValueError, match="belongs to scene"):
        module.handle_apply_light_plan(
            {"action": "ROLLBACK", "transaction_id": applied["transaction_id"]}
        )

    assert applied["transaction_id"] in module._ledger


def test_keep_policy_does_not_restore_unrelated_light_visibility(monkeypatch):
    module, _fake_bpy, _spatial, _dispatcher = _load(monkeypatch)
    runtime, scene = _managed_runtime(module)
    module._ledger.clear()
    module._ledger_loaded = True

    original_data = runtime.data.lights.new("Original Light", type="POINT")
    original = runtime.data.objects.new("Original Light", original_data)
    scene.collection.objects.link(original)
    original.hide_render = False
    original.hide_viewport = False

    applied = module.handle_apply_light_plan(
        {
            "action": "APPLY",
            "plan_id": "warehouse_night",
            "lights": [
                {
                    "id": "red_point",
                    "type": "POINT",
                    "location_world": [1.0, 2.0, 3.0],
                    "energy": 500.0,
                    "color_rgb": [1.0, 0.0, 0.0],
                }
            ],
        }
    )
    assert module._ledger[applied["transaction_id"]]["snapshot"][
        "non_managed_visibility"
    ] == []

    # This is an independent artist edit after the plan; KEEP rollback must
    # not overwrite it.
    original.hide_render = True
    original.hide_viewport = True
    module.handle_apply_light_plan(
        {"action": "ROLLBACK", "transaction_id": applied["transaction_id"]}
    )

    assert original.hide_render is True
    assert original.hide_viewport is True


def test_failed_rollback_restores_safety_snapshot_and_keeps_ledger(monkeypatch):
    module, fake_bpy, _spatial, _dispatcher = _load(monkeypatch)
    module._ledger.clear()
    module._ledger_loaded = True
    module._ledger_source_token = (id(fake_bpy.data.texts), "")
    target = {
        "scene_name": "Scene",
        "collection_name": module.MANAGED_COLLECTION,
        "non_managed_visibility": [],
    }
    entry = {
        "transaction_id": "latest",
        "plan_id": "night",
        "scene_name": "Scene",
        "snapshot": target,
    }
    module._ledger["latest"] = entry
    before_ledger = module._ledger_payload()
    safety = {"safety": True}
    state = {"value": "before"}
    monkeypatch.setattr(module, "_snapshot_scene", lambda *_args, **_kwargs: safety)

    def restore(_scene, snapshot):
        if snapshot is target:
            state["value"] = "partial"
            raise RuntimeError("snapshot world no longer exists")
        assert snapshot is safety
        state["value"] = "before"

    monkeypatch.setattr(module, "_restore_scene", restore)

    with pytest.raises(RuntimeError, match="pre-rollback state was restored"):
        module._rollback({"transaction_id": "latest"})

    assert state["value"] == "before"
    assert module._ledger_payload() == before_ledger
    assert list(module._ledger) == ["latest"]


def test_validate_rejects_duplicate_display_names(monkeypatch):
    module, _bpy, _spatial, _dispatcher = _load(monkeypatch)

    with pytest.raises(ValueError, match="duplicate display names"):
        module.handle_apply_light_plan(
            {
                "action": "VALIDATE",
                "plan_id": "night",
                "lights": [
                    _point_spec("left", name="Same Display Name"),
                    _point_spec("right", name="Same Display Name"),
                ],
            }
        )


def test_strict_preflight_rejects_shared_light_datablock(monkeypatch):
    module, _fake_bpy, _spatial, _dispatcher = _load(monkeypatch)
    _runtime, scene = _managed_runtime(module)
    applied = module.handle_apply_light_plan(
        {
            "action": "APPLY",
            "plan_id": "night",
            "lights": [_point_spec("shared")],
        }
    )
    obj = next(item for item in scene.objects if module._is_managed_light(item))
    obj.data.users = 2

    with pytest.raises(ValueError, match="shares its Light datablock"):
        module.handle_apply_light_plan(
            {
                "action": "VALIDATE",
                "plan_id": "night",
                "lights": [_point_spec("shared")],
                "strict": True,
            }
        )

    assert applied["transaction_id"] in module._ledger


def test_rollback_strict_preflight_rejects_new_unsupported_state(monkeypatch):
    module, _fake_bpy, _spatial, _dispatcher = _load(monkeypatch)
    _runtime, scene = _managed_runtime(module)
    applied = module.handle_apply_light_plan(
        {
            "action": "APPLY",
            "plan_id": "night",
            "lights": [_point_spec("red")],
        }
    )
    obj = next(item for item in scene.objects if module._is_managed_light(item))
    obj.parent = FakeID(name="Artist_Rig")

    with pytest.raises(ValueError, match="unsupported parenting"):
        module.handle_apply_light_plan(
            {
                "action": "ROLLBACK",
                "transaction_id": applied["transaction_id"],
                "strict": True,
            }
        )

    assert obj in scene.objects
    assert applied["transaction_id"] in module._ledger


def test_rollback_preserves_object_identity_and_extra_light_fields(monkeypatch):
    module, _fake_bpy, _spatial, _dispatcher = _load(monkeypatch)
    _runtime, scene = _managed_runtime(module)
    module.handle_apply_light_plan(
        {
            "action": "APPLY",
            "plan_id": "night",
            "lights": [_point_spec("red", energy=250.0)],
        }
    )
    obj = next(item for item in scene.objects if module._is_managed_light(item))
    obj.data.exposure = 2.5
    obj.data.normalize = False
    obj.data.cycles = FakeID(
        max_bounces=17,
        use_multiple_importance_sampling=False,
        is_portal=False,
        is_caustics_light=True,
    )
    obj.data.name = "Separate Light Data Name"
    identity = id(obj)

    patched = module.handle_apply_light_plan(
        {
            "action": "APPLY",
            "plan_id": "night",
            "lights": [_point_spec("red", energy=800.0)],
        }
    )
    obj.data.exposure = 9.0
    obj.data.normalize = True
    obj.data.cycles.max_bounces = 2
    obj.data.cycles.is_caustics_light = False
    obj.data.name = "Changed After Apply"

    module.handle_apply_light_plan(
        {"action": "ROLLBACK", "transaction_id": patched["transaction_id"]}
    )

    restored = next(item for item in scene.objects if module._is_managed_light(item))
    assert id(restored) == identity
    assert restored.data.energy == 250.0
    assert restored.data.exposure == 2.5
    assert restored.data.normalize is False
    assert restored.data.cycles.max_bounces == 17
    assert restored.data.cycles.is_caustics_light is True
    assert restored.data.name == "Separate Light Data Name"


def test_validate_rejects_cross_plan_remove_id(monkeypatch):
    module, _fake_bpy, _spatial, _dispatcher = _load(monkeypatch)
    _runtime, _scene = _managed_runtime(module)
    module.handle_apply_light_plan(
        {
            "action": "APPLY",
            "plan_id": "alpha",
            "lights": [_point_spec("alpha_light")],
        }
    )

    with pytest.raises(ValueError, match="owned by plan 'alpha'"):
        module.handle_apply_light_plan(
            {
                "action": "VALIDATE",
                "plan_id": "beta",
                "remove_ids": ["alpha_light"],
            }
        )


def test_exact_proposal_classifies_idempotent_light_as_unchanged(monkeypatch):
    module, _fake_bpy, _spatial, _dispatcher = _load(monkeypatch)
    _runtime, _scene = _managed_runtime(module)
    spec = _point_spec("red")
    module.handle_apply_light_plan(
        {"action": "APPLY", "plan_id": "night", "lights": [spec]}
    )

    validated = module.handle_apply_light_plan(
        {"action": "VALIDATE", "plan_id": "night", "lights": [spec]}
    )

    assert validated["proposed_changes"]["update_ids"] == []
    assert validated["proposed_changes"]["unchanged_ids"] == ["red"]


def test_apply_freezes_target_object_before_removing_that_anchor(monkeypatch):
    module, _fake_bpy, _spatial, _dispatcher = _load(monkeypatch)
    _runtime, scene = _managed_runtime(module)
    monkeypatch.setattr(module, "_aim_object", MagicMock())
    follower = _light_spec(
        id="follower",
        name="follower",
        target_object="anchor",
    )
    module.handle_apply_light_plan(
        {
            "action": "APPLY",
            "plan_id": "night",
            "mode": "REPLACE_MANAGED",
            "lights": [_point_spec("anchor", location_world=[0.0, 0.0, 0.0])],
        }
    )
    module.handle_apply_light_plan(
        {
            "action": "APPLY",
            "plan_id": "night",
            "mode": "PATCH_MANAGED",
            "lights": [follower],
        }
    )

    applied = module.handle_apply_light_plan(
        {
            "action": "APPLY",
            "plan_id": "night",
            "mode": "REPLACE_MANAGED",
            "lights": [follower],
        }
    )

    assert applied["removed"] == ["anchor"]
    assert [
        obj[module.MANAGED_ID_PROP]
        for obj in scene.objects
        if module._is_managed_light(obj)
    ] == ["follower"]


def test_unmarked_ledger_text_collision_is_never_adopted_or_cleared(monkeypatch):
    module, _fake_bpy, _spatial, _dispatcher = _load(monkeypatch)
    runtime, _scene = _managed_runtime(module)
    artist_text = FakeText(module.LEDGER_TEXT)
    artist_text.write("artist notes")
    runtime.data.texts.append(artist_text)

    with pytest.raises(ValueError, match="not marked as the blend-ai relighting ledger"):
        module.handle_apply_light_plan(
            {"action": "VALIDATE", "plan_id": "night", "lights": []}
        )

    assert artist_text.as_string() == "artist notes"


def test_ledger_write_failure_rolls_back_entire_apply(monkeypatch):
    module, _fake_bpy, _spatial, _dispatcher = _load(monkeypatch)
    runtime, scene = _managed_runtime(module)

    class BrokenText(FakeText):
        def write(self, value):
            raise OSError("disk full")

    runtime.data.texts.new = MagicMock(
        side_effect=lambda name: (
            runtime.data.texts.append(text := BrokenText(name)) or text
        )
    )

    with pytest.raises(RuntimeError, match="failed and was rolled back"):
        module.handle_apply_light_plan(
            {
                "action": "APPLY",
                "plan_id": "night",
                "lights": [_point_spec("red")],
            }
        )

    assert not [obj for obj in scene.objects if module._is_managed_light(obj)]
    assert runtime.data.collections.get(module.MANAGED_COLLECTION) is None
    assert module._ledger == {}


def test_world_override_retains_previous_world_until_rollback(monkeypatch):
    module, _fake_bpy, _spatial, _dispatcher = _load(monkeypatch)
    runtime, scene = _managed_runtime(module)
    original = FakeID(
        name="Artist World",
        users=1,
        use_fake_user=False,
        library=None,
        color=(0.1, 0.1, 0.1),
        node_tree=None,
    )
    scene.world = original
    runtime.data.worlds.append(original)

    def new_world(name):
        world = FakeID(
            name=name,
            users=0,
            use_fake_user=False,
            library=None,
            color=(0.0, 0.0, 0.0),
            node_tree=None,
        )
        runtime.data.worlds.append(world)
        return world

    runtime.data.worlds.new = new_world

    def mark_world(world, _color, _strength, plan_id):
        world[module.MANAGED_PROP] = True
        world[module.MANAGED_PLAN_PROP] = plan_id

    monkeypatch.setattr(module, "_set_managed_world", mark_world)
    applied = module.handle_apply_light_plan(
        {
            "action": "APPLY",
            "plan_id": "night",
            "lights": [],
            "scene_overrides": {
                "world": {
                    "mode": "MANAGED_SOLID",
                    "color_rgb": [0.0, 0.0, 0.02],
                    "strength": 0.1,
                }
            },
        }
    )
    transaction_world = scene.world
    assert original.use_fake_user is True

    module.handle_apply_light_plan(
        {"action": "ROLLBACK", "transaction_id": applied["transaction_id"]}
    )

    assert scene.world is original
    assert original.use_fake_user is False
    assert transaction_world not in runtime.data.worlds


@pytest.mark.parametrize("spot_angle", [0.0, 0.5, 0.999999])
def test_handler_rejects_spot_angles_below_blender_minimum(monkeypatch, spot_angle):
    module, _bpy, _spatial, _dispatcher = _load(monkeypatch)

    with pytest.raises(ValueError, match="must be >= 1.0"):
        module.handle_apply_light_plan(
            {
                "action": "VALIDATE",
                "plan_id": "night",
                "lights": [
                    {
                        "id": "spot",
                        "type": "SPOT",
                        "location_world": [0.0, 0.0, 2.0],
                        "spot_angle_degrees": spot_angle,
                    }
                ],
            }
        )


@pytest.mark.parametrize("stage,failure_index", [(stage, index) for stage in ("create", "update", "remove") for index in (1, 2)])
def test_fault_after_each_managed_mutation_restores_snapshot(
    monkeypatch, stage, failure_index
):
    module, _fake_bpy, _spatial, _dispatcher = _load(monkeypatch)
    _runtime, scene = _managed_runtime(module)
    if stage != "create":
        module.handle_apply_light_plan(
            {
                "action": "APPLY",
                "plan_id": "night",
                "mode": "REPLACE_MANAGED",
                "lights": [_point_spec("one"), _point_spec("two")],
            }
        )
    before = {
        item["id"]: item
        for item in (
            module._serialize_managed_light(obj)
            for obj in scene.objects
            if module._is_managed_light(obj)
        )
    }

    if stage in {"create", "update"}:
        original = module._apply_light_spec
        calls = {"count": 0, "raised": False}

        def fail_once(obj, spec, plan_id):
            calls["count"] += 1
            original(obj, spec, plan_id)
            if calls["count"] == failure_index and not calls["raised"]:
                calls["raised"] = True
                raise RuntimeError(f"fault after {stage} {failure_index}")

        monkeypatch.setattr(module, "_apply_light_spec", fail_once)
        lights = (
            [_point_spec("one"), _point_spec("two")]
            if stage == "create"
            else [
                _point_spec("one", energy=501.0),
                _point_spec("two", energy=502.0),
            ]
        )
    else:
        original = module._remove_light_object
        calls = {"count": 0, "raised": False}

        def fail_once(obj):
            calls["count"] += 1
            original(obj)
            if calls["count"] == failure_index and not calls["raised"]:
                calls["raised"] = True
                raise RuntimeError(f"fault after remove {failure_index}")

        monkeypatch.setattr(module, "_remove_light_object", fail_once)
        lights = []

    with pytest.raises(RuntimeError, match="failed and was rolled back"):
        module.handle_apply_light_plan(
            {
                "action": "APPLY",
                "plan_id": "night",
                "mode": "REPLACE_MANAGED",
                "lights": lights,
            }
        )

    after = {
        item["id"]: item
        for item in (
            module._serialize_managed_light(obj)
            for obj in scene.objects
            if module._is_managed_light(obj)
        )
    }
    assert after == before
