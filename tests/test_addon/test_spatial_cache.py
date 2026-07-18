"""Focused tests for revision tracking and the JSON-only spatial cache."""

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE_PATH = Path(__file__).parents[2] / "addon" / "spatial_cache.py"


def _load(monkeypatch):
    fake_bpy = SimpleNamespace(
        context=SimpleNamespace(scene=SimpleNamespace(frame_current=7)),
        app=SimpleNamespace(
            handlers=SimpleNamespace(depsgraph_update_post=[], load_post=[]),
        ),
    )
    monkeypatch.setitem(__import__("sys").modules, "bpy", fake_bpy)
    spec = importlib.util.spec_from_file_location("spatial_cache_subject", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, fake_bpy


def test_cache_is_detached_json(monkeypatch):
    module, _bpy = _load(monkeypatch)
    original = {"bounds": {"min": [0.0, 1.0, 2.0]}}
    module.put_cached("scene", original)
    original["bounds"]["min"][0] = 99.0

    first = module.get_cached("scene")
    first["bounds"]["min"][1] = 88.0
    second = module.get_cached("scene")

    assert second == {"bounds": {"min": [0.0, 1.0, 2.0]}}
    with pytest.raises(ValueError):
        module.put_cached("nan", {"value": math.nan})


def test_depsgraph_updates_advance_separate_revisions(monkeypatch):
    module, _bpy = _load(monkeypatch)
    mesh = SimpleNamespace(bl_rna=SimpleNamespace(identifier="Mesh"))
    light = SimpleNamespace(bl_rna=SimpleNamespace(identifier="Light"))
    depsgraph = SimpleNamespace(
        updates=[SimpleNamespace(id=mesh), SimpleNamespace(id=light)]
    )

    module.depsgraph_update_post(None, depsgraph)

    assert module.get_revisions() == {
        "geometry_revision": 1,
        "lighting_revision": 1,
    }
    assert module.make_scene_revision() == "g1:l1:f7"


def test_rendered_viewport_shading_updates_do_not_invalidate_geometry(monkeypatch):
    module, _bpy = _load(monkeypatch)
    mesh_object = SimpleNamespace(
        bl_rna=SimpleNamespace(identifier="Object"),
        name="Warehouse_Floor",
        type="MESH",
    )
    mesh_data = SimpleNamespace(
        bl_rna=SimpleNamespace(identifier="Mesh"),
        name="Floor_Mesh",
    )
    updates = [
        SimpleNamespace(
            id=value,
            is_updated_geometry=False,
            is_updated_transform=False,
            is_updated_shading=True,
        )
        for value in (mesh_object, mesh_data)
    ]

    module.depsgraph_update_post(None, SimpleNamespace(updates=updates))

    assert module.get_revisions() == {
        "geometry_revision": 0,
        "lighting_revision": 1,
    }


def test_object_transform_still_invalidates_geometry(monkeypatch):
    module, _bpy = _load(monkeypatch)
    mesh_object = SimpleNamespace(
        bl_rna=SimpleNamespace(identifier="Object"),
        name="Warehouse_Floor",
        type="MESH",
    )
    update = SimpleNamespace(
        id=mesh_object,
        is_updated_geometry=False,
        is_updated_transform=True,
        is_updated_shading=False,
    )

    module.depsgraph_update_post(None, SimpleNamespace(updates=[update]))

    assert module.get_revisions() == {
        "geometry_revision": 1,
        "lighting_revision": 0,
    }


def test_handler_registration_is_idempotent(monkeypatch):
    module, fake_bpy = _load(monkeypatch)

    module.register_handlers()
    module.register_handlers()

    assert fake_bpy.app.handlers.depsgraph_update_post == [module.depsgraph_update_post]
    assert fake_bpy.app.handlers.load_post == [module.load_post]
    module.unregister_handlers()
    assert fake_bpy.app.handlers.depsgraph_update_post == []
    assert fake_bpy.app.handlers.load_post == []


def test_lighting_revision_preserves_geometry_cache(monkeypatch):
    module, _bpy = _load(monkeypatch)
    module.put_cached("geometry", {"instances": ["expensive"]})

    module.mark_lighting_dirty()

    assert module.get_revisions()["lighting_revision"] == 1
    assert module.get_cached("geometry") == {"instances": ["expensive"]}


def test_managed_relighting_ids_do_not_advance_geometry(monkeypatch):
    module, _bpy = _load(monkeypatch)

    def marker(key, default=False):
        return key == module.MANAGED_PROP or default

    collection = SimpleNamespace(
        bl_rna=SimpleNamespace(identifier="Collection"),
        get=marker,
    )
    world = SimpleNamespace(bl_rna=SimpleNamespace(identifier="World"), get=marker)
    scene = SimpleNamespace(
        bl_rna=SimpleNamespace(identifier="Scene"),
        world=world,
    )
    ledger = SimpleNamespace(bl_rna=SimpleNamespace(identifier="Text"))

    module.depsgraph_update_post(
        None,
        SimpleNamespace(
            updates=[
                SimpleNamespace(id=collection),
                SimpleNamespace(id=scene),
                SimpleNamespace(id=ledger),
            ]
        ),
    )

    assert module.get_revisions() == {
        "geometry_revision": 0,
        "lighting_revision": 1,
    }


def test_managed_edit_scope_covers_parent_collection_and_view_layer(monkeypatch):
    module, _bpy = _load(monkeypatch)
    parent = SimpleNamespace(bl_rna=SimpleNamespace(identifier="Collection"))
    view_layer = SimpleNamespace(bl_rna=SimpleNamespace(identifier="ViewLayer"))
    module.begin_managed_edit()

    module.depsgraph_update_post(
        None,
        SimpleNamespace(
            updates=[SimpleNamespace(id=parent), SimpleNamespace(id=view_layer)]
        ),
    )
    module.end_managed_edit()

    assert module.get_revisions() == {
        "geometry_revision": 0,
        "lighting_revision": 1,
    }
    recent = module.get_recent_updates()
    assert len(recent) == 1
    assert recent[0]["managed_grace"] is True
    assert recent[0]["geometry_changed"] is False
    assert all(
        update["suppressed_by_managed_grace"] for update in recent[0]["updates"]
    )


@pytest.mark.parametrize(
    "value",
    [
        SimpleNamespace(bl_rna=SimpleNamespace(identifier="Collection")),
        SimpleNamespace(bl_rna=SimpleNamespace(identifier="ViewLayer")),
        SimpleNamespace(bl_rna=SimpleNamespace(identifier="Scene"), world=None),
    ],
    ids=["collection", "view-layer", "scene"],
)
def test_real_spatial_edit_immediately_after_managed_scope_is_not_hidden(
    monkeypatch, value
):
    module, _bpy = _load(monkeypatch)
    update = SimpleNamespace(
        id=value,
        is_updated_geometry=True,
        is_updated_transform=False,
        is_updated_shading=False,
    )

    module.begin_managed_edit()
    module.depsgraph_update_post(None, SimpleNamespace(updates=[update]))
    module.end_managed_edit()
    module.depsgraph_update_post(None, SimpleNamespace(updates=[update]))

    assert module.get_revisions()["geometry_revision"] == 1


def test_geometry_node_tree_invalidates_geometry(monkeypatch):
    module, _bpy = _load(monkeypatch)
    node_tree = SimpleNamespace(
        bl_rna=SimpleNamespace(identifier="NodeTree"),
        bl_idname="GeometryNodeTree",
    )

    module.depsgraph_update_post(
        None, SimpleNamespace(updates=[SimpleNamespace(id=node_tree)])
    )

    assert module.get_revisions() == {
        "geometry_revision": 1,
        "lighting_revision": 0,
    }


def test_recent_update_diagnostics_are_detached_bounded_and_cleared(monkeypatch):
    module, _bpy = _load(monkeypatch)
    mesh = SimpleNamespace(
        bl_rna=SimpleNamespace(identifier="Mesh"),
        name_full="Warehouse Floor",
    )
    update = SimpleNamespace(
        id=mesh,
        is_updated_geometry=True,
        is_updated_transform=False,
        is_updated_shading=False,
    )

    for _ in range(module.MAX_RECENT_UPDATE_BATCHES + 3):
        module.depsgraph_update_post(None, SimpleNamespace(updates=[update]))

    recent = module.get_recent_updates()
    assert len(recent) == module.MAX_RECENT_UPDATE_BATCHES
    assert recent[-1]["updates"][0] == {
        "kind": "Mesh",
        "name": "Warehouse Floor",
        "object_type": "",
        "managed": False,
        "is_updated_geometry": True,
        "is_updated_transform": False,
        "is_updated_shading": False,
        "classified_geometry": True,
        "classified_lighting": False,
        "geometry": True,
        "lighting": False,
        "suppressed_by_managed_grace": False,
        "suppressed_by_render_grace": False,
    }
    recent[-1]["updates"][0]["name"] = "changed"
    assert module.get_recent_updates()[-1]["updates"][0]["name"] == "Warehouse Floor"

    module.load_post(None)
    assert module.get_recent_updates() == []


def test_render_evaluation_grace_suppresses_camera_tags_only(monkeypatch):
    module, _bpy = _load(monkeypatch)
    camera = SimpleNamespace(
        bl_rna=SimpleNamespace(identifier="Object"),
        name_full="Warehouse Camera",
        type="CAMERA",
    )
    camera_update = SimpleNamespace(
        id=camera,
        is_updated_geometry=True,
        is_updated_transform=False,
        is_updated_shading=False,
    )

    module.begin_render_evaluation()
    module.depsgraph_update_post(None, SimpleNamespace(updates=[camera_update]))

    assert module.get_revisions() == {
        "geometry_revision": 0,
        "lighting_revision": 1,
    }
    recent = module.get_recent_updates()[-1]
    assert recent["render_grace"] is True
    assert recent["updates"][0]["suppressed_by_render_grace"] is True

    module.end_render_evaluation()
    module.depsgraph_update_post(None, SimpleNamespace(updates=[camera_update]))
    assert module.get_revisions()["geometry_revision"] == 1


def test_render_evaluation_grace_does_not_hide_mesh_geometry(monkeypatch):
    module, _bpy = _load(monkeypatch)
    mesh = SimpleNamespace(
        bl_rna=SimpleNamespace(identifier="Mesh"),
        name_full="Warehouse Floor",
    )
    update = SimpleNamespace(
        id=mesh,
        is_updated_geometry=True,
        is_updated_transform=False,
        is_updated_shading=False,
    )

    module.begin_render_evaluation()
    module.depsgraph_update_post(None, SimpleNamespace(updates=[update]))
    module.end_render_evaluation()

    assert module.get_revisions()["geometry_revision"] == 1
    assert (
        module.get_recent_updates()[-1]["updates"][0][
            "suppressed_by_render_grace"
        ]
        is False
    )


def test_light_revision_preserves_geometry_cache_but_geometry_revision_clears_it(monkeypatch):
    module, _bpy = _load(monkeypatch)
    module.put_cached("geometry", {"bounds": [1, 2, 3]})

    module.mark_lighting_dirty()
    assert module.get_cached("geometry") == {"bounds": [1, 2, 3]}

    module.mark_geometry_dirty()
    assert module.get_cached("geometry") is None


def test_cache_age_is_recorded_and_cleared(monkeypatch):
    module, _bpy = _load(monkeypatch)
    module.put_cached("geometry", {"bounds": [1, 2, 3]})

    age = module.get_cache_age_ms("geometry")
    assert age is not None
    assert age >= 0.0

    module.clear_cache()
    assert module.get_cache_age_ms("geometry") is None
