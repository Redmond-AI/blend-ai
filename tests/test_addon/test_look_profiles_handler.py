"""Focused tests for persistent look-profile compilation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest


MODULE_PATH = Path(__file__).parents[2] / "addon" / "handlers" / "look_profiles.py"


def _load_module():
    """Load only this handler, without importing Blender's full handler package."""
    bpy_module = ModuleType("bpy")
    dispatcher = ModuleType("addon.dispatcher")
    dispatcher.register_handler = MagicMock()
    spatial_cache = ModuleType("addon.spatial_cache")
    spatial_cache.get_revisions = lambda: {
        "geometry_revision": 11,
        "lighting_revision": 7,
    }
    look_profile_assets = ModuleType("addon.look_profile_assets")

    def build_profile_assets(
        _scene,
        _collection,
        _world,
        profile,
        *,
        name_namespace=None,
    ):
        return {
            "created": [],
            "inventory": {
                "profile_id": profile["profile_id"],
                "component_count": len(profile.get("atmosphere", []))
                + len(profile.get("lighting", {}).get("lights", [])),
                "name_namespace": name_namespace,
            },
        }

    look_profile_assets.build_profile_assets = build_profile_assets
    look_profile_assets.cleanup_created = lambda _created: []
    look_profile_assets.validate_profile_light_isolation = lambda _scene, _policy: {
        "excluded": [],
        "cloned": [],
    }
    look_compositor = ModuleType("addon.look_compositor")
    look_compositor.API_NODE_GROUP = "NODE_GROUP_5X"
    look_compositor.API_LEGACY = "LEGACY_4X"
    look_compositor.MANAGED_TREE_PROP = "blend_ai_look_compositor_managed"

    class FakeCompositor(dict):
        def __init__(self, name):
            super().__init__()
            self.name = name
            self.nodes = []

    look_compositor.get_compositor_tree = lambda scene: getattr(
        scene, "compositing_node_group", None
    )

    def ensure_compositor(scene, *, name, source_tree=None, copy_source=True):
        tree = FakeCompositor(name)
        tree[look_compositor.MANAGED_TREE_PROP] = True
        scene.compositing_node_group = tree
        return tree

    look_compositor.ensure_unique_managed_compositor = ensure_compositor
    look_compositor.rebind_render_layers_to_scene = lambda _tree, _scene: 0
    look_compositor.owned_nested_compositor_trees = lambda _tree: []
    look_compositor.detect_compositor_api = lambda _scene: look_compositor.API_NODE_GROUP
    look_compositor.build_managed_post_stack = lambda _scene, _post, tree=None: {
        "api": look_compositor.API_NODE_GROUP,
        "tree_name": tree.name,
        "node_names": [],
        "compositor_hash": "c" * 64,
    }
    look_compositor.validate_final_image_linked = lambda _scene, tree=None: {
        "api": look_compositor.API_NODE_GROUP,
        "node_name": "AI_LOOK_OUTPUT",
        "socket_name": "Image",
    }
    look_compositor.compositor_hash = lambda _scene, tree=None: "c" * 64
    addon = ModuleType("addon")
    addon.__path__ = []
    addon.dispatcher = dispatcher
    addon.spatial_cache = spatial_cache
    addon.look_profile_assets = look_profile_assets
    addon.look_compositor = look_compositor
    handlers = ModuleType("addon.handlers")
    handlers.__path__ = []

    temporary_modules = {
        "bpy": bpy_module,
        "addon": addon,
        "addon.dispatcher": dispatcher,
        "addon.spatial_cache": spatial_cache,
        "addon.look_profile_assets": look_profile_assets,
        "addon.look_compositor": look_compositor,
        "addon.handlers": handlers,
    }
    previous = {name: sys.modules.get(name) for name in temporary_modules}
    try:
        sys.modules.update(temporary_modules)
        spec = importlib.util.spec_from_file_location("addon.handlers.look_profiles", MODULE_PATH)
        assert spec is not None and spec.loader is not None
        loaded = importlib.util.module_from_spec(spec)
        sys.modules["addon.handlers.look_profiles"] = loaded
        spec.loader.exec_module(loaded)
        return loaded
    finally:
        sys.modules.pop("addon.handlers.look_profiles", None)
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


module = _load_module()


class FakeID(dict):
    def __init__(self, **attributes):
        super().__init__()
        for name, value in attributes.items():
            setattr(self, name, value)


class FakeNamedValues(list):
    def get(self, name, default=None):
        for value in self:
            if getattr(value, "name", None) == name:
                return value
        return default


class FakeLinks(list):
    def link(self, value):
        if value not in self:
            self.append(value)

    def unlink(self, value):
        if value in self:
            self.remove(value)


class FakeText(FakeID):
    def __init__(self, name):
        super().__init__(name=name)
        self.body = ""

    def as_string(self):
        return self.body

    def clear(self):
        self.body = ""

    def write(self, value):
        self.body += value


def _clone_settings(value):
    return FakeID(**{name: item for name, item in vars(value).items() if not name.startswith("_")})


class FakeRuntime:
    def __init__(self):
        self.scenes = FakeNamedValues()
        self.collections = FakeNamedValues()
        self.worlds = FakeNamedValues()
        self.texts = FakeNamedValues()
        self.text_new_calls = 0

        self.collections.new = self.new_collection
        self.worlds.new = self.new_world
        self.texts.new = self.new_text

        geometry = FakeID(name="Geometry", children=FakeLinks(), objects=FakeLinks())
        self.collections.append(geometry)
        source_root = FakeID(
            name="Source Root",
            children=FakeLinks([geometry]),
            objects=FakeLinks(),
        )
        objects = [FakeID(name="Cube"), FakeID(name="Ground")]
        source_root.objects.extend(objects)

        self.source_world = FakeID(
            name="Source World",
            color=(0.1, 0.2, 0.3),
            use_nodes=True,
            node_tree=FakeID(),
            library=None,
        )
        self.source_world.copy = self.copy_source_world
        self.worlds.append(self.source_world)

        self.source = FakeID(
            name="Base Scene",
            library=None,
            objects=objects,
            collection=source_root,
            world=self.source_world,
            camera=FakeID(name="Camera"),
            render=FakeID(
                engine="BLENDER_EEVEE",
                resolution_x=1920,
                resolution_y=1080,
                resolution_percentage=50,
                film_transparent=False,
                use_motion_blur=False,
                use_compositing=True,
            ),
            cycles=FakeID(samples=64, use_denoising=True),
            eevee=FakeID(taa_render_samples=32),
            view_settings=FakeID(
                view_transform="AgX",
                look="Medium High Contrast",
                exposure=0.25,
                gamma=1.0,
            ),
            compositing_node_group=FakeID(
                name="Source Compositor",
                nodes=FakeNamedValues(),
                links=FakeLinks(),
            ),
        )
        self.source.copy = self.copy_source_scene
        self.scenes.append(self.source)
        self.window = FakeID(scene=self.source)
        self.bpy = SimpleNamespace(
            app=SimpleNamespace(version=(5, 2, 0), version_string="5.2.0 LTS"),
            context=SimpleNamespace(
                window=self.window,
                window_manager=SimpleNamespace(windows=[self.window]),
            ),
            data=SimpleNamespace(
                scenes=self.scenes,
                collections=self.collections,
                worlds=self.worlds,
                texts=self.texts,
            ),
        )

    def new_collection(self, name):
        collection = FakeID(
            name=name,
            children=FakeLinks(),
            objects=FakeLinks(),
            hide_render=False,
            hide_viewport=False,
        )
        self.collections.append(collection)
        return collection

    def new_world(self, name):
        world = FakeID(
            name=name,
            color=(0.0, 0.0, 0.0),
            use_nodes=False,
            node_tree=None,
            library=None,
        )
        self.worlds.append(world)
        return world

    def new_text(self, name):
        self.text_new_calls += 1
        text = FakeText(name)
        self.texts.append(text)
        return text

    def copy_source_world(self):
        world = FakeID(
            name="Source World Copy",
            color=tuple(self.source_world.color),
            use_nodes=self.source_world.use_nodes,
            node_tree=self.source_world.node_tree,
            library=None,
        )
        self.worlds.append(world)
        return world

    def copy_source_scene(self):
        root = FakeID(
            name="Source Root Copy",
            children=FakeLinks(list(self.source.collection.children)),
            objects=FakeLinks(list(self.source.collection.objects)),
        )
        scene = FakeID(
            name="Base Scene Copy",
            library=None,
            objects=list(self.source.objects),
            collection=root,
            world=self.source.world,
            camera=self.source.camera,
            render=_clone_settings(self.source.render),
            cycles=_clone_settings(self.source.cycles),
            eevee=_clone_settings(self.source.eevee),
            view_settings=_clone_settings(self.source.view_settings),
            compositing_node_group=self.source.compositing_node_group,
        )
        self.scenes.append(scene)
        return scene


@pytest.fixture
def runtime(monkeypatch):
    value = FakeRuntime()
    monkeypatch.setattr(module, "bpy", value.bpy)
    return value


def _request(**updates):
    result = {
        "action": "VALIDATE",
        "update_mode": "CREATE_VERSION",
        "source_scene": "Base Scene",
        "profile_id": "summer_day",
        "display_name": "Summer Day",
        "random_seed": 1234,
        "world_settings": {"mode": "COPY_SOURCE"},
        "render_settings": {
            "engine": "CYCLES",
            "resolution_x": 2048,
            "resolution_y": 1024,
            "resolution_percentage": 100,
            "samples": 128,
            "use_denoising": False,
            "film_transparent": True,
        },
        "color_settings": {
            "view_transform": "AgX",
            "look": "High Contrast",
            "exposure": 1.25,
            "gamma": 1.1,
        },
    }
    result.update(updates)
    return result


def _base_snapshot(runtime):
    source = runtime.source
    return {
        "scene_count": len(runtime.scenes),
        "collection_count": len(runtime.collections),
        "world_count": len(runtime.worlds),
        "text_count": len(runtime.texts),
        "root_children": list(source.collection.children),
        "world": source.world,
        "render": dict(vars(source.render)),
        "color": dict(vars(source.view_settings)),
    }


def _assert_base_unchanged(runtime, before):
    source = runtime.source
    assert source in runtime.scenes
    assert source.world is before["world"]
    assert list(source.collection.children) == before["root_children"]
    assert vars(source.render) == before["render"]
    assert vars(source.view_settings) == before["color"]


def test_context_is_read_only_and_requires_explicit_source_scene(runtime):
    before = _base_snapshot(runtime)

    result = module.handle_get_look_profile_context({"source_scene": "Base Scene"})

    assert result["schema_version"] == module.MANIFEST_SCHEMA_VERSION
    assert result["manifest_exists"] is False
    assert result["source_scene"]["name"] == "Base Scene"
    assert result["profiles"] == []
    assert runtime.text_new_calls == 0
    assert _base_snapshot(runtime) == before

    with pytest.raises(ValueError, match="source_scene"):
        module.handle_get_look_profile_context({})
    with pytest.raises(ValueError, match="was not found"):
        module.handle_get_look_profile_context({"source_scene": "Missing"})


def test_validate_is_deterministic_and_has_zero_mutation(runtime):
    before = _base_snapshot(runtime)

    first = module.handle_upsert_look_profile(_request())
    second = module.handle_upsert_look_profile(_request())

    assert first == second
    assert first["action"] == "VALIDATE"
    assert first["mutation_count"] == 0
    assert first["plan"]["compile_mode"] == "LINK_COPY"
    assert first["plan"]["version"] == 1
    assert first["plan"]["revision"] == 1
    assert first["plan"]["scene_name"].startswith("AI_LOOK_summer_day_v001_r001")
    assert runtime.text_new_calls == 0
    assert _base_snapshot(runtime) == before


def test_validate_rejects_collision_without_mutation(runtime):
    validated = module.handle_upsert_look_profile(_request())
    collision_name = validated["plan"]["collection_name"]
    runtime.new_collection(collision_name)
    before = _base_snapshot(runtime)

    with pytest.raises(ValueError, match="already exists"):
        module.handle_upsert_look_profile(_request())

    assert _base_snapshot(runtime) == before
    assert runtime.text_new_calls == 0


def test_validate_preflights_mute_non_managed_collection_safety(
    runtime,
    monkeypatch,
):
    before = _base_snapshot(runtime)
    calls = []

    def reject_shared_collection(scene, policy):
        calls.append((scene, policy))
        raise ValueError("MUTE_NON_MANAGED would hide non-profile content")

    monkeypatch.setattr(
        module.look_profile_assets,
        "validate_profile_light_isolation",
        reject_shared_collection,
    )

    with pytest.raises(ValueError, match="would hide non-profile content"):
        module.handle_upsert_look_profile(_nested_request())

    assert calls == [(runtime.source, "MUTE_NON_MANAGED")]
    assert _base_snapshot(runtime) == before
    assert runtime.text_new_calls == 0


def test_compile_creates_isolated_link_copy_and_persists_manifest(runtime):
    before = _base_snapshot(runtime)
    request = _request(action="COMPILE")

    result = module.handle_upsert_look_profile(request)

    assert result["compiled"] is True
    entry = result["profile"]
    compiled = runtime.scenes.get(entry["scene_name"])
    profile_collection = runtime.collections.get(entry["collection_name"])
    profile_world = runtime.worlds.get(entry["world_name"])
    assert compiled is not None and compiled is not runtime.source
    assert [id(value) for value in compiled.objects] == [
        id(value) for value in runtime.source.objects
    ]
    assert profile_collection in compiled.collection.children
    assert profile_collection not in runtime.source.collection.children
    assert compiled.world is profile_world
    assert profile_world is not runtime.source_world

    for value, role in (
        (compiled, "SCENE"),
        (profile_collection, "COLLECTION"),
        (profile_world, "WORLD"),
    ):
        assert value[module.MANAGED_PROP] is True
        assert value[module.PROFILE_ID_PROP] == "summer_day"
        assert value[module.PROFILE_VERSION_PROP] == 1
        assert value[module.PROFILE_REVISION_PROP] == 1
        assert value[module.SOURCE_SCENE_PROP] == "Base Scene"
        assert value[module.ROLE_PROP] == role
        assert value[module.COMPILE_MODE_PROP] == "LINK_COPY"

    assert compiled.render.engine == "CYCLES"
    assert compiled.render.resolution_x == 2048
    assert compiled.render.resolution_y == 1024
    assert compiled.render.resolution_percentage == 100
    assert compiled.render.film_transparent is True
    assert compiled.cycles.samples == 128
    assert compiled.cycles.use_denoising is False
    assert compiled.view_settings.look == "High Contrast"
    assert compiled.view_settings.exposure == 1.25
    assert compiled.view_settings.gamma == 1.1

    manifest_text = runtime.texts.get(module.MANIFEST_TEXT)
    assert manifest_text is not None
    assert manifest_text[module.MANIFEST_PROP] is True
    manifest = json.loads(manifest_text.as_string())
    assert manifest["schema_version"] == module.MANIFEST_SCHEMA_VERSION
    assert manifest["profiles"] == [entry]
    _assert_base_unchanged(runtime, before)


def test_compile_failure_removes_all_created_artifacts(runtime, monkeypatch):
    before = _base_snapshot(runtime)

    def fail_color(_scene, _settings):
        raise RuntimeError("synthetic color failure")

    monkeypatch.setattr(module, "_apply_color_settings", fail_color)
    with pytest.raises(RuntimeError, match="created artifacts were cleaned up"):
        module.handle_upsert_look_profile(_request(action="COMPILE"))

    assert len(runtime.scenes) == before["scene_count"]
    assert len(runtime.collections) == before["collection_count"]
    assert len(runtime.worlds) == before["world_count"]
    assert len(runtime.texts) == before["text_count"]
    assert runtime.text_new_calls == 0
    _assert_base_unchanged(runtime, before)


def test_link_copy_verification_failure_removes_scene_created_inside_helper(runtime):
    before = _base_snapshot(runtime)
    original_copy = runtime.source.copy

    def make_non_linked_scene():
        scene = original_copy()
        scene.objects = [FakeID(name="Unexpected Copy")]
        return scene

    runtime.source.copy = make_non_linked_scene
    with pytest.raises(RuntimeError, match="created artifacts were cleaned up"):
        module.handle_upsert_look_profile(_request(action="COMPILE"))

    assert len(runtime.scenes) == before["scene_count"]
    assert len(runtime.collections) == before["collection_count"]
    assert len(runtime.worlds) == before["world_count"]
    assert len(runtime.texts) == before["text_count"]
    _assert_base_unchanged(runtime, before)


def test_solid_world_setup_failure_removes_world_before_helper_return(runtime):
    before = _base_snapshot(runtime)

    with pytest.raises(RuntimeError, match="created artifacts were cleaned up"):
        module.handle_upsert_look_profile(
            _request(
                action="COMPILE",
                world_settings={
                    "mode": "SOLID",
                    "color_rgb": [0.01, 0.02, 0.03],
                    "strength": 0.5,
                },
            )
        )

    assert len(runtime.scenes) == before["scene_count"]
    assert len(runtime.collections) == before["collection_count"]
    assert len(runtime.worlds) == before["world_count"]
    assert len(runtime.texts) == before["text_count"]
    _assert_base_unchanged(runtime, before)


def test_replace_draft_creates_immutable_revision_and_supersedes_manifest(runtime):
    first = module.handle_upsert_look_profile(_request(action="COMPILE"))
    second = module.handle_upsert_look_profile(
        _request(
            action="COMPILE",
            update_mode="REPLACE_DRAFT",
            random_seed=5678,
        )
    )

    assert first["profile"]["version"] == second["profile"]["version"] == 1
    assert first["profile"]["revision"] == 1
    assert second["profile"]["revision"] == 2
    assert second["profile"]["scene_name"] != first["profile"]["scene_name"]
    assert second["warnings"]

    manifest = json.loads(runtime.texts.get(module.MANIFEST_TEXT).as_string())
    assert [entry["status"] for entry in manifest["profiles"]] == [
        "SUPERSEDED",
        "DRAFT",
    ]
    assert runtime.scenes.get(first["profile"]["scene_name"]) is not None
    assert runtime.scenes.get(second["profile"]["scene_name"]) is not None


def test_compile_scopes_managed_asset_names_to_each_immutable_scene(runtime):
    first = module.handle_upsert_look_profile(_request(action="COMPILE"))
    second = module.handle_upsert_look_profile(
        _request(
            action="COMPILE",
            update_mode="REPLACE_DRAFT",
            random_seed=5678,
        )
    )

    assert first["profile"]["managed_inventory"]["name_namespace"] == first["profile"]["scene_name"]
    assert (
        second["profile"]["managed_inventory"]["name_namespace"] == second["profile"]["scene_name"]
    )
    assert first["profile"]["scene_name"] != second["profile"]["scene_name"]


def test_manifest_write_failure_restores_manifest_and_cleans_new_revision(runtime, monkeypatch):
    first = module.handle_upsert_look_profile(_request(action="COMPILE"))
    text = runtime.texts.get(module.MANIFEST_TEXT)
    previous_manifest = text.as_string()
    before_counts = (len(runtime.scenes), len(runtime.collections), len(runtime.worlds))
    original_write = text.write
    state = {"failed": False}

    def fail_once(value):
        if not state["failed"]:
            state["failed"] = True
            raise RuntimeError("synthetic manifest failure")
        original_write(value)

    monkeypatch.setattr(text, "write", fail_once)
    with pytest.raises(RuntimeError, match="created artifacts were cleaned up"):
        module.handle_upsert_look_profile(_request(action="COMPILE", update_mode="REPLACE_DRAFT"))

    assert text.as_string() == previous_manifest
    assert (len(runtime.scenes), len(runtime.collections), len(runtime.worlds)) == before_counts
    assert runtime.scenes.get(first["profile"]["scene_name"]) is not None


def test_replace_draft_requires_existing_draft_without_mutation(runtime):
    before = _base_snapshot(runtime)
    with pytest.raises(ValueError, match="requires an existing DRAFT"):
        module.handle_upsert_look_profile(_request(update_mode="REPLACE_DRAFT"))
    assert _base_snapshot(runtime) == before


def test_unmarked_manifest_collision_is_never_adopted(runtime):
    artist_text = runtime.new_text(module.MANIFEST_TEXT)
    artist_text.write("artist notes")
    before = artist_text.as_string()

    with pytest.raises(ValueError, match="not marked"):
        module.handle_get_look_profile_context({"source_scene": "Base Scene"})

    assert artist_text.as_string() == before
    assert artist_text.get(module.MANIFEST_PROP) is None


def test_register_exposes_only_profile_commands(monkeypatch):
    registered = []
    monkeypatch.setattr(
        module.dispatcher,
        "register_handler",
        lambda name, handler: registered.append((name, handler)),
    )

    module.register()

    assert [name for name, _handler in registered] == [
        "get_look_profile_context",
        "upsert_look_profile",
        "activate_look_profile",
        "accept_look_profile",
    ]


def test_request_schema_rejects_unknown_fields_and_invalid_modes(runtime):
    with pytest.raises(ValueError, match="unsupported fields"):
        module.handle_upsert_look_profile(_request(extra=True))
    with pytest.raises(ValueError, match="VALIDATE or COMPILE"):
        module.handle_upsert_look_profile(_request(action="APPLY"))
    with pytest.raises(ValueError, match="CREATE_VERSION or REPLACE_DRAFT"):
        module.handle_upsert_look_profile(_request(update_mode="OVERWRITE"))


def test_validated_definition_hash_changes_with_seed_but_names_do_not(runtime):
    first = module.handle_upsert_look_profile(_request(random_seed=1))["plan"]
    second = module.handle_upsert_look_profile(_request(random_seed=2))["plan"]

    assert first["definition_hash"] != second["definition_hash"]
    assert first["scene_name"] == second["scene_name"]
    assert first["collection_name"] == second["collection_name"]
    assert first["world_name"] == second["world_name"]


def _nested_request(**updates):
    profile = {
        "schema_version": 1,
        "profile_id": "creepy-night",
        "display_name": "Creepy Night",
        "description": "Cold moonlight, fog, grain, and a restrained vignette.",
        "status": "DRAFT",
        "seed": 9001,
        "generator_version": "blender-look-profiles-0.1.0",
        "tags": ["night", "exterior"],
        "lighting": {
            "existing_light_policy": "MUTE_NON_MANAGED",
            "lights": [
                {
                    "id": "moon-key",
                    "type": "SUN",
                    "location_world": [0.0, 0.0, 10.0],
                    "energy": 0.5,
                    "color_rgb": [0.3, 0.45, 0.8],
                }
            ],
        },
        "world": {
            "mode": "MANAGED_SKY",
            "strength": 0.1,
            "sun_elevation_degrees": 20.0,
        },
        "atmosphere": [
            {
                "id": "ground-fog",
                "kind": "FOG_VOLUME",
                "density": 0.006,
                "anisotropy": 0.25,
            }
        ],
        "post": {
            "mode": "MANAGED_STACK",
            "bloom": {"enabled": False, "threshold": 1.0, "strength": 0.0, "radius": 0.5},
            "grain": {"enabled": True, "strength": 0.08, "scale": 1.0, "seed": 91},
            "vignette": {"enabled": True, "strength": 0.2, "feather": 0.6},
        },
        "color_management": {
            "mode": "MANAGED",
            "view_transform": "AgX",
            "look": "Medium High Contrast",
            "exposure": -1.0,
            "gamma": 1.0,
        },
        "camera": {"mode": "KEEP"},
        "render": {
            "engine": "BLENDER_EEVEE_NEXT",
            "samples": 32,
            "denoise": True,
            "resolution_x": 1920,
            "resolution_y": 1080,
            "resolution_percentage": 25,
            "film_transparent": False,
            "use_motion_blur": True,
        },
    }
    result = {
        "action": "VALIDATE",
        "source_scene": "Base Scene",
        "profile": profile,
        "update_mode": "CREATE_VERSION",
        "expected_base_revision": None,
        "expected_profile_revision": 0,
        "expected_geometry_revision": 11,
        "strict": True,
    }
    result.update(updates)
    return result


def test_nested_contract_persists_full_resolved_definition_and_maps_eevee(runtime):
    request = _nested_request(action="COMPILE")
    context = module.handle_get_look_profile_context(
        {"source_scene": "Base Scene", "detail": "FULL"}
    )
    request["expected_base_revision"] = context["base_revision"]

    result = module.handle_upsert_look_profile(request)

    entry = result["profile"]
    definition = entry["definition"]
    assert definition["description"].startswith("Cold moonlight")
    assert definition["lighting"]["lights"][0]["id"] == "moon-key"
    assert definition["atmosphere"][0]["kind"] == "FOG_VOLUME"
    assert definition["post"]["grain"]["enabled"] is True
    assert definition["resolved"]["render"]["engine"] == "BLENDER_EEVEE"
    assert definition["resolved"]["render"]["resolution_x"] == 1920
    assert definition["resolved"]["render"]["resolution_y"] == 1080
    assert definition["resolved"]["render"]["resolution_percentage"] == 25
    assert entry["base_fingerprint"].startswith("sha256:")
    assert entry["geometry_fingerprint"].startswith("sha256:")
    assert entry["managed_inventory"]["profile_id"] == "creepy-night"
    assert entry["managed_inventory"]["component_count"] == 2
    assert entry["managed_inventory"]["compositor"]["adapter"] == ("SCENE_COMPOSITING_NODE_GROUP")
    assert entry["compositor_hash"] == "c" * 64
    compiled = runtime.scenes.get(entry["scene_name"])
    assert compiled.render.engine == "BLENDER_EEVEE"
    assert compiled.render.resolution_x == 1920
    assert compiled.render.resolution_y == 1080
    assert compiled.render.use_motion_blur is True
    assert compiled[module.DEFINITION_HASH_PROP] == entry["definition_hash"]


def test_nested_preconditions_are_deterministic_and_zero_mutation(runtime):
    context = module.handle_get_look_profile_context(
        {
            "source_scene": "Base Scene",
            "profile_id": "creepy-night",
            "detail": "CAPABILITIES",
        }
    )
    request = _nested_request(
        expected_base_revision=context["base_revision"],
        expected_profile_revision=context["profile_revision"],
        expected_geometry_revision=context["geometry_revision"],
    )
    before = _base_snapshot(runtime)
    first = module.handle_upsert_look_profile(request)
    second = module.handle_upsert_look_profile(request)
    assert first == second
    assert first["plan"]["base_fingerprint"] == context["base_fingerprint"]
    assert first["plan"]["geometry_fingerprint"] == context["geometry_fingerprint"]
    assert _base_snapshot(runtime) == before

    with pytest.raises(ValueError, match="Stale expected_base_revision"):
        module.handle_upsert_look_profile(
            _nested_request(expected_base_revision=context["base_revision"] + 1)
        )
    with pytest.raises(ValueError, match="Stale expected_profile_revision"):
        module.handle_upsert_look_profile(_nested_request(expected_profile_revision=99))
    with pytest.raises(ValueError, match="Stale expected_geometry_revision"):
        module.handle_upsert_look_profile(_nested_request(expected_geometry_revision=12))
    assert _base_snapshot(runtime) == before


def test_direct_accepted_upsert_is_rejected_without_mutation(runtime):
    request = _nested_request()
    request["profile"]["status"] = "ACCEPTED"
    before = _base_snapshot(runtime)

    with pytest.raises(ValueError, match="only creates DRAFT"):
        module.handle_upsert_look_profile(request)

    assert _base_snapshot(runtime) == before


def test_context_reports_version_compositor_and_filtered_ownership(runtime):
    compiled = module.handle_upsert_look_profile(_nested_request(action="COMPILE"))

    context = module.handle_get_look_profile_context(
        {
            "source_scene": "Base Scene",
            "profile_id": "creepy-night",
            "detail": "FULL",
        }
    )

    assert context["capabilities"]["blender"]["version"] == [5, 2, 0]
    assert context["capabilities"]["compositor"]["adapter_mode"] == "NODE_GROUP"
    assert context["capabilities"]["compositor"]["authoritative_output"] == "NODE_GROUP_OUTPUT"
    assert context["profiles"] == [compiled["profile"]]
    assert context["ownership"]["valid"] is True
    assert context["ownership"]["profiles"][0]["valid"] is True
    assert context["ownership"]["orphaned_managed_artifacts"] == []


def test_color_look_alias_resolves_to_blender_52_identifier(runtime):
    enum_items = FakeNamedValues(
        [
            FakeID(
                identifier="AgX - Medium High Contrast",
                name="AgX - Medium High Contrast",
                description="",
            ),
            FakeID(identifier="AgX - None", name="AgX - None", description=""),
        ]
    )
    runtime.source.view_settings.bl_rna = FakeID(
        properties=FakeNamedValues([FakeID(name="look", identifier="look", enum_items=enum_items)])
    )

    validated = module.handle_upsert_look_profile(_nested_request(action="VALIDATE"))

    assert (
        validated["plan"]["definition"]["resolved"]["color_management"]["look"]
        == "AgX - Medium High Contrast"
    )


def test_managed_camera_reframe_preserves_exact_source_name(runtime):
    request = _nested_request(action="VALIDATE")
    request["profile"]["camera"] = {
        "mode": "MANAGED_CLONE",
        "source_camera": "Camera ",
        "location_world": [2.0, -22.0, 2.5],
        "target_point": [-3.0, -3.0, 3.0],
    }

    validated = module.handle_upsert_look_profile(request)
    camera = validated["plan"]["definition"]["camera"]

    assert camera["source_camera"] == "Camera "
    assert camera["location_world"] == [2.0, -22.0, 2.5]
    assert camera["target_point"] == [-3.0, -3.0, 3.0]


def test_activate_requires_exact_manifest_and_artifact_ownership_and_never_saves(runtime):
    compiled = module.handle_upsert_look_profile(_nested_request(action="COMPILE"))
    entry = compiled["profile"]

    result = module.handle_activate_look_profile(
        {
            "profile_id": "creepy-night",
            "target_scene": entry["scene_name"],
            "window_index": 0,
            "expected_profile_revision": 1,
            "expected_geometry_revision": 11,
            "strict": True,
        }
    )

    assert result["activated"] is True
    assert result["previous_scene"] == "Base Scene"
    assert result["saved_blend"] is False
    assert runtime.window.scene is runtime.scenes.get(entry["scene_name"])

    with pytest.raises(ValueError, match="not exactly one manifest-owned"):
        module.handle_activate_look_profile(
            {"profile_id": "wrong-profile", "target_scene": entry["scene_name"]}
        )

    world = runtime.worlds.get(entry["world_name"])
    world[module.PROFILE_ID_PROP] = "tampered"
    with pytest.raises(ValueError, match="ownership validation failed"):
        module.handle_activate_look_profile(
            {"profile_id": "creepy-night", "target_scene": entry["scene_name"]}
        )


def test_profile_mutations_refuse_active_render_or_viewport(runtime, monkeypatch):
    monkeypatch.setattr(
        module,
        "_profile_mutation_conflicts",
        lambda: ["a queued or active look render"],
    )

    validated = module.handle_upsert_look_profile(_nested_request(action="VALIDATE"))
    assert validated["valid"] is True
    with pytest.raises(RuntimeError, match="queued or active look render"):
        module.handle_upsert_look_profile(_nested_request(action="COMPILE"))
    with pytest.raises(RuntimeError, match="queued or active look render"):
        module.handle_activate_look_profile(
            {"profile_id": "creepy-night", "target_scene": "anything"}
        )


def _acceptance_evidence(entry, scene, **updates):
    resolved = entry["definition"]["resolved"]["render"]
    payload = {
        "batch_id": "lookbatch-accepted",
        "result_id": "lookresult-accepted",
        "profile_id": entry["profile_id"],
        "target_scene": entry["scene_name"],
        "frame": 1,
        "output_pass": "COMPOSITE",
        "image_source": "COMPOSITE_OUTPUT",
        "artifact_path": "/tmp/accepted-composite.png",
        "artifact_sha256": "a" * 64,
        "artifact_byte_count": 4096,
        "completed_at": 12345.0,
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
        "source_width": round(scene.render.resolution_x * resolved["resolution_percentage"] / 100),
        "source_height": round(scene.render.resolution_y * resolved["resolution_percentage"] / 100),
        "engine": resolved["engine"],
        "samples": resolved["samples"],
        "denoise": resolved["use_denoising"],
        "color_management": entry["definition"]["resolved"]["color_management"],
    }
    payload.update(updates)
    return payload


def test_accept_requires_reviewed_full_composite_and_persists_evidence(runtime, monkeypatch):
    entry = module.handle_upsert_look_profile(_nested_request(action="COMPILE"))["profile"]
    scene = runtime.scenes.get(entry["scene_name"])
    evidence = _acceptance_evidence(entry, scene)
    monkeypatch.setattr(module, "_get_acceptance_evidence", lambda _batch, _result: evidence)

    result = module.handle_accept_look_profile(
        {
            "profile_id": entry["profile_id"],
            "target_scene": entry["scene_name"],
            "batch_id": evidence["batch_id"],
            "result_id": evidence["result_id"],
            "artifact_sha256": evidence["artifact_sha256"],
            "review_acknowledged": True,
            "expected_profile_revision": entry["revision"],
            "expected_geometry_revision": 11,
        }
    )

    assert result["accepted"] is True
    assert result["status"] == "ACCEPTED"
    assert result["saved_blend"] is False
    manifest = json.loads(runtime.texts.get(module.MANIFEST_TEXT).as_string())
    accepted = manifest["profiles"][0]
    assert accepted["status"] == "ACCEPTED"
    assert accepted["acceptance"]["artifact_sha256"] == "a" * 64
    assert accepted["acceptance"]["review_acknowledged"] is True


def test_accept_rejects_missing_review_and_preview_quality(runtime, monkeypatch):
    entry = module.handle_upsert_look_profile(_nested_request(action="COMPILE"))["profile"]
    scene = runtime.scenes.get(entry["scene_name"])
    evidence = _acceptance_evidence(entry, scene, source_width=1, source_height=1)
    monkeypatch.setattr(module, "_get_acceptance_evidence", lambda _batch, _result: evidence)
    params = {
        "profile_id": entry["profile_id"],
        "target_scene": entry["scene_name"],
        "batch_id": evidence["batch_id"],
        "result_id": evidence["result_id"],
        "artifact_sha256": evidence["artifact_sha256"],
        "review_acknowledged": False,
    }
    with pytest.raises(ValueError, match="review_acknowledged"):
        module.handle_accept_look_profile(params)
    params["review_acknowledged"] = True
    with pytest.raises(ValueError, match="full render preset"):
        module.handle_accept_look_profile(params)


def test_activation_strictly_rejects_stale_base_but_non_strict_warns(runtime):
    compiled = module.handle_upsert_look_profile(_nested_request(action="COMPILE"))
    entry = compiled["profile"]
    runtime.source.render.resolution_x = 1280

    with pytest.raises(ValueError, match="base fingerprint is stale"):
        module.handle_activate_look_profile(
            {
                "profile_id": "creepy-night",
                "target_scene": entry["scene_name"],
                "strict": True,
            }
        )

    result = module.handle_activate_look_profile(
        {
            "profile_id": "creepy-night",
            "target_scene": entry["scene_name"],
            "strict": False,
        }
    )
    assert result["warnings"]


def test_late_compile_failure_invokes_profile_asset_cleanup(runtime, monkeypatch):
    created_asset = FakeID(name="Managed Fog")
    cleanup_calls = []
    monkeypatch.setattr(
        module.look_profile_assets,
        "build_profile_assets",
        lambda _scene, _collection, _world, _profile, **_kwargs: {
            "created": [("OBJECT", created_asset)],
            "inventory": {"atmosphere": ["Managed Fog"]},
        },
    )
    monkeypatch.setattr(
        module.look_profile_assets,
        "cleanup_created",
        lambda created: cleanup_calls.append(list(created)) or [],
    )
    monkeypatch.setattr(
        module,
        "_persist_manifest",
        lambda _manifest: (_ for _ in ()).throw(RuntimeError("synthetic commit failure")),
    )
    before = _base_snapshot(runtime)

    with pytest.raises(RuntimeError, match="created artifacts were cleaned up"):
        module.handle_upsert_look_profile(_nested_request(action="COMPILE"))

    assert cleanup_calls[0][0][0] == "NODE_GROUP"
    assert cleanup_calls[-1] == [("OBJECT", created_asset)]
    assert len(runtime.scenes) == before["scene_count"]
    assert len(runtime.collections) == before["collection_count"]
    assert len(runtime.worlds) == before["world_count"]
    _assert_base_unchanged(runtime, before)
