"""Focused tests for asynchronous managed-profile Composite rendering."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
from pathlib import Path
import struct
import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest


MODULE_PATH = Path(__file__).parents[2] / "addon" / "handlers" / "look_profile_rendering.py"


def _load_module():
    bpy_module = ModuleType("bpy")
    dispatcher = ModuleType("addon.dispatcher")
    dispatcher.register_handler = MagicMock()
    look_compositor = ModuleType("addon.look_compositor")
    look_compositor.API_NODE_GROUP = "NODE_GROUP_5X"
    look_compositor.API_LEGACY = "LEGACY_4X"

    def get_compositor_tree(scene):
        group = getattr(scene, "compositing_node_group", None)
        if group is not None:
            return group
        return getattr(scene, "node_tree", None)

    def detect_compositor_api(scene):
        if getattr(scene, "compositing_node_group", None) is not None:
            return look_compositor.API_NODE_GROUP
        if getattr(scene, "node_tree", None) is not None and getattr(scene, "use_nodes", False):
            return look_compositor.API_LEGACY
        raise ValueError("Scene has no compositor API")

    def authoritative_final_output(scene, *, tree=None):
        tree = tree or get_compositor_tree(scene)
        api = detect_compositor_api(scene)
        kinds = (
            {"NodeGroupOutput", "GROUP_OUTPUT"}
            if api == look_compositor.API_NODE_GROUP
            else {"CompositorNodeComposite", "COMPOSITE"}
        )
        label = "Group Output" if api == look_compositor.API_NODE_GROUP else "Composite"
        candidates = [
            node
            for node in tree.nodes
            if getattr(node, "bl_idname", "") in kinds or getattr(node, "type", "") in kinds
        ]
        linked = [
            node
            for node in candidates
            if node.inputs and bool(getattr(node.inputs[0], "is_linked", False))
        ]
        if not linked:
            raise ValueError(f"Managed compositor has no linked {label} Image output")
        active = [node for node in linked if bool(getattr(node, "is_active_output", False))]
        if len(active) == 1:
            node = active[0]
        elif len(linked) == 1:
            node = linked[0]
        else:
            raise ValueError(f"Managed compositor has ambiguous linked {label} outputs")
        return node, node.inputs[0]

    def validate_final_image_linked(scene, *, tree=None):
        node, socket = authoritative_final_output(scene, tree=tree)
        if not socket.is_linked:
            raise ValueError("Authoritative compositor output is unlinked")
        return {"node_name": node.name, "socket_name": socket.name}

    def compositor_hash(scene, *, tree=None):
        tree = tree or get_compositor_tree(scene)
        return tree["semantic_hash"]

    def redirect_file_outputs(tree, directory):
        snapshots = []
        try:
            for node in tree.nodes:
                if getattr(node, "bl_idname", "") != "CompositorNodeOutputFile":
                    continue
                if not hasattr(node, "base_path"):
                    raise ValueError("File Output node does not expose a redirectable base_path")
                snapshots.append({"node": node, "base_path": node.base_path})
                node.base_path = directory
        except Exception:
            restore_file_outputs(tree, snapshots)
            raise
        return snapshots

    def restore_file_outputs(_tree, snapshots):
        for snapshot in snapshots:
            snapshot["node"].base_path = snapshot["base_path"]

    look_compositor.redirect_file_outputs = redirect_file_outputs
    look_compositor.restore_file_outputs = restore_file_outputs
    look_compositor.get_compositor_tree = get_compositor_tree
    look_compositor.detect_compositor_api = detect_compositor_api
    look_compositor.authoritative_final_output = authoritative_final_output
    look_compositor.validate_final_image_linked = validate_final_image_linked
    look_compositor.compositor_hash = compositor_hash

    look_profiles = ModuleType("addon.handlers.look_profiles")
    look_profiles.MANAGED_PROP = "blend_ai_look_managed"
    look_profiles.PROFILE_ID_PROP = "blend_ai_look_profile_id"
    look_profiles.PROFILE_VERSION_PROP = "blend_ai_look_profile_version"
    look_profiles.PROFILE_REVISION_PROP = "blend_ai_look_profile_revision"
    look_profiles.SOURCE_SCENE_PROP = "blend_ai_look_source_scene"
    look_profiles.ROLE_PROP = "blend_ai_look_role"
    look_profiles.COMPILE_MODE_PROP = "blend_ai_look_compile_mode"
    look_profiles.manifest = {"schema_version": 1, "profiles": []}
    look_profiles.runtime_bpy = None

    def load_manifest():
        return look_profiles.manifest

    def entry_ownership(entry):
        scene = look_profiles.runtime_bpy.data.scenes.get(entry["scene_name"])
        conflicts = []
        if scene is None:
            conflicts.append("Missing SCENE artifact")
        else:
            expected = {
                look_profiles.MANAGED_PROP: True,
                look_profiles.PROFILE_ID_PROP: entry["profile_id"],
                look_profiles.PROFILE_VERSION_PROP: entry["version"],
                look_profiles.PROFILE_REVISION_PROP: entry["revision"],
                look_profiles.SOURCE_SCENE_PROP: entry["source_scene"],
                look_profiles.ROLE_PROP: "SCENE",
                look_profiles.COMPILE_MODE_PROP: "LINK_COPY",
            }
            for key, value in expected.items():
                if scene.get(key) != value:
                    conflicts.append(f"SCENE has mismatched {key}")
            try:
                validate_final_image_linked(scene)
                if compositor_hash(scene) != entry["compositor_hash"]:
                    conflicts.append("Profile compositor hash differs from the manifest")
            except Exception as exc:
                conflicts.append(str(exc))
        return {"valid": not conflicts, "conflicts": conflicts}

    look_profiles._load_manifest = load_manifest
    look_profiles._entry_ownership = entry_ownership
    look_profiles._base_fingerprint = lambda scene: scene.base_fingerprint
    look_profiles._geometry_fingerprint = lambda scene: scene.geometry_fingerprint
    addon = ModuleType("addon")
    addon.__path__ = []
    addon.dispatcher = dispatcher
    addon.look_compositor = look_compositor
    handlers = ModuleType("addon.handlers")
    handlers.__path__ = []
    temporary = {
        "bpy": bpy_module,
        "addon": addon,
        "addon.dispatcher": dispatcher,
        "addon.look_compositor": look_compositor,
        "addon.handlers": handlers,
        "addon.handlers.look_profiles": look_profiles,
    }
    previous = {name: sys.modules.get(name) for name in temporary}
    try:
        sys.modules.update(temporary)
        spec = importlib.util.spec_from_file_location(
            "addon.handlers.look_profile_rendering", MODULE_PATH
        )
        assert spec is not None and spec.loader is not None
        loaded = importlib.util.module_from_spec(spec)
        sys.modules["addon.handlers.look_profile_rendering"] = loaded
        spec.loader.exec_module(loaded)
        return loaded, dispatcher
    finally:
        sys.modules.pop("addon.handlers.look_profile_rendering", None)
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


module, mock_dispatcher = _load_module()


class FakeID(dict):
    def __init__(self, **attributes):
        super().__init__()
        for name, value in attributes.items():
            setattr(self, name, value)


class FakeNamedValues(list):
    def get(self, name, default=None):
        return next(
            (value for value in self if getattr(value, "name", None) == name),
            default,
        )


def _png(width=320, height=180):
    return (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\rIHDR"
        + struct.pack(">II", width, height)
        + b"finished-composite"
    )


def _compiled_scene(name="AI Look Summer", profile_id="summer", revision=3):
    image_input = FakeID(name="Image", is_linked=True, default_value=None)
    group_output = FakeID(
        name="Group Output",
        bl_idname="NodeGroupOutput",
        type="GROUP_OUTPUT",
        inputs=FakeNamedValues([image_input]),
        mute=False,
        is_active_output=True,
        node_tree=None,
    )
    file_slot = FakeID(path="artist_beauty_")
    file_output = FakeID(
        name="Artist File Output",
        bl_idname="CompositorNodeOutputFile",
        type="OUTPUT_FILE",
        inputs=FakeNamedValues(),
        mute=False,
        node_tree=None,
        base_path="/artist/original",
        file_slots=[file_slot],
        layer_slots=[],
    )
    tree = FakeID(
        name=f"{name} Compositor",
        nodes=FakeNamedValues([group_output, file_output]),
        links=[],
    )
    tree["semantic_hash"] = "b" * 64
    render = FakeID(
        engine="CYCLES",
        filepath="/artist/original.png",
        image_settings=FakeID(file_format="JPEG", color_depth="8"),
        resolution_x=800,
        resolution_y=600,
        resolution_percentage=55,
        use_compositing=False,
        use_lock_interface=False,
    )
    scene = FakeID(
        name=name,
        library=None,
        render=render,
        cycles=FakeID(samples=19, use_denoising=False),
        camera=FakeID(name="Camera_Main"),
        frame_current=7,
        frame_subframe=0.25,
        view_layers=[FakeID(name="View Layer", use=True)],
        view_settings=FakeID(
            view_transform="AgX", look="Medium High Contrast", exposure=0.5, gamma=1.0
        ),
        display_settings=FakeID(display_device="sRGB"),
        sequencer_colorspace_settings=FakeID(name="sRGB"),
        compositing_node_group=tree,
    )

    scene.frame_set_calls = []

    def frame_set(frame, subframe=0.0):
        scene.frame_set_calls.append((frame, subframe))
        scene.frame_current = frame
        scene.frame_subframe = subframe

    scene.frame_set = frame_set
    scene[module.MANAGED_PROP] = True
    scene[module.PROFILE_ID_PROP] = profile_id
    scene[module.PROFILE_VERSION_PROP] = 2
    scene[module.PROFILE_REVISION_PROP] = revision
    scene[module.SOURCE_SCENE_PROP] = "Base Scene"
    scene[module.ROLE_PROP] = "SCENE"
    scene[module.COMPILE_MODE_PROP] = "LINK_COPY"
    return scene, file_output, file_slot


def _manifest_entry(scene):
    return {
        "profile_id": scene[module.PROFILE_ID_PROP],
        "display_name": scene[module.PROFILE_ID_PROP],
        "source_scene": scene[module.SOURCE_SCENE_PROP],
        "version": scene[module.PROFILE_VERSION_PROP],
        "revision": scene[module.PROFILE_REVISION_PROP],
        "status": "DRAFT",
        "compile_mode": "LINK_COPY",
        "scene_name": scene.name,
        "collection_name": f"{scene.name} Collection",
        "world_name": f"{scene.name} World",
        "compositor_name": scene.compositing_node_group.name,
        "compositor_hash": scene.compositing_node_group["semantic_hash"],
        "compositor_adapter": "SCENE_COMPOSITING_NODE_GROUP",
        "definition_hash": "c" * 64,
        "base_revision": 1234,
        "geometry_revision": 41,
        "base_fingerprint": "sha256:" + "d" * 64,
        "geometry_fingerprint": "sha256:" + "e" * 64,
        "definition": {"profile_id": scene[module.PROFILE_ID_PROP]},
        "managed_inventory": {},
    }


class FakeRuntime:
    def __init__(self):
        self.scene, self.file_output, self.file_slot = _compiled_scene()
        self.source_scene = FakeID(
            name="Base Scene",
            base_fingerprint="sha256:" + "d" * 64,
            geometry_fingerprint="sha256:" + "e" * 64,
        )
        self.scenes = FakeNamedValues([self.scene, self.source_scene])
        self.manifest = {"schema_version": 1, "profiles": [_manifest_entry(self.scene)]}
        self.timer_calls = []
        self.render_calls = []
        self.cancel_calls = 0
        self.render_complete = []
        self.render_cancel = []

        def register_timer(callback, *, first_interval):
            self.timer_calls.append((callback, first_interval))

        def render(*args, **kwargs):
            self.render_calls.append((args, kwargs))
            return {"RUNNING_MODAL"}

        def cancel():
            self.cancel_calls += 1
            return {"FINISHED"}

        self.bpy = SimpleNamespace(
            app=SimpleNamespace(
                timers=SimpleNamespace(register=register_timer),
                handlers=SimpleNamespace(
                    render_complete=self.render_complete,
                    render_cancel=self.render_cancel,
                ),
            ),
            data=SimpleNamespace(scenes=self.scenes, images=None),
            ops=SimpleNamespace(render=SimpleNamespace(render=render, cancel=cancel)),
        )

    def add_scene(self, *, name, profile_id="summer", revision=3):
        scene, _file_output, _file_slot = _compiled_scene(name, profile_id, revision)
        self.scenes.append(scene)
        self.manifest["profiles"].append(_manifest_entry(scene))
        return scene


@pytest.fixture
def runtime(monkeypatch):
    value = FakeRuntime()
    monkeypatch.setattr(module, "bpy", value.bpy)
    module.profile_compiler.runtime_bpy = value.bpy
    module.profile_compiler.manifest = value.manifest
    module._batches.clear()
    module._active_render = None
    module._timer_armed = False
    module._completion_request = None
    module._completion_timer_armed = False
    yield value
    module._batches.clear()
    module._active_render = None
    module._timer_armed = False
    module._completion_request = None
    module._completion_timer_armed = False


def _complete_render(scene):
    module._on_render_complete(scene)
    assert module._completion_timer_armed is True
    module._completion_timer()


def _cancel_render(scene):
    module._on_render_cancel(scene)
    assert module._completion_timer_armed is True
    module._completion_timer()


def _params(root, *, frames=None, profiles=None, scenes=None, pairing="PAIRWISE"):
    profiles = profiles or ["summer"]
    scenes = scenes or ["AI Look Summer"]
    result = {
        "batch": {
            "profile_ids": profiles,
            "target_scenes": scenes,
            "pairing": pairing,
            "frames": {"frames": frames or [1]},
            "output_root": str(root),
            "filename_template": "{scene}-{profile}-{frame}",
            "render_settings": {
                "samples": 64,
                "denoise": True,
                "resolution_percentage": 80,
                "file_format": "PNG",
                "color_depth": "16",
                "existing_file_policy": "ERROR",
            },
            "continue_on_error": False,
        },
        "output_pass": "COMPOSITE",
        "restore_scene_state": True,
        "save_blend": False,
    }
    pair_count = len(profiles) if pairing == "PAIRWISE" else len(profiles) * len(scenes)
    if pair_count == 1:
        result["expected_profile_revision"] = 3
    else:
        result["expected_profile_revisions"] = {scene: 3 for scene in scenes}
    return result


def test_submit_expands_cross_product_and_returns_before_render(runtime, tmp_path):
    runtime.add_scene(name="AI Look Summer B")

    result = module.handle_submit_look_render_batch(
        _params(
            tmp_path,
            frames=[1, 3],
            scenes=["AI Look Summer", "AI Look Summer B"],
            pairing="CROSS_PRODUCT",
        )
    )
    assert result["status"] == "QUEUED"
    assert result["item_count"] == 4
    assert len(result["result_ids"]) == 4
    assert runtime.render_calls == []
    assert runtime.timer_calls == [(module._job_timer, 0.0)]
    assert all(
        Path(item["output_path"]).resolve().is_relative_to(tmp_path.resolve())
        for item in module._batches[result["batch_id"]]["items"]
    )


def test_batch_contract_rejects_multi_profile_cross_product_and_scalar_revision(runtime, tmp_path):
    runtime.add_scene(name="AI Look Night", profile_id="night")
    cross = _params(
        tmp_path / "cross",
        profiles=["summer", "night"],
        scenes=["AI Look Summer", "AI Look Night"],
        pairing="CROSS_PRODUCT",
    )
    with pytest.raises(ValueError, match="exactly one profile_id"):
        module.handle_submit_look_render_batch(cross)

    pairwise = _params(
        tmp_path / "pairwise",
        profiles=["summer", "night"],
        scenes=["AI Look Summer", "AI Look Night"],
        pairing="PAIRWISE",
    )
    pairwise.pop("expected_profile_revisions")
    pairwise["expected_profile_revision"] = 3
    with pytest.raises(ValueError, match="Scalar expected_profile_revision"):
        module.handle_submit_look_render_batch(pairwise)


def test_submit_refuses_retained_cycles_viewport_session(runtime, tmp_path, monkeypatch):
    monkeypatch.setattr(
        module,
        "_active_cycles_viewport_session_ids",
        lambda: ["viewport-session"],
    )

    with pytest.raises(RuntimeError, match="Restore it first"):
        module.handle_submit_look_render_batch(_params(tmp_path))

    assert module._batches == {}
    assert runtime.timer_calls == []


def test_rejects_unmanaged_or_profile_mismatched_scene(runtime, tmp_path):
    runtime.scene[module.PROFILE_ID_PROP] = "night"

    with pytest.raises(ValueError, match="not the compiled LINK_COPY Scene"):
        module.handle_submit_look_render_batch(_params(tmp_path))

    runtime.scene[module.PROFILE_ID_PROP] = "summer"
    del runtime.scene[module.MANAGED_PROP]
    with pytest.raises(ValueError, match="not the compiled LINK_COPY Scene"):
        module.handle_submit_look_render_batch(_params(tmp_path))


def test_rejects_png_32_bit_before_any_job_is_created(runtime, tmp_path):
    params = _params(tmp_path)
    params["batch"]["render_settings"]["color_depth"] = "32"

    with pytest.raises(ValueError, match="PNG render batches require 8- or 16-bit"):
        module.handle_submit_look_render_batch(params)

    assert module._batches == {}
    assert runtime.timer_calls == []


def test_rejects_missing_or_ambiguous_final_composite(runtime, tmp_path):
    output = runtime.scene.compositing_node_group.nodes[0]
    output.inputs[0].is_linked = False
    with pytest.raises(ValueError, match="no linked Group Output"):
        module.handle_submit_look_render_batch(_params(tmp_path))

    output.inputs[0].is_linked = True
    output.is_active_output = False
    duplicate = FakeID(
        name="Other Output",
        bl_idname="NodeGroupOutput",
        type="GROUP_OUTPUT",
        inputs=FakeNamedValues([FakeID(name="Image", is_linked=True)]),
        mute=False,
        is_active_output=False,
        node_tree=None,
    )
    runtime.scene.compositing_node_group.nodes.append(duplicate)
    with pytest.raises(ValueError, match="ambiguous linked Group Output"):
        module.handle_submit_look_render_batch(_params(tmp_path))


def test_requires_one_explicit_view_layer_and_camera(runtime, tmp_path):
    runtime.scene.view_layers.append(FakeID(name="Second Layer", use=True))
    with pytest.raises(ValueError, match="exactly one enabled View Layer"):
        module.handle_submit_look_render_batch(_params(tmp_path))

    runtime.scene.view_layers[-1].use = False
    runtime.scene.camera = None
    with pytest.raises(ValueError, match="has no active camera"):
        module.handle_submit_look_render_batch(_params(tmp_path))


def _review_ready_runtime(runtime):
    scene = runtime.scene
    scene.render.use_compositing = True
    scene.cycles.glossy_bounces = 4
    scene.camera.type = "CAMERA"
    mesh = FakeID(
        name="Reflective Floor",
        type="MESH",
        hide_render=False,
        visible_shadow=True,
        visible_glossy=True,
    )
    light = FakeID(
        name="Managed Moon",
        type="LIGHT",
        data=FakeID(use_shadow=True),
        hide_render=False,
    )
    scene.objects = FakeNamedValues([scene.camera, mesh, light])
    entry = runtime.manifest["profiles"][0]
    entry["definition"]["review_intent"] = {
        "summary": "Cold moonlight with readable reflections",
        "expects_shadows": True,
        "expects_reflections": True,
        "expects_volume": False,
        "expects_compositing": True,
    }
    entry["managed_inventory"] = {
        "lights": [{"object_name": "Managed Moon", "component_id": "moon"}]
    }
    collection = FakeID(name=entry["collection_name"])
    scene.view_layers[0].layer_collection = FakeID(
        collection=FakeID(name="Root"),
        exclude=False,
        children=[FakeID(collection=collection, exclude=False, children=[])],
    )
    return mesh, light


def test_review_audit_passes_and_detects_explicit_failures(runtime):
    mesh, light = _review_ready_runtime(runtime)
    params = {
        "profile_id": "summer",
        "target_scene": "AI Look Summer",
        "camera": "Camera_Main",
        "expected_profile_revision": 3,
        "expected_geometry_revision": 41,
    }

    passed = module.handle_inspect_look_review_state(params)
    assert passed["status"] == "PASS"
    assert passed["failures"] == []

    light.data.use_shadow = False
    failed = module.handle_inspect_look_review_state(params)
    assert failed["status"] == "FAIL"
    assert "MANAGED_LIGHT_SHADOWS_DISABLED" in {item["code"] for item in failed["failures"]}

    light.data.use_shadow = True
    mesh.visible_glossy = False
    failed = module.handle_inspect_look_review_state(params)
    assert "ALL_REFLECTION_VISIBILITY_DISABLED" in {item["code"] for item in failed["failures"]}


def test_review_audit_warns_for_individual_artist_visibility(runtime):
    mesh, _light = _review_ready_runtime(runtime)
    other = FakeID(
        name="Wall",
        type="MESH",
        hide_render=False,
        visible_shadow=True,
        visible_glossy=True,
    )
    runtime.scene.objects.append(other)
    mesh.visible_shadow = False

    result = module.handle_inspect_look_review_state(
        {
            "profile_id": "summer",
            "target_scene": "AI Look Summer",
            "camera": "Camera_Main",
        }
    )
    assert result["status"] == "WARN"
    assert result["warnings"][0]["code"] == "OBJECT_SHADOW_VISIBILITY_DISABLED"


def test_legacy_42_composite_detection_requires_linked_authoritative_output(runtime):
    scene = runtime.scene
    scene.compositing_node_group = None
    scene.use_nodes = True
    composite = FakeID(
        name="Composite",
        bl_idname="CompositorNodeComposite",
        type="COMPOSITE",
        inputs=FakeNamedValues([FakeID(name="Image", is_linked=True)]),
        is_active_output=True,
    )
    scene.node_tree = FakeID(name="Legacy", nodes=[composite], links=[])

    scene.node_tree["semantic_hash"] = "b" * 64
    module.look_compositor.validate_final_image_linked(scene, tree=scene.node_tree)
    api = module.look_compositor.detect_compositor_api(scene)

    assert api == module.look_compositor.API_LEGACY


def test_async_render_uses_explicit_scene_and_restores_every_setting(runtime, tmp_path):
    original = {
        "frame": runtime.scene.frame_current,
        "subframe": runtime.scene.frame_subframe,
        "filepath": runtime.scene.render.filepath,
        "format": runtime.scene.render.image_settings.file_format,
        "depth": runtime.scene.render.image_settings.color_depth,
        "percentage": runtime.scene.render.resolution_percentage,
        "samples": runtime.scene.cycles.samples,
        "denoise": runtime.scene.cycles.use_denoising,
        "use_compositing": runtime.scene.render.use_compositing,
        "use_lock_interface": runtime.scene.render.use_lock_interface,
        "file_base": runtime.file_output.base_path,
        "slot_path": runtime.file_slot.path,
    }
    submitted = module.handle_submit_look_render_batch(_params(tmp_path, frames=[12]))

    module._job_timer()

    assert runtime.render_calls == [
        (
            ("INVOKE_DEFAULT",),
            {"write_still": True, "scene": "AI Look Summer"},
        )
    ]
    assert runtime.scene.frame_current == 12
    assert runtime.scene.frame_subframe == 0.0
    assert runtime.scene.render.image_settings.file_format == "PNG"
    assert runtime.scene.render.image_settings.color_depth == "16"
    assert runtime.scene.render.resolution_percentage == 80
    assert runtime.scene.cycles.samples == 64
    assert runtime.scene.cycles.use_denoising is True
    assert runtime.scene.render.use_compositing is True
    assert runtime.scene.render.use_lock_interface is True
    assert Path(runtime.file_output.base_path).resolve().is_relative_to(tmp_path.resolve())
    assert runtime.file_slot.path == original["slot_path"]
    item = module._batches[submitted["batch_id"]]["items"][0]
    artifact = Path(item["output_path"])
    artifact.write_bytes(_png(640, 480))
    module._on_render_complete(runtime.scene)

    assert item["status"] == "RUNNING"
    assert runtime.scene.render.filepath != original["filepath"]
    assert module._completion_timer_armed is True
    module._completion_timer()

    assert item["status"] == "SUCCEEDED"
    assert runtime.scene.frame_current == original["frame"]
    assert runtime.scene.frame_subframe == original["subframe"]
    assert runtime.scene.render.filepath == original["filepath"]
    assert runtime.scene.render.image_settings.file_format == original["format"]
    assert runtime.scene.render.image_settings.color_depth == original["depth"]
    assert runtime.scene.render.resolution_percentage == original["percentage"]
    assert runtime.scene.cycles.samples == original["samples"]
    assert runtime.scene.cycles.use_denoising == original["denoise"]
    assert runtime.scene.render.use_compositing == original["use_compositing"]
    assert runtime.scene.render.use_lock_interface == original["use_lock_interface"]
    assert runtime.file_output.base_path == original["file_base"]
    assert runtime.file_slot.path == original["slot_path"]


def test_render_skips_noop_frame_updates_during_prepare_and_restore(runtime, tmp_path):
    runtime.scene.frame_current = 18
    runtime.scene.frame_subframe = 0.0
    submitted = module.handle_submit_look_render_batch(_params(tmp_path, frames=[18]))

    module._job_timer()

    item = module._batches[submitted["batch_id"]]["items"][0]
    assert runtime.scene.frame_set_calls == []
    Path(item["output_path"]).write_bytes(_png())
    _complete_render(runtime.scene)
    assert runtime.scene.frame_set_calls == []


def test_poll_result_returns_composite_proxy_and_strict_provenance(runtime, tmp_path):
    from blend_ai.tools.look_profiles import LookRenderResultMetadata

    submitted = module.handle_submit_look_render_batch(_params(tmp_path))
    module._job_timer()
    batch = module._batches[submitted["batch_id"]]
    item = batch["items"][0]
    artifact_bytes = _png(1920, 1080)
    Path(item["output_path"]).write_bytes(artifact_bytes)
    _complete_render(runtime.scene)

    polled = module.handle_get_look_render_batch(
        {
            "batch_id": submitted["batch_id"],
            "include_results": True,
            "max_results": 10,
            "cursor": None,
        }
    )
    fetched = module.handle_get_look_render_result(
        {
            "batch_id": submitted["batch_id"],
            "result_id": item["result_id"],
            "output_pass": "COMPOSITE",
            "proxy_max_size": 1024,
        }
    )

    assert polled["status"] == "SUCCEEDED"
    assert polled["counts"]["succeeded"] == 1
    metadata = fetched["metadata"]
    assert metadata["status"] == "SUCCEEDED"
    assert metadata["scene"] == "AI Look Summer"
    assert metadata["view_layer"] == "View Layer"
    assert metadata["camera"] == "Camera_Main"
    assert metadata["frame"] == 1
    assert metadata["engine"] == "CYCLES"
    assert metadata["samples"] == 64
    assert metadata["image_source"] == "COMPOSITE_OUTPUT"
    assert len(metadata["compositor_hash"]) == 64
    assert metadata["profile_status"] == "DRAFT"
    assert metadata["definition_hash"] == "c" * 64
    assert metadata["base_revision"] == 1234
    assert metadata["geometry_revision"] == 41
    assert metadata["base_fingerprint"] == "sha256:" + "d" * 64
    assert metadata["geometry_fingerprint"] == "sha256:" + "e" * 64
    assert len(metadata["manifest_entry_hash"]) == 64
    assert metadata["artifact_sha256"] == hashlib.sha256(artifact_bytes).hexdigest()
    assert metadata["artifact_byte_count"] == len(artifact_bytes)
    assert metadata["source_width"] == 1920
    assert metadata["proxy_width"] == 1920
    assert metadata["proxy_format"] == "PNG"
    assert metadata["submitted_at"] > 0
    assert metadata["started_at"] > 0
    assert metadata["completed_at"] > 0
    assert base64.b64decode(fetched["image_base64"]) == artifact_bytes
    assert LookRenderResultMetadata.model_validate(metadata).status == "SUCCEEDED"


def test_cancellation_preserves_completed_artifact_and_restores_active_item(runtime, tmp_path):
    submitted = module.handle_submit_look_render_batch(_params(tmp_path, frames=[1, 2]))
    batch = module._batches[submitted["batch_id"]]
    module._job_timer()
    first, second = batch["items"]
    first_bytes = _png()
    Path(first["output_path"]).write_bytes(first_bytes)
    _complete_render(runtime.scene)

    cancelled = module.handle_cancel_look_render_batch({"batch_id": submitted["batch_id"]})

    assert cancelled["status"] == "CANCELLED"
    assert first["status"] == "SUCCEEDED"
    assert Path(first["output_path"]).read_bytes() == first_bytes
    assert second["status"] == "CANCELLED"
    assert runtime.cancel_calls == 0
    assert runtime.scene.render.filepath == "/artist/original.png"


def test_active_cancellation_is_cooperative_until_render_callback(runtime, tmp_path):
    submitted = module.handle_submit_look_render_batch(_params(tmp_path))
    batch = module._batches[submitted["batch_id"]]
    module._job_timer()

    result = module.handle_cancel_look_render_batch({"batch_id": submitted["batch_id"]})

    assert result["cancel_requested"] is True
    assert runtime.cancel_calls == 1
    assert batch["items"][0]["status"] == "RUNNING"
    _cancel_render(runtime.scene)
    assert batch["items"][0]["status"] == "CANCELLED"
    assert runtime.scene.render.filepath == "/artist/original.png"


def test_background_cancellation_marks_queue_without_calling_bpy_ops(
    runtime,
    tmp_path,
):
    submitted = module.handle_submit_look_render_batch(_params(tmp_path, frames=[1, 2]))
    batch = module._batches[submitted["batch_id"]]
    module._job_timer()
    responses = []

    worker = threading.Thread(
        target=lambda: responses.append(
            module.handle_cancel_look_render_batch({"batch_id": submitted["batch_id"]})
        )
    )
    worker.start()
    worker.join(timeout=2.0)

    assert worker.is_alive() is False
    assert responses[0]["cancel_requested"] is True
    assert runtime.cancel_calls == 0
    assert batch["items"][0]["status"] == "RUNNING"
    assert batch["items"][1]["status"] == "CANCELLED"


def test_revalidates_revision_and_refuses_output_that_appears_after_submit(runtime, tmp_path):
    revision_job = module.handle_submit_look_render_batch(_params(tmp_path / "revision"))
    runtime.scene[module.PROFILE_REVISION_PROP] = 4

    module._job_timer()

    revision_item = module._batches[revision_job["batch_id"]]["items"][0]
    assert revision_item["status"] == "FAILED"
    assert "does not match expected revision 3" in revision_item["error"]
    assert runtime.render_calls == []

    runtime.scene[module.PROFILE_REVISION_PROP] = 3
    file_job = module.handle_submit_look_render_batch(_params(tmp_path / "appeared"))
    file_item = module._batches[file_job["batch_id"]]["items"][0]
    original = b"do-not-overwrite"
    Path(file_item["output_path"]).write_bytes(original)

    module._job_timer()

    assert file_item["status"] == "FAILED"
    assert Path(file_item["output_path"]).read_bytes() == original
    assert len(runtime.render_calls) == 0


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("definition_hash", "9" * 64, "provenance changed"),
        ("base_revision", 9999, "provenance changed"),
        ("geometry_revision", 9999, "provenance changed"),
        ("base_fingerprint", "sha256:" + "9" * 64, "base fingerprint changed"),
        (
            "geometry_fingerprint",
            "sha256:" + "8" * 64,
            "geometry fingerprint changed",
        ),
    ],
)
def test_render_start_rejects_any_manifest_provenance_drift(
    runtime, tmp_path, field, replacement, message
):
    submitted = module.handle_submit_look_render_batch(_params(tmp_path))
    runtime.manifest["profiles"][0][field] = replacement

    module._job_timer()

    item = module._batches[submitted["batch_id"]]["items"][0]
    assert item["status"] == "FAILED"
    assert message in item["error"]
    assert runtime.render_calls == []


@pytest.mark.parametrize("fingerprint", ["base_fingerprint", "geometry_fingerprint"])
def test_render_start_rejects_current_source_scene_drift(runtime, tmp_path, fingerprint):
    submitted = module.handle_submit_look_render_batch(_params(tmp_path))
    setattr(runtime.source_scene, fingerprint, "sha256:" + "1" * 64)

    module._job_timer()

    item = module._batches[submitted["batch_id"]]["items"][0]
    assert item["status"] == "FAILED"
    assert "fingerprint changed after profile compilation" in item["error"]
    assert runtime.render_calls == []


def test_render_start_rejects_compositor_hash_drift(runtime, tmp_path):
    submitted = module.handle_submit_look_render_batch(_params(tmp_path))
    runtime.scene.compositing_node_group["semantic_hash"] = "7" * 64

    module._job_timer()

    item = module._batches[submitted["batch_id"]]["items"][0]
    assert item["status"] == "FAILED"
    assert "compositor hash" in item["error"]
    assert runtime.render_calls == []


def test_render_completion_rejects_provenance_drift_and_withholds_result(runtime, tmp_path):
    submitted = module.handle_submit_look_render_batch(_params(tmp_path))
    module._job_timer()
    item = module._batches[submitted["batch_id"]]["items"][0]
    Path(item["output_path"]).write_bytes(_png())
    runtime.manifest["profiles"][0]["geometry_revision"] = 42

    _complete_render(runtime.scene)

    assert item["status"] == "FAILED"
    assert item["metadata"] is None
    fetched = module.handle_get_look_render_result(
        {
            "batch_id": submitted["batch_id"],
            "result_id": item["result_id"],
            "output_pass": "COMPOSITE",
            "proxy_max_size": 1024,
        }
    )
    assert "image_base64" not in fetched


def test_render_completion_rejects_current_source_scene_drift(runtime, tmp_path):
    submitted = module.handle_submit_look_render_batch(_params(tmp_path))
    module._job_timer()
    item = module._batches[submitted["batch_id"]]["items"][0]
    Path(item["output_path"]).write_bytes(_png())
    runtime.source_scene.geometry_fingerprint = "sha256:" + "2" * 64

    _complete_render(runtime.scene)

    assert item["status"] == "FAILED"
    assert "geometry fingerprint changed" in item["error"]
    assert item["metadata"] is None


def test_existing_file_skip_is_terminal_and_never_acceptable(runtime, tmp_path):
    artifact = tmp_path / "AI Look Summer-summer-1.png"
    artifact.write_bytes(_png())
    params = _params(tmp_path)
    params["batch"]["render_settings"]["existing_file_policy"] = "SKIP"

    submitted = module.handle_submit_look_render_batch(params)
    batch = module._batches[submitted["batch_id"]]
    item = batch["items"][0]

    assert submitted["status"] == "SKIPPED"
    assert item["status"] == "SKIPPED"
    assert item["metadata"] is None
    assert runtime.render_calls == []
    assert runtime.timer_calls == []
    with pytest.raises(ValueError, match="SKIPPED"):
        module.get_acceptance_evidence(submitted["batch_id"], item["result_id"])


def test_file_appearing_after_enqueue_is_skipped_without_proxy(runtime, tmp_path):
    params = _params(tmp_path)
    params["batch"]["render_settings"]["existing_file_policy"] = "SKIP"
    submitted = module.handle_submit_look_render_batch(params)
    item = module._batches[submitted["batch_id"]]["items"][0]
    Path(item["output_path"]).write_bytes(_png())

    module._job_timer()

    assert item["status"] == "SKIPPED"
    assert item["proxy_path"] is None
    assert item["metadata"] is None
    assert runtime.render_calls == []


def test_composite_proxy_preserves_display_pixels_without_scene_color_management(
    monkeypatch, tmp_path
):
    source = tmp_path / "composite.png"
    source.write_bytes(b"display-referred-source")
    destination = tmp_path / "proxy.png"
    calls = []

    class FakeSpec:
        width = 3200
        height = 1600
        nchannels = 4

    class FakeImageBuf:
        def __init__(self, path=None):
            self.path = path
            self.has_error = False

        def spec(self):
            return FakeSpec()

        def geterror(self):
            return ""

        def write(self, path, output_type):
            calls.append(("write", path, output_type))
            Path(path).write_bytes(b"display-referred-proxy")
            return True

    class FakeImageBufAlgo:
        @staticmethod
        def resize(destination_image, source_image, filter_name, filter_width, roi):
            calls.append(
                (
                    "resize",
                    source_image.path,
                    filter_name,
                    filter_width,
                    (roi.xbegin, roi.xend, roi.ybegin, roi.yend, roi.chbegin, roi.chend),
                )
            )
            return True

    class FakeROI:
        def __init__(self, xbegin, xend, ybegin, yend, zbegin, zend, chbegin, chend):
            self.xbegin = xbegin
            self.xend = xend
            self.ybegin = ybegin
            self.yend = yend
            self.chbegin = chbegin
            self.chend = chend

    fake_oiio = ModuleType("OpenImageIO")
    fake_oiio.ImageBuf = FakeImageBuf
    fake_oiio.ImageBufAlgo = FakeImageBufAlgo
    fake_oiio.ROI = FakeROI
    fake_oiio.UINT8 = "UINT8"
    monkeypatch.setitem(sys.modules, "OpenImageIO", fake_oiio)

    scene = SimpleNamespace(
        view_settings=SimpleNamespace(
            view_transform="ACES 2.0",
            look="None",
            exposure=-1.05,
            gamma=1.0,
        ),
        render=SimpleNamespace(image_settings=SimpleNamespace(file_format="PNG", color_depth="16")),
    )
    original_view = dict(vars(scene.view_settings))
    original_image_settings = dict(vars(scene.render.image_settings))

    size = module._create_proxy(scene, source, destination)

    assert size == (2048, 1024)
    assert destination.read_bytes() == b"display-referred-proxy"
    assert calls == [
        (
            "resize",
            str(source),
            "lanczos3",
            0.0,
            (0, 2048, 0, 1024, 0, 4),
        ),
        ("write", str(destination), "UINT8"),
    ]
    assert vars(scene.view_settings) == original_view
    assert vars(scene.render.image_settings) == original_image_settings


def test_acceptance_evidence_is_checksum_and_manifest_bound(runtime, tmp_path):
    submitted = module.handle_submit_look_render_batch(_params(tmp_path))
    module._job_timer()
    item = module._batches[submitted["batch_id"]]["items"][0]
    artifact = Path(item["output_path"])
    artifact.write_bytes(_png())
    _complete_render(runtime.scene)

    evidence = module.get_acceptance_evidence(submitted["batch_id"], item["result_id"])

    assert evidence["image_source"] == "COMPOSITE_OUTPUT"
    assert evidence["definition_hash"] == "c" * 64
    assert evidence["geometry_revision"] == 41
    assert evidence["source_width"] == 320
    assert evidence["source_height"] == 180
    assert evidence["samples"] == 64
    assert evidence["denoise"] is True
    assert evidence["engine"] == "CYCLES"
    assert evidence["color_management"]["view_transform"] == "AgX"
    assert evidence["compositor_hash"] == "b" * 64
    assert evidence["compositor_adapter"] == "SCENE_COMPOSITING_NODE_GROUP"
    assert evidence["artifact_sha256"] == hashlib.sha256(_png()).hexdigest()
    runtime.source_scene.base_fingerprint = "sha256:" + "3" * 64
    with pytest.raises(ValueError, match="base fingerprint changed"):
        module.get_acceptance_evidence(submitted["batch_id"], item["result_id"])
    runtime.source_scene.base_fingerprint = "sha256:" + "d" * 64
    artifact.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="checksum changed"):
        module.get_acceptance_evidence(submitted["batch_id"], item["result_id"])


def test_partial_file_output_redirect_is_rolled_back_on_prepare_failure(runtime, tmp_path):
    invalid_output = FakeID(
        name="Unredirectable Output",
        bl_idname="CompositorNodeOutputFile",
        type="OUTPUT_FILE",
        inputs=FakeNamedValues(),
        mute=False,
        node_tree=None,
        file_slots=[],
        layer_slots=[],
    )
    runtime.scene.compositing_node_group.nodes.append(invalid_output)
    submitted = module.handle_submit_look_render_batch(_params(tmp_path))

    module._job_timer()

    item = module._batches[submitted["batch_id"]]["items"][0]
    assert item["status"] == "FAILED"
    assert "redirectable base_path" in item["error"]
    assert runtime.file_output.base_path == "/artist/original"
    assert runtime.file_slot.path == "artist_beauty_"
    assert runtime.scene.render.filepath == "/artist/original.png"
    assert runtime.scene.frame_current == 7
    assert runtime.render_calls == []


def test_review_tile_paths_marks_an_engine_omission_unavailable(tmp_path):
    sidecar = tmp_path / "review"
    sidecar.mkdir()
    (sidecar / "combined_0001.exr").write_bytes(b"combined")
    prepared = {
        "review": {
            "sidecar": str(sidecar),
            "prefixes": {
                "combined": "combined_",
                "cryptomatte_object": "cryptomatte_object_",
            },
        }
    }

    paths = module._review_tile_paths(prepared)

    assert paths["combined"] == sidecar / "combined_0001.exr"
    assert paths["cryptomatte_object"] is None


def test_review_render_layers_are_explicitly_bound_to_compiled_scene():
    profile_scene = SimpleNamespace(name="AI Look Moonlight")
    render_layers = SimpleNamespace(scene=None)

    module._bind_review_render_layers_scene(render_layers, profile_scene)

    assert render_layers.scene is profile_scene


def test_prepare_review_nodes_binds_transient_layers_to_profile_scene(monkeypatch, tmp_path):
    image_socket = SimpleNamespace(name="Image")
    render_layers = SimpleNamespace(
        name="",
        label="",
        scene=None,
        layer="",
        outputs=FakeNamedValues([image_socket]),
    )

    class Nodes(list):
        def new(self, node_type):
            if node_type == "CompositorNodeRLayers":
                node = render_layers
            elif node_type == "ShaderNodeMix":
                node = SimpleNamespace(
                    name="",
                    data_type="",
                    blend_type="",
                    inputs=[SimpleNamespace(default_value=None) for _ in range(8)],
                    outputs=[SimpleNamespace() for _ in range(3)],
                )
            elif node_type == "CompositorNodeCryptomatteV2":
                raise RuntimeError("Cryptomatte unavailable in focused test")
            else:
                raise AssertionError(f"Unexpected node type: {node_type}")
            self.append(node)
            return node

        def remove(self, node):
            if node in self:
                super().remove(node)

    tree = SimpleNamespace(
        nodes=Nodes(),
        links=SimpleNamespace(new=lambda _source, _target: None),
    )
    profile_scene = SimpleNamespace(
        name="AI Look Moonlight",
        view_layers=[SimpleNamespace(name="View Layer", use=True)],
    )
    monkeypatch.setattr(
        module,
        "_configure_review_file_output",
        lambda _tree, _socket, _sidecar, slug: SimpleNamespace(name=slug),
    )

    module._prepare_review_nodes(
        profile_scene,
        tree,
        tmp_path,
        "lookresult-000001-reviewbind",
        [],
    )

    assert render_layers.scene is profile_scene
    assert render_layers.layer == "View Layer"


def test_review_render_layers_refuse_missing_or_rejected_scene_binding():
    profile_scene = SimpleNamespace(name="AI Look Moonlight")

    with pytest.raises(RuntimeError, match="no explicit Scene binding"):
        module._bind_review_render_layers_scene(SimpleNamespace(), profile_scene)

    class RejectingRenderLayers:
        scene = None

        def __setattr__(self, name, value):
            if name == "scene":
                return
            super().__setattr__(name, value)

    with pytest.raises(RuntimeError, match="did not retain"):
        module._bind_review_render_layers_scene(RejectingRenderLayers(), profile_scene)


def test_review_tile_order_uses_camera_depth_instead_of_volume():
    assert module.REVIEW_TILE_ORDER == (
        ("combined", "Combined (pre-compositor)"),
        ("diffuse_direct", "Diffuse Direct"),
        ("glossy", "Glossy Direct + Indirect"),
        ("emission", "Emission"),
        ("depth", "Camera Depth (near=white)"),
        ("cryptomatte_object", "Object Cryptomatte"),
    )


def test_review_tile_paths_rejects_ambiguous_pass_outputs(tmp_path):
    sidecar = tmp_path / "review"
    sidecar.mkdir()
    (sidecar / "combined_0001.exr").write_bytes(b"first")
    (sidecar / "combined_0002.exr").write_bytes(b"second")
    prepared = {
        "review": {
            "sidecar": str(sidecar),
            "prefixes": {"combined": "combined_"},
        }
    }

    with pytest.raises(RuntimeError, match="produced 2 files"):
        module._review_tile_paths(prepared)


def test_registration_is_idempotent_for_callbacks(runtime):
    mock_dispatcher.register_handler.reset_mock()

    module.register()
    module.register()

    assert mock_dispatcher.register_handler.call_count == 10
    assert runtime.render_complete == [module._on_render_complete]
    assert runtime.render_cancel == [module._on_render_cancel]
    module.unregister()
    assert runtime.render_complete == []
    assert runtime.render_cancel == []


def test_unregister_refuses_to_restore_or_remove_callbacks_during_async_render(runtime, tmp_path):
    module.register()
    submitted = module.handle_submit_look_render_batch(_params(tmp_path))
    module._job_timer()

    with pytest.raises(RuntimeError, match="asynchronous render is active"):
        module.unregister()

    assert module._active_render is not None
    assert runtime.scene.render.filepath != "/artist/original.png"
    assert runtime.render_complete == [module._on_render_complete]
    assert runtime.render_cancel == [module._on_render_cancel]
    assert runtime.cancel_calls == 1

    _cancel_render(runtime.scene)
    module.unregister()
    assert module._active_render is None
    assert runtime.scene.render.filepath == "/artist/original.png"
    assert runtime.render_complete == []
    assert runtime.render_cancel == []
    item = module._batches[submitted["batch_id"]]["items"][0]
    assert item["status"] == "CANCELLED"
