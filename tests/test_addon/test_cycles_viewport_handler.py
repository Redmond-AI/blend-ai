"""Tests for visible Cycles viewport preparation, capture, and restoration."""

from __future__ import annotations

import base64
import importlib.util
import os
import struct
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _load_handler_module():
    sys.modules["bpy"].app.handlers.persistent.side_effect = lambda callback: callback
    mock_dispatcher = MagicMock()
    mock_addon = MagicMock()
    mock_addon.dispatcher = mock_dispatcher

    old_addon = sys.modules.get("addon")
    old_dispatcher = sys.modules.get("addon.dispatcher")
    sys.modules["addon"] = mock_addon
    sys.modules["addon.dispatcher"] = mock_dispatcher
    try:
        handler_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "addon",
            "handlers",
            "cycles_viewport.py",
        )
        spec = importlib.util.spec_from_file_location(
            "addon.handlers.cycles_viewport", handler_path
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["addon.handlers.cycles_viewport"] = module
        spec.loader.exec_module(module)
    finally:
        if old_addon is None:
            sys.modules.pop("addon", None)
        else:
            sys.modules["addon"] = old_addon
        if old_dispatcher is None:
            sys.modules.pop("addon.dispatcher", None)
        else:
            sys.modules["addon.dispatcher"] = old_dispatcher
    return module, mock_dispatcher


@pytest.fixture(scope="module")
def cycles_viewport_handler():
    return _load_handler_module()


@pytest.fixture(autouse=True)
def _clear_sessions(cycles_viewport_handler):
    module, _dispatcher = cycles_viewport_handler
    module._sessions.clear()
    module.spatial_cache.reset_mock()
    yield
    module._sessions.clear()


class _Spaces(list):
    def __init__(self, active):
        super().__init__([active])
        self.active = active


def _png(width: int = 1600, height: int = 1200) -> bytes:
    """Return enough of a PNG header for the handler's dimension parser."""
    return b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", width, height)


def _make_bpy(*, screenshot_error: Exception | None = None):
    camera = SimpleNamespace(name="WarehouseCam", type="CAMERA")
    scene = SimpleNamespace(
        name="Mini Warehouse Relighting Fixture",
        render=SimpleNamespace(
            engine="BLENDER_EEVEE_NEXT",
            filepath="original.png",
            image_settings=SimpleNamespace(
                file_format="JPEG",
                color_depth="16",
                compression=50,
            ),
            resolution_x=960,
            resolution_y=540,
            resolution_percentage=100,
        ),
        cycles=SimpleNamespace(
            preview_samples=64,
            use_preview_denoising=False,
            device="CPU",
            samples=64,
            use_denoising=False,
        ),
        camera=camera,
    )

    shading = SimpleNamespace(
        type="SOLID",
        use_scene_lights_render=False,
        use_scene_world_render=False,
        use_scene_lights=False,
        use_scene_world=False,
        render_pass="NORMAL",
        use_compositor="DISABLED",
        use_dof=False,
    )
    overlay = SimpleNamespace(show_overlays=True)
    region_3d = SimpleNamespace(view_perspective="PERSP")
    space = SimpleNamespace(
        type="VIEW_3D",
        shading=shading,
        overlay=overlay,
        region_3d=region_3d,
        camera=None,
        show_gizmo=True,
    )
    window_region = SimpleNamespace(
        type="WINDOW",
        x=50,
        y=50,
        width=740,
        height=540,
    )
    header_region = SimpleNamespace(
        type="HEADER",
        x=10,
        y=590,
        width=800,
        height=30,
    )
    area = SimpleNamespace(
        type="VIEW_3D",
        x=10,
        y=20,
        width=800,
        height=600,
        spaces=_Spaces(space),
        regions=[header_region, window_region],
        tag_redraw=MagicMock(),
    )
    screen = SimpleNamespace(areas=[area])
    workspace = SimpleNamespace(name="Lighting")
    window = SimpleNamespace(
        workspace=workspace,
        screen=screen,
        scene=scene,
        x=0,
        y=0,
        width=800,
        height=600,
    )

    override = MagicMock()
    override.__enter__ = MagicMock(return_value=None)
    override.__exit__ = MagicMock(return_value=False)
    temp_override = MagicMock(return_value=override)

    captured_paths: list[str] = []
    render_states: list[dict] = []

    def screenshot_area(*, filepath):
        captured_paths.append(filepath)
        if screenshot_error is not None:
            raise screenshot_error
        with open(filepath, "wb") as image_file:
            image_file.write(_png())
        return {"FINISHED"}

    def render(*, write_still, scene: str):
        render_states.append(
            {
                "write_still": write_still,
                "scene": scene,
                "samples": fake_bpy.context.scene.cycles.samples,
                "denoise": fake_bpy.context.scene.cycles.use_denoising,
                "resolution_x": fake_bpy.context.scene.render.resolution_x,
                "resolution_y": fake_bpy.context.scene.render.resolution_y,
                "percentage": fake_bpy.context.scene.render.resolution_percentage,
                "format": fake_bpy.context.scene.render.image_settings.file_format,
                "filepath": fake_bpy.context.scene.render.filepath,
            }
        )
        with open(fake_bpy.context.scene.render.filepath, "wb") as image_file:
            image_file.write(_png(320, 180))
        return {"FINISHED"}

    fake_bpy = SimpleNamespace(
        app=SimpleNamespace(
            handlers=SimpleNamespace(load_pre=[], load_post=[]),
        ),
        context=SimpleNamespace(
            window_manager=SimpleNamespace(windows=[window]),
            window=window,
            scene=scene,
            view_layer=SimpleNamespace(update=MagicMock()),
            temp_override=temp_override,
        ),
        data=SimpleNamespace(
            workspaces={"Lighting": workspace},
            objects={"WarehouseCam": camera},
        ),
        ops=SimpleNamespace(
            screen=SimpleNamespace(screenshot_area=MagicMock(side_effect=screenshot_area)),
            render=SimpleNamespace(render=MagicMock(side_effect=render)),
        ),
    )
    state = SimpleNamespace(
        camera=camera,
        scene=scene,
        space=space,
        area=area,
        region=window_region,
        screen=screen,
        workspace=workspace,
        window=window,
        captured_paths=captured_paths,
        render_states=render_states,
        temp_override=temp_override,
    )
    return fake_bpy, state


def _prepare(module, fake_bpy, **overrides):
    module.bpy = fake_bpy
    params = {
        "session_id": "preview-1",
        "workspace_name": "Lighting",
        "area_index": 0,
        "camera_name": "WarehouseCam",
        "preview_samples": 16,
        "denoise": True,
        "device": "GPU",
    }
    params.update(overrides)
    return module.handle_prepare_cycles_viewport(params)


def test_prepare_configures_exact_visible_cycles_view(cycles_viewport_handler):
    module, _dispatcher = cycles_viewport_handler
    fake_bpy, state = _make_bpy()

    result = _prepare(module, fake_bpy, keep_session=True)

    assert result["session_id"] == "preview-1"
    assert result["workspace_name"] == "Lighting"
    assert result["area_index"] == 0
    assert result["camera_name"] == "WarehouseCam"
    assert result["width"] == 800
    assert result["height"] == 600
    assert state.scene.render.engine == "CYCLES"
    assert state.scene.cycles.preview_samples == 16
    assert state.scene.cycles.use_preview_denoising is True
    assert state.scene.cycles.device == "GPU"
    assert state.space.shading.type == "RENDERED"
    assert state.space.shading.use_scene_lights_render is True
    assert state.space.shading.use_scene_world_render is True
    assert state.space.shading.render_pass == "COMBINED"
    assert state.space.shading.use_compositor == "CAMERA"
    assert state.space.shading.use_dof is True
    assert state.space.overlay.show_overlays is False
    assert state.space.show_gizmo is False
    assert state.space.region_3d.view_perspective == "CAMERA"
    assert state.space.camera is state.camera

    override = state.temp_override.call_args.kwargs
    assert override == {
        "window": state.window,
        "screen": state.screen,
        "area": state.area,
        "region": state.region,
        "scene": state.scene,
    }


def test_none_area_index_resolves_to_first_view(cycles_viewport_handler):
    module, _dispatcher = cycles_viewport_handler
    fake_bpy, state = _make_bpy()

    result = _prepare(module, fake_bpy, area_index=None, keep_session=True)

    assert result["area_index"] == 0


def test_none_area_index_rejects_ambiguous_workspace(cycles_viewport_handler):
    module, _dispatcher = cycles_viewport_handler
    fake_bpy, state = _make_bpy()
    second_area = SimpleNamespace(type="VIEW_3D", width=640, height=480)
    state.screen.areas.append(second_area)

    with pytest.raises(ValueError, match=r"0=800x600, 1=640x480"):
        _prepare(module, fake_bpy, area_index=None, keep_session=True)


def test_hidden_named_workspace_fails_without_switching_window(cycles_viewport_handler):
    module, _dispatcher = cycles_viewport_handler
    fake_bpy, state = _make_bpy()
    original_workspace = SimpleNamespace(name="Layout")
    state.window.workspace = original_workspace

    with pytest.raises(RuntimeError, match="not visible in any Blender window"):
        _prepare(module, fake_bpy, keep_session=True)

    assert state.window.workspace is original_workspace
    assert module._sessions == {}


def test_matching_explicit_session_is_reused_and_reconfigured(cycles_viewport_handler):
    module, _dispatcher = cycles_viewport_handler
    fake_bpy, state = _make_bpy()
    _prepare(module, fake_bpy, keep_session=True)
    original_change_count = len(module._sessions["preview-1"]["changes"])

    result = _prepare(
        module,
        fake_bpy,
        preview_samples=8,
        denoise=False,
        device="KEEP",
        keep_session=True,
    )

    assert result["session_id"] == "preview-1"
    assert result["reused_session"] is True
    assert result["preview_samples"] == 8
    assert result["device"] == "GPU"
    assert state.scene.cycles.preview_samples == 8
    assert state.scene.cycles.use_preview_denoising is False
    assert len(module._sessions["preview-1"]["changes"]) == original_change_count

    module.handle_restore_cycles_viewport({"session_id": "preview-1"})
    assert state.scene.cycles.preview_samples == 64
    assert state.scene.cycles.device == "CPU"


def test_omitted_session_id_reuses_sole_matching_session(cycles_viewport_handler):
    module, _dispatcher = cycles_viewport_handler
    fake_bpy, state = _make_bpy()
    _prepare(module, fake_bpy, keep_session=True)

    result = module.handle_prepare_cycles_viewport(
        {
            "session_id": None,
            "workspace_name": "Lighting",
            "area_index": 0,
            "camera_name": None,
            "preview_samples": 4,
            "denoise": True,
            "device": "KEEP",
            "keep_session": True,
        }
    )

    assert result["session_id"] == "preview-1"
    assert result["reused_session"] is True
    assert result["preview_samples"] == 4


def test_different_explicit_session_is_rejected(cycles_viewport_handler):
    module, _dispatcher = cycles_viewport_handler
    fake_bpy, _state = _make_bpy()
    _prepare(module, fake_bpy, keep_session=True)

    with pytest.raises(RuntimeError, match="cannot prepare different session"):
        _prepare(module, fake_bpy, session_id="preview-2", keep_session=True)


def test_restore_returns_every_temporary_setting(cycles_viewport_handler):
    module, _dispatcher = cycles_viewport_handler
    fake_bpy, state = _make_bpy()
    _prepare(module, fake_bpy, keep_session=True)

    result = module.handle_restore_cycles_viewport({"session_id": "preview-1"})

    assert result["restored"] is True
    assert state.scene.render.engine == "BLENDER_EEVEE_NEXT"
    assert state.scene.cycles.preview_samples == 64
    assert state.scene.cycles.use_preview_denoising is False
    assert state.scene.cycles.device == "CPU"
    assert state.space.shading.type == "SOLID"
    assert state.space.shading.use_scene_lights_render is False
    assert state.space.shading.use_scene_world_render is False
    assert state.space.shading.render_pass == "NORMAL"
    assert state.space.shading.use_compositor == "DISABLED"
    assert state.space.shading.use_dof is False
    assert state.space.overlay.show_overlays is True
    assert state.space.show_gizmo is True
    assert state.space.region_3d.view_perspective == "PERSP"


def test_failed_restore_keeps_session_for_retry(cycles_viewport_handler):
    module, _dispatcher = cycles_viewport_handler
    fake_bpy, _state = _make_bpy()
    _prepare(module, fake_bpy, keep_session=True)

    with patch.object(
        module,
        "_restore_changes",
        side_effect=[["engine: temporarily read-only"], []],
    ):
        with pytest.raises(RuntimeError, match="temporarily read-only"):
            module.handle_restore_cycles_viewport({"session_id": "preview-1"})
        assert "preview-1" in module._sessions

        restored = module.handle_restore_cycles_viewport({"session_id": "preview-1"})

    assert restored["restored"] is True
    assert "preview-1" not in module._sessions
    assert module._sessions == {}


def test_capture_returns_png_and_hidpi_scaled_region(cycles_viewport_handler):
    module, _dispatcher = cycles_viewport_handler
    fake_bpy, state = _make_bpy()
    _prepare(module, fake_bpy, keep_session=True)
    state.temp_override.reset_mock()

    result = module.handle_capture_cycles_viewport_area(
        {"session_id": "preview-1", "keep_session": True}
    )

    assert base64.b64decode(result["image_base64"]) == _png()
    assert result["source_width"] == 1600
    assert result["source_height"] == 1200
    assert result["region_rect"] == {
        "x": 80,
        "y": 60,
        "width": 1480,
        "height": 1080,
        "origin": "TOP_LEFT",
    }
    assert result["mode"] == "cycles_rendered_viewport"
    assert result["session_active"] is True
    assert state.temp_override.call_args.kwargs["area"] is state.area
    assert not os.path.exists(state.captured_paths[0])
    assert not os.path.exists(os.path.dirname(state.captured_paths[0]))


def test_metadata_only_capture_returns_macos_window_target(cycles_viewport_handler):
    module, _dispatcher = cycles_viewport_handler
    fake_bpy, state = _make_bpy()
    _prepare(module, fake_bpy, keep_session=False)

    result = module.handle_capture_cycles_viewport_area(
        {
            "session_id": "preview-1",
            "keep_session": False,
            "metadata_only": True,
        }
    )

    assert result["capture_backend"] == "MACOS_SCREENCAPTUREKIT"
    assert result["capture_target"] == {
        "process_id": os.getpid(),
        "window_content_width": 800,
        "window_content_height": 600,
        "window_x": 0,
        "window_y": 0,
        "region_rect": {
            "x": 50,
            "y": 10,
            "width": 740,
            "height": 540,
            "origin": "TOP_LEFT",
        },
    }
    assert result["session_active"] is True
    assert module._sessions["preview-1"]["scene"] is state.scene
    assert state.captured_paths == []


def test_quick_render_is_denoised_bounded_and_restores_settings(
    cycles_viewport_handler,
):
    module, _dispatcher = cycles_viewport_handler
    fake_bpy, state = _make_bpy()
    _prepare(module, fake_bpy, keep_session=True)

    result = module.handle_capture_cycles_quick_render(
        {"session_id": "preview-1", "max_size": 320, "denoise": True}
    )

    assert result["capture_backend"] == "QUICK_CYCLES_RENDER"
    assert result["source_width"] == 320
    assert result["source_height"] == 180
    assert result["denoise"] is True
    assert base64.b64decode(result["image_base64"]).startswith(b"\x89PNG")
    assert state.render_states == [
        {
            "write_still": True,
            "scene": "Mini Warehouse Relighting Fixture",
            "samples": 16,
            "denoise": True,
            "resolution_x": 320,
            "resolution_y": 180,
            "percentage": 100,
            "format": "PNG",
            "filepath": state.render_states[0]["filepath"],
        }
    ]
    assert not os.path.exists(state.render_states[0]["filepath"])
    assert state.scene.render.filepath == "original.png"
    assert state.scene.render.resolution_percentage == 100
    assert state.scene.render.resolution_x == 960
    assert state.scene.render.resolution_y == 540
    assert state.scene.render.image_settings.file_format == "JPEG"
    assert state.scene.cycles.samples == 64
    assert state.scene.cycles.use_denoising is False
    assert module._sessions["preview-1"]["scene"] is state.scene
    module.spatial_cache.begin_render_evaluation.assert_called_once_with()
    module.spatial_cache.end_render_evaluation.assert_called_once_with()
    fake_bpy.context.view_layer.update.assert_called_once_with()


def test_quick_render_reports_primary_and_session_cleanup_failures(
    cycles_viewport_handler,
):
    module, _dispatcher = cycles_viewport_handler
    fake_bpy, _state = _make_bpy()
    _prepare(module, fake_bpy, keep_session=True)
    fake_bpy.ops.render.render.side_effect = RuntimeError("render failed")

    with patch.object(
        module,
        "_restore_session",
        side_effect=RuntimeError("session restore failed"),
    ):
        with pytest.raises(RuntimeError, match="render failed.*session restore failed"):
            module.handle_capture_cycles_quick_render(
                {"session_id": "preview-1", "max_size": 320, "denoise": True}
            )


def test_quick_render_cleanup_failure_prevents_success(cycles_viewport_handler):
    module, _dispatcher = cycles_viewport_handler
    fake_bpy, _state = _make_bpy()
    _prepare(module, fake_bpy, keep_session=True)

    with (
        patch.object(
            module,
            "_restore_changes",
            return_value=["resolution_x: readonly"],
        ),
        patch.object(module, "_restore_session") as restore_session,
    ):
        with pytest.raises(RuntimeError, match="cleanup failed.*resolution_x"):
            module.handle_capture_cycles_quick_render(
                {"session_id": "preview-1", "max_size": 320, "denoise": True}
            )

    restore_session.assert_called_once_with("preview-1")


def test_capture_restores_by_default(cycles_viewport_handler):
    module, _dispatcher = cycles_viewport_handler
    fake_bpy, state = _make_bpy()
    _prepare(module, fake_bpy, keep_session=False)

    result = module.handle_capture_cycles_viewport_area({"session_id": "preview-1"})

    assert result["session_active"] is False
    assert module._sessions == {}
    assert state.scene.render.engine == "BLENDER_EEVEE_NEXT"
    assert state.space.shading.type == "SOLID"


def test_capture_failure_restores_session(cycles_viewport_handler):
    module, _dispatcher = cycles_viewport_handler
    fake_bpy, state = _make_bpy(screenshot_error=RuntimeError("capture failed"))
    _prepare(module, fake_bpy, keep_session=True)

    with pytest.raises(RuntimeError, match="capture failed"):
        module.handle_capture_cycles_viewport_area(
            {"session_id": "preview-1", "keep_session": True}
        )

    assert module._sessions == {}
    assert state.scene.render.engine == "BLENDER_EEVEE_NEXT"
    assert state.space.shading.type == "SOLID"


def test_prepare_context_failure_restores_all_mutations(cycles_viewport_handler):
    module, _dispatcher = cycles_viewport_handler
    fake_bpy, state = _make_bpy()
    fake_bpy.context.temp_override.side_effect = RuntimeError("invalid context")

    with pytest.raises(RuntimeError, match="invalid context"):
        _prepare(module, fake_bpy, keep_session=True)

    assert module._sessions == {}
    assert state.scene.render.engine == "BLENDER_EEVEE_NEXT"
    assert state.scene.cycles.preview_samples == 64
    assert state.space.shading.type == "SOLID"


def test_prepare_cleanup_failure_retains_recovery_session(cycles_viewport_handler):
    module, _dispatcher = cycles_viewport_handler
    fake_bpy, _state = _make_bpy()
    fake_bpy.context.temp_override.side_effect = RuntimeError("invalid context")

    with patch.object(
        module,
        "_restore_changes",
        side_effect=[["engine: temporarily read-only"], []],
    ):
        with pytest.raises(RuntimeError, match="Recovery session 'preview-1'"):
            _prepare(module, fake_bpy, keep_session=True)
        assert "preview-1" in module._sessions

        fake_bpy.context.temp_override.side_effect = None
        restored = module.handle_restore_cycles_viewport({"session_id": "preview-1"})

    assert restored["restored"] is True
    assert module._sessions == {}


@pytest.mark.parametrize("device", ["METAL", "AUTO", 1])
def test_prepare_rejects_unknown_device(cycles_viewport_handler, device):
    module, _dispatcher = cycles_viewport_handler
    fake_bpy, _state = _make_bpy()
    module.bpy = fake_bpy

    with pytest.raises(ValueError, match="device"):
        module.handle_prepare_cycles_viewport({"device": device})


def test_keep_device_preserves_current_cycles_device(cycles_viewport_handler):
    module, _dispatcher = cycles_viewport_handler
    fake_bpy, state = _make_bpy()

    result = _prepare(module, fake_bpy, device="KEEP", keep_session=True)

    assert result["device"] == "CPU"
    assert state.scene.cycles.device == "CPU"


def test_restore_without_id_is_safe_when_nothing_active(cycles_viewport_handler):
    module, _dispatcher = cycles_viewport_handler

    result = module.handle_restore_cycles_viewport({})

    assert result == {
        "restored": False,
        "session_id": None,
        "restored_session_ids": [],
    }


def test_file_load_clears_stale_preview_sessions(cycles_viewport_handler):
    module, _dispatcher = cycles_viewport_handler
    module._sessions["stale"] = {"old_rna": object()}

    module._clear_sessions_on_load(None)

    assert module._sessions == {}


def test_load_pre_restores_then_load_post_hard_clears(cycles_viewport_handler):
    module, _dispatcher = cycles_viewport_handler
    fake_bpy, state = _make_bpy()
    _prepare(module, fake_bpy, keep_session=True)
    assert state.scene.render.engine == "CYCLES"

    module.register()
    assert module._restore_sessions_before_load in fake_bpy.app.handlers.load_pre
    assert module._clear_sessions_on_load in fake_bpy.app.handlers.load_post

    module._restore_sessions_before_load(None, object())
    assert state.scene.render.engine == "BLENDER_EEVEE_NEXT"
    assert module._sessions == {}

    module._sessions["stale"] = {"old_rna": object()}
    module._clear_sessions_on_load(None, object())
    assert module._sessions == {}

    module.unregister()
    assert module._restore_sessions_before_load not in fake_bpy.app.handlers.load_pre
    assert module._clear_sessions_on_load not in fake_bpy.app.handlers.load_post


def test_restore_specific_session_is_idempotent(cycles_viewport_handler):
    module, _dispatcher = cycles_viewport_handler

    result = module.handle_restore_cycles_viewport({"session_id": "already-restored"})

    assert result == {
        "restored": False,
        "session_id": "already-restored",
        "restored_session_ids": [],
    }


def test_registers_all_four_private_wire_commands(cycles_viewport_handler):
    module, mock_dispatcher = cycles_viewport_handler
    mock_dispatcher.reset_mock()

    module.register()

    names = [call.args[0] for call in mock_dispatcher.register_handler.call_args_list]
    assert names == [
        "prepare_cycles_viewport",
        "capture_cycles_viewport_area",
        "capture_cycles_quick_render",
        "restore_cycles_viewport",
    ]
