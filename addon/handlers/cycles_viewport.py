"""Visible Cycles viewport preparation and screenshot handlers.

These handlers deliberately split viewport preparation, capture, and restoration
into separate commands.  Blender's UI must return to its event loop for a
Rendered viewport to redraw and for Cycles to accumulate samples; sleeping in a
handler would block exactly that work.
"""

from __future__ import annotations

import base64
import math
import os
import struct
import sys
import tempfile
import uuid
from typing import Any

import bpy

from .. import dispatcher, spatial_cache


_MIN_PREVIEW_SAMPLES = 1
_MAX_PREVIEW_SAMPLES = 4096
_ALLOWED_DEVICES = {"KEEP", "CPU", "GPU"}

# UI and render settings are global Blender state.  Supporting overlapping
# sessions would make restoration order-dependent, so only one is allowed.
_sessions: dict[str, dict[str, Any]] = {}


def active_session_ids() -> list[str]:
    """Return retained viewport-session ids for cross-handler coordination."""
    return list(_sessions)

try:
    _persistent = bpy.app.handlers.persistent
except (AttributeError, TypeError):
    def _persistent(callback: Any) -> Any:
        return callback


def _load_post_handlers():
    """Return Blender's load-post handler list when the UI build exposes it."""
    return getattr(getattr(getattr(bpy, "app", None), "handlers", None), "load_post", None)


def _load_pre_handlers():
    """Return Blender's load-pre handler list when exposed by this build."""
    return getattr(getattr(getattr(bpy, "app", None), "handlers", None), "load_pre", None)


@_persistent
def _restore_sessions_before_load(_unused: Any, *_handler_args: Any) -> None:
    """Best-effort UI restoration while the current file's RNA is still valid."""
    for session_id in list(reversed(list(_sessions))):
        try:
            _restore_session(session_id)
        except Exception:
            # load_post performs a hard clear after Blender replaces the file.
            # Raising from a load handler can prevent the user's file open.
            continue


@_persistent
def _clear_sessions_on_load(_unused: Any, *_handler_args: Any) -> None:
    """Forget stale RNA references after Blender replaces the open file."""
    _sessions.clear()


def _require_int(params: dict, name: str, default: int, minimum: int, maximum: int) -> int:
    """Read and validate a bounded integer command parameter."""
    value = params.get(name, default)
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"'{name}' must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"'{name}' must be between {minimum} and {maximum}")
    return value


def _workspace_by_name(name: str):
    """Return a Blender workspace by exact name."""
    workspace = bpy.data.workspaces.get(name)
    if workspace is None:
        raise ValueError(f"Workspace '{name}' not found")
    return workspace


def _window_for_workspace(workspace):
    """Resolve a window already showing *workspace* without changing user UI."""
    windows = list(bpy.context.window_manager.windows)
    if not windows:
        raise RuntimeError("No visible Blender window is available")

    context_window = getattr(bpy.context, "window", None)
    matching = [window for window in windows if window.workspace == workspace]
    if context_window in matching:
        return context_window
    if matching:
        return matching[0]
    raise RuntimeError(
        f"Workspace '{workspace.name}' exists but is not visible in any Blender window"
    )


def _resolve_view3d(window, area_index: int | None):
    """Resolve one VIEW_3D area, its active space, and its WINDOW region."""
    screen = window.screen
    areas = [area for area in screen.areas if area.type == "VIEW_3D"]
    if area_index is None:
        if len(areas) != 1:
            available = ", ".join(
                f"{index}={int(area.width)}x{int(area.height)}"
                for index, area in enumerate(areas)
            )
            detail = available or "none"
            raise ValueError(
                "'area_index' is required unless the workspace has exactly one "
                f"VIEW_3D area; available: {detail}"
            )
        area_index = 0
    if area_index >= len(areas):
        raise ValueError(
            f"VIEW_3D area index {area_index} is unavailable; "
            f"workspace has {len(areas)} VIEW_3D area(s)"
        )

    area = areas[area_index]
    space = getattr(area.spaces, "active", None)
    if space is None or space.type != "VIEW_3D":
        space = next((item for item in area.spaces if item.type == "VIEW_3D"), None)
    if space is None:
        raise RuntimeError("Selected VIEW_3D area has no VIEW_3D space")

    regions = [region for region in area.regions if region.type == "WINDOW"]
    if not regions:
        raise RuntimeError("Selected VIEW_3D area has no WINDOW region")
    # A normal 3D View has one WINDOW region.  Choosing the largest is stable
    # for unusual layouts and avoids accidentally selecting a zero-sized one.
    region = max(regions, key=lambda item: int(item.width) * int(item.height))
    return screen, area, space, region, area_index


def _camera_for_scene(scene, camera_name: str | None):
    """Resolve the requested camera, or the scene's active camera."""
    if camera_name:
        camera = bpy.data.objects.get(camera_name)
        if camera is None:
            raise ValueError(f"Camera '{camera_name}' not found")
        if camera.type != "CAMERA":
            raise ValueError(f"Object '{camera_name}' is not a camera")
        return camera

    camera = scene.camera
    if camera is None:
        raise ValueError("Scene has no active camera")
    if camera.type != "CAMERA":
        raise ValueError(f"Active camera object '{camera.name}' is not a camera")
    return camera


def _record_and_set(changes: list[tuple[Any, str, Any]], target, attribute: str, value) -> None:
    """Record a Blender RNA property before assigning a temporary value."""
    changes.append((target, attribute, getattr(target, attribute)))
    setattr(target, attribute, value)


def _record_and_set_if_present(
    changes: list[tuple[Any, str, Any]], target, attribute: str, value
) -> bool:
    """Temporarily set an RNA property when supported by this Blender version."""
    if not hasattr(target, attribute):
        return False
    _record_and_set(changes, target, attribute, value)
    return True


def _restore_changes(changes: list[tuple[Any, str, Any]]) -> list[str]:
    """Restore all recorded properties in reverse mutation order."""
    errors: list[str] = []
    for target, attribute, value in reversed(changes):
        try:
            setattr(target, attribute, value)
        except Exception as exc:  # Blender references can become invalid after UI edits.
            errors.append(f"{attribute}: {exc}")
    return errors


def _flush_dependency_graph_updates() -> None:
    """Best-effort flush while a scoped revision classifier is still active.

    Blender may defer Scene/Camera tags from a synchronous render until the
    view layer is updated.  Flushing before ending the render scope avoids a
    wall-clock suppression window while still allowing the very next artist
    edit to invalidate spatial context normally.
    """
    view_layer = getattr(getattr(bpy, "context", None), "view_layer", None)
    update = getattr(view_layer, "update", None)
    if callable(update):
        try:
            update()
        except (AttributeError, ReferenceError, RuntimeError):
            # Failure is cache-safe: once the scope ends, any deferred update
            # is classified normally and may conservatively invalidate geometry.
            pass


def _tag_redraw(area) -> None:
    """Request a redraw without blocking Blender's main thread."""
    try:
        area.tag_redraw()
    except (AttributeError, ReferenceError):
        pass


def _context_override(session: dict[str, Any]):
    """Build an exact, internally consistent Blender UI context override."""
    return bpy.context.temp_override(
        window=session["window"],
        screen=session["screen"],
        area=session["area"],
        region=session["region"],
        scene=session["scene"],
    )


def _png_dimensions(data: bytes) -> tuple[int, int]:
    """Read PNG pixel dimensions without adding an image-library dependency."""
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise RuntimeError("Blender screenshot is not a valid PNG")
    width, height = struct.unpack(">II", data[16:24])
    if width <= 0 or height <= 0:
        raise RuntimeError("Blender screenshot has invalid dimensions")
    return width, height


def _camera_frame_bounds(session: dict[str, Any]) -> tuple[int, int, int, int] | None:
    """Return the visible camera frame in WINDOW-region coordinates."""
    camera = session.get("camera")
    data = getattr(camera, "data", None)
    view_frame = getattr(data, "view_frame", None)
    matrix = getattr(camera, "matrix_world", None)
    if not callable(view_frame) or matrix is None:
        return None
    try:
        from bpy_extras import view3d_utils

        points = []
        for corner in view_frame(scene=session["scene"]):
            projected = view3d_utils.location_3d_to_region_2d(
                session["region"],
                session["space"].region_3d,
                matrix @ corner,
            )
            if projected is None:
                return None
            points.append((float(projected.x), float(projected.y)))
        minimum_x = max(0, math.floor(min(point[0] for point in points)))
        maximum_x = min(
            int(session["region"].width),
            math.ceil(max(point[0] for point in points)),
        )
        minimum_y = max(0, math.floor(min(point[1] for point in points)))
        maximum_y = min(
            int(session["region"].height),
            math.ceil(max(point[1] for point in points)),
        )
        if maximum_x <= minimum_x or maximum_y <= minimum_y:
            return None
        return minimum_x, minimum_y, maximum_x - minimum_x, maximum_y - minimum_y
    except Exception:
        return None


def _region_rect(session: dict[str, Any], image_width: int, image_height: int) -> dict:
    """Convert the visible camera frame to a Retina-aware top-left PNG crop."""
    area = session["area"]
    region = session["region"]
    area_width = max(int(area.width), 1)
    area_height = max(int(area.height), 1)
    scale_x = image_width / area_width
    scale_y = image_height / area_height

    local_x, local_bottom, local_width, local_height = (
        _camera_frame_bounds(session)
        or (0, 0, int(region.width), int(region.height))
    )
    relative_x = int(region.x) - int(area.x) + local_x
    relative_bottom = int(region.y) - int(area.y) + local_bottom
    relative_top = area_height - (relative_bottom + local_height)

    x = round(relative_x * scale_x)
    y = round(relative_top * scale_y)
    width = round(local_width * scale_x)
    height = round(local_height * scale_y)

    # Clamp for rounding and platform-specific window decoration differences.
    x = max(0, min(x, image_width))
    y = max(0, min(y, image_height))
    width = max(0, min(width, image_width - x))
    height = max(0, min(height, image_height - y))
    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "origin": "TOP_LEFT",
    }


def _window_content_region_rect(session: dict[str, Any]) -> dict[str, Any]:
    """Return the camera frame relative to Blender's window content pixels.

    macOS ScreenCaptureKit captures the complete native window, including its
    title bar, whereas Blender's Window RNA dimensions describe the content
    beneath that chrome.  Returning a content-relative rectangle lets the MCP
    process add the measured title-bar offset after inspecting the native PNG.
    """
    window = session["window"]
    region = session["region"]
    content_width = max(int(window.width), 1)
    content_height = max(int(window.height), 1)
    local_x, local_bottom, local_width, local_height = (
        _camera_frame_bounds(session)
        or (0, 0, int(region.width), int(region.height))
    )
    x = int(region.x) + local_x
    bottom = int(region.y) + local_bottom
    y = content_height - (bottom + local_height)
    x = max(0, min(x, content_width))
    y = max(0, min(y, content_height))
    width = max(0, min(local_width, content_width - x))
    height = max(0, min(local_height, content_height - y))
    if width <= 0 or height <= 0:
        raise RuntimeError("Camera frame lies outside the visible Blender window")
    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "origin": "TOP_LEFT",
    }


def _restore_session(session_id: str) -> None:
    """Restore and forget one prepared viewport session."""
    session = _sessions.get(session_id)
    if session is None:
        raise ValueError(f"Unknown Cycles viewport session '{session_id}'")

    _tag_redraw(session["area"])
    errors = _restore_changes(session["changes"])
    _tag_redraw(session["area"])
    try:
        current_screen = session["window"].screen
        for area in current_screen.areas:
            if area.type == "VIEW_3D":
                _tag_redraw(area)
    except (AttributeError, ReferenceError):
        pass

    if errors:
        raise RuntimeError("Failed to restore Cycles viewport session: " + "; ".join(errors))
    _sessions.pop(session_id, None)


def _configure_existing_session(
    session: dict[str, Any],
    *,
    camera,
    preview_samples: int,
    denoise: bool,
    device: str | None,
    keep_session: bool,
) -> None:
    """Reconfigure a retained session without replacing its original snapshot."""
    scene = session["scene"]
    space = session["space"]
    scene.render.engine = "CYCLES"
    scene.cycles.preview_samples = preview_samples
    scene.cycles.use_preview_denoising = denoise
    if device not in {None, "KEEP"}:
        scene.cycles.device = device

    scene.camera = camera
    if hasattr(space, "camera"):
        space.camera = camera
    space.shading.type = "RENDERED"
    for attribute in (
        "use_scene_lights_render",
        "use_scene_world_render",
        "use_scene_lights",
        "use_scene_world",
    ):
        if hasattr(space.shading, attribute):
            setattr(space.shading, attribute, True)
    if hasattr(space.shading, "render_pass"):
        space.shading.render_pass = "COMBINED"
    if hasattr(space.shading, "use_compositor"):
        space.shading.use_compositor = "CAMERA"
    if hasattr(space.shading, "use_dof"):
        space.shading.use_dof = True
    space.overlay.show_overlays = False
    if hasattr(space, "show_gizmo"):
        space.show_gizmo = False
    space.region_3d.view_perspective = "CAMERA"

    session["camera"] = camera
    session["preview_samples"] = preview_samples
    session["denoise"] = denoise
    session["device"] = str(scene.cycles.device)
    session["keep_session"] = keep_session

    with _context_override(session):
        _tag_redraw(session["area"])


def _prepare_result(session: dict[str, Any], *, reused: bool) -> dict:
    """Build the stable wire result for a new or retained session."""
    return {
        "session_id": session["session_id"],
        "workspace_name": session["workspace_name"],
        "area_index": session["area_index"],
        "camera_name": session["camera"].name,
        "engine": "CYCLES",
        "shading": "RENDERED",
        "preview_samples": session["preview_samples"],
        "denoise": session["denoise"],
        "device": session["device"],
        "width": int(session["area"].width),
        "height": int(session["area"].height),
        "workspace_switched": session["workspace_switched"],
        "keep_session": session["keep_session"],
        "reused_session": reused,
        "capture_backend_required": (
            "MACOS_SCREENCAPTUREKIT"
            if sys.platform == "darwin"
            else "BLENDER_SCREENSHOT_AREA"
        ),
    }


def handle_prepare_cycles_viewport(params: dict) -> dict:
    """Prepare a visible, named VIEW_3D area for camera-view Cycles preview."""
    supplied_session_id = params.get("session_id")
    if supplied_session_id is not None and (
        not isinstance(supplied_session_id, str) or not supplied_session_id.strip()
    ):
        raise ValueError("'session_id' must be a non-empty string when provided")
    supplied_session_id = supplied_session_id.strip() if supplied_session_id else None

    raw_area_index = params.get("area_index")
    area_index = (
        None
        if raw_area_index is None
        else _require_int(params, "area_index", 0, 0, 1024)
    )
    preview_samples = _require_int(
        params,
        "preview_samples",
        16,
        _MIN_PREVIEW_SAMPLES,
        _MAX_PREVIEW_SAMPLES,
    )
    denoise = bool(params.get("denoise", True))
    keep_session = bool(params.get("keep_session", False))

    device = params.get("device")
    if device is not None:
        if not isinstance(device, str) or device.upper() not in _ALLOWED_DEVICES:
            raise ValueError("'device' must be 'KEEP', 'CPU', or 'GPU' when provided")
        device = device.upper()

    workspace_name = params.get("workspace_name")
    if workspace_name is not None and (
        not isinstance(workspace_name, str) or not workspace_name.strip()
    ):
        raise ValueError("'workspace_name' must be a non-empty string when provided")

    active_session = next(iter(_sessions.values()), None)
    if active_session is not None:
        active_session_id = active_session["session_id"]
        if supplied_session_id is not None and supplied_session_id != active_session_id:
            raise RuntimeError(
                f"Cycles viewport session '{active_session_id}' is still active; "
                f"cannot prepare different session '{supplied_session_id}'"
            )
        if workspace_name and workspace_name.strip() != active_session["workspace_name"]:
            raise RuntimeError(
                f"Active session uses workspace '{active_session['workspace_name']}', "
                f"not '{workspace_name.strip()}'"
            )
        if area_index is not None and area_index != active_session["area_index"]:
            raise RuntimeError(
                f"Active session uses VIEW_3D area {active_session['area_index']}, "
                f"not {area_index}"
            )
        if active_session["window"].workspace != active_session["workspace"]:
            _restore_session(active_session_id)
            raise RuntimeError("Retained Cycles viewport workspace is no longer visible")

        scene = active_session["scene"]
        camera = _camera_for_scene(scene, params.get("camera_name"))
        try:
            _configure_existing_session(
                active_session,
                camera=camera,
                preview_samples=preview_samples,
                denoise=denoise,
                device=device,
                keep_session=keep_session,
            )
        except Exception:
            _restore_session(active_session_id)
            raise
        return _prepare_result(active_session, reused=True)

    session_id = supplied_session_id or uuid.uuid4().hex
    changes: list[tuple[Any, str, Any]] = []
    area = None
    session: dict[str, Any] | None = None
    try:
        if workspace_name:
            workspace = _workspace_by_name(workspace_name.strip())
        else:
            context_window = getattr(bpy.context, "window", None)
            if context_window is None:
                windows = list(bpy.context.window_manager.windows)
                if not windows:
                    raise RuntimeError("No visible Blender window is available")
                context_window = windows[0]
            workspace = context_window.workspace

        window = _window_for_workspace(workspace)
        workspace_switched = False

        screen, area, space, region, resolved_area_index = _resolve_view3d(
            window, area_index
        )
        scene = getattr(window, "scene", None) or bpy.context.scene
        camera = _camera_for_scene(scene, params.get("camera_name"))

        _record_and_set(changes, scene.render, "engine", "CYCLES")
        _record_and_set(changes, scene.cycles, "preview_samples", preview_samples)
        _record_and_set(changes, scene.cycles, "use_preview_denoising", denoise)
        if device not in {None, "KEEP"}:
            _record_and_set(changes, scene.cycles, "device", device)

        _record_and_set(changes, scene, "camera", camera)
        _record_and_set_if_present(changes, space, "camera", camera)
        _record_and_set(changes, space.shading, "type", "RENDERED")

        # Blender has separate scene-light/world flags for Material Preview and
        # Rendered modes across supported versions.  Set every available render
        # flag and restore it exactly later.
        _record_and_set_if_present(changes, space.shading, "use_scene_lights_render", True)
        _record_and_set_if_present(changes, space.shading, "use_scene_world_render", True)
        _record_and_set_if_present(changes, space.shading, "use_scene_lights", True)
        _record_and_set_if_present(changes, space.shading, "use_scene_world", True)
        _record_and_set_if_present(changes, space.shading, "render_pass", "COMBINED")
        _record_and_set_if_present(changes, space.shading, "use_compositor", "CAMERA")
        _record_and_set_if_present(changes, space.shading, "use_dof", True)

        _record_and_set(changes, space.overlay, "show_overlays", False)
        _record_and_set_if_present(changes, space, "show_gizmo", False)

        region_3d = space.region_3d
        _record_and_set(changes, region_3d, "view_perspective", "CAMERA")

        session = {
            "session_id": session_id,
            "window": window,
            "screen": screen,
            "area": area,
            "space": space,
            "region": region,
            "scene": scene,
            "camera": camera,
            "workspace": workspace,
            "workspace_name": workspace.name,
            "workspace_switched": workspace_switched,
            "area_index": resolved_area_index,
            "preview_samples": preview_samples,
            "denoise": denoise,
            "device": str(scene.cycles.device),
            "keep_session": keep_session,
            "changes": changes,
        }
        _sessions[session_id] = session

        # Assignments above are context-independent RNA writes.  Entering the
        # exact UI context here validates that the resolved window/screen/area/
        # region combination is usable before returning the session.
        with _context_override(session):
            _tag_redraw(area)

        return _prepare_result(session, reused=False)
    except Exception as prepare_error:
        cleanup_errors = _restore_changes(changes)
        if cleanup_errors and session is None and area is not None:
            session = {
                "session_id": session_id,
                "window": window,
                "screen": screen,
                "area": area,
                "space": space,
                "region": region,
                "scene": scene,
                "camera": camera,
                "workspace": workspace,
                "workspace_name": workspace.name,
                "workspace_switched": workspace_switched,
                "area_index": resolved_area_index,
                "preview_samples": preview_samples,
                "denoise": denoise,
                "device": str(scene.cycles.device),
                "keep_session": keep_session,
                "changes": changes,
            }
        if cleanup_errors and session is not None:
            _sessions[session_id] = session
        else:
            _sessions.pop(session_id, None)
        if area is not None:
            _tag_redraw(area)
        if cleanup_errors:
            raise RuntimeError(
                f"Cycles viewport preparation failed: {prepare_error}; cleanup also "
                f"failed: {'; '.join(cleanup_errors)}. Recovery session "
                f"'{session_id}' was retained for RESTORE."
            ) from prepare_error
        raise


def handle_capture_cycles_viewport_area(params: dict) -> dict:
    """Capture the currently visible pixels of a prepared Cycles viewport."""
    session_id = params.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("'session_id' is required")
    session = _sessions.get(session_id)
    if session is None:
        raise ValueError(f"Unknown Cycles viewport session '{session_id}'")

    keep_session = bool(params.get("keep_session", session["keep_session"]))
    metadata_only = bool(params.get("metadata_only", False))
    try:
        if session["window"].workspace != session["workspace"]:
            raise RuntimeError("Prepared workspace is no longer visible in its Blender window")
        if session["window"].screen != session["screen"]:
            raise RuntimeError("Prepared workspace screen changed before capture")
        if session["area"].type != "VIEW_3D":
            raise RuntimeError("Prepared area is no longer a VIEW_3D area")

        if metadata_only:
            window = session["window"]
            return {
                "session_id": session_id,
                "capture_backend": "MACOS_SCREENCAPTUREKIT",
                "capture_target": {
                    "process_id": os.getpid(),
                    "window_content_width": int(window.width),
                    "window_content_height": int(window.height),
                    "window_x": int(window.x),
                    "window_y": int(window.y),
                    "region_rect": _window_content_region_rect(session),
                },
                "mode": "cycles_rendered_viewport",
                "workspace_name": session["workspace_name"],
                "area_index": session["area_index"],
                "camera_name": session["camera"].name,
                "preview_samples": session["preview_samples"],
                "denoise": session["denoise"],
                "device": session["device"],
                # The external process must capture before restoration.
                "keep_session": True,
                "session_active": True,
                "warnings": [
                    "Using macOS ScreenCaptureKit because Blender screenshot "
                    "operators omit Metal-backed viewport pixels on this build."
                ],
            }

        with tempfile.TemporaryDirectory(prefix="blend_ai_cycles_viewport_") as temp_dir:
            filepath = os.path.join(temp_dir, "cycles_viewport.png")
            with _context_override(session):
                operator_result = bpy.ops.screen.screenshot_area(filepath=filepath)

            if operator_result is not None and "CANCELLED" in operator_result:
                raise RuntimeError("Blender cancelled the viewport screenshot")
            if not os.path.isfile(filepath):
                raise RuntimeError("Blender did not create the viewport screenshot")
            with open(filepath, "rb") as image_file:
                image_data = image_file.read()

        source_width, source_height = _png_dimensions(image_data)
        result = {
            "session_id": session_id,
            "image_base64": base64.b64encode(image_data).decode("ascii"),
            "format": "png",
            "mime_type": "image/png",
            "source_width": source_width,
            "source_height": source_height,
            "region_rect": _region_rect(session, source_width, source_height),
            "mode": "cycles_rendered_viewport",
            "workspace_name": session["workspace_name"],
            "area_index": session["area_index"],
            "camera_name": session["camera"].name,
            "preview_samples": session["preview_samples"],
            "denoise": session["denoise"],
            "device": session["device"],
            "keep_session": keep_session,
        }

        if not keep_session:
            _restore_session(session_id)
        result["session_active"] = keep_session
        return result
    except Exception as capture_error:
        if session_id in _sessions:
            try:
                _restore_session(session_id)
            except Exception as restore_error:
                raise RuntimeError(
                    f"Cycles viewport capture failed: {capture_error}; "
                    f"restoration also failed: {restore_error}"
                ) from capture_error
        raise


def handle_capture_cycles_quick_render(params: dict) -> dict:
    """Render a bounded, denoised Cycles camera preview to a temporary PNG.

    This is an explicit reliability alternative when native window capture is
    unavailable or unwanted. It never saves the blend file, restores every
    render setting it changes, and removes the temporary PNG before returning.
    """
    session_id = params.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("'session_id' is required")
    session = _sessions.get(session_id)
    if session is None:
        raise ValueError(f"Unknown Cycles viewport session '{session_id}'")
    max_size = _require_int(params, "max_size", 1024, 64, 4096)
    denoise = params.get("denoise", session["denoise"])
    if not isinstance(denoise, bool):
        raise ValueError("'denoise' must be a boolean")

    scene = session["scene"]
    changes: list[tuple[Any, str, Any]] = []
    result: dict[str, Any] | None = None
    spatial_cache.begin_render_evaluation()
    try:
        render = scene.render
        image_settings = render.image_settings
        _record_and_set(changes, render, "filepath", "")
        _record_and_set(changes, image_settings, "file_format", "PNG")
        _record_and_set_if_present(changes, image_settings, "color_mode", "RGB")
        _record_and_set_if_present(changes, image_settings, "color_depth", "8")
        _record_and_set_if_present(changes, image_settings, "compression", 15)
        _record_and_set_if_present(changes, render, "use_border", False)
        _record_and_set_if_present(changes, render, "use_crop_to_border", False)
        _record_and_set_if_present(changes, render, "use_compositing", False)
        _record_and_set_if_present(changes, render, "use_sequencer", False)
        _record_and_set(changes, scene.cycles, "samples", session["preview_samples"])
        _record_and_set_if_present(changes, scene.cycles, "use_denoising", denoise)

        resolution_x = max(int(render.resolution_x), 1)
        resolution_y = max(int(render.resolution_y), 1)
        original_percentage = max(int(render.resolution_percentage), 1)
        effective_x = max(1, round(resolution_x * original_percentage / 100))
        effective_y = max(1, round(resolution_y * original_percentage / 100))
        scale = min(1.0, max_size / max(effective_x, effective_y))
        target_x = max(1, min(max_size, round(effective_x * scale)))
        target_y = max(1, min(max_size, round(effective_y * scale)))
        _record_and_set(changes, render, "resolution_x", target_x)
        _record_and_set(changes, render, "resolution_y", target_y)
        _record_and_set(changes, render, "resolution_percentage", 100)

        with tempfile.TemporaryDirectory(prefix="blend_ai_cycles_render_") as temp_dir:
            filepath = os.path.join(temp_dir, "cycles_preview.png")
            render.filepath = filepath
            with _context_override(session):
                operator_result = bpy.ops.render.render(
                    write_still=True,
                    scene=scene.name,
                )
            if operator_result is not None and "CANCELLED" in operator_result:
                raise RuntimeError("Blender cancelled the quick Cycles preview render")
            if not os.path.isfile(filepath):
                raise RuntimeError("Blender did not write the quick Cycles preview PNG")
            if os.path.getsize(filepath) > 64 * 1024 * 1024:
                raise RuntimeError("Quick Cycles preview PNG exceeds 64 MiB")
            with open(filepath, "rb") as image_file:
                image_data = image_file.read()

        source_width, source_height = _png_dimensions(image_data)
        result = {
            "session_id": session_id,
            "image_base64": base64.b64encode(image_data).decode("ascii"),
            "format": "png",
            "mime_type": "image/png",
            "source_width": source_width,
            "source_height": source_height,
            "region_rect": None,
            "capture_backend": "QUICK_CYCLES_RENDER",
            "mode": "quick_cycles_render",
            "workspace_name": session["workspace_name"],
            "area_index": session["area_index"],
            "camera_name": session["camera"].name,
            "preview_samples": session["preview_samples"],
            "denoise": denoise,
            "device": session["device"],
            "keep_session": True,
            "session_active": True,
            "warnings": [
                "Explicit quick-render mode used a temporary low-sample denoised "
                "Cycles PNG. This updates Blender's Render Result."
            ],
        }
    except Exception as capture_error:
        cleanup_errors: list[str] = []
        try:
            cleanup_errors = _restore_changes(changes)
            if session_id in _sessions:
                try:
                    _restore_session(session_id)
                except Exception as restore_error:
                    cleanup_errors.append(f"viewport session: {restore_error}")
        finally:
            _flush_dependency_graph_updates()
            spatial_cache.end_render_evaluation()
        if cleanup_errors:
            raise RuntimeError(
                f"Quick Cycles preview failed: {capture_error}; cleanup also failed: "
                + "; ".join(cleanup_errors)
            ) from capture_error
        raise

    try:
        cleanup_errors = _restore_changes(changes)
    finally:
        _flush_dependency_graph_updates()
        spatial_cache.end_render_evaluation()
    if cleanup_errors:
        if session_id in _sessions:
            try:
                _restore_session(session_id)
            except Exception as restore_error:
                cleanup_errors.append(f"viewport session: {restore_error}")
        raise RuntimeError(
            "Quick Cycles preview completed but cleanup failed: "
            + "; ".join(cleanup_errors)
        )
    assert result is not None
    return result


def handle_restore_cycles_viewport(params: dict) -> dict:
    """Restore one prepared viewport session, or all active sessions."""
    session_id = params.get("session_id")
    if session_id is not None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("'session_id' must be a non-empty string when provided")
        if session_id not in _sessions:
            return {
                "restored": False,
                "session_id": session_id,
                "restored_session_ids": [],
            }
        _restore_session(session_id)
        return {
            "restored": True,
            "session_id": session_id,
            "restored_session_ids": [session_id],
        }

    restored_ids = list(reversed(list(_sessions)))
    for active_session_id in restored_ids:
        _restore_session(active_session_id)
    return {
        "restored": bool(restored_ids),
        "session_id": None,
        "restored_session_ids": restored_ids,
    }


def register() -> None:
    """Register the visible Cycles viewport command handlers."""
    dispatcher.register_handler("prepare_cycles_viewport", handle_prepare_cycles_viewport)
    dispatcher.register_handler(
        "capture_cycles_viewport_area", handle_capture_cycles_viewport_area
    )
    dispatcher.register_handler(
        "capture_cycles_quick_render", handle_capture_cycles_quick_render
    )
    dispatcher.register_handler("restore_cycles_viewport", handle_restore_cycles_viewport)
    load_pre = _load_pre_handlers()
    if load_pre is not None and _restore_sessions_before_load not in load_pre:
        load_pre.append(_restore_sessions_before_load)
    load_post = _load_post_handlers()
    if load_post is not None and _clear_sessions_on_load not in load_post:
        load_post.append(_clear_sessions_on_load)


def unregister() -> None:
    """Best-effort restoration when the add-on is disabled or reloaded."""
    load_pre = _load_pre_handlers()
    if load_pre is not None:
        while _restore_sessions_before_load in load_pre:
            load_pre.remove(_restore_sessions_before_load)
    load_post = _load_post_handlers()
    if load_post is not None:
        while _clear_sessions_on_load in load_post:
            load_post.remove(_clear_sessions_on_load)
    for session_id in list(reversed(list(_sessions))):
        try:
            _restore_session(session_id)
        except Exception:
            # Blender may already have destroyed windows/areas during shutdown.
            _sessions.pop(session_id, None)
