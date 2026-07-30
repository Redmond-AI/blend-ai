"""Start the miniature warehouse in an isolated, visible Blender process.

Launch this script in a *new* Blender process, for example::

    open -na Blender --args --factory-startup \
      --python tests/fixtures/live_blender_bootstrap.py -- \
      --confirm-isolated-process --ready-file /tmp/blend-ai-live-ready.json

The script refuses background mode, a loaded ``.blend`` file, or a launch that
does not include both ``--factory-startup`` and ``--confirm-isolated-process``.
It never saves.  The default extension module assumes Blender's 4.2+ user
extension namespace (``bl_ext.user_default.blend_ai``); override it with
``--addon-module`` if the local repository name differs.

Maximizing a 3D View is a best-effort UI operation because Blender can reject
screen operators while a window is still settling.  The normal factory Layout
already has exactly one large VIEW_3D, so capture remains usable if maximizing
is unavailable.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any

import bpy


DEFAULT_ADDON_MODULE = "bl_ext.user_default.blend_ai"
DEFAULT_PORT = 9876
WORKSPACE_NAME = "AI Preview"


def _script_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-isolated-process",
        action="store_true",
        help="Required safety acknowledgement for a new --factory-startup process.",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--workspace-name", default=WORKSPACE_NAME)
    parser.add_argument("--addon-module", default=DEFAULT_ADDON_MODULE)
    parser.add_argument(
        "--fixture",
        choices=("warehouse", "scale"),
        default="warehouse",
        help="Build the miniature warehouse or the 20,000-instance benchmark scene.",
    )
    parser.add_argument(
        "--ready-file",
        type=Path,
        help="Optional JSON status file written after the localhost server starts.",
    )
    parser.add_argument(
        "--no-maximize",
        action="store_true",
        help="Keep the factory Layout instead of maximizing its single 3D View.",
    )
    trailing = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(trailing)


def _assert_isolated_launch(args: argparse.Namespace) -> None:
    if bpy.app.background:
        raise RuntimeError("Live Cycles acceptance requires a visible Blender process")
    if not args.confirm_isolated_process:
        raise RuntimeError("Refusing to reset a scene without --confirm-isolated-process")
    if "--factory-startup" not in sys.argv:
        raise RuntimeError("Refusing to run unless Blender was launched with --factory-startup")
    if bpy.data.filepath:
        raise RuntimeError(
            f"Refusing to alter loaded file {bpy.data.filepath!r}; use a new factory process"
        )
    if not isinstance(args.port, int) or isinstance(args.port, bool):
        raise RuntimeError("Port must be an integer")
    if not 1024 <= args.port <= 65535:
        raise RuntimeError("Port must be between 1024 and 65535")
    if not args.workspace_name.strip():
        raise RuntimeError("Workspace name must not be empty")


def _assert_port_available(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.25)
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(
                f"127.0.0.1:{port} is already in use; stop the other add-on server first"
            )


def _load_fixture_builder(fixture: str):
    filename = (
        "build_scale_benchmark.py"
        if fixture == "scale"
        else "build_mini_warehouse.py"
    )
    builder_path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location("blend_ai_live_fixture_builder", builder_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load fixture builder at {builder_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_scene


def _enable_extension(module_name: str) -> Any:
    if module_name not in bpy.context.preferences.addons:
        result = bpy.ops.preferences.addon_enable(module=module_name)
        if "FINISHED" not in result:
            raise RuntimeError(f"Blender did not enable extension module {module_name!r}: {result}")
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Enabled extension {module_name!r} could not be imported. "
            "The developer symlink is expected to be named 'blend_ai' in "
            "Blender's user_default extension repository."
        ) from exc


def _preview_workspace(name: str, *, maximize: bool) -> dict[str, Any]:
    window = bpy.context.window
    if window is None:
        raise RuntimeError("Blender has no visible main window")

    workspace = bpy.data.workspaces.get(name)
    if workspace is None:
        workspace = window.workspace
        workspace.name = name
    window.workspace = workspace

    screen = window.screen
    view_areas = [area for area in screen.areas if area.type == "VIEW_3D"]
    if not view_areas:
        area = max(screen.areas, key=lambda item: int(item.width) * int(item.height))
        area.type = "VIEW_3D"
        view_areas = [area]
    if len(view_areas) > 1:
        keep = max(view_areas, key=lambda item: int(item.width) * int(item.height))
        for area in view_areas:
            if area != keep:
                area.type = "PROPERTIES"
        view_areas = [keep]

    area = view_areas[0]
    maximized = False
    maximize_warning = None
    if maximize:
        try:
            with bpy.context.temp_override(window=window, screen=screen, area=area):
                try:
                    result = bpy.ops.screen.screen_full_area(use_hide_panels=True)
                except TypeError:
                    result = bpy.ops.screen.screen_full_area()
            maximized = "FINISHED" in result
            if not maximized:
                maximize_warning = f"screen_full_area returned {result}"
        except (RuntimeError, TypeError, ValueError) as exc:
            maximize_warning = f"screen_full_area was unavailable: {exc}"

    screen = window.screen
    view_areas = [item for item in screen.areas if item.type == "VIEW_3D"]
    if len(view_areas) != 1:
        raise RuntimeError(
            f"Preview workspace has {len(view_areas)} VIEW_3D areas after setup; expected one"
        )
    area = view_areas[0]
    space = area.spaces.active
    if space.type != "VIEW_3D":
        raise RuntimeError("Preview area's active space is not VIEW_3D")
    space.region_3d.view_perspective = "CAMERA"
    if hasattr(space, "lock_camera"):
        space.lock_camera = False
    area.tag_redraw()

    return {
        "workspace_name": workspace.name,
        "view3d_count": len(view_areas),
        "area_width": int(area.width),
        "area_height": int(area.height),
        "maximized": maximized,
        "warning": maximize_warning,
    }


def _write_ready(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, resolved)


def main() -> None:
    args = _script_arguments()
    _assert_isolated_launch(args)
    _assert_port_available(args.port)

    addon = _enable_extension(args.addon_module)
    scene = _load_fixture_builder(args.fixture)()
    if bpy.data.filepath:
        raise RuntimeError("Fixture builder unexpectedly assigned a .blend filepath")
    workspace = _preview_workspace(
        args.workspace_name.strip(),
        maximize=not args.no_maximize,
    )

    addon_server = importlib.import_module(f"{args.addon_module}.server")
    addon_server.start_server(host="127.0.0.1", port=args.port)
    server = addon_server.get_server()
    if not server.is_running:
        raise RuntimeError("The blend-ai add-on server did not start")

    payload = {
        "ready": True,
        "pid": os.getpid(),
        "started_at_unix": time.time(),
        "blender_version": list(bpy.app.version),
        "background": bool(bpy.app.background),
        "filepath": bpy.data.filepath,
        "scene_name": scene.name,
        "fixture_marker": bool(scene.get("blend_ai_fixture", False)),
        "scale_fixture_marker": bool(scene.get("blend_ai_scale_fixture", False)),
        "fixture": args.fixture,
        "camera_name": scene.camera.name if scene.camera else None,
        "addon_module": addon.__name__,
        "server_host": "127.0.0.1",
        "server_port": args.port,
        **workspace,
    }
    _write_ready(args.ready_file, payload)
    print("BLEND_AI_LIVE_READY " + json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
