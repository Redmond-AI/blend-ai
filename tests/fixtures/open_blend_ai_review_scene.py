"""Verify an explicitly opened review scene and start its blend-ai server."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys

import bpy


ADDON_MODULE = "bl_ext.user_default.blend_ai"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-file", required=True, type=Path)
    parser.add_argument("--ready-file", required=True, type=Path)
    parser.add_argument("--port", type=int, default=9876)
    trailing = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(trailing)


def _write(path: Path, payload: dict[str, object]) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, resolved)


def main() -> None:
    args = _arguments()
    expected = args.expected_file.expanduser().resolve()
    actual = Path(bpy.data.filepath).resolve()
    if actual != expected:
        raise RuntimeError(f"Opened file {actual} does not match expected {expected}")
    if ADDON_MODULE not in bpy.context.preferences.addons:
        result = bpy.ops.preferences.addon_enable(module=ADDON_MODULE)
        if "FINISHED" not in result:
            raise RuntimeError(f"Could not enable blend-ai: {result}")
    addon = importlib.import_module(ADDON_MODULE)
    server_module = importlib.import_module(f"{addon.__name__}.server")
    server_module.start_server(host="127.0.0.1", port=args.port)
    if not server_module.get_server().is_running:
        raise RuntimeError("blend-ai server did not start")

    startup_attempts = 0

    def finish_visible_startup() -> float | None:
        nonlocal startup_attempts
        startup_attempts += 1
        # Blender can apply the file's saved workspace after a --python script
        # begins.  Defer the readiness receipt until the first UI timer tick so
        # READY describes the visible final workspace rather than that transient
        # startup state.
        window = bpy.context.window
        if window is None:
            raise RuntimeError("Visible Blender window is unavailable")
        workspace = bpy.data.workspaces.get("AI Preview")
        if workspace is None:
            workspace = window.workspace.copy()
            workspace.name = "AI Preview"
        window.workspace = workspace
        if window.workspace != workspace:
            # Command-line startup can briefly reject workspace changes while
            # the loaded file's screen is being attached to the new window.
            # Keep the timer alive until that transition settles.
            if startup_attempts < 20:
                return 0.25
            raise RuntimeError(
                "Blender did not activate the AI Preview workspace after "
                f"{startup_attempts} attempts"
            )
        view_areas = [area for area in window.screen.areas if area.type == "VIEW_3D"]
        if not view_areas:
            area = max(window.screen.areas, key=lambda item: item.width * item.height)
            area.type = "VIEW_3D"
            view_areas = [area]
        for area in view_areas[1:]:
            area.type = "PROPERTIES"
        view_areas[0].spaces.active.region_3d.view_perspective = "CAMERA"
        payload = {
            "status": "READY",
            "pid": os.getpid(),
            "filepath": str(actual),
            "scene": bpy.context.scene.name,
            "camera": bpy.context.scene.camera.name if bpy.context.scene.camera else None,
            "blender_version": list(bpy.app.version),
            "server_host": "127.0.0.1",
            "server_port": args.port,
            "workspace": window.workspace.name,
            "view3d_count": len(
                [area for area in window.screen.areas if area.type == "VIEW_3D"]
            ),
            "saved": False,
        }
        _write(args.ready_file, payload)
        print("BLEND_AI_REVIEW_READY " + json.dumps(payload, sort_keys=True), flush=True)

    # Startup may finish applying the .blend after this script returns.  A
    # persistent timer survives that load boundary and runs against the final
    # visible window state.
    bpy.app.timers.register(
        finish_visible_startup,
        first_interval=0.5,
        persistent=True,
    )


if __name__ == "__main__":
    main()
