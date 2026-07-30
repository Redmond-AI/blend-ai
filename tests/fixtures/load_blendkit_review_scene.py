"""Load one exact BlenderKit scene with user prefs, save once, and start blend-ai."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback
from typing import Any

import bpy


BLEND_AI_MODULE = "bl_ext.user_default.blend_ai"
BLENDERKIT_MODULE = "bl_ext.user_default.blenderkit"
DEFAULT_PORT = 9876


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-isolated-process", action="store_true")
    parser.add_argument("--asset-base-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ready-file", required=True, type=Path)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=float, default=900.0)
    trailing = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(trailing)


def _write(path: Path, payload: dict[str, Any]) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, resolved)


def _enable(module_name: str) -> Any:
    if module_name not in bpy.context.preferences.addons:
        result = bpy.ops.preferences.addon_enable(module=module_name)
        if "FINISHED" not in result:
            raise RuntimeError(f"Could not enable {module_name}: {result}")
    return importlib.import_module(module_name)


def _asset_scene(asset_base_id: str) -> Any | None:
    for scene in bpy.data.scenes:
        props = getattr(scene, "blenderkit", None)
        if str(getattr(props, "asset_base_id", "")) == asset_base_id:
            return scene
    return None


def _assert_launch(args: argparse.Namespace) -> tuple[Path, Path]:
    if bpy.app.background or not args.confirm_isolated_process:
        raise RuntimeError("BlendKit acceptance requires an authorized visible process")
    if bpy.data.filepath:
        raise RuntimeError("BlendKit acceptance requires a fresh process with no loaded file")
    output = args.output.expanduser().resolve()
    ready = args.ready_file.expanduser().resolve()
    if output.suffix.lower() != ".blend":
        raise RuntimeError("Output must be an explicit .blend path")
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite existing output {output}")
    if not 1024 <= args.port <= 65535:
        raise RuntimeError("Port must be between 1024 and 65535")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if probe.connect_ex(("127.0.0.1", args.port)) == 0:
            raise RuntimeError(f"127.0.0.1:{args.port} is already in use")
    return output, ready


def main() -> None:
    args = _arguments()
    output, ready = _assert_launch(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        _enable(BLENDERKIT_MODULE)
        blend_ai = _enable(BLEND_AI_MODULE)
        scene_settings = bpy.context.window_manager.blenderkit_scene
        scene_settings.append_link = "APPEND"
        scene_settings.switch_after_append = True

        def monitor() -> float | None:
            try:
                scene = _asset_scene(args.asset_base_id)
                if scene is None:
                    if time.monotonic() - started > args.timeout:
                        raise TimeoutError(
                            f"BlendKit scene {args.asset_base_id} did not arrive in time"
                        )
                    return 0.25
                bpy.context.window.scene = scene
                save_result = bpy.ops.wm.save_as_mainfile(filepath=str(output))
                if "FINISHED" not in save_result or not output.is_file():
                    raise RuntimeError(f"Blender did not save {output}: {save_result}")
                addon_server = importlib.import_module(f"{blend_ai.__name__}.server")
                addon_server.start_server(host="127.0.0.1", port=args.port)
                if not addon_server.get_server().is_running:
                    raise RuntimeError("blend-ai server did not start")
                payload = {
                    "status": "READY",
                    "asset_base_id": args.asset_base_id,
                    "asset_type": "scene",
                    "scene_name": scene.name,
                    "camera": scene.camera.name if scene.camera else None,
                    "blend_path": str(output),
                    "blend_bytes": output.stat().st_size,
                    "blender_version": list(bpy.app.version),
                    "pid": os.getpid(),
                    "server_port": args.port,
                    "saved_once": True,
                }
                _write(ready, payload)
                print("BLENDKIT_REVIEW_READY " + json.dumps(payload, sort_keys=True), flush=True)
                return None
            except Exception as exc:
                payload = {
                    "status": "ERROR",
                    "asset_base_id": args.asset_base_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                _write(ready, payload)
                print("BLENDKIT_REVIEW_ERROR " + json.dumps(payload, sort_keys=True), flush=True)
                return None

        def begin_download() -> float | None:
            try:
                client_lib = importlib.import_module(f"{BLENDERKIT_MODULE}.client_lib")
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                    probe.settimeout(0.05)
                    accessible = probe.connect_ex(
                        ("127.0.0.1", int(client_lib.get_port()))
                    ) == 0
                if not accessible:
                    if time.monotonic() - started > args.timeout:
                        raise TimeoutError("BlendKit client did not become accessible")
                    return 0.25
                result = bpy.ops.scene.blenderkit_download(
                    "EXEC_DEFAULT",
                    asset_base_id=args.asset_base_id,
                    invoke_resolution=False,
                    invoke_scene_settings=False,
                )
                if "FINISHED" not in result:
                    raise RuntimeError(f"BlenderKit download operator returned {result}")
                bpy.app.timers.register(monitor, first_interval=0.25, persistent=False)
                return None
            except Exception as exc:
                payload = {
                    "status": "ERROR",
                    "asset_base_id": args.asset_base_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                _write(ready, payload)
                print("BLENDKIT_REVIEW_ERROR " + json.dumps(payload, sort_keys=True), flush=True)
                return None

        bpy.app.timers.register(begin_download, first_interval=0.25, persistent=False)
    except Exception as exc:
        _write(
            ready,
            {
                "status": "ERROR",
                "asset_base_id": args.asset_base_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
