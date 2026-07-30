"""Run the visible-Blender, native-image relighting acceptance sequence.

This script connects only through a freshly launched blend-ai stdio server.  It
refuses to mutate unless the Blender TCP endpoint reports the exact miniature
warehouse scene created by ``fixtures/live_blender_bootstrap.py``.  It never
calls a save tool and always attempts preview restoration and reverse-order
transaction rollback before exiting.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from datetime import timedelta
from io import BytesIO
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, ImageContent, TextContent
from PIL import Image, ImageChops, ImageStat


EXPECTED_SCENE = "Mini Warehouse Relighting Fixture"
EXPECTED_CAMERA = "Warehouse_Camera"
REQUIRED_TOOLS = {
    "get_scene_info",
    "list_lights",
    "get_viewport_screenshot",
    "get_lighting_context",
    "batch_raycast",
    "apply_light_plan",
    "capture_cycles_viewport",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="Explicit directory for three images and acceptance-report.json.",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="blend-ai checkout to launch through uv.",
    )
    parser.add_argument(
        "--uv",
        type=Path,
        default=None,
        help="uv executable (defaults to UV_BIN, PATH, then ~/.local/bin/uv).",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    parser.add_argument("--preview-samples", type=int, default=16)
    parser.add_argument(
        "--capture-mode",
        choices=("VIEWPORT", "QUICK_RENDER"),
        default="VIEWPORT",
        help=(
            "Use native visible-viewport capture or the explicitly authorized "
            "low-sample denoised Cycles render path."
        ),
    )
    return parser.parse_args()


def _resolve_uv(value: Path | None) -> Path:
    candidates = [
        value,
        Path(os.environ["UV_BIN"]) if os.environ.get("UV_BIN") else None,
        Path(found) if (found := shutil.which("uv")) else None,
        Path.home() / ".local/bin/uv",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.expanduser().is_file():
            return candidate.expanduser().resolve()
    raise RuntimeError("uv was not found; pass --uv /absolute/path/to/uv")


def _artifact_directory(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    unsafe = {Path("/"), Path.home().resolve()}
    if resolved in unsafe:
        raise RuntimeError(f"Refusing unsafe artifact directory {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _json_value(result: CallToolResult) -> Any:
    if result.structuredContent is not None:
        value: Any = result.structuredContent
        if isinstance(value, dict) and set(value) == {"result"}:
            value = value["result"]
        return value
    for content in result.content:
        if isinstance(content, TextContent):
            try:
                return json.loads(content.text)
            except json.JSONDecodeError:
                continue
    raise RuntimeError("Tool returned neither structured JSON nor JSON TextContent")


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False, default=str))


async def _call(
    session: ClientSession,
    report: dict[str, Any],
    name: str,
    arguments: dict[str, Any] | None = None,
) -> CallToolResult:
    started = time.perf_counter()
    result = await session.call_tool(name, arguments or {})
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    report["calls"].append(
        {"tool": name, "elapsed_ms": elapsed_ms, "is_error": bool(result.isError)}
    )
    if result.isError:
        messages = [item.text for item in result.content if isinstance(item, TextContent)]
        raise RuntimeError(f"{name} failed: {'; '.join(messages) or 'unknown MCP error'}")
    return result


def _assert_fixture(scene_info: Any) -> None:
    if not isinstance(scene_info, dict):
        raise RuntimeError("get_scene_info returned an invalid object")
    if scene_info.get("scene_name") != EXPECTED_SCENE:
        raise RuntimeError(
            "Safety stop: Blender is not showing the isolated miniature warehouse "
            f"({scene_info.get('scene_name')!r} != {EXPECTED_SCENE!r})"
        )
    objects = scene_info.get("objects")
    names = {
        item.get("name")
        for item in objects
        if isinstance(objects, list) and isinstance(item, dict)
    }
    required = {"Warehouse_Floor", "Warehouse_Camera", "Skylight_Glass_Clear"}
    missing = sorted(required - names)
    if missing:
        raise RuntimeError(f"Safety stop: miniature warehouse is missing {missing}")


def _capture_image(
    result: CallToolResult,
    artifact_dir: Path,
    stem: str,
) -> tuple[dict[str, Any], Path, Image.Image]:
    images = [item for item in result.content if isinstance(item, ImageContent)]
    if len(images) != 1:
        raise RuntimeError(
            f"{stem} capture returned {len(images)} native ImageContent blocks; expected one"
        )
    image_content = images[0]
    try:
        data = base64.b64decode(image_content.data, validate=True)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(f"{stem} ImageContent contains invalid base64") from exc
    suffix_by_mime = {"image/jpeg": ".jpg", "image/png": ".png"}
    suffix = suffix_by_mime.get(image_content.mimeType)
    if suffix is None:
        raise RuntimeError(f"{stem} returned unsupported MIME type {image_content.mimeType!r}")
    if suffix == ".jpg" and not data.startswith(b"\xff\xd8\xff"):
        raise RuntimeError(f"{stem} claimed JPEG but its signature is invalid")
    if suffix == ".png" and not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError(f"{stem} claimed PNG but its signature is invalid")

    path = artifact_dir / f"{stem}{suffix}"
    path.write_bytes(data)
    with Image.open(BytesIO(data)) as loaded:
        loaded.load()
        image = loaded.convert("RGB")
    metadata = _json_value(result)
    if not isinstance(metadata, dict):
        raise RuntimeError(f"{stem} capture metadata is not an object")
    if metadata.get("byte_count") != len(data):
        raise RuntimeError(f"{stem} native image byte count does not match its metadata")
    return _json_safe(metadata), path, image


def _image_metrics(image: Image.Image) -> dict[str, Any]:
    stats = ImageStat.Stat(image)
    red, green, blue = stats.mean[:3]
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    # The fixture camera places the clear skylight contribution across the
    # upper-center/left wall.  A region metric verifies that the cool source is
    # visible rather than merely present in the light plan.
    cool_region = image.crop(
        (
            image.width // 5,
            0,
            image.width * 3 // 5,
            image.height // 2,
        )
    )
    cool_red, cool_green, cool_blue = ImageStat.Stat(cool_region).mean[:3]
    return {
        "width": image.width,
        "height": image.height,
        "mean_rgb": [red, green, blue],
        "mean_luminance": luminance,
        "red_excess": red - ((green + blue) * 0.5),
        "cool_skylight_region_mean_rgb": [cool_red, cool_green, cool_blue],
        "cool_skylight_blue_excess": cool_blue - ((cool_red + cool_green) * 0.5),
    }


def _mean_absolute_difference(left: Image.Image, right: Image.Image) -> float:
    if left.size != right.size:
        right = right.resize(left.size, Image.Resampling.LANCZOS)
    difference = ImageChops.difference(left, right)
    return sum(ImageStat.Stat(difference).mean[:3]) / 3.0


def _night_plan(geometry_revision: int) -> dict[str, Any]:
    return {
        "action": "APPLY",
        "plan_id": "mini_warehouse_night",
        "mode": "REPLACE_MANAGED",
        "expected_geometry_revision": geometry_revision,
        "collection_name": "AI_RELIGHT",
        "strict": True,
        "lights": [
            {
                "id": "moon_clear_skylight",
                "name": "AI Moon Clear Skylight",
                "type": "AREA",
                "location_world": [-10.0, 0.0, 9.6],
                "target_point": [-10.0, 0.0, 0.2],
                "energy": 1500.0,
                "color_rgb": [0.20, 0.34, 0.72],
                "area_shape": "RECTANGLE",
                "size": 4.0,
                "size_y": 3.0,
                "use_shadow": True,
            },
            {
                "id": "red_west",
                "name": "AI Red West",
                "type": "POINT",
                "location_world": [-3.5, 0.0, 2.8],
                "energy": 450.0,
                "color_rgb": [1.0, 0.012, 0.006],
                "radius": 0.12,
                "use_shadow": True,
            },
            {
                "id": "red_east",
                "name": "AI Red East",
                "type": "POINT",
                "location_world": [6.0, 2.0, 2.8],
                "energy": 375.0,
                "color_rgb": [1.0, 0.008, 0.004],
                "radius": 0.12,
                "use_shadow": True,
            },
        ],
        "scene_overrides": {
            "existing_light_policy": "MUTE_NON_MANAGED",
            "world": {
                "mode": "MANAGED_SOLID",
                "color_rgb": [0.002, 0.005, 0.018],
                "strength": 0.025,
            },
            "exposure": -0.7,
        },
    }


def _patch_plan(geometry_revision: int) -> dict[str, Any]:
    return {
        "action": "APPLY",
        "plan_id": "mini_warehouse_night",
        "mode": "PATCH_MANAGED",
        "expected_geometry_revision": geometry_revision,
        "collection_name": "AI_RELIGHT",
        "strict": True,
        "lights": [
            {
                "id": "red_west",
                "name": "AI Red West",
                "type": "POINT",
                "location_world": [-3.5, 0.0, 2.8],
                "energy": 900.0,
                "color_rgb": [1.0, 0.01, 0.004],
                "radius": 0.12,
                "use_shadow": True,
            }
        ],
    }


def _capture_arguments(args: argparse.Namespace, session_id: str | None = None) -> dict[str, Any]:
    values: dict[str, Any] = {
        "action": "CAPTURE",
        "capture_mode": args.capture_mode,
        "workspace_name": "AI Preview",
        "camera_name": EXPECTED_CAMERA,
        "preview_samples": args.preview_samples,
        "settle_seconds": args.settle_seconds,
        "denoise": True,
        "device": "GPU",
        "max_size": 1024,
        "format": "PNG" if args.capture_mode == "QUICK_RENDER" else "JPEG",
        "jpeg_quality": 85,
        "keep_session": True,
    }
    if session_id is not None:
        values["session_id"] = session_id
    return values


async def _run(args: argparse.Namespace, artifact_dir: Path, report: dict[str, Any]) -> None:
    repo = args.repo.expanduser().resolve()
    uv = _resolve_uv(args.uv)
    server = StdioServerParameters(
        command=str(uv),
        args=["run", "--directory", str(repo), "blend-ai"],
        cwd=repo,
    )
    timeout = timedelta(seconds=args.timeout)

    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(
            read_stream,
            write_stream,
            read_timeout_seconds=timeout,
        ) as session:
            await session.initialize()
            listed = await session.list_tools()
            names = sorted(tool.name for tool in listed.tools)
            report["tools"] = names
            missing = sorted(REQUIRED_TOOLS - set(names))
            if missing:
                raise RuntimeError(f"MCP tool listing is missing {missing}")

            preview_session_id: str | None = None
            transactions: list[str] = []
            original_lights: Any = None
            primary_error: BaseException | None = None
            try:
                scene_info = _json_value(await _call(session, report, "get_scene_info"))
                _assert_fixture(scene_info)
                report["scene_info"] = _json_safe(scene_info)

                original_lights = _json_value(await _call(session, report, "list_lights"))
                report["original_lights"] = _json_safe(original_lights)

                baseline_result = await _call(
                    session,
                    report,
                    "capture_cycles_viewport",
                    _capture_arguments(args),
                )
                baseline_meta, baseline_path, baseline_image = _capture_image(
                    baseline_result, artifact_dir, "baseline"
                )
                preview_session_id = baseline_meta.get("session_id")
                if not isinstance(preview_session_id, str) or not preview_session_id:
                    raise RuntimeError("Baseline capture did not return a reusable session_id")

                context = _json_value(
                    await _call(
                        session,
                        report,
                        "get_lighting_context",
                        {
                            "camera_name": EXPECTED_CAMERA,
                            "scope": "CAMERA",
                            "detail": "CANDIDATES",
                            "max_instances": 500,
                            "max_surface_triangles": 200_000,
                            "cache_mode": "REFRESH",
                        },
                    )
                )
                if not isinstance(context, dict):
                    raise RuntimeError("Lighting context is not an object")
                geometry_revision = context.get("geometry_revision")
                if not isinstance(geometry_revision, int) or isinstance(geometry_revision, bool):
                    raise RuntimeError("Lighting context lacks an integer geometry_revision")
                if not context.get("instances"):
                    raise RuntimeError("Lighting context returned no camera-scoped instances")
                instance_ids_by_source: dict[str, set[str]] = {}
                for instance in context.get("instances", []):
                    source_name = instance.get("source_name")
                    revision_scoped_id = instance.get("revision_scoped_id")
                    if isinstance(source_name, str) and isinstance(revision_scoped_id, str):
                        instance_ids_by_source.setdefault(source_name, set()).add(
                            revision_scoped_id
                        )
                collection_names = {
                    item.get("name") for item in context.get("collections", [])
                }
                if "Warehouse_Skylights" not in collection_names:
                    raise RuntimeError("Lighting context lost the skylight collection hierarchy")
                openings = context.get("candidates", {}).get("openings", [])
                clear_openings = [
                    item
                    for item in openings
                    if "Skylight_Glass_Clear" in str(item.get("object_id", ""))
                ]
                if not clear_openings or not all(
                    item.get("heuristic") is True
                    and isinstance(item.get("confidence"), (int, float))
                    and bool(item.get("reasons"))
                    for item in clear_openings
                ):
                    raise RuntimeError(
                        "Clear skylight candidates lack explicit heuristic evidence"
                    )
                report["lighting_context"] = _json_safe(context)

                raycast = _json_value(
                    await _call(
                        session,
                        report,
                        "batch_raycast",
                        {
                            "rays": [
                                {
                                    "id": "clear_skylight",
                                    "origin": [-10.0, 0.0, 12.0],
                                    "target": [-10.0, 0.0, 0.1],
                                },
                                {
                                    "id": "blocked_skylight",
                                    "origin": [10.0, 0.0, 12.0],
                                    "target": [10.0, 0.0, 0.1],
                                },
                                {
                                    "id": "red_west_path",
                                    "origin": [-3.5, 0.0, 2.8],
                                    "target": [-3.5, 0.0, 0.1],
                                },
                                {
                                    "id": "red_east_path",
                                    "origin": [6.0, 2.0, 2.8],
                                    "target": [6.0, 2.0, 2.1],
                                },
                                {
                                    "id": "collection_instance_identity",
                                    "origin": [-2.0, 4.0, 5.0],
                                    "target": [-2.0, 4.0, 0.1],
                                },
                                {
                                    "id": "geometry_nodes_identity",
                                    "origin": [11.0, -6.0, 5.0],
                                    "target": [11.0, -6.0, 0.1],
                                },
                            ],
                            "ignore_object_patterns": ["Skylight_Glass_*"],
                            "ignore_material_patterns": ["*glass*"],
                            "include_ignored_hits": True,
                            "expected_geometry_revision": geometry_revision,
                            "max_hits": 8,
                            "time_budget_ms": 2000,
                        },
                    )
                )
                if not isinstance(raycast, dict) or not raycast.get("complete"):
                    raise RuntimeError("Raycast did not complete")
                rays = {item.get("id"): item for item in raycast.get("rays", [])}
                clear_ray = rays.get("clear_skylight", {})
                blocked_ray = rays.get("blocked_skylight", {})
                clear_blockers = [hit for hit in clear_ray.get("hits", []) if not hit.get("ignored")]
                blocked_blockers = [
                    hit for hit in blocked_ray.get("hits", []) if not hit.get("ignored")
                ]
                if clear_blockers:
                    raise RuntimeError(f"Clear skylight ray found blockers: {clear_blockers}")
                if not blocked_blockers:
                    raise RuntimeError("Blocked skylight ray did not find its opaque cover")
                for ray_id in ("red_west_path", "red_east_path"):
                    if not rays.get(ray_id, {}).get("clear_to_target"):
                        raise RuntimeError(f"Practical-light path {ray_id!r} is blocked")
                rack_hits = rays.get("collection_instance_identity", {}).get("hits", [])
                if not rack_hits or rack_hits[0].get("object_name") != "Instanced_Rack_Source":
                    raise RuntimeError("Collection-instance ray lost evaluated object identity")
                if rack_hits[0].get("material_name") != "Painted_Steel":
                    raise RuntimeError("Collection-instance ray lost evaluated material identity")
                if rack_hits[0].get("object_id") not in instance_ids_by_source.get(
                    "Instanced_Rack_Source", set()
                ):
                    raise RuntimeError(
                        "Collection-instance ray ID does not match lighting context"
                    )
                geometry_nodes_hits = rays.get("geometry_nodes_identity", {}).get("hits", [])
                if (
                    not geometry_nodes_hits
                    or geometry_nodes_hits[0].get("object_name")
                    != "GN_Instanced_Fixtures"
                ):
                    raise RuntimeError("Geometry Nodes ray lost evaluated object identity")
                if geometry_nodes_hits[0].get("object_id") not in instance_ids_by_source.get(
                    "GN_Instanced_Fixtures", set()
                ):
                    raise RuntimeError(
                        "Geometry Nodes ray ID does not match lighting context"
                    )
                report["raycast"] = _json_safe(raycast)

                plan = _night_plan(geometry_revision)
                validate_args = dict(plan)
                validate_args["action"] = "VALIDATE"
                validation = _json_value(
                    await _call(session, report, "apply_light_plan", validate_args)
                )
                if not isinstance(validation, dict) or not validation.get("valid"):
                    raise RuntimeError("Managed night plan did not validate")
                report["validation"] = _json_safe(validation)

                applied = _json_value(await _call(session, report, "apply_light_plan", plan))
                if not isinstance(applied, dict) or not applied.get("applied"):
                    raise RuntimeError("Managed night plan was not applied")
                transaction_id = applied.get("transaction_id")
                if not isinstance(transaction_id, str) or not transaction_id:
                    raise RuntimeError("Initial apply did not return a transaction_id")
                transactions.append(transaction_id)
                report["apply"] = _json_safe(applied)

                relit_result = await _call(
                    session,
                    report,
                    "capture_cycles_viewport",
                    _capture_arguments(args, preview_session_id),
                )
                relit_meta, relit_path, relit_image = _capture_image(
                    relit_result, artifact_dir, "relit"
                )

                patched = _json_value(
                    await _call(
                        session,
                        report,
                        "apply_light_plan",
                        _patch_plan(geometry_revision),
                    )
                )
                if not isinstance(patched, dict) or not patched.get("applied"):
                    raise RuntimeError("Managed patch was not applied")
                patch_transaction = patched.get("transaction_id")
                if not isinstance(patch_transaction, str) or not patch_transaction:
                    raise RuntimeError("Patch apply did not return a transaction_id")
                transactions.append(patch_transaction)
                report["patch"] = _json_safe(patched)

                patched_result = await _call(
                    session,
                    report,
                    "capture_cycles_viewport",
                    _capture_arguments(args, preview_session_id),
                )
                patched_meta, patched_path, patched_image = _capture_image(
                    patched_result, artifact_dir, "patched"
                )

                baseline_metrics = _image_metrics(baseline_image)
                relit_metrics = _image_metrics(relit_image)
                patched_metrics = _image_metrics(patched_image)
                baseline_to_relit = _mean_absolute_difference(baseline_image, relit_image)
                relit_to_patched = _mean_absolute_difference(relit_image, patched_image)
                checks = {
                    "relit_materially_changed": baseline_to_relit > 1.0,
                    "patch_changed_image": relit_to_patched > 0.1,
                    "relit_is_darker": (
                        relit_metrics["mean_luminance"] < baseline_metrics["mean_luminance"]
                    ),
                    "red_separation_increased": (
                        relit_metrics["red_excess"] > baseline_metrics["red_excess"]
                    ),
                    "cool_skylight_contribution_increased": (
                        relit_metrics["cool_skylight_blue_excess"]
                        > baseline_metrics["cool_skylight_blue_excess"] + 3.0
                    ),
                }
                report["captures"] = {
                    "baseline": {
                        "path": str(baseline_path),
                        "metadata": baseline_meta,
                        "metrics": baseline_metrics,
                    },
                    "relit": {
                        "path": str(relit_path),
                        "metadata": relit_meta,
                        "metrics": relit_metrics,
                    },
                    "patched": {
                        "path": str(patched_path),
                        "metadata": patched_meta,
                        "metrics": patched_metrics,
                    },
                    "mean_absolute_difference": {
                        "baseline_to_relit": baseline_to_relit,
                        "relit_to_patched": relit_to_patched,
                    },
                    "checks": checks,
                }
                if not all(checks.values()):
                    failed = sorted(name for name, passed in checks.items() if not passed)
                    raise RuntimeError(f"Visual acceptance checks failed: {failed}")
            except BaseException as exc:
                primary_error = exc
                raise
            finally:
                cleanup_errors: list[str] = []
                if preview_session_id is not None:
                    try:
                        restored = _json_value(
                            await _call(
                                session,
                                report,
                                "capture_cycles_viewport",
                                {"action": "RESTORE", "session_id": preview_session_id},
                            )
                        )
                        report["restore"] = _json_safe(restored)
                    except BaseException as exc:
                        cleanup_errors.append(f"preview restore: {exc}")

                rollback_results = []
                for transaction_id in reversed(transactions):
                    try:
                        rolled_back = _json_value(
                            await _call(
                                session,
                                report,
                                "apply_light_plan",
                                {"action": "ROLLBACK", "transaction_id": transaction_id},
                            )
                        )
                        rollback_results.append(_json_safe(rolled_back))
                    except BaseException as exc:
                        cleanup_errors.append(f"rollback {transaction_id}: {exc}")
                report["rollbacks"] = rollback_results

                if original_lights is not None and not cleanup_errors:
                    try:
                        final_lights = _json_value(
                            await _call(session, report, "list_lights")
                        )
                        report["final_lights"] = _json_safe(final_lights)
                        if _json_safe(final_lights) != _json_safe(original_lights):
                            cleanup_errors.append("list_lights differed after rollback")
                    except BaseException as exc:
                        cleanup_errors.append(f"post-rollback list_lights: {exc}")

                report["cleanup_errors"] = cleanup_errors
                if cleanup_errors and primary_error is None:
                    raise RuntimeError("; ".join(cleanup_errors))


def _write_report(path: Path, report: dict[str, Any]) -> None:
    report["finished_at_unix"] = time.time()
    report["elapsed_seconds"] = report["finished_at_unix"] - report["started_at_unix"]
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    args = _arguments()
    artifact_dir = _artifact_directory(args.artifact_dir)
    report_path = artifact_dir / "acceptance-report.json"
    report: dict[str, Any] = {
        "schema_version": 1,
        "started_at_unix": time.time(),
        "success": False,
        "repo": str(args.repo.expanduser().resolve()),
        "artifact_dir": str(artifact_dir),
        "calls": [],
        "assumptions": {
            "blender_is_separate_visible_factory_process": True,
            "workspace_is_visible_and_unminimized": "AI Preview",
            "addon_tcp_endpoint": "127.0.0.1:9876",
            "no_blend_save_is_performed": True,
            "capture_mode": args.capture_mode,
            "quick_render_replaces_render_result": args.capture_mode == "QUICK_RENDER",
        },
    }
    try:
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise RuntimeError("--timeout must be a finite positive number")
        if not math.isfinite(args.settle_seconds) or not 0 <= args.settle_seconds <= 30:
            raise RuntimeError("--settle-seconds must be finite and between 0 and 30")
        if not 1 <= args.preview_samples <= 4096:
            raise RuntimeError("--preview-samples must be between 1 and 4096")
        asyncio.run(_run(args, artifact_dir, report))
        report["success"] = True
        return_code = 0
    except BaseException as exc:
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        return_code = 1
    finally:
        _write_report(report_path, report)
        print(json.dumps({"success": report["success"], "report": str(report_path)}))
    return return_code


if __name__ == "__main__":
    sys.exit(main())
