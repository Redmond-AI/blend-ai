"""Benchmark relighting tools against the visible 20,000-instance fixture.

Launch ``live_blender_bootstrap.py --fixture scale`` in a separate visible
factory Blender process first.  This harness refuses any scene that lacks the
exact in-memory marker, never saves, and rolls back its temporary 64-light plan.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import timedelta
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, TextContent


SCENE_NAME = "Blend AI Relighting Scale Benchmark"
CAMERA_NAME = "Scale_Benchmark_Camera"
INSTANCE_COUNT = 20_000
SOURCE_TRIANGLES = 256
GRID_COLUMNS = 200
GRID_ROWS = 100
GRID_SPACING = 1.25


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--uv", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=300.0)
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


def _json_value(result: CallToolResult) -> Any:
    value = result.structuredContent
    if value is not None:
        if isinstance(value, dict) and set(value) == {"result"}:
            return value["result"]
        return value
    for content in result.content:
        if isinstance(content, TextContent):
            try:
                return json.loads(content.text)
            except json.JSONDecodeError:
                continue
    raise RuntimeError("MCP result contains no structured JSON")


async def _call(
    session: ClientSession,
    report: dict[str, Any],
    name: str,
    arguments: dict[str, Any] | None = None,
) -> tuple[Any, float]:
    started = time.perf_counter()
    result = await session.call_tool(name, arguments or {})
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    report["calls"].append(
        {"tool": name, "elapsed_ms": elapsed_ms, "is_error": bool(result.isError)}
    )
    if result.isError:
        messages = [item.text for item in result.content if isinstance(item, TextContent)]
        raise RuntimeError(f"{name} failed: {'; '.join(messages) or 'unknown error'}")
    return _json_value(result), elapsed_ms


def _grid_position(index: int) -> tuple[float, float]:
    column = index % GRID_COLUMNS
    row = index // GRID_COLUMNS
    return (
        (column - (GRID_COLUMNS - 1) / 2) * GRID_SPACING,
        (row - (GRID_ROWS - 1) / 2) * GRID_SPACING,
    )


def _context_summary(context: dict[str, Any], elapsed_ms: float) -> dict[str, Any]:
    payload_bytes = len(
        json.dumps(context, separators=(",", ":"), allow_nan=False).encode("utf-8")
    )
    return {
        "elapsed_ms": elapsed_ms,
        "payload_bytes": payload_bytes,
        "geometry_revision": context.get("geometry_revision"),
        "lighting_revision": context.get("lighting_revision"),
        "cache": context.get("cache"),
        "instance_records": len(context.get("instances", [])),
        "next_cursor": bool(context.get("next_cursor")),
        "truncated": bool(context.get("truncated")),
        "timings_ms": context.get("timings_ms"),
    }


async def _run(args: argparse.Namespace, report: dict[str, Any]) -> None:
    repo = args.repo.expanduser().resolve()
    server = StdioServerParameters(
        command=str(_resolve_uv(args.uv)),
        args=["run", "--directory", str(repo), "blend-ai"],
        cwd=repo,
    )
    transaction_id: str | None = None
    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(
            read_stream,
            write_stream,
            read_timeout_seconds=timedelta(seconds=args.timeout),
        ) as session:
            await session.initialize()
            listed = await session.list_tools()
            required = {
                "execute_blender_code",
                "get_lighting_context",
                "batch_raycast",
                "apply_light_plan",
            }
            missing = sorted(required - {tool.name for tool in listed.tools})
            if missing:
                raise RuntimeError(f"MCP tool listing is missing {missing}")

            marker_result, _marker_ms = await _call(
                session,
                report,
                "execute_blender_code",
                {
                    "code": (
                        "import bpy, json; s=bpy.context.scene; "
                        "print(json.dumps({'name': s.name, "
                        "'marker': bool(s.get('blend_ai_scale_fixture', False)), "
                        "'instances': int(s.get('expected_instance_count', 0)), "
                        "'source_triangles': int(s.get('source_triangle_count', 0)), "
                        "'evaluated_triangles': "
                        "int(s.get('expected_evaluated_triangles', 0)), "
                        "'filepath': bpy.data.filepath}))"
                    )
                },
            )
            if not isinstance(marker_result, dict):
                raise RuntimeError("Scale fixture marker result is invalid")
            marker = json.loads(marker_result.get("output", "{}"))
            expected_marker = {
                "name": SCENE_NAME,
                "marker": True,
                "instances": INSTANCE_COUNT,
                "source_triangles": SOURCE_TRIANGLES,
                "evaluated_triangles": INSTANCE_COUNT * SOURCE_TRIANGLES,
                "filepath": "",
            }
            if marker != expected_marker:
                raise RuntimeError(
                    "Safety stop: Blender is not the unsaved scale fixture: "
                    f"{marker!r}"
                )
            report["fixture"] = marker

            context_args = {
                "camera_name": CAMERA_NAME,
                "scope": "SCENE",
                "detail": "BOUNDS",
                "semantic_terms": [],
                "include_hidden": False,
                "max_instances": 500,
                "max_surface_triangles": 0,
            }
            cold, cold_ms = await _call(
                session,
                report,
                "get_lighting_context",
                {**context_args, "cache_mode": "REFRESH"},
            )
            if not isinstance(cold, dict) or cold.get("scene", {}).get("name") != SCENE_NAME:
                raise RuntimeError("Cold context did not return the exact scale fixture")
            geometry_revision = cold.get("geometry_revision")
            if not isinstance(geometry_revision, int) or isinstance(geometry_revision, bool):
                raise RuntimeError("Cold context lacks a geometry revision")

            # The first USE call establishes the USE-keyed cache if the handler
            # implementation separates refresh policy from request identity;
            # the second is the measured warm page.
            await _call(
                session,
                report,
                "get_lighting_context",
                {**context_args, "cache_mode": "USE"},
            )
            warm, warm_ms = await _call(
                session,
                report,
                "get_lighting_context",
                {**context_args, "cache_mode": "USE"},
            )
            if not isinstance(warm, dict):
                raise RuntimeError("Warm context is invalid")
            report["context"] = {
                "cold": _context_summary(cold, cold_ms),
                "warm": _context_summary(warm, warm_ms),
            }

            rays = []
            for row in range(16):
                for column in range(16):
                    index = row * GRID_COLUMNS + column
                    x, y = _grid_position(index)
                    rays.append(
                        {
                            "id": f"ray_{row:02d}_{column:02d}",
                            "origin": [x, y, 3.0],
                            "target": [x, y, -1.0],
                        }
                    )
            raycast, raycast_ms = await _call(
                session,
                report,
                "batch_raycast",
                {
                    "rays": rays,
                    "max_hits": 8,
                    "expected_geometry_revision": geometry_revision,
                    "time_budget_ms": 2000,
                },
            )
            if not isinstance(raycast, dict):
                raise RuntimeError("Scale raycast response is invalid")
            ray_results = raycast.get("rays", [])
            report["raycast"] = {
                "elapsed_ms": raycast_ms,
                "complete": bool(raycast.get("complete")),
                "result_count": len(ray_results),
                "hit_count": sum(bool(ray.get("hits")) for ray in ray_results),
                "timings_ms": raycast.get("timings_ms"),
            }

            lights = []
            for index in range(64):
                x, y = _grid_position(index)
                lights.append(
                    {
                        "id": f"benchmark_light_{index:02d}",
                        "name": f"Benchmark Light {index:02d}",
                        "type": "POINT",
                        "location_world": [x, y, 2.0],
                        "energy": 10.0,
                        "color_rgb": [0.2, 0.3, 1.0],
                        "radius": 0.1,
                    }
                )
            plan = {
                "action": "VALIDATE",
                "plan_id": "scale_benchmark_64",
                "mode": "REPLACE_MANAGED",
                "expected_geometry_revision": geometry_revision,
                "lights": lights,
                "strict": True,
            }
            validated, validate_ms = await _call(
                session, report, "apply_light_plan", plan
            )
            if not isinstance(validated, dict) or not validated.get("valid"):
                raise RuntimeError("Scale light plan did not validate")
            applied, apply_ms = await _call(
                session,
                report,
                "apply_light_plan",
                {**plan, "action": "APPLY"},
            )
            if not isinstance(applied, dict) or not applied.get("applied"):
                raise RuntimeError("Scale light plan did not apply")
            transaction_id = applied.get("transaction_id")
            if not isinstance(transaction_id, str) or not transaction_id:
                raise RuntimeError("Scale light plan lacks a transaction ID")
            report["light_plan"] = {
                "validate_ms": validate_ms,
                "apply_ms": apply_ms,
                "geometry_revision": applied.get("geometry_revision"),
                "transaction_id": transaction_id,
            }

            rolled_back, rollback_ms = await _call(
                session,
                report,
                "apply_light_plan",
                {"action": "ROLLBACK", "transaction_id": transaction_id},
            )
            transaction_id = None
            if not isinstance(rolled_back, dict) or not rolled_back.get("rolled_back"):
                raise RuntimeError("Scale light plan did not roll back")
            report["light_plan"]["rollback_ms"] = rollback_ms

            cold_summary = report["context"]["cold"]
            report["targets"] = {
                "cold_context_under_3000_ms": cold_ms < 3000,
                "warm_context_under_200_ms": warm_ms < 200,
                "default_payload_under_512_kib": cold_summary["payload_bytes"] < 512 * 1024,
                "payload_under_2_mib": cold_summary["payload_bytes"] < 2 * 1024 * 1024,
                "raycast_256_under_1000_ms": raycast_ms < 1000,
                "raycast_complete": bool(raycast.get("complete")),
                "apply_64_under_250_ms": apply_ms < 250,
                "geometry_revision_stable": (
                    applied.get("geometry_revision") == geometry_revision
                ),
            }

    if transaction_id is not None:
        raise RuntimeError(
            "Internal cleanup invariant failed: benchmark transaction remains active"
        )


def _write_report(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    args = _arguments()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    if artifact_dir in {Path("/"), Path.home().resolve()}:
        raise RuntimeError(f"Refusing unsafe artifact directory {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifact_dir / "scale-benchmark-report.json"
    report: dict[str, Any] = {
        "schema_version": 1,
        "started_at_unix": time.time(),
        "success": False,
        "calls": [],
    }
    try:
        asyncio.run(_run(args, report))
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
        report["finished_at_unix"] = time.time()
        report["elapsed_seconds"] = (
            report["finished_at_unix"] - report["started_at_unix"]
        )
        _write_report(report_path, report)
        print(json.dumps({"success": report["success"], "report": str(report_path)}))
    return return_code


if __name__ == "__main__":
    sys.exit(main())
