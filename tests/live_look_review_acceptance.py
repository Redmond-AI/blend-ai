"""Exercise cinematic review through a visible Blender and the native MCP boundary."""

from __future__ import annotations

import argparse
import asyncio
import base64
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, ImageContent, TextContent


EXPECTED_SCENE = "Mini Warehouse Relighting Fixture"
EXPECTED_CAMERA = "Warehouse_Camera"
REQUIRED_TOOLS = {
    "get_scene_info",
    "get_look_profile_context",
    "upsert_look_profile",
    "inspect_look_review_state",
    "submit_look_render_batch",
    "get_look_render_batch",
    "get_look_render_result",
    "review_look_render",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--source-scene", default=EXPECTED_SCENE)
    parser.add_argument("--camera", default=EXPECTED_CAMERA)
    parser.add_argument("--frame", type=int, default=1)
    parser.add_argument("--profile-preset", choices=("fixture", "blendkit"), default="fixture")
    return parser.parse_args()


def _json_value(result: CallToolResult) -> Any:
    if result.structuredContent is not None:
        value: Any = result.structuredContent
        if isinstance(value, dict) and set(value) == {"result"}:
            return value["result"]
        return value
    for item in result.content:
        if isinstance(item, TextContent):
            try:
                return json.loads(item.text)
            except json.JSONDecodeError:
                pass
    raise RuntimeError("MCP result lacks structured JSON")


async def _call(
    session: ClientSession, name: str, arguments: dict[str, Any] | None = None
) -> CallToolResult:
    result = await session.call_tool(name, arguments or {})
    if result.isError:
        messages = [item.text for item in result.content if isinstance(item, TextContent)]
        raise RuntimeError(f"{name} failed: {'; '.join(messages)}")
    return result


def _profile(preset: str = "fixture") -> dict[str, Any]:
    if preset == "blendkit":
        return {
            "schema_version": 1,
            "profile_id": "native-review-smoke",
            "display_name": "Fishermans House Cinematic Dusk",
            "description": "Disposable BlendKit acceptance profile with cool ambience and a warm motivated key",
            "status": "DRAFT",
            "seed": 144,
            "generator_version": "native-review-smoke-v1",
            "tags": ["integration", "cinematic-review", "blendkit"],
            "lighting": {
                "existing_light_policy": "KEEP",
                "lights": [
                    {
                        "id": "key",
                        "name": "AI Cabin Warm Key",
                        "type": "AREA",
                        "location_world": [15.0, -20.0, 25.0],
                        "target_point": [1.94, -1.70, 2.66],
                        "energy": 1000.0,
                        "color_rgb": [1.0, 0.58, 0.28],
                        "area_shape": "DISK",
                        "size": 12.0,
                        "use_shadow": True,
                    },
                    {
                        "id": "moon",
                        "name": "AI Cool Dusk Direction",
                        "type": "SUN",
                        "location_world": [-25.0, 10.0, 40.0],
                        "target_point": [1.94, -1.70, 2.66],
                        "energy": 1.5,
                        "color_rgb": [0.28, 0.43, 1.0],
                        "sun_angle_degrees": 8.0,
                        "use_shadow": True,
                    },
                ],
            },
            "world": {
                "mode": "MANAGED_SOLID",
                "color_rgb": [0.004, 0.01, 0.04],
                "strength": 0.16,
            },
            "atmosphere": [],
            "post": {
                "mode": "MANAGED_STACK",
                "bloom": {"enabled": True, "threshold": 0.9, "strength": 0.14, "radius": 0.5},
                "grain": {"enabled": True, "strength": 0.025, "scale": 1.0, "seed": 144},
                "vignette": {"enabled": True, "strength": 0.1, "feather": 0.72},
            },
            "color_management": {
                "mode": "MANAGED",
                "exposure": -0.3,
                "gamma": 1.0,
            },
            "camera": {"mode": "KEEP"},
            "render": {
                "engine": "CYCLES",
                "samples": 16,
                "denoise": False,
                "resolution_percentage": 25,
                "film_transparent": False,
                "use_motion_blur": False,
            },
            "review_intent": {
                "summary": "A cinematic blue-hour Fishermans House with cool environmental depth, a restrained warm cabin key, readable shadows and reflections, and subtle bloom, grain, and vignette.",
                "expects_shadows": True,
                "expects_reflections": True,
                "expects_volume": False,
                "expects_compositing": True,
            },
        }
    return {
        "schema_version": 1,
        "profile_id": "native-review-smoke",
        "display_name": "Native Review Smoke",
        "description": "Disposable visible-MCP cinematic review acceptance profile",
        "status": "DRAFT",
        "seed": 144,
        "generator_version": "native-review-smoke-v1",
        "tags": ["integration", "cinematic-review"],
        "lighting": {
            "existing_light_policy": "KEEP",
            "lights": [
                {
                    "id": "key",
                    "name": "AI Review Key",
                    "type": "AREA",
                    "location_world": [2.5, -3.0, 5.5],
                    "target_point": [0.0, 0.0, 1.2],
                    "energy": 700.0,
                    "color_rgb": [1.0, 0.72, 0.48],
                    "area_shape": "DISK",
                    "size": 3.0,
                    "use_shadow": True,
                }
            ],
        },
        "world": {
            "mode": "MANAGED_SOLID",
            "color_rgb": [0.008, 0.018, 0.05],
            "strength": 0.12,
        },
        "atmosphere": [],
        "post": {
            "mode": "MANAGED_STACK",
            "bloom": {"enabled": True, "threshold": 0.85, "strength": 0.12, "radius": 0.45},
            "grain": {"enabled": True, "strength": 0.02, "scale": 1.0, "seed": 144},
            "vignette": {"enabled": True, "strength": 0.08, "feather": 0.7},
        },
        "color_management": {"mode": "KEEP"},
        "camera": {"mode": "KEEP"},
        "render": {
            "engine": "CYCLES",
            "samples": 4,
            "denoise": False,
            "resolution_percentage": 25,
            "film_transparent": False,
            "use_motion_blur": False,
        },
        "review_intent": {
            "summary": "Warm directional key with cool depth, readable shadows, reflections, and restrained finishing.",
            "expects_shadows": True,
            "expects_reflections": True,
            "expects_volume": False,
            "expects_compositing": True,
        },
    }


def _mock_reference_packet(image_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a checksum-bound packet for transport acceptance, not visual evaluation."""
    beauty_path = Path(image_records[0]["path"]).resolve()
    source_sha256 = hashlib.sha256(beauty_path.read_bytes()).hexdigest()
    return {
        "packet_id": "native-reference-smoke",
        "brief": "Soft motivated light with readable form and grounded surface response.",
        "targets": ["directional depth", "visible surface variation"],
        "non_targets": [],
        "references": [
            {
                "reference_id": "REF-01",
                "image_path": str(beauty_path),
                "source_sha256": source_sha256,
                "roles": ["LIGHTING_TOPOLOGY"],
                "comparison_focus": "Visible source direction, softness, and falloff.",
            },
            {
                "reference_id": "REF-02",
                "image_path": str(beauty_path),
                "source_sha256": source_sha256,
                "roles": ["MATERIAL_AND_SURFACE"],
                "comparison_focus": "Visible roughness, wear, and tactile variation.",
            },
        ],
    }


async def _render_review(
    session: ClientSession,
    *,
    artifact_dir: Path,
    target_scene: str,
    profile_revision: int,
    timeout: float,
    image_offset: int,
    camera: str = EXPECTED_CAMERA,
    frame: int = 1,
) -> tuple[str, str, list[dict[str, Any]]]:
    batch_request = {
        "profile_ids": ["native-review-smoke"],
        "target_scenes": [target_scene],
        "pairing": "PAIRWISE",
        "frames": {"frames": [frame]},
        "output_root": str(artifact_dir / "renders"),
        "filename_template": "{scene}-{profile}-{camera}-{frame}",
        "render_settings": {
            "samples": 4,
            "denoise": False,
            "resolution_percentage": 25,
            "file_format": "PNG",
            "color_depth": "8",
            "existing_file_policy": "ERROR",
        },
        "continue_on_error": False,
        "include_review_packet": True,
    }
    batch_request["camera_names"] = [camera]
    submitted = _json_value(
        await _call(
            session,
            "submit_look_render_batch",
            {
                "batch": batch_request,
                "expected_profile_revision": profile_revision,
            },
        )
    )
    batch_id = submitted["batch_id"]
    deadline = time.monotonic() + timeout
    batch: dict[str, Any] = {}
    while time.monotonic() < deadline:
        batch = _json_value(
            await _call(
                session,
                "get_look_render_batch",
                {"batch_id": batch_id, "include_results": True},
            )
        )
        if batch.get("status") in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            break
        await asyncio.sleep(0.1)
    if batch.get("status") != "SUCCEEDED":
        raise RuntimeError(f"Native review batch did not succeed: {batch}")
    items = batch.get("items") or batch.get("results")
    if not isinstance(items, list) or len(items) != 1:
        raise RuntimeError(f"Native review batch returned invalid items: {batch}")
    result_id = items[0]["result_id"]
    fetched = await _call(
        session,
        "get_look_render_result",
        {
            "batch_id": batch_id,
            "result_id": result_id,
            "max_size": 1024,
            "format": "JPEG",
            "jpeg_quality": 90,
        },
    )
    images = [item for item in fetched.content if isinstance(item, ImageContent)]
    if len(images) != 2:
        raise RuntimeError(f"Native result returned {len(images)} images, expected two")
    records = []
    for index, (label, item) in enumerate(zip(("beauty", "diagnostics"), images)):
        suffix = ".jpg" if item.mimeType == "image/jpeg" else ".png"
        image_path = artifact_dir / f"{image_offset + index:02d}-{label}{suffix}"
        data = base64.b64decode(item.data, validate=True)
        image_path.write_bytes(data)
        records.append(
            {
                "label": label,
                "mime_type": item.mimeType,
                "bytes": len(data),
                "path": str(image_path),
            }
        )
    return batch_id, result_id, records


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.expanduser().resolve()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
    server = StdioServerParameters(
        command=uv,
        args=[
            "run",
            "--directory",
            str(repo),
            "python",
            str(repo / "tests/helpers/mock_review_mcp_server.py"),
        ],
        cwd=repo,
    )
    async with stdio_client(server) as streams:
        async with ClientSession(
            *streams, read_timeout_seconds=timedelta(seconds=args.timeout)
        ) as session:
            await session.initialize()
            tools = {item.name for item in (await session.list_tools()).tools}
            missing = sorted(REQUIRED_TOOLS - tools)
            if missing:
                raise RuntimeError(f"Native MCP tool list is missing {missing}")

            scene_info = _json_value(await _call(session, "get_scene_info"))
            if scene_info.get("scene_name") != args.source_scene:
                raise RuntimeError(
                    "Safety stop: visible Blender is not the explicitly named source Scene"
                )
            if args.profile_preset == "fixture" and args.source_scene != EXPECTED_SCENE:
                raise RuntimeError("Fixture preset requires the isolated warehouse Scene")
            await _call(
                session,
                "get_look_profile_context",
                {"source_scene": args.source_scene, "detail": "FULL"},
            )
            compile_result = _json_value(
                await _call(
                    session,
                    "upsert_look_profile",
                    {
                        "action": "COMPILE",
                        "source_scene": args.source_scene,
                        "profile": _profile(args.profile_preset),
                        "update_mode": "CREATE_VERSION",
                        "strict": True,
                    },
                )
            )
            entry = compile_result["profile"]
            target_scene = entry["scene_name"]
            profile_revision = entry["revision"]
            audit_arguments = {
                "profile_id": "native-review-smoke",
                "target_scene": target_scene,
                "camera": args.camera,
                "expected_profile_revision": profile_revision,
            }
            audit = _json_value(
                await _call(
                    session,
                    "inspect_look_review_state",
                    audit_arguments,
                )
            )
            if audit.get("status") == "FAIL":
                raise RuntimeError(f"Native audit failed: {audit}")

            batch_id, result_id, image_records = await _render_review(
                session,
                artifact_dir=artifact_dir,
                target_scene=target_scene,
                profile_revision=profile_revision,
                timeout=args.timeout,
                image_offset=1,
                camera=args.camera,
                frame=args.frame,
            )

            realism = _json_value(
                await _call(
                    session,
                    "review_look_render",
                    {
                        "mode": "REALISM",
                        "batch_id": batch_id,
                        "result_id": result_id,
                        "realism_prompt_version": "causal-v1",
                    },
                )
            )
            if realism.get("request_id") != "mock-native-mcp-realism-vision":
                raise RuntimeError(f"Mock realism receipt is invalid: {realism}")
            if realism.get("input_image_count") != 1:
                raise RuntimeError(f"Mock realism image isolation is invalid: {realism}")
            if realism.get("prompt_version") != "causal-v1":
                raise RuntimeError(f"Mock realism prompt receipt is invalid: {realism}")
            if realism.get("provider_call_count") != 2:
                raise RuntimeError(f"Mock realism call count is invalid: {realism}")
            rewrite = realism.get("rewrite", {})
            if rewrite.get("request_id") != "mock-native-mcp-realism-rewrite":
                raise RuntimeError(f"Mock realism rewrite receipt is invalid: {realism}")
            if rewrite.get("input_image_count") != 0:
                raise RuntimeError(f"Mock realism rewrite image isolation is invalid: {realism}")

            reference = _json_value(
                await _call(
                    session,
                    "review_look_render",
                    {
                        "mode": "REFERENCE",
                        "batch_id": batch_id,
                        "result_id": result_id,
                        "reference_packet": _mock_reference_packet(image_records),
                    },
                )
            )
            if reference.get("request_id") != "mock-native-mcp-reference-vision":
                raise RuntimeError(f"Mock reference receipt is invalid: {reference}")
            if reference.get("input_image_count") != 3:
                raise RuntimeError(f"Mock reference image boundary is invalid: {reference}")
            if reference.get("provider_call_count") != 2:
                raise RuntimeError(f"Mock reference call count is invalid: {reference}")
            reference_rewrite = reference.get("rewrite", {})
            if reference_rewrite.get("request_id") != "mock-native-mcp-reference-rewrite":
                raise RuntimeError(f"Mock reference rewrite receipt is invalid: {reference}")
            if reference_rewrite.get("input_image_count") != 0:
                raise RuntimeError(f"Mock reference rewrite image boundary is invalid: {reference}")
            if not reference.get("holistic_result", {}).get("assessments"):
                raise RuntimeError(f"Mock holistic reference context is missing: {reference}")

            critique = _json_value(
                await _call(
                    session,
                    "review_look_render",
                    {"mode": "CRITIQUE", "batch_id": batch_id, "result_id": result_id},
                )
            )
            if critique.get("request_id") != "mock-native-mcp-review":
                raise RuntimeError(f"Mock critique receipt is invalid: {critique}")

            revised_profile = _profile(args.profile_preset)
            revised_profile["display_name"] = "Native Review Smoke - Approved Revision"
            revised_profile["lighting"]["lights"][0]["energy"] = 560.0
            if revised_profile["lighting"]["lights"][0]["id"] != "key":
                raise RuntimeError("Approved revision did not preserve the managed light ID")
            revised_compile = _json_value(
                await _call(
                    session,
                    "upsert_look_profile",
                    {
                        "action": "COMPILE",
                        "source_scene": args.source_scene,
                        "profile": revised_profile,
                        "update_mode": "CREATE_VERSION",
                        "expected_profile_revision": profile_revision,
                        "strict": True,
                    },
                )
            )
            revised_entry = revised_compile["profile"]
            revised_target_scene = revised_entry["scene_name"]
            revised_revision = revised_entry["revision"]
            revised_audit_arguments = {
                "profile_id": "native-review-smoke",
                "target_scene": revised_target_scene,
                "camera": args.camera,
                "expected_profile_revision": revised_revision,
            }
            revised_audit = _json_value(
                await _call(
                    session,
                    "inspect_look_review_state",
                    revised_audit_arguments,
                )
            )
            if revised_audit.get("status") == "FAIL":
                raise RuntimeError(f"Revised native audit failed: {revised_audit}")
            revised_batch_id, revised_result_id, revised_images = await _render_review(
                session,
                artifact_dir=artifact_dir,
                target_scene=revised_target_scene,
                profile_revision=revised_revision,
                timeout=args.timeout,
                image_offset=3,
                camera=args.camera,
                frame=args.frame,
            )
            comparison = _json_value(
                await _call(
                    session,
                    "review_look_render",
                    {
                        "mode": "COMPARE",
                        "batch_id": revised_batch_id,
                        "result_id": revised_result_id,
                        "baseline_batch_id": batch_id,
                        "baseline_result_id": result_id,
                    },
                )
            )
            if comparison.get("request_id") != "mock-native-mcp-compare":
                raise RuntimeError(f"Mock comparison receipt is invalid: {comparison}")
            return {
                "status": "PASS",
                "scene": args.source_scene,
                "baseline": {
                    "target_scene": target_scene,
                    "profile_revision": profile_revision,
                    "audit_status": audit["status"],
                    "batch_id": batch_id,
                    "result_id": result_id,
                    "native_images": image_records,
                },
                "revised": {
                    "target_scene": revised_target_scene,
                    "profile_revision": revised_revision,
                    "audit_status": revised_audit["status"],
                    "batch_id": revised_batch_id,
                    "result_id": revised_result_id,
                    "native_images": revised_images,
                    "approved_patch": {"lighting.lights.key.energy": 560.0},
                },
                "mock_realism": realism,
                "mock_reference": reference,
                "mock_review": critique,
                "mock_comparison": comparison,
                "saved_blend": False,
            }


def main() -> None:
    args = _arguments()
    report = asyncio.run(_run(args))
    path = args.artifact_dir.expanduser().resolve() / "native-mcp-report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
