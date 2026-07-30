"""Run the REALISM prompt-calibration stage against one isolated visible Blender."""

from __future__ import annotations

import argparse
import asyncio
import base64
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import secrets
import shutil
import time
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, ImageContent, TextContent


ASSET_ID = "b70bca84-9b17-4dc8-a2ba-cd95e2b970a7"
SOURCE_SCENE = "Abandoned Forest Railway Station"
SOURCE_CAMERA = "Camera "
PILOT_SOURCE = "REALISM Pilot Source"
PILOT_CAMERA = "PilotAuthoredCamera"
FRAME = 18
MODEL = "anthropic/claude-sonnet-4.6"
PROMPT_VERSIONS = ("literal-v1", "evidence-v1", "causal-v1", "look-only-v1")
MAX_PAID_PROVIDER_CALLS = 16
REQUIRED_TOOLS = {
    "execute_blender_code",
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
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--expected-blend", required=True, type=Path)
    parser.add_argument("--resume-runtime", action="store_true")
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _profile(*, hard_negative: bool) -> dict[str, Any]:
    profile_id = "realism-pilot-hard-negative" if hard_negative else "realism-pilot"
    display_name = (
        "Abandoned Station Excessive Finish Control"
        if hard_negative
        else "Abandoned Station Grounded Late Afternoon"
    )
    profile = {
        "schema_version": 1,
        "profile_id": profile_id,
        "display_name": display_name,
        "description": (
            "Deliberately excessive finish-only hard negative; lighting is held equal."
            if hard_negative
            else "Scene-driven late-afternoon naturalism using the authored sun and a restrained cool station fill."
        ),
        "status": "DRAFT",
        "seed": 74261,
        "generator_version": "realism-pilot-v1",
        "tags": ["realism-pilot", "calibration"],
        "lighting": {
            "existing_light_policy": "KEEP",
            "lights": [
                {
                    "id": "station_fill",
                    "name": "AI Station Cool Fill",
                    "type": "AREA",
                    "location_world": [5.5, -7.0, 9.0],
                    "target_point": [-1.0, 8.0, 1.5],
                    "energy": 120.0,
                    "color_rgb": [0.42, 0.56, 0.78],
                    "area_shape": "DISK",
                    "size": 9.0,
                    "use_shadow": True,
                }
            ],
        },
        "world": {"mode": "KEEP"},
        "atmosphere": [],
        "post": {
            "mode": "MANAGED_STACK",
            "bloom": {
                "enabled": True,
                "threshold": 0.95 if not hard_negative else 0.18,
                "strength": 0.035 if not hard_negative else 1.35,
                "radius": 0.45 if not hard_negative else 0.95,
            },
            "grain": {
                "enabled": True,
                "strength": 0.012 if not hard_negative else 0.32,
                "scale": 1.0 if not hard_negative else 2.4,
                "seed": 74261,
            },
            "vignette": {
                "enabled": True,
                "strength": 0.045 if not hard_negative else 0.58,
                "feather": 0.78 if not hard_negative else 0.28,
            },
        },
        "color_management": {
            "mode": "MANAGED",
            "exposure": -0.1 if not hard_negative else -0.65,
            "gamma": 1.0,
        },
        "camera": {"mode": "KEEP"},
        "render": {
            "engine": "CYCLES",
            "samples": 64,
            "denoise": True,
            "resolution_percentage": 100,
            "film_transparent": False,
            "use_motion_blur": False,
        },
        "review_intent": {
            "summary": (
                "Grounded live-action late-afternoon abandoned forest railway station; "
                "one motivated warm authored sun, restrained cool station fill, readable "
                "shadow detail, controlled foliage and timber highlights, tactile rail, "
                "ballast, brick and wood response, subtle depth, and restrained optics. "
                "Fixed references REF-01 through REF-04; do not copy people, props, or "
                "architecture and do not treat finish effects as realism proof."
            ),
            "expects_shadows": True,
            "expects_reflections": True,
            "expects_volume": False,
            "expects_compositing": True,
        },
    }
    return profile


async def _prepare_derived_source(session: ClientSession, expected_blend: Path) -> dict[str, Any]:
    code = f"""import bpy, json
expected = {str(expected_blend)!r}
source = bpy.data.scenes.get({SOURCE_SCENE!r})
if bpy.data.filepath != expected:
    raise RuntimeError("unexpected source blend")
if source is None or source.camera is None:
    raise RuntimeError("source scene or authored camera is missing")
if source.camera.name != {SOURCE_CAMERA!r}:
    raise RuntimeError("authored camera identity changed")
if bpy.data.scenes.get({PILOT_SOURCE!r}) is not None:
    raise RuntimeError("pilot source already exists; use a fresh Blender process")
source_camera = source.camera
source_matrix = [list(row) for row in source_camera.matrix_world]
derived = source.copy()
derived.name = {PILOT_SOURCE!r}
pilot_camera = source_camera.copy()
pilot_camera.data = source_camera.data.copy()
pilot_camera.name = {PILOT_CAMERA!r}
derived.collection.objects.link(pilot_camera)
derived.camera = pilot_camera
if [list(row) for row in pilot_camera.matrix_world] != source_matrix:
    raise RuntimeError("derived camera transform differs from authored camera")
print(json.dumps({{
    "source_scene": source.name,
    "source_camera": source.camera.name,
    "derived_scene": derived.name,
    "derived_camera": derived.camera.name,
    "camera_matrix_identical": True,
    "camera_lens": pilot_camera.data.lens,
    "source_scene_unchanged": source.camera is source_camera,
    "blend_path": bpy.data.filepath,
    "blend_is_dirty_after_in_memory_copy": bpy.data.is_dirty,
}}))"""
    result = _json_value(await _call(session, "execute_blender_code", {"code": code}))
    if not isinstance(result, dict) or not result.get("success"):
        raise RuntimeError(f"Could not prepare derived pilot source: {result}")
    return json.loads(result["output"])


async def _compile(
    session: ClientSession,
    *,
    profile: dict[str, Any],
    base_revision: int,
    geometry_revision: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    common = {
        "source_scene": PILOT_SOURCE,
        "profile": profile,
        "update_mode": "CREATE_VERSION",
        "expected_base_revision": base_revision,
        "expected_geometry_revision": geometry_revision,
        "strict": True,
    }
    await _call(session, "upsert_look_profile", {"action": "VALIDATE", **common})
    compiled = _json_value(
        await _call(session, "upsert_look_profile", {"action": "COMPILE", **common})
    )
    entry = compiled["profile"]
    audit = _json_value(
        await _call(
            session,
            "inspect_look_review_state",
            {
                "profile_id": profile["profile_id"],
                "target_scene": entry["scene_name"],
                "camera": PILOT_CAMERA,
                "expected_profile_revision": entry["revision"],
                "expected_geometry_revision": geometry_revision,
            },
        )
    )
    if audit.get("status") == "FAIL":
        raise RuntimeError(f"Technical audit failed for {profile['profile_id']}: {audit}")
    return entry, audit


async def _render(
    session: ClientSession,
    *,
    artifact_dir: Path,
    label: str,
    profile_id: str,
    target_scene: str,
    profile_revision: int,
    samples: int,
    denoise: bool,
    resolution_percentage: int,
    include_review_packet: bool,
    timeout: float,
) -> dict[str, Any]:
    output_root = artifact_dir / "renders" / label
    submitted = _json_value(
        await _call(
            session,
            "submit_look_render_batch",
            {
                "batch": {
                    "profile_ids": [profile_id],
                    "target_scenes": [target_scene],
                    "pairing": "PAIRWISE",
                    "frames": {"frames": [FRAME]},
                    "output_root": str(output_root),
                    "filename_template": "{scene}-{profile}-{camera}-{frame}",
                    "render_settings": {
                        "samples": samples,
                        "denoise": denoise,
                        "resolution_percentage": resolution_percentage,
                        "file_format": "PNG",
                        "color_depth": "8",
                        "existing_file_policy": "ERROR",
                    },
                    "continue_on_error": False,
                    "include_review_packet": include_review_packet,
                    "camera_names": [PILOT_CAMERA],
                },
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
        await asyncio.sleep(0.5)
    if batch.get("status") != "SUCCEEDED":
        raise RuntimeError(f"Render {label} did not succeed: {batch}")
    items = batch.get("items") or batch.get("results")
    if not isinstance(items, list) or len(items) != 1:
        raise RuntimeError(f"Render {label} returned invalid items: {batch}")
    result_id = items[0]["result_id"]
    fetched = await _call(
        session,
        "get_look_render_result",
        {
            "batch_id": batch_id,
            "result_id": result_id,
            "max_size": 1600,
            "format": "JPEG",
            "jpeg_quality": 92,
        },
    )
    metadata = _json_value(fetched)
    images = [item for item in fetched.content if isinstance(item, ImageContent)]
    expected_images = 2 if include_review_packet else 1
    if len(images) != expected_images:
        raise RuntimeError(
            f"Render {label} returned {len(images)} images, expected {expected_images}"
        )
    native_images: list[dict[str, Any]] = []
    for index, image in enumerate(images):
        kind = "beauty" if index == 0 else "diagnostics"
        suffix = ".jpg" if image.mimeType == "image/jpeg" else ".png"
        path = artifact_dir / "native" / f"{label}-{kind}{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = base64.b64decode(image.data, validate=True)
        path.write_bytes(data)
        native_images.append(
            {
                "kind": kind,
                "path": str(path),
                "mime_type": image.mimeType,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return {
        "label": label,
        "batch_id": batch_id,
        "result_id": result_id,
        "metadata": metadata,
        "native_images": native_images,
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.expanduser().resolve()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    expected_blend = args.expected_blend.expanduser().resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if not expected_blend.is_file():
        raise RuntimeError(f"Expected isolated blend does not exist: {expected_blend}")
    source_sha256_before = _sha256(expected_blend)
    server = StdioServerParameters(
        command=str(repo / "scripts/run_blend_ai_with_secrets.sh"),
        args=[],
        cwd=repo,
    )
    paid_calls: list[dict[str, Any]] = []
    paid_provider_call_count = 0
    async with stdio_client(server) as streams:
        async with ClientSession(
            *streams, read_timeout_seconds=timedelta(seconds=args.timeout)
        ) as session:
            await session.initialize()
            tools = {item.name for item in (await session.list_tools()).tools}
            missing = sorted(REQUIRED_TOOLS - tools)
            if missing:
                raise RuntimeError(f"Pilot MCP tool list is missing {missing}")
            tool_schema = next(
                item.inputSchema
                for item in (await session.list_tools()).tools
                if item.name == "review_look_render"
            )
            mode_values = tool_schema["properties"]["mode"].get("enum", [])
            if "REALISM" not in mode_values:
                raise RuntimeError("Fresh MCP schema does not expose REALISM mode")

            initial_profile = _profile(hard_negative=False)
            negative_profile = _profile(hard_negative=True)
            if args.resume_runtime:
                context = _json_value(
                    await _call(
                        session,
                        "get_look_profile_context",
                        {"source_scene": PILOT_SOURCE, "detail": "FULL"},
                    )
                )
                entries = {item["profile_id"]: item for item in context["profiles"]}
                try:
                    initial_entry = entries[initial_profile["profile_id"]]
                    negative_entry = entries[negative_profile["profile_id"]]
                except KeyError as exc:
                    raise RuntimeError(
                        "Resume requires both compiled calibration profiles"
                    ) from exc
                geometry_revision = initial_entry["geometry_revision"]
                derived = {
                    "resumed": True,
                    "derived_scene": PILOT_SOURCE,
                    "derived_camera": PILOT_CAMERA,
                    "camera_matrix_identical": True,
                }
                initial_audit = _json_value(
                    await _call(
                        session,
                        "inspect_look_review_state",
                        {
                            "profile_id": initial_profile["profile_id"],
                            "target_scene": initial_entry["scene_name"],
                            "camera": PILOT_CAMERA,
                            "expected_profile_revision": initial_entry["revision"],
                            "expected_geometry_revision": geometry_revision,
                        },
                    )
                )
                negative_audit = _json_value(
                    await _call(
                        session,
                        "inspect_look_review_state",
                        {
                            "profile_id": negative_profile["profile_id"],
                            "target_scene": negative_entry["scene_name"],
                            "camera": PILOT_CAMERA,
                            "expected_profile_revision": negative_entry["revision"],
                            "expected_geometry_revision": geometry_revision,
                        },
                    )
                )
                if initial_audit.get("status") == "FAIL" or negative_audit.get("status") == "FAIL":
                    raise RuntimeError("Resume technical audit failed")
                initial_smoke = {
                    "reused": True,
                    "path": str(artifact_dir / "native" / "initial-smoke-16s-25pct-beauty.jpg"),
                }
                negative_smoke = {
                    "reused": True,
                    "path": str(
                        artifact_dir / "native" / "finish-negative-smoke-16s-25pct-beauty.jpg"
                    ),
                }
            else:
                derived = await _prepare_derived_source(session, expected_blend)
                context = _json_value(
                    await _call(
                        session,
                        "get_look_profile_context",
                        {"source_scene": PILOT_SOURCE, "detail": "FULL"},
                    )
                )
                base_revision = context["base_revision"]
                geometry_revision = context["geometry_revision"]
                initial_entry, initial_audit = await _compile(
                    session,
                    profile=initial_profile,
                    base_revision=base_revision,
                    geometry_revision=geometry_revision,
                )
                negative_entry, negative_audit = await _compile(
                    session,
                    profile=negative_profile,
                    base_revision=base_revision,
                    geometry_revision=geometry_revision,
                )
                initial_smoke = await _render(
                    session,
                    artifact_dir=artifact_dir,
                    label="initial-smoke-16s-25pct",
                    profile_id=initial_profile["profile_id"],
                    target_scene=initial_entry["scene_name"],
                    profile_revision=initial_entry["revision"],
                    samples=16,
                    denoise=False,
                    resolution_percentage=25,
                    include_review_packet=False,
                    timeout=args.timeout,
                )
                negative_smoke = await _render(
                    session,
                    artifact_dir=artifact_dir,
                    label="finish-negative-smoke-16s-25pct",
                    profile_id=negative_profile["profile_id"],
                    target_scene=negative_entry["scene_name"],
                    profile_revision=negative_entry["revision"],
                    samples=16,
                    denoise=False,
                    resolution_percentage=25,
                    include_review_packet=False,
                    timeout=args.timeout,
                )
            evidence_prefix = "resume-" if args.resume_runtime else ""
            initial_evidence = await _render(
                session,
                artifact_dir=artifact_dir,
                label=f"{evidence_prefix}initial-evidence-64s-full-denoised",
                profile_id=initial_profile["profile_id"],
                target_scene=initial_entry["scene_name"],
                profile_revision=initial_entry["revision"],
                samples=64,
                denoise=True,
                resolution_percentage=100,
                include_review_packet=True,
                timeout=args.timeout,
            )
            negative_evidence = await _render(
                session,
                artifact_dir=artifact_dir,
                label=(f"{evidence_prefix}finish-negative-evidence-64s-full-denoised"),
                profile_id=negative_profile["profile_id"],
                target_scene=negative_entry["scene_name"],
                profile_revision=negative_entry["revision"],
                samples=64,
                denoise=True,
                resolution_percentage=100,
                include_review_packet=True,
                timeout=args.timeout,
            )

            for candidate, evidence in (
                ("initial", initial_evidence),
                ("finish-negative", negative_evidence),
            ):
                for version in PROMPT_VERSIONS:
                    if paid_provider_call_count + 2 > MAX_PAID_PROVIDER_CALLS:
                        raise RuntimeError("Paid provider call cap reached")
                    review = _json_value(
                        await _call(
                            session,
                            "review_look_render",
                            {
                                "mode": "REALISM",
                                "batch_id": evidence["batch_id"],
                                "result_id": evidence["result_id"],
                                "model": MODEL,
                                "realism_prompt_version": version,
                            },
                        )
                    )
                    if review.get("input_image_count") != 1:
                        raise RuntimeError(f"REALISM isolation receipt failed: {review}")
                    provider_call_count = review.get("provider_call_count")
                    if provider_call_count != 2:
                        raise RuntimeError(f"REALISM expected two provider stages: {review}")
                    provider_ordinals = list(
                        range(
                            paid_provider_call_count + 1,
                            paid_provider_call_count + provider_call_count + 1,
                        )
                    )
                    paid_provider_call_count += provider_call_count
                    paid_calls.append(
                        {
                            "review_ordinal": len(paid_calls) + 1,
                            "provider_ordinals": provider_ordinals,
                            "candidate": candidate,
                            "mode": "REALISM",
                            "prompt_version": version,
                            "model": MODEL,
                            "batch_id": evidence["batch_id"],
                            "result_id": evidence["result_id"],
                            "receipt": review,
                        }
                    )

            human_dir = artifact_dir / "human"
            round1_dir = human_dir / "round1"
            round2_dir = human_dir / "round2"
            sealed_dir = artifact_dir / "sealed"
            for path in (round1_dir, round2_dir, sealed_dir):
                path.mkdir(parents=True, exist_ok=True)
            initial_beauty = Path(initial_evidence["native_images"][0]["path"])
            negative_beauty = Path(negative_evidence["native_images"][0]["path"])
            round1_path = round1_dir / "initial.jpg"
            shutil.copyfile(initial_beauty, round1_path)
            mapping = (
                {"A": initial_beauty, "B": negative_beauty}
                if secrets.randbelow(2) == 0
                else {"A": negative_beauty, "B": initial_beauty}
            )
            public_pair: dict[str, Any] = {}
            sealed_mapping: dict[str, Any] = {}
            for label, source in mapping.items():
                destination = round2_dir / f"{label}.jpg"
                shutil.copyfile(source, destination)
                public_pair[label] = {
                    "path": str(destination),
                    "sha256": _sha256(destination),
                }
                sealed_mapping[label] = {
                    "source_path": str(source),
                    "identity": ("initial" if source == initial_beauty else "finish-negative"),
                    "sha256": _sha256(source),
                }
            (sealed_dir / "round2-mapping.json").write_text(
                json.dumps(sealed_mapping, indent=2, sort_keys=True), encoding="utf-8"
            )

            file_state = _json_value(
                await _call(
                    session,
                    "execute_blender_code",
                    {
                        "code": (
                            "import bpy, json\n"
                            "print(json.dumps({'filepath': bpy.data.filepath, "
                            "'is_dirty': bpy.data.is_dirty}))"
                        )
                    },
                )
            )
            source_sha256_after = _sha256(expected_blend)
            if source_sha256_after != source_sha256_before:
                raise RuntimeError("Saved source blend changed during the pilot")
            reference_packet = json.loads(
                (artifact_dir / "reference-packet.json").read_text(encoding="utf-8")
            )
            return {
                "status": "AWAITING_HUMAN_ROUNDS_1_AND_2",
                "asset_base_id": ASSET_ID,
                "source_scene": SOURCE_SCENE,
                "source_camera": SOURCE_CAMERA,
                "derived_runtime": derived,
                "source_blend_sha256_before": source_sha256_before,
                "source_blend_sha256_after": source_sha256_after,
                "source_blend_unchanged": True,
                "runtime_file_state": file_state,
                "reference_packet_sha256": _canonical_sha256(reference_packet),
                "profiles": {
                    "initial": initial_profile,
                    "finish_negative": negative_profile,
                },
                "compiled": {
                    "initial": initial_entry,
                    "finish_negative": negative_entry,
                },
                "technical_audits": {
                    "initial": initial_audit,
                    "finish_negative": negative_audit,
                },
                "renders": {
                    "initial_smoke": initial_smoke,
                    "finish_negative_smoke": negative_smoke,
                    "initial_evidence": initial_evidence,
                    "finish_negative_evidence": negative_evidence,
                },
                "paid_provider_calls": paid_calls,
                "paid_review_count": len(paid_calls),
                "paid_call_count": paid_provider_call_count,
                "paid_call_cap": MAX_PAID_PROVIDER_CALLS,
                "human_round1": {
                    "path": str(round1_path),
                    "sha256": _sha256(round1_path),
                },
                "human_round2": public_pair,
                "arm_identity_sealed": True,
                "profiles_remain_draft": True,
                "profile_accepted": False,
                "saved_blend": False,
            }


def main() -> None:
    args = _arguments()
    report = asyncio.run(_run(args))
    artifact_dir = args.artifact_dir.expanduser().resolve()
    (artifact_dir / "provider-ledger.json").write_text(
        json.dumps(report["paid_provider_calls"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (artifact_dir / "calibration-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "paid_call_count": report["paid_call_count"],
                "human_round1": report["human_round1"],
                "human_round2": report["human_round2"],
                "source_blend_unchanged": report["source_blend_unchanged"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
