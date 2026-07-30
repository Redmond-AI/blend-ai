"""Exercise REFERENCE through native MCP using frozen local image evidence."""

from __future__ import annotations

import argparse
import asyncio
from datetime import timedelta
import json
import os
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, TextContent


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--candidate-source-sha256", required=True)
    parser.add_argument("--reference-packet", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
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


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.expanduser().resolve()
    candidate = args.candidate.expanduser().resolve()
    packet = json.loads(args.reference_packet.expanduser().resolve().read_text(encoding="utf-8"))
    child_env = os.environ.copy()
    child_env["BLEND_AI_FROZEN_REVIEW_CANDIDATE"] = str(candidate)
    child_env["BLEND_AI_FROZEN_REVIEW_SOURCE_SHA256"] = args.candidate_source_sha256
    server = StdioServerParameters(
        command=str(repo / ".venv/bin/python"),
        args=[str(repo / "tests/helpers/mock_review_mcp_server.py")],
        cwd=repo,
        env=child_env,
    )
    async with stdio_client(server) as streams:
        async with ClientSession(
            *streams, read_timeout_seconds=timedelta(seconds=args.timeout)
        ) as session:
            await session.initialize()
            result = await session.call_tool(
                "review_look_render",
                {
                    "mode": "REFERENCE",
                    "batch_id": "frozen-native-reference-smoke",
                    "result_id": "round-1",
                    "reference_packet": packet,
                },
            )
            if result.isError:
                messages = [item.text for item in result.content if isinstance(item, TextContent)]
                raise RuntimeError("review_look_render failed: " + "; ".join(messages))
            receipt = _json_value(result)
    expected_images = 1 + len(packet["references"])
    if receipt.get("input_image_count") != expected_images:
        raise RuntimeError("REFERENCE vision image boundary is invalid")
    if receipt.get("provider_call_count") != 2:
        raise RuntimeError("REFERENCE provider call count is invalid")
    if receipt.get("rewrite", {}).get("input_image_count") != 0:
        raise RuntimeError("REFERENCE rewrite received an image")
    if not receipt.get("holistic_result", {}).get("assessments"):
        raise RuntimeError("REFERENCE holistic result is missing")
    return receipt


def main() -> None:
    args = _arguments()
    receipt = asyncio.run(_run(args))
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output),
                "request_id": receipt["request_id"],
                "rewrite_request_id": receipt["rewrite"]["request_id"],
                "input_image_count": receipt["input_image_count"],
                "rewrite_input_image_count": receipt["rewrite"]["input_image_count"],
                "assessment_count": len(receipt["holistic_result"]["assessments"]),
                "look_direction_count": len(receipt["result"]["look_directions"]),
                "deferred_context_count": len(receipt["result"]["deferred_context"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
