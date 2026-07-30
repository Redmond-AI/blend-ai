"""Run one checksum-bound REFERENCE review through the native MCP tool."""

from __future__ import annotations

import argparse
import asyncio
from datetime import timedelta
import json
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, TextContent


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--result-id", required=True)
    parser.add_argument("--reference-packet", required=True, type=Path)
    parser.add_argument("--expected-blend", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=300.0)
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


def _schema_enum_values(value: Any) -> set[str]:
    if isinstance(value, list):
        return set().union(*(_schema_enum_values(item) for item in value))
    if not isinstance(value, dict):
        return set()
    found = {item for item in value.get("enum", []) if isinstance(item, str)}
    for item in value.values():
        found.update(_schema_enum_values(item))
    return found


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.expanduser().resolve()
    packet_path = args.reference_packet.expanduser().resolve()
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    server = StdioServerParameters(
        command=str(repo / "scripts/run_blend_ai_with_secrets.sh"),
        args=[],
        cwd=repo,
    )
    async with stdio_client(server) as streams:
        async with ClientSession(
            *streams, read_timeout_seconds=timedelta(seconds=args.timeout)
        ) as session:
            await session.initialize()
            tools = {item.name: item for item in (await session.list_tools()).tools}
            review_tool = tools.get("review_look_render")
            if review_tool is None:
                raise RuntimeError("Fresh MCP server does not expose review_look_render")
            modes = _schema_enum_values(review_tool.inputSchema["properties"]["mode"])
            if "REFERENCE" not in modes:
                raise RuntimeError("Fresh MCP schema does not expose REFERENCE mode")
            state_result = await session.call_tool(
                "execute_blender_code",
                {
                    "code": (
                        "import bpy, json\n"
                        "print(json.dumps({'filepath': bpy.data.filepath, "
                        "'is_dirty': bpy.data.is_dirty}))"
                    )
                },
            )
            if state_result.isError:
                raise RuntimeError("Could not verify the live Blender file state")
            state = _json_value(state_result)
            if not isinstance(state, dict) or not state.get("success"):
                raise RuntimeError(f"Invalid Blender file-state response: {state}")
            file_state = json.loads(state["output"])
            expected_blend = str(args.expected_blend.expanduser().resolve())
            if file_state != {"filepath": expected_blend, "is_dirty": False}:
                raise RuntimeError(f"Unsafe Blender file state: {file_state}")
            result = await session.call_tool(
                "review_look_render",
                {
                    "mode": "REFERENCE",
                    "batch_id": args.batch_id,
                    "result_id": args.result_id,
                    "reference_packet": packet,
                },
            )
            if result.isError:
                messages = [item.text for item in result.content if isinstance(item, TextContent)]
                raise RuntimeError("review_look_render failed: " + "; ".join(messages))
            receipt = _json_value(result)
            expected_count = 1 + len(packet["references"])
            if receipt.get("input_image_count") != expected_count:
                raise RuntimeError("REFERENCE review violated image-count isolation")
            if receipt.get("reference_image_count") != len(packet["references"]):
                raise RuntimeError("REFERENCE review returned an invalid reference receipt")
            if receipt.get("provider_call_count") != 2:
                raise RuntimeError("REFERENCE review did not preserve both provider stages")
            if receipt.get("rewrite", {}).get("input_image_count") != 0:
                raise RuntimeError("REFERENCE rewrite unexpectedly received an image")
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
                "status": "SUCCEEDED",
                "output": str(output),
                "request_id": receipt["request_id"],
                "assessment_count": len(receipt["holistic_result"]["assessments"]),
                "direction_count": len(receipt["result"]["look_directions"]),
                "deferred_context_count": len(receipt["result"]["deferred_context"]),
                "usage": receipt.get("usage"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
