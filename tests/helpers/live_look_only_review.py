"""Run one receipt-bound constrained REALISM call through the native MCP tool."""

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


PROMPT_VERSIONS = ("literal-v1", "evidence-v1", "causal-v1", "look-only-v1")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--result-id", required=True)
    parser.add_argument("--prompt-version", required=True, choices=PROMPT_VERSIONS)
    parser.add_argument("--expected-blend", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=240.0)
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
            versions = _schema_enum_values(
                review_tool.inputSchema["properties"]["realism_prompt_version"]
            )
            if args.prompt_version not in versions:
                raise RuntimeError(f"Fresh MCP schema does not expose {args.prompt_version}")
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
                    "mode": "REALISM",
                    "batch_id": args.batch_id,
                    "result_id": args.result_id,
                    "realism_prompt_version": args.prompt_version,
                },
            )
            if result.isError:
                messages = [item.text for item in result.content if isinstance(item, TextContent)]
                raise RuntimeError("review_look_render failed: " + "; ".join(messages))
            receipt = _json_value(result)
            if receipt.get("prompt_version") != args.prompt_version:
                raise RuntimeError("Unexpected prompt version in receipt")
            if receipt.get("input_image_count") != 1:
                raise RuntimeError("Constrained REALISM review violated one-image isolation")
            forbidden = {"material", "geometry-asset"}
            changes = receipt.get("result", {}).get("proposed_changes", [])
            returned = {change.get("category") for change in changes}
            if forbidden & returned:
                raise RuntimeError("Constrained REALISM review returned an artist-scene category")
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
                "classification": receipt["result"]["classification"],
                "confidence": receipt["result"]["calibrated_confidence"],
                "usage": receipt.get("usage"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
