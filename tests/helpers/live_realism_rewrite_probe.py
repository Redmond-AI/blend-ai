"""Probe the text-only REALISM rewrite model without spending a vision call."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from blend_ai.tools.look_review import (
    DEFAULT_REALISM_REWRITE_MODEL,
    LookOnlyRealismDraft,
    _post_openrouter,
    _provider_json_schema,
    _realism_rewrite_prompt,
    _response_content,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")
    source = json.loads(args.input.expanduser().resolve().read_text(encoding="utf-8"))
    prompt = _realism_rewrite_prompt(source)
    payload = {
        "model": DEFAULT_REALISM_REWRITE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": 1400,
        "provider": {"require_parameters": True},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "filtered_realism_feedback",
                "strict": True,
                "schema": _provider_json_schema(LookOnlyRealismDraft),
            },
        },
    }
    parsed, request_id, usage = _response_content(
        _post_openrouter(payload, api_key), stage="rewrite"
    )
    result = LookOnlyRealismDraft.model_validate(parsed).model_dump(mode="json")
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"))
    record = {
        "status": "SUCCEEDED",
        "model": DEFAULT_REALISM_REWRITE_MODEL,
        "request_id": request_id,
        "usage": usage,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "input_image_count": 0,
        "response_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "result": result,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: record[key] for key in ("status", "model", "request_id", "usage")}))


if __name__ == "__main__":
    main()
