"""Send one explicit first-pass REALISM payload for provider/schema diagnosis."""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

from blend_ai.tools.look_review import (
    DEFAULT_MODEL,
    RealismResponse,
    _post_openrouter,
    _provider_json_schema,
    _realism_prompt,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument(
        "--prompt-version",
        choices=("literal-v1", "evidence-v1", "causal-v1", "look-only-v1"),
        default="literal-v1",
    )
    args = parser.parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")
    image = args.image.expanduser().resolve().read_bytes()
    response_model = RealismResponse
    payload = {
        "model": DEFAULT_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _realism_prompt(args.prompt_version)},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,"
                            + base64.b64encode(image).decode("ascii")
                        },
                    },
                ],
            }
        ],
        "stream": False,
        "max_tokens": 1800,
        "provider": {"require_parameters": True},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "blind_realism_assessment",
                "strict": True,
                "schema": _provider_json_schema(response_model),
            },
        },
    }
    response = _post_openrouter(payload, api_key)
    record: dict[str, object] = {
        "http_status": response.status_code,
        "request_id": response.headers.get("x-request-id"),
        "prompt_version": args.prompt_version,
        "model": DEFAULT_MODEL,
        "image_count": 1,
    }
    try:
        body = response.json()
    except ValueError:
        body = {"non_json_body": response.text[:2000]}
    if response.is_error:
        record["error"] = body
        print(json.dumps(record, sort_keys=True))
        raise SystemExit(1)
    content = body["choices"][0]["message"]["content"]
    validated = response_model.model_validate(json.loads(content))
    record["usage"] = body.get("usage")
    record["result"] = validated.model_dump(mode="json")
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
