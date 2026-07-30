"""Run blend-ai MCP with a deterministic in-process OpenRouter test double."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import httpx
from mcp.types import ImageContent

from blend_ai import server
from blend_ai.tools import look_review


def _install_frozen_review_packet() -> None:
    """Optionally serve one frozen beauty without requiring a live Blender render."""
    candidate_value = os.environ.get("BLEND_AI_FROZEN_REVIEW_CANDIDATE")
    if candidate_value is None:
        return
    candidate_path = Path(candidate_value).expanduser().resolve()
    source_sha256 = os.environ.get("BLEND_AI_FROZEN_REVIEW_SOURCE_SHA256")
    if not candidate_path.is_file():
        raise RuntimeError("Frozen review candidate does not exist")
    if source_sha256 is None:
        raise RuntimeError("Frozen review source SHA-256 is missing")
    candidate_data = candidate_path.read_bytes()
    candidate = ImageContent(
        type="image",
        data=base64.b64encode(candidate_data).decode("ascii"),
        mimeType="image/jpeg",
    )
    unused_diagnostic = ImageContent(
        type="image",
        data=base64.b64encode(b"unused-diagnostic").decode("ascii"),
        mimeType="image/png",
    )
    metadata = {
        "status": "SUCCEEDED",
        "profile_id": "frozen-native-reference-smoke",
        "source_scene": "frozen-round-1",
        "camera": "authored-camera",
        "frame": 18,
        "artifact_sha256": source_sha256,
        "review_packet": {
            "technical_check": {"status": "PASS", "failures": [], "warnings": []},
            "review_intent": {"summary": "Frozen native REFERENCE transport smoke"},
            "tiles": [],
            "editable_controls": {},
        },
    }

    def frozen_result_packet(*_args: object, **_kwargs: object):
        return metadata, [candidate, unused_diagnostic]

    look_review._result_packet = frozen_result_packet


def _mock_post(payload: dict[str, object], api_key: str) -> httpx.Response:
    if api_key != "native-mcp-smoke-only":
        raise RuntimeError("Visible smoke received an unexpected test credential")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RuntimeError("Visible smoke received an invalid review payload")
    content = messages[0].get("content") if isinstance(messages[0], dict) else None
    image_count = sum(
        isinstance(item, dict) and item.get("type") == "image_url" for item in content or []
    )
    response_format = payload.get("response_format")
    json_schema = response_format.get("json_schema") if isinstance(response_format, dict) else None
    schema_name = json_schema.get("name") if isinstance(json_schema, dict) else None
    if schema_name == "blind_realism_assessment" and image_count == 1:
        result = {
            "classification": "CG",
            "calibrated_confidence": 0.84,
            "summary": "The image has a plausible composition but visibly uniform contact response.",
            "real_cues": [
                {
                    "region": "main lit surfaces",
                    "observation": "Highlight intensity varies across the frame.",
                }
            ],
            "cg_cues": [
                {
                    "region": "floor contact areas",
                    "observation": "Contact shadows repeat with unusually uniform density.",
                }
            ],
            "proposed_changes": [
                {
                    "region": "floor contact areas",
                    "direction": "Reduce broad fill and preserve localized contact contrast.",
                    "expected_effect": "Objects read as seated in the same photographed space.",
                    "category": "lighting",
                },
                {
                    "region": "rear wall",
                    "direction": "Replace the repeating surface texture.",
                    "expected_effect": "Reduce the visible asset repetition.",
                    "category": "material",
                },
            ],
        }
        request_id = "mock-native-mcp-realism-vision"
    elif schema_name == "filtered_realism_feedback" and image_count == 0:
        result = {
            "classification": "CG",
            "calibrated_confidence": 0.76,
            "summary": "The retained lighting evidence shows overly uniform contact response.",
            "real_cues": [
                {
                    "region": "main lit surfaces",
                    "observation": "Highlight intensity varies plausibly across the frame.",
                    "aspect": "highlight-rolloff",
                }
            ],
            "cg_cues": [
                {
                    "region": "floor contact areas",
                    "observation": "Contact shadows repeat with unusually uniform density.",
                    "aspect": "shadows",
                }
            ],
            "proposed_changes": [
                {
                    "region": "floor contact areas",
                    "direction": "Reduce broad fill and preserve localized contact contrast.",
                    "expected_effect": "The lighting gains more convincing depth.",
                    "category": "lighting",
                }
            ],
        }
        request_id = "mock-native-mcp-realism-rewrite"
    elif schema_name == "reference_conditioned_comparison" and image_count == 3:
        result = {
            "summary": "The candidate partly matches the reference lighting and surface context.",
            "assessments": [
                {
                    "reference_id": "REF-01",
                    "role": "LIGHTING_TOPOLOGY",
                    "status": "PARTIAL",
                    "reference_observation": "The reference uses a soft lateral key.",
                    "candidate_observation": "The candidate key is directional but more frontal.",
                    "candidate_region": "main lit surfaces",
                    "delta": "Shift the light direction laterally while retaining soft fill.",
                    "confidence": 0.82,
                },
                {
                    "reference_id": "REF-02",
                    "role": "MATERIAL_AND_SURFACE",
                    "status": "PARTIAL",
                    "reference_observation": "The reference surfaces vary in roughness and wear.",
                    "candidate_observation": "The rear wall has visibly uniform surface response.",
                    "candidate_region": "rear wall",
                    "delta": "The candidate has less visible surface variation.",
                    "confidence": 0.75,
                },
            ],
            "conflicts": [],
            "directions": [
                {
                    "supported_by": ["REF-01"],
                    "candidate_region": "main lit surfaces",
                    "direction": "Move the key toward a softer lateral direction.",
                    "expected_effect": "The form reads with clearer directional depth.",
                    "category": "lighting",
                },
                {
                    "supported_by": ["REF-02"],
                    "candidate_region": "rear wall",
                    "direction": "Increase visible roughness variation.",
                    "expected_effect": "The wall carries more tactile variation.",
                    "category": "material",
                },
            ],
        }
        request_id = "mock-native-mcp-reference-vision"
    elif schema_name == "reference_decision_handoff" and image_count == 0:
        result = {
            "summary": "Use the lateral-key note in the look workflow and retain the surface note as deferred context.",
            "look_directions": [
                {
                    "supported_by": ["REF-01"],
                    "candidate_region": "main lit surfaces",
                    "direction": "Move the key toward a softer lateral direction.",
                    "expected_effect": "The form reads with clearer directional depth.",
                    "category": "lighting",
                }
            ],
            "deferred_context": [
                {
                    "supported_by": ["REF-02"],
                    "candidate_region": "rear wall",
                    "observation": "The surface response is visibly more uniform than the reference.",
                    "category": "material",
                }
            ],
            "conflicts": [],
        }
        request_id = "mock-native-mcp-reference-rewrite"
    elif schema_name == "cinematic_critique" and image_count == 2:
        result = {
            "verdict": "revise",
            "summary": "The evidence pair is valid; lower the managed key slightly.",
            "findings": [
                {
                    "category": "highlight_balance",
                    "region": "main lit surfaces",
                    "evidence": ["beauty", "diffuse_direct"],
                    "observation": "The managed key dominates the value hierarchy.",
                    "hypothesis": "A small energy reduction should preserve direction with more headroom.",
                    "confidence": 0.9,
                    "action": {
                        "path": "lighting.lights.key.energy",
                        "operation": "SET",
                        "value": 560.0,
                        "expected_result": "Retain the warm key while recovering highlight separation.",
                    },
                }
            ],
        }
        request_id = "mock-native-mcp-review"
    elif schema_name == "cinematic_comparison" and image_count == 4:
        result = {
            "winner": "after",
            "regressions": [],
            "remaining_blocker": None,
            "confidence": 0.91,
            "accept": True,
        }
        request_id = "mock-native-mcp-compare"
    else:
        raise RuntimeError(
            f"Visible review received unsupported schema/image combination {schema_name!r}/{image_count}"
        )
    body = {
        "id": request_id,
        "choices": [{"message": {"content": json.dumps(result)}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    return httpx.Response(
        200,
        json=body,
        headers={"x-request-id": request_id},
        request=httpx.Request("POST", look_review.OPENROUTER_URL),
    )


def main() -> None:
    os.environ["OPENROUTER_API_KEY"] = "native-mcp-smoke-only"
    _install_frozen_review_packet()
    look_review._post_openrouter = _mock_post
    server.mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
