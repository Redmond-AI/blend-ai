"""Focused hosted cinematic-review boundary tests."""

from __future__ import annotations

import base64
import hashlib
import json
from unittest.mock import patch

import httpx
from mcp.types import ImageContent
from PIL import Image as PILImage
from pydantic import ValidationError
import pytest

from blend_ai.tools import look_review


def _packet(*, status="PASS", camera="Camera A", frame=1):
    metadata = {
        "status": "SUCCEEDED",
        "profile_id": "night",
        "source_scene": "Base Scene",
        "camera": camera,
        "frame": frame,
        "artifact_sha256": "ab" * 32,
        "review_packet": {
            "review_intent": {"summary": "Cold moonlight"},
            "technical_check": {"status": status, "failures": [], "warnings": []},
            "tiles": [],
            "editable_controls": {
                "world.strength": 0.4,
                "lighting.lights.moon.energy": 2.0,
            },
        },
    }
    image_data = base64.b64encode(b"image").decode("ascii")
    images = [
        ImageContent(type="image", data=image_data, mimeType="image/jpeg"),
        ImageContent(type="image", data=image_data, mimeType="image/png"),
    ]
    return metadata, images


def _response(payload):
    request = httpx.Request("POST", look_review.OPENROUTER_URL)
    return httpx.Response(
        200,
        request=request,
        headers={"x-request-id": "req-test"},
        json={
            "id": "generation-test",
            "choices": [{"message": {"content": json.dumps(payload)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        },
    )


def _raw_realism_assessment():
    return {
        "classification": "CG",
        "calibrated_confidence": 0.78,
        "summary": "The flat shadow fill and repeated wall texture look synthetic.",
        "real_cues": [
            {
                "region": "window highlights",
                "observation": "The brightest reflections have plausible variation.",
            }
        ],
        "cg_cues": [
            {
                "region": "floor contact area",
                "observation": "Contact shadows are too uniform across repeated objects.",
            },
            {
                "region": "left wall",
                "observation": "The material texture visibly repeats.",
            },
        ],
        "proposed_changes": [
            {
                "region": "floor contact area",
                "direction": "Vary the key-to-fill ratio and restore local contact contrast.",
                "expected_effect": "Objects feel seated in the photographed space.",
                "category": "lighting",
            },
            {
                "region": "left wall",
                "direction": "Replace the repeating texture.",
                "expected_effect": "The wall looks less procedural.",
                "category": "material",
            },
        ],
    }


def _filtered_realism_assessment():
    return {
        "classification": "CG",
        "calibrated_confidence": 0.68,
        "summary": "The retained lighting evidence shows overly uniform shadow fill.",
        "real_cues": [
            {
                "region": "window highlights",
                "observation": "Highlight rolloff varies plausibly across the frame.",
                "aspect": "highlight-rolloff",
            }
        ],
        "cg_cues": [
            {
                "region": "floor contact area",
                "observation": "Contact shadows have overly uniform density.",
                "aspect": "shadows",
            }
        ],
        "proposed_changes": [
            {
                "region": "floor contact area",
                "direction": "Vary the key-to-fill ratio and restore local contact contrast.",
                "expected_effect": "The lighting gains more convincing depth.",
                "category": "lighting",
            }
        ],
    }


def _reference_packet(tmp_path):
    references = []
    for index, (reference_id, role, color) in enumerate(
        (
            ("REF-01", "LIGHTING_TOPOLOGY", (220, 160, 80)),
            ("REF-02", "CAMERA_AND_FINISH", (40, 70, 110)),
        ),
        start=1,
    ):
        path = tmp_path / f"private-reference-{index}.jpg"
        PILImage.new("RGB", (32, 24), color=color).save(path, format="JPEG")
        references.append(
            {
                "reference_id": reference_id,
                "image_path": str(path),
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "roles": [role],
                "comparison_focus": f"Narrow focus for {role}",
            }
        )
    return {
        "packet_id": "reference-packet-v1",
        "primary_reference_id": "REF-01",
        "brief": "Warm motivated light with readable cool shadows and restrained optics.",
        "targets": ["coherent direction", "readable shadow floor"],
        "non_targets": ["architecture", "materials"],
        "references": references,
    }


def _reference_comparison():
    return {
        "summary": "The candidate partially matches the lighting reference and misses its finish.",
        "assessments": [
            {
                "reference_id": "REF-01",
                "role": "LIGHTING_TOPOLOGY",
                "status": "PARTIAL",
                "reference_observation": "Warm directional light falls off softly.",
                "candidate_observation": "The key is directional but the transition is abrupt.",
                "candidate_region": "platform under the canopy",
                "delta": "The candidate needs a softer transition and more visible bounce.",
                "confidence": 0.82,
            },
            {
                "reference_id": "REF-02",
                "role": "CAMERA_AND_FINISH",
                "status": "MISS",
                "reference_observation": "Focus falls off gently with restrained edge softness.",
                "candidate_observation": "Near and far detail remain uniformly sharp.",
                "candidate_region": "arch and background foliage",
                "delta": "The candidate lacks the reference's subtle focus separation.",
                "confidence": 0.76,
            },
        ],
        "conflicts": [],
        "directions": [
            {
                "supported_by": ["REF-01"],
                "candidate_region": "platform under the canopy",
                "direction": "Soften the key-to-shadow transition and retain low-level bounce.",
                "expected_effect": "The lighting follows the reference without copying its scene.",
                "category": "lighting",
            }
        ],
    }


def _reference_synthesis():
    return {
        "summary": "The actionable lighting gap is the abrupt canopy transition.",
        "look_directions": [
            {
                "supported_by": ["REF-01"],
                "candidate_region": "platform under the canopy",
                "direction": "Soften the key-to-shadow transition and retain low-level bounce.",
                "expected_effect": "The lighting follows the reference without copying its scene.",
                "category": "lighting",
            }
        ],
        "deferred_context": [
            {
                "supported_by": ["REF-02"],
                "candidate_region": "arch and background foliage",
                "observation": "The authored composition limits focus separation in this framing.",
                "category": "composition-staging",
            }
        ],
        "conflicts": [],
    }


def test_review_action_rejects_non_allowlisted_or_unbounded_values():
    with pytest.raises(ValidationError, match="allowlisted"):
        look_review.ReviewAction(
            path="materials.floor.roughness",
            value=0.2,
            expected_result="shinier floor",
        )
    with pytest.raises(ValidationError, match="between 0 and 1"):
        look_review.ReviewAction(
            path="lighting.lights.moon.color_rgb",
            value=[2.0, 0.2, 0.3],
            expected_result="cooler moon",
        )


def test_critique_finding_accepts_depth_evidence_and_rejects_removed_volume_tile():
    finding = look_review.CritiqueFinding(
        category="spatial_separation",
        region="foreground dock",
        evidence=["beauty", "depth"],
        observation="The foreground and cabin occupy distinct depth bands.",
        confidence=0.9,
    )

    assert finding.evidence == ["beauty", "depth"]
    with pytest.raises(ValidationError):
        look_review.CritiqueFinding(
            category="legacy_tile",
            region="scene",
            evidence=["volume"],
            observation="Legacy evidence should be rejected.",
            confidence=0.5,
        )


def test_critique_summary_allows_detailed_aov_evidence():
    response = look_review.CritiqueResponse(
        verdict="revise",
        summary="x" * 3000,
        findings=[
            look_review.CritiqueFinding(
                category="exposure",
                region="full frame",
                evidence=["beauty", "combined"],
                observation="Midtones are compressed.",
                confidence=0.9,
            )
        ],
    )

    assert len(response.summary) == 3000


@pytest.mark.parametrize("mode", ["REALISM", "CRITIQUE"])
def test_missing_key_fails_before_render_lookup(monkeypatch, mode):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with patch.object(look_review, "_result_packet") as packet:
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            look_review.review_look_render(mode, "batch-1", "result-1")
        packet.assert_not_called()


def test_critique_sends_exactly_two_images_and_strict_schema(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "concealed-test-value")
    critique = {
        "verdict": "revise",
        "summary": "The foreground needs a clearer moon reflection.",
        "findings": [
            {
                "category": "reflection_too_weak",
                "region": "lower center",
                "evidence": ["beauty", "glossy"],
                "observation": "The foreground reads matte.",
                "hypothesis": "The moon contribution is too low.",
                "confidence": 0.86,
                "action": {
                    "path": "lighting.lights.moon.energy",
                    "operation": "SET",
                    "value": 2.8,
                    "expected_result": "stronger foreground separation",
                },
            }
        ],
    }
    captured = {}

    def post(payload, api_key):
        captured["payload"] = payload
        captured["api_key_seen"] = bool(api_key)
        return _response(critique)

    with (
        patch.object(look_review, "_result_packet", return_value=_packet()),
        patch.object(look_review, "_post_openrouter", side_effect=post),
    ):
        result = look_review.review_look_render("CRITIQUE", "batch-1", "result-1")

    content = captured["payload"]["messages"][0]["content"]
    assert len([item for item in content if item["type"] == "image_url"]) == 2
    assert captured["payload"]["response_format"]["json_schema"]["strict"] is True
    assert captured["payload"]["provider"]["require_parameters"] is True
    assert captured["payload"]["max_tokens"] == 3200
    assert result["request_id"] == "req-test"
    assert result["result"]["findings"][0]["action"]["path"].endswith("energy")
    assert "api_key" not in result


def test_provider_schema_strips_only_unsupported_claude_constraints():
    local = look_review.RealismResponse.model_json_schema()
    provider = look_review._provider_json_schema(look_review.RealismResponse)
    assert "minimum" in json.dumps(local)
    serialized = json.dumps(provider)
    for keyword in look_review._UNSUPPORTED_PROVIDER_SCHEMA_CONSTRAINTS:
        assert f'"{keyword}"' not in serialized
    assert '"maxItems": 3' in json.dumps(local)
    assert provider["additionalProperties"] is False
    assert set(provider["required"]) == set(provider["properties"])


@pytest.mark.parametrize("prompt_version", look_review.REALISM_PROMPT_VERSIONS)
def test_realism_sends_one_1600px_beauty_without_provenance_and_returns_receipts(
    monkeypatch, prompt_version
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "concealed-test-value")
    raw_assessment = _raw_realism_assessment()
    filtered_assessment = _filtered_realism_assessment()
    captured = {"payloads": []}

    def packet(batch_id, result_id, **kwargs):
        captured["packet_args"] = (batch_id, result_id, kwargs)
        return _packet()

    def post(payload, _api_key):
        captured["payloads"].append(payload)
        return _response(raw_assessment if len(captured["payloads"]) == 1 else filtered_assessment)

    with (
        patch.object(look_review, "_result_packet", side_effect=packet),
        patch.object(look_review, "_post_openrouter", side_effect=post),
    ):
        result = look_review.review_look_render(
            "REALISM",
            "batch-1",
            "result-1",
            realism_prompt_version=prompt_version,
        )

    assert len(captured["payloads"]) == 2
    vision_payload, rewrite_payload = captured["payloads"]
    content = vision_payload["messages"][0]["content"]
    images = [item for item in content if item["type"] == "image_url"]
    prompt = content[0]["text"]
    assert len(images) == 1
    assert images[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert captured["packet_args"][2] == {"max_size": 1600, "jpeg_quality": 92}
    assert vision_payload["response_format"]["json_schema"]["name"] == "blind_realism_assessment"
    assert rewrite_payload["model"] == look_review.DEFAULT_REALISM_REWRITE_MODEL
    assert rewrite_payload["response_format"]["json_schema"]["name"] == (
        "filtered_realism_feedback"
    )
    assert isinstance(rewrite_payload["messages"][0]["content"], str)
    assert "image_url" not in json.dumps(rewrite_payload)
    assert "repeated wall texture" in rewrite_payload["messages"][0]["content"]
    assert look_review.REALISM_QUESTION in prompt
    for leaked in ("Cold moonlight", "Camera A", "Base Scene", "night", "Blender", "Cycles"):
        assert leaked not in prompt
    assert result["prompt_version"] == prompt_version
    assert result["prompt_sha256"] == look_review.hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    assert result["input_proxy_sha256"] == look_review.hashlib.sha256(b"image").hexdigest()
    assert result["source_artifact_sha256"] == "ab" * 32
    assert result["input_image_count"] == 1
    assert result["request_id"] == "req-test"
    assert result["usage"] == {"prompt_tokens": 100, "completion_tokens": 20}
    assert result["provider_call_count"] == 2
    assert result["rewrite"]["model"] == look_review.DEFAULT_REALISM_REWRITE_MODEL
    assert result["rewrite"]["input_image_count"] == 0
    assert result["result"]["proposed_changes"][0]["scope"] == "relighting"
    assert "material" not in json.dumps(result["result"])
    final_assessment = {
        **filtered_assessment,
        "proposed_changes": [{**filtered_assessment["proposed_changes"][0], "scope": "relighting"}],
    }
    canonical = json.dumps(final_assessment, sort_keys=True, separators=(",", ":"))
    assert (
        result["response_sha256"]
        == look_review.hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    )


def test_realism_prompt_variants_and_mode_specific_arguments_are_strict(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "concealed-test-value")
    for version in look_review.REALISM_PROMPT_VERSIONS:
        prompt = look_review._realism_prompt(version)
        assert look_review.REALISM_QUESTION in prompt
        assert "separate feedback editor handles relevance and scope" in prompt
    assert "not proof of realism" in look_review._realism_prompt("causal-v1")
    assert "direct literal strategy" in look_review._realism_prompt("literal-v1")
    assert "supporting both real and CG" in look_review._realism_prompt("evidence-v1")
    assert "causal analysis strategy" in look_review._realism_prompt("causal-v1")
    look_only = look_review._realism_prompt("look-only-v1")
    assert "actionable cinematic-finishing strategy" in look_only
    assert len(
        {look_review._realism_prompt(version) for version in look_review.REALISM_PROMPT_VERSIONS}
    ) == len(look_review.REALISM_PROMPT_VERSIONS)

    rewrite_prompt = look_review._realism_rewrite_prompt(_raw_realism_assessment())
    assert "Remove feedback about materials" in rewrite_prompt
    assert "without inventing new visible facts" in rewrite_prompt

    look_only_schema = look_review._provider_json_schema(look_review.LookOnlyRealismDraft)
    allowed_categories = look_only_schema["$defs"]["LookOnlyRealismDraftChange"]["properties"][
        "category"
    ]["enum"]
    assert "material" not in allowed_categories
    assert "geometry-asset" not in allowed_categories
    assert "scope" not in look_only_schema["$defs"]["LookOnlyRealismDraftChange"]["properties"]

    with patch.object(look_review, "_result_packet") as packet:
        with pytest.raises(ValueError, match="only in REALISM"):
            look_review.review_look_render(
                "CRITIQUE",
                "batch-1",
                "result-1",
                realism_prompt_version="literal-v1",
            )
        with pytest.raises(ValueError, match="only in REALISM"):
            look_review.review_look_render(
                "CRITIQUE",
                "batch-1",
                "result-1",
                realism_rewrite_model="google/gemini-3.6-flash",
            )
        with pytest.raises(ValueError, match="only in COMPARE"):
            look_review.review_look_render(
                "REALISM",
                "batch-1",
                "result-1",
                baseline_batch_id="batch-0",
                baseline_result_id="result-0",
            )
        with pytest.raises(ValueError, match="supported prompt version"):
            look_review.review_look_render(
                "REALISM",
                "batch-1",
                "result-1",
                realism_prompt_version="custom-v9",  # type: ignore[arg-type]
            )
        packet.assert_not_called()


def test_look_only_realism_uses_constrained_schema(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "concealed-test-value")
    captured = {"payloads": []}

    def post(payload, _api_key):
        captured["payloads"].append(payload)
        return _response(
            _raw_realism_assessment()
            if len(captured["payloads"]) == 1
            else _filtered_realism_assessment()
        )

    with (
        patch.object(look_review, "_result_packet", return_value=_packet()),
        patch.object(look_review, "_post_openrouter", side_effect=post),
    ):
        result = look_review.review_look_render(
            "REALISM",
            "batch-1",
            "result-1",
            realism_prompt_version="look-only-v1",
            realism_rewrite_model="google/gemini-2.5-flash",
        )

    schema = captured["payloads"][1]["response_format"]["json_schema"]
    assert schema["name"] == "filtered_realism_feedback"
    assert result["prompt_version"] == "look-only-v1"
    assert result["rewrite"]["model"] == "google/gemini-2.5-flash"
    assert result["result"]["cg_cues"][0]["aspect"] == "shadows"


def test_raw_realism_schema_accepts_artist_scene_feedback_without_scope():
    schema = look_review.RealismResponse.model_json_schema()
    assert set(schema["required"]) == set(schema["properties"])
    change = look_review.RealismChange(
        region="wall",
        direction="Break up roughness.",
        expected_effect="Less uniform response.",
        category="material",
    )
    assert change.category == "material"
    assert "scope" not in change.model_dump()
    with pytest.raises(ValidationError, match="localized CG cue"):
        look_review.RealismResponse(
            classification="CG",
            calibrated_confidence=0.7,
            summary="Looks synthetic.",
            real_cues=[],
            cg_cues=[],
            proposed_changes=[],
        )


def test_look_only_realism_schema_rejects_artist_scene_changes():
    with pytest.raises(ValidationError):
        look_review.LookOnlyRealismDraftChange(
            region="foliage",
            direction="Vary the leaf cards.",
            expected_effect="Reduce repetition.",
            category="geometry-asset",  # type: ignore[arg-type]
        )


def test_realism_validation_failure_preserves_stage_receipt_and_does_not_retry(
    monkeypatch,
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "concealed-test-value")
    invalid_rewrite = _filtered_realism_assessment()
    invalid_rewrite["proposed_changes"][0]["category"] = "material"
    responses = [_response(_raw_realism_assessment()), _response(invalid_rewrite)]
    with (
        patch.object(look_review, "_result_packet", return_value=_packet()),
        patch.object(look_review, "_post_openrouter", side_effect=responses) as post,
    ):
        with pytest.raises(RuntimeError) as caught:
            look_review.review_look_render("REALISM", "batch-1", "result-1")

    failure = json.loads(str(caught.value))
    assert failure["status"] == "INVALID_PROVIDER_RESPONSE"
    assert failure["stage"] == "rewrite"
    assert failure["request_id"] == "req-test"
    assert failure["usage"] == {"prompt_tokens": 100, "completion_tokens": 20}
    assert len(failure["response_sha256"]) == 64
    assert post.call_count == 2
    with pytest.raises(ValidationError, match="scope must be 'look-profile'"):
        look_review.LookOnlyRealismChange(
            region="frame edges",
            direction="Reduce the vignette.",
            expected_effect="Preserve edge detail.",
            category="camera-post",
            scope="relighting",
        )


def test_critique_validation_failure_preserves_exact_response_and_does_not_retry(
    monkeypatch,
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "concealed-test-value")
    invalid_critique = {
        "verdict": "accept",
        "summary": "The look is acceptable, but the practical should be reduced.",
        "findings": [
            {
                "category": "practical_falloff",
                "region": "left couch",
                "evidence": ["beauty", "emission"],
                "observation": "The practical extends beyond the immediate cushion.",
                "hypothesis": "The practical contribution is too broad.",
                "confidence": 0.82,
                "action": {
                    "path": "lighting.lights.moon.energy",
                    "operation": "SET",
                    "value": 1.8,
                    "expected_result": "Tighter warm practical separation.",
                },
            }
        ],
    }
    with (
        patch.object(look_review, "_result_packet", return_value=_packet()),
        patch.object(
            look_review,
            "_post_openrouter",
            return_value=_response(invalid_critique),
        ) as post,
    ):
        with pytest.raises(RuntimeError) as caught:
            look_review.review_look_render("CRITIQUE", "batch-1", "result-1")

    failure = json.loads(str(caught.value))
    assert failure["status"] == "INVALID_PROVIDER_RESPONSE"
    assert failure["stage"] == "critique"
    assert failure["request_id"] == "req-test"
    assert failure["usage"] == {"prompt_tokens": 100, "completion_tokens": 20}
    assert failure["response"] == invalid_critique
    assert failure["validation_errors"][0]["ctx"]["error"].startswith(
        "an accepted critique must not propose actions"
    )
    assert post.call_count == 1


def test_realism_requires_source_hash_before_provider_call(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "concealed-test-value")
    metadata, images = _packet()
    metadata.pop("artifact_sha256")
    with (
        patch.object(look_review, "_result_packet", return_value=(metadata, images)),
        patch.object(look_review, "_post_openrouter") as post,
    ):
        with pytest.raises(RuntimeError, match="source artifact SHA-256"):
            look_review.review_look_render("REALISM", "batch-1", "result-1")
        post.assert_not_called()


@pytest.mark.parametrize("mode", ["REALISM", "CRITIQUE"])
def test_technical_failure_blocks_provider_call(monkeypatch, mode):
    monkeypatch.setenv("OPENROUTER_API_KEY", "concealed-test-value")
    with (
        patch.object(look_review, "_result_packet", side_effect=RuntimeError("technical audit")),
        patch.object(look_review, "_post_openrouter") as post,
    ):
        with pytest.raises(RuntimeError, match="technical audit"):
            look_review.review_look_render(mode, "batch-1", "result-1")
        post.assert_not_called()


@pytest.mark.parametrize("mode", ["REALISM", "CRITIQUE"])
def test_invalid_json_http_failure_and_timeout_are_not_retried(monkeypatch, mode):
    monkeypatch.setenv("OPENROUTER_API_KEY", "concealed-test-value")
    request = httpx.Request("POST", look_review.OPENROUTER_URL)
    invalid = httpx.Response(
        200,
        request=request,
        json={"choices": [{"message": {"content": "not-json"}}]},
    )
    failed = httpx.Response(503, request=request, text="unavailable")

    with (
        patch.object(look_review, "_result_packet", return_value=_packet()),
        patch.object(look_review, "_post_openrouter", return_value=invalid) as post,
    ):
        with pytest.raises(RuntimeError, match="invalid structured JSON") as caught:
            look_review.review_look_render(mode, "batch-1", "result-1")
        assert post.call_count == 1
        failure = json.loads(str(caught.value))
        assert failure["status"] == "INVALID_PROVIDER_RESPONSE"
        assert failure["stage"] == ("vision" if mode == "REALISM" else "critique")
        assert failure["request_id"] == "unknown"
        assert len(failure["response_sha256"]) == 64
        assert failure["response_text"] == "not-json"

    with (
        patch.object(look_review, "_result_packet", return_value=_packet()),
        patch.object(look_review, "_post_openrouter", return_value=failed) as post,
    ):
        with pytest.raises(httpx.HTTPStatusError):
            look_review.review_look_render(mode, "batch-1", "result-1")
        assert post.call_count == 1

    with (
        patch.object(look_review, "_result_packet", return_value=_packet()),
        patch.object(
            look_review,
            "_post_openrouter",
            side_effect=httpx.ReadTimeout("timed out", request=request),
        ) as post,
    ):
        with pytest.raises(httpx.ReadTimeout):
            look_review.review_look_render(mode, "batch-1", "result-1")
        assert post.call_count == 1


def test_reference_uses_open_single_pass_without_aov_or_private_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "concealed-test-value")
    packet = _reference_packet(tmp_path)
    captured = {"payloads": []}

    def post(payload, _api_key):
        captured["payloads"].append(payload)
        return _response(
            {
                "feedback": (
                    "The candidate is too bright and evenly filled for the requested spooky "
                    "moonlit mood. Reduce the broad blue fill and preserve the small warm pool."
                )
            }
        )

    with (
        patch.object(look_review, "_result_packet", return_value=_packet()),
        patch.object(look_review, "_post_openrouter", side_effect=post),
    ):
        result = look_review.review_look_render(
            "REFERENCE",
            "batch-1",
            "result-1",
            reference_packet=packet,
        )

    assert len(captured["payloads"]) == 1
    payload = captured["payloads"][0]
    content = payload["messages"][0]["content"]
    images = [item for item in content if item["type"] == "image_url"]
    prompt = content[0]["text"]
    assert len(images) == 3
    assert [item["text"] for item in content if item["type"] == "text"][1:] == [
        "CANDIDATE IMAGE",
        "PRIMARY REFERENCE REF-01",
        "SECONDARY REFERENCE REF-02",
    ]
    assert payload["response_format"]["json_schema"]["name"] == ("open_reference_lighting_feedback")
    assert "The goal is for the candidate image to look like this:\nCold moonlight" in prompt
    assert "models and textures cannot be changed" in prompt
    assert "lighting and overall vibe" in prompt
    assert "role" not in prompt.lower()
    assert "private-reference" not in json.dumps(payload)
    assert str(tmp_path) not in json.dumps(payload)
    assert result["input_image_count"] == 3
    assert result["reference_image_count"] == 2
    assert result["provider_call_count"] == 1
    assert result["source_artifact_sha256"] == "ab" * 32
    assert result["result"]["feedback"].startswith("The candidate is too bright")
    assert "holistic_result" not in result
    assert "rewrite" not in result
    assert [item["reference_id"] for item in result["references"]] == ["REF-01", "REF-02"]
    assert result["primary_reference_id"] == "REF-01"
    assert [item["is_primary"] for item in result["references"]] == [True, False]
    assert all(len(item["proxy_sha256"]) == 64 for item in result["references"])


def test_reference_arguments_and_hashes_are_strict(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "concealed-test-value")
    packet = _reference_packet(tmp_path)
    with patch.object(look_review, "_result_packet") as result_packet:
        with pytest.raises(ValueError, match="requires reference_packet"):
            look_review.review_look_render("REFERENCE", "batch-1", "result-1")
        with pytest.raises(ValueError, match="only in REFERENCE"):
            look_review.review_look_render(
                "CRITIQUE", "batch-1", "result-1", reference_packet=packet
            )
        with pytest.raises(ValueError, match="no longer supported"):
            look_review.review_look_render(
                "CRITIQUE",
                "batch-1",
                "result-1",
                reference_rewrite_model="google/gemini-3.6-flash",
            )
        result_packet.assert_not_called()

    missing_primary = json.loads(json.dumps(packet))
    missing_primary.pop("primary_reference_id")
    with pytest.raises(ValidationError, match="primary_reference_id"):
        look_review.ReferenceReviewPacket.model_validate(missing_primary)

    unknown_primary = json.loads(json.dumps(packet))
    unknown_primary["primary_reference_id"] = "UNKNOWN"
    with pytest.raises(ValidationError, match="must identify one packet reference"):
        look_review.ReferenceReviewPacket.model_validate(unknown_primary)

    validated_packet = look_review.ReferenceReviewPacket.model_validate(packet)
    with pytest.raises(ValueError, match="rendered review_intent provenance"):
        look_review._reference_prompt(validated_packet, {})

    bad_hash = json.loads(json.dumps(packet))
    bad_hash["references"][0]["source_sha256"] = "00" * 32
    with (
        patch.object(look_review, "_result_packet", return_value=_packet()),
        patch.object(look_review, "_post_openrouter") as post,
    ):
        with pytest.raises(ValueError, match="source SHA-256 does not match"):
            look_review.review_look_render(
                "REFERENCE", "batch-1", "result-1", reference_packet=bad_hash
            )
        post.assert_not_called()


def test_reference_missing_key_and_technical_audit_block_before_provider(monkeypatch, tmp_path):
    packet = _reference_packet(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with patch.object(look_review, "_result_packet") as result_packet:
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            look_review.review_look_render(
                "REFERENCE", "batch-1", "result-1", reference_packet=packet
            )
        result_packet.assert_not_called()

    monkeypatch.setenv("OPENROUTER_API_KEY", "concealed-test-value")
    with (
        patch.object(look_review, "_result_packet", side_effect=RuntimeError("technical audit")),
        patch.object(look_review, "_post_openrouter") as post,
    ):
        with pytest.raises(RuntimeError, match="technical audit"):
            look_review.review_look_render(
                "REFERENCE", "batch-1", "result-1", reference_packet=packet
            )
        post.assert_not_called()


def test_compare_requires_matching_camera_and_sends_four_images(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "concealed-test-value")
    comparison = {
        "winner": "after",
        "regressions": [],
        "remaining_blocker": None,
        "confidence": 0.82,
        "accept": True,
    }
    packets = [_packet(), _packet()]
    captured = {}

    def post(payload, _api_key):
        captured["payload"] = payload
        return _response(comparison)

    with (
        patch.object(look_review, "_result_packet", side_effect=packets),
        patch.object(look_review, "_post_openrouter", side_effect=post),
    ):
        result = look_review.review_look_render(
            "COMPARE",
            "batch-after",
            "result-after",
            baseline_batch_id="batch-before",
            baseline_result_id="result-before",
        )

    content = captured["payload"]["messages"][0]["content"]
    assert len([item for item in content if item["type"] == "image_url"]) == 4
    assert result["result"]["winner"] == "after"

    mismatched = [_packet(), _packet(camera="Camera B")]
    with (
        patch.object(look_review, "_result_packet", side_effect=mismatched),
        patch.object(look_review, "_post_openrouter") as post,
    ):
        with pytest.raises(ValueError, match="camera"):
            look_review.review_look_render(
                "COMPARE",
                "batch-after",
                "result-after",
                baseline_batch_id="batch-before",
                baseline_result_id="result-before",
            )
        post.assert_not_called()


def test_compare_accepts_same_managed_camera_across_revision_hashes(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "concealed-test-value")
    comparison = {
        "winner": "after",
        "regressions": [],
        "remaining_blocker": None,
        "confidence": 0.82,
        "accept": True,
    }
    packets = [
        _packet(camera="AI_CAMERA_railway_moonlight_camera_2e85517e"),
        _packet(camera="AI_CAMERA_railway_moonlight_camera_deca01d5"),
    ]

    with (
        patch.object(look_review, "_result_packet", side_effect=packets),
        patch.object(look_review, "_post_openrouter", return_value=_response(comparison)),
    ):
        result = look_review.review_look_render(
            "COMPARE",
            "batch-after",
            "result-after",
            baseline_batch_id="batch-before",
            baseline_result_id="result-before",
        )

    assert result["result"]["winner"] == "after"
