"""Run the two-stage REFERENCE provider pipeline on checksum-fixed local images."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

from mcp.types import ImageContent

from blend_ai.tools import look_review


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--candidate-source-sha256", required=True)
    parser.add_argument("--reference-packet", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    candidate_path = args.candidate.expanduser().resolve()
    candidate = candidate_path.read_bytes()
    packet = json.loads(args.reference_packet.expanduser().resolve().read_text(encoding="utf-8"))
    candidate_image = ImageContent(
        type="image",
        data=base64.b64encode(candidate).decode("ascii"),
        mimeType="image/jpeg",
    )
    diagnostic_image = ImageContent(
        type="image",
        data=base64.b64encode(b"unused-diagnostic").decode("ascii"),
        mimeType="image/png",
    )
    metadata = {
        "status": "SUCCEEDED",
        "profile_id": "reference-probe",
        "source_scene": "checksum-bound-probe",
        "camera": "authored-camera",
        "frame": 18,
        "artifact_sha256": args.candidate_source_sha256,
        "review_packet": {
            "technical_check": {"status": "PASS", "failures": [], "warnings": []},
            "review_intent": {"summary": packet["brief"]},
            "tiles": [],
            "editable_controls": {},
        },
    }

    def packet_result(*_args, **_kwargs):
        return metadata, [candidate_image, diagnostic_image]

    look_review._result_packet = packet_result
    receipt = look_review.review_look_render(
        "REFERENCE",
        "reference-probe-batch",
        "reference-probe-result",
        reference_packet=packet,
    )
    if receipt["input_proxy_sha256"] != hashlib.sha256(candidate).hexdigest():
        raise RuntimeError("Candidate proxy hash changed during the probe")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "SUCCEEDED",
                "output": str(output),
                "request_id": receipt["request_id"],
                "rewrite_request_id": receipt["rewrite"]["request_id"],
                "provider_call_count": receipt["provider_call_count"],
                "holistic_assessment_count": len(receipt["holistic_result"]["assessments"]),
                "look_direction_count": len(receipt["result"]["look_directions"]),
                "deferred_context_count": len(receipt["result"]["deferred_context"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
