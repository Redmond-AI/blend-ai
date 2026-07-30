"""Hosted cinematic critique for checksum-bound look-profile review packets."""

from __future__ import annotations

import hashlib
import base64
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Literal

import httpx
from mcp.types import CallToolResult, ImageContent
from PIL import Image as PILImage
from PIL import ImageOps, UnidentifiedImageError
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from blend_ai.server import mcp
from blend_ai.tools.look_profiles import get_look_render_result


DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"
DEFAULT_REALISM_REWRITE_MODEL = "openai/gpt-5.6-luna"
DEFAULT_REFERENCE_REWRITE_MODEL = "google/gemini-3.6-flash"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_FINDINGS = 3
MAX_REFERENCE_IMAGES = 4
MAX_REFERENCE_ASSESSMENTS = 16
REALISM_QUESTION = "Does this look CG or real? If CG, what would you change to make it look real?"
REALISM_PROMPT_VERSIONS = (
    "literal-v1",
    "evidence-v1",
    "causal-v1",
    "look-only-v1",
)
_UNSUPPORTED_PROVIDER_SCHEMA_CONSTRAINTS = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
    }
)
_SAFE_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_MANAGED_CAMERA_NAME = re.compile(r"^(AI_CAMERA_.+)_([0-9a-fA-F]{8})$")
_LIGHT_PATH = re.compile(
    r"^lighting\.lights\.[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\."
    r"(energy|color_rgb|target_point|size|size_y|radius|sun_angle_degrees)$"
)
_ATMOSPHERE_PATH = re.compile(r"^atmosphere\.[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\.density$")
_FIXED_PATHS = {
    "world.strength",
    "color_management.exposure",
    "post.bloom.threshold",
    "post.bloom.strength",
    "post.bloom.radius",
    "post.grain.strength",
    "post.grain.scale",
    "post.vignette.strength",
    "post.vignette.feather",
}


def _camera_comparison_identity(value: Any) -> Any:
    """Return the stable identity embedded in a managed compiled camera name."""
    if not isinstance(value, str):
        return value
    match = _MANAGED_CAMERA_NAME.fullmatch(value)
    return match.group(1) if match else value


def _bounded_id(value: str, field: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{field} is not a safe identifier")
    return value


class ReviewAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    operation: Literal["SET"] = "SET"
    value: float | list[float]
    expected_result: str = Field(min_length=1, max_length=500)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        if (
            value not in _FIXED_PATHS
            and not _LIGHT_PATH.fullmatch(value)
            and not _ATMOSPHERE_PATH.fullmatch(value)
        ):
            raise ValueError("action path is not an allowlisted look-profile control")
        return value

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: float | list[float]) -> float | list[float]:
        values = value if isinstance(value, list) else [value]
        if len(values) not in {1, 3}:
            raise ValueError("action value must be one number or a three-number vector")
        if any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(item)
            or abs(float(item)) > 1_000_000_000.0
            for item in values
        ):
            raise ValueError("action values must be finite and bounded")
        return [float(item) for item in value] if isinstance(value, list) else float(value)

    @model_validator(mode="after")
    def validate_shape_and_range(self) -> ReviewAction:
        vector = self.path.endswith((".color_rgb", ".target_point"))
        if vector != isinstance(self.value, list):
            raise ValueError("color_rgb and target_point require vectors; other paths are scalar")
        values = self.value if isinstance(self.value, list) else [self.value]
        if self.path.endswith(".color_rgb") and any(not 0.0 <= item <= 1.0 for item in values):
            raise ValueError("color_rgb components must be between 0 and 1")
        scalar_limits = {
            "world.strength": (0.0, 1000.0),
            "color_management.exposure": (-32.0, 32.0),
            "post.bloom.threshold": (0.0, 1000.0),
            "post.bloom.strength": (0.0, 100.0),
            "post.bloom.radius": (0.0, 1.0),
            "post.grain.strength": (0.0, 1.0),
            "post.grain.scale": (0.01, 1000.0),
            "post.vignette.strength": (0.0, 1.0),
            "post.vignette.feather": (0.0, 1.0),
        }
        if not isinstance(self.value, list):
            minimum, maximum = scalar_limits.get(self.path, (0.0, 1_000_000_000.0))
            if self.path.endswith(".density"):
                minimum, maximum = (0.0, 1000.0)
            elif self.path.endswith(".sun_angle_degrees"):
                minimum, maximum = (0.0, 180.0)
            elif self.path.endswith((".size", ".size_y", ".radius")):
                minimum, maximum = (0.0, 1_000_000.0)
            if not minimum <= self.value <= maximum:
                raise ValueError(f"action value for {self.path} is outside its safe range")
        return self


class CritiqueFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str = Field(min_length=1, max_length=128)
    region: str = Field(min_length=1, max_length=256)
    evidence: list[
        Literal[
            "beauty",
            "combined",
            "diffuse_direct",
            "glossy",
            "emission",
            "depth",
            "cryptomatte_object",
            "technical_check",
        ]
    ] = Field(min_length=1, max_length=4)
    observation: str = Field(min_length=1, max_length=1000)
    hypothesis: str | None = Field(default=None, max_length=1000)
    confidence: float = Field(ge=0.0, le=1.0)
    action: ReviewAction | None = None


class CritiqueResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["accept", "revise"]
    summary: str = Field(min_length=1, max_length=4000)
    findings: list[CritiqueFinding] = Field(default_factory=list, max_length=MAX_FINDINGS)

    @model_validator(mode="after")
    def validate_verdict(self) -> CritiqueResponse:
        actions = sum(item.action is not None for item in self.findings)
        if actions > MAX_FINDINGS:
            raise ValueError("critique may propose at most three actions")
        if self.verdict == "accept" and actions:
            raise ValueError("an accepted critique must not propose actions")
        if self.verdict == "revise" and not self.findings:
            raise ValueError("a revise verdict requires at least one finding")
        return self


class CompareResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    winner: Literal["before", "after", "tie"]
    regressions: list[str] = Field(default_factory=list, max_length=3)
    remaining_blocker: str | None = Field(default=None, max_length=1000)
    confidence: float = Field(ge=0.0, le=1.0)
    accept: bool


class RealismCue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region: str = Field(min_length=1, max_length=256)
    observation: str = Field(min_length=1, max_length=1000)


_REALISM_SCOPE_BY_CATEGORY = {
    "lighting": "relighting",
    "world": "relighting",
    "atmosphere": "look-profile",
    "camera-post": "look-profile",
    "material": "artist-scene",
    "geometry-asset": "artist-scene",
    "render-artifact": "technical diagnosis",
}
_REFERENCE_SCOPE_BY_CATEGORY = {
    **_REALISM_SCOPE_BY_CATEGORY,
    "composition-staging": "artist-scene",
}


class ReferenceImageSpec(BaseModel):
    """One local, checksum-bound photographic reference and its narrow roles."""

    model_config = ConfigDict(extra="forbid")

    reference_id: str = Field(min_length=1, max_length=128)
    image_path: str = Field(min_length=1, max_length=4096)
    source_sha256: str
    roles: list[
        Literal[
            "LIGHTING_TOPOLOGY",
            "VALUE_AND_COLOR",
            "ATMOSPHERE_AND_DEPTH",
            "CAMERA_AND_FINISH",
            "MATERIAL_AND_SURFACE",
            "GEOMETRY_AND_ASSET",
            "COMPOSITION_AND_STAGING",
        ]
    ] = Field(min_length=1, max_length=7)
    comparison_focus: str = Field(min_length=1, max_length=1000)

    @field_validator("reference_id")
    @classmethod
    def validate_reference_id(cls, value: str) -> str:
        if not _SAFE_ID.fullmatch(value):
            raise ValueError("reference_id is not a safe identifier")
        return value

    @field_validator("source_sha256")
    @classmethod
    def validate_source_sha256(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("source_sha256 must be a SHA-256 hex digest")
        return value.lower()

    @field_validator("roles")
    @classmethod
    def validate_unique_roles(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("reference roles must be unique")
        return value


class ReferenceReviewPacket(BaseModel):
    """Fixed art-direction packet used for a reference-conditioned comparison."""

    model_config = ConfigDict(extra="forbid")

    packet_id: str = Field(min_length=1, max_length=128)
    primary_reference_id: str = Field(min_length=1, max_length=128)
    brief: str = Field(min_length=1, max_length=3000)
    targets: list[str] = Field(min_length=1, max_length=12)
    non_targets: list[str] = Field(default_factory=list, max_length=12)
    references: list[ReferenceImageSpec] = Field(min_length=2, max_length=MAX_REFERENCE_IMAGES)

    @field_validator("packet_id")
    @classmethod
    def validate_packet_id(cls, value: str) -> str:
        if not _SAFE_ID.fullmatch(value):
            raise ValueError("packet_id is not a safe identifier")
        return value

    @model_validator(mode="after")
    def validate_references(self) -> ReferenceReviewPacket:
        ids = [item.reference_id for item in self.references]
        if len(ids) != len(set(ids)):
            raise ValueError("reference IDs must be unique")
        if sum(len(item.roles) for item in self.references) > MAX_REFERENCE_ASSESSMENTS:
            raise ValueError("reference packet declares too many role assessments")
        if self.primary_reference_id not in ids:
            raise ValueError("primary_reference_id must identify one packet reference")
        return self


class ReferenceAssessment(BaseModel):
    """An independent candidate-versus-reference assessment for one declared role."""

    model_config = ConfigDict(extra="forbid")

    reference_id: str = Field(min_length=1, max_length=128)
    role: Literal[
        "LIGHTING_TOPOLOGY",
        "VALUE_AND_COLOR",
        "ATMOSPHERE_AND_DEPTH",
        "CAMERA_AND_FINISH",
        "MATERIAL_AND_SURFACE",
        "GEOMETRY_AND_ASSET",
        "COMPOSITION_AND_STAGING",
    ]
    status: Literal["MATCH", "PARTIAL", "MISS", "OUT_OF_SCOPE"]
    reference_observation: str = Field(min_length=1, max_length=1000)
    candidate_observation: str = Field(min_length=1, max_length=1000)
    candidate_region: str = Field(min_length=1, max_length=256)
    delta: str = Field(min_length=1, max_length=1000)
    confidence: float = Field(ge=0.0, le=1.0)


class ReferenceConflict(BaseModel):
    """A visible disagreement between references that should not be averaged away."""

    model_config = ConfigDict(extra="forbid")

    reference_ids: list[str] = Field(min_length=2, max_length=MAX_REFERENCE_IMAGES)
    aspect: str = Field(min_length=1, max_length=256)
    description: str = Field(min_length=1, max_length=1000)
    handling: Literal[
        "FOLLOW_PRIMARY",
        "FOLLOW_BRIEF",
        "PRESERVE_CANDIDATE",
        "NEEDS_ART_DIRECTION",
    ]


class ReferenceDirectionDraft(BaseModel):
    """Non-executable direction before deterministic scope assignment."""

    model_config = ConfigDict(extra="forbid")

    supported_by: list[str] = Field(min_length=1, max_length=MAX_REFERENCE_IMAGES)
    candidate_region: str = Field(min_length=1, max_length=256)
    direction: str = Field(min_length=1, max_length=1000)
    expected_effect: str = Field(min_length=1, max_length=1000)
    category: Literal[
        "lighting",
        "world",
        "atmosphere",
        "camera-post",
        "material",
        "geometry-asset",
        "composition-staging",
        "render-artifact",
    ]


class ReferenceDirection(ReferenceDirectionDraft):
    """Reference-grounded direction with caller-derived scope."""

    scope: Literal["relighting", "look-profile", "artist-scene", "technical diagnosis"]

    @model_validator(mode="after")
    def validate_scope(self) -> ReferenceDirection:
        expected = _REFERENCE_SCOPE_BY_CATEGORY[self.category]
        if self.scope != expected:
            raise ValueError(f"scope must be {expected!r} for category {self.category!r}")
        return self


class ReferenceComparisonDraft(BaseModel):
    """Provider response for per-reference evidence and shared directions."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=1500)
    assessments: list[ReferenceAssessment] = Field(
        min_length=1, max_length=MAX_REFERENCE_ASSESSMENTS
    )
    conflicts: list[ReferenceConflict] = Field(default_factory=list, max_length=MAX_FINDINGS)
    directions: list[ReferenceDirectionDraft] = Field(default_factory=list, max_length=MAX_FINDINGS)


class ReferenceComparisonResponse(BaseModel):
    """Validated reference comparison with no score, winner, or executable patch."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=1500)
    assessments: list[ReferenceAssessment] = Field(
        min_length=1, max_length=MAX_REFERENCE_ASSESSMENTS
    )
    conflicts: list[ReferenceConflict] = Field(default_factory=list, max_length=MAX_FINDINGS)
    directions: list[ReferenceDirection] = Field(default_factory=list, max_length=MAX_FINDINGS)


class OpenReferenceFeedback(BaseModel):
    """Open-ended lighting and vibe feedback from one reference vision pass."""

    model_config = ConfigDict(extra="forbid")

    feedback: str = Field(min_length=1, max_length=12000)


class ReferenceLookDirectionDraft(BaseModel):
    """Action-facing but non-executable lighting/look direction from the rewrite pass."""

    model_config = ConfigDict(extra="forbid")

    supported_by: list[str] = Field(min_length=1, max_length=MAX_REFERENCE_IMAGES)
    candidate_region: str = Field(min_length=1, max_length=256)
    direction: str = Field(min_length=1, max_length=1000)
    expected_effect: str = Field(min_length=1, max_length=1000)
    category: Literal[
        "lighting",
        "world",
        "atmosphere",
        "camera-post",
    ]


class ReferenceLookDirection(ReferenceLookDirectionDraft):
    scope: Literal["relighting", "look-profile"]

    @model_validator(mode="after")
    def validate_scope(self) -> ReferenceLookDirection:
        expected = _REFERENCE_SCOPE_BY_CATEGORY[self.category]
        if self.scope != expected:
            raise ValueError(f"scope must be {expected!r} for category {self.category!r}")
        return self


class ReferenceDeferredContextDraft(BaseModel):
    """Useful holistic context that the relighting workflow must not execute."""

    model_config = ConfigDict(extra="forbid")

    supported_by: list[str] = Field(min_length=1, max_length=MAX_REFERENCE_IMAGES)
    candidate_region: str = Field(min_length=1, max_length=256)
    observation: str = Field(min_length=1, max_length=1000)
    category: Literal[
        "material",
        "geometry-asset",
        "composition-staging",
        "render-artifact",
    ]


class ReferenceDeferredContext(ReferenceDeferredContextDraft):
    scope: Literal["artist-scene", "technical diagnosis"]

    @model_validator(mode="after")
    def validate_scope(self) -> ReferenceDeferredContext:
        expected = _REFERENCE_SCOPE_BY_CATEGORY[self.category]
        if self.scope != expected:
            raise ValueError(f"scope must be {expected!r} for category {self.category!r}")
        return self


class ReferenceSynthesisDraft(BaseModel):
    """Text-only rewrite that separates look directions from deferred context."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=1500)
    look_directions: list[ReferenceLookDirectionDraft] = Field(
        default_factory=list, max_length=MAX_FINDINGS
    )
    deferred_context: list[ReferenceDeferredContextDraft] = Field(
        default_factory=list, max_length=MAX_FINDINGS
    )
    conflicts: list[ReferenceConflict] = Field(default_factory=list, max_length=MAX_FINDINGS)


class ReferenceSynthesisResponse(BaseModel):
    """Final action-facing rewrite plus explicitly deferred holistic context."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=1500)
    look_directions: list[ReferenceLookDirection] = Field(
        default_factory=list, max_length=MAX_FINDINGS
    )
    deferred_context: list[ReferenceDeferredContext] = Field(
        default_factory=list, max_length=MAX_FINDINGS
    )
    conflicts: list[ReferenceConflict] = Field(default_factory=list, max_length=MAX_FINDINGS)


class RealismChange(BaseModel):
    """Unfiltered first-pass change; the rewrite pass removes artist-scene work."""

    model_config = ConfigDict(extra="forbid")

    region: str = Field(min_length=1, max_length=256)
    direction: str = Field(min_length=1, max_length=1000)
    expected_effect: str = Field(min_length=1, max_length=1000)
    category: Literal[
        "lighting",
        "world",
        "atmosphere",
        "camera-post",
        "material",
        "geometry-asset",
        "render-artifact",
    ]


class RealismResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classification: Literal["REAL", "CG", "INDETERMINATE"]
    calibrated_confidence: float = Field(ge=0.0, le=1.0)
    summary: str = Field(min_length=1, max_length=4000)
    real_cues: list[RealismCue] = Field(max_length=MAX_FINDINGS)
    cg_cues: list[RealismCue] = Field(max_length=MAX_FINDINGS)
    proposed_changes: list[RealismChange] = Field(max_length=MAX_FINDINGS)

    @model_validator(mode="after")
    def validate_cg_diagnosis(self) -> RealismResponse:
        if self.classification == "CG" and not self.cg_cues:
            raise ValueError("a CG classification requires at least one localized CG cue")
        if self.classification == "CG" and not self.proposed_changes:
            raise ValueError("a CG classification requires at least one proposed change")
        return self


class LookOnlyRealismCue(BaseModel):
    """A visible realism cue confined to artist-controllable look qualities."""

    model_config = ConfigDict(extra="forbid")

    region: str = Field(min_length=1, max_length=256)
    observation: str = Field(min_length=1, max_length=1000)
    aspect: Literal[
        "lighting",
        "shadows",
        "lighting-color",
        "lighting-aesthetics",
        "atmosphere",
        "depth-of-field",
        "optical-softness-sharpness",
        "highlight-rolloff",
        "grain",
        "bloom",
        "vignette",
        "chromatic-aberration",
        "grading",
        "other-post-effect",
        "render-artifact",
    ]


class LookOnlyRealismChange(BaseModel):
    """A change limited to lighting, camera/post, atmosphere, or diagnosis."""

    model_config = ConfigDict(extra="forbid")

    region: str = Field(min_length=1, max_length=256)
    direction: str = Field(min_length=1, max_length=1000)
    expected_effect: str = Field(min_length=1, max_length=1000)
    category: Literal[
        "lighting",
        "world",
        "atmosphere",
        "camera-post",
        "render-artifact",
    ]
    scope: Literal["relighting", "look-profile", "technical diagnosis"]

    @model_validator(mode="after")
    def validate_scope(self) -> LookOnlyRealismChange:
        expected = _REALISM_SCOPE_BY_CATEGORY[self.category]
        if self.scope != expected:
            raise ValueError(f"scope must be {expected!r} for category {self.category!r}")
        return self


class LookOnlyRealismDraftChange(BaseModel):
    """Second-pass change before deterministic scope assignment."""

    model_config = ConfigDict(extra="forbid")

    region: str = Field(min_length=1, max_length=256)
    direction: str = Field(min_length=1, max_length=1000)
    expected_effect: str = Field(min_length=1, max_length=1000)
    category: Literal[
        "lighting",
        "world",
        "atmosphere",
        "camera-post",
        "render-artifact",
    ]


class LookOnlyRealismDraft(BaseModel):
    """Text-only rewrite output before deterministic scope assignment."""

    model_config = ConfigDict(extra="forbid")

    classification: Literal["REAL", "CG", "INDETERMINATE"]
    calibrated_confidence: float = Field(ge=0.0, le=1.0)
    summary: str = Field(min_length=1, max_length=1500)
    real_cues: list[LookOnlyRealismCue] = Field(max_length=MAX_FINDINGS)
    cg_cues: list[LookOnlyRealismCue] = Field(max_length=MAX_FINDINGS)
    proposed_changes: list[LookOnlyRealismDraftChange] = Field(max_length=MAX_FINDINGS)

    @model_validator(mode="after")
    def validate_cg_diagnosis(self) -> LookOnlyRealismDraft:
        if self.classification == "CG" and not self.cg_cues:
            raise ValueError("a CG classification requires at least one localized CG cue")
        if self.classification == "CG" and not self.proposed_changes:
            raise ValueError("a CG classification requires at least one proposed change")
        return self


class LookOnlyRealismResponse(BaseModel):
    """Blind realism response that cannot propose material or geometry work."""

    model_config = ConfigDict(extra="forbid")

    classification: Literal["REAL", "CG", "INDETERMINATE"]
    calibrated_confidence: float = Field(ge=0.0, le=1.0)
    summary: str = Field(min_length=1, max_length=1500)
    real_cues: list[LookOnlyRealismCue] = Field(max_length=MAX_FINDINGS)
    cg_cues: list[LookOnlyRealismCue] = Field(max_length=MAX_FINDINGS)
    proposed_changes: list[LookOnlyRealismChange] = Field(max_length=MAX_FINDINGS)

    @model_validator(mode="after")
    def validate_cg_diagnosis(self) -> LookOnlyRealismResponse:
        if self.classification == "CG" and not self.cg_cues:
            raise ValueError("a CG classification requires at least one localized CG cue")
        if self.classification == "CG" and not self.proposed_changes:
            raise ValueError("a CG classification requires at least one proposed change")
        return self


def _result_packet(
    batch_id: str,
    result_id: str,
    *,
    max_size: int = 1024,
    jpeg_quality: int = 90,
) -> tuple[dict[str, Any], list[ImageContent]]:
    result = get_look_render_result(
        _bounded_id(batch_id, "batch_id"),
        _bounded_id(result_id, "result_id"),
        max_size=max_size,
        format="JPEG",
        jpeg_quality=jpeg_quality,
    )
    if not isinstance(result, CallToolResult):
        raise RuntimeError("Look render result did not return MCP content")
    metadata = result.structuredContent
    if not isinstance(metadata, dict) or metadata.get("status") != "SUCCEEDED":
        raise RuntimeError("Cinematic review requires a successful look render result")
    images = [item for item in result.content if isinstance(item, ImageContent)]
    if len(images) != 2 or metadata.get("review_packet") is None:
        raise RuntimeError("Cinematic review requires beauty plus one review contact sheet")
    technical = metadata["review_packet"].get("technical_check")
    if not isinstance(technical, dict) or technical.get("status") == "FAIL":
        raise RuntimeError("Provider review is blocked by the deterministic technical audit")
    return metadata, images


def _image_part(image: ImageContent) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{image.mimeType};base64,{image.data}"},
    }


def _load_reference_image(spec: ReferenceImageSpec) -> tuple[ImageContent, dict[str, Any]]:
    path = Path(spec.image_path).expanduser()
    if not path.is_absolute():
        raise ValueError(f"Reference {spec.reference_id} image_path must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"Reference {spec.reference_id} image does not exist") from exc
    if not resolved.is_file():
        raise ValueError(f"Reference {spec.reference_id} image is not a regular file")
    if resolved.stat().st_size > 20 * 1024 * 1024:
        raise ValueError(f"Reference {spec.reference_id} image exceeds 20 MiB")
    source = resolved.read_bytes()
    source_sha256 = hashlib.sha256(source).hexdigest()
    if source_sha256 != spec.source_sha256:
        raise ValueError(f"Reference {spec.reference_id} source SHA-256 does not match")
    try:
        with PILImage.open(BytesIO(source)) as loaded:
            image = ImageOps.exif_transpose(loaded)
            image.load()
            image = image.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Reference {spec.reference_id} is not a supported image") from exc
    image.thumbnail((1600, 1600), PILImage.Resampling.LANCZOS)
    output = BytesIO()
    image.save(output, format="JPEG", quality=92, optimize=True)
    proxy = output.getvalue()
    return (
        ImageContent(
            type="image",
            data=base64.b64encode(proxy).decode("ascii"),
            mimeType="image/jpeg",
        ),
        {
            "reference_id": spec.reference_id,
            "source_sha256": source_sha256,
            "proxy_sha256": hashlib.sha256(proxy).hexdigest(),
            "width": image.width,
            "height": image.height,
            "roles": spec.roles,
        },
    )


def _reference_prompt(packet: ReferenceReviewPacket, render_metadata: dict[str, Any]) -> str:
    rendered_review_intent = render_metadata.get("review_packet", {}).get("review_intent")
    if not isinstance(rendered_review_intent, dict):
        raise ValueError("REFERENCE mode requires rendered review_intent provenance")
    goal = rendered_review_intent.get("summary")
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("REFERENCE mode requires a non-empty rendered relighting goal")
    return (
        "The goal is for the candidate image to look like this:\n"
        f"{goal.strip()}\n\n"
        "Compare the candidate image to the reference images and explain how the lighting "
        "in the candidate image needs to change to match the references better, given that "
        "relighting goal. The models and textures cannot be changed, so focus your critique "
        "on the lighting and overall vibe.\n\n"
        "Be candid and open-ended. Explain what is working, what is not working, and the most "
        "important lighting changes you would make. Do not suggest changes to models or "
        "textures. The images are ordered as the candidate, the primary reference, and then "
        "the remaining references. Return your critique in the feedback field."
    )


def _reference_rewrite_prompt(raw_comparison: dict[str, Any], packet: ReferenceReviewPacket) -> str:
    known_ids = [reference.reference_id for reference in packet.references]
    return (
        "You are a senior VFX feedback editor. You did not see any images. Convert the supplied "
        "holistic reference comparison into a concise decision handoff without inventing "
        "visible facts, regions, causes, conflicts, or directions. Preserve the full source "
        "comparison outside this rewrite; this handoff should separate what the relighting/look "
        "workflow can act on from useful context it must defer.\n"
        "look_directions may contain only lighting direction, motivation, intensity, falloff, "
        "lighting colour and aesthetics; shadow density, softness, contact and consistency; "
        "world light; value hierarchy and highlight rolloff; atmosphere and depth; depth of "
        "field and focus behaviour; and post-processing such as softness, sharpness, grain, "
        "bloom, vignette, chromatic aberration, and grading. deferred_context may contain only "
        "material, geometry/asset, composition/staging, or visible render-artifact observations "
        "that remain useful but are not executable relighting actions. Do not disguise deferred "
        "context as a look direction. Remove statements about whether "
        "the candidate is real, CG, rendered, photographed, synthetic, or produced by a "
        "particular method. Remove hidden-production speculation such as lights that are not "
        "visibly established, HDRIs, ray counts, samples, passes, or renderer settings. Every "
        "item must remain non-executable, cite only known reference IDs, and contain no exact "
        "parameter values. Keep no score, winner, realism label, or accept/reject verdict. "
        "The primary reference remains binding in this rewrite. Preserve FOLLOW_PRIMARY "
        "conflicts and never emit a secondary-supported direction in a role shared with the "
        "primary unless it also cites the primary. Return only the strict response schema. "
        "Primary reference ID:\n"
        + json.dumps(packet.primary_reference_id)
        + "\nKnown reference IDs:\n"
        + json.dumps(known_ids, sort_keys=True, separators=(",", ":"))
        + "\nSource comparison:\n"
        + json.dumps(raw_comparison, sort_keys=True, separators=(",", ":"))
    )


def _validate_reference_comparison(
    draft: ReferenceComparisonDraft,
    packet: ReferenceReviewPacket,
    *,
    request_id: str,
    usage: Any,
) -> ReferenceComparisonResponse:
    expected = {
        (reference.reference_id, role)
        for reference in packet.references
        for role in reference.roles
    }
    actual_pairs = [(item.reference_id, item.role) for item in draft.assessments]
    known_ids = {item.reference_id for item in packet.references}
    roles_by_id = {item.reference_id: set(item.roles) for item in packet.references}
    primary_id = packet.primary_reference_id
    primary_roles = roles_by_id[primary_id]
    errors: list[str] = []
    if len(actual_pairs) != len(set(actual_pairs)):
        errors.append("assessment pairs must be unique")
    if set(actual_pairs) != expected:
        errors.append("assessments must cover every declared reference-role pair exactly once")
    useful_ids = {
        item.reference_id for item in draft.assessments if item.status in {"PARTIAL", "MISS"}
    }
    for direction in draft.directions:
        if len(direction.supported_by) != len(set(direction.supported_by)):
            errors.append("direction supported_by IDs must be unique")
        if not set(direction.supported_by) <= known_ids:
            errors.append("direction cites an unknown reference ID")
        if not set(direction.supported_by) & useful_ids:
            errors.append("direction must cite a PARTIAL or MISS assessment")
        secondary_ids = [
            item for item in direction.supported_by if item != primary_id and item in roles_by_id
        ]
        secondary_roles = (
            set().union(*(roles_by_id[item] for item in secondary_ids)) if secondary_ids else set()
        )
        if primary_roles & secondary_roles and primary_id not in direction.supported_by:
            errors.append(
                "directions supported by secondary references in primary-governed roles "
                "must also cite the primary reference"
            )
    for conflict in draft.conflicts:
        if len(conflict.reference_ids) != len(set(conflict.reference_ids)):
            errors.append("conflict reference IDs must be unique")
        if not set(conflict.reference_ids) <= known_ids:
            errors.append("conflict cites an unknown reference ID")
        if primary_id in conflict.reference_ids and conflict.handling != "FOLLOW_PRIMARY":
            errors.append("conflicts involving the primary reference must use FOLLOW_PRIMARY")
    if errors:
        canonical = json.dumps(draft.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        raise RuntimeError(
            json.dumps(
                {
                    "status": "INVALID_PROVIDER_RESPONSE",
                    "stage": "reference",
                    "request_id": request_id,
                    "usage": usage,
                    "response_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                    "validation_errors": sorted(set(errors)),
                },
                sort_keys=True,
            )
        )
    directions = [
        ReferenceDirection(
            **item.model_dump(mode="json"),
            scope=_REFERENCE_SCOPE_BY_CATEGORY[item.category],
        )
        for item in draft.directions
    ]
    return ReferenceComparisonResponse(
        summary=draft.summary,
        assessments=draft.assessments,
        conflicts=draft.conflicts,
        directions=directions,
    )


def _validate_reference_synthesis(
    draft: ReferenceSynthesisDraft,
    packet: ReferenceReviewPacket,
    *,
    request_id: str,
    usage: Any,
) -> ReferenceSynthesisResponse:
    known_ids = {item.reference_id for item in packet.references}
    roles_by_id = {item.reference_id: set(item.roles) for item in packet.references}
    primary_id = packet.primary_reference_id
    primary_roles = roles_by_id[primary_id]
    errors: list[str] = []
    for item in [*draft.look_directions, *draft.deferred_context]:
        if len(item.supported_by) != len(set(item.supported_by)):
            errors.append("supported_by IDs must be unique")
        if not set(item.supported_by) <= known_ids:
            errors.append("item cites an unknown reference ID")
    for item in draft.look_directions:
        secondary_ids = [
            reference_id
            for reference_id in item.supported_by
            if reference_id != primary_id and reference_id in roles_by_id
        ]
        secondary_roles = (
            set().union(*(roles_by_id[reference_id] for reference_id in secondary_ids))
            if secondary_ids
            else set()
        )
        if primary_roles & secondary_roles and primary_id not in item.supported_by:
            errors.append(
                "look directions supported by secondary references in primary-governed "
                "roles must also cite the primary reference"
            )
    for conflict in draft.conflicts:
        if len(conflict.reference_ids) != len(set(conflict.reference_ids)):
            errors.append("conflict reference IDs must be unique")
        if not set(conflict.reference_ids) <= known_ids:
            errors.append("conflict cites an unknown reference ID")
        if primary_id in conflict.reference_ids and conflict.handling != "FOLLOW_PRIMARY":
            errors.append("conflicts involving the primary reference must use FOLLOW_PRIMARY")
    if errors:
        canonical = json.dumps(draft.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        raise RuntimeError(
            json.dumps(
                {
                    "status": "INVALID_PROVIDER_RESPONSE",
                    "stage": "reference_rewrite",
                    "request_id": request_id,
                    "usage": usage,
                    "response_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                    "validation_errors": sorted(set(errors)),
                },
                sort_keys=True,
            )
        )
    look_directions = [
        ReferenceLookDirection(
            **item.model_dump(mode="json"),
            scope=_REFERENCE_SCOPE_BY_CATEGORY[item.category],
        )
        for item in draft.look_directions
    ]
    deferred_context = [
        ReferenceDeferredContext(
            **item.model_dump(mode="json"),
            scope=_REFERENCE_SCOPE_BY_CATEGORY[item.category],
        )
        for item in draft.deferred_context
    ]
    return ReferenceSynthesisResponse(
        summary=draft.summary,
        look_directions=look_directions,
        deferred_context=deferred_context,
        conflicts=draft.conflicts,
    )


def _critique_prompt(metadata: dict[str, Any]) -> str:
    packet = metadata["review_packet"]
    prompt_data = {
        "look_intent": packet["review_intent"],
        "camera": metadata["camera"],
        "technical_check": packet["technical_check"],
        "tiles": packet["tiles"],
        "editable_controls": packet["editable_controls"],
    }
    return (
        "You are a senior cinematic lighting and finishing critic. Judge the final "
        "composited beauty against the supplied look intent, using the labeled AOV "
        "contact sheet only as diagnostic evidence. Prioritize lighting hierarchy, "
        "motivated direction, subject separation, depth, color/value structure, atmosphere, "
        "and restraint. The Camera Depth tile is camera-space proximity normalized so "
        "near geometry is white and distant or infinite geometry is black. Cite beauty "
        "or named tiles for every finding. Separate visible "
        "observation from hypothesis. Never claim a Blender switch is disabled unless the "
        "technical_check says so. Propose no more than three SET actions and only paths "
        "present in editable_controls. Return accept when no edit clearly justifies another "
        "render. Input record:\n" + json.dumps(prompt_data, sort_keys=True, separators=(",", ":"))
    )


def _compare_prompt(before: dict[str, Any], after: dict[str, Any]) -> str:
    return (
        "Compare the BEFORE and AFTER composited renders directly against the same look "
        "intent. Images are ordered: before beauty, before AOV sheet, after beauty, after "
        "AOV sheet. Prefer AFTER only if it materially improves the intended cinematic look "
        "without a new regression. Do not use numeric beauty scores. Return the strict "
        "comparison schema. Input record:\n"
        + json.dumps(
            {
                "look_intent": after["review_packet"]["review_intent"],
                "camera": after["camera"],
                "before_technical_check": before["review_packet"]["technical_check"],
                "after_technical_check": after["review_packet"]["technical_check"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _realism_prompt(
    version: Literal["literal-v1", "evidence-v1", "causal-v1", "look-only-v1"],
) -> str:
    common = (
        f"{REALISM_QUESTION}\n"
        "Judge only what is visibly supported by the supplied image. Do not infer its "
        "origin, production method, filename, or hidden context. Use INDETERMINATE when "
        "the visible evidence is insufficient. Calibrate confidence to the strength and "
        "specificity of that evidence. Localize every cue and every proposed change. "
        "Classify each proposed change as lighting, world, atmosphere, camera-post, "
        "material, geometry-asset, or render-artifact. Do not assign implementation "
        "scope; a separate feedback editor handles relevance and scope. Return only the "
        "strict first-pass response schema."
    )
    if version == "literal-v1":
        return common + (
            " Use a direct literal strategy: answer the real-versus-CG question from the "
            "clearest visible look cues without forcing equal evidence on both sides. "
            "Describe what is visible, avoid causal speculation, and propose only changes "
            "that directly address those cues."
        )
    evidence = (
        " Report visible evidence supporting both real and CG interpretations, even when "
        "one side is weak. Keep observations separate from causal assumptions. Favor "
        "specific spatial cues over general aesthetic preference."
    )
    if version == "evidence-v1":
        return common + evidence
    if version == "causal-v1":
        return (
            common
            + evidence
            + (
                " Use a causal analysis strategy. Explicitly test light and shadow consistency, "
                "material and specular response, contact and repetition, atmosphere, optics, "
                "highlight rolloff, and render artifacts. Grain, bloom, darkness, vignetting, "
                "grading, chromatic aberration, and shallow depth of field are not proof of "
                "realism and must not be treated as such. Prioritize causal changes that remove "
                "the strongest visible contradiction."
            )
        )
    return common + (
        " Use an actionable cinematic-finishing strategy. Judge whether the lighting and "
        "camera treatment support believable depth, visual hierarchy, mood, and photographic "
        "restraint. Rank proposed changes by expected realism impact, prefer reversible "
        "artist-controllable lighting, atmosphere, optics, and post adjustments, and return "
        "fewer than three changes when fewer are visibly justified."
    )


def _realism_rewrite_prompt(raw_assessment: dict[str, Any]) -> str:
    return (
        "You are a senior VFX lighting-feedback editor. You did not see the image. Rewrite "
        "and filter the supplied first-pass realism assessment without inventing new visible "
        "facts, regions, causes, or changes. Keep only feedback useful for lighting direction, "
        "motivation, intensity, falloff, lighting color and aesthetics; shadow density, "
        "softness, contact and consistency; atmosphere; camera and optical qualities such as "
        "depth of field and focus behavior; and post-processing such as softness, sharpness, "
        "grain, bloom, vignette, chromatic aberration, grading, highlight rolloff, and visible "
        "render artifacts. Remove feedback about materials, textures, shaders, roughness, "
        "geometry, modeling, assets, scatter systems, repetition, object placement, "
        "architecture, and production design. Remove hidden-production speculation such as "
        "HDRI, ray counts, samples, passes, or renderer settings; when the source contains a "
        "supported visible symptom, rewrite it only as that visible symptom. Do not treat "
        "grain, bloom, darkness, vignetting, grading, chromatic aberration, or shallow depth "
        "of field as proof of realism. Re-evaluate REAL, CG, or INDETERMINATE using only the "
        "retained in-scope evidence and calibrate confidence conservatively. If a CG label is "
        "not supported by at least one retained localized CG cue and one retained change, use "
        "INDETERMINATE. Proposed changes may use only lighting, world, atmosphere, "
        "camera-post, or render-artifact categories. Do not assign scope; the caller derives "
        "it deterministically. Return only the strict rewrite schema. Source feedback:\n"
        + json.dumps(raw_assessment, sort_keys=True, separators=(",", ":"))
    )


def _image_sha256(image: ImageContent) -> str:
    try:
        raw = base64.b64decode(image.data, validate=True)
    except (ValueError, TypeError) as exc:
        raise RuntimeError("Look render returned invalid base64 image data") from exc
    return hashlib.sha256(raw).hexdigest()


def _provider_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Remove constraints Claude cannot compile; Pydantic still validates them locally."""

    def sanitize(value: Any) -> Any:
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        if not isinstance(value, dict):
            return value
        constraints = [
            f"{key}={value[key]}"
            for key in sorted(_UNSUPPORTED_PROVIDER_SCHEMA_CONSTRAINTS)
            if key in value
        ]
        result = {
            key: sanitize(item)
            for key, item in value.items()
            if key not in _UNSUPPORTED_PROVIDER_SCHEMA_CONSTRAINTS
        }
        if constraints:
            local_note = "Validated locally after generation: " + ", ".join(constraints) + "."
            description = result.get("description")
            result["description"] = (
                f"{description} {local_note}" if isinstance(description, str) else local_note
            )
        return result

    return sanitize(model.model_json_schema())


def _post_openrouter(payload: dict[str, Any], api_key: str) -> httpx.Response:
    timeout = httpx.Timeout(180.0, connect=10.0)
    with httpx.Client(timeout=timeout) as client:
        return client.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-Title": "blend-ai cinematic review",
            },
            json=payload,
        )


def _invalid_provider_response(
    *,
    stage: str,
    request_id: str,
    usage: Any,
    response_text: str,
    error: str,
) -> RuntimeError:
    return RuntimeError(
        json.dumps(
            {
                "status": "INVALID_PROVIDER_RESPONSE",
                "stage": stage,
                "request_id": request_id,
                "usage": usage,
                "response_sha256": hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
                "response_text": response_text,
                "error": error,
            },
            sort_keys=True,
        )
    )


def _response_content(response: httpx.Response, *, stage: str) -> tuple[dict[str, Any], str, Any]:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = response.text.strip().replace("\n", " ")[:2000]
        raise httpx.HTTPStatusError(
            f"{exc}; bounded provider detail: {detail}",
            request=exc.request,
            response=exc.response,
        ) from exc
    request_id = str(response.headers.get("x-request-id") or "unknown")
    try:
        body = response.json()
    except ValueError as exc:
        raise _invalid_provider_response(
            stage=stage,
            request_id=request_id,
            usage=None,
            response_text=response.text,
            error="OpenRouter returned a non-JSON chat-completions response",
        ) from exc
    if isinstance(body, dict):
        request_id = str(response.headers.get("x-request-id") or body.get("id") or "unknown")
        usage = body.get("usage")
    else:
        usage = None
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise _invalid_provider_response(
            stage=stage,
            request_id=request_id,
            usage=usage,
            response_text=json.dumps(body, sort_keys=True, separators=(",", ":")),
            error="OpenRouter returned an invalid chat-completions response",
        ) from exc
    if not isinstance(content, str):
        raise _invalid_provider_response(
            stage=stage,
            request_id=request_id,
            usage=usage,
            response_text=json.dumps(content, sort_keys=True, separators=(",", ":")),
            error="OpenRouter response content was not JSON text",
        )
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise _invalid_provider_response(
            stage=stage,
            request_id=request_id,
            usage=usage,
            response_text=content,
            error="OpenRouter returned invalid structured JSON",
        ) from exc
    if not isinstance(parsed, dict):
        raise _invalid_provider_response(
            stage=stage,
            request_id=request_id,
            usage=usage,
            response_text=content,
            error="OpenRouter structured response was not an object",
        )
    return parsed, request_id, usage


def _validate_provider_result(
    response_model: type[BaseModel],
    parsed: dict[str, Any],
    *,
    stage: str,
    request_id: str,
    usage: Any,
) -> BaseModel:
    try:
        return response_model.model_validate(parsed)
    except ValidationError as exc:
        raw = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
        validation_errors = json.loads(
            json.dumps(
                exc.errors(include_url=False, include_input=False),
                default=str,
            )
        )
        failure = {
            "status": "INVALID_PROVIDER_RESPONSE",
            "stage": stage,
            "request_id": request_id,
            "usage": usage,
            "response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "response": parsed,
            "validation_errors": validation_errors,
        }
        raise RuntimeError(json.dumps(failure, sort_keys=True)) from exc


@mcp.tool()
def review_look_render(
    mode: Literal["REALISM", "REFERENCE", "CRITIQUE", "COMPARE"],
    batch_id: str,
    result_id: str,
    baseline_batch_id: str | None = None,
    baseline_result_id: str | None = None,
    model: str = DEFAULT_MODEL,
    realism_prompt_version: Literal["literal-v1", "evidence-v1", "causal-v1", "look-only-v1"]
    | None = None,
    realism_rewrite_model: str | None = None,
    reference_packet: dict[str, Any] | None = None,
    reference_rewrite_model: str | None = None,
) -> dict[str, Any]:
    """Assess realism, compare references, critique diagnostics, or compare revisions."""
    if mode not in {"REALISM", "REFERENCE", "CRITIQUE", "COMPARE"}:
        raise ValueError("mode must be REALISM, REFERENCE, CRITIQUE, or COMPARE")
    if not isinstance(model, str) or not _SAFE_MODEL.fullmatch(model):
        raise ValueError("model is not a safe OpenRouter model identifier")
    if mode != "REALISM" and realism_prompt_version is not None:
        raise ValueError("realism_prompt_version is valid only in REALISM mode")
    if mode != "REALISM" and realism_rewrite_model is not None:
        raise ValueError("realism_rewrite_model is valid only in REALISM mode")
    if mode != "REFERENCE" and reference_packet is not None:
        raise ValueError("reference_packet is valid only in REFERENCE mode")
    if mode == "REFERENCE" and reference_packet is None:
        raise ValueError("REFERENCE mode requires reference_packet")
    if reference_rewrite_model is not None:
        raise ValueError(
            "reference_rewrite_model is no longer supported; REFERENCE is a single-pass review"
        )
    if mode == "REALISM" and realism_prompt_version not in {
        None,
        *REALISM_PROMPT_VERSIONS,
    }:
        raise ValueError("realism_prompt_version is not a supported prompt version")
    if realism_rewrite_model is not None and (
        not isinstance(realism_rewrite_model, str)
        or not _SAFE_MODEL.fullmatch(realism_rewrite_model)
    ):
        raise ValueError("realism_rewrite_model is not a safe OpenRouter model identifier")
    if mode != "COMPARE" and (baseline_batch_id is not None or baseline_result_id is not None):
        raise ValueError("baseline identifiers are valid only in COMPARE mode")
    validated_reference_packet = (
        ReferenceReviewPacket.model_validate(reference_packet)
        if mode == "REFERENCE" and reference_packet is not None
        else None
    )
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured in the MCP process")

    realism_version = realism_prompt_version or "causal-v1"
    realism_editor_model = realism_rewrite_model or DEFAULT_REALISM_REWRITE_MODEL
    packet_options = (
        {"max_size": 1600, "jpeg_quality": 92} if mode in {"REALISM", "REFERENCE"} else {}
    )
    current_metadata, current_images = _result_packet(batch_id, result_id, **packet_options)
    source_sha256: str | None = None
    if mode in {"REALISM", "REFERENCE"}:
        source_sha256 = current_metadata.get("artifact_sha256")
        if not isinstance(source_sha256, str) or not _SHA256.fullmatch(source_sha256):
            raise RuntimeError(f"{mode} review requires the source artifact SHA-256")
    reference_receipts: list[dict[str, Any]] = []
    reference_specs: list[ReferenceImageSpec] = []
    response_model: type[BaseModel]
    if mode == "REALISM":
        prompt = _realism_prompt(realism_version)
        images = current_images[:1]
        response_model = RealismResponse
        schema_name = "blind_realism_assessment"
        max_tokens = 1800
    elif mode == "REFERENCE":
        assert validated_reference_packet is not None
        prompt = _reference_prompt(validated_reference_packet, current_metadata)
        reference_specs = sorted(
            validated_reference_packet.references,
            key=lambda item: item.reference_id != validated_reference_packet.primary_reference_id,
        )
        loaded_references = [_load_reference_image(spec) for spec in reference_specs]
        images = [current_images[0], *(item[0] for item in loaded_references)]
        reference_receipts = [item[1] for item in loaded_references]
        for receipt in reference_receipts:
            receipt["is_primary"] = (
                receipt["reference_id"] == validated_reference_packet.primary_reference_id
            )
        response_model = OpenReferenceFeedback
        schema_name = "open_reference_lighting_feedback"
        max_tokens = 3200
    elif mode == "CRITIQUE":
        prompt = _critique_prompt(current_metadata)
        images = current_images
        response_model = CritiqueResponse
        schema_name = "cinematic_critique"
        max_tokens = 3200
    else:
        if baseline_batch_id is None or baseline_result_id is None:
            raise ValueError("COMPARE mode requires baseline_batch_id and baseline_result_id")
        baseline_metadata, baseline_images = _result_packet(baseline_batch_id, baseline_result_id)
        for field in ("profile_id", "source_scene", "camera", "frame"):
            baseline_value = baseline_metadata.get(field)
            current_value = current_metadata.get(field)
            if field == "camera":
                baseline_value = _camera_comparison_identity(baseline_value)
                current_value = _camera_comparison_identity(current_value)
            if baseline_value != current_value:
                raise ValueError(f"COMPARE results have mismatched {field}")
        prompt = _compare_prompt(baseline_metadata, current_metadata)
        images = baseline_images + current_images
        response_model = CompareResponse
        schema_name = "cinematic_comparison"
        max_tokens = 700

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if mode == "REFERENCE":
        assert validated_reference_packet is not None
        content.extend(
            [
                {"type": "text", "text": "CANDIDATE IMAGE"},
                _image_part(images[0]),
            ]
        )
        for spec, image in zip(reference_specs, images[1:], strict=True):
            label = (
                "PRIMARY REFERENCE"
                if spec.reference_id == validated_reference_packet.primary_reference_id
                else "SECONDARY REFERENCE"
            )
            content.extend(
                [
                    {"type": "text", "text": f"{label} {spec.reference_id}"},
                    _image_part(image),
                ]
            )
    else:
        content.extend(_image_part(image) for image in images)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "max_tokens": max_tokens,
        "provider": {"require_parameters": True},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": _provider_json_schema(response_model),
            },
        },
    }
    response = _post_openrouter(payload, api_key)
    response_stage = "vision" if mode == "REALISM" else mode.lower()
    parsed, request_id, usage = _response_content(response, stage=response_stage)
    validated_model = _validate_provider_result(
        response_model,
        parsed,
        stage=response_stage,
        request_id=request_id,
        usage=usage,
    )
    rewrite_receipt: dict[str, Any] | None = None
    raw_response_sha256: str | None = None
    if mode == "REALISM":
        raw_assessment = validated_model.model_dump(mode="json")
        raw_canonical = json.dumps(raw_assessment, sort_keys=True, separators=(",", ":"))
        raw_response_sha256 = hashlib.sha256(raw_canonical.encode("utf-8")).hexdigest()
        rewrite_prompt = _realism_rewrite_prompt(raw_assessment)
        rewrite_payload = {
            "model": realism_editor_model,
            "messages": [{"role": "user", "content": rewrite_prompt}],
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
        rewrite_response = _post_openrouter(rewrite_payload, api_key)
        rewrite_parsed, rewrite_request_id, rewrite_usage = _response_content(
            rewrite_response, stage="rewrite"
        )
        rewrite_validated = _validate_provider_result(
            LookOnlyRealismDraft,
            rewrite_parsed,
            stage="rewrite",
            request_id=rewrite_request_id,
            usage=rewrite_usage,
        )
        rewrite_draft = rewrite_validated.model_dump(mode="json")
        rewrite_canonical = json.dumps(rewrite_draft, sort_keys=True, separators=(",", ":"))
        final_changes = [
            LookOnlyRealismChange(
                **change,
                scope=_REALISM_SCOPE_BY_CATEGORY[change["category"]],
            )
            for change in rewrite_draft["proposed_changes"]
        ]
        validated_model = LookOnlyRealismResponse(
            classification=rewrite_draft["classification"],
            calibrated_confidence=rewrite_draft["calibrated_confidence"],
            summary=rewrite_draft["summary"],
            real_cues=rewrite_draft["real_cues"],
            cg_cues=rewrite_draft["cg_cues"],
            proposed_changes=final_changes,
        )
        rewrite_receipt = {
            "provider": "openrouter",
            "model": realism_editor_model,
            "request_id": rewrite_request_id,
            "usage": rewrite_usage,
            "prompt_sha256": hashlib.sha256(rewrite_prompt.encode("utf-8")).hexdigest(),
            "input_feedback_sha256": raw_response_sha256,
            "input_image_count": 0,
            "response_sha256": hashlib.sha256(rewrite_canonical.encode("utf-8")).hexdigest(),
        }
    elif mode == "REFERENCE":
        assert isinstance(validated_model, OpenReferenceFeedback)
    if isinstance(validated_model, CritiqueResponse):
        editable = set(current_metadata["review_packet"]["editable_controls"])
        unexpected = [
            finding.action.path
            for finding in validated_model.findings
            if finding.action is not None and finding.action.path not in editable
        ]
        if unexpected:
            raise RuntimeError(
                "OpenRouter proposed controls absent from the rendered profile: "
                + ", ".join(unexpected)
            )
    validated = validated_model.model_dump(mode="json")
    canonical = json.dumps(validated, sort_keys=True, separators=(",", ":"))
    receipt = {
        "mode": mode,
        "provider": "openrouter",
        "model": model,
        "request_id": request_id,
        "usage": usage,
        "response_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "result": validated,
    }
    if mode == "REALISM":
        assert source_sha256 is not None
        assert raw_response_sha256 is not None
        assert rewrite_receipt is not None
        receipt.update(
            {
                "prompt_version": realism_version,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "input_proxy_sha256": _image_sha256(images[0]),
                "source_artifact_sha256": source_sha256.lower(),
                "input_image_count": 1,
                "raw_response_sha256": raw_response_sha256,
                "provider_call_count": 2,
                "rewrite": rewrite_receipt,
            }
        )
    elif mode == "REFERENCE":
        assert source_sha256 is not None
        assert validated_reference_packet is not None
        packet_canonical = json.dumps(
            validated_reference_packet.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        receipt.update(
            {
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "input_proxy_sha256": _image_sha256(images[0]),
                "source_artifact_sha256": source_sha256.lower(),
                "input_image_count": len(images),
                "reference_image_count": len(reference_receipts),
                "reference_packet_id": validated_reference_packet.packet_id,
                "primary_reference_id": validated_reference_packet.primary_reference_id,
                "reference_packet_sha256": hashlib.sha256(
                    packet_canonical.encode("utf-8")
                ).hexdigest(),
                "references": reference_receipts,
                "provider_call_count": 1,
            }
        )
    return receipt
