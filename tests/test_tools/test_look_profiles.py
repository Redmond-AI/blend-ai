"""Focused tests for the client-side managed look-profile MCP surface."""

from __future__ import annotations

import base64
import hashlib
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

from mcp.types import CallToolResult, ImageContent, TextContent
from PIL import Image as PILImage
from pydantic import ValidationError as PydanticValidationError
import pytest

from blend_ai.tools.look_profiles import (
    AtmosphereComponentSpec,
    BatchRenderSettings,
    FrameSelection,
    LookProfileSpec,
    LookRenderBatchSpec,
    WorldProfileSpec,
    _send_look_profile_command,
    activate_look_profile,
    accept_look_profile,
    cancel_look_render_batch,
    get_look_profile_context,
    get_look_render_batch,
    get_look_render_result,
    inspect_look_review_state,
    submit_look_render_batch,
    upsert_look_profile,
)


@pytest.fixture
def mock_conn():
    connection = MagicMock()
    connection.send_command.return_value = {"status": "ok", "result": {"ok": True}}
    with patch("blend_ai.tools.look_profiles.get_connection", return_value=connection):
        yield connection


def _profile(**overrides):
    payload = {
        "profile_id": "summer-day",
        "display_name": "Summer Day",
        "seed": 42,
    }
    payload.update(overrides)
    return payload


def _batch(**overrides):
    payload = {
        "profile_ids": ["summer-day", "creepy-night"],
        "target_scenes": ["Scene Exterior Summer", "Scene Exterior Night"],
        "frames": {"frames": [1, 12]},
        "output_root": "/tmp/look-renders",
    }
    payload.update(overrides)
    return payload


def _png_base64(width=200, height=100):
    image = PILImage.new("RGBA", (width, height), (120, 80, 30, 180))
    output = BytesIO()
    image.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


class TestProfileSchemas:
    def test_minimal_profile_is_deterministic_and_forbids_unknown_fields(self):
        profile = LookProfileSpec.model_validate(_profile())

        assert profile.schema_version == 1
        assert profile.status == "DRAFT"
        assert profile.seed == 42
        assert profile.world.mode == "KEEP"

        with pytest.raises(PydanticValidationError, match="extra"):
            LookProfileSpec.model_validate(_profile(artist_python="bpy.ops.wm.save_mainfile()"))

    def test_render_profile_accepts_explicit_managed_dimensions(self):
        profile = LookProfileSpec.model_validate(
            _profile(
                render={
                    "engine": "CYCLES",
                    "samples": 64,
                    "denoise": True,
                    "resolution_x": 1920,
                    "resolution_y": 1080,
                    "resolution_percentage": 100,
                }
            )
        )

        assert profile.render.resolution_x == 1920
        assert profile.render.resolution_y == 1080

        with pytest.raises(PydanticValidationError, match="explicit render engine"):
            LookProfileSpec.model_validate(
                _profile(render={"engine": "KEEP", "resolution_x": 1920})
            )

    @pytest.mark.parametrize("profile_id", ["", "has space", "slash/path", "x" * 129])
    def test_profile_ids_are_strict(self, profile_id):
        with pytest.raises(PydanticValidationError):
            LookProfileSpec.model_validate(_profile(profile_id=profile_id))

    def test_managed_hdri_requires_absolute_hdr_or_exr(self):
        with pytest.raises(PydanticValidationError, match="required"):
            WorldProfileSpec(mode="MANAGED_HDRI", strength=1.0)
        with pytest.raises(PydanticValidationError, match="absolute"):
            WorldProfileSpec(mode="MANAGED_HDRI", hdri_path="sky.exr")
        with pytest.raises(PydanticValidationError, match=".hdr or .exr"):
            WorldProfileSpec(mode="MANAGED_HDRI", hdri_path="/tmp/sky.png")

        world = WorldProfileSpec(
            mode="MANAGED_HDRI",
            hdri_path="/tmp/sky.exr",
            strength=0.8,
        )
        assert world.hdri_path == str(Path("/tmp/sky.exr").resolve())

    def test_atmosphere_fields_are_discriminated(self):
        fog = AtmosphereComponentSpec(
            id="ground-fog",
            kind="FOG_VOLUME",
            density=0.006,
            anisotropy=0.25,
        )
        assert fog.density == 0.006

        with pytest.raises(PydanticValidationError, match="requires rain_rate"):
            AtmosphereComponentSpec(id="rain", kind="RAIN_RIG")
        with pytest.raises(PydanticValidationError, match="only valid for RAIN_RIG"):
            AtmosphereComponentSpec(
                id="fog",
                kind="FOG_VOLUME",
                density=0.01,
                rain_rate=200,
            )
        with pytest.raises(PydanticValidationError, match="less than or equal to 5000"):
            AtmosphereComponentSpec(
                id="rain",
                kind="RAIN_RIG",
                rain_rate=5001,
                drop_size_m=0.01,
                fall_speed_mps=12.0,
            )

    def test_duplicate_nested_component_ids_are_rejected(self):
        component = {"id": "fog", "kind": "FOG_VOLUME", "density": 0.01}
        with pytest.raises(PydanticValidationError, match="duplicate component"):
            LookProfileSpec.model_validate(
                _profile(atmosphere=[component, component])
            )

    def test_managed_post_is_explicit(self):
        with pytest.raises(PydanticValidationError, match="MANAGED_STACK"):
            LookProfileSpec.model_validate(
                _profile(post={"mode": "KEEP", "bloom": {"enabled": True}})
            )

    def test_camera_clone_does_not_accept_ambiguous_focus(self):
        with pytest.raises(PydanticValidationError, match="not both"):
            LookProfileSpec.model_validate(
                _profile(
                    camera={
                        "mode": "MANAGED_CLONE",
                        "source_camera": "Camera",
                        "focus_object": "Subject",
                        "focus_distance_m": 5.0,
                    }
                )
            )

    def test_camera_clone_preserves_exact_source_and_accepts_managed_reframe(self):
        profile = LookProfileSpec.model_validate(
            _profile(
                camera={
                    "mode": "MANAGED_CLONE",
                    "source_camera": "Camera ",
                    "location_world": [2.0, -22.0, 2.5],
                    "target_point": [-3.0, -3.0, 3.0],
                }
            )
        )

        assert profile.camera.source_camera == "Camera "
        assert profile.camera.location_world == (2.0, -22.0, 2.5)
        assert profile.camera.target_point == (-3.0, -3.0, 3.0)

    def test_review_intent_is_optional_strict_and_persisted(self):
        assert LookProfileSpec.model_validate(_profile()).review_intent is None
        profile = LookProfileSpec.model_validate(
            _profile(
                review_intent={
                    "summary": "Cold moonlit threat with readable silhouettes",
                    "expects_volume": True,
                }
            )
        )
        assert profile.review_intent is not None
        assert profile.review_intent.expects_shadows is True
        assert profile.review_intent.expects_volume is True
        with pytest.raises(PydanticValidationError, match="extra"):
            LookProfileSpec.model_validate(
                _profile(review_intent={"summary": "Night", "score": 10})
            )


class TestBoundaryAndProfileTools:
    @pytest.mark.parametrize(
        ("response", "message"),
        [
            (None, "invalid response"),
            ({"status": "busy"}, "unexpected status"),
            ({"status": "error", "result": "linked data"}, "linked data"),
        ],
    )
    def test_wire_response_shape_is_strict(self, mock_conn, response, message):
        mock_conn.send_command.return_value = response
        with pytest.raises(RuntimeError, match=message):
            _send_look_profile_command("test", {})

    def test_context_forwards_explicit_source_scene_profile_and_detail(self, mock_conn):
        result = get_look_profile_context(
            "Scene Exterior",
            profile_id="summer-day",
            detail="FULL",
        )

        assert result == {"ok": True}
        assert mock_conn.send_command.call_args.args == (
            "get_look_profile_context",
            {
                "source_scene": "Scene Exterior",
                "profile_id": "summer-day",
                "detail": "FULL",
            },
        )

    def test_context_without_profile_filter_requests_summary(self, mock_conn):
        get_look_profile_context("Scene Exterior")
        assert mock_conn.send_command.call_args.args[1] == {
            "source_scene": "Scene Exterior",
            "profile_id": None,
            "detail": "SUMMARY",
        }

    def test_review_state_forwards_explicit_identity_and_revisions(self, mock_conn):
        inspect_look_review_state(
            "summer-day",
            "Scene Exterior",
            "Camera A",
            expected_profile_revision=3,
            expected_geometry_revision=7,
        )
        assert mock_conn.send_command.call_args.args == (
            "inspect_look_review_state",
            {
                "profile_id": "summer-day",
                "target_scene": "Scene Exterior",
                "camera": "Camera A",
                "expected_profile_revision": 3,
                "expected_geometry_revision": 7,
            },
        )

    def test_upsert_validate_serializes_strict_profile_and_revisions(self, mock_conn):
        upsert_look_profile(
            "VALIDATE",
            "Scene Exterior",
            _profile(
                world={
                    "mode": "MANAGED_SKY",
                    "strength": 0.7,
                    "sun_elevation_degrees": 65.0,
                },
                lighting={
                    "lights": [
                        {
                            "id": "summer-sun",
                            "type": "SUN",
                            "location_world": [0, 0, 10],
                            "energy": 3.0,
                        }
                    ]
                },
            ),
            "CREATE_VERSION",
            expected_base_revision=2,
            expected_profile_revision=4,
            expected_geometry_revision=11,
        )

        command, params = mock_conn.send_command.call_args.args
        assert command == "upsert_look_profile"
        assert params["action"] == "VALIDATE"
        assert params["source_scene"] == "Scene Exterior"
        assert params["update_mode"] == "CREATE_VERSION"
        assert params["expected_base_revision"] == 2
        assert params["profile"]["profile_id"] == "summer-day"
        assert params["profile"]["lighting"]["lights"][0]["location_world"] == [
            0.0,
            0.0,
            10.0,
        ]
        assert params["expected_profile_revision"] == 4
        assert params["expected_geometry_revision"] == 11

    @pytest.mark.parametrize("window_index", [True, -1, 129, 1.5])
    def test_activation_window_index_is_bounded(self, mock_conn, window_index):
        with pytest.raises(ValueError, match="window_index"):
            activate_look_profile(
                "creepy-night",
                "Scene Exterior",
                window_index=window_index,
            )
        mock_conn.send_command.assert_not_called()

    def test_activation_forwards_profile_target_and_window(self, mock_conn):
        activate_look_profile(
            "creepy-night",
            "Scene Exterior",
            window_index=1,
            expected_profile_revision=8,
            expected_geometry_revision=20,
        )
        command, params = mock_conn.send_command.call_args.args
        assert command == "activate_look_profile"
        assert params["profile_id"] == "creepy-night"
        assert params["target_scene"] == "Scene Exterior"
        assert params["window_index"] == 1
        assert params["expected_profile_revision"] == 8
        assert params["expected_geometry_revision"] == 20

    def test_acceptance_requires_review_and_forwards_checksum_bound_evidence(self, mock_conn):
        with pytest.raises(ValueError, match="review_acknowledged"):
            accept_look_profile(
                "summer-day",
                "AI Look Summer",
                "lookbatch-1",
                "lookresult-1",
                "a" * 64,
                False,
            )
        mock_conn.send_command.assert_not_called()

        accept_look_profile(
            "summer-day",
            "AI Look Summer",
            "lookbatch-1",
            "lookresult-1",
            "a" * 64,
            True,
            expected_profile_revision=3,
            expected_geometry_revision=11,
        )

        assert mock_conn.send_command.call_args.args == (
            "accept_look_profile",
            {
                "profile_id": "summer-day",
                "target_scene": "AI Look Summer",
                "batch_id": "lookbatch-1",
                "result_id": "lookresult-1",
                "artifact_sha256": "a" * 64,
                "review_acknowledged": True,
                "expected_profile_revision": 3,
                "expected_geometry_revision": 11,
            },
        )

    def test_render_engine_supports_42_and_52_eevee_identifiers(self):
        for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
            profile = LookProfileSpec.model_validate(
                _profile(render={"engine": engine, "samples": 16})
            )
            assert profile.render.engine == engine

    def test_render_formats_reject_unsupported_color_depths(self):
        with pytest.raises(PydanticValidationError, match="JPEG"):
            BatchRenderSettings(file_format="JPEG", color_depth="16")
        with pytest.raises(PydanticValidationError, match="PNG"):
            BatchRenderSettings(file_format="PNG", color_depth="32")
        with pytest.raises(PydanticValidationError, match="OPEN_EXR"):
            BatchRenderSettings(file_format="OPEN_EXR", color_depth="8")


class TestRenderBatchSchemasAndTools:
    def test_frame_selection_requires_list_or_complete_range(self):
        with pytest.raises(PydanticValidationError, match="exactly one"):
            FrameSelection()
        with pytest.raises(PydanticValidationError, match="require start, end, and step"):
            FrameSelection(start=1, end=10)
        with pytest.raises(PydanticValidationError, match="must not contain duplicates"):
            FrameSelection(frames=[1, 1])

        selection = FrameSelection(start=1, end=10, step=3)
        assert selection.count == 4

    def test_batch_requires_explicit_targets_and_bounded_expansion(self):
        assert LookRenderBatchSpec.model_validate(_batch()).pairing == "PAIRWISE"
        with pytest.raises(PydanticValidationError, match="profile_ids"):
            LookRenderBatchSpec.model_validate(_batch(profile_ids=[]))
        with pytest.raises(PydanticValidationError, match="equal"):
            LookRenderBatchSpec.model_validate(
                _batch(pairing="PAIRWISE", target_scenes=["A", "B", "C"])
            )
        with pytest.raises(PydanticValidationError, match="maximum"):
            LookRenderBatchSpec.model_validate(
                _batch(frames={"start": 1, "end": 10000, "step": 1})
            )

    def test_batch_rejects_implicit_overwrite_and_unsafe_paths(self):
        with pytest.raises(PydanticValidationError, match="absolute"):
            LookRenderBatchSpec.model_validate(_batch(output_root="relative/output"))
        with pytest.raises(PydanticValidationError):
            LookRenderBatchSpec.model_validate(
                _batch(render_settings={"existing_file_policy": "OVERWRITE"})
            )

    def test_multi_camera_requires_collision_safe_template(self):
        with pytest.raises(PydanticValidationError, match="require.*camera"):
            LookRenderBatchSpec.model_validate(
                _batch(camera_names=["Camera A", "Camera B"])
            )
        batch = LookRenderBatchSpec.model_validate(
            _batch(
                camera_names=["Camera A", "Camera B"],
                filename_template="{scene}-{profile}-{camera}-{frame}",
                include_review_packet=True,
            )
        )
        assert batch.camera_names == ["Camera A", "Camera B"]
        assert batch.include_review_packet is True

        trailing = LookRenderBatchSpec.model_validate(
            _batch(camera_names=["Camera "])
        )
        assert trailing.camera_names == ["Camera "]

    def test_submit_forces_composite_restore_and_no_save(self, mock_conn):
        expected = {"Scene Exterior Summer": 9, "Scene Exterior Night": 12}
        submit_look_render_batch(_batch(), expected_profile_revisions=expected)

        command, params = mock_conn.send_command.call_args.args
        assert command == "submit_look_render_batch"
        assert params["batch"]["profile_ids"] == ["summer-day", "creepy-night"]
        assert params["batch"]["target_scenes"] == [
            "Scene Exterior Summer",
            "Scene Exterior Night",
        ]
        assert params["output_pass"] == "COMPOSITE"
        assert params["restore_scene_state"] is True
        assert params["save_blend"] is False
        assert params["expected_profile_revision"] is None
        assert params["expected_profile_revisions"] == expected

    def test_scalar_revision_is_rejected_for_multiple_pairs(self):
        with pytest.raises(ValueError, match="Scalar expected_profile_revision"):
            submit_look_render_batch(_batch(), expected_profile_revision=9)

        with pytest.raises(ValueError, match="exactly one entry"):
            submit_look_render_batch(
                _batch(),
                expected_profile_revisions={"Scene Exterior Summer": 9},
            )

    def test_cross_product_rejects_multiple_profile_owners(self):
        with pytest.raises(PydanticValidationError, match="exactly one profile_id"):
            LookRenderBatchSpec.model_validate(_batch(pairing="CROSS_PRODUCT"))

    def test_get_and_cancel_batch_use_bounded_identifiers(self, mock_conn):
        get_look_render_batch("batch-1", include_results=True, max_results=25)
        assert mock_conn.send_command.call_args.args == (
            "get_look_render_batch",
            {
                "batch_id": "batch-1",
                "include_results": True,
                "max_results": 25,
                "cursor": None,
            },
        )

        cancel_look_render_batch("batch-1")
        assert mock_conn.send_command.call_args.args == (
            "cancel_look_render_batch",
            {"batch_id": "batch-1"},
        )


class TestCompositeRenderResult:
    def _wire_result(self, *, status="SUCCEEDED", image=True, **metadata_overrides):
        succeeded = status == "SUCCEEDED"
        encoded_image = _png_base64() if succeeded else None
        metadata = {
            "batch_id": "batch-1",
            "result_id": "result-1",
            "status": status,
            "profile_id": "summer-day",
            "target_scene": "Scene Exterior",
            "frame": 1,
            "output_pass": "COMPOSITE",
            "output_path": "/tmp/look-renders/summer.png",
            "error": None,
            "image_source": "COMPOSITE_OUTPUT" if succeeded else None,
            "artifact_sha256": "a" * 64 if succeeded else None,
            "artifact_byte_count": 123456 if succeeded else None,
            "source_width": 200 if succeeded else None,
            "source_height": 100 if succeeded else None,
            "proxy_width": 200 if succeeded else None,
            "proxy_height": 100 if succeeded else None,
            "proxy_format": "PNG" if succeeded else None,
            "proxy_byte_count": (
                len(base64.b64decode(encoded_image)) if encoded_image is not None else None
            ),
            "scene": "Scene Exterior" if succeeded else None,
            "view_layer": "ViewLayer" if succeeded else None,
            "camera": "Camera" if succeeded else None,
            "engine": "CYCLES" if succeeded else None,
            "samples": 64 if succeeded else None,
            "denoise": True if succeeded else None,
            "color_management": (
                {
                    "view_transform": "AgX",
                    "look": "Medium High Contrast",
                    "exposure": 0.0,
                    "gamma": 1.0,
                }
                if succeeded
                else None
            ),
            "compositor_hash": "b" * 64 if succeeded else None,
            "compositor_adapter": (
                "SCENE_COMPOSITING_NODE_GROUP" if succeeded else None
            ),
            "source_scene": "Scene Base" if succeeded else None,
            "profile_version": 1 if succeeded else None,
            "profile_revision": 2 if succeeded else None,
            "profile_status": "DRAFT" if succeeded else None,
            "definition_hash": "c" * 64 if succeeded else None,
            "base_revision": 1234 if succeeded else None,
            "geometry_revision": 41 if succeeded else None,
            "base_fingerprint": "sha256:" + "d" * 64 if succeeded else None,
            "geometry_fingerprint": "sha256:" + "e" * 64 if succeeded else None,
            "manifest_entry_hash": "f" * 64 if succeeded else None,
            "submitted_at": 10.0 if succeeded else None,
            "started_at": 11.0 if succeeded else None,
            "completed_at": 12.0 if succeeded else None,
            "warnings": [],
        }
        metadata.update(metadata_overrides)
        result = {"metadata": metadata}
        if image:
            result["image_base64"] = encoded_image or _png_base64()
        return {"status": "ok", "result": result}

    def test_success_returns_native_composite_proxy_and_metadata(self, mock_conn):
        mock_conn.send_command.return_value = self._wire_result()

        result = get_look_render_result(
            "batch-1",
            "result-1",
            max_size=64,
            format="JPEG",
        )

        assert isinstance(result, CallToolResult)
        assert isinstance(result.content[0], ImageContent)
        assert result.content[0].mimeType == "image/jpeg"
        assert isinstance(result.content[1], TextContent)
        assert result.structuredContent["output_pass"] == "COMPOSITE"
        assert result.structuredContent["source_width"] == 200
        assert result.structuredContent["source_height"] == 100
        assert result.structuredContent["proxy_width"] == 64
        assert result.structuredContent["proxy_height"] == 32
        assert result.structuredContent["proxy_format"] == "JPEG"
        assert result.structuredContent["image_source"] == "COMPOSITE_OUTPUT"
        assert result.structuredContent["artifact_sha256"] == "a" * 64
        assert result.structuredContent["compositor_hash"] == "b" * 64
        assert len(result.structuredContent["content_sha256"]) == 64
        assert mock_conn.send_command.call_args.args == (
            "get_look_render_result",
            {
                "batch_id": "batch-1",
                "result_id": "result-1",
                "output_pass": "COMPOSITE",
                "proxy_max_size": 64,
            },
        )

        with PILImage.open(BytesIO(base64.b64decode(result.content[0].data))) as image:
            assert image.format == "JPEG"
            assert image.size == (64, 32)

    def test_review_result_returns_beauty_then_labeled_contact_sheet(self, mock_conn):
        slugs = [
            "combined",
            "diffuse_direct",
            "glossy",
            "emission",
            "depth",
            "cryptomatte_object",
        ]
        labels = [
            "Combined (pre-compositor)",
            "Diffuse Direct",
            "Glossy Direct + Indirect",
            "Emission",
            "Camera Depth (near=white)",
            "Object Cryptomatte",
        ]
        tile_bytes = [
            base64.b64decode(_png_base64(64, 36)) for _slug in slugs
        ]
        packet = {
            "schema_version": 1,
            "image_source": "SAME_RENDER_RESULT_PASSES",
            "tile_order": slugs,
            "tiles": [
                {
                    "slug": slug,
                    "label": label,
                    "available": True,
                    "artifact_sha256": hashlib.sha256(data).hexdigest(),
                    "artifact_byte_count": len(data),
                    "width": 64,
                    "height": 36,
                    "mean_energy": 0.1,
                    "nonzero_coverage": 1.0,
                    "magenta_coverage": 0.0,
                }
                for slug, label, data in zip(slugs, labels, tile_bytes, strict=True)
            ],
            "technical_check": {"status": "PASS", "failures": [], "warnings": []},
            "review_intent": {
                "summary": "Cold moonlight",
                "expects_shadows": True,
                "expects_reflections": True,
                "expects_volume": False,
                "expects_compositing": True,
            },
            "editable_controls": {"world.strength": 0.4},
            "beauty_magenta_coverage": 0.0,
        }
        wire = self._wire_result(review_packet=packet)
        wire["result"]["review_tiles"] = [
            {"slug": slug, "image_base64": base64.b64encode(data).decode("ascii")}
            for slug, data in zip(slugs, tile_bytes, strict=True)
        ]
        mock_conn.send_command.return_value = wire

        result = get_look_render_result("batch-1", "result-1", max_size=128)

        assert len(result.content) == 3
        assert isinstance(result.content[0], ImageContent)
        assert isinstance(result.content[1], ImageContent)
        assert result.content[1].mimeType == "image/png"
        assert isinstance(result.content[2], TextContent)
        review = result.structuredContent["review_packet"]
        assert review["contact_sheet_width"] == 1536
        assert review["contact_sheet_height"] == 640
        assert len(review["contact_sheet_sha256"]) == 64

    def test_pending_result_returns_metadata_without_image(self, mock_conn):
        mock_conn.send_command.return_value = self._wire_result(
            status="RUNNING",
            image=False,
            output_path=None,
        )

        result = get_look_render_result("batch-1", "result-1")

        assert len(result.content) == 1
        assert isinstance(result.content[0], TextContent)
        assert result.structuredContent["status"] == "RUNNING"

    def test_skipped_result_returns_non_acceptable_metadata_without_image(self, mock_conn):
        mock_conn.send_command.return_value = self._wire_result(
            status="SKIPPED",
            image=False,
        )

        result = get_look_render_result("batch-1", "result-1")

        assert len(result.content) == 1
        assert result.structuredContent["status"] == "SKIPPED"
        assert result.structuredContent["image_source"] is None
        assert result.structuredContent["artifact_sha256"] is None

        mock_conn.send_command.return_value = self._wire_result(
            status="SKIPPED",
            image=False,
            image_source="COMPOSITE_OUTPUT",
        )
        with pytest.raises(PydanticValidationError, match="must not carry"):
            get_look_render_result("batch-1", "result-1")

    @pytest.mark.parametrize(
        ("wire", "message"),
        [
            ({"metadata": {}}, "validation"),
            (
                {
                    "metadata": {
                        "batch_id": "batch-1",
                        "result_id": "result-1",
                        "status": "SUCCEEDED",
                        "profile_id": "summer-day",
                        "target_scene": "Scene Exterior",
                        "frame": 1,
                        "output_pass": "COMPOSITE",
                    }
                },
                "provenance fields",
            ),
        ],
    )
    def test_success_requires_strict_metadata_and_image(self, mock_conn, wire, message):
        mock_conn.send_command.return_value = {"status": "ok", "result": wire}
        with pytest.raises((RuntimeError, PydanticValidationError), match=message):
            get_look_render_result("batch-1", "result-1")

    def test_success_with_valid_provenance_still_requires_image(self, mock_conn):
        mock_conn.send_command.return_value = self._wire_result(image=False)
        with pytest.raises(RuntimeError, match="image_base64"):
            get_look_render_result("batch-1", "result-1")

    def test_non_success_result_rejects_image(self, mock_conn):
        mock_conn.send_command.return_value = self._wire_result(status="FAILED", image=True)
        with pytest.raises(RuntimeError, match="must not include"):
            get_look_render_result("batch-1", "result-1")

    def test_result_must_be_final_composite(self, mock_conn):
        mock_conn.send_command.return_value = self._wire_result(output_pass="BEAUTY")
        with pytest.raises(PydanticValidationError, match="COMPOSITE"):
            get_look_render_result("batch-1", "result-1")
