"""Tests for the spatial relighting MCP tools and native viewport images."""

from __future__ import annotations

import asyncio
import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Annotated
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, ImageContent, TextContent
from PIL import Image as PILImage
from pydantic import ValidationError as PydanticValidationError

from blend_ai.tools.relighting import (
    BUILT_IN_SEMANTIC_TERMS,
    CaptureMetadata,
    LightSpec,
    RaySpec,
    SceneOverrides,
    _crop_box,
    _capture_macos_window,
    _capture_result,
    _decode_capture_image,
    _encode_capture_image,
    _stage_capture_bytes,
    _send_relighting_command,
    _validate_identifier,
    _validate_safe_strings,
    apply_light_plan,
    batch_raycast,
    capture_cycles_viewport,
    get_lighting_context,
)
from blend_ai.validators import ValidationError

import blend_ai.tools.relighting as relighting_module


def _png_base64(width: int = 200, height: int = 100, mode: str = "RGB") -> str:
    color = (170, 20, 30, 180) if mode == "RGBA" else (170, 20, 30)
    image = PILImage.new(mode, (width, height), color)
    output = BytesIO()
    image.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


@pytest.fixture
def mock_conn():
    connection = MagicMock()
    connection.send_command.return_value = {"status": "ok", "result": {"ok": True}}
    with patch("blend_ai.tools.relighting.get_connection", return_value=connection):
        yield connection


class TestInternalBoundaryValidation:
    @pytest.mark.parametrize("value", [None, "", "x" * 129, "has space"])
    def test_identifier_rejects_invalid_values(self, value):
        with pytest.raises(ValueError):
            _validate_identifier(value, name="id")

    @pytest.mark.parametrize(
        "values",
        ["not-a-list", [""], ["x" * 129], ["line\nbreak"]],
    )
    def test_safe_string_lists_reject_invalid_values(self, values):
        with pytest.raises(ValidationError):
            _validate_safe_strings(values, name="patterns", max_count=4)

    @pytest.mark.parametrize(
        ("response", "message"),
        [
            (None, "invalid response"),
            ({"status": "busy"}, "unexpected status"),
        ],
    )
    def test_wire_response_shape_is_strict(self, mock_conn, response, message):
        mock_conn.send_command.return_value = response
        with pytest.raises(RuntimeError, match=message):
            _send_relighting_command("test", {})

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"image_base64": ""},
            {"image_base64": base64.b64encode(b"").decode("ascii")},
        ],
    )
    def test_capture_image_payload_must_be_nonempty(self, payload):
        with pytest.raises(RuntimeError):
            _decode_capture_image(payload)

    def test_macos_capture_translates_title_bar_and_retina_crop(self):
        def fake_run(arguments, **_kwargs):
            output_path = Path(arguments[-1])
            PILImage.new("RGB", (400, 264), (20, 30, 40)).save(
                output_path,
                format="PNG",
            )
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "windowID": 77,
                        "ownerPID": 42,
                        "width": 400,
                        "height": 264,
                        "bounds": {
                            "x": 0,
                            "y": 30,
                            "width": 200,
                            "height": 132,
                        },
                    }
                ),
                stderr="",
            )

        target = {
            "process_id": 42,
            "window_content_width": 200,
            "window_content_height": 100,
            "window_x": 0,
            "window_y": 0,
            "region_rect": {
                "x": 50,
                "y": 10,
                "width": 100,
                "height": 80,
                "origin": "TOP_LEFT",
            },
        }
        with (
            patch(
                "blend_ai.tools.relighting._ensure_macos_capture_helper",
                return_value=Path("/tmp/fake-capture-helper"),
            ),
            patch("blend_ai.tools.relighting.subprocess.run", side_effect=fake_run),
        ):
            data, width, height, region, metadata = _capture_macos_window(target)

        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        assert (width, height) == (400, 264)
        # Retina scale is 2x; 64 pixels of top chrome remain above content.
        assert region == {
            "x": 100,
            "y": 84,
            "width": 200,
            "height": 160,
            "origin": "TOP_LEFT",
        }
        assert metadata["windowID"] == 77

    def test_macos_capture_rejects_wrong_blender_window_metadata(self):
        def fake_run(arguments, **_kwargs):
            output_path = Path(arguments[-1])
            PILImage.new("RGB", (200, 132), (20, 30, 40)).save(
                output_path,
                format="PNG",
            )
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "windowID": 88,
                        "ownerPID": 42,
                        "width": 200,
                        "height": 132,
                        "bounds": {
                            "x": 900,
                            "y": 900,
                            "width": 200,
                            "height": 132,
                        },
                    }
                ),
                stderr="",
            )

        target = {
            "process_id": 42,
            "window_content_width": 200,
            "window_content_height": 100,
            "window_x": 0,
            "window_y": 0,
            "region_rect": {
                "x": 50,
                "y": 10,
                "width": 100,
                "height": 80,
                "origin": "TOP_LEFT",
            },
        }
        with (
            patch(
                "blend_ai.tools.relighting._ensure_macos_capture_helper",
                return_value=Path("/tmp/fake-capture-helper"),
            ),
            patch("blend_ai.tools.relighting.subprocess.run", side_effect=fake_run),
        ):
            with pytest.raises(RuntimeError, match="outside the requested"):
                _capture_macos_window(target)

    @pytest.mark.parametrize(
        "rect",
        [
            "bad",
            {},
            {"x": 0, "y": 0, "width": 0, "height": 1},
            {"x": 0, "y": 0, "width": 1, "height": 1, "origin": "CENTER"},
        ],
    )
    def test_crop_metadata_is_strict(self, rect):
        with pytest.raises(RuntimeError):
            _crop_box(rect, 10, 10)

    def test_bottom_left_crop_is_converted(self):
        assert _crop_box(
            {"x": 1, "y": 2, "width": 3, "height": 4, "origin": "BOTTOM_LEFT"},
            10,
            10,
        ) == (1, 4, 4, 8)

    def test_image_decoder_rejects_signature_and_pixel_bomb(self, monkeypatch):
        with pytest.raises(RuntimeError, match="supported image"):
            _encode_capture_image(
                b"not an image",
                source_width=None,
                source_height=None,
                region_rect=None,
                max_size=64,
                output_format="PNG",
                jpeg_quality=85,
            )

    def test_macos_helper_compiles_once_and_is_removed(self, tmp_path, monkeypatch):
        source = tmp_path / "capture.swift"
        source.write_text('print("capture")', encoding="utf-8")

        def fake_compile(arguments, **_kwargs):
            Path(arguments[-1]).write_bytes(b"compiled helper")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(relighting_module, "_MACOS_CAPTURE_SOURCE", source)
        monkeypatch.setattr(relighting_module, "_MACOS_CAPTURE_HELPER", None)
        monkeypatch.setattr(relighting_module.tempfile, "gettempdir", lambda: str(tmp_path))
        monkeypatch.setattr(relighting_module.shutil, "which", lambda _name: "/usr/bin/swiftc")
        monkeypatch.setattr(relighting_module.subprocess, "run", fake_compile)
        register = MagicMock()
        monkeypatch.setattr(relighting_module.atexit, "register", register)

        helper = relighting_module._ensure_macos_capture_helper()

        assert helper.is_file()
        assert relighting_module._ensure_macos_capture_helper() == helper
        register.assert_called_once_with(relighting_module._remove_macos_capture_helper)
        relighting_module._remove_macos_capture_helper()
        assert not helper.exists()
        assert relighting_module._MACOS_CAPTURE_HELPER is None

        monkeypatch.setattr(relighting_module, "MAX_IMAGE_PIXELS", 1)
        with pytest.raises(RuntimeError, match="pixel limit"):
            _encode_capture_image(
                base64.b64decode(_png_base64(2, 2)),
                source_width=2,
                source_height=2,
                region_rect=None,
                max_size=64,
                output_format="PNG",
                jpeg_quality=85,
            )


class TestGetLightingContext:
    def test_forwards_approved_defaults(self, mock_conn):
        result = get_lighting_context()

        assert result == {"ok": True}
        mock_conn.send_command.assert_called_once_with(
            "get_lighting_context",
            {
                "camera_name": None,
                "scope": "CAMERA",
                "collection_names": [],
                "detail": "CANDIDATES",
                "semantic_terms": BUILT_IN_SEMANTIC_TERMS,
                "include_hidden": False,
                "max_instances": 500,
                "cursor": None,
                "max_surface_triangles": 200_000,
                "cache_mode": "USE",
            },
        )

    def test_forwards_collection_request(self, mock_conn):
        get_lighting_context(
            camera_name="Warehouse Camera",
            scope="COLLECTIONS",
            collection_names=["Shell", "Props"],
            detail="BOUNDS",
            semantic_terms=["skylight"],
            include_hidden=True,
            max_instances=1000,
            cursor="opaque:2",
            max_surface_triangles=500_000,
            cache_mode="REFRESH",
        )

        params = mock_conn.send_command.call_args.args[1]
        assert params["collection_names"] == ["Shell", "Props"]
        assert params["camera_name"] == "Warehouse Camera"
        assert params["cache_mode"] == "REFRESH"

    def test_preserves_trailing_camera_space(self, mock_conn):
        get_lighting_context(camera_name="Camera ")

        params = mock_conn.send_command.call_args.args[1]
        assert params["camera_name"] == "Camera "

    def test_collections_scope_requires_names(self, mock_conn):
        with pytest.raises(ValidationError, match="collection_names"):
            get_lighting_context(scope="COLLECTIONS")
        mock_conn.send_command.assert_not_called()

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"max_instances": 5_001}, "max_instances"),
            ({"max_surface_triangles": 1_000_001}, "max_surface_triangles"),
            ({"cursor": "x\n"}, "control"),
            ({"semantic_terms": ["x"] * 65}, "semantic_terms"),
        ],
    )
    def test_bounded_inputs(self, mock_conn, kwargs, message):
        with pytest.raises(ValidationError, match=message):
            get_lighting_context(**kwargs)
        mock_conn.send_command.assert_not_called()

    def test_blender_error_is_raised(self, mock_conn):
        mock_conn.send_command.return_value = {"status": "error", "result": "stale"}
        with pytest.raises(RuntimeError, match="stale"):
            get_lighting_context()

    def test_rejects_non_boolean_and_invalid_result(self, mock_conn):
        with pytest.raises(ValidationError, match="include_hidden"):
            get_lighting_context(include_hidden=1)
        mock_conn.send_command.return_value = {"status": "ok", "result": []}
        with pytest.raises(RuntimeError, match="invalid lighting context"):
            get_lighting_context()


class TestRaySpec:
    def test_target_form(self):
        ray = RaySpec(id="to_floor", origin=[0, 0, 2], target=[0, 0, 0])
        assert ray.target == (0.0, 0.0, 0.0)
        assert ray.direction is None

    def test_direction_form(self):
        ray = RaySpec(
            id="moon_path",
            origin=[0, 0, 2],
            direction=[0, 0, -1],
            max_distance=100,
        )
        assert ray.max_distance == 100.0

    def test_explicit_optional_none_values_are_supported(self):
        ray = RaySpec(
            id="to_floor",
            origin=[0, 0, 2],
            target=[0, 0, 0],
            direction=None,
            max_distance=None,
        )
        assert ray.direction is None
        assert ray.max_distance is None

    def test_target_form_rejects_max_distance(self):
        with pytest.raises(PydanticValidationError, match="only valid with direction"):
            RaySpec(
                id="bad_target",
                origin=[0, 0, 2],
                target=[0, 0, 0],
                max_distance=3,
            )

    @pytest.mark.parametrize(
        "payload",
        [
            {"id": "bad", "origin": [0, 0, 0]},
            {
                "id": "bad",
                "origin": [0, 0, 0],
                "target": [0, 0, 1],
                "direction": [0, 0, 1],
            },
            {"id": "bad", "origin": [0, 0, 0], "direction": [0, 0, 1]},
            {"id": "bad", "origin": [0, 0, 0], "direction": [0, 0, 0], "max_distance": 1},
            {"id": "bad", "origin": [0, 0, 0], "target": [0, 0, 0]},
            {"id": "bad", "origin": [0, float("nan"), 0], "target": [0, 0, 1]},
        ],
    )
    def test_invalid_forms(self, payload):
        with pytest.raises(PydanticValidationError):
            RaySpec.model_validate(payload)


class TestBatchRaycast:
    def test_serializes_models_and_defaults(self, mock_conn):
        batch_raycast(
            [
                RaySpec(id="a", origin=[0, 0, 0], target=[0, 0, 1]),
                {
                    "id": "b",
                    "origin": [1, 0, 0],
                    "direction": [0, 0, 1],
                    "max_distance": 10,
                },
            ],
            expected_geometry_revision=7,
        )

        command, params = mock_conn.send_command.call_args.args
        assert command == "batch_raycast"
        assert params["max_hits"] == 8
        assert params["include_ignored_hits"] is True
        assert params["time_budget_ms"] == 2000
        assert params["expected_geometry_revision"] == 7
        assert params["rays"][0] == {
            "id": "a",
            "origin": [0.0, 0.0, 0.0],
            "target": [0.0, 0.0, 1.0],
        }
        assert params["rays"][1]["max_distance"] == 10.0

    def test_duplicate_ids_rejected(self, mock_conn):
        rays = [
            RaySpec(id="same", origin=[0, 0, 0], target=[0, 0, 1]),
            RaySpec(id="same", origin=[1, 0, 0], target=[1, 0, 1]),
        ]
        with pytest.raises(ValidationError, match="unique"):
            batch_raycast(rays)
        mock_conn.send_command.assert_not_called()

    def test_empty_and_oversized_batches_rejected(self, mock_conn):
        with pytest.raises(ValidationError, match="at least one"):
            batch_raycast([])
        ray = RaySpec(id="a", origin=[0, 0, 0], target=[0, 0, 1])
        with pytest.raises(ValidationError, match="at most"):
            batch_raycast([ray] * 513)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"max_hits": 33}, "max_hits"),
            ({"time_budget_ms": 9}, "time_budget_ms"),
            ({"ignore_object_patterns": ["x"] * 33}, "ignore_object_patterns"),
            ({"expected_geometry_revision": -1}, "expected_geometry_revision"),
        ],
    )
    def test_limits(self, mock_conn, kwargs, message):
        ray = RaySpec(id="a", origin=[0, 0, 0], target=[0, 0, 1])
        with pytest.raises(ValidationError, match=message):
            batch_raycast([ray], **kwargs)
        mock_conn.send_command.assert_not_called()

    def test_boolean_and_result_shape_are_strict(self, mock_conn):
        ray = RaySpec(id="a", origin=[0, 0, 0], target=[0, 0, 1])
        with pytest.raises(ValidationError, match="include_ignored_hits"):
            batch_raycast([ray], include_ignored_hits=1)
        mock_conn.send_command.return_value = {"status": "ok", "result": []}
        with pytest.raises(RuntimeError, match="invalid raycast"):
            batch_raycast([ray])


class TestLightSpec:
    def test_area_light(self):
        light = LightSpec(
            id="moon_fill",
            type="AREA",
            location_world=[0, 0, 10],
            target_point=[0, 0, 0],
            area_shape="RECTANGLE",
            size=5,
            size_y=2,
            energy=1200,
            color_rgb=[0.4, 0.5, 1.0],
        )
        assert light.size_y == 2.0

    def test_point_rejects_target(self):
        with pytest.raises(PydanticValidationError, match="do not accept targets"):
            LightSpec(
                id="red_point",
                type="POINT",
                location_world=[0, 0, 1],
                target_point=[0, 0, 0],
            )

    @pytest.mark.parametrize("spot_angle", [0.0, 0.5, 0.999999])
    def test_spot_angle_respects_blender_one_degree_minimum(self, spot_angle):
        with pytest.raises(PydanticValidationError, match="spot_angle_degrees"):
            LightSpec(
                id="tight_spot",
                type="SPOT",
                location_world=[0, 0, 2],
                spot_angle_degrees=spot_angle,
            )

    def test_explicit_optional_none_values_are_supported(self):
        light = LightSpec(
            id="red_point",
            name=None,
            type="POINT",
            location_world=[0, 0, 1],
            target_point=None,
            target_object=None,
            radius=None,
            sun_angle_degrees=None,
            area_shape=None,
            size=None,
            size_y=None,
            spot_angle_degrees=None,
            spot_blend=None,
        )
        assert light.name is None
        assert light.radius is None

    def test_world_override_and_scene_optional_validation(self):
        world = relighting_module.WorldOverride(
            mode="KEEP",
            color_rgb=[0.0, 0.0, 0.0],
            strength=0,
        )
        assert world.strength == 0.0
        scene = SceneOverrides(exposure=None)
        assert scene.exposure is None
        with pytest.raises(PydanticValidationError):
            relighting_module.WorldOverride(color_rgb=[1, 1, 1, 1])

    @pytest.mark.parametrize(
        "payload",
        [
            {"id": "bad", "type": "POINT", "location_world": [0, 0, 0], "use_shadow": 1},
            {"id": "bad", "type": "POINT", "location_world": [0, 0, 0], "name": "bad\nname"},
            {
                "id": "bad",
                "type": "POINT",
                "location_world": [0, 0, 0],
                "color_rgb": [1, 1, 1, 1],
            },
            {
                "id": "bad",
                "type": "POINT",
                "location_world": [0, 0, 0],
                "target_point": [0, 0, 1],
                "target_object": "Target",
            },
            {
                "id": "bad",
                "type": "POINT",
                "location_world": [0, 0, 0],
                "area_shape": "SQUARE",
            },
            {
                "id": "bad",
                "type": "AREA",
                "location_world": [0, 0, 0],
                "radius": 1,
            },
        ],
    )
    def test_additional_type_and_field_guards(self, payload):
        with pytest.raises(PydanticValidationError):
            LightSpec.model_validate(payload)

    @pytest.mark.parametrize(
        "payload",
        [
            {
                "id": "bad",
                "type": "POINT",
                "location_world": [0, 0, 0],
                "sun_angle_degrees": 1,
            },
            {
                "id": "bad",
                "type": "SUN",
                "location_world": [0, 0, 0],
                "spot_blend": 0.5,
            },
            {
                "id": "bad",
                "type": "AREA",
                "location_world": [0, 0, 0],
                "area_shape": "SQUARE",
                "size_y": 2,
            },
            {
                "id": "bad",
                "type": "POINT",
                "location_world": [0, 0, 0],
                "color_rgb": [2, 0, 0],
            },
        ],
    )
    def test_type_specific_validation(self, payload):
        with pytest.raises(PydanticValidationError):
            LightSpec.model_validate(payload)


class TestApplyLightPlan:
    def test_apply_serializes_plan(self, mock_conn):
        result = apply_light_plan(
            "APPLY",
            plan_id="warehouse-night-v1",
            expected_geometry_revision=12,
            lights=[
                {
                    "id": "moon",
                    "name": "Moon Area",
                    "type": "AREA",
                    "location_world": [0, 0, 10],
                    "target_object": "Floor",
                    "area_shape": "RECTANGLE",
                    "size": 8,
                    "size_y": 2,
                    "energy": 1500,
                    "color_rgb": [0.35, 0.45, 1.0],
                }
            ],
            remove_ids=["old_red"],
            scene_overrides={
                "existing_light_policy": "MUTE_NON_MANAGED",
                "world": {
                    "mode": "MANAGED_SOLID",
                    "color_rgb": [0.0, 0.0, 0.01],
                    "strength": 0.02,
                },
                "exposure": -1.0,
            },
        )

        assert result == {"ok": True}
        command, params = mock_conn.send_command.call_args.args
        assert command == "apply_light_plan"
        assert params["action"] == "APPLY"
        assert params["mode"] == "PATCH_MANAGED"
        assert params["lights"][0]["location_world"] == [0.0, 0.0, 10.0]
        assert params["scene_overrides"]["world"]["strength"] == 0.02
        assert params["transaction_id"] is None

    def test_validate_uses_same_envelope(self, mock_conn):
        apply_light_plan(
            "VALIDATE",
            plan_id="check",
            lights=[LightSpec(id="red", type="POINT", location_world=[0, 0, 1])],
        )
        assert mock_conn.send_command.call_args.args[1]["action"] == "VALIDATE"

    def test_rollback_requires_and_forwards_transaction(self, mock_conn):
        apply_light_plan("ROLLBACK", transaction_id="txn-123")
        params = mock_conn.send_command.call_args.args[1]
        assert params["action"] == "ROLLBACK"
        assert params["transaction_id"] == "txn-123"
        assert params["lights"] == []

    def test_rollback_can_select_latest_transaction_for_plan(self, mock_conn):
        apply_light_plan("ROLLBACK", plan_id="warehouse-night-v1")
        params = mock_conn.send_command.call_args.args[1]
        assert params["plan_id"] == "warehouse-night-v1"
        assert params["transaction_id"] is None

    @pytest.mark.parametrize(
        ("args", "kwargs", "message"),
        [
            (("APPLY",), {}, "plan_id"),
            (("ROLLBACK",), {}, "transaction_id or plan_id"),
            (("APPLY",), {"plan_id": "p", "transaction_id": "t"}, "ROLLBACK"),
        ],
    )
    def test_action_discrimination(self, mock_conn, args, kwargs, message):
        with pytest.raises(ValidationError, match=message):
            apply_light_plan(*args, **kwargs)
        mock_conn.send_command.assert_not_called()

    def test_duplicate_and_overlapping_ids_rejected(self, mock_conn):
        lights = [
            LightSpec(id="red", type="POINT", location_world=[0, 0, 1]),
            LightSpec(id="red", type="POINT", location_world=[1, 0, 1]),
        ]
        with pytest.raises(ValidationError, match="unique"):
            apply_light_plan("APPLY", plan_id="p", lights=lights)
        with pytest.raises(ValidationError, match="both"):
            apply_light_plan("APPLY", plan_id="p", lights=lights[:1], remove_ids=["red"])

    def test_scene_override_bounds(self):
        with pytest.raises(PydanticValidationError):
            SceneOverrides(exposure=float("nan"))

    def test_strict_and_result_shape_are_validated(self, mock_conn):
        with pytest.raises(ValidationError, match="strict"):
            apply_light_plan("APPLY", plan_id="p", strict=1)
        mock_conn.send_command.return_value = {"status": "ok", "result": []}
        with pytest.raises(RuntimeError, match="invalid light plan"):
            apply_light_plan("VALIDATE", plan_id="p")


def _capture_side_effect(*, region_rect=None, mode="RGB"):
    def respond(command, params):
        if command == "prepare_cycles_viewport":
            return {
                "status": "ok",
                "result": {
                    "session_id": "preview-123",
                    "workspace_name": "AI Preview",
                    "area_index": 0,
                    "camera_name": "Camera",
                    "engine": "CYCLES",
                    "preview_samples": 16,
                    "denoise": True,
                    "device": "GPU",
                },
            }
        if command == "capture_cycles_viewport_area":
            return {
                "status": "ok",
                "result": {
                    "session_id": "preview-123",
                    "image_base64": _png_base64(200, 100, mode=mode),
                    "source_width": 200,
                    "source_height": 100,
                    "region_rect": region_rect,
                },
            }
        if command == "restore_cycles_viewport":
            return {
                "status": "ok",
                "result": {"session_id": params["session_id"], "restored": True},
            }
        raise AssertionError(command)

    return respond


class TestCaptureCyclesViewport:
    @pytest.mark.asyncio
    async def test_two_phase_capture_returns_native_jpeg_and_metadata(self, mock_conn):
        mock_conn.send_command.side_effect = _capture_side_effect(
            region_rect={"x": 50, "y": 0, "width": 100, "height": 100, "origin": "TOP_LEFT"}
        )
        with patch("blend_ai.tools.relighting.asyncio.sleep", new=AsyncMock()) as sleep:
            result = await capture_cycles_viewport(max_size=64)

        assert isinstance(result, CallToolResult)
        assert isinstance(result.content[0], ImageContent)
        assert result.content[0].mimeType == "image/jpeg"
        assert isinstance(result.content[1], TextContent)
        assert result.structuredContent["action"] == "CAPTURE"
        assert result.structuredContent["session_id"] == "preview-123"
        assert result.structuredContent["source_width"] == 200
        assert result.structuredContent["output_width"] == 64
        assert result.structuredContent["output_height"] == 64
        assert result.structuredContent["keep_session"] is True
        assert result.structuredContent["restored"] is False
        assert result.structuredContent["requested_preview_samples"] == 16
        assert result.structuredContent["settle_basis"] == "elapsed"
        assert result.structuredContent["warnings"] == []
        assert json.loads(result.content[1].text) == result.structuredContent
        sleep.assert_awaited_once_with(2.0)

        image_bytes = base64.b64decode(result.content[0].data)
        with PILImage.open(BytesIO(image_bytes)) as image:
            assert image.format == "JPEG"
            assert image.size == (64, 64)

        calls = mock_conn.send_command.call_args_list
        assert [call.args[0] for call in calls] == [
            "prepare_cycles_viewport",
            "capture_cycles_viewport_area",
        ]
        assert calls[0].args[1] == {
            "session_id": None,
            "workspace_name": "AI Preview",
            "area_index": None,
            "camera_name": None,
            "preview_samples": 16,
            "denoise": True,
            "device": "GPU",
            "keep_session": True,
        }
        assert calls[1].args[1] == {"session_id": "preview-123", "keep_session": True}

    @pytest.mark.asyncio
    async def test_png_capture_resizes_without_changing_aspect(self, mock_conn):
        mock_conn.send_command.side_effect = _capture_side_effect()
        with patch("blend_ai.tools.relighting.asyncio.sleep", new=AsyncMock()):
            result = await capture_cycles_viewport(
                settle_seconds=0,
                max_size=64,
                format="PNG",
                keep_session=False,
            )

        assert result.content[0].mimeType == "image/png"
        assert result.structuredContent["output_width"] == 64
        assert result.structuredContent["output_height"] == 32
        assert result.structuredContent["jpeg_quality"] is None
        assert result.structuredContent["restored"] is True

    @pytest.mark.asyncio
    async def test_capture_atomically_stages_the_exact_returned_bytes(
        self,
        mock_conn,
        tmp_path,
    ):
        mock_conn.send_command.side_effect = _capture_side_effect()
        destination = tmp_path / "capture.png"
        with patch("blend_ai.tools.relighting.asyncio.sleep", new=AsyncMock()):
            result = await capture_cycles_viewport(
                settle_seconds=0,
                max_size=64,
                format="PNG",
                staging_path=str(destination),
            )

        returned = base64.b64decode(result.content[0].data)
        assert destination.read_bytes() == returned
        assert result.structuredContent["staging_path"] == str(destination)
        assert result.structuredContent["staging_sha256"] == hashlib.sha256(returned).hexdigest()
        assert not list(tmp_path.glob(".capture.png.*.tmp"))

    @pytest.mark.asyncio
    async def test_capture_staging_refuses_to_overwrite_before_prepare(
        self,
        mock_conn,
        tmp_path,
    ):
        destination = tmp_path / "capture.png"
        destination.write_bytes(b"existing evidence")

        with pytest.raises(ValidationError, match="never overwrites"):
            await capture_cycles_viewport(format="PNG", staging_path=str(destination))

        assert destination.read_bytes() == b"existing evidence"
        mock_conn.send_command.assert_not_called()

    def test_atomic_stage_helper_does_not_publish_invalid_image(self, tmp_path):
        destination = tmp_path / "capture.png"

        with pytest.raises(RuntimeError, match="image validation"):
            _stage_capture_bytes(b"not an image", destination, "PNG", 64, 64)

        assert not destination.exists()
        assert not list(tmp_path.glob(".capture.png.*.tmp"))

    @pytest.mark.asyncio
    async def test_transparent_source_can_encode_to_jpeg(self, mock_conn):
        mock_conn.send_command.side_effect = _capture_side_effect(mode="RGBA")
        with patch("blend_ai.tools.relighting.asyncio.sleep", new=AsyncMock()):
            result = await capture_cycles_viewport(settle_seconds=0, max_size=64)
        assert result.content[0].mimeType == "image/jpeg"

    @pytest.mark.asyncio
    async def test_macos_backend_captures_before_explicit_restore(self, mock_conn):
        def respond(command, params):
            if command == "prepare_cycles_viewport":
                return {
                    "status": "ok",
                    "result": {
                        "session_id": "preview-123",
                        "capture_backend_required": "MACOS_SCREENCAPTUREKIT",
                    },
                }
            if command == "capture_cycles_viewport_area":
                return {
                    "status": "ok",
                    "result": {
                        "session_id": "preview-123",
                        "capture_target": {"process_id": 42},
                        "warnings": ["Metal viewport capture"],
                    },
                }
            if command == "restore_cycles_viewport":
                return {
                    "status": "ok",
                    "result": {"session_id": params["session_id"], "restored": True},
                }
            raise AssertionError(command)

        source = base64.b64decode(_png_base64(200, 132))
        mock_conn.send_command.side_effect = respond
        with (
            patch("blend_ai.tools.relighting.sys.platform", "darwin"),
            patch("blend_ai.tools.relighting.asyncio.sleep", new=AsyncMock()),
            patch(
                "blend_ai.tools.relighting._capture_macos_window",
                return_value=(
                    source,
                    200,
                    132,
                    {
                        "x": 50,
                        "y": 32,
                        "width": 100,
                        "height": 100,
                        "origin": "TOP_LEFT",
                    },
                    {"windowID": 99},
                ),
            ),
        ):
            result = await capture_cycles_viewport(
                settle_seconds=0,
                max_size=64,
                keep_session=False,
            )

        assert result.structuredContent["capture_backend"] == "MACOS_SCREENCAPTUREKIT"
        assert result.structuredContent["restored"] is True
        assert result.structuredContent["output_width"] == 64
        assert result.structuredContent["output_height"] == 64
        assert "visible Blender window 99" in result.structuredContent["warnings"][-1]
        calls = mock_conn.send_command.call_args_list
        assert [call.args[0] for call in calls] == [
            "prepare_cycles_viewport",
            "capture_cycles_viewport_area",
            "restore_cycles_viewport",
        ]
        assert calls[1].args[1] == {
            "session_id": "preview-123",
            "keep_session": True,
            "metadata_only": True,
        }

    @pytest.mark.asyncio
    async def test_explicit_quick_render_is_denoised_and_skips_settle(self, mock_conn):
        def respond(command, params):
            if command == "prepare_cycles_viewport":
                return {"status": "ok", "result": {"session_id": "preview-123"}}
            if command == "capture_cycles_quick_render":
                assert params == {
                    "session_id": "preview-123",
                    "max_size": 64,
                    "denoise": True,
                }
                return {
                    "status": "ok",
                    "result": {
                        "session_id": "preview-123",
                        "image_base64": _png_base64(64, 36),
                        "source_width": 64,
                        "source_height": 36,
                        "denoise": True,
                        "warnings": ["temporary denoised render"],
                    },
                }
            if command == "restore_cycles_viewport":
                return {
                    "status": "ok",
                    "result": {"session_id": params["session_id"], "restored": True},
                }
            raise AssertionError(command)

        mock_conn.send_command.side_effect = respond
        sleep = AsyncMock()
        with patch("blend_ai.tools.relighting.asyncio.sleep", new=sleep):
            result = await capture_cycles_viewport(
                capture_mode="QUICK_RENDER",
                settle_seconds=2,
                denoise=True,
                max_size=64,
                keep_session=False,
            )

        sleep.assert_not_awaited()
        assert result.structuredContent["capture_backend"] == "QUICK_CYCLES_RENDER"
        assert result.structuredContent["settle_basis"] == "not_applicable"
        assert result.structuredContent["settle_seconds"] == 0.0
        assert result.structuredContent["restored"] is True
        assert [call.args[0] for call in mock_conn.send_command.call_args_list] == [
            "prepare_cycles_viewport",
            "capture_cycles_quick_render",
            "restore_cycles_viewport",
        ]

    @pytest.mark.asyncio
    async def test_restore_action_returns_text_and_structured_metadata(self, mock_conn):
        mock_conn.send_command.return_value = {
            "status": "ok",
            "result": {"session_id": "preview-123", "restored": True},
        }
        result = await capture_cycles_viewport(action="RESTORE", session_id="preview-123")

        assert len(result.content) == 1
        assert isinstance(result.content[0], TextContent)
        assert result.structuredContent["action"] == "RESTORE"
        assert result.structuredContent["restored"] is True
        mock_conn.send_command.assert_called_once_with(
            "restore_cycles_viewport",
            {"session_id": "preview-123"},
        )

    @pytest.mark.asyncio
    async def test_capture_error_restores_prepared_session(self, mock_conn):
        def respond(command, params):
            if command == "prepare_cycles_viewport":
                return {"status": "ok", "result": {"session_id": "preview-123"}}
            if command == "capture_cycles_viewport_area":
                return {"status": "error", "result": "window hidden"}
            return {
                "status": "ok",
                "result": {"session_id": params["session_id"], "restored": True},
            }

        mock_conn.send_command.side_effect = respond
        with patch("blend_ai.tools.relighting.asyncio.sleep", new=AsyncMock()):
            with pytest.raises(RuntimeError, match="window hidden"):
                await capture_cycles_viewport(settle_seconds=0)

        assert [call.args[0] for call in mock_conn.send_command.call_args_list] == [
            "prepare_cycles_viewport",
            "capture_cycles_viewport_area",
            "restore_cycles_viewport",
        ]

    @pytest.mark.asyncio
    async def test_prepare_transport_failure_attempts_restore_all(self, mock_conn):
        def respond(command, params):
            if command == "prepare_cycles_viewport":
                return {"status": "error", "result": "response lost after prepare"}
            assert command == "restore_cycles_viewport"
            assert params == {"session_id": None}
            return {
                "status": "ok",
                "result": {"session_id": None, "restored": False},
            }

        mock_conn.send_command.side_effect = respond
        with pytest.raises(RuntimeError, match="response lost after prepare"):
            await capture_cycles_viewport(settle_seconds=0)
        assert [call.args[0] for call in mock_conn.send_command.call_args_list] == [
            "prepare_cycles_viewport",
            "restore_cycles_viewport",
        ]

    @pytest.mark.asyncio
    async def test_capture_and_restore_failure_are_both_reported(self, mock_conn):
        def respond(command, _params):
            if command == "prepare_cycles_viewport":
                return {"status": "ok", "result": {"session_id": "preview-123"}}
            if command == "capture_cycles_viewport_area":
                return {"status": "error", "result": "capture failed"}
            if command == "restore_cycles_viewport":
                return {"status": "error", "result": "restore failed"}
            raise AssertionError(command)

        mock_conn.send_command.side_effect = respond
        with patch("blend_ai.tools.relighting.asyncio.sleep", new=AsyncMock()):
            with pytest.raises(RuntimeError, match="restoration also failed.*restore failed"):
                await capture_cycles_viewport(settle_seconds=0)

    @pytest.mark.asyncio
    async def test_cancellation_restores_prepared_session(self, mock_conn):
        mock_conn.send_command.side_effect = _capture_side_effect()
        with patch(
            "blend_ai.tools.relighting.asyncio.sleep",
            new=AsyncMock(side_effect=asyncio.CancelledError),
        ):
            with pytest.raises(asyncio.CancelledError):
                await capture_cycles_viewport()

        assert [call.args[0] for call in mock_conn.send_command.call_args_list] == [
            "prepare_cycles_viewport",
            "restore_cycles_viewport",
        ]

    @pytest.mark.asyncio
    async def test_cancellation_during_prepare_waits_then_restores(self, mock_conn):
        prepare_started = threading.Event()
        release_prepare = threading.Event()
        order = []

        def respond(command, params):
            order.append(f"{command}:start")
            if command == "prepare_cycles_viewport":
                prepare_started.set()
                assert release_prepare.wait(timeout=2)
                order.append("prepare_cycles_viewport:done")
                return {"status": "ok", "result": {"session_id": "preview-123"}}
            if command == "restore_cycles_viewport":
                order.append("restore_cycles_viewport:done")
                return {
                    "status": "ok",
                    "result": {"session_id": params["session_id"], "restored": True},
                }
            raise AssertionError(command)

        mock_conn.send_command.side_effect = respond
        task = asyncio.create_task(capture_cycles_viewport(settle_seconds=0))
        assert await asyncio.to_thread(prepare_started.wait, 1)
        task.cancel()
        release_prepare.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert order.index("prepare_cycles_viewport:done") < order.index(
            "restore_cycles_viewport:start"
        )
        assert order[-1] == "restore_cycles_viewport:done"

    @pytest.mark.asyncio
    async def test_cancellation_during_capture_waits_before_restore(self, mock_conn):
        capture_started = threading.Event()
        release_capture = threading.Event()
        order = []

        def respond(command, params):
            order.append(f"{command}:start")
            if command == "prepare_cycles_viewport":
                return {"status": "ok", "result": {"session_id": "preview-123"}}
            if command == "capture_cycles_viewport_area":
                capture_started.set()
                assert release_capture.wait(timeout=2)
                order.append("capture_cycles_viewport_area:done")
                return {
                    "status": "ok",
                    "result": {
                        "session_id": "preview-123",
                        "image_base64": _png_base64(),
                        "source_width": 200,
                        "source_height": 100,
                    },
                }
            if command == "restore_cycles_viewport":
                order.append("restore_cycles_viewport:done")
                return {
                    "status": "ok",
                    "result": {"session_id": params["session_id"], "restored": True},
                }
            raise AssertionError(command)

        mock_conn.send_command.side_effect = respond
        task = asyncio.create_task(capture_cycles_viewport(settle_seconds=0))
        assert await asyncio.to_thread(capture_started.wait, 1)
        task.cancel()
        release_capture.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert order.index("capture_cycles_viewport_area:done") < order.index(
            "restore_cycles_viewport:start"
        )
        assert order[-1] == "restore_cycles_viewport:done"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("capture_patch", "message"),
        [
            ({"image_base64": "%%%"}, "base64"),
            ({"session_id": "wrong"}, "mismatched"),
            ({"source_width": 201}, "source_width"),
            (
                {
                    "region_rect": {
                        "x": 190,
                        "y": 0,
                        "width": 20,
                        "height": 20,
                        "origin": "TOP_LEFT",
                    }
                },
                "outside",
            ),
        ],
    )
    async def test_invalid_capture_payload_restores(self, mock_conn, capture_patch, message):
        default = {
            "session_id": "preview-123",
            "image_base64": _png_base64(),
            "source_width": 200,
            "source_height": 100,
        }
        default.update(capture_patch)

        def respond(command, params):
            if command == "prepare_cycles_viewport":
                return {"status": "ok", "result": {"session_id": "preview-123"}}
            if command == "capture_cycles_viewport_area":
                return {"status": "ok", "result": default}
            return {
                "status": "ok",
                "result": {"session_id": params["session_id"], "restored": True},
            }

        mock_conn.send_command.side_effect = respond
        with patch("blend_ai.tools.relighting.asyncio.sleep", new=AsyncMock()):
            with pytest.raises(RuntimeError, match=message):
                await capture_cycles_viewport(settle_seconds=0)
        assert mock_conn.send_command.call_args_list[-1].args[0] == "restore_cycles_viewport"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"preview_samples": 0}, "preview_samples"),
            ({"settle_seconds": 31}, "settle_seconds"),
            ({"max_size": 63}, "max_size"),
            ({"jpeg_quality": 101}, "jpeg_quality"),
            ({"area_index": -1}, "area_index"),
            ({"capture_mode": "AUTO"}, "capture_mode"),
        ],
    )
    async def test_capture_limits_fail_before_prepare(self, mock_conn, kwargs, message):
        with pytest.raises(ValidationError, match=message):
            await capture_cycles_viewport(**kwargs)
        mock_conn.send_command.assert_not_called()


@pytest.mark.asyncio
async def test_real_fastmcp_preserves_direct_image_result_and_output_schema():
    """Exercise FastMCP's real CallToolResult conversion, not the mocked decorator."""
    server = FastMCP("native-image-test")
    metadata = CaptureMetadata(
        action="CAPTURE",
        session_id="session",
        format="PNG",
        byte_count=1,
    )

    @server.tool()
    def native_image() -> Annotated[CallToolResult, CaptureMetadata]:
        return _capture_result(metadata, base64.b64decode(_png_base64(1, 1)))

    tools = await server.list_tools()
    tool = next(item for item in tools if item.name == "native_image")
    assert tool.outputSchema is not None
    assert tool.outputSchema["properties"]["action"]

    result = await server.call_tool("native_image", {})
    assert isinstance(result, CallToolResult)
    assert isinstance(result.content[0], ImageContent)
    assert result.structuredContent["session_id"] == "session"
