"""Blender-native end-to-end smoke test for the actual look-profile compiler.

Run in a disposable factory-startup process.  The script invokes the public
handler functions, validates Blender-version color enums, compiles a managed
Scene, checks ownership, verifies source preservation and confirms that
profile-local datablocks do not advance the source geometry revision.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import bpy
from mathutils import Vector


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from addon import spatial_cache  # noqa: E402
from addon.handlers import look_profiles  # noqa: E402
from addon.handlers import look_profile_rendering  # noqa: E402


def _aim(obj, point=(0.0, 0.0, 0.0)) -> None:
    obj.rotation_euler = (Vector(point) - obj.location).to_track_quat("-Z", "Y").to_euler()


def _look_alias(scene: bpy.types.Scene) -> tuple[str, str]:
    import PyOpenColorIO as ocio

    current = str(scene.view_settings.look)
    identifiers = list(ocio.GetCurrentConfig().getLookNames())
    target = f"{scene.view_settings.view_transform} - Medium High Contrast"
    preferred = target if target in identifiers else next(
        (identifier for identifier in identifiers if "Medium High Contrast" in identifier),
        current,
    )
    alias = preferred.split(" - ", 1)[-1] if " - " in preferred else preferred
    return alias, preferred


def main() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.name = "Look Profile Handler Source"

    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 1.0))
    cube = bpy.context.object
    cube.name = "Profile Source Cube"
    camera_data = bpy.data.cameras.new("Profile Source Camera Data")
    camera = bpy.data.objects.new("Profile Source Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (6.0, -6.0, 4.0)
    _aim(camera, (0.0, 0.0, 1.0))
    scene.camera = camera
    camera_names = [camera.name]
    for name, location in (
        ("Profile Camera Wide", (8.0, -8.0, 5.0)),
        ("Profile Camera Low", (4.0, -5.0, 1.8)),
    ):
        data = bpy.data.cameras.new(f"{name} Data")
        extra = bpy.data.objects.new(name, data)
        scene.collection.objects.link(extra)
        extra.location = location
        _aim(extra, (0.0, 0.0, 1.0))
        camera_names.append(extra.name)
    scene.render.resolution_x = 160
    scene.render.resolution_y = 96
    scene.render.resolution_percentage = 100

    spatial_cache._reset_for_tests()
    spatial_cache.register_handlers()
    for _index in range(3):
        bpy.context.view_layer.update()
        updater = getattr(bpy.context.evaluated_depsgraph_get(), "update", None)
        if callable(updater):
            updater()
    spatial_cache._reset_for_tests()
    spatial_cache.clear_recent_updates()
    before_revisions = spatial_cache.get_revisions()
    before_base = look_profiles._base_fingerprint(scene)
    before_geometry = look_profiles._geometry_fingerprint(scene)
    requested_look, expected_look = _look_alias(scene)

    profile = {
        "schema_version": 1,
        "profile_id": "native-smoke",
        "display_name": "Native Smoke",
        "status": "DRAFT",
        "seed": 321,
        "generator_version": "integration-v1",
        "tags": ["integration"],
        "lighting": {
            "existing_light_policy": "KEEP",
            "lights": [
                {
                    "id": "key",
                    "type": "AREA",
                    "location_world": [3.0, -3.0, 5.0],
                    "target_point": [0.0, 0.0, 1.0],
                    "energy": 700.0,
                    "color_rgb": [1.0, 0.78, 0.5],
                    "area_shape": "DISK",
                    "size": 3.0,
                }
            ],
        },
        "world": {
            "mode": "MANAGED_SOLID",
            "color_rgb": [0.01, 0.02, 0.05],
            "strength": 0.2,
        },
        "atmosphere": [
            {
                "id": "fog",
                "kind": "FOG_VOLUME",
                "density": 0.004,
                "anisotropy": 0.1,
                "location_world": [0.0, 0.0, 1.5],
                "size_xyz": [8.0, 8.0, 3.0],
            }
        ],
        "post": {
            "mode": "MANAGED_STACK",
            "bloom": {"enabled": True, "threshold": 0.8, "strength": 0.15, "radius": 0.5},
            "grain": {"enabled": True, "strength": 0.03, "scale": 1.0, "seed": 321},
            "vignette": {"enabled": True, "strength": 0.1, "feather": 0.7},
        },
        "color_management": {
            "mode": "MANAGED",
            "view_transform": str(scene.view_settings.view_transform),
            "look": requested_look,
            "exposure": -0.25,
            "gamma": 1.0,
        },
        "camera": {"mode": "KEEP"},
        "render": {
            "engine": "CYCLES",
            "samples": 8,
            "denoise": True,
            "resolution_percentage": 50,
            "film_transparent": False,
            "use_motion_blur": False,
        },
        "review_intent": {
            "summary": "Warm key against a cold, lightly fogged background",
            "expects_shadows": True,
            "expects_reflections": True,
            "expects_volume": True,
            "expects_compositing": True,
        },
    }
    request = {
        "action": "VALIDATE",
        "source_scene": scene.name,
        "profile": profile,
        "update_mode": "CREATE_VERSION",
        "expected_profile_revision": 0,
        "expected_geometry_revision": before_revisions["geometry_revision"],
        "strict": True,
    }
    validated = look_profiles.handle_upsert_look_profile(request)
    resolved_look = validated["plan"]["definition"]["resolved"]["color_management"]["look"]
    if resolved_look != expected_look:
        raise AssertionError(f"Color look alias resolved to {resolved_look!r}, not {expected_look!r}")

    request["action"] = "COMPILE"
    compiled = look_profiles.handle_upsert_look_profile(request)
    entry = compiled["profile"]
    after_revisions = spatial_cache.get_revisions()
    if after_revisions["geometry_revision"] != before_revisions["geometry_revision"]:
        raise AssertionError(
            f"Profile compile advanced geometry revision {before_revisions} -> "
            f"{after_revisions}: {json.dumps(spatial_cache.get_recent_updates(), indent=2)}"
        )
    if look_profiles._base_fingerprint(scene) != before_base:
        raise AssertionError("Profile compile changed the source base fingerprint")
    if look_profiles._geometry_fingerprint(scene) != before_geometry:
        raise AssertionError("Profile compile changed the source geometry fingerprint")

    context = look_profiles.handle_get_look_profile_context(
        {"source_scene": scene.name, "profile_id": "native-smoke", "detail": "FULL"}
    )
    if not context["ownership"]["valid"]:
        raise AssertionError(json.dumps(context["ownership"], indent=2))
    activated = look_profiles.handle_activate_look_profile(
        {
            "profile_id": "native-smoke",
            "target_scene": entry["scene_name"],
            "expected_profile_revision": entry["revision"],
            "expected_geometry_revision": after_revisions["geometry_revision"],
            "strict": True,
        }
    )
    if not activated["activated"] or bpy.context.window.scene.name != entry["scene_name"]:
        raise AssertionError("Compiled profile did not activate in the current window")

    compiled_scene = bpy.data.scenes[entry["scene_name"]]
    audit = look_profile_rendering.handle_inspect_look_review_state(
        {
            "profile_id": "native-smoke",
            "target_scene": entry["scene_name"],
            "camera": compiled_scene.camera.name,
            "expected_profile_revision": entry["revision"],
            "expected_geometry_revision": after_revisions["geometry_revision"],
        }
    )
    if audit["status"] == "FAIL":
        raise AssertionError(f"Review audit failed: {json.dumps(audit, indent=2)}")

    original_state = {
        "camera": compiled_scene.camera.name,
        "frame": compiled_scene.frame_current,
        "filepath": compiled_scene.render.filepath,
        "resolution_percentage": compiled_scene.render.resolution_percentage,
        "use_compositing": compiled_scene.render.use_compositing,
        "compositor_hash": look_profiles.look_compositor.compositor_hash(compiled_scene),
        "view_layer_passes": {
            name: getattr(compiled_scene.view_layers[0], name)
            for name in (
                "use_pass_z",
                "use_pass_diffuse_direct",
                "use_pass_glossy_direct",
                "use_pass_glossy_indirect",
                "use_pass_emit",
                "use_pass_cryptomatte_object",
                "use_pass_cryptomatte_accurate",
            )
        },
    }
    with tempfile.TemporaryDirectory(prefix="blend-ai-review-") as output_root:
        submitted = look_profile_rendering.handle_submit_look_render_batch(
            {
                "batch": {
                    "profile_ids": ["native-smoke"],
                    "target_scenes": [entry["scene_name"]],
                    "pairing": "PAIRWISE",
                    "frames": {"frames": [1]},
                    "output_root": output_root,
                    "filename_template": "{scene}-{profile}-{camera}-{frame}",
                    "render_settings": {
                        "samples": 4,
                        "denoise": False,
                        "resolution_percentage": 50,
                        "file_format": "PNG",
                        "color_depth": "8",
                        "existing_file_policy": "ERROR",
                    },
                    "continue_on_error": False,
                    "include_review_packet": True,
                    "camera_names": camera_names,
                },
                "expected_profile_revision": entry["revision"],
                "expected_profile_revisions": None,
                "output_pass": "COMPOSITE",
                "restore_scene_state": True,
                "save_blend": False,
            }
        )
        for _index in range(len(camera_names)):
            look_profile_rendering._job_timer()
        batch = look_profile_rendering._batches[submitted["batch_id"]]
        if len(batch["items"]) != 3:
            raise AssertionError("Multi-camera review batch did not expand to three items")
        for item in batch["items"]:
            if item["status"] != "SUCCEEDED":
                raise AssertionError(
                    f"Review render did not succeed: {item['status']} {item.get('error')}"
                )
            fetched = look_profile_rendering.handle_get_look_render_result(
                {
                    "batch_id": submitted["batch_id"],
                    "result_id": item["result_id"],
                    "output_pass": "COMPOSITE",
                    "proxy_max_size": 1024,
                }
            )
            if len(fetched.get("review_tiles", [])) != 6:
                raise AssertionError("Review result did not return six diagnostic tiles")
            packet = fetched["metadata"].get("review_packet")
            if not isinstance(packet, dict) or len(packet.get("tiles", [])) != 6:
                raise AssertionError("Review result lacks strict packet metadata")
            if packet["tile_order"][0] != "combined":
                raise AssertionError("Review packet tile order is not canonical")
            if packet["tile_order"][4] != "depth" or "volume" in packet["tile_order"]:
                raise AssertionError("Review packet did not replace Volume with Depth")
            depth = packet["tiles"][4]
            if not depth["available"] or depth["nonzero_coverage"] <= 0.0:
                raise AssertionError(f"Camera Depth tile is not useful: {depth}")

    restored_state = {
        "camera": compiled_scene.camera.name,
        "frame": compiled_scene.frame_current,
        "filepath": compiled_scene.render.filepath,
        "resolution_percentage": compiled_scene.render.resolution_percentage,
        "use_compositing": compiled_scene.render.use_compositing,
        "compositor_hash": look_profiles.look_compositor.compositor_hash(compiled_scene),
        "view_layer_passes": {
            name: getattr(compiled_scene.view_layers[0], name)
            for name in original_state["view_layer_passes"]
        },
    }
    if restored_state != original_state:
        raise AssertionError(f"Review render state was not restored: {restored_state}")

    print(
        json.dumps(
            {
                "status": "PASS",
                "blender": list(bpy.app.version),
                "profile_scene": entry["scene_name"],
                "compositor_hash": entry["compositor_hash"],
                "resolved_look": resolved_look,
                "revisions_before": before_revisions,
                "revisions_after": after_revisions,
                "ownership_valid": True,
                "review_audit": audit["status"],
                "review_tiles": 18,
                "review_cameras": 3,
                "review_state_restored": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
