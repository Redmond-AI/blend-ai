"""Blender-native smoke test for managed profile lights, fog, and rain.

Run with a factory-startup Blender process.  The script never saves a blend.
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

from addon import look_profile_assets  # noqa: E402


def _aim(obj, point=(0.0, 0.0, 0.0)):
    obj.rotation_euler = (Vector(point) - obj.location).to_track_quat("-Z", "Y").to_euler()


def _image_tokens():
    return {image.as_pointer() for image in bpy.data.images}


def _write_hdri_fixture(path: Path) -> None:
    image = bpy.data.images.new("Temporary HDRI Fixture", width=8, height=4)
    image.generated_color = (0.2, 0.35, 0.6, 1.0)
    image.filepath_raw = str(path)
    image.file_format = "PNG"
    image.save()
    bpy.data.images.remove(image, do_unlink=True)
    if not path.is_file() or path.stat().st_size == 0:
        raise AssertionError("HDRI transaction fixture was not written")


def _exercise_hdri_transactions(scene, base_profile, temp_root: Path) -> None:
    hdri_path = temp_root / "transaction-hdri.png"
    _write_hdri_fixture(hdri_path)

    helper_world = bpy.data.worlds.new("HDRI Helper Failure World")
    before_helper = _image_tokens()
    try:
        look_profile_assets.configure_world(
            helper_world,
            {
                "mode": "MANAGED_HDRI",
                "hdri_path": str(hdri_path),
                "rotation_degrees": "invalid-after-load",
                "strength": 0.5,
            },
            profile_id="hdri-helper-failure",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid post-load HDRI configuration unexpectedly succeeded")
    if _image_tokens() != before_helper:
        raise AssertionError("configure_world leaked a newly loaded HDRI image")
    bpy.data.worlds.remove(helper_world, do_unlink=True)

    existing_image = bpy.data.images.load(str(hdri_path), check_existing=True)
    existing_world = bpy.data.worlds.new("Existing HDRI Failure World")
    before_existing = _image_tokens()
    try:
        look_profile_assets.configure_world(
            existing_world,
            {
                "mode": "MANAGED_HDRI",
                "hdri_path": str(hdri_path),
                "rotation_degrees": "invalid-after-reuse",
                "strength": 0.5,
            },
            profile_id="existing-hdri-failure",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid reused-HDRI configuration unexpectedly succeeded")
    if _image_tokens() != before_existing or existing_image.name not in bpy.data.images:
        raise AssertionError("HDRI rollback removed an image it did not create")
    bpy.data.worlds.remove(existing_world, do_unlink=True)
    bpy.data.images.remove(existing_image, do_unlink=True)

    failed_collection = bpy.data.collections.new("HDRI Later Failure Payload")
    scene.collection.children.link(failed_collection)
    failed_world = bpy.data.worlds.new("HDRI Later Failure World")
    failed_profile = dict(base_profile)
    failed_profile.update(
        {
            "profile_id": "hdri-later-failure",
            "world": {
                "mode": "MANAGED_HDRI",
                "hdri_path": str(hdri_path),
                "rotation_degrees": 17.0,
                "strength": 0.5,
            },
            "lighting": {"existing_light_policy": "KEEP", "lights": []},
            "atmosphere": [{"id": "invalid", "kind": "NOT_SUPPORTED"}],
            "camera": {"mode": "KEEP"},
        }
    )
    before_build = _image_tokens()
    try:
        look_profile_assets.build_profile_assets(
            scene,
            failed_collection,
            failed_world,
            failed_profile,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Deliberately invalid post-HDRI asset build unexpectedly succeeded")
    if _image_tokens() != before_build:
        raise AssertionError("Later asset failure leaked a newly loaded HDRI image")
    bpy.data.worlds.remove(failed_world, do_unlink=True)
    bpy.data.collections.remove(failed_collection, do_unlink=True)

    handoff_collection = bpy.data.collections.new("HDRI Compiler Handoff Payload")
    scene.collection.children.link(handoff_collection)
    handoff_world = bpy.data.worlds.new("HDRI Compiler Handoff World")
    handoff_profile = dict(failed_profile)
    handoff_profile["profile_id"] = "hdri-compiler-handoff"
    handoff_profile["atmosphere"] = []
    before_handoff = _image_tokens()
    handoff_result = look_profile_assets.build_profile_assets(
        scene,
        handoff_collection,
        handoff_world,
        handoff_profile,
    )
    if not any(kind == "IMAGE" for kind, _value in handoff_result["created"]):
        raise AssertionError("Successful HDRI build did not hand image ownership to its caller")
    if _image_tokens() == before_handoff:
        raise AssertionError("Successful HDRI build did not retain its World image")
    if look_profile_assets.cleanup_created(handoff_result["created"]):
        raise AssertionError("Compiler-style HDRI cleanup reported an error")
    if _image_tokens() != before_handoff:
        raise AssertionError("Compiler-style later failure leaked its HDRI image")
    bpy.data.worlds.remove(handoff_world, do_unlink=True)
    bpy.data.collections.remove(handoff_collection, do_unlink=True)


def _exercise_rain_limits(scene) -> None:
    zero_collection = bpy.data.collections.new("Zero Rain Payload")
    scene.collection.children.link(zero_collection)
    inventory, created = look_profile_assets.create_atmosphere(
        zero_collection,
        [
            {
                "id": "zero-rain",
                "kind": "RAIN_RIG",
                "rain_rate": 0,
                "drop_size_m": 0.002,
                "fall_speed_mps": 10.0,
                "seed": 11,
            }
        ],
        profile_id="zero-rain-profile",
    )
    zero_curve = bpy.data.curves[inventory[0]["object_name"] + "_CURVE"]
    if inventory[0]["drop_count"] != 0 or len(zero_curve.splines) != 0:
        raise AssertionError("rain_rate=0 did not create an empty deterministic rain rig")
    if look_profile_assets.cleanup_created(created):
        raise AssertionError("Zero-rain cleanup reported an error")
    bpy.data.collections.remove(zero_collection, do_unlink=True)

    before_curves = {curve.as_pointer() for curve in bpy.data.curves}
    try:
        look_profile_assets.create_atmosphere(
            scene.collection,
            [
                {
                    "id": "too-much-rain",
                    "kind": "RAIN_RIG",
                    "rain_rate": look_profile_assets.MAX_RAIN_DROPS + 1,
                    "drop_size_m": 0.002,
                    "fall_speed_mps": 10.0,
                }
            ],
            profile_id="too-much-rain-profile",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Unsupported rain rate was silently clamped")
    if {curve.as_pointer() for curve in bpy.data.curves} != before_curves:
        raise AssertionError("Rejected rain rate left a Curve datablock behind")


def _exercise_camera_helper_atomicity(scene, payload_collection) -> None:
    original_camera = scene.camera
    before_objects = {obj.as_pointer() for obj in bpy.data.objects}
    before_cameras = {camera.as_pointer() for camera in bpy.data.cameras}
    try:
        look_profile_assets.clone_camera(
            scene,
            payload_collection,
            {
                "mode": "MANAGED_CLONE",
                "source_camera": original_camera.name,
                "focus_object": "Definitely Missing Focus Object",
            },
            profile_id="camera-helper-failure",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Deliberately invalid camera clone unexpectedly succeeded")
    if scene.camera is not original_camera:
        raise AssertionError("Failed camera clone did not restore the prior Scene camera")
    if {obj.as_pointer() for obj in bpy.data.objects} != before_objects:
        raise AssertionError("Failed camera clone leaked an Object datablock")
    if {camera.as_pointer() for camera in bpy.data.cameras} != before_cameras:
        raise AssertionError("Failed camera clone leaked a Camera datablock")


def _exercise_safe_light_isolation(scene, ground) -> None:
    safe_a = bpy.data.collections.new("Isolation Safe A")
    safe_b = bpy.data.collections.new("Isolation Safe B")
    scene.collection.children.link(safe_a)
    scene.collection.children.link(safe_b)
    light_data = bpy.data.lights.new("Multi-linked Artist Light Data", type="POINT")
    light = bpy.data.objects.new("Multi-linked Artist Light", light_data)
    safe_a.objects.link(light)
    safe_b.objects.link(light)
    bpy.context.view_layer.update()

    changes = []
    excluded = look_profile_assets.isolate_existing_lights(
        scene,
        "MUTE_NON_MANAGED",
        mutation_sink=changes,
    )
    if not {"Isolation Safe A", "Isolation Safe B"}.issubset(excluded):
        raise AssertionError("Not every multi-linked artist-light path was excluded")
    if not all(layer_collection.exclude for layer_collection, _previous in changes):
        raise AssertionError("A planned artist-light LayerCollection remained renderable")
    if look_profile_assets._restore_layer_exclusions(changes):
        raise AssertionError("Artist-light LayerCollection rollback failed")

    before_objects = {obj.as_pointer() for obj in bpy.data.objects}
    before_lights = {item.as_pointer() for item in bpy.data.lights}
    try:
        look_profile_assets.create_lights(
            scene,
            safe_a,
            {
                "existing_light_policy": "MUTE_NON_MANAGED",
                "lights": [
                    {
                        "id": "invalid-type",
                        "type": "NOT_A_BLENDER_LIGHT_TYPE",
                        "location_world": [0.0, 0.0, 1.0],
                    }
                ],
            },
            profile_id="light-helper-failure",
        )
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError("Deliberately invalid managed light unexpectedly succeeded")
    if {obj.as_pointer() for obj in bpy.data.objects} != before_objects:
        raise AssertionError("Failed managed light creation leaked an Object datablock")
    if {item.as_pointer() for item in bpy.data.lights} != before_lights:
        raise AssertionError("Failed managed light creation leaked a Light datablock")
    if any(layer_collection.exclude for layer_collection, _previous in changes):
        raise AssertionError("Failed managed light creation left artist-light paths excluded")

    safe_c = bpy.data.collections.new("Isolation Safe C")
    unsafe_parent = bpy.data.collections.new("Isolation Unsafe Parent")
    nested_geometry = bpy.data.collections.new("Isolation Nested Geometry")
    scene.collection.children.link(safe_c)
    scene.collection.children.link(unsafe_parent)
    unsafe_parent.children.link(nested_geometry)
    nested_geometry.objects.link(ground)
    unsafe_data = bpy.data.lights.new("Unsafe Multi-linked Light Data", type="POINT")
    unsafe_light = bpy.data.objects.new("Unsafe Multi-linked Light", unsafe_data)
    safe_c.objects.link(unsafe_light)
    unsafe_parent.objects.link(unsafe_light)
    bpy.context.view_layer.update()
    try:
        look_profile_assets.isolate_existing_lights(scene, "MUTE_NON_MANAGED")
    except ValueError:
        pass
    else:
        raise AssertionError("Nested geometry path was excluded with an artist light")
    if any(
        layer_collection.exclude
        for view_layer in scene.view_layers
        for layer_collection in look_profile_assets._walk_layer_collections(
            view_layer.layer_collection
        )
        if layer_collection.collection in {safe_a, safe_b, safe_c, unsafe_parent}
    ):
        raise AssertionError("Failed light-isolation preflight partially mutated a View Layer")

    nested_geometry.objects.unlink(ground)
    bpy.data.objects.remove(light, do_unlink=True)
    bpy.data.objects.remove(unsafe_light, do_unlink=True)
    bpy.data.lights.remove(light_data, do_unlink=True)
    bpy.data.lights.remove(unsafe_data, do_unlink=True)
    for candidate in (nested_geometry, unsafe_parent, safe_c, safe_b, safe_a):
        bpy.data.collections.remove(candidate, do_unlink=True)


def main() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.name = "Look Profile Asset Integration"

    camera_data = bpy.data.cameras.new("Integration Camera Data")
    camera = bpy.data.objects.new("Integration Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (10.0, -10.0, 7.0)
    _aim(camera, (0.0, 0.0, 1.5))
    scene.camera = camera

    bpy.ops.mesh.primitive_plane_add(size=20.0, location=(0.0, 0.0, 0.0))
    ground = bpy.context.object
    ground.name = "Integration Ground"

    collection = bpy.data.collections.new("AI Profile Payload")
    scene.collection.children.link(collection)
    world = bpy.data.worlds.new("AI Profile World")
    scene.world = world

    profile = {
        "profile_id": "rainy-night",
        "world": {
            "mode": "MANAGED_SOLID",
            "color_rgb": [0.015, 0.03, 0.06],
            "strength": 0.12,
        },
        "lighting": {
            "existing_light_policy": "KEEP",
            "lights": [
                {
                    "id": "moon",
                    "type": "SUN",
                    "location_world": [4.0, -2.0, 8.0],
                    "target_point": [0.0, 0.0, 0.0],
                    "energy": 2.0,
                    "color_rgb": [0.28, 0.42, 0.85],
                    "sun_angle_degrees": 3.0,
                },
                {
                    "id": "window-fill",
                    "type": "AREA",
                    "location_world": [-3.0, -4.0, 4.0],
                    "target_point": [0.0, 0.0, 1.0],
                    "energy": 650.0,
                    "color_rgb": [0.3, 0.55, 1.0],
                    "area_shape": "DISK",
                    "size": 3.0,
                },
            ],
        },
        "atmosphere": [
            {
                "id": "ground-fog",
                "kind": "FOG_VOLUME",
                "density": 0.018,
                "anisotropy": 0.2,
                "location_world": [0.0, 0.0, 1.2],
                "size_xyz": [12.0, 12.0, 2.4],
                "color_rgb": [0.45, 0.55, 0.7],
            },
            {
                "id": "rain",
                "kind": "RAIN_RIG",
                "seed": 731,
                "rain_rate": 220,
                "drop_size_m": 0.018,
                "fall_speed_mps": 14.0,
                "wind_vector": [1.5, 0.25, 0.0],
                "location_world": [0.0, 0.0, 4.0],
                "size_xyz": [12.0, 12.0, 8.0],
                "color_rgb": [0.7, 0.85, 1.0],
            },
        ],
        "camera": {"mode": "KEEP"},
        "render": {
            "engine": "BLENDER_EEVEE",
            "samples": 16,
            "denoise": True,
            "resolution_percentage": 100,
        },
        "color_management": {"mode": "KEEP"},
    }

    result = look_profile_assets.build_profile_assets(scene, collection, world, profile)
    inventory = result["inventory"]
    assert len(inventory["lights"]) == 2
    assert {item["kind"] for item in inventory["atmosphere"]} == {
        "FOG_VOLUME",
        "RAIN_RIG",
    }
    assert inventory["atmosphere"][1]["drop_count"] == 220
    assert inventory["atmosphere"][1]["requested_rain_rate"] == 220.0
    assert all(obj.get("blend_ai_profile_id") == "rainy-night" for obj in collection.objects)

    _exercise_rain_limits(scene)
    _exercise_camera_helper_atomicity(scene, collection)
    _exercise_safe_light_isolation(scene, ground)
    with tempfile.TemporaryDirectory(prefix="blend-ai-hdri-transaction-") as temp_dir:
        _exercise_hdri_transactions(scene, profile, Path(temp_dir))

    scene.render.resolution_x = 192
    scene.render.resolution_y = 128
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    artifact = Path(tempfile.gettempdir()) / "blend_ai_look_profile_assets.png"
    scene.render.filepath = str(artifact)
    bpy.ops.render.render(write_still=True, scene=scene.name)
    if not artifact.is_file() or artifact.stat().st_size == 0:
        raise AssertionError("Managed profile asset smoke render was not written")

    print(
        json.dumps(
            {
                "success": True,
                "blender_version": bpy.app.version_string,
                "artifact": str(artifact),
                "artifact_bytes": artifact.stat().st_size,
                "engine": scene.render.engine,
                "inventory": inventory,
                "hardening": {
                    "hdri_transaction_rollback": True,
                    "helper_internal_atomicity": True,
                    "multi_path_light_isolation": True,
                    "nested_geometry_fail_closed": True,
                    "rain_zero_and_max": True,
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
