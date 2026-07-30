"""Build the deterministic miniature warehouse used by live relighting tests.

Run from Blender's scripting workspace, or launch Blender with ``--python``.
The script does not save unless an output path is supplied after ``--`` or in
``BLEND_AI_FIXTURE_OUTPUT``.
"""

from __future__ import annotations

import os
import sys

import bpy
from mathutils import Vector


def _collection(name: str) -> bpy.types.Collection:
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def _move_to_collection(obj: bpy.types.Object, collection: bpy.types.Collection) -> None:
    for current in list(obj.users_collection):
        current.objects.unlink(obj)
    collection.objects.link(obj)


def _cube(
    name: str,
    location: tuple[float, float, float],
    dimensions: tuple[float, float, float],
    collection: bpy.types.Collection,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    _move_to_collection(obj, collection)
    return obj


def _material(name: str, color: tuple[float, float, float, float], metallic=0.0, roughness=0.6):
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = color
    metallic_input = principled.inputs.get("Metallic IOR Level") or principled.inputs.get(
        "Metallic"
    )
    if metallic_input is not None:
        metallic_input.default_value = metallic
    principled.inputs["Roughness"].default_value = roughness
    return material


def _aim(obj: bpy.types.Object, target: tuple[float, float, float]) -> None:
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _add_collection_instance(
    scene_collection: bpy.types.Collection,
    material: bpy.types.Material,
) -> bpy.types.Object:
    """Add one evaluated collection instance without linking its source collection."""
    source = bpy.data.collections.new("Warehouse_Rack_Instance_Source")
    rack = _cube("Instanced_Rack_Source", (0, 0, 1.2), (1.2, 3.0, 2.4), source)
    rack.data.materials.append(material)

    instance = bpy.data.objects.new("Racking_Collection_Instance", None)
    instance.instance_type = "COLLECTION"
    instance.instance_collection = source
    instance.location = (-2.0, 4.0, 0.0)
    instance.rotation_euler[2] = 0.22
    scene_collection.objects.link(instance)
    return instance


def _add_geometry_nodes_instances(
    scene_collection: bpy.types.Collection,
) -> bpy.types.Object:
    """Add a tiny Geometry Nodes instance-on-points fixture for evaluated tests."""
    points = bpy.data.meshes.new("GN_Fixture_Points_Mesh")
    points.from_pydata([(0.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 4.0, 0.0)], [], [])
    points.update()
    obj = bpy.data.objects.new("GN_Instanced_Fixtures", points)
    obj.location = (11.0, -6.0, 2.0)
    scene_collection.objects.link(obj)

    tree = bpy.data.node_groups.new("GN_Instanced_Fixtures_Group", "GeometryNodeTree")
    tree.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    group_input = tree.nodes.new("NodeGroupInput")
    group_output = tree.nodes.new("NodeGroupOutput")
    cube = tree.nodes.new("GeometryNodeMeshCube")
    cube.inputs["Size"].default_value = (0.45, 0.45, 0.45)
    instances = tree.nodes.new("GeometryNodeInstanceOnPoints")
    tree.links.new(group_input.outputs["Geometry"], instances.inputs["Points"])
    tree.links.new(cube.outputs["Mesh"], instances.inputs["Instance"])
    tree.links.new(instances.outputs["Instances"], group_output.inputs["Geometry"])

    modifier = obj.modifiers.new("GN_Instanced_Fixtures", "NODES")
    modifier.node_group = tree
    return obj


def build_scene() -> bpy.types.Scene:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in list(bpy.data.collections):
        if collection.users == 0:
            bpy.data.collections.remove(collection)

    scene = bpy.context.scene
    scene.name = "Mini Warehouse Relighting Fixture"
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        # Blender 5.2 renamed the engine identifier back to BLENDER_EEVEE.
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 960
    scene.render.resolution_y = 540
    scene.render.resolution_percentage = 100

    geometry = _collection("Warehouse_Geometry")
    skylights = _collection("Warehouse_Skylights")
    originals = _collection("Warehouse_Original_Lights")

    concrete = _material("Concrete", (0.16, 0.17, 0.18, 1.0), roughness=0.82)
    steel = _material("Painted_Steel", (0.10, 0.12, 0.14, 1.0), metallic=0.55, roughness=0.38)
    crate_material = _material("Crate_Wood", (0.24, 0.11, 0.04, 1.0), roughness=0.7)

    floor = _cube("Warehouse_Floor", (0, 0, -0.1), (40, 20, 0.2), geometry)
    floor.data.materials.append(concrete)
    floor.data.materials.append(steel)
    if floor.data.polygons:
        floor.data.polygons[0].material_index = 1
    for name, loc, dims in (
        ("Wall_North", (0, 10, 5), (40, 0.25, 10)),
        ("Wall_South", (0, -10, 5), (40, 0.25, 10)),
        ("Wall_East", (20, 0, 5), (0.25, 20, 10)),
        ("Wall_West", (-20, 0, 5), (0.25, 20, 10)),
    ):
        wall = _cube(name, loc, dims, geometry)
        wall.data.materials.append(concrete)

    # Three roof panels leave two real 5m x 4m apertures centered at x=-10/+10.
    for index, (center_x, length_x) in enumerate(
        ((-16.25, 7.5), (-0.0, 15.0), (16.25, 7.5))
    ):
        roof = _cube(f"Roof_Panel_{index}", (center_x, 0, 10.1), (length_x, 20, 0.2), geometry)
        roof.data.materials.append(steel)

    glass = _material("Skylight_Glass", (0.32, 0.52, 0.78, 0.18), roughness=0.08)
    if hasattr(glass, "surface_render_method"):
        glass.surface_render_method = "DITHERED"
    glass.diffuse_color = (0.32, 0.52, 0.78, 0.18)
    glass_principled = glass.node_tree.nodes.get("Principled BSDF")
    transmission = glass_principled.inputs.get("Transmission Weight")
    if transmission is not None:
        transmission.default_value = 0.8
    for center_x in (-10.0, 10.0):
        pane = _cube(
            f"Skylight_Glass_{'Clear' if center_x < 0 else 'Blocked'}",
            (center_x, 0, 10.05),
            (5, 4, 0.05),
            skylights,
        )
        pane.data.materials.append(glass)

    blocker = _cube("Blocked_Skylight_Cover", (10, 0, 10.35), (4.5, 3.5, 0.25), geometry)
    blocker.data.materials.append(steel)

    for x in (-14, -7, 0, 7, 14):
        for y in (-6, 6):
            column = _cube(f"Column_{x}_{y}", (x, y, 5), (0.45, 0.45, 10), geometry)
            column.data.materials.append(steel)

    for index, location in enumerate(((-5, 1, 1), (0, -2, 1), (6, 2, 1))):
        crate = _cube(f"Crate_{index}", location, (2, 2, 2), geometry)
        crate.data.materials.append(crate_material)

    _add_collection_instance(geometry, steel)
    _add_geometry_nodes_instances(geometry)

    light_data = bpy.data.lights.new("Original_Daylight_Fill", type="AREA")
    light_data.energy = 2600
    light_data.shape = "RECTANGLE"
    light_data.size = 16
    light_data.size_y = 8
    original_light = bpy.data.objects.new("Original_Daylight_Fill", light_data)
    originals.objects.link(original_light)
    original_light.location = (0, 0, 9)
    _aim(original_light, (0, 0, 0))

    camera_data = bpy.data.cameras.new("Warehouse_Camera")
    camera = bpy.data.objects.new("Warehouse_Camera", camera_data)
    geometry.objects.link(camera)
    # Keep the acceptance camera inside the shell so the south wall does not
    # occlude the complete warehouse interior.
    camera.location = (17, -8, 5.5)
    camera_data.lens = 28
    _aim(camera, (-6, 2, 2.2))
    scene.camera = camera

    world = bpy.data.worlds.new("Fixture_Daylight_World")
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.38, 0.48, 0.65, 1.0)
    background.inputs["Strength"].default_value = 0.55
    scene.world = world

    scene["blend_ai_fixture"] = True
    scene["clear_skylight_target"] = [-10.0, 0.0, 0.1]
    scene["blocked_skylight_target"] = [10.0, 0.0, 0.1]
    return scene


def _output_path() -> str | None:
    output = os.environ.get("BLEND_AI_FIXTURE_OUTPUT")
    if "--" in sys.argv:
        trailing = sys.argv[sys.argv.index("--") + 1 :]
        if trailing:
            output = trailing[0]
    return os.path.abspath(output) if output else None


if __name__ == "__main__":
    build_scene()
    output_path = _output_path()
    if output_path:
        bpy.ops.wm.save_as_mainfile(filepath=output_path)
        print(f"Saved miniature warehouse fixture: {output_path}")
    else:
        print("Miniature warehouse fixture created in memory; not saved")
