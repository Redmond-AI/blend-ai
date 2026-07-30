"""Build an unsaved 20,000-instance spatial-relighting benchmark scene.

The source mesh contains 256 evaluated triangles, so its 20,000 collection
instances represent 5.12 million evaluated triangles.  This file is intended
for ``live_blender_bootstrap.py --fixture scale`` in a new visible
``--factory-startup`` process; it never saves a blend file.
"""

from __future__ import annotations

import bpy
from mathutils import Vector


SCENE_NAME = "Blend AI Relighting Scale Benchmark"
INSTANCE_COUNT = 20_000
GRID_COLUMNS = 200
GRID_ROWS = 100
GRID_SPACING = 1.25
SOURCE_TRIANGLES = 256


def _reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in list(bpy.data.collections):
        if collection.users == 0:
            bpy.data.collections.remove(collection)


def _source_mesh() -> bpy.types.Mesh:
    """Return a 16 by 8 quad grid (256 evaluated triangles)."""
    x_cells = 16
    y_cells = 8
    vertices = []
    for row in range(y_cells + 1):
        for column in range(x_cells + 1):
            vertices.append(
                (
                    (column / x_cells - 0.5) * 0.8,
                    (row / y_cells - 0.5) * 0.8,
                    0.0,
                )
            )
    faces = []
    stride = x_cells + 1
    for row in range(y_cells):
        for column in range(x_cells):
            lower_left = row * stride + column
            faces.append(
                (
                    lower_left,
                    lower_left + 1,
                    lower_left + stride + 1,
                    lower_left + stride,
                )
            )
    mesh = bpy.data.meshes.new("Scale_Source_256_Triangles")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    if len(mesh.loop_triangles) != SOURCE_TRIANGLES:
        mesh.calc_loop_triangles()
    if len(mesh.loop_triangles) != SOURCE_TRIANGLES:
        raise RuntimeError(
            f"Scale source has {len(mesh.loop_triangles)} triangles; "
            f"expected {SOURCE_TRIANGLES}"
        )
    return mesh


def _aim(obj: bpy.types.Object, target: tuple[float, float, float]) -> None:
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def build_scene() -> bpy.types.Scene:
    _reset_scene()
    scene = bpy.context.scene
    scene.name = SCENE_NAME
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.render.resolution_x = 640
    scene.render.resolution_y = 360
    scene.render.resolution_percentage = 100

    benchmark = bpy.data.collections.new("Scale_Instances_20000")
    scene.collection.children.link(benchmark)
    source = bpy.data.collections.new("Scale_Source_Collection")
    mesh = _source_mesh()
    material = bpy.data.materials.new("Scale_Benchmark_Material")
    material.diffuse_color = (0.18, 0.24, 0.32, 1.0)
    source_object = bpy.data.objects.new("Scale_Source_Mesh", mesh)
    source_object.data.materials.append(material)
    source.objects.link(source_object)

    for index in range(INSTANCE_COUNT):
        column = index % GRID_COLUMNS
        row = index // GRID_COLUMNS
        instance = bpy.data.objects.new(f"Scale_Instance_{index:05d}", None)
        instance.instance_type = "COLLECTION"
        instance.instance_collection = source
        instance.location = (
            (column - (GRID_COLUMNS - 1) / 2) * GRID_SPACING,
            (row - (GRID_ROWS - 1) / 2) * GRID_SPACING,
            0.0,
        )
        benchmark.objects.link(instance)

    camera_data = bpy.data.cameras.new("Scale_Benchmark_Camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 300.0
    camera = bpy.data.objects.new("Scale_Benchmark_Camera", camera_data)
    benchmark.objects.link(camera)
    camera.location = (0.0, -175.0, 190.0)
    _aim(camera, (0.0, 0.0, 0.0))
    scene.camera = camera

    world = bpy.data.worlds.new("Scale_Benchmark_World")
    world.color = (0.02, 0.02, 0.02)
    scene.world = world

    scene["blend_ai_scale_fixture"] = True
    scene["expected_instance_count"] = INSTANCE_COUNT
    scene["source_triangle_count"] = SOURCE_TRIANGLES
    scene["expected_evaluated_triangles"] = INSTANCE_COUNT * SOURCE_TRIANGLES
    bpy.context.view_layer.update()
    return scene


if __name__ == "__main__":
    built = build_scene()
    print(
        f"Built unsaved {built.name}: {INSTANCE_COUNT} instances, "
        f"{INSTANCE_COUNT * SOURCE_TRIANGLES} evaluated triangles"
    )
