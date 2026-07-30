"""Blender-native compositor ownership acceptance for look profiles.

Run in an isolated Blender process, for example::

    /Applications/Blender.app/Contents/MacOS/Blender \
      --background --factory-startup \
      --python tests/integration/run_look_profile_compositor_integration.py \
      -- --confirm-isolated-process

This intentionally lives outside pytest discovery.  It renders, saves a blend,
and reopens that file in-process, so it must never run inside an artist session.
It does not import or modify the add-on or MCP client implementation.

Blender 5.0 introduced reusable compositor node-group datablocks through
``Scene.compositing_node_group``.  Blender 4.2 instead exposes an embedded tree
through ``Scene.use_nodes`` and ``Scene.node_tree``.  The helpers below keep that
compatibility boundary explicit rather than guessing from a version string.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
from typing import Any

import bpy


SOURCE_SCENE_NAME = "Look Profile Source"
LINKED_SCENE_NAME = "Look Profile Linked Copy"
SOURCE_TREE_NAME = "Look_Profile_Source_Compositor"
LINKED_TREE_NAME = "Look_Profile_Linked_Compositor"
RENDER_LAYERS_NODE = "LOOK_PROFILE_RAW_COMBINED"
INVERT_NODE = "LOOK_PROFILE_FINAL_GRADE"
COMPOSITE_NODE = "LOOK_PROFILE_COMPOSITE"
FILE_OUTPUT_NODE = "LOOK_PROFILE_RAW_FILE_OUTPUT"
ORIGINAL_FILE_OUTPUT_PATH = "//artist_original_output"
LINKED_FILE_OUTPUT_PATH = "//linked_profile_output"


def _script_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--confirm-isolated-process",
        action="store_true",
        help="Confirm that replacing Blender's current file is safe.",
    )
    parser.add_argument(
        "--artifact-parent",
        type=Path,
        default=Path(tempfile.gettempdir()),
        help="Parent directory for a new, uniquely named persistent artifact folder.",
    )
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(arguments)


def _require_isolated_process(arguments: argparse.Namespace) -> None:
    if bpy.app.binary_path and not arguments.confirm_isolated_process:
        raise RuntimeError(
            "A Blender executable run requires --confirm-isolated-process"
        )
    if bpy.data.filepath:
        raise RuntimeError(f"Refusing to replace loaded file {bpy.data.filepath!r}")


def _new_artifact_directory(parent: Path) -> Path:
    parent = parent.expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="blend-ai-look-profile-", dir=parent))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _png_dimensions(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError(f"{path} is not a PNG")
    if header[12:16] != b"IHDR":
        raise AssertionError(f"{path} has no PNG IHDR header")
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        raise AssertionError(f"{path} has invalid dimensions {width}x{height}")
    return width, height


def _read_image_pixels(
    path: Path,
    *,
    require_png: bool = False,
) -> tuple[tuple[int, int], list[float]]:
    """Read a generated image back through Blender's image decoder."""
    png_dimensions = _png_dimensions(path) if require_png else None
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        dimensions = (int(image.size[0]), int(image.size[1]))
        pixels = list(image.pixels[:])
    finally:
        bpy.data.images.remove(image)
    if png_dimensions is not None and dimensions != png_dimensions:
        raise AssertionError(
            f"Blender decoded {path} as {dimensions}, not {png_dimensions}"
        )
    expected_values = dimensions[0] * dimensions[1] * 4
    if len(pixels) != expected_values:
        raise AssertionError(
            f"Blender decoded {len(pixels)} values, expected {expected_values}"
        )
    return dimensions, pixels


def _mean_absolute_rgb_difference(first: list[float], second: list[float]) -> float:
    if len(first) != len(second) or len(first) % 4:
        raise AssertionError("RGBA buffers are incompatible")
    difference = sum(
        abs(first[index + channel] - second[index + channel])
        for index in range(0, len(first), 4)
        for channel in range(3)
    )
    return difference / (len(first) // 4 * 3)


def _compositor_api(scene: bpy.types.Scene | None = None) -> str:
    # In Blender 5.2 these extension-RNA properties are visible on Scene
    # instances but, unlike ordinary RNA, not through ``bpy.types.Scene``.
    scene = scene or bpy.context.scene
    if hasattr(scene, "compositing_node_group"):
        return "SCENE_COMPOSITING_NODE_GROUP"
    if hasattr(scene, "node_tree"):
        return "SCENE_EMBEDDED_NODE_TREE"
    raise RuntimeError("This Blender build exposes no supported compositor API")


def _new_compositor_tree(scene: bpy.types.Scene, name: str):
    if _compositor_api(scene) == "SCENE_COMPOSITING_NODE_GROUP":
        tree = bpy.data.node_groups.new(name=name, type="CompositorNodeTree")
        scene.compositing_node_group = tree
        return tree

    scene.use_nodes = True
    tree = scene.node_tree
    if tree is None:
        raise RuntimeError("Blender did not create the Scene compositor tree")
    try:
        tree.name = name
    except (AttributeError, TypeError):
        # Blender 4.2's embedded compositor may not expose a writable ID name.
        pass
    return tree


def _compositor_tree(scene: bpy.types.Scene):
    if _compositor_api(scene) == "SCENE_COMPOSITING_NODE_GROUP":
        tree = scene.compositing_node_group
    else:
        tree = scene.node_tree if scene.use_nodes else None
    if tree is None:
        raise AssertionError(f"Scene {scene.name!r} has no compositor tree")
    return tree


def _find_node(tree, name: str):
    node = tree.nodes.get(name)
    if node is None:
        raise AssertionError(f"Compositor {tree.name!r} has no node {name!r}")
    return node


def _file_output_path_property(node) -> str:
    if hasattr(node, "directory"):
        return "directory"
    if hasattr(node, "base_path"):
        return "base_path"
    raise RuntimeError("File Output node has no supported directory property")


def _file_output_path(node) -> str:
    return str(getattr(node, _file_output_path_property(node)))


def _set_file_output_path(node, value: str) -> None:
    setattr(node, _file_output_path_property(node), value)


def _configure_file_output(node):
    """Configure and return one RGBA input across Blender 5.2 and 4.2."""
    _set_file_output_path(node, ORIGINAL_FILE_OUTPUT_PATH)
    if hasattr(node, "file_output_items"):
        # Blender 5.2 replaced file_slots with typed file_output_items and
        # split base_path into directory + file_name.  The node-level format
        # is multilayer EXR-only; ordinary PNG is an item-level override.
        node.file_name = "raw_combined_"
        node.file_output_items.clear()
        item = node.file_output_items.new(socket_type="RGBA", name="Raw Combined")
        item.override_node_format = True
        item.format.file_format = "PNG"
        item.format.color_mode = "RGBA"
        item.save_as_render = True
        socket = node.inputs.get("Raw Combined")
    else:
        node.format.file_format = "PNG"
        node.format.color_mode = "RGBA"
        node.file_slots[0].path = "raw_combined_"
        socket = node.inputs[0]
    if socket is None:
        raise AssertionError("File Output node did not expose its RGBA input")
    return socket


def _clear_scene_objects(scene: bpy.types.Scene) -> None:
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def _select_eevee_engine(scene: bpy.types.Scene) -> str:
    """Use the renamed engine identifiers on Blender 5.x and 4.2."""
    failures: list[str] = []
    for identifier in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        try:
            scene.render.engine = identifier
            return identifier
        except (TypeError, ValueError) as exc:
            failures.append(f"{identifier}: {exc}")
    raise RuntimeError("No supported EEVEE engine: " + "; ".join(failures))


def _configure_source_scene() -> tuple[bpy.types.Scene, Any]:
    bpy.ops.wm.read_factory_settings(use_empty=False)
    scene = bpy.context.scene
    scene.name = SOURCE_SCENE_NAME
    _clear_scene_objects(scene)

    camera_data = bpy.data.cameras.new("Look Profile Camera Data")
    camera = bpy.data.objects.new("Look Profile Camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera

    world = bpy.data.worlds.new("Look Profile World")
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is None:
        raise AssertionError("World has no Background node")
    background.inputs["Color"].default_value = (0.08, 0.22, 0.55, 1.0)
    background.inputs["Strength"].default_value = 0.8
    scene.world = world

    engine = _select_eevee_engine(scene)
    scene.render.resolution_x = 64
    scene.render.resolution_y = 48
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.use_file_extension = True
    scene.frame_set(1)

    tree = _new_compositor_tree(scene, SOURCE_TREE_NAME)
    tree.nodes.clear()

    render_layers = tree.nodes.new("CompositorNodeRLayers")
    render_layers.name = RENDER_LAYERS_NODE
    render_layers.label = "Raw Combined"

    invert = tree.nodes.new("CompositorNodeInvert")
    invert.name = INVERT_NODE
    invert.label = "Deterministic Final Grade"
    factor_input = invert.inputs.get("Factor") or invert.inputs.get("Fac")
    color_input = invert.inputs.get("Color") or invert.inputs.get("Image")
    if factor_input is None or color_input is None:
        raise AssertionError(
            "Invert node lacks the expected Factor/Fac and Color/Image inputs"
        )
    factor_input.default_value = 1.0

    if _compositor_api(scene) == "SCENE_COMPOSITING_NODE_GROUP":
        # Blender 5.2 removed CompositorNodeComposite.  A top-level reusable
        # compositor group publishes its final result through a group output.
        tree.interface.new_socket(
            name="Image",
            in_out="OUTPUT",
            socket_type="NodeSocketColor",
        )
        composite = tree.nodes.new("NodeGroupOutput")
        composite.name = COMPOSITE_NODE
        composite.label = "Final Composite Group Output"
    else:
        # Blender 4.2 uses the embedded Scene compositor and legacy Composite.
        composite = tree.nodes.new("CompositorNodeComposite")
        composite.name = COMPOSITE_NODE

    file_output = tree.nodes.new("CompositorNodeOutputFile")
    file_output.name = FILE_OUTPUT_NODE
    file_output.label = "Raw Combined Probe"
    file_output_input = _configure_file_output(file_output)

    combined = render_layers.outputs.get("Image")
    if combined is None:
        raise AssertionError("Render Layers node has no Combined/Image output")
    tree.links.new(combined, color_input)
    final_input = composite.inputs.get("Image") or composite.inputs[0]
    tree.links.new(invert.outputs[0], final_input)
    tree.links.new(combined, file_output_input)
    return scene, engine


def _render_and_compare(
    scene: bpy.types.Scene,
    artifact_directory: Path,
) -> dict[str, Any]:
    tree = _compositor_tree(scene)
    file_output = _find_node(tree, FILE_OUTPUT_NODE)
    original_base_path = _file_output_path(file_output)
    original_render_path = str(scene.render.filepath)
    original_use_compositing = bool(scene.render.use_compositing)
    file_output_directory = artifact_directory / "redirected_file_output"
    file_output_directory.mkdir()
    raw_path = artifact_directory / "raw_combined.png"
    final_path = artifact_directory / "final_composite.png"

    try:
        # With compositing disabled, Blender writes the unprocessed Combined
        # render to a normal PNG.  Blender 5.2's multilayer File Output EXR is
        # not directly reloadable through bpy.data.images, so this is also the
        # portable 4.2/5.2 pixel-comparison path.
        scene.render.use_compositing = False
        scene.render.filepath = str(raw_path)
        raw_result = bpy.ops.render.render(write_still=True, scene=scene.name)
        if raw_result is not None and "CANCELLED" in raw_result:
            raise RuntimeError("Blender cancelled the raw Combined render")

        scene.render.use_compositing = True
        _set_file_output_path(file_output, str(file_output_directory))
        scene.render.filepath = str(final_path)
        result = bpy.ops.render.render(write_still=True, scene=scene.name)
        if result is not None and "CANCELLED" in result:
            raise RuntimeError("Blender cancelled the compositor prototype render")
    finally:
        _set_file_output_path(file_output, original_base_path)
        scene.render.filepath = original_render_path
        scene.render.use_compositing = original_use_compositing

    if _file_output_path(file_output) != ORIGINAL_FILE_OUTPUT_PATH:
        raise AssertionError("File Output base_path was not exactly restored")

    file_output_candidates = sorted(
        path for path in file_output_directory.rglob("*") if path.is_file()
    )
    if len(file_output_candidates) != 1:
        raise AssertionError(
            f"Expected one redirected File Output artifact, found "
            f"{len(file_output_candidates)}: {file_output_candidates}"
        )
    file_output_artifact = file_output_candidates[0]

    if not raw_path.is_file():
        raise AssertionError("write_still did not write the raw Combined PNG")
    if not final_path.is_file():
        raise AssertionError("write_still did not write the final Composite PNG")

    raw_dimensions, raw_pixels = _read_image_pixels(raw_path, require_png=True)
    final_dimensions, final_pixels = _read_image_pixels(final_path, require_png=True)
    if raw_dimensions != final_dimensions:
        raise AssertionError(
            f"Raw and final dimensions differ: {raw_dimensions} != {final_dimensions}"
        )
    difference = _mean_absolute_rgb_difference(raw_pixels, final_pixels)
    if difference <= 0.10:
        raise AssertionError(
            f"Final Composite did not materially differ from raw Combined: {difference}"
        )
    raw_digest = _sha256(raw_path)
    final_digest = _sha256(final_path)
    if raw_digest == final_digest:
        raise AssertionError("Raw Combined and final Composite PNGs are byte-identical")

    return {
        "raw_combined_path": str(raw_path),
        "raw_combined_format": "png",
        "final_composite_path": str(final_path),
        "dimensions": list(raw_dimensions),
        "raw_sha256": raw_digest,
        "final_sha256": final_digest,
        "mean_absolute_rgb_difference": difference,
        "file_output_redirected_to": str(file_output_directory),
        "file_output_artifact_path": str(file_output_artifact),
        "file_output_artifact_format": file_output_artifact.suffix.lower().lstrip("."),
        "file_output_path_property": _file_output_path_property(file_output),
        "file_output_restored_to": _file_output_path(file_output),
        "final_png_reloaded_by_blender": True,
    }


def _link_copy_scene(source: bpy.types.Scene) -> bpy.types.Scene:
    window = getattr(bpy.context, "window", None)
    if window is None:
        raise RuntimeError("LINK_COPY requires a Blender window context")
    window.scene = source
    before = {scene.as_pointer() for scene in bpy.data.scenes}
    result = bpy.ops.scene.new(type="LINK_COPY")
    if result is not None and "CANCELLED" in result:
        raise RuntimeError("Blender cancelled bpy.ops.scene.new(type='LINK_COPY')")
    linked = window.scene
    if linked.as_pointer() in before or linked == source:
        raise AssertionError("LINK_COPY did not create and activate a new Scene")
    linked.name = LINKED_SCENE_NAME
    return linked


def _detach_linked_compositor(
    source: bpy.types.Scene,
    linked: bpy.types.Scene,
) -> bool:
    source_tree = _compositor_tree(source)
    linked_tree_before = _compositor_tree(linked)
    shared_before_detach = linked_tree_before == source_tree

    if (
        _compositor_api(source) == "SCENE_COMPOSITING_NODE_GROUP"
        and shared_before_detach
    ):
        owned_tree = linked_tree_before.copy()
        owned_tree.name = LINKED_TREE_NAME
        linked.compositing_node_group = owned_tree
    elif shared_before_detach:
        raise RuntimeError(
            "Blender 4.2 LINK_COPY unexpectedly shared its embedded Scene.node_tree; "
            "a generic embedded-tree clone routine is required before 4.2 support"
        )
    else:
        # Blender 5.2 LINK_COPY currently creates an owned compositor group on
        # its own.  Preserve that native copy so save/reopen verifies the
        # operator's behavior rather than a redundant test-created duplicate.
        try:
            linked_tree_before.name = LINKED_TREE_NAME
        except (AttributeError, TypeError):
            pass

    linked_tree = _compositor_tree(linked)
    if linked_tree == source_tree:
        raise AssertionError("Linked profile Scene still shares its compositor")

    linked_invert = _find_node(linked_tree, INVERT_NODE)
    source_invert = _find_node(source_tree, INVERT_NODE)
    linked_invert.mute = True
    if source_invert.mute:
        raise AssertionError("Linked compositor edit leaked into the source Scene")

    linked_output = _find_node(linked_tree, FILE_OUTPUT_NODE)
    source_output = _find_node(source_tree, FILE_OUTPUT_NODE)
    _set_file_output_path(linked_output, LINKED_FILE_OUTPUT_PATH)
    if _file_output_path(source_output) != ORIGINAL_FILE_OUTPUT_PATH:
        raise AssertionError("Linked File Output path leaked into the source Scene")
    return shared_before_detach


def _save_reopen_and_verify(
    artifact_directory: Path,
    shared_before_detach: bool,
) -> dict[str, Any]:
    blend_path = artifact_directory / "look_profile_compositor_ownership.blend"
    result = bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    if result is not None and "CANCELLED" in result:
        raise RuntimeError("Blender cancelled save_as_mainfile")
    if not blend_path.is_file():
        raise AssertionError("Blender did not save the ownership prototype blend")

    result = bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    if result is not None and "CANCELLED" in result:
        raise RuntimeError("Blender cancelled open_mainfile")

    source = bpy.data.scenes.get(SOURCE_SCENE_NAME)
    linked = bpy.data.scenes.get(LINKED_SCENE_NAME)
    if source is None or linked is None:
        raise AssertionError("Source or linked Scene was lost after save/reopen")
    source_tree = _compositor_tree(source)
    linked_tree = _compositor_tree(linked)
    if source_tree == linked_tree:
        raise AssertionError("Compositor ownership collapsed after save/reopen")

    source_invert = _find_node(source_tree, INVERT_NODE)
    linked_invert = _find_node(linked_tree, INVERT_NODE)
    source_output = _find_node(source_tree, FILE_OUTPUT_NODE)
    linked_output = _find_node(linked_tree, FILE_OUTPUT_NODE)
    if source_invert.mute or not linked_invert.mute:
        raise AssertionError("Profile-specific compositor node state did not persist")
    if _file_output_path(source_output) != ORIGINAL_FILE_OUTPUT_PATH:
        raise AssertionError("Source File Output path did not persist")
    if _file_output_path(linked_output) != LINKED_FILE_OUTPUT_PATH:
        raise AssertionError("Linked File Output path did not persist")

    return {
        "blend_path": str(blend_path),
        "link_copy_shared_compositor_before_detach": shared_before_detach,
        "source_compositor": source_tree.name,
        "linked_compositor": linked_tree.name,
        "source_file_output_path": _file_output_path(source_output),
        "linked_file_output_path": _file_output_path(linked_output),
        "source_invert_muted": bool(source_invert.mute),
        "linked_invert_muted": bool(linked_invert.mute),
        "save_reopen_independent_ownership": True,
    }


def main() -> None:
    arguments = _script_arguments()
    _require_isolated_process(arguments)
    artifact_directory = _new_artifact_directory(arguments.artifact_parent)
    source, engine = _configure_source_scene()
    render_findings = _render_and_compare(source, artifact_directory)
    linked = _link_copy_scene(source)
    shared_before_detach = _detach_linked_compositor(source, linked)
    ownership_findings = _save_reopen_and_verify(
        artifact_directory,
        shared_before_detach,
    )

    findings = {
        "status": "passed",
        "blender_version": bpy.app.version_string,
        "blender_version_tuple": list(bpy.app.version),
        "compositor_api": _compositor_api(bpy.context.scene),
        "scene_has_node_tree": hasattr(bpy.context.scene, "node_tree"),
        "scene_has_compositing_node_group": hasattr(
            bpy.context.scene,
            "compositing_node_group",
        ),
        "render_engine": engine,
        "artifact_directory": str(artifact_directory),
        **render_findings,
        **ownership_findings,
    }
    report_path = artifact_directory / "result.json"
    report_path.write_text(json.dumps(findings, indent=2, sort_keys=True) + "\n")
    print("LOOK_PROFILE_COMPOSITOR_INTEGRATION_RESULT")
    print(json.dumps(findings, indent=2, sort_keys=True))
    print(f"Result report: {report_path}")


if __name__ == "__main__":
    main()
