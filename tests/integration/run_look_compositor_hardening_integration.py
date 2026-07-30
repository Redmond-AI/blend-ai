"""Blender-native smoke for nested compositor ownership and output containment.

Run only in an isolated Blender process::

    /Applications/Blender.app/Contents/MacOS/Blender \
      --background --factory-startup \
      --python tests/integration/run_look_compositor_hardening_integration.py \
      -- --confirm-isolated-process

Blender 4.2's embedded root compositor is covered by focused unit tests and the
broader save/reopen integration.  This smoke targets Blender 5.x reusable
compositor groups, where shallow nested-group ownership is the extra hazard.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

import bpy


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-isolated-process", action="store_true")
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(values)


def _load_adapter():
    path = Path(__file__).resolve().parents[2] / "addon" / "look_compositor.py"
    spec = importlib.util.spec_from_file_location("addon.look_compositor", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load compositor adapter from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configure_file_output(node, *, directory: str, file_name: str, item_name: str):
    node.directory = directory
    node.file_name = file_name
    node.file_output_items.clear()
    item = node.file_output_items.new(socket_type="RGBA", name=item_name)
    if hasattr(item, "path"):
        item.path = f"../../outside/{item_name}_####"
    if hasattr(item, "file_name"):
        item.file_name = f"../../outside/{item_name}_####"
    return item


def _assert_contained_outputs(adapter, tree, root: str) -> None:
    for nested_tree in [tree, *adapter.owned_nested_compositor_trees(tree)]:
        for node in nested_tree.nodes:
            if node.bl_idname != "CompositorNodeOutputFile":
                continue
            if os.path.commonpath((root, node.directory)) != root:
                raise AssertionError(f"File Output escaped root: {node.directory}")
            if "/" in node.file_name or "\\" in node.file_name or ".." in node.file_name:
                raise AssertionError(f"Unsafe 5.x File Output file_name: {node.file_name}")
            for item in node.file_output_items:
                for attribute in ("path", "file_name"):
                    if not hasattr(item, attribute):
                        continue
                    value = str(getattr(item, attribute))
                    if "/" in value or "\\" in value or ".." in value:
                        raise AssertionError(
                            f"Unsafe 5.x File Output item {attribute}: {value}"
                        )


def main() -> None:
    arguments = _arguments()
    if not arguments.confirm_isolated_process:
        raise RuntimeError("This test requires --confirm-isolated-process")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    if not hasattr(scene, "compositing_node_group"):
        print(
            json.dumps(
                {
                    "status": "skipped",
                    "reason": "Blender build has no 5.x compositing_node_group API",
                    "blender_version": bpy.app.version_string,
                },
                sort_keys=True,
            )
        )
        return

    adapter = _load_adapter()
    child = bpy.data.node_groups.new("Artist Nested Group", "CompositorNodeTree")
    nested_grade = child.nodes.new("CompositorNodeInvert")
    nested_grade.name = "Nested Grade"
    nested_output = child.nodes.new("CompositorNodeOutputFile")
    nested_output.name = "Nested File Output"
    _configure_file_output(
        nested_output,
        directory="//artist/nested",
        file_name="../../nested_####",
        item_name="Nested Beauty",
    )

    root_tree = bpy.data.node_groups.new("Artist Root Group", "CompositorNodeTree")
    group_node = root_tree.nodes.new("CompositorNodeGroup")
    group_node.name = "Nested Group"
    group_node.node_tree = child
    root_output = root_tree.nodes.new("CompositorNodeOutputFile")
    root_output.name = "Root File Output"
    _configure_file_output(
        root_output,
        directory="//artist/root",
        file_name="../../root_####",
        item_name="Root Beauty",
    )
    scene.compositing_node_group = root_tree

    owned = adapter.ensure_unique_managed_compositor(
        scene,
        name="AI_LOOK_Hardening_Compositor",
        source_tree=root_tree,
    )
    owned_child = owned.nodes["Nested Group"].node_tree
    if owned is root_tree or owned_child is child:
        raise AssertionError("Managed KEEP copy retained a shared compositor datablock")
    if owned_child.get(adapter.MANAGED_NESTED_TREE_PROP) is not True:
        raise AssertionError("Owned child group lacks its nested ownership marker")

    first_hash = adapter.compositor_hash(scene, tree=owned)
    owned_child.nodes["Nested Grade"].mute = True
    if child.nodes["Nested Grade"].mute:
        raise AssertionError("Owned nested edit leaked into the artist group")
    second_hash = adapter.compositor_hash(scene, tree=owned)
    if first_hash == second_hash:
        raise AssertionError("Nested compositor mutation did not change compositor hash")

    before = adapter.snapshot_file_outputs(owned)
    output_root = tempfile.mkdtemp(prefix="blend-ai-compositor-output-")
    redirected = adapter.redirect_file_outputs(owned, output_root)
    if len(redirected) != 2:
        raise AssertionError(f"Expected two recursive File Outputs, got {len(redirected)}")
    _assert_contained_outputs(adapter, owned, output_root)
    adapter.restore_file_outputs(owned, redirected)
    after = adapter.snapshot_file_outputs(owned)
    if after != before:
        raise AssertionError("Recursive File Output settings were not restored exactly")

    print("LOOK_COMPOSITOR_HARDENING_INTEGRATION_RESULT")
    print(
        json.dumps(
            {
                "status": "passed",
                "blender_version": bpy.app.version_string,
                "nested_groups": [
                    tree.name for tree in adapter.owned_nested_compositor_trees(owned)
                ],
                "file_outputs": len(redirected),
                "hash_changed_after_nested_edit": first_hash != second_hash,
                "output_root": output_root,
                "exact_restore": after == before,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
