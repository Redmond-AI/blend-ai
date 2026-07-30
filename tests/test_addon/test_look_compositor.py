"""Focused tests for managed compositor API compatibility and side effects."""

from __future__ import annotations

import importlib.util
import os
from types import SimpleNamespace

import pytest


def _load_module():
    path = os.path.join(
        os.path.dirname(__file__), "..", "..", "addon", "look_compositor.py"
    )
    spec = importlib.util.spec_from_file_location("addon.look_compositor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def compositor():
    return _load_module()


class _Marked:
    def __init__(self):
        self._custom_properties = {}

    def __getitem__(self, key):
        return self._custom_properties[key]

    def __setitem__(self, key, value):
        self._custom_properties[key] = value

    def __delitem__(self, key):
        del self._custom_properties[key]

    def get(self, key, default=None):
        return self._custom_properties.get(key, default)


class _Socket:
    def __init__(self, name, default_value=None):
        self.name = name
        self.default_value = default_value
        self.is_linked = False
        self.node = None


class _Sockets(list):
    def get(self, name):
        return next((socket for socket in self if socket.name == name), None)


_NODE_LAYOUTS = {
    "CompositorNodeRLayers": ([], ["Image"]),
    "CompositorNodeGlare": (
        ["Image", "Type", "Quality", "Threshold", "Strength", "Size"],
        ["Image"],
    ),
    "CompositorNodeImage": ([], ["Image", "Alpha"]),
    "CompositorNodeScale": (
        ["Image", "Type", "X", "Y", "Frame Type", "Interpolation"],
        ["Image"],
    ),
    "CompositorNodeAlphaOver": (["Background", "Foreground", "Factor"], ["Image"]),
    "CompositorNodeEllipseMask": (
        ["Operation", "Mask", "Value", "Position", "Size", "Rotation"],
        ["Mask"],
    ),
    "CompositorNodeBlur": (["Image", "Size", "Type"], ["Image"]),
    "CompositorNodeInvert": (["Color", "Factor"], ["Color"]),
    "CompositorNodeRGB": ([], ["Color"]),
    "CompositorNodeSetAlpha": (["Image", "Alpha"], ["Image"]),
    "NodeGroupOutput": (["Image"], []),
    "CompositorNodeComposite": (["Image"], []),
    "CompositorNodeMixRGB": (["Fac", "Image", "Image"], ["Image"]),
    "CompositorNodeOutputFile": (["Image"], []),
    "CompositorNodeGroup": (["Image"], ["Image"]),
}


class _Node(_Marked):
    def __init__(self, bl_idname, *, api="5x"):
        super().__init__()
        self.bl_idname = bl_idname
        self.name = bl_idname
        self.label = ""
        self.mute = False
        self.is_active_output = False
        inputs, outputs = _NODE_LAYOUTS[bl_idname]
        if api == "legacy":
            if bl_idname == "CompositorNodeGlare":
                inputs = ["Image"]
            elif bl_idname == "CompositorNodeScale":
                inputs = ["Image", "X", "Y"]
            elif bl_idname == "CompositorNodeAlphaOver":
                inputs = ["Fac", "Image", "Image"]
            elif bl_idname == "CompositorNodeEllipseMask":
                inputs = ["Mask", "Value"]
            elif bl_idname == "CompositorNodeBlur":
                inputs = ["Image"]
            elif bl_idname == "CompositorNodeInvert":
                inputs = ["Fac", "Color"]
        self.inputs = _Sockets(_Socket(name) for name in inputs)
        self.outputs = _Sockets(_Socket(name) for name in outputs)
        for socket in [*self.inputs, *self.outputs]:
            socket.node = self
        self.scene = None
        self.node_tree = None
        if bl_idname == "CompositorNodeRGB":
            self.outputs[0].default_value = (1.0, 1.0, 1.0, 1.0)


class _Nodes(list):
    def __init__(self, api="5x", unsupported=()):
        super().__init__()
        self.api = api
        self.unsupported = set(unsupported)

    def new(self, bl_idname):
        if bl_idname in self.unsupported:
            raise RuntimeError(f"Node type {bl_idname} undefined")
        node = _Node(bl_idname, api=self.api)
        self.append(node)
        return node

    def remove(self, node):
        super().remove(node)

    def get(self, name):
        return next((node for node in self if node.name == name), None)


class _Link:
    def __init__(self, from_socket, to_socket):
        self.from_socket = from_socket
        self.to_socket = to_socket
        self.from_node = from_socket.node
        self.to_node = to_socket.node


class _Links(list):
    def new(self, from_socket, to_socket):
        link = _Link(from_socket, to_socket)
        self.append(link)
        to_socket.is_linked = True
        return link


class _InterfaceItem:
    def __init__(self, name, in_out, socket_type):
        self.name = name
        self.in_out = in_out
        self.socket_type = socket_type
        self.item_type = "SOCKET"


class _Interface:
    def __init__(self):
        self.items_tree = []

    def clear(self):
        self.items_tree.clear()

    def new_socket(self, *, name, in_out, socket_type):
        item = _InterfaceItem(name, in_out, socket_type)
        self.items_tree.append(item)
        return item


class _Tree(_Marked):
    def __init__(self, name="Tree", *, api="5x", unsupported=()):
        super().__init__()
        self.name = name
        self.api = api
        self.nodes = _Nodes(api=api, unsupported=unsupported)
        self.links = _Links()
        if api == "5x":
            self.interface = _Interface()
        self._copy_registry = None

    def copy(self):
        copied = _Tree(f"{self.name}.copy", api=self.api)
        copied._copy_registry = self._copy_registry
        for source in self.nodes:
            node = copied.nodes.new(source.bl_idname)
            node.name = source.name
            node.label = source.label
            node.mute = source.mute
            node.is_active_output = source.is_active_output
            node.scene = source.scene
            node.node_tree = source.node_tree
            node._custom_properties = dict(source._custom_properties)
            for source_socket, target_socket in zip(source.inputs, node.inputs):
                target_socket.default_value = source_socket.default_value
            for source_socket, target_socket in zip(source.outputs, node.outputs):
                target_socket.default_value = source_socket.default_value
        if hasattr(self, "interface"):
            for item in self.interface.items_tree:
                copied.interface.items_tree.append(
                    _InterfaceItem(item.name, item.in_out, item.socket_type)
                )
        if self._copy_registry is not None:
            self._copy_registry.append(copied)
        return copied


class _Pixels(list):
    def foreach_set(self, values):
        self[:] = values


class _Image(_Marked):
    def __init__(self, name, width, height):
        super().__init__()
        self.name = name
        self.size = (width, height)
        self.pixels = _Pixels()
        self.updated = False

    def update(self):
        self.updated = True


class _Images(dict):
    def new(self, name, width, height, **_kwargs):
        image = _Image(name, width, height)
        self[name] = image
        return image


class _NodeGroups(list):
    def new(self, name, _tree_type):
        tree = _Tree(name, api="5x")
        tree._copy_registry = self
        self.append(tree)
        return tree

    def remove(self, tree, **_kwargs):
        super().remove(tree)


class _GroupScene:
    def __init__(self, tree=None):
        self.compositing_node_group = tree
        self.use_compositing = False
        self.render = SimpleNamespace(resolution_x=1920, resolution_y=1080)


class _LegacyScene:
    def __init__(self, tree):
        self.node_tree = tree
        self.use_nodes = False
        self.render = SimpleNamespace(resolution_x=1920, resolution_y=1080)


def _install_fake_bpy(module, *copyable_trees):
    images = _Images()
    node_groups = _NodeGroups()
    for tree in copyable_trees:
        tree._copy_registry = node_groups
    module.bpy = SimpleNamespace(
        data=SimpleNamespace(node_groups=node_groups, images=images)
    )
    return images


def _managed_group(module, *, unsupported=()):
    tree = _Tree(api="5x", unsupported=unsupported)
    scene = _GroupScene(tree)
    tree[module.MANAGED_TREE_PROP] = True
    return scene, tree


def _managed_legacy(module):
    tree = _Tree(api="legacy")
    scene = _LegacyScene(tree)
    tree[module.MANAGED_TREE_PROP] = True
    return scene, tree


def _post(**overrides):
    value = {
        "mode": "MANAGED_STACK",
        "bloom": {"enabled": False},
        "grain": {"enabled": False},
        "vignette": {"enabled": False},
    }
    value.update(overrides)
    return value


def test_detects_scene_api_by_concrete_attributes(compositor):
    assert compositor.detect_compositor_api(_GroupScene()) == compositor.API_NODE_GROUP
    assert (
        compositor.detect_compositor_api(_LegacyScene(_Tree(api="legacy")))
        == compositor.API_LEGACY
    )
    with pytest.raises(compositor.CompositorCompatibilityError, match="neither"):
        compositor.detect_compositor_api(SimpleNamespace())


def test_ensure_unique_group_copies_and_legacy_rejects_sharing(compositor):
    artist = _Tree("Artist")
    _install_fake_bpy(compositor, artist)
    scene = _GroupScene(artist)
    managed = compositor.ensure_unique_managed_compositor(
        scene, name="AI_LOOK_Profile_Compositor", source_tree=artist
    )
    assert managed is not artist
    assert scene.compositing_node_group is managed
    assert managed[compositor.MANAGED_TREE_PROP] is True
    assert scene.use_compositing is True

    legacy_tree = _Tree(api="legacy")
    legacy_scene = _LegacyScene(legacy_tree)
    with pytest.raises(compositor.CompositorCompatibilityError, match="shares"):
        compositor.ensure_unique_managed_compositor(
            legacy_scene, name="AI_LOOK_Legacy", source_tree=legacy_tree
        )


def test_ensure_unique_group_deep_copies_nested_groups_and_preserves_sharing(compositor):
    grandchild = _Tree("Artist Grandchild")
    grade = grandchild.nodes.new("CompositorNodeInvert")
    grade.name = "Nested Grade"

    child = _Tree("Artist Child")
    child_group = child.nodes.new("CompositorNodeGroup")
    child_group.name = "Grandchild Group"
    child_group.node_tree = grandchild

    artist = _Tree("Artist Root")
    first = artist.nodes.new("CompositorNodeGroup")
    first.name = "First Shared Group"
    first.node_tree = child
    second = artist.nodes.new("CompositorNodeGroup")
    second.name = "Second Shared Group"
    second.node_tree = child
    _install_fake_bpy(compositor, artist, child, grandchild)

    scene = _GroupScene(artist)
    managed = compositor.ensure_unique_managed_compositor(
        scene,
        name="AI_LOOK_Profile_Compositor",
        source_tree=artist,
    )

    managed_first = managed.nodes.get("First Shared Group").node_tree
    managed_second = managed.nodes.get("Second Shared Group").node_tree
    assert managed_first is managed_second
    assert managed_first is not child
    managed_grandchild = managed_first.nodes.get("Grandchild Group").node_tree
    assert managed_grandchild is not grandchild
    assert managed[compositor.MANAGED_TREE_PROP] is True
    assert managed_first[compositor.MANAGED_NESTED_TREE_PROP] is True
    assert managed_grandchild[compositor.MANAGED_NESTED_TREE_PROP] is True
    assert managed_first[compositor.MANAGED_PARENT_PROP] == managed.name
    assert compositor.owned_nested_compositor_trees(managed) == [
        managed_first,
        managed_grandchild,
    ]

    managed_grandchild.nodes.get("Nested Grade").mute = True
    assert grandchild.nodes.get("Nested Grade").mute is False


def test_rebind_render_layers_targets_only_managed_copies(compositor):
    artist_scene = SimpleNamespace(name="Artist Scene")
    profile_scene = SimpleNamespace(name="Profile Scene")
    child = _Tree("Artist Child")
    child_layers = child.nodes.new("CompositorNodeRLayers")
    child_layers.scene = artist_scene
    artist = _Tree("Artist Root")
    root_layers = artist.nodes.new("CompositorNodeRLayers")
    root_layers.scene = artist_scene
    group = artist.nodes.new("CompositorNodeGroup")
    group.node_tree = child
    _install_fake_bpy(compositor, artist, child)
    managed = compositor.ensure_unique_managed_compositor(
        _GroupScene(artist),
        name="AI_LOOK_Profile_Compositor",
        source_tree=artist,
    )

    count = compositor.rebind_render_layers_to_scene(managed, profile_scene)

    assert count == 2
    assert managed.nodes.get(root_layers.name).scene is profile_scene
    managed_child = managed.nodes.get(group.name).node_tree
    assert managed_child.nodes.get(child_layers.name).scene is profile_scene
    assert root_layers.scene is artist_scene
    assert child_layers.scene is artist_scene


def test_ensure_unique_legacy_tree_detaches_nested_group_datablocks(compositor):
    child = _Tree("Artist Child", api="legacy")
    grade = child.nodes.new("CompositorNodeInvert")
    grade.name = "Artist Grade"

    source_root = _Tree("Artist Root", api="legacy")
    source_group = source_root.nodes.new("CompositorNodeGroup")
    source_group.name = "Shared Child"
    source_group.node_tree = child

    compiled_root = _Tree("Compiled Root", api="legacy")
    compiled_group = compiled_root.nodes.new("CompositorNodeGroup")
    compiled_group.name = "Shared Child"
    compiled_group.node_tree = child
    _install_fake_bpy(compositor, source_root, compiled_root, child)
    scene = _LegacyScene(compiled_root)

    managed = compositor.ensure_unique_managed_compositor(
        scene,
        name="AI_LOOK_Legacy_Profile",
        source_tree=source_root,
        copy_source=True,
    )

    owned_child = managed.nodes.get("Shared Child").node_tree
    assert owned_child is not child
    assert owned_child[compositor.MANAGED_NESTED_TREE_PROP] is True
    assert owned_child[compositor.MANAGED_PARENT_PROP] == managed.name
    owned_child.nodes.get("Artist Grade").mute = True
    assert child.nodes.get("Artist Grade").mute is False


def test_ensure_unique_group_rejects_nested_cycles_without_allocating(compositor):
    artist = _Tree("Artist Root")
    child = _Tree("Artist Child")
    down = artist.nodes.new("CompositorNodeGroup")
    down.name = "Down"
    down.node_tree = child
    up = child.nodes.new("CompositorNodeGroup")
    up.name = "Up"
    up.node_tree = artist
    _install_fake_bpy(compositor, artist, child)
    groups = compositor.bpy.data.node_groups
    scene = _GroupScene(artist)

    with pytest.raises(compositor.CompositorCompatibilityError, match="cycle"):
        compositor.ensure_unique_managed_compositor(
            scene,
            name="AI_LOOK_Cyclic",
            source_tree=artist,
        )

    assert scene.compositing_node_group is artist
    assert groups == []


class _FailOnceAssignmentScene:
    def __init__(self, tree):
        self._tree = tree
        self.fail_next_assignment = True
        self.use_compositing = False
        self.render = SimpleNamespace(resolution_x=1920, resolution_y=1080)

    @property
    def compositing_node_group(self):
        return self._tree

    @compositing_node_group.setter
    def compositing_node_group(self, value):
        if self.fail_next_assignment:
            self.fail_next_assignment = False
            raise RuntimeError("injected assignment failure")
        self._tree = value


def test_ensure_unique_group_assignment_failure_is_atomic(compositor):
    artist = _Tree("Artist")
    _install_fake_bpy(compositor, artist)
    groups = compositor.bpy.data.node_groups
    scene = _FailOnceAssignmentScene(artist)

    with pytest.raises(compositor.CompositorCompatibilityError, match="assignment failure"):
        compositor.ensure_unique_managed_compositor(
            scene,
            name="AI_LOOK_Assignment_Failure",
            source_tree=artist,
        )

    assert scene.compositing_node_group is artist
    assert scene.use_compositing is False
    assert groups == []


def test_ensure_unique_group_marker_failure_cleans_every_copy(compositor, monkeypatch):
    child = _Tree("Artist Child")
    artist = _Tree("Artist Root")
    group = artist.nodes.new("CompositorNodeGroup")
    group.name = "Child"
    group.node_tree = child
    _install_fake_bpy(compositor, artist, child)
    groups = compositor.bpy.data.node_groups
    scene = _GroupScene(artist)
    original_set = compositor._custom_set

    def fail_schema(value, key, item):
        if key == compositor.SCHEMA_PROP:
            raise compositor.CompositorCompatibilityError("injected marker failure")
        return original_set(value, key, item)

    monkeypatch.setattr(compositor, "_custom_set", fail_schema)
    with pytest.raises(compositor.CompositorCompatibilityError, match="marker failure"):
        compositor.ensure_unique_managed_compositor(
            scene,
            name="AI_LOOK_Marker_Failure",
            source_tree=artist,
        )

    assert scene.compositing_node_group is artist
    assert scene.use_compositing is False
    assert groups == []


def test_managed_group_pass_through_has_authoritative_link_and_stable_hash(compositor):
    _install_fake_bpy(compositor)
    scene, tree = _managed_group(compositor)
    result = compositor.build_managed_post_stack(scene, _post())

    assert result["node_names"] == ["AI_LOOK_OUTPUT", "AI_LOOK_RENDER_LAYERS"]
    assert compositor.validate_final_image_linked(scene)["node_name"] == "AI_LOOK_OUTPUT"
    first = compositor.compositor_hash(scene)
    assert first == compositor.compositor_hash(scene)

    output = tree.nodes.get("AI_LOOK_OUTPUT")
    output.mute = True
    assert compositor.compositor_hash(scene) != first


def test_compositor_hash_recurses_into_nested_groups(compositor):
    child = _Tree("Child")
    nested_grade = child.nodes.new("CompositorNodeInvert")
    nested_grade.name = "Nested Grade"
    root = _Tree("Root")
    group = root.nodes.new("CompositorNodeGroup")
    group.name = "Nested Group"
    group.node_tree = child
    scene = _GroupScene(root)

    first = compositor.compositor_hash(scene)
    nested_grade.mute = True
    assert compositor.compositor_hash(scene) != first


def test_builds_deterministic_group_effect_stack(compositor):
    images = _install_fake_bpy(compositor)
    scene, tree = _managed_group(compositor)
    result = compositor.build_managed_post_stack(
        scene,
        _post(
            bloom={
                "enabled": True,
                "threshold": 1.2,
                "strength": 0.7,
                "radius": 0.6,
            },
            grain={"enabled": True, "strength": 0.08, "scale": 4.0, "seed": 29},
            vignette={"enabled": True, "strength": 0.35, "feather": 0.45},
        ),
    )

    assert result["node_names"] == sorted(
        [
            "AI_LOOK_BLOOM",
            "AI_LOOK_GRAIN_BLEND",
            "AI_LOOK_GRAIN_IMAGE",
            "AI_LOOK_GRAIN_SCALE",
            "AI_LOOK_OUTPUT",
            "AI_LOOK_RENDER_LAYERS",
            "AI_LOOK_VIGNETTE_ALPHA",
            "AI_LOOK_VIGNETTE_BLEND",
            "AI_LOOK_VIGNETTE_BLUR",
            "AI_LOOK_VIGNETTE_COLOR",
            "AI_LOOK_VIGNETTE_INVERT",
            "AI_LOOK_VIGNETTE_MASK",
        ]
    )
    assert len(images) == 1
    image = next(iter(images.values()))
    assert image.updated is True
    first_pixels = image.pixels[:24]
    invert = tree.nodes.get("AI_LOOK_VIGNETTE_INVERT")
    invert_links = [link for link in tree.links if link.to_node is invert]
    assert [link.to_socket.name for link in invert_links] == ["Color"]

    scene_two, _tree_two = _managed_group(compositor)
    compositor.build_managed_post_stack(
        scene_two,
        _post(grain={"enabled": True, "strength": 0.5, "scale": 4.0, "seed": 29}),
    )
    assert len(images) == 1
    assert next(iter(images.values())).pixels[:24] == first_pixels


def test_unsupported_required_node_fails_loudly_without_artist_rewrite(compositor):
    _install_fake_bpy(compositor)
    scene, tree = _managed_group(compositor, unsupported={"CompositorNodeGlare"})
    with pytest.raises(compositor.CompositorCompatibilityError, match="unsupported"):
        compositor.build_managed_post_stack(
            scene,
            _post(bloom={"enabled": True, "threshold": 1.0, "strength": 1.0, "radius": 0.5}),
        )

    artist = _Tree("Artist")
    artist_scene = _GroupScene(artist)
    with pytest.raises(compositor.CompositorCompatibilityError, match="unmarked"):
        compositor.build_managed_post_stack(artist_scene, _post())
    assert len(artist.nodes) == 0


def test_legacy_final_contract_requires_one_linked_composite(compositor):
    _install_fake_bpy(compositor)
    scene, tree = _managed_legacy(compositor)
    compositor.build_managed_post_stack(scene, _post())
    assert compositor.validate_final_image_linked(scene)["api"] == compositor.API_LEGACY

    duplicate = tree.nodes.new("CompositorNodeComposite")
    duplicate.name = "Other Composite"
    duplicate.is_active_output = True
    with pytest.raises(compositor.CompositorCompatibilityError, match="exactly one"):
        compositor.authoritative_final_output(scene)


def test_legacy_effect_stack_uses_duplicate_image_socket_indices(compositor):
    _install_fake_bpy(compositor)
    scene, tree = _managed_legacy(compositor)
    compositor.build_managed_post_stack(
        scene,
        _post(
            bloom={"enabled": True, "threshold": 1.0, "strength": 0.5, "radius": 0.5},
            grain={"enabled": True, "strength": 0.1, "scale": 2.0, "seed": 7},
            vignette={"enabled": True, "strength": 0.2, "feather": 0.4},
        ),
    )
    grain = tree.nodes.get("AI_LOOK_GRAIN_BLEND")
    destinations = [
        grain.inputs.index(link.to_socket)
        for link in tree.links
        if link.to_node is grain
    ]
    assert destinations == [1, 2]
    invert = tree.nodes.get("AI_LOOK_VIGNETTE_INVERT")
    assert [link.to_socket.name for link in tree.links if link.to_node is invert] == ["Color"]


def _format():
    return SimpleNamespace(
        file_format="OPEN_EXR",
        color_mode="RGBA",
        color_depth="16",
        compression=15,
    )


def test_redirect_and_restore_blender_5_file_outputs(compositor):
    tree = _Tree(api="5x")
    node = tree.nodes.new("CompositorNodeOutputFile")
    node.name = "Beauty Output"
    node.directory = "/artist/renders"
    node.file_name = "../../beauty_####"
    node.file_output_items = [
        SimpleNamespace(
            name="Beauty",
            path="../../../outside/beauty_pass_####",
            socket_type="RGBA",
            vector_socket_dimensions=3,
            override_node_format=True,
            save_as_render=True,
            format=_format(),
        )
    ]
    snapshots = compositor.redirect_file_outputs(tree, "/tmp/batch-42")
    assert os.path.commonpath(("/tmp/batch-42", node.directory)) == "/tmp/batch-42"
    assert node.file_name == "beauty_####"
    assert node.file_output_items[0].path == "000_beauty_pass_####"
    node.file_name = "mutated"
    node.file_output_items[0].path = "mutated"
    node.file_output_items[0].save_as_render = False

    compositor.restore_file_outputs(tree, snapshots)
    assert node.directory == "/artist/renders"
    assert node.file_name == "../../beauty_####"
    assert node.file_output_items[0].path == "../../../outside/beauty_pass_####"
    assert node.file_output_items[0].save_as_render is True


def test_redirect_and_restore_recurses_through_nested_shared_groups(compositor):
    nested = _Tree("Nested", api="5x")
    output = nested.nodes.new("CompositorNodeOutputFile")
    output.name = "Nested Output"
    output.directory = "//artist/nested"
    output.file_name = "../../nested_####"
    output.file_output_items = [
        SimpleNamespace(
            name="Nested Beauty",
            path="../../nested/beauty_####",
            override_node_format=False,
            save_as_render=True,
            format=_format(),
        )
    ]

    root = _Tree("Root", api="5x")
    first = root.nodes.new("CompositorNodeGroup")
    first.name = "First Nested"
    first.node_tree = nested
    second = root.nodes.new("CompositorNodeGroup")
    second.name = "Second Nested"
    second.node_tree = nested

    snapshots = compositor.redirect_file_outputs(root, "/tmp/batch-nested")
    assert len(snapshots) == 1
    assert snapshots[0]["tree_path"] == ["First Nested"]
    assert os.path.commonpath(("/tmp/batch-nested", output.directory)) == (
        "/tmp/batch-nested"
    )
    assert "/" not in output.file_name
    assert "\\" not in output.file_name
    assert ".." not in output.file_name
    assert "/" not in output.file_output_items[0].path
    assert "\\" not in output.file_output_items[0].path
    assert ".." not in output.file_output_items[0].path

    compositor.restore_file_outputs(root, snapshots)
    assert output.directory == "//artist/nested"
    assert output.file_name == "../../nested_####"
    assert output.file_output_items[0].path == "../../nested/beauty_####"


def test_redirect_and_restore_blender_42_file_outputs(compositor):
    tree = _Tree(api="legacy")
    node = tree.nodes.new("CompositorNodeOutputFile")
    node.name = "Passes"
    node.base_path = "/artist/passes"
    node.format = _format()
    node.file_slots = [
        SimpleNamespace(
            name="Diffuse",
            path="../../outside/diffuse_####",
            use_node_format=True,
            save_as_render=False,
            format=_format(),
        )
    ]
    snapshots = compositor.redirect_file_outputs(tree, "/tmp/batch-42")
    assert os.path.commonpath(("/tmp/batch-42", node.base_path)) == "/tmp/batch-42"
    assert node.file_slots[0].path == "000_diffuse_####"
    node.file_slots[0].path = "mutated"

    compositor.restore_file_outputs(tree, snapshots)
    assert node.base_path == "/artist/passes"
    assert node.file_slots[0].path == "../../outside/diffuse_####"


class _FailingPathNode(_Node):
    def __init__(self):
        super().__init__("CompositorNodeOutputFile")
        self._directory = "/artist/failing"
        self.file_name = "failing_####"
        self.file_output_items = []
        self.fail_next_directory = True

    @property
    def directory(self):
        return self._directory

    @directory.setter
    def directory(self, value):
        if self.fail_next_directory:
            self.fail_next_directory = False
            raise RuntimeError("injected redirect failure")
        self._directory = value


def test_redirect_failure_restores_already_mutated_nested_outputs(compositor):
    root = _Tree("Root", api="5x")
    first = root.nodes.new("CompositorNodeOutputFile")
    first.name = "A First"
    first.directory = "/artist/first"
    first.file_name = "../../first_####"
    first.file_output_items = []
    failing = _FailingPathNode()
    failing.name = "B Failing"
    root.nodes.append(failing)

    with pytest.raises(compositor.CompositorCompatibilityError, match="redirect failure"):
        compositor.redirect_file_outputs(root, "/tmp/batch-failure")

    assert first.directory == "/artist/first"
    assert first.file_name == "../../first_####"
    assert failing.directory == "/artist/failing"


def test_redirect_requires_declared_absolute_directory(compositor):
    tree = _Tree(api="5x")
    with pytest.raises(ValueError, match="absolute"):
        compositor.redirect_file_outputs(tree, "relative/output")
