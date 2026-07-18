"""Blender command handlers - registers all handlers with the dispatcher."""

from . import (
    scene,
    objects,
    transforms,
    modeling,
    materials,
    lighting,
    camera,
    animation,
    rendering,
    curves,
    sculpting,
    uv,
    physics,
    geometry_nodes,
    armature,
    collections,
    file_ops,
    viewport,
    code_exec,
    booltool,
    mesh_editing,
    mesh_quality,
    gpencil,
    relighting,
    cycles_viewport,
)

_modules = [
    scene,
    objects,
    transforms,
    modeling,
    mesh_editing,
    mesh_quality,
    materials,
    lighting,
    camera,
    animation,
    rendering,
    curves,
    sculpting,
    uv,
    physics,
    geometry_nodes,
    armature,
    collections,
    file_ops,
    viewport,
    code_exec,
    booltool,
    gpencil,
    relighting,
    cycles_viewport,
]


def register():
    for mod in _modules:
        if hasattr(mod, "register"):
            mod.register()


def unregister():
    for mod in reversed(_modules):
        if hasattr(mod, "unregister"):
            mod.unregister()
