"""Blender N-panel UI for blend-ai server control."""

import json

import bpy

from . import server as addon_server


class BLENDAI_PT_MainPanel(bpy.types.Panel):
    """blend-ai MCP Server Control Panel"""
    bl_label = "blend-ai"
    bl_idname = "BLENDAI_PT_main_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "blend-ai"

    def draw(self, context):
        layout = self.layout
        srv = addon_server.get_server()

        if srv.is_running:
            port = srv._port
            layout.label(text=f"Server: Running (port {port})", icon="CHECKMARK")
            layout.operator("blendai.stop_server", text="Stop Server", icon="CANCEL")
        else:
            layout.label(text="Server: Stopped", icon="X")
            layout.prop(context.scene, "blendai_port", text="Port")
            layout.operator("blendai.start_server", text="Start Server", icon="PLAY")


def _look_profile_entries(context):
    """Read visible profile entries without creating or mutating the manifest."""
    text = bpy.data.texts.get("AI_LOOK_PROFILES")
    if text is None or not bool(text.get("blend_ai_look_profile_manifest", False)):
        return []
    try:
        payload = json.loads(text.as_string())
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    entries = payload.get("profiles") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return []
    current = context.scene
    source_name = str(
        current.get("blend_ai_look_source_scene", "") or getattr(current, "name", "")
    )
    result = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("source_scene") != source_name:
            continue
        if entry.get("status") not in {"DRAFT", "ACCEPTED"}:
            continue
        version = entry.get("version")
        revision = entry.get("revision")
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version < 1
            or isinstance(revision, bool)
            or not isinstance(revision, int)
            or revision < 1
            or not isinstance(entry.get("scene_name"), str)
            or not entry["scene_name"]
            or not isinstance(entry.get("profile_id"), str)
            or not entry["profile_id"]
        ):
            continue
        result.append(entry)
    return sorted(
        result,
        key=lambda entry: (
            str(entry.get("profile_id", "")),
            int(entry.get("version", 0)),
            int(entry.get("revision", 0)),
        ),
    )


class BLENDAI_PT_LookProfiles(bpy.types.Panel):
    """Native switcher for compiled managed look-profile Scenes."""

    bl_label = "Look Profiles"
    bl_idname = "BLENDAI_PT_look_profiles"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "blend-ai"
    bl_parent_id = "BLENDAI_PT_main_panel"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        entries = _look_profile_entries(context)
        if not entries:
            layout.label(text="No compiled profiles for this source Scene", icon="INFO")
            return

        active_name = str(getattr(context.scene, "name", ""))
        layout.label(text=f"Active: {active_name}", icon="SCENE_DATA")
        for entry in entries[:32]:
            scene_name = str(entry.get("scene_name", ""))
            display_name = str(entry.get("display_name") or entry.get("profile_id") or "Look")
            version = int(entry.get("version", 0))
            revision = int(entry.get("revision", 0))
            status = str(entry.get("status", "DRAFT"))
            row = layout.row(align=True)
            row.label(text=f"{display_name} v{version:03d} r{revision:03d} [{status}]")
            operator = row.operator(
                "blendai.activate_look_profile",
                text="Active" if scene_name == active_name else "Activate",
                icon="CHECKMARK" if scene_name == active_name else "PLAY",
            )
            operator.target_scene = scene_name
        if len(entries) > 32:
            layout.label(text=f"{len(entries) - 32} more profiles available in Scene menu")


class BLENDAI_OT_StartServer(bpy.types.Operator):
    """Start the blend-ai MCP server"""
    bl_idname = "blendai.start_server"
    bl_label = "Start blend-ai Server"

    def execute(self, context):
        port = context.scene.blendai_port
        addon_server.start_server(port=port)
        self.report({"INFO"}, f"blend-ai server started on 127.0.0.1:{port}")
        return {"FINISHED"}


class BLENDAI_OT_StopServer(bpy.types.Operator):
    """Stop the blend-ai MCP server"""
    bl_idname = "blendai.stop_server"
    bl_label = "Stop blend-ai Server"

    def execute(self, context):
        addon_server.stop_server()
        self.report({"INFO"}, "blend-ai server stopped")
        return {"FINISHED"}


class BLENDAI_OT_ActivateLookProfile(bpy.types.Operator):
    """Activate a compiled profile Scene without saving the blend."""

    bl_idname = "blendai.activate_look_profile"
    bl_label = "Activate Look Profile"
    bl_options = {"REGISTER"}

    target_scene: bpy.props.StringProperty(name="Target Scene")

    def execute(self, context):
        scene = bpy.data.scenes.get(self.target_scene)
        if scene is None:
            self.report({"ERROR"}, f"Look Scene '{self.target_scene}' was not found")
            return {"CANCELLED"}
        if not bool(scene.get("blend_ai_look_managed", False)) or scene.get(
            "blend_ai_look_role"
        ) != "SCENE":
            self.report({"ERROR"}, "Target is not a managed look-profile Scene")
            return {"CANCELLED"}
        profile_id = scene.get("blend_ai_look_profile_id")
        if not isinstance(profile_id, str) or not profile_id:
            self.report({"ERROR"}, "Target has no managed look-profile id")
            return {"CANCELLED"}
        try:
            from .handlers import look_profiles

            look_profiles.handle_activate_look_profile(
                {
                    "profile_id": profile_id,
                    "target_scene": scene.name,
                    "strict": True,
                }
            )
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Activated look Scene '{scene.name}'")
        return {"FINISHED"}


classes = (
    BLENDAI_PT_MainPanel,
    BLENDAI_PT_LookProfiles,
    BLENDAI_OT_StartServer,
    BLENDAI_OT_StopServer,
    BLENDAI_OT_ActivateLookProfile,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.blendai_port = bpy.props.IntProperty(
        name="Port",
        description="TCP port for the blend-ai server",
        default=9876,
        min=1024,
        max=65535,
    )


def unregister():
    if hasattr(bpy.types.Scene, "blendai_port"):
        del bpy.types.Scene.blendai_port

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
