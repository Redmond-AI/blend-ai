"""Real bpy 5.2 save/reopen acceptance for managed relighting transactions.

Run this as a standalone Python process with the PyPI ``bpy`` module.  It is
kept outside pytest discovery so the normal mock-bpy unit suite remains fast
and so opening a temporary blend file cannot perturb another test process.
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import bpy

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from addon.handlers import relighting  # noqa: E402


def _managed_lights() -> list[bpy.types.Object]:
    return [
        obj
        for obj in bpy.context.scene.objects
        if relighting._is_managed_light(obj)
    ]


def _point_spec(energy: float) -> dict[str, object]:
    return {
        "id": "integration_point",
        "name": "Integration Point",
        "type": "POINT",
        "location_world": [1.0, 2.0, 3.0],
        "energy": energy,
        "color_rgb": [0.8, 0.1, 0.05],
        "radius": 0.2,
    }


def _reset_scene() -> bpy.types.Scene:
    bpy.ops.wm.read_factory_settings(use_empty=False)
    scene = bpy.context.scene
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    scene.world = None
    relighting._reset_ledger_state()
    relighting._reset_raycast_index()
    return scene


def main() -> None:
    if tuple(bpy.app.version)[:2] != (5, 2):
        raise RuntimeError(f"Expected bpy 5.2, found {bpy.app.version_string}")
    if bpy.data.filepath:
        raise RuntimeError(f"Refusing to reset loaded file {bpy.data.filepath!r}")
    if bpy.app.binary_path and "--confirm-isolated-process" not in sys.argv:
        raise RuntimeError(
            "A Blender executable run requires --confirm-isolated-process"
        )
    scene = _reset_scene()
    scene.name = "Relighting bpy Integration"
    original_world = bpy.data.worlds.new("Artist Original World")
    original_world.color = (0.12, 0.08, 0.04)
    original_world.use_fake_user = False
    scene.world = original_world

    first = relighting.handle_apply_light_plan(
        {
            "action": "APPLY",
            "plan_id": "integration_night",
            "mode": "REPLACE_MANAGED",
            "lights": [_point_spec(250.0)],
            "scene_overrides": {
                "world": {
                    "mode": "MANAGED_SOLID",
                    "color_rgb": [0.0, 0.0, 0.02],
                    "strength": 0.1,
                },
                "exposure": -1.0,
            },
        }
    )
    assert original_world.use_fake_user is True
    light = _managed_lights()[0]
    light.data.exposure = 2.5
    light.data.normalize = False
    light.data.name = "Integration Point Data"
    cycles_expected: dict[str, object] = {}
    for attribute, value in {
        "max_bounces": 17,
        "use_multiple_importance_sampling": False,
        "is_portal": False,
        "is_caustics_light": True,
    }.items():
        if hasattr(light.data.cycles, attribute):
            setattr(light.data.cycles, attribute, value)
            cycles_expected[attribute] = value

    replacement = _point_spec(800.0)
    replacement.pop("radius")
    replacement.update(
        {
            "type": "AREA",
            "target_point": [1.0, 2.0, 0.0],
            "size": 2.0,
            "area_shape": "DISK",
        }
    )
    second = relighting.handle_apply_light_plan(
        {
            "action": "APPLY",
            "plan_id": "integration_night",
            "mode": "PATCH_MANAGED",
            "lights": [replacement],
        }
    )
    assert light.data.energy == 800.0
    assert light.data.type == "AREA"
    ledger = bpy.data.texts.get(relighting.LEDGER_TEXT)
    assert ledger is not None
    assert bool(ledger.get(relighting.LEDGER_PROP, False))

    with tempfile.TemporaryDirectory(prefix="blend-ai-bpy-") as directory:
        blend_path = Path(directory) / "relighting-save-reopen.blend"
        result = bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
        assert "FINISHED" in result and blend_path.is_file()
        result = bpy.ops.wm.open_mainfile(filepath=str(blend_path))
        assert "FINISHED" in result

        # The registered add-on load handler performs this reset in production.
        relighting._reset_ledger_state()
        relighting.handle_apply_light_plan(
            {"action": "ROLLBACK", "transaction_id": second["transaction_id"]}
        )
        restored = _managed_lights()[0]
        assert restored.data.type == "POINT"
        assert restored.data.energy == 250.0
        assert restored.data.exposure == 2.5
        assert restored.data.normalize is False
        assert restored.data.name == "Integration Point Data"
        for attribute, value in cycles_expected.items():
            assert getattr(restored.data.cycles, attribute) == value

        relighting.handle_apply_light_plan(
            {"action": "ROLLBACK", "transaction_id": first["transaction_id"]}
        )
        assert _managed_lights() == []
        assert bpy.context.scene.world.name == "Artist Original World"
        assert bpy.context.scene.world.use_fake_user is False
        assert bpy.data.collections.get(relighting.MANAGED_COLLECTION) is None

    print("bpy 5.2 relighting save/reopen integration passed")


if __name__ == "__main__":
    main()
