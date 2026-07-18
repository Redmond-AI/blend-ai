---
name: blender-relighting
description: Relight an existing Blender scene through blend-ai using evaluated spatial context, batched raycasts, managed light plans, and native Cycles viewport image critique. Use for non-destructive day-to-night or alternate-lighting work in complex scenes.
---

# Blender relighting

Use the `blend_ai_relighting` MCP server's four relighting tools. Do not replace
the spatial tools with ad hoc `execute_blender_code` unless diagnosing a tool
bug.

## Preconditions

1. Confirm Blender is open, the blend-ai N-panel server is running on port
   `9876`, and a visible unminimized workspace named `AI Preview` contains one
   large 3D View.
2. Read `bpy.data.filepath` through the MCP before any mutation. A narrowly
   scoped, read-only `execute_blender_code` call is allowed for this safety
   preflight. Refuse if the file is unsaved. Require a basename containing
   `_AI_RELIGHT_TEST`, or an explicit user confirmation that this is a duplicate,
   and record the verified absolute path in the handoff.
3. Do not save or overwrite the `.blend` file without explicit user approval.
4. Do not edit geometry, materials, cameras, animation, compositor, or arbitrary
   world nodes. The managed light plan may assign a separate managed World and
   temporarily mute existing lights when the user requested it.

## Required workflow

1. Call `get_lighting_context` once with the active camera, `scope="CAMERA"`,
   `detail="CANDIDATES"`, and the default warehouse semantic terms. Retain the
   returned `geometry_revision` and page through only when needed.
2. Treat every opening candidate as heuristic. Use `batch_raycast` to verify
   clear paths from proposed moon/practical sources to intended targets. Pass
   explicit glass material/object patterns when glass should not block the
   lighting path.
3. Form one coherent initial plan. Prefer a few motivated sources over many
   weak lights. Use stable IDs that remain constant across iterations.
4. Call `apply_light_plan(action="VALIDATE")`. Resolve every warning or conflict
   before applying.
5. Call `apply_light_plan(action="APPLY")` with the same plan and geometry
   revision. Use `REPLACE_MANAGED` for the initial plan and `PATCH_MANAGED` for
   later deltas.
6. Call `capture_cycles_viewport` with `capture_mode="VIEWPORT"`, 16 preview
   samples, two seconds settling, denoising, 1024 maximum size, JPEG 85, and
   `keep_session=true`. If the user has explicitly authorized temporary quick
   renders, `capture_mode="QUICK_RENDER"` is an allowed reliability alternative:
   start with 8 samples, denoising, PNG, and the same size cap; it deletes its
   temporary source PNG, does not save the blend file, and does replace
   Blender's `Render Result`. Never use it as a silent fallback.
7. Critique the returned image for:
   - Immediate night readability.
   - Cool moonlight versus red practical separation.
   - Directional and architectural motivation.
   - Unwanted uniform red wash.
   - Clipping, empty blacks, noisy pools, and implausible spill.
   - Aisle/subject silhouettes and perceived warehouse depth.
8. Patch only lights whose position, target, energy, color or size needs to
   change. Reuse stable IDs; do not rebuild the whole plan for one adjustment.
9. Stop when the image meets the brief or after eight captures. If the look is
   not converging, explain the spatial/scene limitation instead of continuing
   indefinitely.
10. Call `capture_cycles_viewport(action="RESTORE")` before ending, including
    after failures.
11. Report the last light-plan transaction ID and offer
    `apply_light_plan(action="ROLLBACK")`. Do not automatically roll back a look
    the user has accepted. Rollback is global LIFO because snapshots include
    supported scene-wide relighting state: if a requested transaction is not
    the newest ledger entry, roll back newer transactions first.

## Warehouse night defaults

- Start with a dark managed solid world rather than destroying the source
  world's nodes.
- Use a cool SUN or large AREA source only after raycasts support the skylight
  direction.
- Place sparse red AREA/SPOT sources near named fixtures, columns, walls, or
  ceiling structure and aim them at useful aisle/subject regions.
- Preserve readable silhouettes; night does not mean uniformly black.
- Keep red illumination localized and allow neutral/cool shadow regions.

## Failure handling

- A stale geometry revision requires a fresh lighting context before any
  apply/raycast call.
- A linked/read-only conflict must be resolved in Blender or omitted; do not
  bypass strict validation.
- If capture reports no visible workspace, ask the user to expose and
  unminimize `AI Preview`; both modes require successful preview preparation.
- Never switch from `VIEWPORT` to `QUICK_RENDER` silently. Use quick-render mode
  only after explicit user authorization and disclose its `Render Result`
  replacement in the handoff.
- If capture preparation succeeds and a later step fails, restore the preview
  session before doing anything else.
