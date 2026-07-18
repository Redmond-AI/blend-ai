# Spatial, vision-guided relighting

This fork adds a bounded spatial-lighting layer to `blend-ai`. It is based on
upstream commit `621ddc0dc8b379428e027c17ce817e1fc4d9cb36`. At that commit the
latest GitHub release is labelled `v1.2.2`, while `pyproject.toml`, the Blender
manifest, and the add-on metadata identify the code as `1.2.1`.

The original screenshot tool is intentionally unchanged. Relighting uses four
new tools:

- `get_lighting_context` returns evaluated, paginated world bounds, camera
  projection, collection hierarchy, existing lights, and heuristic surface or
  opening candidates.
- `batch_raycast` verifies line of sight through evaluated scene geometry.
- `apply_light_plan` validates, applies, and rolls back lights owned by a
  marked `AI_RELIGHT` collection.
- `capture_cycles_viewport` returns native MCP image content from either a
  visible Cycles Rendered editor or an explicitly requested low-sample,
  denoised Cycles camera render.

## Important guarantees and limits

- "In camera" means the evaluated bounds intersect the camera frustum. It does
  not mean unoccluded.
- An opening candidate is semantic/geometry evidence and is always labelled
  `heuristic: true`. Use raycasts to establish a clear path.
- Light-plan atomicity covers only the supported state recorded in the
  transaction snapshot. It is not a Blender-wide transaction.
- Rollback is global last-in, first-out (LIFO), because each transaction
  snapshots supported scene-wide relighting state. A rollback selected by
  `plan_id` or `transaction_id` succeeds only when that transaction is the
  newest ledger entry; roll back newer transactions first. This prevents an
  older whole-scene snapshot from overwriting newer managed work.
- The viewport result reports a bounded settling interval. Blender does not
  expose a reliable current Cycles viewport sample count, so the tool never
  claims convergence.
- Both capture modes prepare a visible, unminimized Blender preview window.
  `capture_mode="VIEWPORT"` does not use F12, `render.render`, or the existing
  OpenGL screenshot path and does not create or replace `Render Result`.
- `capture_mode="QUICK_RENDER"` is never selected automatically. It renders a
  bounded temporary PNG with the requested sample cap and denoising, removes
  that file after reading it, and restores the render settings it changed. It
  does replace Blender's `Render Result`; use it only when that side effect is
  acceptable.
- Neither capture mode saves the `.blend` file. A client or test harness may
  explicitly persist the returned native image as an audit artifact, but the
  MCP never silently leaves its temporary source PNG on disk.

## Recommended fast loop

1. Query and cache lighting context.
2. Raycast proposed sources to their targets.
3. Validate the whole light plan.
4. Apply the managed plan.
5. Capture a 16-sample, two-second Cycles viewport preview, or explicitly use a
   quick denoised render when the user has authorized `Render Result`
   replacement.
6. Critique the image and patch only changed lights.
7. Restore the preview session and offer rollback before saving.

## Mac development installation

The checkout is expected at:

`/Users/robingraham/Library/CloudStorage/Dropbox/github/blend-ai-relighting`

The developer add-on is a symlink at:

`/Users/robingraham/Library/Application Support/Blender/5.2/extensions/user_default/blend_ai`

Run the MCP server through its locked environment:

```bash
/Users/robingraham/.local/bin/uv run \
  --directory "/Users/robingraham/Library/CloudStorage/Dropbox/github/blend-ai-relighting" \
  blend-ai
```

Keep Blender's add-on port at `9876`. Stop the server and fully restart Blender
after changing add-on source files.

For live capture, make a workspace named `AI Preview` containing exactly one
large 3D View, show it in a second Blender window, and keep that window
unminimized. Configure Metal under Blender's Cycles device preferences; the
tool may select GPU mode for the scene but never changes global device
preferences.

### macOS capture backends

Blender 5.2 on this M4 Max/Metal configuration writes only gray backing pixels
through both `bpy.ops.screen.screenshot_area` and `bpy.ops.screen.screenshot`,
even while the visible editor shows the correct Cycles image. The MCP process
therefore uses a small, locally compiled ScreenCaptureKit helper for
`capture_mode="VIEWPORT"` on macOS. It resolves the exact Blender process and
window, captures the composited window, validates the window/client geometry,
crops the requested 3D View region, and then packages the result as MCP
`ImageContent`. Ambiguous windows fail rather than selecting one arbitrarily.

ScreenCaptureKit requires macOS Screen Recording permission for the capture
process. If preflight fails, enable the process shown by macOS under System
Settings -> Privacy & Security -> Screen & System Audio Recording, then restart
the affected application. The current machine already had working permission,
so a first-denial/first-grant flow has not been reproduced here.

The helper is compiled once into the MCP process's temporary directory and is
removed at process exit. This requires Apple's Swift compiler/Xcode command-line
tools. No screen image is sent off the machine.

### Explicit quick-render example

```text
capture_cycles_viewport(
  capture_mode="QUICK_RENDER",
  preview_samples=8,
  denoise=true,
  max_size=1024,
  format="PNG",
  keep_session=true
)
```

The temporary source PNG is deleted after the MCP process has validated and
encoded it. The returned native MCP image can itself be persisted by the client
or acceptance harness when an audit artifact is wanted.

## Tests

```bash
uv run --extra dev pytest -q tests/test_validators.py tests/test_tools/test_relighting.py
uv run --extra dev pytest -q tests/test_addon
uv run --extra dev pytest -q tests --ignore=tests/test_addon
uv run --extra dev ruff check src addon tests
```

`tests/fixtures/build_mini_warehouse.py` creates a deterministic acceptance
scene. GUI capture must be tested in a visible Blender session. Local Blender
5.2 background mode currently crashes during Metal initialization on this Mac,
so headless local capture is not a release gate.

## Verified M4 Max live acceptance

The deterministic warehouse passed the complete MCP protocol workflow in
separate visible Blender 5.2.0 factory processes on 2026-07-17. No `.blend`
file was saved. Both final runs validated tool discovery, evaluated
collection-instance and Geometry Nodes ray identities, ignored-glass and
opaque-blocker behavior, validate/apply/patch, native MCP images, global LIFO
rollback, preview restoration, and exact post-rollback light state. All five
image checks passed: the relit frame was darker and materially different, red
separation and cool skylight contribution increased, and the patch changed the
image again.

| Mode | Settings | Baseline / relit / patched totals | Geometry revision | Result |
| --- | --- | --- | --- | --- |
| Visible viewport | 16 samples, 2 s settle, denoise, JPEG | 5.963 s / 3.891 s / 3.920 s | 104 throughout apply/patch | Pass; warm overhead target missed |
| Quick render | 8 samples, denoise, 960x540 PNG | 2.805 s / 2.238 s / 2.671 s | 104 throughout apply/patch | Pass |

The viewport run used `MACOS_SCREENCAPTUREKIT`; the quick run used
`QUICK_CYCLES_RENDER`. Every image was native MCP `ImageContent`, the viewport
JPEGs were 24-32 KiB, and the quick PNGs were 302-347 KiB.

The final warm viewport calls took 3.891 s and 3.920 s including the requested
two-second settle, or approximately 1.9 s of warm overhead. Native viewport
capture therefore passes functionally but does not meet the aspirational
under-one-second overhead target.

### Verified M4 Max scale benchmark

The generated 20,000-instance, multi-million-triangle fixture also passed its
functional benchmark on 2026-07-17. Cold context timing includes a 721.50 ms
revision-bound raycast acceleration build. The default context page contained
500 instance records and stayed below the 512 KiB payload target.

| Operation | Measured | Target | Result |
| --- | ---: | ---: | --- |
| Cold context page | 2,354.15 ms | < 3,000 ms | Pass |
| Warm cached context page | 65.09 ms | < 200 ms | Pass |
| Cold context payload | 499,230 bytes | < 512 KiB | Pass |
| 256 rays, 256 hits | 28.49 ms | < 1,000 ms | Pass |
| Validate 64-light plan | 60.83 ms | Informational | Recorded |
| Apply 64-light plan | 395.64 ms | < 250 ms | **Miss** |
| Roll back 64-light plan | 951.64 ms | Informational | Recorded |

The 64-light apply is correct and comfortably below the socket timeout, but it
does not meet the initial 250 ms target. Correct view-layer flushing is retained
instead of weakening transaction or scene-consistency guarantees to improve
that number.

The private real-warehouse acceptance remains an open release gate and cannot
run until a duplicate scene path is supplied.

Evidence is written under ignored local directories:

- `artifacts/relighting/live-warehouse-quick-final/acceptance-report.json`
- `artifacts/relighting/live-warehouse-viewport-final/acceptance-report.json`
- `artifacts/relighting/live-scale-20260717-final4/scale-benchmark-report.json`
