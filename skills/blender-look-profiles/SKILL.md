---
name: blender-look-profiles
description: Build, switch, inspect, and batch-render deterministic full-look profiles in Blender through blend-ai. Use for persistent multi-look Scene catalogs that combine managed lighting, World, fog/rain, camera/render/color settings, and compositing, especially dataset generation across many scenes.
---

# Blender look profiles

Use the `blend_ai_relighting` MCP server's look-profile tools to compile complete
looks as shallow linked Scenes. Use Collections as containers for profile-owned
lights, fog, rain, and helpers; do not treat a Collection toggle as the complete
profile switch. The compiled Scene is the runtime boundary for World, render,
color, camera, and compositor state.

Read [profile-recipes.md](references/profile-recipes.md) before generating a
suite of profiles. Read [composited-acceptance.md](references/composited-acceptance.md)
before previewing, accepting, or batch-rendering profiles.

## Cycles-only rendering invariant

- Use Cycles for every viewport preview, composited preview, beauty render,
  diagnostic contact sheet, hosted-critic input, before/after comparison, and
  acceptance render. Eevee must never be used for rendering or evaluation.
- Every generated profile must explicitly set `render.engine="CYCLES"`.
  `KEEP`, `BLENDER_EEVEE`, and `BLENDER_EEVEE_NEXT` are invalid even for
  drafts or low-cost previews. If the source Scene uses Eevee, the compiled
  profile must override it.
- Use lower Cycles samples, denoising, resolution percentage, or bounded camera
  subsets to accelerate iteration. Never change engines as a speed shortcut.
- Reject any retrieved result whose provenance does not report
  `engine=CYCLES`. Do not send it to a critic, compare it, accept it, or
  describe it as evaluation evidence.

## Safety and preflight

1. Confirm Blender and the blend-ai N-panel server are running.
2. Read `bpy.data.filepath`, Blender version, current Scene, camera, render
   engine, and compositor capability before mutation. A narrow read-only
   `execute_blender_code` call is permitted for this preflight.
3. Require a saved duplicate such as `*_AI_RELIGHT_TEST.blend`, or explicit user
   confirmation that the open file/Scene is disposable. Record that decision.
4. Never save or overwrite the `.blend` without explicit approval. Compiling a
   profile and accepting a render do not imply permission to save.
5. Use explicit `source_scene` and target Scene names in every call. Never infer
   profile ownership from whichever Scene happens to be active.
6. If the new look-profile tools are absent, ask for an add-on/MCP refresh. Do
   not recreate this production workflow with arbitrary `execute_blender_code`.
7. Blender 5.2 has the live profile/compositor gate. On Blender 4.2, run a
   bounded compositor smoke before production and report that equivalent live
   acceptance is still pending; focused adapter tests are not live proof.

## Architecture contract

- Treat the artist source Scene as immutable `BASE` content.
- Persist deterministic intent in the marked `AI_LOOK_PROFILES` manifest.
- Compile each complete look to a managed `LINK_COPY` Scene that shares base
  geometry but owns a unique payload Collection, World, compositor, and settings.
- Use stable `profile_id` and component IDs across iterations.
- Store the root seed and every resolved parameter. Never draw new random values
  during activation or rendering.
- A profile starts as `DRAFT`. It cannot become `ACCEPTED` until a fresh,
  full-quality Composite output has been retrieved and visually reviewed.
- Never mutate a shared material for rainy wetness. Use profile-owned overlay
  geometry or explicitly duplicated objects/materials; otherwise leave wetness
  out and report it.

## Required generation workflow

1. Call `get_look_profile_context(source_scene=..., detail="FULL")`. Retain the
   base/profile/geometry revisions, manifest state, ownership warnings,
   compositor adapter, and active preview/render locks.
2. Call `get_lighting_context` once for the canonical camera with
   `scope="CAMERA"` and `detail="CANDIDATES"`. Retain its geometry revision and
   treat the returned JSON as a plain snapshot. Do not recreate this scan with
   `execute_blender_code` or retain `DepsgraphObjectInstance`, evaluated Object,
   or RNA Matrix references while advancing `depsgraph.object_instances`.
   Blender 5.2 can invalidate those iterator-scoped wrappers and abort the host;
   the hardened handler snapshots stable IDs/plain matrices and reacquires
   evaluated objects for deferred candidate work.
3. Select a coherent suite of archetypes from the recipe reference. For a dozen
   looks, vary motivated conditions, not every numeric knob independently.
4. Derive independent deterministic component seeds from SHA-256 of
   `(asset_id, template_id, variant_index, generator_version, namespace)`.
   Resolve all sampled values before any mutation.
5. Use `batch_raycast` for every Sun/moon/practical direction that depends on
   openings, clear paths, or subject visibility. Treat semantic candidates as
   hints until raycasts verify them.
6. Build a strict profile specification containing lighting, World,
   atmosphere/weather, post, camera, render, and color-management intent. Keep
   IDs stable and keep the camera canonical unless the brief explicitly changes
   optics or DOF. Set `render.engine="CYCLES"` explicitly on every profile,
   including previews.
7. Call `upsert_look_profile(action="VALIDATE", update_mode="CREATE_VERSION")`
   with the retained expected revisions. Resolve every conflict; validation must
   report zero mutation. When `existing_light_policy="MUTE_NON_MANAGED"`, this
   step must pass the fail-closed profile-local isolation preflight. Audit its
   light-only Collection exclusions, managed non-light clones for mixed
   Collections, and copied-root light unlinks. Never replace that plan with
   global object or Collection visibility changes.
8. Call the same tool with `action="COMPILE"`. Confirm the response says the
   source Scene was preserved and audit the managed inventory.
9. Activate the compiled target with `activate_look_profile`. Never activate a
   new Scene while a retained preview session or render job owns stale Scene
   state.
10. Submit a bounded Cycles composited preview through the asynchronous profile
    render path, poll it, and retrieve the completed result. Assert
    `engine=CYCLES` in its provenance. A viewport screenshot or raw Combined
    pass is diagnostic only and cannot be used for look approval.
    For cinematic, dataset-ready, or final-quality review, first call
    `inspect_look_review_state` for every intended camera, then submit with
    `include_review_packet=true`. A technical `FAIL` blocks rendering/provider
    review until the explicit managed-state fault is resolved.
11. Critique lighting, atmosphere, and post together. Patch only the affected
    components by compiling a new immutable draft revision; never silently
    overwrite a dirty/frozen/accepted revision.
12. Submit a render using the exact compiled Cycles engine, dimensions, samples,
    and denoise preset through `submit_look_render_batch`, poll with
    `get_look_render_batch`, and retrieve it with `get_look_render_result`.
    Multi-camera batches use explicit `camera_names` and must include `{camera}`
    in the filename template. Each camera expands to an independent result.
13. Inspect the returned native image and provenance using the composited
    acceptance reference. When a review packet was requested, verify the first
    native image is the final Composite and the second is the labeled contact
    sheet. Before revealing provenance or diagnostics, call
    `review_look_render(mode="REALISM", realism_prompt_version="causal-v1")`
    so exactly one unlabeled 1600px JPEG beauty is assessed, followed by a
    zero-image text-only rewrite that filters the feedback to lighting, shadows,
    lighting color/aesthetics, atmosphere, camera/optical qualities, and
    post-processing. Preserve both audit receipts; choose the vision pass's
    literal, evidence-balanced, causal, or actionable-finishing strategy as
    appropriate. When a fixed photographic packet exists, call
    `review_look_render(mode="REFERENCE", reference_packet=...)`. Keep its
    complete Claude `holistic_result` across lighting, value/color, atmosphere,
    camera/finish, material/surface, geometry/asset, and composition/staging.
    Also retain the zero-image Gemini rewrite, which separates actionable look
    directions from non-executable deferred context. Then call
    `review_look_render(mode="CRITIQUE")` as a separate beauty-plus-AOV causal
    lane. Reconcile the blind result, full holistic comparison, filtered
    reference handoff, and technical critique. Show the exact proposed patch and
    wait for explicit approval before compiling a new version. After
    an approved revision, compare the new result directly with the prior result
    using `mode="COMPARE"`. Only after review passes, call `accept_look_profile` with
    the exact batch ID, result ID, artifact SHA-256, retained revisions, and
    `review_acknowledged=true`.
14. Restore the original active Scene and any retained preview session. Report
    profile IDs, compiled Scene names, revisions, seeds, render result IDs,
    artifact paths/checksums, warnings, and whether the `.blend` remains unsaved.

## Batch dataset workflow

- Materialize and validate looks before production; batch workers should target
  compiled profile Scenes instead of mutating one broad Scene before each frame.
- Prefer `PAIRWISE` when each `profile_id` is paired with its compiled Scene.
  `CROSS_PRODUCT` supports one profile across multiple Scenes that all carry
  that exact profile ownership; it is not a many-profile matrix shortcut.
- Keep all output paths inside one declared absolute `output_root`. Use stable
  filenames containing Scene, profile, and frame.
- Use asynchronous submission/status/cancellation. Cooperative cancellation may
  allow the active frame to finish; preserve and report every completed output.
- The built-in batch registry is process-local, in memory, and serialized. A
  Blender restart/reload loses its job records and one render runs at a time;
  use an external durable scheduler when work must survive the Blender process.
- For hundreds of first-pass looks, use one low-sample denoised Cycles technical
  preview and one Cycles visual scoring render. Spend the deeper eight-capture
  relighting budget only on failures and high-value looks.
- A returned proxy may be downsampled only after the full Composite artifact is
  complete. Record source and returned dimensions and both checksums.

## Failure handling

- Refresh context after any stale base, profile, or geometry revision.
- Refuse linked/read-only mutation, unmanaged name collisions, ambiguous final
  compositor outputs, missing cameras, unsafe output roots, and shared-data
  ownership leaks.
- `VALIDATE` must reject `MUTE_NON_MANAGED` before mutation unless every artist
  light membership can be isolated inside the compiled Scene by a light-only
  Collection exclusion, a managed non-light clone of a mixed Collection, or a
  copied-root unlink. Do not toggle global object/Collection visibility on
  shared base data.
- If fog, rain, post, or camera compilation fails, the transaction must remove
  all temporary datablocks and leave the base fingerprint unchanged.
- If a composited render fails or is stale, keep the profile `DRAFT` and report
  the exact missing acceptance evidence.
- If a profile or result uses Eevee or reports any engine other than `CYCLES`,
  keep it `DRAFT`, discard it as evaluation evidence, and rerender with an
  explicitly compiled Cycles preset.
- Never call OpenRouter after a deterministic review-state `FAIL`, silently
  retry a failed provider request, apply critic actions without approval, or
  auto-accept a profile. Stop after two approved review revisions.
- Treat `SKIPPED` outputs as unproven pre-existing files. They have no image
  proxy or acceptance evidence and must never be described as fresh renders.
- Preview File Output nodes must be redirected and restored exactly. Final File
  Output nodes may write only under the declared output root.
