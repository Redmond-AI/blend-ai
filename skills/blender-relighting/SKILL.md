---
name: blender-relighting
description: Relight an existing Blender scene through blend-ai using evaluated spatial context, batched raycasts, managed light plans, and reference-grounded Cycles image critique. Use for non-destructive day-to-night or alternate-lighting work in complex scenes.
---

# Blender relighting

Use the `blend_ai_relighting` MCP server's four relighting tools. Do not replace
the spatial tools with ad hoc `execute_blender_code` unless diagnosing a tool
bug.

This skill is the reversible, light-only primitive. If the request needs a
persistent catalog of multiple looks, linked profile Scenes, World variants,
fog/rain, camera variants, render presets, compositor/post effects, composited
previews, or automated multi-profile renders, route the work to the
`blender-look-profiles` skill. Continue to use this skill's lighting-context and
raycast rules while the profile skill builds each managed lighting payload.

When the user asks for a cinematic, dataset-ready, composited, or final-quality
judgment, finish the fast viewport loop with the cinematic review workflow in
`blender-look-profiles`. The final judge must see the authoritative Composite
and its diagnostic contact sheet; a viewport capture remains an iteration aid.

All rendering and visual evaluation in this workflow is Cycles-only. Never use
Eevee for a viewport capture, quick render, look-profile preview, final beauty,
diagnostic pass, critic input, comparison, or acceptance render. A source Scene
may use Eevee, but every managed profile must explicitly override it with
`render.engine="CYCLES"`; `KEEP`, `BLENDER_EEVEE`, and
`BLENDER_EEVEE_NEXT` are invalid render choices for this skill.

## Preconditions

1. Confirm Blender is open, the blend-ai N-panel server is running on port
   `9876`, and a visible unminimized workspace named `AI Preview` contains one
   large 3D View. Use the startup procedure below when Blender is not already
   connected; opening a `.blend` normally does not start the N-panel server.
2. Read `bpy.data.filepath` through the MCP before any mutation. A narrowly
   scoped, read-only `execute_blender_code` call is allowed for this safety
   preflight. Refuse if the file is unsaved. Require a basename containing
   `_AI_RELIGHT_TEST`, or an explicit user confirmation that this is a duplicate,
   and record the verified absolute path in the handoff.
3. Do not save or overwrite the `.blend` file without explicit user approval.
4. Do not edit geometry, materials, cameras, animation, compositor, or arbitrary
   world nodes. The managed light plan may assign a separate managed World and
   temporarily mute existing lights when the user requested it.

## Start Blender and the MCP bridge together

Use this procedure before the other preconditions when the first read-only MCP
probe cannot reach Blender. It is the supported launch path for this local
Blender 5.2 developer installation.

1. Probe the existing MCP connection once with a read-only request for
   `bpy.data.filepath`. If it responds, keep that Blender process and do not
   launch another one. Also verify the returned path is the intended scene.
2. Resolve the target `.blend` to a literal absolute path and confirm it exists.
   Refuse an unsaved scene. Before launching, verify that the developer extension
   path
   `/Users/robingraham/Library/Application Support/Blender/5.2/extensions/user_default/blend_ai`
   is a symlink whose exact target is
   `/Users/robingraham/Library/CloudStorage/Dropbox/github/blend-ai-relighting/addon`.
   Never replace an unexpected file, directory, or differently targeted symlink.
3. Check whether `127.0.0.1:9876` already has a listener. If it does, retry the
   MCP probe and diagnose that process; do not kill it or start a competing
   server. If a Blender UI process is already open without the port listener,
   do not quit it because it may contain unsaved work. Start the server from its
   **blend-ai** N-panel, or obtain approval to close and relaunch it.
4. When no Blender process or port listener conflicts, create a run-scoped
   temporary directory and launch a new visible Blender process with the
   repository bootstrap below. Substitute literal absolute paths for
   `<SCENE.blend>` and `<READY.json>`:

   ```bash
   open -na Blender --args \
     "<SCENE.blend>" \
     --python "/Users/robingraham/Library/CloudStorage/Dropbox/github/blend-ai-relighting/tests/fixtures/open_blend_ai_review_scene.py" \
     -- \
     --expected-file "<SCENE.blend>" \
     --ready-file "<READY.json>" \
     --port 9876
   ```

   This bootstrap verifies the opened file, enables
   `bl_ext.user_default.blend_ai`, imports the developer checkout, starts the
   add-on TCP server on localhost, prepares the visible `AI Preview` workspace,
   and writes the readiness receipt. It never saves the scene.
5. Wait up to 30 seconds for the readiness JSON. Require `status="READY"`, the
   exact expected filepath, `server_port=9876`, a visible Blender PID, and
   `workspace="AI Preview"`. Then require a live listener on
   `127.0.0.1:9876` and repeat the MCP filepath probe. The receipt or socket
   alone is not sufficient; the first successful MCP request is the final gate.
6. If bootstrap fails, inspect Blender's launch output or readiness receipt and
   report the exact error. Do not fall back to a background Blender process:
   visible Cycles viewport capture requires a visible, unminimized window.
7. After any add-on source change, stop the add-on server and fully quit that
   Blender process before relaunching with this procedure. `Reload Scripts` is
   insufficient because the old TCP thread and imported modules can survive.

## Cinematic reference packet

For a brief that asks for a cinematic target, realism, or final-quality review,
use user-supplied references when available. Otherwise, after reading the
lighting context and before forming the initial light plan, research a bounded
packet of two to four real photographic references:

1. Search for live-action film frames, unit stills, or cinematographer
   photography that matches the scene type, time/weather, motivated light
   sources, and intended mood. Prefer official studio/film pages, ASC or
   cinematographer material, and verifiable frame databases. Reject generative
   images, concept art, game screenshots, and obviously CG-heavy frames unless
   the user explicitly wants them as style references.
2. Retrieve the actual image pixels and the source page. A search thumbnail,
   URL, caption, or text description alone is not visual evidence. Record a
   stable reference ID, source URL, title/film/creator and year when known,
   provenance confidence, why it was selected, and its exact comparison role.
   Preserve attribution and use the images for analysis, not redistribution.
3. Assign each image one or more narrow roles: `LIGHTING_TOPOLOGY`,
   `VALUE_AND_COLOR`, `ATMOSPHERE_AND_DEPTH`, `CAMERA_AND_FINISH`,
   `MATERIAL_AND_SURFACE`, `GEOMETRY_AND_ASSET`, or
   `COMPOSITION_AND_STAGING`. No reference needs to match the whole scene, and
   the complete packet should cover only roles the retrieved pixels can support.
4. Derive explicit `reference_targets` and `non_targets`. Targets may include
   motivated direction, key/fill hierarchy, shadow density and softness, color
   separation, falloff, highlight rolloff, reflections, depth atmosphere, DOF,
   grain, surface response, asset integration, or composition. Non-targets may
   include unrelated architecture, object identity, actor identity, or other
   qualities that should not be copied. The first comparison is intentionally
   holistic; later routing determines which observations the look workflow can
   act on.
5. Keep the packet and target definitions fixed across revisions. If the brief
   changes, rebuild them and disclose the change. If no trustworthy reference
   can be retrieved, continue from the brief but mark reference comparison
   `NOT_RUN`; never imply that text-only search results were inspected visually.

## Required workflow

1. Call `get_lighting_context` once with the active camera, `scope="CAMERA"`,
   `detail="CANDIDATES"`, and the default warehouse semantic terms. Retain the
   returned `geometry_revision` and page through only when needed.
2. Treat every opening candidate as heuristic. Use `batch_raycast` to verify
   clear paths from proposed moon/practical sources to intended targets. Pass
   explicit glass material/object patterns when glass should not block the
   lighting path.
3. Build the cinematic reference packet when the trigger above applies, then
   form one coherent initial plan against its derived targets. Prefer a few
   motivated sources over many weak lights. Use stable IDs that remain constant
   across iterations.
4. Call `apply_light_plan(action="VALIDATE")`. Resolve every warning or conflict
   before applying.
5. Call `apply_light_plan(action="APPLY")` with the same plan and geometry
   revision. Use `REPLACE_MANAGED` for the initial plan and `PATCH_MANAGED` for
   later deltas.
6. Call `capture_cycles_viewport` with `capture_mode="VIEWPORT"`, 16 preview
   samples, two seconds settling, denoising, 1024 maximum size, JPEG 85, and
   `keep_session=true`. When the capture will enter an evidence or render-history
   ledger, also pass its unique absolute destination as `staging_path`. The tool
   validates the encoded image and atomically creates that exact-byte file,
   returning its SHA-256; never copy MCP base64 through an ad hoc terminal
   decoder. Existing staging paths fail closed and are never overwritten. If the
   user has explicitly authorized temporary quick
   renders, `capture_mode="QUICK_RENDER"` is an allowed reliability alternative:
   start with 8 samples, denoising, PNG, and the same size cap; it deletes its
   temporary source PNG, does not save the blend file, and does replace
   Blender's `Render Result`. Never use it as a silent fallback. Both capture
   modes must remain on Cycles; do not substitute a generic Eevee viewport or
   render for speed.
7. Critique the returned image for:
   - Immediate night readability.
   - Cool moonlight versus red practical separation.
   - Directional and architectural motivation.
   - Unwanted uniform red wash.
   - Clipping, empty blacks, noisy pools, and implausible spill.
   - Aisle/subject silhouettes and perceived warehouse depth.
   - Directional agreement with the reference targets, without penalizing
     unrelated scene content.
   Use the fast capture for look development only. Do not run the final
   real-versus-CG test on a low-sample viewport or quick render whose missing
   finishing, noise, or denoising artifacts could dominate the answer.
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

## Final cinematic review handoff

1. Persist a short `review_intent` on the look profile, including explicit
   shadow, reflection, volume, and compositing expectations. Put the stable
   reference IDs and their derived targets/non-targets in its bounded summary;
   keep full URLs and attribution in the review handoff.
2. Call `inspect_look_review_state` for each intended camera. Do not spend a
   provider call while its status is `FAIL`. Recompile managed drift; report
   artist-owned object visibility warnings without changing those objects.
3. Compile the review profile with `render.engine="CYCLES"`, then render with
   `include_review_packet=true` through the look-profile batch renderer and
   retrieve the exact `SUCCEEDED` result. Require its provenance to report
   `engine=CYCLES`; otherwise discard it and rerender. Expect beauty first and
   one labeled six-tile contact sheet second. Its diagnostic order is Combined,
   Diffuse Direct, Glossy Direct plus Indirect, Emission, normalized Camera
   Depth (near white, distant black), and Object Cryptomatte.
4. Before revealing render provenance, references, or diagnostics, call
   `review_look_render(mode="REALISM", realism_prompt_version="causal-v1")`.
   It receives exactly one 1600px JPEG of the unlabeled Composite beauty. Do not
   send the filename, AOV sheet, engine, Scene metadata, prompt history, or the
   fact that it is a render. Every prompt asks exactly:
   **"Does this look CG or real? If CG, what would you change to make it look
   real?"** The vision pass may diagnose broad realism causes. A required second
   pass sends only that structured text, with zero images or scene metadata, to
   the configured fast rewrite model (`openai/gpt-5.6-luna` by default).
   It must remove material, texture, shader, geometry, asset, scatter,
   repetition, architecture, and production-design feedback; remove hidden
   renderer speculation; and rewrite only supported visible symptoms into the
   permitted lighting/optics/post language. Require the final result to contain:
   - classification `REAL`, `CG`, or `INDETERMINATE`, plus calibrated confidence;
   - up to three localized observable cues for and against photographic realism,
     each classified as lighting, shadows, lighting color/aesthetics,
     atmosphere, depth of field, optical softness/sharpness, highlight rolloff,
     grain, bloom, vignette, chromatic aberration, grading, another post effect,
     or render artifact;
   - up to three changes with region, direction, expected effect, category, and
     deterministic scope: `relighting`, `look-profile`, or
     `technical diagnosis`. Scope is derived by the tool, never by either model.
   Preserve both passes' prompt/input/response hashes, models, requests, usage,
   and the one-image/zero-image receipts. Do not treat the label alone as proof.
   If the one-image boundary fails or context leaks, discard the result instead
   of substituting a model that already knows the answer.
   Vision prompt versions retain different strategies: `literal-v1` is direct
   and non-speculative; `evidence-v1` balances both interpretations;
   `causal-v1` tests broad physical causes; `look-only-v1` prioritizes
   actionable cinematic finishing. The shared rewrite pass applies the same
   final boundary to every version and may return `INDETERMINATE` when retained
   look-only evidence cannot support the vision label. Neither pass retries.
5. Call `review_look_render(mode="REFERENCE", reference_packet=...)` as a
   separate reference-conditioned comparison. The packet contains the fixed
   brief/targets/non-targets and two-to-four local, checksum-bound reference
   images with stable IDs, assigned roles, and comparison focuses. The first
   Claude pass receives the Composite plus the actual reference pixels and
   judges every declared role, including lighting, value/color,
   atmosphere/depth, camera/finish, materials/surfaces, geometry/assets, and
   composition/staging. It must compare each reference independently and return
   `MATCH`, `PARTIAL`, `MISS`, or `OUT_OF_SCOPE`; do not rank the references or
   average them into one style. Every delta cites a reference ID and visible
   candidate region. It returns no aggregate score, winner, real/CG label,
   accept verdict, parameter values, or executable patch.

   Preserve the complete validated Claude analysis from `holistic_result` when
   deciding what to do next. The required second pass sends only that structured
   feedback, with zero images, to `google/gemini-3.6-flash` by default. Gemini
   produces a concise `look_directions` handoff for lighting, world,
   atmosphere, camera, and post while retaining material, geometry/asset,
   composition/staging, and render-artifact findings as non-executable
   `deferred_context`. Do not discard the holistic result and do not disguise
   deferred context as a relighting action. Preserve the reference packet hash,
   candidate/reference image hashes, both request/usage/hash receipts, the
   vision image count, and the zero-image rewrite boundary. A rewrite failure is
   fail-closed but retains the holistic result and its hash in the error.
6. Call `review_look_render(mode="CRITIQUE")` for the separate beauty-plus-AOV
   technical/causal judgment. This lane does not see the web references and is
   neither the blind challenge nor the holistic comparison. Reconcile all four
   records: the blind REALISM result, the complete reference `holistic_result`,
   Gemini's filtered reference handoff, and the AOV-backed CRITIQUE. Prioritize
   no more than three directions supported by visible evidence, classify
   out-of-scope blockers honestly, and present only allowlisted `SET` actions as
   an exact proposed profile patch. Do not apply them until the user explicitly
   approves.
7. After approval, reconstruct the full strict profile, preserve component IDs,
   and compile with `CREATE_VERSION`. Rerender the same cameras, repeat the blind
   and reference-conditioned checks with the same packet and rubric, then call
   `review_look_render(mode="COMPARE")` against the prior result.
8. Never auto-accept. A model's `REAL` label, reference match, or hosted-critic
   accept is advisory evidence, not ground truth. Stop after two user-approved
   revision cycles.

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
- Treat any preview, batch item, review packet, comparison, or acceptance result
  reporting an engine other than `CYCLES` as invalid evidence. Do not show it to
  the critic or use it to approve a look.
- Invalidate a blind realism result if the call included provenance, AOVs,
  references, filenames, or text that disclosed the synthetic source.
- Do not claim a reference comparison when the evaluator saw only captions,
  thumbnails, inaccessible URLs, or text descriptions rather than image pixels.
- If capture preparation succeeds and a later step fails, restore the preview
  session before doing anything else.
