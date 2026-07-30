# Deterministic Blender look profiles

Look profiles turn one artist-authored source Scene into a native Blender catalog
of independently switchable lighting/rendering/compositing variants. The system
is designed for dataset production: roughly a dozen looks per asset and hundreds
of source files, without duplicating all geometry or mutating one broad Scene
immediately before every render.

## Ownership model

Each look is persisted in the marked `AI_LOOK_PROFILES` Text manifest and
compiled to a managed shallow linked Scene:

```text
source BASE Scene (immutable)
  shared geometry, objects, textures, ordinary materials

compiled profile Scene
  shared BASE Collections
  unique profile Collection
    unique lights
    bounded fog
    deterministic seeded rain
    optional camera clone
  unique World
  unique compositor assignment
  profile render/color settings
```

Collections carry profile payloads; selecting the compiled Scene is the complete
look switch. Per-View-Layer Collection exclusion is used only where it is safe.
`VALIDATE` runs the same fail-closed isolation analysis as `COMPILE`, without
changing LayerCollection flags or creating datablocks. In a compiled linked
Scene, light-only Collections are excluded per View Layer, mixed Collections
are replaced by profile-owned clones containing only their non-artist-light
members, and a direct root-level artist light is unlinked only from the copied
Scene root. Shared objects, geometry, source Collections, and global visibility
remain unchanged. The compiler refuses any membership or nested structure it
cannot reproduce and isolate safely.

## Public MCP tools

- `get_look_profile_context`: inspect a named source Scene, catalog, ownership,
  revisions, and Blender compositor capabilities without mutation.
- `upsert_look_profile`: `VALIDATE` or atomically `COMPILE` one deterministic
  profile using `CREATE_VERSION` or `REPLACE_DRAFT`.
- `activate_look_profile`: switch one explicit Blender window to a compiled
  managed Scene. It does not save the file.
- `inspect_look_review_state`: deterministically audit managed shadows,
  reflection transport/visibility, compositing, Collection inclusion, camera
  ownership, and revision provenance before a cinematic review render.
- `accept_look_profile`: promote one DRAFT only from a visually reviewed,
  current, checksum-bound, full-profile-preset Composite result.
- `submit_look_render_batch`: queue bounded, explicit Scene/profile/frame work.
- `get_look_render_batch`: poll aggregate state and paginated item results.
- `cancel_look_render_batch`: cooperatively cancel queued work while retaining
  completed artifacts.
- `get_look_render_result`: return metadata and native MCP image content derived
  from the full artifact's final Composite. Review-enabled results return the
  Composite first and one labeled six-tile diagnostic sheet second.
- `review_look_render`: send one isolated 1600px JPEG beauty for `REALISM`; one
  beauty plus two to four checksum-bound local reference images for `REFERENCE`;
  one beauty/diagnostic pair for `CRITIQUE`; or two matched before/after evidence
  pairs for `COMPARE`. `REALISM` returns localized cues and deterministically
  scoped directions. `REFERENCE` preserves a holistic comparison and adds a
  Gemini-authored decision handoff that separates look directions from deferred
  scene context. Only `CRITIQUE` validates at most three bounded profile-field
  actions.

The MCP client performs strict Pydantic validation before any request reaches
Blender. The add-on repeats ownership, revision, path, compositor, and managed-ID
checks at the mutation boundary.

## Profile lifecycle

1. Save/open a disposable duplicate or explicitly authorize the current file.
2. Inspect the explicit source Scene and call
   `get_lighting_context(detail="CANDIDATES")` plus the raycast tools for
   motivated directions. The Blender 5.2-safe scanner stores only stable object
   identifiers and plain matrix snapshots while iterating the depsgraph, then
   reacquires evaluated objects for deferred mesh work. Iterator-scoped
   `DepsgraphObjectInstance`, evaluated-object, or RNA matrix references must
   never be retained after advancing `depsgraph.object_instances`.
3. Resolve the complete deterministic profile spec, including all component
   seeds and sampled values. When the deliverable aspect or dimensions differ
   from the artist source, set explicit managed `render.resolution_x` and
   `render.resolution_y`; the compiled profile Scene owns those settings while
   the source Scene remains unchanged.
   A `MANAGED_CLONE` camera may likewise carry deterministic
   `location_world` and `target_point` fields for a profile-owned reframe
   without changing the source camera object.
4. Call `upsert_look_profile(action="VALIDATE")`; this creates no datablocks
   and preflights the exact `MUTE_NON_MANAGED` isolation plan, including shared
   exclusions, profile-owned mixed-Collection clones, and copied-root unlinks.
5. Call `upsert_look_profile(action="COMPILE")`; the manifest write is the
   transaction commit point.
6. Activate and inspect a composited preview.
7. Compile a new immutable draft revision for changes; do not rewrite an
   accepted/dirty/frozen profile in place.
8. Submit a full-quality composited render and retrieve it through
   `get_look_render_result`.
9. For cinematic or dataset-ready judgment, audit first and submit with
   `include_review_packet=true`. Call `REALISM` before revealing provenance or
   diagnostics. When a fixed photographic packet exists, call `REFERENCE` with
   its two to four local image paths, SHA-256 values, assigned roles, brief, and
   targets. Reconcile its complete `holistic_result`, filtered look handoff, and
   the separate beauty-plus-AOV `CRITIQUE`; do not let deferred material,
   geometry, composition, or artifact context masquerade as a relighting patch.
   Present the exact proposed patch and do not mutate until the user approves it.
   An approved change becomes a new `CREATE_VERSION`, followed by one matched
   rerender and `COMPARE` call.
10. After visual review, call `accept_look_profile` with the exact batch/result
   IDs and artifact checksum. Preview-resolution or stale evidence is rejected.

The compiler never saves the `.blend`. Saving remains a separate, explicit user
decision.

## Full-Composite contract

Blender 4.2 uses an embedded `Scene.node_tree` and an authoritative Composite
node. Blender 5.x uses `Scene.compositing_node_group`, an `Image` interface
output, and `NodeGroupOutput`. The adapter detects capabilities on the Scene
instance and validates one linked final output before rendering.

Submission is asynchronous: `submit_look_render_batch` returns a batch record,
`get_look_render_batch` is the authoritative polling surface, and
`get_look_render_result` becomes available only for a completed item. A
successful result records:

- `image_source=COMPOSITE_OUTPUT`;
- profile/source Scene/version/revision;
- target Scene, View Layer, camera, frame, engine, samples, and denoise state;
- color-management settings and compositor adapter/hash;
- full artifact path, dimensions, byte count, and SHA-256;
- proxy dimensions/format/byte count, generated only after the full artifact is
  complete.

When `include_review_packet=true`, the renderer temporarily enables the needed
View Layer passes and writes them from the same render evaluation. It restores
the View Layer, Scene, camera, frame, render settings, compositor, and File
Output state afterward. The MCP response contains exactly two native images in
this order:

1. authoritative final Composite beauty;
2. a labeled 3x2 sheet containing raw Combined, Diffuse Direct, Glossy Direct +
   Indirect, Emission, normalized Camera Depth, and stable false-color Object
   Cryptomatte.

Every tile records dimensions, SHA-256, mean energy, and nonzero coverage.
Camera Depth uses a deterministic reciprocal camera-space mapping: the camera
focus distance establishes a dark-mid reference value, nearer geometry
approaches white, and distant or infinite geometry approaches black. A 3x3
median cleanup affects only the display proxy so low-sample transparency noise
does not obscure the layout; the rendered Z data remains unchanged.
Unsupported passes are labeled `UNAVAILABLE` and emit a warning. A conservative
magenta-pixel test is warning-only. The deterministic audit is authoritative
for explicit Blender switches: the image critic may observe or hypothesize a
visual problem, but may not contradict the audited switch state.

Hosted review is opt-in and runs only when `review_look_render` is called. It
uses non-streaming OpenRouter requests with strict structured output, no hidden
retry, and a 180-second read timeout. A technical audit failure prevents every
provider call. `REALISM` accepts `literal-v1`, `evidence-v1`, `causal-v1` (the
default), or `look-only-v1` and intentionally uses two calls. The first sends
exactly one 1600px JPEG beauty to the selected vision model. The second sends
only the structured first-pass feedback to `openai/gpt-5.6-luna` by default;
it receives zero images and rewrites/filters the result into lighting, shadows,
lighting color/aesthetics, atmosphere, camera/optical qualities, and post
effects. Material, geometry-asset, and artist-scene actions cannot survive the
rewrite schema, and scope is assigned deterministically from the retained
category. The vision versions remain distinct: `literal-v1` makes a direct
visible judgment, `evidence-v1` balances evidence for both interpretations,
`causal-v1` tests physical causes broadly, and `look-only-v1` prioritizes
actionable cinematic finishing. The receipt preserves both models, requests,
usage records, prompt/input/response hashes, the one-image vision boundary, and
the zero-image rewrite boundary. Validation failures preserve their stage,
request ID, usage, response hash, and schema errors and are never retried. No
filename, diagnostic sheet, Scene/camera/profile metadata, render engine, or
synthetic provenance is sent in REALISM mode. Credentials remain in the MCP
server child environment and are never sent to Blender or returned in tool
output.

`REFERENCE` is a separate holistic, reference-conditioned lane, not a realism
classifier and not an AOV diagnosis. Its first call sends exactly the candidate
beauty followed by two to four reference JPEG proxies. The caller binds every
local reference to a SHA-256, a stable ID, one or more explicit roles, and a
comparison focus. Supported roles are `LIGHTING_TOPOLOGY`, `VALUE_AND_COLOR`,
`ATMOSPHERE_AND_DEPTH`, `CAMERA_AND_FINISH`, `MATERIAL_AND_SURFACE`,
`GEOMETRY_AND_ASSET`, and `COMPOSITION_AND_STAGING`. The model must assess every
declared reference-role pair once as `MATCH`, `PARTIAL`, `MISS`, or
`OUT_OF_SCOPE`, localize candidate evidence, and record reference conflicts
without averaging them into a generic style. It returns no score, winner,
real/CG label, accept verdict, exact parameter values, or executable patch.

The complete validated first pass remains available as `holistic_result` and is
checksum-bound by `raw_response_sha256`. A second, zero-image call to
`google/gemini-3.6-flash` by default receives only that structured comparison.
It emits up to three lighting/world/atmosphere/camera-post `look_directions` and
keeps material, geometry/asset, composition/staging, and render-artifact notes
as `deferred_context`. Scope is assigned deterministically by the tool. If the
rewrite fails, the call fails closed with `REFERENCE_REWRITE_FAILED` while
including the complete validated holistic result and its hash in the error.
Receipts preserve both request IDs, models, usage, prompt/response hashes, the
five-image maximum vision boundary, the zero-image rewrite boundary, packet
hash, candidate/source hashes, and every reference source/proxy hash. The
provider sees the bounded brief, roles, focuses, and image pixels, but never
local paths, source URLs, filenames, AOVs, Scene/camera/profile metadata, render
engine, or synthetic provenance.

All root and nested File Output nodes are redirected beneath the declared root,
their filename/slot patterns are path-sanitized, and their state is restored
exactly. A pre-existing output handled by `SKIP` is reported as `SKIPPED` and
cannot become acceptance evidence. Blender's Render Result is replaced by real
renders and that side effect should be disclosed to the user.

Acceptance is a separate evidence-gated mutation. Only a current `SUCCEEDED`
Composite result rendered with the compiled profile's exact final engine,
samples, denoise, and resolution preset can promote a DRAFT. The exact batch
ID, result ID, artifact SHA-256, current revisions, unchanged source
fingerprints, valid managed ownership, and `review_acknowledged=true` are
rechecked. Preview, raw Combined, `SKIPPED`, failed, cancelled, stale, or
preset-mismatched results are not acceptance evidence. Neither rendering nor
acceptance saves the `.blend`.

## Blender 5.2 verified behavior

The integration fixtures verify that:

- raw Combined and final Composite sentinel images differ;
- the final saved PNG is the post-compositor output;
- new and legacy File Output paths can be redirected/restored through their
  version-specific APIs;
- nested compositor groups are deep-owned, recursively hashed, and cannot leak
  File Output paths or edits back to the source graph;
- shallow linked Scenes have independent compositor assignments in Blender 5.2;
- divergent compositor/File Output state survives save/reopen;
- managed lights, bounded fog, and deterministic rain render together in a
  profile-local Collection;
- one exact render evaluation produces the Composite and all available review
  tiles, while success, failure, and cancellation restore the touched state;
- three-camera expansion produces collision-safe camera-specific items and the
  same Cryptomatte color assignment for the same object identifiers.

Live Blender 5.2 testing additionally proved that `VALIDATE` rejects unsafe
`MUTE_NON_MANAGED` mixed Collections before mutation, while `KEEP` remains
available for scenes that cannot be reorganized. The camera candidate scan was
also hardened after a large-scene Blender 5.2 host crash exposed that
depsgraph-iterator RNA wrappers are shorter-lived than ordinary Python
references; the current scan snapshots plain values and reacquires evaluated
objects instead of retaining those wrappers.

## Current limitations

- The asynchronous batch registry is process-local and in memory. Blender
  restart/reload loses job records, so retrieve and accept evidence in the same
  live session or use an external durable scheduler around the MCP.
- Rendering is serialized inside one Blender process. Cancellation is
  cooperative: queued items stop, but the active frame may finish before the
  batch becomes cancelled.
- `CROSS_PRODUCT` is intentionally not an arbitrary many-profile by many-Scene
  matrix. It supports one profile across Scenes that already carry that exact
  managed ownership; use `PAIRWISE` for multiple compiled profiles.
- The v1 rain payload is deterministic static Curve streak geometry. It is not
  animated precipitation, a fluid/surface simulation, or shared-material
  wetness; shared materials remain untouched.
- Blender 5.2 is live- and integration-tested. The Blender 4.2 legacy
  compositor adapter has focused compatibility tests but still needs an
  equivalent live 4.2 acceptance run.
- Camera Depth is derived from the same render's Z pass and normalized with the
  active camera's focus distance (or clip-range geometric mean when no useful
  focus distance exists), then display-cleaned with a 3x3 median filter. It is
  not a second render.
- `MUTE_NON_MANAGED` is allowed only when every artist-light membership can be
  isolated inside the compiled Scene by exclusion, managed non-light cloning,
  or copied-root unlinking. Ambiguous or unsafe nesting fails before mutation.

Run the focused Python tests:

```bash
uv run pytest -q \
  tests/test_tools/test_look_profiles.py \
  tests/test_addon/test_look_profiles_handler.py \
  tests/test_addon/test_look_compositor.py \
  tests/test_addon/test_look_profile_assets.py \
  tests/test_addon/test_look_profile_rendering.py \
  tests/test_tools/test_look_review.py
```

Run the destructive save/reopen fixture only in an isolated Blender process:

```bash
/Applications/Blender.app/Contents/MacOS/Blender \
  --background --factory-startup \
  --python tests/integration/run_look_profile_compositor_integration.py \
  -- --confirm-isolated-process
```

The actual compiler and hardened recursive compositor paths also have native
fixtures:

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup \
  --python tests/integration/run_look_profile_handler_integration.py
/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup \
  --python tests/integration/run_look_compositor_hardening_integration.py -- \
  --confirm-isolated-process
```

The installed and repository copies of the `blender-look-profiles` skill encode
the complete generation, preview, batch, acceptance, and failure workflow.
