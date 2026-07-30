# Full-Composite acceptance checklist

The image used for acceptance must be the final output of the profile Scene's
authoritative compositor, never a viewport capture or raw Combined pass.
It must be rendered with Cycles. Any preview, beauty, diagnostic packet,
comparison, or acceptance result reporting Eevee or any engine other than
`CYCLES` is invalid evidence and must be rerendered.

Rendering is asynchronous. `SUBMITTED` or `RUNNING` batch state is not evidence;
retrieve an exact `SUCCEEDED` item through `get_look_render_result`. The built-in
job/evidence registry is process-local, so retrieve and accept the result before
restarting or reloading Blender.

Verify the result reports:

- `image_source=COMPOSITE_OUTPUT` or equivalent final-output provenance;
- profile ID/hash and compositor hash;
- explicit Scene, View Layer, camera, frame, engine, and samples;
- `engine=CYCLES` exactly;
- full artifact path, byte size, dimensions, and SHA-256;
- returned proxy format/dimensions/checksum when a proxy is used;
- color-management view transform, look, exposure, and gamma;
- restored File Output/render/frame/Scene state and any warnings.

For cinematic review, additionally require `include_review_packet=true` and
verify the result returns exactly two native images in this order:

1. the authoritative final Composite beauty;
2. one labeled 3x2 diagnostic contact sheet from that same render.

The contact sheet order is Combined before compositing, Diffuse Direct, Glossy
Direct plus Indirect, Emission, normalized Camera Depth, and stable false-color
Object Cryptomatte. Camera Depth maps geometry at the camera focus distance to
a dark-mid reference, nearer geometry toward white, and distant or infinite
geometry toward black. Its 3x3 median cleanup affects only the display proxy,
not the underlying Z data. An unsupported pass must be labeled `UNAVAILABLE`
with a warning; it must never be silently replaced. Verify every available tile
has dimensions, byte count, SHA-256, mean energy, and nonzero coverage, and that
the packet carries the technical audit and persisted review intent.

Inspect visually for:

- motivated key/fill/practical direction and readable depth;
- bloom/glare threshold, clipping, and highlight rolloff;
- fog integration, transparency/alpha edges, and volume noise;
- deterministic grain level, vignette, banding, and color casts;
- rain density/scale/direction and obvious repetition;
- camera crop, DOF, motion blur, and unintended Scene content;
- raw-versus-composite difference when diagnosing the post path.

Do not accept when the compositor output is missing or ambiguous, the artifact
predates the current profile hash, File Output escaped the declared root, the
proxy came from a second render, the profile/base revision changed after the
render started, or the reported engine is not `CYCLES`.

Do not call the hosted critic while the deterministic audit is `FAIL`. The
critic may return at most three typed edits to existing profile controls. Show
the exact patch and wait for user approval; after an approved new profile
version, compare before versus after on the same camera/frame. Never auto-accept
the critic's verdict and stop after two approved revision cycles.

After the visual check passes, call `accept_look_profile` with the exact batch
ID, result ID, artifact SHA-256, and `review_acknowledged=true`. The handler
rechecks the manifest, source fingerprints, owned compositor, checksum, exact
compiled final dimensions, engine, samples, and denoise state before changing
DRAFT to ACCEPTED. A `SKIPPED`, failed, cancelled, preview-preset, or stale item
is never eligible. Rendering and acceptance never save the `.blend`.
