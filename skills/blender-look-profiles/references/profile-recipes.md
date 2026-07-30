# Deterministic look-profile recipes

Choose correlated archetypes, then resolve bounded values with independent
component seeds. Values below are starting ranges, not reasons to skip camera
context or raycasts.

Every recipe is rendered and evaluated only with Cycles. The resolved profile
must explicitly set `render.engine="CYCLES"`. Reduce samples, enable
denoising, or lower resolution for previews; never substitute Eevee.

## Core three

### Summer day

- Clear managed sky or approved daylight HDRI; strength roughly 0.7-1.5.
- Motivated Sun, elevation 35-75 degrees, small angle 0.3-2 degrees for readable
  sharper shadows.
- High-key exposure/contrast with restrained highlight rolloff.
- Fog Glow above bright highlights; low or zero fog and grain.

### Creepy moonlight

- Cool Sun/large Area motivated by an opening and raycast toward important
  surfaces; preserve silhouette readability.
- Dark blue World, sparse practicals, and bounded low ground fog.
- Cooler grade, mild vignette, deterministic grain, restrained bloom.

### Rainy overcast

- Broad, cool overcast World with no hard Sun unless modeling a storm break.
- Seeded rain rig in the profile WEATHER payload and optional bounded ground
  mist. `rain_rate` is the deterministic drop count and is bounded to 0-5,000;
  zero produces no splines and oversized requests are rejected.
- The current v1 rain rig is static deterministic Curve streak geometry. It
  does not animate precipitation, simulate surface interaction, or make shared
  materials wet; use profile-owned overlay geometry for any additional wetness.
- Low-contrast grade, moderate grain, soft highlights. Use profile-owned overlay
  geometry for puddles/sheens; do not edit shared materials.

## Additional suite candidates

Use these to reach a varied twelve-look catalog without incoherent random knobs:

1. Golden hour: low warm Sun, cool sky fill, longer soft-edged shadows.
2. Blue-hour dusk: dim cool sky, warm practical accents, light haze.
3. Dawn mist: low pastel Sun, denser ground fog, low grain.
4. Flat overcast: soft neutral World, low contrast, no bloom.
5. Storm break: dark clouds/World with one raycast-verified hard shaft.
6. Winter clear: cool high sky, crisp Sun, restrained saturation.
7. Hot dusty noon: high Sun, warm haze, rolled highlights.
8. Foggy night: sparse motivated practicals, deep bounded fog, stronger grain.
9. Neon wet night: colored practical Areas, rain, post glow, overlay-only wetness.

## Stable sampling rules

- Sample Sun azimuth/elevation together, then raycast; do not sample them as
  unrelated decoration.
- Correlate World brightness, exposure, fog density, bloom threshold, and grain
  with the archetype's intended visibility.
- Derive seeds independently for `lighting`, `world`, `atmosphere`, `weather`,
  and `post`; adding a post parameter must not move the lights.
- Store the resolved values in the profile manifest. Re-rendering a compiled
  profile must consume no randomness.
