# Development and installation notes

## Optional cinematic-review credential

The MCP server is launched through `scripts/run_blend_ai_with_secrets.sh`. The
launcher resolves only the `op://` reference in
`~/.codex/secret-env/blend-ai.env` into the child MCP process. It first uses the
Keychain-backed 1Password service account and falls back to the current
1Password CLI session. Blender receives no hosted-provider credential. A real
OpenRouter request still requires an explicit `review_look_render` call; tests
use a local deterministic adapter and never consume paid API usage. The native
acceptance harness exercises the one-image vision plus zero-image text rewrite
stages of `REALISM`; the candidate-plus-two-reference vision and zero-image
Gemini rewrite stages of `REFERENCE`; two-image `CRITIQUE`; and four-image
`COMPARE`. It preserves the native image and receipt artifacts, including the
complete holistic reference result and the filtered decision handoff.

## Provenance

- Public fork: `https://github.com/Redmond-AI/blend-ai`
- `origin`: the public fork above
- `upstream`: `https://github.com/HoldMyBeer-gg/blend-ai.git`
- Development branch: `codex/relighting-tools`
- Pinned upstream base: `621ddc0dc8b379428e027c17ce817e1fc4d9cb36`

The pinned commit is tagged `v1.2.2`; this fork now reports version `1.4.0` in
the Python package, Blender manifest, and legacy `bl_info` metadata to identify
the cinematic-review, persistent look-profile, compositor, and batch-render
feature release.

The six-commit implementation and test tree was tested at fork commit
`c356372d81115d739766b2f2e2d59c07d6ea8402`. The pinned SHA above is the
upstream base, not the tested fork commit. This installation-note-only child
commit does not change executable or test code.

## Verified local installation

The developer checkout is:

`/Users/robingraham/Library/CloudStorage/Dropbox/github/blend-ai-relighting`

The Blender 5.2 developer extension is a symlink:

`/Users/robingraham/Library/Application Support/Blender/5.2/extensions/user_default/blend_ai`

It points to the checkout's `addon` directory. Refuse to replace an unverified
directory or a symlink with a different target. After any add-on source change,
stop the N-panel server, quit Blender completely, and relaunch it; Reload Scripts
is insufficient because the background TCP thread can survive module reloads.

The Codex Desktop MCP entry launches the locked environment with:

```toml
[mcp_servers.blend_ai_relighting]
command = "/Users/robingraham/.local/bin/uv"
args = [
  "run",
  "--directory",
  "/Users/robingraham/Library/CloudStorage/Dropbox/github/blend-ai-relighting",
  "blend-ai"
]
startup_timeout_sec = 30
tool_timeout_sec = 300
enabled = true
```

The server dependency lock currently tests MCP Python SDK `1.28.1`, Pillow
`12.3.0`, and Pydantic `2.12.5` under Python `3.13`. The Blender extension
remains zero-dependency. Blender's add-on server binds only to `127.0.0.1:9876`.

## Acceptance status

The deterministic warehouse has passed both native image paths through the MCP
protocol: an 8-sample denoised PNG quick render and a 16-sample, two-second
denoised JPEG visible-viewport capture. The 20,000-instance scale fixture also
passes functionally. Current measured target misses are:

- Warm visible-viewport overhead is about 1.9 seconds beyond the requested
  settle interval, versus the initial under-one-second target.
- A 64-light apply takes 395.64 ms, versus the initial under-250 ms target.

Detailed final measurements and ignored local evidence paths are in
[`relighting.md`](relighting.md). The real warehouse gate is intentionally still
open until a duplicate scene path is supplied. Do not merge a release or claim
full acceptance before that test, rollback verification, and final commit-SHA
recording are complete.

## Capture and transaction cautions

- `capture_mode="VIEWPORT"` returns visible Cycles editor pixels and leaves
  `Render Result` alone.
- `capture_mode="QUICK_RENDER"` is explicit only. It uses denoising, updates
  `Render Result`, removes its temporary source PNG, and never saves the
  `.blend` file. It is not a silent fallback for failed viewport capture.
- Managed transaction rollback is global LIFO. Whole-scene relighting snapshots
  must be unwound newest first, even when transactions belong to different
  plan IDs.
