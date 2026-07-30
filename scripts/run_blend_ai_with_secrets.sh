#!/bin/sh

set -eu

HELPER=/Users/robingraham/.codex/skills/onepassword-secrets/scripts/op_secrets.py
ENV_FILE=/Users/robingraham/.codex/secret-env/blend-ai.env
KEYCHAIN_SERVICE=codex-deep-research-1password-service-account
UV_BIN=/Users/robingraham/.local/bin/uv
REPO=/Users/robingraham/Library/CloudStorage/Dropbox/github/blend-ai-relighting

if /usr/bin/python3 "$HELPER" check \
    --keychain-service "$KEYCHAIN_SERVICE" \
    --env-file "$ENV_FILE" >/dev/null 2>&1; then
    exec /usr/bin/python3 "$HELPER" run \
        --keychain-service "$KEYCHAIN_SERVICE" \
        --env-file "$ENV_FILE" -- \
        "$UV_BIN" run --directory "$REPO" blend-ai
fi

exec /usr/bin/python3 "$HELPER" run \
    --env-file "$ENV_FILE" -- \
    "$UV_BIN" run --directory "$REPO" blend-ai
