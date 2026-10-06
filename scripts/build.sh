#!/usr/bin/env bash
# Builds the sts-wasm and mimi-wasm WebAssembly packages and assembles the
# deployed site into _site/, matching the layout already live on
# idle-intelligence.github.io/sts-web (pkg/, mimi-pkg/, web/).
#
# The demo's JS loads model shards through a `/hf/...` path during local dev
# (served by web/serve.mjs, which proxies that prefix to the sibling `hf/`
# cache directory). The deployed site has no such proxy, so this script
# rewrites HF_BASE to the real Hugging Face URL in the built copy only —
# the source files keep the dev proxy path.
#
# Usage: ENGINE_BUILD=<tag> scripts/build.sh
# ENGINE_BUILD defaults to "dev" for local builds; CI passes the commit sha.
# It is required on every real deploy - a rebuild with no tag bump keeps
# browsers running the cached worker/pkg.
set -euo pipefail

ENGINE_BUILD="${ENGINE_BUILD:-dev}"
HF_PROD_BASE="https://huggingface.co/idle-intelligence/personaplex-24L-q4_k-webgpu/resolve/main"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Building sts-wasm (wasm feature) for ENGINE_BUILD=$ENGINE_BUILD"
RUSTFLAGS="--remap-path-prefix=$HOME=/home" \
  wasm-pack build crates/sts-wasm --target web --release --no-default-features --features wasm

echo "==> Building mimi-wasm"
RUSTFLAGS="--remap-path-prefix=$HOME=/home" \
  wasm-pack build crates/mimi-wasm --target web --release

# --- Local-path / user-name leak check on built wasm outputs ---
for WASM in crates/sts-wasm/pkg/sts_wasm_bg.wasm crates/mimi-wasm/pkg/mimi_wasm_bg.wasm; do
    LEAKS=$(strings "$WASM" | grep -F -e "$HOME" -e "Code/" -e ".claude/" -e "/Users/" || true)
    USER_HITS=$(strings "$WASM" | grep -Fw -e "$(id -un)" || true)
    if [ -n "$LEAKS$USER_HITS" ]; then
        echo "error: $WASM contains local paths or the user name:" >&2
        printf '%s\n%s\n' "$LEAKS" "$USER_HITS" | grep -v '^$' | head -20 >&2
        exit 1
    fi
done
echo "==> no local paths in built wasm"

# --- Assemble the deployed site into _site/ ---
echo "==> Assembling _site"
rm -rf _site
mkdir -p _site/pkg _site/mimi-pkg _site/web
cp crates/sts-wasm/pkg/sts_wasm.js crates/sts-wasm/pkg/sts_wasm_bg.wasm _site/pkg/
cp crates/mimi-wasm/pkg/mimi_wasm.js crates/mimi-wasm/pkg/mimi_wasm_bg.wasm _site/mimi-pkg/
cp web/index.html web/worker.js web/mimi-worker.js web/audio-playback.js web/audio-processor.js _site/web/

# --- Rewrite the ENGINE_BUILD tag on every loading URL ---
echo "==> Rewriting ENGINE_BUILD tag to $ENGINE_BUILD"
sed -i.bak "s/const ENGINE_BUILD = \"[^\"]*\";/const ENGINE_BUILD = \"$ENGINE_BUILD\";/" \
  _site/web/index.html _site/web/worker.js _site/web/mimi-worker.js
rm -f _site/web/index.html.bak _site/web/worker.js.bak _site/web/mimi-worker.js.bak

COUNT="$(grep -rEo "ENGINE_BUILD = \"[^\"]*\"" _site/web/index.html _site/web/worker.js _site/web/mimi-worker.js | grep -Fc "\"$ENGINE_BUILD\"")"
if [ "$COUNT" -ne 3 ]; then
    echo "error: expected 3 ENGINE_BUILD assignments rewritten to $ENGINE_BUILD, found $COUNT" >&2
    exit 1
fi

# --- Rewrite the dev /hf/ proxy path to the real Hugging Face URL ---
echo "==> Rewriting HF_BASE to the production Hugging Face URL"
sed -i.bak "s#/hf/personaplex-7b-v1-q4_k-webgpu#$HF_PROD_BASE#g" \
  _site/web/index.html _site/web/worker.js
rm -f _site/web/index.html.bak _site/web/worker.js.bak

if grep -q '/hf/personaplex-7b-v1-q4_k-webgpu' _site/web/index.html _site/web/worker.js; then
    echo "error: dev /hf/ proxy path still present in _site" >&2
    exit 1
fi

echo "==> Wrote _site"
