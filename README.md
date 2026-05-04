# sts-web

Browser-native speech-to-speech running 100% client-side via Rust/WASM + WebGPU.

[**Try the demo →**](https://idle-intelligence.github.io/sts-web/web/)

> **Model:** [`idle-intelligence/personaplex-24L-q4_k-webgpu`](https://huggingface.co/idle-intelligence/personaplex-24L-q4_k-webgpu) — a layer-pruned (32L → 24L), LoRA-recovered, Q4_K-quantized derivative of [`nvidia/personaplex-7b-v1`](https://huggingface.co/nvidia/personaplex-7b-v1), built specifically to run in this WebGPU/native runtime. Pruning + recovery + quantization are by [@idle-intelligence](https://huggingface.co/idle-intelligence); the base PersonaPlex weights are NVIDIA's. See the [HF discussion](https://huggingface.co/idle-intelligence/personaplex-24L-q4_k-webgpu/discussions/1) for usage Q&A.

## Status

Work in progress. The pipeline runs end-to-end in Chrome/Edge with WebGPU — walkie-talkie mode functional, voice presets available. Audio quality is poor. Generation runs ~3x slower than realtime on consumer GPUs.

```
Microphone → AudioWorklet (24kHz mono) → Mimi encoder (WASM) → Temporal transformer (WASM/WebGPU) → Depth transformer (WASM/WebGPU) → Mimi decoder (WASM) → AudioWorklet playback
```

## Requirements

- Chrome 113+ or Edge 113+ (WebGPU required)
- HTTPS (required for WebGPU; dev server uses self-signed cert)
- Microphone access for voice input

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/idle-intelligence/sts-web.git
cd sts-web

# 2. Build WASM
wasm-pack build crates/sts-wasm --target web --no-default-features --features wasm

# 3. Start dev server
node web/serve.mjs

# 4. Open https://localhost:8443
```

## Native CLI (`sts`)

Run the model from a terminal — useful for trying `personaplex-24L-q4_k-webgpu` without a browser, scripting batch inference, or smoke-testing changes against `joke.wav`.

```bash
# 1. Download the model (~3.8 GB)
huggingface-cli download idle-intelligence/personaplex-24L-q4_k-webgpu \
    --local-dir personaplex-24L-q4_k-webgpu

# 2. Build and run (release; first build pulls Burn + cubecl, ~5 min)
cargo run --release --features "wgpu,cli" --bin sts -- \
    --model-dir ./personaplex-24L-q4_k-webgpu \
    --input  my_question.wav \
    --output response.wav \
    --voice  NATF2
```

The CLI loads the sharded GGUF, the Mimi codec safetensors, the SentencePiece tokenizer, and a `.pt` voice preset directly — no Python preprocessing. It runs an end-to-end speech-to-speech turn (voice prefill → system prompt → user audio prefill → response generation → Mimi decode) and writes a 24 kHz mono WAV. The model's inner-monologue text is also printed to stdout.

Requirements: Vulkan (Linux/Windows) or Metal (macOS) — wgpu auto-selects. ~4 GB VRAM. Input WAV must be mono; any sample rate is accepted (resampled to 24 kHz). Other voices: `NATF0..3`, `NATM0..3`, `VARF0..4`, `VARM0..4`. Run `sts --help` for all options (sampling temperatures, max frame count, layer count for non-default checkpoints).

## Architecture

- `crates/sts-wasm/` — Temporal transformer (24L pruned, 32L upstream) + depth transformer (6L × 16 steps, 8 generated) in Burn + wgpu, Q4_K GGUF quantization
- **Mimi codec** (`mimi-rs`) — Audio tokenizer/detokenizer, 8 codebooks at 12.5Hz
- `web/` — Standalone demo, Web Workers for inference + Mimi decode, AudioWorklet for playback
- Model weights fetched from HuggingFace at runtime, cached via Cache API
