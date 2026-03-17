# sts-web

Browser-native speech-to-speech running 100% client-side via Rust/WASM + WebGPU.

[**Try the demo →**](https://idle-intelligence.github.io/sts-web/web/)

> **Disclaimer:** This is an experimental port. The model weights are Q4_K-quantized from [nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1), a 24-layer pruned + QLoRA recovery variant of PersonaPlex-7B-v1. Inference quality may differ from the original implementation. This project is not affiliated with or endorsed by NVIDIA.

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

## Architecture

- `crates/sts-wasm/` — Temporal transformer (24L) + depth transformer (6L × 8 steps) in Burn + wgpu, Q4_K GGUF quantization
- **Mimi codec** (`mimi-rs`) — Audio tokenizer/detokenizer, 8 codebooks at 12.5Hz
- `web/` — Standalone demo, Web Workers for inference + Mimi decode, AudioWorklet for playback
- Model weights fetched from HuggingFace at runtime, cached via Cache API
