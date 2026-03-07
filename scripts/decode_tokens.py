#!/usr/bin/env python3
"""Decode audio tokens with the Python Mimi reference implementation.

Usage:
    python scripts/decode_tokens.py tests/reference/joke_output_tokens.json tests/reference/joke_output_python.wav

This loads the Mimi codec from the PersonaPlex checkpoint and decodes
the tokens saved by the Rust integration test. Comparing the output
audio against the Rust-decoded WAV isolates whether garbled audio
is from token generation or from our Mimi decode.
"""

import json
import sys
import torch
import numpy as np
import soundfile as sf

# Add moshi reference to path
sys.path.insert(0, "refs/moshi/moshi")

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <tokens.json> <output.wav>")
        sys.exit(1)

    tokens_path = sys.argv[1]
    output_path = sys.argv[2]

    # Load tokens
    with open(tokens_path) as f:
        data = json.load(f)

    num_codebooks = data["num_codebooks"]
    num_frames = data["num_frames"]
    tokens = data["tokens"]  # list of lists: [num_frames][num_codebooks]

    print(f"Loaded {num_frames} frames, {num_codebooks} codebooks")
    print(f"First 3 frames: {tokens[:3]}")

    # Build codes tensor [1, num_codebooks, num_frames]
    codes_np = np.array(tokens, dtype=np.int64).T  # [num_codebooks, num_frames]
    codes = torch.from_numpy(codes_np).unsqueeze(0)  # [1, num_codebooks, num_frames]
    print(f"Codes tensor shape: {codes.shape}")

    # Load Mimi model
    mimi_path = "/Users/tc/Code/idle-intelligence/hf/personaplex-7b-v1-q4_k-webgpu/tokenizer-e351c8d8-checkpoint125.safetensors"
    print(f"Loading Mimi from {mimi_path}...")

    from moshi.models import loaders
    mimi_model = loaders.get_mimi(mimi_path, device="cpu", num_codebooks=num_codebooks)

    print(f"Mimi loaded, num_codebooks={num_codebooks}")

    # Decode
    with torch.no_grad():
        audio = mimi_model.decode(codes)  # [1, 1, num_samples]

    audio_np = audio.squeeze().numpy()
    print(f"Decoded {len(audio_np)} samples ({len(audio_np)/24000:.2f}s)")
    print(f"Audio stats: min={audio_np.min():.4f}, max={audio_np.max():.4f}, rms={np.sqrt(np.mean(audio_np**2)):.4f}")

    # Save
    sf.write(output_path, audio_np, 24000)
    print(f"Written to {output_path}")


if __name__ == "__main__":
    main()
