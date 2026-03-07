#!/usr/bin/env python3
"""Run BF16 reference inference on joke.wav with PersonaPlex-7B.

Produces reference audio tokens and output WAV for comparison
with the Q4 Rust implementation.
"""

import sys
import os
import json
import numpy as np

# Monkey-patch torch.compile for Python 3.14
import torch
if sys.version_info >= (3, 14):
    torch.compile = lambda fn, *a, **k: fn

from pathlib import Path

MODEL_PATH = Path("/Users/tc/Code/idle-intelligence/hf/personaplex-7b-v1/model.safetensors")
MIMI_PATH = Path("/Users/tc/Code/idle-intelligence/hf/personaplex-7b-v1-q4_0-webgpu/tokenizer-e351c8d8-checkpoint125.safetensors")
WAV_PATH = Path("/Users/tc/Code/idle-intelligence/sts-web/tests/reference/joke.wav")
OUTPUT_DIR = Path("/Users/tc/Code/idle-intelligence/sts-web/tests/reference")

def load_wav(path):
    """Load WAV file as float32 numpy array."""
    import soundfile as sf
    samples, sr = sf.read(str(path), dtype='float32')
    if samples.ndim > 1:
        samples = samples[:, 0]
    return samples, sr

def save_wav(path, samples, sr=24000):
    """Save float32 samples as 16-bit WAV."""
    import wave
    import struct
    samples = np.clip(samples, -1.0, 1.0)
    int_samples = (samples * 32767).astype(np.int16)
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(int_samples.tobytes())

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Mimi codec
    print("Loading Mimi codec...")
    from moshi.models import loaders
    mimi = loaders.get_mimi(MIMI_PATH, device=device)
    mimi.set_num_codebooks(32)
    print(f"  Mimi: {mimi.num_codebooks} codebooks, sample_rate={mimi.sample_rate}")

    # Load PersonaPlex model
    print("Loading PersonaPlex-7B (BF16)...")
    lm = loaders.get_moshi_lm(filename=MODEL_PATH, device=device, lm_kwargs_overrides={'dep_q': 16})
    lm.eval()
    print(f"  Model loaded: dep_q={lm.dep_q}, n_q={lm.n_q}, card={lm.card}")

    # Load test audio
    print(f"Loading audio: {WAV_PATH}")
    samples, sr = load_wav(WAV_PATH)
    print(f"  {len(samples)} samples, {sr}Hz, {len(samples)/sr:.2f}s")

    # Resample to 24kHz if needed
    if sr != 24000:
        from scipy import signal
        samples = signal.resample(samples, int(len(samples) * 24000 / sr))
        sr = 24000
        print(f"  Resampled to 24kHz: {len(samples)} samples")

    # Encode with Mimi
    print("Encoding with Mimi...")
    audio_tensor = torch.from_numpy(samples).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, T]
    with torch.no_grad():
        with mimi.streaming(1):
            # Process in chunks matching Mimi's frame size
            frame_size = 1920  # 80ms at 24kHz
            all_codes = []
            for i in range(0, audio_tensor.shape[-1], frame_size):
                chunk = audio_tensor[:, :, i:i+frame_size]
                if chunk.shape[-1] < frame_size:
                    chunk = torch.nn.functional.pad(chunk, (0, frame_size - chunk.shape[-1]))
                codes = mimi.encode(chunk)  # [1, n_q, T']
                if codes.shape[-1] > 0:
                    all_codes.append(codes.cpu())

    if all_codes:
        all_codes = torch.cat(all_codes, dim=-1)
    else:
        print("No codes produced!")
        return
    print(f"  Encoded: {all_codes.shape} (n_q={all_codes.shape[1]}, T={all_codes.shape[2]})")

    # Run STS inference
    print("\nRunning STS inference (BF16, greedy)...")
    n_frames = min(all_codes.shape[2], int(os.environ.get("MAX_FRAMES", all_codes.shape[2])))
    dep_q = lm.dep_q  # 16
    num_codebooks = lm.num_codebooks  # 17 (1 text + 16 audio)
    card = lm.card  # 2048
    text_card = lm.text_card  # 32000

    # Initial tokens
    audio_pad = card  # 2048 = padding token (card = vocab size)
    text_start = text_card  # 32000 = text start token

    all_text_tokens = []
    all_model_audio_tokens = []

    with torch.no_grad():
        with lm.streaming(1):
            # Previous tokens for delay handling
            prev_text_token = text_start
            prev_model_audio = [audio_pad] * 8
            prev_user_audio = [audio_pad] * 8
            prev_prev_user_audio = [audio_pad] * 8
            prev_prev_model_audio = [audio_pad] * 8

            for frame_idx in range(n_frames):
                user_audio = all_codes[0, :8, frame_idx].tolist()  # 8 codebooks

                # Build input sequence with delays
                # Stream layout: [text, model_sem, model_ac1..7, user_sem, user_ac1..7]
                # delays:        [0,    0,         1..1,          0,        1..1]
                input_tokens = []

                # Stream 0: text (delay=0) -> previous text token
                input_tokens.append(prev_text_token)

                # Stream 1: model semantic (delay=0) -> previous model audio[0]
                input_tokens.append(prev_model_audio[0])

                # Streams 2-8: model acoustic (delay=1) -> prev_prev model audio[1..7]
                for i in range(1, 8):
                    input_tokens.append(prev_prev_model_audio[i])

                # Stream 9: user semantic (delay=0) -> current user audio[0]
                input_tokens.append(user_audio[0])

                # Streams 10-16: user acoustic (delay=1) -> prev user audio[1..7]
                for i in range(1, 8):
                    input_tokens.append(prev_user_audio[i])

                # Convert to tensor [1, 17, 1]
                input_tensor = torch.tensor(input_tokens, dtype=torch.long, device=device).view(1, num_codebooks, 1)

                # Forward through temporal transformer
                transformer_out, text_logits = lm.forward_text(input_tensor)
                # text_logits: [1, 1, 1, text_card_out]
                text_token = text_logits[0, 0, 0].argmax(dim=-1).item()

                # Forward through depth transformer (autoregressive)
                # The depformer needs its own streaming context per time step
                # so its KV cache accumulates across depth steps then resets
                depth_tokens = []
                dep_input = torch.tensor([[[text_token]]], dtype=torch.long, device=device)  # [1, 1, 1]
                with lm.depformer.streaming(1):
                    for cb_idx in range(dep_q):
                        dep_logits = lm.forward_depformer(cb_idx, dep_input, transformer_out)
                        # dep_logits: [1, 1, 1, card]
                        dep_token = dep_logits[0, 0, 0].argmax(dim=-1).item()
                        depth_tokens.append(dep_token)
                        # Next step conditioned on this token
                        dep_input = torch.tensor([[[dep_token]]], dtype=torch.long, device=device)

                model_audio = depth_tokens[:8]

                all_text_tokens.append(text_token)
                all_model_audio_tokens.append(model_audio)

                # Update delay buffers
                prev_prev_user_audio = prev_user_audio[:]
                prev_prev_model_audio = prev_model_audio[:]
                prev_user_audio = user_audio[:]
                prev_model_audio = model_audio[:]
                prev_text_token = text_token

                if frame_idx % 10 == 0:
                    print(f"  Frame {frame_idx}/{n_frames}: text={text_token}, audio={model_audio[:3]}...")

    # Decode model audio with Mimi
    print("\nDecoding model audio with Mimi...")
    model_audio_tensor = torch.tensor(all_model_audio_tokens, dtype=torch.long).T  # [8, T]
    model_audio_tensor = model_audio_tensor.unsqueeze(0).to(device)  # [1, 8, T]

    with torch.no_grad():
        mimi.set_num_codebooks(8)
        with mimi.streaming(1):
            output_pcm_chunks = []
            for t in range(model_audio_tensor.shape[-1]):
                codes_t = model_audio_tensor[:, :, t:t+1]
                pcm = mimi.decode(codes_t)  # [1, 1, samples]
                if pcm.shape[-1] > 0:
                    output_pcm_chunks.append(pcm.cpu().numpy().flatten())

    if output_pcm_chunks:
        output_pcm = np.concatenate(output_pcm_chunks)
    else:
        output_pcm = np.array([])

    print(f"  Output: {len(output_pcm)} samples ({len(output_pcm)/24000:.2f}s)")
    rms = np.sqrt(np.mean(output_pcm**2)) if len(output_pcm) > 0 else 0
    print(f"  RMS: {rms:.4f}")

    # Save outputs
    if len(output_pcm) > 0:
        out_wav = OUTPUT_DIR / "joke_output_bf16_ref.wav"
        save_wav(out_wav, output_pcm)
        print(f"  Saved: {out_wav}")

    # Save token log for comparison
    token_log = {
        "text_tokens": all_text_tokens,
        "model_audio_tokens": all_model_audio_tokens,
    }
    token_log_path = OUTPUT_DIR / "bf16_reference_tokens.json"
    with open(token_log_path, 'w') as f:
        json.dump(token_log, f, indent=2)
    print(f"  Token log: {token_log_path}")

    print("\nDone!")

if __name__ == "__main__":
    main()
