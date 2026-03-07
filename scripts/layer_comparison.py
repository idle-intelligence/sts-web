#!/usr/bin/env python3
"""Layer-by-layer BF16 reference logging for PersonaPlex-7B.

Runs exactly ONE frame of inference with all-padding input tokens,
logging intermediate values at every transformer layer for comparison
with the Q4 quantized Rust implementation.

Output: tests/reference/bf16_layer_log.json

Usage:
    cd /Users/tc/Code/idle-intelligence/sts-web
    .venv/bin/python scripts/layer_comparison.py
"""

import sys
import json

# Monkey-patch torch.compile for Python 3.14 compatibility
import torch
if sys.version_info >= (3, 14):
    torch.compile = lambda fn, *a, **k: fn

from pathlib import Path

MODEL_PATH = Path("/Users/tc/Code/idle-intelligence/hf/personaplex-7b-v1/model.safetensors")
OUTPUT_PATH = Path("/Users/tc/Code/idle-intelligence/sts-web/tests/reference/bf16_layer_log.json")


def tensor_stats(t: torch.Tensor) -> dict:
    """Extract norm and first 10 values from a tensor (flattened, float32)."""
    t_flat = t.detach().float().flatten()
    return {
        "norm": t_flat.norm().item(),
        "first_10": t_flat[:10].tolist(),
    }


def top10_logits(logits_1d: torch.Tensor) -> list:
    """Get top-10 tokens and their logit values from a 1D logit tensor."""
    logits_f = logits_1d.detach().float()
    values, indices = logits_f.topk(10)
    return [{"token": int(idx), "logit": float(val)} for idx, val in zip(indices, values)]


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load PersonaPlex model
    print("Loading PersonaPlex-7B (BF16)...")
    from moshi.models import loaders
    lm = loaders.get_moshi_lm(
        filename=MODEL_PATH, device=device,
        lm_kwargs_overrides={'dep_q': 16}
    )
    lm.eval()
    print(f"  dep_q={lm.dep_q}, n_q={lm.n_q}, card={lm.card}, text_card={lm.text_card}")
    print(f"  num_codebooks={lm.num_codebooks} (1 text + {lm.n_q} audio)")
    print(f"  Temporal: {len(lm.transformer.layers)} layers, dim={lm.dim}")
    print(f"  Depth: {len(lm.depformer.layers)} layers")
    print(f"  depformer_multi_linear={lm.depformer_multi_linear}")
    print(f"  depformer positional_embedding={lm.depformer.positional_embedding}")

    # Result container
    result = {}

    # Build input: all padding tokens
    # text_start=32000, audio_pad=2048 for all 16 audio streams
    text_start = lm.text_card    # 32000
    audio_pad = lm.card          # 2048
    n_q = lm.n_q                 # 16
    num_codebooks = lm.num_codebooks  # 17

    input_tokens = [text_start] + [audio_pad] * n_q
    input_tensor = torch.tensor(input_tokens, dtype=torch.long, device=device).view(1, num_codebooks, 1)
    print(f"\nInput tensor shape: {input_tensor.shape}")
    print(f"  text token: {text_start}")
    print(f"  audio tokens: all {audio_pad}")

    # ========================================================================
    # PHASE 1: Temporal transformer with hooks
    # ========================================================================
    print("\n=== Temporal Transformer ===")

    temporal_layer_outputs = []

    def make_temporal_hook(layer_idx):
        def hook(module, input, output):
            temporal_layer_outputs.append({
                "layer": layer_idx,
                **tensor_stats(output[0] if isinstance(output, tuple) else output),
            })
        return hook

    # Register hooks on each temporal transformer layer
    hooks = []
    for i, layer in enumerate(lm.transformer.layers):
        h = layer.register_forward_hook(make_temporal_hook(i))
        hooks.append(h)

    with torch.no_grad():
        with lm.streaming(1):
            # --- Compute embedding sum manually to log it ---
            # Replicate the logic from forward_text but capture the embedding sum
            sequence = input_tensor  # [1, 17, 1]

            # Compute embeddings exactly as in forward_text
            input_ = None
            for cb_index in range(lm.n_q):  # 16 audio codebooks
                audio_emb = lm.emb[cb_index](sequence[:, cb_index + lm.audio_offset])
                input_ = audio_emb if input_ is None else input_ + audio_emb
            text_emb = lm.text_emb(sequence[:, 0])
            input_ = text_emb if input_ is None else input_ + text_emb

            # Log embedding sum BEFORE transformer
            result["embedding_sum"] = tensor_stats(input_)
            print(f"  Embedding sum: norm={result['embedding_sum']['norm']:.6f}")
            print(f"    first_10: {result['embedding_sum']['first_10'][:5]}...")

            # Now run the full forward_text which will trigger our hooks
            transformer_out, text_logits = lm.forward_text(input_tensor)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Store temporal layer outputs
    result["temporal_layers"] = temporal_layer_outputs
    for entry in temporal_layer_outputs:
        print(f"  Layer {entry['layer']:2d}: norm={entry['norm']:.6f}, "
              f"first_3={entry['first_10'][:3]}")

    # After out_norm: transformer_out is already post-norm from forward_text
    result["after_out_norm"] = tensor_stats(transformer_out)
    print(f"  After out_norm: norm={result['after_out_norm']['norm']:.6f}")

    # Text logits: shape [1, 1, 1, text_card_out]
    text_logits_1d = text_logits[0, 0, 0]
    result["text_logits_top10"] = top10_logits(text_logits_1d)
    text_token = text_logits_1d.argmax(dim=-1).item()
    result["text_token"] = text_token
    print(f"  Text token (greedy): {text_token}")
    print(f"  Text top-3: {result['text_logits_top10'][:3]}")

    # Temporal hidden state passed to depth transformer
    # transformer_out shape: [1, 1, dim] — this is what gets projected into depformer
    result["temporal_hidden"] = tensor_stats(transformer_out)
    print(f"  Temporal hidden: norm={result['temporal_hidden']['norm']:.6f}")

    # ========================================================================
    # PHASE 2: Depth transformer (16 steps, 6 layers each)
    # ========================================================================
    print("\n=== Depth Transformer ===")

    depth_steps = []
    dep_input_token = torch.tensor([[[text_token]]], dtype=torch.long, device=device)

    with torch.no_grad():
        with lm.streaming(1):
            with lm.depformer.streaming(1):
                for step_k in range(lm.dep_q):
                    step_data = {"step": step_k}

                    # --- Manually compute depth input to log it ---
                    # Replicate logic from forward_depformer
                    if lm.depformer_multi_linear:
                        in_index = step_k
                        if lm.depformer_weights_per_step_schedule is not None:
                            in_index = lm.depformer_weights_per_step_schedule[in_index]
                        depformer_proj = lm.depformer_in[in_index](transformer_out)
                    else:
                        depformer_proj = lm.depformer_in[0](transformer_out)

                    if step_k == 0:
                        token_emb = lm.depformer_text_emb(dep_input_token[:, 0])
                    else:
                        token_emb = lm.depformer_emb[step_k - 1](dep_input_token[:, 0])

                    depth_input = depformer_proj + token_emb
                    step_data["input"] = tensor_stats(depth_input)

                    # --- Register hooks on depth transformer layers ---
                    depth_layer_outputs = []

                    def make_depth_hook(layer_idx):
                        def hook(module, inp, output):
                            depth_layer_outputs.append({
                                "layer": layer_idx,
                                **tensor_stats(output if isinstance(output, torch.Tensor) else output[0]),
                            })
                        return hook

                    depth_hooks = []
                    for i, layer in enumerate(lm.depformer.layers):
                        dh = layer.register_forward_hook(make_depth_hook(i))
                        depth_hooks.append(dh)

                    # --- Run actual forward_depformer ---
                    dep_logits = lm.forward_depformer(step_k, dep_input_token, transformer_out)
                    # dep_logits: [1, 1, 1, card]

                    # Remove depth hooks
                    for dh in depth_hooks:
                        dh.remove()

                    step_data["layers"] = depth_layer_outputs

                    # Output logits
                    dep_logits_1d = dep_logits[0, 0, 0]
                    step_data["logits_top10"] = top10_logits(dep_logits_1d)
                    dep_token = dep_logits_1d.argmax(dim=-1).item()
                    step_data["token"] = dep_token

                    depth_steps.append(step_data)

                    print(f"  Step {step_k:2d}: input_norm={step_data['input']['norm']:.4f}, "
                          f"token={dep_token}, "
                          f"top1_logit={step_data['logits_top10'][0]['logit']:.4f}")

                    # Next step input is this token
                    dep_input_token = torch.tensor([[[dep_token]]], dtype=torch.long, device=device)

    result["depth_steps"] = depth_steps

    # ========================================================================
    # Save output
    # ========================================================================
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {OUTPUT_PATH}")

    # Summary
    print("\n=== Summary ===")
    print(f"  Temporal layers logged: {len(result['temporal_layers'])}")
    print(f"  Text token: {result['text_token']}")
    print(f"  Depth steps: {len(result['depth_steps'])}")
    depth_tokens = [s['token'] for s in result['depth_steps']]
    print(f"  Depth tokens: {depth_tokens}")
    print("\nDone!")


if __name__ == "__main__":
    main()
