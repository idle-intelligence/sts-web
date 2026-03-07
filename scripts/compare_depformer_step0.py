#!/usr/bin/env python3
"""Compare step-0 depformer output between Python BF16 and Rust Q4.

Runs the initial token step (all audio = 2048, text = 32000) through both
the temporal transformer and the depformer, logging per-step depformer
intermediates for comparison against tests/reference/q4_layer_log.json.

Usage:
    NO_TORCH_COMPILE=1 .venv/bin/python scripts/compare_depformer_step0.py
"""

import json
import sys
import os
import numpy as np

sys.path.insert(0, "refs/moshi/moshi")

import torch
from moshi.models import loaders

MODEL_PATH = "/Users/tc/Code/idle-intelligence/hf/personaplex-7b-v1/model.safetensors"
RUST_LOG_PATH = "tests/reference/q4_layer_log.json"

PERSONAPLEX_LM_KWARGS = dict(loaders._lm_kwargs)
PERSONAPLEX_LM_KWARGS["dep_q"] = 16

TEXT_START_TOKEN = 32000
AUDIO_INITIAL_TOKEN = 2048


def log_tensor(t):
    v = t.float().flatten().detach().cpu().numpy()
    return {
        "norm": float(np.linalg.norm(v)),
        "first_10": v[:10].tolist(),
    }


def main():
    # Load Rust Q4 log
    with open(RUST_LOG_PATH) as f:
        rust = json.load(f)

    print("Loading PersonaPlex BF16 model...")
    model = loaders.get_moshi_lm(
        MODEL_PATH,
        lm_kwargs=PERSONAPLEX_LM_KWARGS,
        device="cpu",
        dtype=torch.bfloat16,
    )
    model.eval()
    print(f"  dep_q={model.dep_q}, card={model.card}, dim={model.dim}")
    print(f"  depformer_multi_linear={model.depformer_multi_linear}")
    print(f"  num depformer_in: {len(model.depformer_in)}")
    print(f"  num depformer_emb: {len(model.depformer_emb)}")
    print(f"  num linears: {len(model.linears)}")

    # Build initial tokens: [1, 17, 1]
    initial = model._get_initial_token()  # [1, 17, 1]
    print(f"\nInitial tokens: {initial.squeeze().tolist()}")

    with torch.no_grad():
        # Run temporal forward_text (with streaming for KV cache)
        with model.streaming(1):
            transformer_out, text_logits = model.forward_text(initial)

        # Sample text token (greedy)
        text_token_val = text_logits.float().squeeze().argmax().item()
        print(f"Text token (greedy): {text_token_val}")
        print(f"Rust text token:     {rust['text_token']}")

        # Compare temporal output
        py_norm = log_tensor(transformer_out)
        print(f"\nTemporal hidden after norm:")
        print(f"  PY norm={py_norm['norm']:.4f}  first_3={py_norm['first_10'][:3]}")
        print(f"  RS norm={rust['after_out_norm']['norm']:.4f}  first_3={rust['after_out_norm']['first_10'][:3]}")

        # Run depformer for all 16 steps
        print(f"\n{'='*60}")
        print(f"DEPFORMER STEP-BY-STEP COMPARISON")
        print(f"{'='*60}")

        text_token = torch.tensor([text_token_val], dtype=torch.long)
        py_tokens = []

        with model.depformer.streaming(1):
            prev_token = text_token_val

            for cb_index in range(model.dep_q):
                # Manual depformer forward to capture intermediates:
                # 1. Input projection (depformer_in is on the LMModel, not the depformer)
                if model.depformer_multi_linear:
                    projected = model.depformer_in[cb_index](transformer_out)
                else:
                    projected = model.depformer_in[0](transformer_out)
                proj_log = log_tensor(projected)

                # 2. Token embedding
                if cb_index == 0:
                    token_emb = model.depformer_text_emb(
                        torch.tensor([[prev_token]], dtype=torch.long))
                else:
                    token_emb = model.depformer_emb[cb_index - 1](
                        torch.tensor([[prev_token]], dtype=torch.long))
                emb_log = log_tensor(token_emb)

                # 3. Sum
                x = projected + token_emb
                input_log = log_tensor(x)

                # Get Rust input log for this step
                rs_step = rust['depth_steps'][cb_index]
                rs_input_norm = rs_step['input']['norm']
                rs_input_f10 = rs_step['input']['first_10'][:3]

                # 4. Run through depformer transformer (streaming will use KV cache)
                dep_output = model.depformer(x)
                dep_log = log_tensor(dep_output)

                # 5. Apply depformer_norms (Identity by default) and output head
                normed = model.depformer_norms[cb_index](dep_output)
                logits = model.linears[cb_index](normed)
                logits_np = logits.float().squeeze().detach().cpu().numpy()
                top_idx = np.argsort(logits_np)[::-1][:5]
                py_top5 = [(int(i), f"{float(logits_np[i]):.2f}") for i in top_idx]

                rs_top5 = [(t['token'], f"{t['logit']:.2f}") for t in rs_step['logits_top10'][:5]]

                next_token = int(top_idx[0])  # greedy
                py_tokens.append(next_token)

                # Compare per-layer norms
                # Note: dep_output is after all layers; Rust logs per-layer
                # We can only compare the final output vs last Rust layer
                rs_last_layer = rs_step['layers'][-1]

                # Print comparison
                input_rel = abs(input_log['norm'] - rs_input_norm) / max(abs(input_log['norm']), 1e-6) * 100
                dep_rel = abs(dep_log['norm'] - rs_last_layer['norm']) / max(abs(dep_log['norm']), 1e-6) * 100

                print(f"\nStep {cb_index:2d}: input_token={prev_token}")
                print(f"  Input: PY norm={input_log['norm']:.4f}  RS norm={rs_input_norm:.4f}  rel={input_rel:.1f}%")
                print(f"         PY first_3={[f'{v:.4f}' for v in input_log['first_10'][:3]]}")
                print(f"         RS first_3={[f'{v:.4f}' for v in rs_input_f10]}")
                print(f"  After layers: PY norm={dep_log['norm']:.4f}  RS L5 norm={rs_last_layer['norm']:.4f}  rel={dep_rel:.1f}%")
                print(f"  Logits top5:")
                print(f"    PY: {py_top5}")
                print(f"    RS: {rs_top5}")
                print(f"  Token: PY={next_token}  RS={rs_step['logits_top10'][0]['token']}")

                prev_token = next_token

        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"PY tokens: {py_tokens}")
        rs_tokens = [s['logits_top10'][0]['token'] for s in rust['depth_steps']]
        print(f"RS tokens: {rs_tokens}")

        match_count = sum(1 for a, b in zip(py_tokens, rs_tokens) if a == b)
        print(f"Match: {match_count}/{len(py_tokens)}")


if __name__ == "__main__":
    main()
