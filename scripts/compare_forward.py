#!/usr/bin/env python3
"""Compare a single temporal transformer forward pass between Python BF16 and Rust Q4.

Loads the PersonaPlex BF16 model, runs one forward_text step with known input tokens,
and logs intermediate values (embedding sum, per-layer hiddens, text logits top-10).

Usage:
    python scripts/compare_forward.py [--step N]

Outputs JSON to tests/reference/python_forward_log.json for comparison with Rust.
"""

import json
import sys
import os
import numpy as np

sys.path.insert(0, "refs/moshi/moshi")

import torch
from moshi.models import loaders

MODEL_DIR = "/Users/tc/Code/idle-intelligence/hf/personaplex-7b-v1"
MODEL_PATH = os.path.join(MODEL_DIR, "model.safetensors")
MIMI_PATH = os.path.join(MODEL_DIR, "tokenizer-e351c8d8-checkpoint125.safetensors")
OUTPUT_PATH = "tests/reference/python_forward_log.json"

# PersonaPlex config overrides (dep_q=16, not the default 8)
PERSONAPLEX_LM_KWARGS = dict(loaders._lm_kwargs)
PERSONAPLEX_LM_KWARGS["dep_q"] = 16

# Token constants
TEXT_START_TOKEN = 32000
AUDIO_INITIAL_TOKEN = 2048  # = card
TEXT_PADDING_ID = 3
SILENCE_TOKENS = [948, 243, 1178, 546, 1736, 1030, 1978, 2008]
SINE_TOKENS = [430, 1268, 381, 1611, 1095, 1495, 56, 472]

# System prompt tokens
SYSTEM_PROMPT_TOKENS = [
    607, 4831, 578, 493, 298, 272, 3850, 5019, 263, 17453, 6716, 269, 419, 262,
    819, 1182, 261, 409, 4816, 1312, 269, 347, 560, 307, 2498, 263, 17308, 291,
    3398, 263, 1451, 22876, 263, 607, 4831, 578,
]


def log_tensor(name, t):
    """Extract logging info from a tensor."""
    t_flat = t.float().flatten()
    vals = t_flat.detach().cpu().numpy()
    return {
        "name": name,
        "shape": list(t.shape),
        "norm": float(np.linalg.norm(vals)),
        "first_10": vals[:10].tolist(),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
    }


def main():
    print("Loading PersonaPlex BF16 model...")
    model = loaders.get_moshi_lm(
        MODEL_PATH,
        lm_kwargs=PERSONAPLEX_LM_KWARGS,
        device="cpu",
        dtype=torch.bfloat16,
    )
    model.eval()
    print(f"  Model loaded: n_q={model.n_q}, dep_q={model.dep_q}, "
          f"card={model.card}, text_card={model.text_card}, dim={model.dim}")
    print(f"  num_codebooks={model.num_codebooks}, audio_offset={model.audio_offset}")
    print(f"  initial_token_id={model.initial_token_id}, "
          f"text_initial_token_id={model.text_initial_token_id}")
    print(f"  delays={model.delays}")

    # Build input for step 0: all initial tokens
    # Shape: [1, num_codebooks=17, 1]
    # [text_initial(32000), audio_initial(2048) * 16]
    initial = model._get_initial_token()  # [1, 17, 1]
    print(f"\nInitial tokens: {initial.squeeze().tolist()}")

    # --- Run forward_text for the initial step ---
    print("\n=== Step 0: Initial token (all codebooks = initial) ===")
    logs = []

    with torch.no_grad():
        # Manually replicate forward_text to capture intermediates
        input_sequence = initial  # [1, 17, 1]
        B, K, S = input_sequence.shape

        # Compute embeddings exactly as forward_text does
        input_ = None
        emb_logs = []
        for cb_index in range(model.num_audio_codebooks):
            token = input_sequence[:, cb_index + model.audio_offset, :]  # [1, 1]
            audio_emb = model.emb[cb_index](token)  # [1, 1, dim]
            emb_logs.append({
                "stream": f"audio_{cb_index}",
                "token": int(token.item()),
                **log_tensor(f"emb[{cb_index}]", audio_emb),
            })
            input_ = audio_emb if input_ is None else input_ + audio_emb

        text_token = input_sequence[:, 0, :]  # [1, 1]
        text_emb = model.text_emb(text_token)  # [1, 1, dim]
        emb_logs.append({
            "stream": "text",
            "token": int(text_token.item()),
            **log_tensor("text_emb", text_emb),
        })
        input_ = text_emb if input_ is None else input_ + text_emb

        emb_sum_log = log_tensor("embedding_sum", input_)
        print(f"  Embedding sum: norm={emb_sum_log['norm']:.4f}, "
              f"first_3={emb_sum_log['first_10'][:3]}")

        # Run transformer layers
        layer_logs = []
        x = input_
        for i, layer in enumerate(model.transformer.layers):
            x = layer(x)
            ll = log_tensor(f"layer_{i}", x)
            layer_logs.append(ll)
            if i < 3 or i >= 30:
                print(f"  Layer {i:2d}: norm={ll['norm']:.4f}, "
                      f"first_3={ll['first_10'][:3]}")

        # Output norm
        if model.out_norm:
            x = model.out_norm(x)
        hidden_log = log_tensor("hidden_after_norm", x)
        print(f"  After out_norm: norm={hidden_log['norm']:.4f}, "
              f"first_3={hidden_log['first_10'][:3]}")

        # Text logits
        text_logits = model.text_linear(x)  # [1, 1, text_card]
        tl = text_logits.float().squeeze().detach().cpu().numpy()
        top_indices = np.argsort(tl)[::-1][:10]
        text_logits_top10 = [
            {"token": int(idx), "logit": float(tl[idx])} for idx in top_indices
        ]
        print(f"  Text logits top-3: {text_logits_top10[:3]}")

        step0_log = {
            "step": 0,
            "input_tokens": initial.squeeze().tolist(),
            "embedding_logs": emb_logs,
            "embedding_sum": emb_sum_log,
            "layer_logs": layer_logs,
            "hidden_after_norm": hidden_log,
            "text_logits_top10": text_logits_top10,
        }
        logs.append(step0_log)

    # --- Now run the full prefill sequence and log the first generation step ---
    print("\n=== Running full prefill + first generation step ===")

    with torch.no_grad():
        B_cfg = 1
        # Start streaming
        with model.streaming(B_cfg):
            # Build the prefill sequence: silence spacer + system prompt + silence spacer
            silence_frames = 6  # 0.5s at 12.5Hz
            prefill_text_tokens = (
                [TEXT_PADDING_ID] * silence_frames +
                SYSTEM_PROMPT_TOKENS +
                [TEXT_PADDING_ID] * silence_frames
            )

            # For prefill, all audio codebooks are initial tokens
            num_prefill = len(prefill_text_tokens)
            print(f"  Prefill: {num_prefill} frames "
                  f"({silence_frames} silence + {len(SYSTEM_PROMPT_TOKENS)} prompt + {silence_frames} silence)")

            # Run prefill steps one by one
            for t in range(num_prefill):
                # Build input: [1, 17, 1]
                tokens = torch.full((1, model.num_codebooks, 1),
                                   AUDIO_INITIAL_TOKEN, dtype=torch.long)
                tokens[:, 0, :] = prefill_text_tokens[t]
                # All audio = initial token (2048)

                transformer_out, text_logits = model.forward_text(tokens)

                if t == 0:
                    # Log first prefill step
                    tl = text_logits.float().squeeze().detach().cpu().numpy()
                    top_idx = np.argsort(tl)[::-1][:5]
                    top5_str = ", ".join(f"({int(i)}, {float(tl[i]):.2f})" for i in top_idx)
                    print(f"  Prefill step 0: text_logits_top5=[{top5_str}]")

            print(f"  Prefill done ({num_prefill} steps)")

            # Now load user audio and encode with Mimi
            print("\n  Loading user audio and encoding with Mimi...")
            mimi = loaders.get_mimi(MIMI_PATH, device="cpu", num_codebooks=8)

            import soundfile as sf
            wav_path = "tests/reference/joke.wav"
            audio_data, sr = sf.read(wav_path)
            print(f"  Audio: {len(audio_data)} samples, {sr}Hz, {len(audio_data)/sr:.2f}s")

            audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0).unsqueeze(0)
            with mimi.streaming(1):
                mimi_codes = mimi.encode(audio_tensor)  # [1, 8, T]
            user_frames = mimi_codes.squeeze(0).T.tolist()  # [T, 8]
            print(f"  Encoded {len(user_frames)} Mimi frames")

            # User audio prefill: feed user tokens + run depformer
            print(f"\n  Running user audio prefill ({len(user_frames)} frames)...")
            prev_user_tokens = [AUDIO_INITIAL_TOKEN] * 8
            agent_audio_frames = []

            for t, frame in enumerate(user_frames):
                # Build input with acoustic delay for user audio
                # Text = padding, model audio = initial (or previous depformer output)
                tokens = torch.full((1, model.num_codebooks, 1),
                                   AUDIO_INITIAL_TOKEN, dtype=torch.long)
                tokens[:, 0, :] = TEXT_PADDING_ID

                # User audio with delay:
                # cb0 (semantic, delay=0): current frame
                # cb1-7 (acoustic, delay=1): previous frame
                tokens[:, 1 + 8 + 0, :] = frame[0]  # user semantic = current
                for cb in range(1, 8):
                    tokens[:, 1 + 8 + cb, :] = prev_user_tokens[cb]  # user acoustic = prev

                # Model audio (streams 1-8): use initial token for now
                # (In full pipeline, would use previous depformer output)

                transformer_out, text_logits = model.forward_text(tokens)

                # Run depformer to get agent audio
                text_token_sampled = TEXT_PADDING_ID  # During user speech
                agent_tokens = []
                prev_dep_token = text_token_sampled

                with model.depformer.streaming(B_cfg):
                    for cb_index in range(model.dep_q):
                        input_ = torch.tensor([[prev_dep_token]], dtype=torch.long)  # [1, 1]
                        input_ = input_.unsqueeze(1)  # [1, 1, 1]
                        logits = model.forward_depformer(cb_index, input_, transformer_out)
                        # Greedy sample
                        next_token = logits.float().squeeze().argmax().item()
                        agent_tokens.append(next_token)
                        # Use provided user tokens for steps 8-15
                        if cb_index >= 8 and cb_index - 8 < len(frame):
                            prev_dep_token = frame[cb_index - 8]
                        else:
                            prev_dep_token = next_token

                agent_audio_frames.append(agent_tokens[:8])
                prev_user_tokens = frame

                if t == 0:
                    print(f"    User prefill step 0: agent_audio={agent_tokens[:8]}")

            print(f"  User audio prefill done")

            # --- First generation step with full logging ---
            print("\n=== First generation step (with logging) ===")

            # Input: text=padding, model audio=last depformer agent, user audio=last depformer user pred
            tokens = torch.full((1, model.num_codebooks, 1),
                               AUDIO_INITIAL_TOKEN, dtype=torch.long)
            tokens[:, 0, :] = TEXT_PADDING_ID

            # Model audio (streams 1-8): last agent tokens from user prefill
            last_agent = agent_audio_frames[-1] if agent_audio_frames else [AUDIO_INITIAL_TOKEN] * 8
            for cb in range(8):
                tokens[:, 1 + cb, :] = last_agent[cb]

            # User audio (streams 9-16): last depformer user predictions (steps 8-15)
            last_user_pred = agent_tokens[8:16] if len(agent_tokens) >= 16 else [AUDIO_INITIAL_TOKEN] * 8
            for cb in range(8):
                tokens[:, 1 + 8 + cb, :] = last_user_pred[cb]

            print(f"  Input tokens: text={int(tokens[0,0,0])}, "
                  f"model_audio={[int(tokens[0,1+i,0]) for i in range(8)]}, "
                  f"user_audio={[int(tokens[0,9+i,0]) for i in range(8)]}")

            # Manual forward_text with logging
            input_sequence = tokens
            input_ = None
            gen_emb_logs = []
            for cb_index in range(model.num_audio_codebooks):
                tok = input_sequence[:, cb_index + model.audio_offset, :]
                audio_emb = model.emb[cb_index](tok)
                gen_emb_logs.append({
                    "stream": f"audio_{cb_index}",
                    "token": int(tok.item()),
                    **log_tensor(f"emb[{cb_index}]", audio_emb),
                })
                input_ = audio_emb if input_ is None else input_ + audio_emb

            text_tok = input_sequence[:, 0, :]
            text_emb = model.text_emb(text_tok)
            gen_emb_logs.append({
                "stream": "text",
                "token": int(text_tok.item()),
                **log_tensor("text_emb", text_emb),
            })
            input_ = text_emb if input_ is None else input_ + text_emb

            gen_emb_sum = log_tensor("embedding_sum", input_)
            print(f"  Embedding sum: norm={gen_emb_sum['norm']:.4f}, "
                  f"first_3={gen_emb_sum['first_10'][:3]}")

            # Transformer layers
            gen_layer_logs = []
            x = input_
            for i, layer in enumerate(model.transformer.layers):
                x = layer(x)
                ll = log_tensor(f"layer_{i}", x)
                gen_layer_logs.append(ll)
                if i < 3 or i >= 30:
                    print(f"  Layer {i:2d}: norm={ll['norm']:.4f}, "
                          f"first_3=[{ll['first_10'][0]:.4f}, {ll['first_10'][1]:.4f}, {ll['first_10'][2]:.4f}]")

            if model.out_norm:
                x = model.out_norm(x)
            gen_hidden = log_tensor("hidden_after_norm", x)
            print(f"  After out_norm: norm={gen_hidden['norm']:.4f}")

            text_logits = model.text_linear(x)
            tl = text_logits.float().squeeze().detach().cpu().numpy()
            top_idx = np.argsort(tl)[::-1][:10]
            gen_text_top10 = [
                {"token": int(i), "logit": float(tl[i])} for i in top_idx
            ]
            print(f"  Text logits top-5: {gen_text_top10[:5]}")

            # Depformer step
            text_token_sampled = int(top_idx[0])  # greedy for reproducibility
            print(f"  Sampled text token: {text_token_sampled}")

            transformer_out = x.unsqueeze(0) if x.dim() == 2 else x
            dep_tokens = []
            dep_logs = []
            prev_dep_token = text_token_sampled

            with model.depformer.streaming(B_cfg):
                for cb_index in range(model.dep_q):
                    dep_input = torch.tensor([[prev_dep_token]], dtype=torch.long).unsqueeze(1)
                    logits = model.forward_depformer(cb_index, dep_input, transformer_out)
                    logits_np = logits.float().squeeze().detach().cpu().numpy()
                    top_dep_idx = np.argsort(logits_np)[::-1][:10]
                    dep_top10 = [{"token": int(j), "logit": float(logits_np[j])} for j in top_dep_idx]

                    next_token = int(top_dep_idx[0])  # greedy
                    dep_tokens.append(next_token)
                    dep_logs.append({
                        "step": cb_index,
                        "input_token": prev_dep_token,
                        "logits_top10": dep_top10,
                        "sampled_token": next_token,
                    })
                    prev_dep_token = next_token

                    if cb_index < 3:
                        print(f"  Depformer step {cb_index}: "
                              f"input={dep_logs[-1]['input_token']}, "
                              f"top3={dep_top10[:3]}, sampled={next_token}")

            print(f"\n  Generated agent audio: {dep_tokens[:8]}")
            print(f"  Generated user pred:  {dep_tokens[8:]}")

            gen_step_log = {
                "step": "first_gen",
                "input_tokens": tokens.squeeze().tolist(),
                "embedding_logs": gen_emb_logs,
                "embedding_sum": gen_emb_sum,
                "layer_logs": gen_layer_logs,
                "hidden_after_norm": gen_hidden,
                "text_logits_top10": gen_text_top10,
                "text_token_sampled": text_token_sampled,
                "depformer_logs": dep_logs,
                "depformer_tokens": dep_tokens,
            }
            logs.append(gen_step_log)

    # Save logs
    with open(OUTPUT_PATH, "w") as f:
        json.dump(logs, f, indent=2)
    print(f"\nLogs saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
