# Next Steps

Loose backlog. Order by interest, not priority.

## Audio quality (README's main caveat)
- A/B depth-transformer sampling: top-k=25 + temperature=0.8 (current, set in `0cbe285`) vs deterministic argmax (the regression mentioned in `59ed1b8`).
- 24L vs 7b-v1 on identical input + voice — how much of the artifact budget is pruning?
- Mimi roundtrip is clean (`mimi_roundtrip_test`), so the issue is upstream of the codec.

## Prefill latency
- Bench on RTX 3080: prefill = 3.4 s for system prompt + 2.8 s of user audio; single-step generation = 63 ms (0.8× realtime). Prefill is the dominant time-to-first-frame cost.
- Profile cache-build vs transformer-step share inside prefill.
- System-prompt cache could be persisted across runs.

## CLI / UX
- `--max-frames` default of 250 truncated roughly half the voices on `joke.wav` mid-sentence. Raise the default, or stop on natural punctuation / EOS.
- Repetition penalty audit: NATM1 looped "Hey, let me know if you have any questions." 4×.
- `--list-voices` flag.

## Browser / native parity
- All current bench numbers are native (Vulkan on RTX 3080). Re-measure WASM/WebGPU on the same machine to size the gap and decide whether the prefill / generation split looks the same in-browser.
