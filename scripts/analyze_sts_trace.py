#!/usr/bin/env python3
"""Detailed Chrome trace analysis for STS-web WebGPU pipeline.

Analyzes GPU dispatches, WASM overhead, readback stalls, warmup vs steady-state.
"""

import json
import sys
from collections import defaultdict


def load_trace(path):
    with open(path) as f:
        trace = json.load(f)
    return trace.get("traceEvents", trace) if isinstance(trace, dict) else trace


def find_threads(events):
    """Find worker thread, GPU process/thread, and CrGpuMain."""
    threads = {}
    for e in events:
        if e.get("name") == "thread_name":
            name = e.get("args", {}).get("name", "")
            tid = e["tid"]
            pid = e["pid"]
            if "worker" in name.lower() and "dedicatedworker" in name.lower():
                threads["worker"] = (pid, tid, name)
            elif name == "CrGpuMain":
                threads["gpu_main"] = (pid, tid, name)
            elif name == "CrRendererMain":
                threads["renderer"] = (pid, tid, name)
    return threads


def analyze(path):
    print(f"Loading {path}...")
    events = load_trace(path)
    print(f"Total events: {len(events)}")

    threads = find_threads(events)
    for k, v in threads.items():
        print(f"  {k}: pid={v[0]} tid={v[1]} ({v[2]})")

    worker_pid, worker_tid, _ = threads.get("worker", (None, None, None))
    if not worker_tid:
        print("ERROR: no worker thread found")
        # Try to find any DedicatedWorker
        for e in events:
            if e.get("name") == "thread_name":
                name = e.get("args", {}).get("name", "")
                if "worker" in name.lower():
                    print(f"  candidate: tid={e['tid']} name={name}")
        sys.exit(1)

    # ============================================================
    # 1. Categorize all events by type
    # ============================================================

    # GPU tasks (on GPU process)
    gpu_tasks = sorted(
        [e for e in events if e.get("name") == "GPUTask" and e.get("ph") == "X" and e.get("dur", 0) > 0],
        key=lambda e: e["ts"]
    )
    print(f"\nGPU tasks: {len(gpu_tasks)}")
    if gpu_tasks:
        gpu_total_ms = sum(e["dur"] for e in gpu_tasks) / 1000
        print(f"  Total GPU time: {gpu_total_ms:.0f}ms")
        print(f"  Avg GPU task: {gpu_total_ms/len(gpu_tasks):.2f}ms")

    # Worker events
    worker_events = [e for e in events if e.get("tid") == worker_tid and e.get("pid") == worker_pid]
    print(f"Worker events: {len(worker_events)}")

    # Find RunMicrotasks and RunTask on worker
    worker_tasks = sorted(
        [e for e in worker_events if e.get("name") in ("RunMicrotasks", "RunTask", "FunctionCall")
         and e.get("ph") == "X" and e.get("dur", 0) > 1000],
        key=lambda e: e["ts"]
    )

    # ============================================================
    # 2. Look for WebGPU-specific events
    # ============================================================
    webgpu_events = [e for e in events if "webgpu" in str(e.get("name", "")).lower()
                     or "dawn" in str(e.get("name", "")).lower()
                     or "wgpu" in str(e.get("name", "")).lower()]
    print(f"WebGPU/Dawn events: {len(webgpu_events)}")
    if webgpu_events:
        names = defaultdict(int)
        for e in webgpu_events:
            names[e.get("name", "?")] += 1
        for n, c in sorted(names.items(), key=lambda x: -x[1])[:20]:
            print(f"  {n}: {c}")

    # ============================================================
    # 3. Find GPU readback / buffer mapping events
    # ============================================================
    readback_events = [e for e in events
                       if any(kw in str(e.get("name", "")).lower()
                              for kw in ["mapasync", "mapbuffer", "readback", "getmappedrange",
                                         "mapbufferasync", "buffer.map"])]
    print(f"\nBuffer map/readback events: {len(readback_events)}")
    if readback_events:
        names = defaultdict(int)
        for e in readback_events:
            names[e.get("name", "?")] += 1
        for n, c in sorted(names.items(), key=lambda x: -x[1])[:10]:
            print(f"  {n}: {c}")

    # ============================================================
    # 4. Analyze steady-state generation frames
    # ============================================================
    # Look for worker RunMicrotasks bursts > 10ms
    bursts = sorted(
        [e for e in worker_events if e.get("name") == "RunMicrotasks"
         and e.get("ph") == "X" and e.get("dur", 0) > 10000],
        key=lambda e: e["ts"]
    )
    print(f"\nWorker bursts (>10ms): {len(bursts)}")

    # Identify steady-state frames (bursts 17-163 look steady from initial analysis)
    # Find runs of similar-duration bursts
    steady_bursts = [b for b in bursts if 30000 < b.get("dur", 0) < 60000]
    print(f"Steady-state bursts (30-60ms): {len(steady_bursts)}")

    if len(steady_bursts) > 5:
        # Analyze a representative steady-state frame
        print("\n=== STEADY-STATE FRAME ANALYSIS ===")

        # For each steady burst, find what happens in the gap before it
        frame_data = []
        for i in range(1, len(steady_bursts)):
            prev = steady_bursts[i - 1]
            curr = steady_bursts[i]

            gap_start = prev["ts"] + prev["dur"]
            gap_end = curr["ts"]
            gap_ms = (gap_end - gap_start) / 1000

            wasm_ms = curr["dur"] / 1000

            # GPU tasks in gap
            gpu_in_gap = [g for g in gpu_tasks
                          if g["ts"] >= gap_start and g["ts"] + g.get("dur", 0) <= gap_end]
            gpu_ms = sum(g["dur"] / 1000 for g in gpu_in_gap)

            # GPU tasks during WASM burst
            gpu_during_wasm = [g for g in gpu_tasks
                               if g["ts"] >= curr["ts"] and g["ts"] + g.get("dur", 0) <= curr["ts"] + curr["dur"]]
            gpu_during_wasm_ms = sum(g["dur"] / 1000 for g in gpu_during_wasm)

            frame_data.append({
                "wasm_ms": wasm_ms,
                "gap_ms": gap_ms,
                "gpu_in_gap_ms": gpu_ms,
                "gpu_in_gap_count": len(gpu_in_gap),
                "gpu_during_wasm_ms": gpu_during_wasm_ms,
                "gpu_during_wasm_count": len(gpu_during_wasm),
                "cycle_ms": wasm_ms + gap_ms,
                "idle_ms": gap_ms - gpu_ms,
            })

        n = len(frame_data)
        avg = lambda key: sum(f[key] for f in frame_data) / n

        print(f"  Frames analyzed: {n}")
        print(f"  Cycle time:        {avg('cycle_ms'):.1f}ms")
        print(f"  WASM work:         {avg('wasm_ms'):.1f}ms")
        print(f"  Gap (waiting):     {avg('gap_ms'):.1f}ms")
        print(f"  GPU in gap:        {avg('gpu_in_gap_ms'):.1f}ms ({avg('gpu_in_gap_count'):.0f} dispatches)")
        print(f"  GPU during WASM:   {avg('gpu_during_wasm_ms'):.1f}ms ({avg('gpu_during_wasm_count'):.0f} dispatches)")
        print(f"  GPU idle in gap:   {avg('idle_ms'):.1f}ms")
        print(f"  GPU % of cycle:    {(avg('gpu_in_gap_ms') + avg('gpu_during_wasm_ms')) / avg('cycle_ms') * 100:.1f}%")
        print(f"  WASM % of cycle:   {avg('wasm_ms') / avg('cycle_ms') * 100:.1f}%")
        print(f"  Idle % of cycle:   {avg('idle_ms') / avg('cycle_ms') * 100:.1f}%")

    # ============================================================
    # 5. Warmup vs steady-state
    # ============================================================
    print("\n=== WARMUP vs STEADY-STATE ===")
    if len(bursts) > 15:
        warmup = bursts[:10]
        steady = bursts[16:]  # Skip transition

        print("Warmup (first 10 bursts):")
        for i, b in enumerate(warmup):
            print(f"  {i}: {b['dur']/1000:.1f}ms")

        print(f"\nSteady (bursts 16+, n={len(steady)}):")
        durs = [b["dur"] / 1000 for b in steady]
        print(f"  avg: {sum(durs)/len(durs):.1f}ms  min: {min(durs):.1f}ms  max: {max(durs):.1f}ms")

    # ============================================================
    # 6. GPU dispatch pattern within a single frame
    # ============================================================
    print("\n=== GPU DISPATCH PATTERN (sample frame) ===")
    if len(steady_bursts) > 20:
        # Pick frame 20 as representative
        prev = steady_bursts[19]
        curr = steady_bursts[20]
        gap_start = prev["ts"] + prev["dur"]
        gap_end = curr["ts"]

        gpu_in_frame = sorted(
            [g for g in gpu_tasks if g["ts"] >= gap_start and g["ts"] + g.get("dur", 0) <= gap_end],
            key=lambda g: g["ts"]
        )

        print(f"  Frame gap: {(gap_end - gap_start)/1000:.1f}ms, GPU tasks: {len(gpu_in_frame)}")
        if gpu_in_frame:
            # Group by duration to find patterns
            dur_buckets = defaultdict(int)
            for g in gpu_in_frame:
                dur_us = g["dur"]
                if dur_us < 100:
                    dur_buckets["<0.1ms"] += 1
                elif dur_us < 500:
                    dur_buckets["0.1-0.5ms"] += 1
                elif dur_us < 1000:
                    dur_buckets["0.5-1ms"] += 1
                elif dur_us < 5000:
                    dur_buckets["1-5ms"] += 1
                elif dur_us < 10000:
                    dur_buckets["5-10ms"] += 1
                else:
                    dur_buckets[">10ms"] += 1

            print("  Duration distribution:")
            for bucket in ["<0.1ms", "0.1-0.5ms", "0.5-1ms", "1-5ms", "5-10ms", ">10ms"]:
                if bucket in dur_buckets:
                    print(f"    {bucket}: {dur_buckets[bucket]}")

            # Show gaps between GPU tasks
            gpu_gaps = [(gpu_in_frame[i]["ts"] - (gpu_in_frame[i-1]["ts"] + gpu_in_frame[i-1]["dur"])) / 1000
                        for i in range(1, len(gpu_in_frame))]
            if gpu_gaps:
                print(f"  Inter-GPU-task gaps: avg={sum(gpu_gaps)/len(gpu_gaps):.2f}ms min={min(gpu_gaps):.2f}ms max={max(gpu_gaps):.2f}ms")

            # Show first and last few
            print("  First 5 GPU tasks (relative to gap start):")
            for i, g in enumerate(gpu_in_frame[:5]):
                offset = (g["ts"] - gap_start) / 1000
                print(f"    +{offset:.1f}ms: {g['dur']/1000:.2f}ms")
            print("  Last 5 GPU tasks:")
            for g in gpu_in_frame[-5:]:
                offset = (g["ts"] - gap_start) / 1000
                print(f"    +{offset:.1f}ms: {g['dur']/1000:.2f}ms")

    # ============================================================
    # 7. Look for long events (potential Mimi decode)
    # ============================================================
    print("\n=== LONG WORKER EVENTS (potential Mimi) ===")
    # Look at all worker events that are long
    long_worker = sorted(
        [e for e in worker_events if e.get("ph") == "X" and e.get("dur", 0) > 5000
         and e.get("name") not in ("RunMicrotasks",)],
        key=lambda e: -e.get("dur", 0)
    )
    names = defaultdict(lambda: {"count": 0, "total_us": 0})
    for e in long_worker:
        n = e.get("name", "?")
        names[n]["count"] += 1
        names[n]["total_us"] += e.get("dur", 0)
    print(f"Long worker events (>5ms, excl RunMicrotasks): {len(long_worker)}")
    for n, d in sorted(names.items(), key=lambda x: -x[1]["total_us"])[:15]:
        print(f"  {n}: {d['count']}x, total={d['total_us']/1000:.0f}ms, avg={d['total_us']/d['count']/1000:.1f}ms")

    # ============================================================
    # 8. Find all unique event names on worker thread
    # ============================================================
    print("\n=== WORKER EVENT CATEGORIES ===")
    worker_cats = defaultdict(lambda: {"count": 0, "total_us": 0})
    for e in worker_events:
        if e.get("ph") == "X" and e.get("dur", 0) > 0:
            n = e.get("name", "?")
            worker_cats[n]["count"] += 1
            worker_cats[n]["total_us"] += e.get("dur", 0)
    for n, d in sorted(worker_cats.items(), key=lambda x: -x[1]["total_us"])[:25]:
        print(f"  {n}: {d['count']}x, total={d['total_us']/1000:.0f}ms, avg={d['total_us']/d['count']/1000:.2f}ms")

    # ============================================================
    # 9. Identify the big 9650ms and 9311ms bursts
    # ============================================================
    print("\n=== LARGE BURSTS ANALYSIS ===")
    big_bursts = [b for b in bursts if b["dur"] > 1_000_000]  # > 1s
    for i, b in enumerate(big_bursts):
        ts_s = (b["ts"] - bursts[0]["ts"]) / 1_000_000
        dur_ms = b["dur"] / 1000
        print(f"  Burst at t={ts_s:.1f}s: {dur_ms:.0f}ms")

        # What's happening inside this burst?
        inner = [e for e in worker_events
                 if e.get("ph") == "X" and e.get("dur", 0) > 100
                 and e["ts"] >= b["ts"] and e["ts"] + e.get("dur", 0) <= b["ts"] + b["dur"]
                 and e.get("name") != "RunMicrotasks"]
        inner_names = defaultdict(lambda: {"count": 0, "total_us": 0})
        for e in inner:
            n = e.get("name", "?")
            inner_names[n]["count"] += 1
            inner_names[n]["total_us"] += e.get("dur", 0)
        for n, d in sorted(inner_names.items(), key=lambda x: -x[1]["total_us"])[:10]:
            print(f"    {n}: {d['count']}x, total={d['total_us']/1000:.0f}ms")

    # ============================================================
    # 10. Overall time breakdown
    # ============================================================
    print("\n=== OVERALL TIME BREAKDOWN ===")
    if bursts:
        total_span = (bursts[-1]["ts"] + bursts[-1]["dur"] - bursts[0]["ts"]) / 1000
        total_wasm = sum(b["dur"] for b in bursts) / 1000
        total_gpu = sum(g["dur"] for g in gpu_tasks) / 1000

        # GPU during bursts
        gpu_during_bursts = 0
        for b in bursts:
            for g in gpu_tasks:
                if g["ts"] >= b["ts"] and g["ts"] + g.get("dur", 0) <= b["ts"] + b["dur"]:
                    gpu_during_bursts += g["dur"]
        gpu_during_bursts /= 1000

        print(f"  Total span: {total_span:.0f}ms ({total_span/1000:.1f}s)")
        print(f"  WASM work: {total_wasm:.0f}ms ({total_wasm/total_span*100:.1f}%)")
        print(f"  GPU total: {total_gpu:.0f}ms ({total_gpu/total_span*100:.1f}%)")
        print(f"  GPU during WASM: {gpu_during_bursts:.0f}ms")
        print(f"  GPU in gaps: {total_gpu - gpu_during_bursts:.0f}ms")
        print(f"  Neither (idle): {total_span - total_wasm - (total_gpu - gpu_during_bursts):.0f}ms ({(total_span - total_wasm - (total_gpu - gpu_during_bursts))/total_span*100:.1f}%)")

    # ============================================================
    # 11. Prefill vs Generation phase identification
    # ============================================================
    print("\n=== PHASE IDENTIFICATION ===")
    # The initial bursts (0-9) are warmup, bursts 10-14 have big ones (prefill?),
    # then 15+ are steady generation
    if len(bursts) > 15:
        # Look for phase boundaries based on burst patterns
        for i, b in enumerate(bursts[:20]):
            dur_ms = b["dur"] / 1000
            ts_s = (b["ts"] - bursts[0]["ts"]) / 1_000_000
            marker = " <-- LARGE" if dur_ms > 1000 else ""
            print(f"  Burst {i:3d}: t={ts_s:6.1f}s  dur={dur_ms:8.1f}ms{marker}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <trace.json>")
        sys.exit(1)
    analyze(sys.argv[1])
