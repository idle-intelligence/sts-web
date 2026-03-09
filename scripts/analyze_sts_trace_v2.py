#!/usr/bin/env python3
"""Comprehensive Chrome trace analysis for STS-web WebGPU pipeline.

Analyzes generation frames: temporal transformer, depth transformer,
readback stalls, WASM overhead, and identifies bottlenecks.

Usage: python3 scripts/analyze_sts_trace_v2.py <trace.json> [--frame N] [--verbose]
"""

import json
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FrameMetrics:
    """Metrics for a single generation frame."""
    index: int
    total_ms: float
    wasm_ms: float
    wasm_gpu_overlap_ms: float  # WASM and GPU running concurrently
    temporal_compute_ms: float
    temporal_wall_ms: float
    temporal_dispatches: int
    readback_stall_ms: float  # worker idle waiting for mapAsync
    depth_compute_ms: float
    depth_wall_ms: float
    depth_dispatches: int
    depth_steps: int
    depth_ipc_overhead_ms: float  # depth_wall - depth_compute
    idle_after_depth_ms: float


def load_trace(path):
    with open(path) as f:
        trace = json.load(f)
    return trace.get("traceEvents", trace) if isinstance(trace, dict) else trace


def find_worker_thread(events):
    """Find the DedicatedWorker thread (the one running WASM)."""
    candidates = []
    for e in events:
        if e.get("name") == "thread_name":
            name = e.get("args", {}).get("name", "")
            if "dedicatedworker" in name.lower():
                candidates.append((e["pid"], e["tid"], name))
    if not candidates:
        return None, None
    # If multiple, pick the one with most FunctionCall events
    if len(candidates) == 1:
        return candidates[0][0], candidates[0][1]
    # Count events per candidate
    counts = {}
    for pid, tid, _ in candidates:
        counts[(pid, tid)] = sum(1 for e in events
                                  if e.get("tid") == tid and e.get("pid") == pid
                                  and e.get("name") == "FunctionCall")
    best = max(counts, key=counts.get)
    return best[0], best[1]


def find_generation_calls(events, worker_pid, worker_tid):
    """Find the generation step FunctionCalls on the worker thread."""
    func_calls = sorted(
        [e for e in events if e.get("tid") == worker_tid and e.get("pid") == worker_pid
         and e.get("name") == "FunctionCall" and e.get("ph") == "X"
         and e.get("dur", 0) > 10000],  # >10ms
        key=lambda e: e["ts"]
    )
    # Find prefill (the large initial call, >200ms)
    prefill_end = 0
    for f in func_calls:
        if f["dur"] > 200000:
            prefill_end = f["ts"] + f["dur"]
    # Generation calls come after prefill
    gen_calls = [f for f in func_calls if f["ts"] > prefill_end]
    return gen_calls, prefill_end


def analyze_frame(call, next_call_start, gpu_tasks, call_start_ref):
    """Analyze a single generation frame."""
    call_start = call["ts"]
    call_end = call["ts"] + call["dur"]

    # Get all GPU tasks in the full frame
    frame_gpu = sorted(
        [g for g in gpu_tasks if g["ts"] >= call_start and g["ts"] < next_call_start],
        key=lambda g: g["ts"]
    )
    if len(frame_gpu) < 10:
        return None

    # Find phase boundaries (>5ms gaps between GPU tasks)
    phase_boundaries = []
    for i in range(1, len(frame_gpu)):
        gap = (frame_gpu[i]["ts"] - (frame_gpu[i - 1]["ts"] + frame_gpu[i - 1]["dur"])) / 1000
        if gap > 5:
            phase_boundaries.append(i)

    if len(phase_boundaries) < 1:
        return None

    # Temporal = first phase, Depth = last phase
    temporal = frame_gpu[:phase_boundaries[0]]
    depth = frame_gpu[phase_boundaries[-1]:]

    temporal_end = temporal[-1]["ts"] + temporal[-1]["dur"]
    depth_start_ts = depth[0]["ts"]
    depth_end_ts = depth[-1]["ts"] + depth[-1]["dur"]

    # Compute temporal metrics
    temporal_compute = sum(g["dur"] for g in temporal) / 1000
    temporal_wall = (temporal_end - temporal[0]["ts"]) / 1000

    # Compute depth metrics
    depth_compute = sum(g["dur"] for g in depth) / 1000
    depth_wall = (depth_end_ts - depth_start_ts) / 1000

    # Count depth steps (>1ms gaps within depth)
    depth_steps = 1
    for i in range(1, len(depth)):
        gap = (depth[i]["ts"] - (depth[i - 1]["ts"] + depth[i - 1]["dur"])) / 1000
        if gap > 1.0:
            depth_steps += 1

    # Readback stall: from temporal GPU end to depth GPU start
    readback = (depth_start_ts - temporal_end) / 1000

    # WASM/GPU overlap
    overlap_start = max(call_start, temporal[0]["ts"])
    overlap_end = min(call_end, temporal_end)
    overlap = max(0, (overlap_end - overlap_start)) / 1000

    # Idle after depth
    idle = (next_call_start - depth_end_ts) / 1000

    total = (next_call_start - call_start) / 1000

    return FrameMetrics(
        index=0,
        total_ms=total,
        wasm_ms=call["dur"] / 1000,
        wasm_gpu_overlap_ms=overlap,
        temporal_compute_ms=temporal_compute,
        temporal_wall_ms=temporal_wall,
        temporal_dispatches=len(temporal),
        readback_stall_ms=readback,
        depth_compute_ms=depth_compute,
        depth_wall_ms=depth_wall,
        depth_dispatches=len(depth),
        depth_steps=depth_steps,
        depth_ipc_overhead_ms=depth_wall - depth_compute,
        idle_after_depth_ms=idle,
    )


def print_frame_detail(frame_idx, call, next_start, gpu_tasks, events, worker_pid, worker_tid):
    """Print detailed timeline for a single frame."""
    call_start = call["ts"]
    call_end = call["ts"] + call["dur"]

    frame_gpu = sorted(
        [g for g in gpu_tasks if g["ts"] >= call_start and g["ts"] < next_start],
        key=lambda g: g["ts"]
    )

    phase_boundaries = []
    for i in range(1, len(frame_gpu)):
        gap = (frame_gpu[i]["ts"] - (frame_gpu[i - 1]["ts"] + frame_gpu[i - 1]["dur"])) / 1000
        if gap > 5:
            phase_boundaries.append(i)

    temporal = frame_gpu[:phase_boundaries[0]]
    depth = frame_gpu[phase_boundaries[-1]:]
    temporal_end = temporal[-1]["ts"] + temporal[-1]["dur"]

    print(f"\n=== FRAME {frame_idx} DETAILED TIMELINE ===")
    print(f"t=0.0ms:           WASM starts (command encoding)")
    print(f"t={(temporal[0]['ts'] - call_start)/1000:.1f}ms:          First GPU dispatch (temporal)")
    print(f"t={call['dur']/1000:.1f}ms:          WASM ends")
    print(f"t={(temporal_end - call_start)/1000:.1f}ms:         Temporal GPU ends ({len(temporal)} dispatches)")
    print(f"                   --- READBACK STALL ({(depth[0]['ts'] - temporal_end)/1000:.1f}ms) ---")
    print(f"                   Worker idle: waiting for buffer.mapAsync to resolve")
    print(f"t={(depth[0]['ts'] - call_start)/1000:.1f}ms:         Depth starts")

    # Show depth sub-phases
    depth_sub = []
    start_idx = 0
    for i in range(1, len(depth)):
        gap = (depth[i]["ts"] - (depth[i - 1]["ts"] + depth[i - 1]["dur"])) / 1000
        if gap > 1.0:
            depth_sub.append(depth[start_idx:i])
            start_idx = i
    depth_sub.append(depth[start_idx:])

    for i, sub in enumerate(depth_sub):
        if not sub:
            continue
        t = (sub[0]["ts"] - call_start) / 1000
        compute = sum(g["dur"] for g in sub) / 1000
        gap = ""
        if i > 0 and depth_sub[i - 1]:
            prev_end = depth_sub[i - 1][-1]["ts"] + depth_sub[i - 1][-1]["dur"]
            g = (sub[0]["ts"] - prev_end) / 1000
            gap = f" (gap={g:.1f}ms)"
        print(f"  Step {i:2d}: t={t:.1f}ms  compute={compute:.2f}ms  dispatches={len(sub)}{gap}")

    depth_end = depth[-1]["ts"] + depth[-1]["dur"]
    print(f"t={(depth_end - call_start)/1000:.1f}ms:         Depth ends")
    print(f"t={(next_start - call_start)/1000:.1f}ms:         Next frame starts")


def analyze(path, detail_frame=None, verbose=False):
    print(f"Loading {path}...")
    events = load_trace(path)
    print(f"Total events: {len(events)}")

    worker_pid, worker_tid = find_worker_thread(events)
    if not worker_tid:
        print("ERROR: No DedicatedWorker thread found")
        sys.exit(1)
    print(f"Worker: pid={worker_pid} tid={worker_tid}")

    # Get GPU tasks
    gpu_tasks = sorted(
        [e for e in events if e.get("name") == "GPUTask" and e.get("ph") == "X" and e.get("dur", 0) > 0],
        key=lambda e: e["ts"]
    )
    print(f"GPU tasks: {len(gpu_tasks)}")

    # Find generation calls
    gen_calls, prefill_end = find_generation_calls(events, worker_pid, worker_tid)
    print(f"Generation frames: {len(gen_calls)}")

    if not gen_calls:
        print("No generation frames found!")
        sys.exit(1)

    # Skip first 5 frames (warmup/JIT)
    skip = 5
    frames: List[FrameMetrics] = []
    for i in range(skip, len(gen_calls) - 1):
        m = analyze_frame(gen_calls[i], gen_calls[i + 1]["ts"], gpu_tasks, gen_calls[0]["ts"])
        if m:
            m.index = i
            frames.append(m)

    if not frames:
        print("No valid frames to analyze!")
        sys.exit(1)

    n = len(frames)
    avg = lambda lst: sum(lst) / len(lst)

    # ================================================================
    # Summary statistics
    # ================================================================
    total = avg([f.total_ms for f in frames])
    wasm = avg([f.wasm_ms for f in frames])
    overlap = avg([f.wasm_gpu_overlap_ms for f in frames])
    wasm_exclusive = wasm - overlap

    print(f"\n{'='*60}")
    print(f"  STEADY-STATE ANALYSIS ({n} frames, skipping first {skip})")
    print(f"{'='*60}")
    print(f"\nTOTAL FRAME TIME:  {total:.1f}ms  ({1000/total:.1f} FPS)")
    print(f"Target (80ms):     {'MISS' if total > 80 else 'HIT'} by {abs(total - 80):.0f}ms")

    print(f"\n--- Time Budget ---")
    print(f"{'Component':<30} {'Time':>8} {'% Frame':>8}")
    print(f"{'-'*48}")
    items = [
        ("WASM CPU (exclusive)", wasm_exclusive, wasm_exclusive / total * 100),
        ("WASM/GPU overlap", overlap, overlap / total * 100),
        ("Temporal GPU", avg([f.temporal_wall_ms for f in frames]),
         avg([f.temporal_wall_ms for f in frames]) / total * 100),
        ("Readback stall", avg([f.readback_stall_ms for f in frames]),
         avg([f.readback_stall_ms for f in frames]) / total * 100),
        ("Depth GPU (wall)", avg([f.depth_wall_ms for f in frames]),
         avg([f.depth_wall_ms for f in frames]) / total * 100),
        ("  - compute", avg([f.depth_compute_ms for f in frames]), None),
        ("  - IPC overhead", avg([f.depth_ipc_overhead_ms for f in frames]), None),
        ("Idle", avg([f.idle_after_depth_ms for f in frames]),
         avg([f.idle_after_depth_ms for f in frames]) / total * 100),
    ]
    for name, ms, pct in items:
        if pct is not None:
            print(f"{name:<30} {ms:>7.1f}ms {pct:>7.1f}%")
        else:
            print(f"{name:<30} {ms:>7.1f}ms")

    print(f"\n--- Temporal Transformer ---")
    print(f"GPU compute:  {avg([f.temporal_compute_ms for f in frames]):.1f}ms")
    print(f"Wall time:    {avg([f.temporal_wall_ms for f in frames]):.1f}ms")
    print(f"Utilization:  {avg([f.temporal_compute_ms for f in frames])/avg([f.temporal_wall_ms for f in frames])*100:.1f}%")
    print(f"Dispatches:   {avg([f.temporal_dispatches for f in frames]):.0f}")
    big_dispatch = avg([f.temporal_compute_ms for f in frames]) / 32
    print(f"Per-layer:    {big_dispatch:.1f}ms (32 layers)")

    print(f"\n--- Depth Transformer ---")
    print(f"GPU compute:  {avg([f.depth_compute_ms for f in frames]):.1f}ms")
    print(f"Wall time:    {avg([f.depth_wall_ms for f in frames]):.1f}ms")
    print(f"Utilization:  {avg([f.depth_compute_ms for f in frames])/avg([f.depth_wall_ms for f in frames])*100:.1f}%")
    print(f"Dispatches:   {avg([f.depth_dispatches for f in frames]):.0f}")
    print(f"Steps:        {avg([f.depth_steps for f in frames]):.1f} (EXPECTED: 8)")
    print(f"Per-step:")
    print(f"  Wall:       {avg([f.depth_wall_ms for f in frames])/avg([f.depth_steps for f in frames]):.1f}ms")
    print(f"  Compute:    {avg([f.depth_compute_ms for f in frames])/avg([f.depth_steps for f in frames]):.2f}ms")
    print(f"  IPC/idle:   {avg([f.depth_ipc_overhead_ms for f in frames])/avg([f.depth_steps for f in frames]):.1f}ms")

    # Depth step distribution
    step_counts = Counter(f.depth_steps for f in frames)
    print(f"\n--- Depth Steps Distribution ---")
    for steps, count in sorted(step_counts.items()):
        bar = '#' * count
        print(f"  {steps:2d} steps: {count:3d} frames {bar}")

    # ================================================================
    # Bottleneck analysis
    # ================================================================
    readback_avg = avg([f.readback_stall_ms for f in frames])
    depth_ipc_avg = avg([f.depth_ipc_overhead_ms for f in frames])
    temporal_avg = avg([f.temporal_wall_ms for f in frames])

    print(f"\n{'='*60}")
    print(f"  TOP 3 BOTTLENECKS")
    print(f"{'='*60}")

    bottlenecks = [
        (readback_avg, "READBACK STALL",
         f"Worker thread sits idle for {readback_avg:.0f}ms waiting for buffer.mapAsync.\n"
         f"  This is the temporal->depth handoff: temporal output tokens must be\n"
         f"  read back to CPU before depth transformer can run.\n"
         f"  FIX: GPU-side argmax + embedding lookup. Keep tokens on GPU.\n"
         f"  POTENTIAL GAIN: ~{readback_avg - 2:.0f}ms (replace with ~2ms GPU-side ops)"),
        (depth_ipc_avg, "DEPTH IPC OVERHEAD",
         f"Each depth step needs a CPU round-trip (submit -> GPU -> readback -> submit).\n"
         f"  {avg([f.depth_steps for f in frames]):.0f} steps x {depth_ipc_avg/avg([f.depth_steps for f in frames]):.1f}ms IPC = "
         f"{depth_ipc_avg:.0f}ms overhead for only {avg([f.depth_compute_ms for f in frames]):.0f}ms compute.\n"
         f"  Also: running {avg([f.depth_steps for f in frames]):.0f} steps instead of 8.\n"
         f"  FIX 1: Fix depth step count to 8 (save {(avg([f.depth_steps for f in frames]) - 8) * depth_ipc_avg/avg([f.depth_steps for f in frames]):.0f}ms)\n"
         f"  FIX 2: Batch depth work to reduce round-trips\n"
         f"  POTENTIAL GAIN: ~{depth_ipc_avg * 0.6:.0f}ms"),
        (temporal_avg, "TEMPORAL GPU COMPUTE",
         f"32-layer temporal transformer: {temporal_avg:.0f}ms, {big_dispatch:.1f}ms/layer.\n"
         f"  Already 95%+ GPU-utilized. This is the Q4K matmul cost.\n"
         f"  FIX: Optimize Q4K kernel, or model reduction (layer pruning).\n"
         f"  POTENTIAL GAIN: 10-30% = {temporal_avg * 0.2:.0f}ms with kernel work"),
    ]

    for rank, (ms, name, desc) in enumerate(sorted(bottlenecks, key=lambda x: -x[0]), 1):
        print(f"\n{rank}. {name}: {ms:.0f}ms ({ms/total*100:.0f}% of frame)")
        print(f"  {desc}")

    # ================================================================
    # Sequence diagram
    # ================================================================
    print(f"\n{'='*60}")
    print(f"  FRAME SEQUENCE DIAGRAM")
    print(f"{'='*60}")
    print(f"""
Worker:  |==WASM({wasm:.0f}ms)==|.....idle({readback_avg:.0f}ms).....|=D=|=D=|...|=D=| next
         |  cmd encode  |  waiting mapAsync  |depth steps     |
         |              |                     |1.6ms each      |
         |              |                     |~{avg([f.depth_steps for f in frames]):.0f} steps       |
GPU:     |    ...submit.|===TEMPORAL({temporal_avg:.0f}ms)===|...|=d=|..|=d=|..|=d=|
         |              |  32 layers x {big_dispatch:.1f}ms  |   |depth GPU    |
         |              |  {avg([f.temporal_dispatches for f in frames]):.0f} dispatches      |   |{avg([f.depth_compute_ms for f in frames]):.0f}ms total  |
         |              |  95% utilized       |   |14% utilized |
         0         {wasm:.0f}         {wasm+temporal_avg:.0f}      {wasm+temporal_avg+readback_avg:.0f}       {total:.0f}ms

Critical path: WASM -> Temporal -> Readback -> Depth = {total:.0f}ms
Target: 80ms (real-time audio)
""")

    # ================================================================
    # Detailed frame view
    # ================================================================
    if detail_frame is not None and detail_frame < len(gen_calls) - 1:
        print_frame_detail(detail_frame, gen_calls[detail_frame],
                           gen_calls[detail_frame + 1]["ts"],
                           gpu_tasks, events, worker_pid, worker_tid)

    # Percentile analysis
    if verbose:
        print(f"\n{'='*60}")
        print(f"  PERCENTILE ANALYSIS")
        print(f"{'='*60}")
        totals = sorted([f.total_ms for f in frames])
        for p in [10, 25, 50, 75, 90, 95]:
            idx = int(len(totals) * p / 100)
            print(f"  P{p}: {totals[min(idx, len(totals)-1)]:.1f}ms")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze STS-web Chrome trace")
    parser.add_argument("trace", help="Path to Chrome trace JSON file")
    parser.add_argument("--frame", type=int, default=10, help="Frame index for detailed view")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    analyze(args.trace, detail_frame=args.frame, verbose=args.verbose)
