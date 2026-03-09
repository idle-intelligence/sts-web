// Fused single-token attention kernel (Flash-decoding style).
//
// For single-token generation:
//   Q: [num_heads, head_dim]       (one query token, batch=1)
//   K: [num_heads, S, head_dim]    (KV cache)
//   V: [num_heads, S, head_dim]    (KV cache)
//   Output: [num_heads, head_dim]
//
// Dispatch: num_heads workgroups, 256 threads each.
// One workgroup per head. 256 threads cooperate on the S dimension.
//
// Algorithm: Online softmax with 2-phase V reduction.
//   Phase 1: Each thread processes ceil(S/256) KV positions with online softmax.
//   Phase 2: Parallel reduction of (max, sum) across threads.
//   Phase 3: V reduction — 256 threads reduce per-dim partial sums.
//            With head_dim <= 128 and 256 threads, we assign 2 threads per dim,
//            then tree-reduce pairwise. Only 2 barriers needed.
//
// Bindings:
//   @group(0) @binding(0) q:      f32 [num_heads * head_dim]
//   @group(0) @binding(1) k:      f32 [num_heads * seq_len * head_dim]
//   @group(0) @binding(2) v:      f32 [num_heads * seq_len * head_dim]
//   @group(0) @binding(3) output: f32 [num_heads * head_dim]
//   @group(0) @binding(4) info:   u32 [4]: {num_heads, head_dim, seq_len, scale_bits}

@group(0) @binding(0) var<storage, read> q_buf: array<f32>;
@group(0) @binding(1) var<storage, read> k_buf: array<f32>;
@group(0) @binding(2) var<storage, read> v_buf: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_buf: array<f32>;
@group(0) @binding(4) var<storage, read> info: array<u32>;

const WG: u32 = 256u;
const MAX_HD: u32 = 128u;

// Shared memory for reductions.
var<workgroup> sh_max: array<f32, 256>;
var<workgroup> sh_sum: array<f32, 256>;
// For V reduction: 256 * 128 is too large for workgroup memory.
// Instead we use a chunked approach with a 256-slot shared array.
var<workgroup> sh_v: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let head = wg_id.x;
    let tid = lid.x;

    let head_dim = info[1];
    let seq_len = info[2];
    let scale = bitcast<f32>(info[3]);

    let q_off = head * head_dim;
    let kv_off = head * seq_len * head_dim;

    // =================================================================
    // Phase 1: Online softmax over assigned KV positions.
    // =================================================================

    // Load Q into registers
    var q_reg: array<f32, 128>;
    for (var d = 0u; d < head_dim; d++) {
        q_reg[d] = q_buf[q_off + d];
    }

    var p_max: f32 = -3.402823e+38f;
    var p_sum: f32 = 0.0f;
    var p_v: array<f32, 128>;
    for (var d = 0u; d < MAX_HD; d++) {
        p_v[d] = 0.0f;
    }

    var pos = tid;
    while (pos < seq_len) {
        var dot: f32 = 0.0f;
        let k_off = kv_off + pos * head_dim;
        for (var d = 0u; d < head_dim; d++) {
            dot += q_reg[d] * k_buf[k_off + d];
        }
        dot *= scale;

        let old_max = p_max;
        p_max = max(p_max, dot);
        let alpha = exp(old_max - p_max);
        let w = exp(dot - p_max);

        p_sum = p_sum * alpha + w;

        let v_off = kv_off + pos * head_dim;
        for (var d = 0u; d < head_dim; d++) {
            p_v[d] = p_v[d] * alpha + w * v_buf[v_off + d];
        }

        pos += WG;
    }

    // =================================================================
    // Phase 2: Cross-thread reduction of online-softmax (max, sum).
    // =================================================================

    sh_max[tid] = p_max;
    sh_sum[tid] = p_sum;
    workgroupBarrier();

    for (var stride = WG / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            let other = tid + stride;
            let m1 = sh_max[tid];
            let m2 = sh_max[other];
            let s1 = sh_sum[tid];
            let s2 = sh_sum[other];
            let new_max = max(m1, m2);
            let a1 = exp(m1 - new_max);
            let a2 = exp(m2 - new_max);
            sh_max[tid] = new_max;
            sh_sum[tid] = s1 * a1 + s2 * a2;
        }
        workgroupBarrier();
    }

    let global_max = sh_max[0];
    let global_sum = sh_sum[0];

    // =================================================================
    // Phase 3: Correct V to global_max, then reduce across threads.
    //
    // We process head_dim dims in chunks of (WG/REDUCE_FACTOR) dims.
    // For each chunk, threads map to dims within the chunk.
    // With REDUCE_FACTOR=2 (256 threads, 128 dims), each dim gets
    // exactly 2 threads. We sum the pair, then thread 0 of the pair writes.
    //
    // For head_dim=64 (depth transformer), each dim gets 4 threads,
    // requiring a 2-level tree reduction.
    //
    // General approach: process head_dim dims per chunk. Each dim gets
    // (WG / head_dim) threads. Tree-reduce those threads.
    // =================================================================

    let correction = exp(p_max - global_max);
    for (var d = 0u; d < head_dim; d++) {
        p_v[d] *= correction;
    }

    let out_off = head * head_dim;
    let threads_per_dim = WG / head_dim;  // 2 for hd=128, 4 for hd=64

    // Each thread maps to a dim: dim_idx = tid % head_dim
    // And a "slot" within that dim: slot = tid / head_dim
    let dim_idx = tid % head_dim;
    let slot = tid / head_dim;

    // Each thread sums all its p_v values, but reassigned by dim_idx.
    // Thread tid has p_v[0..head_dim], but we want to sum p_v[dim_idx]
    // across all threads that share the same dim_idx.
    // That's not right — each thread has partial sums for ALL dims.
    // We need to sum thread 0's p_v[d] + thread 1's p_v[d] + ... for each d.

    // Strategy: process dims in chunks of head_dim. In each chunk,
    // thread tid writes its value for dim (tid % head_dim) into shared mem.
    // Threads with the same dim_idx but different slots tree-reduce.

    // With 256 threads and head_dim dims, we have threads_per_dim slots per dim.
    // Process all dims in ceil(256 / head_dim) = threads_per_dim rounds per dim.
    // Actually, we need to sum across ALL 256 threads for each dim.

    // Correct approach: for each dim d, we need sum over all 256 threads of p_v[d].
    // We can't fit 256*128 in shared. So we chunk.
    // Process head_dim dims at a time:
    //   Round 0: thread tid handles dim (tid % head_dim), contributes p_v[tid % head_dim]
    //   But we have 256 values and only head_dim dims, so threads_per_dim threads
    //   contribute to each dim.
    // Wait — each thread has a DIFFERENT p_v for each dim. We need ALL 256 threads'
    // p_v[d] summed for each d.

    // The simplest efficient approach: chunk the head_dim dims.
    // Process 1 dim at a time, sum 256 values via tree reduction.
    // That's 128 rounds of tree reduction = 128 * 9 barriers.
    // On GPU this is ~1152 * 50ns = ~58 microseconds. Negligible vs 274ms frame.
    // Let's just do it.

    for (var d = 0u; d < head_dim; d++) {
        sh_v[tid] = p_v[d];
        workgroupBarrier();

        for (var stride = WG / 2u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                sh_v[tid] += sh_v[tid + stride];
            }
            workgroupBarrier();
        }

        if (tid == 0u) {
            out_buf[out_off + d] = sh_v[0] / global_sum;
        }
        workgroupBarrier();
    }
}
