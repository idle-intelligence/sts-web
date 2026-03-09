// Fused single-token attention kernel (Flash-decoding style).
//
// For single-token generation (batch=1, q_seq=1):
//   Q: [num_heads, head_dim]            (contiguous, from [1,1,H,D] reinterpreted)
//   K: [1, num_heads, S, head_dim]      (may be non-contiguous cache slice)
//   V: [1, num_heads, S, head_dim]      (may be non-contiguous cache slice)
//   Output: [num_heads, head_dim]
//
// K and V may come from a KV cache slice where the head stride is larger
// than seq_len * head_dim (because the cache has max_len slots). The kernel
// uses kv_head_stride from the info buffer to compute correct offsets,
// avoiding expensive into_contiguous() copy dispatches.
//
// Dispatch: num_heads workgroups, 256 threads each.
// One workgroup per head. 256 threads cooperate on the S dimension.
//
// Algorithm: Online softmax with 2-phase V reduction.
//   Phase 1: Each thread processes ceil(S/256) KV positions with online softmax.
//   Phase 2: Parallel reduction of (max, sum) across threads.
//   Phase 3: V reduction — 256 threads reduce per-dim partial sums.
//
// Bindings:
//   @group(0) @binding(0) q:      f32 [num_heads * head_dim]
//   @group(0) @binding(1) k:      f32 (stride-addressed via kv_head_stride)
//   @group(0) @binding(2) v:      f32 (stride-addressed via kv_head_stride)
//   @group(0) @binding(3) output: f32 [num_heads * head_dim]
//   @group(0) @binding(4) info:   u32 [5]: {num_heads, head_dim, seq_len, scale_bits, kv_head_stride}

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
    let kv_head_stride = info[4];  // actual stride between heads in K/V buffers

    let q_off = head * head_dim;
    // Use kv_head_stride instead of seq_len * head_dim for K/V offset.
    // This handles non-contiguous cache slices where head stride > seq_len * head_dim.
    let kv_off = head * kv_head_stride;

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
    // =================================================================

    let correction = exp(p_max - global_max);
    for (var d = 0u; d < head_dim; d++) {
        p_v[d] *= correction;
    }

    let out_off = head * head_dim;

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
