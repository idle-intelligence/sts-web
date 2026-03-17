// Fused RMSNorm Compute Shader
//
// Replaces 6-8 Burn dispatches with a single GPU dispatch per RMSNorm call.
// Called ~256 times per generation step (2 norms x 32 temporal + 2 norms x 6 depth x 16).
//
// Algorithm per (b, s) position:
//   1. Each thread accumulates partial sum-of-squares over its D/256 elements
//   2. Workgroup reduction via shared memory to get total sum-of-squares
//   3. Compute rms = sqrt(sum/D + eps)
//   4. Each thread: output[i] = (input[i] / rms) * alpha[i]
//
// Bindings:
//   @binding(0) input:  f32 array [B * S * D]
//   @binding(1) alpha:  f32 array [D]
//   @binding(2) output: f32 array [B * S * D]
//   @binding(3) info:   u32 array [3] = { D, eps_bits, total_rows (B*S) }
//
// Dispatch: workgroups = (B * S, 1, 1), workgroup_size = (256, 1, 1)
// Each workgroup handles one (b, s) row of D elements.

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> alpha: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> info: array<u32>;

var<workgroup> shared_sums: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let D = info[0];
    let eps_bits = info[1];
    let total_rows = info[2];
    let eps = bitcast<f32>(eps_bits);

    let row = wid.x;
    if (row >= total_rows) {
        return;
    }

    let tid = lid.x;
    let row_offset = row * D;

    // Phase 1: Each thread accumulates sum of squares for its elements
    var partial_sum: f32 = 0.0;
    var i = tid;
    loop {
        if (i >= D) {
            break;
        }
        let val = input[row_offset + i];
        partial_sum += val * val;
        i += 256u;
    }

    shared_sums[tid] = partial_sum;
    workgroupBarrier();

    // Phase 2: Tree reduction in shared memory
    // 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
    if (tid < 128u) { shared_sums[tid] += shared_sums[tid + 128u]; }
    workgroupBarrier();
    if (tid < 64u) { shared_sums[tid] += shared_sums[tid + 64u]; }
    workgroupBarrier();
    if (tid < 32u) { shared_sums[tid] += shared_sums[tid + 32u]; }
    workgroupBarrier();
    if (tid < 16u) { shared_sums[tid] += shared_sums[tid + 16u]; }
    workgroupBarrier();
    if (tid < 8u) { shared_sums[tid] += shared_sums[tid + 8u]; }
    workgroupBarrier();
    if (tid < 4u) { shared_sums[tid] += shared_sums[tid + 4u]; }
    workgroupBarrier();
    if (tid < 2u) { shared_sums[tid] += shared_sums[tid + 2u]; }
    workgroupBarrier();
    if (tid == 0u) { shared_sums[0] = shared_sums[0] + shared_sums[1]; }
    workgroupBarrier();

    // Phase 3: Compute rms = sqrt(sum_sq / D + eps)
    let rms = sqrt(shared_sums[0] / f32(D) + eps);
    let inv_rms = 1.0 / rms;

    // Phase 4: Normalize and scale
    i = tid;
    loop {
        if (i >= D) {
            break;
        }
        output[row_offset + i] = input[row_offset + i] * inv_rms * alpha[i];
        i += 256u;
    }
}
