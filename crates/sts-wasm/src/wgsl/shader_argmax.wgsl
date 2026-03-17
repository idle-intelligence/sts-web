// GPU Argmax Compute Shader
//
// Finds the index of the maximum value in a 1D float array using parallel
// reduction with workgroup shared memory.
//
// Input:  logits[V] — flat f32 array (vocab size, e.g. 2048)
// Output: result[0] — u32 index of the maximum element
//
// With 256 threads per workgroup, each thread scans V/256 elements (e.g. 8
// for V=2048), then we do a shared-memory reduction to find the global max.
// Single workgroup dispatch — no multi-workgroup coordination needed for V<=65536.

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<u32>;
@group(0) @binding(2) var<storage, read> info: array<u32>;

var<workgroup> shared_vals: array<f32, 256>;
var<workgroup> shared_idxs: array<u32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let V = info[0];

    // Phase 1: Each thread finds max of its strided chunk
    let stride = 256u;
    var local_max: f32 = -3.402823466e+38; // -FLT_MAX
    var local_idx: u32 = 0u;

    var i = tid;
    while (i < V) {
        let val = logits[i];
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
        i = i + stride;
    }

    shared_vals[tid] = local_max;
    shared_idxs[tid] = local_idx;
    workgroupBarrier();

    // Phase 2: Tree reduction in shared memory
    // 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
    var half = 128u;
    while (half > 0u) {
        if (tid < half) {
            if (shared_vals[tid + half] > shared_vals[tid]) {
                shared_vals[tid] = shared_vals[tid + half];
                shared_idxs[tid] = shared_idxs[tid + half];
            }
        }
        workgroupBarrier();
        half = half >> 1u;
    }

    // Thread 0 writes the result
    if (tid == 0u) {
        result[0] = shared_idxs[0];
    }
}
