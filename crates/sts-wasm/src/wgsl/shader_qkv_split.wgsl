// Fused QKV Split Compute Shader
//
// Replaces 2-3 clone+slice operations (which may trigger contiguous copy dispatches)
// with a single GPU dispatch. Reads from a packed QKV buffer [B*S*(q_dim + 2*kv_dim)]
// and writes to 3 separate contiguous output buffers.
//
// Layout in QKV buffer (per row of B*S rows):
//   [0..q_dim] = Q, [q_dim..q_dim+kv_dim] = K, [q_dim+kv_dim..q_dim+2*kv_dim] = V
//
// Bindings:
//   @binding(0) qkv:   f32 array [total_elements] (packed QKV)
//   @binding(1) q_out: f32 array [B*S*q_dim]
//   @binding(2) k_out: f32 array [B*S*kv_dim]
//   @binding(3) v_out: f32 array [B*S*kv_dim]
//   @binding(4) info:  u32 array [3] = { q_dim, kv_dim, num_rows (B*S) }
//
// Dispatch: workgroups = ceil(max(q_dim, kv_dim) * num_rows / 256)

@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> q_out: array<f32>;
@group(0) @binding(2) var<storage, read_write> k_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> v_out: array<f32>;
@group(0) @binding(4) var<storage, read> info: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q_dim = info[0];
    let kv_dim = info[1];
    let num_rows = info[2];
    let row_stride = q_dim + 2u * kv_dim;
    let max_dim = max(q_dim, kv_dim);
    let total_threads = max_dim * num_rows;

    let tid = gid.x;
    if (tid >= total_threads) {
        return;
    }

    let row = tid / max_dim;
    let col = tid % max_dim;

    let qkv_row_offset = row * row_stride;

    // Write Q element
    if (col < q_dim) {
        q_out[row * q_dim + col] = qkv[qkv_row_offset + col];
    }

    // Write K element
    if (col < kv_dim) {
        k_out[row * kv_dim + col] = qkv[qkv_row_offset + q_dim + col];
    }

    // Write V element
    if (col < kv_dim) {
        v_out[row * kv_dim + col] = qkv[qkv_row_offset + q_dim + kv_dim + col];
    }
}
