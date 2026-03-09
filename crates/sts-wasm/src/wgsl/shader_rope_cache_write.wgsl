// Fused RoPE + KV Cache Write Compute Shader
//
// Applies RoPE rotation to K and writes the result directly into the KV cache
// buffer at the correct ring buffer position. This fuses two operations:
// 1. fused_rope_inplace(K) — RoPE rotation
// 2. cache.update(K) — slice_assign into ring buffer
//
// Input K layout:  [B, S, H, D] contiguous (from reshape after in_proj slice)
// Output layout:   [B, H, max_len, D] (KV cache ring buffer, BHSD)
//
// For generation (S=1), this eliminates:
// - The swap_dims(1,2) that makes K non-contiguous
// - The into_contiguous copy dispatch
// - The slice_assign dispatch
//
// Bindings:
//   @binding(0) input:     f32 array [B * S * H * D] (read, K after reshape)
//   @binding(1) cos_table: f32 array [max_seq_len * half_D]
//   @binding(2) sin_table: f32 array [max_seq_len * half_D]
//   @binding(3) cache:     f32 array [B * H * max_len * D] (read_write, K cache)
//   @binding(4) info:      u32 array [6] = { pos_offset, B, S, H, half_D, max_len }
//
// Dispatch: workgroups = ceil(B * S * H * half_D / 256), workgroup_size = 256

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> cos_table: array<f32>;
@group(0) @binding(2) var<storage, read> sin_table: array<f32>;
@group(0) @binding(3) var<storage, read_write> cache: array<f32>;
@group(0) @binding(4) var<storage, read> info: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pos_offset = info[0];
    let B = info[1];
    let S = info[2];
    let H = info[3];
    let half_D = info[4];
    let max_len = info[5];
    let D = half_D * 2u;

    let total_elements = B * S * H * half_D;
    let idx = gid.x;
    if (idx >= total_elements) {
        return;
    }

    // Decompose flat index into (b, s, h, d) where d is the pair index
    let d = idx % half_D;
    let remaining = idx / half_D;
    let h = remaining % H;
    let remaining2 = remaining / H;
    let s = remaining2 % S;
    let b = remaining2 / S;

    // Position in cos/sin table: absolute position = pos_offset + s
    let abs_pos = pos_offset + s;
    let table_idx = abs_pos * half_D + d;

    let cos_val = cos_table[table_idx];
    let sin_val = sin_table[table_idx];

    // Input layout: [B, S, H, D] contiguous
    // base_in = ((b * S + s) * H + h) * D + d * 2
    let base_in = ((b * S + s) * H + h) * D + d * 2u;
    let x_r = input[base_in];
    let x_i = input[base_in + 1u];

    // Apply RoPE rotation
    let out_r = x_r * cos_val - x_i * sin_val;
    let out_i = x_r * sin_val + x_i * cos_val;

    // Cache layout: [B, H, max_len, D]
    // Write position in ring buffer: (pos_offset + s) % max_len
    let cache_pos = (pos_offset + s) % max_len;
    // base_out = ((b * H + h) * max_len + cache_pos) * D + d * 2
    let base_out = ((b * H + h) * max_len + cache_pos) * D + d * 2u;

    cache[base_out] = out_r;
    cache[base_out + 1u] = out_i;
}
