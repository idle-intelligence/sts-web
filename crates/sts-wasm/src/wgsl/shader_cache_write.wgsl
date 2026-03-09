// V Cache Write Compute Shader
//
// Copies V from [B, S, H, D] layout directly into the KV cache buffer
// in [B, H, max_len, D] layout at the correct ring buffer position.
// This fuses swap_dims + into_contiguous + slice_assign into one dispatch.
//
// Bindings:
//   @binding(0) input: f32 array [B * S * H * D] (read, V after reshape)
//   @binding(1) cache: f32 array [B * H * max_len * D] (read_write, V cache)
//   @binding(2) info:  u32 array [5] = { pos_offset, B, S, H, D, max_len }
//
// Dispatch: workgroups = ceil(B * S * H * D / 256), workgroup_size = 256

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> cache: array<f32>;
@group(0) @binding(2) var<storage, read> info: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pos_offset = info[0];
    let B = info[1];
    let S = info[2];
    let H = info[3];
    let D = info[4];
    let max_len = info[5];

    let total_elements = B * S * H * D;
    let idx = gid.x;
    if (idx >= total_elements) {
        return;
    }

    // Decompose flat index into (b, s, h, d)
    let d = idx % D;
    let remaining = idx / D;
    let h = remaining % H;
    let remaining2 = remaining / H;
    let s = remaining2 % S;
    let b = remaining2 / S;

    // Input layout: [B, S, H, D] contiguous
    // Just read linearly since idx maps directly
    let val = input[idx];

    // Cache layout: [B, H, max_len, D]
    let cache_pos = (pos_offset + s) % max_len;
    let base_out = ((b * H + h) * max_len + cache_pos) * D + d;

    cache[base_out] = val;
}
