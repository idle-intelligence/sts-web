// Q4_K Dequantization + Matrix Multiplication Compute Shader
//
// One-thread-per-output-element kernel, no shared memory.
// Q4_K uses 144-byte super-blocks covering 256 weights each,
// with 8 sub-blocks of 32 weights and packed 6-bit scales/mins.
// Compatible with WebGPU's 256 workgroup invocation limit (16x16=256).
//
// Computes: output[B, M, N] = input[B, M, K] x weights[N, K]^T
//
// Q4_K block layout (144 bytes per 256 weights):
//   Bytes  0-1:   d     (f16 super-scale)
//   Bytes  2-3:   dmin  (f16 super-min)
//   Bytes  4-15:  scales (12 bytes, packed 6-bit scale/min per sub-block)
//   Bytes 16-143: qs    (128 bytes, packed 4-bit weights)

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> info: array<u32>;

// Read a u32 from an arbitrary byte offset in the weights buffer.
// Handles unaligned access by combining two adjacent u32 words.
fn read_u32_unaligned(byte_offset: u32) -> u32 {
    let word = byte_offset >> 2u;
    let shift = (byte_offset & 3u) << 3u;
    if (shift == 0u) {
        return weights[word];
    }
    return (weights[word] >> shift) | (weights[word + 1u] << (32u - shift));
}

// Read a single byte from the weights buffer.
fn read_byte(byte_offset: u32) -> u32 {
    let word = byte_offset >> 2u;
    let shift = (byte_offset & 3u) << 3u;
    return (weights[word] >> shift) & 0xFFu;
}

// Read an f16 value stored at byte_offset (occupies 2 bytes).
fn read_f16(byte_offset: u32) -> f32 {
    let bits = read_u32_unaligned(byte_offset) & 0xFFFFu;
    return unpack2x16float(bits).x;
}

// Unpack the 6-bit scale and min for sub-block j (0..7) within a Q4_K block.
// The 12 bytes of packed scales start at block_byte_offset + 4.
// Returns vec2(scale, min) as floats.
fn get_scale_min_k4(j: u32, block_byte_offset: u32) -> vec2<f32> {
    let scales_offset = block_byte_offset + 4u;
    var sc: u32;
    var m: u32;
    if (j < 4u) {
        sc = read_byte(scales_offset + j) & 63u;
        m  = read_byte(scales_offset + j + 4u) & 63u;
    } else {
        // High sub-blocks: 6-bit values split across two locations
        sc = (read_byte(scales_offset + j + 4u) & 0xFu)
           | ((read_byte(scales_offset + j - 4u) >> 6u) << 4u);
        m  = (read_byte(scales_offset + j + 4u) >> 4u)
           | ((read_byte(scales_offset + j) >> 6u) << 4u);
    }
    return vec2<f32>(f32(sc), f32(m));
}

@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let B = info[0];
    let M = info[1];
    let K = info[2];
    let N = info[3];
    let blocks_per_row = info[4]; // K / 256

    let n = gid.x;   // output column
    let bm = gid.y;  // flattened (batch * M + m)
    let m = bm % M;
    let b = bm / M;

    if (n >= N || b >= B) {
        return;
    }

    var acc: f32 = 0.0;
    let input_base = b * M * K + m * K;

    // Iterate over super-blocks (each covers 256 weights)
    for (var blk: u32 = 0u; blk < blocks_per_row; blk = blk + 1u) {
        let global_block = n * blocks_per_row + blk;
        let block_byte = global_block * 144u;
        let k_base = blk * 256u;

        // Read super-block header
        let d    = read_f16(block_byte);        // super-scale
        let dmin = read_f16(block_byte + 2u);   // super-min

        // qs data starts at byte 16 within the block
        let qs_offset = block_byte + 16u;

        // 4 iterations, each processing 2 sub-blocks (64 weights total).
        // Each iteration reads 32 bytes of qs: low nibbles → even sub-block,
        // high nibbles → odd sub-block.
        for (var it: u32 = 0u; it < 4u; it = it + 1u) {
            // Sub-block indices for this iteration
            let sb_lo = 2u * it;       // low-nibble sub-block
            let sb_hi = 2u * it + 1u;  // high-nibble sub-block

            // Unpack 6-bit scales and mins for both sub-blocks
            let sm_lo = get_scale_min_k4(sb_lo, block_byte);
            let sm_hi = get_scale_min_k4(sb_hi, block_byte);

            let d1  = d * sm_lo.x;      // scale for low-nibble sub-block
            let m1  = dmin * sm_lo.y;    // min for low-nibble sub-block
            let d2  = d * sm_hi.x;      // scale for high-nibble sub-block
            let m2  = dmin * sm_hi.y;    // min for high-nibble sub-block

            // Base weight index within K for this iteration's sub-blocks
            let k_lo = input_base + k_base + sb_lo * 32u;
            let k_hi = input_base + k_base + sb_hi * 32u;

            // Each iteration reads 32 bytes of qs (covering 32 low + 32 high nibble weights)
            let qs_iter_offset = qs_offset + it * 32u;

            // Process 4 bytes at a time using u32 reads (8 iterations for 32 bytes)
            for (var wi: u32 = 0u; wi < 8u; wi = wi + 1u) {
                let packed = read_u32_unaligned(qs_iter_offset + wi * 4u);
                let byte0 = packed & 0xFFu;
                let byte1 = (packed >> 8u) & 0xFFu;
                let byte2 = (packed >> 16u) & 0xFFu;
                let byte3 = (packed >> 24u) & 0xFFu;

                let base_j = wi * 4u;

                // Low nibbles -> sub-block sb_lo weights
                let nib_lo = vec4<f32>(
                    f32(byte0 & 0xFu),
                    f32(byte1 & 0xFu),
                    f32(byte2 & 0xFu),
                    f32(byte3 & 0xFu)
                );
                let in_lo = vec4<f32>(
                    input[k_lo + base_j],
                    input[k_lo + base_j + 1u],
                    input[k_lo + base_j + 2u],
                    input[k_lo + base_j + 3u]
                );
                // weight = d1 * nibble - m1
                acc += dot(nib_lo * d1 - vec4<f32>(m1), in_lo);

                // High nibbles -> sub-block sb_hi weights
                let nib_hi = vec4<f32>(
                    f32((byte0 >> 4u) & 0xFu),
                    f32((byte1 >> 4u) & 0xFu),
                    f32((byte2 >> 4u) & 0xFu),
                    f32((byte3 >> 4u) & 0xFu)
                );
                let in_hi = vec4<f32>(
                    input[k_hi + base_j],
                    input[k_hi + base_j + 1u],
                    input[k_hi + base_j + 2u],
                    input[k_hi + base_j + 3u]
                );
                // weight = d2 * nibble - m2
                acc += dot(nib_hi * d2 - vec4<f32>(m2), in_hi);
            }
        }
    }

    output[b * M * N + m * N + n] = acc;
}
