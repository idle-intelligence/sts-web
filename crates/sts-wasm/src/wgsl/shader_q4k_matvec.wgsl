// Q4_K Matrix-Vector Multiply with Shared Memory Input Caching (M=1)
//
// For single-token generation, the input vector x[K] is identical across all N
// output elements. This kernel caches x in shared memory so that all threads
// in a workgroup read from fast on-chip memory instead of global memory.
//
// Each thread computes one complete output element (full dot product over K).
// The workgroup processes TILE_N outputs in parallel, all sharing the same
// cached input vector.
//
// Computes: output[B, 1, N] = input[B, 1, K] x weights[N, K]^T
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

// Workgroup: 256 threads, each computes one output element.
// Input vector cached in shared memory in chunks.
const WG_SIZE: u32 = 256u;
const INPUT_CHUNK: u32 = 4096u;  // max floats cached at once (16KB shared mem)

var<workgroup> shared_input: array<f32, 4096>;

// Read a u32 from an arbitrary byte offset in the weights buffer.
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

// Read an f16 value stored at byte_offset.
fn read_f16(byte_offset: u32) -> f32 {
    let bits = read_u32_unaligned(byte_offset) & 0xFFFFu;
    return unpack2x16float(bits).x;
}

// Unpack the 6-bit scale and min for sub-block j (0..7) within a Q4_K block.
fn get_scale_min_k4(j: u32, block_byte_offset: u32) -> vec2<f32> {
    let scales_offset = block_byte_offset + 4u;
    var sc: u32;
    var m: u32;
    if (j < 4u) {
        sc = read_byte(scales_offset + j) & 63u;
        m  = read_byte(scales_offset + j + 4u) & 63u;
    } else {
        sc = (read_byte(scales_offset + j + 4u) & 0xFu)
           | ((read_byte(scales_offset + j - 4u) >> 6u) << 4u);
        m  = (read_byte(scales_offset + j + 4u) >> 4u)
           | ((read_byte(scales_offset + j) >> 6u) << 4u);
    }
    return vec2<f32>(f32(sc), f32(m));
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let B = info[0];
    let K = info[2];
    let N = info[3];
    let blocks_per_row = info[4]; // K / 256

    let n = gid.x;       // output column (thread-level, up to N)
    let b = wg_id.y;     // batch index
    let tid = local_id.x;

    if (b >= B) {
        return;
    }

    let input_base = b * K;
    var acc: f32 = 0.0;

    // Process K in chunks that fit in shared memory
    let num_chunks = (K + INPUT_CHUNK - 1u) / INPUT_CHUNK;

    for (var chunk: u32 = 0u; chunk < num_chunks; chunk = chunk + 1u) {
        let chunk_k_start = chunk * INPUT_CHUNK;
        let chunk_k_end = min(chunk_k_start + INPUT_CHUNK, K);
        let chunk_len = chunk_k_end - chunk_k_start;

        // All 256 threads cooperatively load input chunk into shared memory
        for (var i: u32 = tid; i < chunk_len; i = i + WG_SIZE) {
            shared_input[i] = input[input_base + chunk_k_start + i];
        }
        workgroupBarrier();

        // Each thread processes the super-blocks that fall in this chunk
        if (n < N) {
            let first_blk = chunk_k_start / 256u;
            let last_blk = (chunk_k_end - 1u) / 256u;

            for (var blk: u32 = first_blk; blk <= last_blk; blk = blk + 1u) {
                let global_block = n * blocks_per_row + blk;
                let block_byte = global_block * 144u;
                let k_base = blk * 256u;

                // Read super-block header
                let d    = read_f16(block_byte);
                let dmin = read_f16(block_byte + 2u);

                let qs_offset = block_byte + 16u;

                // Process 8 sub-blocks (4 iterations, 2 sub-blocks per iteration)
                for (var it: u32 = 0u; it < 4u; it = it + 1u) {
                    let sb_lo = 2u * it;
                    let sb_hi = 2u * it + 1u;

                    let sm_lo = get_scale_min_k4(sb_lo, block_byte);
                    let sm_hi = get_scale_min_k4(sb_hi, block_byte);

                    let d1  = d * sm_lo.x;
                    let m1  = dmin * sm_lo.y;
                    let d2  = d * sm_hi.x;
                    let m2  = dmin * sm_hi.y;

                    // Weight indices within K
                    let k_lo = k_base + sb_lo * 32u;
                    let k_hi = k_base + sb_hi * 32u;

                    // Shared memory offsets (relative to chunk start)
                    let sh_lo = k_lo - chunk_k_start;
                    let sh_hi = k_hi - chunk_k_start;

                    let qs_iter_offset = qs_offset + it * 32u;

                    for (var wi: u32 = 0u; wi < 8u; wi = wi + 1u) {
                        let packed = read_u32_unaligned(qs_iter_offset + wi * 4u);
                        let byte0 = packed & 0xFFu;
                        let byte1 = (packed >> 8u) & 0xFFu;
                        let byte2 = (packed >> 16u) & 0xFFu;
                        let byte3 = (packed >> 24u) & 0xFFu;

                        let base_j = wi * 4u;

                        // Low nibbles -> sub-block sb_lo
                        let nib_lo = vec4<f32>(
                            f32(byte0 & 0xFu),
                            f32(byte1 & 0xFu),
                            f32(byte2 & 0xFu),
                            f32(byte3 & 0xFu)
                        );
                        let in_lo = vec4<f32>(
                            shared_input[sh_lo + base_j],
                            shared_input[sh_lo + base_j + 1u],
                            shared_input[sh_lo + base_j + 2u],
                            shared_input[sh_lo + base_j + 3u],
                        );
                        acc += dot(nib_lo * d1 - vec4<f32>(m1), in_lo);

                        // High nibbles -> sub-block sb_hi
                        let nib_hi = vec4<f32>(
                            f32((byte0 >> 4u) & 0xFu),
                            f32((byte1 >> 4u) & 0xFu),
                            f32((byte2 >> 4u) & 0xFu),
                            f32((byte3 >> 4u) & 0xFu)
                        );
                        let in_hi = vec4<f32>(
                            shared_input[sh_hi + base_j],
                            shared_input[sh_hi + base_j + 1u],
                            shared_input[sh_hi + base_j + 2u],
                            shared_input[sh_hi + base_j + 3u],
                        );
                        acc += dot(nib_hi * d2 - vec4<f32>(m2), in_hi);
                    }
                }
            }
        }

        workgroupBarrier();
    }

    // Write result (M=1, so output is [B, 1, N])
    if (n < N) {
        output[b * N + n] = acc;
    }
}
