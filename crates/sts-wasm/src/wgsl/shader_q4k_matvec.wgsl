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
// Q4_K block layout (144 bytes = 36 u32s per 256 weights):
//   Bytes  0-1:   d     (f16 super-scale)
//   Bytes  2-3:   dmin  (f16 super-min)
//   Bytes  4-15:  scales (12 bytes = 3 u32s, packed 6-bit scale/min per sub-block)
//   Bytes 16-143: qs    (128 bytes = 32 u32s, packed 4-bit weights)
//
// All fields are u32-aligned because block size (144) is divisible by 4.

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> info: array<u32>;

// Workgroup: 256 threads, each computes one output element.
// Input vector cached in shared memory in chunks.
const WG_SIZE: u32 = 256u;
const INPUT_CHUNK: u32 = 4096u;  // max floats cached at once (16KB shared mem)
const BLOCK_WORDS: u32 = 36u;    // 144 bytes / 4

var<workgroup> shared_input: array<f32, 4096>;

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
                let block_word = global_block * BLOCK_WORDS;
                let k_base = blk * 256u;

                // Read d (f16) and dmin (f16) packed in one u32 at word 0
                let d_dmin_bits = weights[block_word];
                let d_dmin = unpack2x16float(d_dmin_bits);
                let d    = d_dmin.x;
                let dmin = d_dmin.y;

                // Bulk-load 12 scale bytes as 3 u32s (words 1-3)
                let sc0 = weights[block_word + 1u];
                let sc1 = weights[block_word + 2u];
                let sc2 = weights[block_word + 3u];

                // Extract all 8 scale/min pairs from the 3 registers.
                // Bytes layout of sc0..sc2 (12 bytes, indices 0..11):
                //   sc0: bytes [0,1,2,3]  sc1: bytes [4,5,6,7]  sc2: bytes [8,9,10,11]
                //
                // For j < 4:  scale[j] = byte[j] & 63,  min[j] = byte[j+4] & 63
                // For j >= 4: scale[j] = (byte[j+4] & 0xF) | ((byte[j-4] >> 6) << 4)
                //             min[j]   = (byte[j+4] >> 4)   | ((byte[j]   >> 6) << 4)

                // Extract individual bytes from the 3 u32s
                let b0  = sc0 & 0xFFu;
                let b1  = (sc0 >> 8u) & 0xFFu;
                let b2  = (sc0 >> 16u) & 0xFFu;
                let b3  = (sc0 >> 24u) & 0xFFu;
                let b4  = sc1 & 0xFFu;
                let b5  = (sc1 >> 8u) & 0xFFu;
                let b6  = (sc1 >> 16u) & 0xFFu;
                let b7  = (sc1 >> 24u) & 0xFFu;
                let b8  = sc2 & 0xFFu;
                let b9  = (sc2 >> 8u) & 0xFFu;
                let b10 = (sc2 >> 16u) & 0xFFu;
                let b11 = (sc2 >> 24u) & 0xFFu;

                // Sub-blocks 0-3: scale = byte[j] & 63, min = byte[j+4] & 63
                let scale0 = f32(b0 & 63u);  let min0 = f32(b4 & 63u);
                let scale1 = f32(b1 & 63u);  let min1 = f32(b5 & 63u);
                let scale2 = f32(b2 & 63u);  let min2 = f32(b6 & 63u);
                let scale3 = f32(b3 & 63u);  let min3 = f32(b7 & 63u);

                // Sub-blocks 4-7: scale = (byte[j+4] & 0xF) | ((byte[j-4] >> 6) << 4)
                //                  min   = (byte[j+4] >> 4)  | ((byte[j]   >> 6) << 4)
                let scale4 = f32((b8  & 0xFu) | ((b0 >> 6u) << 4u));
                let min4   = f32((b8  >> 4u)   | ((b4 >> 6u) << 4u));
                let scale5 = f32((b9  & 0xFu) | ((b1 >> 6u) << 4u));
                let min5   = f32((b9  >> 4u)   | ((b5 >> 6u) << 4u));
                let scale6 = f32((b10 & 0xFu) | ((b2 >> 6u) << 4u));
                let min6   = f32((b10 >> 4u)   | ((b6 >> 6u) << 4u));
                let scale7 = f32((b11 & 0xFu) | ((b3 >> 6u) << 4u));
                let min7   = f32((b11 >> 4u)   | ((b7 >> 6u) << 4u));

                // qs starts at word 4 within the block (byte 16)
                let qs_word = block_word + 4u;

                // Process 8 sub-blocks in 4 iterations (2 sub-blocks per iteration)
                // Each iteration: 32 bytes of qs = 8 u32s
                // Low nibbles -> even sub-block, high nibbles -> odd sub-block

                // --- Iteration 0: sub-blocks 0,1 ---
                {
                    let d1 = d * scale0;  let m1 = dmin * min0;
                    let d2 = d * scale1;  let m2 = dmin * min1;
                    let sh_lo = k_base - chunk_k_start;
                    let sh_hi = sh_lo + 32u;
                    let qw = qs_word;

                    // Load 8 u32s as 2 vec4<u32>
                    let q0 = vec4<u32>(weights[qw], weights[qw + 1u], weights[qw + 2u], weights[qw + 3u]);
                    let q1 = vec4<u32>(weights[qw + 4u], weights[qw + 5u], weights[qw + 6u], weights[qw + 7u]);

                    // Process q0 (16 weights: 4 per component)
                    acc += dot(vec4<f32>(f32(q0.x & 0xFu), f32((q0.x >> 8u) & 0xFu), f32((q0.x >> 16u) & 0xFu), f32((q0.x >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                              vec4<f32>(shared_input[sh_lo], shared_input[sh_lo + 1u], shared_input[sh_lo + 2u], shared_input[sh_lo + 3u]));
                    acc += dot(vec4<f32>(f32((q0.x >> 4u) & 0xFu), f32((q0.x >> 12u) & 0xFu), f32((q0.x >> 20u) & 0xFu), f32((q0.x >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                              vec4<f32>(shared_input[sh_hi], shared_input[sh_hi + 1u], shared_input[sh_hi + 2u], shared_input[sh_hi + 3u]));

                    acc += dot(vec4<f32>(f32(q0.y & 0xFu), f32((q0.y >> 8u) & 0xFu), f32((q0.y >> 16u) & 0xFu), f32((q0.y >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                              vec4<f32>(shared_input[sh_lo + 4u], shared_input[sh_lo + 5u], shared_input[sh_lo + 6u], shared_input[sh_lo + 7u]));
                    acc += dot(vec4<f32>(f32((q0.y >> 4u) & 0xFu), f32((q0.y >> 12u) & 0xFu), f32((q0.y >> 20u) & 0xFu), f32((q0.y >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                              vec4<f32>(shared_input[sh_hi + 4u], shared_input[sh_hi + 5u], shared_input[sh_hi + 6u], shared_input[sh_hi + 7u]));

                    acc += dot(vec4<f32>(f32(q0.z & 0xFu), f32((q0.z >> 8u) & 0xFu), f32((q0.z >> 16u) & 0xFu), f32((q0.z >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                              vec4<f32>(shared_input[sh_lo + 8u], shared_input[sh_lo + 9u], shared_input[sh_lo + 10u], shared_input[sh_lo + 11u]));
                    acc += dot(vec4<f32>(f32((q0.z >> 4u) & 0xFu), f32((q0.z >> 12u) & 0xFu), f32((q0.z >> 20u) & 0xFu), f32((q0.z >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                              vec4<f32>(shared_input[sh_hi + 8u], shared_input[sh_hi + 9u], shared_input[sh_hi + 10u], shared_input[sh_hi + 11u]));

                    acc += dot(vec4<f32>(f32(q0.w & 0xFu), f32((q0.w >> 8u) & 0xFu), f32((q0.w >> 16u) & 0xFu), f32((q0.w >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                              vec4<f32>(shared_input[sh_lo + 12u], shared_input[sh_lo + 13u], shared_input[sh_lo + 14u], shared_input[sh_lo + 15u]));
                    acc += dot(vec4<f32>(f32((q0.w >> 4u) & 0xFu), f32((q0.w >> 12u) & 0xFu), f32((q0.w >> 20u) & 0xFu), f32((q0.w >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                              vec4<f32>(shared_input[sh_hi + 12u], shared_input[sh_hi + 13u], shared_input[sh_hi + 14u], shared_input[sh_hi + 15u]));

                    // Process q1 (next 16 weights)
                    acc += dot(vec4<f32>(f32(q1.x & 0xFu), f32((q1.x >> 8u) & 0xFu), f32((q1.x >> 16u) & 0xFu), f32((q1.x >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                              vec4<f32>(shared_input[sh_lo + 16u], shared_input[sh_lo + 17u], shared_input[sh_lo + 18u], shared_input[sh_lo + 19u]));
                    acc += dot(vec4<f32>(f32((q1.x >> 4u) & 0xFu), f32((q1.x >> 12u) & 0xFu), f32((q1.x >> 20u) & 0xFu), f32((q1.x >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                              vec4<f32>(shared_input[sh_hi + 16u], shared_input[sh_hi + 17u], shared_input[sh_hi + 18u], shared_input[sh_hi + 19u]));

                    acc += dot(vec4<f32>(f32(q1.y & 0xFu), f32((q1.y >> 8u) & 0xFu), f32((q1.y >> 16u) & 0xFu), f32((q1.y >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                              vec4<f32>(shared_input[sh_lo + 20u], shared_input[sh_lo + 21u], shared_input[sh_lo + 22u], shared_input[sh_lo + 23u]));
                    acc += dot(vec4<f32>(f32((q1.y >> 4u) & 0xFu), f32((q1.y >> 12u) & 0xFu), f32((q1.y >> 20u) & 0xFu), f32((q1.y >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                              vec4<f32>(shared_input[sh_hi + 20u], shared_input[sh_hi + 21u], shared_input[sh_hi + 22u], shared_input[sh_hi + 23u]));

                    acc += dot(vec4<f32>(f32(q1.z & 0xFu), f32((q1.z >> 8u) & 0xFu), f32((q1.z >> 16u) & 0xFu), f32((q1.z >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                              vec4<f32>(shared_input[sh_lo + 24u], shared_input[sh_lo + 25u], shared_input[sh_lo + 26u], shared_input[sh_lo + 27u]));
                    acc += dot(vec4<f32>(f32((q1.z >> 4u) & 0xFu), f32((q1.z >> 12u) & 0xFu), f32((q1.z >> 20u) & 0xFu), f32((q1.z >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                              vec4<f32>(shared_input[sh_hi + 24u], shared_input[sh_hi + 25u], shared_input[sh_hi + 26u], shared_input[sh_hi + 27u]));

                    acc += dot(vec4<f32>(f32(q1.w & 0xFu), f32((q1.w >> 8u) & 0xFu), f32((q1.w >> 16u) & 0xFu), f32((q1.w >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                              vec4<f32>(shared_input[sh_lo + 28u], shared_input[sh_lo + 29u], shared_input[sh_lo + 30u], shared_input[sh_lo + 31u]));
                    acc += dot(vec4<f32>(f32((q1.w >> 4u) & 0xFu), f32((q1.w >> 12u) & 0xFu), f32((q1.w >> 20u) & 0xFu), f32((q1.w >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                              vec4<f32>(shared_input[sh_hi + 28u], shared_input[sh_hi + 29u], shared_input[sh_hi + 30u], shared_input[sh_hi + 31u]));
                }

                // --- Iteration 1: sub-blocks 2,3 ---
                {
                    let d1 = d * scale2;  let m1 = dmin * min2;
                    let d2 = d * scale3;  let m2 = dmin * min3;
                    let sh_lo = k_base + 64u - chunk_k_start;
                    let sh_hi = sh_lo + 32u;
                    let qw = qs_word + 8u;

                    let q0 = vec4<u32>(weights[qw], weights[qw + 1u], weights[qw + 2u], weights[qw + 3u]);
                    let q1 = vec4<u32>(weights[qw + 4u], weights[qw + 5u], weights[qw + 6u], weights[qw + 7u]);

                    acc += dot(vec4<f32>(f32(q0.x & 0xFu), f32((q0.x >> 8u) & 0xFu), f32((q0.x >> 16u) & 0xFu), f32((q0.x >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                              vec4<f32>(shared_input[sh_lo], shared_input[sh_lo + 1u], shared_input[sh_lo + 2u], shared_input[sh_lo + 3u]));
                    acc += dot(vec4<f32>(f32((q0.x >> 4u) & 0xFu), f32((q0.x >> 12u) & 0xFu), f32((q0.x >> 20u) & 0xFu), f32((q0.x >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                              vec4<f32>(shared_input[sh_hi], shared_input[sh_hi + 1u], shared_input[sh_hi + 2u], shared_input[sh_hi + 3u]));

                    acc += dot(vec4<f32>(f32(q0.y & 0xFu), f32((q0.y >> 8u) & 0xFu), f32((q0.y >> 16u) & 0xFu), f32((q0.y >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                              vec4<f32>(shared_input[sh_lo + 4u], shared_input[sh_lo + 5u], shared_input[sh_lo + 6u], shared_input[sh_lo + 7u]));
                    acc += dot(vec4<f32>(f32((q0.y >> 4u) & 0xFu), f32((q0.y >> 12u) & 0xFu), f32((q0.y >> 20u) & 0xFu), f32((q0.y >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                              vec4<f32>(shared_input[sh_hi + 4u], shared_input[sh_hi + 5u], shared_input[sh_hi + 6u], shared_input[sh_hi + 7u]));

                    acc += dot(vec4<f32>(f32(q0.z & 0xFu), f32((q0.z >> 8u) & 0xFu), f32((q0.z >> 16u) & 0xFu), f32((q0.z >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                              vec4<f32>(shared_input[sh_lo + 8u], shared_input[sh_lo + 9u], shared_input[sh_lo + 10u], shared_input[sh_lo + 11u]));
                    acc += dot(vec4<f32>(f32((q0.z >> 4u) & 0xFu), f32((q0.z >> 12u) & 0xFu), f32((q0.z >> 20u) & 0xFu), f32((q0.z >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                              vec4<f32>(shared_input[sh_hi + 8u], shared_input[sh_hi + 9u], shared_input[sh_hi + 10u], shared_input[sh_hi + 11u]));

                    acc += dot(vec4<f32>(f32(q0.w & 0xFu), f32((q0.w >> 8u) & 0xFu), f32((q0.w >> 16u) & 0xFu), f32((q0.w >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                              vec4<f32>(shared_input[sh_lo + 12u], shared_input[sh_lo + 13u], shared_input[sh_lo + 14u], shared_input[sh_lo + 15u]));
                    acc += dot(vec4<f32>(f32((q0.w >> 4u) & 0xFu), f32((q0.w >> 12u) & 0xFu), f32((q0.w >> 20u) & 0xFu), f32((q0.w >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                              vec4<f32>(shared_input[sh_hi + 12u], shared_input[sh_hi + 13u], shared_input[sh_hi + 14u], shared_input[sh_hi + 15u]));

                    acc += dot(vec4<f32>(f32(q1.x & 0xFu), f32((q1.x >> 8u) & 0xFu), f32((q1.x >> 16u) & 0xFu), f32((q1.x >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                              vec4<f32>(shared_input[sh_lo + 16u], shared_input[sh_lo + 17u], shared_input[sh_lo + 18u], shared_input[sh_lo + 19u]));
                    acc += dot(vec4<f32>(f32((q1.x >> 4u) & 0xFu), f32((q1.x >> 12u) & 0xFu), f32((q1.x >> 20u) & 0xFu), f32((q1.x >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                              vec4<f32>(shared_input[sh_hi + 16u], shared_input[sh_hi + 17u], shared_input[sh_hi + 18u], shared_input[sh_hi + 19u]));

                    acc += dot(vec4<f32>(f32(q1.y & 0xFu), f32((q1.y >> 8u) & 0xFu), f32((q1.y >> 16u) & 0xFu), f32((q1.y >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                              vec4<f32>(shared_input[sh_lo + 20u], shared_input[sh_lo + 21u], shared_input[sh_lo + 22u], shared_input[sh_lo + 23u]));
                    acc += dot(vec4<f32>(f32((q1.y >> 4u) & 0xFu), f32((q1.y >> 12u) & 0xFu), f32((q1.y >> 20u) & 0xFu), f32((q1.y >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                              vec4<f32>(shared_input[sh_hi + 20u], shared_input[sh_hi + 21u], shared_input[sh_hi + 22u], shared_input[sh_hi + 23u]));

                    acc += dot(vec4<f32>(f32(q1.z & 0xFu), f32((q1.z >> 8u) & 0xFu), f32((q1.z >> 16u) & 0xFu), f32((q1.z >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                              vec4<f32>(shared_input[sh_lo + 24u], shared_input[sh_lo + 25u], shared_input[sh_lo + 26u], shared_input[sh_lo + 27u]));
                    acc += dot(vec4<f32>(f32((q1.z >> 4u) & 0xFu), f32((q1.z >> 12u) & 0xFu), f32((q1.z >> 20u) & 0xFu), f32((q1.z >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                              vec4<f32>(shared_input[sh_hi + 24u], shared_input[sh_hi + 25u], shared_input[sh_hi + 26u], shared_input[sh_hi + 27u]));

                    acc += dot(vec4<f32>(f32(q1.w & 0xFu), f32((q1.w >> 8u) & 0xFu), f32((q1.w >> 16u) & 0xFu), f32((q1.w >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                              vec4<f32>(shared_input[sh_lo + 28u], shared_input[sh_lo + 29u], shared_input[sh_lo + 30u], shared_input[sh_lo + 31u]));
                    acc += dot(vec4<f32>(f32((q1.w >> 4u) & 0xFu), f32((q1.w >> 12u) & 0xFu), f32((q1.w >> 20u) & 0xFu), f32((q1.w >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                              vec4<f32>(shared_input[sh_hi + 28u], shared_input[sh_hi + 29u], shared_input[sh_hi + 30u], shared_input[sh_hi + 31u]));
                }

                // --- Iteration 2: sub-blocks 4,5 ---
                {
                    let d1 = d * scale4;  let m1 = dmin * min4;
                    let d2 = d * scale5;  let m2 = dmin * min5;
                    let sh_lo = k_base + 128u - chunk_k_start;
                    let sh_hi = sh_lo + 32u;
                    let qw = qs_word + 16u;

                    let q0 = vec4<u32>(weights[qw], weights[qw + 1u], weights[qw + 2u], weights[qw + 3u]);
                    let q1 = vec4<u32>(weights[qw + 4u], weights[qw + 5u], weights[qw + 6u], weights[qw + 7u]);

                    acc += dot(vec4<f32>(f32(q0.x & 0xFu), f32((q0.x >> 8u) & 0xFu), f32((q0.x >> 16u) & 0xFu), f32((q0.x >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                              vec4<f32>(shared_input[sh_lo], shared_input[sh_lo + 1u], shared_input[sh_lo + 2u], shared_input[sh_lo + 3u]));
                    acc += dot(vec4<f32>(f32((q0.x >> 4u) & 0xFu), f32((q0.x >> 12u) & 0xFu), f32((q0.x >> 20u) & 0xFu), f32((q0.x >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                              vec4<f32>(shared_input[sh_hi], shared_input[sh_hi + 1u], shared_input[sh_hi + 2u], shared_input[sh_hi + 3u]));

                    acc += dot(vec4<f32>(f32(q0.y & 0xFu), f32((q0.y >> 8u) & 0xFu), f32((q0.y >> 16u) & 0xFu), f32((q0.y >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                              vec4<f32>(shared_input[sh_lo + 4u], shared_input[sh_lo + 5u], shared_input[sh_lo + 6u], shared_input[sh_lo + 7u]));
                    acc += dot(vec4<f32>(f32((q0.y >> 4u) & 0xFu), f32((q0.y >> 12u) & 0xFu), f32((q0.y >> 20u) & 0xFu), f32((q0.y >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                              vec4<f32>(shared_input[sh_hi + 4u], shared_input[sh_hi + 5u], shared_input[sh_hi + 6u], shared_input[sh_hi + 7u]));

                    acc += dot(vec4<f32>(f32(q0.z & 0xFu), f32((q0.z >> 8u) & 0xFu), f32((q0.z >> 16u) & 0xFu), f32((q0.z >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                              vec4<f32>(shared_input[sh_lo + 8u], shared_input[sh_lo + 9u], shared_input[sh_lo + 10u], shared_input[sh_lo + 11u]));
                    acc += dot(vec4<f32>(f32((q0.z >> 4u) & 0xFu), f32((q0.z >> 12u) & 0xFu), f32((q0.z >> 20u) & 0xFu), f32((q0.z >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                              vec4<f32>(shared_input[sh_hi + 8u], shared_input[sh_hi + 9u], shared_input[sh_hi + 10u], shared_input[sh_hi + 11u]));

                    acc += dot(vec4<f32>(f32(q0.w & 0xFu), f32((q0.w >> 8u) & 0xFu), f32((q0.w >> 16u) & 0xFu), f32((q0.w >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                              vec4<f32>(shared_input[sh_lo + 12u], shared_input[sh_lo + 13u], shared_input[sh_lo + 14u], shared_input[sh_lo + 15u]));
                    acc += dot(vec4<f32>(f32((q0.w >> 4u) & 0xFu), f32((q0.w >> 12u) & 0xFu), f32((q0.w >> 20u) & 0xFu), f32((q0.w >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                              vec4<f32>(shared_input[sh_hi + 12u], shared_input[sh_hi + 13u], shared_input[sh_hi + 14u], shared_input[sh_hi + 15u]));

                    acc += dot(vec4<f32>(f32(q1.x & 0xFu), f32((q1.x >> 8u) & 0xFu), f32((q1.x >> 16u) & 0xFu), f32((q1.x >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                              vec4<f32>(shared_input[sh_lo + 16u], shared_input[sh_lo + 17u], shared_input[sh_lo + 18u], shared_input[sh_lo + 19u]));
                    acc += dot(vec4<f32>(f32((q1.x >> 4u) & 0xFu), f32((q1.x >> 12u) & 0xFu), f32((q1.x >> 20u) & 0xFu), f32((q1.x >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                              vec4<f32>(shared_input[sh_hi + 16u], shared_input[sh_hi + 17u], shared_input[sh_hi + 18u], shared_input[sh_hi + 19u]));

                    acc += dot(vec4<f32>(f32(q1.y & 0xFu), f32((q1.y >> 8u) & 0xFu), f32((q1.y >> 16u) & 0xFu), f32((q1.y >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                              vec4<f32>(shared_input[sh_lo + 20u], shared_input[sh_lo + 21u], shared_input[sh_lo + 22u], shared_input[sh_lo + 23u]));
                    acc += dot(vec4<f32>(f32((q1.y >> 4u) & 0xFu), f32((q1.y >> 12u) & 0xFu), f32((q1.y >> 20u) & 0xFu), f32((q1.y >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                              vec4<f32>(shared_input[sh_hi + 20u], shared_input[sh_hi + 21u], shared_input[sh_hi + 22u], shared_input[sh_hi + 23u]));

                    acc += dot(vec4<f32>(f32(q1.z & 0xFu), f32((q1.z >> 8u) & 0xFu), f32((q1.z >> 16u) & 0xFu), f32((q1.z >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                              vec4<f32>(shared_input[sh_lo + 24u], shared_input[sh_lo + 25u], shared_input[sh_lo + 26u], shared_input[sh_lo + 27u]));
                    acc += dot(vec4<f32>(f32((q1.z >> 4u) & 0xFu), f32((q1.z >> 12u) & 0xFu), f32((q1.z >> 20u) & 0xFu), f32((q1.z >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                              vec4<f32>(shared_input[sh_hi + 24u], shared_input[sh_hi + 25u], shared_input[sh_hi + 26u], shared_input[sh_hi + 27u]));

                    acc += dot(vec4<f32>(f32(q1.w & 0xFu), f32((q1.w >> 8u) & 0xFu), f32((q1.w >> 16u) & 0xFu), f32((q1.w >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                              vec4<f32>(shared_input[sh_lo + 28u], shared_input[sh_lo + 29u], shared_input[sh_lo + 30u], shared_input[sh_lo + 31u]));
                    acc += dot(vec4<f32>(f32((q1.w >> 4u) & 0xFu), f32((q1.w >> 12u) & 0xFu), f32((q1.w >> 20u) & 0xFu), f32((q1.w >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                              vec4<f32>(shared_input[sh_hi + 28u], shared_input[sh_hi + 29u], shared_input[sh_hi + 30u], shared_input[sh_hi + 31u]));
                }

                // --- Iteration 3: sub-blocks 6,7 ---
                {
                    let d1 = d * scale6;  let m1 = dmin * min6;
                    let d2 = d * scale7;  let m2 = dmin * min7;
                    let sh_lo = k_base + 192u - chunk_k_start;
                    let sh_hi = sh_lo + 32u;
                    let qw = qs_word + 24u;

                    let q0 = vec4<u32>(weights[qw], weights[qw + 1u], weights[qw + 2u], weights[qw + 3u]);
                    let q1 = vec4<u32>(weights[qw + 4u], weights[qw + 5u], weights[qw + 6u], weights[qw + 7u]);

                    acc += dot(vec4<f32>(f32(q0.x & 0xFu), f32((q0.x >> 8u) & 0xFu), f32((q0.x >> 16u) & 0xFu), f32((q0.x >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                              vec4<f32>(shared_input[sh_lo], shared_input[sh_lo + 1u], shared_input[sh_lo + 2u], shared_input[sh_lo + 3u]));
                    acc += dot(vec4<f32>(f32((q0.x >> 4u) & 0xFu), f32((q0.x >> 12u) & 0xFu), f32((q0.x >> 20u) & 0xFu), f32((q0.x >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                              vec4<f32>(shared_input[sh_hi], shared_input[sh_hi + 1u], shared_input[sh_hi + 2u], shared_input[sh_hi + 3u]));

                    acc += dot(vec4<f32>(f32(q0.y & 0xFu), f32((q0.y >> 8u) & 0xFu), f32((q0.y >> 16u) & 0xFu), f32((q0.y >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                              vec4<f32>(shared_input[sh_lo + 4u], shared_input[sh_lo + 5u], shared_input[sh_lo + 6u], shared_input[sh_lo + 7u]));
                    acc += dot(vec4<f32>(f32((q0.y >> 4u) & 0xFu), f32((q0.y >> 12u) & 0xFu), f32((q0.y >> 20u) & 0xFu), f32((q0.y >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                              vec4<f32>(shared_input[sh_hi + 4u], shared_input[sh_hi + 5u], shared_input[sh_hi + 6u], shared_input[sh_hi + 7u]));

                    acc += dot(vec4<f32>(f32(q0.z & 0xFu), f32((q0.z >> 8u) & 0xFu), f32((q0.z >> 16u) & 0xFu), f32((q0.z >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                              vec4<f32>(shared_input[sh_lo + 8u], shared_input[sh_lo + 9u], shared_input[sh_lo + 10u], shared_input[sh_lo + 11u]));
                    acc += dot(vec4<f32>(f32((q0.z >> 4u) & 0xFu), f32((q0.z >> 12u) & 0xFu), f32((q0.z >> 20u) & 0xFu), f32((q0.z >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                              vec4<f32>(shared_input[sh_hi + 8u], shared_input[sh_hi + 9u], shared_input[sh_hi + 10u], shared_input[sh_hi + 11u]));

                    acc += dot(vec4<f32>(f32(q0.w & 0xFu), f32((q0.w >> 8u) & 0xFu), f32((q0.w >> 16u) & 0xFu), f32((q0.w >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                              vec4<f32>(shared_input[sh_lo + 12u], shared_input[sh_lo + 13u], shared_input[sh_lo + 14u], shared_input[sh_lo + 15u]));
                    acc += dot(vec4<f32>(f32((q0.w >> 4u) & 0xFu), f32((q0.w >> 12u) & 0xFu), f32((q0.w >> 20u) & 0xFu), f32((q0.w >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                              vec4<f32>(shared_input[sh_hi + 12u], shared_input[sh_hi + 13u], shared_input[sh_hi + 14u], shared_input[sh_hi + 15u]));

                    acc += dot(vec4<f32>(f32(q1.x & 0xFu), f32((q1.x >> 8u) & 0xFu), f32((q1.x >> 16u) & 0xFu), f32((q1.x >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                              vec4<f32>(shared_input[sh_lo + 16u], shared_input[sh_lo + 17u], shared_input[sh_lo + 18u], shared_input[sh_lo + 19u]));
                    acc += dot(vec4<f32>(f32((q1.x >> 4u) & 0xFu), f32((q1.x >> 12u) & 0xFu), f32((q1.x >> 20u) & 0xFu), f32((q1.x >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                              vec4<f32>(shared_input[sh_hi + 16u], shared_input[sh_hi + 17u], shared_input[sh_hi + 18u], shared_input[sh_hi + 19u]));

                    acc += dot(vec4<f32>(f32(q1.y & 0xFu), f32((q1.y >> 8u) & 0xFu), f32((q1.y >> 16u) & 0xFu), f32((q1.y >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                              vec4<f32>(shared_input[sh_lo + 20u], shared_input[sh_lo + 21u], shared_input[sh_lo + 22u], shared_input[sh_lo + 23u]));
                    acc += dot(vec4<f32>(f32((q1.y >> 4u) & 0xFu), f32((q1.y >> 12u) & 0xFu), f32((q1.y >> 20u) & 0xFu), f32((q1.y >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                              vec4<f32>(shared_input[sh_hi + 20u], shared_input[sh_hi + 21u], shared_input[sh_hi + 22u], shared_input[sh_hi + 23u]));

                    acc += dot(vec4<f32>(f32(q1.z & 0xFu), f32((q1.z >> 8u) & 0xFu), f32((q1.z >> 16u) & 0xFu), f32((q1.z >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                              vec4<f32>(shared_input[sh_lo + 24u], shared_input[sh_lo + 25u], shared_input[sh_lo + 26u], shared_input[sh_lo + 27u]));
                    acc += dot(vec4<f32>(f32((q1.z >> 4u) & 0xFu), f32((q1.z >> 12u) & 0xFu), f32((q1.z >> 20u) & 0xFu), f32((q1.z >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                              vec4<f32>(shared_input[sh_hi + 24u], shared_input[sh_hi + 25u], shared_input[sh_hi + 26u], shared_input[sh_hi + 27u]));

                    acc += dot(vec4<f32>(f32(q1.w & 0xFu), f32((q1.w >> 8u) & 0xFu), f32((q1.w >> 16u) & 0xFu), f32((q1.w >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                              vec4<f32>(shared_input[sh_lo + 28u], shared_input[sh_lo + 29u], shared_input[sh_lo + 30u], shared_input[sh_lo + 31u]));
                    acc += dot(vec4<f32>(f32((q1.w >> 4u) & 0xFu), f32((q1.w >> 12u) & 0xFu), f32((q1.w >> 20u) & 0xFu), f32((q1.w >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                              vec4<f32>(shared_input[sh_hi + 28u], shared_input[sh_hi + 29u], shared_input[sh_hi + 30u], shared_input[sh_hi + 31u]));
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
