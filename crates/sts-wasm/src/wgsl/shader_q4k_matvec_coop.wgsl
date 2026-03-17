// Q4_K Matrix-Vector Multiply — K-Cooperative Approach (M=1)
//
// Instead of 256 threads each computing a full row, groups of THREADS_PER_ROW
// threads cooperate on the K dimension of a single output element.
//
// Thread mapping:
//   row_in_wg = tid / THREADS_PER_ROW   (which of the 16 rows this thread works on)
//   k_lane    = tid % THREADS_PER_ROW   (which K-slice within the row)
//
// Each thread processes super-blocks at stride THREADS_PER_ROW:
//   blocks k_lane, k_lane+THREADS_PER_ROW, k_lane+2*THREADS_PER_ROW, ...
//
// After accumulation, tree-reduce partial sums within each group of 16 threads.
// Thread 0 of each group writes the final output.
//
// Dispatch: ceil(N / ROWS_PER_WG) workgroups in X, B in Y.
//
// Q4_K block layout (144 bytes = 36 u32s per 256 weights):
//   Bytes  0-1:   d     (f16 super-scale)
//   Bytes  2-3:   dmin  (f16 super-min)
//   Bytes  4-15:  scales (12 bytes = 3 u32s, packed 6-bit scale/min per sub-block)
//   Bytes 16-143: qs    (128 bytes = 32 u32s, packed 4-bit weights)

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> info: array<u32>;

const WG_SIZE: u32 = 256u;
const THREADS_PER_ROW: u32 = 16u;
const ROWS_PER_WG: u32 = 16u;  // 256 / 16
const BLOCK_WORDS: u32 = 36u;  // 144 bytes / 4

// Shared memory for partial sum reduction
var<workgroup> partial_sums: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let B = info[0];
    let K = info[2];
    let N = info[3];
    let blocks_per_row = info[4]; // K / 256

    let tid = local_id.x;
    let b = wg_id.y;          // batch index

    if (b >= B) {
        return;
    }

    let row_in_wg = tid / THREADS_PER_ROW;  // 0..15
    let k_lane = tid % THREADS_PER_ROW;     // 0..15

    let n = wg_id.x * ROWS_PER_WG + row_in_wg;  // global output row

    let input_base = b * K;
    var acc: f32 = 0.0;

    // Each thread processes super-blocks at stride THREADS_PER_ROW
    if (n < N) {
        var blk = k_lane;
        for (; blk < blocks_per_row; blk = blk + THREADS_PER_ROW) {
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

            // Sub-blocks 4-7
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

            // --- Iteration 0: sub-blocks 0,1 ---
            {
                let d1 = d * scale0;  let m1 = dmin * min0;
                let d2 = d * scale1;  let m2 = dmin * min1;
                let inp_lo = input_base + k_base;
                let inp_hi = inp_lo + 32u;
                let qw = qs_word;

                let q0 = vec4<u32>(weights[qw], weights[qw + 1u], weights[qw + 2u], weights[qw + 3u]);
                let q1 = vec4<u32>(weights[qw + 4u], weights[qw + 5u], weights[qw + 6u], weights[qw + 7u]);

                acc += dot(vec4<f32>(f32(q0.x & 0xFu), f32((q0.x >> 8u) & 0xFu), f32((q0.x >> 16u) & 0xFu), f32((q0.x >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                          vec4<f32>(input[inp_lo], input[inp_lo + 1u], input[inp_lo + 2u], input[inp_lo + 3u]));
                acc += dot(vec4<f32>(f32((q0.x >> 4u) & 0xFu), f32((q0.x >> 12u) & 0xFu), f32((q0.x >> 20u) & 0xFu), f32((q0.x >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                          vec4<f32>(input[inp_hi], input[inp_hi + 1u], input[inp_hi + 2u], input[inp_hi + 3u]));

                acc += dot(vec4<f32>(f32(q0.y & 0xFu), f32((q0.y >> 8u) & 0xFu), f32((q0.y >> 16u) & 0xFu), f32((q0.y >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                          vec4<f32>(input[inp_lo + 4u], input[inp_lo + 5u], input[inp_lo + 6u], input[inp_lo + 7u]));
                acc += dot(vec4<f32>(f32((q0.y >> 4u) & 0xFu), f32((q0.y >> 12u) & 0xFu), f32((q0.y >> 20u) & 0xFu), f32((q0.y >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                          vec4<f32>(input[inp_hi + 4u], input[inp_hi + 5u], input[inp_hi + 6u], input[inp_hi + 7u]));

                acc += dot(vec4<f32>(f32(q0.z & 0xFu), f32((q0.z >> 8u) & 0xFu), f32((q0.z >> 16u) & 0xFu), f32((q0.z >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                          vec4<f32>(input[inp_lo + 8u], input[inp_lo + 9u], input[inp_lo + 10u], input[inp_lo + 11u]));
                acc += dot(vec4<f32>(f32((q0.z >> 4u) & 0xFu), f32((q0.z >> 12u) & 0xFu), f32((q0.z >> 20u) & 0xFu), f32((q0.z >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                          vec4<f32>(input[inp_hi + 8u], input[inp_hi + 9u], input[inp_hi + 10u], input[inp_hi + 11u]));

                acc += dot(vec4<f32>(f32(q0.w & 0xFu), f32((q0.w >> 8u) & 0xFu), f32((q0.w >> 16u) & 0xFu), f32((q0.w >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                          vec4<f32>(input[inp_lo + 12u], input[inp_lo + 13u], input[inp_lo + 14u], input[inp_lo + 15u]));
                acc += dot(vec4<f32>(f32((q0.w >> 4u) & 0xFu), f32((q0.w >> 12u) & 0xFu), f32((q0.w >> 20u) & 0xFu), f32((q0.w >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                          vec4<f32>(input[inp_hi + 12u], input[inp_hi + 13u], input[inp_hi + 14u], input[inp_hi + 15u]));

                acc += dot(vec4<f32>(f32(q1.x & 0xFu), f32((q1.x >> 8u) & 0xFu), f32((q1.x >> 16u) & 0xFu), f32((q1.x >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                          vec4<f32>(input[inp_lo + 16u], input[inp_lo + 17u], input[inp_lo + 18u], input[inp_lo + 19u]));
                acc += dot(vec4<f32>(f32((q1.x >> 4u) & 0xFu), f32((q1.x >> 12u) & 0xFu), f32((q1.x >> 20u) & 0xFu), f32((q1.x >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                          vec4<f32>(input[inp_hi + 16u], input[inp_hi + 17u], input[inp_hi + 18u], input[inp_hi + 19u]));

                acc += dot(vec4<f32>(f32(q1.y & 0xFu), f32((q1.y >> 8u) & 0xFu), f32((q1.y >> 16u) & 0xFu), f32((q1.y >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                          vec4<f32>(input[inp_lo + 20u], input[inp_lo + 21u], input[inp_lo + 22u], input[inp_lo + 23u]));
                acc += dot(vec4<f32>(f32((q1.y >> 4u) & 0xFu), f32((q1.y >> 12u) & 0xFu), f32((q1.y >> 20u) & 0xFu), f32((q1.y >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                          vec4<f32>(input[inp_hi + 20u], input[inp_hi + 21u], input[inp_hi + 22u], input[inp_hi + 23u]));

                acc += dot(vec4<f32>(f32(q1.z & 0xFu), f32((q1.z >> 8u) & 0xFu), f32((q1.z >> 16u) & 0xFu), f32((q1.z >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                          vec4<f32>(input[inp_lo + 24u], input[inp_lo + 25u], input[inp_lo + 26u], input[inp_lo + 27u]));
                acc += dot(vec4<f32>(f32((q1.z >> 4u) & 0xFu), f32((q1.z >> 12u) & 0xFu), f32((q1.z >> 20u) & 0xFu), f32((q1.z >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                          vec4<f32>(input[inp_hi + 24u], input[inp_hi + 25u], input[inp_hi + 26u], input[inp_hi + 27u]));

                acc += dot(vec4<f32>(f32(q1.w & 0xFu), f32((q1.w >> 8u) & 0xFu), f32((q1.w >> 16u) & 0xFu), f32((q1.w >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                          vec4<f32>(input[inp_lo + 28u], input[inp_lo + 29u], input[inp_lo + 30u], input[inp_lo + 31u]));
                acc += dot(vec4<f32>(f32((q1.w >> 4u) & 0xFu), f32((q1.w >> 12u) & 0xFu), f32((q1.w >> 20u) & 0xFu), f32((q1.w >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                          vec4<f32>(input[inp_hi + 28u], input[inp_hi + 29u], input[inp_hi + 30u], input[inp_hi + 31u]));
            }

            // --- Iteration 1: sub-blocks 2,3 ---
            {
                let d1 = d * scale2;  let m1 = dmin * min2;
                let d2 = d * scale3;  let m2 = dmin * min3;
                let inp_lo = input_base + k_base + 64u;
                let inp_hi = inp_lo + 32u;
                let qw = qs_word + 8u;

                let q0 = vec4<u32>(weights[qw], weights[qw + 1u], weights[qw + 2u], weights[qw + 3u]);
                let q1 = vec4<u32>(weights[qw + 4u], weights[qw + 5u], weights[qw + 6u], weights[qw + 7u]);

                acc += dot(vec4<f32>(f32(q0.x & 0xFu), f32((q0.x >> 8u) & 0xFu), f32((q0.x >> 16u) & 0xFu), f32((q0.x >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                          vec4<f32>(input[inp_lo], input[inp_lo + 1u], input[inp_lo + 2u], input[inp_lo + 3u]));
                acc += dot(vec4<f32>(f32((q0.x >> 4u) & 0xFu), f32((q0.x >> 12u) & 0xFu), f32((q0.x >> 20u) & 0xFu), f32((q0.x >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                          vec4<f32>(input[inp_hi], input[inp_hi + 1u], input[inp_hi + 2u], input[inp_hi + 3u]));

                acc += dot(vec4<f32>(f32(q0.y & 0xFu), f32((q0.y >> 8u) & 0xFu), f32((q0.y >> 16u) & 0xFu), f32((q0.y >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                          vec4<f32>(input[inp_lo + 4u], input[inp_lo + 5u], input[inp_lo + 6u], input[inp_lo + 7u]));
                acc += dot(vec4<f32>(f32((q0.y >> 4u) & 0xFu), f32((q0.y >> 12u) & 0xFu), f32((q0.y >> 20u) & 0xFu), f32((q0.y >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                          vec4<f32>(input[inp_hi + 4u], input[inp_hi + 5u], input[inp_hi + 6u], input[inp_hi + 7u]));

                acc += dot(vec4<f32>(f32(q0.z & 0xFu), f32((q0.z >> 8u) & 0xFu), f32((q0.z >> 16u) & 0xFu), f32((q0.z >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                          vec4<f32>(input[inp_lo + 8u], input[inp_lo + 9u], input[inp_lo + 10u], input[inp_lo + 11u]));
                acc += dot(vec4<f32>(f32((q0.z >> 4u) & 0xFu), f32((q0.z >> 12u) & 0xFu), f32((q0.z >> 20u) & 0xFu), f32((q0.z >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                          vec4<f32>(input[inp_hi + 8u], input[inp_hi + 9u], input[inp_hi + 10u], input[inp_hi + 11u]));

                acc += dot(vec4<f32>(f32(q0.w & 0xFu), f32((q0.w >> 8u) & 0xFu), f32((q0.w >> 16u) & 0xFu), f32((q0.w >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                          vec4<f32>(input[inp_lo + 12u], input[inp_lo + 13u], input[inp_lo + 14u], input[inp_lo + 15u]));
                acc += dot(vec4<f32>(f32((q0.w >> 4u) & 0xFu), f32((q0.w >> 12u) & 0xFu), f32((q0.w >> 20u) & 0xFu), f32((q0.w >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                          vec4<f32>(input[inp_hi + 12u], input[inp_hi + 13u], input[inp_hi + 14u], input[inp_hi + 15u]));

                acc += dot(vec4<f32>(f32(q1.x & 0xFu), f32((q1.x >> 8u) & 0xFu), f32((q1.x >> 16u) & 0xFu), f32((q1.x >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                          vec4<f32>(input[inp_lo + 16u], input[inp_lo + 17u], input[inp_lo + 18u], input[inp_lo + 19u]));
                acc += dot(vec4<f32>(f32((q1.x >> 4u) & 0xFu), f32((q1.x >> 12u) & 0xFu), f32((q1.x >> 20u) & 0xFu), f32((q1.x >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                          vec4<f32>(input[inp_hi + 16u], input[inp_hi + 17u], input[inp_hi + 18u], input[inp_hi + 19u]));

                acc += dot(vec4<f32>(f32(q1.y & 0xFu), f32((q1.y >> 8u) & 0xFu), f32((q1.y >> 16u) & 0xFu), f32((q1.y >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                          vec4<f32>(input[inp_lo + 20u], input[inp_lo + 21u], input[inp_lo + 22u], input[inp_lo + 23u]));
                acc += dot(vec4<f32>(f32((q1.y >> 4u) & 0xFu), f32((q1.y >> 12u) & 0xFu), f32((q1.y >> 20u) & 0xFu), f32((q1.y >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                          vec4<f32>(input[inp_hi + 20u], input[inp_hi + 21u], input[inp_hi + 22u], input[inp_hi + 23u]));

                acc += dot(vec4<f32>(f32(q1.z & 0xFu), f32((q1.z >> 8u) & 0xFu), f32((q1.z >> 16u) & 0xFu), f32((q1.z >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                          vec4<f32>(input[inp_lo + 24u], input[inp_lo + 25u], input[inp_lo + 26u], input[inp_lo + 27u]));
                acc += dot(vec4<f32>(f32((q1.z >> 4u) & 0xFu), f32((q1.z >> 12u) & 0xFu), f32((q1.z >> 20u) & 0xFu), f32((q1.z >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                          vec4<f32>(input[inp_hi + 24u], input[inp_hi + 25u], input[inp_hi + 26u], input[inp_hi + 27u]));

                acc += dot(vec4<f32>(f32(q1.w & 0xFu), f32((q1.w >> 8u) & 0xFu), f32((q1.w >> 16u) & 0xFu), f32((q1.w >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                          vec4<f32>(input[inp_lo + 28u], input[inp_lo + 29u], input[inp_lo + 30u], input[inp_lo + 31u]));
                acc += dot(vec4<f32>(f32((q1.w >> 4u) & 0xFu), f32((q1.w >> 12u) & 0xFu), f32((q1.w >> 20u) & 0xFu), f32((q1.w >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                          vec4<f32>(input[inp_hi + 28u], input[inp_hi + 29u], input[inp_hi + 30u], input[inp_hi + 31u]));
            }

            // --- Iteration 2: sub-blocks 4,5 ---
            {
                let d1 = d * scale4;  let m1 = dmin * min4;
                let d2 = d * scale5;  let m2 = dmin * min5;
                let inp_lo = input_base + k_base + 128u;
                let inp_hi = inp_lo + 32u;
                let qw = qs_word + 16u;

                let q0 = vec4<u32>(weights[qw], weights[qw + 1u], weights[qw + 2u], weights[qw + 3u]);
                let q1 = vec4<u32>(weights[qw + 4u], weights[qw + 5u], weights[qw + 6u], weights[qw + 7u]);

                acc += dot(vec4<f32>(f32(q0.x & 0xFu), f32((q0.x >> 8u) & 0xFu), f32((q0.x >> 16u) & 0xFu), f32((q0.x >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                          vec4<f32>(input[inp_lo], input[inp_lo + 1u], input[inp_lo + 2u], input[inp_lo + 3u]));
                acc += dot(vec4<f32>(f32((q0.x >> 4u) & 0xFu), f32((q0.x >> 12u) & 0xFu), f32((q0.x >> 20u) & 0xFu), f32((q0.x >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                          vec4<f32>(input[inp_hi], input[inp_hi + 1u], input[inp_hi + 2u], input[inp_hi + 3u]));

                acc += dot(vec4<f32>(f32(q0.y & 0xFu), f32((q0.y >> 8u) & 0xFu), f32((q0.y >> 16u) & 0xFu), f32((q0.y >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                          vec4<f32>(input[inp_lo + 4u], input[inp_lo + 5u], input[inp_lo + 6u], input[inp_lo + 7u]));
                acc += dot(vec4<f32>(f32((q0.y >> 4u) & 0xFu), f32((q0.y >> 12u) & 0xFu), f32((q0.y >> 20u) & 0xFu), f32((q0.y >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                          vec4<f32>(input[inp_hi + 4u], input[inp_hi + 5u], input[inp_hi + 6u], input[inp_hi + 7u]));

                acc += dot(vec4<f32>(f32(q0.z & 0xFu), f32((q0.z >> 8u) & 0xFu), f32((q0.z >> 16u) & 0xFu), f32((q0.z >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                          vec4<f32>(input[inp_lo + 8u], input[inp_lo + 9u], input[inp_lo + 10u], input[inp_lo + 11u]));
                acc += dot(vec4<f32>(f32((q0.z >> 4u) & 0xFu), f32((q0.z >> 12u) & 0xFu), f32((q0.z >> 20u) & 0xFu), f32((q0.z >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                          vec4<f32>(input[inp_hi + 8u], input[inp_hi + 9u], input[inp_hi + 10u], input[inp_hi + 11u]));

                acc += dot(vec4<f32>(f32(q0.w & 0xFu), f32((q0.w >> 8u) & 0xFu), f32((q0.w >> 16u) & 0xFu), f32((q0.w >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                          vec4<f32>(input[inp_lo + 12u], input[inp_lo + 13u], input[inp_lo + 14u], input[inp_lo + 15u]));
                acc += dot(vec4<f32>(f32((q0.w >> 4u) & 0xFu), f32((q0.w >> 12u) & 0xFu), f32((q0.w >> 20u) & 0xFu), f32((q0.w >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                          vec4<f32>(input[inp_hi + 12u], input[inp_hi + 13u], input[inp_hi + 14u], input[inp_hi + 15u]));

                acc += dot(vec4<f32>(f32(q1.x & 0xFu), f32((q1.x >> 8u) & 0xFu), f32((q1.x >> 16u) & 0xFu), f32((q1.x >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                          vec4<f32>(input[inp_lo + 16u], input[inp_lo + 17u], input[inp_lo + 18u], input[inp_lo + 19u]));
                acc += dot(vec4<f32>(f32((q1.x >> 4u) & 0xFu), f32((q1.x >> 12u) & 0xFu), f32((q1.x >> 20u) & 0xFu), f32((q1.x >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                          vec4<f32>(input[inp_hi + 16u], input[inp_hi + 17u], input[inp_hi + 18u], input[inp_hi + 19u]));

                acc += dot(vec4<f32>(f32(q1.y & 0xFu), f32((q1.y >> 8u) & 0xFu), f32((q1.y >> 16u) & 0xFu), f32((q1.y >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                          vec4<f32>(input[inp_lo + 20u], input[inp_lo + 21u], input[inp_lo + 22u], input[inp_lo + 23u]));
                acc += dot(vec4<f32>(f32((q1.y >> 4u) & 0xFu), f32((q1.y >> 12u) & 0xFu), f32((q1.y >> 20u) & 0xFu), f32((q1.y >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                          vec4<f32>(input[inp_hi + 20u], input[inp_hi + 21u], input[inp_hi + 22u], input[inp_hi + 23u]));

                acc += dot(vec4<f32>(f32(q1.z & 0xFu), f32((q1.z >> 8u) & 0xFu), f32((q1.z >> 16u) & 0xFu), f32((q1.z >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                          vec4<f32>(input[inp_lo + 24u], input[inp_lo + 25u], input[inp_lo + 26u], input[inp_lo + 27u]));
                acc += dot(vec4<f32>(f32((q1.z >> 4u) & 0xFu), f32((q1.z >> 12u) & 0xFu), f32((q1.z >> 20u) & 0xFu), f32((q1.z >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                          vec4<f32>(input[inp_hi + 24u], input[inp_hi + 25u], input[inp_hi + 26u], input[inp_hi + 27u]));

                acc += dot(vec4<f32>(f32(q1.w & 0xFu), f32((q1.w >> 8u) & 0xFu), f32((q1.w >> 16u) & 0xFu), f32((q1.w >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                          vec4<f32>(input[inp_lo + 28u], input[inp_lo + 29u], input[inp_lo + 30u], input[inp_lo + 31u]));
                acc += dot(vec4<f32>(f32((q1.w >> 4u) & 0xFu), f32((q1.w >> 12u) & 0xFu), f32((q1.w >> 20u) & 0xFu), f32((q1.w >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                          vec4<f32>(input[inp_hi + 28u], input[inp_hi + 29u], input[inp_hi + 30u], input[inp_hi + 31u]));
            }

            // --- Iteration 3: sub-blocks 6,7 ---
            {
                let d1 = d * scale6;  let m1 = dmin * min6;
                let d2 = d * scale7;  let m2 = dmin * min7;
                let inp_lo = input_base + k_base + 192u;
                let inp_hi = inp_lo + 32u;
                let qw = qs_word + 24u;

                let q0 = vec4<u32>(weights[qw], weights[qw + 1u], weights[qw + 2u], weights[qw + 3u]);
                let q1 = vec4<u32>(weights[qw + 4u], weights[qw + 5u], weights[qw + 6u], weights[qw + 7u]);

                acc += dot(vec4<f32>(f32(q0.x & 0xFu), f32((q0.x >> 8u) & 0xFu), f32((q0.x >> 16u) & 0xFu), f32((q0.x >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                          vec4<f32>(input[inp_lo], input[inp_lo + 1u], input[inp_lo + 2u], input[inp_lo + 3u]));
                acc += dot(vec4<f32>(f32((q0.x >> 4u) & 0xFu), f32((q0.x >> 12u) & 0xFu), f32((q0.x >> 20u) & 0xFu), f32((q0.x >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                          vec4<f32>(input[inp_hi], input[inp_hi + 1u], input[inp_hi + 2u], input[inp_hi + 3u]));

                acc += dot(vec4<f32>(f32(q0.y & 0xFu), f32((q0.y >> 8u) & 0xFu), f32((q0.y >> 16u) & 0xFu), f32((q0.y >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                          vec4<f32>(input[inp_lo + 4u], input[inp_lo + 5u], input[inp_lo + 6u], input[inp_lo + 7u]));
                acc += dot(vec4<f32>(f32((q0.y >> 4u) & 0xFu), f32((q0.y >> 12u) & 0xFu), f32((q0.y >> 20u) & 0xFu), f32((q0.y >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                          vec4<f32>(input[inp_hi + 4u], input[inp_hi + 5u], input[inp_hi + 6u], input[inp_hi + 7u]));

                acc += dot(vec4<f32>(f32(q0.z & 0xFu), f32((q0.z >> 8u) & 0xFu), f32((q0.z >> 16u) & 0xFu), f32((q0.z >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                          vec4<f32>(input[inp_lo + 8u], input[inp_lo + 9u], input[inp_lo + 10u], input[inp_lo + 11u]));
                acc += dot(vec4<f32>(f32((q0.z >> 4u) & 0xFu), f32((q0.z >> 12u) & 0xFu), f32((q0.z >> 20u) & 0xFu), f32((q0.z >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                          vec4<f32>(input[inp_hi + 8u], input[inp_hi + 9u], input[inp_hi + 10u], input[inp_hi + 11u]));

                acc += dot(vec4<f32>(f32(q0.w & 0xFu), f32((q0.w >> 8u) & 0xFu), f32((q0.w >> 16u) & 0xFu), f32((q0.w >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                          vec4<f32>(input[inp_lo + 12u], input[inp_lo + 13u], input[inp_lo + 14u], input[inp_lo + 15u]));
                acc += dot(vec4<f32>(f32((q0.w >> 4u) & 0xFu), f32((q0.w >> 12u) & 0xFu), f32((q0.w >> 20u) & 0xFu), f32((q0.w >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                          vec4<f32>(input[inp_hi + 12u], input[inp_hi + 13u], input[inp_hi + 14u], input[inp_hi + 15u]));

                acc += dot(vec4<f32>(f32(q1.x & 0xFu), f32((q1.x >> 8u) & 0xFu), f32((q1.x >> 16u) & 0xFu), f32((q1.x >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                          vec4<f32>(input[inp_lo + 16u], input[inp_lo + 17u], input[inp_lo + 18u], input[inp_lo + 19u]));
                acc += dot(vec4<f32>(f32((q1.x >> 4u) & 0xFu), f32((q1.x >> 12u) & 0xFu), f32((q1.x >> 20u) & 0xFu), f32((q1.x >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                          vec4<f32>(input[inp_hi + 16u], input[inp_hi + 17u], input[inp_hi + 18u], input[inp_hi + 19u]));

                acc += dot(vec4<f32>(f32(q1.y & 0xFu), f32((q1.y >> 8u) & 0xFu), f32((q1.y >> 16u) & 0xFu), f32((q1.y >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                          vec4<f32>(input[inp_lo + 20u], input[inp_lo + 21u], input[inp_lo + 22u], input[inp_lo + 23u]));
                acc += dot(vec4<f32>(f32((q1.y >> 4u) & 0xFu), f32((q1.y >> 12u) & 0xFu), f32((q1.y >> 20u) & 0xFu), f32((q1.y >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                          vec4<f32>(input[inp_hi + 20u], input[inp_hi + 21u], input[inp_hi + 22u], input[inp_hi + 23u]));

                acc += dot(vec4<f32>(f32(q1.z & 0xFu), f32((q1.z >> 8u) & 0xFu), f32((q1.z >> 16u) & 0xFu), f32((q1.z >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                          vec4<f32>(input[inp_lo + 24u], input[inp_lo + 25u], input[inp_lo + 26u], input[inp_lo + 27u]));
                acc += dot(vec4<f32>(f32((q1.z >> 4u) & 0xFu), f32((q1.z >> 12u) & 0xFu), f32((q1.z >> 20u) & 0xFu), f32((q1.z >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                          vec4<f32>(input[inp_hi + 24u], input[inp_hi + 25u], input[inp_hi + 26u], input[inp_hi + 27u]));

                acc += dot(vec4<f32>(f32(q1.w & 0xFu), f32((q1.w >> 8u) & 0xFu), f32((q1.w >> 16u) & 0xFu), f32((q1.w >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                          vec4<f32>(input[inp_lo + 28u], input[inp_lo + 29u], input[inp_lo + 30u], input[inp_lo + 31u]));
                acc += dot(vec4<f32>(f32((q1.w >> 4u) & 0xFu), f32((q1.w >> 12u) & 0xFu), f32((q1.w >> 20u) & 0xFu), f32((q1.w >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                          vec4<f32>(input[inp_hi + 28u], input[inp_hi + 29u], input[inp_hi + 30u], input[inp_hi + 31u]));
            }
        }
    }

    // Store partial sum to shared memory
    partial_sums[tid] = acc;
    workgroupBarrier();

    // Tree reduction within each group of THREADS_PER_ROW threads
    // Stride 8: threads 0-7 add from threads 8-15
    if (k_lane < 8u) {
        partial_sums[tid] = partial_sums[tid] + partial_sums[tid + 8u];
    }
    workgroupBarrier();

    // Stride 4
    if (k_lane < 4u) {
        partial_sums[tid] = partial_sums[tid] + partial_sums[tid + 4u];
    }
    workgroupBarrier();

    // Stride 2
    if (k_lane < 2u) {
        partial_sums[tid] = partial_sums[tid] + partial_sums[tid + 2u];
    }
    workgroupBarrier();

    // Stride 1
    if (k_lane < 1u) {
        partial_sums[tid] = partial_sums[tid] + partial_sums[tid + 1u];
    }
    workgroupBarrier();

    // Thread 0 of each group writes the final result
    if (k_lane == 0u && n < N) {
        output[b * N + n] = partial_sums[tid];
    }
}
