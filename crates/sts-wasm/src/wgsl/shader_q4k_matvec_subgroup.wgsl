// Q4_K Matrix-Vector Multiply with Subgroup Reduction (M=1)
//
// Uses subgroupAdd() to reduce partial sums across threads in a subgroup,
// replacing shared memory tree reduction with hardware SIMD operations.
//
// Strategy: Each subgroup (32 threads on Apple Silicon) cooperatively computes
// one output row by splitting the K dimension. Each thread processes K/32
// elements, then subgroupAdd() sums across the subgroup.
//
// With workgroup size 256 and subgroup size 32, we get 8 subgroups per workgroup,
// each computing one output row = 8 rows per workgroup.
//
// Computes: output[B, 1, N] = input[B, 1, K] x weights[N, K]^T
//
// Q4_K block layout (144 bytes = 36 u32s per 256 weights):
//   Bytes  0-1:   d     (f16 super-scale)
//   Bytes  2-3:   dmin  (f16 super-min)
//   Bytes  4-15:  scales (12 bytes = 3 u32s, packed 6-bit scale/min per sub-block)
//   Bytes 16-143: qs    (128 bytes = 32 u32s, packed 4-bit weights)

enable subgroups;

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> info: array<u32>;

const WG_SIZE: u32 = 256u;
const SUBGROUP_SIZE: u32 = 32u;
const ROWS_PER_WG: u32 = 8u;  // WG_SIZE / SUBGROUP_SIZE
const BLOCK_WORDS: u32 = 36u;  // 144 bytes / 4

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_id: u32,
) {
    let B = info[0];
    let K = info[2];
    let N = info[3];
    let blocks_per_row = info[4]; // K / 256

    let tid = local_id.x;
    let b = wg_id.y;     // batch index

    if (b >= B) {
        return;
    }

    // Which subgroup am I in within this workgroup?
    // Use compile-time constants (not runtime sg_size) to match dispatch grid.
    let sg_idx = tid / SUBGROUP_SIZE;
    // Which output row does my subgroup compute?
    let n = wg_id.x * ROWS_PER_WG + sg_idx;

    if (n >= N) {
        return;
    }

    let input_base = b * K;
    var acc: f32 = 0.0;

    // Each thread in the subgroup processes a subset of Q4K blocks for this row.
    // blocks_per_row = K / 256. Distribute blocks across subgroup threads.
    // Thread sg_id processes blocks: sg_id, sg_id + sg_size, sg_id + 2*sg_size, ...
    for (var blk: u32 = sg_id; blk < blocks_per_row; blk = blk + SUBGROUP_SIZE) {
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
            let k_lo = k_base;
            let k_hi = k_base + 32u;
            let qw = qs_word;

            let q0 = vec4<u32>(weights[qw], weights[qw + 1u], weights[qw + 2u], weights[qw + 3u]);
            let q1 = vec4<u32>(weights[qw + 4u], weights[qw + 5u], weights[qw + 6u], weights[qw + 7u]);

            acc += dot(vec4<f32>(f32(q0.x & 0xFu), f32((q0.x >> 8u) & 0xFu), f32((q0.x >> 16u) & 0xFu), f32((q0.x >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                      vec4<f32>(input[input_base + k_lo], input[input_base + k_lo + 1u], input[input_base + k_lo + 2u], input[input_base + k_lo + 3u]));
            acc += dot(vec4<f32>(f32((q0.x >> 4u) & 0xFu), f32((q0.x >> 12u) & 0xFu), f32((q0.x >> 20u) & 0xFu), f32((q0.x >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                      vec4<f32>(input[input_base + k_hi], input[input_base + k_hi + 1u], input[input_base + k_hi + 2u], input[input_base + k_hi + 3u]));

            acc += dot(vec4<f32>(f32(q0.y & 0xFu), f32((q0.y >> 8u) & 0xFu), f32((q0.y >> 16u) & 0xFu), f32((q0.y >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                      vec4<f32>(input[input_base + k_lo + 4u], input[input_base + k_lo + 5u], input[input_base + k_lo + 6u], input[input_base + k_lo + 7u]));
            acc += dot(vec4<f32>(f32((q0.y >> 4u) & 0xFu), f32((q0.y >> 12u) & 0xFu), f32((q0.y >> 20u) & 0xFu), f32((q0.y >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                      vec4<f32>(input[input_base + k_hi + 4u], input[input_base + k_hi + 5u], input[input_base + k_hi + 6u], input[input_base + k_hi + 7u]));

            acc += dot(vec4<f32>(f32(q0.z & 0xFu), f32((q0.z >> 8u) & 0xFu), f32((q0.z >> 16u) & 0xFu), f32((q0.z >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                      vec4<f32>(input[input_base + k_lo + 8u], input[input_base + k_lo + 9u], input[input_base + k_lo + 10u], input[input_base + k_lo + 11u]));
            acc += dot(vec4<f32>(f32((q0.z >> 4u) & 0xFu), f32((q0.z >> 12u) & 0xFu), f32((q0.z >> 20u) & 0xFu), f32((q0.z >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                      vec4<f32>(input[input_base + k_hi + 8u], input[input_base + k_hi + 9u], input[input_base + k_hi + 10u], input[input_base + k_hi + 11u]));

            acc += dot(vec4<f32>(f32(q0.w & 0xFu), f32((q0.w >> 8u) & 0xFu), f32((q0.w >> 16u) & 0xFu), f32((q0.w >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                      vec4<f32>(input[input_base + k_lo + 12u], input[input_base + k_lo + 13u], input[input_base + k_lo + 14u], input[input_base + k_lo + 15u]));
            acc += dot(vec4<f32>(f32((q0.w >> 4u) & 0xFu), f32((q0.w >> 12u) & 0xFu), f32((q0.w >> 20u) & 0xFu), f32((q0.w >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                      vec4<f32>(input[input_base + k_hi + 12u], input[input_base + k_hi + 13u], input[input_base + k_hi + 14u], input[input_base + k_hi + 15u]));

            acc += dot(vec4<f32>(f32(q1.x & 0xFu), f32((q1.x >> 8u) & 0xFu), f32((q1.x >> 16u) & 0xFu), f32((q1.x >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                      vec4<f32>(input[input_base + k_lo + 16u], input[input_base + k_lo + 17u], input[input_base + k_lo + 18u], input[input_base + k_lo + 19u]));
            acc += dot(vec4<f32>(f32((q1.x >> 4u) & 0xFu), f32((q1.x >> 12u) & 0xFu), f32((q1.x >> 20u) & 0xFu), f32((q1.x >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                      vec4<f32>(input[input_base + k_hi + 16u], input[input_base + k_hi + 17u], input[input_base + k_hi + 18u], input[input_base + k_hi + 19u]));

            acc += dot(vec4<f32>(f32(q1.y & 0xFu), f32((q1.y >> 8u) & 0xFu), f32((q1.y >> 16u) & 0xFu), f32((q1.y >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                      vec4<f32>(input[input_base + k_lo + 20u], input[input_base + k_lo + 21u], input[input_base + k_lo + 22u], input[input_base + k_lo + 23u]));
            acc += dot(vec4<f32>(f32((q1.y >> 4u) & 0xFu), f32((q1.y >> 12u) & 0xFu), f32((q1.y >> 20u) & 0xFu), f32((q1.y >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                      vec4<f32>(input[input_base + k_hi + 20u], input[input_base + k_hi + 21u], input[input_base + k_hi + 22u], input[input_base + k_hi + 23u]));

            acc += dot(vec4<f32>(f32(q1.z & 0xFu), f32((q1.z >> 8u) & 0xFu), f32((q1.z >> 16u) & 0xFu), f32((q1.z >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                      vec4<f32>(input[input_base + k_lo + 24u], input[input_base + k_lo + 25u], input[input_base + k_lo + 26u], input[input_base + k_lo + 27u]));
            acc += dot(vec4<f32>(f32((q1.z >> 4u) & 0xFu), f32((q1.z >> 12u) & 0xFu), f32((q1.z >> 20u) & 0xFu), f32((q1.z >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                      vec4<f32>(input[input_base + k_hi + 24u], input[input_base + k_hi + 25u], input[input_base + k_hi + 26u], input[input_base + k_hi + 27u]));

            acc += dot(vec4<f32>(f32(q1.w & 0xFu), f32((q1.w >> 8u) & 0xFu), f32((q1.w >> 16u) & 0xFu), f32((q1.w >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                      vec4<f32>(input[input_base + k_lo + 28u], input[input_base + k_lo + 29u], input[input_base + k_lo + 30u], input[input_base + k_lo + 31u]));
            acc += dot(vec4<f32>(f32((q1.w >> 4u) & 0xFu), f32((q1.w >> 12u) & 0xFu), f32((q1.w >> 20u) & 0xFu), f32((q1.w >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                      vec4<f32>(input[input_base + k_hi + 28u], input[input_base + k_hi + 29u], input[input_base + k_hi + 30u], input[input_base + k_hi + 31u]));
        }

        // --- Iteration 1: sub-blocks 2,3 ---
        {
            let d1 = d * scale2;  let m1 = dmin * min2;
            let d2 = d * scale3;  let m2 = dmin * min3;
            let k_lo = k_base + 64u;
            let k_hi = k_lo + 32u;
            let qw = qs_word + 8u;

            let q0 = vec4<u32>(weights[qw], weights[qw + 1u], weights[qw + 2u], weights[qw + 3u]);
            let q1 = vec4<u32>(weights[qw + 4u], weights[qw + 5u], weights[qw + 6u], weights[qw + 7u]);

            acc += dot(vec4<f32>(f32(q0.x & 0xFu), f32((q0.x >> 8u) & 0xFu), f32((q0.x >> 16u) & 0xFu), f32((q0.x >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                      vec4<f32>(input[input_base + k_lo], input[input_base + k_lo + 1u], input[input_base + k_lo + 2u], input[input_base + k_lo + 3u]));
            acc += dot(vec4<f32>(f32((q0.x >> 4u) & 0xFu), f32((q0.x >> 12u) & 0xFu), f32((q0.x >> 20u) & 0xFu), f32((q0.x >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                      vec4<f32>(input[input_base + k_hi], input[input_base + k_hi + 1u], input[input_base + k_hi + 2u], input[input_base + k_hi + 3u]));

            acc += dot(vec4<f32>(f32(q0.y & 0xFu), f32((q0.y >> 8u) & 0xFu), f32((q0.y >> 16u) & 0xFu), f32((q0.y >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                      vec4<f32>(input[input_base + k_lo + 4u], input[input_base + k_lo + 5u], input[input_base + k_lo + 6u], input[input_base + k_lo + 7u]));
            acc += dot(vec4<f32>(f32((q0.y >> 4u) & 0xFu), f32((q0.y >> 12u) & 0xFu), f32((q0.y >> 20u) & 0xFu), f32((q0.y >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                      vec4<f32>(input[input_base + k_hi + 4u], input[input_base + k_hi + 5u], input[input_base + k_hi + 6u], input[input_base + k_hi + 7u]));

            acc += dot(vec4<f32>(f32(q0.z & 0xFu), f32((q0.z >> 8u) & 0xFu), f32((q0.z >> 16u) & 0xFu), f32((q0.z >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                      vec4<f32>(input[input_base + k_lo + 8u], input[input_base + k_lo + 9u], input[input_base + k_lo + 10u], input[input_base + k_lo + 11u]));
            acc += dot(vec4<f32>(f32((q0.z >> 4u) & 0xFu), f32((q0.z >> 12u) & 0xFu), f32((q0.z >> 20u) & 0xFu), f32((q0.z >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                      vec4<f32>(input[input_base + k_hi + 8u], input[input_base + k_hi + 9u], input[input_base + k_hi + 10u], input[input_base + k_hi + 11u]));

            acc += dot(vec4<f32>(f32(q0.w & 0xFu), f32((q0.w >> 8u) & 0xFu), f32((q0.w >> 16u) & 0xFu), f32((q0.w >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                      vec4<f32>(input[input_base + k_lo + 12u], input[input_base + k_lo + 13u], input[input_base + k_lo + 14u], input[input_base + k_lo + 15u]));
            acc += dot(vec4<f32>(f32((q0.w >> 4u) & 0xFu), f32((q0.w >> 12u) & 0xFu), f32((q0.w >> 20u) & 0xFu), f32((q0.w >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                      vec4<f32>(input[input_base + k_hi + 12u], input[input_base + k_hi + 13u], input[input_base + k_hi + 14u], input[input_base + k_hi + 15u]));

            acc += dot(vec4<f32>(f32(q1.x & 0xFu), f32((q1.x >> 8u) & 0xFu), f32((q1.x >> 16u) & 0xFu), f32((q1.x >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                      vec4<f32>(input[input_base + k_lo + 16u], input[input_base + k_lo + 17u], input[input_base + k_lo + 18u], input[input_base + k_lo + 19u]));
            acc += dot(vec4<f32>(f32((q1.x >> 4u) & 0xFu), f32((q1.x >> 12u) & 0xFu), f32((q1.x >> 20u) & 0xFu), f32((q1.x >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                      vec4<f32>(input[input_base + k_hi + 16u], input[input_base + k_hi + 17u], input[input_base + k_hi + 18u], input[input_base + k_hi + 19u]));

            acc += dot(vec4<f32>(f32(q1.y & 0xFu), f32((q1.y >> 8u) & 0xFu), f32((q1.y >> 16u) & 0xFu), f32((q1.y >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                      vec4<f32>(input[input_base + k_lo + 20u], input[input_base + k_lo + 21u], input[input_base + k_lo + 22u], input[input_base + k_lo + 23u]));
            acc += dot(vec4<f32>(f32((q1.y >> 4u) & 0xFu), f32((q1.y >> 12u) & 0xFu), f32((q1.y >> 20u) & 0xFu), f32((q1.y >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                      vec4<f32>(input[input_base + k_hi + 20u], input[input_base + k_hi + 21u], input[input_base + k_hi + 22u], input[input_base + k_hi + 23u]));

            acc += dot(vec4<f32>(f32(q1.z & 0xFu), f32((q1.z >> 8u) & 0xFu), f32((q1.z >> 16u) & 0xFu), f32((q1.z >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                      vec4<f32>(input[input_base + k_lo + 24u], input[input_base + k_lo + 25u], input[input_base + k_lo + 26u], input[input_base + k_lo + 27u]));
            acc += dot(vec4<f32>(f32((q1.z >> 4u) & 0xFu), f32((q1.z >> 12u) & 0xFu), f32((q1.z >> 20u) & 0xFu), f32((q1.z >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                      vec4<f32>(input[input_base + k_hi + 24u], input[input_base + k_hi + 25u], input[input_base + k_hi + 26u], input[input_base + k_hi + 27u]));

            acc += dot(vec4<f32>(f32(q1.w & 0xFu), f32((q1.w >> 8u) & 0xFu), f32((q1.w >> 16u) & 0xFu), f32((q1.w >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                      vec4<f32>(input[input_base + k_lo + 28u], input[input_base + k_lo + 29u], input[input_base + k_lo + 30u], input[input_base + k_lo + 31u]));
            acc += dot(vec4<f32>(f32((q1.w >> 4u) & 0xFu), f32((q1.w >> 12u) & 0xFu), f32((q1.w >> 20u) & 0xFu), f32((q1.w >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                      vec4<f32>(input[input_base + k_hi + 28u], input[input_base + k_hi + 29u], input[input_base + k_hi + 30u], input[input_base + k_hi + 31u]));
        }

        // --- Iteration 2: sub-blocks 4,5 ---
        {
            let d1 = d * scale4;  let m1 = dmin * min4;
            let d2 = d * scale5;  let m2 = dmin * min5;
            let k_lo = k_base + 128u;
            let k_hi = k_lo + 32u;
            let qw = qs_word + 16u;

            let q0 = vec4<u32>(weights[qw], weights[qw + 1u], weights[qw + 2u], weights[qw + 3u]);
            let q1 = vec4<u32>(weights[qw + 4u], weights[qw + 5u], weights[qw + 6u], weights[qw + 7u]);

            acc += dot(vec4<f32>(f32(q0.x & 0xFu), f32((q0.x >> 8u) & 0xFu), f32((q0.x >> 16u) & 0xFu), f32((q0.x >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                      vec4<f32>(input[input_base + k_lo], input[input_base + k_lo + 1u], input[input_base + k_lo + 2u], input[input_base + k_lo + 3u]));
            acc += dot(vec4<f32>(f32((q0.x >> 4u) & 0xFu), f32((q0.x >> 12u) & 0xFu), f32((q0.x >> 20u) & 0xFu), f32((q0.x >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                      vec4<f32>(input[input_base + k_hi], input[input_base + k_hi + 1u], input[input_base + k_hi + 2u], input[input_base + k_hi + 3u]));

            acc += dot(vec4<f32>(f32(q0.y & 0xFu), f32((q0.y >> 8u) & 0xFu), f32((q0.y >> 16u) & 0xFu), f32((q0.y >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                      vec4<f32>(input[input_base + k_lo + 4u], input[input_base + k_lo + 5u], input[input_base + k_lo + 6u], input[input_base + k_lo + 7u]));
            acc += dot(vec4<f32>(f32((q0.y >> 4u) & 0xFu), f32((q0.y >> 12u) & 0xFu), f32((q0.y >> 20u) & 0xFu), f32((q0.y >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                      vec4<f32>(input[input_base + k_hi + 4u], input[input_base + k_hi + 5u], input[input_base + k_hi + 6u], input[input_base + k_hi + 7u]));

            acc += dot(vec4<f32>(f32(q0.z & 0xFu), f32((q0.z >> 8u) & 0xFu), f32((q0.z >> 16u) & 0xFu), f32((q0.z >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                      vec4<f32>(input[input_base + k_lo + 8u], input[input_base + k_lo + 9u], input[input_base + k_lo + 10u], input[input_base + k_lo + 11u]));
            acc += dot(vec4<f32>(f32((q0.z >> 4u) & 0xFu), f32((q0.z >> 12u) & 0xFu), f32((q0.z >> 20u) & 0xFu), f32((q0.z >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                      vec4<f32>(input[input_base + k_hi + 8u], input[input_base + k_hi + 9u], input[input_base + k_hi + 10u], input[input_base + k_hi + 11u]));

            acc += dot(vec4<f32>(f32(q0.w & 0xFu), f32((q0.w >> 8u) & 0xFu), f32((q0.w >> 16u) & 0xFu), f32((q0.w >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                      vec4<f32>(input[input_base + k_lo + 12u], input[input_base + k_lo + 13u], input[input_base + k_lo + 14u], input[input_base + k_lo + 15u]));
            acc += dot(vec4<f32>(f32((q0.w >> 4u) & 0xFu), f32((q0.w >> 12u) & 0xFu), f32((q0.w >> 20u) & 0xFu), f32((q0.w >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                      vec4<f32>(input[input_base + k_hi + 12u], input[input_base + k_hi + 13u], input[input_base + k_hi + 14u], input[input_base + k_hi + 15u]));

            acc += dot(vec4<f32>(f32(q1.x & 0xFu), f32((q1.x >> 8u) & 0xFu), f32((q1.x >> 16u) & 0xFu), f32((q1.x >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                      vec4<f32>(input[input_base + k_lo + 16u], input[input_base + k_lo + 17u], input[input_base + k_lo + 18u], input[input_base + k_lo + 19u]));
            acc += dot(vec4<f32>(f32((q1.x >> 4u) & 0xFu), f32((q1.x >> 12u) & 0xFu), f32((q1.x >> 20u) & 0xFu), f32((q1.x >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                      vec4<f32>(input[input_base + k_hi + 16u], input[input_base + k_hi + 17u], input[input_base + k_hi + 18u], input[input_base + k_hi + 19u]));

            acc += dot(vec4<f32>(f32(q1.y & 0xFu), f32((q1.y >> 8u) & 0xFu), f32((q1.y >> 16u) & 0xFu), f32((q1.y >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                      vec4<f32>(input[input_base + k_lo + 20u], input[input_base + k_lo + 21u], input[input_base + k_lo + 22u], input[input_base + k_lo + 23u]));
            acc += dot(vec4<f32>(f32((q1.y >> 4u) & 0xFu), f32((q1.y >> 12u) & 0xFu), f32((q1.y >> 20u) & 0xFu), f32((q1.y >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                      vec4<f32>(input[input_base + k_hi + 20u], input[input_base + k_hi + 21u], input[input_base + k_hi + 22u], input[input_base + k_hi + 23u]));

            acc += dot(vec4<f32>(f32(q1.z & 0xFu), f32((q1.z >> 8u) & 0xFu), f32((q1.z >> 16u) & 0xFu), f32((q1.z >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                      vec4<f32>(input[input_base + k_lo + 24u], input[input_base + k_lo + 25u], input[input_base + k_lo + 26u], input[input_base + k_lo + 27u]));
            acc += dot(vec4<f32>(f32((q1.z >> 4u) & 0xFu), f32((q1.z >> 12u) & 0xFu), f32((q1.z >> 20u) & 0xFu), f32((q1.z >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                      vec4<f32>(input[input_base + k_hi + 24u], input[input_base + k_hi + 25u], input[input_base + k_hi + 26u], input[input_base + k_hi + 27u]));

            acc += dot(vec4<f32>(f32(q1.w & 0xFu), f32((q1.w >> 8u) & 0xFu), f32((q1.w >> 16u) & 0xFu), f32((q1.w >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                      vec4<f32>(input[input_base + k_lo + 28u], input[input_base + k_lo + 29u], input[input_base + k_lo + 30u], input[input_base + k_lo + 31u]));
            acc += dot(vec4<f32>(f32((q1.w >> 4u) & 0xFu), f32((q1.w >> 12u) & 0xFu), f32((q1.w >> 20u) & 0xFu), f32((q1.w >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                      vec4<f32>(input[input_base + k_hi + 28u], input[input_base + k_hi + 29u], input[input_base + k_hi + 30u], input[input_base + k_hi + 31u]));
        }

        // --- Iteration 3: sub-blocks 6,7 ---
        {
            let d1 = d * scale6;  let m1 = dmin * min6;
            let d2 = d * scale7;  let m2 = dmin * min7;
            let k_lo = k_base + 192u;
            let k_hi = k_lo + 32u;
            let qw = qs_word + 24u;

            let q0 = vec4<u32>(weights[qw], weights[qw + 1u], weights[qw + 2u], weights[qw + 3u]);
            let q1 = vec4<u32>(weights[qw + 4u], weights[qw + 5u], weights[qw + 6u], weights[qw + 7u]);

            acc += dot(vec4<f32>(f32(q0.x & 0xFu), f32((q0.x >> 8u) & 0xFu), f32((q0.x >> 16u) & 0xFu), f32((q0.x >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                      vec4<f32>(input[input_base + k_lo], input[input_base + k_lo + 1u], input[input_base + k_lo + 2u], input[input_base + k_lo + 3u]));
            acc += dot(vec4<f32>(f32((q0.x >> 4u) & 0xFu), f32((q0.x >> 12u) & 0xFu), f32((q0.x >> 20u) & 0xFu), f32((q0.x >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                      vec4<f32>(input[input_base + k_hi], input[input_base + k_hi + 1u], input[input_base + k_hi + 2u], input[input_base + k_hi + 3u]));

            acc += dot(vec4<f32>(f32(q0.y & 0xFu), f32((q0.y >> 8u) & 0xFu), f32((q0.y >> 16u) & 0xFu), f32((q0.y >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                      vec4<f32>(input[input_base + k_lo + 4u], input[input_base + k_lo + 5u], input[input_base + k_lo + 6u], input[input_base + k_lo + 7u]));
            acc += dot(vec4<f32>(f32((q0.y >> 4u) & 0xFu), f32((q0.y >> 12u) & 0xFu), f32((q0.y >> 20u) & 0xFu), f32((q0.y >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                      vec4<f32>(input[input_base + k_hi + 4u], input[input_base + k_hi + 5u], input[input_base + k_hi + 6u], input[input_base + k_hi + 7u]));

            acc += dot(vec4<f32>(f32(q0.z & 0xFu), f32((q0.z >> 8u) & 0xFu), f32((q0.z >> 16u) & 0xFu), f32((q0.z >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                      vec4<f32>(input[input_base + k_lo + 8u], input[input_base + k_lo + 9u], input[input_base + k_lo + 10u], input[input_base + k_lo + 11u]));
            acc += dot(vec4<f32>(f32((q0.z >> 4u) & 0xFu), f32((q0.z >> 12u) & 0xFu), f32((q0.z >> 20u) & 0xFu), f32((q0.z >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                      vec4<f32>(input[input_base + k_hi + 8u], input[input_base + k_hi + 9u], input[input_base + k_hi + 10u], input[input_base + k_hi + 11u]));

            acc += dot(vec4<f32>(f32(q0.w & 0xFu), f32((q0.w >> 8u) & 0xFu), f32((q0.w >> 16u) & 0xFu), f32((q0.w >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                      vec4<f32>(input[input_base + k_lo + 12u], input[input_base + k_lo + 13u], input[input_base + k_lo + 14u], input[input_base + k_lo + 15u]));
            acc += dot(vec4<f32>(f32((q0.w >> 4u) & 0xFu), f32((q0.w >> 12u) & 0xFu), f32((q0.w >> 20u) & 0xFu), f32((q0.w >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                      vec4<f32>(input[input_base + k_hi + 12u], input[input_base + k_hi + 13u], input[input_base + k_hi + 14u], input[input_base + k_hi + 15u]));

            acc += dot(vec4<f32>(f32(q1.x & 0xFu), f32((q1.x >> 8u) & 0xFu), f32((q1.x >> 16u) & 0xFu), f32((q1.x >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                      vec4<f32>(input[input_base + k_lo + 16u], input[input_base + k_lo + 17u], input[input_base + k_lo + 18u], input[input_base + k_lo + 19u]));
            acc += dot(vec4<f32>(f32((q1.x >> 4u) & 0xFu), f32((q1.x >> 12u) & 0xFu), f32((q1.x >> 20u) & 0xFu), f32((q1.x >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                      vec4<f32>(input[input_base + k_hi + 16u], input[input_base + k_hi + 17u], input[input_base + k_hi + 18u], input[input_base + k_hi + 19u]));

            acc += dot(vec4<f32>(f32(q1.y & 0xFu), f32((q1.y >> 8u) & 0xFu), f32((q1.y >> 16u) & 0xFu), f32((q1.y >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                      vec4<f32>(input[input_base + k_lo + 20u], input[input_base + k_lo + 21u], input[input_base + k_lo + 22u], input[input_base + k_lo + 23u]));
            acc += dot(vec4<f32>(f32((q1.y >> 4u) & 0xFu), f32((q1.y >> 12u) & 0xFu), f32((q1.y >> 20u) & 0xFu), f32((q1.y >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                      vec4<f32>(input[input_base + k_hi + 20u], input[input_base + k_hi + 21u], input[input_base + k_hi + 22u], input[input_base + k_hi + 23u]));

            acc += dot(vec4<f32>(f32(q1.z & 0xFu), f32((q1.z >> 8u) & 0xFu), f32((q1.z >> 16u) & 0xFu), f32((q1.z >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                      vec4<f32>(input[input_base + k_lo + 24u], input[input_base + k_lo + 25u], input[input_base + k_lo + 26u], input[input_base + k_lo + 27u]));
            acc += dot(vec4<f32>(f32((q1.z >> 4u) & 0xFu), f32((q1.z >> 12u) & 0xFu), f32((q1.z >> 20u) & 0xFu), f32((q1.z >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                      vec4<f32>(input[input_base + k_hi + 24u], input[input_base + k_hi + 25u], input[input_base + k_hi + 26u], input[input_base + k_hi + 27u]));

            acc += dot(vec4<f32>(f32(q1.w & 0xFu), f32((q1.w >> 8u) & 0xFu), f32((q1.w >> 16u) & 0xFu), f32((q1.w >> 24u) & 0xFu)) * d1 - vec4<f32>(m1),
                      vec4<f32>(input[input_base + k_lo + 28u], input[input_base + k_lo + 29u], input[input_base + k_lo + 30u], input[input_base + k_lo + 31u]));
            acc += dot(vec4<f32>(f32((q1.w >> 4u) & 0xFu), f32((q1.w >> 12u) & 0xFu), f32((q1.w >> 20u) & 0xFu), f32((q1.w >> 28u) & 0xFu)) * d2 - vec4<f32>(m2),
                      vec4<f32>(input[input_base + k_hi + 28u], input[input_base + k_hi + 29u], input[input_base + k_hi + 30u], input[input_base + k_hi + 31u]));
        }
    }

    // Subgroup reduction: sum partial products across all threads in the subgroup
    let row_sum = subgroupAdd(acc);

    // Only the first thread in each subgroup writes the result
    if (sg_id == 0u) {
        output[b * N + n] = row_sum;
    }
}
