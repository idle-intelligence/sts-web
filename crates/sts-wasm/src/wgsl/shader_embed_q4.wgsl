// GPU Q4_0 Embedding Lookup Shader
//
// Dequantizes a single row from a Q4_0 embedding table on GPU.
// Each Q4_0 block: 2 bytes f16 scale + 16 bytes packed nibbles = 18 bytes, 32 values.
//
// Input:
//   token_buf[0] — u32 token ID (reinterpreted from f32 bits)
//   q4_data[]    — raw Q4_0 bytes packed as u32 array
//   info[0]      — dim (embedding dimension)
//   info[1]      — bytes_per_row (= (dim/32) * 18)
//   info[2]      — vocab_size
//
// Output:
//   output[0..dim-1] — f32 embedding vector (dequantized)
//
// Dispatch: ceil(dim/256) workgroups × 1 × 1, workgroup_size(256)
// Each thread handles one output element.

@group(0) @binding(0) var<storage, read> token_buf: array<u32>;
@group(0) @binding(1) var<storage, read> q4_data: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> info: array<u32>;

// Read a single byte from a u32 array at byte position `pos`.
fn read_byte(pos: u32) -> u32 {
    let word_idx = pos >> 2u;
    let byte_idx = pos & 3u;
    return (q4_data[word_idx] >> (byte_idx * 8u)) & 0xFFu;
}

// Convert f16 bits (u32 with only lower 16 bits valid) to f32.
fn f16_to_f32(bits: u32) -> f32 {
    let sign = (bits >> 15u) & 1u;
    let expo = (bits >> 10u) & 0x1Fu;
    let mant = bits & 0x3FFu;

    if (expo == 0u) {
        if (mant == 0u) {
            // Zero
            if (sign == 1u) { return -0.0; }
            return 0.0;
        }
        // Subnormal: value = (-1)^sign * 2^(-14) * (mant/1024)
        let f = f32(mant) / 1024.0 * 0.00006103515625; // 2^(-14)
        if (sign == 1u) { return -f; }
        return f;
    }
    if (expo == 31u) {
        // Inf or NaN — treat as 0 for safety
        return 0.0;
    }
    // Normal: value = (-1)^sign * 2^(expo-15) * (1 + mant/1024)
    let f = (1.0 + f32(mant) / 1024.0) * exp2(f32(expo) - 15.0);
    if (sign == 1u) { return -f; }
    return f;
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let elem = gid.x;
    let dim = info[0];
    let bytes_per_row = info[1];
    let vocab_size = info[2];

    if (elem >= dim) {
        return;
    }

    // Read token ID (stored as u32 bits in an f32-typed buffer by the argmax kernel)
    let token_id = token_buf[0];

    // Out-of-vocab → zero
    if (token_id >= vocab_size) {
        output[elem] = 0.0;
        return;
    }

    // Which Q4_0 block and position within it?
    let block_idx = elem / 32u;
    let pos_in_block = elem % 32u;

    // Byte offset of this block in the row
    let row_byte_offset = token_id * bytes_per_row;
    let block_byte_offset = row_byte_offset + block_idx * 18u;

    // Read f16 scale (2 bytes at block start)
    let scale_lo = read_byte(block_byte_offset);
    let scale_hi = read_byte(block_byte_offset + 1u);
    let scale_bits = scale_lo | (scale_hi << 8u);
    let scale = f16_to_f32(scale_bits);

    // Read the nibble for this element
    // Nibbles are packed: byte j (0..15) contains elements j (low nibble) and j+16 (high nibble)
    var nibble: u32;
    if (pos_in_block < 16u) {
        let byte_val = read_byte(block_byte_offset + 2u + pos_in_block);
        nibble = byte_val & 0xFu;
    } else {
        let byte_val = read_byte(block_byte_offset + 2u + pos_in_block - 16u);
        nibble = (byte_val >> 4u) & 0xFu;
    }

    // Dequantize: (nibble - 8) * scale
    let value = (f32(nibble) - 8.0) * scale;
    output[elem] = value;
}
