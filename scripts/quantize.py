#!/usr/bin/env python3
"""Convert nvidia/personaplex-7b-v1 safetensors weights to Q4_K GGUF format.

Usage:
    python scripts/quantize.py [--model-dir DIR] [--output DIR] [--max-shard-size MB]

Reads model.safetensors from the model directory, quantizes weight matrices
to Q4_K (4-bit with super-block size 256), keeps norms as F32, embeddings
as Q4_0 (for CPU row lookups), and saves as sharded GGUF files for browser
inference (each shard < 2GB for WASM ArrayBuffer limit).

Two-pass approach for low memory usage:
  Pass 1: Scan safetensors header to build tensor index (names, shapes, types)
  Pass 2: Write GGUF header, then stream each tensor through quantization

Output: <output-dir>/personaplex-7b-v1-q4_k.gguf (+ shard files if needed)

## Tensor Summary (475 tensors, 8.37B params, all BF16)

**Temporal Transformer** (32 layers):
- `emb.{0-15}.weight` - [2049, 4096] (16 audio codebook embeddings, Q4_0)
- `text_emb.weight` - [32001, 4096] (Q4_0)
- `transformer.layers.{0-31}.norm1.alpha` - [1, 1, 4096] (F32)
- `transformer.layers.{i}.self_attn.in_proj_weight` - [12288, 4096] (Q4_K)
- `transformer.layers.{i}.self_attn.out_proj.weight` - [4096, 4096] (Q4_K)
- `transformer.layers.{i}.norm2.alpha` - [1, 1, 4096] (F32)
- `transformer.layers.{i}.gating.linear_in.weight` - [22528, 4096] (Q4_K)
- `transformer.layers.{i}.gating.linear_out.weight` - [4096, 11264] (Q4_K)
- `out_norm.alpha` - [1, 1, 4096] (F32)
- `text_linear.weight` - [32000, 4096] (Q4_K)

**Depth Transformer** (6 layers, multi-linear with 16 codebook-specific gatings):
- `depformer_emb.{0-14}.weight` - [2049, 1024] (Q4_0)
- `depformer_text_emb.weight` - [32001, 1024] (Q4_0)
- `depformer_in.{0-15}.weight` - [1024, 4096] (Q4_K)
- `depformer.layers.{0-5}.self_attn.in_proj_weight` - [49152, 1024] (Q4_K)
- `depformer.layers.{i}.self_attn.out_proj.weight` - [16384, 1024] (Q4_K)
- `depformer.layers.{i}.gating.{0-15}.linear_in.weight` - [5632, 1024] (Q4_K)
- `depformer.layers.{i}.gating.{0-15}.linear_out.weight` - [1024, 2816] (Q4_K)
- `depformer.layers.{i}.norm1.alpha` - [1, 1, 1024] (F32)
- `depformer.layers.{i}.norm2.alpha` - [1, 1, 1024] (F32)
- `linears.{0-15}.weight` - [2048, 1024] (Q4_K)

Quantization strategy:
- Q4_K: All weight matrices (in_proj, out_proj, gating, linears, text_linear)
- Q4_0: Embeddings (*_emb.*.weight, emb.*.weight) for CPU row-lookup efficiency
- F32: Norm alpha parameters only (*.alpha)
"""

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# GGUF constants
GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3
GGML_TYPE_F32 = 0
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_K = 12

# Q4_0 block size
Q4_BLOCK_SIZE = 32

# Q4_K constants
QK_K = 256          # super-block size (256 weights)
Q4K_BLOCK_BYTES = 144  # 2 (d) + 2 (dmin) + 12 (scales) + 128 (qs)


# ---------------------------------------------------------------------------
# BF16 conversion
# ---------------------------------------------------------------------------

def bf16_to_f32(bf16_bytes: bytes) -> np.ndarray:
    """Convert raw BF16 bytes to float32 numpy array.

    BF16 is the upper 16 bits of IEEE 754 float32, so we just shift left
    by 16 bits and reinterpret as float32.
    """
    bf16 = np.frombuffer(bf16_bytes, dtype=np.uint16)
    f32_bits = bf16.astype(np.uint32) << 16
    return f32_bits.view(np.float32)


# ---------------------------------------------------------------------------
# Q4_0 quantization
# ---------------------------------------------------------------------------

def quantize_q4_0(data: np.ndarray) -> bytes:
    """Quantize a float32 array to Q4_0 format.

    Q4_0 format: For each block of 32 values:
    - 1 x f16 scale factor (2 bytes)
    - 32 x 4-bit quantized values packed into 16 bytes
    - Total: 18 bytes per block

    Nibble packing: byte[j] = (element[j+16] + 8) << 4 | (element[j] + 8)
    """
    data = data.flatten().astype(np.float32)

    # Pad to multiple of 32
    remainder = len(data) % Q4_BLOCK_SIZE
    if remainder != 0:
        padding = Q4_BLOCK_SIZE - remainder
        data = np.pad(data, (0, padding), mode='constant', constant_values=0.0)

    num_blocks = len(data) // Q4_BLOCK_SIZE
    output = bytearray()

    for i in range(num_blocks):
        block = data[i * Q4_BLOCK_SIZE:(i + 1) * Q4_BLOCK_SIZE]

        # Scale factor = max absolute value / 7
        abs_max = np.abs(block).max()
        scale = abs_max / 7.0 if abs_max > 0 else 1.0

        # Quantize to 4-bit signed [-8, 7]
        quantized = np.round(block / scale).astype(np.int8)
        quantized = np.clip(quantized, -8, 7)

        # Pack scale as f16
        scale_f16 = np.float16(scale)
        output.extend(struct.pack('<e', scale_f16))

        # Pack nibbles: byte[j] = (element[j+16] + 8) << 4 | (element[j] + 8)
        for j in range(16):
            v_lo = (int(quantized[j]) + 8) & 0x0F
            v_hi = (int(quantized[j + 16]) + 8) & 0x0F
            output.append((v_hi << 4) | v_lo)

    return bytes(output)


def dequantize_q4_0(q4_bytes: bytes, num_elements: int) -> np.ndarray:
    """Dequantize Q4_0 bytes back to float32 (for verification)."""
    num_blocks = len(q4_bytes) // 18
    output = np.zeros(num_blocks * Q4_BLOCK_SIZE, dtype=np.float32)

    for i in range(num_blocks):
        offset = i * 18
        scale = np.frombuffer(q4_bytes[offset:offset + 2], dtype=np.float16)[0]
        scale = float(scale)

        for j in range(16):
            byte = q4_bytes[offset + 2 + j]
            v_lo = (byte & 0x0F) - 8
            v_hi = ((byte >> 4) & 0x0F) - 8
            output[i * Q4_BLOCK_SIZE + j] = v_lo * scale
            output[i * Q4_BLOCK_SIZE + j + 16] = v_hi * scale

    return output[:num_elements]


# ---------------------------------------------------------------------------
# Q4_K quantization
# ---------------------------------------------------------------------------

def quantize_q4_k(data: np.ndarray) -> bytes:
    """Quantize a float32 array to Q4_K format (vectorized).

    Q4_K format: For each super-block of 256 values:
    - 2 bytes: d (f16) - super-block scale
    - 2 bytes: dmin (f16) - super-block min
    - 12 bytes: packed scales and mins for 8 sub-blocks
    - 128 bytes: packed 4-bit quantized values (nibble pairs)
    - Total: 144 bytes per super-block
    """
    data = data.flatten().astype(np.float32)

    # Pad to multiple of 256
    remainder = len(data) % QK_K
    if remainder != 0:
        padding = QK_K - remainder
        data = np.pad(data, (0, padding), mode='constant', constant_values=0.0)

    num_blocks = len(data) // QK_K
    # Reshape: [num_blocks, 8, 32]
    blocks = data.reshape(num_blocks, 8, 32)

    # Step 1: Per-sub-block min/max → [num_blocks, 8]
    sub_mins = blocks.min(axis=2)
    sub_maxs = blocks.max(axis=2)
    raw_scale = np.where(sub_maxs > sub_mins, (sub_maxs - sub_mins) / 15.0, 0.0)
    raw_min_val = -sub_mins  # >= 0 when weights are negative

    # Step 2: Super-block d, dmin → [num_blocks]
    max_scale = raw_scale.max(axis=1)
    max_min_val = raw_min_val.max(axis=1)
    d_arr = np.where(max_scale > 0, max_scale / 63.0, 1.0)
    dmin_arr = np.where(max_min_val > 0, max_min_val / 63.0, 1.0)

    # Convert to f16 and back for exact match with dequant
    d_f16 = d_arr.astype(np.float16)
    dmin_f16 = dmin_arr.astype(np.float16)
    d_eff = d_f16.astype(np.float32)
    dmin_eff = dmin_f16.astype(np.float32)

    # Step 3: Quantize sub-block scales/mins to 6-bit → [num_blocks, 8]
    sc = np.where(d_eff[:, None] > 0,
                  np.clip(np.round(raw_scale / d_eff[:, None]), 0, 63),
                  0).astype(np.uint8)
    m = np.where(dmin_eff[:, None] > 0,
                 np.clip(np.round(raw_min_val / dmin_eff[:, None]), 0, 63),
                 0).astype(np.uint8)

    # Step 4: Quantize weights → [num_blocks, 8, 32]
    eff_scale = d_eff[:, None] * sc.astype(np.float32)    # [num_blocks, 8]
    eff_min = dmin_eff[:, None] * m.astype(np.float32)     # [num_blocks, 8]
    quants = np.where(eff_scale[:, :, None] > 0,
                      np.clip(np.round((blocks + eff_min[:, :, None]) / eff_scale[:, :, None]), 0, 15),
                      0).astype(np.uint8)

    # Step 5: Build output buffer
    output = bytearray(num_blocks * Q4K_BLOCK_BYTES)
    offset = 0

    for i in range(num_blocks):
        # d and dmin as f16
        struct.pack_into('<e', output, offset, d_f16[i])
        struct.pack_into('<e', output, offset + 2, dmin_f16[i])

        # Pack scales/mins into 12 bytes
        sci = sc[i]
        mi = m[i]
        sb = offset + 4
        output[sb + 0] = int((sci[0] & 63) | ((sci[4] >> 4) << 6))
        output[sb + 1] = int((sci[1] & 63) | ((sci[5] >> 4) << 6))
        output[sb + 2] = int((sci[2] & 63) | ((sci[6] >> 4) << 6))
        output[sb + 3] = int((sci[3] & 63) | ((sci[7] >> 4) << 6))
        output[sb + 4] = int((mi[0] & 63) | ((mi[4] >> 4) << 6))
        output[sb + 5] = int((mi[1] & 63) | ((mi[5] >> 4) << 6))
        output[sb + 6] = int((mi[2] & 63) | ((mi[6] >> 4) << 6))
        output[sb + 7] = int((mi[3] & 63) | ((mi[7] >> 4) << 6))
        output[sb + 8] = int((sci[4] & 0xF) | ((mi[4] & 0xF) << 4))
        output[sb + 9] = int((sci[5] & 0xF) | ((mi[5] & 0xF) << 4))
        output[sb + 10] = int((sci[6] & 0xF) | ((mi[6] & 0xF) << 4))
        output[sb + 11] = int((sci[7] & 0xF) | ((mi[7] & 0xF) << 4))

        # Pack nibbles: 4 iterations of 2 sub-blocks
        qi = quants[i]  # [8, 32]
        qo = offset + 16
        for it in range(4):
            lo = qi[2 * it]      # [32] low nibble sub-block
            hi = qi[2 * it + 1]  # [32] high nibble sub-block
            packed = (lo & 0xF) | ((hi & 0xF) << 4)
            output[qo + it * 32:qo + it * 32 + 32] = packed.tobytes()

        offset += Q4K_BLOCK_BYTES

    return bytes(output)


def get_scale_min_k4(j: int, scales_bytes: bytes) -> tuple:
    """Unpack scale and min for sub-block j from Q4_K packed scales."""
    if j < 4:
        sc = scales_bytes[j] & 63
        m = scales_bytes[j + 4] & 63
    else:
        sc = (scales_bytes[j + 4] & 0xF) | ((scales_bytes[j - 4] >> 6) << 4)
        m = (scales_bytes[j + 4] >> 4) | ((scales_bytes[j] >> 6) << 4)
    return sc, m


def dequantize_q4_k(q4k_bytes: bytes, num_elements: int) -> np.ndarray:
    """Dequantize Q4_K bytes back to float32 (for verification)."""
    num_blocks = len(q4k_bytes) // Q4K_BLOCK_BYTES
    output = np.zeros(num_blocks * QK_K, dtype=np.float32)

    for i in range(num_blocks):
        offset = i * Q4K_BLOCK_BYTES
        d = float(np.frombuffer(q4k_bytes[offset:offset + 2], dtype=np.float16)[0])
        dmin = float(np.frombuffer(q4k_bytes[offset + 2:offset + 4], dtype=np.float16)[0])
        scales_bytes = q4k_bytes[offset + 4:offset + 16]
        qs = q4k_bytes[offset + 16:offset + 144]

        for it in range(4):
            is_sb = 2 * it
            sc0, m0 = get_scale_min_k4(is_sb, scales_bytes)
            sc1, m1 = get_scale_min_k4(is_sb + 1, scales_bytes)

            d0 = d * sc0
            dm0 = dmin * m0
            d1 = d * sc1
            dm1 = dmin * m1

            for j in range(32):
                byte_val = qs[it * 32 + j]
                lo = byte_val & 0xF
                hi = (byte_val >> 4) & 0xF
                output[i * QK_K + is_sb * 32 + j] = lo * d0 - dm0
                output[i * QK_K + (is_sb + 1) * 32 + j] = hi * d1 - dm1

    return output[:num_elements]


# ---------------------------------------------------------------------------
# Safetensors header parsing
# ---------------------------------------------------------------------------

def parse_safetensors_header(path: Path) -> Dict:
    """Parse safetensors header without loading tensor data.

    Returns dict of {name: {"shape": [...], "dtype": str, "data_offsets": [start, end]}}.
    """
    with open(path, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))

    # Remove __metadata__ key if present
    header.pop('__metadata__', None)
    return header


def load_tensor_bf16(path: Path, name: str, header: Dict) -> np.ndarray:
    """Load a single tensor from safetensors as float32.

    Reads raw BF16 bytes and converts to float32 in-place.
    """
    info = header[name]
    start, end = info['data_offsets']
    shape = info['shape']

    # Header size prefix (8 bytes) + header JSON length
    with open(path, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        data_start = 8 + header_len

    with open(path, 'rb') as f:
        f.seek(data_start + start)
        raw = f.read(end - start)

    f32_data = bf16_to_f32(raw)
    return f32_data.reshape(shape)


# ---------------------------------------------------------------------------
# Quantization decisions
# ---------------------------------------------------------------------------

def is_embedding(name: str) -> bool:
    """Return True for embedding tensors that should stay Q4_0.

    Embeddings use Q4_0 for efficient CPU row-lookup (single-row dequant).
    Matches names containing '_emb.' or 'emb.' and ending with '.weight'.
    """
    if not name.endswith('.weight'):
        return False
    if '_emb.' in name or name.startswith('emb.'):
        return True
    return False


def should_quantize(name: str) -> bool:
    """Return True for weight matrices to quantize (Q4_K or Q4_0).

    Returns False for norm alpha parameters (keep as F32).
    """
    if '.alpha' in name:
        return False
    if name.endswith('.weight') or name.endswith('_weight'):
        return True
    return False


def q4_byte_size(num_elements: int) -> int:
    """Compute Q4_0 byte size for a given number of elements."""
    num_elements_padded = ((num_elements + Q4_BLOCK_SIZE - 1) // Q4_BLOCK_SIZE) * Q4_BLOCK_SIZE
    return (num_elements_padded // Q4_BLOCK_SIZE) * 18


def q4k_byte_size(num_elements: int) -> int:
    """Compute Q4_K byte size for a given number of elements."""
    num_elements_padded = ((num_elements + QK_K - 1) // QK_K) * QK_K
    return (num_elements_padded // QK_K) * Q4K_BLOCK_BYTES


# ---------------------------------------------------------------------------
# GGUF writing
# ---------------------------------------------------------------------------

def write_gguf_string(f, s: str):
    """Write a GGUF string (u64 length + UTF-8 bytes)."""
    encoded = s.encode('utf-8')
    f.write(struct.pack('<Q', len(encoded)))
    f.write(encoded)


def write_gguf_metadata(f, metadata: Dict):
    """Write GGUF metadata key-value pairs."""
    f.write(struct.pack('<Q', len(metadata)))

    for key, value in metadata.items():
        write_gguf_string(f, key)

        if isinstance(value, str):
            f.write(struct.pack('<I', 8))  # GGUF_TYPE_STRING
            write_gguf_string(f, value)
        elif isinstance(value, int):
            f.write(struct.pack('<I', 4))  # GGUF_TYPE_UINT32
            f.write(struct.pack('<I', value))
        elif isinstance(value, float):
            f.write(struct.pack('<I', 6))  # GGUF_TYPE_FLOAT32
            f.write(struct.pack('<f', value))
        else:
            raise ValueError(f"Unsupported metadata type: {type(value)}")


class TensorEntry:
    """Metadata for a tensor to be written to GGUF."""
    __slots__ = ('name', 'shape', 'ggml_type', 'num_elements', 'byte_size')

    def __init__(self, name: str, shape: List[int], ggml_type: int):
        self.name = name
        self.shape = shape
        self.ggml_type = ggml_type
        self.num_elements = 1
        for s in shape:
            self.num_elements *= s

        if ggml_type == GGML_TYPE_Q4_0:
            self.byte_size = q4_byte_size(self.num_elements)
        elif ggml_type == GGML_TYPE_Q4_K:
            self.byte_size = q4k_byte_size(self.num_elements)
        elif ggml_type == GGML_TYPE_F32:
            self.byte_size = self.num_elements * 4
        else:
            raise ValueError(f"Unsupported GGML type: {ggml_type}")


def write_gguf_header(f, entries: List[TensorEntry], metadata: Dict):
    """Write GGUF v3 header with tensor index."""
    # Magic and version
    f.write(struct.pack('<I', GGUF_MAGIC))
    f.write(struct.pack('<I', GGUF_VERSION))

    # Tensor count and metadata
    f.write(struct.pack('<Q', len(entries)))
    write_gguf_metadata(f, metadata)

    # Tensor index
    offset = 0
    for entry in entries:
        write_gguf_string(f, entry.name)

        # Number of dimensions
        ndims = len(entry.shape)
        f.write(struct.pack('<I', ndims))

        # Dimensions (reversed for GGUF: innermost first)
        for dim in reversed(entry.shape):
            f.write(struct.pack('<Q', dim))

        # GGML type
        f.write(struct.pack('<I', entry.ggml_type))

        # Offset from start of data section
        f.write(struct.pack('<Q', offset))

        offset += entry.byte_size

    # Align to 32 bytes
    header_end = f.tell()
    alignment = 32
    padding = (alignment - (header_end % alignment)) % alignment
    f.write(b'\x00' * padding)


# ---------------------------------------------------------------------------
# Main quantization pipeline
# ---------------------------------------------------------------------------

def build_tensor_index(safetensors_header: Dict) -> List[TensorEntry]:
    """Build sorted list of TensorEntry from safetensors header.

    Determines quantization type for each tensor:
    - Norms (.alpha) -> F32
    - Embeddings (*_emb.*, emb.*) -> Q4_0 (for CPU row lookups)
    - All other weights -> Q4_K

    Returns entries sorted by name.
    """
    entries = []
    for name in sorted(safetensors_header.keys()):
        info = safetensors_header[name]
        shape = info['shape']

        if not should_quantize(name):
            ggml_type = GGML_TYPE_F32
        elif is_embedding(name):
            ggml_type = GGML_TYPE_Q4_0
        else:
            ggml_type = GGML_TYPE_Q4_K

        entries.append(TensorEntry(name, shape, ggml_type))

    return entries


def build_metadata() -> Dict:
    """Build GGUF metadata for PersonaPlex-7B."""
    return {
        "general.architecture": "personaplex",
        "general.name": "personaplex-7b-v1",
        # Temporal transformer
        "personaplex.context_length": 4096,
        "personaplex.embedding_length": 4096,
        "personaplex.block_count": 32,
        "personaplex.feed_forward_length": 11264,
        "personaplex.attention.head_count": 32,
        "personaplex.text_vocab_size": 32001,
        "personaplex.audio_vocab_size": 2049,
        "personaplex.num_codebooks": 16,
        "personaplex.rope.freq_base": 10000.0,
        # Depth transformer
        "personaplex.depformer.embedding_length": 1024,
        "personaplex.depformer.block_count": 6,
        "personaplex.depformer.feed_forward_length": 2816,
        "personaplex.depformer.attention.head_count": 16,
        "personaplex.depformer.num_codebooks": 16,
    }


def quantize_model(model_dir: Path, output_dir: Path, max_shard_mb: int):
    """Two-pass quantization: scan index, then stream-write GGUF."""
    safetensors_path = model_dir / "model.safetensors"
    if not safetensors_path.exists():
        print(f"Error: {safetensors_path} not found", file=sys.stderr)
        sys.exit(1)

    # --- Pass 1: Build tensor index from header ---
    print("Pass 1: Scanning safetensors header...")
    st_header = parse_safetensors_header(safetensors_path)
    entries = build_tensor_index(st_header)

    q4k_count = sum(1 for e in entries if e.ggml_type == GGML_TYPE_Q4_K)
    q4_count = sum(1 for e in entries if e.ggml_type == GGML_TYPE_Q4_0)
    f32_count = sum(1 for e in entries if e.ggml_type == GGML_TYPE_F32)
    total_data = sum(e.byte_size for e in entries)
    print(f"  {len(entries)} tensors: {q4k_count} Q4_K, {q4_count} Q4_0, {f32_count} F32")
    print(f"  Estimated output size: {total_data / (1024**3):.2f} GB")

    # --- Pass 2: Write GGUF with streaming quantization ---
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "personaplex-7b-v1-q4_k.gguf"

    print(f"\nPass 2: Writing GGUF to {output_path}")
    metadata = build_metadata()

    with open(output_path, 'wb') as f:
        # Write header
        write_gguf_header(f, entries, metadata)
        data_start = f.tell()
        print(f"  Header size: {data_start} bytes")

        # Stream each tensor: load BF16 -> convert F32 -> quantize -> write
        for idx, entry in enumerate(entries):
            tensor_data = load_tensor_bf16(safetensors_path, entry.name, st_header)

            if entry.ggml_type == GGML_TYPE_Q4_K:
                data = quantize_q4_k(tensor_data)
                tag = "Q4_K"
            elif entry.ggml_type == GGML_TYPE_Q4_0:
                data = quantize_q4_0(tensor_data)
                tag = "Q4_0"
            else:
                data = tensor_data.flatten().astype(np.float32).tobytes()
                tag = "F32"

            assert len(data) == entry.byte_size, \
                f"Size mismatch for {entry.name}: expected {entry.byte_size}, got {len(data)}"

            f.write(data)

            if (idx + 1) % 50 == 0 or idx == len(entries) - 1:
                pct = (idx + 1) / len(entries) * 100
                print(f"  [{pct:5.1f}%] {idx + 1}/{len(entries)} tensors written")

    file_size = output_path.stat().st_size
    file_size_gb = file_size / (1024**3)
    print(f"\nWrote {file_size_gb:.2f} GB to {output_path}")

    # --- Shard if needed ---
    max_shard_bytes = max_shard_mb * 1024 * 1024
    if file_size > max_shard_bytes:
        print(f"\nSharding into <{max_shard_mb} MB chunks...")
        shard_dir = output_dir
        shard_prefix = "personaplex-7b-v1-q4_k"

        with open(output_path, 'rb') as f_in:
            shard_idx = 0
            while True:
                chunk = f_in.read(max_shard_bytes)
                if not chunk:
                    break

                shard_name = f"{shard_prefix}.gguf.shard-{shard_idx:02d}"
                shard_path = shard_dir / shard_name

                with open(shard_path, 'wb') as f_out:
                    f_out.write(chunk)

                shard_size_mb = len(chunk) / (1024**2)
                print(f"  {shard_name}: {shard_size_mb:.1f} MB")
                shard_idx += 1

        # Remove the monolithic file after sharding
        output_path.unlink()
        print(f"Created {shard_idx} shards, removed monolithic GGUF")

    # --- Copy tokenizer files ---
    tokenizer_src = model_dir / "tokenizer-e351c8d8-checkpoint125.safetensors"
    if tokenizer_src.exists():
        import shutil
        tokenizer_dst = output_dir / tokenizer_src.name
        if not tokenizer_dst.exists():
            shutil.copy2(tokenizer_src, tokenizer_dst)
            print(f"Copied {tokenizer_src.name} ({tokenizer_src.stat().st_size / (1024**2):.0f} MB)")

    spm_src = model_dir / "tokenizer_spm_32k_3.model"
    if spm_src.exists():
        import shutil
        spm_dst = output_dir / spm_src.name
        if not spm_dst.exists():
            shutil.copy2(spm_src, spm_dst)
            print(f"Copied {spm_src.name}")

    print("\nDone! Quantization complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Quantize PersonaPlex-7B to Q4_K GGUF for browser inference"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/Users/tc/Code/idle-intelligence/hf/personaplex-7b-v1"),
        help="Directory containing model.safetensors",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/Users/tc/Code/idle-intelligence/hf/personaplex-7b-v1-q4_k-webgpu"),
        help="Output directory for quantized GGUF files",
    )
    parser.add_argument(
        "--max-shard-size",
        type=int,
        default=512,
        help="Maximum shard size in MB (default: 512)",
    )

    args = parser.parse_args()
    quantize_model(args.model_dir, args.output, args.max_shard_size)


if __name__ == "__main__":
    main()
