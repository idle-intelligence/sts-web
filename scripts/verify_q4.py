#!/usr/bin/env python3
"""Compare Q4_0 quantized GGUF tensors against original BF16 safetensors.

Dequantizes Q4_0 blocks and computes error metrics vs the BF16 originals.
No PyTorch needed — uses numpy + safetensors only.
"""

import struct
import numpy as np
from safetensors import safe_open
from pathlib import Path


def read_gguf_tensors(shard_paths):
    """Read GGUF tensor info and data from sharded files."""
    # Concatenate all shards
    all_data = bytearray()
    for p in shard_paths:
        all_data.extend(Path(p).read_bytes())
    data = bytes(all_data)

    off = 0
    magic = data[off:off+4]; off += 4
    version = struct.unpack_from('<I', data, off)[0]; off += 4
    n_tensors = struct.unpack_from('<Q', data, off)[0]; off += 8
    n_kv = struct.unpack_from('<Q', data, off)[0]; off += 8

    def read_string():
        nonlocal off
        length = struct.unpack_from('<Q', data, off)[0]; off += 8
        s = data[off:off+length].decode('utf-8'); off += length
        return s

    def skip_value(vtype):
        nonlocal off
        if vtype in (0, 1, 7): off += 1
        elif vtype in (2, 3): off += 2
        elif vtype in (4, 5, 6): off += 4
        elif vtype == 8: read_string()
        elif vtype == 9:
            atype = struct.unpack_from('<I', data, off)[0]; off += 4
            alen = struct.unpack_from('<Q', data, off)[0]; off += 8
            for _ in range(alen): skip_value(atype)
        elif vtype in (10, 11, 12): off += 8

    # Skip KV pairs
    for _ in range(n_kv):
        read_string()
        vtype = struct.unpack_from('<I', data, off)[0]; off += 4
        skip_value(vtype)

    # Read tensor info
    tensors = {}
    for _ in range(n_tensors):
        name = read_string()
        n_dims = struct.unpack_from('<I', data, off)[0]; off += 4
        dims = []
        for _ in range(n_dims):
            dims.append(struct.unpack_from('<Q', data, off)[0]); off += 8
        dtype = struct.unpack_from('<I', data, off)[0]; off += 4
        tensor_offset = struct.unpack_from('<Q', data, off)[0]; off += 8
        tensors[name] = {'dims': dims, 'dtype': dtype, 'offset': tensor_offset}

    # Data starts after header, aligned to 32 bytes
    header_end = off
    alignment = 32
    data_start = (header_end + alignment - 1) // alignment * alignment

    return tensors, data, data_start


def dequantize_q4_0(q4_bytes, shape):
    """Dequantize Q4_0 bytes to float32. shape = [rows, cols] in GGUF order."""
    rows, cols = shape[0], shape[1] if len(shape) > 1 else 1
    n_elements = rows * cols
    n_blocks = n_elements // 32
    block_size = 18  # 2 bytes scale (f16) + 16 bytes data (32 nibbles)

    result = np.zeros(n_elements, dtype=np.float32)

    for b in range(n_blocks):
        boff = b * block_size
        # Scale is float16 (2 bytes)
        scale = np.frombuffer(q4_bytes[boff:boff+2], dtype=np.float16)[0].astype(np.float32)
        # 16 bytes = 32 nibbles
        for j in range(16):
            byte_val = q4_bytes[boff + 2 + j]
            lo = (byte_val & 0x0F) - 8
            hi = ((byte_val >> 4) & 0x0F) - 8
            result[b * 32 + j] = lo * scale
            result[b * 32 + j + 16] = hi * scale

    return result.reshape(rows, cols) if len(shape) > 1 else result


def bf16_to_f32(bf16_bytes):
    """Convert BF16 bytes to float32 numpy array."""
    bf16 = np.frombuffer(bf16_bytes, dtype=np.uint16)
    # BF16 -> F32: shift left by 16 bits
    f32_bits = bf16.astype(np.uint32) << 16
    return f32_bits.view(np.float32)


def main():
    orig_path = "/Users/tc/Code/idle-intelligence/hf/personaplex-7b-v1/model.safetensors"
    gguf_dir = Path("/Users/tc/Code/idle-intelligence/hf/personaplex-7b-v1-q4_0-webgpu")

    # Find GGUF shards
    shard_paths = sorted(gguf_dir.glob("*.shard-*"))
    print(f"Loading {len(shard_paths)} GGUF shards...")
    gguf_tensors, gguf_data, gguf_data_start = read_gguf_tensors(shard_paths)
    print(f"GGUF: {len(gguf_tensors)} tensors")

    # Load original safetensors (lazy) - use raw bytes since numpy doesn't support BF16
    print(f"Opening original safetensors: {orig_path}")
    orig = safe_open(orig_path, framework="numpy", device="cpu")
    # Get keys by reading header
    from safetensors import SafetensorError
    import json
    with open(orig_path, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header_json = f.read(header_size).decode('utf-8')
    orig_header = json.loads(header_json)
    orig_keys = set(k for k in orig_header.keys() if k != '__metadata__')
    print(f"Original: {len(orig_keys)} tensors")

    # Compare a selection of tensors
    test_tensors = [
        # Temporal attention
        "transformer.layers.0.self_attn.in_proj_weight",
        "transformer.layers.0.self_attn.out_proj.weight",
        "transformer.layers.0.gating.linear_in.weight",
        "transformer.layers.15.self_attn.in_proj_weight",
        "transformer.layers.31.gating.linear_out.weight",
        # Depth attention
        "depformer.layers.0.self_attn.in_proj_weight",
        "depformer.layers.0.gating.0.linear_in.weight",
        "depformer.layers.5.self_attn.out_proj.weight",
        # Embeddings & heads
        "text_linear.weight",
        "linears.0.weight",
        "linears.7.weight",
        "depformer_in.0.weight",
        "depformer_in.7.weight",
        # Projections
        "emb.0.weight",
        "text_emb.weight",
    ]

    print(f"\n{'Tensor':<55} {'Shape':<20} {'MAE':>10} {'MaxErr':>10} {'RMSE':>10} {'Corr':>8}")
    print("-" * 120)

    errors = []
    for name in test_tensors:
        if name not in gguf_tensors:
            print(f"{name:<55} NOT IN GGUF")
            continue
        if name not in orig_keys:
            print(f"{name:<55} NOT IN ORIGINAL")
            continue

        info = gguf_tensors[name]
        # dtype 2 = Q4_0, dtype 0 = F32
        if info['dtype'] == 2:  # Q4_0
            shape = info['dims']  # GGUF stores [cols, rows] for 2D
            n_elements = 1
            for d in shape:
                n_elements *= d
            n_blocks = n_elements // 32
            n_bytes = n_blocks * 18
            q4_bytes = gguf_data[gguf_data_start + info['offset']:gguf_data_start + info['offset'] + n_bytes]
            q4_values = dequantize_q4_0(q4_bytes, shape)
        elif info['dtype'] == 0:  # F32
            shape = info['dims']
            n_elements = 1
            for d in shape:
                n_elements *= d
            f32_bytes = gguf_data[gguf_data_start + info['offset']:gguf_data_start + info['offset'] + n_elements * 4]
            q4_values = np.frombuffer(f32_bytes, dtype=np.float32).reshape(shape)
        else:
            print(f"{name:<55} UNSUPPORTED dtype={info['dtype']}")
            continue

        # Get original BF16 tensor — read raw bytes and convert manually
        orig_info = orig_header[name]
        orig_shape = orig_info['shape']
        orig_dtype = orig_info['dtype']
        offsets = orig_info['data_offsets']
        with open(orig_path, 'rb') as f:
            header_size_bytes = struct.unpack('<Q', f.read(8))[0]
            f.seek(8 + header_size_bytes + offsets[0])
            raw_bytes = f.read(offsets[1] - offsets[0])

        if orig_dtype == 'BF16':
            orig_tensor = bf16_to_f32(raw_bytes).reshape(orig_shape)
        elif orig_dtype == 'F32':
            orig_tensor = np.frombuffer(raw_bytes, dtype=np.float32).reshape(orig_shape)
        elif orig_dtype == 'F16':
            orig_tensor = np.frombuffer(raw_bytes, dtype=np.float16).astype(np.float32).reshape(orig_shape)
        else:
            print(f"{name:<55} UNSUPPORTED orig dtype={orig_dtype}")
            continue

        # GGUF stores transposed: [cols, rows] for weight matrices
        # Safetensors stores [rows, cols]
        # Compare the raw values regardless of layout
        q4_flat = q4_values.flatten().astype(np.float64)
        orig_flat = orig_tensor.flatten().astype(np.float64)

        if len(q4_flat) != len(orig_flat):
            print(f"{name:<55} SIZE MISMATCH: q4={len(q4_flat)} orig={len(orig_flat)}")
            continue

        diff = np.abs(q4_flat - orig_flat)
        mae = np.mean(diff)
        max_err = np.max(diff)
        rmse = np.sqrt(np.mean(diff**2))

        # Correlation
        if np.std(q4_flat) > 0 and np.std(orig_flat) > 0:
            corr = np.corrcoef(q4_flat, orig_flat)[0, 1]
        else:
            corr = 0.0

        shape_str = f"{list(orig_shape)}"
        print(f"{name:<55} {shape_str:<20} {mae:>10.6f} {max_err:>10.4f} {rmse:>10.6f} {corr:>8.4f}")
        errors.append((name, mae, max_err, rmse, corr))

    if errors:
        avg_mae = np.mean([e[1] for e in errors])
        avg_corr = np.mean([e[4] for e in errors])
        print(f"\n{'AVERAGE':<55} {'':<20} {avg_mae:>10.6f} {'':<10} {'':<10} {avg_corr:>8.4f}")

        # Flag any suspicious tensors
        for name, mae, max_err, rmse, corr in errors:
            if corr < 0.9:
                print(f"\n  WARNING: {name} has low correlation {corr:.4f} — quantization may be broken!")
            if mae > 1.0:
                print(f"\n  WARNING: {name} has high MAE {mae:.4f} — quantization may be broken!")


if __name__ == "__main__":
    main()
