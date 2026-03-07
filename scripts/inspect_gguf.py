#!/usr/bin/env python3
"""Inspect a GGUF file: list all tensors with their types and sizes."""

import struct
import sys
from pathlib import Path

GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
    18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
}

def read_string(f):
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8')

def skip_value(f, vtype):
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if vtype == 8:
        read_string(f)
    elif vtype == 9:
        elem_type = struct.unpack('<I', f.read(4))[0]
        count = struct.unpack('<Q', f.read(8))[0]
        for _ in range(count):
            skip_value(f, elem_type)
    elif vtype in sizes:
        f.read(sizes[vtype])
    else:
        raise ValueError(f"Unknown value type: {vtype}")

def read_metadata_value(f, vtype):
    if vtype == 4:  # UINT32
        return struct.unpack('<I', f.read(4))[0]
    elif vtype == 5:  # INT32
        return struct.unpack('<i', f.read(4))[0]
    elif vtype == 6:  # FLOAT32
        return struct.unpack('<f', f.read(4))[0]
    elif vtype == 8:  # STRING
        return read_string(f)
    elif vtype == 7:  # BOOL
        return bool(struct.unpack('<B', f.read(1))[0])
    elif vtype == 10:  # UINT64
        return struct.unpack('<Q', f.read(8))[0]
    elif vtype == 9:  # ARRAY
        elem_type = struct.unpack('<I', f.read(4))[0]
        count = struct.unpack('<Q', f.read(8))[0]
        return [read_metadata_value(f, elem_type) for _ in range(count)]
    else:
        skip_value(f, vtype)
        return f"<type {vtype}>"

def inspect_gguf(path):
    with open(path, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        assert magic == 0x46554747, f"Bad magic: {magic:#x}"
        version = struct.unpack('<I', f.read(4))[0]
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        metadata_count = struct.unpack('<Q', f.read(8))[0]

        print(f"GGUF v{version}, {tensor_count} tensors, {metadata_count} metadata entries")
        print()

        # Read metadata
        metadata = {}
        for _ in range(metadata_count):
            key = read_string(f)
            vtype = struct.unpack('<I', f.read(4))[0]
            value = read_metadata_value(f, vtype)
            metadata[key] = value
            print(f"  {key} = {value}")

        print()

        # Read tensor index
        type_counts = {}
        tensors = []
        for _ in range(tensor_count):
            name = read_string(f)
            ndims = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(ndims)]
            dtype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            type_name = GGML_TYPE_NAMES.get(dtype, f"?{dtype}")
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
            num_elements = 1
            for d in dims:
                num_elements *= d
            tensors.append((name, dims, type_name, num_elements))

        # Print summary
        print(f"Tensor type distribution:")
        for tname, count in sorted(type_counts.items()):
            print(f"  {tname}: {count} tensors")
        print()

        # Print all tensors
        for name, dims, type_name, num_elements in tensors:
            dims_str = "x".join(str(d) for d in dims)
            size_mb = num_elements * 4 / 1e6 if type_name == "F32" else num_elements * 2 / 1e6 if type_name == "F16" else num_elements / 2 / 1e6
            print(f"  {type_name:6s} [{dims_str:>20s}] {size_mb:8.2f} MB  {name}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/Users/tc/Code/idle-intelligence/hf/personaplex-7b-v1-q4_k/model-q4_k.gguf"
    inspect_gguf(path)
