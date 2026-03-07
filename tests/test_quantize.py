"""Tests for the Q4_0 GGUF quantization pipeline.

Tests cover:
- BF16 to F32 conversion
- Q4_0 quantization round-trip accuracy
- Q4_0 byte size calculations
- Tensor classification (quantize vs F32)
- GGUF header writing and parsing
- End-to-end mini-model quantization
"""

import io
import json
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add scripts dir to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from quantize import (
    GGML_TYPE_F32,
    GGML_TYPE_Q4_0,
    GGUF_MAGIC,
    GGUF_VERSION,
    Q4_BLOCK_SIZE,
    TensorEntry,
    bf16_to_f32,
    build_tensor_index,
    dequantize_q4_0,
    q4_byte_size,
    quantize_q4_0,
    should_quantize,
    write_gguf_header,
    write_gguf_string,
)


# ---------------------------------------------------------------------------
# BF16 conversion
# ---------------------------------------------------------------------------

class TestBF16Conversion:
    def test_zeros(self):
        """BF16 zero converts to F32 zero."""
        bf16_bytes = np.array([0], dtype=np.uint16).tobytes()
        result = bf16_to_f32(bf16_bytes)
        assert result[0] == 0.0

    def test_one(self):
        """BF16 representation of 1.0 converts correctly."""
        # 1.0 in float32 = 0x3F800000, BF16 = upper 16 bits = 0x3F80
        bf16_bytes = np.array([0x3F80], dtype=np.uint16).tobytes()
        result = bf16_to_f32(bf16_bytes)
        assert result[0] == 1.0

    def test_negative(self):
        """BF16 representation of -1.0 converts correctly."""
        # -1.0 in float32 = 0xBF800000, BF16 = 0xBF80
        bf16_bytes = np.array([0xBF80], dtype=np.uint16).tobytes()
        result = bf16_to_f32(bf16_bytes)
        assert result[0] == -1.0

    def test_small_value(self):
        """BF16 representation of 0.5 converts correctly."""
        # 0.5 in float32 = 0x3F000000, BF16 = 0x3F00
        bf16_bytes = np.array([0x3F00], dtype=np.uint16).tobytes()
        result = bf16_to_f32(bf16_bytes)
        assert result[0] == 0.5

    def test_batch(self):
        """Multiple BF16 values convert correctly."""
        # [1.0, -1.0, 0.0, 0.5]
        bf16_vals = np.array([0x3F80, 0xBF80, 0x0000, 0x3F00], dtype=np.uint16)
        result = bf16_to_f32(bf16_vals.tobytes())
        np.testing.assert_array_equal(result, [1.0, -1.0, 0.0, 0.5])

    def test_roundtrip_via_float32(self):
        """F32 -> BF16 -> F32 roundtrip preserves values approximately."""
        original = np.array([3.14, -2.71, 100.0, 0.001], dtype=np.float32)
        # Convert F32 to BF16 (truncate lower 16 bits)
        bf16_bits = (original.view(np.uint32) >> 16).astype(np.uint16)
        recovered = bf16_to_f32(bf16_bits.tobytes())
        # BF16 has ~3 decimal digits of precision
        np.testing.assert_allclose(recovered, original, rtol=1e-2)


# ---------------------------------------------------------------------------
# Q4_0 quantization
# ---------------------------------------------------------------------------

class TestQ4Quantization:
    def test_zeros(self):
        """Zeros quantize and dequantize to zeros."""
        data = np.zeros(64, dtype=np.float32)
        q4 = quantize_q4_0(data)
        assert len(q4) == 2 * 18  # 2 blocks of 32
        result = dequantize_q4_0(q4, 64)
        np.testing.assert_array_equal(result, np.zeros(64))

    def test_block_size(self):
        """Q4_0 produces 18 bytes per block of 32 elements."""
        data = np.random.randn(32).astype(np.float32)
        q4 = quantize_q4_0(data)
        assert len(q4) == 18

    def test_multiple_blocks(self):
        """Multiple blocks produce correct total byte count."""
        data = np.random.randn(128).astype(np.float32)
        q4 = quantize_q4_0(data)
        assert len(q4) == 4 * 18  # 128 / 32 = 4 blocks

    def test_padding(self):
        """Non-multiple-of-32 inputs are padded correctly."""
        data = np.random.randn(50).astype(np.float32)
        q4 = quantize_q4_0(data)
        # 50 -> padded to 64 -> 2 blocks
        assert len(q4) == 2 * 18

    def test_roundtrip_accuracy(self):
        """Q4_0 roundtrip error is within expected bounds.

        For Q4_0 with 4-bit quantization (16 levels), the max relative error
        per element is about 1/7 ~= 14.3%. We test that RMS error is reasonable.
        """
        np.random.seed(42)
        data = np.random.randn(1024).astype(np.float32)
        q4 = quantize_q4_0(data)
        recovered = dequantize_q4_0(q4, 1024)

        # RMS error should be small relative to the data range
        rms_error = np.sqrt(np.mean((data - recovered) ** 2))
        data_rms = np.sqrt(np.mean(data ** 2))
        relative_error = rms_error / data_rms
        # Q4_0 typically has ~5-8% relative RMS error on Gaussian data
        assert relative_error < 0.15, f"Relative RMS error too high: {relative_error:.3f}"

    def test_max_values_preserved(self):
        """The maximum magnitude value in each block is well-preserved."""
        np.random.seed(123)
        data = np.random.randn(32).astype(np.float32)
        q4 = quantize_q4_0(data)
        recovered = dequantize_q4_0(q4, 32)

        # The value with max magnitude should be close to original
        max_idx = np.argmax(np.abs(data))
        if abs(data[max_idx]) > 1e-6:
            rel_err = abs(data[max_idx] - recovered[max_idx]) / abs(data[max_idx])
            assert rel_err < 0.15, f"Max value relative error: {rel_err:.3f}"

    def test_nibble_packing_order(self):
        """Verify the GGML nibble packing order: lo=0-15, hi=16-31."""
        # Create a block where we know exact quantized values
        data = np.zeros(32, dtype=np.float32)
        data[0] = 7.0   # Should be quantized to 7 -> nibble = 15 = 0xF
        data[16] = -7.0  # Should be quantized to -7 -> nibble = 1

        q4 = quantize_q4_0(data)

        # Scale = 7.0 / 7.0 = 1.0
        scale = np.frombuffer(q4[:2], dtype=np.float16)[0]
        assert abs(float(scale) - 1.0) < 0.01

        # First data byte: lo nibble for element 0, hi nibble for element 16
        byte0 = q4[2]
        lo = byte0 & 0x0F       # element[0] + 8 = 7 + 8 = 15 = 0xF
        hi = (byte0 >> 4) & 0x0F  # element[16] + 8 = -7 + 8 = 1
        assert lo == 15, f"Expected lo nibble 15, got {lo}"
        assert hi == 1, f"Expected hi nibble 1, got {hi}"

    def test_large_tensor(self):
        """Test quantization of a larger tensor (simulating a weight matrix)."""
        np.random.seed(0)
        # Simulate a [4096, 4096] weight matrix row
        data = np.random.randn(4096).astype(np.float32) * 0.02
        q4 = quantize_q4_0(data)
        expected_blocks = 4096 // 32
        assert len(q4) == expected_blocks * 18
        recovered = dequantize_q4_0(q4, 4096)
        assert recovered.shape == (4096,)


# ---------------------------------------------------------------------------
# Byte size calculations
# ---------------------------------------------------------------------------

class TestByteSizes:
    def test_exact_multiple(self):
        """Exact multiple of 32 gives correct size."""
        assert q4_byte_size(32) == 18
        assert q4_byte_size(64) == 36
        assert q4_byte_size(1024) == 1024 // 32 * 18

    def test_needs_padding(self):
        """Non-multiple of 32 is padded up."""
        assert q4_byte_size(33) == 2 * 18  # Padded to 64
        assert q4_byte_size(1) == 18       # Padded to 32

    def test_large_tensor_sizes(self):
        """Size calculations for actual PersonaPlex tensor shapes."""
        # transformer.layers.*.self_attn.in_proj_weight: [12288, 4096]
        assert q4_byte_size(12288 * 4096) == (12288 * 4096 // 32) * 18

        # transformer.layers.*.gating.linear_in.weight: [22528, 4096]
        assert q4_byte_size(22528 * 4096) == (22528 * 4096 // 32) * 18

        # emb.*.weight: [2049, 4096]
        # 2049 * 4096 = 8,392,704. Not divisible by 32, needs padding.
        n = 2049 * 4096
        padded = ((n + 31) // 32) * 32
        assert q4_byte_size(n) == (padded // 32) * 18


# ---------------------------------------------------------------------------
# Tensor classification
# ---------------------------------------------------------------------------

class TestShouldQuantize:
    def test_weight_tensors(self):
        """Weight matrices should be quantized."""
        assert should_quantize("transformer.layers.0.self_attn.in_proj_weight") is True
        assert should_quantize("transformer.layers.0.self_attn.out_proj.weight") is True
        assert should_quantize("transformer.layers.0.gating.linear_in.weight") is True
        assert should_quantize("transformer.layers.0.gating.linear_out.weight") is True
        assert should_quantize("text_linear.weight") is True
        assert should_quantize("text_emb.weight") is True
        assert should_quantize("emb.0.weight") is True
        assert should_quantize("linears.0.weight") is True

    def test_depformer_weights(self):
        """Depformer weight matrices should be quantized."""
        assert should_quantize("depformer.layers.0.self_attn.in_proj_weight") is True
        assert should_quantize("depformer.layers.0.self_attn.out_proj.weight") is True
        assert should_quantize("depformer.layers.0.gating.0.linear_in.weight") is True
        assert should_quantize("depformer.layers.0.gating.15.linear_out.weight") is True
        assert should_quantize("depformer_emb.0.weight") is True
        assert should_quantize("depformer_text_emb.weight") is True
        assert should_quantize("depformer_in.0.weight") is True

    def test_norm_tensors(self):
        """Norm alpha parameters should NOT be quantized."""
        assert should_quantize("transformer.layers.0.norm1.alpha") is False
        assert should_quantize("transformer.layers.0.norm2.alpha") is False
        assert should_quantize("out_norm.alpha") is False
        assert should_quantize("depformer.layers.0.norm1.alpha") is False
        assert should_quantize("depformer.layers.0.norm2.alpha") is False


# ---------------------------------------------------------------------------
# TensorEntry
# ---------------------------------------------------------------------------

class TestTensorEntry:
    def test_q4_entry(self):
        """Q4 tensor entry computes correct byte size."""
        entry = TensorEntry("test.weight", [4096, 4096], GGML_TYPE_Q4_0)
        assert entry.num_elements == 4096 * 4096
        assert entry.byte_size == q4_byte_size(4096 * 4096)

    def test_f32_entry(self):
        """F32 tensor entry computes correct byte size."""
        entry = TensorEntry("test.alpha", [1, 1, 4096], GGML_TYPE_F32)
        assert entry.num_elements == 4096
        assert entry.byte_size == 4096 * 4

    def test_3d_shape(self):
        """3D tensor shape (norm alpha) works correctly."""
        entry = TensorEntry("norm.alpha", [1, 1, 1024], GGML_TYPE_F32)
        assert entry.num_elements == 1024
        assert entry.byte_size == 1024 * 4


# ---------------------------------------------------------------------------
# GGUF header writing
# ---------------------------------------------------------------------------

class TestGGUFHeader:
    def test_magic_and_version(self):
        """GGUF header starts with correct magic and version."""
        buf = io.BytesIO()
        entries = [TensorEntry("test.weight", [32, 32], GGML_TYPE_Q4_0)]
        write_gguf_header(buf, entries, {})

        buf.seek(0)
        magic = struct.unpack('<I', buf.read(4))[0]
        version = struct.unpack('<I', buf.read(4))[0]
        assert magic == GGUF_MAGIC
        assert version == GGUF_VERSION

    def test_tensor_count(self):
        """GGUF header has correct tensor count."""
        buf = io.BytesIO()
        entries = [
            TensorEntry("a.weight", [32, 32], GGML_TYPE_Q4_0),
            TensorEntry("b.alpha", [32], GGML_TYPE_F32),
            TensorEntry("c.weight", [64, 32], GGML_TYPE_Q4_0),
        ]
        write_gguf_header(buf, entries, {})

        buf.seek(0)
        buf.read(8)  # Skip magic + version
        tensor_count = struct.unpack('<Q', buf.read(8))[0]
        assert tensor_count == 3

    def test_alignment(self):
        """Header is padded to 32-byte alignment."""
        buf = io.BytesIO()
        entries = [TensorEntry("t.weight", [32], GGML_TYPE_Q4_0)]
        write_gguf_header(buf, entries, {"key": "value"})

        header_size = buf.tell()
        assert header_size % 32 == 0

    def test_metadata(self):
        """Metadata key-value pairs are written correctly."""
        buf = io.BytesIO()
        metadata = {
            "general.architecture": "test",
            "test.dim": 4096,
            "test.theta": 100000.0,
        }
        entries = [TensorEntry("t.weight", [32], GGML_TYPE_Q4_0)]
        write_gguf_header(buf, entries, metadata)

        buf.seek(0)
        buf.read(8)  # magic + version
        buf.read(8)  # tensor count

        # Metadata KV count
        kv_count = struct.unpack('<Q', buf.read(8))[0]
        assert kv_count == 3

    def test_tensor_offsets_sequential(self):
        """Tensor offsets in the index are sequential."""
        buf = io.BytesIO()
        entries = [
            TensorEntry("a.weight", [64, 64], GGML_TYPE_Q4_0),
            TensorEntry("b.alpha", [64], GGML_TYPE_F32),
        ]
        write_gguf_header(buf, entries, {})

        # Parse back: skip to tensor index
        buf.seek(0)
        buf.read(8)  # magic + version
        buf.read(8)  # tensor count
        buf.read(8)  # metadata kv count = 0

        # First tensor
        name_len = struct.unpack('<Q', buf.read(8))[0]
        buf.read(name_len)
        ndims = struct.unpack('<I', buf.read(4))[0]
        for _ in range(ndims):
            buf.read(8)
        buf.read(4)  # ggml type
        offset1 = struct.unpack('<Q', buf.read(8))[0]

        # Second tensor
        name_len2 = struct.unpack('<Q', buf.read(8))[0]
        buf.read(name_len2)
        ndims2 = struct.unpack('<I', buf.read(4))[0]
        for _ in range(ndims2):
            buf.read(8)
        buf.read(4)  # ggml type
        offset2 = struct.unpack('<Q', buf.read(8))[0]

        assert offset1 == 0
        assert offset2 == entries[0].byte_size


# ---------------------------------------------------------------------------
# Build tensor index
# ---------------------------------------------------------------------------

class TestBuildTensorIndex:
    def test_classifies_correctly(self):
        """build_tensor_index classifies weights vs norms correctly."""
        st_header = {
            "transformer.layers.0.self_attn.in_proj_weight": {
                "shape": [12288, 4096], "dtype": "BF16", "data_offsets": [0, 100]
            },
            "transformer.layers.0.norm1.alpha": {
                "shape": [1, 1, 4096], "dtype": "BF16", "data_offsets": [100, 200]
            },
        }
        entries = build_tensor_index(st_header)
        assert len(entries) == 2

        # Sorted by name: norm1.alpha comes first
        assert entries[0].name == "transformer.layers.0.norm1.alpha"
        assert entries[0].ggml_type == GGML_TYPE_F32

        assert entries[1].name == "transformer.layers.0.self_attn.in_proj_weight"
        assert entries[1].ggml_type == GGML_TYPE_Q4_0


# ---------------------------------------------------------------------------
# End-to-end mini model test
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def _create_mini_safetensors(self, tmp_path: Path) -> Path:
        """Create a minimal safetensors file with BF16 tensors."""
        # We need to create a valid safetensors file with BF16 data
        # Safetensors format: u64 header_len + JSON header + tensor data
        tensors = {
            "test.weight": {
                "shape": [64, 32],
                "data": np.random.randn(64, 32).astype(np.float32),
            },
            "test_norm.alpha": {
                "shape": [1, 1, 32],
                "data": np.ones((1, 1, 32), dtype=np.float32),
            },
        }

        # Convert to BF16 bytes
        data_parts = {}
        offset = 0
        for name, info in sorted(tensors.items()):
            bf16_bits = (info["data"].view(np.uint32).flatten() >> 16).astype(np.uint16)
            raw = bf16_bits.tobytes()
            data_parts[name] = {
                "raw": raw,
                "start": offset,
                "end": offset + len(raw),
                "shape": info["shape"],
            }
            offset += len(raw)

        # Build header
        header = {}
        for name, info in sorted(data_parts.items()):
            header[name] = {
                "dtype": "BF16",
                "shape": info["shape"],
                "data_offsets": [info["start"], info["end"]],
            }

        header_json = json.dumps(header).encode('utf-8')
        header_len = len(header_json)

        # Write file
        sf_path = tmp_path / "model.safetensors"
        with open(sf_path, 'wb') as f:
            f.write(struct.pack('<Q', header_len))
            f.write(header_json)
            for name in sorted(data_parts.keys()):
                f.write(data_parts[name]["raw"])

        return sf_path, tensors

    def test_end_to_end_quantize(self, tmp_path):
        """Full pipeline: safetensors -> GGUF with Q4_0 quantization."""
        sf_path, original_tensors = self._create_mini_safetensors(tmp_path)

        # Parse header
        from quantize import parse_safetensors_header, load_tensor_bf16, build_metadata

        st_header = parse_safetensors_header(sf_path)
        assert len(st_header) == 2

        # Build tensor index
        entries = build_tensor_index(st_header)
        assert len(entries) == 2

        # Write GGUF
        output_path = tmp_path / "test.gguf"
        metadata = {"general.architecture": "test"}

        with open(output_path, 'wb') as f:
            write_gguf_header(f, entries, metadata)
            for entry in entries:
                tensor_data = load_tensor_bf16(sf_path, entry.name, st_header)
                if entry.ggml_type == GGML_TYPE_Q4_0:
                    data = quantize_q4_0(tensor_data)
                else:
                    data = tensor_data.flatten().astype(np.float32).tobytes()
                assert len(data) == entry.byte_size
                f.write(data)

        # Verify GGUF file is valid
        with open(output_path, 'rb') as f:
            magic = struct.unpack('<I', f.read(4))[0]
            version = struct.unpack('<I', f.read(4))[0]
            tensor_count = struct.unpack('<Q', f.read(8))[0]

        assert magic == GGUF_MAGIC
        assert version == GGUF_VERSION
        assert tensor_count == 2
        assert output_path.stat().st_size > 0

    def test_bf16_load_accuracy(self, tmp_path):
        """Loading BF16 tensors produces correct F32 values."""
        sf_path, original_tensors = self._create_mini_safetensors(tmp_path)
        from quantize import parse_safetensors_header, load_tensor_bf16

        st_header = parse_safetensors_header(sf_path)

        for name, info in original_tensors.items():
            loaded = load_tensor_bf16(sf_path, name, st_header)
            # BF16 truncates lower 16 bits, so we compare with BF16 precision
            original_bf16 = info["data"].astype(np.float32)
            bf16_bits = (original_bf16.view(np.uint32) >> 16).astype(np.uint16)
            expected = (bf16_bits.astype(np.uint32) << 16).view(np.float32)

            np.testing.assert_array_equal(
                loaded.flatten(), expected.flatten(),
                err_msg=f"BF16 load mismatch for {name}"
            )
