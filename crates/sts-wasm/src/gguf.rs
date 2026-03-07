//! Q4 GGUF weight loader and WGSL dequantization shaders.
//!
//! Pipeline: GGUF file -> parse header/tensors -> store Q4 blocks as raw bytes
//! on GPU -> dequantize via WGSL compute shader -> matmul.
//!
//! Forked from stt-web. Key patterns:
//! - `ShardedCursor`: Read+Seek over Vec<Vec<u8>> for multi-shard GGUF
//!   (stays under 2GB per-allocation limit in WASM)
//! - Two-phase loading: parse GGUF, drop reader, finalize tensors
//!   (stays under 4GB address space)
//! - Naive WGSL kernel for WASM (tiled kernel is native-only)

use anyhow::{bail, ensure, Context, Result};
use burn::backend::wgpu::{
    into_contiguous, AutoCompiler, CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate,
    WgpuDevice, WgpuRuntime,
};
use burn::backend::Wgpu;
use burn::tensor::{DType, Tensor, TensorData, TensorPrimitive};
use byteorder::{LittleEndian, ReadBytesExt};
use cubecl::prelude::KernelId;
use cubecl::server::{Bindings, CubeCount, Handle};
use cubecl::{CubeTask, Runtime};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" little-endian
const ALIGNMENT: u64 = 32;

// Naive kernel workgroup sizes (16x16 = 256, the WebGPU limit)
const NAIVE_WG_X: u32 = 16;
const NAIVE_WG_Y: u32 = 16;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert IEEE 754 half-precision (f16) bits to f32.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            f32::from_bits(sign << 31)
        } else {
            let mut e = 1u32;
            let mut m = mantissa;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            m &= 0x3FF;
            let f32_exp = 127u32.wrapping_sub(15 + e - 1);
            f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13))
        }
    } else if exponent == 31 {
        f32::from_bits((sign << 31) | (0xFF << 23) | (mantissa << 13))
    } else {
        let f32_exp = (exponent as i32 - 15 + 127) as u32;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (mantissa << 13))
    }
}

fn align_up(offset: u64, alignment: u64) -> u64 {
    offset.div_ceil(alignment) * alignment
}

pub fn reverse_gguf_dims(gguf_dims: &[u64]) -> Vec<usize> {
    gguf_dims.iter().rev().map(|&d| d as usize).collect()
}

// ---------------------------------------------------------------------------
// GGUF String / Value helpers
// ---------------------------------------------------------------------------

fn read_gguf_string<R: Read>(reader: &mut R) -> Result<String> {
    let len = reader.read_u64::<LittleEndian>()? as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    String::from_utf8(buf).context("Invalid UTF-8 in GGUF string")
}

fn skip_gguf_value<R: Read + Seek>(reader: &mut R, value_type: u32) -> Result<()> {
    match value_type {
        0 => { reader.read_u8()?; }
        1 => { reader.read_i8()?; }
        2 => { reader.seek(SeekFrom::Current(2))?; }
        3 => { reader.seek(SeekFrom::Current(2))?; }
        4 => { reader.seek(SeekFrom::Current(4))?; }
        5 => { reader.seek(SeekFrom::Current(4))?; }
        6 => { reader.seek(SeekFrom::Current(4))?; }
        7 => { reader.read_u8()?; }
        8 => { let _ = read_gguf_string(reader)?; }
        9 => {
            let elem_type = reader.read_u32::<LittleEndian>()?;
            let count = reader.read_u64::<LittleEndian>()?;
            for _ in 0..count {
                skip_gguf_value(reader, elem_type)?;
            }
        }
        10 => { reader.seek(SeekFrom::Current(8))?; }
        11 => { reader.seek(SeekFrom::Current(8))?; }
        12 => { reader.seek(SeekFrom::Current(8))?; }
        other => bail!("Unknown GGUF metadata value type: {other}"),
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// GgmlDtype
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum GgmlDtype {
    F32,
    F16,
    Q4_0,
    Q4_K,
}

impl GgmlDtype {
    fn from_u32(v: u32) -> Result<Self> {
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            12 => Ok(Self::Q4_K),
            other => bail!("Unsupported GGML dtype code: {other}"),
        }
    }

    pub fn byte_size(&self, num_elements: u64) -> u64 {
        match self {
            Self::F32 => num_elements * 4,
            Self::F16 => num_elements * 2,
            Self::Q4_0 => {
                let num_blocks = num_elements / 32;
                num_blocks * 18
            }
            Self::Q4_K => {
                let num_blocks = num_elements / 256;
                num_blocks * 144
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GgufTensorInfo
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    dimensions: Vec<u64>,
    dtype: GgmlDtype,
    offset: u64,
}

impl GgufTensorInfo {
    pub fn shape(&self) -> &[u64] {
        &self.dimensions
    }

    pub fn dtype(&self) -> GgmlDtype {
        self.dtype
    }

    pub fn num_elements(&self) -> u64 {
        self.dimensions.iter().product()
    }

    pub fn byte_size(&self) -> u64 {
        self.dtype.byte_size(self.num_elements())
    }

    pub fn offset(&self) -> u64 {
        self.offset
    }
}

// ---------------------------------------------------------------------------
// GgufReader
// ---------------------------------------------------------------------------

pub struct GgufReader<R: Read + Seek> {
    reader: R,
    version: u32,
    tensor_count: u64,
    tensors: HashMap<String, GgufTensorInfo>,
    data_section_offset: u64,
}

impl<R: Read + Seek> GgufReader<R> {
    pub fn open(mut reader: R) -> Result<Self> {
        let magic = reader
            .read_u32::<LittleEndian>()
            .context("Failed to read GGUF magic")?;
        if magic != GGUF_MAGIC {
            bail!("Invalid GGUF magic: 0x{magic:08X} (expected 0x{GGUF_MAGIC:08X})");
        }

        let version = reader
            .read_u32::<LittleEndian>()
            .context("Failed to read GGUF version")?;
        if version != 2 && version != 3 {
            bail!("Unsupported GGUF version: {version} (expected 2 or 3)");
        }

        let tensor_count = reader
            .read_u64::<LittleEndian>()
            .context("Failed to read tensor count")?;
        let metadata_kv_count = reader
            .read_u64::<LittleEndian>()
            .context("Failed to read metadata KV count")?;

        for i in 0..metadata_kv_count {
            let _key = read_gguf_string(&mut reader)
                .with_context(|| format!("Failed to read metadata key {i}"))?;
            let value_type = reader
                .read_u32::<LittleEndian>()
                .with_context(|| format!("Failed to read metadata value type {i}"))?;
            skip_gguf_value(&mut reader, value_type)
                .with_context(|| format!("Failed to skip metadata value {i}"))?;
        }

        let mut tensors = HashMap::with_capacity(tensor_count as usize);
        for i in 0..tensor_count {
            let name = read_gguf_string(&mut reader)
                .with_context(|| format!("Failed to read tensor name {i}"))?;
            let ndims = reader
                .read_u32::<LittleEndian>()
                .with_context(|| format!("Failed to read ndims for tensor {i}"))?;
            let mut dimensions = Vec::with_capacity(ndims as usize);
            for d in 0..ndims {
                dimensions.push(
                    reader
                        .read_u64::<LittleEndian>()
                        .with_context(|| format!("Failed to read dim {d} for tensor {i}"))?,
                );
            }
            let dtype = GgmlDtype::from_u32(
                reader
                    .read_u32::<LittleEndian>()
                    .with_context(|| format!("Failed to read dtype for tensor {i}"))?,
            )?;
            let offset = reader
                .read_u64::<LittleEndian>()
                .with_context(|| format!("Failed to read offset for tensor {i}"))?;

            tensors.insert(
                name.clone(),
                GgufTensorInfo {
                    name,
                    dimensions,
                    dtype,
                    offset,
                },
            );
        }

        let current_pos = reader.stream_position()?;
        let data_section_offset = align_up(current_pos, ALIGNMENT);

        Ok(Self {
            reader,
            version,
            tensor_count,
            tensors,
            data_section_offset,
        })
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn tensor_count(&self) -> u64 {
        self.tensor_count
    }

    pub fn tensor_info(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.get(name)
    }

    pub fn tensor_data(&mut self, name: &str) -> Result<Vec<u8>> {
        let info = self
            .tensors
            .get(name)
            .with_context(|| format!("Tensor '{name}' not found in GGUF"))?
            .clone();
        let byte_size = info.byte_size() as usize;
        let abs_offset = self.data_section_offset + info.offset;
        self.reader.seek(SeekFrom::Start(abs_offset))?;
        let mut buf = vec![0u8; byte_size];
        self.reader.read_exact(&mut buf)?;
        Ok(buf)
    }
}

// ---------------------------------------------------------------------------
// ShardedCursor — Read + Seek over multiple buffers
// ---------------------------------------------------------------------------

/// A cursor that provides `Read + Seek` over multiple contiguous byte buffers.
///
/// Each shard is kept as a separate `Vec<u8>` to stay under the WASM32
/// `isize::MAX` (~2 GB) per-allocation limit while supporting total sizes > 2 GB.
pub struct ShardedCursor {
    shards: Vec<Vec<u8>>,
    ends: Vec<u64>,
    pos: u64,
    total_len: u64,
}

impl ShardedCursor {
    pub fn new(shards: Vec<Vec<u8>>) -> Self {
        let mut ends = Vec::with_capacity(shards.len());
        let mut total: u64 = 0;
        for s in &shards {
            total += s.len() as u64;
            ends.push(total);
        }
        Self {
            shards,
            ends,
            pos: 0,
            total_len: total,
        }
    }

    fn shard_for_offset(&self, offset: u64) -> Option<(usize, usize)> {
        if offset >= self.total_len {
            return None;
        }
        let shard_idx = self.ends.partition_point(|&end| end <= offset);
        let shard_start = if shard_idx > 0 {
            self.ends[shard_idx - 1]
        } else {
            0
        };
        Some((shard_idx, (offset - shard_start) as usize))
    }
}

impl Read for ShardedCursor {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.pos >= self.total_len {
            return Ok(0);
        }
        let (shard_idx, local_offset) = self.shard_for_offset(self.pos).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "no shard found for offset {} (total_len={})",
                    self.pos, self.total_len
                ),
            )
        })?;
        let shard = &self.shards[shard_idx];
        let available = shard.len() - local_offset;
        let to_read = buf.len().min(available);
        buf[..to_read].copy_from_slice(&shard[local_offset..local_offset + to_read]);
        self.pos += to_read as u64;
        Ok(to_read)
    }
}

impl Seek for ShardedCursor {
    fn seek(&mut self, style: SeekFrom) -> std::io::Result<u64> {
        let new_pos = match style {
            SeekFrom::Start(offset) => offset as i64,
            SeekFrom::End(offset) => self.total_len as i64 + offset,
            SeekFrom::Current(offset) => self.pos as i64 + offset,
        };
        if new_pos < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "seek to negative position",
            ));
        }
        self.pos = new_pos as u64;
        Ok(self.pos)
    }
}

// ---------------------------------------------------------------------------
// Q4Tensor — GPU buffer of Q4_0 blocks
// ---------------------------------------------------------------------------

/// A Q4_0 quantized weight tensor living on GPU.
pub struct Q4Tensor {
    pub(crate) handle: Handle,
    shape: [usize; 2],
    num_blocks: usize,
}

impl Q4Tensor {
    pub fn from_q4_bytes(raw_bytes: &[u8], shape: [usize; 2], device: &WgpuDevice) -> Result<Self> {
        let [n, k] = shape;
        let num_elements = k * n;
        ensure!(
            num_elements % 32 == 0,
            "Q4_0 requires element count divisible by 32, got {num_elements}"
        );
        let num_blocks = num_elements / 32;
        let expected_bytes = num_blocks * 18;
        ensure!(
            raw_bytes.len() == expected_bytes,
            "Q4_0 byte count mismatch: expected {expected_bytes} for {num_blocks} blocks, got {}",
            raw_bytes.len()
        );

        let client = WgpuRuntime::client(device);

        let handle = if !raw_bytes.len().is_multiple_of(4) {
            let pad = 4 - (raw_bytes.len() % 4);
            let mut buf = raw_bytes.to_vec();
            buf.resize(raw_bytes.len() + pad, 0);
            client.create_from_slice(&buf)
        } else {
            // Already 4-byte aligned — upload directly without copying
            client.create_from_slice(raw_bytes)
        };

        Ok(Self {
            handle,
            shape,
            num_blocks,
        })
    }

    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }
}

// ---------------------------------------------------------------------------
// Q4Linear
// ---------------------------------------------------------------------------

/// A linear layer with Q4_0 quantized weights.
pub struct Q4Linear {
    weights: Q4Tensor,
    bias: Option<Tensor<Wgpu, 1>>,
}

impl Q4Linear {
    pub fn new(weights: Q4Tensor, bias: Option<Tensor<Wgpu, 1>>) -> Self {
        Self { weights, bias }
    }

    pub fn forward(&self, x: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
        let out = q4_matmul(x, &self.weights);
        match &self.bias {
            Some(bias) => out + bias.clone().unsqueeze::<3>(),
            None => out,
        }
    }
}

// ---------------------------------------------------------------------------
// Q4KTensor — GPU buffer of Q4_K blocks
// ---------------------------------------------------------------------------

/// A Q4_K quantized weight tensor living on GPU.
pub struct Q4KTensor {
    pub(crate) handle: Handle,
    shape: [usize; 2],
    num_blocks: usize,
}

impl Q4KTensor {
    pub fn from_q4k_bytes(raw_bytes: &[u8], shape: [usize; 2], device: &WgpuDevice) -> Result<Self> {
        let [n, k] = shape;
        let num_elements = n * k;
        ensure!(
            num_elements % 256 == 0,
            "Q4_K requires element count divisible by 256, got {num_elements}"
        );
        let num_blocks = num_elements / 256;
        let expected_bytes = num_blocks * 144;
        ensure!(
            raw_bytes.len() == expected_bytes,
            "Q4_K byte count mismatch: expected {expected_bytes} for {num_blocks} blocks, got {}",
            raw_bytes.len()
        );

        let client = WgpuRuntime::client(device);

        let handle = if !raw_bytes.len().is_multiple_of(4) {
            let pad = 4 - (raw_bytes.len() % 4);
            let mut buf = raw_bytes.to_vec();
            buf.resize(raw_bytes.len() + pad, 0);
            client.create_from_slice(&buf)
        } else {
            // Already 4-byte aligned — upload directly without copying
            client.create_from_slice(raw_bytes)
        };

        Ok(Self {
            handle,
            shape,
            num_blocks,
        })
    }

    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }
}

// ---------------------------------------------------------------------------
// Q4KLinear
// ---------------------------------------------------------------------------

/// A linear layer with Q4_K quantized weights.
pub struct Q4KLinear {
    weights: Q4KTensor,
    bias: Option<Tensor<Wgpu, 1>>,
}

impl Q4KLinear {
    pub fn new(weights: Q4KTensor, bias: Option<Tensor<Wgpu, 1>>) -> Self {
        Self { weights, bias }
    }

    pub fn forward(&self, x: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
        let out = q4k_matmul(x, &self.weights);
        match &self.bias {
            Some(bias) => out + bias.clone().unsqueeze::<3>(),
            None => out,
        }
    }
}

/// A dense (F32) linear layer for precision-sensitive weights.
pub struct DenseLinear {
    weight_t: Tensor<Wgpu, 2>, // [in_features, out_features] (pre-transposed)
}

impl DenseLinear {
    pub fn new(weight: Tensor<Wgpu, 2>) -> Self {
        // Pre-transpose at construction to avoid clone+transpose every forward call
        Self { weight_t: weight.transpose() }
    }

    pub fn forward(&self, x: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
        // x: [B, M, K], weight_t: [K, N] → output: [B, M, N]
        burn::tensor::Tensor::matmul(x, self.weight_t.clone().unsqueeze::<3>())
    }
}

/// A linear layer that can be Q4, Q4_K (quantized), or Dense (F32).
pub enum Linear {
    Q4(Q4Linear),
    Q4K(Q4KLinear),
    Dense(DenseLinear),
}

impl Linear {
    pub fn forward(&self, x: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
        match self {
            Linear::Q4(l) => l.forward(x),
            Linear::Q4K(l) => l.forward(x),
            Linear::Dense(l) => l.forward(x),
        }
    }
}

// ---------------------------------------------------------------------------
// Q4 matmul kernel dispatch
// ---------------------------------------------------------------------------

struct Q4MatmulNaiveKernel {
    workgroup_size_x: u32,
    workgroup_size_y: u32,
}

impl KernelSource for Q4MatmulNaiveKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("wgsl/shader_naive.wgsl"))
            .register("workgroup_size_x", self.workgroup_size_x.to_string())
            .register("workgroup_size_y", self.workgroup_size_y.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(self.workgroup_size_x * 1000 + self.workgroup_size_y)
    }
}

/// Fused Q4_0 dequant+matmul on GPU.
pub fn q4_matmul(input: Tensor<Wgpu, 3>, weights: &Q4Tensor) -> Tensor<Wgpu, 3> {
    let cube_input: CubeTensor<WgpuRuntime> = input.into_primitive().tensor();
    let cube_input = into_contiguous(cube_input);

    assert_eq!(cube_input.shape.num_dims(), 3, "Input must be 3D [B, M, K]");
    let b = cube_input.shape.dims[0];
    let m = cube_input.shape.dims[1];
    let k = cube_input.shape.dims[2];
    let [n, wk] = weights.shape();
    assert_eq!(
        k, wk,
        "K dimension mismatch: input has {k}, weights have {wk}"
    );

    let client = cube_input.client.clone();
    let device = cube_input.device.clone();
    let blocks_per_row = k / 32;

    let output_handle = client.empty(b * m * n * 4);

    let info: [u32; 5] = [
        b as u32,
        m as u32,
        k as u32,
        n as u32,
        blocks_per_row as u32,
    ];
    let info_bytes: Vec<u8> = info.iter().flat_map(|v| v.to_le_bytes()).collect();
    let info_handle = client.create_from_slice(&info_bytes);

    let bindings = Bindings::new()
        .with_buffer(weights.handle.clone().binding())
        .with_buffer(cube_input.handle.clone().binding())
        .with_buffer(output_handle.clone().binding())
        .with_buffer(info_handle.binding());

    let kernel = SourceKernel::new(
        Q4MatmulNaiveKernel {
            workgroup_size_x: NAIVE_WG_X,
            workgroup_size_y: NAIVE_WG_Y,
        },
        CubeDim::new_2d(NAIVE_WG_X, NAIVE_WG_Y),
    );
    let wg_x = n.div_ceil(NAIVE_WG_X as usize) as u32;
    let wg_y = (b * m).div_ceil(NAIVE_WG_Y as usize) as u32;
    client
        .launch(
            Box::new(kernel) as Box<dyn CubeTask<AutoCompiler>>,
            CubeCount::new_2d(wg_x, wg_y),
            bindings,
        )
        .expect("Q4 naive matmul kernel launch failed");

    let output_tensor = CubeTensor::new_contiguous(
        client,
        device,
        burn::prelude::Shape::from(vec![b, m, n]),
        output_handle,
        DType::F32,
    );
    Tensor::from_primitive(TensorPrimitive::Float(output_tensor))
}

// ---------------------------------------------------------------------------
// Q4_K matmul kernel dispatch
// ---------------------------------------------------------------------------

struct Q4KMatmulNaiveKernel {
    workgroup_size_x: u32,
    workgroup_size_y: u32,
}

impl KernelSource for Q4KMatmulNaiveKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("wgsl/shader_q4k.wgsl"))
            .register("workgroup_size_x", self.workgroup_size_x.to_string())
            .register("workgroup_size_y", self.workgroup_size_y.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(self.workgroup_size_x * 1000 + self.workgroup_size_y)
    }
}

/// Fused Q4_K dequant+matmul on GPU.
pub fn q4k_matmul(input: Tensor<Wgpu, 3>, weights: &Q4KTensor) -> Tensor<Wgpu, 3> {
    let cube_input: CubeTensor<WgpuRuntime> = input.into_primitive().tensor();
    let cube_input = into_contiguous(cube_input);

    assert_eq!(cube_input.shape.num_dims(), 3, "Input must be 3D [B, M, K]");
    let b = cube_input.shape.dims[0];
    let m = cube_input.shape.dims[1];
    let k = cube_input.shape.dims[2];
    let [n, wk] = weights.shape();
    assert_eq!(
        k, wk,
        "K dimension mismatch: input has {k}, weights have {wk}"
    );

    let client = cube_input.client.clone();
    let device = cube_input.device.clone();
    let blocks_per_row = k / 256;

    let output_handle = client.empty(b * m * n * 4);

    let info: [u32; 5] = [
        b as u32,
        m as u32,
        k as u32,
        n as u32,
        blocks_per_row as u32,
    ];
    let info_bytes: Vec<u8> = info.iter().flat_map(|v| v.to_le_bytes()).collect();
    let info_handle = client.create_from_slice(&info_bytes);

    let bindings = Bindings::new()
        .with_buffer(weights.handle.clone().binding())
        .with_buffer(cube_input.handle.clone().binding())
        .with_buffer(output_handle.clone().binding())
        .with_buffer(info_handle.binding());

    let kernel = SourceKernel::new(
        Q4KMatmulNaiveKernel {
            workgroup_size_x: NAIVE_WG_X,
            workgroup_size_y: NAIVE_WG_Y,
        },
        CubeDim::new_2d(NAIVE_WG_X, NAIVE_WG_Y),
    );
    let wg_x = n.div_ceil(NAIVE_WG_X as usize) as u32;
    let wg_y = (b * m).div_ceil(NAIVE_WG_Y as usize) as u32;
    client
        .launch(
            Box::new(kernel) as Box<dyn CubeTask<AutoCompiler>>,
            CubeCount::new_2d(wg_x, wg_y),
            bindings,
        )
        .expect("Q4_K naive matmul kernel launch failed");

    let output_tensor = CubeTensor::new_contiguous(
        client,
        device,
        burn::prelude::Shape::from(vec![b, m, n]),
        output_handle,
        DType::F32,
    );
    Tensor::from_primitive(TensorPrimitive::Float(output_tensor))
}

// ---------------------------------------------------------------------------
// EmbeddingStore — Q4 embeddings for token lookups
// ---------------------------------------------------------------------------

/// Q4 embedding table stored as CPU bytes for efficient row lookups.
pub struct EmbeddingStore {
    cpu_bytes: Vec<u8>,
    vocab_size: usize,
    dim: usize,
}

impl EmbeddingStore {
    pub fn new(cpu_bytes: Vec<u8>, vocab_size: usize, dim: usize) -> Self {
        Self {
            cpu_bytes,
            vocab_size,
            dim,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Dequantize a single row into an existing CPU buffer (for accumulation).
    ///
    /// If `id` is out of vocabulary range, this is a no-op (zero contribution),
    /// matching the reference behavior for invalid/padding tokens.
    pub fn embed_id_add_cpu(&self, id: u32, out_buf: &mut [f32]) {
        assert_eq!(out_buf.len(), self.dim);
        if id as usize >= self.vocab_size {
            return; // Out-of-vocab token — add nothing (zero contribution)
        }
        let blocks_per_row = self.dim / 32;
        let bytes_per_row = blocks_per_row * 18;
        let row_offset = (id as usize) * bytes_per_row;
        let row_bytes = &self.cpu_bytes[row_offset..row_offset + bytes_per_row];

        for block in 0..blocks_per_row {
            let bo = block * 18;
            let d = f16_to_f32(u16::from_le_bytes([row_bytes[bo], row_bytes[bo + 1]]));
            let base = block * 32;
            for j in 0..16 {
                let byte = row_bytes[bo + 2 + j];
                out_buf[base + j] += ((byte & 0x0F) as f32 - 8.0) * d;
                out_buf[base + j + 16] += (((byte >> 4) & 0x0F) as f32 - 8.0) * d;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GgufTensorIndex — header-only metadata (no shard data retained)
// ---------------------------------------------------------------------------

/// Parsed GGUF header containing tensor metadata but not tensor data.
///
/// Used for incremental shard loading: parse the header from shard 0 once,
/// then process each shard's tensors independently without keeping all
/// shard data in memory simultaneously.
pub struct GgufTensorIndex {
    pub tensors: HashMap<String, GgufTensorInfo>,
    pub data_section_offset: u64,
    pub version: u32,
    pub tensor_count: u64,
}

impl GgufTensorIndex {
    /// Parse a GGUF header from raw bytes (typically shard 0).
    ///
    /// Extracts all tensor metadata (names, shapes, dtypes, offsets) and the
    /// data section offset. Does not read any tensor data.
    pub fn parse(data: &[u8]) -> Result<Self> {
        let reader = GgufReader::open(std::io::Cursor::new(data))?;
        Ok(Self {
            tensors: reader.tensors,
            data_section_offset: reader.data_section_offset,
            version: reader.version,
            tensor_count: reader.tensor_count,
        })
    }

    /// Look up a tensor by name.
    pub fn tensor_info(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.get(name)
    }

    /// Return all tensors whose data falls within an absolute byte range.
    ///
    /// A tensor is "in" a shard if its absolute data range
    /// `[data_section_offset + offset, data_section_offset + offset + byte_size)`
    /// overlaps with `[shard_abs_start, shard_abs_end)`.
    pub fn tensors_in_range(&self, shard_abs_start: u64, shard_abs_end: u64) -> Vec<&GgufTensorInfo> {
        self.tensors
            .values()
            .filter(|t| {
                let t_start = self.data_section_offset + t.offset;
                let t_end = t_start + t.byte_size();
                // Overlaps if t_start < shard_end AND t_end > shard_start
                t_start < shard_abs_end && t_end > shard_abs_start
            })
            .collect()
    }

    /// Read tensor data from a shard byte slice.
    ///
    /// The shard covers absolute byte range `[shard_abs_start, shard_abs_start + shard_data.len())`.
    /// Returns `None` if the tensor's data does not fully reside within this shard.
    pub fn read_tensor_from_shard(
        &self,
        tensor: &GgufTensorInfo,
        shard_data: &[u8],
        shard_abs_start: u64,
    ) -> Option<Vec<u8>> {
        let t_abs_start = self.data_section_offset + tensor.offset;
        let t_byte_size = tensor.byte_size();
        let t_abs_end = t_abs_start + t_byte_size;
        let shard_abs_end = shard_abs_start + shard_data.len() as u64;

        // Tensor must be fully contained within this shard
        if t_abs_start < shard_abs_start || t_abs_end > shard_abs_end {
            return None;
        }

        let local_start = (t_abs_start - shard_abs_start) as usize;
        let local_end = local_start + t_byte_size as usize;
        Some(shard_data[local_start..local_end].to_vec())
    }
}

// ---------------------------------------------------------------------------
// Q4ModelLoader
// ---------------------------------------------------------------------------

pub struct Q4ModelLoader<R: Read + Seek> {
    reader: GgufReader<R>,
}

impl Q4ModelLoader<ShardedCursor> {
    pub fn from_shards(shards: Vec<Vec<u8>>) -> Result<Self> {
        let reader = GgufReader::open(ShardedCursor::new(shards))?;
        Ok(Self { reader })
    }
}

impl<R: Read + Seek> Q4ModelLoader<R> {
    pub fn version(&self) -> u32 {
        self.reader.version()
    }

    pub fn tensor_count(&self) -> u64 {
        self.reader.tensor_count()
    }

    pub fn tensor_info(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.reader.tensor_info(name)
    }

    pub fn tensor_data(&mut self, name: &str) -> Result<Vec<u8>> {
        self.reader.tensor_data(name)
    }

    /// Check if a tensor exists in the GGUF.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.reader.tensor_info(name).is_some()
    }

    /// Load a Q4 linear layer from a named tensor.
    pub fn load_q4_linear(&mut self, name: &str, device: &WgpuDevice) -> Result<Q4Linear> {
        let info = self
            .reader
            .tensor_info(name)
            .with_context(|| format!("Tensor '{name}' not found"))?
            .clone();

        if info.dtype() != GgmlDtype::Q4_0 {
            bail!("Expected Q4_0 for '{name}', got {:?}", info.dtype());
        }

        let shape = reverse_gguf_dims(info.shape());
        let bytes = self.reader.tensor_data(name)?;
        let q4 = Q4Tensor::from_q4_bytes(&bytes, [shape[0], shape[1]], device)?;
        Ok(Q4Linear::new(q4, None))
    }

    /// Load an f32/f16 tensor as a 1D f32 tensor (for norm weights).
    pub fn load_f32_tensor(&mut self, name: &str, device: &WgpuDevice) -> Result<Tensor<Wgpu, 1>> {
        let info = self
            .reader
            .tensor_info(name)
            .with_context(|| format!("Tensor '{name}' not found"))?
            .clone();
        let num_elements = info.num_elements();
        let bytes = self.reader.tensor_data(name)?;
        let data: Vec<f32> = match info.dtype() {
            GgmlDtype::F32 => bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            GgmlDtype::F16 => bytes
                .chunks_exact(2)
                .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect(),
            GgmlDtype::Q4_0 => bail!("Cannot load Q4_0 tensor '{name}' as f32"),
            GgmlDtype::Q4_K => bail!("Cannot load Q4_K tensor '{name}' as f32"),
        };
        Ok(Tensor::from_data(
            TensorData::new(data, [num_elements as usize]),
            device,
        ))
    }

    /// Load an RmsNormLayer from a named tensor.
    ///
    /// The GGUF stores norm weights as f32 or f16 with shape [1, 1, dim].
    /// We squeeze to [dim] for the RmsNormLayer.
    pub fn load_rms_norm(
        &mut self,
        name: &str,
        eps: f64,
        device: &WgpuDevice,
    ) -> Result<crate::model::RmsNormLayer> {
        let alpha = self.load_f32_tensor(name, device)?;
        Ok(crate::model::RmsNormLayer::new(alpha, eps))
    }

    /// Load an F32 dense linear layer from a named tensor.
    ///
    /// The tensor can be stored as F32 or Q4_0 in the GGUF. If Q4_0, it is
    /// dequantized to F32 on CPU and uploaded as a dense GPU tensor.
    pub fn load_dense_linear(&mut self, name: &str, device: &WgpuDevice) -> Result<DenseLinear> {
        let info = self
            .reader
            .tensor_info(name)
            .with_context(|| format!("Tensor '{name}' not found"))?
            .clone();

        let shape = reverse_gguf_dims(info.shape());
        let rows = shape[0];
        let cols = shape[1];
        let bytes = self.reader.tensor_data(name)?;

        let data: Vec<f32> = match info.dtype() {
            GgmlDtype::F32 => bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            GgmlDtype::F16 => bytes
                .chunks_exact(2)
                .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect(),
            GgmlDtype::Q4_0 => {
                // Dequantize Q4_0 to F32 on CPU
                let num_elements = rows * cols;
                let block_size = 32;
                let num_blocks = num_elements.div_ceil(block_size);
                let mut output = vec![0.0f32; num_blocks * block_size];
                for i in 0..num_blocks {
                    let offset = i * 18;
                    let scale = f16_to_f32(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]));
                    for j in 0..16 {
                        let byte = bytes[offset + 2 + j];
                        let v_lo = ((byte & 0x0F) as i8 - 8) as f32;
                        let v_hi = (((byte >> 4) & 0x0F) as i8 - 8) as f32;
                        output[i * block_size + j] = v_lo * scale;
                        output[i * block_size + j + 16] = v_hi * scale;
                    }
                }
                output.truncate(num_elements);
                output
            }
            GgmlDtype::Q4_K => {
                bail!("Cannot load Q4_K tensor '{name}' as dense linear; use load_q4k_linear instead");
            }
        };

        let weight = Tensor::from_data(
            TensorData::new(data, [rows, cols]),
            device,
        );
        Ok(DenseLinear::new(weight))
    }

    /// Load a Q4_K linear layer from a named tensor.
    pub fn load_q4k_linear(&mut self, name: &str, device: &WgpuDevice) -> Result<Q4KLinear> {
        let info = self
            .reader
            .tensor_info(name)
            .with_context(|| format!("Tensor '{name}' not found"))?
            .clone();

        if info.dtype() != GgmlDtype::Q4_K {
            bail!("Expected Q4_K for '{name}', got {:?}", info.dtype());
        }

        let shape = reverse_gguf_dims(info.shape());
        let bytes = self.reader.tensor_data(name)?;
        let q4k = Q4KTensor::from_q4k_bytes(&bytes, [shape[0], shape[1]], device)?;
        Ok(Q4KLinear::new(q4k, None))
    }

    /// Load a linear layer, using F32 precision if the GGUF tensor is stored as F32,
    /// otherwise use Q4 or Q4_K.
    pub fn load_linear_auto(&mut self, name: &str, device: &WgpuDevice) -> Result<Linear> {
        let info = self
            .reader
            .tensor_info(name)
            .with_context(|| format!("Tensor '{name}' not found"))?
            .clone();

        match info.dtype() {
            GgmlDtype::F32 | GgmlDtype::F16 => {
                let dense = self.load_dense_linear(name, device)?;
                Ok(Linear::Dense(dense))
            }
            GgmlDtype::Q4_0 => {
                let q4 = self.load_q4_linear(name, device)?;
                Ok(Linear::Q4(q4))
            }
            GgmlDtype::Q4_K => {
                let q4k = self.load_q4k_linear(name, device)?;
                Ok(Linear::Q4K(q4k))
            }
        }
    }

    /// Load an embedding table as raw Q4 bytes (deferred creation).
    ///
    /// Returns (raw_bytes, [vocab_size, dim]) for later EmbeddingStore construction.
    pub fn load_embedding_bytes(&mut self, name: &str) -> Result<(Vec<u8>, [usize; 2])> {
        let info = self
            .reader
            .tensor_info(name)
            .with_context(|| format!("Tensor '{name}' not found"))?
            .clone();
        let shape = reverse_gguf_dims(info.shape());
        let bytes = self.reader.tensor_data(name)?;
        Ok((bytes, [shape[0], shape[1]]))
    }
}
