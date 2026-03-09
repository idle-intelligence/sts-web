//! Temporal transformer model: embeddings, attention, RoPE, KV cache.
//!
//! The main ~7B parameter model that processes time steps autoregressively.
//! Architecture (PersonaPlex / Moshi):
//! - 17 input embedding streams summed per step:
//!   stream 0 = text (32001 vocab), streams 1-8 = model audio codebooks,
//!   streams 9-16 = user audio codebooks (2049 vocab each)
//! - 32-layer transformer: RMSNorm -> MHA (32 heads) -> RMSNorm -> SwiGLU
//! - RoPE positional encoding (theta=10000)
//! - Ring-buffer KV cache (3000 steps)
//! - Output: hidden state [1, 1, 4096] for depth transformer,
//!   plus text logits via text_linear head
//!
//! All linear projections use Q4Linear from the gguf module.

use burn::backend::wgpu::{
    into_contiguous, AutoCompiler, CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate,
    Wgpu, WgpuDevice, WgpuRuntime,
};
use burn::tensor::activation::softmax;
use burn::tensor::{DType, Tensor, TensorPrimitive};
use cubecl::prelude::KernelId;
use cubecl::server::{Bindings, CubeCount, Handle};
use cubecl::{CubeTask, Runtime};
use std::cell::RefCell;

use crate::gguf::{EmbeddingStore, Linear};
use crate::StsConfig;

// ---------------------------------------------------------------------------
// RoPE -- Rotary Position Embeddings (fused GPU kernel)
// ---------------------------------------------------------------------------

struct RopeKernelSource;

impl KernelSource for RopeKernelSource {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("wgsl/shader_rope.wgsl"))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

/// Apply fused RoPE in-place on a [B, S, H, D] tensor.
///
/// Launches a single GPU dispatch instead of ~6-8 Burn operations.
/// The tensor is modified in-place (safe since Q/K are fresh projections).
fn fused_rope_inplace(
    x: Tensor<Wgpu, 4>,
    cos_handle: &Handle,
    sin_handle: &Handle,
    offset: usize,
    half_dim: usize,
) -> Tensor<Wgpu, 4> {
    let cube_x: CubeTensor<WgpuRuntime> = x.into_primitive().tensor();
    let cube_x = into_contiguous(cube_x);

    let b = cube_x.shape.dims[0];
    let s = cube_x.shape.dims[1];
    let h = cube_x.shape.dims[2];
    // d = cube_x.shape.dims[3], but we use half_dim from the RoPE tables

    let total_elements = b * s * h * half_dim;

    let client = cube_x.client.clone();
    let device = cube_x.device.clone();

    // Info buffer: [pos_offset, B, S, H, half_D]
    let info: [u32; 5] = [
        offset as u32,
        b as u32,
        s as u32,
        h as u32,
        half_dim as u32,
    ];
    let info_bytes: Vec<u8> = info.iter().flat_map(|v| v.to_le_bytes()).collect();
    let info_handle = client.create_from_slice(&info_bytes);

    let bindings = Bindings::new()
        .with_buffer(cube_x.handle.clone().binding())
        .with_buffer(cos_handle.clone().binding())
        .with_buffer(sin_handle.clone().binding())
        .with_buffer(info_handle.binding());

    let kernel = SourceKernel::new(RopeKernelSource, CubeDim::new_1d(256));

    let num_workgroups = (total_elements as u32).div_ceil(256);
    client
        .launch(
            Box::new(kernel) as Box<dyn CubeTask<AutoCompiler>>,
            CubeCount::new_1d(num_workgroups),
            bindings,
        )
        .expect("RoPE kernel launch failed");

    // Return the same buffer as a tensor (modified in-place)
    let output_tensor = CubeTensor::new_contiguous(
        client,
        device,
        burn::prelude::Shape::from(vec![b, s, h, half_dim * 2]),
        cube_x.handle,
        DType::F32,
    );
    Tensor::from_primitive(TensorPrimitive::Float(output_tensor))
}

/// Rotary Position Embeddings with precomputed cos/sin tables on GPU.
///
/// Uses a fused WGSL compute shader: 1 dispatch per tensor instead of ~6-8
/// Burn operations (reshape, slice, broadcast, mul, sub, add, cat, reshape).
pub struct RoPE {
    cos_handle: Handle,
    sin_handle: Handle,
    half_dim: usize,
}

impl RoPE {
    /// Create RoPE with precomputed frequencies.
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f64, device: &WgpuDevice) -> Self {
        let half_dim = head_dim / 2;

        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / (theta as f32).powf((2 * i) as f32 / head_dim as f32))
            .collect();

        let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();

        let mut freqs = vec![0.0f32; max_seq_len * half_dim];
        for i in 0..max_seq_len {
            for j in 0..half_dim {
                freqs[i * half_dim + j] = positions[i] * inv_freq[j];
            }
        }

        let freqs = Tensor::<Wgpu, 1>::from_floats(freqs.as_slice(), device)
            .reshape([max_seq_len, half_dim]);

        let cos = freqs.clone().cos();
        let sin = freqs.sin();

        // Extract GPU handles for the fused kernel
        let cos_handle = cos.into_primitive().tensor().handle;
        let sin_handle = sin.into_primitive().tensor().handle;

        RoPE {
            cos_handle,
            sin_handle,
            half_dim,
        }
    }

    /// Apply rotary embeddings to Q and K tensors using fused GPU kernel.
    ///
    /// q, k shape: [batch, seq, heads, head_dim]
    /// 2 GPU dispatches total (1 for Q, 1 for K) instead of ~12-16 Burn ops.
    pub fn apply(
        &self,
        q: Tensor<Wgpu, 4>,
        k: Tensor<Wgpu, 4>,
        offset: usize,
    ) -> (Tensor<Wgpu, 4>, Tensor<Wgpu, 4>) {
        let q_rot = fused_rope_inplace(
            q,
            &self.cos_handle,
            &self.sin_handle,
            offset,
            self.half_dim,
        );
        let k_rot = fused_rope_inplace(
            k,
            &self.cos_handle,
            &self.sin_handle,
            offset,
            self.half_dim,
        );
        (q_rot, k_rot)
    }
}

// ---------------------------------------------------------------------------
// KVCache
// ---------------------------------------------------------------------------

/// Ring-buffer KV cache for autoregressive decoding.
pub struct KVCache {
    k: Option<Tensor<Wgpu, 4>>,
    v: Option<Tensor<Wgpu, 4>>,
    offset: usize,
    len: usize,
    write_pos: usize,
    max_len: usize,
    n_kv_heads: usize,
    head_dim: usize,
    device: WgpuDevice,
}

impl KVCache {
    pub fn new(max_len: usize, n_kv_heads: usize, head_dim: usize, device: &WgpuDevice) -> Self {
        Self {
            k: None,
            v: None,
            offset: 0,
            len: 0,
            write_pos: 0,
            max_len,
            n_kv_heads,
            head_dim,
            device: device.clone(),
        }
    }

    pub fn update(
        &mut self,
        k: Tensor<Wgpu, 4>,
        v: Tensor<Wgpu, 4>,
    ) -> (Tensor<Wgpu, 4>, Tensor<Wgpu, 4>) {
        let [b, _h, _seq, _d] = k.dims();

        if self.k.is_none() {
            self.k = Some(Tensor::zeros(
                [b, self.n_kv_heads, self.max_len, self.head_dim],
                &self.device,
            ));
            self.v = Some(Tensor::zeros(
                [b, self.n_kv_heads, self.max_len, self.head_dim],
                &self.device,
            ));
        }

        let k_buf = self.k.take().unwrap();
        let v_buf = self.v.take().unwrap();
        let [b, h, _, hd] = k_buf.dims();
        let pos = self.write_pos;

        let k_buf = k_buf.slice_assign([0..b, 0..h, pos..pos + 1, 0..hd], k);
        let v_buf = v_buf.slice_assign([0..b, 0..h, pos..pos + 1, 0..hd], v);

        self.write_pos = (self.write_pos + 1) % self.max_len;
        self.offset += 1;
        self.len = (self.len + 1).min(self.max_len);

        let result = if self.len < self.max_len {
            let k_view = k_buf.clone().slice([0..b, 0..h, 0..self.len, 0..hd]);
            let v_view = v_buf.clone().slice([0..b, 0..h, 0..self.len, 0..hd]);
            (k_view, v_view)
        } else {
            (k_buf.clone(), v_buf.clone())
        };

        self.k = Some(k_buf);
        self.v = Some(v_buf);

        result
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn seq_len(&self) -> usize {
        self.len
    }

    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
        self.offset = 0;
        self.len = 0;
        self.write_pos = 0;
    }

    pub fn reset_keep_buffers(&mut self) {
        if let Some(ref k) = self.k {
            let shape = k.dims();
            self.k = Some(Tensor::zeros(shape, &self.device));
        }
        if let Some(ref v) = self.v {
            let shape = v.dims();
            self.v = Some(Tensor::zeros(shape, &self.device));
        }
        self.offset = 0;
        self.len = 0;
        self.write_pos = 0;
    }

    /// Roll back the cache by `n` steps (undo the last `n` updates).
    ///
    /// The stale KV data at the rolled-back positions remains in the buffer
    /// but will be overwritten by subsequent updates and is excluded from
    /// the valid region via `len`.
    pub fn rollback(&mut self, n: usize) {
        let n = n.min(self.len);
        self.offset -= n;
        self.len -= n;
        if self.write_pos >= n {
            self.write_pos -= n;
        } else {
            // wrap around ring buffer
            self.write_pos = self.max_len - (n - self.write_pos);
        }
    }
}

/// Collection of KV caches for all layers.
pub struct LayerCaches {
    caches: Vec<KVCache>,
}

impl LayerCaches {
    pub fn new(
        n_layers: usize,
        max_len: usize,
        n_kv_heads: usize,
        head_dim: usize,
        device: &WgpuDevice,
    ) -> Self {
        Self {
            caches: (0..n_layers)
                .map(|_| KVCache::new(max_len, n_kv_heads, head_dim, device))
                .collect(),
        }
    }

    pub fn get_mut(&mut self, layer: usize) -> Option<&mut KVCache> {
        self.caches.get_mut(layer)
    }

    pub fn seq_len(&self) -> usize {
        self.caches.first().map(|c| c.seq_len()).unwrap_or(0)
    }

    pub fn reset(&mut self) {
        for cache in &mut self.caches {
            cache.reset();
        }
    }

    pub fn reset_keep_buffers(&mut self) {
        for cache in &mut self.caches {
            cache.reset_keep_buffers();
        }
    }

    /// Roll back all layer caches by `n` steps.
    pub fn rollback(&mut self, n: usize) {
        for cache in &mut self.caches {
            cache.rollback(n);
        }
    }
}

// ---------------------------------------------------------------------------
// Fused RMSNorm kernel — 1 dispatch instead of 6-8
// ---------------------------------------------------------------------------

struct RmsNormKernelSource;

impl KernelSource for RmsNormKernelSource {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("wgsl/shader_rmsnorm.wgsl"))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

/// Fused RMSNorm on GPU: output = (x / rms(x)) * alpha, single dispatch.
///
/// `x`: [B, S, D], `alpha_handle`: GPU buffer of [D] f32 weights.
/// Returns [B, S, D] normalized tensor.
fn fused_rms_norm(
    x: Tensor<Wgpu, 3>,
    alpha_handle: &Handle,
    _eps: f32,
    info_handle: &Handle,
) -> Tensor<Wgpu, 3> {
    let cube_input: CubeTensor<WgpuRuntime> = x.into_primitive().tensor();
    let cube_input = into_contiguous(cube_input);

    let b = cube_input.shape.dims[0];
    let s = cube_input.shape.dims[1];
    let d = cube_input.shape.dims[2];
    let total_rows = b * s;

    let client = cube_input.client.clone();
    let device = cube_input.device.clone();

    let output_handle = client.empty(total_rows * d * 4);

    let bindings = Bindings::new()
        .with_buffer(cube_input.handle.clone().binding())
        .with_buffer(alpha_handle.clone().binding())
        .with_buffer(output_handle.clone().binding())
        .with_buffer(info_handle.clone().binding());

    let kernel = SourceKernel::new(RmsNormKernelSource, CubeDim::new_1d(256));

    client
        .launch(
            Box::new(kernel) as Box<dyn CubeTask<AutoCompiler>>,
            CubeCount::new_1d(total_rows as u32),
            bindings,
        )
        .expect("RMSNorm kernel launch failed");

    let output_tensor = CubeTensor::new_contiguous(
        client,
        device,
        burn::prelude::Shape::from(vec![b, s, d]),
        output_handle,
        DType::F32,
    );
    Tensor::from_primitive(TensorPrimitive::Float(output_tensor))
}

// ---------------------------------------------------------------------------
// Fused SwiGLU kernel — 1 dispatch instead of 3+ (slice, silu, mul)
// ---------------------------------------------------------------------------

struct SwigluKernelSource;

impl KernelSource for SwigluKernelSource {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("wgsl/shader_swiglu.wgsl"))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

/// Fused SwiGLU activation on GPU: output[i] = silu(input[i]) * input[i + N].
///
/// `x`: [B, S, 2*N] from linear_in. Returns [B, S, N].
/// Single dispatch replaces split + silu + mul.
pub fn fused_swiglu(x: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
    let cube_input: CubeTensor<WgpuRuntime> = x.into_primitive().tensor();
    let cube_input = into_contiguous(cube_input);

    let b = cube_input.shape.dims[0];
    let s = cube_input.shape.dims[1];
    let total_dim = cube_input.shape.dims[2];
    let half = total_dim / 2;

    let client = cube_input.client.clone();
    let device = cube_input.device.clone();

    // Total elements = B * S * N (each row of N elements processed independently)
    let total_elements = b * s * half;

    let output_handle = client.empty(total_elements * 4);

    // Info buffer: [N] where N = half of total_dim
    let info: [u32; 1] = [total_elements as u32];
    let info_bytes: Vec<u8> = info.iter().flat_map(|v| v.to_le_bytes()).collect();
    let info_handle = client.create_from_slice(&info_bytes);

    let bindings = Bindings::new()
        .with_buffer(cube_input.handle.clone().binding())
        .with_buffer(output_handle.clone().binding())
        .with_buffer(info_handle.binding());

    let kernel = SourceKernel::new(SwigluKernelSource, CubeDim::new_1d(256));

    let num_workgroups = (total_elements as u32).div_ceil(256);
    client
        .launch(
            Box::new(kernel) as Box<dyn CubeTask<AutoCompiler>>,
            CubeCount::new_1d(num_workgroups),
            bindings,
        )
        .expect("SwiGLU kernel launch failed");

    let output_tensor = CubeTensor::new_contiguous(
        client,
        device,
        burn::prelude::Shape::from(vec![b, s, half]),
        output_handle,
        DType::F32,
    );
    Tensor::from_primitive(TensorPrimitive::Float(output_tensor))
}

pub struct RmsNormLayer {
    alpha_handle: Handle,
    eps: f32,
    dim: usize,
    /// Cached info buffer (D, eps_bits, total_rows). Keyed by total_rows.
    info_cache: RefCell<Option<(usize, Handle)>>,
    device: WgpuDevice,
}

impl RmsNormLayer {
    pub fn new(alpha: Tensor<Wgpu, 1>, eps: f64) -> Self {
        let dim = alpha.dims()[0];
        let device = alpha.device();
        // Extract the GPU handle from the alpha tensor
        let alpha_handle = alpha.into_primitive().tensor().handle;
        Self {
            alpha_handle,
            eps: eps as f32,
            dim,
            info_cache: RefCell::new(None),
            device,
        }
    }

    /// Get or create a cached info buffer for the given total_rows.
    fn get_or_create_info(&self, total_rows: usize) -> Handle {
        let mut cache = self.info_cache.borrow_mut();
        if let Some((cached_rows, ref handle)) = *cache {
            if cached_rows == total_rows {
                return handle.clone();
            }
        }
        let client = WgpuRuntime::client(&self.device);
        let info: [u32; 3] = [
            self.dim as u32,
            self.eps.to_bits(),
            total_rows as u32,
        ];
        let info_bytes: Vec<u8> = info.iter().flat_map(|v| v.to_le_bytes()).collect();
        let handle = client.create_from_slice(&info_bytes);
        *cache = Some((total_rows, handle.clone()));
        handle
    }

    pub fn forward(&self, x: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
        let [b, s, _d] = x.dims();
        let total_rows = b * s;
        let info_handle = self.get_or_create_info(total_rows);
        fused_rms_norm(x, &self.alpha_handle, self.eps, &info_handle)
    }
}

// ---------------------------------------------------------------------------
// Causal mask
// ---------------------------------------------------------------------------

fn apply_causal_mask(
    scores: Tensor<Wgpu, 4>,
    q_len: usize,
    kv_len: usize,
    offset: usize,
) -> Tensor<Wgpu, 4> {
    // Single-step decode: all KV entries are in the past, no mask needed.
    if q_len == 1 {
        return scores;
    }
    let device = scores.device();
    let mut mask_data = vec![0.0f32; q_len * kv_len];
    for i in 0..q_len {
        let actual_pos = offset + i;
        for j in 0..kv_len {
            if j > actual_pos {
                mask_data[i * kv_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    let mask: Tensor<Wgpu, 1> = Tensor::from_floats(mask_data.as_slice(), &device);
    let mask: Tensor<Wgpu, 2> = mask.reshape([q_len, kv_len]);
    let mask: Tensor<Wgpu, 4> = mask.unsqueeze_dim::<3>(0).unsqueeze_dim(0);
    scores + mask
}

// ---------------------------------------------------------------------------
// Fused Attention -- single-token decode (Flash-decoding style)
// ---------------------------------------------------------------------------

struct FusedAttentionKernelSource;

impl KernelSource for FusedAttentionKernelSource {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("wgsl/shader_attention.wgsl"))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

/// Fused single-token attention: QK^T + softmax + V in one GPU dispatch.
///
/// Only valid for single-token decode (q_seq_len == 1).
/// q shape: [1, num_heads, 1, head_dim] -> reshaped to [num_heads, head_dim]
/// k shape: [1, num_heads, S, head_dim] -> [num_heads, S, head_dim]
/// v shape: [1, num_heads, S, head_dim] -> [num_heads, S, head_dim]
/// Returns: [1, num_heads, 1, head_dim]
///
/// Replaces: matmul(Q, K^T), scale, softmax, matmul(attn, V) — ~6 dispatches -> 1.
pub fn fused_attention(
    q: Tensor<Wgpu, 4>,
    k: Tensor<Wgpu, 4>,
    v: Tensor<Wgpu, 4>,
    scale: f32,
) -> Tensor<Wgpu, 4> {
    let [_b, num_heads, _one, head_dim] = q.dims();
    let [_b2, _h2, seq_len, _hd2] = k.dims();

    // Reshape to 2D/3D for the kernel (drop batch and singleton dims)
    let q = q.reshape([num_heads, head_dim]);
    let k = k.reshape([num_heads, seq_len, head_dim]);
    let v = v.reshape([num_heads, seq_len, head_dim]);

    let cube_q: CubeTensor<WgpuRuntime> = q.into_primitive().tensor();
    let cube_k: CubeTensor<WgpuRuntime> = k.into_primitive().tensor();
    let cube_v: CubeTensor<WgpuRuntime> = v.into_primitive().tensor();

    let cube_q = into_contiguous(cube_q);
    let cube_k = into_contiguous(cube_k);
    let cube_v = into_contiguous(cube_v);

    let client = cube_q.client.clone();
    let device = cube_q.device.clone();

    let output_size = num_heads * head_dim;
    let output_handle = client.empty(output_size * 4);

    // Info buffer: [num_heads, head_dim, seq_len, scale_bits]
    let info: [u32; 4] = [
        num_heads as u32,
        head_dim as u32,
        seq_len as u32,
        scale.to_bits(),
    ];
    let info_bytes: Vec<u8> = info.iter().flat_map(|v| v.to_le_bytes()).collect();
    let info_handle = client.create_from_slice(&info_bytes);

    let bindings = Bindings::new()
        .with_buffer(cube_q.handle.clone().binding())
        .with_buffer(cube_k.handle.clone().binding())
        .with_buffer(cube_v.handle.clone().binding())
        .with_buffer(output_handle.clone().binding())
        .with_buffer(info_handle.binding());

    let kernel = SourceKernel::new(FusedAttentionKernelSource, CubeDim::new_1d(256));

    // One workgroup per head
    client
        .launch(
            Box::new(kernel) as Box<dyn CubeTask<AutoCompiler>>,
            CubeCount::new_1d(num_heads as u32),
            bindings,
        )
        .expect("Fused attention kernel launch failed");

    let output_tensor = CubeTensor::new_contiguous(
        client,
        device,
        burn::prelude::Shape::from(vec![1, num_heads, 1, head_dim]),
        output_handle,
        DType::F32,
    );
    Tensor::from_primitive(TensorPrimitive::Float(output_tensor))
}

// ---------------------------------------------------------------------------
// Q4Attention -- MHA with combined QKV projection
// ---------------------------------------------------------------------------

/// Multi-head attention (32 Q heads, 32 KV heads = standard MHA).
///
/// Weight layout:
/// - `in_proj_weight`: [3 * dim, dim] = [12288, 4096] (combined QKV)
/// - `out_proj.weight`: [dim, dim] = [4096, 4096]
pub struct Q4Attention {
    in_proj: Linear,
    out_proj: Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    dim: usize,
    scale: f32,
}

impl Q4Attention {
    pub fn new(
        in_proj: Linear,
        out_proj: Linear,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let dim = n_heads * head_dim;
        Self {
            in_proj,
            out_proj,
            n_heads,
            n_kv_heads,
            head_dim,
            dim,
            scale: (head_dim as f32).powf(-0.5),
        }
    }

    fn split_qkv(
        &self,
        qkv: Tensor<Wgpu, 3>,
    ) -> (Tensor<Wgpu, 3>, Tensor<Wgpu, 3>, Tensor<Wgpu, 3>) {
        let [batch, seq, _] = qkv.dims();
        let kv_dim = self.n_kv_heads * self.head_dim;
        let q = qkv
            .clone()
            .slice([0..batch, 0..seq, 0..self.dim]);
        let k = qkv
            .clone()
            .slice([0..batch, 0..seq, self.dim..self.dim + kv_dim]);
        let v = qkv.slice([
            0..batch,
            0..seq,
            self.dim + kv_dim..self.dim + 2 * kv_dim,
        ]);
        (q, k, v)
    }

    pub fn forward_with_cache(
        &self,
        x: Tensor<Wgpu, 3>,
        rope: &RoPE,
        cache: &mut KVCache,
    ) -> Tensor<Wgpu, 3> {
        let [batch, seq_len, _] = x.dims();
        let offset = cache.offset();

        let qkv = self.in_proj.forward(x);
        let (q, k, v) = self.split_qkv(qkv);

        let q = q.reshape([batch, seq_len, self.n_heads, self.head_dim]);
        let k = k.reshape([batch, seq_len, self.n_kv_heads, self.head_dim]);
        let v = v.reshape([batch, seq_len, self.n_kv_heads, self.head_dim]);

        let (q, k) = rope.apply(q, k, offset);

        // [batch, heads, seq, head_dim]
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        let (k, v) = cache.update(k, v);
        let total_seq_len = cache.seq_len();

        // Expand K, V for GQA if n_heads != n_kv_heads
        let (k, v) = if self.n_heads != self.n_kv_heads {
            let repeat_factor = self.n_heads / self.n_kv_heads;
            let [b, nkv, s, hd] = k.dims();
            let k = k
                .unsqueeze_dim::<5>(2)
                .repeat_dim(2, repeat_factor)
                .reshape([b, nkv * repeat_factor, s, hd]);
            let v = v
                .unsqueeze_dim::<5>(2)
                .repeat_dim(2, repeat_factor)
                .reshape([b, nkv * repeat_factor, s, hd]);
            (k, v)
        } else {
            (k, v)
        };

        // Fused attention for single-token decode (generation mode).
        // Replaces: Q@K^T, scale, causal_mask, softmax, @V — ~6 dispatches -> 1.
        let out = if seq_len == 1 && self.n_heads == self.n_kv_heads {
            fused_attention(q, k, v, self.scale)
        } else {
            let k_t = k.swap_dims(2, 3);
            let scores = q.matmul(k_t) * self.scale;
            let scores = apply_causal_mask(scores, seq_len, total_seq_len, offset);
            let attn = softmax(scores, 3);
            attn.matmul(v)
        };

        let out = out.swap_dims(1, 2);
        let out = out.reshape([batch, seq_len, self.n_heads * self.head_dim]);
        self.out_proj.forward(out)
    }

    /// Forward with fused residual addition in the out_proj matmul.
    pub fn forward_with_cache_fused_residual(
        &self,
        x: Tensor<Wgpu, 3>,
        residual: Tensor<Wgpu, 3>,
        rope: &RoPE,
        cache: &mut KVCache,
    ) -> Tensor<Wgpu, 3> {
        let [batch, seq_len, _] = x.dims();
        let offset = cache.offset();

        let qkv = self.in_proj.forward(x);
        let (q, k, v) = self.split_qkv(qkv);

        let q = q.reshape([batch, seq_len, self.n_heads, self.head_dim]);
        let k = k.reshape([batch, seq_len, self.n_kv_heads, self.head_dim]);
        let v = v.reshape([batch, seq_len, self.n_kv_heads, self.head_dim]);

        let (q, k) = rope.apply(q, k, offset);

        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        let (k, v) = cache.update(k, v);
        let total_seq_len = cache.seq_len();

        let (k, v) = if self.n_heads != self.n_kv_heads {
            let repeat_factor = self.n_heads / self.n_kv_heads;
            let [b, nkv, s, hd] = k.dims();
            let k = k
                .unsqueeze_dim::<5>(2)
                .repeat_dim(2, repeat_factor)
                .reshape([b, nkv * repeat_factor, s, hd]);
            let v = v
                .unsqueeze_dim::<5>(2)
                .repeat_dim(2, repeat_factor)
                .reshape([b, nkv * repeat_factor, s, hd]);
            (k, v)
        } else {
            (k, v)
        };

        let out = if seq_len == 1 && self.n_heads == self.n_kv_heads {
            fused_attention(q, k, v, self.scale)
        } else {
            let k_t = k.swap_dims(2, 3);
            let scores = q.matmul(k_t) * self.scale;
            let scores = apply_causal_mask(scores, seq_len, total_seq_len, offset);
            let attn = softmax(scores, 3);
            attn.matmul(v)
        };

        let out = out.swap_dims(1, 2);
        let out = out.reshape([batch, seq_len, self.n_heads * self.head_dim]);
        self.out_proj.forward_with_residual(out, residual)
    }
}

// ---------------------------------------------------------------------------
// Q4FeedForward (SwiGLU)
// ---------------------------------------------------------------------------

/// Gated MLP with Q4-quantized weights (SwiGLU).
///
/// Weight layout:
/// - `gating.linear_in.weight`: [2 * intermediate, dim] = [22528, 4096]
/// - `gating.linear_out.weight`: [dim, intermediate] = [4096, 11264]
///
/// Forward: `linear_out(silu(gate) * value)` where gate, value = split(linear_in(x))
pub struct Q4FeedForward {
    linear_in: Linear,
    linear_out: Linear,
}

impl Q4FeedForward {
    pub fn new(linear_in: Linear, linear_out: Linear) -> Self {
        Self {
            linear_in,
            linear_out,
        }
    }

    pub fn forward(&self, x: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
        let combined = self.linear_in.forward(x);
        let activated = fused_swiglu(combined);
        self.linear_out.forward(activated)
    }

    /// Forward with fused residual addition in the linear_out matmul.
    pub fn forward_with_residual(&self, x: Tensor<Wgpu, 3>, residual: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
        let combined = self.linear_in.forward(x);
        let activated = fused_swiglu(combined);
        self.linear_out.forward_with_residual(activated, residual)
    }
}

// ---------------------------------------------------------------------------
// Q4TransformerBlock
// ---------------------------------------------------------------------------

/// Pre-LN transformer block.
///
/// Architecture: RMSNorm -> MHA -> residual -> RMSNorm -> SwiGLU -> residual
pub struct Q4TransformerBlock {
    norm1: RmsNormLayer,
    attention: Q4Attention,
    norm2: RmsNormLayer,
    ffn: Q4FeedForward,
}

impl Q4TransformerBlock {
    pub fn new(
        norm1: RmsNormLayer,
        attention: Q4Attention,
        norm2: RmsNormLayer,
        ffn: Q4FeedForward,
    ) -> Self {
        Self {
            norm1,
            attention,
            norm2,
            ffn,
        }
    }

    pub fn forward_with_cache(
        &self,
        x: Tensor<Wgpu, 3>,
        rope: &RoPE,
        cache: &mut KVCache,
    ) -> Tensor<Wgpu, 3> {
        // Fuse residual addition into out_proj matmul (saves 1 dispatch)
        let residual = x.clone();
        let x = self.norm1.forward(x);
        let x = self.attention.forward_with_cache_fused_residual(x, residual, rope, cache);

        // Fuse residual addition into linear_out matmul (saves 1 dispatch)
        let residual = x.clone();
        let x = self.norm2.forward(x);
        self.ffn.forward_with_residual(x, residual)
    }

    pub fn norm1(&self) -> &RmsNormLayer { &self.norm1 }
    pub fn norm2(&self) -> &RmsNormLayer { &self.norm2 }
    pub fn attention(&self) -> &Q4Attention { &self.attention }
    pub fn ffn(&self) -> &Q4FeedForward { &self.ffn }
}

// ---------------------------------------------------------------------------
// TemporalTransformer
// ---------------------------------------------------------------------------

/// The temporal transformer (~7B params).
///
/// Processes time steps autoregressively. Each step receives the sum of 17
/// stream embeddings:
/// - Stream 0: text token (inner monologue)
/// - Streams 1-8: model's audio codebook tokens
/// - Streams 9-16: user's audio codebook tokens
///
/// Outputs hidden state [1, 1, 4096] for the depth transformer,
/// and text logits via the text_linear head.
pub struct TemporalTransformer {
    /// Text input embedding (vocab 32001)
    text_emb: EmbeddingStore,
    /// Per-codebook audio embeddings for model stream (8 codebooks, vocab 2049 each)
    model_audio_emb: Vec<EmbeddingStore>,
    /// Per-codebook audio embeddings for user stream (8 codebooks, vocab 2049 each)
    user_audio_emb: Vec<EmbeddingStore>,
    /// Transformer layers
    layers: Vec<Q4TransformerBlock>,
    /// Shared RoPE
    rope: RoPE,
    /// Output normalization
    out_norm: RmsNormLayer,
    /// Text output head: hidden_size -> text_vocab_size
    text_linear: Linear,
    config: StsConfig,
    device: WgpuDevice,
}

impl TemporalTransformer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        text_emb: EmbeddingStore,
        model_audio_emb: Vec<EmbeddingStore>,
        user_audio_emb: Vec<EmbeddingStore>,
        layers: Vec<Q4TransformerBlock>,
        rope: RoPE,
        out_norm: RmsNormLayer,
        text_linear: Linear,
        config: StsConfig,
        device: WgpuDevice,
    ) -> Self {
        Self {
            text_emb,
            model_audio_emb,
            user_audio_emb,
            layers,
            rope,
            out_norm,
            text_linear,
            config,
            device,
        }
    }

    /// Single-step forward pass.
    ///
    /// `user_audio_tokens`: 8 token IDs from Mimi encoder (user stream).
    /// `model_audio_tokens`: 8 token IDs from previous depth step (model stream).
    /// `text_token`: previous text token from inner monologue.
    /// `cache`: KV cache (updated in place).
    ///
    /// Returns (hidden_state [1, 1, hidden_size], text_logits [1, 1, text_vocab_size]).
    pub fn forward(
        &self,
        user_audio_tokens: &[i32],
        model_audio_tokens: &[i32],
        text_token: i32,
        cache: &mut LayerCaches,
    ) -> (Tensor<Wgpu, 3>, Tensor<Wgpu, 3>) {
        let dim = self.config.hidden_size;

        // Sum all 17 stream embeddings on CPU, then upload once.
        // Tokens with value -1 are masked (contribute zero), matching Swift's
        // delay handling where unwritten cache positions contain -1.
        let mut sum = vec![0.0f32; dim];

        // Stream 0: text embedding (skip if -1)
        if text_token >= 0 {
            self.text_emb.embed_id_add_cpu(text_token as u32, &mut sum);
        }

        // Streams 1-8: model audio codebook embeddings (skip -1)
        for (i, &token) in model_audio_tokens.iter().enumerate() {
            if token >= 0 && i < self.model_audio_emb.len() {
                self.model_audio_emb[i].embed_id_add_cpu(token as u32, &mut sum);
            }
        }

        // Streams 9-16: user audio codebook embeddings (skip -1)
        for (i, &token) in user_audio_tokens.iter().enumerate() {
            if token >= 0 && i < self.user_audio_emb.len() {
                self.user_audio_emb[i].embed_id_add_cpu(token as u32, &mut sum);
            }
        }

        let input = Tensor::<Wgpu, 3>::from_data(
            burn::tensor::TensorData::new(sum, [1, 1, dim]),
            &self.device,
        );

        // Run 32 transformer layers with KV cache
        let mut x = input;
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(c) = cache.get_mut(i) {
                x = layer.forward_with_cache(x, &self.rope, c);
            }
        }

        // Output norm
        let hidden = self.out_norm.forward(x);

        // Text logits head
        let text_logits = self.text_linear.forward(hidden.clone());

        (hidden, text_logits)
    }

    /// Forward pass with a pre-computed embedding (bypasses token lookup).
    ///
    /// Used for voice preset prefill where embeddings are pre-computed
    /// and stored externally (e.g. NATF2.pt).
    ///
    /// `input` shape: [1, 1, hidden_size]
    pub fn forward_embeddings(
        &self,
        input: Tensor<Wgpu, 3>,
        cache: &mut LayerCaches,
    ) -> (Tensor<Wgpu, 3>, Tensor<Wgpu, 3>) {
        let mut x = input;
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(c) = cache.get_mut(i) {
                x = layer.forward_with_cache(x, &self.rope, c);
            }
        }
        let hidden = self.out_norm.forward(x);
        let text_logits = self.text_linear.forward(hidden.clone());
        (hidden, text_logits)
    }

    /// Create KV caches for this model.
    pub fn create_cache(&self) -> LayerCaches {
        let head_dim = self.config.hidden_size / self.config.num_heads;
        LayerCaches::new(
            self.config.num_layers,
            self.config.max_seq_len,
            self.config.num_kv_heads,
            head_dim,
            &self.device,
        )
    }

    pub fn config(&self) -> &StsConfig {
        &self.config
    }

    pub fn device(&self) -> &WgpuDevice {
        &self.device
    }

    pub fn text_emb(&self) -> &EmbeddingStore {
        &self.text_emb
    }

    pub fn model_audio_emb(&self) -> &[EmbeddingStore] {
        &self.model_audio_emb
    }

    pub fn user_audio_emb(&self) -> &[EmbeddingStore] {
        &self.user_audio_emb
    }

    pub fn layers(&self) -> &[Q4TransformerBlock] {
        &self.layers
    }

    pub fn rope(&self) -> &RoPE {
        &self.rope
    }

    /// Single-step forward pass with intermediate logging.
    ///
    /// Same logic as `forward`, but collects per-layer hidden states
    /// for debugging / comparison against a BF16 reference.
    ///
    /// Returns (hidden_state, text_logits, log) where `log` is a
    /// `TemporalForwardLog` containing embedding sum, per-layer hidden
    /// states, and post-norm hidden state.
    ///
    /// NOT available in WASM: uses sync GPU readback (`.to_data()`).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn forward_with_logging(
        &self,
        user_audio_tokens: &[i32],
        model_audio_tokens: &[i32],
        text_token: i32,
        cache: &mut LayerCaches,
    ) -> (Tensor<Wgpu, 3>, Tensor<Wgpu, 3>, TemporalForwardLog) {
        let dim = self.config.hidden_size;

        // Sum all 17 stream embeddings on CPU, then upload once.
        let mut sum = vec![0.0f32; dim];

        // Stream 0: text embedding (skip if -1)
        if text_token >= 0 {
            self.text_emb.embed_id_add_cpu(text_token as u32, &mut sum);
        }

        // Streams 1-8: model audio codebook embeddings (skip -1)
        for (i, &token) in model_audio_tokens.iter().enumerate() {
            if token >= 0 && i < self.model_audio_emb.len() {
                self.model_audio_emb[i].embed_id_add_cpu(token as u32, &mut sum);
            }
        }

        // Streams 9-16: user audio codebook embeddings (skip -1)
        for (i, &token) in user_audio_tokens.iter().enumerate() {
            if token >= 0 && i < self.user_audio_emb.len() {
                self.user_audio_emb[i].embed_id_add_cpu(token as u32, &mut sum);
            }
        }

        // Log embedding sum (CPU, no GPU readback needed)
        let emb_norm = sum.iter().map(|x| x * x).sum::<f32>().sqrt();
        let emb_first_10: Vec<f32> = sum[..10.min(dim)].to_vec();

        let input = Tensor::<Wgpu, 3>::from_data(
            burn::tensor::TensorData::new(sum, [1, 1, dim]),
            &self.device,
        );

        // Run 32 transformer layers with KV cache, logging after each
        let mut x = input;
        let mut layer_logs = Vec::with_capacity(self.layers.len());
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(c) = cache.get_mut(i) {
                x = layer.forward_with_cache(x, &self.rope, c);
            }
            // Read back hidden state for logging
            let flat: Tensor<Wgpu, 1> = x.clone().reshape([dim]);
            let vals: Vec<f32> = flat.to_data().to_vec().expect("layer hidden to_vec");
            let norm = vals.iter().map(|v| v * v).sum::<f32>().sqrt();
            let first_10: Vec<f32> = vals[..10.min(dim)].to_vec();
            layer_logs.push(LayerLog {
                layer: i,
                norm,
                first_10,
            });
        }

        // Output norm
        let hidden = self.out_norm.forward(x);

        // Log post-norm hidden state
        let flat: Tensor<Wgpu, 1> = hidden.clone().reshape([dim]);
        let after_norm_vals: Vec<f32> = flat.to_data().to_vec().expect("after_norm to_vec");
        let after_norm_norm = after_norm_vals.iter().map(|v| v * v).sum::<f32>().sqrt();
        let after_norm_first_10: Vec<f32> = after_norm_vals[..10.min(dim)].to_vec();

        // Text logits head
        let text_logits = self.text_linear.forward(hidden.clone());

        let log = TemporalForwardLog {
            embedding_sum_norm: emb_norm,
            embedding_sum_first_10: emb_first_10,
            layer_logs,
            after_out_norm_norm: after_norm_norm,
            after_out_norm_first_10: after_norm_first_10,
        };

        (hidden, text_logits, log)
    }
}

/// Per-layer log entry for debugging (native only).
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, serde::Serialize)]
pub struct LayerLog {
    pub layer: usize,
    pub norm: f32,
    pub first_10: Vec<f32>,
}

/// Full log of temporal transformer forward pass intermediates (native only).
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, serde::Serialize)]
pub struct TemporalForwardLog {
    pub embedding_sum_norm: f32,
    pub embedding_sum_first_10: Vec<f32>,
    pub layer_logs: Vec<LayerLog>,
    pub after_out_norm_norm: f32,
    pub after_out_norm_first_10: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Sampling utilities
// ---------------------------------------------------------------------------

/// Top-k sampling with temperature.
///
/// Returns the sampled token ID as u32 (requires readback from GPU).
pub async fn sample_top_k(
    logits: Tensor<Wgpu, 3>,
    top_k: usize,
    temperature: f32,
) -> u32 {
    sample_top_k_with_penalty(logits, top_k, temperature, &[], 1.0).await
}

/// Top-k sampling with temperature and repetition penalty.
///
/// `logits`: [1, 1, vocab_size] tensor
/// `top_k`: number of top tokens to consider
/// `temperature`: sampling temperature
/// `past_tokens`: recent token history for penalty
/// `penalty`: repetition penalty factor (e.g. 1.2). Values > 1.0 discourage
///   repetition; 1.0 disables the penalty.
///
/// Penalty is applied before temperature scaling and top-K filtering:
///   - if logit > 0: divide by penalty
///   - if logit < 0: multiply by penalty
///
/// Returns the sampled token ID as u32 (requires readback from GPU).
pub async fn sample_top_k_with_penalty(
    logits: Tensor<Wgpu, 3>,
    top_k: usize,
    temperature: f32,
    past_tokens: &[u32],
    penalty: f32,
) -> u32 {
    let [_b, _s, vocab] = logits.dims();
    // Squeeze to 1D: [vocab]
    let logits_1d: Tensor<Wgpu, 1> = logits.reshape([vocab]);

    // Read logits to CPU for sampling
    let mut logits_vec: Vec<f32> = logits_1d
        .into_data_async()
        .await
        .expect("GPU readback failed")
        .to_vec()
        .expect("logits to_vec failed");

    // Apply repetition penalty (before temperature)
    if penalty != 1.0 && !past_tokens.is_empty() {
        let seen: std::collections::HashSet<u32> = past_tokens.iter().copied().collect();
        for &tok in &seen {
            let idx = tok as usize;
            if idx < vocab {
                if logits_vec[idx] > 0.0 {
                    logits_vec[idx] /= penalty;
                } else {
                    logits_vec[idx] *= penalty;
                }
            }
        }
    }

    // Apply temperature
    if temperature != 1.0 {
        for v in &mut logits_vec {
            *v /= temperature;
        }
    }

    // Find top-k indices using partial sort (O(n) partition + O(k log k) sort)
    let mut indexed: Vec<(usize, f32)> = logits_vec.iter().copied().enumerate().collect();
    if indexed.len() > top_k {
        indexed.select_nth_unstable_by(top_k, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        indexed.truncate(top_k);
    }
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Softmax over top-k
    let max_val = indexed[0].1;
    let exp_sum: f32 = indexed.iter().map(|(_, v)| (v - max_val).exp()).collect::<Vec<f32>>().iter().sum();
    let probs: Vec<(usize, f32)> = indexed
        .iter()
        .map(|&(idx, v)| (idx, (v - max_val).exp() / exp_sum))
        .collect();

    // Sample from distribution (simple linear scan)
    let r: f32 = pseudo_random();
    let mut cumulative = 0.0f32;
    for &(idx, p) in &probs {
        cumulative += p;
        if r < cumulative {
            return idx as u32;
        }
    }
    probs.last().map(|&(idx, _)| idx as u32).unwrap_or(0)
}

/// Greedy argmax sampling (no temperature/top-k).
pub async fn sample_greedy(logits: Tensor<Wgpu, 3>) -> u32 {
    let [_b, _s, vocab] = logits.dims();
    let logits_1d: Tensor<Wgpu, 1> = logits.reshape([vocab]);
    let argmax = logits_1d.argmax(0);
    let data: Vec<i32> = argmax
        .into_data_async()
        .await
        .expect("GPU readback failed")
        .to_vec()
        .expect("argmax to_vec failed");
    data[0] as u32
}

/// Simple pseudo-random number in [0, 1) for sampling.
///
/// Uses a basic LCG seeded from a global counter. Not cryptographic,
/// but sufficient for top-k sampling diversity.
pub(crate) fn pseudo_random() -> f32 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEED: AtomicU64 = AtomicU64::new(0);

    // Lazy init: seed from time on first call
    let s = SEED.fetch_add(1, Ordering::Relaxed);
    if s == 0 {
        #[cfg(target_family = "wasm")]
        let time_seed = (js_sys::Date::now() * 1000.0) as u64;
        #[cfg(not(target_family = "wasm"))]
        let time_seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        SEED.store(time_seed, Ordering::Relaxed);
        return pseudo_random(); // re-enter with seeded value
    }
    // LCG: (a * s + c) mod m
    let next = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    SEED.store(next, Ordering::Relaxed);
    // Map to [0, 1)
    (next >> 33) as f32 / (1u64 << 31) as f32
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_device() -> WgpuDevice {
        WgpuDevice::default()
    }

    #[test]
    fn test_rope_shape() {
        pollster::block_on(async {
            let device = test_device();
            let rope = RoPE::new(128, 100, 10000.0, &device);

            // Create dummy Q, K: [1, 5, 4, 128]
            let q = Tensor::<Wgpu, 4>::zeros([1, 5, 4, 128], &device);
            let k = Tensor::<Wgpu, 4>::zeros([1, 5, 4, 128], &device);

            let (q_rot, k_rot) = rope.apply(q, k, 0);

            assert_eq!(q_rot.dims(), [1, 5, 4, 128]);
            assert_eq!(k_rot.dims(), [1, 5, 4, 128]);
        });
    }

    #[test]
    fn test_rope_offset() {
        pollster::block_on(async {
            let device = test_device();
            let rope = RoPE::new(128, 100, 10000.0, &device);

            let q = Tensor::<Wgpu, 4>::zeros([1, 1, 4, 128], &device);
            let k = Tensor::<Wgpu, 4>::zeros([1, 1, 4, 128], &device);

            // Should not panic with offset
            let (q_rot, k_rot) = rope.apply(q, k, 50);
            assert_eq!(q_rot.dims(), [1, 1, 4, 128]);
            assert_eq!(k_rot.dims(), [1, 1, 4, 128]);
        });
    }

    #[test]
    fn test_kv_cache_ring_buffer() {
        pollster::block_on(async {
            let device = test_device();
            let mut cache = KVCache::new(4, 2, 8, &device);

            assert_eq!(cache.seq_len(), 0);
            assert_eq!(cache.offset(), 0);

            // Insert 3 entries
            for i in 0..3 {
                let k = Tensor::<Wgpu, 4>::ones([1, 2, 1, 8], &device) * (i as f64 + 1.0);
                let v = Tensor::<Wgpu, 4>::ones([1, 2, 1, 8], &device) * (i as f64 + 1.0);
                let (k_all, v_all) = cache.update(k, v);
                assert_eq!(cache.seq_len(), i + 1);
                assert_eq!(k_all.dims()[2], i + 1);
                assert_eq!(v_all.dims()[2], i + 1);
            }

            assert_eq!(cache.offset(), 3);

            // Fill up (4th entry = full)
            let k = Tensor::<Wgpu, 4>::ones([1, 2, 1, 8], &device) * 4.0;
            let v = Tensor::<Wgpu, 4>::ones([1, 2, 1, 8], &device) * 4.0;
            let (k_all, _) = cache.update(k, v);
            assert_eq!(cache.seq_len(), 4);
            assert_eq!(k_all.dims()[2], 4);

            // 5th entry: wraps around
            let k = Tensor::<Wgpu, 4>::ones([1, 2, 1, 8], &device) * 5.0;
            let v = Tensor::<Wgpu, 4>::ones([1, 2, 1, 8], &device) * 5.0;
            let (k_all, _) = cache.update(k, v);
            assert_eq!(cache.seq_len(), 4); // still max_len
            assert_eq!(k_all.dims()[2], 4);
            assert_eq!(cache.offset(), 5);
        });
    }

    #[test]
    fn test_kv_cache_reset() {
        pollster::block_on(async {
            let device = test_device();
            let mut cache = KVCache::new(4, 2, 8, &device);

            let k = Tensor::<Wgpu, 4>::ones([1, 2, 1, 8], &device);
            let v = Tensor::<Wgpu, 4>::ones([1, 2, 1, 8], &device);
            cache.update(k, v);

            cache.reset();
            assert_eq!(cache.seq_len(), 0);
            assert_eq!(cache.offset(), 0);
        });
    }

    #[test]
    fn test_kv_cache_reset_keep_buffers() {
        pollster::block_on(async {
            let device = test_device();
            let mut cache = KVCache::new(4, 2, 8, &device);

            let k = Tensor::<Wgpu, 4>::ones([1, 2, 1, 8], &device);
            let v = Tensor::<Wgpu, 4>::ones([1, 2, 1, 8], &device);
            cache.update(k, v);

            cache.reset_keep_buffers();
            assert_eq!(cache.seq_len(), 0);
            assert_eq!(cache.offset(), 0);
            // Buffers should still be allocated
            assert!(cache.k.is_some());
            assert!(cache.v.is_some());
        });
    }

    #[test]
    fn test_layer_caches() {
        pollster::block_on(async {
            let device = test_device();
            let mut caches = LayerCaches::new(3, 10, 2, 8, &device);

            assert_eq!(caches.seq_len(), 0);

            // Update first layer
            if let Some(c) = caches.get_mut(0) {
                let k = Tensor::<Wgpu, 4>::ones([1, 2, 1, 8], &device);
                let v = Tensor::<Wgpu, 4>::ones([1, 2, 1, 8], &device);
                c.update(k, v);
            }

            assert_eq!(caches.seq_len(), 1);

            caches.reset();
            assert_eq!(caches.seq_len(), 0);
        });
    }

    #[test]
    fn test_rms_norm_shape() {
        pollster::block_on(async {
            let device = test_device();
            let alpha = Tensor::<Wgpu, 1>::ones([64], &device);
            let norm = RmsNormLayer::new(alpha, 1e-5);

            let x = Tensor::<Wgpu, 3>::ones([1, 1, 64], &device) * 3.0;
            let out = norm.forward(x);

            assert_eq!(out.dims(), [1, 1, 64]);
        });
    }

    #[test]
    fn test_rms_norm_unit() {
        pollster::block_on(async {
            let device = test_device();
            let alpha = Tensor::<Wgpu, 1>::ones([4], &device);
            let norm = RmsNormLayer::new(alpha, 1e-8);

            let data = vec![1.0f32, 2.0, 3.0, 4.0];
            let x = Tensor::<Wgpu, 3>::from_data(
                burn::tensor::TensorData::new(data.clone(), [1, 1, 4]),
                &device,
            );
            let out = norm.forward(x);
            let result: Vec<f32> = out.reshape([4]).to_data().to_vec().expect("to_vec");

            // RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ~ 2.7386
            let rms = (data.iter().map(|x| x * x).sum::<f32>() / 4.0).sqrt();
            for (i, &v) in result.iter().enumerate() {
                let expected = data[i] / rms;
                assert!((v - expected).abs() < 1e-4, "idx {i}: got {v}, expected {expected}");
            }
        });
    }

    #[test]
    fn test_sample_greedy() {
        pollster::block_on(async {
            let device = test_device();
            // Logits where idx 3 is highest
            let mut data = vec![-10.0f32; 10];
            data[3] = 100.0;
            let logits = Tensor::<Wgpu, 3>::from_data(
                burn::tensor::TensorData::new(data, [1, 1, 10]),
                &device,
            );
            let token = sample_greedy(logits).await;
            assert_eq!(token, 3);
        });
    }

    #[test]
    fn test_sample_top_k_returns_valid_token() {
        pollster::block_on(async {
            let device = test_device();
            let mut data = vec![0.0f32; 100];
            data[42] = 50.0;
            let logits = Tensor::<Wgpu, 3>::from_data(
                burn::tensor::TensorData::new(data, [1, 1, 100]),
                &device,
            );
            let token = sample_top_k(logits, 5, 1.0).await;
            // With such strong logit at 42, top-k should return 42
            assert_eq!(token, 42);
        });
    }

    #[test]
    fn test_sample_top_k_with_penalty_suppresses_repeated() {
        pollster::block_on(async {
            let device = test_device();
            // Token 42 has the highest logit, but if it's in the penalty history
            // its logit should be divided by the penalty factor, potentially
            // allowing token 10 (second-highest) to win.
            let mut data = vec![0.0f32; 100];
            data[42] = 5.0;
            data[10] = 4.9;
            let logits = Tensor::<Wgpu, 3>::from_data(
                burn::tensor::TensorData::new(data.clone(), [1, 1, 100]),
                &device,
            );
            // Without penalty: greedy should pick 42
            let token_no_penalty = sample_top_k_with_penalty(logits, 1, 1.0, &[], 1.0).await;
            assert_eq!(token_no_penalty, 42);

            // With penalty on token 42: logit becomes 5.0/2.0 = 2.5, token 10's 4.9 wins
            let logits2 = Tensor::<Wgpu, 3>::from_data(
                burn::tensor::TensorData::new(data, [1, 1, 100]),
                &device,
            );
            let token_with_penalty = sample_top_k_with_penalty(logits2, 1, 1.0, &[42], 2.0).await;
            assert_eq!(token_with_penalty, 10);
        });
    }

    #[test]
    fn test_sample_top_k_with_penalty_negative_logits() {
        pollster::block_on(async {
            let device = test_device();
            // For negative logits, penalty multiplies (makes more negative)
            let mut data = vec![-10.0f32; 50];
            data[5] = -1.0; // least negative = highest
            data[10] = -1.1;
            let logits = Tensor::<Wgpu, 3>::from_data(
                burn::tensor::TensorData::new(data.clone(), [1, 1, 50]),
                &device,
            );
            // Without penalty: token 5 wins
            let token = sample_top_k_with_penalty(logits, 1, 1.0, &[], 1.0).await;
            assert_eq!(token, 5);

            // With penalty on token 5: -1.0 * 2.0 = -2.0, token 10 at -1.1 wins
            let logits2 = Tensor::<Wgpu, 3>::from_data(
                burn::tensor::TensorData::new(data, [1, 1, 50]),
                &device,
            );
            let token = sample_top_k_with_penalty(logits2, 1, 1.0, &[5], 2.0).await;
            assert_eq!(token, 10);
        });
    }

    #[test]
    fn test_pseudo_random_range() {
        for _ in 0..100 {
            let r = pseudo_random();
            assert!(r >= 0.0 && r < 1.0, "out of range: {r}");
        }
    }
}
