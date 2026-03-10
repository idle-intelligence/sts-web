//! Custom WebGPU depth engine: bypasses Burn/CubeCL to encode all 8 depth
//! steps as ONE command buffer.
//!
//! The existing depth transformer path goes through Burn -> CubeCL -> wgpu for
//! every dispatch, creating ~150+ command buffers per frame. This engine
//! pre-creates all pipelines and bind groups, then records one command buffer
//! containing all 8 depth steps (embedding lookup, 6 transformer layers each,
//! output linear, argmax) and submits it in a single `queue.submit()`.
//!
//! Architecture (per frame):
//!   1. Encode 8 input projections (temporal_hidden @ input_proj weights)
//!   2. For each of 8 steps:
//!      a. Embedding lookup (Q4 dequant) for previous token
//!      b. Vec add: projected + embedding
//!      c. 6 layers of: RMSNorm → Attention (Q/K/V proj, flash-decode, O proj) → Residual → RMSNorm → SwiGLU FFN → Residual
//!      d. Output linear (logits)
//!      e. Argmax (greedy sampling)
//!   3. Copy 8 token results to staging buffer
//!   4. Submit ONE command buffer, mapAsync, return 8 u32 tokens

use std::num::NonZeroU64;
use std::sync::Arc;

use burn::backend::wgpu::{WgpuDevice, WgpuResource};

use crate::depth::DepthTransformer;
use crate::gguf::handle_to_wgpu_resource;
use crate::StsConfig;

// ---------------------------------------------------------------------------
// Shader sources (included at compile time)
// ---------------------------------------------------------------------------

const SHADER_Q4K_MATVEC: &str = include_str!("wgsl/shader_q4k_matvec.wgsl");
const SHADER_RMSNORM: &str = include_str!("wgsl/shader_rmsnorm.wgsl");
const SHADER_ATTENTION: &str = include_str!("wgsl/shader_attention.wgsl");
const SHADER_EMBED_Q4: &str = include_str!("wgsl/shader_embed_q4.wgsl");
const SHADER_ARGMAX: &str = include_str!("wgsl/shader_argmax.wgsl");
const SHADER_SWIGLU: &str = include_str!("wgsl/shader_swiglu.wgsl");
const SHADER_VEC_ADD: &str = include_str!("wgsl/shader_vec_add.wgsl");
const SHADER_CACHE_WRITE: &str = include_str!("wgsl/shader_cache_write.wgsl");

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Static depth engine configuration derived from StsConfig.
#[derive(Clone, Debug)]
pub struct DepthEngineConfig {
    /// Number of depth steps to run (8 for generation).
    pub num_steps: usize,
    /// Hidden dimension of the depth transformer (1024).
    pub dim: usize,
    /// Number of transformer layers (6).
    pub num_layers: usize,
    /// Number of attention heads (16).
    pub num_heads: usize,
    /// Number of KV heads (16, MHA).
    pub num_kv_heads: usize,
    /// Head dimension (dim / num_heads = 64).
    pub head_dim: usize,
    /// FFN intermediate size (2816, gating = 5632).
    pub ffn_intermediate: usize,
    /// QKV projection output size (3 * dim = 3072).
    pub qkv_dim: usize,
    /// Attention scale factor.
    pub attn_scale: f32,
    /// RMSNorm epsilon.
    pub rms_eps: f32,
    /// KV cache max length (depformer_context + 1 = 9).
    pub kv_cache_max_len: usize,
    /// Audio vocab size (2048 for output heads).
    pub audio_vocab_size: usize,
    /// Text vocab size (32000 for output head 0).
    pub text_vocab_size: usize,
    /// Text embedding vocab size (32001).
    pub text_emb_vocab_size: usize,
    /// Audio embedding vocab size (2049).
    pub audio_emb_vocab_size: usize,
    /// Temporal hidden dimension (4096, for input projections).
    pub temporal_dim: usize,
}

impl DepthEngineConfig {
    pub fn from_sts_config(config: &StsConfig) -> Self {
        let dim = config.depth_hidden_size;
        let num_heads = config.depth_num_heads;
        Self {
            num_steps: config.depth_gen_steps,
            dim,
            num_layers: config.depth_num_layers,
            num_heads,
            num_kv_heads: config.depth_num_kv_heads,
            head_dim: dim / num_heads,
            ffn_intermediate: config.depth_intermediate_size,
            qkv_dim: 3 * dim,
            attn_scale: (dim as f32 / num_heads as f32).powf(-0.5),
            rms_eps: 1e-8,
            kv_cache_max_len: 9, // depformer_context=8 + 1 current step
            audio_vocab_size: 2048,
            text_vocab_size: config.text_vocab_size,
            text_emb_vocab_size: config.text_in_vocab_size,
            audio_emb_vocab_size: config.audio_vocab_size,
            temporal_dim: config.hidden_size,
        }
    }
}

// ---------------------------------------------------------------------------
// Weight buffer references extracted from Burn/CubeCL
// ---------------------------------------------------------------------------

/// All weight buffers for one transformer layer at one depth step.
struct LayerStepWeights {
    /// Attention QKV in_proj: Q4K weights buffer [3072, 1024].
    attn_in_proj: Arc<wgpu::Buffer>,
    attn_in_proj_offset: u64,
    attn_in_proj_size: u64,

    /// Attention output projection: Q4K weights buffer [1024, 1024].
    attn_out_proj: Arc<wgpu::Buffer>,
    attn_out_proj_offset: u64,
    attn_out_proj_size: u64,

    /// FFN linear_in: Q4K weights buffer [5632, 1024] (gate + value packed).
    ffn_linear_in: Arc<wgpu::Buffer>,
    ffn_linear_in_offset: u64,
    ffn_linear_in_size: u64,

    /// FFN linear_out: Q4K weights buffer [1024, 2816].
    ffn_linear_out: Arc<wgpu::Buffer>,
    ffn_linear_out_offset: u64,
    ffn_linear_out_size: u64,

    /// RMSNorm 1 alpha: f32 buffer [dim].
    norm1_alpha: Arc<wgpu::Buffer>,
    norm1_alpha_offset: u64,
    norm1_alpha_size: u64,

    /// RMSNorm 2 alpha: f32 buffer [dim].
    norm2_alpha: Arc<wgpu::Buffer>,
    norm2_alpha_offset: u64,
    norm2_alpha_size: u64,
}

/// Per-step (codebook) weight buffers.
struct StepWeights {
    /// Input projection: Q4K weights [1024, 4096].
    input_proj: Arc<wgpu::Buffer>,
    input_proj_offset: u64,
    input_proj_size: u64,

    /// Output linear: Q4K weights [vocab, 1024].
    output_linear: Arc<wgpu::Buffer>,
    output_linear_offset: u64,
    output_linear_size: u64,
    output_vocab: usize,
}

/// Embedding data for GPU lookup.
struct EmbeddingData {
    /// Q4_0 data buffer on GPU.
    q4_data: Arc<wgpu::Buffer>,
    q4_data_offset: u64,
    q4_data_size: u64,
    vocab_size: usize,
    dim: usize,
    bytes_per_row: usize,
}

// ---------------------------------------------------------------------------
// DepthEngine
// ---------------------------------------------------------------------------

/// Custom WebGPU depth engine that encodes all 8 depth steps as one command buffer.
pub struct DepthEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,

    // ---- Compute pipelines (created once) ----
    q4k_matvec_pipeline: wgpu::ComputePipeline,
    rmsnorm_pipeline: wgpu::ComputePipeline,
    attention_pipeline: wgpu::ComputePipeline,
    embed_q4_pipeline: wgpu::ComputePipeline,
    argmax_pipeline: wgpu::ComputePipeline,
    swiglu_pipeline: wgpu::ComputePipeline,
    vec_add_pipeline: wgpu::ComputePipeline,
    cache_write_pipeline: wgpu::ComputePipeline,

    // ---- Weight buffers (persistent, extracted from Burn) ----

    /// Per-step weights: input_projs and output_linears.
    /// Index: [step] for steps 0..num_steps.
    step_weights: Vec<StepWeights>,

    /// Per-layer, per-step weights for transformer blocks.
    /// Index: [layer][step].
    layer_step_weights: Vec<Vec<LayerStepWeights>>,

    /// Text embedding (step 0 input).
    text_emb: EmbeddingData,

    /// Audio embeddings (steps 1..num_steps input).
    /// Index: [step - 1] for steps 1..num_steps.
    audio_embs: Vec<EmbeddingData>,

    // ---- Scratch buffers (pre-allocated, reused each frame) ----

    /// Per-step projected temporal hidden: [dim] f32 each.
    projected_bufs: Vec<wgpu::Buffer>,

    /// Per-step embedding output: [dim] f32 each.
    embed_out_bufs: Vec<wgpu::Buffer>,

    /// Hidden state buffer A (ping): [dim] f32.
    hidden_a: wgpu::Buffer,

    /// Hidden state buffer B (pong): [dim] f32.
    hidden_b: wgpu::Buffer,

    /// RMSNorm output: [dim] f32.
    norm_out: wgpu::Buffer,

    /// QKV projection output: [3 * dim] f32.
    qkv_out: wgpu::Buffer,

    /// Attention output: [dim] f32 (num_heads * head_dim).
    attn_out: wgpu::Buffer,

    /// Attention + out_proj output (before residual): [dim] f32.
    attn_proj_out: wgpu::Buffer,

    /// FFN linear_in output: [2 * ffn_intermediate] f32 (gate + value).
    ffn_gate_out: wgpu::Buffer,

    /// SwiGLU activation output: [ffn_intermediate] f32.
    swiglu_out: wgpu::Buffer,

    /// FFN linear_out output (before residual): [dim] f32.
    ffn_proj_out: wgpu::Buffer,

    /// Output logits buffer: [max_vocab] f32.
    logits_buf: wgpu::Buffer,

    /// Per-step argmax result: one 4-byte buffer per step.
    /// Separate buffers avoid WebGPU's 256-byte offset alignment requirement
    /// (storage buffer bindings must be aligned to minStorageBufferOffsetAlignment).
    token_result_bufs: Vec<wgpu::Buffer>,

    /// Staging buffer for readback: 8 x u32 = 32 bytes.
    staging_buf: wgpu::Buffer,

    // ---- KV cache buffers ----
    // Per-layer K and V caches: [num_heads, kv_cache_max_len, head_dim] f32.
    // Index: [layer].

    /// K cache buffers per layer.
    k_caches: Vec<wgpu::Buffer>,

    /// V cache buffers per layer.
    v_caches: Vec<wgpu::Buffer>,

    // ---- Pre-computed info buffers ----

    /// Q4K matvec info for input_proj: [B=1, M=1, K=temporal_dim, N=dim, blocks_per_row].
    info_input_proj: wgpu::Buffer,

    /// Q4K matvec info for attn in_proj: [B=1, M=1, K=dim, N=3*dim, blocks_per_row].
    info_attn_in_proj: wgpu::Buffer,

    /// Q4K matvec info for attn out_proj: [B=1, M=1, K=dim, N=dim, blocks_per_row].
    info_attn_out_proj: wgpu::Buffer,

    /// Q4K matvec info for ffn linear_in: [B=1, M=1, K=dim, N=2*ffn_intermediate, blocks_per_row].
    info_ffn_linear_in: wgpu::Buffer,

    /// Q4K matvec info for ffn linear_out: [B=1, M=1, K=ffn_intermediate, N=dim, blocks_per_row].
    info_ffn_linear_out: wgpu::Buffer,

    /// Per-step Q4K matvec info for output linear (vocab varies per step).
    info_output_linear: Vec<wgpu::Buffer>,

    /// RMSNorm info: [D, eps_bits, total_rows=1].
    info_rmsnorm: wgpu::Buffer,

    /// Per-step attention info: [num_heads, head_dim, seq_len_for_step, scale_bits, kv_head_stride].
    /// seq_len increments with each step (step 0 = 1, step 1 = 2, etc.).
    info_attention: Vec<wgpu::Buffer>,

    /// SwiGLU info: [half_dim = ffn_intermediate].
    info_swiglu: wgpu::Buffer,

    /// Vec add info for dim-sized adds: [len = dim].
    info_vec_add_dim: wgpu::Buffer,

    /// Per-step embedding info: [dim, bytes_per_row, vocab_size].
    info_embed: Vec<wgpu::Buffer>,

    /// Per-step argmax info: [vocab_size].
    info_argmax: Vec<wgpu::Buffer>,

    /// Per-step, per-layer cache write info: [pos_offset, B=1, S=1, H=num_heads, D=head_dim, max_len].
    /// pos_offset varies by step. Stored as [step][layer] but layer dimension is constant
    /// so we just store [step].
    info_cache_write: Vec<wgpu::Buffer>,

    // ---- Config ----
    config: DepthEngineConfig,
}

// ---------------------------------------------------------------------------
// Pipeline creation helper
// ---------------------------------------------------------------------------

fn create_pipeline(
    device: &wgpu::Device,
    shader_source: &str,
    label: &str,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None, // Auto-layout from shader reflection
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

// ---------------------------------------------------------------------------
// Info buffer creation helpers
// ---------------------------------------------------------------------------

/// Create a GPU buffer initialized with the given u32 data.
fn create_info_buffer_init(device: &wgpu::Device, label: &str, data: &[u32]) -> wgpu::Buffer {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    {
        let mut view = buf.slice(..).get_mapped_range_mut();
        view.copy_from_slice(&bytes);
    }
    buf.unmap();
    buf
}

/// Create a storage buffer (read_write) of the given size in bytes.
fn create_storage_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Extract a WgpuResource from a CubeCL Handle and return the Arc<Buffer>, offset, size.
fn extract_buffer(
    handle: &cubecl::server::Handle,
    wgpu_device: &WgpuDevice,
) -> (Arc<wgpu::Buffer>, u64, u64) {
    let res = handle_to_wgpu_resource(handle, wgpu_device);
    (Arc::new(res.buffer), res.offset, res.size)
}

// ---------------------------------------------------------------------------
// Bind group helpers
// ---------------------------------------------------------------------------

/// Helper to create a bind group entry with buffer binding.
fn buf_entry(
    binding: u32,
    buffer: &wgpu::Buffer,
    offset: u64,
    size: u64,
) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer,
            offset,
            size: Some(NonZeroU64::new(size).expect("buffer size must be > 0")),
        }),
    }
}

/// Helper for a whole-buffer bind group entry.
fn buf_entry_whole(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer,
            offset: 0,
            size: None, // entire buffer
        }),
    }
}

// ---------------------------------------------------------------------------
// DepthEngine implementation
// ---------------------------------------------------------------------------

impl DepthEngine {
    /// Create a new DepthEngine by extracting weight buffers from a DepthTransformer
    /// and pre-creating all compute pipelines and scratch buffers.
    ///
    /// # Panics
    ///
    /// Panics if the raw wgpu device/queue are not initialized (call `initWgpuDevice` first),
    /// or if the depth transformer doesn't have GPU buffers initialized.
    pub fn new(depth: &DepthTransformer, wgpu_device: &WgpuDevice, sts_config: &StsConfig) -> Self {
        let device = crate::web::raw_wgpu_device()
            .expect("raw wgpu device not initialized")
            .clone();
        let queue = crate::web::raw_wgpu_queue()
            .expect("raw wgpu queue not initialized")
            .clone();

        let config = DepthEngineConfig::from_sts_config(sts_config);

        // ---- Create compute pipelines ----
        let q4k_matvec_pipeline = create_pipeline(&device, SHADER_Q4K_MATVEC, "depth_q4k_matvec");
        let rmsnorm_pipeline = create_pipeline(&device, SHADER_RMSNORM, "depth_rmsnorm");
        let attention_pipeline = create_pipeline(&device, SHADER_ATTENTION, "depth_attention");
        let embed_q4_pipeline = create_pipeline(&device, SHADER_EMBED_Q4, "depth_embed_q4");
        let argmax_pipeline = create_pipeline(&device, SHADER_ARGMAX, "depth_argmax");
        let swiglu_pipeline = create_pipeline(&device, SHADER_SWIGLU, "depth_swiglu");
        let vec_add_pipeline = create_pipeline(&device, SHADER_VEC_ADD, "depth_vec_add");
        let cache_write_pipeline = create_pipeline(&device, SHADER_CACHE_WRITE, "depth_cache_write");

        // ---- Extract weight buffers ----
        let num_steps = config.num_steps;
        let num_layers = config.num_layers;
        let dim = config.dim;
        let temporal_dim = config.temporal_dim;
        let ffn_intermediate = config.ffn_intermediate;

        // Per-step weights (input_projs and output_linears)
        let step_weights: Vec<StepWeights> = (0..num_steps)
            .map(|step| {
                let ip = Self::extract_linear_weights(&depth.input_projs[step], wgpu_device);
                let ol = Self::extract_linear_weights(&depth.output_linears[step], wgpu_device);
                StepWeights {
                    input_proj: ip.0,
                    input_proj_offset: ip.1,
                    input_proj_size: ip.2,
                    output_linear: ol.0,
                    output_linear_offset: ol.1,
                    output_linear_size: ol.2,
                    output_vocab: depth.output_linears[step].output_dim(),
                }
            })
            .collect();

        // Per-layer, per-step transformer weights
        let layer_step_weights: Vec<Vec<LayerStepWeights>> = (0..num_layers)
            .map(|layer_idx| {
                (0..num_steps)
                    .map(|step| {
                        Self::extract_layer_step_weights(depth, layer_idx, step, wgpu_device)
                    })
                    .collect()
            })
            .collect();

        // Embeddings
        let text_emb = Self::extract_embedding(&depth.text_emb, wgpu_device);
        let audio_embs: Vec<EmbeddingData> = depth
            .audio_embs
            .iter()
            .map(|emb| Self::extract_embedding(emb, wgpu_device))
            .collect();

        // ---- Create scratch buffers ----
        let f32_bytes = 4u64;
        let dim_bytes = dim as u64 * f32_bytes;

        let projected_bufs: Vec<wgpu::Buffer> = (0..num_steps)
            .map(|i| create_storage_buffer(&device, &format!("depth_proj_{i}"), dim_bytes))
            .collect();

        let embed_out_bufs: Vec<wgpu::Buffer> = (0..num_steps)
            .map(|i| create_storage_buffer(&device, &format!("depth_emb_out_{i}"), dim_bytes))
            .collect();

        let hidden_a = create_storage_buffer(&device, "depth_hidden_a", dim_bytes);
        let hidden_b = create_storage_buffer(&device, "depth_hidden_b", dim_bytes);
        let norm_out = create_storage_buffer(&device, "depth_norm_out", dim_bytes);
        let qkv_out = create_storage_buffer(
            &device,
            "depth_qkv_out",
            config.qkv_dim as u64 * f32_bytes,
        );
        let attn_out = create_storage_buffer(&device, "depth_attn_out", dim_bytes);
        let attn_proj_out = create_storage_buffer(&device, "depth_attn_proj_out", dim_bytes);
        let ffn_gate_out = create_storage_buffer(
            &device,
            "depth_ffn_gate_out",
            (2 * ffn_intermediate) as u64 * f32_bytes,
        );
        let swiglu_out = create_storage_buffer(
            &device,
            "depth_swiglu_out",
            ffn_intermediate as u64 * f32_bytes,
        );
        let ffn_proj_out = create_storage_buffer(&device, "depth_ffn_proj_out", dim_bytes);

        // Max output vocab across all steps
        let max_vocab = step_weights.iter().map(|s| s.output_vocab).max().unwrap_or(2048);
        let logits_buf = create_storage_buffer(
            &device,
            "depth_logits",
            max_vocab as u64 * f32_bytes,
        );

        // Token results: one 4-byte buffer per step (separate to avoid 256-byte alignment issues)
        let token_result_bufs: Vec<wgpu::Buffer> = (0..num_steps)
            .map(|step| {
                create_storage_buffer(&device, &format!("depth_token_result_{step}"), 4)
            })
            .collect();
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("depth_staging"),
            size: num_steps as u64 * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // ---- KV cache buffers ----
        let kv_cache_max_len = config.kv_cache_max_len;
        let num_heads = config.num_heads;
        let head_dim = config.head_dim;
        let cache_size = (num_heads * kv_cache_max_len * head_dim) as u64 * f32_bytes;

        let k_caches: Vec<wgpu::Buffer> = (0..num_layers)
            .map(|i| create_storage_buffer(&device, &format!("depth_k_cache_{i}"), cache_size))
            .collect();
        let v_caches: Vec<wgpu::Buffer> = (0..num_layers)
            .map(|i| create_storage_buffer(&device, &format!("depth_v_cache_{i}"), cache_size))
            .collect();

        // ---- Info buffers ----

        // Q4K matvec info: [B, M, K, N, blocks_per_row]
        let info_input_proj = create_info_buffer_init(
            &device,
            "info_input_proj",
            &[1, 1, temporal_dim as u32, dim as u32, (temporal_dim / 256) as u32],
        );

        let info_attn_in_proj = create_info_buffer_init(
            &device,
            "info_attn_in_proj",
            &[1, 1, dim as u32, (3 * dim) as u32, (dim / 256) as u32],
        );

        let info_attn_out_proj = create_info_buffer_init(
            &device,
            "info_attn_out_proj",
            &[1, 1, dim as u32, dim as u32, (dim / 256) as u32],
        );

        let info_ffn_linear_in = create_info_buffer_init(
            &device,
            "info_ffn_linear_in",
            &[
                1,
                1,
                dim as u32,
                (2 * ffn_intermediate) as u32,
                (dim / 256) as u32,
            ],
        );

        let info_ffn_linear_out = create_info_buffer_init(
            &device,
            "info_ffn_linear_out",
            &[
                1,
                1,
                ffn_intermediate as u32,
                dim as u32,
                (ffn_intermediate / 256) as u32,
            ],
        );

        let info_output_linear: Vec<wgpu::Buffer> = (0..num_steps)
            .map(|step| {
                let vocab = step_weights[step].output_vocab;
                create_info_buffer_init(
                    &device,
                    &format!("info_output_linear_{step}"),
                    &[1, 1, dim as u32, vocab as u32, (dim / 256) as u32],
                )
            })
            .collect();

        // RMSNorm info: [D, eps_bits, total_rows=1]
        let info_rmsnorm = create_info_buffer_init(
            &device,
            "info_rmsnorm",
            &[dim as u32, config.rms_eps.to_bits(), 1],
        );

        // Per-step attention info: [num_heads, head_dim, seq_len, scale_bits, kv_head_stride]
        // seq_len = step + 1 (step 0 sees 1 token, step 1 sees 2, etc.)
        // kv_head_stride = kv_cache_max_len * head_dim (cache layout)
        let kv_head_stride = (kv_cache_max_len * head_dim) as u32;
        let info_attention: Vec<wgpu::Buffer> = (0..num_steps)
            .map(|step| {
                let seq_len = (step + 1) as u32;
                create_info_buffer_init(
                    &device,
                    &format!("info_attention_{step}"),
                    &[
                        num_heads as u32,
                        head_dim as u32,
                        seq_len,
                        config.attn_scale.to_bits(),
                        kv_head_stride,
                    ],
                )
            })
            .collect();

        // SwiGLU info: [half_dim]
        let info_swiglu = create_info_buffer_init(
            &device,
            "info_swiglu",
            &[ffn_intermediate as u32],
        );

        // Vec add info: [len = dim]
        let info_vec_add_dim = create_info_buffer_init(
            &device,
            "info_vec_add_dim",
            &[dim as u32],
        );

        // Per-step embedding info: [dim, bytes_per_row, vocab_size]
        let mut info_embed: Vec<wgpu::Buffer> = Vec::with_capacity(num_steps);
        // Step 0: text embedding
        info_embed.push(create_info_buffer_init(
            &device,
            "info_embed_text",
            &[
                text_emb.dim as u32,
                text_emb.bytes_per_row as u32,
                text_emb.vocab_size as u32,
            ],
        ));
        // Steps 1..num_steps: audio embeddings
        for step in 1..num_steps {
            let emb_idx = step - 1;
            if emb_idx < audio_embs.len() {
                let emb = &audio_embs[emb_idx];
                info_embed.push(create_info_buffer_init(
                    &device,
                    &format!("info_embed_audio_{step}"),
                    &[
                        emb.dim as u32,
                        emb.bytes_per_row as u32,
                        emb.vocab_size as u32,
                    ],
                ));
            } else {
                // Shouldn't happen for 8 steps with 7 audio embs, but be safe
                info_embed.push(create_info_buffer_init(
                    &device,
                    &format!("info_embed_audio_{step}"),
                    &[dim as u32, 0, 0],
                ));
            }
        }

        // Per-step argmax info: [vocab_size]
        let info_argmax: Vec<wgpu::Buffer> = (0..num_steps)
            .map(|step| {
                let vocab = step_weights[step].output_vocab;
                create_info_buffer_init(
                    &device,
                    &format!("info_argmax_{step}"),
                    &[vocab as u32],
                )
            })
            .collect();

        // Per-step cache write info: [pos_offset, B=1, S=1, H=num_heads, D=head_dim, max_len]
        let info_cache_write: Vec<wgpu::Buffer> = (0..num_steps)
            .map(|step| {
                create_info_buffer_init(
                    &device,
                    &format!("info_cache_write_{step}"),
                    &[
                        step as u32, // pos_offset = step index within the frame
                        1,           // B
                        1,           // S
                        num_heads as u32,
                        head_dim as u32,
                        kv_cache_max_len as u32,
                    ],
                )
            })
            .collect();

        DepthEngine {
            device,
            queue,
            q4k_matvec_pipeline,
            rmsnorm_pipeline,
            attention_pipeline,
            embed_q4_pipeline,
            argmax_pipeline,
            swiglu_pipeline,
            vec_add_pipeline,
            cache_write_pipeline,
            step_weights,
            layer_step_weights,
            text_emb,
            audio_embs,
            projected_bufs,
            embed_out_bufs,
            hidden_a,
            hidden_b,
            norm_out,
            qkv_out,
            attn_out,
            attn_proj_out,
            ffn_gate_out,
            swiglu_out,
            ffn_proj_out,
            logits_buf,
            token_result_bufs,
            staging_buf,
            k_caches,
            v_caches,
            info_input_proj,
            info_attn_in_proj,
            info_attn_out_proj,
            info_ffn_linear_in,
            info_ffn_linear_out,
            info_output_linear,
            info_rmsnorm,
            info_attention,
            info_swiglu,
            info_vec_add_dim,
            info_embed,
            info_argmax,
            info_cache_write,
            config,
        }
    }

    /// Extract the raw wgpu buffer from a Linear enum (Q4K only for depth transformer).
    fn extract_linear_weights(
        linear: &crate::gguf::Linear,
        wgpu_device: &WgpuDevice,
    ) -> (Arc<wgpu::Buffer>, u64, u64) {
        match linear {
            crate::gguf::Linear::Q4K(l) => {
                extract_buffer(l.weights.raw_handle(), wgpu_device)
            }
            crate::gguf::Linear::Q4(l) => {
                extract_buffer(&l.weights.handle, wgpu_device)
            }
            crate::gguf::Linear::Dense(_) => {
                panic!("DepthEngine does not support dense linear layers");
            }
        }
    }

    /// Extract per-layer, per-step weight buffers from the DepthTransformer.
    fn extract_layer_step_weights(
        depth: &DepthTransformer,
        layer_idx: usize,
        step: usize,
        wgpu_device: &WgpuDevice,
    ) -> LayerStepWeights {
        let layer = &depth.layers[layer_idx];

        // Attention weights (multi-linear: indexed by step)
        let attn_in = Self::extract_linear_weights(&layer.attention.in_projs[step], wgpu_device);
        let attn_out = Self::extract_linear_weights(&layer.attention.out_projs[step], wgpu_device);

        // FFN weights (multi-linear: indexed by step)
        let ffn_in = Self::extract_linear_weights(&layer.ffn.linear_ins[step], wgpu_device);
        let ffn_out = Self::extract_linear_weights(&layer.ffn.linear_outs[step], wgpu_device);

        // RMSNorm alpha handles (shared across steps — same norm weights)
        let norm1_res = handle_to_wgpu_resource(&layer.norm1.alpha_handle, wgpu_device);
        let norm2_res = handle_to_wgpu_resource(&layer.norm2.alpha_handle, wgpu_device);

        LayerStepWeights {
            attn_in_proj: attn_in.0,
            attn_in_proj_offset: attn_in.1,
            attn_in_proj_size: attn_in.2,
            attn_out_proj: attn_out.0,
            attn_out_proj_offset: attn_out.1,
            attn_out_proj_size: attn_out.2,
            ffn_linear_in: ffn_in.0,
            ffn_linear_in_offset: ffn_in.1,
            ffn_linear_in_size: ffn_in.2,
            ffn_linear_out: ffn_out.0,
            ffn_linear_out_offset: ffn_out.1,
            ffn_linear_out_size: ffn_out.2,
            norm1_alpha: Arc::new(norm1_res.buffer),
            norm1_alpha_offset: norm1_res.offset,
            norm1_alpha_size: norm1_res.size,
            norm2_alpha: Arc::new(norm2_res.buffer),
            norm2_alpha_offset: norm2_res.offset,
            norm2_alpha_size: norm2_res.size,
        }
    }

    /// Extract embedding data from an EmbeddingStore.
    fn extract_embedding(
        emb: &crate::gguf::EmbeddingStore,
        wgpu_device: &WgpuDevice,
    ) -> EmbeddingData {
        let (handle, emb_dim, vocab_size, bytes_per_row, _dev) = emb.gpu_lookup_params();
        let res = handle_to_wgpu_resource(handle, wgpu_device);
        EmbeddingData {
            q4_data: Arc::new(res.buffer),
            q4_data_offset: res.offset,
            q4_data_size: res.size,
            vocab_size,
            dim: emb_dim,
            bytes_per_row,
        }
    }

    /// Run all 8 depth steps in a single command buffer.
    ///
    /// # Arguments
    ///
    /// * `temporal_hidden` - The temporal transformer's hidden state as a WgpuResource
    ///   pointing to [1, 1, 4096] f32 data on GPU.
    /// * `text_token` - The text token from temporal argmax as a WgpuResource
    ///   pointing to a single u32 (stored as f32 bits) on GPU.
    ///
    /// # Returns
    ///
    /// 8 u32 token IDs (one per depth step / codebook).
    pub async fn generate(
        &self,
        temporal_hidden: &WgpuResource,
        text_token: &WgpuResource,
    ) -> Vec<u32> {
        let config = &self.config;
        let num_steps = config.num_steps;
        let dim = config.dim;
        let num_heads = config.num_heads;
        let head_dim = config.head_dim;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("depth_engine"),
            });

        // ===========================================================
        // Phase 1: Encode all 8 input projections
        //
        // projected[step] = temporal_hidden @ input_proj[step]^T
        // Q4K matvec bindings:
        //   @binding(0) weights: Q4K packed data
        //   @binding(1) input: f32 vector
        //   @binding(2) output: f32 vector
        //   @binding(3) info: [B, M, K, N, blocks_per_row]
        // ===========================================================
        for step in 0..num_steps {
            let sw = &self.step_weights[step];
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.q4k_matvec_pipeline.get_bind_group_layout(0),
                entries: &[
                    buf_entry(0, &sw.input_proj, sw.input_proj_offset, sw.input_proj_size),
                    buf_entry(1, &temporal_hidden.buffer, temporal_hidden.offset, temporal_hidden.size),
                    buf_entry_whole(2, &self.projected_bufs[step]),
                    buf_entry_whole(3, &self.info_input_proj),
                ],
            });
            let wg_x = (dim as u32).div_ceil(256);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("input_proj"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.q4k_matvec_pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(wg_x, 1, 1);
        }

        // ===========================================================
        // Phase 2: 8 autoregressive depth steps
        // ===========================================================
        for step in 0..num_steps {
            // ---- Embedding lookup ----
            // embed_q4 bindings:
            //   @binding(0) token_buf: u32 token ID
            //   @binding(1) q4_data: Q4_0 embedding table
            //   @binding(2) output: f32 embedding vector
            //   @binding(3) info: [dim, bytes_per_row, vocab_size]
            {
                let (token_buf, token_offset, token_size) = if step == 0 {
                    // Step 0: use text token from temporal
                    (&text_token.buffer, text_token.offset, text_token.size)
                } else {
                    // Steps 1+: use previous step's argmax result
                    (&self.token_result_bufs[step - 1], 0u64, 4u64)
                };

                let (emb_buf, emb_offset, emb_size) = if step == 0 {
                    (&self.text_emb.q4_data, self.text_emb.q4_data_offset, self.text_emb.q4_data_size)
                } else {
                    let emb_idx = step - 1;
                    let emb = &self.audio_embs[emb_idx.min(self.audio_embs.len() - 1)];
                    (&emb.q4_data, emb.q4_data_offset, emb.q4_data_size)
                };

                let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.embed_q4_pipeline.get_bind_group_layout(0),
                    entries: &[
                        buf_entry(0, token_buf, token_offset, token_size),
                        buf_entry(1, emb_buf, emb_offset, emb_size),
                        buf_entry_whole(2, &self.embed_out_bufs[step]),
                        buf_entry_whole(3, &self.info_embed[step]),
                    ],
                });
                let wg_x = (dim as u32).div_ceil(256);
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("embed_lookup"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.embed_q4_pipeline);
                pass.set_bind_group(0, Some(&bg), &[]);
                pass.dispatch_workgroups(wg_x, 1, 1);
            }

            // ---- Vec add: hidden = projected + embedding ----
            // vec_add bindings:
            //   @binding(0) a: f32 array
            //   @binding(1) b: f32 array
            //   @binding(2) output: f32 array
            //   @binding(3) info: [len]
            {
                let dim_bytes = (dim * 4) as u64;
                let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.vec_add_pipeline.get_bind_group_layout(0),
                    entries: &[
                        buf_entry(0, &self.projected_bufs[step], 0, dim_bytes),
                        buf_entry(1, &self.embed_out_bufs[step], 0, dim_bytes),
                        buf_entry(2, &self.hidden_a, 0, dim_bytes),
                        buf_entry_whole(3, &self.info_vec_add_dim),
                    ],
                });
                let wg_x = (dim as u32).div_ceil(256);
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("vec_add_proj_emb"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.vec_add_pipeline);
                pass.set_bind_group(0, Some(&bg), &[]);
                pass.dispatch_workgroups(wg_x, 1, 1);
            }

            // ---- 6 transformer layers ----
            // We ping-pong between hidden_a and hidden_b.
            // After vec_add, hidden_a has the input.
            // After each layer, the result is in the "other" buffer.
            let mut cur_hidden = &self.hidden_a;
            let mut alt_hidden = &self.hidden_b;

            for layer_idx in 0..config.num_layers {
                let lw = &self.layer_step_weights[layer_idx][step];

                // ---- RMSNorm 1 ----
                // rmsnorm bindings:
                //   @binding(0) input: f32 [D]
                //   @binding(1) alpha: f32 [D]
                //   @binding(2) output: f32 [D]
                //   @binding(3) info: [D, eps_bits, total_rows]
                {
                    let dim_bytes = (dim * 4) as u64;
                    let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.rmsnorm_pipeline.get_bind_group_layout(0),
                        entries: &[
                            buf_entry(0, cur_hidden, 0, dim_bytes),
                            buf_entry(1, &lw.norm1_alpha, lw.norm1_alpha_offset, lw.norm1_alpha_size),
                            buf_entry(2, &self.norm_out, 0, dim_bytes),
                            buf_entry_whole(3, &self.info_rmsnorm),
                        ],
                    });
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("rmsnorm1"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.rmsnorm_pipeline);
                    pass.set_bind_group(0, Some(&bg), &[]);
                    // One workgroup per row, we have 1 row
                    pass.dispatch_workgroups(1, 1, 1);
                }

                // ---- Attention QKV projection ----
                // Q4K matvec: norm_out @ attn_in_proj^T -> qkv_out [3*dim]
                {
                    let dim_bytes = (dim * 4) as u64;
                    let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.q4k_matvec_pipeline.get_bind_group_layout(0),
                        entries: &[
                            buf_entry(0, &lw.attn_in_proj, lw.attn_in_proj_offset, lw.attn_in_proj_size),
                            buf_entry(1, &self.norm_out, 0, dim_bytes),
                            buf_entry_whole(2, &self.qkv_out),
                            buf_entry_whole(3, &self.info_attn_in_proj),
                        ],
                    });
                    let wg_x = ((3 * dim) as u32).div_ceil(256);
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("attn_qkv"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.q4k_matvec_pipeline);
                    pass.set_bind_group(0, Some(&bg), &[]);
                    pass.dispatch_workgroups(wg_x, 1, 1);
                }

                // ---- Cache write K ----
                // cache_write bindings:
                //   @binding(0) input: f32 [B*S*H*D] (K portion of QKV)
                //   @binding(1) cache: f32 [B*H*max_len*D]
                //   @binding(2) info: [pos_offset, B, S, H, D, max_len]
                {
                    // K is the second dim-sized chunk of qkv_out (after Q)
                    let k_offset = (dim * 4) as u64; // Q is first dim floats
                    let k_size = (dim * 4) as u64;   // K is next dim floats
                    let cache_size = (num_heads * config.kv_cache_max_len * head_dim * 4) as u64;

                    let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.cache_write_pipeline.get_bind_group_layout(0),
                        entries: &[
                            buf_entry(0, &self.qkv_out, k_offset, k_size),
                            buf_entry(1, &self.k_caches[layer_idx], 0, cache_size),
                            buf_entry_whole(2, &self.info_cache_write[step]),
                        ],
                    });
                    let total_elements = (num_heads * head_dim) as u32;
                    let wg_x = total_elements.div_ceil(256);
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("cache_write_k"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.cache_write_pipeline);
                    pass.set_bind_group(0, Some(&bg), &[]);
                    pass.dispatch_workgroups(wg_x, 1, 1);
                }

                // ---- Cache write V ----
                {
                    let v_offset = (2 * dim * 4) as u64; // V is third dim-sized chunk
                    let v_size = (dim * 4) as u64;
                    let cache_size = (num_heads * config.kv_cache_max_len * head_dim * 4) as u64;

                    let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.cache_write_pipeline.get_bind_group_layout(0),
                        entries: &[
                            buf_entry(0, &self.qkv_out, v_offset, v_size),
                            buf_entry(1, &self.v_caches[layer_idx], 0, cache_size),
                            buf_entry_whole(2, &self.info_cache_write[step]),
                        ],
                    });
                    let total_elements = (num_heads * head_dim) as u32;
                    let wg_x = total_elements.div_ceil(256);
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("cache_write_v"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.cache_write_pipeline);
                    pass.set_bind_group(0, Some(&bg), &[]);
                    pass.dispatch_workgroups(wg_x, 1, 1);
                }

                // ---- Flash-decode attention ----
                // attention bindings:
                //   @binding(0) q: f32 [num_heads * head_dim]
                //   @binding(1) k: f32 cache (stride-addressed)
                //   @binding(2) v: f32 cache (stride-addressed)
                //   @binding(3) output: f32 [num_heads * head_dim]
                //   @binding(4) info: [num_heads, head_dim, seq_len, scale_bits, kv_head_stride]
                {
                    let q_size = (dim * 4) as u64;
                    let cache_size = (num_heads * config.kv_cache_max_len * head_dim * 4) as u64;

                    let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.attention_pipeline.get_bind_group_layout(0),
                        entries: &[
                            // Q is the first dim-sized chunk of qkv_out
                            buf_entry(0, &self.qkv_out, 0, q_size),
                            buf_entry(1, &self.k_caches[layer_idx], 0, cache_size),
                            buf_entry(2, &self.v_caches[layer_idx], 0, cache_size),
                            buf_entry_whole(3, &self.attn_out),
                            buf_entry_whole(4, &self.info_attention[step]),
                        ],
                    });
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("attention"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.attention_pipeline);
                    pass.set_bind_group(0, Some(&bg), &[]);
                    // One workgroup per head
                    pass.dispatch_workgroups(num_heads as u32, 1, 1);
                }

                // ---- Attention output projection ----
                // Q4K matvec: attn_out @ out_proj^T -> attn_proj_out [dim]
                {
                    let dim_bytes = (dim * 4) as u64;
                    let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.q4k_matvec_pipeline.get_bind_group_layout(0),
                        entries: &[
                            buf_entry(0, &lw.attn_out_proj, lw.attn_out_proj_offset, lw.attn_out_proj_size),
                            buf_entry(1, &self.attn_out, 0, dim_bytes),
                            buf_entry_whole(2, &self.attn_proj_out),
                            buf_entry_whole(3, &self.info_attn_out_proj),
                        ],
                    });
                    let wg_x = (dim as u32).div_ceil(256);
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("attn_out_proj"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.q4k_matvec_pipeline);
                    pass.set_bind_group(0, Some(&bg), &[]);
                    pass.dispatch_workgroups(wg_x, 1, 1);
                }

                // ---- Residual 1: hidden = attn_proj_out + hidden ----
                // vec_add: a=attn_proj_out, b=cur_hidden, output=alt_hidden
                {
                    let dim_bytes = (dim * 4) as u64;
                    let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.vec_add_pipeline.get_bind_group_layout(0),
                        entries: &[
                            buf_entry(0, &self.attn_proj_out, 0, dim_bytes),
                            buf_entry(1, cur_hidden, 0, dim_bytes),
                            buf_entry(2, alt_hidden, 0, dim_bytes),
                            buf_entry_whole(3, &self.info_vec_add_dim),
                        ],
                    });
                    let wg_x = (dim as u32).div_ceil(256);
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("residual1"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.vec_add_pipeline);
                    pass.set_bind_group(0, Some(&bg), &[]);
                    pass.dispatch_workgroups(wg_x, 1, 1);
                }

                // Swap: now alt_hidden has the post-attention hidden state
                std::mem::swap(&mut cur_hidden, &mut alt_hidden);

                // ---- RMSNorm 2 ----
                {
                    let dim_bytes = (dim * 4) as u64;
                    let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.rmsnorm_pipeline.get_bind_group_layout(0),
                        entries: &[
                            buf_entry(0, cur_hidden, 0, dim_bytes),
                            buf_entry(1, &lw.norm2_alpha, lw.norm2_alpha_offset, lw.norm2_alpha_size),
                            buf_entry(2, &self.norm_out, 0, dim_bytes),
                            buf_entry_whole(3, &self.info_rmsnorm),
                        ],
                    });
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("rmsnorm2"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.rmsnorm_pipeline);
                    pass.set_bind_group(0, Some(&bg), &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }

                // ---- FFN linear_in (gate + value) ----
                // Q4K matvec: norm_out @ ffn_linear_in^T -> ffn_gate_out [2*ffn_intermediate]
                {
                    let dim_bytes = (dim * 4) as u64;
                    let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.q4k_matvec_pipeline.get_bind_group_layout(0),
                        entries: &[
                            buf_entry(0, &lw.ffn_linear_in, lw.ffn_linear_in_offset, lw.ffn_linear_in_size),
                            buf_entry(1, &self.norm_out, 0, dim_bytes),
                            buf_entry_whole(2, &self.ffn_gate_out),
                            buf_entry_whole(3, &self.info_ffn_linear_in),
                        ],
                    });
                    let wg_x = ((2 * config.ffn_intermediate) as u32).div_ceil(256);
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("ffn_linear_in"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.q4k_matvec_pipeline);
                    pass.set_bind_group(0, Some(&bg), &[]);
                    pass.dispatch_workgroups(wg_x, 1, 1);
                }

                // ---- SwiGLU activation ----
                // swiglu bindings:
                //   @binding(0) input: f32 [2 * half_dim]
                //   @binding(1) output: f32 [half_dim]
                //   @binding(2) info: [half_dim]
                {
                    let gate_size = (2 * config.ffn_intermediate * 4) as u64;
                    let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.swiglu_pipeline.get_bind_group_layout(0),
                        entries: &[
                            buf_entry(0, &self.ffn_gate_out, 0, gate_size),
                            buf_entry_whole(1, &self.swiglu_out),
                            buf_entry_whole(2, &self.info_swiglu),
                        ],
                    });
                    let wg_x = (config.ffn_intermediate as u32).div_ceil(256);
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("swiglu"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.swiglu_pipeline);
                    pass.set_bind_group(0, Some(&bg), &[]);
                    pass.dispatch_workgroups(wg_x, 1, 1);
                }

                // ---- FFN linear_out ----
                // Q4K matvec: swiglu_out @ ffn_linear_out^T -> ffn_proj_out [dim]
                {
                    let ffn_size = (config.ffn_intermediate * 4) as u64;
                    let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.q4k_matvec_pipeline.get_bind_group_layout(0),
                        entries: &[
                            buf_entry(0, &lw.ffn_linear_out, lw.ffn_linear_out_offset, lw.ffn_linear_out_size),
                            buf_entry(1, &self.swiglu_out, 0, ffn_size),
                            buf_entry_whole(2, &self.ffn_proj_out),
                            buf_entry_whole(3, &self.info_ffn_linear_out),
                        ],
                    });
                    let wg_x = (dim as u32).div_ceil(256);
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("ffn_linear_out"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.q4k_matvec_pipeline);
                    pass.set_bind_group(0, Some(&bg), &[]);
                    pass.dispatch_workgroups(wg_x, 1, 1);
                }

                // ---- Residual 2: hidden = ffn_proj_out + hidden ----
                {
                    let dim_bytes = (dim * 4) as u64;
                    let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.vec_add_pipeline.get_bind_group_layout(0),
                        entries: &[
                            buf_entry(0, &self.ffn_proj_out, 0, dim_bytes),
                            buf_entry(1, cur_hidden, 0, dim_bytes),
                            buf_entry(2, alt_hidden, 0, dim_bytes),
                            buf_entry_whole(3, &self.info_vec_add_dim),
                        ],
                    });
                    let wg_x = (dim as u32).div_ceil(256);
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("residual2"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.vec_add_pipeline);
                    pass.set_bind_group(0, Some(&bg), &[]);
                    pass.dispatch_workgroups(wg_x, 1, 1);
                }

                // Swap again
                std::mem::swap(&mut cur_hidden, &mut alt_hidden);
            }

            // ---- Output linear: hidden -> logits ----
            // Q4K matvec: cur_hidden @ output_linear^T -> logits_buf [vocab]
            {
                let sw = &self.step_weights[step];
                let dim_bytes = (dim * 4) as u64;
                let vocab_bytes = (sw.output_vocab * 4) as u64;
                let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.q4k_matvec_pipeline.get_bind_group_layout(0),
                    entries: &[
                        buf_entry(0, &sw.output_linear, sw.output_linear_offset, sw.output_linear_size),
                        buf_entry(1, cur_hidden, 0, dim_bytes),
                        buf_entry(2, &self.logits_buf, 0, vocab_bytes),
                        buf_entry_whole(3, &self.info_output_linear[step]),
                    ],
                });
                let wg_x = (sw.output_vocab as u32).div_ceil(256);
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("output_linear"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.q4k_matvec_pipeline);
                pass.set_bind_group(0, Some(&bg), &[]);
                pass.dispatch_workgroups(wg_x, 1, 1);
            }

            // ---- Argmax ----
            // argmax bindings:
            //   @binding(0) logits: f32 [V]
            //   @binding(1) result: u32 [1]
            //   @binding(2) info: [V]
            {
                let sw = &self.step_weights[step];
                let vocab_bytes = (sw.output_vocab * 4) as u64;

                let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.argmax_pipeline.get_bind_group_layout(0),
                    entries: &[
                        buf_entry(0, &self.logits_buf, 0, vocab_bytes),
                        buf_entry_whole(1, &self.token_result_bufs[step]),
                        buf_entry_whole(2, &self.info_argmax[step]),
                    ],
                });
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("argmax"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.argmax_pipeline);
                pass.set_bind_group(0, Some(&bg), &[]);
                // Single workgroup (256 threads do the reduction)
                pass.dispatch_workgroups(1, 1, 1);
            }
        }

        // ===========================================================
        // Phase 3: Copy per-step token results to staging buffer
        // ===========================================================
        for step in 0..num_steps {
            encoder.copy_buffer_to_buffer(
                &self.token_result_bufs[step],
                0,
                &self.staging_buf,
                (step * 4) as u64,
                4,
            );
        }

        // ===========================================================
        // Phase 4: Submit and readback
        // ===========================================================
        self.queue.submit(std::iter::once(encoder.finish()));

        // Map the staging buffer and read results
        let slice = self.staging_buf.slice(..);
        let (sender, receiver) = async_channel::bounded(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.try_send(result);
        });

        // Wait for the GPU to complete
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.device.poll(wgpu::PollType::Wait).unwrap();
        }

        // async_channel::recv() is async — on WASM it yields to the event loop,
        // allowing wgpu's auto-polling to fire the map_async callback.
        // On native, the poll(Wait) above completes synchronously before we get here.
        receiver
            .recv()
            .await
            .expect("map_async callback never fired")
            .expect("GPU buffer mapping failed");

        let data = slice.get_mapped_range();
        let tokens: Vec<u32> = data
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        drop(data);
        self.staging_buf.unmap();

        tokens
    }

    /// Reset KV caches for a new frame. Call before each `generate()`.
    ///
    /// This clears the KV cache buffers by writing zeros. The depth transformer
    /// builds up its cache from scratch each frame (depformer_context = 8 steps).
    pub fn reset_caches(&self) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("depth_cache_reset"),
            });

        for cache in self.k_caches.iter().chain(self.v_caches.iter()) {
            encoder.clear_buffer(cache, 0, None);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_from_defaults() {
        let sts_config = StsConfig::default();
        let config = DepthEngineConfig::from_sts_config(&sts_config);

        assert_eq!(config.num_steps, 8);
        assert_eq!(config.dim, 1024);
        assert_eq!(config.num_layers, 6);
        assert_eq!(config.num_heads, 16);
        assert_eq!(config.num_kv_heads, 16);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.ffn_intermediate, 2816);
        assert_eq!(config.qkv_dim, 3072);
        assert_eq!(config.kv_cache_max_len, 9);
        assert_eq!(config.temporal_dim, 4096);
    }
}
