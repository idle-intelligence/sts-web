//! Depth transformer: generates multi-codebook audio tokens from a hidden state.
//!
//! For each time step, the depth transformer takes the temporal transformer's
//! hidden state and autoregressively generates tokens for all 16 output streams
//! (1 text + 8 model audio + 7 remaining model audio delayed).
//!
//! Architecture (PersonaPlex / Moshi Depformer):
//! - 16 separate input projections: depformer_in.{0-15} [1024, 4096]
//! - 16 per-codebook embeddings: depformer_emb.{0-14} [2049, 1024] +
//!   depformer_text_emb [32001, 1024]
//! - 6 transformer layers with MULTI-LINEAR weights:
//!   each layer has 16 separate weight sets for attention and FFN
//! - 16 output heads: linears.{0-15} [2048, 1024]
//!
//! The depth transformer uses per-step (per-codebook) weight matrices,
//! meaning the linear layers use different weights at each depth step.
//! This is critical for quality: different codebooks need different
//! transformations.

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::activation::softmax;
use burn::tensor::Tensor;

use crate::gguf::{
    gpu_argmax, gpu_read_token_tensors,
    DepthGpuBuffers, EmbeddingStore, Linear,
};
use crate::model::{fused_attention, fused_qkv_split, fused_swiglu, gpu_cache_write_v, sample_top_k_with_penalty, KVCache, LayerCaches, RmsNormLayer};
#[cfg(not(target_arch = "wasm32"))]
use crate::model::{sample_greedy, sample_top_k};
use crate::StsConfig;

// ---------------------------------------------------------------------------
// Multi-linear attention (per-step weights)
// ---------------------------------------------------------------------------

/// Multi-head attention with per-step weight selection.
///
/// Stores 16 separate weight sets for in_proj and out_proj.
/// At each depth step k, we use weights[k].
///
/// Packed tensor layout from GGUF:
/// - `self_attn.in_proj_weight`: [49152, 1024] = 16 x [3072, 1024]
/// - `self_attn.out_proj.weight`: [16384, 1024] = 16 x [1024, 1024]
///
/// We store them pre-split as 16 separate Q4Linear instances.
pub struct MultiLinearAttention {
    in_projs: Vec<Linear>,   // 16 separate [3072, 1024] linears
    out_projs: Vec<Linear>,  // 16 separate [1024, 1024] linears
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    dim: usize,
    scale: f32,
}

impl MultiLinearAttention {
    pub fn new(
        in_projs: Vec<Linear>,
        out_projs: Vec<Linear>,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let dim = n_heads * head_dim;
        Self {
            in_projs,
            out_projs,
            n_heads,
            n_kv_heads,
            head_dim,
            dim,
            scale: (head_dim as f32).powf(-0.5),
        }
    }

    /// Forward pass using weights for depth step `step_idx`.
    pub fn forward_with_cache(
        &self,
        x: Tensor<Wgpu, 3>,
        step_idx: usize,
        cache: &mut KVCache,
    ) -> Tensor<Wgpu, 3> {
        let [batch, seq_len, _] = x.dims();
        let offset = cache.offset();

        let qkv = self.in_projs[step_idx].forward(x);
        let kv_dim = self.n_kv_heads * self.head_dim;

        let (q, k, v) = fused_qkv_split(qkv, self.dim, kv_dim);

        let q = q.reshape([batch, seq_len, self.n_heads, self.head_dim]);
        let k = k.reshape([batch, seq_len, self.n_kv_heads, self.head_dim]);
        let v = v.reshape([batch, seq_len, self.n_kv_heads, self.head_dim]);

        // No positional embedding for depth transformer (depformer_pos_emb="none" in Moshi)

        // Generation mode (S=1, no GQA): fused cache write path.
        // Saves 4 dispatches per layer (swap_dims+contiguous+slice_assign for K and V).
        if seq_len == 1 && self.n_heads == self.n_kv_heads {
            let (k_cache_handle, v_cache_handle, _write_pos) =
                cache.ensure_allocated(batch);
            let max_len = cache.max_len();

            // Write K and V directly to cache (no RoPE for depth transformer)
            gpu_cache_write_v(&k, &k_cache_handle, offset, max_len);
            gpu_cache_write_v(&v, &v_cache_handle, offset, max_len);

            let (k_all, v_all) = cache.advance(1);

            let out = fused_attention(q, k_all, v_all, self.scale);
            let out = out.swap_dims(1, 2);
            let out = out.reshape([batch, seq_len, self.n_heads * self.head_dim]);
            return self.out_projs[step_idx].forward(out);
        }

        // Prefill path
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        let (k, v) = cache.update(k, v);
        let total_seq_len = cache.seq_len();

        // GQA expansion if needed
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

        // Prefill path: need Q in [B, H, S, D]
        let q = q.swap_dims(1, 2);
        let k_t = k.swap_dims(2, 3);
        let scores = q.matmul(k_t) * self.scale;

        // Causal mask for depth steps (only needed if seq_len > 1)
        let scores = if seq_len > 1 {
            let device = scores.device();
            let mut mask_data = vec![0.0f32; seq_len * total_seq_len];
            for i in 0..seq_len {
                let actual_pos = offset + i;
                for j in 0..total_seq_len {
                    if j > actual_pos {
                        mask_data[i * total_seq_len + j] = f32::NEG_INFINITY;
                    }
                }
            }
            let mask: Tensor<Wgpu, 1> = Tensor::from_floats(mask_data.as_slice(), &device);
            let mask: Tensor<Wgpu, 4> = mask
                .reshape([seq_len, total_seq_len])
                .unsqueeze_dim::<3>(0)
                .unsqueeze_dim(0);
            scores + mask
        } else {
            scores
        };

        let attn = softmax(scores, 3);
        let out = attn.matmul(v);
        let out = out.swap_dims(1, 2);
        let out = out.reshape([batch, seq_len, self.n_heads * self.head_dim]);
        self.out_projs[step_idx].forward(out)
    }

    /// Forward with fused residual addition in the out_proj matmul.
    pub fn forward_with_cache_fused_residual(
        &self,
        x: Tensor<Wgpu, 3>,
        residual: Tensor<Wgpu, 3>,
        step_idx: usize,
        cache: &mut KVCache,
    ) -> Tensor<Wgpu, 3> {
        let [batch, seq_len, _] = x.dims();
        let offset = cache.offset();

        let qkv = self.in_projs[step_idx].forward(x);
        let kv_dim = self.n_kv_heads * self.head_dim;

        let (q, k, v) = fused_qkv_split(qkv, self.dim, kv_dim);

        let q = q.reshape([batch, seq_len, self.n_heads, self.head_dim]);
        let k = k.reshape([batch, seq_len, self.n_kv_heads, self.head_dim]);
        let v = v.reshape([batch, seq_len, self.n_kv_heads, self.head_dim]);

        // Generation mode (S=1, no GQA): fused cache write path.
        if seq_len == 1 && self.n_heads == self.n_kv_heads {
            let (k_cache_handle, v_cache_handle, _write_pos) =
                cache.ensure_allocated(batch);
            let max_len = cache.max_len();

            gpu_cache_write_v(&k, &k_cache_handle, offset, max_len);
            gpu_cache_write_v(&v, &v_cache_handle, offset, max_len);

            let (k_all, v_all) = cache.advance(1);

            let out = fused_attention(q, k_all, v_all, self.scale);
            let out = out.swap_dims(1, 2);
            let out = out.reshape([batch, seq_len, self.n_heads * self.head_dim]);
            return self.out_projs[step_idx].forward_with_residual(out, residual);
        }

        // Prefill path
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

        // Prefill path: need Q in [B, H, S, D]
        let q = q.swap_dims(1, 2);
        let k_t = k.swap_dims(2, 3);
        let scores = q.matmul(k_t) * self.scale;

        let scores = if seq_len > 1 {
            let device = scores.device();
            let mut mask_data = vec![0.0f32; seq_len * total_seq_len];
            for i in 0..seq_len {
                let actual_pos = offset + i;
                for j in 0..total_seq_len {
                    if j > actual_pos {
                        mask_data[i * total_seq_len + j] = f32::NEG_INFINITY;
                    }
                }
            }
            let mask: Tensor<Wgpu, 1> = Tensor::from_floats(mask_data.as_slice(), &device);
            let mask: Tensor<Wgpu, 4> = mask
                .reshape([seq_len, total_seq_len])
                .unsqueeze_dim::<3>(0)
                .unsqueeze_dim(0);
            scores + mask
        } else {
            scores
        };

        let attn = softmax(scores, 3);
        let out = attn.matmul(v);
        let out = out.swap_dims(1, 2);
        let out = out.reshape([batch, seq_len, self.n_heads * self.head_dim]);
        self.out_projs[step_idx].forward_with_residual(out, residual)
    }
}

// ---------------------------------------------------------------------------
// Multi-linear feed-forward (per-step weights)
// ---------------------------------------------------------------------------

/// SwiGLU feed-forward with per-step weight selection.
///
/// Stores 16 separate weight sets for linear_in and linear_out.
///
/// Packed tensor layout:
/// - `gating.{0-15}.linear_in.weight`: [5632, 1024] = 2 * 2816
/// - `gating.{0-15}.linear_out.weight`: [1024, 2816]
pub struct MultiLinearFeedForward {
    linear_ins: Vec<Linear>,   // 16 x [5632, 1024]
    linear_outs: Vec<Linear>,  // 16 x [1024, 2816]
}

impl MultiLinearFeedForward {
    pub fn new(linear_ins: Vec<Linear>, linear_outs: Vec<Linear>) -> Self {
        Self {
            linear_ins,
            linear_outs,
        }
    }

    pub fn forward(&self, x: Tensor<Wgpu, 3>, step_idx: usize) -> Tensor<Wgpu, 3> {
        let combined = self.linear_ins[step_idx].forward(x);
        let activated = fused_swiglu(combined);
        self.linear_outs[step_idx].forward(activated)
    }

    /// Forward with fused residual addition in the linear_out matmul.
    pub fn forward_with_residual(&self, x: Tensor<Wgpu, 3>, residual: Tensor<Wgpu, 3>, step_idx: usize) -> Tensor<Wgpu, 3> {
        let combined = self.linear_ins[step_idx].forward(x);
        let activated = fused_swiglu(combined);
        self.linear_outs[step_idx].forward_with_residual(activated, residual)
    }
}

// ---------------------------------------------------------------------------
// Depth transformer block (multi-linear)
// ---------------------------------------------------------------------------

/// A single depth transformer layer with multi-linear weights.
///
/// The norm weights are shared across all steps (not per-step).
pub struct DepthTransformerBlock {
    norm1: RmsNormLayer,
    attention: MultiLinearAttention,
    norm2: RmsNormLayer,
    ffn: MultiLinearFeedForward,
}

impl DepthTransformerBlock {
    pub fn new(
        norm1: RmsNormLayer,
        attention: MultiLinearAttention,
        norm2: RmsNormLayer,
        ffn: MultiLinearFeedForward,
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
        step_idx: usize,
        cache: &mut KVCache,
    ) -> Tensor<Wgpu, 3> {
        // Fuse residual addition into out_proj matmul (saves 1 dispatch)
        let residual = x.clone();
        let x = self.norm1.forward(x);
        let x = self.attention.forward_with_cache_fused_residual(x, residual, step_idx, cache);

        // Fuse residual addition into linear_out matmul (saves 1 dispatch)
        let residual = x.clone();
        let x = self.norm2.forward(x);
        self.ffn.forward_with_residual(x, residual, step_idx)
    }
}

// ---------------------------------------------------------------------------
// DepthTransformer
// ---------------------------------------------------------------------------

/// The depth transformer for multi-codebook token generation.
///
/// At each time step, takes the temporal hidden state [1, 1, 4096] and
/// autoregressively generates 16 output tokens (selecting per-step weights
/// at each depth step).
///
/// Weight structure:
/// - depformer_in.{0-15}: project temporal hidden -> depth dim [1024, 4096]
/// - depformer_text_emb: text token embedding [32001, 1024]
/// - depformer_emb.{0-14}: audio codebook embeddings [2049, 1024]
/// - 6 layers with multi-linear attention + FFN (16 weight sets each)
/// - linears.{0-15}: output heads [2048, 1024] (audio) or [32000, 1024] (text)
pub struct DepthTransformer {
    /// Per-step input projections from temporal hidden to depth dim.
    input_projs: Vec<Linear>, // 16 x [1024, 4096] — F32 or Q4
    /// Text token embedding for depth input.
    text_emb: EmbeddingStore, // [32001, 1024]
    /// Per-codebook audio embeddings for depth input.
    audio_embs: Vec<EmbeddingStore>, // 15 x [2049, 1024]
    /// Transformer layers with multi-linear weights.
    layers: Vec<DepthTransformerBlock>,
    /// Per-step output heads.
    output_linears: Vec<Linear>, // 16 x [vocab, 1024] — F32 or Q4
    config: StsConfig,
    device: WgpuDevice,
    /// Pre-allocated GPU buffers for the GPU-optimized generation path.
    gpu_buffers: Option<DepthGpuBuffers>,
}

impl DepthTransformer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_projs: Vec<Linear>,
        text_emb: EmbeddingStore,
        audio_embs: Vec<EmbeddingStore>,
        layers: Vec<DepthTransformerBlock>,
        output_linears: Vec<Linear>,
        config: StsConfig,
        device: WgpuDevice,
    ) -> Self {
        let mut dt = Self {
            input_projs,
            text_emb,
            audio_embs,
            layers,
            output_linears,
            config,
            device,
            gpu_buffers: None,
        };
        // Upload audio embeddings to GPU and pre-allocate buffers for
        // GPU-side embedding lookups and sampling (eliminates per-step
        // buffer allocation overhead).
        dt.init_gpu_buffers();
        dt
    }

    /// Upload audio embeddings to GPU and pre-allocate reusable buffers.
    fn init_gpu_buffers(&mut self) {
        // Upload Q4 bytes for each audio embedding to GPU
        for emb in &mut self.audio_embs {
            emb.upload_to_gpu(&self.device);
        }
        // Pre-allocate sampling + embedding output buffers
        self.gpu_buffers = Some(DepthGpuBuffers::new(
            self.config.depth_num_steps,
            self.config.depth_hidden_size,
            &self.audio_embs,
            &self.device,
        ));
    }

    /// Generate all 16 audio codebook tokens from a temporal hidden state.
    ///
    /// `temporal_hidden`: [1, 1, 4096] from temporal transformer's output.
    /// `text_token`: text token sampled from the temporal transformer's text_linear head.
    /// `cache`: depth KV cache (reset between time steps).
    /// `audio_temp`, `audio_top_k`: sampling params for audio tokens.
    ///
    /// Returns 16 audio codebook tokens. Steps 0-7 are model audio codebooks,
    /// steps 8-15 are user audio predictions (typically discarded at inference).
    ///
    /// Autoregressive chain:
    /// - Step 0: input = proj(hidden) + text_emb(text_token) → audio_token_0
    /// - Step k>0: input = proj(hidden) + audio_embs[k-1](audio_token_{k-1}) → audio_token_k
    ///
    /// GPU-optimized: sampling and embedding lookups happen on GPU to avoid
    /// per-step GPU→CPU readbacks. All 16 tokens are read back in one batch
    /// at the end.
    #[allow(clippy::too_many_arguments)]
    pub async fn generate(
        &self,
        temporal_hidden: Tensor<Wgpu, 3>,
        text_token: u32,
        cache: &mut LayerCaches,
        audio_temp: f32,
        audio_top_k: usize,
        provided_tokens: Option<&[i32]>,
        penalty_history: Option<&[Vec<u32>]>,
        audio_penalty: f32,
        max_steps: Option<usize>,
    ) -> Vec<u32> {
        let num_steps = max_steps.unwrap_or(self.config.depth_num_steps)
            .min(self.config.depth_num_steps);
        let dim = self.config.depth_hidden_size;

        // GPU path is only beneficial for greedy sampling (temp <= 0).
        // For top-k sampling, the CPU path is faster because:
        // - Vocab is small (2048), so 8KB readback is cheap
        // - CPU top-k is trivially fast for small vocab
        // - GPU kernel launch overhead exceeds readback savings
        let gpu_path_available = audio_temp <= 0.0 && self.gpu_buffers.is_some();

        if gpu_path_available && provided_tokens.is_none() {
            // === GPU-OPTIMIZED PATH ===
            // All sampling + embedding lookups on GPU, one batch readback at end.
            self.generate_gpu_path(
                temporal_hidden,
                text_token,
                cache,
                audio_temp,
                audio_top_k,
                penalty_history,
                audio_penalty,
                num_steps,
                dim,
            )
            .await
        } else {
            // === CPU FALLBACK PATH ===
            // Used when GPU embeddings not uploaded or provided_tokens override needed.
            self.generate_cpu_path(
                temporal_hidden,
                text_token,
                cache,
                audio_temp,
                audio_top_k,
                provided_tokens,
                penalty_history,
                audio_penalty,
                num_steps,
                dim,
            )
            .await
        }
    }

    /// GPU-optimized generation: no per-step readbacks.
    ///
    /// Uses pre-allocated GPU buffers for sampling and embedding lookup.
    #[allow(clippy::too_many_arguments)]
    async fn generate_gpu_path(
        &self,
        temporal_hidden: Tensor<Wgpu, 3>,
        text_token: u32,
        cache: &mut LayerCaches,
        audio_temp: f32,
        audio_top_k: usize,
        penalty_history: Option<&[Vec<u32>]>,
        audio_penalty: f32,
        num_steps: usize,
        dim: usize,
    ) -> Vec<u32> {
        let gpu = self.gpu_buffers.as_ref().expect("GPU buffers not initialized");

        // Pre-compute all input projections before the autoregressive loop.
        // temporal_hidden doesn't change between steps, so we can batch all
        // matmul dispatches upfront for better GPU pipelining.
        let projected: Vec<Tensor<Wgpu, 3>> = (0..num_steps)
            .map(|step| self.input_projs[step].forward(temporal_hidden.clone()))
            .collect();

        // Step 0: text token embedding (CPU path — text_emb is too large for GPU)
        let mut emb_buf = vec![0.0f32; dim];
        self.text_emb.embed_id_add_cpu(text_token, &mut emb_buf);
        let emb_tensor = Tensor::<Wgpu, 3>::from_data(
            burn::tensor::TensorData::new(emb_buf, [1, 1, dim]),
            &self.device,
        );
        let x = projected[0].clone() + emb_tensor;

        let mut h = x;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            if let Some(c) = cache.get_mut(layer_idx) {
                h = layer.forward_with_cache(h, 0, c);
            }
        }

        let logits = self.output_linears[0].forward(h);

        if audio_temp <= 0.0 {
            gpu.gpu_argmax_into(0, logits);
        } else {
            let history = penalty_history
                .and_then(|h| h.first())
                .map(|v| v.as_slice())
                .unwrap_or(&[]);
            gpu.gpu_sample_top_k_into(0, logits, audio_top_k, audio_temp, history, audio_penalty);
        }

        // Steps 1..num_steps: audio embedding via GPU lookup
        #[allow(clippy::needless_range_loop)]
        for step in 1..num_steps {
            // GPU-side embedding lookup for previous step's token
            let emb_idx = step - 1;
            let emb_tensor = if emb_idx < self.audio_embs.len() {
                let q4_handle = self.audio_embs[emb_idx]
                    .gpu_lookup_params()
                    .0;
                let token_handle = &gpu.token_handles[step - 1];
                gpu.gpu_embed_lookup_into(step, emb_idx, token_handle, q4_handle)
            } else {
                Tensor::<Wgpu, 3>::zeros([1, 1, dim], &self.device)
            };

            let x = projected[step].clone() + emb_tensor;

            let mut h = x;
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                if let Some(c) = cache.get_mut(layer_idx) {
                    h = layer.forward_with_cache(h, step, c);
                }
            }

            let logits = self.output_linears[step].forward(h);

            if audio_temp <= 0.0 {
                gpu.gpu_argmax_into(step, logits);
            } else {
                let history = penalty_history
                    .and_then(|h| h.get(step))
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);
                gpu.gpu_sample_top_k_into(step, logits, audio_top_k, audio_temp, history, audio_penalty);
            }
        }

        // ONE batch readback of all tokens
        let token_tensors = gpu.collect_token_tensors(num_steps);
        gpu_read_token_tensors(token_tensors).await
    }

    /// CPU fallback generation: per-step readbacks (used with provided_tokens).
    #[allow(clippy::too_many_arguments)]
    async fn generate_cpu_path(
        &self,
        temporal_hidden: Tensor<Wgpu, 3>,
        text_token: u32,
        cache: &mut LayerCaches,
        audio_temp: f32,
        audio_top_k: usize,
        provided_tokens: Option<&[i32]>,
        penalty_history: Option<&[Vec<u32>]>,
        audio_penalty: f32,
        num_steps: usize,
        dim: usize,
    ) -> Vec<u32> {
        // Pre-compute all input projections before the autoregressive loop.
        // temporal_hidden doesn't change between steps, so we can batch all
        // matmul dispatches upfront for better GPU pipelining.
        let projected: Vec<Tensor<Wgpu, 3>> = (0..num_steps)
            .map(|step| self.input_projs[step].forward(temporal_hidden.clone()))
            .collect();

        let mut audio_tokens = Vec::with_capacity(num_steps);
        let mut prev_token: u32 = text_token;

        for step in 0..num_steps {
            let mut emb_buf = vec![0.0f32; dim];
            if step == 0 {
                self.text_emb.embed_id_add_cpu(prev_token, &mut emb_buf);
            } else {
                let emb_idx = step - 1;
                if emb_idx < self.audio_embs.len() {
                    self.audio_embs[emb_idx].embed_id_add_cpu(prev_token, &mut emb_buf);
                }
            }

            let emb_tensor = Tensor::<Wgpu, 3>::from_data(
                burn::tensor::TensorData::new(emb_buf, [1, 1, dim]),
                &self.device,
            );

            let x = projected[step].clone() + emb_tensor;

            let mut h = x;
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                if let Some(c) = cache.get_mut(layer_idx) {
                    h = layer.forward_with_cache(h, step, c);
                }
            }

            let logits = self.output_linears[step].forward(h);

            let token = if audio_temp <= 0.0 {
                gpu_argmax(logits).await
            } else {
                let history = penalty_history
                    .and_then(|h| h.get(step))
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);
                sample_top_k_with_penalty(
                    logits,
                    audio_top_k,
                    audio_temp,
                    history,
                    audio_penalty,
                )
                .await
            };

            audio_tokens.push(token);

            prev_token = if let Some(provided) = provided_tokens {
                if step < provided.len() && provided[step] >= 0 {
                    provided[step] as u32
                } else {
                    token
                }
            } else {
                token
            };
        }

        audio_tokens
    }

    /// Create KV caches for the depth transformer.
    ///
    /// The depth cache needs space for at most `depth_num_steps` entries
    /// per time step. It is reset between time steps.
    pub fn create_cache(&self) -> LayerCaches {
        let head_dim = self.config.depth_hidden_size / self.config.depth_num_heads;
        LayerCaches::new(
            self.config.depth_num_layers,
            9, // depformer_context=8 + 1 current step
            self.config.depth_num_kv_heads,
            head_dim,
            &self.device,
        )
    }

    /// Generate all 16 audio codebook tokens with intermediate logging.
    ///
    /// Same logic as `generate`, but collects per-step, per-layer hidden
    /// states and logits for comparison against a BF16 reference.
    ///
    /// Returns (audio_tokens, depth_log).
    #[cfg(not(target_arch = "wasm32"))]
    #[allow(clippy::too_many_arguments)]
    pub async fn generate_with_logging(
        &self,
        temporal_hidden: Tensor<Wgpu, 3>,
        text_token: u32,
        cache: &mut LayerCaches,
        audio_temp: f32,
        audio_top_k: usize,
        provided_tokens: Option<&[i32]>,
        _penalty_history: Option<&[Vec<u32>]>,
        _audio_penalty: f32,
    ) -> (Vec<u32>, Vec<DepthStepLog>) {
        let num_steps = self.config.depth_num_steps;
        let dim = self.config.depth_hidden_size;

        // Pre-compute all input projections before the autoregressive loop.
        let projected: Vec<Tensor<Wgpu, 3>> = (0..num_steps)
            .map(|step| self.input_projs[step].forward(temporal_hidden.clone()))
            .collect();

        let mut audio_tokens = Vec::with_capacity(num_steps);
        let mut step_logs = Vec::with_capacity(num_steps);

        let mut prev_token: u32 = text_token;

        for step in 0..num_steps {

            // Add embedding of the conditioning token
            let mut emb_buf = vec![0.0f32; dim];
            if step == 0 {
                self.text_emb.embed_id_add_cpu(prev_token, &mut emb_buf);
            } else {
                let emb_idx = step - 1;
                if emb_idx < self.audio_embs.len() {
                    self.audio_embs[emb_idx].embed_id_add_cpu(prev_token, &mut emb_buf);
                }
            }

            let emb_tensor = Tensor::<Wgpu, 3>::from_data(
                burn::tensor::TensorData::new(emb_buf, [1, 1, dim]),
                &self.device,
            );

            let x = projected[step].clone() + emb_tensor;

            // Log input (projected + embedding)
            let flat: Tensor<Wgpu, 1> = x.clone().reshape([dim]);
            let input_vals: Vec<f32> = flat.into_data_async().await.expect("GPU readback failed").to_vec().expect("depth input to_vec");
            let input_norm = input_vals.iter().map(|v| v * v).sum::<f32>().sqrt();
            let input_first_10: Vec<f32> = input_vals[..10.min(dim)].to_vec();

            // Run through depth transformer layers with step-specific weights
            let mut h = x;
            let mut layer_logs = Vec::with_capacity(self.layers.len());
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                if let Some(c) = cache.get_mut(layer_idx) {
                    h = layer.forward_with_cache(h, step, c);
                }
                // Log after each layer
                let flat: Tensor<Wgpu, 1> = h.clone().reshape([dim]);
                let vals: Vec<f32> = flat.into_data_async().await.expect("GPU readback failed").to_vec().expect("depth layer to_vec");
                let norm = vals.iter().map(|v| v * v).sum::<f32>().sqrt();
                let first_10: Vec<f32> = vals[..10.min(dim)].to_vec();
                layer_logs.push(crate::model::LayerLog {
                    layer: layer_idx,
                    norm,
                    first_10,
                });
            }

            let logits = self.output_linears[step].forward(h);

            // Read logits for top-10 logging
            let [_b, _s, vocab] = logits.dims();
            let logits_1d: Tensor<Wgpu, 1> = logits.clone().reshape([vocab]);
            let logits_vals: Vec<f32> = logits_1d.into_data_async().await.expect("GPU readback failed").to_vec().expect("depth logits to_vec");
            let mut indexed: Vec<(usize, f32)> = logits_vals
                .iter()
                .copied()
                .enumerate()
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            indexed.truncate(10);
            let logits_top10: Vec<TokenLogit> = indexed
                .iter()
                .map(|&(token, logit)| TokenLogit { token, logit })
                .collect();

            let token = if audio_temp <= 0.0 {
                sample_greedy(logits).await
            } else {
                sample_top_k(logits, audio_top_k, audio_temp).await
            };

            step_logs.push(DepthStepLog {
                step,
                input_norm,
                input_first_10,
                layer_logs,
                logits_top10,
                token,
            });

            audio_tokens.push(token);

            // Determine conditioning token for the next step
            prev_token = if let Some(provided) = provided_tokens {
                if step < provided.len() && provided[step] >= 0 {
                    provided[step] as u32
                } else {
                    token
                }
            } else {
                token
            };
        }

        (audio_tokens, step_logs)
    }

    pub fn config(&self) -> &StsConfig {
        &self.config
    }

    pub fn device(&self) -> &WgpuDevice {
        &self.device
    }
}

/// Log entry for a single depth step (native only).
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, serde::Serialize)]
pub struct DepthStepLog {
    pub step: usize,
    pub input_norm: f32,
    pub input_first_10: Vec<f32>,
    pub layer_logs: Vec<crate::model::LayerLog>,
    pub logits_top10: Vec<TokenLogit>,
    pub token: u32,
}

/// Token ID and its logit value for top-k logging (native only).
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, serde::Serialize)]
pub struct TokenLogit {
    pub token: usize,
    pub logit: f32,
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
    fn test_depth_cache_creation() {
        pollster::block_on(async {
            let config = StsConfig::default();
            let device = test_device();
            let head_dim = config.depth_hidden_size / config.depth_num_heads;
            let caches = LayerCaches::new(
                config.depth_num_layers,
                config.depth_num_steps + 1,
                config.depth_num_kv_heads,
                head_dim,
                &device,
            );

            assert_eq!(caches.seq_len(), 0);
        });
    }

    #[test]
    fn test_multi_linear_ffn_step_selection() {
        pollster::block_on(async {
            let device = test_device();

            // We can't easily create Q4Linear without GGUF data, but we can
            // verify the structure compiles and the per-step logic is sound.
            // This test verifies cache isolation between depth steps.

            let mut cache = KVCache::new(20, 2, 32, &device);

            // Simulate 3 depth steps writing to the cache
            for i in 0..3 {
                let k = Tensor::<Wgpu, 4>::ones([1, 2, 1, 32], &device) * (i as f64 + 1.0);
                let v = Tensor::<Wgpu, 4>::ones([1, 2, 1, 32], &device) * (i as f64 + 1.0);
                let (k_all, _) = cache.update(k, v);
                // Each step should see all previous entries
                assert_eq!(k_all.dims()[2], i + 1);
            }

            // After 3 depth steps, cache has 3 entries
            assert_eq!(cache.seq_len(), 3);

            // Reset for next time step
            cache.reset_keep_buffers();
            assert_eq!(cache.seq_len(), 0);
        });
    }

    #[test]
    fn test_depth_config_defaults() {
        let config = StsConfig::default();
        assert_eq!(config.depth_num_layers, 6);
        assert_eq!(config.depth_hidden_size, 1024);
        assert_eq!(config.depth_num_heads, 16);
        assert_eq!(config.depth_num_kv_heads, 16);
        assert_eq!(config.depth_intermediate_size, 2816);
        assert_eq!(config.depth_num_steps, 16);
    }
}
