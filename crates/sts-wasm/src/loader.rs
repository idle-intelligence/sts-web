//! Two-phase GGUF weight loading for the STS model.
//!
//! Phase 1 (`load_deferred`): Parse GGUF, load Q4 linear weights onto GPU,
//! store embedding bytes on CPU. The GGUF reader is dropped after this phase
//! to free its memory.
//!
//! Phase 2 (`finalize`): Create EmbeddingStore objects from raw bytes and
//! assemble the TemporalTransformer + DepthTransformer.
//!
//! Tensor naming conventions (PersonaPlex-7B GGUF):
//!
//! Temporal transformer:
//!   transformer.layers.{N}.self_attn.in_proj_weight   [12288, 4096]
//!   transformer.layers.{N}.self_attn.out_proj.weight   [4096, 4096]
//!   transformer.layers.{N}.gating.linear_in.weight     [22528, 4096]
//!   transformer.layers.{N}.gating.linear_out.weight    [4096, 11264]
//!   transformer.layers.{N}.norm1.alpha                 [1, 1, 4096]
//!   transformer.layers.{N}.norm2.alpha                 [1, 1, 4096]
//!   out_norm.alpha                                     [1, 1, 4096]
//!   text_linear.weight                                 [32000, 4096]
//!
//! Embeddings:
//!   text_emb.weight                        [32001, 4096]
//!   emb.{0-7}.weight                       [2049, 4096] (model audio)
//!   emb.{8-15}.weight                      [2049, 4096] (user audio)
//!
//! Depth transformer (Depformer):
//!   depformer.layers.{N}.self_attn.in_proj_weight       [49152, 1024] packed 16x[3072,1024]
//!   depformer.layers.{N}.self_attn.out_proj.weight      [16384, 1024] packed 16x[1024,1024]
//!   depformer.layers.{N}.gating.{S}.linear_in.weight    [5632, 1024]
//!   depformer.layers.{N}.gating.{S}.linear_out.weight   [1024, 2816]
//!   depformer.layers.{N}.norm1.alpha                    [1, 1, 1024]
//!   depformer.layers.{N}.norm2.alpha                    [1, 1, 1024]
//!   depformer_in.{0-15}.weight                          [1024, 4096]
//!   depformer_text_emb.weight                           [32001, 1024]
//!   depformer_emb.{0-14}.weight                         [2049, 1024]
//!   linears.{0-15}.weight                               [vocab, 1024]

use anyhow::{bail, Context, Result};
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::{Tensor, TensorData};
use std::collections::HashMap;

use crate::depth::{
    DepthTransformer, DepthTransformerBlock, MultiLinearAttention, MultiLinearFeedForward,
};
use crate::gguf::{
    f16_to_f32, reverse_gguf_dims, EmbeddingStore, GgmlDtype, GgufTensorIndex, GgufTensorInfo,
    Linear, Q4KLinear, Q4KTensor, Q4Linear, Q4ModelLoader, Q4Tensor, DenseLinear,
};
use crate::model::{
    Q4Attention, Q4FeedForward, Q4TransformerBlock, RoPE, RmsNormLayer, TemporalTransformer,
};
use crate::StsConfig;
use std::io::{Read, Seek};

// ---------------------------------------------------------------------------
// StsModelParts — intermediate state between phase 1 and 2
// ---------------------------------------------------------------------------

/// All Q4 model components with embeddings still in raw Q4 form.
///
/// Drop the `Q4ModelLoader` before calling `finalize` to free GGUF memory.
pub struct StsModelParts {
    // Temporal transformer
    pub temporal_layers: Vec<Q4TransformerBlock>,
    pub temporal_rope: RoPE,
    pub temporal_out_norm: RmsNormLayer,
    pub text_linear: Linear,

    // Temporal embeddings (raw Q4 bytes for deferred EmbeddingStore creation)
    pub text_emb_bytes: Vec<u8>,
    pub text_emb_shape: [usize; 2],
    pub model_audio_emb_bytes: Vec<Vec<u8>>,
    pub model_audio_emb_shapes: Vec<[usize; 2]>,
    pub user_audio_emb_bytes: Vec<Vec<u8>>,
    pub user_audio_emb_shapes: Vec<[usize; 2]>,

    // Depth transformer
    pub depth_layers: Vec<DepthTransformerBlock>,
    pub depth_input_projs: Vec<Linear>,
    pub depth_output_linears: Vec<Linear>,

    // Depth embeddings (raw Q4 bytes)
    pub depth_text_emb_bytes: Vec<u8>,
    pub depth_text_emb_shape: [usize; 2],
    pub depth_audio_emb_bytes: Vec<Vec<u8>>,
    pub depth_audio_emb_shapes: Vec<[usize; 2]>,

    pub config: StsConfig,
}

impl StsModelParts {
    /// Phase 2: Assemble the final models from deferred parts.
    pub fn finalize(
        self,
        device: &WgpuDevice,
    ) -> Result<(TemporalTransformer, DepthTransformer)> {
        // -- Temporal embeddings --
        let text_emb = EmbeddingStore::new(
            self.text_emb_bytes,
            self.text_emb_shape[0],
            self.text_emb_shape[1],
        );

        let model_audio_emb: Vec<EmbeddingStore> = self
            .model_audio_emb_bytes
            .into_iter()
            .zip(self.model_audio_emb_shapes.iter())
            .map(|(bytes, &[vocab, dim])| EmbeddingStore::new(bytes, vocab, dim))
            .collect();

        let user_audio_emb: Vec<EmbeddingStore> = self
            .user_audio_emb_bytes
            .into_iter()
            .zip(self.user_audio_emb_shapes.iter())
            .map(|(bytes, &[vocab, dim])| EmbeddingStore::new(bytes, vocab, dim))
            .collect();

        let temporal = TemporalTransformer::new(
            text_emb,
            model_audio_emb,
            user_audio_emb,
            self.temporal_layers,
            self.temporal_rope,
            self.temporal_out_norm,
            self.text_linear,
            self.config.clone(),
            device.clone(),
        );

        // -- Depth embeddings --
        let depth_text_emb = EmbeddingStore::new(
            self.depth_text_emb_bytes,
            self.depth_text_emb_shape[0],
            self.depth_text_emb_shape[1],
        );

        let depth_audio_embs: Vec<EmbeddingStore> = self
            .depth_audio_emb_bytes
            .into_iter()
            .zip(self.depth_audio_emb_shapes.iter())
            .map(|(bytes, &[vocab, dim])| EmbeddingStore::new(bytes, vocab, dim))
            .collect();

        let depth = DepthTransformer::new(
            self.depth_input_projs,
            depth_text_emb,
            depth_audio_embs,
            self.depth_layers,
            self.depth_output_linears,
            self.config,
            device.clone(),
        );

        Ok((temporal, depth))
    }
}

// ---------------------------------------------------------------------------
// Loading implementation
// ---------------------------------------------------------------------------

/// Load the full STS model from a Q4ModelLoader (phase 1).
///
/// This loads all Q4 weights onto GPU and stores embedding bytes on CPU.
/// Drop the loader after this to free GGUF memory, then call `finalize`.
pub fn load_sts_model_deferred<R: Read + Seek>(
    loader: &mut Q4ModelLoader<R>,
    config: &StsConfig,
    device: &WgpuDevice,
) -> Result<StsModelParts> {
    tracing::info!(
        version = loader.version(),
        tensors = loader.tensor_count(),
        "Loading Q4 STS model from GGUF (deferred)"
    );

    // -----------------------------------------------------------------------
    // Temporal transformer layers
    // -----------------------------------------------------------------------
    let head_dim = config.hidden_size / config.num_heads;
    let temporal_rope = RoPE::new(head_dim, config.max_seq_len, config.rope_theta, device);

    let mut temporal_layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let layer = load_temporal_layer(loader, i, config, device)
            .with_context(|| format!("Failed to load temporal layer {i}"))?;
        temporal_layers.push(layer);
    }

    let temporal_out_norm = loader.load_rms_norm("out_norm.alpha", 1e-8, device)?;
    let text_linear = loader.load_linear_auto("text_linear.weight", device)?;

    // -----------------------------------------------------------------------
    // Temporal embeddings (raw bytes, deferred)
    // -----------------------------------------------------------------------
    let (text_emb_bytes, text_emb_shape) =
        loader.load_embedding_bytes("text_emb.weight")?;

    let mut model_audio_emb_bytes = Vec::with_capacity(config.num_codebooks);
    let mut model_audio_emb_shapes = Vec::with_capacity(config.num_codebooks);
    for i in 0..config.num_codebooks {
        let name = format!("emb.{i}.weight");
        let (bytes, shape) = loader.load_embedding_bytes(&name)?;
        model_audio_emb_bytes.push(bytes);
        model_audio_emb_shapes.push(shape);
    }

    let mut user_audio_emb_bytes = Vec::with_capacity(config.num_codebooks);
    let mut user_audio_emb_shapes = Vec::with_capacity(config.num_codebooks);
    for i in 0..config.num_codebooks {
        let idx = config.num_codebooks + i;
        let name = format!("emb.{idx}.weight");
        let (bytes, shape) = loader.load_embedding_bytes(&name)?;
        user_audio_emb_bytes.push(bytes);
        user_audio_emb_shapes.push(shape);
    }

    // -----------------------------------------------------------------------
    // Depth transformer layers
    // -----------------------------------------------------------------------
    // Note: depth transformer has no positional embedding (depformer_pos_emb="none")
    let mut depth_layers = Vec::with_capacity(config.depth_num_layers);
    for i in 0..config.depth_num_layers {
        let layer = load_depth_layer(loader, i, config, device)
            .with_context(|| format!("Failed to load depth layer {i}"))?;
        depth_layers.push(layer);
    }

    // -----------------------------------------------------------------------
    // Depth input projections and output heads
    // -----------------------------------------------------------------------
    let num_steps = config.depth_num_steps;

    let mut depth_input_projs = Vec::with_capacity(num_steps);
    for s in 0..num_steps {
        let name = format!("depformer_in.{s}.weight");
        let proj = loader.load_linear_auto(&name, device)
            .with_context(|| format!("Failed to load depth input proj {s}"))?;
        depth_input_projs.push(proj);
    }

    let mut depth_output_linears = Vec::with_capacity(num_steps);
    for s in 0..num_steps {
        let name = format!("linears.{s}.weight");
        let linear = loader.load_linear_auto(&name, device)
            .with_context(|| format!("Failed to load depth output linear {s}"))?;
        depth_output_linears.push(linear);
    }

    // -----------------------------------------------------------------------
    // Depth embeddings (raw bytes, deferred)
    // -----------------------------------------------------------------------
    let (depth_text_emb_bytes, depth_text_emb_shape) =
        loader.load_embedding_bytes("depformer_text_emb.weight")?;

    // depformer_emb.{0-14} — 15 audio codebook embeddings for depth
    let num_depth_audio_embs = num_steps - 1; // step 0 is text, steps 1-15 are audio
    let mut depth_audio_emb_bytes = Vec::with_capacity(num_depth_audio_embs);
    let mut depth_audio_emb_shapes = Vec::with_capacity(num_depth_audio_embs);
    for i in 0..num_depth_audio_embs {
        let name = format!("depformer_emb.{i}.weight");
        let (bytes, shape) = loader.load_embedding_bytes(&name)?;
        depth_audio_emb_bytes.push(bytes);
        depth_audio_emb_shapes.push(shape);
    }

    tracing::info!("Q4 STS model loaded (embeddings deferred)");

    Ok(StsModelParts {
        temporal_layers,
        temporal_rope,
        temporal_out_norm,
        text_linear,
        text_emb_bytes,
        text_emb_shape,
        model_audio_emb_bytes,
        model_audio_emb_shapes,
        user_audio_emb_bytes,
        user_audio_emb_shapes,
        depth_layers,
        depth_input_projs,
        depth_output_linears,
        depth_text_emb_bytes,
        depth_text_emb_shape,
        depth_audio_emb_bytes,
        depth_audio_emb_shapes,
        config: config.clone(),
    })
}

// ---------------------------------------------------------------------------
// Layer loaders
// ---------------------------------------------------------------------------

fn load_temporal_layer<R: Read + Seek>(
    loader: &mut Q4ModelLoader<R>,
    layer_idx: usize,
    config: &StsConfig,
    device: &WgpuDevice,
) -> Result<Q4TransformerBlock> {
    let prefix = format!("transformer.layers.{layer_idx}");

    let norm1 = loader.load_rms_norm(&format!("{prefix}.norm1.alpha"), 1e-8, device)?;
    let in_proj = loader.load_linear_auto(&format!("{prefix}.self_attn.in_proj_weight"), device)?;
    let out_proj =
        loader.load_linear_auto(&format!("{prefix}.self_attn.out_proj.weight"), device)?;

    let head_dim = config.hidden_size / config.num_heads;
    let attention = Q4Attention::new(in_proj, out_proj, config.num_heads, config.num_kv_heads, head_dim);

    let norm2 = loader.load_rms_norm(&format!("{prefix}.norm2.alpha"), 1e-8, device)?;
    let linear_in =
        loader.load_linear_auto(&format!("{prefix}.gating.linear_in.weight"), device)?;
    let linear_out =
        loader.load_linear_auto(&format!("{prefix}.gating.linear_out.weight"), device)?;
    let ffn = Q4FeedForward::new(linear_in, linear_out);

    Ok(Q4TransformerBlock::new(norm1, attention, norm2, ffn))
}

/// Create a Linear from raw bytes, dispatching on dtype.
fn make_linear_from_bytes(
    bytes: &[u8],
    shape: [usize; 2],
    dtype: GgmlDtype,
    device: &WgpuDevice,
) -> Result<Linear> {
    match dtype {
        GgmlDtype::Q4_0 => {
            let t = Q4Tensor::from_q4_bytes(bytes, shape, device)?;
            Ok(Linear::Q4(Q4Linear::new(t, None)))
        }
        GgmlDtype::Q4_K => {
            let t = Q4KTensor::from_q4k_bytes(bytes, shape, device)?;
            Ok(Linear::Q4K(Q4KLinear::new(t, None)))
        }
        other => bail!("Unsupported dtype {:?} for packed attention tensor", other),
    }
}

fn load_depth_layer<R: Read + Seek>(
    loader: &mut Q4ModelLoader<R>,
    layer_idx: usize,
    config: &StsConfig,
    device: &WgpuDevice,
) -> Result<DepthTransformerBlock> {
    let prefix = format!("depformer.layers.{layer_idx}");
    let num_steps = config.depth_num_steps;

    let norm1 = loader.load_rms_norm(&format!("{prefix}.norm1.alpha"), 1e-8, device)?;
    let norm2 = loader.load_rms_norm(&format!("{prefix}.norm2.alpha"), 1e-8, device)?;

    // Multi-linear attention: the packed tensors [49152, 1024] and [16384, 1024]
    // need to be split into 16 separate Linear instances.
    //
    // in_proj_weight: [49152, 1024] = 16 x [3072, 1024]
    // out_proj.weight: [16384, 1024] = 16 x [1024, 1024]
    let in_proj_name = format!("{prefix}.self_attn.in_proj_weight");
    let out_proj_name = format!("{prefix}.self_attn.out_proj.weight");

    let in_proj_dtype = loader
        .tensor_info(&in_proj_name)
        .with_context(|| format!("Tensor '{in_proj_name}' not found"))?
        .dtype();
    let out_proj_dtype = loader
        .tensor_info(&out_proj_name)
        .with_context(|| format!("Tensor '{out_proj_name}' not found"))?
        .dtype();

    let in_proj_packed = loader.tensor_data(&in_proj_name)?;
    let out_proj_packed = loader.tensor_data(&out_proj_name)?;

    let k_dim = config.depth_hidden_size;
    let in_proj_per_step_rows = 3 * k_dim; // 3072 for 1024-dim
    let out_proj_per_step_rows = k_dim; // 1024

    let in_proj_per_step_elements = in_proj_per_step_rows * k_dim;
    let in_proj_per_step_bytes = in_proj_dtype.byte_size(in_proj_per_step_elements as u64) as usize;

    let out_proj_per_step_elements = out_proj_per_step_rows * k_dim;
    let out_proj_per_step_bytes = out_proj_dtype.byte_size(out_proj_per_step_elements as u64) as usize;

    let mut in_projs = Vec::with_capacity(num_steps);
    let mut out_projs = Vec::with_capacity(num_steps);

    for s in 0..num_steps {
        let in_start = s * in_proj_per_step_bytes;
        let in_end = in_start + in_proj_per_step_bytes;
        let in_bytes = &in_proj_packed[in_start..in_end];
        let in_linear = make_linear_from_bytes(
            in_bytes,
            [in_proj_per_step_rows, k_dim],
            in_proj_dtype,
            device,
        )?;
        in_projs.push(in_linear);

        let out_start = s * out_proj_per_step_bytes;
        let out_end = out_start + out_proj_per_step_bytes;
        let out_bytes = &out_proj_packed[out_start..out_end];
        let out_linear = make_linear_from_bytes(
            out_bytes,
            [out_proj_per_step_rows, k_dim],
            out_proj_dtype,
            device,
        )?;
        out_projs.push(out_linear);
    }

    let head_dim = config.depth_hidden_size / config.depth_num_heads;
    let attention = MultiLinearAttention::new(
        in_projs,
        out_projs,
        config.depth_num_heads,
        config.depth_num_kv_heads,
        head_dim,
    );

    // Multi-linear FFN: per-step gating weights
    let mut linear_ins = Vec::with_capacity(num_steps);
    let mut linear_outs = Vec::with_capacity(num_steps);
    for s in 0..num_steps {
        let li = loader.load_linear_auto(
            &format!("{prefix}.gating.{s}.linear_in.weight"),
            device,
        )?;
        let lo = loader.load_linear_auto(
            &format!("{prefix}.gating.{s}.linear_out.weight"),
            device,
        )?;
        linear_ins.push(li);
        linear_outs.push(lo);
    }

    let ffn = MultiLinearFeedForward::new(linear_ins, linear_outs);

    Ok(DepthTransformerBlock::new(norm1, attention, norm2, ffn))
}

// ===========================================================================
// Incremental (shard-at-a-time) model loading for WASM
// ===========================================================================

/// Processed tensor variants accumulated across shards.
///
/// Linear/norm weights are uploaded to GPU immediately so the raw bytes can
/// be freed. Embedding bytes stay on CPU for row-lookup at inference time.
enum ProcessedTensor {
    /// A quantized or dense linear layer already on GPU.
    Linear(Linear),
    /// An RMS-norm weight tensor on GPU.
    Norm(RmsNormLayer),
    /// Raw embedding bytes on CPU with shape [vocab, dim].
    Embedding(Vec<u8>, [usize; 2]),
}

/// Incremental model loader that processes one shard at a time.
///
/// This keeps only ~512 MB of shard data in WASM memory at once (plus model
/// structures on GPU), avoiding the 4 GB address space crash.
///
/// Usage:
///   1. `new()` with the parsed tensor index from shard 0
///   2. `process_shard()` for each shard (including shard 0)
///   3. `finalize()` to assemble the model
pub struct IncrementalModelLoader {
    index: GgufTensorIndex,
    config: StsConfig,
    device: WgpuDevice,
    /// Accumulated processed tensors keyed by GGUF tensor name.
    processed: HashMap<String, ProcessedTensor>,
    /// Tracks cumulative shard sizes to compute each shard's absolute start.
    shard_sizes: Vec<u64>,
    /// Partial tensor bytes for tensors that span shard boundaries.
    /// Key: tensor name, Value: (partial bytes so far, total expected byte size).
    partial_tensors: HashMap<String, (Vec<u8>, u64)>,
}

impl IncrementalModelLoader {
    pub fn new(index: GgufTensorIndex, config: StsConfig, device: WgpuDevice) -> Self {
        Self {
            index,
            config,
            device,
            processed: HashMap::new(),
            shard_sizes: Vec::new(),
            partial_tensors: HashMap::new(),
        }
    }

    /// Process all tensors whose data resides in this shard.
    ///
    /// `shard_data` is the raw bytes of the shard. `shard_index` is 0-based.
    /// For shard 0 this should be called after parsing the header (the header
    /// bytes are part of shard 0's data).
    pub fn process_shard(&mut self, shard_data: &[u8], shard_index: usize) -> Result<()> {
        // Compute absolute start for this shard from accumulated sizes.
        // Shards must be processed in order (0, 1, 2, ...).
        if shard_index != self.shard_sizes.len() {
            bail!(
                "Shard index {shard_index} out of order (expected {})",
                self.shard_sizes.len()
            );
        }
        let shard_abs_start: u64 = self.shard_sizes.iter().sum();
        let shard_abs_end = shard_abs_start + shard_data.len() as u64;
        self.shard_sizes.push(shard_data.len() as u64);

        // Find all tensors in this shard's byte range.
        let tensors_in_shard = self.index.tensors_in_range(shard_abs_start, shard_abs_end);
        tracing::info!(
            shard = shard_index,
            tensors = tensors_in_shard.len(),
            abs_range = format!("{}..{}", shard_abs_start, shard_abs_end),
            "Processing shard"
        );

        // First, complete any partial tensors from the previous shard.
        let partial_names: Vec<String> = self.partial_tensors.keys().cloned().collect();
        for name in partial_names {
            let tensor_info = match self.index.tensor_info(&name) {
                Some(t) => t.clone(),
                None => continue,
            };
            let t_abs_start = self.index.data_section_offset + tensor_info.offset();
            let _t_byte_size = tensor_info.byte_size();

            // How many bytes remain to read from this shard?
            let (ref mut partial_buf, total_size) = self.partial_tensors.get_mut(&name).unwrap();
            let already_have = partial_buf.len() as u64;
            let remaining = *total_size - already_have;

            // The remaining bytes start at the beginning of this shard
            // (tensor data continues from where previous shard ended).
            let local_start = (t_abs_start + already_have - shard_abs_start) as usize;
            let local_end = local_start + remaining as usize;

            if local_end <= shard_data.len() {
                partial_buf.extend_from_slice(&shard_data[local_start..local_end]);
                let (completed_bytes, _) = self.partial_tensors.remove(&name).unwrap();
                let entries = self.process_tensor(&tensor_info, &completed_bytes)?;
                for (entry_name, entry) in entries {
                    self.processed.insert(entry_name, entry);
                }
            }
            // else: tensor spans 3+ shards (very unlikely), keep buffering
        }

        for tensor_info in tensors_in_shard {
            // Skip tensors we already processed or are partially buffered.
            if self.processed.contains_key(&tensor_info.name)
                || self.partial_tensors.contains_key(&tensor_info.name)
            {
                continue;
            }

            let t_abs_start = self.index.data_section_offset + tensor_info.offset();
            let t_byte_size = tensor_info.byte_size();
            let t_abs_end = t_abs_start + t_byte_size;

            // Read the tensor data from the shard.
            if t_abs_start >= shard_abs_start && t_abs_end <= shard_abs_end {
                // Tensor fully within this shard
                let local_start = (t_abs_start - shard_abs_start) as usize;
                let local_end = local_start + t_byte_size as usize;
                let bytes = shard_data[local_start..local_end].to_vec();

                let entries = self.process_tensor(tensor_info, &bytes)?;
                for (entry_name, entry) in entries {
                    self.processed.insert(entry_name, entry);
                }
            } else if t_abs_start >= shard_abs_start && t_abs_start < shard_abs_end {
                // Tensor starts in this shard but extends past its end — buffer partial
                let local_start = (t_abs_start - shard_abs_start) as usize;
                let mut partial = Vec::with_capacity(t_byte_size as usize);
                partial.extend_from_slice(&shard_data[local_start..]);
                self.partial_tensors
                    .insert(tensor_info.name.clone(), (partial, t_byte_size));
                tracing::info!(
                    name = tensor_info.name,
                    have = shard_data.len() - local_start,
                    need = t_byte_size,
                    "Tensor spans shard boundary, buffering partial"
                );
            }
        }

        Ok(())
    }

    /// Classify and process a single tensor based on its name and dtype.
    ///
    /// Returns a list of (name, ProcessedTensor) pairs. Usually a single entry,
    /// but packed depth attention tensors are split into `num_steps` sub-entries.
    fn process_tensor(
        &self,
        info: &GgufTensorInfo,
        bytes: &[u8],
    ) -> Result<Vec<(String, ProcessedTensor)>> {
        let name = &info.name;

        // Embeddings: store raw bytes on CPU
        if is_embedding_tensor(name) {
            let shape = reverse_gguf_dims(info.shape());
            return Ok(vec![(
                name.clone(),
                ProcessedTensor::Embedding(bytes.to_vec(), [shape[0], shape[1]]),
            )]);
        }

        // Norm tensors: load as f32 tensor
        if name.ends_with(".alpha") {
            let norm = self.load_norm_from_bytes(info, bytes)?;
            return Ok(vec![(name.clone(), ProcessedTensor::Norm(norm))]);
        }

        // Packed depth attention tensors: split into per-step sub-tensors
        if is_packed_depth_attn(name) {
            return self.split_packed_depth_attn(info, bytes);
        }

        // Everything else is a linear weight — upload to GPU
        let linear = self.load_linear_from_bytes(info, bytes)?;
        Ok(vec![(name.clone(), ProcessedTensor::Linear(linear))])
    }

    /// Split a packed depth attention tensor into `num_steps` sub-Linear entries.
    ///
    /// Packed tensors:
    ///   `depformer.layers.{N}.self_attn.in_proj_weight`  [49152, K] = 16 x [3072, K]
    ///   `depformer.layers.{N}.self_attn.out_proj.weight`  [16384, K] = 16 x [K, K]
    fn split_packed_depth_attn(
        &self,
        info: &GgufTensorInfo,
        bytes: &[u8],
    ) -> Result<Vec<(String, ProcessedTensor)>> {
        let name = &info.name;
        let num_steps = self.config.depth_num_steps;
        let dtype = info.dtype();

        let shape = reverse_gguf_dims(info.shape());
        let total_rows = shape[0];
        let cols = shape[1];
        let per_step_rows = total_rows / num_steps;
        let per_step_elements = (per_step_rows * cols) as u64;
        let per_step_bytes = dtype.byte_size(per_step_elements) as usize;

        let mut results = Vec::with_capacity(num_steps);
        for s in 0..num_steps {
            let start = s * per_step_bytes;
            let end = start + per_step_bytes;
            let step_bytes = &bytes[start..end];
            let step_linear = make_linear_from_bytes(
                step_bytes,
                [per_step_rows, cols],
                dtype,
                &self.device,
            )?;
            let split_name = format!("{name}.split.{s}");
            results.push((split_name, ProcessedTensor::Linear(step_linear)));
        }

        Ok(results)
    }

    /// Load a norm tensor from raw bytes.
    fn load_norm_from_bytes(
        &self,
        info: &GgufTensorInfo,
        bytes: &[u8],
    ) -> Result<RmsNormLayer> {
        let num_elements = info.num_elements();
        let data: Vec<f32> = match info.dtype() {
            GgmlDtype::F32 => bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            GgmlDtype::F16 => bytes
                .chunks_exact(2)
                .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect(),
            other => bail!("Unsupported dtype {:?} for norm tensor '{}'", other, info.name),
        };
        let alpha = Tensor::<Wgpu, 1>::from_data(
            TensorData::new(data, [num_elements as usize]),
            &self.device,
        );
        Ok(RmsNormLayer::new(alpha, 1e-8))
    }

    /// Load a linear layer from raw bytes, dispatching on dtype.
    fn load_linear_from_bytes(
        &self,
        info: &GgufTensorInfo,
        bytes: &[u8],
    ) -> Result<Linear> {
        let shape = reverse_gguf_dims(info.shape());
        match info.dtype() {
            GgmlDtype::Q4_0 => {
                let t = Q4Tensor::from_q4_bytes(bytes, [shape[0], shape[1]], &self.device)?;
                Ok(Linear::Q4(Q4Linear::new(t, None)))
            }
            GgmlDtype::Q4_K => {
                let t = Q4KTensor::from_q4k_bytes(bytes, [shape[0], shape[1]], &self.device)?;
                Ok(Linear::Q4K(Q4KLinear::new(t, None)))
            }
            GgmlDtype::F32 | GgmlDtype::F16 => {
                let rows = shape[0];
                let cols = shape[1];
                let data: Vec<f32> = match info.dtype() {
                    GgmlDtype::F32 => bytes
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect(),
                    GgmlDtype::F16 => bytes
                        .chunks_exact(2)
                        .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                        .collect(),
                    _ => unreachable!(),
                };
                let weight = Tensor::<Wgpu, 2>::from_data(
                    TensorData::new(data, [rows, cols]),
                    &self.device,
                );
                Ok(Linear::Dense(DenseLinear::new(weight)))
            }
        }
    }

    /// Take a processed tensor out of the accumulator, or error.
    fn take_linear(&mut self, name: &str) -> Result<Linear> {
        match self.processed.remove(name) {
            Some(ProcessedTensor::Linear(l)) => Ok(l),
            Some(_) => bail!("Expected Linear for '{name}', got different type"),
            None => bail!("Tensor '{name}' not found in processed tensors"),
        }
    }

    fn take_norm(&mut self, name: &str) -> Result<RmsNormLayer> {
        match self.processed.remove(name) {
            Some(ProcessedTensor::Norm(n)) => Ok(n),
            Some(_) => bail!("Expected Norm for '{name}', got different type"),
            None => bail!("Tensor '{name}' not found in processed tensors"),
        }
    }

    fn take_embedding(&mut self, name: &str) -> Result<(Vec<u8>, [usize; 2])> {
        match self.processed.remove(name) {
            Some(ProcessedTensor::Embedding(bytes, shape)) => Ok((bytes, shape)),
            Some(_) => bail!("Expected Embedding for '{name}', got different type"),
            None => bail!("Tensor '{name}' not found in processed tensors"),
        }
    }

    /// Number of tensors processed so far.
    pub fn processed_count(&self) -> usize {
        self.processed.len()
    }

    /// Total number of tensors expected (from the GGUF index).
    pub fn total_tensor_count(&self) -> u64 {
        self.index.tensor_count
    }

    /// Assemble the final model from all processed tensors.
    ///
    /// Must be called after all shards have been processed.
    pub fn finalize(mut self) -> Result<(TemporalTransformer, DepthTransformer)> {
        let config = self.config.clone();
        let device = self.device.clone();

        // ---------------------------------------------------------------
        // Temporal transformer
        // ---------------------------------------------------------------
        let head_dim = config.hidden_size / config.num_heads;
        let temporal_rope = RoPE::new(head_dim, config.max_seq_len, config.rope_theta, &device);

        let mut temporal_layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let prefix = format!("transformer.layers.{i}");

            let norm1 = self.take_norm(&format!("{prefix}.norm1.alpha"))
                .with_context(|| format!("Missing norm1 for temporal layer {i}"))?;
            let in_proj = self.take_linear(&format!("{prefix}.self_attn.in_proj_weight"))
                .with_context(|| format!("Missing in_proj for temporal layer {i}"))?;
            let out_proj = self.take_linear(&format!("{prefix}.self_attn.out_proj.weight"))
                .with_context(|| format!("Missing out_proj for temporal layer {i}"))?;
            let attention = Q4Attention::new(
                in_proj, out_proj, config.num_heads, config.num_kv_heads, head_dim,
            );

            let norm2 = self.take_norm(&format!("{prefix}.norm2.alpha"))
                .with_context(|| format!("Missing norm2 for temporal layer {i}"))?;
            let linear_in = self.take_linear(&format!("{prefix}.gating.linear_in.weight"))
                .with_context(|| format!("Missing ffn linear_in for temporal layer {i}"))?;
            let linear_out = self.take_linear(&format!("{prefix}.gating.linear_out.weight"))
                .with_context(|| format!("Missing ffn linear_out for temporal layer {i}"))?;
            let ffn = Q4FeedForward::new(linear_in, linear_out);

            temporal_layers.push(Q4TransformerBlock::new(norm1, attention, norm2, ffn));
        }

        let temporal_out_norm = self.take_norm("out_norm.alpha")?;
        let text_linear = self.take_linear("text_linear.weight")?;

        // Temporal embeddings
        let (text_emb_bytes, text_emb_shape) = self.take_embedding("text_emb.weight")?;
        let text_emb = EmbeddingStore::new(text_emb_bytes, text_emb_shape[0], text_emb_shape[1]);

        let mut model_audio_emb = Vec::with_capacity(config.num_codebooks);
        for i in 0..config.num_codebooks {
            let (bytes, shape) = self.take_embedding(&format!("emb.{i}.weight"))?;
            model_audio_emb.push(EmbeddingStore::new(bytes, shape[0], shape[1]));
        }

        let mut user_audio_emb = Vec::with_capacity(config.num_codebooks);
        for i in 0..config.num_codebooks {
            let idx = config.num_codebooks + i;
            let (bytes, shape) = self.take_embedding(&format!("emb.{idx}.weight"))?;
            user_audio_emb.push(EmbeddingStore::new(bytes, shape[0], shape[1]));
        }

        let temporal = TemporalTransformer::new(
            text_emb,
            model_audio_emb,
            user_audio_emb,
            temporal_layers,
            temporal_rope,
            temporal_out_norm,
            text_linear,
            config.clone(),
            device.clone(),
        );

        // ---------------------------------------------------------------
        // Depth transformer
        // ---------------------------------------------------------------
        let num_steps = config.depth_num_steps;

        let mut depth_layers = Vec::with_capacity(config.depth_num_layers);
        for i in 0..config.depth_num_layers {
            let prefix = format!("depformer.layers.{i}");

            let norm1 = self.take_norm(&format!("{prefix}.norm1.alpha"))
                .with_context(|| format!("Missing norm1 for depth layer {i}"))?;
            let norm2 = self.take_norm(&format!("{prefix}.norm2.alpha"))
                .with_context(|| format!("Missing norm2 for depth layer {i}"))?;

            // The packed attention tensors were processed as single Linear entries.
            // We need to split them into per-step Linear instances.
            let in_proj_name = format!("{prefix}.self_attn.in_proj_weight");
            let out_proj_name = format!("{prefix}.self_attn.out_proj.weight");

            // Packed depth attention tensors were split into per-step sub-Linear
            // entries during process_shard, keyed as "{name}.split.{s}".

            let mut in_projs = Vec::with_capacity(num_steps);
            let mut out_projs = Vec::with_capacity(num_steps);

            for s in 0..num_steps {
                let in_split_name = format!("{in_proj_name}.split.{s}");
                let in_linear = self.take_linear(&in_split_name)
                    .with_context(|| format!("Missing {in_split_name}"))?;
                in_projs.push(in_linear);

                let out_split_name = format!("{out_proj_name}.split.{s}");
                let out_linear = self.take_linear(&out_split_name)
                    .with_context(|| format!("Missing {out_split_name}"))?;
                out_projs.push(out_linear);
            }

            let depth_head_dim = config.depth_hidden_size / config.depth_num_heads;
            let attention = MultiLinearAttention::new(
                in_projs,
                out_projs,
                config.depth_num_heads,
                config.depth_num_kv_heads,
                depth_head_dim,
            );

            // Multi-linear FFN: per-step gating weights
            let mut linear_ins = Vec::with_capacity(num_steps);
            let mut linear_outs = Vec::with_capacity(num_steps);
            for s in 0..num_steps {
                let li = self.take_linear(&format!("{prefix}.gating.{s}.linear_in.weight"))
                    .with_context(|| format!("Missing gating.{s}.linear_in for depth layer {i}"))?;
                let lo = self.take_linear(&format!("{prefix}.gating.{s}.linear_out.weight"))
                    .with_context(|| format!("Missing gating.{s}.linear_out for depth layer {i}"))?;
                linear_ins.push(li);
                linear_outs.push(lo);
            }

            let ffn = MultiLinearFeedForward::new(linear_ins, linear_outs);
            depth_layers.push(DepthTransformerBlock::new(norm1, attention, norm2, ffn));
        }

        // Depth input projections and output heads
        let mut depth_input_projs = Vec::with_capacity(num_steps);
        for s in 0..num_steps {
            let proj = self.take_linear(&format!("depformer_in.{s}.weight"))
                .with_context(|| format!("Missing depth input proj {s}"))?;
            depth_input_projs.push(proj);
        }

        let mut depth_output_linears = Vec::with_capacity(num_steps);
        for s in 0..num_steps {
            let linear = self.take_linear(&format!("linears.{s}.weight"))
                .with_context(|| format!("Missing depth output linear {s}"))?;
            depth_output_linears.push(linear);
        }

        // Depth embeddings
        let (depth_text_emb_bytes, depth_text_emb_shape) =
            self.take_embedding("depformer_text_emb.weight")?;
        let depth_text_emb = EmbeddingStore::new(
            depth_text_emb_bytes,
            depth_text_emb_shape[0],
            depth_text_emb_shape[1],
        );

        let num_depth_audio_embs = num_steps - 1;
        let mut depth_audio_embs = Vec::with_capacity(num_depth_audio_embs);
        for i in 0..num_depth_audio_embs {
            let (bytes, shape) = self.take_embedding(&format!("depformer_emb.{i}.weight"))?;
            depth_audio_embs.push(EmbeddingStore::new(bytes, shape[0], shape[1]));
        }

        let depth = DepthTransformer::new(
            depth_input_projs,
            depth_text_emb,
            depth_audio_embs,
            depth_layers,
            depth_output_linears,
            config,
            device,
        );

        Ok((temporal, depth))
    }
}

/// Returns true if this tensor name is a packed depth attention tensor
/// that needs to be split into per-step sub-tensors.
fn is_packed_depth_attn(name: &str) -> bool {
    // depformer.layers.{N}.self_attn.in_proj_weight
    // depformer.layers.{N}.self_attn.out_proj.weight
    name.starts_with("depformer.layers.")
        && (name.ends_with(".self_attn.in_proj_weight")
            || name.ends_with(".self_attn.out_proj.weight"))
}

/// Returns true if this tensor name is an embedding table (CPU-resident raw bytes).
fn is_embedding_tensor(name: &str) -> bool {
    // text_emb.weight, emb.{N}.weight, depformer_text_emb.weight, depformer_emb.{N}.weight
    if name == "text_emb.weight" || name == "depformer_text_emb.weight" {
        return true;
    }
    // emb.{N}.weight
    if let Some(rest) = name.strip_prefix("emb.") {
        if rest.ends_with(".weight") {
            return true;
        }
    }
    // depformer_emb.{N}.weight
    if let Some(rest) = name.strip_prefix("depformer_emb.") {
        if rest.ends_with(".weight") {
            return true;
        }
    }
    false
}
