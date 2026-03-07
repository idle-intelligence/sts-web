//! Mimi audio codec wrapper using mimi-rs (Candle-based, CPU).
//!
//! Thin wrapper around the mimi-rs library providing both encode (PCM -> tokens)
//! and decode (tokens -> PCM) paths for the STS pipeline. Runs on CPU via Candle
//! while the main STS model uses Burn/wgpu on GPU.
//!
//! Data flows between the two backends as u32 token IDs, so no cross-framework
//! tensor conversion is needed.

use std::collections::HashMap;

/// Unified Mimi codec: encoder + decoder in a single model.
///
/// Wraps mimi-rs's `MimiModel` with both encoder and decoder state for
/// streaming operation. The full model is loaded once and used for both
/// encoding user audio and decoding model-generated audio tokens.
pub struct MimiCodec {
    model: mimi_rs::mimi::MimiModel,
    encoder_state: mimi_rs::mimi::MimiEncoderState,
    decoder_state: mimi_rs::mimi::MimiState,
    num_codebooks: usize,
    sample_rate: usize,
}

impl MimiCodec {
    /// Load from safetensors bytes (full Mimi model, e.g. kyutai/mimi).
    ///
    /// Applies key remapping from full-Mimi naming to mimi-rs naming,
    /// then loads both encoder and decoder components.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        let remapped = remap_mimi_weights(data);
        let vb = candle_nn::VarBuilder::from_buffered_safetensors(
            remapped,
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .map_err(|e| format!("Failed to create VarBuilder: {e}"))?;

        let cfg = mimi_rs::config::MimiConfig::mimi_v1_0_0();
        let model = mimi_rs::mimi::MimiModel::load(vb, &cfg)
            .map_err(|e| format!("Failed to load Mimi model: {e}"))?;

        let encoder_state = model
            .init_encoder_state(1, &candle_core::Device::Cpu)
            .map_err(|e| format!("Failed to init encoder state: {e}"))?;

        let decoder_state = model
            .init_state(1, &candle_core::Device::Cpu)
            .map_err(|e| format!("Failed to init decoder state: {e}"))?;

        Ok(Self {
            model,
            encoder_state,
            decoder_state,
            num_codebooks: cfg.num_codebooks,
            sample_rate: cfg.sample_rate,
        })
    }

    /// Number of codebooks (32 for standard Mimi).
    pub fn num_codebooks(&self) -> usize {
        self.num_codebooks
    }

    /// Sample rate (24000 for standard Mimi).
    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// Streaming encode: feed PCM audio samples, get back codec tokens.
    ///
    /// Returns a flat `Vec<u32>` in frame-major order:
    /// `[frame0_tok0..tok31, frame1_tok0..tok31, ...]`
    /// May return empty if not enough audio has accumulated for a frame.
    pub fn encode(&mut self, samples: &[f32]) -> Vec<u32> {
        if samples.is_empty() {
            return Vec::new();
        }

        let tensor = match candle_core::Tensor::from_vec(
            samples.to_vec(),
            (1, 1, samples.len()),
            &candle_core::Device::Cpu,
        ) {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };

        // Streaming encode: audio -> latent
        let latent = match self.model.encode_streaming(&tensor, &mut self.encoder_state) {
            Ok(l) => l,
            Err(_) => return Vec::new(),
        };

        // Check if any frames were produced
        let n_frames = latent.dim(2).unwrap_or(0);
        if n_frames == 0 {
            return Vec::new();
        }

        // Quantize latent -> token IDs [1, n_q, T']
        let codes = match self.model.quantize_to_codes(&latent) {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };

        // Flatten [1, n_q, T'] to frame-major order
        let codes_flat = match codes.squeeze(0) {
            Ok(c) => match c.to_vec2::<u32>() {
                Ok(v) => v,
                Err(_) => return Vec::new(),
            },
            Err(_) => return Vec::new(),
        };

        let n_q = codes_flat.len();
        let mut tokens = Vec::with_capacity(n_frames * n_q);
        for f in 0..n_frames {
            for cb in &codes_flat {
                tokens.push(cb[f]);
            }
        }
        tokens
    }

    /// Streaming decode: convert codec tokens back to PCM audio.
    ///
    /// `tokens` should be `num_codebooks` tokens for a single frame,
    /// ordered as `[tok_codebook0, tok_codebook1, ..., tok_codebook31]`.
    ///
    /// Returns PCM audio samples (mono, 24kHz). May return empty if
    /// the decoder hasn't accumulated enough frames yet.
    pub fn decode(&mut self, tokens: &[u32]) -> Vec<f32> {
        self.decode_n(tokens, tokens.len())
    }

    /// Decode using only the first `n_active` codebooks.
    ///
    /// When the model generates fewer codebook tokens than Mimi's full 32,
    /// this avoids adding garbage from unused codebook entries (which would
    /// otherwise contribute their index-0 vector to the reconstruction).
    pub fn decode_n(&mut self, tokens: &[u32], n_active: usize) -> Vec<f32> {
        if tokens.is_empty() || n_active == 0 {
            return Vec::new();
        }

        let n_tokens = tokens.len();

        // Build codes tensor [1, n_tokens, 1]
        let codes = match candle_core::Tensor::from_vec(
            tokens.to_vec(),
            (1, n_tokens, 1),
            &candle_core::Device::Cpu,
        ) {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };

        // Decode using only the first n_active codebooks
        let audio = match self.model.decode_from_codes_n(
            &codes,
            n_active.min(n_tokens),
            &mut self.decoder_state,
        ) {
            Ok(a) => a,
            Err(_) => return Vec::new(),
        };

        // Flatten [1, 1, T] -> Vec<f32>
        match audio.flatten_all() {
            Ok(flat) => flat.to_vec1::<f32>().unwrap_or_default(),
            Err(_) => Vec::new(),
        }
    }

    /// Reset encoder streaming state for a new session.
    pub fn reset_encoder(&mut self) {
        if let Ok(new_state) = self
            .model
            .init_encoder_state(1, &candle_core::Device::Cpu)
        {
            self.encoder_state = new_state;
        }
    }

    /// Reset decoder streaming state for a new session.
    pub fn reset_decoder(&mut self) {
        if let Ok(new_state) = self.model.init_state(1, &candle_core::Device::Cpu) {
            self.decoder_state = new_state;
        }
    }

    /// Reset both encoder and decoder state.
    pub fn reset(&mut self) {
        self.reset_encoder();
        self.reset_decoder();
    }
}

// ---------------------------------------------------------------------------
// Key remapping: full-Mimi safetensors -> mimi-rs naming
// ---------------------------------------------------------------------------
// Ported from stt-web's mimi_remap.rs. Handles:
// 1. SEANet: encoder.layers.{n} -> encoder.model.{n}
// 2. Transformer: separate Q/K/V -> combined in_proj
// 3. Norms, FFN, layer scales renamed
// 4. Downsample/upsample extra .conv. prefix

fn remap_key(name: &str) -> Option<String> {
    let mut name = name.to_string();

    // PersonaPlex checkpoint has an extra nesting level for conv/convtr layers:
    //   file: encoder.model.0.conv.conv.weight → mimi-rs: encoder.model.0.conv.weight
    //   file: downsample.conv.conv.conv.weight → mimi-rs: downsample.conv.conv.weight
    // Strip exactly one level of .conv.conv. → .conv. and .convtr.convtr. → .convtr.
    name = name.replacen(".conv.conv.", ".conv.", 1);
    name = name.replacen(".convtr.convtr.", ".convtr.", 1);

    // SEANet layers
    name = name.replace("encoder.layers.", "encoder.model.");
    name = name.replace("decoder.layers.", "decoder.model.");

    // ProjectedTransformer wrapper adds .transformer. prefix
    if name.starts_with("encoder_transformer.layers.") {
        name = name.replace(
            "encoder_transformer.layers.",
            "encoder_transformer.transformer.layers.",
        );
    }
    if name.starts_with("decoder_transformer.layers.") {
        name = name.replace(
            "decoder_transformer.layers.",
            "decoder_transformer.transformer.layers.",
        );
    }

    // Attention: o_proj -> out_proj
    name = name.replace(".self_attn.o_proj.", ".self_attn.out_proj.");
    // PersonaPlex: in_proj_weight (flat) -> in_proj.weight (nested VarBuilder)
    name = name.replace(".self_attn.in_proj_weight", ".self_attn.in_proj.weight");

    // PersonaPlex quantizer naming: rvq_first -> semantic_residual_vector_quantizer,
    // rvq_rest -> acoustic_residual_vector_quantizer
    name = name.replace(
        "quantizer.rvq_first",
        "quantizer.semantic_residual_vector_quantizer",
    );
    name = name.replace(
        "quantizer.rvq_rest",
        "quantizer.acoustic_residual_vector_quantizer",
    );

    // Norms
    name = name.replace(".input_layernorm.", ".norm1.");
    name = name.replace(".post_attention_layernorm.", ".norm2.");

    // FFN
    name = name.replace(".mlp.fc1.", ".linear1.");
    name = name.replace(".mlp.fc2.", ".linear2.");

    // Layer scales
    name = name.replace(".self_attn_layer_scale.", ".layer_scale_1.");
    name = name.replace(".mlp_layer_scale.", ".layer_scale_2.");

    // Downsample/upsample: full Mimi has `downsample.conv.*` while mimi-rs
    // expects `downsample.conv.conv.*`
    if name.starts_with("downsample.conv.") && !name.starts_with("downsample.conv.conv.") {
        name = name.replacen("downsample.conv.", "downsample.conv.conv.", 1);
    }
    if name.starts_with("upsample.convtr.") && !name.starts_with("upsample.convtr.convtr.") {
        name = name.replacen("upsample.convtr.", "upsample.convtr.convtr.", 1);
    }

    // Quantizer internals: vq.layers → layers, _codebook → codebook,
    // embedding_sum → embed_sum
    name = name.replace(".vq.layers.", ".layers.");
    name = name.replace("._codebook.", ".codebook.");
    name = name.replace(".embedding_sum", ".embed_sum");

    // Skip Q/K/V projections — they get combined into in_proj separately
    if name.contains(".self_attn.q_proj.")
        || name.contains(".self_attn.k_proj.")
        || name.contains(".self_attn.v_proj.")
    {
        return None;
    }

    // Skip _initialized sentinel tensors (not needed for inference)
    if name.ends_with("._initialized") {
        return None;
    }

    Some(name)
}

struct QkvWeights {
    q_data: Vec<u8>,
    q_shape: Vec<usize>,
    k_data: Vec<u8>,
    v_data: Vec<u8>,
    dtype: safetensors::Dtype,
}

fn collect_qkv_weights(st: &safetensors::SafeTensors) -> HashMap<String, QkvWeights> {
    let mut qkv_map: HashMap<String, QkvWeights> = HashMap::new();

    for (name, tensor) in st.iter() {
        let (proj_kind, layer_prefix) = if let Some(rest) = name.strip_suffix(".weight") {
            if rest.ends_with(".self_attn.q_proj") {
                ("q", rest.strip_suffix(".self_attn.q_proj").unwrap())
            } else if rest.ends_with(".self_attn.k_proj") {
                ("k", rest.strip_suffix(".self_attn.k_proj").unwrap())
            } else if rest.ends_with(".self_attn.v_proj") {
                ("v", rest.strip_suffix(".self_attn.v_proj").unwrap())
            } else {
                continue;
            }
        } else {
            continue;
        };

        let remapped_prefix = if layer_prefix.starts_with("encoder_transformer.layers.") {
            layer_prefix.replace(
                "encoder_transformer.layers.",
                "encoder_transformer.transformer.layers.",
            )
        } else if layer_prefix.starts_with("decoder_transformer.layers.") {
            layer_prefix.replace(
                "decoder_transformer.layers.",
                "decoder_transformer.transformer.layers.",
            )
        } else {
            layer_prefix.to_string()
        };

        let entry = qkv_map
            .entry(remapped_prefix)
            .or_insert_with(|| QkvWeights {
                q_data: Vec::new(),
                q_shape: Vec::new(),
                k_data: Vec::new(),
                v_data: Vec::new(),
                dtype: tensor.dtype(),
            });

        match proj_kind {
            "q" => {
                entry.q_data = tensor.data().to_vec();
                entry.q_shape = tensor.shape().to_vec();
            }
            "k" => {
                entry.k_data = tensor.data().to_vec();
            }
            "v" => {
                entry.v_data = tensor.data().to_vec();
            }
            _ => unreachable!(),
        }
    }

    qkv_map
}

struct OwnedView {
    data: Vec<u8>,
    shape: Vec<usize>,
    dtype: safetensors::Dtype,
}

/// Remap full-Mimi safetensors to mimi-rs naming convention.
pub fn remap_mimi_weights(buffer: &[u8]) -> Vec<u8> {
    let st = match safetensors::SafeTensors::deserialize(buffer) {
        Ok(st) => st,
        Err(_) => return buffer.to_vec(),
    };

    let mut views: Vec<(String, OwnedView)> = Vec::new();
    let qkv_weights = collect_qkv_weights(&st);

    for (name, tensor) in st.iter() {
        let remapped = match remap_key(name) {
            Some(r) => r,
            None => continue,
        };

        views.push((
            remapped,
            OwnedView {
                data: tensor.data().to_vec(),
                shape: tensor.shape().to_vec(),
                dtype: tensor.dtype(),
            },
        ));
    }

    for (prefix, qkv) in &qkv_weights {
        let mut combined_data =
            Vec::with_capacity(qkv.q_data.len() + qkv.k_data.len() + qkv.v_data.len());
        combined_data.extend_from_slice(&qkv.q_data);
        combined_data.extend_from_slice(&qkv.k_data);
        combined_data.extend_from_slice(&qkv.v_data);

        let out_dim = qkv.q_shape[0];
        let in_dim = qkv.q_shape[1];

        views.push((
            format!("{prefix}.self_attn.in_proj.weight"),
            OwnedView {
                data: combined_data,
                shape: vec![3 * out_dim, in_dim],
                dtype: qkv.dtype,
            },
        ));
    }

    let view_refs: Vec<(&str, safetensors::tensor::TensorView<'_>)> = views
        .iter()
        .map(|(name, v)| {
            (
                name.as_str(),
                safetensors::tensor::TensorView::new(v.dtype, v.shape.clone(), &v.data)
                    .expect("invalid tensor view"),
            )
        })
        .collect();

    safetensors::tensor::serialize(view_refs, &None).expect("serialization failed")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remap_key_seanet() {
        assert_eq!(
            remap_key("encoder.layers.0.conv.weight"),
            Some("encoder.model.0.conv.weight".to_string())
        );
        assert_eq!(
            remap_key("decoder.layers.0.conv.weight"),
            Some("decoder.model.0.conv.weight".to_string())
        );
    }

    #[test]
    fn test_remap_key_transformer() {
        assert_eq!(
            remap_key("encoder_transformer.layers.0.norm1.weight"),
            Some("encoder_transformer.transformer.layers.0.norm1.weight".to_string())
        );
        assert_eq!(
            remap_key("decoder_transformer.layers.0.norm1.weight"),
            Some("decoder_transformer.transformer.layers.0.norm1.weight".to_string())
        );
    }

    #[test]
    fn test_remap_key_attention() {
        // o_proj -> out_proj
        assert_eq!(
            remap_key("encoder_transformer.layers.0.self_attn.o_proj.weight"),
            Some(
                "encoder_transformer.transformer.layers.0.self_attn.out_proj.weight".to_string()
            )
        );
        // Q/K/V should be skipped
        assert_eq!(
            remap_key("encoder_transformer.layers.0.self_attn.q_proj.weight"),
            None
        );
    }

    #[test]
    fn test_remap_key_norms_ffn() {
        assert_eq!(
            remap_key("encoder_transformer.layers.0.input_layernorm.weight"),
            Some("encoder_transformer.transformer.layers.0.norm1.weight".to_string())
        );
        assert_eq!(
            remap_key("encoder_transformer.layers.0.post_attention_layernorm.weight"),
            Some("encoder_transformer.transformer.layers.0.norm2.weight".to_string())
        );
        assert_eq!(
            remap_key("encoder_transformer.layers.0.mlp.fc1.weight"),
            Some("encoder_transformer.transformer.layers.0.linear1.weight".to_string())
        );
    }

    #[test]
    fn test_remap_key_layer_scales() {
        assert_eq!(
            remap_key("encoder_transformer.layers.0.self_attn_layer_scale.scale"),
            Some("encoder_transformer.transformer.layers.0.layer_scale_1.scale".to_string())
        );
        assert_eq!(
            remap_key("encoder_transformer.layers.0.mlp_layer_scale.scale"),
            Some("encoder_transformer.transformer.layers.0.layer_scale_2.scale".to_string())
        );
    }

    #[test]
    fn test_remap_key_downsample_upsample() {
        assert_eq!(
            remap_key("downsample.conv.weight"),
            Some("downsample.conv.conv.weight".to_string())
        );
        assert_eq!(
            remap_key("upsample.convtr.weight"),
            Some("upsample.convtr.convtr.weight".to_string())
        );
        // Already correct format should not double-prefix
        assert_eq!(
            remap_key("downsample.conv.conv.weight"),
            Some("downsample.conv.conv.weight".to_string())
        );
    }

    #[test]
    fn test_codec_api_shape() {
        // Verify MimiCodec struct has the right fields without loading weights
        // (actual loading requires safetensors file)
        let cfg = mimi_rs::config::MimiConfig::mimi_v1_0_0();
        assert_eq!(cfg.num_codebooks, 32);
        assert_eq!(cfg.sample_rate, 24000);
        assert_eq!(cfg.codebook_bins, 2048);
    }
}
