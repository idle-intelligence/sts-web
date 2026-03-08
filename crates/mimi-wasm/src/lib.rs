//! Standalone Mimi audio codec WASM module.
//!
//! Wraps mimi-rs as a self-contained WASM module intended to run in its own
//! Web Worker, separate from the main STS inference engine. This allows CPU
//! codec work (encode/decode) to overlap with GPU transformer inference.

use std::collections::HashMap;
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// WASM entry points
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct MimiEngine {
    model: mimi_rs::mimi::MimiModel,
    encoder_state: mimi_rs::mimi::MimiEncoderState,
    decoder_state: mimi_rs::mimi::MimiState,
    num_codebooks: usize,
    sample_rate: usize,
}

#[wasm_bindgen]
impl MimiEngine {
    /// Load Mimi model from safetensors bytes.
    ///
    /// `data` — raw bytes of the Mimi safetensors file.
    /// `num_codebooks` — how many codebooks to use (typically 8 for STS, 32 for full Mimi).
    #[wasm_bindgen]
    pub fn load(data: &[u8], num_codebooks: usize) -> Result<MimiEngine, JsError> {
        console_error_panic_hook::set_once();

        let remapped = remap_mimi_weights(data);
        let vb = candle_nn::VarBuilder::from_buffered_safetensors(
            remapped,
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .map_err(|e| JsError::new(&format!("Failed to create VarBuilder: {e}")))?;

        let cfg = mimi_rs::config::MimiConfig::mimi_v1_0_0();
        let model = mimi_rs::mimi::MimiModel::load(vb, &cfg)
            .map_err(|e| JsError::new(&format!("Failed to load Mimi model: {e}")))?;

        let encoder_state = model
            .init_encoder_state(1, &candle_core::Device::Cpu)
            .map_err(|e| JsError::new(&format!("Failed to init encoder state: {e}")))?;

        let decoder_state = model
            .init_state(1, &candle_core::Device::Cpu)
            .map_err(|e| JsError::new(&format!("Failed to init decoder state: {e}")))?;

        let actual_codebooks = if num_codebooks > 0 {
            num_codebooks
        } else {
            cfg.num_codebooks
        };

        Ok(MimiEngine {
            model,
            encoder_state,
            decoder_state,
            num_codebooks: actual_codebooks,
            sample_rate: cfg.sample_rate,
        })
    }

    /// Number of codebooks this engine was configured with.
    #[wasm_bindgen(getter)]
    pub fn num_codebooks(&self) -> usize {
        self.num_codebooks
    }

    /// Sample rate (24000 Hz for standard Mimi).
    #[wasm_bindgen(getter)]
    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// Decode audio tokens to PCM samples.
    ///
    /// `tokens` — flat array of u32, `num_codebooks` tokens per frame.
    ///            For multiple frames: [f0_cb0, f0_cb1, ..., f1_cb0, f1_cb1, ...].
    /// `num_codebooks` — number of codebooks per frame (determines how tokens are split).
    ///
    /// Returns PCM f32 samples (mono, 24kHz).
    #[wasm_bindgen]
    pub fn decode(&mut self, tokens: &[u32], num_codebooks: usize) -> Vec<f32> {
        if tokens.is_empty() || num_codebooks == 0 {
            return Vec::new();
        }

        let n_cb = num_codebooks;
        let n_frames = tokens.len() / n_cb;
        let mut all_pcm = Vec::new();

        for f in 0..n_frames {
            let frame_tokens = &tokens[f * n_cb..(f + 1) * n_cb];
            let pcm = self.decode_frame(frame_tokens, n_cb);
            all_pcm.extend_from_slice(&pcm);
        }

        all_pcm
    }

    /// Encode PCM samples to audio tokens.
    ///
    /// `pcm` — mono f32 samples at 24kHz.
    ///
    /// Returns flat u32 array in frame-major order:
    /// [f0_cb0, f0_cb1, ..., f0_cb31, f1_cb0, ...].
    /// May return empty if not enough audio has accumulated for a frame.
    #[wasm_bindgen]
    pub fn encode(&mut self, pcm: &[f32]) -> Vec<u32> {
        if pcm.is_empty() {
            return Vec::new();
        }

        let tensor = match candle_core::Tensor::from_vec(
            pcm.to_vec(),
            (1, 1, pcm.len()),
            &candle_core::Device::Cpu,
        ) {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };

        let latent = match self.model.encode_streaming(&tensor, &mut self.encoder_state) {
            Ok(l) => l,
            Err(_) => return Vec::new(),
        };

        let n_frames = latent.dim(2).unwrap_or(0);
        if n_frames == 0 {
            return Vec::new();
        }

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

    /// Reset both encoder and decoder streaming state.
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        if let Ok(new_enc) = self
            .model
            .init_encoder_state(1, &candle_core::Device::Cpu)
        {
            self.encoder_state = new_enc;
        }
        if let Ok(new_dec) = self.model.init_state(1, &candle_core::Device::Cpu) {
            self.decoder_state = new_dec;
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

impl MimiEngine {
    /// Decode a single frame of tokens.
    fn decode_frame(&mut self, tokens: &[u32], n_active: usize) -> Vec<f32> {
        let n_tokens = tokens.len();

        let codes = match candle_core::Tensor::from_vec(
            tokens.to_vec(),
            (1, n_tokens, 1),
            &candle_core::Device::Cpu,
        ) {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };

        let audio = match self.model.decode_from_codes_n(
            &codes,
            n_active.min(n_tokens),
            &mut self.decoder_state,
        ) {
            Ok(a) => a,
            Err(_) => return Vec::new(),
        };

        match audio.flatten_all() {
            Ok(flat) => flat.to_vec1::<f32>().unwrap_or_default(),
            Err(_) => Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Key remapping: full-Mimi safetensors -> mimi-rs naming
// ---------------------------------------------------------------------------
// Ported from crates/sts-wasm/src/mimi.rs. Handles:
// 1. SEANet: encoder.layers.{n} -> encoder.model.{n}
// 2. Transformer: separate Q/K/V -> combined in_proj
// 3. Norms, FFN, layer scales renamed
// 4. Downsample/upsample extra .conv. prefix

fn remap_key(name: &str) -> Option<String> {
    let mut name = name.to_string();

    // Strip extra nesting: .conv.conv. -> .conv. and .convtr.convtr. -> .convtr.
    name = name.replacen(".conv.conv.", ".conv.", 1);
    name = name.replacen(".convtr.convtr.", ".convtr.", 1);

    // SEANet layers
    name = name.replace("encoder.layers.", "encoder.model.");
    name = name.replace("decoder.layers.", "decoder.model.");

    // ProjectedTransformer wrapper
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
    // in_proj_weight (flat) -> in_proj.weight (nested)
    name = name.replace(".self_attn.in_proj_weight", ".self_attn.in_proj.weight");

    // Quantizer naming
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

    // Downsample/upsample
    if name.starts_with("downsample.conv.") && !name.starts_with("downsample.conv.conv.") {
        name = name.replacen("downsample.conv.", "downsample.conv.conv.", 1);
    }
    if name.starts_with("upsample.convtr.") && !name.starts_with("upsample.convtr.convtr.") {
        name = name.replacen("upsample.convtr.", "upsample.convtr.convtr.", 1);
    }

    // Quantizer internals
    name = name.replace(".vq.layers.", ".layers.");
    name = name.replace("._codebook.", ".codebook.");
    name = name.replace(".embedding_sum", ".embed_sum");

    // Skip Q/K/V projections — combined into in_proj separately
    if name.contains(".self_attn.q_proj.")
        || name.contains(".self_attn.k_proj.")
        || name.contains(".self_attn.v_proj.")
    {
        return None;
    }

    // Skip _initialized sentinel tensors
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
fn remap_mimi_weights(buffer: &[u8]) -> Vec<u8> {
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
