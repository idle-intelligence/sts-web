//! Native loader for PersonaPlex `.pt` voice preset files.
//!
//! Each `.pt` archive holds two tensors:
//!   - `embeddings`: shape `[num_frames, 1, 1, hidden_dim]`, bf16
//!   - `cache`:      shape `[1, num_streams, num_positions]`, int64
//!
//! The temporal transformer consumes the embeddings frame-by-frame to build
//! KV-cache state, and the cache snapshot replaces the most-recent positions
//! of the streaming token cache (see `StsStream::prefill_voice_preset`).
//!
//! Equivalent of `scripts/convert_voices.py`, kept native so the CLI does not
//! require a Python preprocessing step.

use anyhow::{anyhow, Context, Result};
use candle_core::{pickle, DType, Tensor};
use std::path::Path;

/// Voice preset decoded from a `.pt` archive.
pub struct VoicePreset {
    /// Row-major `[num_frames, dim]` f32 embeddings, ready for
    /// `StsStream::prefill_voice_preset`.
    pub embeddings: Vec<f32>,
    pub num_frames: usize,
    pub dim: usize,
    /// `[num_streams][num_positions]` token snapshot.
    pub cache: Vec<Vec<i32>>,
}

impl VoicePreset {
    /// Read a voice preset from a `.pt` file.
    pub fn from_pt_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let tensors = pickle::read_all(path)
            .with_context(|| format!("failed to parse pickle archive {}", path.display()))?;

        let mut embeddings_t: Option<Tensor> = None;
        let mut cache_t: Option<Tensor> = None;
        for (name, t) in tensors {
            match name.as_str() {
                "embeddings" => embeddings_t = Some(t),
                "cache" => cache_t = Some(t),
                _ => {}
            }
        }

        let embeddings_t = embeddings_t
            .ok_or_else(|| anyhow!("voice preset {} missing 'embeddings' tensor", path.display()))?;
        let cache_t = cache_t
            .ok_or_else(|| anyhow!("voice preset {} missing 'cache' tensor", path.display()))?;

        // Embeddings: [num_frames, 1, 1, dim] (bf16) → [num_frames, dim] f32 row-major.
        let dims = embeddings_t.dims().to_vec();
        if dims.len() != 4 || dims[1] != 1 || dims[2] != 1 {
            return Err(anyhow!(
                "embeddings shape {:?} unexpected (need [N, 1, 1, D])",
                dims
            ));
        }
        let num_frames = dims[0];
        let dim = dims[3];
        let embeddings_f32 = embeddings_t
            .to_dtype(DType::F32)
            .context("convert embeddings to f32")?
            .reshape((num_frames, dim))
            .context("reshape embeddings")?
            .contiguous()
            .context("make embeddings contiguous")?;
        let embeddings: Vec<f32> = embeddings_f32
            .flatten_all()
            .context("flatten embeddings")?
            .to_vec1::<f32>()
            .context("read embeddings as f32 vec")?;

        // Cache: [1, num_streams, num_positions] int64 → Vec<Vec<i32>>.
        let cache_dims = cache_t.dims().to_vec();
        if cache_dims.len() != 3 || cache_dims[0] != 1 {
            return Err(anyhow!(
                "cache shape {:?} unexpected (need [1, S, P])",
                cache_dims
            ));
        }
        let num_streams = cache_dims[1];
        let num_positions = cache_dims[2];
        let cache_squeezed = cache_t
            .squeeze(0)
            .context("squeeze cache leading dim")?
            .contiguous()
            .context("make cache contiguous")?;
        let cache_i64: Vec<i64> = cache_squeezed
            .flatten_all()
            .context("flatten cache")?
            .to_vec1::<i64>()
            .context("read cache as i64 vec")?;

        let mut cache: Vec<Vec<i32>> = Vec::with_capacity(num_streams);
        for s in 0..num_streams {
            let row = &cache_i64[s * num_positions..(s + 1) * num_positions];
            cache.push(row.iter().map(|&v| v as i32).collect());
        }

        Ok(Self {
            embeddings,
            num_frames,
            dim,
            cache,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn ref_dir() -> PathBuf {
        // Tests run with crate root as cwd.
        PathBuf::from("../../tests/reference")
    }

    fn voice_pt_path() -> Option<PathBuf> {
        // The .pt fixtures ship with the model (not the repo). Try the well-known
        // local mirror; skip the test if unavailable.
        let p = PathBuf::from("/data/Code/claude/hf/personaplex-24L-q4_k-webgpu/voices/NATF2.pt");
        if p.exists() {
            Some(p)
        } else {
            None
        }
    }

    #[test]
    fn natf2_pt_matches_reference_fixtures() {
        let Some(pt_path) = voice_pt_path() else {
            eprintln!("Skipping: NATF2.pt not available locally");
            return;
        };

        let preset = VoicePreset::from_pt_file(&pt_path).expect("parse NATF2.pt");

        // -- Cross-check embeddings against tests/reference/NATF2_embeddings.bin --
        let emb_ref_bytes = std::fs::read(ref_dir().join("NATF2_embeddings.bin"))
            .expect("read NATF2_embeddings.bin");
        assert_eq!(
            emb_ref_bytes.len(),
            preset.embeddings.len() * 4,
            "embedding byte count mismatch"
        );
        let emb_ref: Vec<f32> = emb_ref_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // bf16 → f32 conversion is exact for values stored as bf16 originally,
        // so the parser output must match the reference bit-for-bit (or extremely
        // close — allow a tiny tolerance to be safe).
        let mut max_diff = 0.0f32;
        for (a, b) in preset.embeddings.iter().zip(emb_ref.iter()) {
            let d = (a - b).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
        assert!(
            max_diff <= 1e-6,
            "embeddings differ from reference: max_diff={max_diff}"
        );

        // -- Cross-check cache snapshot --
        let cache_ref_str = std::fs::read_to_string(ref_dir().join("NATF2_cache.json"))
            .expect("read NATF2_cache.json");
        let cache_ref_json: serde_json::Value =
            serde_json::from_str(&cache_ref_str).expect("parse cache json");
        let ref_num_frames = cache_ref_json["num_frames"].as_u64().unwrap() as usize;
        let ref_cache: Vec<Vec<i32>> = cache_ref_json["cache"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| {
                s.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_i64().unwrap() as i32)
                    .collect()
            })
            .collect();

        assert_eq!(preset.num_frames, ref_num_frames);
        assert_eq!(preset.cache.len(), ref_cache.len());
        for (i, (got, want)) in preset.cache.iter().zip(ref_cache.iter()).enumerate() {
            assert_eq!(got, want, "stream {i} cache mismatch");
        }
    }
}
