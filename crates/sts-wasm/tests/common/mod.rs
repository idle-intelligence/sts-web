//! Shared test fixtures and resource resolution.
//!
//! Tests in this crate resolve external resources via the workspace
//! `assets/` directory (gitignored). Symlink local data into it:
//!
//! ```bash
//! mkdir -p assets/audio
//! ln -s /path/to/personaplex-XX-q4_k-webgpu  assets/model
//! ln -s /path/to/your.wav                    assets/audio/joke.wav
//! ```
//!
//! Tests gracefully skip with an `eprintln!` notice when a symlink is
//! missing, so the suite stays runnable on a fresh checkout (or on CI
//! without GPU resources).
//!
//! Reference fixtures (`tests/reference/NATF2_*`) ship with the repo and
//! resolve repo-relative — no setup needed.
//!
//! Test outputs (debug logs, generated WAVs) go to `target/test-output/`
//! (gitignored via Cargo's `target/`).

#![allow(dead_code)] // Helpers are used by some test files but not all.

use std::fs;
use std::path::{Path, PathBuf};

/// Crate-relative manifest dir (`crates/sts-wasm`).
pub fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Workspace root.
pub fn workspace_root() -> PathBuf {
    let mut d = manifest_dir();
    d.pop(); // out of `crates/sts-wasm`
    d.pop(); // out of `crates`
    d
}

/// `<workspace>/assets` — the place to symlink local model + audio resources.
pub fn assets_dir() -> PathBuf {
    workspace_root().join("assets")
}

/// `<workspace>/tests/reference` — repo-checked-in reference fixtures.
pub fn reference_dir() -> PathBuf {
    workspace_root().join("tests/reference")
}

/// `<workspace>/target/test-output` — auto-created scratch dir for test outputs.
pub fn test_output_dir() -> PathBuf {
    let dir = workspace_root().join("target/test-output");
    let _ = fs::create_dir_all(&dir);
    dir
}

/// Resolve `assets/model` to a directory. Returns `None` (with a skip
/// notice on stderr) if the symlink is missing or broken.
pub fn model_dir() -> Option<PathBuf> {
    let p = assets_dir().join("model");
    if p.is_dir() {
        Some(p)
    } else {
        eprintln!(
            "Skipping: {} is not a directory (symlink it to a personaplex model checkout)",
            p.display()
        );
        None
    }
}

/// Resolve `assets/audio/joke.wav` to a file. Returns `None` (with a skip
/// notice on stderr) if the symlink is missing or broken.
pub fn audio_wav() -> Option<PathBuf> {
    let p = assets_dir().join("audio/joke.wav");
    if p.is_file() {
        Some(p)
    } else {
        eprintln!(
            "Skipping: {} is not a file (symlink it to a mono test WAV)",
            p.display()
        );
        None
    }
}

/// Number of temporal-transformer layers in the configured test model.
///
/// Reads `assets/model/config.json` if present (the 24L pruned checkpoint
/// ships one); otherwise falls back to 32 (the original 7b-v1 layout).
pub fn num_layers() -> usize {
    let cfg_path = assets_dir().join("model/config.json");
    if let Ok(s) = fs::read_to_string(&cfg_path) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
            if let Some(n) = v
                .get("architecture")
                .and_then(|a| a.get("temporal_transformer"))
                .and_then(|t| t.get("layers"))
                .and_then(|l| l.as_u64())
            {
                return n as usize;
            }
        }
    }
    32
}

/// Read all GGUF shards from a model directory.
///
/// Supports both layouts in use:
///   * 7b-v1 — shards directly in `model_dir/`
///   * 24L   — shards under `model_dir/shards/`
///
/// Panics on read errors (the dir is expected to exist; missing-resource
/// gating belongs upstream of this call via [`model_dir`]).
pub fn load_shards(model_dir: &Path) -> Vec<Vec<u8>> {
    let candidates = [model_dir.join("shards"), model_dir.to_path_buf()];
    for dir in &candidates {
        if !dir.is_dir() {
            continue;
        }
        let mut paths: Vec<PathBuf> = fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.contains("shard-"))
                    .unwrap_or(false)
            })
            .collect();
        if !paths.is_empty() {
            paths.sort();
            return paths
                .iter()
                .map(|p| fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display())))
                .collect();
        }
    }
    panic!(
        "no GGUF shards found in {} (tried {0} and {0}/shards)",
        model_dir.display()
    );
}

/// Find the Mimi safetensors file (`tokenizer-*.safetensors`) inside a
/// model directory. Returns the first match.
pub fn find_mimi_weights(model_dir: &Path) -> Option<PathBuf> {
    let entries = fs::read_dir(model_dir).ok()?;
    for e in entries.flatten() {
        let n = e.file_name().to_string_lossy().to_string();
        if n.starts_with("tokenizer-") && n.ends_with(".safetensors") {
            return Some(e.path());
        }
    }
    None
}

/// Read a mono WAV file as `Vec<f32>`. Supports int and float sample formats;
/// returned samples are normalised to `[-1, 1]`.
pub fn read_wav_mono_f32(path: &Path) -> (Vec<f32>, hound::WavSpec) {
    let reader = hound::WavReader::open(path)
        .unwrap_or_else(|e| panic!("open WAV {}: {e}", path.display()));
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.unwrap() as f32 / max)
                .collect()
        }
        hound::SampleFormat::Float => reader.into_samples::<f32>().map(|s| s.unwrap()).collect(),
    };
    (samples, spec)
}
