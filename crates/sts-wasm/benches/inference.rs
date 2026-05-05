//! Criterion benchmarks for the STS inference pipeline.
//!
//! Measures frame generation (single step, batch of 10) and prefill time.
//! Requires model weights on disk — skips gracefully if not found.

use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};

use burn::backend::wgpu::WgpuDevice;
use sts_wasm::depth::DepthTransformer;
use sts_wasm::gguf::Q4ModelLoader;
use sts_wasm::loader::load_sts_model_deferred;
use sts_wasm::mimi::MimiCodec;
use sts_wasm::model::TemporalTransformer;
use sts_wasm::stream::StsStream;
use sts_wasm::StsConfig;

fn workspace_root() -> PathBuf {
    let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.pop(); // out of crates/sts-wasm
    d.pop(); // out of crates
    d
}

fn model_dir() -> Option<PathBuf> {
    let p = workspace_root().join("assets/model");
    if p.is_dir() {
        Some(p)
    } else {
        eprintln!(
            "Benchmark skipped: {} is not a directory (symlink it to a personaplex model checkout)",
            p.display()
        );
        None
    }
}

fn num_layers() -> usize {
    let cfg_path = workspace_root().join("assets/model/config.json");
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

fn audio_wav() -> Option<PathBuf> {
    let p = workspace_root().join("assets/audio/joke.wav");
    if p.is_file() {
        Some(p)
    } else {
        eprintln!(
            "Benchmark skipped: {} is not a file (symlink it to a mono test WAV)",
            p.display()
        );
        None
    }
}

fn find_mimi_weights(model_dir: &std::path::Path) -> Option<PathBuf> {
    fs::read_dir(model_dir).ok()?.flatten().find_map(|e| {
        let n = e.file_name().to_string_lossy().to_string();
        (n.starts_with("tokenizer-") && n.ends_with(".safetensors")).then(|| e.path())
    })
}

fn load_shards(model_dir: &std::path::Path) -> Vec<Vec<u8>> {
    for dir in [model_dir.join("shards"), model_dir.to_path_buf()] {
        if !dir.is_dir() {
            continue;
        }
        let mut paths: Vec<PathBuf> = fs::read_dir(&dir)
            .unwrap()
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
            return paths.iter().map(|p| fs::read(p).unwrap()).collect();
        }
    }
    panic!("no GGUF shards found in {}", model_dir.display())
}

fn load_model() -> Option<(TemporalTransformer, DepthTransformer, StsConfig)> {
    let dir = model_dir()?;

    let device = WgpuDevice::default();
    let config = StsConfig {
        num_layers: num_layers(),
        ..StsConfig::default()
    };

    let shards = load_shards(&dir);
    let mut loader = Q4ModelLoader::from_shards(shards).unwrap();
    let parts = load_sts_model_deferred(&mut loader, &config, &device).unwrap();
    drop(loader);
    let (temporal, depth) = parts.finalize(&device).unwrap();

    Some((temporal, depth, config))
}

fn encode_user_audio(config: &StsConfig) -> Option<Vec<Vec<u32>>> {
    let wav_path = audio_wav()?;
    let dir = model_dir()?;
    let mimi_path = find_mimi_weights(&dir)?;

    let mimi_data = fs::read(&mimi_path).unwrap();
    let mut mimi = MimiCodec::from_bytes(&mimi_data).unwrap();
    mimi.reset();

    let reader = hound::WavReader::open(&wav_path).unwrap();
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.unwrap() as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => reader.into_samples::<f32>().map(|s| s.unwrap()).collect(),
    };

    let frame_size = 1920;
    let mut frames: Vec<Vec<u32>> = Vec::new();
    for chunk in samples.chunks(frame_size) {
        let mut frame = chunk.to_vec();
        if frame.len() < frame_size {
            frame.resize(frame_size, 0.0);
        }
        let tokens = mimi.encode(&frame);
        if tokens.is_empty() {
            continue;
        }
        let n = mimi.num_codebooks().min(config.num_codebooks);
        frames.push(tokens[..n].to_vec());
    }
    Some(frames)
}

/// Create a stream with prefill done + warmup frames generated.
fn make_warm_stream(
    temporal: &TemporalTransformer,
    depth: &DepthTransformer,
    config: &StsConfig,
    user_audio_frames: Option<&[Vec<u32>]>,
    warmup_frames: usize,
) -> StsStream {
    let temporal_cache = temporal.create_cache();
    let depth_cache = depth.create_cache();
    let mut stream = StsStream::new(config.clone(), temporal_cache, depth_cache);
    stream.set_sampling_params(0.0, 1, 0.0, 1);

    stream.prefill(temporal);

    if let Some(frames) = user_audio_frames {
        pollster::block_on(stream.prefill_user_audio(frames, temporal, depth));
    }

    for _ in 0..warmup_frames {
        pollster::block_on(stream.step(temporal, depth));
    }

    stream
}

fn bench_frame_generation(c: &mut Criterion) {
    let Some((temporal, depth, config)) = load_model() else {
        eprintln!("Skipping frame_generation benchmark: model not loaded");
        return;
    };

    let user_audio = encode_user_audio(&config);
    let mut stream = make_warm_stream(
        &temporal,
        &depth,
        &config,
        user_audio.as_deref(),
        5,
    );

    let mut group = c.benchmark_group("frame_generation");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_secs(30));

    group.bench_function("single_step", |b| {
        b.iter(|| {
            let out = pollster::block_on(stream.step(&temporal, &depth));
            std::hint::black_box(&out);
        });
    });

    group.finish();
}

fn bench_frame_generation_batch(c: &mut Criterion) {
    let Some((temporal, depth, config)) = load_model() else {
        eprintln!("Skipping frame_generation_batch benchmark: model not loaded");
        return;
    };

    let user_audio = encode_user_audio(&config);
    let mut stream = make_warm_stream(
        &temporal,
        &depth,
        &config,
        user_audio.as_deref(),
        5,
    );

    let mut group = c.benchmark_group("frame_generation_batch");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_secs(60));

    group.bench_function("10_consecutive_steps", |b| {
        b.iter(|| {
            for _ in 0..10 {
                let out = pollster::block_on(stream.step(&temporal, &depth));
                std::hint::black_box(&out);
            }
        });
    });

    group.finish();
}

fn bench_prefill(c: &mut Criterion) {
    let Some((temporal, depth, config)) = load_model() else {
        eprintln!("Skipping prefill benchmark: model not loaded");
        return;
    };

    let user_audio = encode_user_audio(&config);

    let mut group = c.benchmark_group("prefill");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_secs(60));

    group.bench_function("system_prompt_and_user_audio", |b| {
        b.iter(|| {
            let temporal_cache = temporal.create_cache();
            let depth_cache = depth.create_cache();
            let mut stream = StsStream::new(config.clone(), temporal_cache, depth_cache);
            stream.set_sampling_params(0.0, 1, 0.0, 1);

            stream.prefill(&temporal);

            if let Some(ref frames) = user_audio {
                pollster::block_on(stream.prefill_user_audio(frames, &temporal, &depth));
            }

            std::hint::black_box(stream.frame_count())
        });
    });

    group.finish();
}

criterion_group!(benches, bench_frame_generation, bench_frame_generation_batch, bench_prefill);
criterion_main!(benches);
