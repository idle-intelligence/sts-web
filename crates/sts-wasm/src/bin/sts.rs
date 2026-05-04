//! `sts` — native CLI for running PersonaPlex 24L Q4_K speech-to-speech.
//!
//! Loads the model from a local model directory (`shards/`, voice presets,
//! Mimi codec, SentencePiece tokenizer) and runs a single user-audio →
//! agent-audio turn end-to-end. Designed as the canonical "how do I run
//! this model?" entry point referenced from the HuggingFace model card.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use burn::backend::wgpu::WgpuDevice;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};

use sts_wasm::gguf::Q4ModelLoader;
use sts_wasm::loader::load_sts_model_deferred;
use sts_wasm::mimi::MimiCodec;
use sts_wasm::stream::StsStream;
use sts_wasm::tokenizer::SpmDecoder;
use sts_wasm::voice::VoicePreset;
use sts_wasm::StsConfig;

#[derive(Parser, Debug)]
#[command(
    name = "sts",
    version,
    about = "Run PersonaPlex-24L (Q4_K) speech-to-speech locally on a WebGPU/Vulkan GPU.",
    long_about = "Loads a PersonaPlex-24L model directory (shards/, voices/, Mimi safetensors, \
                  SentencePiece .model) and produces a response WAV from an input WAV."
)]
struct Args {
    /// Path to the model directory (the `personaplex-24L-q4_k-webgpu` checkout).
    #[arg(long, value_name = "DIR")]
    model_dir: PathBuf,

    /// Input user-audio WAV (mono, will be resampled to 24 kHz).
    #[arg(long, value_name = "WAV")]
    input: PathBuf,

    /// Output WAV (24 kHz mono, 16-bit PCM).
    #[arg(long, value_name = "WAV")]
    output: PathBuf,

    /// Voice preset name. Resolved to `<model-dir>/voices/<NAME>.pt`.
    #[arg(long, default_value = "NATF2")]
    voice: String,

    /// Number of temporal-transformer layers in the checkpoint (24 for the 24L pruned model).
    #[arg(long, default_value_t = 24)]
    num_layers: usize,

    /// Text-token sampling temperature (0 = greedy).
    #[arg(long, default_value_t = 0.7)]
    temp_text: f32,

    /// Audio-token sampling temperature.
    #[arg(long, default_value_t = 0.8)]
    temp_audio: f32,

    /// Maximum number of generated frames (12.5 Hz; 250 ≈ 20 s).
    #[arg(long, default_value_t = 250)]
    max_frames: usize,

    /// Comma-separated list of system-prompt token IDs. If omitted, uses the
    /// default "wise and friendly teacher" prompt baked into the config.
    #[arg(long, value_name = "IDS")]
    system_prompt_tokens: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let t0 = Instant::now();

    // -- Resolve model files --
    let shards_dir = args.model_dir.join("shards");
    if !shards_dir.is_dir() {
        bail!(
            "model directory missing 'shards/': {}",
            args.model_dir.display()
        );
    }
    let voice_path = args.model_dir.join("voices").join(format!("{}.pt", args.voice));
    if !voice_path.is_file() {
        bail!("voice preset not found: {}", voice_path.display());
    }
    let mimi_path = find_unique_file(&args.model_dir, "tokenizer-", ".safetensors")
        .context("locate Mimi codec safetensors")?;
    let spm_path = find_unique_file(&args.model_dir, "tokenizer_spm_", ".model")
        .context("locate SentencePiece tokenizer model")?;

    if !args.input.is_file() {
        bail!("input wav not found: {}", args.input.display());
    }

    println!("sts: model_dir={}", args.model_dir.display());
    println!("     voice={}  layers={}", args.voice, args.num_layers);

    // -- Build config --
    let mut config = StsConfig {
        num_layers: args.num_layers,
        ..StsConfig::default()
    };
    if let Some(s) = args.system_prompt_tokens.as_ref() {
        let parsed: Result<Vec<u32>> = s
            .split(',')
            .map(|t| {
                t.trim()
                    .parse::<u32>()
                    .map_err(|e| anyhow!("invalid token id '{t}': {e}"))
            })
            .collect();
        config.system_prompt_tokens = parsed?;
    }

    // -- Load shards --
    let shards = load_shards(&shards_dir).context("load GGUF shards")?;
    let total_bytes: usize = shards.iter().map(|s| s.len()).sum();
    println!(
        "     shards={}  size={:.2} GB",
        shards.len(),
        total_bytes as f64 / 1e9
    );

    // -- Build model --
    let device = WgpuDevice::default();
    let pb = spinner("Parsing GGUF and uploading weights to GPU");
    let mut loader = Q4ModelLoader::from_shards(shards).context("parse GGUF")?;
    let parts = load_sts_model_deferred(&mut loader, &config, &device).context("load STS model")?;
    drop(loader);
    let (temporal, depth) = parts.finalize(&device).context("finalize STS model")?;
    pb.finish_with_message(format!(
        "Model loaded ({} temporal layers, {} depth layers, {:.1}s)",
        config.num_layers,
        config.depth_num_layers,
        t0.elapsed().as_secs_f32()
    ));

    // -- Mimi codec --
    let pb = spinner("Loading Mimi codec");
    let mimi_data = fs::read(&mimi_path).with_context(|| format!("read {}", mimi_path.display()))?;
    let mut mimi = MimiCodec::from_bytes(&mimi_data)
        .map_err(|e| anyhow!("load Mimi codec: {e}"))?;
    pb.finish_with_message(format!(
        "Mimi codec loaded ({} codebooks, {} Hz)",
        mimi.num_codebooks(),
        mimi.sample_rate()
    ));

    // -- SentencePiece (decoder only — used for the inner-monologue dump) --
    let spm_data =
        fs::read(&spm_path).with_context(|| format!("read {}", spm_path.display()))?;
    let spm = SpmDecoder::from_bytes(&spm_data);

    // -- Voice preset --
    let pb = spinner(&format!("Loading voice preset {}", args.voice));
    let preset = VoicePreset::from_pt_file(&voice_path)
        .with_context(|| format!("parse voice preset {}", voice_path.display()))?;
    if preset.dim != config.hidden_size {
        bail!(
            "voice preset dim {} != model hidden_size {}",
            preset.dim,
            config.hidden_size
        );
    }
    pb.finish_with_message(format!(
        "Voice preset {}: {} frames × {}",
        args.voice, preset.num_frames, preset.dim
    ));

    // -- Read input WAV --
    let user_pcm = read_wav_24k_mono(&args.input)
        .with_context(|| format!("read input wav {}", args.input.display()))?;
    println!(
        "     input audio: {:.2}s ({} samples)",
        user_pcm.len() as f64 / 24_000.0,
        user_pcm.len()
    );

    // -- Stream setup --
    let temporal_cache = temporal.create_cache();
    let depth_cache = depth.create_cache();
    let mut stream = StsStream::new(config.clone(), temporal_cache, depth_cache);
    stream.set_sampling_params(args.temp_text, 25, args.temp_audio, 25);

    // Phase 1: voice preset prefill
    let pb = spinner("Voice preset prefill (temporal KV cache)");
    stream.prefill_voice_preset(&preset.embeddings, preset.num_frames, &preset.cache, &temporal);
    pb.finish_with_message("Voice preset prefill done");

    // Phases 2-4: silence + system prompt + silence
    let pb = spinner("System prompt prefill");
    stream.prefill(&temporal);
    pb.finish_with_message("System prompt prefill done");

    // Mimi parity reset before encoding user audio.
    mimi.reset();

    // Phase 5: encode user audio with Mimi → tokens
    let frame_size = 1920; // 80 ms @ 24 kHz
    let mut user_audio_frames: Vec<Vec<u32>> = Vec::new();
    let chunks: Vec<&[f32]> = user_pcm.chunks(frame_size).collect();
    let pb = bar(chunks.len() as u64, "Mimi encode user audio");
    for chunk in chunks {
        let mut frame = chunk.to_vec();
        if frame.len() < frame_size {
            frame.resize(frame_size, 0.0);
        }
        let tokens = mimi.encode(&frame);
        if !tokens.is_empty() {
            let nq = mimi.num_codebooks().min(config.num_codebooks);
            user_audio_frames.push(tokens[..nq].to_vec());
        }
        pb.inc(1);
    }
    pb.finish_with_message(format!("Mimi encode done ({} frames)", user_audio_frames.len()));

    let pb = spinner("User audio prefill (temporal + depth)");
    pollster::block_on(stream.prefill_user_audio(&user_audio_frames, &temporal, &depth));
    pb.finish_with_message(format!(
        "User audio prefill done ({} frames in KV cache)",
        stream.frame_count()
    ));

    // -- Generation --
    let mut response_frames: Vec<Vec<u32>> = Vec::new();
    let mut text_tokens: Vec<u32> = Vec::new();
    let pb = bar(args.max_frames as u64, "Generating response");
    let gen_start = Instant::now();
    for i in 0..args.max_frames {
        let out = pollster::block_on(stream.step(&temporal, &depth));
        response_frames.push(out.model_audio_tokens.clone());
        if out.text_token != config.text_padding_id && out.text_token != 0 {
            text_tokens.push(out.text_token);
        }
        pb.inc(1);
        if stream.should_stop() {
            pb.finish_with_message(format!(
                "Stopped early after {} frames (silence streak {})",
                i + 1,
                stream.consecutive_silence_frames()
            ));
            break;
        }
    }
    if !pb.is_finished() {
        pb.finish_with_message(format!("Reached max_frames={}", args.max_frames));
    }
    let gen_elapsed = gen_start.elapsed();
    println!(
        "     generation: {} frames in {:.2}s ({:.1} ms/frame)",
        response_frames.len(),
        gen_elapsed.as_secs_f32(),
        gen_elapsed.as_millis() as f64 / response_frames.len().max(1) as f64
    );

    if !text_tokens.is_empty() {
        let text = spm.decode(&text_tokens);
        println!("     text: {}", text.trim());
    }

    // -- Decode response audio with Mimi --
    let pb = bar(response_frames.len() as u64, "Mimi decode response");
    let mut output_pcm: Vec<f32> = Vec::with_capacity(response_frames.len() * 1920);
    let n_active = config.num_codebooks;
    for tokens in &response_frames {
        let pcm = mimi.decode_n(tokens, n_active);
        output_pcm.extend_from_slice(&pcm);
        pb.inc(1);
    }
    pb.finish_with_message(format!(
        "Mimi decode done ({} samples, {:.2}s)",
        output_pcm.len(),
        output_pcm.len() as f64 / 24_000.0
    ));

    // -- Write WAV --
    write_wav_24k_mono(&args.output, &output_pcm)
        .with_context(|| format!("write {}", args.output.display()))?;
    let rms = if output_pcm.is_empty() {
        0.0
    } else {
        (output_pcm.iter().map(|s| s * s).sum::<f32>() / output_pcm.len() as f32).sqrt()
    };
    println!(
        "     wrote {} ({:.2}s, RMS={:.4})",
        args.output.display(),
        output_pcm.len() as f64 / 24_000.0,
        rms
    );
    println!("done in {:.2}s", t0.elapsed().as_secs_f32());

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn spinner(msg: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("{spinner} {msg}")
            .unwrap()
            .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ "),
    );
    pb.set_message(msg.to_string());
    pb.enable_steady_tick(std::time::Duration::from_millis(80));
    pb
}

fn bar(len: u64, msg: &str) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::with_template(
            "{msg} [{bar:30.cyan/blue}] {pos}/{len} ({eta})",
        )
        .unwrap()
        .progress_chars("=> "),
    );
    pb.set_message(msg.to_string());
    pb
}

fn find_unique_file(dir: &Path, prefix: &str, suffix: &str) -> Result<PathBuf> {
    let mut hits = Vec::new();
    for e in fs::read_dir(dir).with_context(|| format!("read_dir {}", dir.display()))? {
        let e = e?;
        let name = e.file_name();
        let n = name.to_string_lossy();
        if n.starts_with(prefix) && n.ends_with(suffix) {
            hits.push(e.path());
        }
    }
    match hits.len() {
        0 => bail!(
            "no file matching {prefix}*{suffix} in {}",
            dir.display()
        ),
        1 => Ok(hits.pop().unwrap()),
        n => bail!(
            "found {n} files matching {prefix}*{suffix} in {}; expected exactly 1",
            dir.display()
        ),
    }
}

fn load_shards(dir: &Path) -> Result<Vec<Vec<u8>>> {
    let mut shard_paths: Vec<PathBuf> = fs::read_dir(dir)
        .with_context(|| format!("read_dir {}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.contains(".shard-"))
                .unwrap_or(false)
        })
        .collect();
    if shard_paths.is_empty() {
        bail!("no `*.shard-NN` files in {}", dir.display());
    }
    shard_paths.sort();
    let mut shards = Vec::with_capacity(shard_paths.len());
    for p in shard_paths {
        let bytes = fs::read(&p).with_context(|| format!("read {}", p.display()))?;
        shards.push(bytes);
    }
    Ok(shards)
}

fn read_wav_24k_mono(path: &Path) -> Result<Vec<f32>> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels != 1 {
        bail!(
            "input wav must be mono, got {} channels ({})",
            spec.channels,
            path.display()
        );
    }

    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .collect::<std::result::Result<Vec<i32>, _>>()?
                .into_iter()
                .map(|s| s as f32 / max)
                .collect()
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<std::result::Result<Vec<f32>, _>>()?,
    };

    if spec.sample_rate == 24_000 {
        return Ok(raw);
    }
    Ok(resample_linear(&raw, spec.sample_rate, 24_000))
}

/// Cheap linear resampler for arbitrary sample-rate input WAVs. Audio quality
/// is not critical for the prompt path — Mimi accepts any 24 kHz mono signal.
fn resample_linear(samples: &[f32], in_rate: u32, out_rate: u32) -> Vec<f32> {
    if in_rate == out_rate || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = out_rate as f64 / in_rate as f64;
    let out_len = ((samples.len() as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = i as f64 / ratio;
        let i0 = src.floor() as usize;
        let i1 = (i0 + 1).min(samples.len() - 1);
        let frac = (src - i0 as f64) as f32;
        let s = samples[i0] * (1.0 - frac) + samples[i1] * frac;
        out.push(s);
    }
    out
}

fn write_wav_24k_mono(path: &Path, samples: &[f32]) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            fs::create_dir_all(parent)?;
        }
    }
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        w.write_sample(v)?;
    }
    w.finalize()?;
    Ok(())
}
