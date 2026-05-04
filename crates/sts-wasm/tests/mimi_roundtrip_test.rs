//! Mimi codec roundtrip test: encode audio to tokens, decode back to audio.
//!
//! This isolates the codec from the model. If the roundtrip audio is clean,
//! the garbled-audio bug is in the model (or the token plumbing around it).
//! If the roundtrip audio is also garbled, the codec or weight remapping is
//! the problem.
//!
//! Set `STS_TEST_MODEL_DIR` (model dir holding the Mimi safetensors) and
//! `STS_TEST_AUDIO_WAV` (mono test WAV) to enable.

mod common;

use std::fs;

use sts_wasm::mimi::MimiCodec;

#[test]
fn mimi_roundtrip() {
    let Some(model_dir) = common::model_dir() else {
        return;
    };
    let Some(input_wav) = common::audio_wav() else {
        return;
    };
    let Some(mimi_weights) = common::find_mimi_weights(&model_dir) else {
        eprintln!(
            "Skipping: no `tokenizer-*.safetensors` in {}",
            model_dir.display()
        );
        return;
    };

    let output_dir = common::test_output_dir();
    let output_wav = output_dir.join("joke_roundtrip.wav");

    // ── 1. Load Mimi codec ───────────────────────────────────────────
    println!("Loading Mimi codec from {} ...", mimi_weights.display());
    let mimi_data = fs::read(&mimi_weights).expect("failed to read Mimi weights");
    let mut mimi = MimiCodec::from_bytes(&mimi_data).expect("failed to load Mimi codec");
    println!(
        "  Codec ready: {} codebooks, {} Hz sample rate",
        mimi.num_codebooks(),
        mimi.sample_rate()
    );

    // ── 2. Load test WAV ─────────────────────────────────────────────
    println!("Loading test WAV from {} ...", input_wav.display());
    let (samples, spec) = common::read_wav_mono_f32(&input_wav);
    println!(
        "  WAV spec: {} Hz, {} ch, {} bits, {:?}",
        spec.sample_rate, spec.channels, spec.bits_per_sample, spec.sample_format
    );
    let duration_s = samples.len() as f64 / spec.sample_rate as f64;
    println!("  Loaded {} samples ({:.3}s)", samples.len(), duration_s);

    // ── 3. Streaming encode → immediate decode (frame by frame) ──────
    let frame_size: usize = 1920; // 80 ms at 24 kHz
    let num_codebooks = mimi.num_codebooks();
    let mut all_output: Vec<f32> = Vec::new();
    let mut frames_encoded: usize = 0;
    let mut total_tokens: usize = 0;
    let mut frames_with_no_tokens: usize = 0;
    let mut frames_decoded: usize = 0;

    println!(
        "\nStreaming encode→decode ({} sample frames, {} codebooks) ...",
        frame_size, num_codebooks
    );

    for (i, chunk) in samples.chunks(frame_size).enumerate() {
        // Pad the last chunk if shorter than a full frame
        let frame: Vec<f32> = if chunk.len() < frame_size {
            let mut f = chunk.to_vec();
            f.resize(frame_size, 0.0);
            f
        } else {
            chunk.to_vec()
        };

        // Encode
        let tokens = mimi.encode(&frame);

        if tokens.is_empty() {
            frames_with_no_tokens += 1;
            if i < 5 || i % 50 == 0 {
                println!("  Frame {}: encode returned 0 tokens (buffering)", i);
            }
            continue;
        }

        let n_frames_in_batch = tokens.len() / num_codebooks;
        frames_encoded += n_frames_in_batch;
        total_tokens += tokens.len();

        // Decode each frame in the batch
        for f_idx in 0..n_frames_in_batch {
            let frame_tokens = &tokens[f_idx * num_codebooks..(f_idx + 1) * num_codebooks];
            let pcm = mimi.decode(frame_tokens);
            frames_decoded += 1;

            if i < 5 || i % 50 == 0 {
                println!(
                    "  Frame {} (sub {}): {} tokens → {} PCM samples | first tokens: {:?}",
                    i,
                    f_idx,
                    num_codebooks,
                    pcm.len(),
                    &frame_tokens[..4.min(frame_tokens.len())]
                );
            }
            all_output.extend_from_slice(&pcm);
        }
    }

    // ── 4. Statistics ────────────────────────────────────────────────
    let input_samples = samples.len();
    let output_samples = all_output.len();
    let tokens_per_frame = if frames_encoded > 0 {
        total_tokens as f64 / frames_encoded as f64
    } else {
        0.0
    };

    println!("\n===== ROUNDTRIP STATISTICS =====");
    println!("  Input samples:        {}", input_samples);
    println!("  Output samples:       {}", output_samples);
    println!(
        "  Sample ratio (out/in): {:.4}",
        output_samples as f64 / input_samples.max(1) as f64
    );
    println!("  Frames encoded:       {}", frames_encoded);
    println!("  Frames decoded:       {}", frames_decoded);
    println!("  Frames w/o tokens:    {}", frames_with_no_tokens);
    println!("  Total tokens:         {}", total_tokens);
    println!("  Tokens per frame:     {:.1}", tokens_per_frame);
    println!(
        "  Input duration:       {:.3}s",
        input_samples as f64 / 24000.0
    );
    println!(
        "  Output duration:      {:.3}s",
        output_samples as f64 / 24000.0
    );

    // Amplitude stats
    if !all_output.is_empty() {
        let min = all_output.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = all_output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let rms = (all_output.iter().map(|s| s * s).sum::<f32>() / all_output.len() as f32).sqrt();
        let zeros = all_output.iter().filter(|&&s| s == 0.0).count();
        println!("  Amplitude min:        {:.6}", min);
        println!("  Amplitude max:        {:.6}", max);
        println!("  RMS:                  {:.6}", rms);
        println!(
            "  Zero samples:         {} ({:.1}%)",
            zeros,
            zeros as f64 / all_output.len() as f64 * 100.0
        );
    }
    println!("================================\n");

    // ── 5. Write output WAV (16-bit PCM, 24kHz, mono) ──────────────
    if !all_output.is_empty() {
        let out_spec = hound::WavSpec {
            channels: 1,
            sample_rate: 24000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer =
            hound::WavWriter::create(&output_wav, out_spec).expect("failed to create output WAV");
        for &s in &all_output {
            let val = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
            writer.write_sample(val).unwrap();
        }
        writer.finalize().unwrap();
        println!("Roundtrip WAV written to: {}", output_wav.display());
    } else {
        println!("WARNING: No output audio produced — nothing written.");
    }

    // ── 6. Assertions ───────────────────────────────────────────────
    assert!(
        frames_encoded > 0,
        "Expected at least one frame to be encoded"
    );
    assert!(
        !all_output.is_empty(),
        "Expected non-empty decoded audio output"
    );
    assert_eq!(
        tokens_per_frame as usize, num_codebooks,
        "Expected {} tokens per frame (got {:.1})",
        num_codebooks, tokens_per_frame
    );
}

/// Test that decoding with only 8 out of 32 codebooks produces recognisable
/// (lower-quality but not garbled) audio, compared to a full 32-codebook decode.
#[test]
fn test_mimi_8codebook_roundtrip() {
    let Some(model_dir) = common::model_dir() else {
        return;
    };
    let Some(input_wav) = common::audio_wav() else {
        return;
    };
    let Some(mimi_weights) = common::find_mimi_weights(&model_dir) else {
        eprintln!(
            "Skipping: no `tokenizer-*.safetensors` in {}",
            model_dir.display()
        );
        return;
    };

    let output_dir = common::test_output_dir();
    let output_wav_8cb = output_dir.join("joke_roundtrip_8cb.wav");
    let output_wav_32cb = output_dir.join("joke_roundtrip_32cb.wav");

    // ── 1. Load Mimi codec ───────────────────────────────────────────
    println!("Loading Mimi codec from {} ...", mimi_weights.display());
    let mimi_data = fs::read(&mimi_weights).expect("failed to read Mimi weights");
    let mut mimi = MimiCodec::from_bytes(&mimi_data).expect("failed to load Mimi codec");
    let num_codebooks = mimi.num_codebooks();
    println!(
        "  Codec ready: {} codebooks, {} Hz sample rate",
        num_codebooks,
        mimi.sample_rate()
    );
    assert_eq!(num_codebooks, 32, "Expected 32 codebooks from Mimi");

    // ── 2. Load test WAV ─────────────────────────────────────────────
    println!("Loading test WAV from {} ...", input_wav.display());
    let (samples, spec) = common::read_wav_mono_f32(&input_wav);
    println!(
        "  WAV spec: {} Hz, {} ch, {} bits, {:?}",
        spec.sample_rate, spec.channels, spec.bits_per_sample, spec.sample_format
    );
    let duration_s = samples.len() as f64 / spec.sample_rate as f64;
    println!("  Loaded {} samples ({:.3}s)", samples.len(), duration_s);

    // ── 3. Encode all frames and collect tokens ──────────────────────
    let frame_size: usize = 1920; // 80 ms at 24 kHz
    let mut all_tokens: Vec<Vec<u32>> = Vec::new(); // one Vec<u32> per codec frame (32 tokens each)

    println!(
        "\nEncoding {} sample frames ({} codebooks) ...",
        frame_size, num_codebooks
    );

    for chunk in samples.chunks(frame_size) {
        let frame: Vec<f32> = if chunk.len() < frame_size {
            let mut f = chunk.to_vec();
            f.resize(frame_size, 0.0);
            f
        } else {
            chunk.to_vec()
        };

        let tokens = mimi.encode(&frame);
        if tokens.is_empty() {
            continue;
        }

        let n_frames_in_batch = tokens.len() / num_codebooks;
        for f_idx in 0..n_frames_in_batch {
            let frame_tokens =
                tokens[f_idx * num_codebooks..(f_idx + 1) * num_codebooks].to_vec();
            all_tokens.push(frame_tokens);
        }
    }

    println!(
        "  Encoded {} codec frames ({} tokens total)",
        all_tokens.len(),
        all_tokens.len() * num_codebooks
    );
    assert!(!all_tokens.is_empty(), "Expected at least one encoded frame");

    // ── 4a. Decode with ALL 32 codebooks ─────────────────────────────
    println!("\nDecoding with all 32 codebooks ...");
    mimi.reset_decoder();
    let mut output_32cb: Vec<f32> = Vec::new();
    for frame_tokens in &all_tokens {
        let pcm = mimi.decode(frame_tokens);
        output_32cb.extend_from_slice(&pcm);
    }
    println!(
        "  32cb output: {} samples ({:.3}s)",
        output_32cb.len(),
        output_32cb.len() as f64 / 24000.0
    );

    // ── 4b. Decode with ONLY first 8 codebooks ──────────────────────
    println!("Decoding with only first 8 codebooks ...");
    mimi.reset_decoder();
    let n_active: usize = 8;
    let mut output_8cb: Vec<f32> = Vec::new();
    for frame_tokens in &all_tokens {
        let partial_tokens = &frame_tokens[..n_active];
        let pcm = mimi.decode_n(partial_tokens, n_active);
        output_8cb.extend_from_slice(&pcm);
    }
    println!(
        "  8cb output: {} samples ({:.3}s)",
        output_8cb.len(),
        output_8cb.len() as f64 / 24000.0
    );

    // ── 5. Compute signal quality stats ──────────────────────────────
    fn compute_rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    }

    fn compute_peak(samples: &[f32]) -> f32 {
        samples.iter().map(|s| s.abs()).fold(0.0_f32, f32::max)
    }

    let rms_32 = compute_rms(&output_32cb);
    let rms_8 = compute_rms(&output_8cb);
    let peak_32 = compute_peak(&output_32cb);
    let peak_8 = compute_peak(&output_8cb);

    // Compute RMS of the difference (noise introduced by dropping codebooks)
    let min_len = output_32cb.len().min(output_8cb.len());
    let diff: Vec<f32> = output_32cb[..min_len]
        .iter()
        .zip(output_8cb[..min_len].iter())
        .map(|(a, b)| a - b)
        .collect();
    let rms_diff = compute_rms(&diff);

    // SNR: signal is 32cb output, noise is the difference
    let snr_db = if rms_diff > 0.0 {
        20.0 * (rms_32 / rms_diff).log10()
    } else {
        f32::INFINITY
    };

    println!("\n===== 8-CODEBOOK vs 32-CODEBOOK COMPARISON =====");
    println!("  32cb RMS:             {:.6}", rms_32);
    println!("  8cb  RMS:             {:.6}", rms_8);
    println!("  RMS ratio (8/32):     {:.4}", rms_8 / rms_32.max(1e-10));
    println!("  32cb peak:            {:.6}", peak_32);
    println!("  8cb  peak:            {:.6}", peak_8);
    println!("  Difference RMS:       {:.6}", rms_diff);
    println!("  SNR (32cb vs diff):   {:.2} dB", snr_db);
    println!("  Overlap samples:      {}", min_len);
    println!("  32cb total samples:   {}", output_32cb.len());
    println!("  8cb  total samples:   {}", output_8cb.len());
    println!("================================================\n");

    // ── 6. Write output WAVs (16-bit PCM for correct mono playback) ──
    let out_spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    if !output_32cb.is_empty() {
        let mut writer = hound::WavWriter::create(&output_wav_32cb, out_spec)
            .expect("failed to create 32cb WAV");
        for &s in &output_32cb {
            let val = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
            writer.write_sample(val).unwrap();
        }
        writer.finalize().unwrap();
        println!("32cb roundtrip WAV written to: {}", output_wav_32cb.display());
    }

    if !output_8cb.is_empty() {
        let mut writer = hound::WavWriter::create(&output_wav_8cb, out_spec)
            .expect("failed to create 8cb WAV");
        for &s in &output_8cb {
            let val = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
            writer.write_sample(val).unwrap();
        }
        writer.finalize().unwrap();
        println!("8cb roundtrip WAV written to: {}", output_wav_8cb.display());
    }

    // ── 7. Assertions ───────────────────────────────────────────────
    assert!(!output_32cb.is_empty(), "32cb decode produced no audio");
    assert!(!output_8cb.is_empty(), "8cb decode produced no audio");

    // The 8cb output should have non-trivial energy (not silence)
    assert!(
        rms_8 > 0.001,
        "8cb RMS too low ({:.6}) — output might be silence",
        rms_8
    );

    // The 8cb output should have roughly similar energy to 32cb
    // (within an order of magnitude — it's lossy, not silent)
    let ratio = rms_8 / rms_32.max(1e-10);
    assert!(
        ratio > 0.1 && ratio < 10.0,
        "8cb/32cb RMS ratio out of expected range: {:.4}",
        ratio
    );

    println!("All assertions passed. 8-codebook partial decode produces audio with recognisable energy.");
}
