//! Integration test: load quantized GGUF model, run inference on test audio.
//!
//! Set `STS_TEST_MODEL_DIR` (path to a `personaplex-*-q4_k-webgpu` checkout)
//! to enable model-loading tests. Set `STS_TEST_AUDIO_WAV` (path to a mono
//! WAV) for end-to-end audio tests. Override `STS_TEST_NUM_LAYERS=24` when
//! pointing at the layer-pruned 24L checkpoint.

mod common;

use std::fs;

use serde_json::json;

#[test]
fn test_load_gguf_shards() {
    let Some(model_dir) = common::model_dir() else {
        return;
    };

    let shards = common::load_shards(&model_dir);
    let total: usize = shards.iter().map(|s| s.len()).sum();
    println!("Total shard data: {:.2} GB", total as f64 / 1e9);

    use sts_wasm::gguf::Q4ModelLoader;
    let loader = Q4ModelLoader::from_shards(shards).unwrap();
    println!(
        "GGUF parsed: version={}, tensors={}",
        loader.version(),
        loader.tensor_count()
    );
    // Sanity check — the 7b-v1 model has 475 tensors, the 24L has fewer.
    // Don't pin the exact count, just assert it parsed something nontrivial.
    assert!(loader.tensor_count() > 100);
}

#[test]
fn test_load_full_model() {
    let Some(model_dir) = common::model_dir() else {
        return;
    };

    use burn::backend::wgpu::WgpuDevice;
    use sts_wasm::gguf::Q4ModelLoader;
    use sts_wasm::loader::load_sts_model_deferred;
    use sts_wasm::StsConfig;

    let shards = common::load_shards(&model_dir);
    let device = WgpuDevice::default();
    let config = StsConfig {
        num_layers: common::num_layers(),
        ..StsConfig::default()
    };

    println!("Parsing GGUF...");
    let mut loader = Q4ModelLoader::from_shards(shards).unwrap();

    println!("Loading model (phase 1)...");
    let parts = load_sts_model_deferred(&mut loader, &config, &device).unwrap();
    drop(loader);

    println!("Finalizing model (phase 2)...");
    let (temporal, depth) = parts.finalize(&device).unwrap();

    println!("Model loaded successfully!");
    println!("  Temporal layers: {}", config.num_layers);
    println!("  Depth layers: {}", config.depth_num_layers);

    // Quick sanity: create caches and run one step with dummy tokens
    use sts_wasm::stream::StsStream;
    let temporal_cache = temporal.create_cache();
    let depth_cache = depth.create_cache();
    let mut stream = StsStream::new(config.clone(), temporal_cache, depth_cache);

    let _dummy_tokens = vec![0u32; config.num_codebooks];
    println!("Running one inference step...");
    let out = pollster::block_on(stream.step(&temporal, &depth));
    println!("  text_token: {}", out.text_token);
    println!("  model_audio_tokens: {:?}", out.model_audio_tokens);
    println!("Inference step completed!");
}

#[test]
fn test_compare_reference() {
    // Compare first-step greedy output against Python reference (BF16 model).
    // Reference values computed with dep_q=16, initial tokens: text=32000, audio=2048.
    // Expected (BF16 greedy): text=3, audio=[1049, 1948, 936, 1297, 1999, 136, 595, 986]
    let Some(model_dir) = common::model_dir() else {
        return;
    };

    use burn::backend::wgpu::WgpuDevice;
    use sts_wasm::gguf::Q4ModelLoader;
    use sts_wasm::loader::load_sts_model_deferred;
    use sts_wasm::stream::StsStream;
    use sts_wasm::StsConfig;

    let shards = common::load_shards(&model_dir);
    let device = WgpuDevice::default();
    let config = StsConfig {
        num_layers: common::num_layers(),
        ..StsConfig::default()
    };

    let mut loader = Q4ModelLoader::from_shards(shards).unwrap();
    let parts = load_sts_model_deferred(&mut loader, &config, &device).unwrap();
    drop(loader);
    let (temporal, depth) = parts.finalize(&device).unwrap();

    let temporal_cache = temporal.create_cache();
    let depth_cache = depth.create_cache();
    let mut stream = StsStream::new(config.clone(), temporal_cache, depth_cache);
    // Greedy for deterministic comparison
    stream.set_sampling_params(0.0, 1, 0.0, 1);

    // Match Moshi reference initial tokens: all audio = 2048 (padding)
    let audio_pad = config.audio_vocab_size as u32 - 1; // 2048
    let user_tokens = vec![audio_pad; config.num_codebooks];

    // Print embedding sum for debugging (compare against Python BF16 reference)
    {
        let dim = config.hidden_size;
        let mut sum = vec![0.0f32; dim];
        // Replicate temporal forward embedding accumulation
        // text_emb(32000) - the start token
        temporal
            .text_emb()
            .embed_id_add_cpu(config.text_start_token, &mut sum);
        // model audio embs (all 2048)
        for i in 0..config.num_codebooks {
            temporal.model_audio_emb()[i].embed_id_add_cpu(audio_pad, &mut sum);
        }
        // user audio embs (all 2048)
        for i in 0..config.num_codebooks {
            temporal.user_audio_emb()[i].embed_id_add_cpu(audio_pad, &mut sum);
        }
        println!("Q4 embedding sum first 10: {:?}", &sum[..10]);
        let norm: f32 = sum.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!("Q4 embedding sum norm: {:.4}", norm);
        println!(
            "BF16 ref first 10: [-0.0229, -0.0028, -0.0334, 0.0101, -0.0163, 0.0230, -0.0202, 0.0113, 0.0329, -0.0248]"
        );
        println!("BF16 ref norm: 2.4808");
    }
    let _ = &user_tokens;

    let out = pollster::block_on(stream.step(&temporal, &depth));

    println!("Q4 Text token (greedy): {}", out.text_token);
    println!("Q4 Audio tokens: {:?}", out.model_audio_tokens);
    println!();
    println!("BF16 reference: text=3, audio=[1049, 1948, 936, 1297, 1999, 136, 595, 986]");

    // Q4 should be close but may differ slightly from BF16
    // For now just print both for visual comparison
}

#[test]
fn test_temporal_layer0_debug() {
    // Debug: trace through layer 0 step-by-step to find where output goes to zero.
    let Some(model_dir) = common::model_dir() else {
        return;
    };

    use burn::backend::wgpu::WgpuDevice;
    use burn::tensor::Tensor;
    use sts_wasm::gguf::Q4ModelLoader;
    use sts_wasm::loader::load_sts_model_deferred;
    use sts_wasm::StsConfig;

    let shards = common::load_shards(&model_dir);
    let device = WgpuDevice::default();
    let config = StsConfig {
        num_layers: common::num_layers(),
        ..StsConfig::default()
    };

    let mut loader = Q4ModelLoader::from_shards(shards).unwrap();
    let parts = load_sts_model_deferred(&mut loader, &config, &device).unwrap();
    drop(loader);
    let (temporal, _depth) = parts.finalize(&device).unwrap();

    // Build embedding input
    let dim = config.hidden_size;
    let audio_pad = config.audio_vocab_size as u32 - 1;
    let text_token = config.text_start_token;
    let mut sum = vec![0.0f32; dim];
    temporal.text_emb().embed_id_add_cpu(text_token, &mut sum);
    for i in 0..config.num_codebooks {
        temporal.model_audio_emb()[i].embed_id_add_cpu(audio_pad, &mut sum);
    }
    for i in 0..config.num_codebooks {
        temporal.user_audio_emb()[i].embed_id_add_cpu(audio_pad, &mut sum);
    }
    let emb_norm: f32 = sum.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("Embedding norm: {emb_norm:.4}");

    let x = Tensor::<burn::backend::Wgpu, 3>::from_data(
        burn::tensor::TensorData::new(sum, [1, 1, dim]),
        &device,
    );

    // Manually run layer 0 of temporal transformer
    let mut cache = temporal.create_cache();
    let layer0 = &temporal.layers()[0];

    // Step 1: norm1
    let normed = layer0.norm1().forward(x.clone());
    let flat: Tensor<burn::backend::Wgpu, 1> = normed.clone().reshape([dim]);
    let vals: Vec<f32> = flat.to_data().to_vec().expect("norm to_vec");
    let norm = vals.iter().map(|v| v * v).sum::<f32>().sqrt();
    println!("After norm1: norm={norm:.4}, first5={:?}", &vals[..5]);

    // Step 2: attention
    let attn_out = layer0.attention().forward_with_cache(
        normed,
        temporal.rope(),
        cache.get_mut(0).unwrap(),
    );
    let flat: Tensor<burn::backend::Wgpu, 1> = attn_out.clone().reshape([dim]);
    let vals: Vec<f32> = flat.to_data().to_vec().expect("attn to_vec");
    let norm = vals.iter().map(|v| v * v).sum::<f32>().sqrt();
    println!("After attention: norm={norm:.4}, first5={:?}", &vals[..5]);

    // Step 3: residual
    let post_attn = attn_out + x.clone();
    let flat: Tensor<burn::backend::Wgpu, 1> = post_attn.clone().reshape([dim]);
    let vals: Vec<f32> = flat.to_data().to_vec().expect("post_attn to_vec");
    let norm = vals.iter().map(|v| v * v).sum::<f32>().sqrt();
    println!("After attn+residual: norm={norm:.4}, first5={:?}", &vals[..5]);

    // Step 4: norm2
    let normed2 = layer0.norm2().forward(post_attn.clone());
    let flat: Tensor<burn::backend::Wgpu, 1> = normed2.clone().reshape([dim]);
    let vals: Vec<f32> = flat.to_data().to_vec().expect("norm2 to_vec");
    let norm = vals.iter().map(|v| v * v).sum::<f32>().sqrt();
    println!("After norm2: norm={norm:.4}, first5={:?}", &vals[..5]);

    // Step 5: FFN
    let ffn_out = layer0.ffn().forward(normed2);
    let flat: Tensor<burn::backend::Wgpu, 1> = ffn_out.clone().reshape([dim]);
    let vals: Vec<f32> = flat.to_data().to_vec().expect("ffn to_vec");
    let norm = vals.iter().map(|v| v * v).sum::<f32>().sqrt();
    println!("After FFN: norm={norm:.4}, first5={:?}", &vals[..5]);

    // Step 6: final residual
    let out = ffn_out + post_attn;
    let flat: Tensor<burn::backend::Wgpu, 1> = out.clone().reshape([dim]);
    let vals: Vec<f32> = flat.to_data().to_vec().expect("out to_vec");
    let norm = vals.iter().map(|v| v * v).sum::<f32>().sqrt();
    println!("After FFN+residual: norm={norm:.4}, first5={:?}", &vals[..5]);
}

#[test]
fn test_q4k_matmul_nonzero() {
    // Verify Q4_K matmul produces non-zero output.
    // Load a single Q4_K weight tensor and run matmul with a non-trivial input.
    let Some(model_dir) = common::model_dir() else {
        return;
    };

    use burn::backend::wgpu::WgpuDevice;
    use burn::tensor::Tensor;
    use sts_wasm::gguf::Q4ModelLoader;

    let shards = common::load_shards(&model_dir);
    let device = WgpuDevice::default();
    let mut loader = Q4ModelLoader::from_shards(shards).unwrap();

    // Load transformer.layers.0.self_attn.in_proj_weight (Q4_K, [12288, 4096])
    let linear = loader
        .load_linear_auto("transformer.layers.0.self_attn.in_proj_weight", &device)
        .unwrap();

    // Create a non-trivial input [1, 1, 4096]
    let input_data: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.001) - 2.0).collect();
    let input = Tensor::<burn::backend::Wgpu, 3>::from_data(
        burn::tensor::TensorData::new(input_data, [1, 1, 4096]),
        &device,
    );

    let output = linear.forward(input);
    let [b, m, n] = output.dims();
    println!("Output shape: [{b}, {m}, {n}]");

    let out_flat: Tensor<burn::backend::Wgpu, 1> = output.reshape([n]);
    let out_vals: Vec<f32> = out_flat.to_data().to_vec().expect("output to_vec");
    let norm: f32 = out_vals.iter().map(|v| v * v).sum::<f32>().sqrt();
    println!("Output norm: {norm:.4}");
    println!("Output first 10: {:?}", &out_vals[..10]);
    println!("Output last 10: {:?}", &out_vals[n - 10..]);

    assert!(norm > 0.0, "Q4_K matmul output should be non-zero");
    assert!(norm > 1.0, "Q4_K matmul output should have meaningful magnitude");
}

#[test]
fn test_q4k_cooperative_vs_naive() {
    // Compare cooperative matvec (M=1) vs naive (M=2, first row) output.
    // If cooperative shader has a bug, results will differ.
    let Some(model_dir) = common::model_dir() else {
        return;
    };

    use burn::backend::wgpu::WgpuDevice;
    use burn::tensor::Tensor;
    use sts_wasm::gguf::Q4ModelLoader;

    let shards = common::load_shards(&model_dir);
    let device = WgpuDevice::default();
    let mut loader = Q4ModelLoader::from_shards(shards).unwrap();

    // Load a Q4_K tensor
    let linear = loader
        .load_linear_auto("transformer.layers.0.self_attn.in_proj_weight", &device)
        .unwrap();

    let k = 4096usize;
    let n = 12288usize;

    // Create input vector
    let input_data: Vec<f32> = (0..k).map(|i| (i as f32 * 0.001) - 2.0).collect();

    // M=1 path → cooperative shader
    let input_m1 = Tensor::<burn::backend::Wgpu, 3>::from_data(
        burn::tensor::TensorData::new(input_data.clone(), [1, 1, k]),
        &device,
    );
    let out_m1 = linear.forward(input_m1);
    let out_m1_flat: Tensor<burn::backend::Wgpu, 1> = out_m1.reshape([n]);
    let vals_m1: Vec<f32> = pollster::block_on(out_m1_flat.into_data_async())
        .expect("readback")
        .to_vec()
        .expect("to_vec");

    // M=2 path → naive shader (duplicate the row)
    let mut input_data_m2 = input_data.clone();
    input_data_m2.extend_from_slice(&input_data);
    let input_m2 = Tensor::<burn::backend::Wgpu, 3>::from_data(
        burn::tensor::TensorData::new(input_data_m2, [1, 2, k]),
        &device,
    );
    let out_m2 = linear.forward(input_m2);
    // Take first row only
    let out_m2_row0 = out_m2.slice([0..1, 0..1, 0..n]);
    let out_m2_flat: Tensor<burn::backend::Wgpu, 1> = out_m2_row0.reshape([n]);
    let vals_m2: Vec<f32> = pollster::block_on(out_m2_flat.into_data_async())
        .expect("readback")
        .to_vec()
        .expect("to_vec");

    // Compare
    let mut max_diff: f32 = 0.0;
    let mut sum_diff: f32 = 0.0;
    let mut num_mismatches = 0;
    for i in 0..n {
        let diff = (vals_m1[i] - vals_m2[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        sum_diff += diff;
        if diff > 0.01 {
            num_mismatches += 1;
            if num_mismatches <= 10 {
                println!(
                    "Mismatch at [{i}]: cooperative={:.6}, naive={:.6}, diff={:.6}",
                    vals_m1[i], vals_m2[i], diff
                );
            }
        }
    }

    let norm_m1: f32 = vals_m1.iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm_m2: f32 = vals_m2.iter().map(|v| v * v).sum::<f32>().sqrt();
    println!("Cooperative norm: {norm_m1:.4}, Naive norm: {norm_m2:.4}");
    println!(
        "Max diff: {max_diff:.6}, Avg diff: {:.6}",
        sum_diff / n as f32
    );
    println!("Mismatches (>0.01): {num_mismatches} / {n}");

    assert!(
        max_diff < 0.1,
        "Cooperative vs naive max difference too large: {max_diff}"
    );
    assert!(
        num_mismatches == 0,
        "Cooperative vs naive have {num_mismatches} mismatches"
    );
}

#[test]
fn test_end_to_end_audio() {
    let Some(model_dir) = common::model_dir() else {
        return;
    };
    let Some(wav_path) = common::audio_wav() else {
        return;
    };

    use burn::backend::wgpu::WgpuDevice;
    use sts_wasm::gguf::Q4ModelLoader;
    use sts_wasm::loader::load_sts_model_deferred;
    use sts_wasm::mimi::MimiCodec;
    use sts_wasm::stream::StsStream;
    use sts_wasm::StsConfig;

    let device = WgpuDevice::default();
    let config = StsConfig {
        num_layers: common::num_layers(),
        ..StsConfig::default()
    };

    // Load the STS model
    println!("Loading STS model...");
    let shards = common::load_shards(&model_dir);
    let mut loader = Q4ModelLoader::from_shards(shards).unwrap();
    let parts = load_sts_model_deferred(&mut loader, &config, &device).unwrap();
    drop(loader);
    let (temporal, depth) = parts.finalize(&device).unwrap();

    // Load Mimi codec
    println!("Loading Mimi codec...");
    let mimi_path = common::find_mimi_weights(&model_dir)
        .expect("no `tokenizer-*.safetensors` in model dir");
    let mimi_data = fs::read(&mimi_path).unwrap();
    let mut mimi = MimiCodec::from_bytes(&mimi_data).unwrap();
    println!(
        "  Mimi loaded: {} codebooks, {}Hz",
        mimi.num_codebooks(),
        mimi.sample_rate()
    );

    // Load test WAV
    println!("Loading test audio: {}", wav_path.display());
    let (samples, spec) = common::read_wav_mono_f32(&wav_path);
    println!(
        "  WAV: {}Hz, {} channels, {} bits",
        spec.sample_rate, spec.channels, spec.bits_per_sample
    );
    println!(
        "  Loaded {} samples ({:.2}s)",
        samples.len(),
        samples.len() as f64 / spec.sample_rate as f64
    );

    // Create inference stream
    let temporal_cache = temporal.create_cache();
    let depth_cache = depth.create_cache();
    let mut stream = StsStream::new(config.clone(), temporal_cache, depth_cache);

    // Phase 1: Voice preset (optional, but improves output quality)
    let ref_dir = common::reference_dir();
    let voice_preset_path = ref_dir.join("NATF2_embeddings.bin");
    let voice_cache_path = ref_dir.join("NATF2_cache.json");
    if voice_preset_path.exists() && voice_cache_path.exists() {
        println!("Loading voice preset...");
        let emb_bytes = fs::read(&voice_preset_path).unwrap();
        assert_eq!(emb_bytes.len() % 4, 0, "Embeddings file size must be multiple of 4");
        let embeddings: Vec<f32> = emb_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        let cache_json: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&voice_cache_path).unwrap()).unwrap();
        let num_frames = cache_json["num_frames"].as_u64().unwrap() as usize;
        let cache_snapshot: Vec<Vec<i32>> = cache_json["cache"]
            .as_array()
            .unwrap()
            .iter()
            .map(|stream| {
                stream
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_i64().unwrap() as i32)
                    .collect()
            })
            .collect();

        assert_eq!(embeddings.len(), num_frames * config.hidden_size);
        stream.prefill_voice_preset(&embeddings, num_frames, &cache_snapshot, &temporal);
        println!("  Voice preset done: {} frames processed", num_frames);
    }

    // Phase 2-4: System prompt prefill (silence + text + silence)
    println!("Running prefill (system prompt + silence spacers)...");
    stream.prefill(&temporal);

    // Reset Mimi streaming state after prefill, before user audio encoding
    // (matches Python: mimi.reset_streaming() after step_system_prompts)
    mimi.reset();
    println!("  Prefill done, Mimi reset");

    // Encode user audio -> Mimi tokens for prefill
    let frame_size = 1920; // 80ms at 24kHz
    let mut total_output_samples = 0;
    let mut total_text_tokens = 0;
    let mut output_audio = Vec::new();
    let mut all_model_audio_tokens: Vec<Vec<u32>> = Vec::new();

    println!("Encoding user audio with Mimi...");
    let mut user_audio_frames: Vec<Vec<u32>> = Vec::new();
    for chunk in samples.chunks(frame_size) {
        // Pad last chunk if needed
        let mut frame = chunk.to_vec();
        if frame.len() < frame_size {
            frame.resize(frame_size, 0.0);
        }

        // Mimi encode
        let tokens = mimi.encode(&frame);
        if tokens.is_empty() {
            continue;
        }
        let num_codebooks = mimi.num_codebooks();
        let user_tokens: Vec<u32> = tokens[..num_codebooks.min(config.num_codebooks)].to_vec();
        user_audio_frames.push(user_tokens);
    }
    println!("  Encoded {} audio frames", user_audio_frames.len());

    // Phase 3: User audio prefill (feeds through temporal + depformer with provided tokens)
    println!(
        "Running user audio prefill ({} frames)...",
        user_audio_frames.len()
    );
    pollster::block_on(stream.prefill_user_audio(&user_audio_frames, &temporal, &depth));
    let input_frame_count = user_audio_frames.len();
    println!(
        "  User audio prefill done: {} total frames in KV cache",
        stream.frame_count()
    );

    // -----------------------------------------------------------------------
    // Response generation phase: continue with sine tokens as user audio
    // (user is silent, model should generate its spoken response)
    // -----------------------------------------------------------------------
    let max_response_frames = 100; // ~8 seconds at 12.5 Hz
    let _sine_tokens = config.sine_tokens.to_vec();
    let mut response_frame_count = 0;
    let mut response_text_tokens: Vec<u32> = Vec::new();
    let mut early_stopped = false;

    println!(
        "\nRunning response generation (up to {} frames = {:.1}s)...",
        max_response_frames,
        max_response_frames as f64 / config.frame_rate as f64
    );

    let gen_start = std::time::Instant::now();
    for frame_idx in 0..max_response_frames {
        let frame_start = std::time::Instant::now();
        let out = pollster::block_on(stream.step(&temporal, &depth));
        let frame_ms = frame_start.elapsed().as_millis();

        // Collect model audio tokens
        all_model_audio_tokens.push(out.model_audio_tokens.clone());
        response_text_tokens.push(out.text_token);

        if out.text_token != config.text_padding_id && out.text_token != 0 {
            total_text_tokens += 1;
        }

        // Log every 10 frames, plus last few before stop
        if frame_idx % 10 == 0 || stream.consecutive_silence_frames() >= 13 {
            println!(
                "  Response frame {frame_idx}: text={}, audio={:?}, silence_streak={}, {frame_ms}ms",
                out.text_token,
                &out.model_audio_tokens[..3],
                stream.consecutive_silence_frames(),
            );
        }

        response_frame_count += 1;

        if stream.should_stop() {
            println!(
                "  Early stop: {} consecutive silence frames at response frame {frame_idx}",
                stream.consecutive_silence_frames(),
            );
            early_stopped = true;
            break;
        }
    }

    // -----------------------------------------------------------------------
    // Decode response audio tokens with Mimi (only response phase, not prefill)
    // -----------------------------------------------------------------------
    println!(
        "\nDecoding {} response frames with Mimi...",
        all_model_audio_tokens.len()
    );
    let n_active = config.num_codebooks; // 8 model audio codebooks
    for tokens in &all_model_audio_tokens {
        let pcm = mimi.decode_n(tokens, n_active);
        output_audio.extend_from_slice(&pcm);
        total_output_samples += pcm.len();
    }

    // -----------------------------------------------------------------------
    // Print results
    // -----------------------------------------------------------------------
    println!("\n=== Results ===");
    println!(
        "  Input: {:.2}s audio ({} frames, prefilled)",
        samples.len() as f64 / 24000.0,
        input_frame_count
    );
    println!(
        "  Response: {} frames generated ({:.2}s)",
        response_frame_count,
        response_frame_count as f64 / config.frame_rate as f64
    );
    let gen_total_ms = gen_start.elapsed().as_millis();
    let ms_per_frame = gen_total_ms as f64 / response_frame_count as f64;
    println!(
        "  Generation time: {gen_total_ms}ms total, {ms_per_frame:.1}ms/frame (need <80ms for real-time)"
    );
    println!("  Silence early-stop triggered: {early_stopped}");
    println!(
        "  Total output: {total_output_samples} samples ({:.2}s audio)",
        total_output_samples as f64 / 24000.0
    );
    println!("  Text tokens (non-padding): {total_text_tokens}");

    // Print response text tokens to see inner monologue
    let response_text_nonpad: Vec<u32> = response_text_tokens
        .iter()
        .copied()
        .filter(|&t| t != config.text_padding_id && t != 0)
        .collect();
    println!(
        "  Response text tokens (non-padding, {} total): {:?}",
        response_text_nonpad.len(),
        response_text_nonpad
    );

    // Print first/last few response audio token frames
    if response_frame_count > 0 {
        println!("  First 5 response audio frames:");
        for i in 0..5.min(response_frame_count) {
            println!("    frame {i}: {:?}", all_model_audio_tokens[i]);
        }
        if response_frame_count > 5 {
            println!("  Last 5 response audio frames:");
            for i in (response_frame_count.saturating_sub(5))..response_frame_count {
                println!("    frame {i}: {:?}", all_model_audio_tokens[i]);
            }
        }
    }

    // Save generated tokens to JSON for cross-validation with Python Mimi
    let out_dir = common::test_output_dir();
    let tokens_json_path = out_dir.join("joke_output_tokens.json");
    {
        let json_tokens: Vec<Vec<u32>> = all_model_audio_tokens.clone();
        let json_obj = json!({
            "num_codebooks": config.num_codebooks,
            "num_frames": json_tokens.len(),
            "frame_rate": config.frame_rate,
            "tokens": json_tokens,
        });
        fs::write(&tokens_json_path, serde_json::to_string_pretty(&json_obj).unwrap()).unwrap();
        println!("  Tokens JSON written to: {}", tokens_json_path.display());
    }

    // Write output WAV for manual inspection
    let out_wav_path = out_dir.join("joke_output.wav");
    if !output_audio.is_empty() {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 24000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&out_wav_path, spec).unwrap();
        for &s in &output_audio {
            let val = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
            writer.write_sample(val).unwrap();
        }
        writer.finalize().unwrap();
        println!("  Output WAV written to: {}", out_wav_path.display());
    }

    // Basic assertions
    assert!(!output_audio.is_empty(), "Expected non-empty audio output");
    assert!(total_output_samples > 0, "Expected some output samples");
    assert!(response_frame_count > 0, "Expected at least one response frame");
}

#[test]
fn test_layer_comparison_log() {
    // Log intermediate layer values from the Q4 model for comparison
    // against a BF16 Python reference. Runs one frame with all padding
    // tokens and greedy decoding, capturing per-layer hidden states.
    let Some(model_dir) = common::model_dir() else {
        return;
    };

    use burn::backend::wgpu::WgpuDevice;
    use sts_wasm::gguf::Q4ModelLoader;
    use sts_wasm::loader::load_sts_model_deferred;
    use sts_wasm::model::sample_greedy;
    use sts_wasm::StsConfig;

    let shards = common::load_shards(&model_dir);
    let device = WgpuDevice::default();
    let config = StsConfig {
        num_layers: common::num_layers(),
        ..StsConfig::default()
    };

    println!("Parsing GGUF...");
    let mut loader = Q4ModelLoader::from_shards(shards).unwrap();

    println!("Loading model (phase 1)...");
    let parts = load_sts_model_deferred(&mut loader, &config, &device).unwrap();
    drop(loader);

    println!("Finalizing model (phase 2)...");
    let (temporal, depth) = parts.finalize(&device).unwrap();

    // -- Set up initial tokens: all padding --
    let audio_pad = config.audio_vocab_size as i32 - 1; // 2048
    let text_token = config.text_start_token as i32; // 32000
    let user_audio_tokens = vec![audio_pad; config.num_codebooks];
    let model_audio_tokens = vec![audio_pad; config.num_codebooks];

    // -- Temporal transformer forward with logging --
    println!("Running temporal forward with logging...");
    let mut temporal_cache = temporal.create_cache();
    let (hidden, text_logits, temporal_log) = temporal.forward_with_logging(
        &user_audio_tokens,
        &model_audio_tokens,
        text_token,
        &mut temporal_cache,
    );

    // -- Sample text token (greedy) --
    let text_logits_clone = text_logits.clone();
    let sampled_text_token = pollster::block_on(sample_greedy(text_logits));

    // -- Extract text logits top-10 --
    let [_b, _s, vocab] = text_logits_clone.dims();
    let text_logits_1d: burn::tensor::Tensor<burn::backend::Wgpu, 1> =
        text_logits_clone.reshape([vocab]);
    let text_logits_vals: Vec<f32> = text_logits_1d
        .to_data()
        .to_vec()
        .expect("text logits to_vec");
    let mut text_indexed: Vec<(usize, f32)> = text_logits_vals
        .iter()
        .copied()
        .enumerate()
        .collect();
    text_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    text_indexed.truncate(10);

    // -- Log temporal hidden state passed to depth --
    let dim = config.hidden_size;
    let hidden_flat: burn::tensor::Tensor<burn::backend::Wgpu, 1> =
        hidden.clone().reshape([dim]);
    let hidden_vals: Vec<f32> = hidden_flat
        .to_data()
        .to_vec()
        .expect("temporal hidden to_vec");
    let hidden_norm = hidden_vals.iter().map(|v| v * v).sum::<f32>().sqrt();
    let hidden_first_10: Vec<f32> = hidden_vals[..10.min(dim)].to_vec();

    // -- Depth transformer generate with logging --
    println!("Running depth generate with logging...");
    let mut depth_cache = depth.create_cache();
    let (audio_tokens, depth_step_logs) = pollster::block_on(depth.generate_with_logging(
        hidden,
        sampled_text_token,
        &mut depth_cache,
        0.0, // greedy
        1,
        None, // No provided tokens
        None, // No penalty history
        1.0,
    ));

    // -- Print summary --
    println!("\n=== TEMPORAL TRANSFORMER ===");
    println!(
        "Embedding sum: norm={:.6}, first_10={:?}",
        temporal_log.embedding_sum_norm, temporal_log.embedding_sum_first_10
    );
    for ll in &temporal_log.layer_logs {
        println!(
            "  Layer {:2}: norm={:.6}, first_10={:?}",
            ll.layer, ll.norm, ll.first_10
        );
    }
    println!(
        "After out_norm: norm={:.6}, first_10={:?}",
        temporal_log.after_out_norm_norm, temporal_log.after_out_norm_first_10
    );
    println!(
        "Text logits top-10: {:?}",
        text_indexed
            .iter()
            .map(|(t, l)| format!("{}:{:.4}", t, l))
            .collect::<Vec<_>>()
    );
    println!("Text token (greedy): {sampled_text_token}");
    println!(
        "Temporal hidden: norm={:.6}, first_10={:?}",
        hidden_norm, hidden_first_10
    );

    println!("\n=== DEPTH TRANSFORMER ===");
    for ds in &depth_step_logs {
        println!(
            "Step {:2}: input_norm={:.6}, token={}",
            ds.step, ds.input_norm, ds.token
        );
        for ll in &ds.layer_logs {
            println!(
                "    Layer {:2}: norm={:.6}, first_10={:?}",
                ll.layer, ll.norm, ll.first_10
            );
        }
        println!(
            "    Logits top-10: {:?}",
            ds.logits_top10
                .iter()
                .map(|tl| format!("{}:{:.4}", tl.token, tl.logit))
                .collect::<Vec<_>>()
        );
    }
    println!("\nAudio tokens: {audio_tokens:?}");

    // -- Build JSON output --
    let json_output = json!({
        "embedding_sum": {
            "norm": temporal_log.embedding_sum_norm,
            "first_10": temporal_log.embedding_sum_first_10,
        },
        "temporal_layers": temporal_log.layer_logs.iter().map(|ll| {
            json!({
                "layer": ll.layer,
                "norm": ll.norm,
                "first_10": ll.first_10,
            })
        }).collect::<Vec<_>>(),
        "after_out_norm": {
            "norm": temporal_log.after_out_norm_norm,
            "first_10": temporal_log.after_out_norm_first_10,
        },
        "text_logits_top10": text_indexed.iter().map(|&(token, logit)| {
            json!({ "token": token, "logit": logit })
        }).collect::<Vec<_>>(),
        "text_token": sampled_text_token,
        "temporal_hidden": {
            "norm": hidden_norm,
            "first_10": hidden_first_10,
        },
        "depth_steps": depth_step_logs.iter().map(|ds| {
            json!({
                "step": ds.step,
                "input": {
                    "norm": ds.input_norm,
                    "first_10": ds.input_first_10,
                },
                "layers": ds.layer_logs.iter().map(|ll| {
                    json!({
                        "layer": ll.layer,
                        "norm": ll.norm,
                        "first_10": ll.first_10,
                    })
                }).collect::<Vec<_>>(),
                "logits_top10": ds.logits_top10.iter().map(|tl| {
                    json!({ "token": tl.token, "logit": tl.logit })
                }).collect::<Vec<_>>(),
                "token": ds.token,
            })
        }).collect::<Vec<_>>(),
    });

    // -- Write JSON to file --
    let out_path = common::test_output_dir().join("q4_layer_log.json");
    let json_str = serde_json::to_string_pretty(&json_output).unwrap();
    fs::write(&out_path, &json_str).unwrap();
    println!("\nJSON log written to: {}", out_path.display());

    // -- Basic assertions --
    assert_eq!(
        temporal_log.layer_logs.len(),
        config.num_layers,
        "Expected {} temporal layers",
        config.num_layers,
    );
    assert_eq!(depth_step_logs.len(), 16, "Expected 16 depth steps");
    for ds in &depth_step_logs {
        assert_eq!(ds.layer_logs.len(), 6, "Expected 6 depth layers per step");
    }
    assert_eq!(audio_tokens.len(), 16, "Expected 16 audio tokens");
    assert!(
        temporal_log.embedding_sum_norm > 0.0,
        "Embedding norm should be positive"
    );
    println!("\nAll assertions passed.");
}
