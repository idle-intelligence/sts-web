//! WASM bindings for STS using Q4 GGUF weights and wgpu (WebGPU) backend.
//!
//! Provides JavaScript-callable APIs for GPU-accelerated Q4 inference
//! in browsers with WebGPU support.
//!
//! Streaming pipeline:
//!   1. feedAudio()      — encode PCM with Mimi, prefill each frame through temporal+depth
//!   2. generateStart()  — prepare for autoregressive generation
//!   3. generateStep()   — one generation step, returns audio chunk + text (or null when done)

use wasm_bindgen::prelude::*;

use std::sync::OnceLock;

use burn::backend::wgpu::WgpuDevice;

use crate::depth::DepthTransformer;
use crate::gguf::{GgufTensorIndex, Q4ModelLoader};
use crate::loader::{load_sts_model_deferred, IncrementalModelLoader};
use crate::mimi::MimiCodec;
use crate::model::TemporalTransformer;
use crate::stream::StsStream;
use crate::tokenizer::SpmDecoder;
use crate::StsConfig;

/// Device initialized by `initWgpuDevice()` — used by `StsEngine` instances.
static WGPU_DEVICE: OnceLock<WgpuDevice> = OnceLock::new();

fn wasm_log(msg: &str) {
    #[cfg(target_family = "wasm")]
    web_sys::console::log_1(&msg.into());
    #[cfg(not(target_family = "wasm"))]
    let _ = msg;
}

/// Cross-platform millisecond timer.
fn now_ms() -> f64 {
    #[cfg(target_family = "wasm")]
    {
        js_sys::Date::now()
    }
    #[cfg(not(target_family = "wasm"))]
    {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
            * 1000.0
    }
}

/// Cumulative metrics tracked across the session.
struct SessionMetrics {
    total_frames: usize,
    total_temporal_ms: f64,
    total_depth_ms: f64,
    total_mimi_ms: f64,
    total_ms: f64,
}

impl SessionMetrics {
    fn new() -> Self {
        Self {
            total_frames: 0,
            total_temporal_ms: 0.0,
            total_depth_ms: 0.0,
            total_mimi_ms: 0.0,
            total_ms: 0.0,
        }
    }

    fn reset(&mut self) {
        *self = Self::new();
    }
}

/// Initialize panic hook for better error messages in browser console.
#[cfg_attr(target_family = "wasm", wasm_bindgen(start))]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// Initialize the WebGPU device asynchronously.
///
/// Must be called (and awaited) before creating `StsEngine`.
#[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = initWgpuDevice))]
pub async fn init_wgpu_device() {
    use burn::backend::wgpu::{init_device, RuntimeOptions, WgpuSetup};

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU,
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .expect("No WebGPU adapter found");

    let info = adapter.get_info();
    let adapter_limits = adapter.limits();
    wasm_log(&format!(
        "[wgpu] Adapter: {} ({:?}), backend: {:?}",
        info.name, info.device_type, info.backend
    ));
    wasm_log(&format!(
        "[wgpu] Adapter limits: max_compute_invocations_per_workgroup={}, max_buffer_size={}",
        adapter_limits.max_compute_invocations_per_workgroup,
        adapter_limits.max_buffer_size,
    ));

    let features = adapter.features() - wgpu::Features::MAPPABLE_PRIMARY_BUFFERS;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("sts-wgpu"),
            required_features: features,
            required_limits: adapter_limits,
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("Failed to create WebGPU device");

    wasm_log(&format!(
        "[wgpu] Device created: max_compute_invocations_per_workgroup={}",
        device.limits().max_compute_invocations_per_workgroup,
    ));

    let setup = WgpuSetup {
        instance,
        adapter,
        device,
        queue,
        backend: info.backend,
    };

    let wgpu_device = init_device(setup, RuntimeOptions::default());
    WGPU_DEVICE.set(wgpu_device).ok();
}

/// Browser-facing STS engine combining Mimi codec + STS transformers.
///
/// Streaming pipeline:
///   feedAudio(samples)  — encode PCM with Mimi, prefill each frame incrementally
///   generateStart()     — prepare for autoregressive generation
///   generateStep()      — one step → { audio, text } or null when done
#[cfg_attr(target_family = "wasm", wasm_bindgen)]
pub struct StsEngine {
    temporal: Option<TemporalTransformer>,
    depth: Option<DepthTransformer>,
    stream: Option<StsStream>,
    mimi: Option<MimiCodec>,
    tokenizer: Option<SpmDecoder>,
    config: StsConfig,
    device: WgpuDevice,
    shard_bufs: Vec<Vec<u8>>,
    metrics: SessionMetrics,

    /// Incremental model loader (lives between loadModelBegin and loadModelFinish).
    incremental_loader: Option<IncrementalModelLoader>,

    // Audio input buffering
    pcm_buffer: Vec<f32>,
    prefilled: bool,

    // Generation state
    generating: bool,
    gen_step: usize,
    max_gen_frames: usize,

    // Voice preset (pre-computed embeddings + token cache snapshot)
    voice_preset_embeddings: Option<Vec<f32>>,
    voice_preset_num_frames: usize,
    voice_preset_cache: Option<Vec<Vec<i32>>>,
}

#[cfg_attr(target_family = "wasm", wasm_bindgen)]
impl StsEngine {
    /// Create a new StsEngine instance.
    ///
    /// Call `initWgpuDevice()` first, then create this, then load weights.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(constructor))]
    pub fn new() -> Self {
        let device = WGPU_DEVICE
            .get()
            .cloned()
            .unwrap_or_else(WgpuDevice::default);
        Self {
            temporal: None,
            depth: None,
            stream: None,
            mimi: None,
            tokenizer: None,
            config: StsConfig::default(),
            device,
            shard_bufs: Vec::new(),
            metrics: SessionMetrics::new(),
            incremental_loader: None,
            pcm_buffer: Vec::new(),
            prefilled: false,
            generating: false,
            gen_step: 0,
            max_gen_frames: 50, // ~4s at 12.5 Hz
            voice_preset_embeddings: None,
            voice_preset_num_frames: 0,
            voice_preset_cache: None,
        }
    }

    /// Append a model weight shard (for multi-shard GGUF loading).
    ///
    /// Call this for each shard before calling `loadModel`.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = appendModelShard))]
    pub fn append_model_shard(&mut self, shard: &[u8]) {
        self.shard_bufs.push(shard.to_vec());
        wasm_log(&format!(
            "[sts] Shard appended ({} bytes, {} total shards)",
            shard.len(),
            self.shard_bufs.len()
        ));
    }

    /// Load the STS model from previously appended shards.
    ///
    /// Uses two-phase loading: parse GGUF -> drop reader -> finalize tensors on GPU.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = loadModel))]
    pub fn load_model(&mut self) -> Result<(), JsError> {
        if self.shard_bufs.is_empty() {
            return Err(JsError::new(
                "No shards appended. Call appendModelShard first.",
            ));
        }

        wasm_log("[sts] Phase 1: Parsing GGUF and loading Q4 tensors...");

        let shards = std::mem::take(&mut self.shard_bufs);
        let parts = {
            let mut loader = Q4ModelLoader::from_shards(shards)
                .map_err(|e| JsError::new(&format!("Failed to parse GGUF: {e}")))?;
            load_sts_model_deferred(&mut loader, &self.config, &self.device)
                .map_err(|e| JsError::new(&format!("Failed to load Q4 model: {e}")))?
            // loader (and its shard data) dropped here
        };

        wasm_log("[sts] Phase 2: Finalizing model on GPU...");

        let (temporal, depth) = parts
            .finalize(&self.device)
            .map_err(|e| JsError::new(&format!("Failed to finalize model: {e}")))?;

        let temporal_cache = temporal.create_cache();
        let depth_cache = depth.create_cache();
        let stream = StsStream::new(self.config.clone(), temporal_cache, depth_cache);

        self.temporal = Some(temporal);
        self.depth = Some(depth);
        self.stream = Some(stream);

        wasm_log("[sts] Model loaded successfully");
        Ok(())
    }

    // ------------------------------------------------------------------
    // Incremental shard-at-a-time loading (WASM-friendly)
    // ------------------------------------------------------------------

    /// Begin incremental model loading: parse the GGUF header from shard 0
    /// and process shard 0's tensors.
    ///
    /// After this call, shard 0's bytes can be freed by JS garbage collection.
    /// Call `loadModelShard(data, i)` for shards 1..N-1, then `loadModelFinish()`.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = loadModelBegin))]
    pub fn load_model_begin(&mut self, shard0: &[u8]) -> Result<(), JsError> {
        wasm_log(&format!(
            "[sts] loadModelBegin: parsing GGUF header from shard 0 ({} bytes)",
            shard0.len()
        ));

        let index = GgufTensorIndex::parse(shard0)
            .map_err(|e| JsError::new(&format!("Failed to parse GGUF header: {e}")))?;

        wasm_log(&format!(
            "[sts] GGUF v{}: {} tensors, data_section_offset={}",
            index.version, index.tensor_count, index.data_section_offset
        ));

        let mut loader = IncrementalModelLoader::new(
            index,
            self.config.clone(),
            self.device.clone(),
        );

        loader.process_shard(shard0, 0)
            .map_err(|e| JsError::new(&format!("Failed to process shard 0: {e}")))?;

        wasm_log(&format!(
            "[sts] Shard 0 processed ({}/{} tensors so far)",
            loader.processed_count(),
            loader.total_tensor_count()
        ));

        self.incremental_loader = Some(loader);
        Ok(())
    }

    /// Process tensors from shard N (called for shards 1, 2, ..., N-1).
    ///
    /// After this call returns, the shard's bytes can be freed by JS GC.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = loadModelShard))]
    pub fn load_model_shard(&mut self, data: &[u8], shard_index: u32) -> Result<(), JsError> {
        let loader = self.incremental_loader.as_mut().ok_or_else(|| {
            JsError::new("loadModelBegin must be called before loadModelShard")
        })?;

        wasm_log(&format!(
            "[sts] Processing shard {} ({} bytes)...",
            shard_index,
            data.len()
        ));

        loader
            .process_shard(data, shard_index as usize)
            .map_err(|e| JsError::new(&format!("Failed to process shard {shard_index}: {e}")))?;

        wasm_log(&format!(
            "[sts] Shard {} processed ({}/{} tensors so far)",
            shard_index,
            loader.processed_count(),
            loader.total_tensor_count()
        ));

        Ok(())
    }

    /// Assemble the final model from all processed shards.
    ///
    /// Must be called after `loadModelBegin` + all `loadModelShard` calls.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = loadModelFinish))]
    pub fn load_model_finish(&mut self) -> Result<(), JsError> {
        let loader = self.incremental_loader.take().ok_or_else(|| {
            JsError::new("loadModelBegin must be called before loadModelFinish")
        })?;

        wasm_log(&format!(
            "[sts] Assembling model from {} processed tensors...",
            loader.processed_count()
        ));

        let (temporal, depth) = loader
            .finalize()
            .map_err(|e| JsError::new(&format!("Failed to finalize model: {e}")))?;

        let temporal_cache = temporal.create_cache();
        let depth_cache = depth.create_cache();
        let stream = StsStream::new(self.config.clone(), temporal_cache, depth_cache);

        self.temporal = Some(temporal);
        self.depth = Some(depth);
        self.stream = Some(stream);

        wasm_log("[sts] Model loaded successfully (incremental)");
        Ok(())
    }

    /// Initialize the Mimi audio codec from pre-fetched safetensors bytes.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = loadMimi))]
    pub fn load_mimi(&mut self, data: &[u8]) -> Result<(), JsError> {
        wasm_log(&format!("[sts] Loading Mimi codec ({} bytes)...", data.len()));
        let mimi = MimiCodec::from_bytes(data)
            .map_err(|e| JsError::new(&format!("Failed to load Mimi: {e}")))?;
        wasm_log(&format!(
            "[sts] Mimi codec loaded ({} codebooks, {}Hz)",
            mimi.num_codebooks(),
            mimi.sample_rate()
        ));
        self.mimi = Some(mimi);
        Ok(())
    }

    /// Load the SentencePiece tokenizer from raw `.model` bytes.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = loadTokenizer))]
    pub fn load_tokenizer(&mut self, data: &[u8]) -> Result<(), JsError> {
        let tok = SpmDecoder::from_bytes(data);
        wasm_log(&format!(
            "[sts] Tokenizer loaded ({} vocab entries)",
            tok.vocab_len()
        ));
        self.tokenizer = Some(tok);
        Ok(())
    }

    /// Load a voice preset from pre-computed embeddings and cache snapshot.
    ///
    /// `embeddings_bin`: raw f32le bytes (num_frames × hidden_size × 4 bytes)
    /// `cache_json`: JSON string with format:
    ///   `{"num_frames": 51, "dim": 4096, "cache": [[...], ...]}`
    ///
    /// Call this after model loading but before `flush()`.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = loadVoicePreset))]
    pub fn load_voice_preset(&mut self, embeddings_bin: &[u8], cache_json: &str) -> Result<(), JsError> {
        let parsed: serde_json::Value = serde_json::from_str(cache_json)
            .map_err(|e| JsError::new(&format!("Invalid cache JSON: {e}")))?;

        let num_frames = parsed["num_frames"]
            .as_u64()
            .ok_or_else(|| JsError::new("Missing num_frames"))? as usize;

        let cache_snapshot: Vec<Vec<i32>> = parsed["cache"]
            .as_array()
            .ok_or_else(|| JsError::new("Missing cache array"))?
            .iter()
            .map(|stream| {
                stream
                    .as_array()
                    .unwrap_or(&vec![])
                    .iter()
                    .map(|v| v.as_i64().unwrap_or(0) as i32)
                    .collect()
            })
            .collect();

        if !embeddings_bin.len().is_multiple_of(4) {
            return Err(JsError::new("Embeddings size must be multiple of 4"));
        }
        let embeddings: Vec<f32> = embeddings_bin
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        let expected = num_frames * self.config.hidden_size;
        if embeddings.len() != expected {
            return Err(JsError::new(&format!(
                "Expected {} floats ({} frames × {}), got {}",
                expected, num_frames, self.config.hidden_size, embeddings.len()
            )));
        }

        self.voice_preset_embeddings = Some(embeddings);
        self.voice_preset_num_frames = num_frames;
        self.voice_preset_cache = Some(cache_snapshot);

        wasm_log(&format!(
            "[sts] Voice preset loaded: {} frames × {} dim",
            num_frames, self.config.hidden_size
        ));
        Ok(())
    }

    /// Feed PCM audio samples (f32, 24kHz mono).
    ///
    /// Encodes audio with Mimi and immediately prefills each frame through
    /// the temporal+depth transformers. This means prefill happens while the
    /// user is still speaking, not after they stop.
    ///
    /// Returns the number of frames prefilled in this call.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = feedAudio))]
    pub async fn feed_audio(&mut self, samples: &[f32]) -> u32 {
        // Ensure system prompt is prefilled (lazy, first call only)
        self.ensure_prefilled();

        let mimi = match self.mimi.as_mut() {
            Some(m) => m,
            None => {
                wasm_log("[sts] Warning: feedAudio called before Mimi loaded");
                return 0;
            }
        };

        // Append incoming PCM
        self.pcm_buffer.extend_from_slice(samples);

        let frame_size = 1920; // 80ms at 24kHz
        let num_codebooks = self.config.num_codebooks; // 8

        // Encode complete frames with Mimi and collect tokens
        let mut new_frames: Vec<Vec<u32>> = Vec::new();
        while self.pcm_buffer.len() >= frame_size {
            let frame_pcm: Vec<f32> = self.pcm_buffer.drain(..frame_size).collect();

            let t0 = now_ms();
            let tokens = mimi.encode(&frame_pcm);
            self.metrics.total_mimi_ms += now_ms() - t0;

            let mimi_codebooks = mimi.num_codebooks();
            if tokens.len() >= num_codebooks {
                let n_frames = tokens.len() / mimi_codebooks;
                for f in 0..n_frames {
                    let offset = f * mimi_codebooks;
                    let frame_tokens: Vec<u32> =
                        tokens[offset..offset + num_codebooks].to_vec();
                    new_frames.push(frame_tokens);
                }
            }
        }

        // Prefill each new frame incrementally through temporal+depth
        let num_new = new_frames.len() as u32;
        if !new_frames.is_empty() {
            let temporal = match self.temporal.as_ref() {
                Some(t) => t,
                None => return 0,
            };
            let depth = match self.depth.as_ref() {
                Some(d) => d,
                None => return 0,
            };
            let stream = match self.stream.as_mut() {
                Some(s) => s,
                None => return 0,
            };

            let t0 = now_ms();
            stream.prefill_user_audio(&new_frames, temporal, depth).await;
            let elapsed = now_ms() - t0;
            self.metrics.total_ms += elapsed;
        }

        num_new
    }

    /// Run voice preset + system prompt prefill (lazy, idempotent).
    fn ensure_prefilled(&mut self) {
        if self.prefilled {
            return;
        }

        let temporal = match self.temporal.as_ref() {
            Some(t) => t,
            None => return,
        };
        let stream = match self.stream.as_mut() {
            Some(s) => s,
            None => return,
        };

        let t0 = now_ms();

        // Voice preset (optional)
        if let (Some(embeddings), Some(cache_snapshot)) = (
            self.voice_preset_embeddings.as_ref(),
            self.voice_preset_cache.as_ref(),
        ) {
            wasm_log(&format!(
                "[sts] Running voice preset ({} frames)...",
                self.voice_preset_num_frames
            ));
            stream.prefill_voice_preset(
                embeddings,
                self.voice_preset_num_frames,
                cache_snapshot,
                temporal,
            );
            wasm_log("[sts] Voice preset complete");
        }

        wasm_log("[sts] Running system prompt prefill...");
        stream.prefill(temporal);
        self.metrics.total_temporal_ms += now_ms() - t0;
        self.prefilled = true;

        // Reset Mimi streaming state after prefill (matches Python)
        if let Some(mimi) = self.mimi.as_mut() {
            mimi.reset();
        }
        wasm_log("[sts] Prefill complete");
    }

    /// Prepare for autoregressive generation after user stops speaking.
    ///
    /// Call this after the last `feedAudio()`, before the `generateStep()` loop.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = generateStart))]
    pub fn generate_start(&mut self) {
        self.ensure_prefilled();
        self.generating = true;
        self.gen_step = 0;
        self.metrics.reset();
        wasm_log("[sts] Generation started");
    }

    /// Run one generation step. Returns a JS object with audio + text,
    /// or null if generation is complete (silence stop or max frames).
    ///
    /// JS usage:
    /// ```js
    /// engine.generateStart();
    /// while (true) {
    ///     const chunk = await engine.generateStep();
    ///     if (!chunk) break;
    ///     playAudio(chunk.audio);
    ///     showText(chunk.text);
    /// }
    /// ```
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = generateStep))]
    pub async fn generate_step(&mut self) -> JsValue {
        if !self.generating {
            return JsValue::NULL;
        }

        let temporal = match self.temporal.as_ref() {
            Some(t) => t,
            None => return JsValue::NULL,
        };
        let depth = match self.depth.as_ref() {
            Some(d) => d,
            None => return JsValue::NULL,
        };
        let stream = match self.stream.as_mut() {
            Some(s) => s,
            None => return JsValue::NULL,
        };

        // Run one step
        let t0 = now_ms();
        let output = stream.step(temporal, depth).await;
        let elapsed = now_ms() - t0;
        self.metrics.total_ms += elapsed;
        self.metrics.total_frames += 1;

        let i = self.gen_step;
        if i < 5 || i.is_multiple_of(50) {
            let toks: Vec<String> = output.model_audio_tokens.iter().map(|t| t.to_string()).collect();
            wasm_log(&format!(
                "[sts] Frame {i}: text={} audio=[{}] ({:.0}ms)",
                output.text_token, toks.join(","), elapsed
            ));
        }

        // Check if we should stop
        self.gen_step += 1;
        let done = stream.should_stop() || self.gen_step >= self.max_gen_frames;
        if done {
            if stream.should_stop() {
                wasm_log(&format!("[sts] Silence early stop at frame {i}"));
            }
            wasm_log(&format!(
                "[sts] Generated {} frames in {:.1}s",
                self.metrics.total_frames,
                self.metrics.total_ms / 1000.0
            ));
            self.generating = false;
        }

        // Decode audio with Mimi
        let mimi = match self.mimi.as_mut() {
            Some(m) => m,
            None => return JsValue::NULL,
        };
        let num_codebooks = self.config.num_codebooks;
        let t0 = now_ms();
        let pcm = mimi.decode_n(&output.model_audio_tokens, num_codebooks);
        self.metrics.total_mimi_ms += now_ms() - t0;

        // Build result: { audio: Float32Array, text: string|null, done: bool, step: number }
        let result = js_sys::Object::new();

        let audio = js_sys::Float32Array::new_with_length(pcm.len() as u32);
        audio.copy_from(&pcm);
        js_sys::Reflect::set(&result, &"audio".into(), &audio.into()).ok();

        // Decode text token
        let text_token = output.text_token;
        if text_token != self.config.text_padding_id {
            let text_str = if let Some(tok) = &self.tokenizer {
                tok.decode(&[text_token])
            } else {
                text_token.to_string()
            };
            if !text_str.is_empty() {
                js_sys::Reflect::set(&result, &"text".into(), &JsValue::from_str(&text_str)).ok();
            }
        }

        js_sys::Reflect::set(&result, &"done".into(), &JsValue::from_bool(done)).ok();
        js_sys::Reflect::set(&result, &"step".into(), &JsValue::from_f64(i as f64)).ok();

        result.into()
    }

    /// Run warmup passes to pre-compile WebGPU shader pipelines.
    ///
    /// Runs the prefill + a few generation steps to trigger shader compilation,
    /// then resets state so the real inference starts fresh.
    #[cfg_attr(target_family = "wasm", wasm_bindgen)]
    pub async fn warmup(&mut self) -> Result<(), JsError> {
        let temporal = self
            .temporal
            .as_ref()
            .ok_or_else(|| JsError::new("Model not loaded. Call loadModel first."))?;
        let depth = self
            .depth
            .as_ref()
            .ok_or_else(|| JsError::new("Model not loaded. Call loadModel first."))?;
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| JsError::new("Stream not initialized."))?;

        let t0 = now_ms();

        // Run prefill to warm up temporal transformer shaders
        stream.prefill(temporal);

        // Run a few generation steps to warm up depth transformer shaders
        for _ in 0..3 {
            stream.step(temporal, depth).await;
        }

        // Reset state but keep GPU KV cache buffers allocated
        stream.reset_keep_buffers();
        self.prefilled = false;
        self.metrics.reset();

        let elapsed = now_ms() - t0;
        wasm_log(&format!("[sts] Warmup complete ({elapsed:.0}ms)"));
        Ok(())
    }

    /// Get timing metrics from the session.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = getMetrics))]
    pub fn get_metrics(&self) -> JsValue {
        let obj = js_sys::Object::new();
        let set = |k: &str, v: f64| {
            js_sys::Reflect::set(&obj, &JsValue::from_str(k), &JsValue::from_f64(v)).ok();
        };
        set("temporal_ms", self.metrics.total_temporal_ms);
        set("depth_ms", self.metrics.total_depth_ms);
        set("mimi_ms", self.metrics.total_mimi_ms);
        set("total_ms", self.metrics.total_ms);
        set("total_frames", self.metrics.total_frames as f64);
        obj.into()
    }

    /// Reset all state for a new conversation session.
    #[cfg_attr(target_family = "wasm", wasm_bindgen)]
    pub fn reset(&mut self) {
        if let Some(stream) = &mut self.stream {
            stream.reset_keep_buffers();
        }
        if let Some(mimi) = &mut self.mimi {
            mimi.reset();
        }
        self.pcm_buffer.clear();
        self.prefilled = false;
        self.generating = false;
        self.gen_step = 0;
        self.metrics.reset();
    }

    /// Check if the model is loaded and ready.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = isReady))]
    pub fn is_ready(&self) -> bool {
        self.temporal.is_some() && self.depth.is_some() && self.mimi.is_some()
    }
}

impl Default for StsEngine {
    fn default() -> Self {
        Self::new()
    }
}
