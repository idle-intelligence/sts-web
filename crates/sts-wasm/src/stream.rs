//! Multi-stream inference loop for speech-to-speech.
//!
//! Manages the full STS pipeline per time step:
//! 1. Receive user audio tokens from Mimi encoder (8 codebooks)
//! 2. Feed user + model tokens + text token into temporal transformer
//! 3. Get hidden state + text logits
//! 4. Run depth transformer to generate model audio tokens
//! 5. Return model audio tokens for Mimi decoder + text token
//!
//! # Token Cache Layout
//!
//! A flat `TokenCache` with 17 streams and delay offsets (matching Swift reference):
//! - Tokens are written at `write_pos + delays[stream]`
//! - Tokens are read at `step - 1` for all streams
//! - Unwritten positions contain -1 (zero embedding contribution)
//!
//! ```text
//! Stream  | Content                    | Delay | Vocab offset
//! --------|----------------------------|-------|-------------
//!  0      | text                       |   0   | raw text tokens
//!  1      | model audio codebook 0     |   0   | + audio_offset (2048)
//!  2-8    | model audio codebooks 1-7  |   1   | + audio_offset
//!  9      | user audio codebook 0      |   0   | + audio_offset
//! 10-16   | user audio codebooks 1-7   |   1   | + audio_offset
//! ```
//!
//! # Counters: `write_pos` vs `step`
//!
//! - `write_pos`: where the next token will be written in the cache. Advances
//!   during both prefill and generation.
//! - `step`: the temporal transformer's step index. At each step, tokens at
//!   position `step - 1` are read from the cache, embedded, and fed to the
//!   transformer. Equals `write_pos` in normal operation.
//! - `prompt_len`: number of steps consumed by system prompt prefill
//!   (silence1 + text + silence2). User audio frames start at this offset.
//!
//! # Reset Methods
//!
//! - `reset_keep_buffers()`: full reset for a new conversation. Clears token
//!   cache, all counters, stop-detection state, and resets KV caches (keeps
//!   GPU buffers allocated). After this, `prefill()` must be called again.
//! - `reset_generation_state(pipelined_steps)`: partial reset between turns
//!   in the same conversation. Clears stop-detection counters, token histories,
//!   and rolls back any pre-submitted pipelined temporal cache entries. Does
//!   NOT touch the token cache, `write_pos`, `step`, or `prompt_len` — the KV
//!   cache retains the full conversation history.
//! - `undo_warmup_generation(gen_steps, pipelined_steps)`: rolls back warmup
//!   generation steps while preserving the prefill KV cache. Used by
//!   `StsEngine::warmup()` to compile shaders without polluting the cache.
//!
//! # Pipelining
//!
//! After each generation step's depformer finishes, the NEXT frame's temporal
//! forward pass is pre-submitted to the GPU (`pending_temporal`). This lets the
//! GPU compute while the caller runs Mimi decode on the CPU (~35ms overlap).
//! The pipelined result is consumed at the start of the next `step()` call.

use burn::backend::wgpu::{Wgpu, WgpuRuntime};
use burn::tensor::Tensor;
use cubecl::Runtime;

use crate::depth::DepthTransformer;
use crate::gguf::{gpu_argmax_buffer, gpu_sample_top_k_with_penalty};
#[cfg(feature = "wasm")]
use crate::gguf::gpu_read_token_tensor;
use crate::model::{LayerCaches, TemporalTransformer};
use crate::StsConfig;

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

/// Output of a single STS inference step.
pub struct StepOutput {
    /// Model audio codebook tokens (8 tokens, one per codebook).
    /// These are fed into the Mimi decoder to produce audio.
    pub model_audio_tokens: Vec<u32>,
    /// Text token from inner monologue (always produced).
    pub text_token: u32,
    /// Time spent in temporal transformer forward pass (ms).
    pub temporal_ms: f64,
    /// Time spent in depth transformer forward pass (ms).
    pub depth_ms: f64,
}

/// Flat token cache with delay offsets.
///
/// Mirrors Swift's `tokenCache[stream][position]`:
/// - `num_streams` = 17 (1 text + 8 agent + 8 user)
/// - Tokens written at `pos + delays[stream]`
/// - Tokens read at `step - 1`
/// - Unwritten = -1 (masked to zero in embedding sum)
struct TokenCache {
    /// cache[stream * max_len + position] = token (i32, -1 = unwritten)
    data: Vec<i32>,
    num_streams: usize,
    max_len: usize,
}

impl TokenCache {
    fn new(num_streams: usize, max_len: usize) -> Self {
        Self {
            data: vec![-1i32; num_streams * max_len],
            num_streams,
            max_len,
        }
    }

    /// Write a token to cache[stream][position]. No-op if out of bounds.
    fn write(&mut self, stream: usize, position: usize, token: i32) {
        if stream < self.num_streams && position < self.max_len {
            self.data[stream * self.max_len + position] = token;
        }
    }

    /// Read a token from cache[stream][position]. Returns -1 if out of bounds.
    fn read(&self, stream: usize, position: usize) -> i32 {
        if stream < self.num_streams && position < self.max_len {
            self.data[stream * self.max_len + position]
        } else {
            -1
        }
    }

    /// Read all 17 stream tokens at a given position.
    /// Returns (text_token, model_audio[8], user_audio[8]) as i32.
    fn read_all(&self, position: usize) -> (i32, Vec<i32>, Vec<i32>) {
        let text = self.read(0, position);
        let nq = (self.num_streams - 1) / 2; // 8
        let model: Vec<i32> = (0..nq).map(|cb| self.read(1 + cb, position)).collect();
        let user: Vec<i32> = (0..nq).map(|cb| self.read(1 + nq + cb, position)).collect();
        (text, model, user)
    }

    fn reset(&mut self) {
        self.data.fill(-1);
    }
}

/// Streaming STS decoder implementing the multi-stream inference loop.
pub struct StsStream {
    config: StsConfig,
    temporal_cache: LayerCaches,
    depth_cache: LayerCaches,

    /// Token cache: tokenCache[stream][position], matching Swift reference.
    token_cache: TokenCache,

    /// Current write position (next position to write tokens at).
    /// This is the `pos` counter that advances through prefill and generation.
    write_pos: usize,

    /// Current step (temporal transformer step index).
    /// At each step, we read `token_cache[stream][step - 1]` and run the transformer.
    step: usize,

    /// Length of the non-voice prefill (silence1 + text + silence2).
    prompt_len: usize,

    /// Length of the full prefill (prompt_len + user_audio_frames).
    /// Currently informational only — used for debugging.
    _prefill_len: usize,

    /// Track last model audio tokens for silence detection
    last_model_audio_tokens: Vec<u32>,

    // Sampling parameters
    text_temperature: f32,
    text_top_k: usize,
    audio_temperature: f32,
    audio_top_k: usize,

    // Repetition penalty
    text_token_history: Vec<u32>,
    /// Per-codebook audio token history for repetition penalty.
    audio_token_history: Vec<Vec<u32>>,
    repetition_penalty: f32,
    penalty_window: usize,

    // Silence early stopping
    consecutive_silence_frames: usize,
    silence_early_stop_frames: usize,

    // Text-based stopping: stop if text has been padding for N frames after real text
    consecutive_text_pad_frames: usize,
    text_pad_stop_frames: usize,
    has_generated_text: bool,

    /// Pre-submitted temporal result for pipelining.
    /// After depformer finishes, we enqueue the NEXT frame's temporal forward
    /// so the GPU works while the caller does Mimi decode on CPU (~40ms overlap).
    pending_temporal: Option<(Tensor<Wgpu, 3>, Tensor<Wgpu, 3>)>,

    /// Wall-clock time for the pre-submitted temporal forward (ms).
    /// Saved when the work is enqueued, reported on the next step.
    pending_temporal_ms: f64,

    /// Accumulated temporal_ms during prefill_user_audio.
    pub prefill_temporal_ms: f64,
    /// Accumulated depth_ms during prefill_user_audio.
    pub prefill_depth_ms: f64,

    /// Custom WebGPU depth engine (WASM only). When set, bypasses Burn/CubeCL
    /// depth path in step() and submits all 8 depth steps as one command buffer.
    #[cfg(feature = "wasm")]
    pub depth_engine: Option<crate::depth_engine::DepthEngine>,
}

impl StsStream {
    pub fn new(
        config: StsConfig,
        temporal_cache: LayerCaches,
        depth_cache: LayerCaches,
    ) -> Self {
        // Max length: generous upper bound for token cache.
        // prompt (~50) + user audio (~100) + generation (~500) + delay margin
        let max_len = config.max_seq_len + 10;
        let num_streams = 1 + 2 * config.num_codebooks; // 17
        let token_cache = TokenCache::new(num_streams, max_len);

        Self {
            last_model_audio_tokens: config.silence_tokens.to_vec(),
            write_pos: 0,
            step: 0,
            prompt_len: 0,
            _prefill_len: 0,
            text_temperature: 0.7,
            text_top_k: 25,
            audio_temperature: 0.8,
            audio_top_k: 250,
            text_token_history: Vec::new(),
            audio_token_history: vec![Vec::new(); config.depth_num_steps],
            repetition_penalty: 1.2,
            penalty_window: 30,
            consecutive_silence_frames: 0,
            silence_early_stop_frames: 8, // ~0.64s of silence triggers stop
            consecutive_text_pad_frames: 0,
            text_pad_stop_frames: 6, // ~0.5s of text padding after real text → stop
            has_generated_text: false,
            pending_temporal: None,
            pending_temporal_ms: 0.0,
            prefill_temporal_ms: 0.0,
            prefill_depth_ms: 0.0,
            #[cfg(feature = "wasm")]
            depth_engine: None,
            config,
            temporal_cache,
            depth_cache,
            token_cache,
        }
    }

    /// Set sampling parameters.
    pub fn set_sampling_params(
        &mut self,
        text_temperature: f32,
        text_top_k: usize,
        audio_temperature: f32,
        audio_top_k: usize,
    ) {
        self.text_temperature = text_temperature;
        self.text_top_k = text_top_k;
        self.audio_temperature = audio_temperature;
        self.audio_top_k = audio_top_k;
    }

    // -----------------------------------------------------------------------
    // Token cache helpers
    // -----------------------------------------------------------------------

    /// Write prompt-phase tokens into the cache at position `pos` with delays.
    ///
    /// text_token, silence/sine for audio — written at pos + delays[stream].
    fn write_prompt_tokens(&mut self, pos: usize, text_token: i32) {
        let delays = &self.config.delays;
        let nq = self.config.num_codebooks;

        // Text
        self.token_cache.write(0, pos + delays[0], text_token);

        // Agent audio: silence tokens
        for cb in 0..nq {
            let stream = 1 + cb;
            self.token_cache.write(stream, pos + delays[stream], self.config.silence_tokens[cb] as i32);
        }

        // User audio: sine tokens
        for cb in 0..nq {
            let stream = 1 + nq + cb;
            self.token_cache.write(stream, pos + delays[stream], self.config.sine_tokens[cb] as i32);
        }
    }

    /// Write user audio tokens into the cache at position `pos` with delays.
    ///
    /// text = padding, agent = silence, user = real Mimi tokens.
    fn write_user_audio_tokens(&mut self, pos: usize, user_tokens: &[u32]) {
        let delays = &self.config.delays;
        let nq = self.config.num_codebooks;

        // Text: padding
        self.token_cache.write(0, pos + delays[0], self.config.text_padding_id as i32);

        // Agent audio: silence tokens
        for cb in 0..nq {
            let stream = 1 + cb;
            self.token_cache.write(stream, pos + delays[stream], self.config.silence_tokens[cb] as i32);
        }

        // User audio: real Mimi tokens
        for (cb, &tok) in user_tokens.iter().enumerate().take(nq) {
            let stream = 1 + nq + cb;
            self.token_cache.write(stream, pos + delays[stream], tok as i32);
        }
    }

    /// Write depformer output tokens into the cache at position `step` (NO delay).
    ///
    /// Critical: depformer outputs are written without delay offset,
    /// matching Swift/Python behavior. This ensures the next step immediately
    /// reads all depformer outputs regardless of their stream's delay value.
    fn write_depformer_output(
        &mut self,
        step: usize,
        text_token: u32,
        agent_tokens: &[u32],
        user_tokens: Option<&[u32]>,
    ) {
        let nq = self.config.num_codebooks;

        // Text: no delay
        self.token_cache.write(0, step, text_token as i32);

        // Agent audio (streams 1-8): no delay
        for (cb, &tok) in agent_tokens.iter().enumerate() {
            self.token_cache.write(1 + cb, step, tok as i32);
        }

        // User audio predictions (streams 9-16): only during generation
        // (during user audio prefill, real user audio stays in cache)
        if let Some(user) = user_tokens {
            for (cb, &tok) in user.iter().enumerate() {
                self.token_cache.write(1 + nq + cb, step, tok as i32);
            }
        }
    }

    /// Read input tokens for step `step` from the cache.
    /// Returns (text_token, model_audio[8], user_audio[8]) as i32.
    fn read_input_tokens(&self, step: usize) -> (i32, Vec<i32>, Vec<i32>) {
        if step == 0 {
            // At step 0, use initial/BOS tokens (matching Python's _get_initial_token):
            // text = text_initial_token_id (32000), all audio = initial_token_id (2048)
            let nq = self.config.num_codebooks;
            let audio_bos = self.config.audio_initial_token_id as i32;
            (
                self.config.text_start_token as i32,
                vec![audio_bos; nq],
                vec![audio_bos; nq],
            )
        } else {
            self.token_cache.read_all(step - 1)
        }
    }

    // -----------------------------------------------------------------------
    // Prefill: system prompt + silence spacers
    // -----------------------------------------------------------------------

    /// Prefill voice preset from pre-computed embeddings.
    ///
    /// Feeds `num_frames` pre-computed embeddings (from a voice preset file
    /// like NATF2.pt) through the temporal transformer to build KV cache state.
    /// Then overwrites the token cache with the saved cache snapshot.
    ///
    /// The depformer is NOT run during voice preset — the token cache is
    /// entirely replaced by the snapshot, so depformer outputs would be wasted.
    ///
    /// Must be called BEFORE `prefill()`.
    pub fn prefill_voice_preset(
        &mut self,
        embeddings: &[f32],
        num_frames: usize,
        cache_snapshot: &[Vec<i32>],
        temporal: &TemporalTransformer,
    ) {
        let dim = self.config.hidden_size;
        assert_eq!(embeddings.len(), num_frames * dim);

        // Process each embedding through temporal transformer (builds KV cache)
        for frame in 0..num_frames {
            let offset = frame * dim;
            let emb_data = embeddings[offset..offset + dim].to_vec();
            let input = burn::tensor::Tensor::<burn::backend::wgpu::Wgpu, 3>::from_data(
                burn::tensor::TensorData::new(emb_data, [1, 1, dim]),
                temporal.device(),
            );
            let (_hidden, _text_logits) =
                temporal.forward_embeddings(input, &mut self.temporal_cache);
        }

        self.step = num_frames;
        self.write_pos = num_frames;

        // Overwrite token cache with saved snapshot.
        //
        // Python uses a circular cache of size 4, indexed by offset % 4.
        // After 51 step_embeddings calls, Python's offset = num_frames + 1 = 52
        // (includes the init step at offset 0 that bumps to 1 without inference).
        //
        // Map the 4 circular positions back to our flat positions:
        //   most recent (k=0): python_pos = (offset_end - 1) % 4 → rust_pos = step - 1
        //   k-th back:         python_pos = (offset_end - 1 - k) % 4 → rust_pos = step - 1 - k
        let num_snapshot_positions = cache_snapshot.first().map(|v| v.len()).unwrap_or(0);
        let offset_end = num_frames + 1; // Python's state.offset after voice preset

        for k in 0..num_snapshot_positions {
            let python_pos = (offset_end - 1 - k) % num_snapshot_positions;
            let rust_pos = self.step - 1 - k;
            for (stream, snapshot_stream) in cache_snapshot.iter().enumerate() {
                if stream < self.token_cache.num_streams {
                    self.token_cache
                        .write(stream, rust_pos, snapshot_stream[python_pos]);
                }
            }
        }
    }

    /// Run the system prompt prefill sequence.
    ///
    /// This populates the token cache and feeds the temporal transformer with:
    ///   Phase 2: silence spacer (0.5s)
    ///   Phase 3: system prompt text tokens
    ///   Phase 4: silence spacer (0.5s)
    ///
    /// If voice preset was applied first, this continues from the current
    /// step/write_pos. Otherwise starts from 0.
    pub fn prefill(
        &mut self,
        temporal: &TemporalTransformer,
    ) {
        let silence_frames = (0.5 * self.config.frame_rate) as usize; // 6
        let prompt_tokens = self.config.system_prompt_tokens.clone();
        let prompt_len = silence_frames + prompt_tokens.len() + silence_frames;

        // Phase 2: Silence spacer 1
        let mut pos = self.write_pos;
        for _ in 0..silence_frames {
            self.write_prompt_tokens(pos, self.config.text_padding_id as i32);
            pos += 1;
        }

        // Phase 3: System prompt text tokens
        for &tok in &prompt_tokens {
            self.write_prompt_tokens(pos, tok as i32);
            pos += 1;
        }

        // Phase 4: Silence spacer 2
        for _ in 0..silence_frames {
            self.write_prompt_tokens(pos, self.config.text_padding_id as i32);
            pos += 1;
        }

        self.write_pos = pos;
        self.prompt_len = self.step + prompt_len;

        // Now run temporal transformer for each step to build KV cache
        let start_step = self.step;
        for step in start_step..start_step + prompt_len {
            let (text, model, user) = self.read_input_tokens(step);
            let (_hidden, _logits) = temporal.forward(
                &user,
                &model,
                text,
                &mut self.temporal_cache,
            );
        }

        self.step = start_step + prompt_len;
    }

    // -----------------------------------------------------------------------
    // Phase 5: User audio prefill
    // -----------------------------------------------------------------------

    /// Feed user audio frames through temporal transformer + depformer.
    ///
    /// Each frame is [codebook_0, ..., codebook_7] from Mimi encoder.
    pub async fn prefill_user_audio(
        &mut self,
        user_audio_frames: &[Vec<u32>],
        temporal: &TemporalTransformer,
        depth: &DepthTransformer,
    ) {
        // Write all user audio tokens into cache with delays
        let start_pos = self.write_pos;
        for (i, frame) in user_audio_frames.iter().enumerate() {
            self.write_user_audio_tokens(start_pos + i, frame);
        }
        self.write_pos = start_pos + user_audio_frames.len();
        self._prefill_len = self.write_pos;

        // Process each user audio step through temporal + depformer
        let nq = self.config.num_codebooks;
        self.prefill_temporal_ms = 0.0;
        self.prefill_depth_ms = 0.0;
        for _i in 0..user_audio_frames.len() {
            let step = self.step;
            let (text, model, user) = self.read_input_tokens(step);

            // Temporal forward
            let t_temporal = now_ms();
            let (hidden, _text_logits) = temporal.forward(
                &user,
                &model,
                text,
                &mut self.temporal_cache,
            );
            self.prefill_temporal_ms += now_ms() - t_temporal;

            // Force text=PAD during user audio prefill (matches Python)
            let text_token = self.config.text_padding_id;

            // Build provided tokens: read user audio from cache at position `step`
            // (matches Swift: tokenCache[userStreamIdx][step])
            let mut provided = vec![-1i32; self.config.depth_num_steps];
            for cb in 0..nq {
                let stream = 1 + nq + cb;
                let tok = self.token_cache.read(stream, step);
                if tok >= 0 {
                    provided[nq + cb] = tok;
                }
            }

            // Run depformer
            let t_depth = now_ms();
            self.depth_cache.reset_keep_buffers();
            let all_audio_tokens = depth.generate(
                hidden,
                text_token,
                &mut self.depth_cache,
                self.audio_temperature,
                self.audio_top_k,
                Some(&provided),
                None, // No penalty during prefill
                1.0,
                Some(self.config.depth_gen_steps), // Only agent steps; user predictions discarded
            )
            .await;
            self.prefill_depth_ms += now_ms() - t_depth;

            // Write depformer output to cache at position `step` (no delay).
            // Agent audio tokens are written; user audio predictions are NOT written
            // during user audio prefill (real user audio stays in cache).
            let agent = &all_audio_tokens[..nq.min(all_audio_tokens.len())];
            self.write_depformer_output(step, text_token, agent, None);

            // Update last model audio for silence detection
            self.last_model_audio_tokens = agent.to_vec();
            self.last_model_audio_tokens.resize(nq, 0);

            self.step += 1;
        }
    }

    // -----------------------------------------------------------------------
    // Generation
    // -----------------------------------------------------------------------

    /// Process one generation step (one 80ms frame at 12.5 Hz).
    ///
    /// Uses pipelining: after depformer finishes, enqueues the NEXT frame's
    /// temporal forward pass on the GPU and flushes commands. The caller can
    /// then do Mimi decode (CPU, ~40ms) while the GPU computes temporal (~113ms),
    /// saving ~35ms per frame.
    pub async fn step(
        &mut self,
        temporal: &TemporalTransformer,
        depth: &DepthTransformer,
    ) -> StepOutput {
        let nq = self.config.num_codebooks;
        let step = self.step;

        // Step 1: Get temporal result — either from pre-submitted pipeline
        // (GPU already computed while caller was doing Mimi decode) or compute now.
        let (hidden, text_logits, temporal_ms) = if let Some(pending) = self.pending_temporal.take() {
            let ms = self.pending_temporal_ms;
            self.pending_temporal_ms = 0.0;
            let (h, l) = pending;
            (h, l, ms)
        } else {
            let t0 = now_ms();
            let (text, model, user) = self.read_input_tokens(step);
            let result = temporal.forward(&user, &model, text, &mut self.temporal_cache);
            (result.0, result.1, now_ms() - t0)
        };

        // Step 2: Submit text token sampling to GPU (DON'T await yet).
        // The argmax/sampling kernel is enqueued on GPU. We keep both the Handle
        // (for GPU-side text embedding in depth step 0) and the Tensor (for
        // batch readback at the end). No CPU readback happens here.
        let (text_token_handle, text_token_tensor) = if self.text_temperature <= 0.0 {
            gpu_argmax_buffer(text_logits)
        } else {
            let text_history: Vec<u32> = self
                .text_token_history
                .iter()
                .rev()
                .take(self.penalty_window)
                .copied()
                .collect();
            gpu_sample_top_k_with_penalty(
                text_logits,
                self.text_top_k,
                self.text_temperature,
                &text_history,
                self.repetition_penalty,
            )
        };

        // Step 3: Reset depth cache for this time step
        self.depth_cache.reset_keep_buffers();

        // Step 4: Depth transformer generates agent audio tokens (skip user predictions).
        let gen_steps = self.config.depth_gen_steps;
        let t_depth = now_ms();

        // Custom DepthEngine path (WASM only): submits all 8 depth steps as one
        // command buffer, bypassing Burn/CubeCL overhead (~150 dispatches → 1).
        #[cfg(feature = "wasm")]
        let (mut all_audio_tokens, text_token) = if let Some(engine) = self.depth_engine.as_ref() {
            // Flush CubeCL pending work so buffers are valid for raw wgpu access.
            let device = temporal.device();
            let client = WgpuRuntime::client(device);
            client.flush();

            // Extract WgpuResource from the temporal hidden Tensor<Wgpu, 3>.
            let hidden_handle = hidden.into_primitive().tensor().handle;
            let hidden_resource = crate::gguf::handle_to_wgpu_resource(&hidden_handle, device);

            // Extract WgpuResource from the text token Handle.
            let text_resource = crate::gguf::handle_to_wgpu_resource(&text_token_handle, device);

            // Reset the custom engine's KV caches for this frame.
            engine.reset_caches();

            // Run all 8 depth steps in a single command buffer.
            let penalty_hist = &self.audio_token_history;
            let audio_tokens = engine.generate(
                &hidden_resource,
                &text_resource,
                self.audio_temperature,
                self.audio_top_k,
                self.repetition_penalty,
                Some(penalty_hist),
            ).await;

            // Read back text token separately (the custom engine only returns audio).
            let text_token_val = gpu_read_token_tensor(text_token_tensor).await;

            (audio_tokens, text_token_val)
        } else {
            // Standard Burn/CubeCL depth path (WASM fallback).
            let penalty_hist = &self.audio_token_history;
            depth.generate_deferred(
                hidden,
                text_token_tensor,
                text_token_handle,
                &mut self.depth_cache,
                self.audio_temperature,
                self.audio_top_k,
                None, // No provided tokens during generation
                Some(penalty_hist),
                self.repetition_penalty,
                Some(gen_steps), // Only run agent audio steps, skip user predictions
            )
            .await
        };
        #[cfg(not(feature = "wasm"))]
        let (mut all_audio_tokens, text_token) = {
            // Standard Burn/CubeCL depth path (native).
            let penalty_hist = &self.audio_token_history;
            depth.generate_deferred(
                hidden,
                text_token_tensor,
                text_token_handle,
                &mut self.depth_cache,
                self.audio_temperature,
                self.audio_top_k,
                None, // No provided tokens during generation
                Some(penalty_hist),
                self.repetition_penalty,
                Some(gen_steps), // Only run agent audio steps, skip user predictions
            )
            .await
        };
        let depth_ms = now_ms() - t_depth;

        // Update text token history (was deferred from step 2)
        self.text_token_history.push(text_token);
        if self.text_token_history.len() > self.penalty_window {
            self.text_token_history.drain(..self.text_token_history.len() - self.penalty_window);
        }

        // Fill remaining positions (user audio predictions) with sine tokens.
        // These are unused for Mimi decode but needed in the token cache for
        // conditioning the next temporal step.
        let full_steps = self.config.depth_num_steps;
        if all_audio_tokens.len() < full_steps {
            for i in all_audio_tokens.len()..full_steps {
                // Steps 0-7 are agent audio, steps 8-15 are user audio.
                // User codebook index = i - nq (maps to sine_tokens[0..7]).
                let user_cb = i - nq;
                let sine_tok = if user_cb < self.config.sine_tokens.len() {
                    self.config.sine_tokens[user_cb]
                } else {
                    self.config.sine_tokens[0] // fallback
                };
                all_audio_tokens.push(sine_tok);
            }
        }

        // Split depformer output into agent (0-7) and user (8-15)
        let agent_end = nq.min(all_audio_tokens.len());
        let agent = all_audio_tokens[..agent_end].to_vec();
        let user_start = nq;
        let user_end = (2 * nq).min(all_audio_tokens.len());
        let user_pred = if user_end > user_start {
            Some(&all_audio_tokens[user_start..user_end])
        } else {
            None
        };

        // Write depformer output to cache at position `step` (no delay).
        self.write_depformer_output(step, text_token, &agent, user_pred);

        // Update per-codebook audio token history for repetition penalty
        for (s, &token) in all_audio_tokens.iter().enumerate() {
            if s < self.audio_token_history.len() {
                let hist = &mut self.audio_token_history[s];
                hist.push(token);
                if hist.len() > self.penalty_window {
                    hist.drain(..hist.len() - self.penalty_window);
                }
            }
        }

        // Build output
        let mut model_audio_out = agent.clone();
        model_audio_out.resize(nq, 0);
        self.last_model_audio_tokens = model_audio_out.clone();

        self.step += 1;

        // Track silence for early stopping
        if self.is_silence() {
            self.consecutive_silence_frames += 1;
        } else {
            self.consecutive_silence_frames = 0;
        }

        // Track text padding for early stopping
        if text_token == self.config.text_padding_id {
            if self.has_generated_text {
                self.consecutive_text_pad_frames += 1;
            }
        } else {
            self.has_generated_text = true;
            self.consecutive_text_pad_frames = 0;
        }

        // Step 5: Pipeline — pre-submit NEXT frame's temporal forward pass.
        // The GPU starts computing while the caller does Mimi decode on CPU.
        // We enqueue the work and flush to ensure the GPU command is submitted.
        {
            let next_step = self.step;
            let (text, model, user) = self.read_input_tokens(next_step);
            let t_pipe = now_ms();
            let result = temporal.forward(&user, &model, text, &mut self.temporal_cache);
            self.pending_temporal_ms = now_ms() - t_pipe;
            self.pending_temporal = Some(result);

            // Flush GPU command buffer so the temporal work starts immediately.
            // Without this, the commands sit in the encoder until the next readback,
            // which would be too late for pipelining.
            let client = WgpuRuntime::client(temporal.device());
            client.flush();
        }

        StepOutput {
            model_audio_tokens: model_audio_out,
            text_token,
            temporal_ms,
            depth_ms,
        }
    }

    // -----------------------------------------------------------------------
    // Silence detection and early stopping
    // -----------------------------------------------------------------------

    /// Check if the last model output is silence (all 8 codebooks match silence tokens).
    pub fn is_silence(&self) -> bool {
        self.last_model_audio_tokens.len() >= 2
            && self.last_model_audio_tokens[1] == self.config.silence_tokens[1]
    }

    /// Check if generation should stop due to sustained silence or text completion.
    pub fn should_stop(&self) -> bool {
        self.consecutive_silence_frames >= self.silence_early_stop_frames
    }

    /// Get the number of consecutive silence frames so far.
    pub fn consecutive_silence_frames(&self) -> usize {
        self.consecutive_silence_frames
    }

    /// Get the number of consecutive text padding frames after real text.
    pub fn consecutive_text_pad_frames(&self) -> usize {
        self.consecutive_text_pad_frames
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Get the current frame count (steps since start of user audio + generation).
    pub fn frame_count(&self) -> usize {
        self.step.saturating_sub(self.prompt_len)
    }

    /// Get the last generated text token.
    pub fn last_text_token(&self) -> u32 {
        // Read the text token from the most recently written step
        if self.step > 0 {
            let tok = self.token_cache.read(0, self.step - 1);
            if tok >= 0 { tok as u32 } else { self.config.text_padding_id }
        } else {
            self.config.text_start_token
        }
    }

    /// Get the last generated model audio tokens.
    pub fn last_model_audio_tokens(&self) -> &[u32] {
        &self.last_model_audio_tokens
    }

    /// Reset all state for a new conversation.
    pub fn reset(&mut self) {
        self.token_cache.reset();
        self.write_pos = 0;
        self.step = 0;
        self.prompt_len = 0;
        self._prefill_len = 0;
        self.last_model_audio_tokens = self.config.silence_tokens.to_vec();
        self.text_token_history.clear();
        for h in &mut self.audio_token_history { h.clear(); }
        self.consecutive_silence_frames = 0;
        self.consecutive_text_pad_frames = 0;
        self.has_generated_text = false;
        self.pending_temporal = None;
        self.pending_temporal_ms = 0.0;
        self.prefill_temporal_ms = 0.0;
        self.prefill_depth_ms = 0.0;
        self.temporal_cache.reset();
        self.depth_cache.reset();
    }

    /// Undo warmup generation steps, keeping the prefill KV cache intact.
    ///
    /// After warmup runs prefill + N generation steps (to compile depth shaders),
    /// this rolls back the generation state so the next `feedAudio()` or
    /// `generateStart()` continues from where prefill ended.
    ///
    /// `gen_steps` is the number of generation steps to undo (e.g. 3).
    /// `pipelined_steps` is the number of extra temporal forward passes that
    /// were pre-submitted by the pipelining logic (typically 1).
    pub fn undo_warmup_generation(&mut self, gen_steps: usize, pipelined_steps: usize) {
        // Roll back the temporal KV cache: generation steps + pipelined step
        let total_temporal_rollback = gen_steps + pipelined_steps;
        self.temporal_cache.rollback(total_temporal_rollback);

        // Clear token cache entries written during generation
        // (depformer writes at positions step-gen_steps..step without delay)
        let gen_start = self.step - gen_steps;
        for pos in gen_start..self.step {
            for stream in 0..self.token_cache.num_streams {
                self.token_cache.write(stream, pos, -1);
            }
        }

        // Roll back step and write_pos to where prefill ended
        self.step = gen_start;
        self.write_pos = gen_start;

        // Clear generation-related state
        self.last_model_audio_tokens = self.config.silence_tokens.to_vec();
        self.text_token_history.clear();
        for h in &mut self.audio_token_history { h.clear(); }
        self.consecutive_silence_frames = 0;
        self.consecutive_text_pad_frames = 0;
        self.has_generated_text = false;
        self.pending_temporal = None;
        self.pending_temporal_ms = 0.0;
        self.prefill_temporal_ms = 0.0;
        self.prefill_depth_ms = 0.0;

        // Depth cache is reset every step anyway, just clear it
        self.depth_cache.reset_keep_buffers();
    }

    /// Reset generation-related state for a new turn, keeping the KV cache
    /// and token cache intact so the conversation continues.
    ///
    /// Clears: silence/text-pad counters, token histories, pending pipelined
    /// temporal result. Also rolls back the temporal cache by `pipelined_steps`
    /// entries to undo the pre-submitted temporal forward from pipelining.
    pub fn reset_generation_state(&mut self, pipelined_steps: usize) {
        self.consecutive_silence_frames = 0;
        self.consecutive_text_pad_frames = 0;
        self.has_generated_text = false;
        self.text_token_history.clear();
        for h in &mut self.audio_token_history { h.clear(); }
        self.last_model_audio_tokens = self.config.silence_tokens.to_vec();

        // Drop stale pipelined temporal result and roll back its cache entry
        if self.pending_temporal.take().is_some() && pipelined_steps > 0 {
            self.temporal_cache.rollback(pipelined_steps);
        }
        self.pending_temporal_ms = 0.0;
        self.prefill_temporal_ms = 0.0;
        self.prefill_depth_ms = 0.0;

        // Sync write_pos to step so the next prefill_user_audio writes tokens
        // at the correct position (after generated frames, not after turn 1's
        // user audio). Without this, turn 2's user audio overwrites turn 1's
        // generated tokens and temporal reads from the wrong positions.
        self.write_pos = self.step;
    }

    /// Reset state but keep GPU KV cache buffers allocated.
    pub fn reset_keep_buffers(&mut self) {
        self.token_cache.reset();
        self.write_pos = 0;
        self.step = 0;
        self.prompt_len = 0;
        self._prefill_len = 0;
        self.last_model_audio_tokens = self.config.silence_tokens.to_vec();
        self.text_token_history.clear();
        for h in &mut self.audio_token_history { h.clear(); }
        self.consecutive_silence_frames = 0;
        self.consecutive_text_pad_frames = 0;
        self.has_generated_text = false;
        self.pending_temporal = None;
        self.pending_temporal_ms = 0.0;
        self.prefill_temporal_ms = 0.0;
        self.prefill_depth_ms = 0.0;
        self.temporal_cache.reset_keep_buffers();
        self.depth_cache.reset_keep_buffers();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::LayerCaches;
    use burn::backend::wgpu::WgpuDevice;

    fn test_device() -> WgpuDevice {
        WgpuDevice::default()
    }

    fn make_stream(config: &StsConfig) -> StsStream {
        let device = test_device();
        let head_dim = config.hidden_size / config.num_heads;
        let temporal_cache = LayerCaches::new(
            config.num_layers,
            config.max_seq_len,
            config.num_kv_heads,
            head_dim,
            &device,
        );
        let depth_head_dim = config.depth_hidden_size / config.depth_num_heads;
        let depth_cache = LayerCaches::new(
            config.depth_num_layers,
            config.depth_num_steps + 1,
            config.depth_num_kv_heads,
            depth_head_dim,
            &device,
        );
        StsStream::new(config.clone(), temporal_cache, depth_cache)
    }

    #[test]
    fn test_stream_creation_and_reset() {
        let config = StsConfig::default();
        let mut stream = make_stream(&config);

        assert_eq!(stream.frame_count(), 0);
        assert_eq!(stream.last_model_audio_tokens().len(), config.num_codebooks);
        assert_eq!(stream.last_model_audio_tokens(), &config.silence_tokens[..]);

        // Test reset
        stream.step = 5;
        stream.last_model_audio_tokens = vec![100; 8];
        stream.reset();
        assert_eq!(stream.frame_count(), 0);
        assert_eq!(stream.last_model_audio_tokens(), &config.silence_tokens[..]);
    }

    #[test]
    fn test_token_cache_delays() {
        // Verify that the token cache correctly applies delays.
        let config = StsConfig::default();
        let mut stream = make_stream(&config);
        let delays = config.delays;

        // Write prompt tokens at position 0
        stream.write_prompt_tokens(0, config.text_padding_id as i32);

        // Text (delay=0): should be written at position 0
        assert_eq!(stream.token_cache.read(0, 0), config.text_padding_id as i32);

        // Agent cb0 (stream 1, delay=0): silence at position 0
        assert_eq!(stream.token_cache.read(1, 0), config.silence_tokens[0] as i32);

        // Agent cb1 (stream 2, delay=1): silence at position 1
        assert_eq!(stream.token_cache.read(2, 0), -1); // NOT written at 0
        assert_eq!(stream.token_cache.read(2, 1), config.silence_tokens[1] as i32);

        // User cb0 (stream 9, delay=0): sine at position 0
        assert_eq!(stream.token_cache.read(9, 0), config.sine_tokens[0] as i32);

        // User cb1 (stream 10, delay=1): sine at position 1
        assert_eq!(stream.token_cache.read(10, 0), -1); // NOT at 0
        assert_eq!(stream.token_cache.read(10, 1), config.sine_tokens[1] as i32);

        // Verify delays match expected pattern
        assert_eq!(delays[0], 0);  // text
        assert_eq!(delays[1], 0);  // agent cb0
        assert_eq!(delays[2], 1);  // agent cb1
        assert_eq!(delays[9], 0);  // user cb0
        assert_eq!(delays[10], 1); // user cb1
    }

    #[test]
    fn test_read_input_step0() {
        // At step 0, read_input_tokens returns initial/BOS tokens
        // (matching Python's _get_initial_token: text=32000, audio=2048)
        let config = StsConfig::default();
        let stream = make_stream(&config);

        let (text, model, user) = stream.read_input_tokens(0);
        assert_eq!(text, config.text_start_token as i32); // 32000
        assert!(model.iter().all(|&t| t == config.audio_initial_token_id as i32)); // 2048
        assert!(user.iter().all(|&t| t == config.audio_initial_token_id as i32)); // 2048
    }

    #[test]
    fn test_read_input_after_prompt_write() {
        let config = StsConfig::default();
        let mut stream = make_stream(&config);

        // Write prompt tokens at position 0 and 1
        stream.write_prompt_tokens(0, config.text_padding_id as i32);
        stream.write_prompt_tokens(1, 42); // some text token

        // At step 1, read position 0:
        // - text (delay=0): padding (written at pos 0)
        // - agent cb0 (delay=0): silence (written at pos 0)
        // - agent cb1 (delay=1): -1 (written at pos 0+1=1, but we read pos 0)
        // - user cb0 (delay=0): sine (written at pos 0)
        // - user cb1 (delay=1): -1 (written at pos 0+1=1, but we read pos 0)
        let (text, model, user) = stream.read_input_tokens(1);
        assert_eq!(text, config.text_padding_id as i32);
        assert_eq!(model[0], config.silence_tokens[0] as i32); // delay=0
        assert_eq!(model[1], -1); // delay=1, not visible yet
        assert_eq!(user[0], config.sine_tokens[0] as i32); // delay=0
        assert_eq!(user[1], -1); // delay=1, not visible yet

        // At step 2, read position 1:
        // - text: 42 (written at pos 1 + delay 0 = 1)
        // - agent cb0: silence (written at pos 1 + 0 = 1)
        // - agent cb1: silence (written at pos 0 + 1 = 1, from pos 0's write)
        // - user cb0: sine (written at pos 1 + 0 = 1)
        // - user cb1: sine (written at pos 0 + 1 = 1, from pos 0's write)
        let (text, model, user) = stream.read_input_tokens(2);
        assert_eq!(text, 42);
        assert_eq!(model[0], config.silence_tokens[0] as i32);
        assert_eq!(model[1], config.silence_tokens[1] as i32); // now visible!
        assert_eq!(user[0], config.sine_tokens[0] as i32);
        assert_eq!(user[1], config.sine_tokens[1] as i32); // now visible!
    }

    #[test]
    fn test_depformer_output_no_delay() {
        // Verify depformer output is written at `step` (no delay offset)
        let config = StsConfig::default();
        let mut stream = make_stream(&config);

        let agent = vec![100u32, 200, 300, 400, 500, 600, 700, 800];
        let user = vec![900u32, 1000, 1100, 1200, 1300, 1400, 1500, 1600];
        stream.write_depformer_output(5, 42, &agent, Some(&user));

        // All written at position 5, no delay
        assert_eq!(stream.token_cache.read(0, 5), 42); // text
        assert_eq!(stream.token_cache.read(1, 5), 100); // agent cb0
        assert_eq!(stream.token_cache.read(2, 5), 200); // agent cb1 (no delay!)
        assert_eq!(stream.token_cache.read(9, 5), 900); // user cb0
        assert_eq!(stream.token_cache.read(10, 5), 1000); // user cb1 (no delay!)

        // Reading at step 6 sees everything:
        let (text, model, user_read) = stream.read_input_tokens(6);
        assert_eq!(text, 42);
        assert_eq!(model, vec![100, 200, 300, 400, 500, 600, 700, 800i32]);
        assert_eq!(user_read, vec![900, 1000, 1100, 1200, 1300, 1400, 1500, 1600i32]);
    }

    #[test]
    fn test_voice_preset_cache_snapshot_mapping() {
        // Verify that prefill_voice_preset correctly maps the circular cache
        // snapshot to flat cache positions.
        let config = StsConfig::default();
        let mut stream = make_stream(&config);

        // Simulate voice preset without actual model (just test cache mapping).
        // With 51 frames, after voice preset:
        //   step = 51
        //   Python offset_end = 52
        //   Snapshot pos 3 = most recent (at rust pos 50)
        //   Snapshot pos 2 = 2nd most recent (at rust pos 49)
        //   Snapshot pos 1 = 3rd most recent (at rust pos 48)
        //   Snapshot pos 0 = 4th most recent (at rust pos 47)
        let num_frames = 51;

        // Manually set step (normally done by forward_embeddings loop)
        stream.step = num_frames;
        stream.write_pos = num_frames;

        // Create a test snapshot: 17 streams × 4 positions
        let mut cache_snapshot: Vec<Vec<i32>> = Vec::new();
        for s in 0..17 {
            cache_snapshot.push(vec![
                (s * 100) as i32,       // pos 0
                (s * 100 + 1) as i32,   // pos 1
                (s * 100 + 2) as i32,   // pos 2
                (s * 100 + 3) as i32,   // pos 3
            ]);
        }

        // Apply just the snapshot mapping part
        let num_snapshot_positions = 4;
        let offset_end = num_frames + 1;
        for k in 0..num_snapshot_positions {
            let python_pos = (offset_end - 1 - k) % num_snapshot_positions;
            let rust_pos = stream.step - 1 - k;
            for (s, snapshot_stream) in cache_snapshot.iter().enumerate() {
                if s < stream.token_cache.num_streams {
                    stream.token_cache.write(s, rust_pos, snapshot_stream[python_pos]);
                }
            }
        }

        // Verify: reading at step 51 (reads pos 50) should get python pos 3 (most recent)
        let (text, _model, _user) = stream.read_input_tokens(51);
        assert_eq!(text, 3); // stream 0, python pos 3

        // Reading at step 50 (reads pos 49) should get python pos 2
        let (text, _model, _user) = stream.read_input_tokens(50);
        assert_eq!(text, 2); // stream 0, python pos 2

        // Reading at step 49 (reads pos 48) should get python pos 1
        let (text, _model, _user) = stream.read_input_tokens(49);
        assert_eq!(text, 1); // stream 0, python pos 1

        // Reading at step 48 (reads pos 47) should get python pos 0
        let (text, _model, _user) = stream.read_input_tokens(48);
        assert_eq!(text, 0); // stream 0, python pos 0

        // Check a model audio stream (stream 1)
        let (_text, model, _user) = stream.read_input_tokens(51);
        assert_eq!(model[0], 103); // stream 1, python pos 3
    }

    #[test]
    fn test_stream_sampling_params() {
        let config = StsConfig::default();
        let mut stream = make_stream(&config);

        assert!((stream.text_temperature - 0.7).abs() < 1e-6);
        assert_eq!(stream.text_top_k, 25);
        assert!((stream.audio_temperature - 0.8).abs() < 1e-6);
        assert_eq!(stream.audio_top_k, 250);

        stream.set_sampling_params(0.5, 10, 0.9, 100);
        assert!((stream.text_temperature - 0.5).abs() < 1e-6);
        assert_eq!(stream.text_top_k, 10);
    }

    #[test]
    fn test_step_output_structure() {
        let output = StepOutput {
            model_audio_tokens: vec![1, 2, 3, 4, 5, 6, 7, 8],
            text_token: 42,
            temporal_ms: 0.0,
            depth_ms: 0.0,
        };
        assert_eq!(output.model_audio_tokens.len(), 8);
        assert_eq!(output.text_token, 42);
    }

    #[test]
    fn test_silence_detection() {
        let config = StsConfig::default();
        let mut stream = make_stream(&config);

        // Initially, last_model_audio_tokens are silence tokens
        assert!(stream.is_silence());
        assert!(!stream.should_stop());

        // Simulate non-silence
        stream.last_model_audio_tokens = vec![100, 200, 300, 400, 500, 600, 700, 800];
        assert!(!stream.is_silence());
    }

    #[test]
    fn test_reset_clears_history() {
        let config = StsConfig::default();
        let mut stream = make_stream(&config);

        stream.text_token_history.push(42);
        stream.text_token_history.push(99);
        stream.consecutive_silence_frames = 10;

        stream.reset();
        assert!(stream.text_token_history.is_empty());
        assert_eq!(stream.consecutive_silence_frames(), 0);
        assert!(!stream.should_stop());

        stream.text_token_history.push(42);
        stream.consecutive_silence_frames = 20;
        stream.reset_keep_buffers();
        assert!(stream.text_token_history.is_empty());
        assert_eq!(stream.consecutive_silence_frames(), 0);
    }
}
