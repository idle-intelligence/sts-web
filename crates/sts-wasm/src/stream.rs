//! Multi-stream inference loop for speech-to-speech.
//!
//! Manages the full STS pipeline per time step:
//! 1. Receive user audio tokens from Mimi encoder (8 codebooks)
//! 2. Feed user + model tokens + text token into temporal transformer
//! 3. Get hidden state + text logits
//! 4. Run depth transformer to generate model audio tokens
//! 5. Return model audio tokens for Mimi decoder + text token
//!
//! Uses a flat token cache with delay offsets (matching Swift reference):
//! - Tokens are written at `pos + delays[stream]`
//! - Tokens are read at `step - 1` for all streams
//! - Unwritten positions contain -1 (zero embedding contribution)
//!
//! Stream layout (17 sub-sequences at each time step):
//!   Stream 0:  text (delay=0)
//!   Stream 1:  model audio codebook 0 (delay=0)
//!   Streams 2-8: model audio codebooks 1-7 (delay=1)
//!   Stream 9:  user audio codebook 0 (delay=0)
//!   Streams 10-16: user audio codebooks 1-7 (delay=1)

use crate::depth::DepthTransformer;
use crate::model::{sample_greedy, sample_top_k_with_penalty, LayerCaches, TemporalTransformer};
use crate::StsConfig;

/// Output of a single STS inference step.
pub struct StepOutput {
    /// Model audio codebook tokens (8 tokens, one per codebook).
    /// These are fed into the Mimi decoder to produce audio.
    pub model_audio_tokens: Vec<u32>,
    /// Text token from inner monologue (always produced).
    pub text_token: u32,
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
            silence_early_stop_frames: 8, // ~0.6s of silence triggers stop
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
        for _i in 0..user_audio_frames.len() {
            let step = self.step;
            let (text, model, user) = self.read_input_tokens(step);

            // Temporal forward
            let (hidden, _text_logits) = temporal.forward(
                &user,
                &model,
                text,
                &mut self.temporal_cache,
            );

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
            )
            .await;

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
    pub async fn step(
        &mut self,
        temporal: &TemporalTransformer,
        depth: &DepthTransformer,
    ) -> StepOutput {
        let nq = self.config.num_codebooks;
        let step = self.step;

        // Read input tokens from cache (reads position step - 1)
        let (text, model, user) = self.read_input_tokens(step);

        // Step 1: Temporal transformer forward pass
        let (hidden, text_logits) = temporal.forward(
            &user,
            &model,
            text,
            &mut self.temporal_cache,
        );

        // Step 2: Sample text token with repetition penalty
        let text_token = if self.text_temperature <= 0.0 {
            sample_greedy(text_logits).await
        } else {
            let text_history: Vec<u32> = self
                .text_token_history
                .iter()
                .rev()
                .take(self.penalty_window)
                .copied()
                .collect();
            sample_top_k_with_penalty(
                text_logits,
                self.text_top_k,
                self.text_temperature,
                &text_history,
                self.repetition_penalty,
            )
            .await
        };
        self.text_token_history.push(text_token);
        if self.text_token_history.len() > self.penalty_window {
            self.text_token_history.drain(..self.text_token_history.len() - self.penalty_window);
        }

        // Step 3: Reset depth cache for this time step
        self.depth_cache.reset_keep_buffers();

        // Step 4: Depth transformer generates 16 audio tokens
        let penalty_hist = &self.audio_token_history;
        let all_audio_tokens = depth.generate(
            hidden,
            text_token,
            &mut self.depth_cache,
            self.audio_temperature,
            self.audio_top_k,
            None, // No provided tokens during generation
            Some(penalty_hist),
            self.repetition_penalty,
        )
        .await;

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
        // Both agent and user predictions are written during generation.
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

        StepOutput {
            model_audio_tokens: model_audio_out,
            text_token,
        }
    }

    // -----------------------------------------------------------------------
    // Silence detection and early stopping
    // -----------------------------------------------------------------------

    /// Check if the last model output is silence (all 8 codebooks match silence tokens).
    pub fn is_silence(&self) -> bool {
        self.last_model_audio_tokens
            .iter()
            .zip(self.config.silence_tokens.iter())
            .all(|(&got, &expected)| got == expected)
    }

    /// Check if generation should stop due to sustained silence.
    pub fn should_stop(&self) -> bool {
        self.consecutive_silence_frames >= self.silence_early_stop_frames
    }

    /// Get the number of consecutive silence frames so far.
    pub fn consecutive_silence_frames(&self) -> usize {
        self.consecutive_silence_frames
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
        self.temporal_cache.reset();
        self.depth_cache.reset();
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
