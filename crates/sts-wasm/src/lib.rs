//! STS 7B — browser-native speech-to-speech.
//!
//! Multi-stream transformer consuming Mimi audio codec tokens and producing
//! both text and audio output tokens. Uses a temporal transformer (~7B params)
//! for time-step processing and a depth transformer (~300M params) for
//! multi-codebook audio token generation.
//!
//! Uses Burn's wgpu backend for GPU inference — works natively (Vulkan/Metal)
//! and in the browser (WASM + WebGPU).

#[cfg(feature = "wgpu")]
pub mod model;

#[cfg(feature = "wgpu")]
pub mod depth;

#[cfg(feature = "wgpu")]
pub mod gguf;

#[cfg(feature = "wgpu")]
pub mod stream;

pub mod mimi;

pub mod tokenizer;

#[cfg(feature = "wgpu")]
pub mod loader;

#[cfg(feature = "wasm")]
pub mod depth_engine;

#[cfg(feature = "wasm")]
pub mod web;

/// Model configuration for the STS 7B speech-to-speech model.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct StsConfig {
    // -- Temporal transformer --

    /// Number of temporal transformer layers.
    pub num_layers: usize,
    /// Hidden dimension of the temporal transformer.
    pub hidden_size: usize,
    /// Number of attention heads (queries) in the temporal transformer.
    pub num_heads: usize,
    /// Number of key-value heads (for GQA).
    pub num_kv_heads: usize,
    /// Feed-forward intermediate size in the temporal transformer.
    pub intermediate_size: usize,
    /// RoPE base frequency.
    pub rope_theta: f64,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Sliding window size for attention.
    pub sliding_window: usize,

    // -- Depth transformer --

    /// Number of depth transformer layers.
    pub depth_num_layers: usize,
    /// Hidden dimension of the depth transformer.
    pub depth_hidden_size: usize,
    /// Number of attention heads in the depth transformer.
    pub depth_num_heads: usize,
    /// Number of KV heads in the depth transformer.
    pub depth_num_kv_heads: usize,
    /// Feed-forward intermediate size in the depth transformer.
    pub depth_intermediate_size: usize,
    /// Number of multi-linear steps in the depth transformer (one per output stream).
    pub depth_num_steps: usize,
    /// Number of depth steps to run during generation (model speaking).
    /// Steps beyond this are filled with sine tokens. Default 8 (skip user audio predictions).
    pub depth_gen_steps: usize,

    // -- Audio / text vocab --

    /// Text output vocabulary size.
    pub text_vocab_size: usize,
    /// Text input vocabulary size (text_emb rows).
    pub text_in_vocab_size: usize,
    /// Number of audio codebooks (Mimi input).
    pub num_codebooks: usize,
    /// Audio codebook vocabulary size.
    pub audio_vocab_size: usize,
    /// Number of output audio codebooks (Mimi output, may differ from input).
    pub num_output_codebooks: usize,

    // -- Streaming --

    /// Delayed-streams text offset in frames.
    pub text_delay: usize,
    /// Text padding token ID.
    pub text_padding_id: u32,
    /// Text start token ID.
    pub text_start_token: u32,
    /// Audio initial/padding token ID (= card = 2048, last entry in audio embedding).
    /// Used as the "start of sequence" token for all audio codebooks.
    pub audio_initial_token_id: u32,

    // -- Token constants --

    /// Per-codebook silence tokens for agent audio during non-speech.
    pub silence_tokens: [u32; 8],
    /// Per-codebook sine tokens for user audio during non-speech.
    pub sine_tokens: [u32; 8],
    /// Delay pattern for all 17 streams.
    /// [text, agent_0..7, user_0..7]
    pub delays: [usize; 17],
    /// Frame rate in Hz (12.5 for Mimi).
    pub frame_rate: f32,

    // -- Prefill --

    /// Default system prompt token IDs (pre-tokenized "assistant" persona).
    pub system_prompt_tokens: Vec<u32>,
}

impl Default for StsConfig {
    fn default() -> Self {
        // Values from actual PersonaPlex-7B / Moshi architecture tensor inspection.
        Self {
            // Temporal transformer (Helium backbone)
            num_layers: 32,
            hidden_size: 4096,
            num_heads: 32,
            num_kv_heads: 32, // MHA, not GQA
            intermediate_size: 11264, // in_proj is [22528, 4096] = 2 * 11264
            rope_theta: 10000.0,
            max_seq_len: 3000, // ~4 minutes at 12.5 Hz
            sliding_window: 3000, // full context, no sliding window

            // Depth transformer (Depformer)
            depth_num_layers: 6,
            depth_hidden_size: 1024,
            depth_num_heads: 16,
            depth_num_kv_heads: 16, // MHA
            depth_intermediate_size: 2816, // gating [5632, 1024] = 2 * 2816
            depth_num_steps: 16, // multi-linear: 16 weight sets
            depth_gen_steps: 8, // only run 8 steps during generation (skip user audio predictions)

            // Audio / text vocab
            text_vocab_size: 32000,
            text_in_vocab_size: 32001, // +1 for padding
            num_codebooks: 8,
            audio_vocab_size: 2049, // 2048 + 1 for padding
            num_output_codebooks: 8,

            // Streaming
            text_delay: 0, // PersonaPlex uses delay=0 for fine-tuned model
            text_padding_id: 3,
            text_start_token: 32000,
            audio_initial_token_id: 2048, // = card = audio_vocab_size - 1

            // Token constants (from PersonaPlex / ivan-digital Swift reference)
            silence_tokens: [948, 243, 1178, 546, 1736, 1030, 1978, 2008],
            sine_tokens: [430, 1268, 381, 1611, 1095, 1495, 56, 472],
            delays: [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
            frame_rate: 12.5,

            // PersonaPlex default: "<system> You are a wise and friendly teacher.
            // Answer questions or provide advice in a clear and engaging way. <system>"
            system_prompt_tokens: vec![
                607, 4831, 578, 493, 298, 272, 11821, 267, 7514, 3290, 263, 506, 1292, 307,
                775, 3574, 271, 272, 1195, 267, 12250, 488, 263, 607, 4831, 578,
            ],
        }
    }
}
