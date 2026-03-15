/**
 * Web Worker: loads WASM modules, orchestrates the STS pipeline.
 *
 * All inference runs here -- never on the main thread.
 *
 * ## Message Protocol
 *
 *   Main -> Worker:
 *     { type: 'load' }                                     -- init WASM + WebGPU, download + load model
 *     { type: 'new-conversation' }                         -- full reset (clears KV cache, re-prefills)
 *     { type: 'start-listening' }                          -- begin accepting audio chunks
 *     { type: 'audio', samples: Float32Array }             -- feed mic audio chunk (24kHz mono)
 *     { type: 'stop' }                                     -- stop recording, run generation
 *     { type: 'audio-port', port: MessagePort }            -- direct port from mic AudioWorklet
 *     { type: 'playback-port', port: MessagePort }         -- port to stream PCM to AudioWorklet (fallback)
 *     { type: 'mimi-worker-port', port: MessagePort }      -- port to Mimi worker for offloaded decode
 *     { type: 'load-voice-preset', name: string }          -- load voice preset ('random' = no preset)
 *
 *   Worker -> Main:
 *     { type: 'status', text, ready?, progress? }          -- loading/ready/generating status
 *     { type: 'turn-complete' }                            -- generation finished, ready for next turn
 *     { type: 'audio-chunk', step, done }                  -- notifies main of each generated frame
 *     { type: 'transcript', text, final? }                 -- streaming text tokens (inner monologue)
 *     { type: 'metrics', framesPerSec, avgFrameMs, ... }   -- performance metrics after generation
 *     { type: 'error', message }                           -- unrecoverable error
 *     { type: 'voice-loaded', name }                       -- voice preset loaded successfully
 *     { type: 'voice-error', name, error }                 -- voice preset load failed
 *
 * ## Expected Call Sequence
 *
 *   load → [ready] → start-listening → audio* → stop → [turn-complete]
 *                                              ↑________________________|  (multi-turn)
 *   At any time: new-conversation → [re-prefills, back to ready]
 *
 * ## State Variables
 *
 *   engine         -- StsEngine WASM instance (null until loaded)
 *   listening      -- true between start-listening and stop
 *   stopped        -- true after stop received (drops late audio messages)
 *   busy           -- serialization lock for async engine access
 *   msgQueue       -- FIFO queue for serialized message processing
 *   totalSamples   -- PCM samples fed in current turn (for duration calc)
 *   prefillFrames  -- Mimi frames prefilled in current turn
 *   audioPort      -- direct MessagePort from mic AudioWorklet (bypasses main)
 *   playbackPort   -- MessagePort to AudioWorklet for PCM playback (fallback)
 *   mimiWorkerPort -- MessagePort to Mimi worker for offloaded decode
 */

const HF_BASE = '/hf/personaplex-24L-q4_k-webgpu';

let engine = null;
let stsWasm = null;
let totalSamples = 0;
let recordingStart = 0;
let prefillFrames = 0;
let listening = false;

// Benchmark mode: per-frame timing collection
let benchmarkMode = false;
let benchmarkFrames = [];

// Port to stream audio directly to AudioWorklet (legacy, used when no Mimi worker)
let playbackPort = null;

// Port to the Mimi worker for offloaded audio decode
let mimiWorkerPort = null;

function logState(msg) {
    const t = ((performance.now()) / 1000).toFixed(2);
    console.log(`[worker +${t}s] ${msg}`);
}

// Serialize all engine access to prevent wasm-bindgen "recursive use" errors.
let busy = false;
let stopped = false;
let audioChunkCount = 0;
const msgQueue = [];

async function drainQueue() {
    if (busy) return;
    busy = true;
    while (msgQueue.length > 0) {
        const { type, data } = msgQueue.shift();
        try {
            // Drop audio messages when not listening or after stop
            if (type === 'audio' && (!listening || stopped)) continue;
            switch (type) {
                case 'load':
                    logState('Loading model...');
                    if (data.benchmark) benchmarkMode = true;
                    await handleLoad(data.config || {});
                    break;
                case 'start-listening':
                    logState('Listening for audio...');
                    listening = true;
                    stopped = false;
                    totalSamples = 0;
                    prefillFrames = 0;
                    audioChunkCount = 0;
                    break;
                case 'audio':
                    await handleAudio(data);
                    break;
                case 'stop':
                    logState(`Stop received (${audioChunkCount} chunks, ${prefillFrames} frames prefilled)`);
                    stopped = true;
                    listening = false;
                    await handleStop();
                    break;
                case 'new-conversation':
                    logState('New conversation');
                    stopped = false;
                    listening = false;
                    audioChunkCount = 0;
                    prefillFrames = 0;
                    totalSamples = 0;
                    handleNewConversation();
                    break;
                case 'load-voice-preset':
                    await handleLoadVoicePreset(data.name);
                    break;
                default:
                    console.warn('[worker] Unknown message type:', type);
            }
        } catch (err) {
            self.postMessage({ type: 'error', message: err.message || String(err) });
        }
    }
    busy = false;
}

// Direct MessagePort from AudioWorklet (bypasses main thread throttling).
let audioPort = null;

self.onmessage = (e) => {
    const { type, ...data } = e.data;

    if (type === 'audio-port') {
        if (audioPort) {
            audioPort.onmessage = null;
            audioPort.close();
        }
        audioPort = data.port;
        audioPort.onmessage = (ev) => {
            if (ev.data.type === 'audio') {
                msgQueue.push({ type: 'audio', data: { samples: ev.data.samples } });
                drainQueue();
            }
        };
        return;
    }

    if (type === 'playback-port') {
        playbackPort = data.port;
        logState('Playback port received');
        return;
    }

    if (type === 'mimi-worker-port') {
        mimiWorkerPort = data.port;
        logState('Mimi worker port received');
        return;
    }

    msgQueue.push({ type, data });
    drainQueue();
};

// ---------------------------------------------------------------------------
// Cache helpers
// ---------------------------------------------------------------------------

const CACHE_NAME = 'sts-model-v1';

async function isCached(url) {
    try {
        const cache = await caches.open(CACHE_NAME);
        return !!(await cache.match(url));
    } catch { return false; }
}

async function cachedFetch(url, label) {
    const cache = await caches.open(CACHE_NAME);

    const cached = await cache.match(url);
    if (cached) {
        self.postMessage({ type: 'status', text: `${label} (cached)` });
        return await cached.arrayBuffer();
    }

    const resp = await fetch(url);
    if (!resp.ok) {
        throw new Error(`Failed to fetch ${url}: ${resp.status} ${resp.statusText}`);
    }

    const contentLength = parseInt(resp.headers.get('Content-Length') || '0', 10);
    const reader = resp.body.getReader();
    const chunks = [];
    let loaded = 0;

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.byteLength;

        self.postMessage({
            type: 'status',
            text: label,
            progress: { loaded, total: contentLength },
        });
    }

    const buf = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
        buf.set(chunk, offset);
        offset += chunk.byteLength;
    }

    try {
        const cacheResp = new Response(buf.buffer, {
            headers: { 'Content-Type': 'application/octet-stream' },
        });
        await cache.put(url, cacheResp);
    } catch (cacheErr) {
        console.warn('[worker] Could not cache:', cacheErr);
    }

    return buf.buffer;
}

// ---------------------------------------------------------------------------
// Message handlers
// ---------------------------------------------------------------------------

async function handleLoad(config) {
    const base = (config.baseUrl || '').replace(/\/+$/, '');

    // 1. Import WASM module.
    self.postMessage({ type: 'status', text: 'Loading WASM module...' });
    const wasmJsUrl = base ? (base + '/pkg/sts_wasm.js') : new URL('../pkg/sts_wasm.js', import.meta.url).href;
    const wasmBgUrl = base ? (base + '/pkg/sts_wasm_bg.wasm') : new URL('../pkg/sts_wasm_bg.wasm', import.meta.url).href;
    stsWasm = await import(wasmJsUrl);
    await stsWasm.default(wasmBgUrl);

    // 2. Initialize WebGPU device.
    self.postMessage({ type: 'status', text: 'Initializing WebGPU device...' });
    await stsWasm.initWgpuDevice();

    // 3. Create engine instance.
    engine = new stsWasm.StsEngine();

    // 4. Download and process model shards.
    const shardUrls = config.shardList && config.shardList.length > 0
        ? config.shardList
        : Array.from({ length: 8 }, (_, i) =>
            `${HF_BASE}/shards/personaplex-24L-q4_k.gguf.shard-0${i}`
        );

    const allCached = await isCached(shardUrls[0]);
    const verb = allCached ? 'Loading' : 'Downloading';
    self.postMessage({ type: 'status', text: `${verb} model...` });

    {
        const url = shardUrls[0];
        const label = `${verb} model (1/${shardUrls.length})`;
        const buf = await cachedFetch(url, label);
        self.postMessage({ type: 'status', text: 'Parsing model header...' });
        engine.loadModelBegin(new Uint8Array(buf));
    }

    for (let i = 1; i < shardUrls.length; i++) {
        const url = shardUrls[i];
        const label = `${verb} model (${i + 1}/${shardUrls.length})`;
        const buf = await cachedFetch(url, label);
        self.postMessage({ type: 'status', text: `Loading shard ${i + 1}/${shardUrls.length}...` });
        engine.loadModelShard(new Uint8Array(buf), i);
    }

    // 5. Assemble final model.
    self.postMessage({ type: 'status', text: 'Assembling model...' });
    engine.loadModelFinish();

    // 6. Load Mimi codec.
    const mimiUrl = config.mimiUrl || `${HF_BASE}/tokenizer-e351c8d8-checkpoint125.safetensors`;
    const mimiVerb = await isCached(mimiUrl) ? 'Loading' : 'Downloading';
    const mimiBuf = await cachedFetch(mimiUrl, `${mimiVerb} audio codec`);
    self.postMessage({ type: 'status', text: 'Loading audio codec...' });
    engine.loadMimi(new Uint8Array(mimiBuf));

    // 7. Load tokenizer.
    const tokenizerUrl = config.tokenizerUrl || `${HF_BASE}/tokenizer_spm_32k_3.model`;
    const tokVerb = await isCached(tokenizerUrl) ? 'Loading' : 'Downloading';
    const tokBuf = await cachedFetch(tokenizerUrl, `${tokVerb} tokenizer`);
    self.postMessage({ type: 'status', text: 'Loading tokenizer...' });
    engine.loadTokenizer(new Uint8Array(tokBuf));

    // 8. Warm up GPU pipelines (includes system prompt prefill).
    self.postMessage({ type: 'status', text: 'Warming up GPU + prefilling...' });
    await engine.warmup();

    // 9. Signal ready.
    logState('Model loaded, ready');
    self.postMessage({ type: 'status', text: 'Ready', ready: true });
}

// ---------------------------------------------------------------------------
// Audio handlers
// ---------------------------------------------------------------------------

async function handleAudio({ samples }) {
    if (!engine) return;

    const audioData = samples instanceof Float32Array
        ? samples
        : new Float32Array(samples);

    if (totalSamples === 0) {
        recordingStart = performance.now();
        logState('Receiving audio...');
    }
    totalSamples += audioData.length;
    audioChunkCount++;

    // feedAudio does Mimi encode + incremental prefill (async)
    const framesProcessed = await engine.feedAudio(audioData);
    prefillFrames += framesProcessed;

    // Main thread timer handles "Listening... Xs" display — don't send
    // redundant/inaccurate status messages from here.
}

function sendMetrics(now) {
    if (!engine) return;
    const m = engine.getMetrics();
    const elapsed = (now - recordingStart) / 1000;
    const audioDuration = totalSamples / 24000;
    self.postMessage({
        type: 'metrics',
        framesPerSec: m.total_frames > 0 ? m.total_frames / elapsed : 0,
        avgFrameMs: m.total_frames > 0 ? m.total_ms / m.total_frames : 0,
        temporalMs: m.total_frames > 0 ? (m.temporal_ms || 0) / m.total_frames : 0,
        depthMs: m.total_frames > 0 ? (m.depth_ms || 0) / m.total_frames : 0,
        mimiMs: m.total_frames > 0 ? (m.mimi_ms || 0) / m.total_frames : 0,
        rtf: elapsed / audioDuration,
        audioDuration,
    });
}

async function handleStop() {
    if (!engine) return;

    if (totalSamples === 0) {
        logState('Stop: no audio received');
        self.postMessage({ type: 'transcript', text: '', final: true });
        self.postMessage({ type: 'turn-complete' });
        return;
    }

    const audioDuration = totalSamples / 24000;
    logState(`Starting generation (${audioDuration.toFixed(1)}s input, ${prefillFrames} frames prefilled)`);
    self.postMessage({ type: 'status', text: 'Generating response...' });

    // Start streaming generation.
    //
    // With Mimi worker: generateStepInference() returns audio_tokens which are
    // sent to the Mimi worker for decode. The inference worker does NOT wait for
    // decode — it immediately starts the next GPU step. This overlaps Mimi CPU
    // decode (~35ms) with GPU temporal compute (~115ms).
    //
    // Without Mimi worker (fallback): decodeStepAudio() runs locally between
    // inference steps, same as before.
    engine.generateStart();

    let transcript = '';
    if (benchmarkMode) benchmarkFrames = [];
    const genStartTime = performance.now();
    // Track cumulative metrics for per-frame deltas in benchmark mode
    let prevTemporalMs = 0, prevDepthMs = 0, prevMimiMs = 0;

    let frameStart = performance.now();
    let result = await engine.generateStepInference();

    while (result) {
        const frameEnd = performance.now();
        const frameTotalMs = frameEnd - frameStart;

        if (mimiWorkerPort) {
            // Offloaded path: send audio tokens to Mimi worker for decode.
            // Don't wait — GPU is already computing the next temporal step.
            mimiWorkerPort.postMessage({
                type: 'decode',
                tokens: result.audio_tokens,
                numCodebooks: result.num_codebooks,
                step: result.step,
                done: result.done,
            });
        } else {
            // Fallback: decode locally (same thread, blocks next inference step)
            const audio = engine.decodeStepAudio();
            if (audio && audio.length > 0 && playbackPort) {
                playbackPort.postMessage({ type: 'audio', samples: audio }, [audio.buffer]);
            }
        }

        self.postMessage({
            type: 'audio-chunk',
            step: result.step,
            done: result.done,
        });

        if (result.audio_tokens) {
            const silenceTokens = [948, 243, 1178, 546, 1736, 1030, 1978, 2008];
            const audio = Array.from(result.audio_tokens);
            const matches = audio.filter((t, i) => t === silenceTokens[i]).length;
            if (matches >= 3) {
                console.log(`[silence-check] Frame ${result.step}: ${matches}/8 match, audio=[${audio}]`);
            }
        }

        if (result.text) {
            transcript += result.text;
            self.postMessage({ type: 'transcript', text: result.text, final: false });
        }

        self.postMessage({
            type: 'status',
            text: `Generating... frame ${result.step + 1}`,
        });

        // Send live metrics every frame so the UI updates from frame 0
        sendMetrics(performance.now());

        // Collect per-frame timing in benchmark mode using cumulative metric deltas
        if (benchmarkMode) {
            const m = engine.getMetrics();
            benchmarkFrames.push({
                frameIdx: result.step,
                totalMs: frameTotalMs,
                temporalMs: (m.temporal_ms || 0) - prevTemporalMs,
                depthMs: (m.depth_ms || 0) - prevDepthMs,
                mimiMs: (m.mimi_ms || 0) - prevMimiMs,
            });
            prevTemporalMs = m.temporal_ms || 0;
            prevDepthMs = m.depth_ms || 0;
            prevMimiMs = m.mimi_ms || 0;
        }

        if (result.done) break;

        // Yield to the event loop so MessagePort messages (to Mimi worker)
        // actually flush cross-thread. Without this, the WASM Promise from
        // generateStepInference() resolves as a microtask, and the loop
        // resumes immediately — MessagePort postMessage calls queue up but
        // never get dispatched until the entire loop finishes.
        await new Promise(r => setTimeout(r, 0));

        // Next inference step — if Mimi worker is active, GPU gets full overlap
        frameStart = performance.now();
        result = await engine.generateStepInference();
    }

    const totalGenMs = performance.now() - genStartTime;
    logState(`Generation complete: "${transcript.substring(0, 80)}${transcript.length > 80 ? '...' : ''}"`);
    sendMetrics(performance.now());

    // In benchmark mode, post detailed results
    if (benchmarkMode) {
        const m = engine.getMetrics();
        const frameCount = benchmarkFrames.length;
        const avgFrameMs = frameCount > 0 ? totalGenMs / frameCount : 0;
        const avgTemporalMs = frameCount > 0 ? benchmarkFrames.reduce((s, f) => s + f.temporalMs, 0) / frameCount : 0;
        const avgDepthMs = frameCount > 0 ? benchmarkFrames.reduce((s, f) => s + f.depthMs, 0) / frameCount : 0;
        const avgMimiMs = frameCount > 0 ? benchmarkFrames.reduce((s, f) => s + f.mimiMs, 0) / frameCount : 0;
        const audioDuration = totalSamples / 24000;
        self.postMessage({
            type: 'benchmark-complete',
            results: {
                frames: benchmarkFrames,
                avgFrameMs,
                temporalMs: avgTemporalMs,
                depthMs: avgDepthMs,
                mimiMs: avgMimiMs,
                framesPerSec: frameCount > 0 ? (frameCount / (totalGenMs / 1000)) : 0,
                rtf: audioDuration > 0 ? (totalGenMs / 1000) / audioDuration : 0,
                totalGenMs,
                frameCount,
            },
        });
    }

    self.postMessage({ type: 'transcript', text: '', final: true });

    // Prepare for next turn: reset Mimi codec but keep KV cache
    engine.prepareTurn();
    if (mimiWorkerPort) {
        mimiWorkerPort.postMessage({ type: 'reset' });
    }
    totalSamples = 0;
    prefillFrames = 0;
    stopped = false;
    logState('Turn complete, ready for next turn');
    self.postMessage({ type: 'turn-complete' });
}

// ---------------------------------------------------------------------------
// Voice preset loader
// ---------------------------------------------------------------------------

async function handleLoadVoicePreset(name) {
    if (name === 'random') {
        self.postMessage({ type: 'voice-loaded', name: 'random' });
        return;
    }

    try {
        const embeddingsUrl = `${HF_BASE}/voices/${name}.embeddings.bin`;
        const cacheJsonUrl = `${HF_BASE}/voices/${name}.cache.json`;

        const [embeddingsBuffer, cacheJsonBuffer] = await Promise.all([
            cachedFetch(embeddingsUrl, `Loading voice ${name} embeddings`),
            cachedFetch(cacheJsonUrl, `Loading voice ${name} cache`),
        ]);

        const cacheJson = new TextDecoder().decode(cacheJsonBuffer);
        engine.loadVoicePreset(new Uint8Array(embeddingsBuffer), cacheJson);
        self.postMessage({ type: 'voice-loaded', name });
    } catch (err) {
        self.postMessage({ type: 'voice-error', name, error: err.message || String(err) });
    }
}

// ---------------------------------------------------------------------------
// Conversation reset
// ---------------------------------------------------------------------------

function handleNewConversation() {
    totalSamples = 0;
    prefillFrames = 0;
    audioChunkCount = 0;
    if (!engine) return;
    engine.reset();
}
