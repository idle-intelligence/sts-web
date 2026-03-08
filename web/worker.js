/**
 * Web Worker: loads WASM modules, orchestrates the STS pipeline.
 *
 * All inference runs here -- never on the main thread.
 *
 * Protocol:
 *   Main -> Worker:
 *     { type: 'load' }                          -- initialize WASM + WebGPU, download model
 *     { type: 'duplex-start' }                  -- start full-duplex loop
 *     { type: 'duplex-stop' }                   -- stop full-duplex loop
 *     { type: 'new-conversation' }              -- clear all state for a fresh conversation
 *     { type: 'start-listening' }               -- (walkie-talkie) begin accepting audio
 *     { type: 'audio', samples: Float32Array }  -- feed mic audio chunk
 *     { type: 'stop' }                          -- (walkie-talkie) stop recording, generate
 *     { type: 'playback-port', port: MessagePort } -- port to stream audio to AudioWorklet
 *
 *   Worker -> Main:
 *     { type: 'status', text: string, ready?: boolean, progress?: { loaded, total } }
 *     { type: 'turn-complete' }                 -- generation finished (walkie-talkie)
 *     { type: 'audio-chunk', step: number, done: boolean }
 *     { type: 'transcript', text: string, final?: boolean }
 *     { type: 'audio-complete', samples: Float32Array }
 *     { type: 'metrics', ... }
 *     { type: 'error', message: string }
 */

const HF_BASE = '/hf/personaplex-7b-v1-q4_k-webgpu';

let engine = null;
let stsWasm = null;
let totalSamples = 0;
let recordingStart = 0;
let prefillFrames = 0;
let listening = false; // Only process audio when explicitly listening (walkie-talkie)

// Port to stream audio directly to AudioWorklet
let playbackPort = null;

// Duplex state
let duplexRunning = false;
const duplexAudioBuffer = [];

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
            // Drop audio messages when not listening or after stop (walkie-talkie only)
            if (type === 'audio' && (!listening || stopped)) continue;
            switch (type) {
                case 'load':
                    logState('Loading model...');
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
                case 'duplex-start':
                    await handleDuplexStart();
                    break;
                case 'duplex-stop':
                    handleDuplexStop();
                    break;
                case 'new-conversation':
                    logState('New conversation');
                    stopped = false;
                    listening = false;
                    audioChunkCount = 0;
                    prefillFrames = 0;
                    totalSamples = 0;
                    duplexAudioBuffer.length = 0;
                    handleNewConversation();
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
                if (duplexRunning) {
                    // In duplex mode, buffer audio directly (bypass queue serialization)
                    duplexAudioBuffer.push(ev.data.samples);
                } else {
                    msgQueue.push({ type: 'audio', data: { samples: ev.data.samples } });
                    drainQueue();
                }
            }
        };
        return;
    }

    if (type === 'playback-port') {
        playbackPort = data.port;
        logState('Playback port received');
        return;
    }

    // Handle duplex stop/reset directly (don't queue — drainQueue is blocked)
    if (type === 'duplex-stop' && duplexRunning) {
        handleDuplexStop();
        return;
    }
    if (type === 'new-conversation' && duplexRunning) {
        handleDuplexStop();
        // Queue the reset for after the loop exits
        msgQueue.push({ type, data });
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
        : Array.from({ length: 9 }, (_, i) =>
            `${HF_BASE}/personaplex-7b-v1-q4_k.gguf.shard-0${i}`
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

    // 8. Warm up GPU pipelines.
    self.postMessage({ type: 'status', text: 'Warming up GPU...' });
    await engine.warmup();

    // 9. Prefill system prompt (builds KV cache, ~20s one-time cost).
    self.postMessage({ type: 'status', text: 'Prefilling system prompt...' });
    engine.prefill();

    // 10. Signal ready.
    logState('Model loaded, ready');
    self.postMessage({ type: 'status', text: 'Ready', ready: true });
}

// ---------------------------------------------------------------------------
// Walkie-talkie handlers (kept for file/example WAV fallback)
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

    const audioDur = (totalSamples / 24000).toFixed(1);
    self.postMessage({ type: 'status', text: `Listening... ${audioDur}s` });
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

    // Start streaming generation
    engine.generateStart();

    let transcript = '';
    const allAudioChunks = [];
    while (true) {
        const chunk = await engine.generateStep();
        if (!chunk) break;

        // Stream audio chunk to playback port (Worker -> AudioWorklet)
        if (chunk.audio && chunk.audio.length > 0) {
            allAudioChunks.push(new Float32Array(chunk.audio));

            if (playbackPort) {
                playbackPort.postMessage({ type: 'audio', samples: chunk.audio }, [chunk.audio.buffer]);
            }
            self.postMessage({
                type: 'audio-chunk',
                step: chunk.step,
                done: chunk.done,
            });
        }

        if (chunk.text) {
            transcript += chunk.text;
            self.postMessage({ type: 'transcript', text: chunk.text, final: false });
        }

        self.postMessage({
            type: 'status',
            text: `Generating... frame ${chunk.step + 1}`,
        });

        if (chunk.done) break;
    }

    logState(`Generation complete: "${transcript.substring(0, 80)}${transcript.length > 80 ? '...' : ''}"`);
    sendMetrics(performance.now());

    // Send complete audio for WAV download
    if (allAudioChunks.length > 0) {
        let totalLen = 0;
        for (const c of allAudioChunks) totalLen += c.length;
        const fullAudio = new Float32Array(totalLen);
        let offset = 0;
        for (const c of allAudioChunks) {
            fullAudio.set(c, offset);
            offset += c.length;
        }
        self.postMessage({ type: 'audio-complete', samples: fullAudio }, [fullAudio.buffer]);
    }

    self.postMessage({ type: 'transcript', text: '', final: true });

    // Prepare for next turn: reset Mimi codec but keep KV cache
    engine.prepareTurn();
    totalSamples = 0;
    prefillFrames = 0;
    stopped = false;
    logState('Turn complete, ready for next turn');
    self.postMessage({ type: 'turn-complete' });
}

// ---------------------------------------------------------------------------
// Duplex handlers
// ---------------------------------------------------------------------------

async function handleDuplexStart() {
    if (!engine) {
        self.postMessage({ type: 'error', message: 'Engine not loaded' });
        return;
    }
    if (duplexRunning) {
        logState('Duplex already running');
        return;
    }
    logState('Starting duplex loop');
    // Run the loop but don't await it here -- let drainQueue finish so the
    // worker can keep processing other messages (like duplex-stop).
    runDuplexLoop().catch(err => {
        self.postMessage({ type: 'error', message: err.message || String(err) });
    });
}

function handleDuplexStop() {
    logState('Stopping duplex loop');
    duplexRunning = false;
}

async function runDuplexLoop() {
    engine.duplexStart();
    duplexRunning = true;
    busy = true; // Prevent drainQueue from touching engine while duplex runs
    let step = 0;

    while (duplexRunning) {
        // Drain audio buffer into engine (Mimi encode only, fast)
        // Take a snapshot to avoid re-entrancy issues
        const pending = duplexAudioBuffer.splice(0, duplexAudioBuffer.length);
        for (const samples of pending) {
            engine.feedAudioDuplex(samples);
        }

        // Run one duplex step (~250ms)
        const result = await engine.duplexStep();
        if (!result) break;

        // Stream audio to playback
        if (result.audio && result.audio.length > 0 && playbackPort) {
            playbackPort.postMessage({ type: 'audio', samples: result.audio }, [result.audio.buffer]);
        }

        // Forward text
        if (result.text) {
            self.postMessage({ type: 'transcript', text: result.text, final: false });
        }

        // Status
        self.postMessage({
            type: 'status',
            text: result.userQueueDepth > 0
                ? `Listening... (${result.userQueueDepth} frames buffered)`
                : `Active (step ${step})`,
        });

        step++;
    }

    engine.duplexStop();
    duplexRunning = false;
    busy = false; // Allow drainQueue to process messages again
    duplexAudioBuffer.length = 0;
    logState('Duplex loop ended');
    self.postMessage({ type: 'status', text: 'Ready', ready: true });
    // Process any queued messages (e.g. new-conversation) that arrived during loop
    drainQueue();
}

// ---------------------------------------------------------------------------
// Conversation reset
// ---------------------------------------------------------------------------

function handleNewConversation() {
    totalSamples = 0;
    prefillFrames = 0;
    audioChunkCount = 0;
    duplexAudioBuffer.length = 0;
    if (!engine) return;
    engine.reset();
}
