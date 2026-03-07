/**
 * Web Worker: loads WASM modules, orchestrates the STS pipeline.
 *
 * All inference runs here -- never on the main thread.
 *
 * Protocol:
 *   Main -> Worker:
 *     { type: 'load' }                          -- initialize WASM + WebGPU, download model
 *     { type: 'audio', samples: Float32Array }   -- feed mic audio chunk
 *     { type: 'stop' }                           -- end of conversation
 *     { type: 'reset' }                          -- clear state for new session
 *
 *   Worker -> Main:
 *     { type: 'status', text: string, ready?: boolean, progress?: { loaded, total } }
 *     { type: 'audio-out', samples: Float32Array }  -- output audio PCM chunk
 *     { type: 'transcript', text: string, final?: boolean }  -- inner monologue text
 *     { type: 'metrics', ... }                      -- performance metrics
 *     { type: 'error', message: string }
 */

// Model base URL — local dev server serves from /hf/ route
// For production, change to HuggingFace: https://huggingface.co/idle-intelligence/personaplex-7b-v1-q4_k-webgpu/resolve/main
const HF_BASE = '/hf/personaplex-7b-v1-q4_k-webgpu';

let engine = null;
let stsWasm = null;
let totalSamples = 0;
let recordingStart = 0;

let lastMetricsSent = 0;
const METRICS_INTERVAL_MS = 1000;

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
            if (type === 'audio' && stopped) continue;
            switch (type) {
                case 'load':
                    logState('Loading model...');
                    await handleLoad(data.config || {});
                    break;
                case 'audio':
                    await handleAudio(data);
                    break;
                case 'stop':
                    logState(`Stop received (${audioChunkCount} chunks)`);
                    stopped = true;
                    await handleStop();
                    break;
                case 'reset':
                    logState('Reset -- starting new session');
                    stopped = false;
                    audioChunkCount = 0;
                    handleReset();
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

/**
 * Fetch a URL with caching via the Cache API.
 * Returns the Response body as an ArrayBuffer.
 * Reports download progress via postMessage.
 */
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

    // 4. Download and process model shards one at a time (incremental loading).
    //    This keeps only ~512MB of shard data in WASM memory at once,
    //    avoiding the 4GB WASM address space limit.
    const shardUrls = config.shardList && config.shardList.length > 0
        ? config.shardList
        : Array.from({ length: 9 }, (_, i) =>
            `${HF_BASE}/personaplex-7b-v1-q4_k.gguf.shard-0${i}`
        );

    // Check if first shard is cached to pick the right verb
    const allCached = await isCached(shardUrls[0]);
    const verb = allCached ? 'Loading' : 'Downloading';
    self.postMessage({ type: 'status', text: `${verb} model...` });

    // Shard 0: parse GGUF header + process shard 0's tensors
    {
        const url = shardUrls[0];
        const label = `${verb} model (1/${shardUrls.length})`;
        const buf = await cachedFetch(url, label);
        self.postMessage({ type: 'status', text: 'Parsing model header...' });
        engine.loadModelBegin(new Uint8Array(buf));
        // buf can now be GC'd
    }

    // Shards 1..N-1: process each shard's tensors individually
    for (let i = 1; i < shardUrls.length; i++) {
        const url = shardUrls[i];
        const label = `${verb} model (${i + 1}/${shardUrls.length})`;
        const buf = await cachedFetch(url, label);
        self.postMessage({ type: 'status', text: `Loading shard ${i + 1}/${shardUrls.length}...` });
        engine.loadModelShard(new Uint8Array(buf), i);
        // buf can now be GC'd
    }

    // 5. Assemble final model from processed tensors.
    self.postMessage({ type: 'status', text: 'Assembling model...' });
    engine.loadModelFinish();

    // 6. Load Mimi codec weights.
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

    // 9. Signal ready.
    logState('Model loaded, ready to receive audio');
    self.postMessage({ type: 'status', text: 'Ready', ready: true });
}

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

    // Feed audio to the STS engine — encodes with Mimi, accumulates frames.
    // No model inference happens here.
    engine.feedAudio(audioData);

    const audioDur = (totalSamples / 24000).toFixed(1);
    self.postMessage({ type: 'status', text: `Recording... ${audioDur}s` });
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
        return;
    }

    const audioDuration = totalSamples / 24000;
    logState(`Generating response (${totalSamples} samples = ${audioDuration.toFixed(1)}s input)...`);
    self.postMessage({ type: 'status', text: 'Generating response...' });

    // flush() runs the full PersonaPlex pipeline:
    //   1. System prompt prefill (silence → text → silence)
    //   2. User audio prefill (Mimi tokens through temporal + depformer)
    //   3. Autoregressive generation until silence or max frames
    //   4. Mimi decode of response audio tokens
    logState('Calling engine.flush()...');
    const result = await engine.flush();
    logState(`flush() returned: ${result ? JSON.stringify({
        hasAudio: !!(result.audio && result.audio.length),
        audioLen: result.audio ? result.audio.length : 0,
        hasText: !!result.text,
        textLen: result.text ? result.text.length : 0,
    }) : 'null/undefined'}`);

    if (result && result.audio && result.audio.length > 0) {
        const outDur = (result.audio.length / 24000).toFixed(1);
        logState(`Sending ${outDur}s of audio (${result.audio.length} samples) to main thread`);
        self.postMessage({ type: 'audio-out', samples: result.audio });
    } else {
        logState('No audio in result');
    }
    if (result && result.text) {
        logState(`Transcript: "${result.text.substring(0, 100)}${result.text.length > 100 ? '...' : ''}"`);
        self.postMessage({ type: 'transcript', text: result.text, final: false });
    }

    // Final metrics
    sendMetrics(performance.now());

    self.postMessage({ type: 'transcript', text: '', final: true });
    totalSamples = 0;
    stopped = false;
    logState('Generation complete, ready for next input');
    self.postMessage({ type: 'status', text: 'Ready', ready: true });
}

function handleReset() {
    totalSamples = 0;
    lastMetricsSent = 0;
    audioChunkCount = 0;
    if (!engine) return;
    engine.reset();
}
