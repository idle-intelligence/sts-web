/**
 * Mimi Worker: loads the Mimi audio codec WASM module and decodes audio tokens
 * to PCM, sending decoded audio directly to the AudioWorklet via MessagePort.
 *
 * This worker runs Mimi decode (~35ms CPU) in parallel with GPU inference in
 * the main inference worker, overlapping CPU and GPU compute.
 *
 * ## Message Protocol
 *
 *   Main -> MimiWorker:
 *     { type: 'load', url?: string }                      -- load Mimi WASM module (default: /mimi-pkg/)
 *     { type: 'load-model', data: ArrayBuffer, numCodebooks?: number }  -- load safetensors model (default 8 codebooks)
 *     { type: 'set-playback-port', port: MessagePort }    -- direct port to AudioWorklet for PCM streaming
 *     { type: 'set-mimi-port', port: MessagePort }        -- port from inference worker (decode commands arrive here)
 *     { type: 'reset' }                                   -- reset codec streaming state between turns
 *
 *   InferenceWorker -> MimiWorker (via mimi-port):
 *     { type: 'decode', tokens: Uint32Array, numCodebooks, step, done }  -- decode one frame
 *     { type: 'reset' }                                   -- reset codec state
 *
 *   MimiWorker -> Main:
 *     { type: 'status', text: string }            -- loading progress
 *     { type: 'ready' }                           -- Mimi loaded and ready for decode
 *     { type: 'decoded', step, done, samples }    -- decoded PCM (also sent to AudioWorklet)
 *     { type: 'error', message: string }          -- unrecoverable error
 *
 * ## Initialization Sequence
 *
 *   1. Main sends 'set-playback-port' and 'set-mimi-port' (can be in any order)
 *   2. Main sends 'load' -> worker loads WASM module
 *   3. Main sends 'load-model' with safetensors bytes -> worker creates MimiEngine
 *   4. Worker sends 'ready' -> decode commands can now be received on mimi-port
 *
 * ## State Variables
 *
 *   mimiWasm       -- imported WASM module (null until 'load')
 *   mimiEngine     -- MimiEngine instance (null until 'load-model')
 *   playbackPort   -- MessagePort to AudioWorklet (PCM streaming)
 *   inferencePort  -- MessagePort from inference worker (decode commands)
 */

let mimiWasm = null;
let mimiEngine = null;
let playbackPort = null;
let inferencePort = null;
let wasmInitPromise = null;

function logState(msg) {
    const t = ((performance.now()) / 1000).toFixed(2);
    console.log(`[mimi-worker +${t}s] ${msg}`);
}

/**
 * Handle a decode or reset message (from either main thread or inference worker port).
 */
function handleDecodeMessage(msg) {
    try {
        switch (msg.type) {
            case 'decode': {
                if (!mimiEngine) {
                    throw new Error('Mimi model not loaded. Send "load-model" first.');
                }

                const tokens = msg.tokens;
                const numCodebooks = msg.numCodebooks || 8;
                const step = msg.step;
                const done = msg.done;

                const t0 = performance.now();
                const pcm = mimiEngine.decode(tokens, numCodebooks);
                const elapsed = performance.now() - t0;

                if (step < 5 || step % 50 === 0) {
                    logState(`Decoded frame ${step}: ${pcm.length} samples (${elapsed.toFixed(1)}ms)`);
                }

                if (pcm && pcm.length > 0) {
                    // Copy for main thread before transferring buffer to AudioWorklet
                    const mainCopy = new Float32Array(pcm);

                    // Send PCM directly to AudioWorklet (bypasses main thread)
                    if (playbackPort) {
                        playbackPort.postMessage(
                            { type: 'audio', samples: pcm },
                            [pcm.buffer]
                        );
                    }

                    // Notify main thread for UI updates and WAV assembly
                    self.postMessage({
                        type: 'decoded',
                        step,
                        done,
                        samples: mainCopy,
                        decodeMs: elapsed,
                    }, [mainCopy.buffer]);
                }
                break;
            }

            case 'reset': {
                if (mimiEngine) {
                    mimiEngine.reset();
                    logState('Mimi codec reset');
                }
                break;
            }
        }
    } catch (err) {
        console.error('[mimi-worker]', err);
        self.postMessage({ type: 'error', message: err.message || String(err) });
    }
}

self.onmessage = async (e) => {
    const { type, ...data } = e.data;

    try {
        switch (type) {
            case 'load': {
                const wasmJsUrl = data.url || new URL('../mimi-pkg/mimi_wasm.js', import.meta.url).href;
                const wasmBgUrl = wasmJsUrl.replace(/\.js$/, '_bg.wasm');
                logState(`Loading Mimi WASM from ${wasmJsUrl}`);
                self.postMessage({ type: 'status', text: 'Loading Mimi WASM module...' });
                wasmInitPromise = (async () => {
                    mimiWasm = await import(wasmJsUrl);
                    await mimiWasm.default({ module_or_path: wasmBgUrl });
                })();
                await wasmInitPromise;
                logState('Mimi WASM module loaded');
                self.postMessage({ type: 'status', text: 'Mimi WASM module loaded' });
                break;
            }

            case 'load-model': {
                if (wasmInitPromise) await wasmInitPromise;
                if (!mimiWasm) {
                    throw new Error('Mimi WASM module not loaded. Send "load" first.');
                }
                const modelBytes = new Uint8Array(data.data);
                const numCodebooks = data.numCodebooks || 8;
                logState(`Loading Mimi model (${(modelBytes.byteLength / (1024 * 1024)).toFixed(1)} MB, ${numCodebooks} codebooks)...`);
                self.postMessage({ type: 'status', text: 'Loading Mimi codec...' });
                mimiEngine = mimiWasm.MimiEngine.load(modelBytes, numCodebooks);
                logState('Mimi model loaded');
                self.postMessage({ type: 'ready' });
                break;
            }

            case 'decode':
            case 'reset':
                handleDecodeMessage(e.data);
                break;

            case 'set-playback-port': {
                playbackPort = data.port;
                logState('Playback port received');
                break;
            }

            case 'set-mimi-port': {
                inferencePort = data.port;
                inferencePort.onmessage = (ev) => handleDecodeMessage(ev.data);
                logState('Inference worker port received');
                break;
            }

            default:
                console.warn('[mimi-worker] Unknown message type:', type);
        }
    } catch (err) {
        console.error('[mimi-worker]', err);
        self.postMessage({ type: 'error', message: err.message || String(err) });
    }
};
