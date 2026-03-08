/**
 * AudioWorklet processor: plays back PCM audio chunks from the inference worker.
 *
 * Receives audio via MessagePort from the worker, buffers it in a ring buffer,
 * and outputs 128-sample render quantums to the audio context.
 *
 * Prebuffers ~0.25s (1 frame) before starting playback for low-latency duplex.
 * This is the output counterpart to audio-processor.js (input).
 */

class AudioPlayback extends AudioWorkletProcessor {
    constructor() {
        super();
        // Ring buffer: ~2s at 24kHz (enough headroom for RTF ~3-4x)
        this._bufferSize = 48000;
        this._buffer = new Float32Array(this._bufferSize);
        this._readPos = 0;
        this._writePos = 0;
        this._available = 0;
        // Buffer ~0.25s before starting playback (1 generation frame)
        this._prebufferThreshold = 6000;
        this._prebuffering = true;

        this.port.onmessage = (e) => {
            if (e.data && e.data.type === 'audio') {
                this._enqueue(e.data.samples);
            } else if (e.data && e.data.type === 'port') {
                // Direct port from Worker for streaming audio (bypasses main thread)
                const directPort = e.data.port;
                directPort.onmessage = (ev) => {
                    if (ev.data && ev.data.type === 'audio') {
                        this._enqueue(ev.data.samples);
                    }
                };
            } else if (e.data && e.data.type === 'flush') {
                // Stop prebuffering — play whatever we have (generation done)
                this._prebuffering = false;
            } else if (e.data && e.data.type === 'reset') {
                // Clear ring buffer for new conversation
                this._readPos = 0;
                this._writePos = 0;
                this._available = 0;
                this._prebuffering = true;
            }
        };
    }

    _enqueue(samples) {
        for (let i = 0; i < samples.length; i++) {
            this._buffer[this._writePos] = samples[i];
            this._writePos = (this._writePos + 1) % this._bufferSize;
            if (this._available < this._bufferSize) {
                this._available++;
            } else {
                // Overwrite oldest sample if buffer is full
                this._readPos = (this._readPos + 1) % this._bufferSize;
            }
        }
        // Once we have enough buffered, start playing
        if (this._prebuffering && this._available >= this._prebufferThreshold) {
            this._prebuffering = false;
        }
    }

    process(inputs, outputs, parameters) {
        const output = outputs[0];
        if (!output || output.length === 0) return true;
        const channel = output[0];

        // Hold silence while prebuffering
        if (this._prebuffering) {
            for (let i = 0; i < channel.length; i++) channel[i] = 0;
            return true;
        }

        for (let i = 0; i < channel.length; i++) {
            if (this._available > 0) {
                channel[i] = this._buffer[this._readPos];
                this._readPos = (this._readPos + 1) % this._bufferSize;
                this._available--;
            } else {
                channel[i] = 0; // Silence on underrun
            }
        }

        return true;
    }
}

registerProcessor('audio-playback', AudioPlayback);
