/**
 * AudioWorklet processor: plays back PCM audio chunks from the inference worker.
 *
 * Receives audio via MessagePort from the worker, buffers it in a ring buffer,
 * and outputs 128-sample render quantums to the audio context.
 *
 * This is the output counterpart to audio-processor.js (input).
 */

class AudioPlayback extends AudioWorkletProcessor {
    constructor() {
        super();
        // Ring buffer: ~500ms at 24kHz
        this._bufferSize = 12000;
        this._buffer = new Float32Array(this._bufferSize);
        this._readPos = 0;
        this._writePos = 0;
        this._available = 0;

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
    }

    process(inputs, outputs, parameters) {
        const output = outputs[0];
        if (!output || output.length === 0) return true;
        const channel = output[0];

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
