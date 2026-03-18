/**
 * AudioWorklet processor: captures mic input, resamples to 24kHz, and sends
 * mono PCM chunks to the Web Worker.
 *
 * Forked from stt-web. Output is buffered into 1920-sample chunks (~80ms at
 * 24kHz) matching the Mimi codec's frame size.
 */

class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this._targetRate = 24000;
        this._buffer = new Float32Array(1920);
        this._writePos = 0;
        this._active = true;
        this._resamplePos = 0;
        this._directPort = null;

        this.port.onmessage = (e) => {
            if (e.data && e.data.type === 'stop') {
                this._active = false;
            } else if (e.data && e.data.type === 'port') {
                this._directPort = e.data.port;
            }
        };
    }

    process(inputs, outputs, parameters) {
        const port = this._directPort || this.port;

        if (!this._active) {
            if (this._writePos > 0) {
                const remaining = this._buffer.slice(0, this._writePos);
                port.postMessage({ type: 'audio', samples: remaining }, [remaining.buffer]);
                this._writePos = 0;
            }
            port.postMessage({ type: 'done' });
            return false;
        }

        const input = inputs[0];
        if (!input || input.length === 0) return true;
        const channel = input[0];
        if (!channel || channel.length === 0) return true;

        const step = sampleRate / this._targetRate;
        let srcPos = this._resamplePos;

        while (srcPos < channel.length) {
            const idx = Math.floor(srcPos);
            const frac = srcPos - idx;
            const s0 = channel[idx];
            const s1 = idx + 1 < channel.length ? channel[idx + 1] : s0;
            this._buffer[this._writePos++] = s0 + frac * (s1 - s0);

            if (this._writePos >= this._buffer.length) {
                const chunk = new Float32Array(this._buffer);
                port.postMessage({ type: 'audio', samples: chunk }, [chunk.buffer]);
                this._writePos = 0;
            }

            srcPos += step;
        }
        this._resamplePos = srcPos - channel.length;

        return true;
    }
}

registerProcessor('audio-processor', AudioProcessor);
