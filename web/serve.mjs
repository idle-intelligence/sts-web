#!/usr/bin/env node
/**
 * Dev server for STS browser demo.
 *
 * Serves the web app, WASM pkg, and model shards.
 * Forked from stt-web.
 *
 * Usage: node web/serve.mjs [--port 8080]
 */

import { createServer } from "node:http";
import { createReadStream, existsSync, statSync } from "node:fs";
import { join, extname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(fileURLToPath(import.meta.url), "../..");
const PORT = parseInt(process.argv.find((_, i, a) => a[i - 1] === "--port") ?? "8443");

const MIME = {
    ".html": "text/html",
    ".js":   "text/javascript",
    ".mjs":  "text/javascript",
    ".wasm": "application/wasm",
    ".json": "application/json",
    ".wav":  "audio/wav",
    ".css":  "text/css",
    ".bin":  "application/octet-stream",
    ".safetensors": "application/octet-stream",
    ".gguf": "application/octet-stream",
    ".ico":  "image/x-icon",
    ".png":  "image/png",
};

const server = createServer((req, res) => {
    const url = new URL(req.url, `http://${req.headers.host}`);
    const pathname = decodeURIComponent(url.pathname);

    // CORS headers (needed for WASM + WebGPU in cross-origin workers)
    res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
    res.setHeader("Cross-Origin-Embedder-Policy", "credentialless");
    res.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");

    let filePath;
    if (pathname === "/" || pathname === "/index.html") {
        filePath = join(ROOT, "web/index.html");
    } else if (pathname.startsWith("/pkg/")) {
        filePath = join(ROOT, "crates/sts-wasm", pathname);
    } else if (pathname.startsWith("/mimi-pkg/")) {
        filePath = join(ROOT, "crates/mimi-wasm/pkg", pathname.replace("/mimi-pkg/", ""));
    } else if (pathname.startsWith("/hf/")) {
        filePath = join(ROOT, "..", pathname);
    } else {
        filePath = join(ROOT, "web", pathname);
    }

    if (!existsSync(filePath) || !statSync(filePath).isFile()) {
        res.writeHead(404);
        res.end("Not found: " + pathname);
        return;
    }

    const ext = extname(filePath);
    const mime = MIME[ext] ?? "application/octet-stream";
    const stat = statSync(filePath);

    // Support range requests for large shard files
    const range = req.headers.range;
    if (range && stat.size > 1_000_000) {
        const match = range.match(/bytes=(\d+)-(\d*)/);
        if (match) {
            const start = parseInt(match[1]);
            const end = match[2] ? parseInt(match[2]) : stat.size - 1;
            res.writeHead(206, {
                "Content-Type": mime,
                "Content-Range": `bytes ${start}-${end}/${stat.size}`,
                "Content-Length": end - start + 1,
                "Accept-Ranges": "bytes",
            });
            createReadStream(filePath, { start, end }).pipe(res);
            return;
        }
    }

    res.writeHead(200, {
        "Content-Type": mime,
        "Content-Length": stat.size,
        "Accept-Ranges": "bytes",
    });
    createReadStream(filePath).pipe(res);
});

server.listen(PORT, "0.0.0.0", () => {
    console.log(`\nSTS dev server running:`);
    console.log(`  Local:   http://localhost:${PORT}`);
    console.log(`  Model:   /hf/personaplex-24L-q4_k-webgpu/ (local)\n`);
});
