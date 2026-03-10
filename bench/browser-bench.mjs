#!/usr/bin/env node
/**
 * Browser benchmark for STS inference.
 *
 * Launches Chromium with WebGPU, navigates to ?benchmark, waits for
 * window.__benchmarkResults, and prints the results JSON.
 *
 * Usage: node bench/browser-bench.mjs [--url http://localhost:8443] [--browser /path/to/chrome]
 */

import { chromium } from 'playwright';

const url = process.argv.find((_, i, a) => a[i - 1] === '--url') ?? 'http://localhost:8443';
const browserPath = process.argv.find((_, i, a) => a[i - 1] === '--browser');

const launchOptions = {
    headless: false,
    args: [
        '--enable-unsafe-webgpu',
        '--enable-features=WebGPU',
        '--ignore-gpu-blocklist',
        '--ignore-certificate-errors',
    ],
};
if (browserPath) launchOptions.executablePath = browserPath;

const browser = await chromium.launch(launchOptions);

const context = await browser.newContext({
    ignoreHTTPSErrors: true,
});
const page = await context.newPage();
page.setDefaultTimeout(300_000);

page.on('console', msg => {
    const text = msg.text();
    console.log(`  [browser] ${text}`);
});

page.on('pageerror', err => {
    console.error('[page error]', err.message);
});

console.log(`Navigating to ${url}/?benchmark ...`);
await page.goto(`${url}/?benchmark`, { timeout: 60_000 });

// Wait up to 5 minutes for model load + inference to complete
console.log('Waiting for benchmark to complete (up to 5 min)...');
const results = await page.waitForFunction(
    () => window.__benchmarkResults,
    null,
    { timeout: 300_000 }
);

const data = await results.jsonValue();

if (data.error) {
    console.error('Benchmark failed:', data.error);
    await browser.close();
    process.exit(1);
}

console.log('\n=== Benchmark Results ===');
console.log(`Frames:        ${data.frameCount}`);
console.log(`Total gen:     ${data.totalGenMs?.toFixed(0)} ms`);
console.log(`Avg frame:     ${data.avgFrameMs?.toFixed(1)} ms`);
console.log(`Temporal:      ${data.temporalMs?.toFixed(1)} ms`);
console.log(`Depth:         ${data.depthMs?.toFixed(1)} ms`);
console.log(`Mimi:          ${data.mimiMs?.toFixed(1)} ms`);
console.log(`Frames/sec:    ${data.framesPerSec?.toFixed(2)}`);
console.log(`RTF:           ${data.rtf?.toFixed(3)}`);
console.log('========================\n');

// Also print full JSON for piping
console.log(JSON.stringify(data, null, 2));

await browser.close();
process.exit(0);
