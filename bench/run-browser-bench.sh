#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "Starting dev server..."
node web/serve.mjs &
SERVER_PID=$!

# Give the server a moment to start
sleep 2

echo "Running browser benchmark..."
node bench/browser-bench.mjs "$@"
EXIT=$?

kill $SERVER_PID 2>/dev/null || true
exit $EXIT
