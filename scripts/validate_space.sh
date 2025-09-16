#!/usr/bin/env bash
set -euo pipefail
BASE=${BASE:-http://localhost:7860}

echo "Health:"
curl -s ${BASE}/health | jq . || true

echo "Metrics:"
curl -s ${BASE}/metrics | jq . || true

echo "Jobs:"
curl -s ${BASE}/api/v1/jobs | jq . || true

# Demo run via UI is manual; for API, you can run with a public sample
# curl -s -X POST ${BASE}/api/v1/process/auto -H 'Content-Type: application/json' -d '{"engine":"sota","source_url":"https://..."}' | jq .