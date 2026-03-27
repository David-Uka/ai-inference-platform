#!/usr/bin/env bash

set -euo pipefail

TARGET_URL="${1:-http://localhost:8000/v1/infer}"
ITERATIONS="${ITERATIONS:-20}"
PAYLOAD='{"text":"This platform design is thoughtful, practical, and very promising."}'

echo "Sending ${ITERATIONS} inference requests to ${TARGET_URL}"

for ((i = 1; i <= ITERATIONS; i++)); do
  curl -sS \
    -X POST "${TARGET_URL}" \
    -H "Content-Type: application/json" \
    -d "${PAYLOAD}" >/dev/null
  echo "Request ${i} completed"
done
