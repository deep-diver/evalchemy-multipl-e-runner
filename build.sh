#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Build script for local Docker image with Google GenAI SDK
# ============================================================================

IMAGE="${IMAGE:-deeepdiver/evalchemy-multipl-e:0.1-google}"
PLATFORM="${PLATFORM:-linux/amd64}"

echo "[build] Building Docker image: ${IMAGE}"
echo "[build] Platform: ${PLATFORM}"

docker buildx build \
  --platform "${PLATFORM}" \
  --tag "${IMAGE}" \
  --load \
  --file Dockerfile \
  .

echo "[build] Done! Image built: ${IMAGE}"
echo ""
echo "To use this image, run:"
echo "  IMAGE=${IMAGE} PROVIDER=google-direct ./run.sh"
