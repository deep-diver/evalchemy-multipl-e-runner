#!/usr/bin/env bash
set -euo pipefail
# -e: exit on error
# -u: error on unset variables
# -o pipefail: fail pipelines if any part fails

# ----------------------------
# Provider selection
#   PROVIDER=openai            (default)
#   PROVIDER=anthropic-curator (recommended for Claude)
# ----------------------------
PROVIDER="${PROVIDER:-openai}"

# ----------------------------
# Common config (override via env vars)
# ----------------------------
IMAGE="${IMAGE:-deeepdiver/evalchemy-multipl-e:0.1}"
TASKS="${TASKS:-MultiPLE}"
NUM_CONCURRENT="${NUM_CONCURRENT:-8}"
PLATFORM="${PLATFORM:-linux/amd64}"

# Model name meaning depends on PROVIDER:
# - openai: OpenAI model id (e.g., gpt-5.2)
# - anthropic-curator: Claude model id accepted by curator/litellm (e.g., claude-* or anthropic/* depending on your setup)
MODEL="${MODEL:-gpt-5.2}"

# ----------------------------
# Paths (relative to repo)
# ----------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PATCH_OPENAI="${ROOT}/patches/openai_completions.py"
JAVATUPLES="${ROOT}/deps/javatuples-1.2.jar"
LOGS="${ROOT}/logs"
HOST_MULTIPLE_DIR="${HOST_MULTIPLE_DIR:-${ROOT}/multipl}"

# lm_eval OpenAI adapter path inside container (for hot-patch)
DEST_OPENAI="${DEST_OPENAI:-/usr/local/lib/python3.10/dist-packages/lm_eval/models/openai_completions.py}"

mkdir -p "${LOGS}"

# ----------------------------
# Pre-flight checks (common)
# ----------------------------
if [[ ! -f "${JAVATUPLES}" ]]; then
  echo "ERROR: missing dependency jar: ${JAVATUPLES}"
  exit 1
fi

if [[ ! -d "${HOST_MULTIPLE_DIR}" ]]; then
  echo "ERROR: missing MultiPLE override dir: ${HOST_MULTIPLE_DIR}"
  echo "      (set HOST_MULTIPLE_DIR=/path/to/MultiPLE if needed)"
  exit 1
fi

# ----------------------------
# Provider-specific CLI flags / env vars
# ----------------------------
MODEL_BACKEND=""
MODEL_ARGS=""

# Extra docker args that only some providers need
EXTRA_DOCKER_ARGS=()

if [[ "${PROVIDER}" == "openai" ]]; then
  # OpenAI backend uses lm-eval's openai-chat-completions wrapper. :contentReference[oaicite:1]{index=1}
  MODEL_BACKEND="openai-chat-completions"
  MODEL_ARGS="model=${MODEL},num_concurrent=${NUM_CONCURRENT}"

  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set"
    exit 1
  fi

  if [[ ! -f "${PATCH_OPENAI}" ]]; then
    echo "ERROR: missing patch file: ${PATCH_OPENAI}"
    exit 1
  fi

  # Hot-patch OpenAI adapter to avoid unsupported stop for GPT-5.* in your setup
  EXTRA_DOCKER_ARGS+=(
    -e OPENAI_API_KEY
    -e PYTHONPYCACHEPREFIX=/tmp/pycache
    -v "${PATCH_OPENAI}:${DEST_OPENAI}:ro"
  )

elif [[ "${PROVIDER}" == "anthropic-curator" ]]; then
  # Curator backend provides multi-provider API access (Claude/Anthropic included). :contentReference[oaicite:2]{index=2}
  MODEL_BACKEND="curator"

  # Evalchemy docs show curator uses `pretrained=` in --model_args. :contentReference[oaicite:3]{index=3}
  # If your curator build expects `model_name=...` instead, change this line accordingly.
  MODEL_ARGS="pretrained=${MODEL},num_concurrent=${NUM_CONCURRENT}"

  if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ERROR: ANTHROPIC_API_KEY is not set"
    exit 1
  fi

  EXTRA_DOCKER_ARGS+=(
    -e ANTHROPIC_API_KEY
  )

else
  echo "ERROR: unknown PROVIDER='${PROVIDER}'"
  echo "       Use PROVIDER=openai or PROVIDER=anthropic-curator"
  exit 1
fi

# ----------------------------
# Build docker args (array-safe)
# ----------------------------
DOCKER_ARGS=(
  --rm
  -it
  --platform "${PLATFORM}"

  # Mount your MultiPLE override folder (patched benchmark code)
  -v "${HOST_MULTIPLE_DIR}:/workspace/evalchemy/eval/chat_benchmarks/MultiPLE"

  # Persist logs/results
  -v "${LOGS}:/app/logs"

  # Java dependency expected by MultiPL-E Java runner
  -v "${JAVATUPLES}:/usr/multiple/javatuples-1.2.jar:ro"
)

# Append provider-specific docker args (keys/patches)
DOCKER_ARGS+=("${EXTRA_DOCKER_ARGS[@]}")

# ----------------------------
# Build eval args
# ----------------------------
EVAL_ARGS=(
  python3 -m eval.eval
  --model "${MODEL_BACKEND}"
  --tasks "${TASKS}"
  --model_args "${MODEL_ARGS}"
  --output_path /app/logs
)

# OpenAI chat-completions requires chat template in your runs
# (Evalchemy warns to use --apply_chat_template for chat-completions wrapper). :contentReference[oaicite:4]{index=4}
if [[ "${PROVIDER}" == "openai" ]]; then
  EVAL_ARGS+=(--apply_chat_template)
fi

# Optional smoke test
if [[ -n "${DEBUG:-}" ]]; then
  EVAL_ARGS+=(--debug)
fi

# ----------------------------
# Print config
# ----------------------------
echo "[config] PROVIDER=${PROVIDER}"
echo "[config] IMAGE=${IMAGE}"
echo "[config] MODEL_BACKEND=${MODEL_BACKEND}"
echo "[config] MODEL=${MODEL}"
echo "[config] TASKS=${TASKS}"
echo "[config] NUM_CONCURRENT=${NUM_CONCURRENT}"

# ----------------------------
# Execute
# ----------------------------
exec docker run "${DOCKER_ARGS[@]}" "${IMAGE}" "${EVAL_ARGS[@]}"

