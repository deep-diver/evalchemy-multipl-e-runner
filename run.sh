#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Provider selection
#   PROVIDER=openai            : OpenAI via lm-eval openai-chat-completions (patched)
#   PROVIDER=anthropic-direct  : Anthropic via lm-eval anthropic-chat-completions (patched)
#   PROVIDER=anthropic-curator : Anthropic via curator/LiteLLM (may remap model ids)
# ============================================================================
PROVIDER="${PROVIDER:-openai}"

# ============================================================================
# Common configuration (override via host env vars)
# ============================================================================
IMAGE="${IMAGE:-deeepdiver/evalchemy-multipl-e:0.1}"
TASKS="${TASKS:-MultiPLE}"
NUM_CONCURRENT="${NUM_CONCURRENT:-4}"
PLATFORM="${PLATFORM:-linux/amd64}"
MODEL="${MODEL:-gpt-5.2}"
BATCH_SIZE="${BATCH_SIZE:-1}"

# If set on the host, it will be forwarded into the container.
# Example: export MULTIPLE_LANGUAGES="java,python,rs"
MULTIPLE_LANGUAGES="${MULTIPLE_LANGUAGES:-}"

# ============================================================================
# Repo-relative paths
# ============================================================================
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PATCH_OPENAI="${ROOT}/patches/openai_completions.py"
PATCH_ANTHROPIC="${ROOT}/patches/anthropic_completions.py"

# Python auto-import hook to patch LANUGUAGES at runtime (no code changes in MultiPLE)
PATCH_SITECUSTOMIZE="${ROOT}/patches/sitecustomize.py"

JAVATUPLES="${ROOT}/deps/javatuples-1.2.jar"
LOGS="${ROOT}/logs"
HOST_MULTIPLE_DIR="${HOST_MULTIPLE_DIR:-${ROOT}/multipl}"

DEST_OPENAI="${DEST_OPENAI:-/usr/local/lib/python3.10/dist-packages/lm_eval/models/openai_completions.py}"
DEST_ANTHROPIC="${DEST_ANTHROPIC:-/usr/local/lib/python3.10/dist-packages/lm_eval/models/anthropic_llms.py}"

# ============================================================================
# Pre-flight checks (common)
# ============================================================================
mkdir -p "${LOGS}"

if [[ ! -f "${JAVATUPLES}" ]]; then
  echo "ERROR: missing dependency jar: ${JAVATUPLES}"
  exit 1
fi

if [[ ! -d "${HOST_MULTIPLE_DIR}" ]]; then
  echo "ERROR: missing MultiPLE override dir: ${HOST_MULTIPLE_DIR}"
  echo "      (set HOST_MULTIPLE_DIR=/absolute/path/to/MultiPLE if needed)"
  exit 1
fi

# If MULTIPLE_LANGUAGES is set on the host, create a patched MultiPLE folder and mount that instead.
if [[ -n "${MULTIPLE_LANGUAGES:-}" ]]; then
  PATCHED_MULTIPLE_DIR="$(python3 "${ROOT}/tools/patch_multipl_languages.py")"
  echo "[patch] Using patched MultiPLE dir: ${PATCHED_MULTIPLE_DIR}"
  HOST_MULTIPLE_DIR="${PATCHED_MULTIPLE_DIR}"
fi

# ============================================================================
# Provider-specific settings
# ============================================================================
MODEL_BACKEND=""
MODEL_ARGS=""
TIMEOUT="${TIMEOUT:-300}"
EXTRA_DOCKER_ARGS=()

case "${PROVIDER}" in
  openai)
    MODEL_BACKEND="openai-chat-completions"
    MODEL_ARGS="model=${MODEL},num_concurrent=${NUM_CONCURRENT},timeout=${TIMEOUT}"

    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
      echo "ERROR: OPENAI_API_KEY is not set"
      exit 1
    fi
    if [[ ! -f "${PATCH_OPENAI}" ]]; then
      echo "ERROR: missing patch file: ${PATCH_OPENAI}"
      exit 1
    fi

    EXTRA_DOCKER_ARGS+=(
      -e OPENAI_API_KEY
      -e PYTHONPYCACHEPREFIX=/tmp/pycache
      -v "${PATCH_OPENAI}:${DEST_OPENAI}:ro"
    )
    ;;

  anthropic-direct)
    MODEL_BACKEND="anthropic-chat-completions"
    MODEL_ARGS="model=${MODEL},num_concurrent=${NUM_CONCURRENT}"

    if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
      echo "ERROR: ANTHROPIC_API_KEY is not set"
      exit 1
    fi
    if [[ ! -f "${PATCH_ANTHROPIC}" ]]; then
      echo "ERROR: missing patch file: ${PATCH_ANTHROPIC}"
      exit 1
    fi

    EXTRA_DOCKER_ARGS+=(
      -e ANTHROPIC_API_KEY
      -e PYTHONPYCACHEPREFIX=/tmp/pycache
      -v "${PATCH_ANTHROPIC}:${DEST_ANTHROPIC}:ro"
    )
    ;;

  anthropic-curator)
    MODEL_BACKEND="curator"

    if [[ "${MODEL}" != */* ]]; then
      MODEL="anthropic/${MODEL}"
    fi
    MODEL_ARGS="pretrained=${MODEL},num_concurrent=${NUM_CONCURRENT}"

    if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
      echo "ERROR: ANTHROPIC_API_KEY is not set"
      exit 1
    fi

    EXTRA_DOCKER_ARGS+=(
      -e ANTHROPIC_API_KEY
    )
    ;;

  *)
    echo "ERROR: unknown PROVIDER='${PROVIDER}'"
    echo "       Allowed: openai | anthropic-direct | anthropic-curator"
    exit 1
    ;;
esac

# ============================================================================
# Optional runtime language override via sitecustomize.py (no MultiPLE code changes)
#
# How it works:
# - Python automatically imports `sitecustomize` if it is importable on PYTHONPATH.
# - We mount `patches/sitecustomize.py` into /workspace/patches
# - We add /workspace/patches to PYTHONPATH
# - sitecustomize reads env var MULTIPLE_LANGUAGES and patches MultiPLE.LANUGUAGES
# ============================================================================
if [[ -n "${MULTIPLE_LANGUAGES}" ]]; then
  EXTRA_DOCKER_ARGS+=(
    -e MULTIPLE_LANGUAGES
    -e PYTHONPATH="/workspace/patches:${PYTHONPATH:-}"
    -v "${ROOT}/patches:/workspace/patches:ro"
  )
fi

# ============================================================================
# Build docker args (array-safe)
# ============================================================================
DOCKER_ARGS=(
  # --rm
  -it
  --platform "${PLATFORM}"

  # Override the MultiPLE benchmark code inside the container
  -v "${HOST_MULTIPLE_DIR}:/workspace/evalchemy/eval/chat_benchmarks/MultiPLE"

  # Persist logs/results on the host
  -v "${LOGS}:/app/logs"

  # Provide Java dependency at hardcoded location expected by MultiPL-E
  -v "${JAVATUPLES}:/usr/multiple/javatuples-1.2.jar:ro"
)

DOCKER_ARGS+=("${EXTRA_DOCKER_ARGS[@]}")

# ============================================================================
# Build eval args
# ============================================================================
EVAL_ARGS=(
  python3 -m eval.eval
  --model "${MODEL_BACKEND}"
  --tasks "${TASKS}"
  --model_args "${MODEL_ARGS}"
  --batch_size "${BATCH_SIZE}"
  --output_path /app/logs
)

# Apply chat template for known chat backends
if [[ "${PROVIDER}" == "openai" || "${PROVIDER}" == "anthropic-direct" ]]; then
  EVAL_ARGS+=(--apply_chat_template)
fi

# Optional smoke test
if [[ -n "${DEBUG:-}" ]]; then
  EVAL_ARGS+=(--debug)
fi

# ============================================================================
# Print effective config
# ============================================================================
echo "[config] PROVIDER=${PROVIDER}"
echo "[config] IMAGE=${IMAGE}"
echo "[config] MODEL_BACKEND=${MODEL_BACKEND}"
echo "[config] MODEL=${MODEL}"
echo "[config] TASKS=${TASKS}"
echo "[config] NUM_CONCURRENT=${NUM_CONCURRENT}"
echo "[config] PLATFORM=${PLATFORM}"
echo "[config] HOST_MULTIPLE_DIR=${HOST_MULTIPLE_DIR}"
echo "[config] BATCH_SIZE=${BATCH_SIZE}"
echo "[config] TIMEOUT=${TIMEOUT}"
if [[ -n "${MULTIPLE_LANGUAGES}" ]]; then
  echo "[config] MULTIPLE_LANGUAGES=${MULTIPLE_LANGUAGES}"
fi

# ============================================================================
# Execute
# ============================================================================
exec docker run "${DOCKER_ARGS[@]}" "${IMAGE}" "${EVAL_ARGS[@]}"
