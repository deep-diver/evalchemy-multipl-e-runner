#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Provider selection
#   PROVIDER=openai            : OpenAI via lm-eval openai-chat-completions (patched)
#   PROVIDER=anthropic-direct  : Anthropic via lm-eval anthropic-chat-completions (patched)
#   PROVIDER=anthropic-curator : Anthropic via curator/LiteLLM (may remap model ids)
#   PROVIDER=openrouter        : OpenRouter(OpenAI-compatible; Gemini etc.) via local-chat-completions
#   PROVIDER=google-direct     : Google via lm-eval google-chat-completions (patched)
#   PROVIDER=vllm              : vLLM completions API (for pretrain tasks like humaneval)
#   PROVIDER=vllm-chat         : vLLM chat completions API (for instruct tasks like CodeElo)
#   PROVIDER=vertex-direct     : Vertex AI via patched vertex_completions
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
TIMEOUT="${TIMEOUT:-300}"

# If set on the host, it will be forwarded into the container.
# Example: export MULTIPLE_LANGUAGES="java,python,rs"
MULTIPLE_LANGUAGES="${MULTIPLE_LANGUAGES:-}"

# ============================================================================
# Repo-relative paths
# ============================================================================
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PATCH_OPENAI="${ROOT}/patches/openai_completions.py"
PATCH_ANTHROPIC="${ROOT}/patches/anthropic_completions.py"
PATCH_GOOGLE="${ROOT}/patches/google_completions.py"
PATCH_MODELS_INIT="${ROOT}/patches/__init__.py"
PATCH_EVAL="${ROOT}/patches/eval.py"

# Optional runtime language override
PATCH_SITECUSTOMIZE="${ROOT}/patches/sitecustomize.py"

JAVATUPLES="${ROOT}/deps/javatuples-1.2.jar"
LOGS="${ROOT}/logs"
HOST_MULTIPLE_DIR="${HOST_MULTIPLE_DIR:-${ROOT}/multipl}"

DEST_OPENAI="${DEST_OPENAI:-/usr/local/lib/python3.10/dist-packages/lm_eval/models/openai_completions.py}"
DEST_ANTHROPIC="${DEST_ANTHROPIC:-/usr/local/lib/python3.10/dist-packages/lm_eval/models/anthropic_llms.py}"
DEST_GOOGLE="${DEST_GOOGLE:-/usr/local/lib/python3.10/dist-packages/lm_eval/models/google_completions.py}"
DEST_MODELS_INIT="${DEST_MODELS_INIT:-/usr/local/lib/python3.10/dist-packages/lm_eval/models/__init__.py}"

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
EXTRA_DOCKER_ARGS=()

# OpenRouter defaults (OpenAI-compatible)
# Docs: base URL + auth are OpenAI-style. :contentReference[oaicite:2]{index=2}
OPENROUTER_BASE_URL="${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}"

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
    MODEL_ARGS="model=${MODEL},num_concurrent=${NUM_CONCURRENT},timeout=${TIMEOUT}"

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

  openrouter)
    # OpenRouter is OpenAI-compatible.
    # - Base URL: https://openrouter.ai/api/v1 (or docs may show openrouter.co/v1). :contentReference[oaicite:3]{index=3}
    # - Auth header: Authorization: Bearer <key>. :contentReference[oaicite:4]{index=4}
    MODEL_BACKEND="local-chat-completions"
    MODEL_ARGS="model=${MODEL},base_url=${OPENROUTER_BASE_URL}/chat/completions,num_concurrent=${NUM_CONCURRENT},timeout=${TIMEOUT}"

    if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
      echo "ERROR: OPENROUTER_API_KEY is not set"
      exit 1
    fi

    # lm_eval's local-* backends read OPENAI_API_KEY, so we map OpenRouter key into OPENAI_API_KEY.
    export OPENAI_API_KEY="${OPENROUTER_API_KEY}"

    # Optional attribution headers exist on OpenRouter side, but we only forward envs for now. :contentReference[oaicite:5]{index=5}
    EXTRA_DOCKER_ARGS+=(
      -e OPENAI_API_KEY
      -e OPENROUTER_API_KEY
      -e OPENROUTER_BASE_URL
      -e OPENROUTER_SITE_URL
      -e OPENROUTER_APP_NAME
      -e PYTHONPYCACHEPREFIX=/tmp/pycache
      -v "${PATCH_OPENAI}:${DEST_OPENAI}:ro"
    )
    ;;

  google-direct)
    # Use Gemini's OpenAI-compatible endpoint
    # https://ai.google.dev/gemini-api/docs/openai
    # Note: lm-eval posts directly to base_url, so we need the full path
    MODEL_BACKEND="local-chat-completions"
    MODEL_ARGS="model=${MODEL},base_url=https://generativelanguage.googleapis.com/v1beta/openai/chat/completions,num_concurrent=${NUM_CONCURRENT},timeout=${TIMEOUT}"

    if [[ -z "${GOOGLE_API_KEY:-}" ]]; then
      if [[ -z "${GEMINI_API_KEY:-}" ]]; then
        echo "ERROR: GOOGLE_API_KEY or GEMINI_API_KEY is not set"
        exit 1
      fi
    fi

    # local-chat-completions reads OPENAI_API_KEY, so we map Google key to it
    export OPENAI_API_KEY="${GOOGLE_API_KEY:-${GEMINI_API_KEY}}"

    EXTRA_DOCKER_ARGS+=(
      -e OPENAI_API_KEY
      -e GOOGLE_API_KEY
      -e GEMINI_API_KEY
      -e PYTHONPYCACHEPREFIX=/tmp/pycache
      -v "${PATCH_OPENAI}:${DEST_OPENAI}:ro"
    )
    ;;

  vertex-direct)
    # Use Vertex AI endpoint (better performance and rate limits)
    # Requires Google Cloud credentials (gcloud auth application-default login)
    # https://cloud.google.com/vertex-ai/generative-ai/docs/sdks/overview
    MODEL_BACKEND="vertex-chat-completions"
    MODEL_ARGS="model=${MODEL},num_concurrent=${NUM_CONCURRENT},timeout=${TIMEOUT},max_gen_toks=65536,max_length=32768"

    # Optional: Set project and location
    if [[ -n "${GOOGLE_CLOUD_PROJECT:-}" ]]; then
      MODEL_ARGS="${MODEL_ARGS},project=${GOOGLE_CLOUD_PROJECT}"
    fi
    if [[ -n "${GOOGLE_CLOUD_LOCATION:-}" ]]; then
      MODEL_ARGS="${MODEL_ARGS},location=${GOOGLE_CLOUD_LOCATION}"
    fi

    # Mount vertex_completions.py patch and use wrapper script
    PATCH_VERTEX="${ROOT}/patches/vertex_completions.py"
    DEST_VERTEX="/usr/local/lib/python3.10/dist-packages/lm_eval/models/vertex_completions.py"
    WRAPPER_SCRIPT="${ROOT}/patches/run_with_vertex.py"
    DEST_WRAPPER="/app/run_with_vertex.py"

    # Handle service account credentials file
    # Mount to a separate location outside of /root/.config/gcloud to avoid conflicts
    if [[ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]]; then
      # Get the filename from the path
      CREDS_FILENAME="$(basename "${GOOGLE_APPLICATION_CREDENTIALS}")"
      DEST_CREDS="/app/gcp-credentials/${CREDS_FILENAME}"
      EXTRA_DOCKER_ARGS+=(
        -v "${GOOGLE_APPLICATION_CREDENTIALS}:${DEST_CREDS}:ro"
        -e GOOGLE_APPLICATION_CREDENTIALS="${DEST_CREDS}"
      )
    fi

    EXTRA_DOCKER_ARGS+=(
      -e GOOGLE_CLOUD_PROJECT
      -e GOOGLE_CLOUD_LOCATION
      # Mount gcloud config for ADC
      -v "/Users/deep-diver/.config/gcloud:/root/.config/gcloud:ro"
      -v "${PATCH_VERTEX}:${DEST_VERTEX}:ro"
      -v "${WRAPPER_SCRIPT}:${DEST_WRAPPER}:ro"
      -v "${PATCH_OPENAI}:${DEST_OPENAI}:ro"
      -e PYTHONPYCACHEPREFIX=/tmp/pycache
      -e PYTHONDONTWRITEBYTECODE=1
    )
    ;;

  vllm)
    # vLLM provides OpenAI-compatible API (completions endpoint)
    # https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
    # Set VLLM_BASE_URL to your vLLM endpoint (e.g., http://localhost:8000)
    # Use this for pretrain tasks like humaneval, mbpp, etc.
    VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000}"
    MODEL_BACKEND="local-completions"
    MODEL_ARGS="model=${MODEL},base_url=${VLLM_BASE_URL}/v1/completions,num_concurrent=${NUM_CONCURRENT},timeout=${TIMEOUT},max_gen_toks=4096,max_length=4096"

    export OPENAI_API_KEY="${VLLM_API_KEY:-dummy-key}"

    EXTRA_DOCKER_ARGS+=(
      -e OPENAI_API_KEY
      -e VLLM_API_KEY
      -e VLLM_BASE_URL
      -e PYTHONPYCACHEPREFIX=/tmp/pycache
    )
    ;;

  vllm-chat)
    # vLLM provides OpenAI-compatible API (chat completions endpoint)
    # Use this for instruct/chat tasks like CodeElo, MultiPLE, etc.
    VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000}"
    MODEL_BACKEND="openai-chat-completions"
    MODEL_ARGS="model=${MODEL},base_url=${VLLM_BASE_URL}/v1/chat/completions,num_concurrent=${NUM_CONCURRENT},timeout=${TIMEOUT},max_gen_toks=4096,max_length=4096"

    # Qwen3-specific generation parameters
    # Note: Thinking mode is automatically disabled in the Python patch for Qwen3 models
    # To enable thinking mode, you would need to modify the patch or use server-level config
    if [[ "${MODEL}" == *"Qwen3"* ]]; then
      # Temperature (controlled by VLLM_TEMPERATURE)
      if [[ -n "${VLLM_TEMPERATURE:-}" ]]; then
        MODEL_ARGS="${MODEL_ARGS},temperature=${VLLM_TEMPERATURE}"
      fi

      # Top-p (controlled by VLLM_TOP_P)
      if [[ -n "${VLLM_TOP_P:-}" ]]; then
        MODEL_ARGS="${MODEL_ARGS},top_p=${VLLM_TOP_P}"
      fi

      # Top-k (controlled by VLLM_TOP_K)
      if [[ -n "${VLLM_TOP_K:-}" ]]; then
        MODEL_ARGS="${MODEL_ARGS},top_k=${VLLM_TOP_K}"
      fi
    fi

    export OPENAI_API_KEY="${VLLM_API_KEY:-dummy-key}"

    if [[ ! -f "${PATCH_OPENAI}" ]]; then
      echo "ERROR: missing patch file: ${PATCH_OPENAI}"
      exit 1
    fi

    # Pass Qwen3 environment variables into container if set
    if [[ "${MODEL}" == *"Qwen3"* ]]; then
      EXTRA_DOCKER_ARGS+=(
        -e VLLM_TEMPERATURE
        -e VLLM_TOP_P
        -e VLLM_TOP_K
      )
    fi

    EXTRA_DOCKER_ARGS+=(
      -e OPENAI_API_KEY
      -e VLLM_API_KEY
      -e VLLM_BASE_URL
      -e PYTHONPYCACHEPREFIX=/tmp/pycache
      -v "${PATCH_OPENAI}:${DEST_OPENAI}:ro"
    )
    ;;

  *)
    echo "ERROR: unknown PROVIDER='${PROVIDER}'"
    echo "       Allowed: openai | anthropic-direct | anthropic-curator | openrouter | google-direct | vertex-direct | vllm | vllm-chat"
    exit 1
    ;;
esac

# ============================================================================
# Optional runtime language override via sitecustomize.py
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
  --platform "${PLATFORM}"

  # Override the MultiPLE benchmark code inside the container
  -v "${HOST_MULTIPLE_DIR}:/workspace/evalchemy/eval/chat_benchmarks/MultiPLE"

  # Persist logs/results on the host
  -v "${LOGS}:/app/logs"

  # Provide Java dependency at hardcoded location expected by MultiPL-E
  -v "${JAVATUPLES}:/usr/multiple/javatuples-1.2.jar:ro"

  # Patch eval.py to add confirm_run_unsafe_code parameter
  -v "${PATCH_EVAL}:/workspace/evalchemy/eval/eval.py:ro"

  # Allow code execution for HumanEval and other code benchmarks
  -e HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"
  -e CONFIRM_RUN_UNSAFE_CODE="${CONFIRM_RUN_UNSAFE_CODE:-True}"
)

DOCKER_ARGS+=("${EXTRA_DOCKER_ARGS[@]}")

# ============================================================================
# Build eval args
# ============================================================================
if [[ "${PROVIDER}" == "vertex-direct" ]]; then
  # Use wrapper script for vertex-direct to load vertex_completions module
  EVAL_ARGS=(
    python3 /app/run_with_vertex.py
    --model "${MODEL_BACKEND}"
    --tasks "${TASKS}"
    --model_args "${MODEL_ARGS}"
    --batch_size "${BATCH_SIZE}"
    --output_path /app/logs
  )
else
  EVAL_ARGS=(
    python3 -m eval.eval
    --model "${MODEL_BACKEND}"
    --tasks "${TASKS}"
    --model_args "${MODEL_ARGS}"
    --batch_size "${BATCH_SIZE}"
    --output_path /app/logs
  )
fi

# Apply chat template for chat backends
if [[ "${PROVIDER}" == "openai" || "${PROVIDER}" == "anthropic-direct" || "${PROVIDER}" == "openrouter" || "${PROVIDER}" == "google-direct" || "${PROVIDER}" == "vertex-direct" || "${PROVIDER}" == "vllm-chat" ]]; then
  EVAL_ARGS+=(--apply_chat_template)
fi

# Optional smoke test
if [[ -n "${DEBUG:-}" ]]; then
  EVAL_ARGS+=(--debug)
fi

# Allow unsafe code execution for HumanEval and other code benchmarks
EVAL_ARGS+=(--confirm_run_unsafe_code)

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
if [[ "${PROVIDER}" == "openrouter" ]]; then
  echo "[config] OPENROUTER_BASE_URL=${OPENROUTER_BASE_URL}"
fi
if [[ -n "${MULTIPLE_LANGUAGES}" ]]; then
  echo "[config] MULTIPLE_LANGUAGES=${MULTIPLE_LANGUAGES}"
fi

# ============================================================================
# Execute
# ============================================================================
exec docker run "${DOCKER_ARGS[@]}" "${IMAGE}" "${EVAL_ARGS[@]}"
