cat > run.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-deepdiver/evalchemy-multipl-e:0.1}"
MODEL="${MODEL:-gpt-5.2}"
TASKS="${TASKS:-MultiPLE}"
NUM_CONCURRENT="${NUM_CONCURRENT:-8}"
PLATFORM="${PLATFORM:-linux/amd64}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PATCH_OPENAI="${ROOT}/patches/openai_completions.py"
JAVATUPLES="${ROOT}/deps/javatuples-1.2.jar"
LOGS="${ROOT}/logs"
HOST_MULTIPLE_DIR="${HOST_MULTIPLE_DIR:-${ROOT}/multipl}"

DEST="${DEST:-/usr/local/lib/python3.10/dist-packages/lm_eval/models/openai_completions.py}"

mkdir -p "${LOGS}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set"
  exit 1
fi

if [[ ! -f "${PATCH_OPENAI}" ]]; then
  echo "ERROR: missing ${PATCH_OPENAI}"
  exit 1
fi

if [[ ! -f "${JAVATUPLES}" ]]; then
  echo "ERROR: missing ${JAVATUPLES}"
  exit 1
fi

if [[ ! -d "${HOST_MULTIPLE_DIR}" ]]; then
  echo "ERROR: missing ${HOST_MULTIPLE_DIR} (set HOST_MULTIPLE_DIR to your MultiPLE dir)"
  exit 1
fi

exec docker run --rm -it --platform "${PLATFORM}" \
  -e OPENAI_API_KEY \
  -e PYTHONPYCACHEPREFIX=/tmp/pycache \
  -v "${HOST_MULTIPLE_DIR}:/workspace/evalchemy/eval/chat_benchmarks/MultiPLE" \
  -v "${PATCH_OPENAI}:${DEST}:ro" \
  -v "${LOGS}:/app/logs" \
  -v "${JAVATUPLES}:/usr/multiple/javatuples-1.2.jar:ro" \
  "${IMAGE}" \
  python3 -m eval.eval \
    --model openai-chat-completions \
    --tasks "${TASKS}" \
    --model_args "model=${MODEL},num_concurrent=${NUM_CONCURRENT}" \
    --apply_chat_template \
    --output_path /app/logs \
    ${DEBUG:+--debug}
SH

chmod +x run.sh

