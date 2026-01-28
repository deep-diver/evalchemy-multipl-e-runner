#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Download all benchmark datasets for evalchemy-multipl-e-runner
# Downloads datasets for MultiPLE, HumanEval, and MBPP tasks
# ============================================================================

echo "=========================================="
echo "Downloading Benchmark Datasets"
echo "=========================================="

# Create data directories
mkdir -p multiple_data
mkdir -p mbpp_data
mkdir -p humaneval_data

# ============================================================================
# MultiPL-E Datasets
# ============================================================================
echo ""
echo "[1/3] Downloading MultiPL-E datasets..."

# Available languages in MultiPL-E
LANGUAGES="adb clj cpp cs dart elixir go hs java js julia lua php pl r racket rs ruby scala sh swift ts"

# Base URL for MultiPL-E raw GitHub content
MULTIPLE_BASE_URL="https://raw.githubusercontent.com/nusddebson/MultiPL-E/main/eval/chat_benchmarks/MultiPLE/data"

multiple_count=0
for lang in $LANGUAGES; do
    echo -n "  Downloading multipl-e-${lang}.json... "
    if curl -s -L -o "multiple_data/multipl-e-${lang}.json" \
            "${MULTIPLE_BASE_URL}/multipl-e-${lang}.json" 2>/dev/null; then
        # Verify it's a valid JSON (not HTML/404)
        if head -c 10 "multiple_data/multipl-e-${lang}.json" | grep -q '{"'; then
            echo "✓"
            ((multiple_count++)) || true
        else
            echo "✗ (invalid)"
            rm -f "multiple_data/multipl-e-${lang}.json"
        fi
    else
        echo "✗ (failed)"
    fi
done

echo "  Downloaded ${multiple_count} MultiPL-E datasets"

# ============================================================================
# MBPP Datasets
# ============================================================================
echo ""
echo "[2/3] Downloading MBPP datasets..."

# MBPP from HuggingFace (reliable source)
MBPP_BASE_URL="https://huggingface.co/datasets/jash404/mbpp/resolve/main"

if curl -s -L -o "mbpp_data/mbpp.jsonl" "${MBPP_BASE_URL}/mbpp.jsonl" && \
   curl -s -L -o "mbpp_data/mbpp_test.jsonl" "${MBPP_BASE_URL}/mbpp_test.jsonl"; then
    # Verify
    if head -c 10 "mbpp_data/mbpp.jsonl" | grep -q '{"' && \
       head -c 10 "mbpp_data/mbpp_test.jsonl" | grep -q '{"'; then
        echo "  ✓ Downloaded mbpp.jsonl and mbpp_test.jsonl"
    else
        echo "  ✗ MBPP files invalid"
        rm -f mbpp_data/*.jsonl
    fi
else
    echo "  ✗ Failed to download MBPP datasets"
fi

# ============================================================================
# HumanEval Datasets
# ============================================================================
echo ""
echo "[3/3] Downloading HumanEval datasets..."

# HumanEval from OpenAI's official repository
HUMANEVAL_BASE_URL="https://raw.githubusercontent.com/openai/human-eval/master/data"

if curl -s -L -o "humaneval_data/humaneval.jsonl" "${HUMANEVAL_BASE_URL}/HumanEval.jsonl.gz"; then
    # HumanEval is distributed as gzipped jsonl
    if gunzip -c "humaneval_data/humaneval.jsonl" > "humaneval_data/humaneval.jsonl.tmp" 2>/dev/null; then
        mv "humaneval_data/humaneval.jsonl.tmp" "humaneval_data/humaneval.jsonl"
        rm -f "humaneval_data/humaneval.jsonl.gz" 2>/dev/null || true
        echo "  ✓ Downloaded humaneval.jsonl"
    else
        # Try uncompressed version
        if curl -s -L -o "humaneval_data/humaneval.jsonl" "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl"; then
            echo "  ✓ Downloaded humaneval.jsonl (uncompressed)"
        else
            echo "  ✗ Failed to download HumanEval"
            rm -f humaneval_data/humaneval.jsonl*
        fi
    fi
else
    echo "  ✗ Failed to download HumanEval"
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=========================================="
echo "Download Summary"
echo "=========================================="

multi_count=$(ls multiple_data/multipl-e-*.json 2>/dev/null | wc -l)
mbpp_count=$(ls mbpp_data/*.jsonl 2>/dev/null | wc -l)
humaneval_count=$(ls humaneval_data/*.jsonl 2>/dev/null | wc -l)

echo "MultiPL-E: ${multi_count} datasets"
echo "MBPP:      ${mbpp_count} files"
echo "HumanEval: ${humaneval_count} file(s)"

echo ""
echo "Dataset directories:"
echo "  - multiple_data/   (MultiPL-E)"
echo "  - mbpp_data/       (MBPP)"
echo "  - humaneval_data/  (HumanEval)"
echo ""
echo "You can now run benchmarks with ./run.sh"
