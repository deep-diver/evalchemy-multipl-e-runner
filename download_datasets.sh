#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Download all benchmark datasets for evalchemy-multipl-e-runner
# Downloads datasets for MultiPL-E, HumanEval, and MBPP tasks
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

echo "  Note: The official MultiPL-E repository (nuprl/MultiPL-E) has"
echo "  a different structure. For this setup, you need to either:"
echo ""
echo "  1. Clone and copy from an existing setup:"
echo "     git clone https://github.com/nuprl/MultiPL-E.git /tmp/multipl"
echo "     cp /tmp/multipl/multipl_e/*/multipl-e-*.json multiple_data/"
echo ""
echo "  2. Or use the datasets from the evalchemy container directly"
echo ""
echo "  Skipping MultiPL-E download for now."
echo ""

# ============================================================================
# MBPP Datasets
# ============================================================================
echo "[2/3] Downloading MBPP datasets..."

# MBPP from HuggingFace Muennighoff/mbpp (reliable source)
MBPP_BASE_URL="https://huggingface.co/datasets/Muennighoff/mbpp/resolve/main/data"

echo "  Downloading mbpp.jsonl from HuggingFace..."
if curl -s -L -o "mbpp_data/mbpp.jsonl" "${MBPP_BASE_URL}/mbpp.jsonl"; then
    if head -c 10 "mbpp_data/mbpp.jsonl" | grep -q '{"'; then
        echo "  ✓ Downloaded mbpp.jsonl"
    else
        echo "  ✗ mbpp.jsonl invalid"
        rm -f "mbpp_data/mbpp.jsonl"
    fi
else
    echo "  ✗ Failed to download mbpp.jsonl"
fi

# For mbpp_test.jsonl, we create it from mbpp.jsonl (lines 11-510)
if [ -f "mbpp_data/mbpp.jsonl" ]; then
    echo "  Creating mbpp_test.jsonl (test samples 11-510)..."
    if python3 -c "
import json
with open('mbpp_data/mbpp.jsonl', 'r') as f:
    lines = f.readlines()
# MBPP uses samples 11-510 for testing (500 samples)
test_samples = lines[10:510]
with open('mbpp_data/mbpp_test.jsonl', 'w') as f:
    f.writelines(test_samples)
print(f'Created mbpp_test.jsonl with {len(test_samples)} samples')
" 2>/dev/null; then
        echo "  ✓ Created mbpp_test.jsonl"
    else
        echo "  ✗ Failed to create mbpp_test.jsonl"
    fi
fi

# ============================================================================
# HumanEval Datasets
# ============================================================================
echo ""
echo "[3/3] Downloading HumanEval datasets..."

# HumanEval from OpenAI's official repository (uncompressed)
HUMANEVAL_URL="https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl"

echo "  Downloading humaneval.jsonl..."
if curl -s -L -o "humaneval_data/humaneval.jsonl" "${HUMANEVAL_URL}"; then
    if head -c 10 "humaneval_data/humaneval.jsonl" | grep -q '{"'; then
        echo "  ✓ Downloaded humaneval.jsonl"
    else
        echo "  ✗ humaneval.jsonl invalid"
        rm -f "humaneval_data/humaneval.jsonl"
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

multi_count=$(ls multiple_data/multipl-e-*.json 2>/dev/null | wc -l | tr -d ' ')
mbpp_count=$(ls mbpp_data/*.jsonl 2>/dev/null | wc -l | tr -d ' ')
humaneval_count=$(ls humaneval_data/*.jsonl 2>/dev/null | wc -l | tr -d ' ')

echo "MultiPL-E: ${multi_count} datasets"
echo "MBPP:      ${mbpp_count} files"
echo "HumanEval: ${humaneval_count} file(s)"

echo ""
echo "Dataset directories:"
echo "  - multiple_data/   (MultiPL-E)"
echo "  - mbpp_data/       (MBPP)"
echo "  - humaneval_data/  (HumanEval)"
echo ""
echo "For MultiPL-E datasets, please manually copy from an existing installation"
echo "or clone: git clone https://github.com/nuprl/MultiPL-E.git"
echo ""
