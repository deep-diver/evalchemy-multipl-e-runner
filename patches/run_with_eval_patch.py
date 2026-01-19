#!/usr/bin/env python3
"""Patch eval.py to add confirm_run_unsafe_code parameter before running evaluation."""
import subprocess
import sys

# Apply the patch
patch_cmd = [
    "patch",
    "-p0",
    "-f",  # force, don't ask questions
    "/workspace/evalchemy/eval/eval.py",
    "/app/eval.py.patch"
]

result = subprocess.run(patch_cmd, capture_output=True, text=True)
if result.returncode != 0:
    # Patch might already be applied, or patch utility might not be available
    # Try alternative: just run the eval directly
    pass

# Now run the actual eval with original args
sys.argv = ["eval.eval"] + sys.argv[1:]
from eval import eval
eval.cli_evaluate()
