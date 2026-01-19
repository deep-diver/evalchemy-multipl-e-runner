#!/usr/bin/env python3
"""
Wrapper script to run evalchemy with Vertex AI support.

This script imports the vertex_completions module to register the Vertex AI models
before running the eval.
"""
import sys

# Import vertex_completions first to register the model
try:
    from lm_eval.models import vertex_completions
except ImportError:
    pass

# Then run the normal eval entry point
from eval.eval import cli_evaluate
sys.exit(cli_evaluate())
