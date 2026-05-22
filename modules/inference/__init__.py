"""Inference determinism capability — bitwise-deterministic vLLM (the c3 config)."""
from modules.inference.api import C3_ENV, run_inference, verify_runs

__all__ = ["C3_ENV", "run_inference", "verify_runs"]
