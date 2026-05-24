#!/usr/bin/env python3
"""vLLM execution backend for the deterministic runner.

Loads a model via vLLM's offline LLM class with batch invariance,
executes requests, and returns structured observables.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any


def _set_deterministic_env(knobs: dict[str, Any], *, tp_size: int = 1, pp_size: int = 1) -> dict[str, str]:
    """Set environment variables for deterministic execution. Returns the env snapshot."""
    env = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_LAUNCH_BLOCKING": str(int(knobs.get("cuda_launch_blocking", True))),
        "PYTHONHASHSEED": "0",
    }
    # Pin NCCL collective algorithms for distributed determinism
    if tp_size > 1 or pp_size > 1:
        env["NCCL_ALGO"] = "Ring"
        env["NCCL_PROTO"] = "Simple"
        env["NCCL_DEBUG"] = "WARN"
    # Multi-node: force NCCL over TCP sockets, disable local shortcuts
    if pp_size > 1 or os.getenv("VLLM_MULTI_NODE"):
        env["NCCL_SOCKET_IFNAME"] = os.getenv("NCCL_SOCKET_IFNAME", "eth0")
        env["NCCL_NET"] = "Socket"
        env["NCCL_P2P_DISABLE"] = "1"
        env["NCCL_SHM_DISABLE"] = "1"
        env["NCCL_BUFFSIZE"] = "8388608"
    for key, value in env.items():
        os.environ[key] = value
    return env


def _resolve_model_path(manifest: dict[str, Any], lockfile: dict[str, Any]) -> str:
    """Determine the model path/name for vLLM.

    Prefers a local model directory if RUNNER_MODEL_PATH is set,
    otherwise uses the HF model ID from the manifest source field.
    """
    env_path = os.getenv("RUNNER_MODEL_PATH")
    if env_path and Path(env_path).is_dir():
        return env_path

    source = manifest["model"]["source"]
    if source.startswith("hf://"):
        return source[len("hf://"):]
    return source


def run_vllm(
    manifest: dict[str, Any],
    lockfile: dict[str, Any],
) -> dict[str, Any]:
    """Execute vLLM inference and return observables.

    Returns dict with keys: request_outputs, engine_events, frames, env_info
    """
    # These imports are deferred so the module can be imported on machines without vLLM
    # (e.g. for schema validation or synthetic mode).
    import torch
    from vllm import LLM, SamplingParams

    runtime = manifest["runtime"]
    knobs = runtime["deterministic_knobs"]
    batch_inv = runtime.get("batch_invariance", {})
    serving_engine = runtime.get("serving_engine", {})

    tp = serving_engine.get("tensor_parallel_size") or 1
    pp = serving_engine.get("pipeline_parallel_size") or 1

    resolved_env = _set_deterministic_env(knobs, tp_size=tp, pp_size=pp)

    attn_backend = serving_engine.get("attention_backend")
    if attn_backend:
        os.environ["VLLM_ATTENTION_BACKEND"] = attn_backend
        resolved_env["VLLM_ATTENTION_BACKEND"] = attn_backend

    # vLLM batch invariance — use env var (works across vLLM versions)
    if batch_inv.get("enabled", False):
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
        resolved_env["VLLM_BATCH_INVARIANT"] = "1"

    if knobs.get("torch_deterministic", False):
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_path = _resolve_model_path(manifest, lockfile)
    seed = knobs.get("seed", 42)

    engine_kwargs: dict[str, Any] = {
        "model": model_path,
        "seed": seed,
        "dtype": serving_engine.get("dtype", "auto"),
        "trust_remote_code": bool(manifest["model"].get("trust_remote_code", False)),
        "gpu_memory_utilization": float(os.getenv("RUNNER_GPU_MEM_UTIL",
                                                   str(serving_engine.get("gpu_memory_utilization", 0.90)))),
    }

    if batch_inv.get("enforce_eager", False):
        engine_kwargs["enforce_eager"] = True

    max_model_len = os.getenv("RUNNER_MAX_MODEL_LEN") or serving_engine.get("max_model_len")
    if max_model_len:
        engine_kwargs["max_model_len"] = int(max_model_len)

    max_num_seqs = serving_engine.get("max_num_seqs")
    if max_num_seqs:
        engine_kwargs["max_num_seqs"] = int(max_num_seqs)

    if tp > 1:
        engine_kwargs["tensor_parallel_size"] = tp
    if pp > 1:
        engine_kwargs["pipeline_parallel_size"] = pp
    if serving_engine.get("disable_custom_all_reduce") is True:
        engine_kwargs["disable_custom_all_reduce"] = True
    if attn_backend:
        engine_kwargs["attention_backend"] = attn_backend

    distributed_backend = serving_engine.get("distributed_executor_backend")
    if distributed_backend:
        engine_kwargs["distributed_executor_backend"] = distributed_backend

    llm = LLM(**engine_kwargs)

    # Prepare requests
    prompts = []
    sampling_params_list = []
    request_ids = []
    for req in manifest["requests"]:
        prompts.append(req["prompt"])
        request_ids.append(req["id"])
        sampling_params_list.append(
            SamplingParams(
                temperature=req["temperature"],
                max_tokens=req["max_new_tokens"],
                logprobs=20,
                seed=seed,
            )
        )

    # Execute inference
    t0 = time.monotonic()
    outputs = llm.generate(prompts, sampling_params_list)
    inference_time = time.monotonic() - t0

    # Extract observables
    request_outputs = []

    for idx, (req_id, output) in enumerate(zip(request_ids, outputs)):
        result = output.outputs[0]
        tokens = list(result.token_ids)

        # Extract logprobs as flat list of floats (log probability of the chosen token)
        logits: list[float] = []
        if result.logprobs:
            for step_logprobs in result.logprobs:
                chosen_token = tokens[len(logits)] if len(logits) < len(tokens) else 0
                if chosen_token in step_logprobs:
                    logits.append(round(float(step_logprobs[chosen_token].logprob), 8))
                else:
                    logits.append(0.0)

        request_outputs.append({
            "id": req_id,
            "tokens": tokens,
            "logits": logits,
            "text": result.text,
            "finish_reason": result.finish_reason,
        })

    # Collect environment info from the running vLLM instance
    gpu_inventory = []
    driver_version = manifest["hardware_profile"]["gpu"]["driver_version"]
    try:
        import subprocess
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        for line in smi.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if parts:
                gpu_inventory.append(parts[0])
                if len(parts) > 1:
                    driver_version = parts[1]
    except Exception:
        gpu_inventory = [manifest["hardware_profile"]["gpu"]["model"]]

    env_info = {
        "vllm_version": _get_vllm_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "unknown",
        "driver_version": driver_version,
        "gpu_inventory": gpu_inventory,
        "inference_time_s": round(inference_time, 3),
    }

    return {
        "request_outputs": request_outputs,
        "env_info": env_info,
        "resolved_env": resolved_env,
    }


def _get_vllm_version() -> str:
    try:
        import vllm
        return getattr(vllm, "__version__", "unknown")
    except Exception:
        return "unknown"
