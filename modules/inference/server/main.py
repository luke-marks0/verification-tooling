#!/usr/bin/env python3
"""Deterministic vLLM serving wrapper.

Validates manifest + lockfile + hardware conformance at boot,
then starts vLLM's OpenAI-compatible server with batch invariance.
All requests/responses are logged to an append-only capture file.

POST /manifest accepts a new manifest, validates it, and (re)starts
vLLM to serve that configuration.
GET /manifest returns the active manifest and server state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError

from pydantic import ValidationError as PydanticValidationError

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.core.common.contracts import ValidationError, validate_with_schema
from modules.core.common.deterministic import (
    canonical_json_bytes,
    canonical_json_text,
    compute_lockfile_digest,
    sha256_file,
    sha256_prefixed,
    utc_now_iso,
)
from modules.attestation.e2e import (
    commit_token,
    commit_token_stream,
    extract_input_token_ids,
    extract_output_token_ids,
)
from modules.attestation.e2e.extract import TokenIdExtractionError
from modules.inference.manifest.model import GpuProbe, HardwareConformance, Manifest
from modules.network.networkdet import DeterministicNetStack
from modules.network.networkdet.config import NetStackConfig
from modules.network.networkdet.backend_sim import SimulatedBackend
from modules.network.networkdet.warden import ActiveWarden
from modules.network.networkdet.capture import CaptureRing


def _hardware_fingerprint(hw: dict[str, Any]) -> str:
    return sha256_prefixed(canonical_json_bytes(hw))


def _validate_boot(manifest: Manifest, lockfile: dict[str, Any]) -> dict[str, Any]:
    """Validate manifest and lockfile at server boot. Returns conformance record."""
    validate_with_schema("manifest.v1.schema.json", manifest.model_dump(exclude_none=True))
    validate_with_schema("lockfile.v1.schema.json", lockfile)

    manifest_digest = sha256_prefixed(canonical_json_bytes(manifest.model_dump(exclude_none=True)))
    if lockfile["manifest_digest"] != manifest_digest:
        raise ValidationError("Lockfile manifest_digest mismatch")

    expected_lockfile_digest = compute_lockfile_digest(lockfile)
    if lockfile["canonicalization"]["lockfile_digest"] != expected_lockfile_digest:
        raise ValidationError("Lockfile canonicalization.lockfile_digest mismatch")

    return {"manifest_digest": manifest_digest, "lockfile_digest": expected_lockfile_digest}


def _probe_gpu() -> GpuProbe:
    """Query the GPU environment. Pure observation, no manifest comparison."""
    try:
        import torch
    except ImportError:
        return GpuProbe()

    if not torch.cuda.is_available():
        return GpuProbe()

    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)

    driver = ""
    try:
        import ctypes
        nvml = ctypes.CDLL("libnvidia-ml.so.1")
        nvml.nvmlInit()
        buf = ctypes.create_string_buffer(80)
        nvml.nvmlSystemGetDriverVersion(buf, 80)
        driver = buf.value.decode()
        nvml.nvmlShutdown()
    except Exception:
        pass

    vllm_ver = ""
    try:
        import vllm
        vllm_ver = getattr(vllm, "__version__", "unknown")
    except ImportError:
        pass

    return GpuProbe(
        available=True,
        name=name,
        count=torch.cuda.device_count(),
        compute_capability=f"{cc[0]}.{cc[1]}",
        driver_version=driver,
        cuda_version=torch.version.cuda or "unknown",
        torch_version=torch.__version__,
        vllm_version=vllm_ver,
    )


def _check_hardware(manifest: Manifest, probe: GpuProbe) -> HardwareConformance:
    """Compare manifest hardware_profile against a GPU probe. Pure function."""
    if not probe.available:
        return HardwareConformance(status="no_gpu", probe=probe)

    warnings: list[str] = []
    gpu = manifest.hardware_profile.gpu

    if gpu.count != probe.count:
        warnings.append(f"GPU count: manifest={gpu.count}, actual={probe.count}")

    if gpu.model.lower() not in probe.name.lower() and probe.name.lower() not in gpu.model.lower():
        warnings.append(f"GPU model: manifest='{gpu.model}', actual='{probe.name}'")

    if gpu.driver_version and probe.driver_version and gpu.driver_version != probe.driver_version:
        warnings.append(f"GPU driver: manifest={gpu.driver_version}, actual={probe.driver_version}")
    elif gpu.driver_version and not probe.driver_version:
        warnings.append("Could not query GPU driver version")

    if gpu.cuda_driver_version and probe.cuda_version and gpu.cuda_driver_version != probe.cuda_version:
        warnings.append(f"CUDA version: manifest={gpu.cuda_driver_version}, actual={probe.cuda_version}")

    status = "mismatch" if warnings else "conformant"
    return HardwareConformance(status=status, probe=probe, warnings=warnings)


def _enforce_model_revision(manifest: Manifest) -> str | None:
    """Return the --revision flag value if the manifest pins a specific commit."""
    return manifest.model.weights_revision


def _validate_requests(manifest: Manifest) -> list[str]:
    """Validate that all requests are servable with the declared engine config."""
    errors = []
    max_len = manifest.runtime.serving_engine.max_model_len

    for req in manifest.requests:
        if req.max_new_tokens > max_len:
            errors.append(
                f"Request '{req.id}': max_new_tokens={req.max_new_tokens} "
                f"exceeds max_model_len={max_len}"
            )
    return errors


def _build_vllm_cmd(manifest: Manifest, host: str, port: int) -> list[str]:
    """Build the vllm serve command from ALL manifest settings."""
    runtime = manifest.runtime
    knobs = runtime.deterministic_knobs
    model = manifest.model
    model_source = model.source
    model_id = model_source.removeprefix("hf://") if model_source.startswith("hf://") else model_source

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", os.getenv("RUNNER_MODEL_PATH", model_id),
        "--host", host,
        "--port", str(port),
        "--seed", str(knobs.seed),
        "--disable-log-stats",
    ]

    # Model revision pinning -- use the exact commit from the manifest
    revision = _enforce_model_revision(manifest)
    if revision:
        cmd.extend(["--revision", revision])
        cmd.extend(["--tokenizer-revision", model.tokenizer_revision])

    # Serving engine -- every field applied
    engine = runtime.serving_engine

    cmd.extend(["--dtype", engine.dtype.value])
    cmd.extend(["--max-model-len", str(engine.max_model_len)])
    cmd.extend(["--gpu-memory-utilization", str(engine.gpu_memory_utilization)])

    cmd.extend(["--max-num-seqs", str(engine.max_num_seqs)])

    cmd.extend(["--attention-backend", engine.attention_backend.value])

    # Batch invariance
    if runtime.batch_invariance.enforce_eager:
        cmd.append("--enforce-eager")

    # Optional serving_engine flags
    if engine.quantization:
        cmd.extend(["--quantization", engine.quantization.value])

    if engine.load_format:
        cmd.extend(["--load-format", engine.load_format.value])

    if engine.kv_cache_dtype:
        cmd.extend(["--kv-cache-dtype", engine.kv_cache_dtype.value])

    if engine.max_num_batched_tokens:
        cmd.extend(["--max-num-batched-tokens", str(engine.max_num_batched_tokens)])

    if engine.block_size:
        cmd.extend(["--block-size", str(engine.block_size)])

    if engine.enable_prefix_caching is True:
        cmd.append("--enable-prefix-caching")
    elif engine.enable_prefix_caching is False:
        cmd.append("--no-enable-prefix-caching")

    if engine.enable_chunked_prefill is True:
        cmd.append("--enable-chunked-prefill")
    elif engine.enable_chunked_prefill is False:
        cmd.append("--no-enable-chunked-prefill")

    if engine.scheduling_policy:
        cmd.extend(["--scheduling-policy", engine.scheduling_policy.value])

    if engine.disable_sliding_window is True:
        cmd.append("--disable-sliding-window")
    elif engine.disable_sliding_window is False:
        cmd.append("--no-disable-sliding-window")

    tp = engine.tensor_parallel_size
    if tp and tp > 1:
        cmd.extend(["--tensor-parallel-size", str(tp)])

    pp = engine.pipeline_parallel_size
    if pp and pp > 1:
        cmd.extend(["--pipeline-parallel-size", str(pp)])

    if engine.disable_custom_all_reduce is True:
        cmd.append("--disable-custom-all-reduce")
    elif engine.disable_custom_all_reduce is False:
        cmd.append("--no-disable-custom-all-reduce")

    if engine.distributed_executor_backend:
        cmd.extend(["--distributed-executor-backend", engine.distributed_executor_backend])

    # Trust remote code
    if model.trust_remote_code:
        cmd.append("--trust-remote-code")

    api_key = os.getenv("VLLM_API_KEY")
    if api_key:
        cmd.extend(["--api-key", api_key])

    return cmd


# ---------------------------------------------------------------------------
# ServerState
# ---------------------------------------------------------------------------

class ServerState:
    """Holds the active manifest, vLLM process, and capture log."""

    def __init__(
        self,
        manifest: Manifest,
        vllm_proc: subprocess.Popen | None,
        vllm_port: int,
        capture_log: CaptureLog,
        out_dir: Path,
    ) -> None:
        self.manifest = manifest
        self.vllm_proc = vllm_proc
        self.vllm_port = vllm_port
        self.capture_log = capture_log
        self.out_dir = out_dir
        self.lock = threading.Lock()
        self.applied_at = utc_now_iso()

    @property
    def manifest_digest(self) -> str:
        return sha256_prefixed(canonical_json_bytes(self.manifest.model_dump(exclude_none=True)))


def _set_deterministic_env(manifest: Manifest) -> None:
    """Set deterministic environment variables from a manifest."""
    knobs = manifest.runtime.deterministic_knobs
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = knobs.cublas_workspace_config or ":4096:8"
    os.environ["CUDA_LAUNCH_BLOCKING"] = str(int(knobs.cuda_launch_blocking))
    os.environ["PYTHONHASHSEED"] = knobs.pythonhashseed or "0"

    if knobs.torch_deterministic:
        os.environ["TORCH_CUDNN_DETERMINISTIC"] = "1"
        os.environ["TORCH_CUDNN_BENCHMARK"] = "0"
    else:
        os.environ.pop("TORCH_CUDNN_DETERMINISTIC", None)
        os.environ.pop("TORCH_CUDNN_BENCHMARK", None)

    if manifest.runtime.batch_invariance.enabled:
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
    else:
        os.environ.pop("VLLM_BATCH_INVARIANT", None)


def _verify_closure(manifest: Manifest, report: dict[str, Any]) -> None:
    """Check nix closure hash if declared in the manifest."""
    expected = manifest.runtime.closure_hash
    if expected:
        actual = os.environ.get("CLOSURE_HASH", "")
        if actual and actual != expected:
            raise ValidationError(
                f"Closure hash mismatch: expected {expected}, got {actual}"
            )
        if actual:
            report["enforced"].append(f"closure hash verified: {actual[:24]}...")
        else:
            report["warnings"].append("CLOSURE_HASH env var not set, cannot verify closure")


def _verify_model_artifacts(manifest: Manifest, report: dict[str, Any]) -> None:
    """Verify model file digests from artifact_inputs against local cache."""
    revision = manifest.model.weights_revision
    if not revision:
        report["warnings"].append("No weights_revision -- skipping file verification")
        return

    # Check RUNNER_MODEL_PATH first, then HF cache
    model_path = os.environ.get("RUNNER_MODEL_PATH")
    if model_path and Path(model_path).is_dir():
        cache_path = Path(model_path)
    else:
        repo_id = manifest.model.source.removeprefix("hf://")
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        cache_path = cache_dir / f"models--{repo_id.replace('/', '--')}" / "snapshots" / revision
        if not cache_path.is_dir():
            report["warnings"].append(f"HF cache not found for {repo_id}@{revision[:12]}")
            return

    model_types = {"model_weights", "model_config", "tokenizer", "generation_config", "chat_template"}
    for artifact in manifest.artifact_inputs:
        if artifact.artifact_type.value not in model_types:
            continue
        expected = artifact.expected_digest
        art_path = artifact.path
        if not expected or not art_path:
            continue

        file_path = cache_path / art_path
        if not file_path.is_file():
            report["warnings"].append(f"File not found in cache: {art_path}")
            continue

        # Quick size check
        expected_size = artifact.size_bytes
        actual_size = file_path.stat().st_size
        if expected_size and actual_size != expected_size:
            raise ValidationError(
                f"File size mismatch for {art_path}: expected {expected_size}, got {actual_size} "
                f"(possible incomplete download)"
            )

        # Full hash check
        actual = sha256_file(file_path)
        if actual != expected:
            raise ValidationError(
                f"File digest mismatch for {art_path}: expected {expected[:24]}..., got {actual[:24]}..."
            )
        report["enforced"].append(f"verified {art_path}")


def _start_vllm(state: ServerState, manifest: Manifest) -> dict[str, Any]:
    """Enforce manifest, stop old vLLM, start fresh, wait for health.

    Returns a report of what was enforced/validated.
    Must be called with state.lock held.
    """
    report: dict[str, Any] = {"enforced": [], "warnings": []}

    # 0. Container image digest verification
    _verify_closure(manifest, report)

    # 1. Validate requests are servable
    req_errors = _validate_requests(manifest)
    if req_errors:
        raise ValidationError("Requests incompatible with engine config: " + "; ".join(req_errors))
    report["enforced"].append(f"validated {len(manifest.requests)} requests against engine config")

    # 2. Enforce hardware profile
    probe = _probe_gpu()
    conformance = _check_hardware(manifest, probe)
    if manifest.runtime.strict_hardware and conformance.warnings:
        raise ValidationError("Hardware mismatch: " + "; ".join(conformance.warnings))
    report["warnings"].extend(conformance.warnings)
    for w in conformance.warnings:
        print(f"[manifest] WARNING: {w}")
    report["enforced"].append(f"hardware: {conformance.status} (gpu={probe.name or 'unavailable'})")

    # 3. Set deterministic env from manifest knobs
    _set_deterministic_env(manifest)
    knobs = manifest.runtime.deterministic_knobs
    report["enforced"].append(
        f"deterministic env: seed={knobs.seed}, "
        f"torch_deterministic={knobs.torch_deterministic}, "
        f"cuda_launch_blocking={knobs.cuda_launch_blocking}"
    )

    # 4. Batch invariance
    batch_inv = manifest.runtime.batch_invariance
    if batch_inv.enabled:
        report["enforced"].append("batch invariance: ENABLED, enforce_eager=" + str(batch_inv.enforce_eager))
    else:
        report["enforced"].append("batch invariance: disabled")

    # 5. Model revision pinning
    revision = _enforce_model_revision(manifest)
    if revision:
        report["enforced"].append(f"model revision pinned: {revision[:12]}...")
    else:
        report["warnings"].append("model revision not pinned -- may load latest")

    # 5b. Verify model file digests from artifact_inputs
    _verify_model_artifacts(manifest, report)

    # 6. Terminate old process
    if state.vllm_proc is not None and state.vllm_proc.poll() is None:
        print("[vllm] Stopping current instance...")
        state.vllm_proc.terminate()
        try:
            state.vllm_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            state.vllm_proc.kill()
            state.vllm_proc.wait(timeout=5)

    # 7. Build command from manifest and launch
    vllm_cmd = _build_vllm_cmd(manifest, "127.0.0.1", state.vllm_port)
    print(f"[vllm] Starting: {' '.join(vllm_cmd)}")
    state.vllm_proc = subprocess.Popen(
        vllm_cmd, stdout=sys.stdout, stderr=sys.stderr,
    )

    print("[vllm] Waiting for health...")
    if not _wait_for_vllm(state.vllm_port):
        raise RuntimeError("vLLM failed to become healthy")

    # 8. Update state
    state.manifest = manifest
    state.applied_at = utc_now_iso()
    state.capture_log = CaptureLog(state.out_dir / "capture.jsonl")

    # 9. Log what was enforced
    engine = manifest.runtime.serving_engine
    report["enforced"].append(
        f"vLLM started: model={manifest.model.source}, "
        f"max_model_len={engine.max_model_len}, "
        f"dtype={engine.dtype.value}, "
        f"attention_backend={engine.attention_backend.value}, "
        f"max_num_seqs={engine.max_num_seqs}"
    )
    comparison_keys = list(manifest.comparison.model_dump(exclude_none=True).keys())
    report["enforced"].append(f"comparison config stored: {comparison_keys}")
    report["enforced"].append(f"artifact_inputs: {len(manifest.artifact_inputs)} artifacts declared")

    print(f"[vllm] Ready -- {len(report['enforced'])} checks enforced, {len(report['warnings'])} warnings")
    return report


# ---------------------------------------------------------------------------
# CaptureLog
# ---------------------------------------------------------------------------

class CaptureLog:
    """Append-only request/response log for provenance with egress hashing."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seq = 0
        self._egress_hasher = hashlib.sha256()
        self._egress_count = 0
        with open(self.path, "w") as f:
            f.write("")

    def next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def append(self, entry: dict[str, Any]) -> None:
        with self._lock:
            entry["captured_at"] = utc_now_iso()
            with open(self.path, "a") as f:
                f.write(canonical_json_text(entry))

    def record_egress(self, payload_digest: str) -> None:
        with self._lock:
            self._egress_hasher.update(bytes.fromhex(payload_digest))
            self._egress_count += 1

    @property
    def egress_digest(self) -> str:
        with self._lock:
            return f"sha256:{self._egress_hasher.hexdigest()}"

    @property
    def egress_count(self) -> int:
        with self._lock:
            return self._egress_count


# ---------------------------------------------------------------------------
# ProxyHandler
# ---------------------------------------------------------------------------

class ProxyHandler(BaseHTTPRequestHandler):
    """Reverse proxy with /manifest endpoint and capture logging."""

    server_state: ServerState | None = None
    api_key: str | None = None

    def _check_auth(self) -> bool:
        if not self.api_key:
            return True
        auth = self.headers.get("Authorization", "")
        if auth == f"Bearer {self.api_key}":
            return True
        self._send_json(401, {"error": "Unauthorized"})
        return False

    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    @property
    def _vllm_port(self) -> int:
        return self.server_state.vllm_port if self.server_state else 8001

    @property
    def _capture_log(self) -> CaptureLog | None:
        return self.server_state.capture_log if self.server_state else None

    # -- POST --

    def do_POST(self):
        if not self._check_auth():
            return

        if self.path == "/manifest":
            return self._handle_post_manifest()

        if self.path == "/run":
            return self._handle_post_run()

        if self.path == "/replay":
            return self._handle_post_replay()

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""

        capture_log = self._capture_log
        arrival_seq = capture_log.next_seq() if capture_log else 0

        try:
            request_data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            request_data = {"raw": body.decode("utf-8", errors="replace")}

        url = f"http://127.0.0.1:{self._vllm_port}{self.path}"
        req = Request(url, data=body, method="POST")
        for key in ["Content-Type", "Authorization"]:
            val = self.headers.get(key)
            if val:
                req.add_header(key, val)

        try:
            with urlopen(req) as resp:
                resp_body = resp.read()
                status = resp.status

                try:
                    response_data = json.loads(resp_body)
                except json.JSONDecodeError:
                    response_data = {"raw": resp_body.decode("utf-8", errors="replace")}

                if capture_log and self.path.startswith("/v1/"):
                    capture_log.append({
                        "seq": arrival_seq,
                        "endpoint": self.path,
                        "request": request_data,
                        "response": response_data,
                        "status": status,
                    })

                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)

        except URLError as exc:
            self._send_json(502, {"error": str(exc)})

    def _handle_post_manifest(self) -> None:
        """POST /manifest -- validate and (re)start vLLM for this manifest."""
        state = self.server_state
        if state is None:
            return self._send_json(500, {"error": "Server state not initialized"})

        # Parse
        content_length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(content_length) if content_length > 0 else b""
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            return self._send_json(400, {"error": f"Invalid JSON: {exc}"})

        if not body:
            return self._send_json(400, {"error": "Empty request body"})

        # Validate with Pydantic
        try:
            manifest = Manifest.model_validate(body)
        except PydanticValidationError as exc:
            return self._send_json(422, {"error": str(exc)})

        # Acquire lock (409 if another manifest is being applied)
        if not state.lock.acquire(blocking=False):
            return self._send_json(409, {"error": "Server is busy applying another manifest"})

        try:
            report = _start_vllm(state, manifest)
        except (ValidationError, RuntimeError) as exc:
            return self._send_json(500, {"error": str(exc)})
        except Exception as exc:
            return self._send_json(500, {"error": f"Failed to start vLLM: {exc}"})
        finally:
            state.lock.release()

        return self._send_json(200, {
            "status": "ok",
            "manifest_digest": state.manifest_digest,
            "model": state.manifest.model.source,
            "run_id": state.manifest.run_id,
            "applied_at": state.applied_at,
            "enforced": report["enforced"],
            "warnings": report["warnings"],
            "requests": len(state.manifest.requests),
            "comparison": list(state.manifest.comparison.model_dump(exclude_none=True).keys()),
        })

    def _call_vllm_chat(
        self,
        *,
        model_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        seed: int,
        want_token_ids: bool,
    ) -> dict[str, Any]:
        """Send one chat completion to the managed vLLM. Returns parsed JSON.

        When `want_token_ids` is True, asks vLLM to include output token IDs
        on the response so the caller can compute per-token commitments.
        """
        state = self.server_state
        body: dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
        }
        if want_token_ids:
            body["return_token_ids"] = True
        vllm_req = Request(
            f"http://127.0.0.1:{state.vllm_port}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(vllm_req, timeout=300) as resp:
            return json.loads(resp.read())

    def _handle_post_run(self) -> None:
        """POST /run -- execute the active manifest's requests, return a run bundle.

        Sends each request in manifest.requests to vLLM, feeds the responses
        through the deterministic net stack and warden, and returns:
        - inference results (tokens, content hash per request)
        - packet output (warden-normalized frames, digest)
        - manifest and closure metadata

        When `manifest.audit.token_commitment.enabled` is true, the server
        also asks vLLM to return both prompt and output token IDs, computes
        an HMAC-SHA256 commitment per token on both sides, and includes a
        `token_commitments` map in the bundle shaped as
        `{request_id: {"input": [...], "output": [...]}}`. Committing both
        sides keeps the audit surface free of plaintext prompts or
        completions — an auditor sees only commitments and can challenge
        any position on either side via POST /replay.
        """
        state = self.server_state
        if state is None:
            return self._send_json(500, {"error": "Server state not initialized"})

        manifest = state.manifest
        model_id = manifest.model.source.removeprefix("hf://")
        seed = manifest.runtime.deterministic_knobs.seed
        audit_enabled = bool(
            manifest.audit and manifest.audit.token_commitment.enabled
        )

        # Set up packet pipeline
        config = NetStackConfig(
            mtu=1500, mss=1460, tso=False, gso=False,
            checksum_offload=False, thread_affinity=(0,),
            tx_queues=1, rx_queues=1, queue_mapping_policy="fixed_core_queue",
            ring_tx=512, ring_rx=512,
            internal_batching_enabled=False, internal_batching_max_burst=1,
            security_mode="plaintext", egress_reproducibility=True,
            src_ip="10.0.0.1", dst_ip="10.0.0.2",
            src_mac="02:00:00:00:00:01", dst_mac="02:00:00:00:00:02",
            src_port=8000, dst_port=80,
        )
        net = DeterministicNetStack(config, run_id=manifest.run_id, backend=SimulatedBackend())
        warden = ActiveWarden(secret=b"deterministic-warden-key")
        warden_capture = CaptureRing()

        inference_results = []
        errors = []
        token_commitments: dict[str, dict[str, list[str]]] = {}

        for i, req in enumerate(manifest.requests):
            try:
                resp_data = self._call_vllm_chat(
                    model_id=model_id,
                    prompt=req.prompt,
                    max_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    seed=seed,
                    want_token_ids=audit_enabled,
                )
            except Exception as exc:
                errors.append({"request_id": req.id, "error": str(exc)})
                continue

            content = resp_data["choices"][0]["message"]["content"]
            tokens = resp_data["usage"]["completion_tokens"]
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            inference_results.append({
                "id": req.id,
                "tokens": tokens,
                "content_hash": content_hash,
            })

            if audit_enabled:
                try:
                    input_token_ids = extract_input_token_ids(resp_data)
                    output_token_ids = extract_output_token_ids(resp_data)
                except TokenIdExtractionError as exc:
                    errors.append({"request_id": req.id, "error": str(exc)})
                    continue
                token_commitments[req.id] = {
                    "input": commit_token_stream(input_token_ids),
                    "output": commit_token_stream(output_token_ids),
                }

            # Feed through packet pipeline — strip nondeterministic API fields
            import copy
            det_resp = copy.deepcopy(resp_data)
            for key in ("id", "created", "system_fingerprint"):
                det_resp.pop(key, None)
            response_bytes = canonical_json_bytes(det_resp)
            frames = net.process_response(conn_index=i, response_bytes=response_bytes)

            for frame in frames:
                normalized = warden.normalize(frame)
                if normalized is not None:
                    warden_capture.record(normalized)

        # Build the bundle
        inference_digest = hashlib.sha256(
            "".join(r["content_hash"] for r in inference_results).encode()
        ).hexdigest()

        bundle = {
            "manifest_digest": state.manifest_digest,
            "closure_hash": os.environ.get("CLOSURE_HASH", ""),
            "run_id": manifest.run_id,
            "inference": {
                "digest": f"sha256:{inference_digest}",
                "results": inference_results,
            },
            "packets": {
                "digest": warden_capture.digest(),
                "frame_count": warden_capture.frame_count,
                "frames": warden_capture.frames_as_hex(),
            },
        }

        if audit_enabled:
            bundle["token_commitments"] = token_commitments
            bundle["audit"] = {
                "algorithm": manifest.audit.token_commitment.algorithm,
                "key_source": manifest.audit.token_commitment.key_source,
            }

        if errors:
            bundle["errors"] = errors

        return self._send_json(200, bundle)

    def _handle_post_replay(self) -> None:
        """POST /replay -- return the HMAC commitment for a challenged token.

        Body: {
            "request_id": "<id>",
            "token_position": <1-indexed int>,
            "side": "input" | "output"   # optional, defaults to "output"
        }

        Looks up `request_id` in the active manifest and returns the HMAC
        commitment for the token at position `token_position` on the
        requested `side`. The caller compares this against the matching
        stream from /run's `token_commitments[request_id][side]`.

        For `side = "output"` the server sends vLLM a single-shot chat
        completion with `max_tokens = token_position` and the same seed
        used by /run, then commits the final generated token.

        For `side = "input"` the server asks vLLM for the prompt
        tokenization only (max_tokens = 1, prompt_token_ids returned)
        and commits the token at `token_position` of the prompt. No
        generation work is needed to verify a prompt token, but routing
        the request through vLLM guarantees the same tokenizer as /run.

        Returns 400 on bad input, 404 for unknown request_id, 409 if the
        manifest has not enabled token commitments, 502 if vLLM is down.
        This endpoint is stateless with respect to prior /run calls.
        """
        state = self.server_state
        if state is None:
            return self._send_json(500, {"error": "Server state not initialized"})
        manifest = state.manifest

        if manifest.audit is None or not manifest.audit.token_commitment.enabled:
            return self._send_json(409, {
                "error": "Audit/token_commitment is not enabled on the active manifest"
            })

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length <= 0:
            return self._send_json(400, {"error": "Empty request body"})
        try:
            body = json.loads(self.rfile.read(content_length))
        except json.JSONDecodeError as exc:
            return self._send_json(400, {"error": f"Invalid JSON: {exc}"})

        request_id = body.get("request_id")
        token_position = body.get("token_position")
        side = body.get("side", "output")
        if not isinstance(request_id, str) or not request_id:
            return self._send_json(400, {"error": "`request_id` must be a non-empty string"})
        if not isinstance(token_position, int) or isinstance(token_position, bool):
            return self._send_json(400, {"error": "`token_position` must be an integer"})
        if side not in ("input", "output"):
            return self._send_json(400, {
                "error": "`side` must be 'input' or 'output'"
            })

        original = next((r for r in manifest.requests if r.id == request_id), None)
        if original is None:
            known = [r.id for r in manifest.requests]
            return self._send_json(404, {
                "error": f"Unknown request_id {request_id!r}",
                "known": known,
            })

        # Output-side positions are bounded by max_new_tokens; input-side
        # positions are bounded by the prompt's token length, which we
        # only learn from vLLM. We validate the upper bound after the call.
        if token_position < 1:
            return self._send_json(400, {
                "error": f"`token_position` must be >= 1, got {token_position}"
            })
        if side == "output" and token_position > original.max_new_tokens:
            return self._send_json(400, {
                "error": (
                    f"`token_position` must be 1..{original.max_new_tokens} "
                    f"for request {request_id!r} (side=output), got {token_position}"
                )
            })

        model_id = manifest.model.source.removeprefix("hf://")
        seed = manifest.runtime.deterministic_knobs.seed
        # For input-side challenges only the tokenizer output matters, so
        # we request the minimum (1 generated token). For output-side we
        # need to generate up to `token_position` to reveal that token.
        replay_max_tokens = token_position if side == "output" else 1
        try:
            resp_data = self._call_vllm_chat(
                model_id=model_id,
                prompt=original.prompt,
                max_tokens=replay_max_tokens,
                temperature=original.temperature,
                seed=seed,
                want_token_ids=True,
            )
        except URLError as exc:
            return self._send_json(502, {"error": f"vLLM unreachable: {exc}"})
        except Exception as exc:
            return self._send_json(500, {"error": f"vLLM call failed: {exc}"})

        try:
            if side == "output":
                token_ids = extract_output_token_ids(resp_data)
            else:
                token_ids = extract_input_token_ids(resp_data)
        except TokenIdExtractionError as exc:
            return self._send_json(500, {"error": str(exc)})
        if len(token_ids) < token_position:
            return self._send_json(400, {
                "error": (
                    f"`token_position`={token_position} exceeds the {side} "
                    f"token length ({len(token_ids)}) for request {request_id!r}"
                ),
            })

        challenged_token = token_ids[token_position - 1]
        commitment = commit_token(challenged_token)
        return self._send_json(200, {
            "request_id": request_id,
            "token_position": token_position,
            "side": side,
            "commitment": commitment,
            "algorithm": manifest.audit.token_commitment.algorithm,
        })

    # -- GET --

    def do_GET(self):
        if self.path != "/health" and not self._check_auth():
            return

        if self.path == "/manifest":
            return self._handle_get_manifest()

        if self.path == "/flake":
            return self._handle_get_flake()

        url = f"http://127.0.0.1:{self._vllm_port}{self.path}"
        try:
            with urlopen(url) as resp:
                resp_body = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)
        except URLError as exc:
            self._send_json(502, {"error": str(exc)})

    def _handle_get_manifest(self) -> None:
        """GET /manifest -- return the active manifest and server state."""
        state = self.server_state
        if state is None:
            return self._send_json(500, {"error": "Server state not initialized"})

        vllm_healthy = False
        try:
            with urlopen(f"http://127.0.0.1:{state.vllm_port}/health") as resp:
                vllm_healthy = resp.status == 200
        except Exception:
            pass

        m = state.manifest
        comparison_dump = m.comparison.model_dump(exclude_none=True)
        return self._send_json(200, {
            "manifest": m.model_dump(exclude_none=True),
            "manifest_digest": state.manifest_digest,
            "applied_at": state.applied_at,
            "vllm_healthy": vllm_healthy,
            "active_config": {
                "model": m.model.source,
                "revision": m.model.weights_revision or "unpinned",
                "run_id": m.run_id,
                "seed": m.runtime.deterministic_knobs.seed,
                "batch_invariance": m.runtime.batch_invariance.enabled,
                "max_model_len": m.runtime.serving_engine.max_model_len,
                "attention_backend": m.runtime.serving_engine.attention_backend.value,
                "dtype": m.runtime.serving_engine.dtype.value,
                "strict_hardware": m.runtime.strict_hardware,
                "requests": len(m.requests),
                "artifact_inputs": len(m.artifact_inputs),
                "comparison_modes": {k: v["mode"] for k, v in comparison_dump.items()},
            },
        })

    def _handle_get_flake(self) -> None:
        """GET /flake -- return the flake.nix that built this container."""
        flake_path = os.environ.get("FLAKE_NIX_PATH")
        if not flake_path or not Path(flake_path).is_file():
            return self._send_json(404, {
                "error": "flake.nix not available (FLAKE_NIX_PATH not set or file missing)"
            })

        result = {"closure_hash": os.environ.get("CLOSURE_HASH", "unknown")}

        result["flake.nix"] = Path(flake_path).read_text(encoding="utf-8")

        lock_path = os.environ.get("FLAKE_LOCK_PATH")
        if lock_path and Path(lock_path).is_file():
            result["flake.lock"] = json.loads(Path(lock_path).read_text(encoding="utf-8"))

        return self._send_json(200, result)

    def log_message(self, format, *args):
        pass


def _wait_for_vllm(port: int, timeout: int = 300) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urlopen(f"http://127.0.0.1:{port}/health") as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic vLLM serving wrapper")
    parser.add_argument("--manifest", required=True, help="Manifest JSON path")
    parser.add_argument("--lockfile", help="Lockfile JSON path (omit to skip boot validation)")
    parser.add_argument("--skip-boot-validation", action="store_true", help="Skip lockfile/hardware checks at boot")
    parser.add_argument("--out-dir", default="/tmp/deterministic-server", help="Output directory")
    parser.add_argument("--host", default="0.0.0.0", help="Listen host")
    parser.add_argument("--port", type=int, default=8000, help="Listen port (proxy)")
    parser.add_argument("--vllm-port", type=int, default=8001, help="Internal vLLM port")
    args = parser.parse_args()

    manifest_data = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    manifest = Manifest.model_validate(manifest_data)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Deterministic Server Boot ===")
    print(f"Model: {manifest.model.source}")
    print(f"Run ID: {manifest.run_id}")
    print()

    if args.skip_boot_validation or not args.lockfile:
        print("[boot] Skipping lockfile/hardware validation")
    else:
        lockfile = json.loads(Path(args.lockfile).read_text(encoding="utf-8"))
        print("[boot] Validating manifest and lockfile...")
        digests = _validate_boot(manifest, lockfile)
        print(f"  manifest_digest: {digests['manifest_digest']}")
        print(f"  lockfile_digest: {digests['lockfile_digest']}")

        print("[boot] Probing hardware...")
        probe = _probe_gpu()
        conformance = _check_hardware(manifest, probe)
        print(f"  GPU: {probe.name}")
        print(f"  Compute capability: {probe.compute_capability}")
        print(f"  Status: {conformance.status}")
        for w in conformance.warnings:
            print(f"  WARNING: {w}")

        if conformance.status != "conformant":
            if manifest.runtime.strict_hardware:
                print(f"ERROR: Hardware conformance failed: {conformance.status}")
                return 1
            print(f"WARNING: Hardware non-conformant ({conformance.status}), continuing")

    # Start vLLM
    _set_deterministic_env(manifest)
    vllm_cmd = _build_vllm_cmd(manifest, "127.0.0.1", args.vllm_port)
    print(f"[boot] Starting vLLM: {' '.join(vllm_cmd)}")

    vllm_proc = subprocess.Popen(vllm_cmd, stdout=sys.stdout, stderr=sys.stderr)

    print("[boot] Waiting for vLLM to be ready...")
    if not _wait_for_vllm(args.vllm_port):
        print("ERROR: vLLM server failed to start within timeout")
        vllm_proc.terminate()
        return 1

    print("[boot] vLLM is ready\n")

    # Create state and start proxy
    capture_log = CaptureLog(out_dir / "capture.jsonl")
    state = ServerState(
        manifest=manifest,
        vllm_proc=vllm_proc,
        vllm_port=args.vllm_port,
        capture_log=capture_log,
        out_dir=out_dir,
    )

    ProxyHandler.server_state = state
    ProxyHandler.api_key = os.getenv("VLLM_API_KEY")

    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    proxy = ThreadedHTTPServer((args.host, args.port), ProxyHandler)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()

    batch_inv = manifest.runtime.batch_invariance
    print("=== Server ready ===")
    print("  POST /manifest to load a new manifest")
    print("  GET  /manifest to inspect active state")
    print(f"  Model: {manifest.model.source}")
    print(f"  Batch invariance: {'ON' if batch_inv.enabled else 'OFF'}")
    print()

    def shutdown(signum, frame):
        print("\n[shutdown] Stopping server...")
        proxy.shutdown()
        if state.vllm_proc and state.vllm_proc.poll() is None:
            state.vllm_proc.terminate()
            try:
                state.vllm_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                state.vllm_proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Wait for vLLM -- loop to handle restarts from POST /manifest
    try:
        while True:
            state.vllm_proc.wait()
            with state.lock:
                if state.vllm_proc.poll() is None:
                    continue  # restart happened, new process running
                break
    except KeyboardInterrupt:
        shutdown(None, None)

    return state.vllm_proc.returncode or 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
