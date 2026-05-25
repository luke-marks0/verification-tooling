#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.core.common.contracts import ValidationError, validate_with_schema
from modules.core.common.deterministic import (
    canonical_json_bytes,
    canonical_json_text,
    compute_bundle_digest,
    compute_lockfile_digest,
    sha256_prefixed,
    utc_now_iso,
)
from modules.inference.manifest.model import Manifest
from modules.network.networkdet import create_net_stack


PCI_ID_RE = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")


def _seed_for_request(run_id: str, req_id: str, prompt: str) -> int:
    digest = sha256_prefixed(canonical_json_bytes({"run_id": run_id, "id": req_id, "prompt": prompt}))
    return int(digest.split(":", 1)[1][:16], 16)


def _tokens(seed: int, count: int = 8) -> list[int]:
    vals = []
    value = seed
    for _ in range(count):
        value = (1103515245 * value + 12345) % (2**31)
        vals.append(value % 50000)
    return vals


def _logits(tokens: list[int]) -> list[float]:
    return [round((tok % 997) / 997.0, 8) for tok in tokens]


def _network_frame_hex_legacy(seed: int, req_id: str) -> str:
    """Legacy synthetic network frame (kept for --network-backend=legacy)."""
    payload = canonical_json_text({"req_id": req_id, "seed": seed}).encode("utf-8")
    return payload.hex()


def _write_json(path: Path, data: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json_text(data), encoding="utf-8")
    return sha256_prefixed(path.read_bytes())


def _hardware_fingerprint(hardware_profile: dict[str, Any]) -> str:
    return sha256_prefixed(canonical_json_bytes(hardware_profile))


def _value_repr(value: Any) -> str:
    return canonical_json_text(value).strip()


def _diff_hardware(expected: Any, actual: Any, path: str, out: list[dict[str, str]]) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            out.append({"path": path, "expected": _value_repr(expected), "actual": _value_repr(actual)})
            return
        keys = sorted(set(expected.keys()) | set(actual.keys()))
        for key in keys:
            key_path = f"{path}.{key}"
            if key not in expected:
                out.append({"path": key_path, "expected": "<absent>", "actual": _value_repr(actual[key])})
                continue
            if key not in actual:
                out.append({"path": key_path, "expected": _value_repr(expected[key]), "actual": "<absent>"})
                continue
            _diff_hardware(expected[key], actual[key], key_path, out)
        return

    if isinstance(expected, list):
        if not isinstance(actual, list) or expected != actual:
            out.append({"path": path, "expected": _value_repr(expected), "actual": _value_repr(actual)})
        return

    if expected != actual:
        out.append({"path": path, "expected": _value_repr(expected), "actual": _value_repr(actual)})


def _normalize_pci_id(value: str) -> str:
    lowered = value.strip().lower()
    if PCI_ID_RE.match(lowered):
        return lowered
    return lowered


def _parse_json_object(raw: str, *, name: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{name} was not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValidationError(f"{name} must decode to a JSON object")
    return parsed


def _env_hardware_profile(expected_hardware: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    observed = copy.deepcopy(expected_hardware)
    evidence: list[str] = []

    overrides = {
        ("gpu", "model"): "RUNNER_GPU_MODEL",
        ("gpu", "driver_version"): "RUNNER_GPU_DRIVER_VERSION",
        ("gpu", "cuda_driver_version"): "RUNNER_GPU_CUDA_DRIVER_VERSION",
    }
    for (section, key), env_key in overrides.items():
        value = os.getenv(env_key)
        if value is None or value.strip() == "":
            continue
        observed[section][key] = value.strip()
        evidence.append(env_key)

    int_overrides = {
        ("gpu", "count"): "RUNNER_GPU_COUNT",
    }
    for (section, key), env_key in int_overrides.items():
        value = os.getenv(env_key)
        if value is None or value.strip() == "":
            continue
        try:
            observed[section][key] = int(value)
        except ValueError as exc:
            raise ValidationError(f"{env_key} must be an integer") from exc
        evidence.append(env_key)

    if len(evidence) == 0:
        return (None, [])
    return (observed, evidence)


def _probe_with_nvidia_smi(expected_hardware: dict[str, Any]) -> tuple[dict[str, Any], list[str]] | None:
    if shutil.which("nvidia-smi") is None:
        return None

    cmd = ["nvidia-smi", "--query-gpu=name,driver_version,pci.bus_id", "--format=csv,noheader"]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        return None

    rows = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if len(rows) == 0:
        return None

    observed = copy.deepcopy(expected_hardware)
    gpu_names: list[str] = []
    driver_version = expected_hardware["gpu"]["driver_version"]
    pci_ids: list[str] = []
    for row in rows:
        parts = [part.strip() for part in row.split(",")]
        if len(parts) < 3:
            continue
        gpu_names.append(parts[0])
        driver_version = parts[1] or driver_version
        pci_ids.append(_normalize_pci_id(parts[2]))

    if len(gpu_names) == 0:
        return None

    observed["gpu"]["model"] = gpu_names[0]
    observed["gpu"]["count"] = len(gpu_names)
    observed["gpu"]["driver_version"] = driver_version
    observed["gpu"]["pci_ids"] = pci_ids
    return (observed, ["nvidia-smi --query-gpu=name,driver_version,pci.bus_id"])


def _probe_runtime_hardware(
    expected_hardware: dict[str, Any],
    runtime_hardware: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if runtime_hardware is not None:
        if not isinstance(runtime_hardware, dict):
            raise ValidationError("runtime_hardware must be a JSON object when provided")
        return (
            runtime_hardware,
            {
                "source": "runtime_hardware_file",
                "evidence": ["--runtime-hardware"],
            },
        )

    env_profile, env_evidence = _env_hardware_profile(expected_hardware)
    if env_profile is not None:
        return (
            env_profile,
            {
                "source": "env_probe",
                "evidence": env_evidence,
            },
        )

    if os.getenv("RUNNER_ENABLE_HOST_PROBE", "").strip().lower() in {"1", "true", "yes"}:
        nvidia_probe = _probe_with_nvidia_smi(expected_hardware)
        if nvidia_probe is not None:
            observed, evidence = nvidia_probe
            return (
                observed,
                {
                    "source": "nvidia_smi",
                    "evidence": evidence,
                },
            )

    return (
        copy.deepcopy(expected_hardware),
        {
            "source": "expected_profile_fallback",
            "evidence": ["Host probe disabled or unavailable; using manifest hardware_profile"],
        },
    )


def _hardware_conformance_record(
    expected_hardware: dict[str, Any],
    observed_hardware: dict[str, Any],
    *,
    strict_hardware: bool,
) -> dict[str, Any]:
    diffs: list[dict[str, str]] = []
    _diff_hardware(expected_hardware, observed_hardware, "$.hardware_profile", diffs)
    status = "conformant" if len(diffs) == 0 else "non_conformant"
    record = {
        "status": status,
        "strict_hardware": strict_hardware,
        "expected_fingerprint": _hardware_fingerprint(expected_hardware),
        "actual_fingerprint": _hardware_fingerprint(observed_hardware),
        "diffs": diffs,
    }
    if strict_hardware and len(diffs) > 0:
        first = diffs[0]
        raise ValidationError(
            "Hardware conformance failed (strict_hardware=true): "
            f"{first['path']} expected={first['expected']} actual={first['actual']}"
        )
    return record


def _artifact_by_type(lockfile: dict[str, Any], artifact_type: str) -> dict[str, Any]:
    for item in lockfile["artifacts"]:
        if item["artifact_type"] == artifact_type:
            return item
    raise ValidationError(f"Lockfile missing required artifact type: {artifact_type}")


def _env_or_default(cli_value: str | None, env_key: str, default: str) -> str:
    if cli_value is not None and cli_value.strip() != "":
        return cli_value
    env_value = os.getenv(env_key)
    if env_value is not None and env_value.strip() != "":
        return env_value
    return default


def _mock_observables(
    m: Manifest,
    manifest_dict: dict[str, Any],
    lockfile: dict[str, Any],
    replica_id: str,
    *,
    network_backend: str = "sim",
    dpdk_port: int = 0,
    dpdk_eal_args: list[str] | None = None,
) -> tuple[list[dict[str, Any]], Any]:
    """Generate mock (stub) observables for testing/CI — no GPU, NOT real inference.

    Returns (request_outputs, net_stack). net_stack is the DeterministicNetStack
    used to generate frames (or None if legacy backend).
    """
    use_legacy = network_backend == "legacy"
    if use_legacy:
        net = None
    else:
        backend_kwargs: dict[str, Any] = {}
        if network_backend == "dpdk":
            backend_kwargs["dpdk_port"] = dpdk_port
            backend_kwargs["dpdk_eal_args"] = dpdk_eal_args or []
        net = create_net_stack(
            manifest_dict, lockfile, backend=network_backend, **backend_kwargs,
        )

    request_outputs: list[dict[str, Any]] = []

    for idx, req in enumerate(m.requests):
        seed = _seed_for_request(m.run_id, req.id, req.prompt)
        toks = _tokens(seed)
        lgt = _logits(toks)

        request_outputs.append({"id": req.id, "tokens": toks, "logits": lgt})

        if not use_legacy:
            response_bytes = canonical_json_bytes({"id": req.id, "tokens": toks, "logits": lgt})
            net.process_response(conn_index=idx, response_bytes=response_bytes)

    return request_outputs, net


def _vllm_observables(
    manifest_dict: dict[str, Any],
    lockfile: dict[str, Any],
    replica_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run real vLLM inference and return observables + env_info."""
    import importlib.util

    _vllm_runner_path = Path(__file__).resolve().parent / "vllm_runner.py"
    _spec = importlib.util.spec_from_file_location("vllm_runner", _vllm_runner_path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    run_vllm = _mod.run_vllm

    result = run_vllm(manifest_dict, lockfile)

    request_outputs = result["request_outputs"]
    return request_outputs, result["env_info"]


def run(
    manifest_dict: dict[str, Any],
    lockfile: dict[str, Any],
    out_dir: Path,
    replica_id: str,
    *,
    mode: str = "vllm",
    network_backend: str = "sim",
    runtime_hardware: dict[str, Any] | None = None,
    pod_manifest_path: str,
    pod_lockfile_path: str,
    pod_runtime_closure_path: str,
    pod_name: str,
    node_name: str,
    namespace: str,
    invocation_argv: list[str],
    dpdk_port: int = 0,
    dpdk_eal_args: list[str] | None = None,
    dpdk_loopback_port: int | None = None,
) -> dict[str, Any]:
    validate_with_schema("manifest.v1.schema.json", manifest_dict)
    validate_with_schema("lockfile.v1.schema.json", lockfile)

    # Parse into typed Manifest for dot-access reads.
    # Keep manifest_dict for canonical JSON serialization (model_dump adds defaults).
    m = Manifest.model_validate(manifest_dict)

    manifest_digest = sha256_prefixed(canonical_json_bytes(manifest_dict))
    if lockfile["manifest_digest"] != manifest_digest:
        raise ValidationError("Lockfile manifest_digest mismatch")

    expected_lockfile_digest = compute_lockfile_digest(lockfile)
    if lockfile["canonicalization"]["lockfile_digest"] != expected_lockfile_digest:
        raise ValidationError("Lockfile canonicalization.lockfile_digest mismatch")

    lock_artifacts_by_id = {item["artifact_id"]: item for item in lockfile["artifacts"]}
    for artifact_input in m.artifact_inputs:
        artifact_id = artifact_input.artifact_id
        if artifact_id not in lock_artifacts_by_id:
            raise ValidationError(f"Lockfile missing artifact required by manifest: {artifact_id}")
        expected_digest = artifact_input.expected_digest
        if expected_digest is not None:
            actual_digest = lock_artifacts_by_id[artifact_id]["digest"]
            if actual_digest != expected_digest:
                raise ValidationError(
                    f"Artifact digest mismatch for {artifact_id}: expected={expected_digest} actual={actual_digest}"
                )

    expected_hardware = manifest_dict["hardware_profile"]
    observed_hardware, hardware_probe = _probe_runtime_hardware(expected_hardware, runtime_hardware)
    hardware_conformance = _hardware_conformance_record(
        expected_hardware,
        observed_hardware,
        strict_hardware=m.runtime.strict_hardware,
    )
    observed_gpu = observed_hardware.get("gpu", expected_hardware["gpu"])

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_copy = out_dir / "manifest.json"
    lockfile_copy = out_dir / "lockfile.json"
    manifest_copy.write_text(canonical_json_text(manifest_dict), encoding="utf-8")
    lockfile_copy.write_text(canonical_json_text(lockfile), encoding="utf-8")

    vllm_env_info: dict[str, Any] | None = None
    tx_report = None

    net = None
    if mode == "vllm":
        request_outputs, vllm_env_info = _vllm_observables(
            manifest_dict, lockfile, replica_id,
        )
        # Generate frames from vLLM output too
        net = create_net_stack(manifest_dict, lockfile, backend="sim")
        for idx, r in enumerate(request_outputs):
            response_bytes = canonical_json_bytes({"id": r["id"], "tokens": r["tokens"], "logits": r["logits"]})
            net.process_response(conn_index=idx, response_bytes=response_bytes)
    else:
        request_outputs, net = _mock_observables(
            m, manifest_dict, lockfile, replica_id,
            network_backend=network_backend,
            dpdk_port=dpdk_port,
            dpdk_eal_args=dpdk_eal_args,
        )

    # Write network frames
    observables_dir = out_dir / "observables"
    network_path = observables_dir / "network_egress.json"

    if net is not None:
        tx_report = net.flush()
        frames = net.capture_frames_hex()
        network_digest = _write_json(network_path, frames)
        network_frame_count = net.frame_count()
        network_capture_digest = net.capture_digest()
        net.close()

        # Loopback verification (Level 2): capture frames on RX port after TX.
        if dpdk_loopback_port is not None and tx_report is not None:
            from modules.network.networkdet.backend_dpdk import DPDKBackend
            from modules.network.networkdet.tx_report import TxReport
            backend = net._backend
            if isinstance(backend, DPDKBackend):
                rx_frames, rx_digest = backend.recv_loopback(timeout_ms=2000)
                tx_report = TxReport(
                    pre_enqueue_digest=tx_report.pre_enqueue_digest,
                    tx_completion_digest=tx_report.tx_completion_digest,
                    frames_submitted=tx_report.frames_submitted,
                    frames_confirmed=tx_report.frames_confirmed,
                    rx_loopback_digest=rx_digest,
                    rx_loopback_count=len(rx_frames),
                )
    else:
        # Legacy fallback
        frames = []
        for req in m.requests:
            seed = _seed_for_request(m.run_id, req.id, req.prompt)
            frames.append({"request_id": req.id, "frame_hex": _network_frame_hex_legacy(seed, req.id)})
        network_digest = _write_json(network_path, frames)
        network_frame_count = len(frames)
        network_capture_digest = network_digest

    tokens_path = observables_dir / "tokens.json"
    logits_path = observables_dir / "logits.json"

    tokens_digest = _write_json(tokens_path, [{"id": r["id"], "tokens": r["tokens"]} for r in request_outputs])
    logits_digest = _write_json(logits_path, [{"id": r["id"], "logits": r["logits"]} for r in request_outputs])

    rerun_metadata = {
        "entrypoint": str(Path(__file__).resolve()),
        "argv": invocation_argv,
        "replica_id": replica_id,
        "manifest_digest": manifest_digest,
        "lockfile_digest": expected_lockfile_digest,
        "runtime_closure_digest": lockfile["runtime_closure_digest"],
        "artifact_count": len(lockfile["artifacts"]),
        "attestation_digests": [item["statement_digest"] for item in lockfile.get("attestations", [])],
    }

    run_bundle: dict[str, Any] = {
        "run_bundle_version": "v1",
        "run_id": m.run_id,
        "created_at": utc_now_iso(),
        "manifest_copy": {
            "path": str(manifest_copy.relative_to(out_dir)),
            "digest": sha256_prefixed(manifest_copy.read_bytes()),
        },
        "lockfile_copy": {
            "path": str(lockfile_copy.relative_to(out_dir)),
            "digest": sha256_prefixed(lockfile_copy.read_bytes()),
        },
        "runtime_closure_digest": lockfile["runtime_closure_digest"],
        "resolved_artifact_digests": [
            {
                "artifact_id": a["artifact_id"],
                "artifact_type": a["artifact_type"],
                "digest": a["digest"],
            }
            for a in lockfile["artifacts"]
        ],
        "environment_info": {
            "vllm_version": vllm_env_info["vllm_version"] if vllm_env_info else "0.1.0-mock",
            "torch_version": vllm_env_info["torch_version"] if vllm_env_info else "2.5.0-mock",
            "cuda_version": vllm_env_info["cuda_version"] if vllm_env_info else "12.4",
            "driver_version": vllm_env_info["driver_version"] if vllm_env_info else observed_gpu.get("driver_version", m.hardware_profile.gpu.driver_version),
            "gpu_inventory": vllm_env_info["gpu_inventory"] if vllm_env_info else [observed_gpu.get("model", m.hardware_profile.gpu.model)],
            "hardware_fingerprint": hardware_conformance["actual_fingerprint"],
        },
        "hardware_probe": hardware_probe,
        "hardware_conformance": hardware_conformance,
        "execution_context": {
            "entrypoint": str(Path(__file__).resolve()),
            "argv": invocation_argv,
            "cwd": str(Path.cwd()),
            "replica_id": replica_id,
            "pod": {
                "name": pod_name,
                "node_name": node_name,
                "namespace": namespace,
            },
            "input_mounts": {
                "manifest_path": pod_manifest_path,
                "lockfile_path": pod_lockfile_path,
                "runtime_closure_path": pod_runtime_closure_path,
            },
        },
        "execution_trace_metadata": {
            "resolved_args": {
                "strict_hardware": str(m.runtime.strict_hardware).lower(),
                "replica_id": replica_id,
            },
            "resolved_env": vllm_env_info.get("resolved_env", {}) if vllm_env_info else {
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "CUDA_LAUNCH_BLOCKING": str(int(m.runtime.deterministic_knobs.cuda_launch_blocking)),
            },
        },
        "rerun_metadata": rerun_metadata,
        "network_provenance": {
            "capture_path": str(network_path.relative_to(out_dir)),
            "capture_digest": network_capture_digest,
            "frame_count": network_frame_count,
            "capture_mode": "userspace_pre_enqueue",
            "capture_isolation": "pre_enqueue_mirror",
            "capture_non_perturbing": True,
            "route_mode": "dpdk_kernel_bypass" if network_backend == "dpdk" else "deterministic_userspace_stack",
            **({"egress_verification": {
                "backend": network_backend,
                "level": tx_report.level,
                "pre_enqueue_digest": tx_report.pre_enqueue_digest,
                "tx_completion_digest": tx_report.tx_completion_digest,
                "frames_submitted": tx_report.frames_submitted,
                "frames_confirmed": tx_report.frames_confirmed,
                "match": tx_report.match,
                **({"rx_loopback_digest": tx_report.rx_loopback_digest,
                    "rx_loopback_count": tx_report.rx_loopback_count,
                    } if tx_report.rx_loopback_digest is not None else {}),
            }} if tx_report is not None else {}),
        },
        "observables": {
            "tokens": {
                "path": str(tokens_path.relative_to(out_dir)),
                "digest": tokens_digest,
            },
            "logits": {
                "path": str(logits_path.relative_to(out_dir)),
                "digest": logits_digest,
            },
            "network_egress": {
                "path": str(network_path.relative_to(out_dir)),
                "digest": network_digest,
            },
        },
        "attestations": [
            {
                "attestation_type": "run_provenance",
                "signer": "runner@deterministic-serving-stack",
                "statement_digest": sha256_prefixed(
                    canonical_json_bytes(
                        {
                            "run_id": m.run_id,
                            "replica_id": replica_id,
                            "runtime_closure_digest": lockfile["runtime_closure_digest"],
                            "manifest_digest": manifest_digest,
                        }
                    )
                ),
                "timestamp": utc_now_iso(),
            }
        ],
        "bundle_digest": "sha256:" + ("0" * 64),
    }

    run_bundle["bundle_digest"] = compute_bundle_digest(run_bundle)
    validate_with_schema("run_bundle.v1.schema.json", run_bundle)
    _write_json(out_dir / "run_bundle.v1.json", run_bundle)
    return run_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic runner scaffold")
    parser.add_argument("--manifest", required=True, help="Manifest JSON path")
    parser.add_argument("--lockfile", required=True, help="Lockfile JSON path")
    parser.add_argument("--out-dir", required=True, help="Output bundle directory")
    parser.add_argument("--replica-id", default="replica-0", help="Replica identifier")
    parser.add_argument("--mode", default="vllm", choices=["mock", "vllm"],
                        help="Execution mode: vllm (real inference, default) or mock "
                             "(no-GPU stub — wiring only, NOT a determinism proof)")
    parser.add_argument("--network-backend", default="sim", choices=["sim", "dpdk", "legacy"],
                        help="Network backend: sim (deterministic frames), dpdk (real NIC), legacy (synthetic hex)")
    parser.add_argument("--runtime-hardware", help="Optional observed runtime hardware profile JSON path")
    parser.add_argument("--dpdk-port", type=int, default=0,
                        help="DPDK port ID for NIC transmission")
    parser.add_argument("--dpdk-eal-args", default="",
                        help="DPDK EAL arguments (space-separated)")
    parser.add_argument("--dpdk-loopback-port", type=int, default=None,
                        help="DPDK RX port for loopback verification (Level 2)")
    parser.add_argument("--pod-manifest-path", help="Mounted manifest path inside the pod")
    parser.add_argument("--pod-lockfile-path", help="Mounted lockfile path inside the pod")
    parser.add_argument("--pod-runtime-closure-path", help="Mounted runtime closure digest path inside the pod")
    parser.add_argument("--pod-name", help="Pod name for provenance recording")
    parser.add_argument("--node-name", help="Node name for provenance recording")
    parser.add_argument("--namespace", help="Namespace for provenance recording")
    args = parser.parse_args()

    manifest_dict = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    lockfile = json.loads(Path(args.lockfile).read_text(encoding="utf-8"))
    runtime_hardware = None
    if args.runtime_hardware:
        runtime_hardware = json.loads(Path(args.runtime_hardware).read_text(encoding="utf-8"))

    run(
        manifest_dict,
        lockfile,
        Path(args.out_dir),
        args.replica_id,
        mode=args.mode,
        network_backend=args.network_backend,
        runtime_hardware=runtime_hardware,
        pod_manifest_path=_env_or_default(args.pod_manifest_path, "RUNNER_POD_MANIFEST_PATH", args.manifest),
        pod_lockfile_path=_env_or_default(args.pod_lockfile_path, "RUNNER_POD_LOCKFILE_PATH", args.lockfile),
        pod_runtime_closure_path=_env_or_default(
            args.pod_runtime_closure_path,
            "RUNNER_POD_RUNTIME_CLOSURE_PATH",
            "lockfile.runtime_closure_digest",
        ),
        pod_name=_env_or_default(args.pod_name, "RUNNER_POD_NAME", "local-pod"),
        node_name=_env_or_default(args.node_name, "RUNNER_NODE_NAME", "local-node"),
        namespace=_env_or_default(args.namespace, "RUNNER_NAMESPACE", "default"),
        invocation_argv=[str(Path(__file__).resolve()), *sys.argv[1:]],
        dpdk_port=args.dpdk_port,
        dpdk_eal_args=args.dpdk_eal_args.split() if args.dpdk_eal_args else [],
        dpdk_loopback_port=args.dpdk_loopback_port,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(str(exc))
        raise SystemExit(1)
