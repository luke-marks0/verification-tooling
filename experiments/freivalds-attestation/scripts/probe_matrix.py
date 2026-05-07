#!/usr/bin/env python3
"""Adversarial probe matrix S0-S6 from plan.md.

Each scenario produces a Response that the verifier judges. We report
the verdict, the max_abs_diff observed, and the wall-clock time so the
detection-margin table can be assembled.

S0 honest        : real GPU matmul
S1 cached_stale  : prover returns C from a different (older) seed
S2 zeros         : prover returns zero matrix
S3 random        : prover returns random bytes (not a real matmul)
S4 dropped_rows  : prover computes correctly for first half, zeros rest
S5 quantized     : prover quantizes B to ~int4 before matmul, claims bf16
S6 stub_kernel   : prover launches a busy-loop kernel, returns garbage

Output: ``data/probe_matrix_v1.json`` and a markdown summary.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pkg.common.deterministic import canonical_json_text, utc_now_iso
from pkg.freivalds import (
    Challenge,
    ComparisonMode,
    MatmulSpec,
    Tolerance,
    execute_challenge,
    verify_response,
)
from pkg.freivalds import prng
from pkg.freivalds.spec import MatmulResult, Response, INTEGER_DTYPES


def _import_torch_backend():
    from pkg.freivalds.backends.torch_backend import TorchBackend
    return TorchBackend


def _build_challenge(dim: int, dtype: str, comparison: ComparisonMode) -> Challenge:
    is_int = dtype in INTEGER_DTYPES
    spec = MatmulSpec(
        id=f"probe-{dtype}-{dim}",
        M=dim, K=dim, N=dim,
        dtype_a=dtype, dtype_b=dtype,
        dtype_acc="int32" if is_int else "fp32",
        dtype_c="int32" if is_int else dtype,
        seed_a=987_654_321 + dim,
        seed_b=123_456_789 + dim,
        comparison=comparison,
        tolerance=None if comparison is ComparisonMode.BITWISE else Tolerance(atol=1e-1, rtol=1e-1),
    )
    return Challenge(challenge_id=f"probe.{dtype}.{dim}", matmuls=(spec,))


# --- adversaries: each takes (challenge, backend) and returns a Response ---

def _restamp_results(response: Response, **per_id_overrides) -> Response:
    """Return a copy of ``response`` with per-id MatmulResult overrides."""
    new = []
    for r in response.results:
        ov = per_id_overrides.get(r.id, {})
        new.append(MatmulResult(
            id=r.id,
            digest_a=ov.get("digest_a", r.digest_a),
            digest_b=ov.get("digest_b", r.digest_b),
            digest_c=ov.get("digest_c", r.digest_c),
            c_b64=ov.get("c_b64", r.c_b64),
            wall_time_ms=ov.get("wall_time_ms", r.wall_time_ms),
            device=ov.get("device", r.device),
            device_name=ov.get("device_name", r.device_name),
            nvml_clock_mhz=ov.get("nvml_clock_mhz", r.nvml_clock_mhz),
            nvml_temp_c=ov.get("nvml_temp_c", r.nvml_temp_c),
        ))
    return Response(challenge_id=response.challenge_id, backend=response.backend, results=tuple(new))


def adv_honest(ch: Challenge, backend) -> Response:
    return execute_challenge(ch, backend)


def adv_cached_stale(ch: Challenge, backend) -> Response:
    """Prover ran the matmul for an OLDER seed and returns that C verbatim."""
    # Build a stale challenge with different seeds and compute it for real.
    stale_specs = []
    for spec in ch.matmuls:
        stale_specs.append(MatmulSpec(
            id=spec.id, M=spec.M, K=spec.K, N=spec.N,
            dtype_a=spec.dtype_a, dtype_b=spec.dtype_b,
            dtype_acc=spec.dtype_acc, dtype_c=spec.dtype_c,
            seed_a=spec.seed_a + 999_999, seed_b=spec.seed_b + 999_999,
            comparison=spec.comparison, tolerance=spec.tolerance,
        ))
    stale = Challenge(challenge_id=ch.challenge_id, matmuls=tuple(stale_specs))
    stale_resp = execute_challenge(stale, backend)
    # Now lie: pretend digest_a / digest_b correspond to the *current* seeds
    # so the prng-drift check passes; only Freivalds will catch us.
    overrides = {}
    for spec, sr in zip(ch.matmuls, stale_resp.results):
        ab = prng.gen_matrix_bytes(spec.seed_a, spec.dtype_a, spec.M, spec.K)
        bb = prng.gen_matrix_bytes(spec.seed_b, spec.dtype_b, spec.K, spec.N)
        overrides[spec.id] = {
            "digest_a": prng.matrix_digest(ab),
            "digest_b": prng.matrix_digest(bb),
            # digest_c and c_b64 are still from the stale matmul.
        }
    return _restamp_results(stale_resp, **overrides)


def adv_zeros(ch: Challenge, backend) -> Response:
    honest = execute_challenge(ch, backend)
    overrides = {}
    for spec, r in zip(ch.matmuls, honest.results):
        Z = backend.zeros_matrix(spec.M, spec.N, spec.dtype_c)
        Z_bytes = backend.write_matrix_to_bytes(Z, spec.dtype_c)
        overrides[spec.id] = {
            "c_b64": base64.b64encode(Z_bytes).decode("ascii"),
            "digest_c": prng.matrix_digest(Z_bytes),
        }
    return _restamp_results(honest, **overrides)


def adv_random(ch: Challenge, backend) -> Response:
    """Prover returns deterministic-but-unrelated bytes for C."""
    import os
    honest = execute_challenge(ch, backend)
    overrides = {}
    for spec, r in zip(ch.matmuls, honest.results):
        n_bytes = spec.M * spec.N * prng.bytes_per_elem(spec.dtype_c)
        # Use a fixed seed so results are reproducible across runs.
        rb = prng.gen_matrix_bytes(seed=999, dtype=spec.dtype_c, rows=spec.M, cols=spec.N)
        overrides[spec.id] = {
            "c_b64": base64.b64encode(rb).decode("ascii"),
            "digest_c": prng.matrix_digest(rb),
        }
    return _restamp_results(honest, **overrides)


def adv_dropped_rows(ch: Challenge, backend) -> Response:
    """Compute correctly for the first half, zero the second half."""
    import torch
    honest = execute_challenge(ch, backend)
    overrides = {}
    for spec, r in zip(ch.matmuls, honest.results):
        c_bytes = bytearray(base64.b64decode(r.c_b64))
        bpe = prng.bytes_per_elem(spec.dtype_c)
        per_row = spec.N * bpe
        half = spec.M // 2
        # Zero out rows [half, M).
        for i in range(half, spec.M):
            start = i * per_row
            for j in range(per_row):
                c_bytes[start + j] = 0
        new_bytes = bytes(c_bytes)
        overrides[spec.id] = {
            "c_b64": base64.b64encode(new_bytes).decode("ascii"),
            "digest_c": prng.matrix_digest(new_bytes),
        }
    return _restamp_results(honest, **overrides)


def adv_quantized(ch: Challenge, backend) -> Response:
    """Prover quantizes B much harder than declared (e.g., bf16 -> int4-ish)."""
    import torch
    honest_results = []
    for spec in ch.matmuls:
        A, A_bytes = backend.gen_matrix(spec.seed_a, spec.dtype_a, spec.M, spec.K)
        B, B_bytes = backend.gen_matrix(spec.seed_b, spec.dtype_b, spec.K, spec.N)
        # Cheap "quantization": round B to nearest 1/8 (loses precision).
        B_q = (B.to(torch.float32) * 8.0).round() / 8.0
        if spec.dtype_b == "bf16":
            B_q = B_q.to(torch.bfloat16)
        elif spec.dtype_b == "fp16":
            B_q = B_q.to(torch.float16)
        else:
            B_q = B_q.to(B.dtype)
        t0 = time.perf_counter()
        C = backend.matmul(A, B_q, spec.dtype_a, spec.dtype_b, spec.dtype_acc, spec.dtype_c)
        if backend.device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        C_bytes = backend.write_matrix_to_bytes(C, spec.dtype_c)
        honest_results.append(MatmulResult(
            id=spec.id,
            digest_a=prng.matrix_digest(A_bytes),
            digest_b=prng.matrix_digest(B_bytes),  # *claim* honest B, lie about it
            digest_c=prng.matrix_digest(C_bytes),
            c_b64=base64.b64encode(C_bytes).decode("ascii"),
            wall_time_ms=(t1 - t0) * 1000.0,
            device=str(backend.device),
            device_name=backend.device_info().get("device_name", ""),
        ))
    return Response(challenge_id=ch.challenge_id, backend=backend.name, results=tuple(honest_results))


def adv_stub_kernel(ch: Challenge, backend) -> Response:
    """Launch a busy-loop unrelated to the matmul, return garbage C with timing
    that resembles the honest case (best-effort)."""
    import torch
    results = []
    for spec in ch.matmuls:
        A, A_bytes = backend.gen_matrix(spec.seed_a, spec.dtype_a, spec.M, spec.K)
        B, B_bytes = backend.gen_matrix(spec.seed_b, spec.dtype_b, spec.K, spec.N)
        # Garbage: use a noise tensor of the right shape.
        C_garb = torch.randn(spec.M, spec.N, dtype=A.dtype if A.is_floating_point() else torch.float32, device=backend.device)
        if spec.dtype_c in INTEGER_DTYPES:
            C_garb = C_garb.to(torch.int32)
        # Stall the GPU with unrelated busy work to mimic honest timing.
        t0 = time.perf_counter()
        x = torch.zeros(1024, 1024, device=backend.device, dtype=torch.float32)
        for _ in range(10):
            x = x + 1.0
        if backend.device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        C_bytes = backend.write_matrix_to_bytes(C_garb, spec.dtype_c)
        results.append(MatmulResult(
            id=spec.id,
            digest_a=prng.matrix_digest(A_bytes),
            digest_b=prng.matrix_digest(B_bytes),
            digest_c=prng.matrix_digest(C_bytes),
            c_b64=base64.b64encode(C_bytes).decode("ascii"),
            wall_time_ms=(t1 - t0) * 1000.0,
            device=str(backend.device), device_name=backend.device_info().get("device_name", ""),
        ))
    return Response(challenge_id=ch.challenge_id, backend=backend.name, results=tuple(results))


SCENARIOS = [
    ("S0_honest",       adv_honest,        "real GPU matmul"),
    ("S1_cached_stale", adv_cached_stale,  "C from a stale seed"),
    ("S2_zeros",        adv_zeros,         "C = 0"),
    ("S3_random",       adv_random,        "C = unrelated bytes"),
    ("S4_dropped_rows", adv_dropped_rows,  "first half correct, second half zeroed"),
    ("S5_quantized",    adv_quantized,     "B aggressively quantized then matmul"),
    ("S6_stub_kernel",  adv_stub_kernel,   "busy-loop kernel; C is noise"),
]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--md-out", default=None, help="Optional markdown summary path")
    p.add_argument("--dim", type=int, default=4096)
    p.add_argument("--dtype", default="bf16",
                   help="Probe dtype (bf16 by default; int8 for bitwise)")
    args = p.parse_args(argv)

    TorchBackend = _import_torch_backend()
    backend = TorchBackend(device="cuda")

    is_int = args.dtype in INTEGER_DTYPES
    comp = ComparisonMode.BITWISE if is_int else ComparisonMode.TOLERANCE
    ch = _build_challenge(args.dim, args.dtype, comp)

    rows = []
    print(f"[probe] dim={args.dim} dtype={args.dtype} comparison={comp.value}")
    for name, fn, desc in SCENARIOS:
        try:
            t0 = time.perf_counter()
            resp = fn(ch, backend)
            t1 = time.perf_counter()
            report = verify_response(ch, resp, backend, r_seed_source=lambda: 7777)
            v = report.matmuls[0]
            rows.append({
                "scenario": name,
                "description": desc,
                "verdict_passed": v.passed,
                "max_abs_diff": v.max_abs_diff,
                "cr_inf_norm": v.cr_inf_norm,
                "wall_time_ms": v.wall_time_ms,
                "scenario_total_ms": (t1 - t0) * 1000.0,
                "reason": v.reason,
                "digest_a_match": v.digest_a_match,
                "digest_b_match": v.digest_b_match,
            })
            mark = "PASS" if v.passed else "FAIL"
            print(f"  {name:<18} verdict={mark}  diff={v.max_abs_diff:.3g}  "
                  f"prover_time={v.wall_time_ms:.1f}ms  reason={v.reason[:60]}")
        except Exception as exc:
            rows.append({"scenario": name, "description": desc, "error": str(exc)})
            print(f"  {name:<18} ERROR: {exc}")

    out = {
        "probe_matrix_version": "v1",
        "generated_at": utc_now_iso(),
        "challenge": {"dim": args.dim, "dtype": args.dtype, "comparison": comp.value},
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(canonical_json_text(out), encoding="utf-8")
    print(f"[probe] wrote {args.out}")

    if args.md_out:
        lines = [
            "# Adversarial probe matrix",
            "",
            f"Hardware: {backend.device_info().get('device_name', '?')}  ",
            f"Challenge: dim={args.dim} dtype={args.dtype} comparison={comp.value}  ",
            f"Generated: {out['generated_at']}",
            "",
            "| Scenario | Description | Verdict | max_abs_diff | prover_time_ms | reason |",
            "|---|---|---|---|---|---|",
        ]
        for r in rows:
            if "error" in r:
                lines.append(f"| {r['scenario']} | {r['description']} | ERROR | — | — | `{r['error'][:60]}` |")
                continue
            verdict = "PASS" if r["verdict_passed"] else "FAIL"
            lines.append(
                f"| {r['scenario']} | {r['description']} | **{verdict}** | "
                f"{r['max_abs_diff']:.3g} | {r['wall_time_ms']:.1f} | {r['reason'][:60]} |"
            )
        Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md_out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[probe] wrote {args.md_out}")

    # The headline soundness assertion: S0 must pass, S1-S6 must fail.
    s0 = next((r for r in rows if r["scenario"] == "S0_honest"), None)
    fails_caught = sum(1 for r in rows if r["scenario"] != "S0_honest" and not r.get("verdict_passed", True))
    total_adv = sum(1 for r in rows if r["scenario"] != "S0_honest" and "error" not in r)
    print(f"[probe] honest=PASS:{s0 and s0.get('verdict_passed')}  adversarial_caught={fails_caught}/{total_adv}")
    if not (s0 and s0.get("verdict_passed")) or fails_caught != total_adv:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
