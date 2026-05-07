"""Prover side: execute a :class:`Challenge` and emit a :class:`Response`.

Honest prover. Adversarial behaviours (cached / zero / random / dropped)
live as separate hooks in ``experiments/freivalds-attestation/scripts/``.
"""
from __future__ import annotations

import base64

from pkg.freivalds.spec import Challenge, MatmulResult, Response
from pkg.freivalds import prng


def execute_challenge(challenge: Challenge, backend) -> Response:
    """Run every matmul in ``challenge`` with ``backend`` and bundle the answers."""
    results: list[MatmulResult] = []
    info = backend.device_info()
    for spec in challenge.matmuls:
        # Materialise A and B from seeds. The bytes here are the canonical
        # bytes — same on any backend that follows the prng spec.
        A, A_bytes = backend.gen_matrix(spec.seed_a, spec.dtype_a, spec.M, spec.K)
        B, B_bytes = backend.gen_matrix(spec.seed_b, spec.dtype_b, spec.K, spec.N)

        t0 = backend.perf_time_ms()
        C = backend.matmul(A, B, spec.dtype_a, spec.dtype_b, spec.dtype_acc, spec.dtype_c)
        t1 = backend.perf_time_ms()

        C_bytes = backend.write_matrix_to_bytes(C, spec.dtype_c)

        results.append(MatmulResult(
            id=spec.id,
            digest_a=prng.matrix_digest(A_bytes),
            digest_b=prng.matrix_digest(B_bytes),
            digest_c=prng.matrix_digest(C_bytes),
            c_b64=base64.b64encode(C_bytes).decode("ascii"),
            wall_time_ms=t1 - t0,
            device=str(info.get("device", "")),
            device_name=str(info.get("device_name", "")),
            nvml_clock_mhz=info.get("nvml_clock_mhz"),
            nvml_temp_c=info.get("nvml_temp_c"),
        ))

    return Response(
        challenge_id=challenge.challenge_id,
        backend=backend.name,
        results=tuple(results),
    )
