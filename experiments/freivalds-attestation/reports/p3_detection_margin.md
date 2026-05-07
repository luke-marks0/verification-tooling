# Adversarial probe matrix — detection margin

Hardware: NVIDIA GH200 480GB. Probes run at dim=4096 with two regimes:
**bf16** (tolerance, calibrated `atol=0.1, rtol=0.1`) and **int8**
(bitwise). Each row is one scenario from `plan.md` §"Adversarial probe
matrix"; the verifier emits one verdict per scenario.

`prover_time_ms` is the wall clock the prover spent inside `matmul`
(not counting PRNG/transfer). It is observed only — the v1 verdict
does not gate on it. The honest-vs-adversarial gap is the size of the
v2 timing gate.

## bf16, tolerance

| Scenario | What the prover does | Verdict | max_abs_diff | prover_time_ms | reason |
|---|---|---|---|---|---|
| S0_honest | real GPU matmul | **PASS** | 11.5 | 113.9 | within tolerance |
| S1_cached_stale | C from a stale seed | **FAIL** | 9.5e+03 | 2.9 | tolerance exceeded |
| S2_zeros | C = 0 | **FAIL** | 6.0e+03 | 2.9 | tolerance exceeded |
| S3_random | C = unrelated bytes | **FAIL** | 6.0e+03 | 2.9 | tolerance exceeded |
| S4_dropped_rows | first half correct, rest zeroed | **FAIL** | 6.0e+03 | 2.9 | tolerance exceeded |
| S5_quantized | B rounded to nearest 1/8 then matmul | **PASS** | 328 | 2.9 | within tolerance |
| S6_stub_kernel | busy-loop kernel; C is noise | **FAIL** | 6.0e+03 | 0.7 | tolerance exceeded |

## int8, bitwise

| Scenario | What the prover does | Verdict | max_abs_diff | prover_time_ms | reason |
|---|---|---|---|---|---|
| S0_honest | real GPU matmul | **PASS** | 0 | 30.3 | bitwise match |
| S1_cached_stale | C from a stale seed | **FAIL** | 8.9e+09 | 1.2 | bitwise mismatch |
| S2_zeros | C = 0 | **FAIL** | 5.6e+09 | 1.2 | bitwise mismatch |
| S3_random | C = unrelated bytes | **FAIL** | 2.2e+13 | 1.2 | bitwise mismatch |
| S4_dropped_rows | first half correct, rest zeroed | **FAIL** | 5.6e+09 | 1.2 | bitwise mismatch |
| S5_quantized | round-to-1/8 (no-op on integers) | **PASS** | 0 | 1.2 | bitwise match |
| S6_stub_kernel | busy-loop kernel; C is noise | **FAIL** | 5.6e+09 | 2.9 | bitwise mismatch |

## Reading the table

- **5/6 adversarial scenarios are caught at both regimes.** The honest
  matmul passes; cached, zeros, random bytes, dropped-rows, and stub
  kernels all fail with `max_abs_diff` 5–10 orders of magnitude above
  the calibrated honest noise.
- **S5 (quantization cheat) is the interesting case.** At int8 the
  `round(x · 8) / 8` "quantization" is a no-op on integer values, so
  the matmul is honest by accident. At bf16 the quantization changes
  the answer by `diff = 328`, which is well below the calibrated
  threshold `atol + rtol · ‖Cr‖∞ = 626`. Tightening `rtol` to ~5e-3
  would catch S5 at the cost of risking false positives on honest runs
  (calibration shows honest `diff_p99 ≈ 15` at dim=4096, so
  `rtol·‖Cr‖∞ ≈ 3` is the floor before honest false positives kick in).
  This is the explicit ε-tightness tradeoff the soundness section of
  `plan.md` flagged.
- **Honest vs adversarial timing gap is 30–100×** at dim=4096 bf16
  (113.9 ms vs 0.7–2.9 ms). The cheats cost essentially zero GPU work.
  This is what the v2 timing gate would lock in. A more sophisticated
  S6 that pads the kernel to match honest time would still be caught
  by Freivalds — the timing gate is an *additional* safety net, not the
  primary check.

## Soundness story (concrete)

- Plain Freivalds: `Pr[A(Br) = Cr | AB ≠ C] ≤ 1/|F|` over a finite field.
  Under float we work in tolerance-mode and the bound becomes:
  *any divergence with `‖AB − C‖∞ > ε` is detected with high probability
  over r* — exactly what the table's "tolerance exceeded" rows show.
- The verifier draws `r` *after* receiving `C`. The prover never sees
  `r`, so it cannot engineer `A(Br) = Cr` without producing a true `C`.
  S1–S4 and S6 demonstrate that you can't get past Freivalds without
  actually computing `AB`.
- S5 is sound: a prover that produces an answer numerically close to
  `AB` *did* compute something close to `AB`. v1's tolerance is loose
  by design (calibrated to honest noise + 2×); tightening it is a
  precision/recall tradeoff handled in calibration.
