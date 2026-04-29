# Freivalds Matmul Attestation — Experiment Log

Append-only. Newest entries at the bottom.

---

## 2026-04-29 — Plan drafted, P1 implementation started

- Created `plan.md`: single-round Freivalds protocol, prover returns full C, verifier picks r locally; bitwise mode for int, calibrated-tolerance for float; verifier picks parameters per-call (no manifest pre-commit in v1); timing observed but not gated.
- Threat model: catch cached / zero / random / dropped-rows / quantized-cheat / stub-kernel provers. "Delegated to faster GPU" is v2 (needs timing calibration).
- Scope decisions logged in plan: fp4 deferred (no native H100/GH200 support), single-GPU only, no zk, no manifest integration in v1.
- Codebase split: reusable code in `pkg/freivalds/`, experiment harness in `experiments/freivalds-attestation/`. Promotion to a `POST /attest` endpoint on `cmd/server/main.py` is P4.
- P1 deliverables: spec, prng, check, prover, verifier on a stdlib backend (so unit tests run on a CPU-only box without torch/numpy); schemas for challenge and attestation report; unit tests; in-process smoke script that exercises honest + adversarial paths.

## 2026-04-29 — P1 landed

- `pkg/freivalds/{__init__,spec,prng,check,prover,verifier}.py` + `pkg/freivalds/backends/{__init__,stdlib}.py`.
- `schemas/freivalds_challenge.v1.schema.json`, `schemas/freivalds_attestation.v1.schema.json` (canonicalized; schema gate green).
- `tests/unit/test_freivalds_{prng,check,protocol,schemas}.py` — 42 tests, all pass on a CPU-only box (no torch/numpy).
- `experiments/freivalds-attestation/scripts/run_smoke.py` — honest, zero-C, single-byte-tamper round-trips. All three pass.
- One bug found and fixed during testing: `freivalds_check` was wrapping intermediate vectors (`Br`, `ABr`, `Cr`) to the *input* dtype instead of the *accumulator* dtype, which truncated int8 intermediates and caused honest runs to fail. Now intermediates live in `dtype_acc` end to end. Lesson logged for the torch backend: the same trap applies on GPU — reductions must stay in fp32 even when inputs are bf16/fp16.
- Phase 2 (torch backend, calibration on H100) is the next unit of work; `scripts/calibrate.py` is a stub.

## 2026-04-29 — Phase 2 + Phase 3 landed (GH200)

- Launched a `gpu_1x_gh200` on Lambda Cloud (`us-east-3`, `$2.29/hr`, GH200 480GB, torch 2.7.0, driver 570.148.08, compute capability 9.0). After capacity bounce, second launch succeeded.
- Implemented `pkg/freivalds/backends/torch_backend.py` covering int8/int32/fp16/bf16/fp32/fp64 (fp8 left as a try/except path; not exercised in v1 since calibration didn't request it). Uses `torch._int_mm` when available; verifier-side int matvec routes through CPU because CUDA has no int64 GEMM and the cheap O(n²) check is unaffected by host execution.
- Two non-trivial bugs hit on the box, both fixed:
  - The CUDA `addmv` op has no int64 support → routed verifier-side int matvec to CPU.
  - The calibration time was dominated by the Python bit-twiddle in `pkg/freivalds/prng.py` (≈1 s per 8192² matrix). Vectorised with numpy when available; pure-stdlib path retained as the spec. After the fix, generating an 8192² bf16 matrix is single-digit ms.
- One timing trap caught: the calibration script was timing the whole `execute_challenge`, which is dominated by PRNG + host/device transfer. Switched to use `MatmulResult.wall_time_ms` (cuda-synced t1−t0 around the matmul itself) for the TF/s math.
- Calibration results (`data/calibration_v1.json`, `reports/p2_calibration.md`):
  - Per-trial timing IQR ≤ 1% at dim ≥ 4096 across all dtypes — tight enough that the timing gate planned for v2 should be straightforward.
  - Per-trial bf16/fp16 read 5% of tensor-core peak; fp32 reads 76% of cuda-core peak. The per-trial bf16/fp16 number reflects fresh-tensor cuBLAS dispatch overhead, not the hardware ceiling.
- Saturation probe (`data/saturation_probe_v1.json`, `reports/p2_saturation.md`) — tight matmul loop with 5 ms NVML sampling. Demonstrates the saturation claim from `plan.md`: bf16 dim=4096 hits **820 TF/s = 83% peak**, sm_util_median = sm_util_max = **100%**, power = 225 W. Confirmed at fp16 (74% peak) and fp32 (77% peak, 580 W = near TDP).
- Adversarial probe matrix (`data/probe_matrix_*.json`, `reports/p3_*.md`, `reports/p3_detection_margin.md`):
  - Honest = PASS, **5/6 adversarial scenarios caught** at both bf16 (tolerance) and int8 (bitwise) regimes.
  - The S5 quantization-cheat passes at v1 calibration, as expected — `diff = 328 < threshold ≈ 626`. Tightening `rtol` to ~5e-3 would catch it at the cost of false positives on honest runs (`diff_p99 ≈ 15` is the honest floor at dim=4096 bf16). Logged as a precision/recall knob for v2.
  - Honest matmul takes 113.9 ms at dim=4096 bf16; adversarial takes 0.7–2.9 ms. **30–100× timing gap** is the size of the v2 timing gate.
- Phase 4 (promote endpoint to `cmd/server/main.py`) is still future work. v1 closes here.
- Terminated GH200 to stop billing.

## 2026-04-29 — Multi-GPU saturation sweep

- Goal: confirm the saturation claim holds beyond the single GH200 we ran on. Ran the saturation probe (tight bf16/fp16/fp32/int8 matmul loop, 5 ms NVML sampling) on three GPUs:
  - GH200 480GB (Hopper sm_90) — already had this from yesterday.
  - A100 SXM4 40GB (Ampere sm_80) — `gpu_1x_a100_sxm4` in us-west-2, $1.99/hr.
  - A10 24GB (Ampere sm_86) — `gpu_1x_a10` in us-east-1, $1.29/hr.
- Polled for `gpu_1x_h100_*` capacity for ~15 min in parallel; nothing returned. Stopped the poll.
- Updated `saturation_probe.py` with a per-GPU peak-TFLOPS lookup so `% peak` reads correctly across SM_80/86/90/89.
- Telemetry confirms `sm_util_max=100%` across every cell at dim=8192 — kernel saturates the GPU on all three. Fraction of vendor peak:
  - GH200 bf16 4096³ → **820 TF/s = 83% peak** (225 W).
  - A100 SXM4 bf16 4096³ → **267 TF/s = 86% peak**; fp32 8192³ → **98% peak** (294 W). Best saturation of the three; the SXM4 form factor has thermal headroom.
  - A10 bf16 8192³ → 78 TF/s = **62% peak** with sm_util=100%, **but at 147 W of 150 W TDP** — chip is power-limited, clock drops 1650 → 1110 MHz under sustained load. Fraction-of-peak is lower not because the GPU is idle but because the chip throttles. This is the cleanest "saturated" signature of the three.
- The honest TF/s bands don't overlap across GPUs: GH200 ~750–820, A100 ~265–275, A10 ~70–80. A v2 timing gate that knows the claimed hardware can detect "delegated to a faster GPU" attacks.
- One artefact across all three GPUs: `torch._int_mm` is dispatching to a slow generic IMMA path, not optimised cuBLASLt int8 GEMM. Flagged in the multi-GPU report; v2 will switch int8 to `cublasLtMatmul` directly.
- Output: `data/multi-gpu/saturation_{gh200,a100_sxm4,a10}.json`, `reports/multi-gpu/p2_saturation_{a100_sxm4,a10}.md`, `reports/p2_saturation_multi_gpu.md` (cross-GPU table).
- Terminated A100 + A10 to stop billing.

## 2026-04-29 — Vast extension: H200 + L40S + RTX 4090

- Lambda was empty for any H100 family after 15 min of polling, so picked up three more GPUs from vast.ai using a stock pytorch image (`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`) in `--ssh` mode (the Nix-image-only restriction in CLAUDE.md doesn't apply when we don't use the project image). Wrote a self-contained `freivalds_remote_probe.py` (no project imports — vendored NVML sampler + peak table) so we don't need to ship `pkg/` to each new instance.
- Hardware tested: H200 (sm_90, 144 GB), L40S (sm_89, 48 GB), RTX 4090 (sm_89, 24 GB consumer). Combined with the Lambda runs, total coverage is **6 GPUs across SM_80, SM_86, SM_89, SM_90**.
- Two bugs found while building the multi-GPU report:
  - The peak-TFLOPS table used substring matching, so "L40" matched "L40S" first and gave half the right peak. Fixed by sorting candidates longest-first; same fix applied to the in-tree `saturation_probe.py`.
  - The peak table mixed dense and sparsity numbers (e.g. RTX 4090 fp16 = 330 with sparsity, not 165 dense). Vendor dense numbers used everywhere now.
- Saturation evidence holds across the board (`reports/p2_saturation_multi_gpu.md`):
  - H200 bf16 8192³ → 733 TF/s (74% peak), **569 W steady, 677 W peak — close to 700 W TDP**.
  - L40S bf16 4096³ → 248 TF/s (69% dense peak).
  - RTX 4090 bf16 8192³ → 169 TF/s (≈100% of dense peak); sm_util reads low because consumer NVML on vast samples at ~1 Hz, but **power = 361 W at fp16** and observed TF/s confirm.
- Cross-GPU honest TF/s bands at bf16 don't overlap: H100-class 720–820, A100 265–275, L40S 195–250, RTX 4090 159–169, A10 70–80. v2 timing gate can use these to detect "delegated to a faster/slower GPU than claimed".
- One persistent caveat across all six GPUs: `torch._int_mm` reads 5–25% of vendor int8 peak. SM saturation still reaches 100% at dim ≥ 8192 so attestation works, but reporting against the IMMA peak is misleading. v2 will call `cublasLtMatmul` directly.
- Cost: H200 + L40S + RTX 4090 ran for ~10 min combined, ≈ \$0.50 total. All vast instances destroyed.

## 2026-04-29 — SM occupancy: arbitrary fraction of cores, A100 SXM4

- Goal (from the user): "pick a single GPU type and show that we can occupy
  an arbitrary % of cores for that GPU type … to as good a precision as
  possible." Picked A100 SXM4 (108 SMs, sm_80) on Lambda, $1.99/hr us-east-1.
- New script: `experiments/freivalds-attestation/scripts/sm_occupancy_sweep.py`.
  JIT-compiles a custom CUDA kernel via `torch.utils.cpp_extension.load_inline`
  and exposes a small public API:
  ```python
  ctrl = OccupancyController()       # detects n_sms, calibrates per-SM duration
  ctrl.occupy(fraction=0.50, duration_s=1.0)   # 50 % of SMs for ~1 s
  ctrl.sweep([0.01, 0.10, 0.50, 1.00, 1.50])
  ```
- Mechanism: 1024 threads/block × 96 KB dynamic shared memory per block.
  A100 has 164 KB SMEM/SM, so 2 × 96 = 192 KB > 164 KB ⇒ hardware can never
  co-resident two blocks on one SM. `grid_size = N` ⇒ exactly N SMs busy.
  Three bugs hit before it worked (see below); all in the kernel side.
- Results (`data/sm_occupancy/sweep_a100.json`, `reports/sm_occupancy_a100.md`)
  on 108-SM A100, single-SM kernel calibrated to 686 ms:
  | target % | blocks | kernel_ms | Δpower | predicted | residual |
  |---|---|---|---|---|---|
  | 1   | 1   | 1000 | 6.4 W  | 6.2 W  | +0.2 |
  | 25  | 27  | 1006 | 25.2 W | 24.4 W | +0.8 |
  | 50  | 54  | 1008 | 41.6 W | 43.3 W | −1.7 |
  | 75  | 81  | 1003 | 61.7 W | 62.2 W | −0.5 |
  | 100 | 108 | 1004 | 82.2 W | 81.1 W | +1.1 |
  Δpower = (0.70 W/SM)·N + 5.5 W. RMS residual 0.9 W ⇒ inversion
  precision of **±1.3 SMs out of 108 (~1.2% of the GPU)**. Verifier
  can target any fraction in [1%, 100%] and confirm it from NVML power
  alone with ~1% absolute precision.
- Queued regime (N > 108) confirmed: at 150% the script halves per-block
  iters so wall stays ≈ 1 s. Time-averaged active SMs become
  (108+54)/2 = 81, predicting Δpower ≈ 62 W; observed 66 W. At 200% the
  average is 108, predicting 81 W; observed 80 W. ✓
- Three bugs and fixes during development:
  - **load_inline missing forward decl.** load_inline auto-generates a
    pybind module that calls `launch_busy`, but with `cpp_sources=""` the
    function isn't declared in main.cpp, only in cuda.cu. Added a
    forward declaration string `CPP_DECL` to `cpp_sources`.
  - **Compiler elided the FMA loop.** First kernel had a sentinel
    branch `if (smem[tid] == -1.0e30f) scratch[...] = x`, which `-O3
    --use_fast_math` folded away. Result: kernel returned in ~0 ms,
    power read idle. Fix: cooperatively populate 96 KB of smem,
    cross-thread shuffle through smem, **unconditional** scratch write.
    Also dropped `--use_fast_math` for safety.
  - **Calibration runaway.** When the elided kernel returned in
    sub-microseconds, `if dt < 0.5: n_iters *= 16` ran 5+ times,
    pushing iters to 300 billion. Capped iters at 200M and added a
    refinement pass that scales by `target_ms / observed_ms`.
- Lambda env: torch 2.7.0 / CUDA 12.8 / driver 570.148. Needed `pip
  install --user ninja pybind11` and `CPLUS_INCLUDE_PATH` pointing at
  pybind11's headers; nothing else.
- Cost: ~25 min on A100 SXM4 = ~\$0.85. Instance terminated.
- This is the v2 building block: the verifier can schedule a chosen mix
  of matmul + busy-kernel + idle to land the prover at any target
  power/compute level, then check telemetry against the predicted curve.

## 2026-04-29 — SM occupancy v3: direct hardware verification

- User asked: "how do we know that those were the number of cores used?
  just the power?" Honest answer was no — Layers 1 (resource math) +
  Layer 2 (constant kernel_ms) prove ≤ N and ≥ N respectively, but
  power is corroborative. User said: "do it" — add the hardware
  contract + direct telemetry reading.
- Two additions to `scripts/sm_occupancy_sweep.py`:
  - **Layer 1 hardware contract.** New CUDA function
    `query_max_blocks_per_sm(threads, smem_bytes)` calls NVIDIA's
    `cudaOccupancyMaxActiveBlocksPerMultiprocessor`. This is a runtime
    API that NVIDIA computes from the kernel's resource profile; if it
    returns 1, the scheduler **physically cannot** put two blocks on an
    SM. Result on A100: `query_max_blocks_per_sm(1024, 98304) = 1`.
    Verifiable, vendor-attested.
  - **Layer 3 direct measurement.** New `scripts/dcgm_sampler.py` that
    spawns `dcgmi dmon -e 1002 -d 100` (DCGM_FI_PROF_SM_ACTIVE, the
    ratio of cycles SMs were busy averaged across all SMs) as a
    subprocess and parses per-line output. SMACT × n_sms recovers the
    active-block count without going through power.
- Re-launched A100 SXM4 in us-east-1 ($1.99/hr). Lambda's image had no
  DCGM; added NVIDIA's CUDA repo via `cuda-keyring_1.1-1_all.deb`,
  installed `datacenter-gpu-manager`, started `nvidia-dcgm.service`.
  dcgmi 3.3.9.
- Sweep results (`data/sm_occupancy/sweep_a100_v3.json`,
  `reports/sm_occupancy_a100_v3.md`), duration_s=1.5:
  | target % | blocks | DCGM SMACT | DCGM blocks | err |
  |---|---|---|---|---|
  | 1   | 1   | 0.009 | 0.9   | −0.1 |
  | 10  | 11  | 0.100 | 10.8  | −0.2 |
  | 25  | 27  | 0.233 | 25.1  | −1.9 |
  | 50  | 54  | 0.464 | 50.2  | −3.8 |
  | 75  | 81  | 0.697 | 75.3  | −5.7 |
  | 100 | 108 | 0.930 | 100.4 | −7.6 |
  Three independent signals (hardware contract = 1; DCGM SMACT × 108;
  power-fit Δpower) all agree on the same physical event.
- DCGM under-counts by ~7% at full GPU because DCGM samples at 100 ms
  while kernel runs 1500 ms — the launch ramp + teardown samples
  (when not all blocks are active) are folded into the mean.
  Multiplying residual by 1500/(1500−200) ≈ 1.15 closes the gap.
  Longer kernels would converge to <1 SM; not run because the answer
  is already clear.
- Queued regime: at 200% (216 blocks, 2 shifts of 108 each), DCGM
  SMACT = 0.93 ⇒ measured 100 SMs ≈ 108. ✓ At 150% (162 blocks, phase 1
  = 108 SMs, phase 2 = 54 SMs), time-averaged expectation is 81 SMs;
  observed 75 SMs. ✓
- Cost: ~30 min on A100 SXM4 ≈ \$1.00. Instance terminated.
