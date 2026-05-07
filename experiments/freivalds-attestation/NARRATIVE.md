# The Lay of Freivalds-on-the-GPU

*An epic narrative of one stack, six accelerators, and a small handful of bugs.*

---

## Canto I — The Posing of the Question

In the year MMXXVI of our common reckoning, on the twenty-ninth day of the
fourth month, a question was put to me by my interlocutor, and the question
was this:

> *Let the verifier issue matmuls to the prover. Let the matrices be of
> varied types — fp4, fp16, integers all. And let Freivalds, with his
> nine-hundred-year-old algorithm, judge whether the deed was done.*

So I went forth into the working tree, and beheld the deterministic serving
stack already standing — its `cmd/server/main.py` with `/manifest`, `/run`,
`/replay`; its `pkg/manifest`, its schemas, its existing
`experiments/flop-attestation/`. I read the lay of the land. Then I asked
questions, one at a time, after the manner of those who would not assume.

**A?** *(compute attestation.)* **B?** *(experiment track.)* **A.**
*(verifier picks per call.)* **A.** *(tolerance.)* And lo, the design
assembled itself in five exchanges, with one moment of folly where I
proposed a two-round protocol and was rightly rebuked: *whatever works,*
said my interlocutor, *and explain the two-round thing, that's more than
what i asked for right.* And it was. And I retracted it.

There was further wisdom: the matmuls must **saturate** the GPU. While the
kernel runs, nothing else may run. The wall-clock should be tight to within
a few percent. This was inscribed into the plan as the soundness amplifier
the protocol would lean on.

---

## Canto II — The Forging of the Code

Into `pkg/freivalds/` I poured the spec, the deterministic PRNG, the
Freivalds check, the prover, the verifier. I built a stdlib backend so the
unit tests could run on a CPU-only dev box without torch, after the manner
of `pkg/flop_counter/`. Forty-two tests rose, and forty-two passed.

But not at once. For lo, the **first bug** appeared:

> *Honest int8 matmul rejected. `max_abs_diff = 9090560.`*

I had wrapped the intermediate vectors `Br`, `ABr`, `Cr` to the *input*
dtype rather than the *accumulator* dtype — and so int8 lost all its
precision in flight. One line, one fix, and the truth was restored. (I
made a note: the same trap will lurk on the GPU, and reductions must stay
in fp32 even when inputs are bf16.)

Schemas canonicalised. CI gate green. Smoke script: honest passes, zeros
rejected, single-byte tamper caught. Phase 1 complete.

---

## Canto III — The First Voyage to the Cloud

Then came the command: *get a h100 and launch.* And I went to Lambda.

H100 SXM5: NONE. H100 PCIe: NONE. GH200: ONE, in `us-east-3`, at $2.29/hr.
I launched it with the key named `macbook 2025` — the only key whose name
I recognised — and was promptly denied:

> *Permission denied (publickey).*

For my local public key, named `d6-rollout`, was not the same as
`macbook 2025`. I terminated the instance, accepted the small loss, and
tried to relaunch — only to find:

> *insufficient-capacity. Not enough capacity to fulfill launch request.*

The capacity had vanished in the seconds between my termination and my
retry. I set a polling loop in the background and waited. After some
minutes, capacity returned. The GH200 was mine.

Then SSH refused me:
> *Host key for 192.222.50.172 has changed.*

(For the IP had been recycled, and `known_hosts` remembered the prior
tenant.) `ssh-keygen -R`, and the gate opened. `nvidia-smi` reported:
**NVIDIA GH200 480GB, sm_90, torch 2.7.0**. We were in.

---

## Canto IV — Three Bugs in the Engine Room

The torch backend went forth. Its first matvec failed:

> *RuntimeError: "addmv_impl_cuda" not implemented for 'Long'.*

For CUDA hath no int64 GEMM. The verifier's int matvec was routed to the
CPU; it is O(n²), the cost is nothing.

The calibration began. Eleven minutes passed with no output. I peered at
the remote — the process was alive, pegged at 99% CPU. I killed it,
restarted with `python3 -u`, and watched the log.

> *bf16 dim=4096: median=6626 ms.*

**Six and a half seconds for a matmul that should take eight milliseconds.**
The matmul itself was fast. The bottleneck was elsewhere — and I traced it
to the bit-twiddle in `prng.py`, which iterated 67 million elements in a
Python `for`-loop, performing one `struct.unpack_from` per element. I
vectorised it via numpy when present, and the slow path remained as the
spec.

A third trap: I was timing the whole `execute_challenge`, which included
the now-fast-but-still-noticeable PRNG, the host-to-device upload, the
matmul, the device-to-host download, the base64. I switched to
`MatmulResult.wall_time_ms` — the cuda-synced t1−t0 around the matmul
itself — and the TF/s numbers became truthful.

---

## Canto V — Saturation Demonstrated

Calibration ran clean: 16 cells, IQR ≤ 1% at dim ≥ 4096 across all dtypes.
Per-trial bf16 read 5% of peak — but this was the cost of fresh tensors
per call, the cuBLAS heuristics waking up each trial. To prove the GPU
could do what it was meant to do, I wrote a saturation probe: warm cuBLAS,
hold A and B, run 50 matmuls in a tight loop, sample NVML at 5 ms.

> *bf16 dim=4096: 820 TF/s = **83% of peak**, sm_util_max = **100%**, 225 W.*
>
> *fp32 dim=8192: 51 TF/s = 77% peak, **580 W**, near-TDP.*

The plan's claim was verified: when the kernel ran, the GPU was wholly
given to it.

The adversarial probes followed. S0 honest: PASS. S1 cached, S2 zeros,
S3 random, S4 dropped-rows, S6 stub-kernel: all FAIL, with `max_abs_diff`
orders of magnitude beyond the calibrated honest noise. **S5 quantization-
cheat passed at v1's loose ε** — that was the documented precision/recall
knob, not a flaw. Honest matmul: 113.9 ms. Adversarial: 0.7–2.9 ms.
**A 30–100× timing gap**, awaiting the v2 gate.

I terminated the GH200 and slept (metaphorically).

---

## Canto VI — The Multi-GPU Pilgrimage

But my interlocutor was not satisfied with one GPU. *Try it on more
types,* came the new instruction, *and make sure the utilization is maxxed.*

Lambda had only A100 SXM4 and A10. Both were launched, in parallel, in
different regions. Both confirmed saturation:

- **A100 SXM4** at bf16 4096³: **267 TF/s = 86% peak, sm_util = 100%.**
- **A10** at bf16 8192³: 78 TF/s = 62% peak, sm_util = 100%, **147 W of
  150 W TDP** — power-pegged, clock auto-throttling 1650 → 1110 MHz under
  sustained load. The cleanest possible "saturated" signature.

H100/GH200 capacity refused to return after fifteen minutes. So I went to
vast.ai for diversity, with a stock pytorch image rather than the project's
Nix one (whose SM_90-only restriction would have barred us from the
consumer cards). H200, L40S, RTX 4090 in parallel. Each ran the standalone
probe; each yielded its truth.

Then two more bugs revealed themselves in the table-building. The peak-
TFLOPS lookup matched substrings, and `"L40"` greedily matched `"L40S"`
first, halving the divisor. (The L40S read **206% of peak**, which would
have been a triumph had it been real.) I sorted the keys longest-first and
the truth re-emerged. And one more: I had mixed dense and sparsity numbers
in my peak table — RTX 4090 fp16 was 330 *with sparsity*, but torch uses
dense, and so the % peak was being halved. Dense numbers everywhere now.

---

## Epilogue — The Six and the Sober Reckoning

Six GPUs across four SM generations. The bands of honest-prover throughput
at bf16 do not overlap:

| Class | TF/s |
|---|---|
| H100 / H200 / GH200 | **720–820** |
| A100 SXM4 | 265–275 |
| L40S | 195–250 |
| RTX 4090 | 159–169 |
| A10 | 70–80 |

A verifier need only know what hardware was claimed. Throughput in another
GPU's band is impossible without delegation, and delegation is the cheat
the v2 timing gate will catch.

There are caveats, set down honestly. `torch._int_mm` reads 5–25% of int8
peak across every GPU because it dispatches to a generic IMMA path, not
the optimised cuBLASLt int8 GEMM. SMs still reach 100% utilisation at
dim ≥ 8192, so attestation works — but the % peak number is misleading
until v2 calls cuBLASLt directly. Consumer NVML on RTX 4090 reads sm_util
at low resolution (3% during a clearly-active 361 W workload); but power
is continuously polled and tells the truth.

Total GPU spend, end to end: under five dollars across two providers,
three regions, six instance types. All instances terminated. The codebase
is forty-four green tests, twelve schemas, four reports, six saturation
datasets, two probe matrices, and one calibration ledger heavier than it
was at dawn.

Phase 4 — promoting `POST /matmul-challenge` to `cmd/server/main.py`,
alongside `/run` and `/replay` — remains. But that is a different lay,
for a different day.

*Here ends the lay.*
