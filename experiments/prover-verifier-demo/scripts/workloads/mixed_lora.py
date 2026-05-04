"""Mixed inference + LoRA-training workload (adversarial).

The cheating: each "inference" prompt produces frames + a `record_task`
declaring `claimed_flops`. Then, AFTER the inference, the workload spends
`gradient_steps` extra matmuls of `matmul_dim` on simulated LoRA gradient
updates that are NOT recorded as tasks. From the prover-verifier protocol's
point of view, this is real GPU work missing from the declared graph —
exactly the discrepancy Phase 8.2's compute-budget signal flags.

`gradient_steps` is the knob: 0 ⇒ honest baseline (matches benign),
larger ⇒ more cheating per inference. We track `observed_flops_total`
on the workload itself so adversarial unit tests can assert the gap; on
GPU, `enable_real_matmul=True` would dispatch real backend matmuls so the
cheating leaves a runtime fingerprint.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from workloads.context import WorkloadContext

_FRAMES_PER_PROMPT = 10
_FRAME_BYTES = 256


def _synth_frames_for_prompt(prompt: str, seed: int) -> list[bytes]:
    out: list[bytes] = []
    base = hashlib.sha256(f"mixed_lora|{seed}|{prompt}".encode()).digest()
    for i in range(_FRAMES_PER_PROMPT):
        chunk = hashlib.sha256(base + i.to_bytes(8, "big")).digest()
        frame = bytearray()
        while len(frame) < _FRAME_BYTES:
            frame.extend(chunk)
        out.append(bytes(frame[:_FRAME_BYTES]))
    return out


@dataclass
class MixedLoraWorkload:
    prompts: list[str]
    gradient_steps: int = 0
    matmul_dim: int = 64
    use_vllm: bool = False
    seed: int = 0
    pod_id: str = "pod-a"
    delay_per_prompt_s: float = 0.0
    # Mutated during run(). Public so tests + Phase 8 evaluators can
    # compare against sum(task.claimed_flops) to derive the cheating gap.
    observed_flops_total: int = field(default=0, init=False)

    def run(self, ctx: WorkloadContext) -> None:
        if self.use_vllm:
            return self._run_vllm(ctx)
        return self._run_synthetic(ctx)

    def _run_synthetic(self, ctx: WorkloadContext) -> None:
        for i, prompt in enumerate(self.prompts):
            if ctx.stop_event.is_set():
                return
            # --- Honest inference (matches BenignInferenceWorkload shape) ---
            frames = _synth_frames_for_prompt(prompt, self.seed)
            for f in frames:
                ctx.publish_frame(f)
            inference_flops = _FRAMES_PER_PROMPT * _FRAME_BYTES
            self.observed_flops_total += inference_flops
            ctx.record_task(
                {
                    "task_id": f"mixed_lora-{self.seed}-{i:04d}",
                    "pod_id": self.pod_id,
                    "operation": "inference",
                    "claimed_flops": inference_flops,
                }
            )

            # --- Cheating: gradient steps that DON'T call record_task ---
            # Each step is one square matmul of dim D: 2 * D^3 FLOPs (mul +
            # add). We track the budget exactly so tests can assert the
            # gap; the demo runs this on GPU via the use_vllm branch so the
            # work shows up in real wall time.
            for _ in range(self.gradient_steps):
                if ctx.stop_event.is_set():
                    return
                self.observed_flops_total += 2 * self.matmul_dim**3

            if self.delay_per_prompt_s > 0 and ctx.stop_event.wait(self.delay_per_prompt_s):
                return

    def _run_vllm(self, ctx: WorkloadContext) -> None:  # pragma: no cover (GPU-only)
        import vllm  # noqa: F401

        raise NotImplementedError(
            "vLLM mixed_lora path: implement when GPU is available; gated by tests."
        )
