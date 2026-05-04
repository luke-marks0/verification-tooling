"""Benign inference workload.

Two-tier:
  - use_vllm=False  → CPU-friendly synthetic frames; runs anywhere.
  - use_vllm=True   → real vLLM inference; gated on a GPU + vLLM at the
                       call site (test gates with _has_gpu()/_has_vllm()).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from workloads.context import WorkloadContext

_FRAMES_PER_PROMPT = 10
_FRAME_BYTES = 256


def _synth_frames_for_prompt(prompt: str, seed: int) -> list[bytes]:
    out: list[bytes] = []
    base = hashlib.sha256(f"benign|{seed}|{prompt}".encode()).digest()
    for i in range(_FRAMES_PER_PROMPT):
        chunk = hashlib.sha256(base + i.to_bytes(8, "big")).digest()
        # Tile to _FRAME_BYTES.
        frame = bytearray()
        while len(frame) < _FRAME_BYTES:
            frame.extend(chunk)
        out.append(bytes(frame[:_FRAME_BYTES]))
    return out


@dataclass
class BenignInferenceWorkload:
    prompts: list[str]
    use_vllm: bool = False
    seed: int = 0
    pod_id: str = "pod-a"

    def run(self, ctx: WorkloadContext) -> None:
        if self.use_vllm:
            return self._run_vllm(ctx)
        return self._run_synthetic(ctx)

    def _run_synthetic(self, ctx: WorkloadContext) -> None:
        for i, prompt in enumerate(self.prompts):
            if ctx.stop_event.is_set():
                return
            frames = _synth_frames_for_prompt(prompt, self.seed)
            for f in frames:
                ctx.publish_frame(f)
            # One task record per prompt. Synthetic FLOPs ~= bytes-on-wire
            # — stand-in for the eventual real inference accounting.
            ctx.record_task(
                {
                    "task_id": f"benign-{self.seed}-{i:04d}",
                    "pod_id": self.pod_id,
                    "operation": "inference",
                    "claimed_flops": _FRAMES_PER_PROMPT * _FRAME_BYTES,
                }
            )

    def _run_vllm(self, ctx: WorkloadContext) -> None:  # pragma: no cover (GPU-only)
        # Lazy-import vLLM only on the GPU path; tests that don't have
        # vllm installed must NOT pay the import cost when use_vllm=False.
        import vllm  # noqa: F401

        # The real path lives behind a _has_gpu() / _has_vllm() gate at the
        # call site. We don't try to provide a CPU fallback for vLLM here;
        # it doesn't exist. If you reached this branch on CPU, you misconfigured.
        raise NotImplementedError(
            "vLLM workload path: implement when GPU is available; gated by tests."
        )
