"""D4-BOI: Batch & Order Invariance under Tensor Parallelism.

Proves that TP inference outputs are identical regardless of:
  1. Request ordering (shuffled vs original)
  2. Batch size (max_num_seqs changed)

For each model, we run 100 requests twice — once in order with batch=64,
once shuffled with batch=16 — and compare outputs matched by request ID.
"""
from __future__ import annotations

import json
import os
import random
import tempfile
import unittest
from pathlib import Path

from tests.helpers import read_json, run_cmd

TOPICS = [
    "Explain how {} works in detail:",
    "Write a technical overview of {}:",
    "Describe the history and evolution of {}:",
    "Compare and contrast {} with alternatives:",
    "What are the key challenges in {}?",
]
SUBJECTS = [
    "TCP/IP networking", "quantum computing", "neural network backpropagation",
    "RSA encryption", "garbage collection in programming languages",
    "distributed consensus algorithms", "compiler optimization passes",
    "GPU shader programming", "database indexing strategies",
    "operating system virtual memory", "HTTP/2 protocol multiplexing",
    "Transformer attention mechanisms", "elliptic curve cryptography",
    "Linux kernel scheduling", "WebAssembly runtime design",
    "RAID storage architectures", "DNS resolution process",
    "TLS 1.3 handshake protocol", "MapReduce computation model",
    "binary search tree balancing",
]


def _generate_requests(n: int = 100) -> list[dict]:
    requests = []
    for i in range(n):
        prompt = TOPICS[i % len(TOPICS)].format(SUBJECTS[i % len(SUBJECTS)])
        requests.append({
            "id": f"req-{i:03d}",
            "prompt": prompt,
            "max_new_tokens": 128,
            "temperature": 0,
        })
    return requests


def _make_manifest_pair(
    base_manifest_path: str, tag: str, out_dir: Path
) -> tuple[Path, Path]:
    """Create two manifests: ordered (batch=64) and shuffled (batch=16)."""
    base = json.loads(Path(base_manifest_path).read_text())
    requests = _generate_requests(100)

    # A: ordered, batch=64
    manifest_a = json.loads(json.dumps(base))
    manifest_a["run_id"] = f"{tag}-boi-ordered"
    manifest_a["requests"] = requests
    manifest_a["runtime"]["serving_engine"]["max_num_seqs"] = 64
    a_path = out_dir / "manifest_a.json"
    a_path.write_text(
        json.dumps(manifest_a, sort_keys=True, separators=(",", ":")) + "\n"
    )

    # B: shuffled, batch=16
    rng = random.Random(12345)
    shuffled = list(requests)
    rng.shuffle(shuffled)
    manifest_b = json.loads(json.dumps(base))
    manifest_b["run_id"] = f"{tag}-boi-shuffled"
    manifest_b["requests"] = shuffled
    manifest_b["runtime"]["serving_engine"]["max_num_seqs"] = 16
    b_path = out_dir / "manifest_b.json"
    b_path.write_text(
        json.dumps(manifest_b, sort_keys=True, separators=(",", ":")) + "\n"
    )

    return a_path, b_path


def _run_pipeline(manifest: str, out_dir: Path) -> Path:
    """resolve -> build -> run(vllm), return run dir."""
    lock_resolved = out_dir / "resolved.lock.json"
    lock_built = out_dir / "built.lock.json"
    run_dir = out_dir / "run"

    run_cmd(["python3", "modules/inference/resolver/main.py",
             "--manifest", str(manifest),
             "--lockfile-out", str(lock_resolved)])
    run_cmd(["python3", "modules/build/builder/main.py",
             "--lockfile", str(lock_resolved),
             "--lockfile-out", str(lock_built)])
    run_cmd(["python3", "modules/inference/runner/main.py",
             "--manifest", str(manifest),
             "--lockfile", str(lock_built),
             "--out-dir", str(run_dir),
             "--mode", "vllm",
             "--replica-id", "replica-0"])
    return run_dir


def _compare_by_request_id(run_a: Path, run_b: Path) -> dict:
    """Compare outputs matched by request ID, not position."""
    tokens_a = read_json(run_a / "observables" / "tokens.json")
    tokens_b = read_json(run_b / "observables" / "tokens.json")

    a_by_id = {r["id"]: r["tokens"] for r in tokens_a}
    b_by_id = {r["id"]: r["tokens"] for r in tokens_b}

    assert set(a_by_id.keys()) == set(b_by_id.keys()), "Request ID sets differ"

    matches = 0
    mismatches = []
    total_tokens = 0

    for rid in sorted(a_by_id.keys()):
        ta, tb = a_by_id[rid], b_by_id[rid]
        total_tokens += len(ta)
        if ta == tb:
            matches += 1
        else:
            first_diff = next(
                (i for i, (x, y) in enumerate(zip(ta, tb)) if x != y),
                min(len(ta), len(tb)),
            )
            mismatches.append({"id": rid, "pos": first_diff})

    return {
        "matches": matches,
        "mismatches": mismatches,
        "total_tokens": total_tokens,
    }


@unittest.skipUnless(
    os.getenv("RUNNER_TP_TEST", "").lower() in ("1", "true"),
    "TP batch/order invariance test requires multi-GPU and RUNNER_TP_TEST=1",
)
class TestD4BatchOrderInvariance(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import torch
            self.gpu_count = torch.cuda.device_count()
        except Exception:
            self.gpu_count = 0
        if self.gpu_count < 4:
            self.skipTest(f"Need >= 4 GPUs, found {self.gpu_count}")

        os.environ["NCCL_ALGO"] = "Ring"
        os.environ["NCCL_PROTO"] = "Simple"
        os.environ["NCCL_DEBUG"] = "WARN"

    def _run_invariance_test(self, manifest_path: str, tag: str) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            a_manifest, b_manifest = _make_manifest_pair(manifest_path, tag, tdir)

            run_a = _run_pipeline(a_manifest, tdir / "a")
            run_b = _run_pipeline(b_manifest, tdir / "b")

            result = _compare_by_request_id(run_a, run_b)
            self.assertEqual(
                len(result["mismatches"]), 0,
                f"{tag}: {len(result['mismatches'])} requests differ. "
                f"First: {result['mismatches'][:3] if result['mismatches'] else 'N/A'}",
            )

    def test_dense_batch_order_invariance(self) -> None:
        self._run_invariance_test(
            "modules/inference/manifests/qwen2.5-32b-tp4.manifest.json", "qwen2.5-32b-dense"
        )

    def test_moe_batch_order_invariance(self) -> None:
        self._run_invariance_test(
            "modules/inference/manifests/qwen3-30b-moe-tp4.manifest.json", "qwen3-30b-moe"
        )


if __name__ == "__main__":
    unittest.main()
