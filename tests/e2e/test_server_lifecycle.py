"""End-to-end server lifecycle tests.

Spins up a deterministic server, runs test cases, shuts it down.
Fully self-contained — no pre-existing server needed.

Usage on a GPU node:
    python3 -m pytest tests/e2e/test_server_lifecycle.py -v -s --timeout=600

Environment:
    RUNNER_MODEL_PATH: local model path (skips HF download if set)
    RUNNER_MAX_MODEL_LEN: max context (default: 4096)
    DETERMINISTIC_TEST_PORT: server port (default: 8100, avoids conflicting with running server)
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[2]
PORT = int(os.getenv("DETERMINISTIC_TEST_PORT", "8100"))
VLLM_PORT = PORT + 1
MAX_MODEL_LEN = os.getenv("RUNNER_MAX_MODEL_LEN", "4096")


def _has_gpu() -> bool:
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def _has_vllm() -> bool:
    try:
        r = subprocess.run(
            [sys.executable, "-c", "import vllm; print(vllm.__version__)"],
            capture_output=True, timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


def _wait_healthy(port: int, timeout: int = 300) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def _chat(port: int, prompt: str, max_tokens: int = 32, seed: int = 42) -> dict:
    body = json.dumps({
        "model": "Qwen/Qwen3-1.7B",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "seed": seed,
    }).encode()
    req = Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                  data=body, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


@unittest.skipUnless(_has_gpu() and _has_vllm(), "Requires GPU + vLLM")
class TestServerLifecycle(unittest.TestCase):
    """Spins up a server, runs determinism checks, tears it down."""

    server_proc: subprocess.Popen | None = None
    run_dir: str = ""
    manifest_path: str = ""
    lockfile_path: str = ""

    @classmethod
    def setUpClass(cls):
        cls.run_dir = tempfile.mkdtemp(prefix="det-test-")
        run_dir = Path(cls.run_dir)
        manifest_src = REPO_ROOT / "modules" / "inference" / "manifests" / "qwen3-1.7b.manifest.json"

        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT)
        env["RUNNER_MAX_MODEL_LEN"] = MAX_MODEL_LEN

        # Resolve
        lockfile = run_dir / "lockfile.v1.json"
        resolved_manifest = run_dir / "manifest.resolved.json"
        subprocess.run([
            sys.executable, str(REPO_ROOT / "modules/inference/resolver/main.py"),
            "--manifest", str(manifest_src),
            "--lockfile-out", str(lockfile),
            "--manifest-out", str(resolved_manifest),
            "--resolve-hf", "--hf-resolution-mode", "online",
        ], check=True, env=env, capture_output=True)

        # Build
        built_lockfile = run_dir / "lockfile.built.v1.json"
        subprocess.run([
            sys.executable, str(REPO_ROOT / "modules/build/builder/main.py"),
            "--lockfile", str(lockfile),
            "--lockfile-out", str(built_lockfile),
            "--builder-system", "equivalent",
        ], check=True, env=env, capture_output=True)

        cls.manifest_path = str(resolved_manifest)
        cls.lockfile_path = str(built_lockfile)

        # Start server — set batch invariance env before spawn
        env["VLLM_BATCH_INVARIANT"] = "1"
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        env["PYTHONHASHSEED"] = "0"
        # Use less GPU memory in case another server is already running
        env["RUNNER_GPU_MEM_UTIL"] = os.getenv("RUNNER_GPU_MEM_UTIL", "0.90")

        cls.server_proc = subprocess.Popen(
            [
                sys.executable, str(REPO_ROOT / "modules/inference/server/main.py"),
                "--manifest", cls.manifest_path,
                "--lockfile", cls.lockfile_path,
                "--out-dir", str(run_dir / "server"),
                "--host", "127.0.0.1",
                "--port", str(PORT),
                "--vllm-port", str(VLLM_PORT),
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        if not _wait_healthy(PORT):
            # Dump logs on failure
            cls.server_proc.kill()
            out, _ = cls.server_proc.communicate(timeout=5)
            raise RuntimeError(f"Server failed to start:\n{out.decode()[-2000:]}")

    @classmethod
    def tearDownClass(cls):
        if cls.server_proc:
            cls.server_proc.send_signal(signal.SIGTERM)
            try:
                cls.server_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                cls.server_proc.kill()
                cls.server_proc.wait()

    # ---- Test cases ----

    def test_01_health(self):
        """Server responds to health check."""
        with urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
            self.assertEqual(r.status, 200)

    def test_02_models(self):
        """Server lists the correct model."""
        with urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5) as r:
            data = json.loads(r.read())
        model_ids = [m["id"] for m in data["data"]]
        self.assertIn("Qwen/Qwen3-1.7B", model_ids)

    def test_03_single_request_determinism(self):
        """Same request twice → same content."""
        r1 = _chat(PORT, "What is 2+2?", max_tokens=16)
        r2 = _chat(PORT, "What is 2+2?", max_tokens=16)
        self.assertEqual(
            r1["choices"][0]["message"]["content"],
            r2["choices"][0]["message"]["content"],
        )

    def test_04_batch_determinism(self):
        """8 requests sent twice → all outputs identical."""
        import concurrent.futures

        prompts = [
            "What is the capital of France?",
            "Explain photosynthesis.",
            "What is 7*8?",
            "Name the largest planet.",
            "Who wrote Hamlet?",
            "What is H2O?",
            "What is the speed of light?",
            "Define entropy.",
        ]

        def run_batch():
            results = {}
            def do(i, p):
                return i, _chat(PORT, p, max_tokens=32)["choices"][0]["message"]["content"]
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                for fut in concurrent.futures.as_completed(
                    [pool.submit(do, i, p) for i, p in enumerate(prompts)]
                ):
                    i, c = fut.result()
                    results[i] = c
            return [results[i] for i in range(len(prompts))]

        batch1 = run_batch()
        batch2 = run_batch()

        for i, (a, b) in enumerate(zip(batch1, batch2)):
            self.assertEqual(a, b, f"Request {i} ('{prompts[i]}') differs")

    def test_05_egress_digest_determinism(self):
        """Batch egress digest matches across runs."""
        prompts = ["Hello", "World", "Test", "Determinism"]

        def run_and_digest():
            contents = []
            for p in prompts:
                r = _chat(PORT, p, max_tokens=16)
                # Strip nondeterministic fields
                resp = json.loads(json.dumps(r))
                resp.pop("id", None)
                resp.pop("created", None)
                resp.pop("system_fingerprint", None)
                canonical = json.dumps(resp, sort_keys=True, separators=(",", ":")).encode()
                contents.append(canonical)
            h = hashlib.sha256()
            for c in contents:
                h.update(hashlib.sha256(c).digest())
            return f"sha256:{h.hexdigest()}"

        d1 = run_and_digest()
        d2 = run_and_digest()
        self.assertEqual(d1, d2)

    def test_06_capture_log_written(self):
        """Capture proxy logs requests."""
        capture_path = Path(self.run_dir) / "server" / "capture.jsonl"
        # Send a request to ensure at least one entry
        _chat(PORT, "capture test", max_tokens=4)
        time.sleep(0.5)
        self.assertTrue(capture_path.exists(), "capture.jsonl not found")
        lines = [l for l in capture_path.read_text().splitlines() if l.strip()]
        self.assertGreater(len(lines), 0, "capture.jsonl is empty")

    def test_07_boot_record_exists(self):
        """Boot record written with hardware info."""
        boot_path = Path(self.run_dir) / "server" / "boot_record.json"
        self.assertTrue(boot_path.exists())
        boot = json.loads(boot_path.read_text())
        self.assertIn("hardware", boot)
        self.assertEqual(boot["hardware"]["status"], "conformant")
        self.assertIn("vllm_version", boot["hardware"])

    def test_08_long_output_determinism(self):
        """Longer generation (256 tokens) is deterministic."""
        prompt = "Write a detailed paragraph about the history of mathematics."
        r1 = _chat(PORT, prompt, max_tokens=256)
        r2 = _chat(PORT, prompt, max_tokens=256)
        self.assertEqual(
            r1["choices"][0]["message"]["content"],
            r2["choices"][0]["message"]["content"],
        )


if __name__ == "__main__":
    unittest.main()
