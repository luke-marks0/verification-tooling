from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from tests.proverdet._helpers import (
    REPO_ROOT,
    http_post_json,
    read_bound_port,
    sandbox_env,
)


def _expected_concatenation(seed: int, count: int, size_bytes: int) -> bytes:
    out = bytearray()
    for i in range(count):
        seed_bytes = seed.to_bytes(8, "big", signed=False)
        idx_bytes = i.to_bytes(8, "big", signed=False)
        digest = hashlib.sha256(seed_bytes + idx_bytes).digest()
        frame = bytearray()
        while len(frame) < size_bytes:
            frame.extend(digest)
        out.extend(frame[:size_bytes])
    return bytes(out)


class TestTrafficE2E(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.prover_dir = Path(self.tmp.name) / "prover"
        self.verifier_dir = Path(self.tmp.name) / "verifier"
        self.prover_dir.mkdir()
        self.verifier_dir.mkdir()

        verifier_port_file = self.verifier_dir / "port"
        prover_port_file = self.prover_dir / "port"

        # Start verifier first; prover needs its URL.
        self.verifier = subprocess.Popen(
            [
                sys.executable,
                "modules/attestation/verifier_server/main.py",
                "--host",
                "127.0.0.1",
                "--port",
                "0",
                "--port-file",
                str(verifier_port_file),
                "--out-dir",
                str(self.verifier_dir),
                "--prover-base-url",
                "http://127.0.0.1:1",  # circular; we set --no-scheduler
                "--no-scheduler",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(REPO_ROOT),
            env=sandbox_env(),
        )
        try:
            self.verifier_port = read_bound_port(verifier_port_file, timeout_s=10.0)
        except Exception:
            self.verifier.terminate()
            self.fail("verifier never bound")

        verifier_url = f"http://127.0.0.1:{self.verifier_port}"

        self.prover = subprocess.Popen(
            [
                sys.executable,
                "modules/attestation/prover/main.py",
                "--host",
                "127.0.0.1",
                "--port",
                "0",
                "--port-file",
                str(prover_port_file),
                "--run-id",
                "e2e-traffic",
                "--out-dir",
                str(self.prover_dir),
                "--verifier-url",
                verifier_url,
                "--debug-mode",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(REPO_ROOT),
            env=sandbox_env(),
        )
        try:
            self.prover_port = read_bound_port(prover_port_file, timeout_s=10.0)
        except Exception:
            self.prover.terminate()
            self.verifier.terminate()
            self.fail("prover never bound")

    def tearDown(self) -> None:
        for p in (self.prover, self.verifier):
            if p is not None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(timeout=5)
        self.tmp.cleanup()

    def test_synthetic_frames_round_trip_to_verifier_digest(self) -> None:
        seed = 42
        count = 16
        size_bytes = 256
        # Trigger publish.
        status, body = http_post_json(
            f"http://127.0.0.1:{self.prover_port}/debug/emit-frames",
            {"seed": seed, "count": count, "size_bytes": size_bytes},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["published"], count)

        # Wait briefly for the publisher to drain (max_batch_bytes default
        # is 64 KiB > 16*256, so this only flushes on the timer or stop).
        time.sleep(0.5)

        # Finalize.
        status, fbody = http_post_json(
            f"http://127.0.0.1:{self.verifier_port}/traffic/finalize", {}
        )
        self.assertEqual(status, 200)

        # The publisher only flushes when the buffer fills or the timer
        # ticks. We give it up to 2s to land everything.
        expected_size = count * size_bytes
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if fbody["size_bytes"] >= expected_size:
                break
            time.sleep(0.1)
            _, fbody = http_post_json(f"http://127.0.0.1:{self.verifier_port}/traffic/finalize", {})
        self.assertEqual(fbody["size_bytes"], expected_size)

        expected = _expected_concatenation(seed, count, size_bytes)
        expected_digest = "sha256:" + hashlib.sha256(expected).hexdigest()
        self.assertEqual(fbody["digest"], expected_digest)

        # And the on-disk file matches.
        bin_path = self.verifier_dir / "traffic.bin"
        self.assertEqual(bin_path.read_bytes(), expected)


if __name__ == "__main__":
    unittest.main()
