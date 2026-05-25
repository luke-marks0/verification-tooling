from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.helpers import read_json, run_cmd


class TestLongRunDeterminism(unittest.TestCase):
    def test_thirty_repeated_runs_have_identical_observable_digests(self) -> None:
        manifest = "tests/fixtures/positive/manifest.v1.example.json"
        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            lock_resolved = tdir / "resolved.lock.json"
            lock_built = tdir / "built.lock.json"
            run_cmd(["python3", "modules/inference/resolver/main.py", "--manifest", manifest, "--lockfile-out", str(lock_resolved)])
            run_cmd(["python3", "modules/build/builder/main.py", "--lockfile", str(lock_resolved), "--lockfile-out", str(lock_built)])

            digest_sets = []
            for idx in range(30):
                out = tdir / f"run-{idx:02d}"
                run_cmd(["python3", "modules/inference/runner/main.py", "--mode", "mock", "--manifest", manifest, "--lockfile", str(lock_built), "--out-dir", str(out)])
                bundle = read_json(out / "run_bundle.v1.json")
                digest_sets.append(
                    {
                        "tokens": bundle["observables"]["tokens"]["digest"],
                        "logits": bundle["observables"]["logits"]["digest"],
                    }
                )

            first = digest_sets[0]
            for item in digest_sets[1:]:
                self.assertEqual(first, item)


if __name__ == "__main__":
    unittest.main()
