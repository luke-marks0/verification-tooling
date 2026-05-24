from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.helpers import read_json, run_cmd


class TestD0ManifestToLockfile(unittest.TestCase):
    def test_resolver_is_byte_deterministic(self) -> None:
        manifest = "tests/fixtures/positive/manifest.v1.example.json"
        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            lock1 = tdir / "lock1.json"
            lock2 = tdir / "lock2.json"

            run_cmd(["python3", "modules/inference/resolver/main.py", "--manifest", manifest, "--lockfile-out", str(lock1)])
            run_cmd(["python3", "modules/inference/resolver/main.py", "--manifest", manifest, "--lockfile-out", str(lock2)])

            self.assertEqual(lock1.read_bytes(), lock2.read_bytes())
            lock_data = read_json(lock1)
            self.assertIn("canonicalization", lock_data)


if __name__ == "__main__":
    unittest.main()
