from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.helpers import read_json, run_cmd


class TestD1BuilderRuntimeDigest(unittest.TestCase):
    def test_builder_digest_is_deterministic(self) -> None:
        manifest = "tests/fixtures/positive/manifest.v1.example.json"
        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            resolved = tdir / "resolved.lock.json"
            built1 = tdir / "built1.lock.json"
            built2 = tdir / "built2.lock.json"

            run_cmd(["python3", "modules/inference/resolver/main.py", "--manifest", manifest, "--lockfile-out", str(resolved)])
            run_cmd(["python3", "modules/build/builder/main.py", "--lockfile", str(resolved), "--lockfile-out", str(built1)])
            run_cmd(["python3", "modules/build/builder/main.py", "--lockfile", str(resolved), "--lockfile-out", str(built2)])

            left = read_json(built1)
            right = read_json(built2)
            self.assertEqual(left["runtime_closure_digest"], right["runtime_closure_digest"])
            self.assertEqual(left["canonicalization"]["lockfile_digest"], right["canonicalization"]["lockfile_digest"])


if __name__ == "__main__":
    unittest.main()
