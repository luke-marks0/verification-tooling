from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.helpers import read_json, run_cmd


class TestD3ReplicatedDispatch(unittest.TestCase):
    def test_dispatcher_mapping_is_stable(self) -> None:
        manifest = "tests/fixtures/positive/manifest.v1.example.json"
        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            out1 = tdir / "dispatch1.json"
            out2 = tdir / "dispatch2.json"
            replicas = "replica-0,replica-1,replica-2,replica-3"

            run_cmd(["python3", "modules/inference/runner/dispatcher.py", "--manifest", manifest, "--replicas", replicas, "--out", str(out1)])
            run_cmd(["python3", "modules/inference/runner/dispatcher.py", "--manifest", manifest, "--replicas", replicas, "--out", str(out2)])

            self.assertEqual(out1.read_bytes(), out2.read_bytes())
            mapping = read_json(out1)
            self.assertGreater(len(mapping), 0)


if __name__ == "__main__":
    unittest.main()
