from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pkg.proverdet.verdict import emit_verdict


class TestVerdictStub(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.transcript = Path(self.tmp.name) / "transcript.jsonl"
        self.transcript.write_text("", encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_empty_transcript_yields_inference(self) -> None:
        # No replay/verdict entries, no summaries, no traffic digest — every
        # signal returns passed, so the binary combiner emits inference.
        result = emit_verdict(self.transcript)
        self.assertEqual(result["verdict"], "inference")
        self.assertEqual(result["reasons"], [])

    def test_transcript_without_failing_signals_yields_inference(self) -> None:
        self.transcript.write_text(
            '{"seq":1,"direction":"sent","endpoint":"/graph",'
            '"timestamp":"2026-05-04T12:00:00Z",'
            '"payload_digest":"sha256:' + "0" * 64 + '"}\n',
            encoding="utf-8",
        )
        result = emit_verdict(self.transcript)
        self.assertEqual(result["verdict"], "inference")


class TestVerdictCli(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.transcript = Path(self.tmp.name) / "transcript.jsonl"
        self.transcript.write_text("", encoding="utf-8")
        self.out = Path(self.tmp.name) / "verdict.json"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_cli_writes_verdict_json(self) -> None:
        import subprocess
        import sys

        from tests.proverdet._helpers import REPO_ROOT, sandbox_env

        rc = subprocess.run(
            [
                sys.executable,
                "cmd/verifier_cli/main.py",
                "--transcript",
                str(self.transcript),
                "--out",
                str(self.out),
            ],
            cwd=str(REPO_ROOT),
            env=sandbox_env(),
            capture_output=True,
        )
        self.assertEqual(rc.returncode, 0, rc.stderr.decode())
        self.assertTrue(self.out.exists())
        loaded = json.loads(self.out.read_text(encoding="utf-8"))
        self.assertEqual(loaded["verdict"], "inference")
        self.assertEqual(loaded["reasons"], [])

    def test_cli_output_matches_function(self) -> None:
        import subprocess
        import sys

        from tests.proverdet._helpers import REPO_ROOT, sandbox_env

        subprocess.run(
            [
                sys.executable,
                "cmd/verifier_cli/main.py",
                "--transcript",
                str(self.transcript),
                "--out",
                str(self.out),
            ],
            cwd=str(REPO_ROOT),
            env=sandbox_env(),
            check=True,
        )
        cli_result = json.loads(self.out.read_text(encoding="utf-8"))
        fn_result = emit_verdict(self.transcript)
        self.assertEqual(cli_result, fn_result)


if __name__ == "__main__":
    unittest.main()
