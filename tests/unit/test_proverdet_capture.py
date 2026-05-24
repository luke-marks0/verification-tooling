from __future__ import annotations

import json
import re
import tempfile
import threading
import unittest
from pathlib import Path

from modules.attestation.proverdet.capture import ProverCaptureLog


class TestProverCaptureLog(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "capture.jsonl"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_appends_each_entry_with_monotonic_seq(self) -> None:
        log = ProverCaptureLog(self.path)
        log.record(direction="received", endpoint="/replay", payload=b'{"a":1}')
        log.record(direction="sent", endpoint="/replay", payload=b'{"ok": true}', status_code=200)
        log.record(direction="received", endpoint="/graph", payload=b"")
        lines = self.path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 3)
        seqs = [json.loads(line)["seq"] for line in lines]
        self.assertEqual(seqs, [1, 2, 3])

    def test_payload_digest_is_sha256_prefixed(self) -> None:
        log = ProverCaptureLog(self.path)
        log.record(direction="received", endpoint="/replay", payload=b"hi")
        line = self.path.read_text(encoding="utf-8").splitlines()[0]
        d = json.loads(line)["payload_digest"]
        self.assertRegex(d, r"^sha256:[0-9a-f]{64}$")

    def test_writes_canonical_json_lines(self) -> None:
        log = ProverCaptureLog(self.path)
        log.record(direction="sent", endpoint="/graph", payload=b"{}", status_code=200)
        line = self.path.read_text(encoding="utf-8").splitlines()[0]
        # Canonical JSON: keys sorted, no spaces between separators.
        self.assertNotIn(", ", line)
        self.assertNotIn(": ", line)
        # Sorted keys: "direction" < "endpoint" < "payload_digest" < "seq" < ...
        keys = [m for m in re.findall(r'"([a-z_]+)":', line)]
        self.assertEqual(keys, sorted(keys))

    def test_records_status_code_only_when_provided(self) -> None:
        log = ProverCaptureLog(self.path)
        log.record(direction="received", endpoint="/replay", payload=b"")
        log.record(direction="sent", endpoint="/replay", payload=b"{}", status_code=200)
        lines = [json.loads(line) for line in self.path.read_text().splitlines()]
        self.assertNotIn("status_code", lines[0])
        self.assertEqual(lines[1]["status_code"], 200)

    def test_records_payload_path_when_provided(self) -> None:
        log = ProverCaptureLog(self.path)
        log.record(
            direction="received",
            endpoint="/replay",
            payload=b"hi",
            payload_path="payloads/r-1.json",
        )
        line = json.loads(self.path.read_text().splitlines()[0])
        self.assertEqual(line["payload_path"], "payloads/r-1.json")

    def test_threadsafe_under_concurrent_appends(self) -> None:
        log = ProverCaptureLog(self.path)

        def worker(start: int) -> None:
            for i in range(50):
                log.record(
                    direction="sent",
                    endpoint="/graph",
                    payload=f"{start}-{i}".encode(),
                )

        threads = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        lines = [json.loads(line) for line in self.path.read_text().splitlines()]
        self.assertEqual(len(lines), 4 * 50)
        seqs = sorted(line["seq"] for line in lines)
        self.assertEqual(seqs, list(range(1, 4 * 50 + 1)))


if __name__ == "__main__":
    unittest.main()
