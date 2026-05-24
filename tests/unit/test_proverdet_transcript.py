from __future__ import annotations

import json
import re
import tempfile
import threading
import unittest
from pathlib import Path

from modules.core.common.contracts import ValidationError
from modules.attestation.proverdet.transcript import TranscriptLog


class TestTranscriptLog(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "transcript.jsonl"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_appends_each_entry_with_monotonic_seq(self) -> None:
        log = TranscriptLog(self.path)
        log.record(direction="sent", endpoint="/graph", payload=b"{}")
        log.record(direction="received", endpoint="/graph", payload=b'{"x":1}', status_code=200)
        seqs = [json.loads(line)["seq"] for line in self.path.read_text().splitlines()]
        self.assertEqual(seqs, [1, 2])

    def test_records_payload_path_when_provided(self) -> None:
        log = TranscriptLog(self.path)
        log.record(
            direction="received",
            endpoint="/traffic",
            payload=b"hello",
            payload_path="traffic-1.bin",
        )
        line = json.loads(self.path.read_text().splitlines()[0])
        self.assertEqual(line["payload_path"], "traffic-1.bin")

    def test_validates_against_schema_on_append(self) -> None:
        log = TranscriptLog(self.path)
        # Direction "internal" is not in the enum — should raise before write.
        with self.assertRaises(ValidationError):
            log.record(direction="internal", endpoint="/graph", payload=b"")
        # File should still be empty.
        self.assertEqual(self.path.read_text(), "")

    def test_writes_canonical_json_lines(self) -> None:
        log = TranscriptLog(self.path)
        log.record(direction="sent", endpoint="/graph", payload=b"{}", status_code=200)
        line = self.path.read_text().splitlines()[0]
        self.assertNotIn(", ", line)
        self.assertNotIn(": ", line)
        keys = re.findall(r'"([a-z_]+)":', line)
        self.assertEqual(keys, sorted(keys))

    def test_threadsafe_under_concurrent_appends(self) -> None:
        log = TranscriptLog(self.path)

        def worker(start: int) -> None:
            for i in range(100):
                log.record(direction="sent", endpoint="/graph", payload=f"{start}-{i}".encode())

        threads = [threading.Thread(target=worker, args=(k,)) for k in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        lines = [json.loads(line) for line in self.path.read_text().splitlines()]
        self.assertEqual(len(lines), 10 * 100)
        seqs = sorted(line["seq"] for line in lines)
        self.assertEqual(seqs, list(range(1, 10 * 100 + 1)))


if __name__ == "__main__":
    unittest.main()
