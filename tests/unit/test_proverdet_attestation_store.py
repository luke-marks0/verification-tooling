from __future__ import annotations

import threading
import unittest

from pkg.proverdet.attestation_store import AttestationStore


class TestAttestationStore(unittest.TestCase):
    def test_put_then_get_round_trips(self) -> None:
        store = AttestationStore()
        body = {"matmul_id": "m-0", "answer": 42}
        store.put("att-1", body)
        self.assertEqual(store.get("att-1"), body)

    def test_get_unknown_returns_none(self) -> None:
        store = AttestationStore()
        self.assertIsNone(store.get("nope"))

    def test_put_overwrites(self) -> None:
        store = AttestationStore()
        store.put("a", {"v": 1})
        store.put("a", {"v": 2})
        self.assertEqual(store.get("a"), {"v": 2})

    def test_put_is_threadsafe(self) -> None:
        store = AttestationStore()
        n_threads = 8
        per_thread = 64

        def worker(tid: int) -> None:
            for i in range(per_thread):
                store.put(f"t{tid}-{i}", {"tid": tid, "i": i})

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for tid in range(n_threads):
            for i in range(per_thread):
                self.assertEqual(store.get(f"t{tid}-{i}"), {"tid": tid, "i": i})


if __name__ == "__main__":
    unittest.main()
