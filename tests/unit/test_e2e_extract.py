"""Unit tests for modules.attestation.e2e.extract."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.attestation.e2e.extract import (
    TokenIdExtractionError,
    extract_input_token_ids,
    extract_output_token_ids,
)


def _resp(**kw):
    base = {"choices": [{"index": 0, "message": {"role": "assistant", "content": ""}}]}
    base.update(kw)
    return base


class TestExtractOutputTokenIds(unittest.TestCase):

    def test_reads_top_level_token_ids(self):
        self.assertEqual(extract_output_token_ids(_resp(token_ids=[1, 2, 3])), [1, 2, 3])

    def test_reads_nested_token_ids(self):
        r = _resp()
        r["choices"][0]["token_ids"] = [9, 8]
        self.assertEqual(extract_output_token_ids(r), [9, 8])

    def test_raises_when_absent(self):
        with self.assertRaises(TokenIdExtractionError):
            extract_output_token_ids(_resp())


class TestExtractInputTokenIds(unittest.TestCase):

    def test_reads_top_level_prompt_token_ids(self):
        self.assertEqual(
            extract_input_token_ids(_resp(prompt_token_ids=[5, 6, 7])),
            [5, 6, 7],
        )

    def test_reads_nested_prompt_token_ids(self):
        r = _resp()
        r["choices"][0]["prompt_token_ids"] = [100, 200]
        self.assertEqual(extract_input_token_ids(r), [100, 200])

    def test_raises_when_absent(self):
        with self.assertRaises(TokenIdExtractionError):
            extract_input_token_ids(_resp())

    def test_empty_list_is_preserved(self):
        """An empty prompt_token_ids list is still a list, not 'missing'."""
        self.assertEqual(extract_input_token_ids(_resp(prompt_token_ids=[])), [])


if __name__ == "__main__":
    unittest.main()
