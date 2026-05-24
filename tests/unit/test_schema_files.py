from __future__ import annotations

import json
import pathlib
import unittest

from modules.core.common.jsonschema_compat import DefaultValidator


class TestSchemaFiles(unittest.TestCase):
    def test_all_schema_files_are_valid_json_schema(self) -> None:
        schema_files = sorted(pathlib.Path("modules/core/schemas").glob("*.schema.json"))
        self.assertGreater(len(schema_files), 0)
        for schema_file in schema_files:
            with self.subTest(file=str(schema_file)):
                content = json.loads(schema_file.read_text(encoding="utf-8"))
                DefaultValidator.check_schema(content)


if __name__ == "__main__":
    unittest.main()
