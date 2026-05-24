import pathlib
import unittest


class TestRepoLayout(unittest.TestCase):
    def test_required_directories_exist(self) -> None:
        required = [
            "modules/inference/resolver",
            "modules/build/builder",
            "modules/inference/runner",
            "modules/inference/server",
            "modules/attestation/verifier",
            "modules/inference/manifest",
            "modules/inference/manifests",
            "modules/network/networkdet",
            "modules/core/common",
            "modules/core/schemas",
            "tests/unit",
            "tests/integration",
            "tests/e2e",
            "tests/determinism",
            "tests/fixtures",
        ]
        for rel in required:
            with self.subTest(path=rel):
                self.assertTrue(pathlib.Path(rel).is_dir(), f"Missing directory: {rel}")


if __name__ == "__main__":
    unittest.main()
