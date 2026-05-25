"""Tests to verify the critical bugs identified in docs/conformance/SPEC_REVIEW.md.

Each test demonstrates the bug by exercising the code path that would fail.
Tests are expected to FAIL (or raise) until the bugs are fixed.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from tests.helpers import read_json, write_json


class TestBug1RunnerSyntheticMode(unittest.TestCase):
    """Verify the runner produces a valid bundle in synthetic mode."""

    def test_run_in_synthetic_mode_produces_bundle(self) -> None:
        from tests.helpers import run_cmd

        manifest_path = "tests/fixtures/positive/manifest.v1.example.json"

        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            lockfile = tdir / "lock.json"
            built = tdir / "built.json"
            out = tdir / "run"

            run_cmd(["python3", "modules/inference/resolver/main.py",
                     "--manifest", manifest_path,
                     "--lockfile-out", str(lockfile)])
            run_cmd(["python3", "modules/build/builder/main.py",
                     "--lockfile", str(lockfile),
                     "--lockfile-out", str(built)])

            import subprocess
            result = subprocess.run(
                ["python3", "modules/inference/runner/main.py",
                 "--manifest", manifest_path,
                 "--lockfile", str(built),
                 "--out-dir", str(out)],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, f"Runner failed: {result.stderr[-300:]}")

            bundle = read_json(out / "run_bundle.v1.json")
            self.assertIn("observables", bundle)
            self.assertIn("tokens", bundle["observables"])
            self.assertIn("logits", bundle["observables"])


class TestBug3UlpComparisonWrong(unittest.TestCase):
    """Bug #3: ULP comparison uses fixed epsilon instead of magnitude-relative.

    A ULP (unit in the last place) depends on the magnitude of the value.
    Using tol = ulp * 1e-7 is only correct near 1.0. For large or small
    values, this gives incorrect results.
    """

    def test_ulp_comparison_near_zero(self) -> None:
        """Near zero, 1 ULP is ~1.4e-45 for float32. Fixed 1e-7 is way too large."""
        # Inline the comparison logic to avoid cmd import issues
        import importlib
        import sys
        # Load verifier module directly by path
        spec = importlib.util.spec_from_file_location(
            "verifier_main", "modules/attestation/verifier/main.py",
            submodule_search_locations=[],
        )
        mod = importlib.util.module_from_spec(spec)
        # Patch sys.modules to avoid cmd conflict
        old = sys.modules.get("cmd")
        sys.modules.pop("cmd", None)
        try:
            spec.loader.exec_module(mod)
        finally:
            if old is not None:
                sys.modules["cmd"] = old

        _compare_observable = mod._compare_observable

        # Two values that differ by much more than 1 ULP near zero,
        # but less than 1e-7
        baseline = [1e-10]
        candidate = [2e-10]
        comp = {"mode": "ulp", "ulp": 1}

        result = _compare_observable("ulp", baseline, candidate, comp)

        if result:
            self.fail(
                "Bug #3 confirmed: ULP comparison passed for values that differ "
                "by billions of ULPs near zero (1e-10 vs 2e-10 with ulp=1). "
                "The fixed epsilon 1e-7 is too permissive."
            )

    def test_ulp_comparison_large_values(self) -> None:
        """At large magnitude, values 1 ULP apart should pass with ulp=1."""
        import importlib, sys, struct
        spec = importlib.util.spec_from_file_location(
            "verifier_main", "modules/attestation/verifier/main.py",
            submodule_search_locations=[],
        )
        mod = importlib.util.module_from_spec(spec)
        old = sys.modules.get("cmd")
        sys.modules.pop("cmd", None)
        try:
            spec.loader.exec_module(mod)
        finally:
            if old is not None:
                sys.modules["cmd"] = old

        _compare_observable = mod._compare_observable

        # Construct two values exactly 1 ULP apart at large magnitude
        val = 1e10
        bits = struct.unpack(">q", struct.pack(">d", val))[0]
        next_val = struct.unpack(">d", struct.pack(">q", bits + 1))[0]

        baseline = [val]
        candidate = [next_val]
        comp = {"mode": "ulp", "ulp": 1}

        result = _compare_observable("ulp", baseline, candidate, comp)

        self.assertTrue(
            result,
            f"ULP comparison should pass for values 1 ULP apart "
            f"({val} vs {next_val}, diff={next_val - val})"
        )

        # And 2 ULPs apart should fail with ulp=1
        two_away = struct.unpack(">d", struct.pack(">q", bits + 2))[0]
        result2 = _compare_observable("ulp", [val], [two_away], comp)
        self.assertFalse(
            result2,
            f"ULP comparison should fail for values 2 ULPs apart with ulp=1"
        )


class TestBug4RelativeSchemaPath(unittest.TestCase):
    """Bug #4: SCHEMA_DIR uses relative path, breaks when cwd != repo root."""

    def test_schema_validation_from_different_cwd(self) -> None:
        from modules.core.common.contracts import validate_with_schema

        manifest = read_json(Path("tests/fixtures/positive/manifest.v1.example.json"))

        original_cwd = os.getcwd()
        try:
            os.chdir(tempfile.gettempdir())
            try:
                validate_with_schema("manifest.v1.schema.json", manifest)
            except FileNotFoundError:
                self.fail(
                    "Bug #4 confirmed: validate_with_schema fails when cwd "
                    f"is {os.getcwd()} instead of repo root. "
                    "SCHEMA_DIR is a relative path."
                )
            except Exception as e:
                if "No such file" in str(e) or "not found" in str(e).lower():
                    self.fail(f"Bug #4 confirmed: schema path resolution failed: {e}")
                raise
        finally:
            os.chdir(original_cwd)


class TestBug5OciImageCmd(unittest.TestCase):
    """Bug #5: modules/build/nix/images/runtime-image.nix hardcodes /app/modules/inference/server/main.py
    but the flake puts code under /nix/store/...-deterministic-serving-stack/.

    We can't test the actual Nix build here, but we can verify the paths
    are inconsistent between the two nix files.
    """

    def test_legacy_image_cmd_path_matches_flake(self) -> None:
        legacy_image = Path("modules/build/nix/images/runtime-image.nix").read_text()
        flake = Path("flake.nix").read_text()

        # The legacy image uses a hardcoded /app path
        legacy_uses_app = "/app/cmd/" in legacy_image
        # The flake uses ${appSrc}/cmd/ which resolves to /nix/store/...
        flake_uses_nix_store = "${appSrc}" in flake

        if legacy_uses_app and flake_uses_nix_store:
            self.fail(
                "Bug #5 confirmed: modules/build/nix/images/runtime-image.nix uses '/app/cmd/...' "
                "but flake.nix puts code at '${appSrc}/cmd/...' "
                "(/nix/store/...-deterministic-serving-stack/cmd/...). "
                "The legacy image Cmd would fail with file-not-found."
            )


if __name__ == "__main__":
    unittest.main()
