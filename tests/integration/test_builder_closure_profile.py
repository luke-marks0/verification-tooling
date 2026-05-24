from __future__ import annotations

import os
import re
import tempfile
import unittest
from pathlib import Path

from tests.helpers import read_json, run_cmd


class TestBuilderClosureProfile(unittest.TestCase):
    def _resolve_and_build(
        self,
        *,
        manifest: str,
        resolved: Path,
        built: Path,
        builder_system: str = "nix",
    ) -> dict:
        run_cmd(["python3", "modules/inference/resolver/main.py", "--manifest", manifest, "--lockfile-out", str(resolved)])
        cmd = ["python3", "modules/build/builder/main.py", "--lockfile", str(resolved), "--lockfile-out", str(built)]
        if builder_system != "nix":
            cmd.extend(["--builder-system", builder_system])
        run_cmd(cmd)
        return read_json(built)

    def test_builder_emits_closure_profile(self) -> None:
        manifest = "tests/fixtures/positive/manifest.v1.example.json"
        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            resolved = tdir / "resolved.lock.json"
            built = tdir / "built.lock.json"
            lockfile = self._resolve_and_build(manifest=manifest, resolved=resolved, built=built)

            self.assertIn("build", lockfile)
            build = lockfile["build"]
            self.assertEqual(build["builder_system"], "nix")
            self.assertEqual(build["closure_inputs_digest"], lockfile["runtime_closure_digest"])
            self.assertTrue(re.fullmatch(r"nix://closure/[a-f0-9]{64}", build["closure_uri"]))
            self.assertEqual(build["nix_closure"]["source"], "reference_descriptor")
            self.assertEqual(build["nix_closure"]["closure_digest"], lockfile["runtime_closure_digest"])
            self.assertGreaterEqual(build["nix_closure"]["closure_size_bytes"], 1)

            expected_components = {
                "serving_stack",
                "cuda_userspace_or_container",
                "kernel_libraries",
            }
            component_names = {item["name"] for item in build["components"]}
            self.assertEqual(component_names, expected_components)

            artifacts_by_id = {item["artifact_id"]: item for item in lockfile["artifacts"]}
            for component in build["components"]:
                self.assertEqual(component["artifact_count"], len(component["artifact_ids"]))
                for artifact_id in component["artifact_ids"]:
                    self.assertIn(artifact_id, artifacts_by_id)

            self.assertGreaterEqual(len(build["oci_artifacts"]), 1)
            self.assertEqual(build["collective_stack_artifacts"], [])
            self.assertTrue(build["oci_image"]["image_ref"].startswith("oci://"))
            self.assertTrue(
                any(item["attestation_type"] == "build_provenance" for item in lockfile["attestations"])
            )

    def test_builder_is_idempotent_on_already_built_lockfile(self) -> None:
        manifest = "tests/fixtures/positive/manifest.v1.example.json"
        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            resolved = tdir / "resolved.lock.json"
            built1 = tdir / "built1.lock.json"
            built2 = tdir / "built2.lock.json"

            self._resolve_and_build(manifest=manifest, resolved=resolved, built=built1)
            run_cmd(["python3", "modules/build/builder/main.py", "--lockfile", str(built1), "--lockfile-out", str(built2)])

            left = read_json(built1)
            right = read_json(built2)
            self.assertEqual(left["runtime_closure_digest"], right["runtime_closure_digest"])
            self.assertEqual(left["canonicalization"]["lockfile_digest"], right["canonicalization"]["lockfile_digest"])
            self.assertEqual(left["build"], right["build"])

    def test_builder_supports_equivalent_builder_mode(self) -> None:
        manifest = "tests/fixtures/positive/manifest.v1.example.json"
        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            resolved = tdir / "resolved.lock.json"
            built = tdir / "built.lock.json"
            lockfile = self._resolve_and_build(
                manifest=manifest,
                resolved=resolved,
                built=built,
                builder_system="equivalent",
            )

            self.assertEqual(lockfile["build"]["builder_system"], "equivalent")
            self.assertTrue(lockfile["build"]["closure_uri"].startswith("equivalent://closure/"))
            self.assertEqual(lockfile["build"]["nix_closure"]["source"], "equivalent_descriptor")

    def test_builder_uses_nix_cli_metadata_when_store_paths_are_provided(self) -> None:
        manifest = "tests/fixtures/positive/manifest.v1.example.json"
        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            bin_dir = tdir / "bin"
            bin_dir.mkdir()
            fake_nix = bin_dir / "nix"
            fake_nix.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env python3",
                        "import json",
                        "import sys",
                        "paths = [arg for arg in sys.argv[1:] if arg.startswith('/nix/store/')]",
                        "payload = {",
                        "    path: {",
                        "        'narHash': 'sha256-fakehash',",
                        "        'narSize': 2048,",
                        "        'references': [],",
                        "        'deriver': '/nix/store/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-runtime.drv',",
                        "    }",
                        "    for path in paths",
                        "}",
                        "print(json.dumps(payload, sort_keys=True))",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            fake_nix.chmod(0o755)

            resolved = tdir / "resolved.lock.json"
            built = tdir / "built.lock.json"
            run_cmd(["python3", "modules/inference/resolver/main.py", "--manifest", manifest, "--lockfile-out", str(resolved)])
            run_cmd(
                [
                    "python3",
                    "modules/build/builder/main.py",
                    "--lockfile",
                    str(resolved),
                    "--lockfile-out",
                    str(built),
                    "--nix-store-path",
                    "/nix/store/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-runtime",
                ],
                env={"PATH": f"{bin_dir}:{os.environ['PATH']}"},
            )

            lockfile = read_json(built)
            nix_closure = lockfile["build"]["nix_closure"]
            self.assertEqual(nix_closure["source"], "nix_cli")
            self.assertEqual(nix_closure["store_paths"], ["/nix/store/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-runtime"])
            self.assertEqual(
                nix_closure["derivation_paths"],
                ["/nix/store/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-runtime.drv"],
            )
            self.assertEqual(nix_closure["closure_digest"], lockfile["runtime_closure_digest"])


if __name__ == "__main__":
    unittest.main()
