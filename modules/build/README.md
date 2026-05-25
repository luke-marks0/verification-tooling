# build — deterministic / reproducible build

**Purpose.** Produce a hermetic runtime so the *software* under which inference
runs is itself reproducible — every shared library (glibc, libstdc++, CUDA
kernels, vLLM) built from source against a pinned Nix toolchain. The closure
digest is the build's identity, recorded in the lockfile and attested at boot.

**Interface (commands).**

```bash
nix build .#oci       # OCI image tarball (sshd + Python + vLLM + app); docker load < result
nix build .#closure   # hermetic runtime closure (symlink join)
nix build .#app       # app source tree (modules/, workflows/)
```

In-pipeline (records the closure into the lockfile):

```python
from modules import Pipeline
pipe = Pipeline.from_manifest(manifest).resolve().build()   # wraps modules/build/builder
pipe.lockfile["runtime_closure_digest"]   # sha256:...
```

**Artifacts.** Consumes `lockfile.v1`; enriches it with `build{}` —
`builder_system`, `closure_uri`, `closure_inputs_digest`, and `nix_closure`
(the software-stack pin). The real content-addressed closure is recorded when
built against an actual Nix closure (`--nix-store-path` / `--closure-digest`, as
the deploy path does); the default path records closure metadata only.

**Requirements.** Nix with flakes. Clean build is 30–60 min; cached builds hit
the Nix store. The OCI image pre-sets `VLLM_BATCH_INVARIANT=1`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=0`.

**Underlying code.** `lockfiles/` + `nix/` (co-located in this module),
`modules/build/builder/main.py`, and `flake.nix` (at the repo root — the flake's
`src = self` packages repo-wide code and callers invoke `.#`, so it must root the
repo; `nix/` isn't referenced by the flake, so it lives here in the module).

**Status.** Production-grade. Python facade in `modules/build/api.py`:
`build_runtime(lockfile)` runs anywhere (used by `Pipeline.build`);
`build_oci()`/`build_closure()`/`nix_build(attr)` shell out to `nix build`
(require Nix).
