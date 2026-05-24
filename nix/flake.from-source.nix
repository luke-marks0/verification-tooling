{
  description = "Deterministic vLLM serving stack — fully hermetic, built from source";

  # ───────────────────────────────────────────────────────────────────────────
  # Build strategy
  # ───────────────────────────────────────────────────────────────────────────
  # This flake builds PyTorch and vLLM entirely from source inside Nix so that
  # every shared library (including CUDA kernels) links against Nix's glibc,
  # libstdc++, and CUDA toolkit.  No manylinux wheels, no autoPatchelfHook,
  # no FHS escape hatches.
  #
  # Trade-off: a clean build takes 30–60 min on a beefy machine (torch alone
  # is ~20 min with parallelism).  Subsequent builds hit the Nix store cache.
  #
  # Placeholder hashes are marked "TODO: replace after first build" — Nix will
  # tell you the correct hash on the first attempt.
  # ───────────────────────────────────────────────────────────────────────────

  inputs = {
    # nixos-unstable has better CUDA / torch support than 24.11, especially
    # on aarch64-linux where 24.11's torch lacks CUDA entirely.
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" ] (system:
      let
        # ── nixpkgs with CUDA enabled ──────────────────────────────────────
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;          # CUDA is unfree
            cudaSupport = true;          # propagate to all packages
            cudaCapabilities = [ "9.0" ]; # H100 / Hopper — trim fat from other archs
          };
        };

        python = pkgs.python310;
        pythonPackages = pkgs.python310Packages;

        # ── PyTorch from source (via nixpkgs) ─────────────────────────────
        # With cudaSupport = true in the nixpkgs config, python310Packages.torch
        # is already built from source with CUDA.  nixpkgs handles the cmake
        # build, NCCL, cuDNN, etc.  We just reference it.
        #
        # If the nixpkgs version doesn't match the exact torch version you
        # need, override the source below.
        torch = pythonPackages.torch;

        # ── vLLM 0.17.1 from source ───────────────────────────────────────
        # vLLM is not in nixpkgs, so we write a buildPythonPackage derivation.
        # vLLM's build system uses cmake + ninja to compile CUDA/C++ extensions
        # (attention kernels, paged-attention, moe kernels, etc.).
        vllmSrc = pkgs.fetchFromGitHub {
          owner = "vllm-project";
          repo = "vllm";
          rev = "v0.17.1";
          # TODO: replace after first build — Nix will print the correct hash
          hash = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=";
          fetchSubmodules = true;  # vLLM vendors cutlass, flashinfer, etc.
        };

        vllm = pythonPackages.buildPythonPackage rec {
          pname = "vllm";
          version = "0.17.1";
          format = "setuptools";

          src = vllmSrc;

          # ── Build-time dependencies ────────────────────────────────────
          nativeBuildInputs = [
            pkgs.cmake
            pkgs.ninja
            pkgs.which
            pkgs.git            # setup.py shells out to git for version
            pythonPackages.setuptools
            pythonPackages.setuptools-scm
            pythonPackages.wheel
            pythonPackages.packaging
          ];

          # ── Propagated runtime + build dependencies ────────────────────
          # These are both needed at build time (for cmake find_package /
          # header includes) and at runtime (Python imports).
          buildInputs = [
            # CUDA toolkit components
            pkgs.cudaPackages.cuda_cudart
            pkgs.cudaPackages.cuda_nvcc
            pkgs.cudaPackages.cuda_nvrtc
            pkgs.cudaPackages.cuda_cupti
            pkgs.cudaPackages.libcublas
            pkgs.cudaPackages.libcusolver
            pkgs.cudaPackages.libcusparse
            pkgs.cudaPackages.libcufft
            pkgs.cudaPackages.libcurand
            pkgs.cudaPackages.nccl
            pkgs.cudaPackages.cudnn

            # System libs
            pkgs.stdenv.cc.cc.lib  # libstdc++
            pkgs.zlib
            pkgs.openssl
          ];

          propagatedBuildInputs = [
            torch
            pythonPackages.numpy
            pythonPackages.transformers
            pythonPackages.tokenizers
            pythonPackages.sentencepiece
            pythonPackages.huggingface-hub
            pythonPackages.safetensors
            pythonPackages.requests
            pythonPackages.pyyaml
            pythonPackages.tqdm
            pythonPackages.filelock
            pythonPackages.typing-extensions
            pythonPackages.packaging
            pythonPackages.psutil
            pythonPackages.py-cpuinfo
            pythonPackages.pydantic
            pythonPackages.fastapi
            pythonPackages.uvicorn
            pythonPackages.uvloop
            pythonPackages.prometheus-client
            pythonPackages.aiohttp
            pythonPackages.ray
            pythonPackages.msgpack
            pythonPackages.pillow
            pythonPackages.openai
            pythonPackages.lm-format-enforcer or null
            pythonPackages.outlines or null
          ];

          # ── Build environment ──────────────────────────────────────────
          # Tell cmake where to find CUDA and which GPU architectures to
          # compile for.  TORCH_CUDA_ARCH_LIST must match cudaCapabilities.
          env = {
            CUDA_HOME = "${pkgs.cudaPackages.cuda_nvcc}";
            TORCH_CUDA_ARCH_LIST = "9.0";
            # Limit parallel compilation to avoid OOM on smaller machines.
            # Adjust as needed — 8 is reasonable for 64 GB RAM.
            MAX_JOBS = "8";
            # Tell vLLM's setup.py to skip the version check against git
            SETUPTOOLS_SCM_PRETEND_VERSION = version;
          };

          # vLLM's setup.py invokes cmake directly; we need CUDA on PATH
          preBuild = ''
            export PATH="${pkgs.cudaPackages.cuda_nvcc}/bin:$PATH"
            export CUDA_HOME="${pkgs.cudaPackages.cuda_nvcc}"
            # Ensure torch's cmake modules are findable
            export CMAKE_PREFIX_PATH="${torch}/${python.sitePackages}/torch/share/cmake:$CMAKE_PREFIX_PATH"
          '';

          # The CUDA kernel compilation can't run in the sandbox without
          # /dev/nvidia* but the *compilation* itself only needs nvcc, not a
          # GPU.  The sandbox is fine.

          # Skip tests — they require a live GPU
          doCheck = false;

          # Some optional deps may not be in nixpkgs; filter nulls
          postFixup = ''
            # vLLM installs some scripts; ensure they point to our python
            for f in $out/bin/*; do
              if [ -f "$f" ]; then
                substituteInPlace "$f" \
                  --replace "/usr/bin/env python" "${python}/bin/python3" || true
              fi
            done
          '';

          meta = with pkgs.lib; {
            description = "High-throughput LLM serving engine";
            homepage = "https://github.com/vllm-project/vllm";
            license = licenses.asl20;
          };
        };

        # ── Python environment ─────────────────────────────────────────────
        # All packages are from-source via nixpkgs.  torch and vllm pull in
        # most transitive deps; we add a few extras for our app.
        pythonEnv = python.withPackages (ps: [
          # Core ML stack (from source)
          torch
          vllm

          # Additional app dependencies
          ps.numpy
          ps.jsonschema
          ps.requests
          ps.pyyaml
          ps.huggingface-hub
          ps.filelock
          ps.tqdm
          ps.typing-extensions
          ps.packaging
        ]);

        # ── Application source ─────────────────────────────────────────────
        appSrc = pkgs.stdenv.mkDerivation {
          pname = "deterministic-serving-stack";
          version = "0.1.0";
          src = self;
          dontBuild = true;
          installPhase = ''
            mkdir -p $out
            # Capability layer holds all runtime code plus schemas (modules/core/schemas)
            # and model manifests (modules/inference/manifests).
            cp -r modules $out/modules
            cp -r workflows $out/workflows
          '';
        };

        # ── Full runtime closure ───────────────────────────────────────────
        # Every .so in this closure links against Nix's glibc — no manylinux
        # wheels, no FHS compat layer.  `nix path-info -rsSh` will show you
        # the exact set of store paths.
        runtimeClosure = pkgs.symlinkJoin {
          name = "deterministic-serving-runtime-closure";
          version = "0.1.0";
          paths = [
            pythonEnv
            appSrc
            pkgs.bash
            pkgs.coreutils
            pkgs.cacert
          ];
        };

        # ── OCI image ──────────────────────────────────────────────────────
        ociImage = pkgs.dockerTools.buildLayeredImage {
          name = "deterministic-serving-runtime";
          tag = self.rev or "dev";
          contents = [ runtimeClosure ];
          config = {
            Cmd = [ "${pythonEnv}/bin/python3" "${appSrc}/modules/inference/server/main.py" ];
            WorkingDir = "/workspace";
            Env = [
              "PYTHONPATH=${appSrc}:${pythonEnv}/${python.sitePackages}"
              # Determinism knobs
              "VLLM_BATCH_INVARIANT=1"
              "CUBLAS_WORKSPACE_CONFIG=:4096:8"
              "PYTHONHASHSEED=0"
              # CUDA visible to the container
              "NVIDIA_VISIBLE_DEVICES=all"
              "NVIDIA_DRIVER_CAPABILITIES=compute,utility"
            ];
          };
        };

      in {
        packages = {
          default = runtimeClosure;
          closure = runtimeClosure;
          app = appSrc;
          oci = ociImage;
          # Expose individual components for debugging / testing
          inherit torch vllm;
        };

        devShells.default = pkgs.mkShell {
          packages = [
            pythonEnv
            pkgs.bash
            pkgs.jq
            pkgs.ripgrep
          ];
          shellHook = ''
            export PYTHONPATH="$PWD:$PYTHONPATH"
          '';
        };
      }
    );
}
