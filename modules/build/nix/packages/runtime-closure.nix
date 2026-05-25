# Runtime closure derivation.
#
# This is the legacy interface — prefer `nix build .#closure` via flake.nix.
# Kept for compatibility with the builder's --nix-store-path workflow.
#
# Note: vLLM, PyTorch, and CUDA are external artifacts tracked by digest
# in the lockfile, not included in the Nix closure.
{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python310;

  pythonEnv = python.withPackages (ps: [
    ps.jsonschema ps.requests ps.pyyaml ps.filelock
    ps.tqdm ps.typing-extensions ps.packaging ps.numpy
  ]);
in
pkgs.symlinkJoin {
  name = "deterministic-serving-runtime-closure";
  version = "0.1.0";
  paths = [
    pythonEnv
    pkgs.bash
    pkgs.coreutils
    pkgs.cacert
  ];

  meta = {
    description = "Hermetic runtime closure for deterministic vLLM serving";
  };
}
