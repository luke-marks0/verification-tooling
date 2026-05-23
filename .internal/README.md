# Internal working notes

Project workspace: spec drafts, build plans, development notes, hardware
recon, and rollout logs that predate or support the public artifacts in the
top-level repo. Not part of the public-facing contract and not required to
reproduce the paper's results.

## Contents

- `plan/` — project specification (`spec.md`), phased build plan, feature/bug
  notes. The spec is the historical source of truth referenced from
  `docs/conformance/`.
- `recon/` — hardware recon field notes (NIC models, PCI topology) gathered
  during experiment setup.
- `experiments/multinode-determinism/` — Lambda/vast rollout plans and logs
  for the multinode determinism runs. The public-facing experiment code
  and data live under `experiments/multinode-determinism/` at the repo root.
