.PHONY: lint schema test-fast test-full test-nightly test-release ci-pr ci-main ci-nightly ci-release build-libnetdet lint-proverdet typecheck-proverdet test-proverdet

lint:
	bash scripts/ci/lint.sh

schema:
	bash scripts/ci/schema_gate.sh

test-fast:
	bash scripts/ci/test_fast.sh

test-full:
	bash scripts/ci/test_full.sh

test-nightly:
	bash scripts/ci/test_nightly.sh

test-release:
	bash scripts/ci/test_release.sh

ci-pr: lint schema test-fast

ci-main: lint schema test-full

ci-nightly: lint schema test-nightly

ci-release: lint schema test-release

build-libnetdet:
	cd native/libnetdet && make

# Scoped tooling for the prover-verifier demo (experiments/prover-verifier-demo).
# Keeps the existing tree's looser conventions intact while letting the new code
# sit under stricter ruff + pyright. Edit ruff.toml / pyrightconfig.json under
# experiments/prover-verifier-demo/ to tune.
PROVERDET_PATHS := pkg/proverdet cmd/prover cmd/verifier_server cmd/verifier_cli

lint-proverdet:
	bash scripts/ci/run_proverdet_lint.sh

typecheck-proverdet:
	bash scripts/ci/run_proverdet_typecheck.sh

test-proverdet:
	bash scripts/ci/run_proverdet_tests.sh
