#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

CATALOG_PATH = Path("tests/conformance/spec_requirements.v1.json")
BLOCKERS_PATH = Path("tests/conformance/RELEASE_BLOCKERS.json")
ID_RE = re.compile(r"^SPEC-[0-9]+(\.[0-9]+)?-[A-Z0-9_-]+$")
VALID_MODALITY = {"MUST", "SHOULD", "MAY"}
VALID_STATUS = {"implemented", "partial", "planned", "scaffolding"}


def fail(msg: str) -> int:
    print(msg, file=sys.stderr)
    return 1


def _validate_catalog(catalog: dict) -> tuple[dict[str, dict], list[str]]:
    errors: list[str] = []
    reqs = catalog.get("requirements")
    if not isinstance(reqs, list) or not reqs:
        errors.append("Conformance catalog requires non-empty requirements list")
        return ({}, errors)

    by_id: dict[str, dict] = {}
    for idx, req in enumerate(reqs):
        if not isinstance(req, dict):
            errors.append(f"Requirement at index {idx} must be object")
            continue

        rid = req.get("id")
        section = req.get("section")
        modality = req.get("modality")
        text = req.get("text")
        status = req.get("status")
        verification = req.get("verification")

        if not isinstance(rid, str) or not ID_RE.match(rid):
            errors.append(f"Invalid requirement id at index {idx}: {rid}")
            continue
        if rid in by_id:
            errors.append(f"Duplicate requirement id: {rid}")
            continue

        if not isinstance(section, str) or not section:
            errors.append(f"{rid}: missing section")
        if modality not in VALID_MODALITY:
            errors.append(f"{rid}: invalid modality {modality}")
        if not isinstance(text, str) or not text:
            errors.append(f"{rid}: missing text")
        if status not in VALID_STATUS:
            errors.append(f"{rid}: invalid status {status}")
        if not isinstance(verification, dict):
            errors.append(f"{rid}: verification must be object")
            verification = {}

        targets = verification.get("targets", [])
        if not isinstance(targets, list):
            errors.append(f"{rid}: verification.targets must be list")
            targets = []

        if modality == "MUST" and status == "implemented" and len(targets) == 0:
            errors.append(f"{rid}: implemented MUST requirement must define verification targets")

        for t in targets:
            if not isinstance(t, str) or not t:
                errors.append(f"{rid}: invalid verification target {t}")
                continue
            path = Path(t)
            if not path.exists():
                errors.append(f"{rid}: verification target does not exist: {t}")

        by_id[rid] = req

    return (by_id, errors)


def _validate_blockers(by_id: dict[str, dict]) -> list[str]:
    errors: list[str] = []
    blockers = json.loads(BLOCKERS_PATH.read_text(encoding="utf-8"))
    ids = blockers.get("required_conformance_ids", [])
    if not isinstance(ids, list) or not ids:
        errors.append("Release blockers must define non-empty required_conformance_ids")
        return errors

    for rid in ids:
        if rid not in by_id:
            errors.append(f"Release blocker references unknown ID: {rid}")
            continue
        req = by_id[rid]
        if req["modality"] != "MUST":
            errors.append(f"Release blocker must reference MUST requirement: {rid}")
        if req["status"] not in {"implemented", "scaffolding"}:
            errors.append(f"Release blocker must reference implemented/scaffolding requirement: {rid} ({req['status']})")

    return errors


def main() -> int:
    if not CATALOG_PATH.exists():
        return fail(f"Missing conformance catalog: {CATALOG_PATH}")
    if not BLOCKERS_PATH.exists():
        return fail(f"Missing release blockers file: {BLOCKERS_PATH}")

    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    by_id, errors = _validate_catalog(catalog)
    errors.extend(_validate_blockers(by_id))

    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        return 1

    must_total = sum(1 for req in by_id.values() if req["modality"] == "MUST")
    must_implemented = sum(1 for req in by_id.values() if req["modality"] == "MUST" and req["status"] == "implemented")
    print(
        f"Conformance catalog valid: {len(by_id)} requirements, MUST implemented {must_implemented}/{must_total}, release blockers verified"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
