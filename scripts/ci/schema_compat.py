#!/usr/bin/env python3
import json
import pathlib
import re
import sys
from collections import defaultdict

PATTERN = re.compile(r"^(?P<name>[a-z_]+)\.v(?P<version>\d+)\.schema\.json$")


def required_set(schema: dict) -> set[str]:
    req = schema.get("required", [])
    return set(req) if isinstance(req, list) else set()


def main() -> int:
    grouped: dict[str, list[tuple[int, pathlib.Path]]] = defaultdict(list)
    for p in sorted(pathlib.Path("modules/core/schemas").glob("*.schema.json")):
        m = PATTERN.match(p.name)
        if not m:
            print(f"Unexpected schema filename: {p.name}", file=sys.stderr)
            return 1
        grouped[m.group("name")].append((int(m.group("version")), p))

    for name, versions in grouped.items():
        versions.sort(key=lambda t: t[0])
        found = [v for v, _ in versions]
        expected = list(range(found[0], found[-1] + 1))
        if found != expected:
            print(f"Schema versions are not contiguous for {name}: {found}", file=sys.stderr)
            return 1

        prev_required = None
        prev_path = None
        for _, path in versions:
            schema = json.loads(path.read_text(encoding="utf-8"))
            current_required = required_set(schema)
            if prev_required is not None and not prev_required.issubset(current_required):
                print(
                    f"Potential incompatible required-key removal between {prev_path.name} and {path.name}",
                    file=sys.stderr,
                )
                return 1
            prev_required = current_required
            prev_path = path

    print(f"Schema compatibility check passed for {len(grouped)} schema family(ies)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
