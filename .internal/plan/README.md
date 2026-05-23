# Planning and Development Notes

This directory contains the project specification, implementation plan, and ongoing development notes.

## Files

- `spec.md`: authoritative technical specification.
- `BUILD_PLAN.md`: phased implementation and test strategy.
- `notes/README.md`: notes structure and rules.
- `notes/INDEX.md`: index of feature/bug/issue note directories.

## Working Rule

After any large change (architecture, subsystem behavior, test strategy, deployment model, or data format), add a short entry under `notes/` with:

1. `notes/features/<feature-name>/` for feature changes.
2. `notes/bugs/<bug-name-or-id>/` for bug work.
3. `notes/issues/<issue-name-or-id>/` for investigations/cross-cutting issues.

Each note entry must include:

1. What changed.
2. Why it changed.
3. Risks/tradeoffs.
4. Next validation steps.
5. `Commit:` hash (current `HEAD`, or `NO_COMMIT_YET` if repo has no commits yet).

Keep entries concise and date-stamped.

If no existing note file matches the work, create a new note subdirectory and markdown file.
