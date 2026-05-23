# Notes Structure

Notes are organized by directory so they can be tracked and ignored per topic area.

## Categories

- `features/`: design and implementation notes for feature development.
- `bugs/`: debugging notes, root-cause analysis, and fixes.
- `issues/`: broader technical issues, decisions, and investigations.

## Required Workflow

After any large change, add a date-stamped markdown note in the correct category.

Each note must include:

1. What changed.
2. Why it changed.
3. Risks/tradeoffs.
4. Validation/test follow-up.
5. `Commit:` hash (current `HEAD`, or `NO_COMMIT_YET` if no commit exists yet).

If existing note files do not fit your work, create a new subdirectory and a new note file.

## Ignoring Notes by Directory

Use category-level `.gitignore` files to ignore specific topic subdirectories:

- `features/.gitignore`
- `bugs/.gitignore`
- `issues/.gitignore`

Add one directory path per line in the relevant category file.

Example:

```gitignore
experimental-parser/
```

This ignores `plan/notes/features/experimental-parser/`.

## Commit Hash Helper

Use `scripts/notes/current_commit.sh` to print the value for note entries.
