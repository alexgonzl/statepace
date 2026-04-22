---
name: commit-readiness
description: Pre-commit audit aligned with CLAUDE.md commit discipline. Checks for runtime test, scope creep, unsolicited files, dead parameters, public-API drift, and forbidden language (pass/fail labels). Run before every commit.
---

# Commit Readiness

Enforce CLAUDE.md standing rules on the staged diff.

## Checklist

1. **Runtime test.** Has a runtime test against real fixture data been run since the last change? If not, block unless the user explicitly acknowledges the risk.
2. **Scope.** Is every hunk in the diff part of the requested change? Flag unsolicited edits: unrelated refactors, formatting-only changes, new files not asked for.
3. **Public API.** Did any symbol exported by `statepace/__init__.py` change signature or disappear? If yes, require explicit confirmation.
4. **Dead parameters.** Any function parameter added but not used, or kept after refactor but no longer reachable? Remove.
5. **Renormalization.** Any implicit renormalization of a signal or score (scaling, centering, reweighting) that wasn't asked for? Block.
6. **Mocking.** Any test that mocks internals instead of using real fixture data? Block.
7. **IDE files.** `.idea/`, `.vscode/`, `*.swp`, `.DS_Store` in the diff? Remove.
8. **Language hygiene.** Any "pass", "fail", "success" labels on results in new docs, comments, or log lines? Replace with numbers and deltas.
9. **Commit message.** Describes what and why, not how. No scope beyond the diff.

## Output

- **Diff summary:** files and hunks.
- **Findings:** numbered, each pointing to file:line.
- **Blocking vs warning:** classify each finding.
- **Verdict:** ready / blocked — enumerate what must change.
