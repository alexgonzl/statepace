---
name: commit-readiness
description: Pre-commit audit aligned with CLAUDE.md commit discipline. Checks for runtime test, scope creep, unsolicited files, dead parameters, public-API drift, forbidden language (pass/fail labels), canonical-source contradictions, governance-doc sync, and vocabulary discipline in ADRs/plans. Run before every commit.
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
10. **Canonical-source rule (advisory).** Scan added lines in any `docs/*.md` or `docs/**/*.md` for numeric literals near known hyperparameter names: `warmup_days`, `max_consecutive_rest_days`, `train_days`, `test_days`, `d_Z`, `d_P`, `d_X`, `d_E`, `n_athletes`, `n_days`. These should appear as parameter names in specs, not as concrete values. Warn with file:line; do not block.
11. **Governance-doc sync (advisory).** If the diff touches files in a governed directory, check that its governance doc is either updated in the same commit or already correctly describes the new state. Mapping: `statepace/**` → `docs/architecture_map.md`; `docs/**` → `docs/README.md`; `docs/decisions/**` → `modeling-decision-record` skill template (no in-repo governance file); `docs/plans/**` → `docs/plans/README.md`; `tests/**` → `tests/README.md`. Warn if the governance doc seems stale relative to the added/removed files.
12. **Vocabulary discipline (advisory).** If the diff touches `docs/decisions/**` or `docs/plans/**`, scan added lines for forbidden architecture-leak symbols: standalone `μ`, `α`, `β`, `σ_X`, `σ_f`, `σ_g`, `Z*` (asterisk form), `α·`, `β·`, or any phrasing that commits the model to a parametric family (linear, Gaussian, additive, mean-reversion, etc.) without an explicit decision record introducing those terms. Warn with file:line.

## Output

- **Diff summary:** files and hunks.
- **Findings:** numbered, each pointing to file:line.
- **Blocking vs warning:** classify each finding. Checks 1–9 may block; checks 10–12 are advisory only.
- **Verdict:** ready / blocked — enumerate what must change. Advisory findings are listed but do not gate the commit.
