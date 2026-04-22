# CLAUDE.md — statepace

Research codebase. Scientific correctness > engineering polish.

---

## What This Project Does

Per-athlete cardiorespiratory state estimation from longitudinal run data:

- **Observation model** `p(X | Z, P, E)` → `observation.py`
- **State transitions** `f` (workout), `g` (rest, bounded) → `transitions.py`; inference over Z in `filter.py`
- **Prediction** = filter → forward → observation at reference conditions → `predict.py`

Current state: scaffold only — package exports constants from `_constants.py`. Planned module layout lives in `docs/architecture_map.md`. Legacy names (`cardiac_cost.py`, `riegel.py`, `state_estimation.py`, etc.) are reference-only; do not import from new modules.

---

## Environment

```bash
conda env create -f environment.yml
conda activate statepace
python -c "import statepace; print(len(statepace.__all__), 'exports')"
```

---

## Read First

| Document | When |
|---|---|
| `docs/theoretical_framework.md` | Before any modeling decision — the DAG is the guiding principle |
| `docs/data_contract.md` | Before touching I/O — expected input schemas |
| `docs/conventions.md` | For cross-cutting conventions (warm-up, bounds, deconfounding template) |
| `docs/decisions/` | Before revisiting a prior modeling decision — check if an ADR already settled it |
| `README.md` | For layout + setup |

---

## User Preferences

- Python is primary.
- To-the-point, succinct answers. Minimize explanation unless asked.
- Always show math when appropriate.
- Ask before opening unnecessary files or running tools that fill context.
- One question at a time with a recommendation when scoping.

---

## Working Principles

### 1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them.
- If a simpler approach exists, say so.
- If something is unclear, stop. Name what's confusing.

### 2. Simplicity First
Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" that wasn't requested.
- No error handling for impossible scenarios.
- If 200 lines could be 50, rewrite it.

### 3. Surgical Changes
Touch only what you must. Clean up only your own mess.

- Don't "improve" adjacent code or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice dead code outside the refactor scope, surface it — don't delete it. Inside an explicit refactor, removing dead/unused code is expected.

### 4. Goal-Driven Execution
Define success criteria. Loop until verified.

Transform tasks into verifiable goals:
- "Add validation" → "Write tests, then make them pass"
- "Fix the bug" → "Reproduce it in a test, then fix"
- "Refactor X" → "Ensure tests pass before and after"

---

## Standing Rules

- No unsolicited files, no scope creep, no renormalization unless asked.
- Preserve public APIs (what `__init__.py` exports) unless explicitly asked to change.
- Test with real fixture data; no mocking internals.
- No IDE config files.
- Commit discipline: never commit without a runtime test or explicit acknowledgement of risk.
- No dead parameters — remove unused/obsolete when explicitly refactoring; otherwise surface, don't delete.
- Report numbers and deltas; don't label pass/fail/success.

---

## Keeping This File Current

CLAUDE.md is an index, not a journal. Full rationale lives in `docs/decisions/` (ADRs via `modeling-decision-record`), channel calls in `docs/channel_assignment.md`, conventions in `docs/conventions.md`.

**At end-of-session, propose a CLAUDE.md diff if any of the following changed during that session:**

1. *What This Project Does* — a module moved from scaffold to implemented, or the three-bullet map drifted.
2. *Standing Rules* or *Working Principles* — a new rule was established, or an existing one was amended/retired.
3. *Read First* pointers — a doc was added, moved, or deprecated under `docs/`.
4. `statepace/__init__.py` exports — the public API changed meaningfully.

If none of these changed, say nothing. Do not propose cosmetic edits.
