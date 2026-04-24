# tests/fixtures/reference_impls/ — organization schema

One factory module per reference implementation spec under `docs/reference_impls/`. Each factory produces synthetic `Channels` (and supporting objects) with the exact X/P/E component schema the spec declares.

```
tests/fixtures/reference_impls/
├── README.md          ← this file
├── __init__.py        ← package marker; keep empty
└── <slug>.py          ← one factory module per spec (slug matches docs/reference_impls/<slug>.md)
```

## Why this directory exists

Reference-impl specs commit to specific X/P/E component schemas (e.g., the 10-component X of `riegel-score-hrstep`). Tests for each reference impl need fixtures matching that schema. Those fixtures are spec-scoped — they carry no project-wide meaning — so they live alongside the spec they serve, not in the general `synthetic.py`.

## Rules

1. **One file per spec.** File slug matches the spec slug. Example: `docs/reference_impls/riegel-score-hrstep.md` → `tests/fixtures/reference_impls/riegel_score_hrstep.py`. Note Python import names use underscores, not hyphens.
2. **Plain factory functions**, same rule as `tests/fixtures/synthetic.py` (see `tests/README.md`). Explicit arguments, no defaults, no RNG state held internally, no side effects.
3. **Produces synthetic `Channels`** with the exact X/P/E `names` tuples the spec declares. Values must respect domain constraints the spec's pre-transforms assume (e.g., non-negative for components entering `log(1 + x)`).
4. **Rest-day rows** are NaN in X per the existing `Channels` contract, per `statepace/channels.py`.
5. **No production code.** Factories construct project dataclasses; they do not define them.

## Adding a new spec fixture

1. Add the file: `tests/fixtures/reference_impls/<slug>.py`.
2. The factory's docstring names the spec it serves.
3. Test file that uses it: `tests/test_observation_<slug>.py` (or the matching test for whichever Protocol the spec implements).
