---
name: dag-audit
description: Audit a code change, proposed model, or analysis step against the project DAG. Classify touched variables (confounder / mediator / collider / exogenous), identify which conditional distribution(s) of the factorization are affected, and flag identification risks. Use before merging a modeling change or when reviewing a result that touches state or observation estimation.
---

# DAG Audit

Read `docs/theoretical_framework.md` §2 and §5 before running the audit. Keep the DAG in mind: `Z_{t-1} → P_t → X_t`, `Z_{t-1} → X_t`, `E_t → X_t`, `X_t → Z_t`, `Z_{t-1} → Z_t`.

## When to trigger

- Proposed new covariate or regressor.
- Proposed new observation channel.
- New state estimator or transition model.
- Result that looks "too good" on something state-related.
- Any time ACWR, chronic load, or Banister-style decomposition is mentioned.

## Checklist

1. **Scope.** Name the change in one sentence. Which of the four conditional distributions does it touch? (`p(P|Z)`, `p(X|Z,P,E)`, `f`, `g`)
2. **Variable classification.** For every variable the change adds, removes, or conditions on, classify it as:
   - Confounder of the edge being estimated → conditioning removes bias.
   - Mediator on the causal path → conditioning blocks the effect.
   - Collider → conditioning introduces bias.
   - Exogenous → optional for identification, can improve precision.
   Sketch the local sub-DAG in text: `A → B ← C`.
3. **DAG edges touched.** Does the change implicitly add, remove, or cross an edge not in §2? (e.g., treating chronic load as a direct predictor of state is an illicit `chronic_load → Z` edge.)
4. **Hard-rule check.** Confirm none of CLAUDE.md's standing modeling rules are violated: ACWR-as-confounder only, chronic load never standalone, no Banister, cardiac drift modeled not truncated, best-effort independence preserved.
5. **Identification.** Does the change create a parameter that's only constrained by one data regime or by a collinear set? If yes, invoke identifiability-auditor.
6. **Warm-up.** If the change affects scoring, does it respect the 6-month warm-up (A8)?

## Output

- **Scope:** one-sentence summary.
- **Edges / distributions touched:** list.
- **Variable classification table:** variable | role | action | rationale.
- **Violations:** numbered, ordered by severity. Each: the issue, what would break, what to do.
- **Verdict:** one of — clean, clean with caveats, blocked pending fix, blocked pending broader review.
