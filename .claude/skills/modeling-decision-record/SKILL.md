---
name: modeling-decision-record
description: Generate an ADR-style record for a non-trivial modeling decision. Captures decision, alternatives, DAG implications, and invalidation conditions. Use after the decision is made (not as planning) to build institutional memory under docs/decisions/.
---

# Modeling Decision Record

When a non-trivial modeling decision is made (estimator choice, covariate addition, transition form, evaluation metric), capture it under `docs/decisions/NNNN-short-slug.md`.

## Trigger

- Switching an estimator default (e.g., EWM halflife).
- Adding, removing, or changing a covariate in deconfounding.
- Changing `f` or `g` form.
- Changing the evaluation scoring target or reference template.
- Rejecting a candidate approach after investigation.

Do NOT generate for trivial tweaks (bug fixes, parameter tuning within a sweep, cosmetic refactors).

## Template

```markdown
# NNNN: <decision in imperative voice>

**Date:** YYYY-MM-DD
**Status:** accepted | superseded by NNNN

## Decision

One to three sentences. What was decided, in imperative voice.

## Context

Why this came up. Cite the DAG section or module that motivated it (`docs/theoretical_framework.md §N`, file:line).

## Alternatives considered

- **Alt A:** one-sentence description. Why rejected.
- **Alt B:** one-sentence description. Why rejected.

## DAG implications

Which conditional distribution(s) of the factorization are affected. Which edges. Which assumptions (A1-A10) are relied on or strained.

## Empirical grounding

Numbers and deltas that informed the decision. Reference the script or notebook that produced them.

## What would invalidate this

Concrete conditions that would force a revisit: new data regime, failed assumption, superseding result.

## Follow-ups

Open questions or deferred work, if any.
```

## Checklist before filing

1. Decision is stated in one sentence without hedging.
2. At least one alternative is cited with a specific reason for rejection.
3. DAG section or assumption is named.
4. Numbers are deltas (not pass/fail labels).
5. Invalidation conditions are concrete, not vague.
