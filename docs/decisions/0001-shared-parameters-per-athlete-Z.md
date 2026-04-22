# 0001: Fit parameters shared across athletes; absorb between-athlete variation into per-athlete `Z`

**Date:** 2026-04-22
**Status:** accepted

## Decision

One set of parameters — for the observation model `p(X | Z, P, E)`, the workout transition `f`, and the rest transition `g` — is fit jointly across all training-cohort athletes; each athlete carries their own `Z` trajectory (initial condition `Z_0`, dynamics, and stable-trait dimensions within `Z`) as the locus of between-athlete variation.

## Context

The training/validation split planned in `/Users/alex/.claude/plans/serialized-singing-wadler.md` requires a clear statement of what is shared and what is per-athlete at fit time. `docs/theoretical_framework.md` §A7 (line 215) establishes that the DAG's *functional forms* are universal across athletes and that between-athlete variation "is absorbed into `Z`" — but that is a statement about the DAG, not about fitting procedure. A7 is silent on whether *fitted parameters* of the observation model and the transitions are pooled, hierarchical, or athlete-specific. This ADR records the procedural choice.

## Alternatives considered

- **Per-athlete fits**: one parameter set per athlete. Rejected: insufficient per-athlete data volume to identify the parameters of the observation model and transitions individually, especially without committing to functional forms that would allow heavy regularization.
- **Hierarchical / random-effects pooling**: parameters drawn from an athlete-level prior, with athlete-specific deviations. Rejected for this pass: adds a second layer of identifiability concerns and requires a functional-form commitment to specify the hierarchy. Worth revisiting once a first estimator family is in place (see invalidation).
- **Full pooling with no per-athlete state variation**: collapse `Z` across athletes too. Rejected: directly contradicts A7's consequence that same-`Z`, same-`(P, E)` athletes should produce the same `X` distribution — i.e., all observed heterogeneity would be unexplained noise.

## DAG implications

- Affects the prior structure on `Z_0` (A8): per-athlete `Z_0` draws, shared diffuse form.
- Relies on A7 in its strongest reading — the functional forms are universal, and the *fitted* parameter vector is the same object across athletes.
- Does not affect the factorization in §3 of the theoretical framework; affects only the estimation procedure.
- Implications for A10 (stationarity): parameters are stationary per athlete *and* stationary across athletes under this choice. Regime changes within or across athletes all collapse into `Z`-trajectory motion.

## Empirical grounding

None for this ADR — taken as a procedural starting point before any estimator is implemented. The test-vs-validation gap reported in the future ADR 0002 (first-estimator verdict) will be the first empirical check on whether parameter sharing holds up.

## What would invalidate this

- A dataset with per-athlete volume large enough that per-athlete parameter fits (or hierarchical partial-pooling) become identifiable, i.e., per-athlete effective sample size grows past the parameter dimension by a reasonable margin.
- A substantial and consistent gap between test score (within-training-cohort held-out) and validation score (held-out athletes) that cannot be closed by refining `Z` — a failure of the "`Z` absorbs between-athlete differences" claim at the chosen `Z` expressiveness.
- Evidence of systematic per-athlete responder-type clusters that cannot be represented by any `Z` trajectory of the chosen dimensionality.

## Follow-ups

- Dimensionality of `Z` (`d_Z`) is load-bearing here — too low and shared parameters cannot absorb heterogeneity; too high and identifiability suffers. Tracked as decision D9 in the sequencing plan.
- A second ADR (0002) will record the verdict on the first estimator family and the first empirical check against this assumption.
