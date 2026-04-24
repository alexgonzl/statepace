# docs/reference_impls/ — organization schema

One markdown file per reference implementation of a Protocol in `statepace/`. A reference impl is additive: the framework commits to no privileged functional form (ADR 0002), and each impl carries its own spec here.

```
docs/reference_impls/
├── README.md                  ← this file
└── <slug>.md                  ← one spec per reference impl
```

## What belongs here

A spec for a concrete implementation of one of the scaffold Protocols: `ObservationModel`, `WorkoutTransition`, `RestTransition`, `StateEstimator`. The spec is written *before* the implementation lands and is what focused-engineer implements against.

What does *not* belong here:
- Modeling decisions that affect the framework or multiple impls — those are ADRs in `docs/decisions/`.
- Identifiability claims about the framework as a whole — those live in `docs/identifiability_baseline.md`.

## Required structure

Every spec file has these sections (in order):

1. **`# <Impl name>`** — descriptive, not a slug.
2. **`**Protocol:**`** — which scaffold Protocol this implements (`ObservationModel`, etc.).
3. **`**Status:**`** — one of `draft`, `accepted`, `implemented`, `superseded by <slug>`.
4. **`## Summary`** — 1–3 sentences on the family and its posture.
5. **`## Channel composition`** — P, E, X components the impl consumes/produces. Reference-impl scope only; does not commit the project to a channel assignment (CLAUDE.md standing rule defers project-wide channel calls to M9/M10).
6. **`## Functionals`** (observation impls only, per ADR 0005) — declare `π_obs(X_t)` and `π_stim(X_t)` explicitly.
7. **`## Family`** — parametric family, dimensionality choices (`d_Z`, etc.), X-parameterization stance, Z-gauge choice.
8. **`## Missingness`** — per-component missingness policy; how the likelihood handles NaN.
9. **`## Coherence with other impls`** — what this spec commits to about how it composes with transition / estimator impls (e.g., `π_stim`'s role in `f`'s input).
10. **`## What would invalidate this`** — empirical conditions under which the impl should be revisited.

## Naming

File slug is the impl's short name, lowercased with hyphens. Choose a slug that identifies the impl unambiguously within the directory; the spec's `## Family` section documents the parametric commitments.

## Relationship to ADRs

A reference-impl spec is not an ADR. Specs describe *what is being built*; ADRs record *decisions that affect the framework or cross-cut multiple impls*. If a spec needs to take a decision that will outlive its own impl (e.g., channel-assignment commitment, framework-level semantics), that decision spins out into a separate ADR and the spec links to it.
