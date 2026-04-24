# 0005: `X_t` is a single node; observation and transition edges read distinct functionals

**Date:** 2026-04-23
**Status:** accepted

## Decision

`X_t` stays a single node — the within-session execution trajectory. Reference impls declare two functionals:
- `π_obs(X_t)` entering `p(X_t | Z_{t-1}, P_t, E_t)`.
- `π_stim(X_t)` entering `f(Z_{t-1}, X_t)`.

Graph unchanged; §4(c) rewritten to make this explicit.

## Context

A proposal to split `X_t` into `X_perf` (observation child of `Z`) and `X_stim` (transition parent of `Z`) went through three review rounds. Arbiter (master-statistician) settled it on Pearl SCM semantics: node-hood requires independent intervenability. Two deterministic functionals of one session trajectory cannot be independently `do(·)`-targeted — one node, two projections.

The §4(c) text as written conflated three claims: (a) no information beyond `X_t`, (b) the stimulus equals full `X_t`, (c) no privileged scalar loads. (b) is physiologically wrong (peak speed during a long run doesn't drive multi-day adaptation). The rewrite keeps (a) and (c), replaces (b) with the functionals view.

## Alternatives considered

- **Split into `X_perf` + `X_stim`.** Rejected: violates SCM semantics; severs identification leverage from joint (performance, stimulus) variation at matched `(Z_{t-1}, P, E)`; forces rewriting A1 and §5.
- **Leave §4(c) as written.** Rejected: its "full execution is the stimulus" reading biases `f` by feeding it capability readouts.

## DAG implications

No edges changed. `π_obs`/`π_stim` are per-family assignments (analogous to A1's `P`/`X` channel split), specified by each reference impl.

## What would invalidate this

- A reference impl needs to intervene on `π_stim` while holding `π_obs` fixed (or vice versa) — would mean they're independently intervenable after all.
- The bundled-`X` bias shows up in practice despite the functionals discipline — would mean the discipline is too weak and a node split is needed in practice.

## Follow-ups

- Rewrite §4(c) in `docs/theoretical_framework.md` (same commit or adjacent).
- Extend A1: channel assignment includes choice of functional per edge.
- Reference-impl specs under `docs/reference_impls/` declare both functionals.
