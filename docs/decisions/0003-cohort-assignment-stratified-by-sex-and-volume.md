# 0003: Assign cohorts by stratified random split on (sex, volume bucket)

**Date:** 2026-04-22
**Status:** accepted

## Decision

Athletes are partitioned into `training_cohort` and `validation_cohort` by a seeded, stratified random split, with `validation_fraction=0.10` and strata defined by the Cartesian product of `sex` and a volume bucket derived from `Channels`. Volume bucket edges are supplied as a fixed hyperparameter (`volume_bucket_edges`), not computed from the cohort. A new `AthleteMeta` container carries athlete-invariant fields (`subject_id`, `sex`) and is passed alongside `Mapping[str, Channels]` into cohort assignment.

Target signature (lives in `statepace/evaluation/harness.py`):

```python
def assign_cohorts(
    athletes: Mapping[str, Channels],
    meta: Mapping[str, AthleteMeta],
    *,
    validation_fraction: float,
    seed: int,
    volume_bucket_edges: Sequence[float],
) -> Mapping[str, Literal["train", "validation"]]: ...
```

## Context

ADR 0001 (shared parameters, per-athlete `Z`) and ADR 0002 (cohort-in-fit, channels-in-infer) together require a concrete procedure for partitioning athletes into `training_cohort` and `validation_cohort` — the split-structure contract in `docs/plans/post-scaffold-sequencing.md` depends on it, and M2 cannot land without it. The question was (a) how cohort assignment is randomized, and (b) on what data.

The user directive: 10% validation, randomization stratified on volume and sex. That rules out unstratified random assignment and rules out deterministic hashing (hash-of-`subject_id` is incompatible with balanced strata at small cohort sizes). Because stratification needs athlete-invariant data not currently on `Channels`, a new container is introduced.

The session-vs-athlete boundary is load-bearing here: `Channels` is per-session (one row per day), `AthleteMeta` is per-athlete-invariant (one record per `subject_id`). Conflating them would push athlete-level fields into every session row and require duplication/consistency checks at every read.

## Alternatives considered

- **Plain seeded random split (no stratification).** Rejected: user directive explicitly calls for balancing on sex and volume. Small validation cohorts (10% of a modest cohort) will have high between-draw variance in stratum composition without stratification.
- **Deterministic hash-of-`subject_id` assignment.** Rejected: pure hashing has the appealing property that adding an athlete doesn't reshuffle existing ones, but it's incompatible with per-stratum balancing at the validation fraction. Stratification requires the draw to see the whole cohort's stratum counts.
- **Stratification deferred to a later milestone.** Rejected: stratification is the user's intended partitioning criterion. Deferring would mean the first end-to-end run (M9, M10) uses an unrepresentative validation cohort, and the fix would retroactively reshuffle cohorts across a result already recorded.
- **Volume buckets computed from cohort (tertiles-per-cohort).** Rejected: makes assignment non-deterministic under cohort change — adding a single athlete can re-bucket others, so the validation cohort is not stable as data arrives. Fixed edges, passed in, keep the partition reproducible and the hyperparameter explicit.
- **Add `sex` as a scalar field on `Channels`.** Rejected: mixes athlete-invariant and session-level data in one container, forces `sex` to be replicated in every row, and breaks the session-vs-athlete boundary without a corresponding gain. A sibling container is the cleaner contract and was confirmed by master-architect placement review (see Follow-ups).
- **Pre-computed stratum label passed per athlete.** Rejected: punts the stratification logic to the caller and means cohort assignment can't be run from `(Channels, AthleteMeta)` alone. Contradicts the "no hardcoded values, hyperparameters surfaced by name" plan discipline by burying the stratum rule outside the function.

## DAG implications

- No DAG edge is added, removed, or altered. `AthleteMeta.sex` does not participate in any conditional distribution of the factorization — it is used only at the split boundary, before `fit` is called.
- No reliance on assumptions A1–A10 is changed. Cohort assignment is pre-inferential and does not touch the observation model, transitions, or the state estimator.
- Boundary rule compliance: `AthleteMeta` is input-side; cohort assignment lives in `evaluation/harness.py`; `statepace/` does not import from `evaluation/` (architecture_map §4, rule 2). A new boundary rule (§4, rule 13 — session-vs-athlete-level) is introduced to enforce the container split going forward.

## Empirical grounding

None. Architectural decision; fixed ahead of any estimator run. The first empirical check on whether the stratified split produces balanced validation scores is the test-vs-validation-score gap and per-stratum breakdown that will be reported after M10.

## What would invalidate this

- **A stratum with too few athletes.** If (`sex`, volume bucket) cells drop below ~2 athletes, stratified assignment becomes degenerate (a whole stratum lands entirely in one cohort). Mitigation at that point is either coarser bucket edges or dropping the stratification axis that's too sparse — both are hyperparameter changes, not ADR invalidations. ADR invalidation occurs only if stratification at real-cohort sizes becomes structurally infeasible.
- **A new stratification axis is required** (e.g., age, training history) — an ADR extension, not a replacement. `AthleteMeta` grows; `assign_cohorts` gains a named hyperparameter for the new axis.
- **Sex coding widens beyond `Literal["F", "M"]`** — handled as a data-contract change (non-binary or missing values), not an invalidation of the procedure.
- **Volume definition needs to change** — the window over which volume is summarized from `Channels` is an `assign_cohorts` implementation detail and a named hyperparameter; re-specifying it does not invalidate the procedure.

## Follow-ups

- **Architecture-map update must land before M2 implementation.** Master-architect drafted the diff (session-vs-athlete boundary as rule 13; `AthleteMeta` dataclass in `statepace/channels.py` adjacent to `Channels`; `make_splits` signature extended to take `meta`). The diff lands in its own commit before the `assign_cohorts` implementation.
- **Public-API decision deferred.** `AthleteMeta` and `Channels` are either both exported from `statepace/__init__.py` at M2 implementation time or neither is; exporting one without the other is asymmetric. No action this ADR.
- **Volume summary definition.** `assign_cohorts` computes volume inline from `Channels`; the exact summary (sum over which `P` component, over what window — warm-up + train only, or full history) is an implementation detail surfaced as a named argument. Not pinned here because it's an M2 scoping call, not an architectural one.
- **Bucket-edge calibration.** Fixed edges require the caller to choose reasonable values. For the synthetic cohort (M9), any monotonic edges are acceptable. For real data (M10), edges should reflect the actual volume distribution; a one-time calibration step at M1b landing is expected but not part of this ADR.
- **Stratum-imbalance diagnostic.** The M2 verification test should assert that every non-empty stratum sends at least one athlete to each cohort when the cohort is large enough to support it; a small-cohort degenerate case is acceptable to flag, not to prevent.
