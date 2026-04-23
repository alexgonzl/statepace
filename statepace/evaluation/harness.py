"""Evaluation harness: split wiring, warm-up enforcement, and estimator invocation (architecture_map §3.7)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np

from statepace.channels import AthleteMeta, Channels, Z, X, Array
from statepace.filter import StateEstimator, Prior
from statepace.observation import ObservationModel
from statepace.transitions import WorkoutTransition, RestTransition

Cohort = Literal["train", "test", "validation"]


@dataclass(frozen=True)
class EvalSplit:
    """Per-subject fit/score split with cohort label (ADR 0001/0002). Respects warm-up mask (conventions)."""
    subject_id: str
    cohort: Cohort
    fit_idx: Array    # int, indices into this athlete's Channels.dates
    score_idx: Array  # int, indices into this athlete's Channels.dates


def assign_cohorts(
    athletes: Mapping[str, Channels],
    meta: Mapping[str, AthleteMeta],
    *,
    validation_fraction: float,
    seed: int,
    volume_bucket_edges: Sequence[float],
    warmup_days: int,
    train_days: int,
    volume_component: str,
) -> Mapping[str, Literal["train", "validation"]]:
    """Partition athletes into training and validation cohorts by stratified random split.

    Strata are the Cartesian product of (sex, volume bucket), where volume is
    the sum of channels.P[volume_component] over the train window
    [warmup_days, warmup_days + train_days). Bucket assignment uses
    np.digitize with the supplied edges.

    Within each non-empty stratum, n_validation = ceil(stratum_size *
    validation_fraction) athletes are drawn into the validation cohort. This is
    a lower bound: every non-empty stratum contributes at least ceil(...) to
    validation. Draws are deterministic given seed and input: subjects are
    sorted lexicographically within each stratum before sampling.

    Args:
        athletes: per-athlete Channels, keyed by subject_id.
        meta: per-athlete AthleteMeta, keyed by subject_id. Must match athletes keys exactly.
        validation_fraction: fraction of each stratum to assign to validation.
        seed: integer seed for np.random.default_rng.
        volume_bucket_edges: monotone edge values for np.digitize bucketing.
        warmup_days: number of warm-up days at the start of each timeline.
        train_days: number of training days immediately following warmup.
        volume_component: name of the P channel component used for volume.

    Returns:
        Mapping from subject_id to "train" or "validation".

    Raises:
        ValueError: if athletes and meta key sets differ, or volume_component
            is not in P.names.
    """
    if set(athletes.keys()) != set(meta.keys()):
        only_athletes = set(athletes.keys()) - set(meta.keys())
        only_meta = set(meta.keys()) - set(athletes.keys())
        raise ValueError(
            f"athletes and meta key sets differ. "
            f"Only in athletes: {only_athletes}. Only in meta: {only_meta}."
        )

    # Verify volume_component exists in at least one athlete's P.names.
    # All athletes share the same channel schema so checking one suffices, but
    # we check the first one here; if the cohort is empty we skip.
    if athletes:
        sample = next(iter(athletes.values()))
        if volume_component not in sample.P.names:
            raise ValueError(
                f"volume_component '{volume_component}' not in P.names {sample.P.names}"
            )

    edges = list(volume_bucket_edges)

    # Build stratum -> [subject_id] mapping.
    strata: dict[tuple[str, int], list[str]] = {}
    for sid, channels in athletes.items():
        j = channels.P.names.index(volume_component)
        train_slice = channels.P.values[warmup_days: warmup_days + train_days, j]
        volume = float(np.sum(train_slice))
        bucket = int(np.digitize(volume, edges))
        sex = meta[sid].sex
        key = (sex, bucket)
        strata.setdefault(key, [])
        strata[key].append(sid)

    rng = np.random.default_rng(seed)
    assignment: dict[str, Literal["train", "validation"]] = {}

    for key in sorted(strata.keys()):
        members = sorted(strata[key])  # lexicographic for determinism
        n = len(members)
        n_validation = math.ceil(n * validation_fraction)
        chosen = rng.choice(n, size=n_validation, replace=False)
        validation_set = {members[i] for i in chosen}
        for sid in members:
            assignment[sid] = "validation" if sid in validation_set else "train"

    return assignment


def make_splits(
    cohort: Mapping[str, Channels],
    meta: Mapping[str, AthleteMeta],
    *,
    warmup_days: int,
    train_days: int,
    test_days: int,
    cohort_assignment: Mapping[str, Literal["train", "validation"]],
) -> Iterable[EvalSplit]:
    """Produce per-athlete EvalSplit objects from a pre-computed cohort assignment.

    fit_idx always covers [0, warmup_days + train_days). score_idx and the
    cohort label depend on cohort_assignment[sid]:

    - "train" athletes emit TWO splits:
        1. In-sample ("train" label): score_idx = [warmup_days, warmup_days + train_days).
        2. Out-of-sample ("test" label): score_idx = [warmup_days + train_days,
           warmup_days + train_days + test_days).
    - "validation" athletes emit ONE split ("validation" label): score_idx =
      [warmup_days + train_days, warmup_days + train_days + test_days).

    Total splits = 2 * n_training_cohort + 1 * n_validation_cohort.

    Args:
        cohort: per-athlete Channels, keyed by subject_id.
        meta: per-athlete AthleteMeta, keyed by subject_id.
        warmup_days: number of warm-up days at the start of each timeline.
        train_days: number of training days following warm-up.
        test_days: number of scoring days following training.
        cohort_assignment: output of assign_cohorts; maps subject_id to
            "train" or "validation".

    Returns:
        Iterable of EvalSplit; two per training-cohort athlete, one per validation-cohort athlete.

    Raises:
        ValueError: if cohort, meta, and cohort_assignment key sets differ, or
            any athlete's timeline is shorter than warmup_days + train_days + test_days.
    """
    if not (set(cohort.keys()) == set(meta.keys()) == set(cohort_assignment.keys())):
        raise ValueError(
            "cohort, meta, and cohort_assignment must have identical key sets."
        )

    required_len = warmup_days + train_days + test_days
    fit_idx = np.arange(0, warmup_days + train_days, dtype=int)
    train_score_idx = np.arange(warmup_days, warmup_days + train_days, dtype=int)
    test_score_idx = np.arange(warmup_days + train_days, required_len, dtype=int)

    splits = []
    for sid, channels in cohort.items():
        if len(channels.dates) < required_len:
            raise ValueError(
                f"Athlete '{sid}' has {len(channels.dates)} days; "
                f"need at least {required_len} (warmup={warmup_days} + "
                f"train={train_days} + test={test_days})."
            )
        if cohort_assignment[sid] == "train":
            splits.append(EvalSplit(
                subject_id=sid,
                cohort="train",
                fit_idx=fit_idx,
                score_idx=train_score_idx,
            ))
            splits.append(EvalSplit(
                subject_id=sid,
                cohort="test",
                fit_idx=fit_idx,
                score_idx=test_score_idx,
            ))
        else:
            splits.append(EvalSplit(
                subject_id=sid,
                cohort="validation",
                fit_idx=fit_idx,
                score_idx=test_score_idx,
            ))

    return splits


def run_evaluation(
    cohort: Mapping[str, Channels],
    splits: Iterable[EvalSplit],
    estimator: StateEstimator,
    observation: ObservationModel,
    workout_transition: WorkoutTransition,
    rest_transition: RestTransition,
    *,
    prior: Prior | Mapping[str, Prior],
) -> "EvalResult": ...


@dataclass(frozen=True)
class EvalResult:
    Z_hat: Mapping[str, Z]                      # keyed by subject_id; estimator output on score_idx
    X_pred: Mapping[str, X]                     # keyed by subject_id; one-step observation predictions on score_idx
    cohort: Mapping[str, Cohort]                # subject_id -> cohort label
    rest_bound_violations: Mapping[str, Array]  # keyed by subject_id; bool, shape (T,), flags A5 overruns


@dataclass(frozen=True)
class FormsBundle:
    """One concrete wiring of the Protocols for a single evaluation run.

    Groups the four Protocol implementations and their prior so a sweep
    can vary them together. `label` keys the bundle into SweepResult and
    must be unique within a given sweep (enforced by run_sweep, not here).
    """
    label: str
    observation: ObservationModel
    workout_transition: WorkoutTransition
    rest_transition: RestTransition
    estimator: StateEstimator
    prior: Prior | Mapping[str, Prior]


def run_sweep(
    cohort: Mapping[str, Channels],
    splits: Iterable[EvalSplit],
    bundles: Sequence[FormsBundle],
) -> "SweepResult":
    """Invoke run_evaluation once per bundle, keying results by bundle label.

    All bundles share the same cohort and splits; only the Protocol wiring
    varies across runs. Duplicate labels in `bundles` are a caller error
    (validated at M4 alongside the body).
    """
    ...


@dataclass(frozen=True)
class SweepResult:
    results: Mapping[str, EvalResult]    # keyed by bundle label
    bundles: Mapping[str, FormsBundle]   # keyed by bundle label; same keys as results
