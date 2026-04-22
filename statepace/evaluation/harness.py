"""Evaluation harness: split wiring, warm-up enforcement, and estimator invocation (architecture_map §3.7)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping
from statepace.channels import Channels, Z, X, Array
from statepace.filter import StateEstimator
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


def make_splits(
    cohort: Mapping[str, Channels],
    warmup_days: int,
) -> Iterable[EvalSplit]: ...

def run_evaluation(
    cohort: Mapping[str, Channels],
    splits: Iterable[EvalSplit],
    estimator: StateEstimator,
    observation: ObservationModel,
    workout_transition: WorkoutTransition,
    rest_transition: RestTransition,
) -> "EvalResult": ...


@dataclass(frozen=True)
class EvalResult:
    Z_hat: Mapping[str, Z]                      # keyed by subject_id; estimator output on score_idx
    X_pred: Mapping[str, X]                     # keyed by subject_id; one-step observation predictions on score_idx
    cohort: Mapping[str, Cohort]                # subject_id -> cohort label
    rest_bound_violations: Mapping[str, Array]  # keyed by subject_id; bool, shape (T,), flags A5 overruns
