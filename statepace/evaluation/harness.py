"""Evaluation harness: split wiring, warm-up enforcement, and estimator invocation (architecture_map §3.7)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
from statepace.channels import Channels, Z, X, Array
from statepace.filter import StateEstimator
from statepace.observation import ObservationModel
from statepace.transitions import WorkoutTransition, RestTransition


@dataclass(frozen=True)
class EvalSplit:
    """Per-subject fit/score split. Respects warm-up mask (conventions)."""
    subject_id: str
    fit_idx: Array    # int, indices into Channels.dates
    score_idx: Array  # int, indices into Channels.dates


def make_splits(channels: Channels, warmup_days: int) -> Iterable[EvalSplit]: ...

def run_evaluation(
    channels: Channels,
    splits: Iterable[EvalSplit],
    estimator: StateEstimator,
    observation: ObservationModel,
    workout_transition: WorkoutTransition,
    rest_transition: RestTransition,
) -> "EvalResult": ...


@dataclass(frozen=True)
class EvalResult:
    Z_hat: Z          # estimator output on score_idx
    X_pred: X         # one-step observation predictions on score_idx
    rest_bound_violations: Array  # bool, shape (T,), flags A5 overruns
