"""Prediction integral composition: filter output -> forward propagation -> observation inverse (architecture_map §3.6)."""
from __future__ import annotations
from statepace.channels import Channels, P, E, X, Array
from statepace.filter import StateEstimator, ZPosterior
from statepace.forward import ForwardSchedule
from statepace.observation import ObservationModel, ConditioningSpec
from statepace.transitions import WorkoutTransition, RestTransition


def predict_session(
    history: Channels,
    estimator: StateEstimator,
    observation: ObservationModel,
    workout_transition: WorkoutTransition,
    rest_transition: RestTransition,
    schedule: ForwardSchedule,
    P_target: P,
    E_target: E,
    X_partial: X,
    spec: ConditioningSpec,
) -> Array: ...
