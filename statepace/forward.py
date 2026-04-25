"""τ-step state-forward propagation using workout and rest transitions (architecture_map §3.5)."""
from __future__ import annotations
from dataclasses import dataclass
from statepace.channels import Z, X as X_channel, Array
from statepace.filter import ZPosterior
from statepace.transitions import WorkoutTransition, RestTransition


@dataclass(frozen=True)
class ForwardSchedule:
    """τ-day schedule of workout/rest days ahead of t.

    For workout days, X_future provides the (hypothetical or planned) X
    driving f. For rest days, X_future rows are masked; consecutive rest
    counts are derived internally and must respect A5's rest bound (`max_consecutive_rest_days`).
    """
    is_rest: Array          # bool, shape (tau,)
    X_future: X_channel     # shape (tau, d_X); NaN on rest days


def forward_state(
    Z_t: ZPosterior,
    schedule: ForwardSchedule,
    workout_transition: WorkoutTransition,
    rest_transition: RestTransition,
) -> ZPosterior: ...
