"""State transition protocols f (workout) and g (rest) for the DAG dynamics (architecture_map §3.3)."""
from __future__ import annotations
from typing import Protocol
from statepace.channels import X, Z, Array


class WorkoutTransition(Protocol):
    """p(Z_t | Z_{t-1}, X_t). Edge Z_{t-1} -> Z_t with X_t -> Z_t."""
    d_Z: int
    def step(self, Z_prev: Z, X_t: X) -> Z: ...
    def log_prob(self, Z_prev: Z, X_t: X, Z_t: Z) -> Array: ...


class RestTransition(Protocol):
    """p(Z_t | Z_{t-1}). Edge Z_{t-1} -> Z_t on rest days.

    Contract: valid only for consecutive-rest-day counts in
    [1, max_consecutive_rest_days]. Beyond the bound, callers must
    treat Z as undefined (A5).
    """
    d_Z: int
    max_consecutive_rest_days: int
    def step(self, Z_prev: Z, n_rest_days: int) -> Z: ...
    def log_prob(self, Z_prev: Z, n_rest_days: int, Z_t: Z) -> Array: ...
