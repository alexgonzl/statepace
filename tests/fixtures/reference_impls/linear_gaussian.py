"""Fixture factory for the linear-gaussian reference impl spec.

Serves: docs/reference_impls/linear-gaussian.md
Used by: tests/test_transitions_linear_gaussian.py
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from statepace.channels import Z
from tests.fixtures.reference_impls.riegel_score_hrstep import make_riegel_hrstep_cohort
from statepace.channels import Channels


def make_linear_gaussian_cohort(
    n_athletes: int,
    n_days: int,
    d_Z: int,
    seed: int,
) -> tuple[Mapping[str, Channels], Mapping[str, Z]]:
    """Synthetic cohort for the linear-gaussian reference impl.

    Delegates to make_riegel_hrstep_cohort to get Channels; this guarantees
    the X schema matches the paired observation impl. Z trajectories are
    drawn independently as Z_t ~ N(0, I) — no consistency with the transition
    model is imposed. This fixture supports OLS mechanics testing, not
    parameter recovery.

    Args:
        n_athletes: number of athletes.
        n_days: number of days (rows) per athlete.
        d_Z: latent dimensionality; determines Z trajectory shape.
        seed: integer seed for np.random.default_rng.

    Returns:
        (cohort, z_trajectories) where cohort is the M4-style Channels mapping
        and z_trajectories is a parallel mapping of Z dataclasses, one per
        athlete, each with mean shape (n_days, d_Z).
    """
    cohort = make_riegel_hrstep_cohort(n_athletes, n_days, seed=seed)

    rng = np.random.default_rng(seed + 1000)

    z_trajectories: dict[str, Z] = {}
    for sid, ch in cohort.items():
        mean = rng.standard_normal((n_days, d_Z))
        z_trajectories[sid] = Z(mean=mean, cov=None, dates=ch.dates.copy())

    return cohort, z_trajectories
