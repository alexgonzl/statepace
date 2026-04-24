"""Fixture factory for the riegel-score-hrstep reference impl spec.

Serves: docs/reference_impls/riegel-score-hrstep.md
Used by: tests/test_observation_riegel.py (task 3).
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from statepace.channels import Channels
from statepace.channels import E as E_channel
from statepace.channels import P as P_channel
from statepace.channels import X as X_channel

# Spec-exact channel names (riegel-score-hrstep.md §Channel composition).
_P_NAMES: tuple[str, ...] = (
    "total_distance",
    "total_duration",
    "elevation",
    "is_track",
    "time_of_day_sin",
    "time_of_day_cos",
)

_E_NAMES: tuple[str, ...] = ("wet_bulb_temp",)

_X_NAMES: tuple[str, ...] = (
    "best_effort_riegel_speed_score",
    "best_effort_grade",
    "best_effort_mean_HR",
    "best_effort_hr_drift",
    "best_effort_speed_cadence_ratio",
    "hr_load",
    "step_load",
    "total_elevation_gain",
    "total_elevation_lost",
    "heat_exposure",
)

# Rest-day fraction: ~20% of days are rest days.
_REST_FRAC = 0.20


def _sample_P(rng: np.random.Generator, T: int) -> np.ndarray:
    """Sample P values (T, 6). All values respect domain constraints."""
    total_distance = rng.uniform(2_000, 20_000, size=T)        # metres
    total_duration = rng.uniform(600, 7_200, size=T)           # seconds
    elevation = rng.uniform(0, 2_500, size=T)                  # metres ASL
    is_track = rng.integers(0, 2, size=T).astype(float)        # {0, 1}

    # time_of_day as an angle; sin and cos guaranteed to satisfy sin²+cos²=1.
    tod_angle = rng.uniform(0, 2 * np.pi, size=T)
    time_of_day_sin = np.sin(tod_angle)
    time_of_day_cos = np.cos(tod_angle)

    return np.column_stack([
        total_distance,
        total_duration,
        elevation,
        is_track,
        time_of_day_sin,
        time_of_day_cos,
    ])


def _sample_E(rng: np.random.Generator, T: int) -> np.ndarray:
    """Sample E values (T, 1). wet_bulb_temp in [0, 30] °C."""
    wet_bulb_temp = rng.uniform(0, 30, size=(T, 1))
    return wet_bulb_temp


def _sample_X(
    rng: np.random.Generator,
    T: int,
    total_duration_s: np.ndarray,
    wet_bulb_temp: np.ndarray,
) -> np.ndarray:
    """Sample X values (T, 10). Respects all pre-transform domain constraints.

    Args:
        rng: seeded RNG.
        T: number of rows.
        total_duration_s: shape (T,), session duration in seconds (from P).
        wet_bulb_temp: shape (T,), wet-bulb temperature °C (from E).

    Returns:
        Array of shape (T, 10).
    """
    # best_effort_riegel_speed_score > 0: lognormal centred near 1.0.
    score = rng.lognormal(mean=0.0, sigma=0.15, size=T)

    # best_effort_grade: signed, typically (-0.1, +0.1).
    grade = rng.uniform(-0.10, 0.10, size=T)

    # best_effort_mean_HR: plausible HR range 120–180 bpm.
    mean_hr = rng.uniform(120, 180, size=T)

    # best_effort_hr_drift: signed, typically (-15, +15) bpm.
    hr_drift = rng.uniform(-15, 15, size=T)

    # best_effort_speed_cadence_ratio: stride length proxy, positive, ~0.8–1.6 m.
    speed_cad_ratio = rng.uniform(0.8, 1.6, size=T)

    # hr_load: raw values ~1500–22000 so that hr_load/7200 ∈ ~[0.2, 3.0].
    hr_load = rng.uniform(1_500, 22_000, size=T)

    # step_load: raw values ~3000–50000 so that step_load/16200 ∈ ~[0.2, 3.0].
    step_load = rng.uniform(3_000, 50_000, size=T)

    # total_elevation_gain/lost: non-negative, 0–1000m per session.
    elev_gain = rng.uniform(0, 1_000, size=T)
    elev_lost = rng.uniform(0, 1_000, size=T)

    # heat_exposure = ∫(wet_bulb − 18)_+ dt, zero on cool sessions.
    # Approximate as (wet_bulb − 18)_+ × duration_hours with small noise.
    duration_hours = total_duration_s / 3_600.0
    excess_temp = np.maximum(0.0, wet_bulb_temp - 18.0)
    heat_exposure = np.maximum(0.0, excess_temp * duration_hours * rng.uniform(0.8, 1.2, size=T))

    return np.column_stack([
        score,
        grade,
        mean_hr,
        hr_drift,
        speed_cad_ratio,
        hr_load,
        step_load,
        elev_gain,
        elev_lost,
        heat_exposure,
    ])


def make_riegel_hrstep_cohort(
    n_athletes: int,
    n_days: int,
    seed: int,
) -> Mapping[str, Channels]:
    """Synthetic cohort for the riegel-score-hrstep reference impl.

    Args:
        n_athletes: number of athletes in the returned dict.
        n_days: number of days (rows) per athlete.
        seed: integer seed passed to np.random.default_rng.

    Returns:
        Dict keyed by subject_id (e.g. "A_00"), each value a Channels with
        P.names = _P_NAMES (6 components),
        E.names = _E_NAMES (1 component),
        X.names = _X_NAMES (10 components).
        X rows for rest days are NaN per the Channels contract.
    """
    rng = np.random.default_rng(seed)

    start = np.datetime64("2023-01-01", "D")
    dates = np.arange(start, start + n_days, dtype="datetime64[D]").astype("datetime64[ns]")

    cohort: dict[str, Channels] = {}

    for i in range(n_athletes):
        sid = f"A_{i:02d}"

        is_rest = rng.random(n_days) < _REST_FRAC

        P_values = _sample_P(rng, n_days)
        E_values = _sample_E(rng, n_days)

        total_duration_s = P_values[:, 1]   # column index 1 = total_duration
        wet_bulb_1d = E_values[:, 0]        # column index 0 = wet_bulb_temp

        X_values = _sample_X(rng, n_days, total_duration_s, wet_bulb_1d)

        # Enforce NaN-on-rest invariant.
        X_values = X_values.copy().astype(float)
        X_values[is_rest, :] = np.nan

        cohort[sid] = Channels(
            subject_id=sid,
            dates=dates.copy(),
            P=P_channel(values=P_values, names=_P_NAMES),
            X=X_channel(values=X_values, names=_X_NAMES, is_rest=is_rest),
            E=E_channel(values=E_values, names=_E_NAMES),
        )

    return cohort
