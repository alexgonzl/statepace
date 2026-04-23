"""Synthetic fixtures for pipeline-wiring tests; no ground-truth modeling parameters."""
from __future__ import annotations

from typing import Mapping

import numpy as np

from statepace.channels import Array, AthleteMeta, Channels
from statepace.channels import E as E_channel
from statepace.channels import P as P_channel
from statepace.channels import X as X_channel


def make_athlete_meta(subject_id: str, sex: str) -> AthleteMeta:
    """Assemble a single AthleteMeta. Purely structural."""
    return AthleteMeta(subject_id=subject_id, sex=sex)


def make_channels(
    subject_id: str,
    dates: Array,
    P_values: Array,
    P_names: tuple[str, ...],
    X_values: Array,
    X_names: tuple[str, ...],
    is_rest: Array,
    E_values: Array,
    E_names: tuple[str, ...],
) -> Channels:
    """Assemble a single-athlete Channels from caller-supplied arrays.

    Purely structural. All values and schedule come from the caller; this
    function does not generate, sample, or default any content. Its sole
    invariant enforcement is: rows of X_values marked by is_rest are
    replaced with NaN before packaging, so the scaffold's "X absent on
    rest days" contract (channels.py) is guaranteed regardless of what
    the caller passed in for those rows.

    Contract:
    - All arrays are 2D with shape (T, d) where T = len(dates), even when
      d == 1. (The caller is responsible for producing 2D arrays.)
    - dates: datetime64[ns], shape (T,).
    - is_rest: bool, shape (T,).
    - P_values, X_values, E_values: shape (T, d_P), (T, d_X), (T, d_E).
    - len(P_names) == d_P; len(X_names) == d_X; len(E_names) == d_E.
    - NaN-on-rest invariant: X_values[is_rest, :] is overwritten with
      np.nan. Other channels and non-rest rows are passed through
      unchanged.

    Raises ValueError on any contract violation above.
    """
    T = len(dates)

    # --- dtype checks ---
    if dates.dtype != np.dtype("datetime64[ns]"):
        raise ValueError(
            f"dates.dtype must be datetime64[ns], got {dates.dtype}"
        )
    if is_rest.dtype != np.dtype(bool):
        raise ValueError(
            f"is_rest.dtype must be bool, got {is_rest.dtype}"
        )

    # --- 2D shape checks ---
    for arr, label in (
        (P_values, "P_values"),
        (X_values, "X_values"),
        (E_values, "E_values"),
    ):
        if arr.ndim != 2:
            raise ValueError(
                f"{label} must be 2D, got shape {arr.shape}"
            )

    # --- consistent T across all arrays ---
    for arr, label in (
        (P_values, "P_values"),
        (X_values, "X_values"),
        (E_values, "E_values"),
        (is_rest, "is_rest"),
    ):
        if len(arr) != T:
            raise ValueError(
                f"{label} first dim is {len(arr)}, expected {T} (len(dates))"
            )

    # --- names length matches trailing dim ---
    if len(P_names) != P_values.shape[1]:
        raise ValueError(
            f"len(P_names)={len(P_names)} does not match P_values.shape[1]={P_values.shape[1]}"
        )
    if len(X_names) != X_values.shape[1]:
        raise ValueError(
            f"len(X_names)={len(X_names)} does not match X_values.shape[1]={X_values.shape[1]}"
        )
    if len(E_names) != E_values.shape[1]:
        raise ValueError(
            f"len(E_names)={len(E_names)} does not match E_values.shape[1]={E_values.shape[1]}"
        )

    # NaN-on-rest: copy so caller's array is not mutated, then mask rest rows.
    X_vals = X_values.copy().astype(float)
    X_vals[is_rest, :] = np.nan

    return Channels(
        subject_id=subject_id,
        dates=dates,
        P=P_channel(values=P_values, names=P_names),
        X=X_channel(values=X_vals, names=X_names, is_rest=is_rest),
        E=E_channel(values=E_values, names=E_names),
    )


def make_m2_test_cohort() -> tuple[Mapping[str, Channels], Mapping[str, AthleteMeta]]:
    """N=50 synthetic cohort for M2 tests. 38 F, 12 M; volumes split evenly per sex across two buckets at edge 100.0.

    Each athlete gets T=360 days. P component `dist_km` carries a per-athlete
    constant distance value chosen so the train-window (indices 90..300) sum
    lands in a specific bucket relative to edge 100.0:
    - Low-volume athletes: P dist_km = 0.3 per day -> train-window sum ~= 63.0 (below edge)
    - High-volume athletes: P dist_km = 0.7 per day -> train-window sum ~= 147.0 (above edge)

    Within each sex, half are low-volume and half are high-volume:
    - F: 19 low, 19 high (38 total)
    - M: 6 low, 6 high (12 total)

    Subject IDs: F_00..F_37 and M_00..M_11.
    """
    T = 360
    dates = np.arange("2024-01-01", "2024-12-26", dtype="datetime64[D]").astype("datetime64[ns]")
    assert len(dates) == T

    is_rest = np.zeros(T, dtype=bool)
    X_values = np.ones((T, 1), dtype=float)
    E_values = np.ones((T, 1), dtype=float)
    P_names = ("dist_km",)
    X_names = ("pace_s_km",)
    E_names = ("temp_c",)

    LOW_DIST = 0.3   # sum over 210 days ~= 63.0
    HIGH_DIST = 0.7  # sum over 210 days ~= 147.0

    # F_00..F_18 low (19), F_19..F_37 high (19)
    # M_00..M_05 low (6), M_06..M_11 high (6)
    specs: list[tuple[str, float]] = []
    for i in range(19):
        specs.append((f"F_{i:02d}", LOW_DIST))
    for i in range(19, 38):
        specs.append((f"F_{i:02d}", HIGH_DIST))
    for i in range(6):
        specs.append((f"M_{i:02d}", LOW_DIST))
    for i in range(6, 12):
        specs.append((f"M_{i:02d}", HIGH_DIST))

    cohort: dict[str, Channels] = {}
    meta: dict[str, AthleteMeta] = {}

    for sid, dist_val in specs:
        sex = sid[0]  # "F" or "M"
        P_values = np.full((T, 1), dist_val, dtype=float)
        channels = make_channels(
            subject_id=sid,
            dates=dates.copy(),
            P_values=P_values,
            P_names=P_names,
            X_values=X_values.copy(),
            X_names=X_names,
            is_rest=is_rest.copy(),
            E_values=E_values.copy(),
            E_names=E_names,
        )
        cohort[sid] = channels
        meta[sid] = AthleteMeta(subject_id=sid, sex=sex)

    return cohort, meta
