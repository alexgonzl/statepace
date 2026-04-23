"""Synthetic fixtures for pipeline-wiring tests; no ground-truth modeling parameters."""
from __future__ import annotations

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
