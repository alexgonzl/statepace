"""DAG primitives and the single entry point for raw frames (architecture_map §3.1)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Protocol
import numpy as np
import pandas as pd

Array = np.ndarray


@dataclass(frozen=True)
class SessionFrame:
    """Typed view of the session-level table (docs/data_contract.md)."""
    subject_id: str
    dates: Array              # datetime64[ns], shape (T,)
    activity_num: Array       # int, shape (T,)
    raw: pd.DataFrame         # the original frame; read-only downstream


@dataclass(frozen=True)
class StepFrame:
    """Typed view of the 10-second step table (docs/data_contract.md)."""
    subject_id: str
    activity_num: Array       # int, shape (N,)
    t_elapsed_s: Array        # float, shape (N,)
    raw: pd.DataFrame


@dataclass(frozen=True)
class P:
    """Session shape (DAG §1). Observed."""
    values: Array             # shape (T, d_P)
    names: tuple[str, ...]


@dataclass(frozen=True)
class X:
    """Execution (DAG §1). Observed; absent on rest days."""
    values: Array             # shape (T, d_X); NaN rows on rest days
    names: tuple[str, ...]
    is_rest: Array            # bool, shape (T,)


@dataclass(frozen=True)
class E:
    """Exogenous environment (DAG §1, §A2)."""
    values: Array             # shape (T, d_E)
    names: tuple[str, ...]


@dataclass(frozen=True)
class Z:
    """Latent state (DAG §1, §A4). Dimension is per-family."""
    mean: Array               # shape (T, d_Z)
    cov: Array | None         # shape (T, d_Z, d_Z) or None if point estimate
    dates: Array              # datetime64[ns], shape (T,)


@dataclass(frozen=True)
class Channels:
    """Bundle of typed channels aligned on a common daily index."""
    subject_id: str
    dates: Array              # datetime64[ns], shape (T,), daily
    P: P
    X: X
    E: E


@dataclass(frozen=True)
class AthleteMeta:
    """Per-athlete invariant metadata. Not on any DAG edge.

    `Channels` is session-level (one row per day per athlete). `AthleteMeta`
    is athlete-level: one record per `subject_id`, stable across time. Used
    by split/cohort machinery (evaluation/harness.py) for stratified cohort
    assignment; never by observation, transitions, filter, forward, or
    predict.
    """
    subject_id: str
    sex: Literal["F", "M"]


class ChannelAssignment(Protocol):
    """Per-family rule mapping data-contract columns to (P, X, E).

    The concrete mapping lives in docs/channel_assignment.md
    (Deliverable #2). This Protocol is the in-code contract.
    """
    def split(self, session: SessionFrame, step: StepFrame) -> Channels: ...


def load_session_frame(df: pd.DataFrame, subject_id: str) -> SessionFrame: ...
def load_step_frame(df: pd.DataFrame, subject_id: str) -> StepFrame: ...

def to_channels(
    session: SessionFrame,
    step: StepFrame,
    assignment: ChannelAssignment,
) -> Channels: ...
