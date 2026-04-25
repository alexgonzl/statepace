"""Evaluation harness: split wiring, warm-up enforcement, and estimator invocation (architecture_map §3.7)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np

from statepace.channels import AthleteMeta, Channels, X, Array
from statepace.filter import Prior, StateEstimator, ZPosterior
from statepace.observation import ObservationModel
from statepace.transitions import RestTransition, WorkoutTransition

Cohort = Literal["train", "test", "validation"]


@dataclass(frozen=True)
class EvalSplit:
    """Per-subject fit/score split with cohort label (ADR 0001/0002). Respects warm-up mask (conventions)."""
    subject_id: str
    cohort: Cohort
    fit_idx: Array    # int, indices into this athlete's Channels.dates
    score_idx: Array  # int, indices into this athlete's Channels.dates


def assign_cohorts(
    athletes: Mapping[str, Channels],
    meta: Mapping[str, AthleteMeta],
    *,
    validation_fraction: float,
    seed: int,
    volume_bucket_edges: Sequence[float],
    warmup_days: int,
    train_days: int,
    volume_component: str,
) -> Mapping[str, Literal["train", "validation"]]:
    """Partition athletes into training and validation cohorts by stratified random split.

    Strata are the Cartesian product of (sex, volume bucket), where volume is
    the sum of channels.P[volume_component] over the train window
    [warmup_days, warmup_days + train_days). Bucket assignment uses
    np.digitize with the supplied edges.

    Within each non-empty stratum, n_validation = ceil(stratum_size *
    validation_fraction) athletes are drawn into the validation cohort. This is
    a lower bound: every non-empty stratum contributes at least ceil(...) to
    validation. Draws are deterministic given seed and input: subjects are
    sorted lexicographically within each stratum before sampling.

    Args:
        athletes: per-athlete Channels, keyed by subject_id.
        meta: per-athlete AthleteMeta, keyed by subject_id. Must match athletes keys exactly.
        validation_fraction: fraction of each stratum to assign to validation.
        seed: integer seed for np.random.default_rng.
        volume_bucket_edges: monotone edge values for np.digitize bucketing.
        warmup_days: number of warm-up days at the start of each timeline.
        train_days: number of training days immediately following warmup.
        volume_component: name of the P channel component used for volume.

    Returns:
        Mapping from subject_id to "train" or "validation".

    Raises:
        ValueError: if athletes and meta key sets differ, or volume_component
            is not in P.names.
    """
    if set(athletes.keys()) != set(meta.keys()):
        only_athletes = set(athletes.keys()) - set(meta.keys())
        only_meta = set(meta.keys()) - set(athletes.keys())
        raise ValueError(
            f"athletes and meta key sets differ. "
            f"Only in athletes: {only_athletes}. Only in meta: {only_meta}."
        )

    # Verify volume_component exists in at least one athlete's P.names.
    # All athletes share the same channel schema so checking one suffices, but
    # we check the first one here; if the cohort is empty we skip.
    if athletes:
        sample = next(iter(athletes.values()))
        if volume_component not in sample.P.names:
            raise ValueError(
                f"volume_component '{volume_component}' not in P.names {sample.P.names}"
            )

    edges = list(volume_bucket_edges)

    # Build stratum -> [subject_id] mapping.
    strata: dict[tuple[str, int], list[str]] = {}
    for sid, channels in athletes.items():
        j = channels.P.names.index(volume_component)
        train_slice = channels.P.values[warmup_days: warmup_days + train_days, j]
        volume = float(np.sum(train_slice))
        bucket = int(np.digitize(volume, edges))
        sex = meta[sid].sex
        key = (sex, bucket)
        strata.setdefault(key, [])
        strata[key].append(sid)

    rng = np.random.default_rng(seed)
    assignment: dict[str, Literal["train", "validation"]] = {}

    for key in sorted(strata.keys()):
        members = sorted(strata[key])  # lexicographic for determinism
        n = len(members)
        n_validation = math.ceil(n * validation_fraction)
        chosen = rng.choice(n, size=n_validation, replace=False)
        validation_set = {members[i] for i in chosen}
        for sid in members:
            assignment[sid] = "validation" if sid in validation_set else "train"

    return assignment


def make_splits(
    cohort: Mapping[str, Channels],
    meta: Mapping[str, AthleteMeta],
    *,
    warmup_days: int,
    train_days: int,
    test_days: int,
    cohort_assignment: Mapping[str, Literal["train", "validation"]],
) -> Iterable[EvalSplit]:
    """Produce per-athlete EvalSplit objects from a pre-computed cohort assignment.

    fit_idx always covers [0, warmup_days + train_days). score_idx and the
    cohort label depend on cohort_assignment[sid]:

    - "train" athletes emit TWO splits:
        1. In-sample ("train" label): score_idx = [warmup_days, warmup_days + train_days).
        2. Out-of-sample ("test" label): score_idx = [warmup_days + train_days,
           warmup_days + train_days + test_days).
    - "validation" athletes emit ONE split ("validation" label): score_idx =
      [warmup_days + train_days, warmup_days + train_days + test_days).

    Total splits = 2 * n_training_cohort + 1 * n_validation_cohort.

    Args:
        cohort: per-athlete Channels, keyed by subject_id.
        meta: per-athlete AthleteMeta, keyed by subject_id.
        warmup_days: number of warm-up days at the start of each timeline.
        train_days: number of training days following warm-up.
        test_days: number of scoring days following training.
        cohort_assignment: output of assign_cohorts; maps subject_id to
            "train" or "validation".

    Returns:
        Iterable of EvalSplit; two per training-cohort athlete, one per validation-cohort athlete.

    Raises:
        ValueError: if cohort, meta, and cohort_assignment key sets differ, or
            any athlete's timeline is shorter than warmup_days + train_days + test_days.
    """
    if not (set(cohort.keys()) == set(meta.keys()) == set(cohort_assignment.keys())):
        raise ValueError(
            "cohort, meta, and cohort_assignment must have identical key sets."
        )

    required_len = warmup_days + train_days + test_days
    fit_idx = np.arange(0, warmup_days + train_days, dtype=int)
    train_score_idx = np.arange(warmup_days, warmup_days + train_days, dtype=int)
    test_score_idx = np.arange(warmup_days + train_days, required_len, dtype=int)

    splits = []
    for sid, channels in cohort.items():
        if len(channels.dates) < required_len:
            raise ValueError(
                f"Athlete '{sid}' has {len(channels.dates)} days; "
                f"need at least {required_len} (warmup={warmup_days} + "
                f"train={train_days} + test={test_days})."
            )
        if cohort_assignment[sid] == "train":
            splits.append(EvalSplit(
                subject_id=sid,
                cohort="train",
                fit_idx=fit_idx,
                score_idx=train_score_idx,
            ))
            splits.append(EvalSplit(
                subject_id=sid,
                cohort="test",
                fit_idx=fit_idx,
                score_idx=test_score_idx,
            ))
        else:
            splits.append(EvalSplit(
                subject_id=sid,
                cohort="validation",
                fit_idx=fit_idx,
                score_idx=test_score_idx,
            ))

    return splits


def run_evaluation(
    cohort: Mapping[str, Channels],
    splits: Iterable[EvalSplit],
    estimator: StateEstimator,
    observation: ObservationModel,
    workout_transition: WorkoutTransition,
    rest_transition: RestTransition,
    *,
    prior: Prior | Mapping[str, Prior],
    n_samples: int = 200,
    rng: np.random.Generator | None = None,
) -> "EvalResult":
    """Fit the estimator on the training cohort, run per-athlete inference, and produce EvalResult.

    Steps:
      1. Fit estimator on training-cohort athletes (those whose split carries cohort=="train").
      2. For every athlete in the cohort, call estimator.infer to obtain a ZPosterior.
      3. For each athlete's score window (EvalSplit.score_idx), produce an XPredictive
         carrying observation-space mean and samples shaped (n_samples, T_score, d_X).
      4. Populate rest_bound_violations per athlete over the score window.
      5. Derive the cohort mapping from the EvalSplit objects.

    Args:
        cohort: per-athlete Channels, keyed by subject_id.
        splits: iterable of EvalSplit; each carries subject_id, cohort label, fit_idx, score_idx.
        estimator: StateEstimator to fit and use for inference.
        observation: ObservationModel for forward predictions.
        workout_transition: WorkoutTransition (passed to estimator.fit).
        rest_transition: RestTransition (passed to estimator.fit; provides max_consecutive_rest_days).
        prior: Prior or per-athlete mapping of Prior; passed to estimator.fit and infer.
        n_samples: number of latent trajectory samples drawn per athlete for XPredictive.samples.
        rng: numpy Generator for reproducibility; defaults to default_rng(0).

    Returns:
        EvalResult with Z_hat, X_pred, cohort, rest_bound_violations populated.
    """
    from statepace.channels import Z as Z_channel
    from statepace.filter import _count_consecutive_rest

    if rng is None:
        rng = np.random.default_rng(0)

    splits_list = list(splits)

    # Build cohort label mapping and per-subject score_idx from splits.
    # A subject may appear in multiple splits (train + test). Use the first occurrence
    # to record the cohort label; score_idx is per-split, handled below.
    cohort_labels: dict[str, Cohort] = {}
    # Map subject_id -> list of (cohort_label, score_idx) for all splits.
    subject_splits: dict[str, list[EvalSplit]] = {}
    for sp in splits_list:
        subject_splits.setdefault(sp.subject_id, []).append(sp)
        # Cohort label: prefer "train" if present (two splits for training athletes).
        if sp.subject_id not in cohort_labels or sp.cohort == "train":
            cohort_labels[sp.subject_id] = sp.cohort

    # Step 1: identify training-cohort athletes and fit.
    train_cohort = {
        sid: cohort[sid]
        for sid, label in cohort_labels.items()
        if label == "train"
    }
    fitted_estimator = estimator.fit(
        train_cohort, observation, workout_transition, rest_transition, prior
    )

    # Take the fitted observation model from the estimator's public surface
    # (Protocol-level: every StateEstimator answers `fitted_observation()`).
    # For families that fit observation parameters jointly (JointMLEKalman),
    # this returns a fresh populated instance built from the estimator's
    # parameters. For families that consume a pre-fit observation without
    # modifying it, this returns that same instance. The harness does not
    # mutate the caller's `observation` argument.
    observation = fitted_estimator.fitted_observation()

    # Step 2: infer per-athlete posteriors (all athletes, not only training).
    Z_hat: dict[str, object] = {}
    for sid in cohort:
        athlete_prior = prior[sid] if isinstance(prior, Mapping) else prior
        Z_hat[sid] = fitted_estimator.infer(cohort[sid], prior=athlete_prior)

    max_rest = rest_transition.max_consecutive_rest_days

    # Steps 3 & 4: produce XPredictive and rest_bound_violations per split.
    # Each EvalSplit gets its own XPredictive keyed by subject_id. When a subject
    # has two splits (train + test), the last one written wins; the caller gets one
    # XPredictive per subject_id. To preserve both, the spec says keyed by subject_id —
    # training athletes have two splits but a single Z_hat entry. We use the
    # test/score split's score_idx for the XPredictive (the harness iterates all splits
    # and we keep the last, which for training athletes is the "test" split appended second).
    # For simplicity, collect all (sid, score_idx) pairs and produce one entry per
    # unique (sid, cohort_label) — but the EvalResult fields are keyed by subject_id only.
    # Resolution: use the non-train split when both exist (test > train for predictive).
    score_split_per_sid: dict[str, EvalSplit] = {}
    for sp in splits_list:
        if sp.subject_id not in score_split_per_sid or sp.cohort != "train":
            score_split_per_sid[sp.subject_id] = sp

    X_pred: dict[str, XPredictive] = {}
    rest_bound_violations: dict[str, np.ndarray] = {}

    for sid, sp in score_split_per_sid.items():
        channels = cohort[sid]
        posterior = Z_hat[sid]
        score_idx = sp.score_idx          # shape (T_score,)
        T_score = len(score_idx)

        # --- XPredictive mean ---
        # Slice posterior mean at score_idx rows and wrap in Z_channel.
        posterior_mean_full = posterior.mean()   # (T, d_Z)
        posterior_cov_full = posterior.cov        # (T, d_Z, d_Z)
        z_mean_score = posterior_mean_full[score_idx]   # (T_score, d_Z)
        z_cov_score = posterior_cov_full[score_idx]     # (T_score, d_Z, d_Z)
        z_score = Z_channel(
            mean=z_mean_score,
            cov=z_cov_score,
            dates=channels.dates[score_idx],
        )
        from statepace.channels import P as P_channel, E as E_channel, X as X_channel
        p_score = P_channel(
            values=channels.P.values[score_idx],
            names=channels.P.names,
        )
        e_score = E_channel(
            values=channels.E.values[score_idx],
            names=channels.E.names,
        )
        x_mean = observation.forward(z_score, p_score, e_score)   # X dataclass

        # --- XPredictive samples ---
        # Draw n_samples from ZPosterior.sample over the full timeline, then slice.
        z_samples_full = posterior.sample(n_samples, rng)   # (n_samples, T, d_Z)
        d_X = x_mean.values.shape[1]
        samples = np.empty((n_samples, T_score, d_X), dtype=float)
        for k in range(n_samples):
            z_k = Z_channel(
                mean=z_samples_full[k][score_idx],    # (T_score, d_Z)
                cov=None,
                dates=channels.dates[score_idx],
            )
            x_k = observation.forward(z_k, p_score, e_score)
            samples[k] = x_k.values

        # Propagate rest-day NaN to samples (rest rows are NaN in x_mean).
        is_rest_score = channels.X.is_rest[score_idx]   # (T_score,)
        samples[:, is_rest_score, :] = np.nan

        X_pred[sid] = XPredictive(
            mean=x_mean,
            samples=samples,
            n_samples=n_samples,
        )

        # --- rest_bound_violations ---
        consec = _count_consecutive_rest(channels.X.is_rest)   # (T,)
        violations = consec[score_idx] > max_rest              # (T_score,) bool
        rest_bound_violations[sid] = violations

    return EvalResult(
        Z_hat=Z_hat,
        X_pred=X_pred,
        cohort=cohort_labels,
        rest_bound_violations=rest_bound_violations,
    )


@dataclass(frozen=True)
class XPredictive:
    """Per-day observation-space predictive distribution summary on score_idx.

    Carries both a point prediction (`mean`, raw-X space) and a sample-based
    representation of the predictive distribution (`samples`). The harness
    populates both using only Protocol-level operations on `ObservationModel`
    and `ZPosterior`:

    - `mean` is `observation.forward(Z_prev=posterior_mean, P_score, E_score)`.
    - `samples` are produced by drawing `n_samples` latent trajectories from
      `ZPosterior.sample(n_samples, rng)`, then pushing each through
      `observation.forward` at the score-window's `(P_score, E_score)`. This
      is family-agnostic: any `ZPosterior` subclass (Gaussian, sample-based,
      mixture) and any `ObservationModel` (linear-Gaussian, non-linear,
      heavy-tailed) compose without harness changes.

    Downstream callers (M7 `predict.py`, M8 `evaluation/metrics.py`) compute
    point residuals from `mean` and prediction intervals / coverage from
    `samples`. The harness does not pre-summarize `samples` into intervals
    because the choice of summary (quantile bands, HDR, per-component
    covariance) is metric-side.

    Attributes:
        mean: `X` carrying conditional-mean predictions over score_idx in
            raw observation space; `is_rest` mirrors the score-window's
            rest mask (the harness predicts on workout days only and sets
            rest rows to NaN with `is_rest=True`).
        samples: shape (n_samples, T_score, d_X) in raw observation space.
            Rows aligned with `mean.values`. Rest rows are NaN across all
            samples.
        n_samples: number of trajectories drawn from `ZPosterior.sample`.
            Recorded so consumers can detect under-sampling.
    """
    mean: X
    samples: Array
    n_samples: int


@dataclass(frozen=True)
class EvalResult:
    """Per-athlete evaluation outputs over a single estimator/observation wiring.

    Attributes:
        Z_hat: subject_id -> `ZPosterior` (latent posterior over score_idx;
            mean + covariance + dates). The full posterior is retained so
            M8 calibration diagnostics can score latent-coverage without
            re-running inference.
        X_pred: subject_id -> `XPredictive` (observation-space predictive
            distribution summary on score_idx; mean + samples).
        cohort: subject_id -> cohort label ("train", "test", "validation").
        rest_bound_violations: subject_id -> bool array of shape (T,);
            True at score_idx rows where the consecutive-rest count exceeded
            `RestTransition.max_consecutive_rest_days` (A5 overrun).
    """
    Z_hat: Mapping[str, ZPosterior]
    X_pred: Mapping[str, XPredictive]
    cohort: Mapping[str, Cohort]
    rest_bound_violations: Mapping[str, Array]


@dataclass(frozen=True)
class FormsBundle:
    """One concrete wiring of the Protocols for a single evaluation run.

    Groups the four Protocol implementations and their prior so a sweep
    can vary them together. `label` keys the bundle into SweepResult and
    must be unique within a given sweep (enforced by run_sweep, not here).
    """
    label: str
    observation: ObservationModel
    workout_transition: WorkoutTransition
    rest_transition: RestTransition
    estimator: StateEstimator
    prior: Prior | Mapping[str, Prior]


def run_sweep(
    cohort: Mapping[str, Channels],
    splits: Iterable[EvalSplit],
    bundles: Sequence[FormsBundle],
    *,
    n_samples: int = 200,
    rng: np.random.Generator | None = None,
) -> "SweepResult":
    """Invoke run_evaluation once per bundle, keying results by bundle label.

    All bundles share the same cohort and splits; only the Protocol wiring
    varies across runs. Duplicate labels in `bundles` are a caller error
    (validated here).

    Args:
        cohort: per-athlete Channels, keyed by subject_id.
        splits: iterable of EvalSplit; consumed once and shared across all bundles.
        bundles: sequence of FormsBundle; labels must be unique.
        n_samples: passed through to run_evaluation.
        rng: passed through to run_evaluation.

    Returns:
        SweepResult with results and bundles dicts keyed by bundle label.

    Raises:
        ValueError: if any two bundles share the same label.
    """
    labels = [b.label for b in bundles]
    seen: set[str] = set()
    for label in labels:
        if label in seen:
            raise ValueError(
                f"run_sweep: duplicate bundle label '{label}'. "
                "Each bundle label must be unique within a sweep."
            )
        seen.add(label)

    splits_list = list(splits)

    results: dict[str, EvalResult] = {}
    bundles_map: dict[str, FormsBundle] = {}

    for bundle in bundles:
        result = run_evaluation(
            cohort=cohort,
            splits=splits_list,
            estimator=bundle.estimator,
            observation=bundle.observation,
            workout_transition=bundle.workout_transition,
            rest_transition=bundle.rest_transition,
            prior=bundle.prior,
            n_samples=n_samples,
            rng=rng,
        )
        results[bundle.label] = result
        bundles_map[bundle.label] = bundle

    return SweepResult(results=results, bundles=bundles_map)


@dataclass(frozen=True)
class SweepResult:
    results: Mapping[str, EvalResult]    # keyed by bundle label
    bundles: Mapping[str, FormsBundle]   # keyed by bundle label; same keys as results
