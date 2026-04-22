---
name: identifiability-checklist
description: Mechanical identifiability quick-check for a model family against the DAG. Parameter-vs-data accounting, dimensionality, f/obs confounding, g boundedness, warm-up. Faster than invoking the identifiability-auditor agent for routine review.
---

# Identifiability Checklist

Read `docs/theoretical_framework.md` §6 before running. The four distributions (`p(P|Z)`, `p(X|Z,P,E)`, `f`, `g`) are NOT jointly identified at framework level; identification relies on per-family restrictions.

## Checklist

1. **Parameter accounting.** List every parameter the family introduces. For each, name the data regime that identifies it:
   - Parameters of `f` → workout-day transitions only.
   - Parameters of `g` → rest-day transitions only (within 10-day bound).
   - Parameters of `p(X | Z, P, E)` → variation in `(P, E)` at matched `Z_{t-1}`.
   - Parameters of `p(P | Z)` → joint over selection and state (required for causal claims, not for filtering).
   Flag any parameter with no clear identification source.
2. **Dimensionality.** `Z ∈ ℝ^d`. Is `d` compatible with the information content of observations? Scalar noisy observation with high CV + large `d` → likely under-determined.
3. **f / obs confounding.** Do `f` and the observation model share parameters (tied coefficients, shared basis, etc.)? If yes, workout-day data identifies a mixture — state what is actually identified.
4. **g boundedness.** Is `g` used only within 10 consecutive rest days (A5)? Past that, mixing recovery and detraining biases `g`.
5. **Constant-X problem (§4d).** Does the model degrade gracefully when `X_t` is near-constant over a window? If not, the `X → Z` edge is unobservable in that regime.
6. **Warm-up.** Is scoring gated to ≥6 months / 1 year per athlete (A8)?
7. **Stationarity (A10).** Does the family assume constant parameters across the window? Are there known regime shifts (injury, long layoff) in the data that violate this?
8. **Exchange symmetries.** Swap two parameter blocks / two `Z` dimensions — do you get an equivalent likelihood? If yes, the family is under-identified without a symmetry-breaking restriction.

## Output

- **Family:** name or one-sentence description.
- **Parameter accounting table:** parameter | identification source | flag.
- **Concerns:** numbered, ordered by severity.
- **Verdict:** identified / partially identified / under-identified / unidentifiable.
- **Fixes:** restrictions, pooling, or priors that would resolve each concern.

For deep audits, escalate to the identifiability-auditor agent.
