---
name: eval-audit
description: Audit an evaluation pipeline against the §7 prediction factorization. Checks target leakage, reference-template matching, gap handling, warm-up, and weighting. Use before trusting any new harness or when results across specs look suspiciously flat.
---

# Evaluation Audit

Read `docs/theoretical_framework.md` §6 and §7 before running. Per §7, prediction = state-forward-project + observation-at-reference-conditions.

## Checklist

1. **Target.** Is the scorer predicting the unobserved component of `X_{t+τ}` given known `(P_{t+τ}, E_{t+τ})`? If `P` or `E` leak training-time values, the score is biased.
2. **Reference template.** Are source `Z_{t+τ-1}` estimate and target `X_{t+τ}` deconfounded to the **same** reference template (elevation, wet-bulb, distance, ToD)? Mismatch produces invisible bias.
3. **Gap.** Is `τ` a stratifier, not a regressor? Gap is a state-process variable; using it as a covariate in the observation-side deconfounding is a DAG violation.
4. **Warm-up.** Are scored predictions gated to ≥1 year of athlete data (A8)? Any warm-up-window score leaks prior.
5. **Weighting.** If Riegel weights are used, are they on fitting the observation model (correct) or on the evaluation (requires explicit estimand justification)?
6. **Independence.** Is the best-effort rule respected — one effort per `activity_num`? Multiple efforts per session break independence.
7. **Aggregation.** Any max-in-fold or monthly aggregation collapses estimator differences to the noise floor. Flag.
8. **Metric coverage.** Is a single MAE hiding structure? Check residual autocorrelation (ac1), rank correlation (rho), and gap-stratified breakdown.

## Output

- **Pipeline:** path or spec being audited.
- **Target check:** pass/problem — describe.
- **Template check:** pass/problem.
- **Gap check:** pass/problem.
- **Warm-up check:** pass/problem.
- **Weighting / independence / aggregation:** findings.
- **Recommended fixes:** numbered, each with file:line reference.
- **Verdict:** clean / clean with caveats / blocked pending fix.

(No pass/fail labels on the pipeline itself — report what is wrong and its consequence.)
