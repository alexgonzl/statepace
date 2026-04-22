---
name: channel-assignment
description: Decide whether a new observable channel belongs to P (session shape), X (execution), or E (exogenous) per the DAG §A1-A2. Use when adding a new data field, integrating a new signal, or revisiting an ambiguous assignment.
---

# P / X / E Channel Assignment

Read `docs/theoretical_framework.md` §A1, §A2 before running.

## Decision tree

1. **Is the channel selected by the athlete in response to prior state `Z_{t-1}`?**
   - Yes → candidate for `P` (session shape).
   - No → go to 2.
2. **Is the channel produced by the body during execution?**
   - Yes → `X`.
   - No → go to 3.
3. **Is the channel genuinely exogenous (`⊥ Z_{t-1}`, `⊥ P_t`) — weather, time-of-day, ambient conditions not selected for interaction with state?**
   - Yes → `E`.
   - No → the channel does not fit the framework as stated. Escalate before integration.

## Failure modes to check

- **"Exogenous" that is actually selected.** Did the athlete pick the start time to avoid heat? Then ToD is not purely `E` — it has a selection component that partly belongs to `P`. Flag.
- **"P" that is actually executed.** Planned distance vs realized distance: planned is `P`, realized is `X`. If only realized is recorded, the split is unrecoverable — note this.
- **"X" that is actually structural.** Average pace over a long run captures both route and execution; it mixes `P` and `X`. Decompose before assigning.

## Output

- **Channel:** name, source, units.
- **Assignment:** `P` / `X` / `E` with one-sentence rationale.
- **Exogeneity check** (if `E`): `⊥ Z_{t-1}` yes/no; `⊥ P_t` yes/no.
- **Data-contract update:** proposed diff to `docs/data_contract.md`.
- **Downstream consumers to update:** list (observation model, deconfounding, state estimation, eval harness).
- **Risks / caveats.**
