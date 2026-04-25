# docs/plans/ — organization schema

Governance doc for `docs/plans/`. One markdown file per sequencing or coordination plan; live-maintained throughout the work it describes.

```
docs/plans/
├── README.md                       ← this file
├── <slug>.md                       ← one plan, e.g. post-scaffold-sequencing.md
└── ...
```

## What belongs here

Plans that sequence multiple milestones, span multiple files or modules, or coordinate decisions over more than one session. A plan is the canonical current state of an in-flight effort.

What does *not* belong here:
- Single-decision records — those go in `docs/decisions/` as ADRs.
- Per-session task lists — those are ephemeral; use the harness's task tracker.
- Aspirational roadmaps with no near-term commitment — write them when work is about to start.

## Required structure

Every plan file has these sections (in order):

1. **`# <Title>`** — descriptive name, not a slug.
2. **`**Status:**`** — one of `active`, `complete`, `superseded by <slug>`, `paused`.
3. **`**Last updated:**`** — ISO date.
4. **`## Current state`** — Completed / Active / Audits landed. Update at every milestone boundary.
5. **`## Constraints`** — load-bearing rules the plan operates under.
6. **`## Milestones`** — dependency-ordered. Each marked done / active / gated.
7. **`## Open decisions`** — table; flip status as decisions resolve.
8. **`## Verification`** — per-milestone exit criteria.
9. **`## Out of scope`** — explicit, so drift is visible.
10. **`## Maintenance`** — the live-update obligations specific to this plan.

## Naming

- Slug describes what is sequenced, not when or by whom: `post-scaffold-sequencing.md`, `evaluation-redesign.md`. No dates in filenames; the **Last updated** line carries that.
- Snake-case-with-hyphens, lowercase.

## Live-maintenance rules (PM-enforced)

- **At each milestone completion:** move the milestone from Active → Completed; record the commit SHA.
- **At each audit landing:** add a line under "Audits landed" with date and one-sentence finding.
- **When a decision resolves:** flip the status in the Open decisions table.
- **When scope drifts:** update Out of scope explicitly before the drift lands in code.
- **Plan edits are their own commits** — small, named, tied to the milestone they record. Do not bundle plan updates with unrelated work.
- **Audit-round budget per planning step: cap at three rounds.** A "round" is one full pass by an auditor (identifiability, senior-scientist, master-statistician, or other Tier-2 specialist) producing findings the plan must address. Diminishing returns past round three: if the third round still produces structural changes — pivots in family choice, fundamental parameterization shifts, identifiability reframings — the plan is fundamentally wrong and must be redrafted from scratch, not iterated further. Narrow asks (mechanical fixes, framing tweaks, cross-references) do not count against the cap. The PM enforces the cap; auditors do not self-cap.

## Retirement

Retired plans stay in `docs/plans/` with `**Status:** complete` or `**Status:** superseded by <slug>`. Do not delete; the historical sequencing record is part of the project's institutional memory.

## Adding a new plan

1. Pick a slug per the naming rules.
2. Copy the required structure above; fill it in.
3. Update CLAUDE.md's Read-First table only if the new plan changes which plan is the *current* canonical one. Otherwise the `docs/plans/` pointer in CLAUDE.md is sufficient.
