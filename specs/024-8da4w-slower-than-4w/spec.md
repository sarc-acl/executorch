# Feature Specification: Why 8da4w Is Slower Than 4w On The Tiled (No-WMMA) Path

**Feature Branch**: `024-8da4w-slower-than-4w`

**Created**: 2026-07-11

**Status**: Answered (mechanism identified; not yet re-verified with a fresh multi-run microbench)

**Input**: Recurring question across this workstream — `specs/025-8da4w-parameter-sweep`
cites this spec's "premise that `8da4w` underperforms `4w` on this hardware" (comparing
`4w`'s autotuned coopmat best, 2518.77 GFLOP/s, against `8da4w`'s, 1731.0 GFLOP/s), and a
2026-07-09 JIRA "personal note" (GFXSW-69499) states "I am still in middle of investigating
why 8da4w is slower than 4w" with a raw microbench table showing `8da4w` tiled consistently
slower than `4w` tiled at every shape. This spec did not exist as a formal artifact until
now — it was referenced by `specs/025` but never created. This document fills that gap.

## Context

Two distinct comparisons get conflated in casual discussion and need to stay separate:

1. **Tiled (no WMMA/coopmat) baseline, per-quant-scheme**: `8da4w` is *slower* than `4w`
   here — this is the subject of this spec.
2. **Coopmat/WMMA e2e**: `8da4w` can be *faster* than `4w` here (e.g. GFXSW-69499
   2026-06-11: B-coopmat 4w=79.3 tok/s vs 8da4w=85.1 tok/s) — real int8-dot hardware
   offsets 8da4w's extra bookkeeping. Not a contradiction with (1); different shader path.

This spec is about (1) only.

## User Scenarios & Testing

### User Story 1 — Understand why 8da4w-tiled underperforms 4w-tiled (Priority: P1)

As an engineer reading the workstream's speedup tables, I need to know why the `8da4w`
tiled baseline is consistently slower than `4w`'s at every GEMM shape and every model size,
so I can explain the ratio in a report without it looking like a measurement error.

**Acceptance criteria**: a shader/dispatch-level mechanism is identified and traced to
actual source (file:line), consistent with the raw microbench data already collected
(GFXSW-69499, 2026-07-09).

## Success Criteria

- **SC-001**: The extra work `8da4w`'s tiled kernel does that `4w`'s tiled kernel doesn't is
  enumerated with source citations. — **Met**, see `research.md`.
- **SC-002**: The mechanism explains why the *tiled* comparison goes one way while the
  *coopmat* comparison can go the other way. — **Met**, see `research.md`.
- **SC-003 (not yet met)**: A fresh, 3-run-mean+CoV microbench (matching this workstream's
  usual rigor, e.g. `specs/023`/`specs/025`'s methodology) confirming the magnitude at all
  13 real Llama GEMM shapes. The existing evidence (JIRA Table 1, 2026-07-09) is single-shot
  per shape, no CoV — sufficient to confirm *direction*, not yet to a report-grade
  confidence level. Flagged as follow-up, out of scope for this pass.
