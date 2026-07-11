---

description: "Task list for MiniPC RDNA3 Handoff Report"
---

# Tasks: MiniPC RDNA3 Handoff Report

**Input**: Design documents from `/specs/013-minipc-handoff-report/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md (all present; no `contracts/`, by design -- no external interface)

**Tests**: Not applicable -- this feature produces a document, not code.
Validation is spot-checking cited numbers against source files (FR-002),
covered in Polish.

**Organization**: Tasks are grouped by user story. This is the
lightest-weight feature in the workstream alongside `011` -- one report
file, no scripts, no build targets. Per the user's own guidance, the
report stays high-level; tasks here are correspondingly light.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files/sections, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)

## Path Conventions

- `specs/013-minipc-handoff-report/results/handoff-report.md` — the single deliverable
- `specs/001-minipc-baseline-benchmarks/` through `specs/012-decode-wmma-feasibility/` — read-only sources, cited not modified

---

## Phase 1: Setup

- [X] T001 Create `specs/013-minipc-handoff-report/results/` directory

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Confirm the grounding already established during planning is
still current -- there is no code to write, only sourcing to verify.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T002 [P] Re-confirm each of the twelve headline numbers in research.md Decision 1 against their source results files (still current, no drift since planning) — all reconfirmed exact matches (007, 010, 011, 012 grep verbatim; 009's source wording is "X% faster"/"Y% slower" without +/- signs, factually identical to research.md's citation)
- [X] T003 [P] Re-confirm the repo state (`git branch --show-current`, `git log --oneline -5`, `git status --short`) is still consistent with research.md Decision 3 — reconfirmed identical: branch `quant-perf-optimization`, last commit `d8800fb02e`, 71 changed files

**Checkpoint**: Foundation ready — all source numbers and repo state reconfirmed current

---

## Phase 3: User Story 1 - One consolidated findings report, not twelve spec folders (Priority: P1) 🎯 MVP

**Goal**: One document synthesizing every finding from `001`-`012`.

**Independent Test**: Someone who hasn't read the twelve prior specs can
read only this report and correctly state which schemes benefit from
linear coopmat, whether SDPA coopmat helps at e2e, and whether decode is
worth accelerating.

- [X] T004 [US1] Write the Consolidated Findings table into `specs/013-minipc-handoff-report/results/handoff-report.md`: one row per spec `001`-`012`, each with its headline, tier (microbenchmark/e2e/n/a), and cited source file (research.md Decision 1, data-model.md) (depends on Foundational checkpoint)
- [X] T005 [US1] Write the Open Items section into the report: the deferred `8da4w` gating decision, decode SDPA's unverified-but-expected-identical conclusion, the two not-attempted directions (further quantization, batching/speculative decoding), and Samsung/Xclipse validation itself as still-pending (research.md Decision 2) (depends on T004) -- all 4 items written, now traced to FR-003 directly (amended per /speckit-analyze finding I1)

**Checkpoint**: US1 complete — the consolidated report answers the core
questions on its own

---

## Phase 4: User Story 2 - Repo state and prerequisites (Priority: P2)

**Goal**: State plainly what a clone of this repo would and wouldn't
contain today.

**Independent Test**: `git status`/`git log` match the report's
description exactly.

- [X] T006 [US2] Write the Repo Handoff State section into the report: current branch, last commit, uncommitted file count and scope (research.md Decision 3), and commit/push named as the required prerequisite -- not performed by this task (depends on T003)

**Checkpoint**: US2 complete — the repo's actual state is visible before
anyone clones it elsewhere

---

## Phase 5: User Story 3 - A starting runbook for the Samsung/Xclipse machine (Priority: P3)

**Goal**: A starting checklist, not a tested pipeline, for what differs
moving to an `adb`-deployed Android target.

**Independent Test**: The runbook names, for each methodology step,
whether it carries over unchanged, needs adaptation, or must be newly
established -- without asserting untested commands as proven.

- [X] T007 [US3] Write the Samsung/Xclipse Runbook section into the report: one row per methodology step (export, build, dispatch-confirm, benchmark, report; data-model.md), each tagged carries-over-unchanged / needs-adaptation / newly-established (depends on T004)
- [X] T008 [US3] Within the runbook, name confirming Xclipse's cooperative-matrix support and tile dimensions as the first newly-established check, ahead of assuming any RDNA3 finding transfers (FR-006) (depends on T007)

**Checkpoint**: US3 complete — there's a starting point for the next
session instead of a blank page

---

## Phase 6: Polish & Cross-Cutting Concerns

- [X] T009 Spot-check every headline number in the report against its cited source file and confirm an exact match (FR-002, SC-002) -- all 8 numeric claims (007 x2, 008, 009 x2, 010, 011, 012) plus 004's case counts and 006's cross-session framing verified as exact matches, zero transcription errors
- [X] T010 Self-review against SC-001 through SC-004: a naive reader gets the right answers from the report alone; every number traces to its source; the repo-clone consequence is stated plainly; every runbook step's carry-over status is clear (depends on T005, T006, T008) -- all four pass: SC-001 (the "In one sentence" summary + table answer all three questions directly), SC-002 (T009), SC-003 (Repo Handoff State names the exact clone consequence and the commit/push prerequisite), SC-004 (every Runbook row has an explicit status, "newly established" rows say so plainly rather than asserting an untested command)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies
- **Foundational (Phase 2)**: Depends on Setup. BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational checkpoint
- **User Story 2 (Phase 4)**: Depends on T003 only -- independent of US1
- **User Story 3 (Phase 5)**: Depends on T004 (writes into the same report
  file after US1's table exists)
- **Polish (Phase 6)**: Depends on US1, US2, US3

### Parallel Opportunities

- T002, T003 (Foundational) can run in parallel
- T006 (US2) has no dependency on US1's own tasks beyond T003, and could
  proceed alongside T004/T005 if desired -- all three user stories write
  into different sections of the same file, so treat as logically
  parallel even though they share one output file

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1 — the consolidated findings, on their own, already deliver most of this feature's value
4. **STOP and VALIDATE**: Confirm the report alone answers the core
   questions before adding the repo-state and runbook sections

### Incremental Delivery

1. Setup + Foundational → sourcing reconfirmed
2. US1 → the findings, consolidated
3. US2 → repo state made visible
4. US3 → a starting runbook, not a blank page
5. Polish → spot-check + final self-review

---

## Notes

- No commits until the user explicitly asks, per repo convention -- this
  applies doubly here, since US2's own content is *about* the fact that
  nothing has been committed yet.
- Per the user's guidance, keep the report high-level throughout --
  resist the urge to over-specify untested Android/`adb` commands in the
  runbook; pointers and carry-over status are sufficient (Clarifications).
