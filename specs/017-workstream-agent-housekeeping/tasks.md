---

description: "Task list for Workstream Agent Housekeeping"
---

# Tasks: Workstream Agent Housekeeping

**Input**: Design documents from `specs/017-workstream-agent-housekeeping/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md

**Tests**: Not requested for this feature (documentation-only; verification is the SC-001..SC-004 manual checks in quickstart.md, folded into Polish below).

**Organization**: Tasks are grouped by user story (US1, US2, US3 per spec.md), each independently verifiable per its own quickstart check.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- All paths are relative to `/local/yanwen.xu/workspace/quant-perf-optimization/executorch`

---

## Phase 1: Setup

**Purpose**: Confirm a clean baseline before editing shared, non-feature-scoped files.

- [X] T001 Run `git status --short` and `git diff --stat` at repo root; confirm no pending uncommitted changes to `CLAUDE.md` or `.specify/memory/constitution.md` already exist from other in-flight work before this feature starts editing them.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Gather the exact insertion points and conventions each user story's edit depends on, before touching any file.

**⚠️ CRITICAL**: Complete before starting US1/US2/US3.

- [X] T002 [P] Read `.specify/memory/constitution.md`'s Governance section: record its current version number and the exact Sync Impact Report format used by prior amendments, to reuse verbatim for US3's version bump.
- [X] T003 [P] Read `CLAUDE.md`'s current heading structure (confirm `# ExecuTorch` is the first heading and `## Skills` immediately follows it) to confirm the insertion point research.md Decision 4 specifies for US1.

**Checkpoint**: Insertion points and version-bump format confirmed — user story edits can begin.

---

## Phase 3: User Story 1 - `CLAUDE.md` points a fresh agent at this workstream's real operating knowledge (Priority: P1) 🎯 MVP

**Goal**: A fresh agent reading only `CLAUDE.md` learns, within the first few lines, which document governs this folder, the M5 EVT1 target, and where `.shared-context/` is.

**Independent Test**: open `CLAUDE.md` with no other file loaded; confirm it names the constitution, M5 EVT1, and `.shared-context/` within the first ~20 lines (quickstart.md check 1 / SC-001).

### Implementation for User Story 1

- [X] T004 [US1] Add a pointer block to `CLAUDE.md`, after the `# ExecuTorch` heading and before `## Skills`, naming `.specify/memory/constitution.md` as this folder's governing document, the M5 EVT1 / Samsung Xclipse target, and workspace-root `.shared-context/` (with the one-sentence caveat that a cited `.shared-context/instruction-for-ai/` mechanism can be wrong for this repo, pointing to `.specify/memory/gotchas.md` — per data-model.md's "CLAUDE.md Pointer Block" and research.md Decision 4). Leave all existing content (Skills list, `.wiki/` pointer, Quick Reference, Naming, Commits, Code Style) untouched below it.

**Checkpoint**: `head -20 CLAUDE.md` satisfies quickstart.md check 1 (SC-001) — verify manually before moving on.

---

## Phase 4: User Story 2 - A single, authoritative doc consolidates this session's gotchas (Priority: P1) 🎯 MVP

**Goal**: `.specify/memory/gotchas.md` exists, holding all 10 gotchas (symptom/root cause/fix/citation each) plus a header establishing it as a living, append-as-you-go document.

**Independent Test**: read `.specify/memory/gotchas.md` once; confirm all 10 `G1`-`G10` entries and the append-convention header are present, without needing to cross-reference `specs/014-016` (quickstart.md check 2 / SC-002).

### Implementation for User Story 2

- [X] T005 [US2] Create `.specify/memory/gotchas.md` with a header section (before any `G<N>` entry) explaining: this doc's purpose, that it holds only mechanism-level findings and never volatile/time-sensitive facts (research.md Decision 3), the append convention for future sessions — add a new `G<N+1>` entry, same schema, id never reused even if an entry is later marked resolved (data-model.md Lifecycle; FR-004a) — and a standing note flagging that a future `install_executorch.sh` re-sync could silently overwrite `CLAUDE.md`'s pointer block (spec.md Edge Cases), so a future session notices if it recurs.
- [X] T006 [US2] Append gotcha entries `G1` through `G10` to `.specify/memory/gotchas.md`, each with `symptom` / `root_cause` / `fix_or_workaround` / `citation` / `status`, per data-model.md's G1-G10 table and spec.md FR-003's ten items verbatim (build-install staling, `ET_VK_FORCE_BUFFER` nonexistence, non-editable `.venv` no-op, PTE export location, `/tmp` size + `rm -rf` denial, ETDump `kernel_name` unreliability, the two SDPA harnesses, the prefiltered-grep trap, the pre-hardware-validated retune + mismatched baseline mechanism, M5 EVT1 device sharing). `status: open` for G1, G2, G3, G5, G6, G7, G8, G9, G10; **G4 gets `status: resolved as of Constitution v2.3.0`** — the underlying issue (no single documented PTE-export location) was actually fixed when that rule was added to the constitution, per data-model.md's Lifecycle rule.

**Checkpoint**: `grep -c '^### G' .specify/memory/gotchas.md` returns 10, satisfying quickstart.md check 2 (SC-002) — verify manually before moving on.

---

## Phase 5: User Story 3 - The constitution cross-references the new gotchas doc (Priority: P2)

**Goal**: Constitution Principles VI and X each point to `.specify/memory/gotchas.md`; version bumped with a Sync Impact Report entry.

**Independent Test**: `grep -n "gotchas.md" .specify/memory/constitution.md` returns at least two hits (Principle VI, Principle X) (quickstart.md check 3 / SC-003).

**Depends on**: User Story 2 (T005/T006) — the file being referenced must exist first.

### Implementation for User Story 3

- [X] T007 [US3] Amend Principle VI ("Verify With Tools, Never Assume") in `.specify/memory/constitution.md`: add one to two sentences citing the ETDump per-event `kernel_name`-attribution finding (`G6`) as a concrete example, pointing to `.specify/memory/gotchas.md` for the full list of similar findings.
- [X] T008 [US3] Amend Principle X ("Consult `instruction-for-ai` Before Acting") in `.specify/memory/constitution.md`: add one to two sentences warning that a cited `.shared-context/instruction-for-ai/` mechanism/command can be actively wrong for this repo specifically (citing `G2`, the `ET_VK_FORCE_BUFFER` example), pointing to `.specify/memory/gotchas.md`.
- [X] T009 [US3] Add a short "Gotchas Reference" paragraph to the constitution's Development Workflow (or existing Issue & Open-Question Tracking subsection) introducing `.specify/memory/gotchas.md`, its append convention, and its parallel (not replacement) relationship to `.shared-context/report-for-human/open-questions.md`.
- [X] T010 [US3] Bump `.specify/memory/constitution.md`'s version by MINOR (per its own Governance rule: principle materially expanded) and prepend a new Sync Impact Report entry documenting this amendment (files touched, principles amended, rationale), matching the exact format recorded in T002.

**Checkpoint**: `grep -n "gotchas.md" .specify/memory/constitution.md` returns ≥2 hits, satisfying quickstart.md check 3 (SC-003).

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Confirm the feature's scope boundary held and run the full quickstart end-to-end.

- [X] T011 [P] Run `git status --short` and `git diff --stat` at repo root; confirm exactly `CLAUDE.md` (modified), `.specify/memory/constitution.md` (modified), and `.specify/memory/gotchas.md` (new) are touched outside this feature's own `specs/017-workstream-agent-housekeeping/` artifacts — no production code, shader, build script, or `.shared-context/` file appears (quickstart.md check 4 / SC-004 / FR-006).
- [X] T012 Run all four quickstart.md checks in sequence (`head -20 CLAUDE.md`; `grep -c '^### G'` + header check on `gotchas.md`; `grep -n "gotchas.md"` on `constitution.md`; scope `git diff --stat`) and confirm each passes as documented in quickstart.md's "Expected outcome".

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately.
- **Foundational (Phase 2)**: Depends on Setup — blocks all user stories.
- **User Story 1 (Phase 3)**: Depends on Foundational only — independent of US2/US3.
- **User Story 2 (Phase 4)**: Depends on Foundational only — independent of US1; must complete before US3.
- **User Story 3 (Phase 5)**: Depends on Foundational AND User Story 2 (references the file US2 creates).
- **Polish (Phase 6)**: Depends on US1 + US2 + US3 all being complete.

### Parallel Opportunities

- T002 and T003 (Foundational) touch different files — run in parallel.
- US1 (T004) and US2 (T005-T006) touch different files and have no ordering dependency on each other — can proceed in parallel once Foundational completes.
- US3 (T007-T010) all touch the same file (`constitution.md`) — run sequentially, not in parallel.
- T011 in Polish has no file-write dependency on T012 — could run in parallel, but T012 re-derives the same facts, so sequential is simpler in practice.

---

## Implementation Strategy

### MVP First (User Stories 1 + 2 only)

1. Complete Phase 1 (Setup) and Phase 2 (Foundational).
2. Complete Phase 3 (US1) and Phase 4 (US2) — together these close the exact gap the user described (a fresh agent finds real operating knowledge from `CLAUDE.md`, and the knowledge itself exists in one place).
3. **STOP and VALIDATE**: run quickstart.md checks 1 and 2.
4. This is already a complete, shippable improvement even without US3.

### Incremental Delivery

1. Setup + Foundational → baseline confirmed.
2. Add US1 → validate (check 1) → CLAUDE.md now points somewhere real.
3. Add US2 → validate (check 2) → the gotchas doc it points to now exists and is complete.
4. Add US3 → validate (check 3) → the constitution's own governance discipline now also points to the gotchas doc.
5. Polish → validate (check 4 + full quickstart) → scope boundary confirmed, feature done.
