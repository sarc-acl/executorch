---

description: "Task list for specs/031-release13-4w-crash-survey"
---

# Tasks: Release/1.3 Vanilla 4w Crash Survey on M5 EVT1 (Floating Clocks)

**Input**: Design documents from `specs/031-release13-4w-crash-survey/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md (all present; no `contracts/` — this feature has no external interface)

**Tests**: Not applicable in the unit/contract-test sense — this feature has no source code. The
"test" of correctness is the runner's own JSON stats line (completed) vs. a device drop-off
(crashed), exactly as defined in `data-model.md`'s Benchmark Attempt.

**Organization**: Tasks are grouped by user story (US1, US2 from `spec.md`), both Priority P1.
Device-execution tasks are inherently sequential (one shared M5 EVT1 board) — `[P]` is only used
where two tasks genuinely touch independent files/state.

## Format: `[ID] [P?] [Story] Description`

## Path Conventions

All artifacts for this feature live under `specs/031-release13-4w-crash-survey/` (docs-and-results
structure, per `plan.md`'s Project Structure — no `src/`/`tests/` tree exists for this feature).

---

## Phase 1: Setup

**Purpose**: Confirm the on-device prerequisites this survey depends on, and create the results
artifact tasks will append to.

- [X] T001 [P] Create `specs/031-release13-4w-crash-survey/results/raw-attempts.md` with the Benchmark
  Attempt table header (fields per `data-model.md`: model, rep_index, driver_md5_pre,
  clock_range_pre, outcome, prefill_tok_s, decode_tok_s, prompt_tokens, generated_tokens,
  crash_event_id) plus an empty Crash Event log section below it.
- [X] T002 [P] Verify `llama_main_rel1.3` and all three `4w` texture `ctx3072` `.pte` files
  (`llama3_2_1b_4w_texture_ctx3072.pte`, `llama3_2_3b_4w_texture_ctx3072.pte`,
  `llama3_1_8b_4w_texture_ctx3072.pte`) are present at `$D` on the M5 EVT1 device (`adb -s $S
  shell ls -la $D/`), and record the confirmation (or what was missing and re-pushed) in
  `specs/031-release13-4w-crash-survey/results/raw-attempts.md`'s provenance section.

**Checkpoint**: Results file exists; all on-device assets confirmed present — ready to start
measurement.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Establish the driver-identity and clock-floating baseline that every model's rep
sequence depends on (Principle VIII / spec FR-004).

**⚠️ CRITICAL**: No model's rep sequence (Phase 3) may start until this passes.

- [X] T003 Verify on-device driver hash via `adb -s $S shell md5sum
  /vendor/lib64/hw/vulkan.samsung.so` matches the documented default
  (`c9861e9906d03fa2c7d48b804e1a1c80`); if it doesn't, stop and escalate per spec Edge Cases rather
  than proceeding. Record the confirmed hash in
  `specs/031-release13-4w-crash-survey/results/raw-attempts.md`.
- [X] T004 Verify (and set if needed) GPU clocks are floating — sysfs `min_freq`/`max_freq` on
  `/sys/class/devfreq/23400000.sgpu/` read `255000`/`980000` — recording the readback in
  `specs/031-release13-4w-crash-survey/results/raw-attempts.md`.

**Checkpoint**: Driver hash and floating-clock state confirmed — model rep sequences (US1) may
begin.

---

## Phase 3: User Story 1 - Establish the crash/normal boundary across model sizes (Priority: P1) 🎯 MVP

**Goal**: Attempt 3 reps each of 3B, 1B, and 8B (in that order — see `research.md`'s run-order
decision) on vanilla `release/1.3`, floating clocks, recording a completed/crashed outcome for
every single attempt, recovering via `fastboot reboot` on any crash.

**Independent Test**: `specs/031-release13-4w-crash-survey/results/raw-attempts.md` contains
exactly 3 recorded attempts for each of 3B, 1B, and 8B, each with either a completed measurement
or a crash record — independently verifiable without reading the report table from US2.

### Implementation for User Story 1

- [X] T005 [US1] Run the 3-rep sequence for **3B** per `quickstart.md` (driver/clock check already
  done in Phase 2 covers the first rep; coherence check with `--prompt='The capital of France is'
  --seq_len=48`; then 3× full `--prompt_file=$D/p2048_exact.txt --max_new_tokens=1024
  --ignore_eos` runs). On any crash: confirm via `lsusb`/`fastboot devices` it's in
  `S5E9975_LK_Bootloader`, run `fastboot -s $S reboot`, poll `sys.boot_completed` until `1`,
  re-verify driver hash + clock range before retrying that rep. Record every attempt (completed or
  crashed) in `specs/031-release13-4w-crash-survey/results/raw-attempts.md`.
- [X] T006 [US1] Run the 3-rep sequence for **1B** per `quickstart.md` (re-verify driver hash +
  clock range first, since T005 may have triggered crash recoveries; same coherence-check +
  3-rep + crash-recovery procedure as T005). Record every attempt in
  `specs/031-release13-4w-crash-survey/results/raw-attempts.md`. Depends on T005 (single shared
  device, sequential).
- [X] T007 [US1] Run the 3-rep sequence for **8B** per `quickstart.md` (re-verify driver hash +
  clock range first; same procedure as T005/T006). Record every attempt in
  `specs/031-release13-4w-crash-survey/results/raw-attempts.md`. Depends on T006 (single shared
  device, sequential).

**Checkpoint**: `raw-attempts.md` has 9 (or more, if retried attempts beyond the first crash are
also logged) recorded Benchmark Attempts covering all three models — User Story 1 is independently
verifiable at this point without US2's report table existing yet.

---

## Phase 4: User Story 2 - Produce the end-of-day report table (Priority: P1)

**Goal**: Turn `raw-attempts.md`'s per-attempt records into the single deliverable report table:
one row per model with prefill/decode tok/s (median) + CoV over completed reps, and an explicit
crash annotation.

**Independent Test**: `specs/031-release13-4w-crash-survey/results/report.md` can be read on its
own (without the raw-attempts log) and tells a reader, for each of 1B/3B/8B, whether it's safe to
benchmark under this exact configuration.

### Implementation for User Story 2

- [X] T008 [US2] Derive the Model Row summary for each of 3B, 1B, 8B from
  `specs/031-release13-4w-crash-survey/results/raw-attempts.md` per `data-model.md`'s Model Row
  fields (completed_count, crashed_count, prefill/decode tok/s median + CoV% over completed
  attempts only, crash_annotation), and write to
  `specs/031-release13-4w-crash-survey/results/report.md`: (a) the headline table, (b) the
  driver-hash provenance note, AND (c) the full raw per-attempt table (every Benchmark Attempt's
  rep_index/outcome/prefill_tok_s/decode_tok_s/driver_md5_pre, copied/embedded from
  `raw-attempts.md`, not merely referenced) — `report.md` MUST be readable and self-contained on
  its own, per FR-006/SC-004 and this story's Independent Test, without requiring a reader to also
  open `raw-attempts.md`.
- [X] T009 [US2] Validate `specs/031-release13-4w-crash-survey/results/report.md` against spec
  Success Criteria SC-001–SC-004 (all 3 models present; every model backed by 3 attempted reps;
  crash/partial-failure patterns explicit; every number traceable to a confirmed-matching driver
  hash) and against `quickstart.md`'s "Expected outcome" checklist; fix any gap found before
  reporting the survey done. Depends on T008.

**Checkpoint**: `report.md` is the complete, self-contained end-of-day deliverable.

---

## Phase 5: Polish & Cross-Cutting Concerns

- [X] T010 [P] Restore GPU clocks to the workspace's pinned default (509/2730/663 MHz per
  `.shared-context/instruction-for-ai/README.md` §Conventions) now that the survey is complete,
  verified via sysfs readback, recorded in
  `specs/031-release13-4w-crash-survey/results/raw-attempts.md`.
- [X] T011 Re-read `specs/031-release13-4w-crash-survey/results/report.md` end to end as if a
  colleague with no session context were seeing it for the first time; fix any place a value,
  acronym, or crash annotation would be ambiguous without this conversation's history.

---

## Phase 6: Extension (Same Day) — 4w Pinned Gap-Fill + Full 8da4w Matrix [US1/US2 continued]

**Goal**: Extend the answered survey to cover the 2 remaining `4w` cells (3B pinned, 8B pinned)
and all 6 `8da4w` cells (3 models × 2 clock policies), using the identical empirical-then-
threshold-fallback methodology, per FR-008/FR-009/SC-005/SC-006.

- [X] T012 [US1] Build fresh `llama_main_nodethresh` from `release13-node-threshold/executorch`
  (two-step Android cross-build per `.shared-context/instruction-for-ai/setup/README.md`),
  verify via `strings <bin> | grep ET_VK_EXECUTE_NODE_THRESHOLD` and arch check, stage to NFS
  (replacing the stale 2026-07-10 build) and push to device.
- [X] T013 [US1] Run 3B pinned 4w: vanilla first (crashed 3/3, confirmed reproducible), then
  `threshold=64` (3/3 completed). Record in `raw-attempts.md`.
- [X] T014 [US1] Re-collect 8B floating 4w fresh at `threshold=64` (3/3 completed, superseding
  the original single `threshold=32` data point per the user's threshold-policy change).
- [X] T015 [US1] Run 8B pinned 4w: `threshold=64` first (crashed 2/2, confirmed insufficient),
  fell back to `threshold=32` (3/3 completed across 2 pre-extension reps + 1 new rep). Record in
  `raw-attempts.md`.
- [X] T016 [US1] Push all three `8da4w` texture `ctx3072` PTEs to device (already exported,
  no new export needed).
- [X] T017 [US1] Run the full `8da4w` matrix (1B/3B/8B × floating/pinned), vanilla first per
  cell, threshold fallback only on confirmed crash (64 for 3B-pinned/8B-floating, 32 for
  8B-pinned per T015's established pattern). Record every attempt in `raw-attempts.md`.
- [X] T018 [US2] Rewrite `report.md` as the full combined, self-contained deliverable: 12
  Model Rows (both quant schemes × both clock policies), headline table, full raw per-attempt
  table, complete crash event log (CE1–CE20), bottom-line synthesis of the
  model/clock/threshold interaction finding.
- [X] T019 Update `spec.md`, `plan.md`, `research.md`, `data-model.md` to reflect the extended
  scope (new FRs/SCs, resolved Complexity Tracking deviation, new data-model fields
  `quant_scheme`/`clocks`/`node_threshold`, extension-specific research decisions) — this task.

**Checkpoint**: All 12 (model × quant × clocks) cells covered, `report.md` is the complete,
self-contained deliverable for the full extended scope.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately.
- **Foundational (Phase 2)**: Depends on Setup — BLOCKS Phase 3.
- **User Story 1 (Phase 3)**: Depends on Foundational. T005 → T006 → T007 strictly sequential
  (one shared device; run order per `research.md`).
- **User Story 2 (Phase 4)**: Depends on User Story 1 being complete (T008 reads the full
  `raw-attempts.md`, including all three models — no partial/incremental report is meaningful
  here, unlike a typical independent-user-story feature).
- **Polish (Phase 5)**: Depends on Phase 4 (T010 can technically run any time after Phase 3, but
  is sequenced last so clocks stay floating for the full duration of measurement).
- **Extension (Phase 6)**: Depends on Phase 5 having completed (the original scope was fully
  answered and the board was left in its default pinned state before the extension began — T012
  re-verifies driver/clock state independently rather than assuming Phase 5's end state persisted
  across the gap). T013→T014→T015 (4w gap-fill) and T016→T017 (8da4w matrix) are each internally
  sequential (one shared device); T018/T019 depend on all of T013–T017 being complete.

### User Story Dependencies

Unlike a typical multi-story feature, **US2 is not independently startable before US1** — the
report table's entire content is derived from US1's raw data (this is explicit in US2's
Independent Test wording and in `data-model.md`'s Model Row derivation). US1 alone is a valid,
independently-verifiable MVP checkpoint (the crash/normal boundary is known even before the
report is written up); US2 adds the presentation layer on top.

### Parallel Opportunities

- T001/T002 (Setup) can run in parallel — different concerns (file creation vs. device asset
  check).
- T010 is marked `[P]` relative to T011 (independent: clock restore vs. report re-read).
- T005/T006/T007 are explicitly **not** parallel — single shared device, and T006/T007 each
  re-verify state that only makes sense after the prior model's sequence (including any crash
  recoveries) has fully settled.

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 (Setup) + Phase 2 (Foundational).
2. Complete Phase 3 (US1) — this alone already answers the original question ("which models crash
   vs. run normally") with repeated evidence, even before the report table exists.
3. **STOP and VALIDATE**: confirm `raw-attempts.md` has 3 recorded attempts per model.

### Incremental Delivery

1. Setup + Foundational → device ready, results file initialized.
2. US1 (T005–T007) → crash/normal boundary known, backed by 3 reps/model → shareable as-is if the
   end-of-day deadline is tight.
3. US2 (T008–T009) → the requested report table exists → full deliverable.
4. Polish (T010–T011) → board left in its default pinned state, report double-checked for a
   reader with no session context.
