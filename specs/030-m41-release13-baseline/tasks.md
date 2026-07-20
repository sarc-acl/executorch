---

description: "Task list for M41 Release/1.3 Baseline Clock & Quant-Mode Study"
---

# Tasks: M41 Release/1.3 Baseline Clock & Quant-Mode Study

**Input**: Design documents from `specs/030-m41-release13-baseline/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md

**Tests**: Not requested — this is a hardware measurement feature; verification is the
crash-attribution, pin-verification, and quickstart checks below, not a code test suite.

**Organization**: Tasks are grouped by user story (US1=preserve existing 4w-floating data +
driver hash, US2=4w-pinned sweep, US3=8da4w floating+pinned sweep, per spec.md's priorities
P1/P2/P3).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to
- All device paths use `S=000009b44fd4abd3` on `ssh xgpusw-debug07`; device working dir
  `/data/local/tmp/llama_vk/`

---

## Phase 1: Setup

**Purpose**: Confirm the device, driver, and staged assets before spending any measurement time.

- [X] T001 Confirm M41 (`S=000009b44fd4abd3`) is reachable and re-verify on-device driver identity (`adb -s $S shell md5sum /vendor/lib64/hw/vulkan.samsung.so`) — expect `d5d76f1bacf404b1a07d87ec8e479bdf`, and record that no documented known-good reference hash exists for this SoC family (spec FR-001).
- [X] T002 Confirm all 6 PTEs + `tokenizer.model` + `p2048_exact.txt` are staged in `/data/local/tmp/llama_vk/` on-device (`adb -s $S shell ls -la /data/local/tmp/llama_vk/`) — re-push from `/sarc-c/gpusw/users/yanwen.xu/android-run/models/` only if anything is missing.
- [X] T003 [P] Create `specs/030-m41-release13-baseline/results/m41-release13-baseline-report.md` skeleton with the SC-006 "M41 is a secondary/cross-device reference, not Samsung M5 EVT1" statement, a one-line methodology note recording storage type (texture/T-tiled) and runner binary (`llama_main_rel1.3`) per FR-008, and four empty table shells (4w-pinned, 4w-floating, 8da4w-pinned, 8da4w-floating), each pre-sized for 3 models × 3 reps with three possible per-cell states — number, "CRASHED", or "DVFS-ARTIFACT" (spec FR-012).

**Checkpoint**: Device/driver/assets confirmed, report skeleton exists — measurement work can begin.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Resolve the node-threshold question and get clocks into a known, verified state
before any user story's real reps run.

**⚠️ CRITICAL**: Complete before starting User Story 2 or 3 (User Story 1 has no device
dependency beyond T001 and can proceed in parallel with this phase).

- [X] T004 Run the one-throwaway-run node-threshold probe (research.md Decision 3): 8B pinned with `ET_VK_EXECUTE_NODE_THRESHOLD=16` set, not counted as a reported rep — record whether `llama_main_rel1.3` recognizes the env var at all (crashes the same way vs. behaves differently).
- [X] T005 Pin clocks (`S=000009b44fd4abd3 pin_freqs.sh`) and confirm via sysfs readback (`cat .../23400000.sgpu/{min,max}_freq`) that the pin values took effect — first half of spec FR-009's two-part verification (second half is the per-run throughput cross-check in US2/US3).
- [X] T006 [P] Record the already-probed HW devfreq ranges (sgpu 226000–980000, mif 676000–5333000, int 133000–800000) in the report skeleton's methodology section, for use when unpinning to floating in US3.

**Checkpoint**: Node-threshold behavior known, clocks pinned and sysfs-verified — US2/US3's real reps can begin.

---

## Phase 3: User Story 1 - Preserve already-collected baseline with device provenance (Priority: P1) 🎯 MVP

**Goal**: The report contains a trustworthy, complete record of the driver hash and the
already-collected 4w-floating dataset, including the one documented crash.

**Independent Test**: Reading the report confirms the driver md5, device identity, and all 9
already-run (3 models × 3 reps) 4w-floating prefill/decode numbers, matching this session's raw
command log.

**Depends on**: Setup (T001 driver hash, T003 report skeleton) — does not need Phase 2's
pin/threshold work, since it reports pre-existing floating data, not a new run.

### Implementation for User Story 1

- [X] T007 [US1] Write the FR-001 driver-hash statement into the report (from T001): current hash, "no documented known-good reference for s5e9965" note.
- [X] T008 [US1] Fill the 4w-floating table's 9 rep-cells (data-model.md's "Known values" table: 1B×3 ok, 3B×3 ok, 8B rep1/rep3 ok, 8B rep2 CRASHED) per spec FR-012's number-or-CRASHED format.
- [X] T009 [US1] Compute and record CoV (FR-011) for 1B and 3B (n=3) and 8B (n=2, from its 2 valid reps) prefill/decode tok/s; add the floating-mean thermal-drift caveat required by FR-007.
- [X] T010 [US1] Apply research.md Decision 4's `dmesg`/`/proc/meminfo` retroactive check to 8B rep 2's crash if logs are still available on-device; update `crash_cause` from `unknown` to `gpu_watchdog`/`host_oom` if the evidence supports it, otherwise leave `unknown` and say so explicitly rather than guessing.
- [X] T011 [US1] Add the FR-010/SC-006 "M41 is a secondary/cross-device reference, not the M5 EVT1 active target" statement immediately before the first results table (may already exist from T003's skeleton — confirm it's actually filled in, not still a placeholder).

**Checkpoint**: 4w-floating table complete and trustworthy; driver provenance recorded — this alone is a valid, deliverable increment even if US2/US3 run out of time.

---

## Phase 4: User Story 2 - Compare pinned vs. floating clocks for the 4w baseline (Priority: P2)

**Goal**: All 9 rep-cells of the 4w-pinned table are filled (number, CRASHED, or DVFS-ARTIFACT),
each pinned run verified via FR-009, and the pinned-vs-floating comparison is stated.

**Independent Test**: Running the pinned-clock 4w-texture baseline for 1B/3B/8B (3 reps each,
continuing through any crash) produces 9 rep-cells that are each a number, CRASHED, or
DVFS-ARTIFACT, with CoV computed from ≥2 `outcome=ok` reps.

**Depends on**: Phase 2 (Foundational) complete. Practically sequential with US3 — both contend
for the same single physical M41 device (no true parallelism between user stories).

### Implementation for User Story 2

- [X] T012 [US2] Run 3 pinned reps for 1B on `llama3_2_1b_4w_texture_ctx3072.pte`; record prefill/decode tok/s per rep (or CRASHED, per FR-006 — continue regardless).
- [X] T013 [US2] Run 3 pinned reps for 3B on `llama3_2_3b_4w_texture_ctx3072.pte`, applying T004's node-threshold finding only if it was confirmed both present and needed for this model per gotcha G12's per-config logic (G12 found 3B T-tiled is *harmed* by the threshold on M5 EVT1 — do not blanket-apply).
- [X] T014 [US2] Run 3 pinned reps for 8B on `llama3_1_8b_4w_texture_ctx3072.pte`, applying T004's node-threshold finding if it helps (G12 found 8B T-tiled *requires* the threshold on M5 EVT1 — likely relevant here too, but confirmed via T004, not assumed).
- [X] T015 [US2] For every successful pinned rep from T012–T014, apply FR-009's throughput cross-check against the corresponding already-collected 4w-floating number (data-model.md's known values): if prefill_tok_s exceeds 70% of the floating number, set that Run's `outcome=dvfs_artifact` (not `ok`) and record it in the report as "DVFS-ARTIFACT" with its measured throughput, per FR-012 — it is excluded from that cell's mean/CoV either way.
- [X] T016 [US2] For every crash from T012–T014, run research.md Decision 4's `dmesg`/`/proc/meminfo` attribution check and confirm device responsiveness before the next rep — never pause the sweep for it (FR-006).
- [X] T017 [US2] Compute per-model CoV (FR-011) for the 4w-pinned cells with ≥2 valid reps; write the complete 4w-pinned table (9 cells) into the report per FR-012.
- [X] T018 [US2] Write the pinned-vs-floating comparison narrative for 4w into the report (relative difference per model, and whether pinned was viable at all for each model size) — spec Acceptance Scenario 2.

**Checkpoint**: 4w-pinned table complete; 4w quant-mode fully covered across both clock modes.

---

## Phase 5: User Story 3 - Extend the pinned/floating comparison to 8da4w (Priority: P3)

**Goal**: All 18 rep-cells across the 8da4w-floating and 8da4w-pinned tables are filled, and all
four tables (spec SC-007) are complete.

**Independent Test**: Running the 8da4w-texture baseline for 1B/3B/8B at both pinned and floating
clocks (3 reps each) produces 6 ModelSummaries' worth of rep data, each showing all 3 reps as a
number, CRASHED, or (pinned only) DVFS-ARTIFACT, with CoV where applicable.

**Depends on**: Phase 2 (Foundational) for the pinned portion; the floating portion only needs
T001/T002 (device/assets) plus an unpin step (T006's recorded HW ranges). Practically sequential
with US2 (shared device).

### Implementation for User Story 3

- [X] T019 [US3] Unpin clocks to floating using T006's recorded HW min/max devfreq values; run 3 floating reps each for 1B/3B/8B on the `8da4w_texture_ctx3072.pte` files.
- [X] T020 [US3] For any crash in T019, apply research.md Decision 4's `dmesg`/`meminfo` attribution check; continue to the next rep without pausing (FR-006).
- [X] T021 [US3] Compute per-model CoV for the 8da4w-floating cells with ≥2 valid reps; write the complete 8da4w-floating table (9 cells) with the FR-007 thermal-drift-caveated mean.
- [X] T022 [US3] Re-pin clocks (`pin_freqs.sh`, re-verify sysfs per T005's method); run 3 pinned reps each for 1B/3B/8B on the `8da4w_texture_ctx3072.pte` files, applying T004's node-threshold finding per model (per-config, not blanket, per G12).
- [X] T023 [US3] For every successful 8da4w-pinned rep, apply FR-009's throughput cross-check against T021's just-collected 8da4w-floating numbers (same 70% threshold as T015) — mark any rep exceeding it `outcome=dvfs_artifact`/"DVFS-ARTIFACT" per FR-012; for any crash, apply the Decision 4 attribution check and continue (FR-006).
- [X] T024 [US3] Compute per-model CoV for the 8da4w-pinned cells with ≥2 valid reps; write the complete 8da4w-pinned table (9 cells).
- [X] T025 [US3] Confirm all four tables use the same 3 models and comparable format (spec Acceptance Scenario 2) — no schema drift between tables written in different phases.

**Checkpoint**: All four tables (36 rep-slots) complete — the full deliverable exists.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final validation that the report actually satisfies spec SC-007 before calling this done.

- [X] T026 Run quickstart.md's full validation pass: confirm all 36 rep-slots across the 4 tables show a number, "CRASHED", or "DVFS-ARTIFACT" (FR-012), with zero omitted cells.
- [X] T027 Confirm every "pinned" cell in the final report has both a sysfs pin-readback record and a throughput cross-check on file (SC-005) — no cell labeled pinned on sysfs-write success alone.
- [X] T028 Confirm the SC-006 M41-secondary-device framing statement appears before/alongside the report's first table.
- [X] T029 Timestamp the report's completion and confirm it landed within the end-of-day 2026-07-14 target (SC-007); if any cell remains unmeasured at that point, say so explicitly in the report rather than silently omitting it.
- [X] T030 Confirm SC-004 self-containment: re-read the finished report as if this session's conversation never happened — it must explain the device, driver, workload, and clock-mode meanings on its own, with no unstated assumption carried over from chat context.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately.
- **Foundational (Phase 2)**: Depends on Setup — blocks User Stories 2 and 3 (not US1).
- **User Story 1 (Phase 3)**: Depends on T001 only — can run in parallel with Phase 2.
- **User Story 2 (Phase 4)**: Depends on Foundational (Phase 2) — practically sequential with US3 (shared device).
- **User Story 3 (Phase 5)**: Depends on Foundational (Phase 2) — practically sequential with US2 (shared device); its pinned portion (T022–T024) additionally depends on its own floating portion (T019–T021) completing first, since T023's cross-check needs T021's numbers.
- **Polish (Phase 6)**: Depends on User Stories 1, 2, AND 3 all completing — needs all four tables to validate completeness.

### Parallel Opportunities

- T003/T006 (file/report-skeleton work) can run in parallel with device-side tasks — no file overlap, no device dependency.
- **US1 is independent of Phase 2** and can be completed first/in-parallel while Phase 2's device setup happens, since it only reports pre-existing data.
- **US2 and US3 are logically independent** (different quant modes) but **not practically parallelizable** — both contend for the same single physical M41 device. Run sequentially: US2 (4w-pinned) → US3 (8da4w-floating → 8da4w-pinned), per the shared-device pattern already established in this workstream's other measurement specs (e.g. `specs/018`).
- Within US3, the floating portion must complete before the pinned portion (T023 needs T021's numbers for the cross-check) — no internal parallelism there either.

---

## Implementation Strategy

### MVP First (User Story 1 only)

1. Complete Phase 1 (Setup).
2. Complete Phase 3 (US1 — preserve the already-collected 4w-floating data + driver hash).
3. **STOP and VALIDATE**: confirm the report's 4w-floating table and driver-hash section are complete and accurate.
4. This alone delivers the lowest-risk, already-in-hand value even if device time runs out before US2/US3.

### Incremental Delivery

1. Setup (+ Foundational, in parallel with US1) → device/driver/assets confirmed, node-threshold question resolved, clocks pinned and verified.
2. Add US1 → validate → driver provenance + existing floating data locked in.
3. Add US2 (4w-pinned) → validate → 4w quant-mode fully covered both clock modes.
4. Add US3 (8da4w-floating, then 8da4w-pinned) → validate → all four tables complete.
5. Polish → final SC-007 completeness/deadline confirmation.

### Sequential Device-Time Strategy

Since almost every task after Phase 1 shares one physical device, there is no multi-developer
parallel strategy here (unlike a typical multi-service feature) — the real constraint is device
time, not staffing. Order (US1 → Foundational → US2 → US3-floating → US3-pinned → Polish)
minimizes clock-mode toggling (only 2 pin/unpin transitions total: pin for US2+early Foundational,
unpin for US3's floating half, re-pin for US3's pinned half).
