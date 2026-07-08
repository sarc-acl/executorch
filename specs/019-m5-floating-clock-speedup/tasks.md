---

description: "Task list for M5 EVT1 Floating-Clock Speedup Table"
---

# Tasks: M5 EVT1 Floating-Clock Speedup Table

**Input**: Design documents from `specs/019-m5-floating-clock-speedup/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md

**Tests**: Not requested — this is a hardware measurement feature; verification is the sysfs floating-state check and the quickstart checks below, not a code test suite.

**Organization**: Tasks are grouped by user story (US1=1B, US2=3B, US3=8B, US4=consolidated report, per spec.md's priorities P1/P2/P3/P2).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to
- All device paths use the env block from `.shared-context/instruction-for-ai/README.md` §Conventions (`$S`/`$D`/`$PTE`/`$NFS`/`$SC`)
- All 12 PTEs are reused verbatim from `specs/015-m5-e2e-wmma-validation` (full-stack optimal) and `specs/018-m5-8da4w-t-tiled-baseline` (T-tiled baseline) — no new export in this feature

---

## Phase 1: Setup

**Purpose**: Confirm the shared device is usable and every PTE this feature needs already exists, before touching clock state.

- [X] T001 Confirm M5 EVT1 is free (gotcha G10) and re-verify on-device driver identity (`adb -s $S shell md5sum /vendor/lib64/hw/vulkan.samsung.so`, per constitution Principle VIII).
- [X] T002 [P] Confirm all 12 PTEs are present/re-stageable: 6 T-tiled baselines (`*_4w_texture_ctx3072.pte` x3 models, `*_8da4w_texture_ctx3072.pte` x3 models, from `specs/018`) and 6 full-stack optimal (`*_4w_buffer_ctx3072.pte` / `*_8da4w_buffer_ctx3072.pte` x3 models, from `specs/015`) in `/local/yanwen.xu/workspace/.pte_out`.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Establish and verify a genuinely-floating clock state — this applies once per session and blocks every measurement.

**⚠️ CRITICAL**: Complete before starting any user story. Do not proceed past this phase until T004's sysfs readback confirms floating.

- [X] T003 Read the device's own hardware min/max frequency for GPU, MIF, and INT (e.g. `adb -s $S shell cat /sys/kernel/gpu/available_frequencies` and the `devfreq` equivalents for MIF/INT) — do not assume or hardcode a value (research.md Decision 2).
- [X] T004 Write hardware min → `min_freq` and hardware max → `max_freq` for all three domains (`/sys/kernel/gpu/{min,max}_freq`, `/sys/class/devfreq/23400000.sgpu/{min,max}_freq`, `/sys/class/devfreq/17000010.devfreq_mif/scaling_devfreq_{min,max}`, `/sys/class/devfreq/17000020.devfreq_int/scaling_devfreq_{min,max}`), then read every value back and confirm it reflects the hardware's full range, NOT the pinned 509000/2730000/663000 triple (research.md Decision 3). If any value still matches the pinned triple, stop and re-issue the write — do not proceed.

**Checkpoint**: Genuinely-floating clock state confirmed via sysfs — per-model measurement can begin.

---

## Phase 3: User Story 1 - 1B floating-clock table (Priority: P1) 🎯 MVP

**Goal**: All 4 of 1B's configs (`4w`/`8da4w` × T-tiled baseline/full-stack optimal) measured under confirmed-floating clocks, 3 reps each, per-rep values published.

**Independent Test**: `specs/019-m5-floating-clock-speedup/results/1b-floating-results.md` exists, shows all 3 per-rep values for each of 1B's 4 configs (not just a mean), and each cold-start speedup ratio is computed per research.md Decision 5.

### Implementation for User Story 1

- [X] T005 [US1] Run 3 timed reps of 1B/`4w` T-tiled baseline. **DONE**: prefill 502.823/506.304/502.207, decode 14.7877/15.2065/14.6528, `throttle_observed=false` (<1% spread).
- [X] T006 [US1] Run 3 timed reps of 1B/`4w` full-stack optimal. **DONE**: prefill 979.904/935.587/943.779, decode 14.6128/14.7425/15.111, `throttle_observed=true` (~4.6% spread -- modest, well below the -19%/-27% tiled precedent).
- [X] T007 [US1] Run 3 timed reps of 1B/`8da4w` T-tiled baseline. **DONE**: prefill 389.132/372.161/372.635, decode 15.9851/15.9799/16.1545, `throttle_observed=true` (~4.3% spread).
- [X] T008 [US1] Run 3 timed reps of 1B/`8da4w` full-stack optimal. **DONE**: prefill 806.617/788.299/944.649, decode 15.441/15.1083/15.4637, `throttle_observed=true` (~19.8% spread, non-monotonic -- flagged as anomalous). Note: a truncated/corrupt PTE copy was caught and fixed before this run (adb host `/tmp` filled up mid-`scp`) -- `generated_tokens=1023` confirmed for all 3 reps on the re-verified complete PTE.
- [X] T009 [US1] Compute `speedup_vs_baseline_coldstart` for both schemes. **DONE**: `4w` 1.95x (502.823->979.904), `8da4w` 2.07x (389.132->806.617) -- both lower than their pinned counterparts (2.60x/3.25x), see results file for the DVFS-boost-asymmetry explanation.
- [X] T010 [US1] Publish `specs/019-m5-floating-clock-speedup/results/1b-floating-results.md`. **DONE**.

**Checkpoint**: 1B's floating-clock table is measured, published, and ready to feed into US4.

---

## Phase 4: User Story 2 - 3B floating-clock table (Priority: P2)

**Goal**: Same as User Story 1, for LLaMA 3.2 3B.

**Independent Test**: `results/3b-floating-results.md` exists with the same shape as 1B's.

**Depends on**: Foundational phase only — shares the same physical device with US1, so executes sequentially after it in practice, not concurrently.

### Implementation for User Story 2

- [X] T011 [US2] Run 3 timed reps of 3B/`4w` T-tiled baseline. **DONE**: prefill 190.017/194.271/194.105, decode 6.14507/6.00994/6.05885, `throttle_observed=false` (~2.2% spread).
- [X] T012 [US2] Run 3 timed reps of 3B/`4w` full-stack optimal (`llama3_2_3b_4w_buffer_ctx3072.pte`, `ET_VK_SDPA_COOPMAT=1`) with the same threshold env var. **DONE**: prefill 473.307/500.244/499.878, decode 5.90296/5.91737/5.88283, `throttle_observed=true` (~5.4% spread, rep1 was the low outlier).
- [X] T013 [US2] Run 3 timed reps of 3B/`8da4w` T-tiled baseline (`llama3_2_3b_8da4w_texture_ctx3072.pte`) with the threshold env var. **DONE**: prefill 140.111/139.652/139.32, decode 6.16681/6.24374/6.24413, `throttle_observed=false` (~0.6% spread).
- [X] T014 [US2] Run 3 timed reps of 3B/`8da4w` full-stack optimal (`llama3_2_3b_8da4w_buffer_ctx3072.pte`, `ET_VK_SDPA_COOPMAT=1`) with the threshold env var. **DONE**: prefill 352.072/367.42/359.993, decode 6.05358/5.97439/6.049, `throttle_observed=true` (~4.2% spread).
- [X] T015 [US2] Compute cold-start speedup ratios and `throttle_observed` for 3B's both schemes. **DONE**: `4w` 2.49x (190.017->473.307), `8da4w` 2.51x (140.111->352.072) -- both lower than pinned (2.97x/3.59x), same DVFS-boost-asymmetry pattern as 1B.
- [X] T016 [US2] Publish `results/3b-floating-results.md`. **DONE**.

**Checkpoint**: 3B's floating-clock table is measured, published, and ready to feed into US4.

---

## Phase 5: User Story 3 - 8B floating-clock table (Priority: P3)

**Goal**: Same as User Stories 1-2, for LLaMA 3.1 8B — the model constitution Principle VII's own -19%/-27% throttle observation was originally measured on.

**Independent Test**: `results/8b-floating-results.md` exists, explicitly noting whether the previously-observed throttle magnitude reproduced for the tiled-baseline configs.

**Depends on**: Foundational phase only — sequenced last (slowest model, highest device-time cost, and per gotcha G11 the model most exposed to on-device memory pressure if runs aren't spaced out).

### Implementation for User Story 3

- [X] T017 [US3] Check on-device memory headroom (`/proc/meminfo`, per gotcha G11) before starting 8B's runs; clean up any already-consumed `.etdp`/log files from prior sessions if `MemAvailable` is tight. **DONE**: MemAvailable=8.73 GB, well above caution threshold -- no cleanup needed.
- [ ] T018 [US3] Run 3 timed reps of 8B/`4w` T-tiled baseline (`llama3_1_8b_4w_texture_ctx3072.pte`) with `ET_VK_EXECUTE_NODE_THRESHOLD=16`.
- [ ] T019 [US3] Run 3 timed reps of 8B/`4w` full-stack optimal (`llama3_1_8b_4w_buffer_ctx3072.pte`, `ET_VK_SDPA_COOPMAT=1`) with the threshold env var.
- [ ] T020 [US3] Run 3 timed reps of 8B/`8da4w` T-tiled baseline (`llama3_1_8b_8da4w_texture_ctx3072.pte`) with the threshold env var.
- [ ] T021 [US3] Run 3 timed reps of 8B/`8da4w` full-stack optimal (`llama3_1_8b_8da4w_buffer_ctx3072.pte`, `ET_VK_SDPA_COOPMAT=1`) with the threshold env var.
- [ ] T022 [US3] Compute cold-start speedup ratios for 8B's both schemes; explicitly compare each tiled-baseline config's rep-to-rep spread against Principle VII's documented -19%/-27% precedent and note whether it reproduced.
- [ ] T023 [US3] Publish `results/8b-floating-results.md`.

**Checkpoint**: 8B's floating-clock table is measured, published, and ready to feed into US4.

---

## Phase 6: User Story 4 - Consolidated floating-vs-pinned report (Priority: P2)

**Goal**: A six-row consolidated floating speedup table exists alongside the existing pinned one, with an explicit comparison-basis caveat.

**Independent Test**: `results/floating-vs-pinned-report.md` exists, shows all 6 rows (3 models × 2 schemes) with floating cold-start speedup ratios, links to the pinned report for direct comparison, and states its cold-start-vs-steady-state methodology choice explicitly.

**Depends on**: User Stories 1, 2, AND 3 (needs all three models' floating data to complete the table).

### Implementation for User Story 4

- [ ] T024 [US4] Compile the 6-row consolidated table (model × scheme, T-tiled baseline cold-start, full-stack optimal cold-start, cold-start speedup ratio) from T009/T015/T022's results.
- [ ] T025 [US4] Write the comparison-basis caveat paragraph: cold-start (rep 1) values were used for the ratio per research.md Decision 5, note which configs (if any) showed `throttle_observed=true` and by how much, and point to each per-model results file for the full per-rep data.
- [ ] T026 [US4] Publish `specs/019-m5-floating-clock-speedup/results/floating-vs-pinned-report.md`, explicitly linking to `specs/015`'s and `specs/018`'s pinned tables for side-by-side reading — do not merge into or overwrite the pinned report (FR-003).
- [ ] T027 [US4] Update this feature's own `data-model.md` with the final measured values for all 12 configs.

**Checkpoint**: Floating-clock speedup table fully published, sitting alongside the pinned one.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Restore the default clock state and do a final end-to-end validation.

- [ ] T028 Re-pin clocks back to 509/2730/663 MHz (`pin_freqs.sh`) once all floating measurements are complete — pinned is the constitution's stated default for every reported number going forward; do not leave the device in a floating state for the next session.
- [ ] T029 Run all four `quickstart.md` steps end-to-end as a final check; confirm every published number is labeled floating, every config shows 3 per-rep values (not a mean-only entry), and the consolidated table sits alongside (not replacing) the pinned one.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately.
- **Foundational (Phase 2)**: Depends on Setup — blocks all user stories. Do not proceed past T004 until floating state is sysfs-confirmed.
- **User Story 1 (Phase 3)**: Depends on Foundational only.
- **User Story 2 (Phase 4)**: Depends on Foundational only — structurally independent of US1, but shares the same physical device, so executes after US1 in practice.
- **User Story 3 (Phase 5)**: Same as US2 — depends on Foundational only, sequenced last (highest device-time cost, per research precedent).
- **User Story 4 (Phase 6)**: Depends on User Stories 1, 2, AND 3 all completing.
- **Polish (Phase 7)**: Depends on User Story 4 — T028 (re-pin) should run regardless of whether US4's report tasks are still in progress, to avoid leaving the device floating longer than necessary.

### Parallel Opportunities

- T001 and T002 (Setup) can run in parallel — different concerns, no file overlap.
- T003 (Foundational) must complete before T004 (need the hardware range before writing it) — sequential.
- **US1/US2/US3 are logically independent** (different PTEs, different result rows) but **not practically parallelizable** — all three contend for the same single physical M5 EVT1 device. Run sequentially: US1 (1B) → US2 (3B) → US3 (8B).
- Within each user story, the 4 configs' measurement tasks (e.g. T005-T008) are inherently sequential on shared hardware — no internal parallelism.

---

## Implementation Strategy

### MVP First (User Story 1 only)

1. Complete Phase 1 (Setup) and Phase 2 (Foundational) — including the sysfs floating-state confirmation.
2. Complete Phase 3 (US1 — 1B's 4 configs).
3. **STOP and VALIDATE**: confirm `1b-floating-results.md` shows genuine per-rep spread data and a real cold-start ratio.
4. This alone proves the floating-clock methodology (unpin, verify, per-rep report) works before committing device time to the larger, slower models.

### Incremental Delivery

1. Setup + Foundational → floating state confirmed.
2. Add US1 (1B) → validate → cheapest floating table in hand.
3. Add US2 (3B) → validate → second floating table in hand.
4. Add US3 (8B) → validate → all three raw floating tables in hand, throttle precedent checked.
5. Add US4 (consolidated report) → validate → the actual deliverable (floating table alongside pinned) is done.
6. Polish (re-pin + final quickstart check) → device restored to its default state, feature done.
