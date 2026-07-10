---

description: "Task list for M5 EVT1 8da4w T-tiled Baseline"
---

# Tasks: M5 EVT1 8da4w T-tiled Baseline

**Input**: Design documents from `specs/018-m5-8da4w-t-tiled-baseline/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md

**Tests**: Not requested — this is a hardware measurement feature; verification is the dispatch-confirmation and quickstart checks below, not a code test suite.

**Organization**: Tasks are grouped by user story (US1=1B, US2=3B, US3=8B, US4=report update, per spec.md's priorities P1/P2/P3/P2).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to
- All device paths use the env block from `.shared-context/instruction-for-ai/README.md` §Conventions (`$S`/`$D`/`$PTE`/`$NFS`/`$SC`)

---

## Phase 1: Setup

**Purpose**: Confirm the shared device is actually usable before spending any export/measurement effort.

- [X] T001 Confirm M5 EVT1 is free (gotcha G10 — confirm with the user, don't assume continuity from a prior session) and re-verify on-device driver identity (`adb -s $S shell md5sum /vendor/lib64/hw/vulkan.samsung.so`, expect current hash in `.shared-context/ACTIVE-STATUS.md`, per constitution Principle VIII).
- [X] T002 Pin clocks (`pin_freqs.sh`) and verify the pin bound via a quick GFLOP/s cross-check against an already-recorded pinned baseline (constitution Principle VII).

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Confirm the export mechanism and runner are ready before touching any model.

**⚠️ CRITICAL**: Complete before starting any user story.

- [X] T003 Locate this repo's existing `8da4w` buffer-storage export config (used for the already-exported `*_8da4w_buffer_ctx3072.pte` files) and confirm the only change needed for a T-tiled export is omitting `backend.vulkan.storage_override` entirely (research.md Decision 1) — do not set it to any value, including `texture`, since omission is what the codebase's default behavior actually is.
- [X] T004 Confirm this repo's already-built `llama_main` and ETDump-enabled runner are current and staged on `$D` (no rebuild needed — this feature changes no source, per plan.md's Technical Context).

**Checkpoint**: Export config and runner confirmed — per-model work can begin.

---

## Phase 3: User Story 1 - 1B `8da4w` T-tiled baseline (Priority: P1) 🎯 MVP

**Goal**: A real, measured, dispatch-confirmed `8da4w` T-tiled prefill/decode tok/s number exists for LLaMA 3.2 1B.

**Independent Test**: `llama3_2_1b_8da4w_texture_ctx3072.pte` exists in `.pte_out`, was measured 3x at the standard workload, and its dispatch is confirmed tiled via ETDump.

### Implementation for User Story 1

- [X] T005 [US1] Export `llama3_2_1b_8da4w_texture_ctx3072.pte` into `/local/yanwen.xu/workspace/.pte_out` using the default (no `storage_override`) config confirmed in T003 (constitution Default Scope — export lands directly in `.pte_out`, never `/tmp`/scratch, per gotcha G4).
- [X] T006 [US1] Push the new PTE plus `tokenizer.model`/`p2048_exact.txt` (if not already staged) to `$D` on M5 EVT1.
- [X] T007 [US1] Run 3 timed reps at the standard 2048-prefill/1024-decode workload (`--warmup=true`, matching the existing `4w` T-tiled baseline's methodology per research.md Decision 5); record prefill/decode tok/s per rep. **DONE**: prefill 221.597/222.754/222.536, decode 13.6946/13.907/13.9095
- [X] T008 [US1] Compute the 3-run mean and CoV for both prefill and decode tok/s. **DONE**: prefill mean=222.30 (CoV=0.28%), decode mean=13.84
- [X] T009 [US1] Run one separate, short (`--max_new_tokens=4 --warmup=false`) ETDump capture for dispatch confirmation, per Principle IV's "never the same run used for the reported number." **DONE**
- [X] T010 [US1] Analyze the capture (`analyze_etdump_shaders.py --by kernel`); confirm the linear kernel family is 100% `linear_dq8ca_q4gsw_tiled_*` with zero `_coopmat_` entries (research.md Decision 4). **DONE**: `linear_dq8ca_q4gsw_tiled_texture3d_texture2d_half` 112/112, zero coopmat entries; `dispatch_status=tiled_confirmed`.

**Checkpoint**: 1B's T-tiled `8da4w` baseline is measured, dispatch-confirmed, and ready to feed into US4.

---

## Phase 4: User Story 2 - 3B `8da4w` T-tiled baseline (Priority: P2)

**Goal**: Same as User Story 1, for LLaMA 3.2 3B.

**Independent Test**: `llama3_2_3b_8da4w_texture_ctx3072.pte` exists and is measured/confirmed the same way as US1.

**Depends on**: Foundational phase only (not on US1's completion) — but shares the same physical M5 EVT1 device, so in practice runs *after* US1 sequentially, not concurrently (research.md Decision 3).

### Implementation for User Story 2

- [X] T011 [US2] Export `llama3_2_3b_8da4w_texture_ctx3072.pte` into `.pte_out` (same config as T005).
- [X] T012 [US2] Push the new PTE to `$D`.
- [X] T013 [US2] Run 3 timed reps at the standard workload **with `ET_VK_EXECUTE_NODE_THRESHOLD=16`** (jira ticket #001 documents the 2048-prefill GPU watchdog risk for 8B **and** 3B, not 8B-only, per spec.md Edge Cases); record prefill/decode tok/s per rep. **DONE**: prefill 79.8472/80.0625/79.785, decode 6.80489/6.84656/6.87024
- [X] T014 [US2] Compute the 3-run mean and CoV. **DONE**: prefill mean=79.83 (CoV=0.21%), decode mean=6.84
- [X] T015 [US2] Run the separate short ETDump dispatch-confirmation capture. **DONE**
- [X] T016 [US2] Analyze; confirm 100% tiled dispatch, zero coopmat entries. **DONE**: `linear_dq8ca_q4gsw_tiled_texture3d_texture2d_half` 196/196, zero coopmat; `dispatch_status=tiled_confirmed`.

**Checkpoint**: 3B's T-tiled `8da4w` baseline is measured, dispatch-confirmed, and ready to feed into US4.

---

## Phase 5: User Story 3 - 8B `8da4w` T-tiled baseline (Priority: P3)

**Goal**: Same as User Stories 1-2, for LLaMA 3.1 8B — including replacing the existing, wrong-context-length `_ctx2304` texture PTE (research.md Decision 2).

**Independent Test**: `llama3_1_8b_8da4w_texture_ctx3072.pte` exists and is measured/confirmed the same way as US1-2, with the established 8B watchdog workaround applied.

**Depends on**: Foundational phase only — sequenced last per research.md Decision 3 (highest device-time cost, highest watchdog risk).

### Implementation for User Story 3

- [X] T017 [US3] Export `llama3_1_8b_8da4w_texture_ctx3072.pte` into `.pte_out` (same config as T005/T011) — this replaces the stale `llama3_1_8b_8da4w_texture_ctx2304.pte`, which is the wrong context length for this workload and must not be reused (research.md Decision 2).
- [X] T018 [US3] Push the new PTE to `$D`.
- [X] T019 [US3] Run 3 timed reps at the standard workload with `ET_VK_EXECUTE_NODE_THRESHOLD=16` (established 8B prefill-watchdog workaround, spec.md Edge Cases); record prefill/decode tok/s per rep. **DONE**: prefill 35.1365/35.2259/35.1624, decode 3.84753/3.84734/3.85054
- [X] T020 [US3] Compute the 3-run mean and CoV. **DONE**: prefill mean=35.17 (CoV=0.13%), decode mean=3.85
- [X] T021 [US3] Run the separate short ETDump dispatch-confirmation capture (with the same threshold env var). **DONE**
- [X] T022 [US3] Analyze; confirm 100% tiled dispatch, zero coopmat entries. **DONE**: `linear_dq8ca_q4gsw_tiled_texture3d_texture2d_half` 224/224, zero coopmat; `dispatch_status=tiled_confirmed`.

**Checkpoint**: 8B's T-tiled `8da4w` baseline is measured, dispatch-confirmed, and ready to feed into US4.

---

## Phase 6: User Story 4 - Speedup table shows real ratios for all six configs (Priority: P2)

**Goal**: Every `8da4w` row in the consolidated report shows a real speedup ratio, closing the gap this feature exists to close.

**Independent Test**: `m5-e2e-validation-report.md` and the per-model `results/*.md` files show a numeric `<baseline> -> <optimized>, N.NNx` for every `8da4w` row — zero "no baseline yet" cells remain.

**Depends on**: User Stories 1, 2, AND 3 (needs all three baselines to complete the table).

### Implementation for User Story 4

- [X] T023 [US4] Compute `speedup_vs_optimized` for each model: existing optimized full-stack tok/s (1B 723.00, 3B 286.31, 8B 130.05, per `specs/015-m5-e2e-wmma-validation/data-model.md`) divided by this feature's own T009/T015/T021-confirmed baseline means.
- [X] T024 [US4] Update **both** `8da4w` rows per model in `specs/015-m5-e2e-wmma-validation/results/1b-results.md`, `3b-results.md`, and `8b-results.md` with the real baseline number and computed ratio: the full-stack (linear+SDPA) row this feature was triggered by, AND the pre-existing linear-only `8da4w` row (currently "None -- no prior M5 EVT1 `8da4w` baseline exists") -- same baseline number serves both (spec.md FR-005).
- [X] T025 [US4] Update `specs/015-m5-e2e-wmma-validation/results/m5-e2e-validation-report.md`'s consolidated 12-row table so all six `8da4w`-labeled rows (three linear-only, three full-stack) show a real ratio, matching the format already used for every `4w` row.
- [X] T026 [US4] Update this feature's own `data-model.md` seeded-rows table with the final measured values (pte_status=exported, dispatch_status=tiled_confirmed, populated tok/s fields).

**Checkpoint**: Report fully concluded — every `8da4w` row has a real ratio.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Final validation that the feature actually closed the gap it set out to close.

- [X] T027 Run all four `quickstart.md` steps end-to-end as a final check; confirm the "Expected outcome" (zero "no baseline yet" cells) holds. **DONE**: found and fixed one stale reference in `m5-e2e-validation-report.md`'s Comparison-type transparency section; zero remain now.
- [X] T028 Confirm no `unexpected_coopmat` dispatch occurred across all three models (T010/T016/T022). **DONE**: 112/112, 196/196, 224/224 all tiled, zero coopmat entries -- no new gotcha/open-question needed.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately.
- **Foundational (Phase 2)**: Depends on Setup — blocks all user stories.
- **User Story 1 (Phase 3)**: Depends on Foundational only.
- **User Story 2 (Phase 4)**: Depends on Foundational only — structurally independent of US1, but shares the same physical M5 EVT1 device, so executes *after* US1 in practice (one adb session, sequential, per research.md Decision 3), not concurrently.
- **User Story 3 (Phase 5)**: Same as US2 — depends on Foundational only, executes last due to shared device + highest device-time cost.
- **User Story 4 (Phase 6)**: Depends on User Stories 1, 2, AND 3 all completing — needs all three baselines to fill the table.
- **Polish (Phase 7)**: Depends on User Story 4.

### Parallel Opportunities

- T001/T002 (Setup) are sequential (clock pin verification depends on the device being confirmed free/correct first).
- T003/T004 (Foundational) can run in parallel — different concerns, no file overlap.
- **US1/US2/US3 are logically independent** (different PTEs, different result rows) but **not practically parallelizable** — all three contend for the same single physical M5 EVT1 device. Run them sequentially: US1 (1B) → US2 (3B) → US3 (8B), per research.md Decision 3.
- Within each user story, the sequence is inherently sequential (export → push → run → analyze) — no internal parallelism.

---

## Implementation Strategy

### MVP First (User Story 1 only)

1. Complete Phase 1 (Setup) and Phase 2 (Foundational).
2. Complete Phase 3 (US1 — 1B baseline).
3. **STOP and VALIDATE**: confirm the 1B row in `data-model.md` shows `tiled_confirmed` with a real mean+CoV.
4. This alone proves the export+measure+verify methodology works before spending device time on the larger models.

### Incremental Delivery

1. Setup + Foundational → device and methodology confirmed ready.
2. Add US1 (1B) → validate → cheapest baseline in hand.
3. Add US2 (3B) → validate → second baseline in hand.
4. Add US3 (8B) → validate → all three raw baselines in hand.
5. Add US4 (report update) → validate → the actual deliverable (a fully concluded report) is done.
6. Polish → final end-to-end confirmation.
