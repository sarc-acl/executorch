---

description: "Task list for M5 EVT1 End-to-End WMMA Validation"
---

# Tasks: M5 EVT1 End-to-End WMMA Validation (Linear 4w/8da4w + SDPA)

**Input**: Design documents from `/specs/015-m5-e2e-wmma-validation/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md (all present; no `contracts/`, per plan.md's Project Structure)

**Tests**: Not a separate automated suite — this feature's correctness
signal is dispatch confirmation via ETDump (Principle VI), matching how
prior e2e-measurement specs in this workstream (`009`, `011`) validated
inline rather than via a new test phase.

**Organization**: Tasks are grouped by user story per spec.md (US1 =
dispatch-confirm mechanism proof, US2 = linear e2e, US3 = SDPA e2e, US4 =
consolidated report). **Per explicit user instruction, execution within
US2/US3 is sequenced 1B → 3B → 8B** (lowest GPU-watchdog risk first), with
a report/publish task immediately after each model's numbers exist —
never batched until the end. US1's own MVP is proving the whole pipeline
(export → build → deploy → dispatch-confirm → e2e) on the single
lowest-risk configuration (1B, `4w`) before spending device time on the
other eight.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files/resources, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4)
- Paths are relative to the repository root

## Path Conventions

- `.pte_out/` — shared export dir; 4w buffer/texture PTEs for all 3 models already exist, 8da4w buffer PTEs are new
- `cmake-out-android-vk/examples/models/llama/llama_main` — this repo's own runner (NOT `_origcm`, `research.md` Decision 2)
- `cmake-out-android-vk-etdump/` — new build dir for the ETDump-enabled runner variant
- `specs/015-m5-e2e-wmma-validation/results/` — per-model result files, raw capture logs, and the final consolidated report

---

## Phase 1: Setup

- [ ] T001 Create `specs/015-m5-e2e-wmma-validation/results/raw/` directory

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Confirm every mechanism this feature depends on is actually present and current before spending device time on any configuration

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T002 [P] Re-verify M5 EVT1 driver identity (`research.md` Decision 4): `ssh yanwen.xu@sj1-dmckee-d01`, `adb -s 0000088f8e579c33 shell md5sum /vendor/lib64/hw/vulkan.samsung.so`, confirm it matches known-good `f14c51b6f8` (or `c0d117aaf2`) per `.shared-context/instruction-for-ai/flash-sumd-driver.md` — do not assume `specs/014`'s end-of-session state still holds (Principle VIII)
- [ ] T003 [P] Confirm (or build via `./build_etdump_android.sh`) this repo's ETDump-enabled Android runner; verify it's current relative to `HEAD` (`98549f93c` or later)
- [ ] T004 [P] Confirm this repo's `cmake-out-android-vk/examples/models/llama/llama_main` is current relative to `HEAD`; rebuild (`cmake --build cmake-out-android-vk --target install` then the `examples/models/llama` sub-build per `build.md`) if stale
- [ ] T005 [P] Confirm this repo's `.venv` is active and `executorch.extension.llm.export.export_llm` imports cleanly (`research.md` Decision 1)
- [ ] T006 Confirm the six existing `.pte_out/llama3_{1_8b,2_1b,2_3b}_4w_{texture,buffer}_ctx3072.pte` files are present and readable (spot-check file size/existence only — content validity is confirmed later via each configuration's own coherence check)
- [ ] T007 If `/data/vendor/gpu/amdPalSettings.cfg` is present and active on the device, ask the user for explicit approval before moving it aside (`.shared-context/instruction-for-ai/commands.md` §10) — do not do this unilaterally

**Checkpoint**: Foundation ready — driver verified, runners built, export environment confirmed, existing exports confirmed present

---

## Phase 3: User Story 1 - Prove WMMA dispatches on M5 EVT1 from this repo's own build (Priority: P1) 🎯 MVP

**Goal**: Prove the entire pipeline (deploy → coherence → dispatch-confirm → e2e capture) works end-to-end on one representative, lowest-risk configuration before scaling to the other eight.

**Independent Test**: Push 1B's existing `4w` buffer PTE, confirm coherent output, confirm via a separate ETDump run that the linear coopmat kernel actually dispatched, then capture one e2e prefill/decode number — all independent of the other eight configurations.

- [ ] T008 [US1] Stage 1B's `4w` buffer PTE, this repo's `llama_main` + ETDump runner, `tokenizer.model`, and `p2048_exact.txt` to the NFS run-kit, then push all to `$D` on M5 EVT1 (depends on T002-T006)
- [ ] T009 [US1] Coherence check: run 1B/`4w` with a short prompt (`"The capital of France is"`), confirm coherent output before proceeding
- [ ] T010 [US1] Dispatch-confirm 1B/`4w`: separate ETDump run (`--max_new_tokens=4`), pull the trace, run `analyze_etdump_shaders.py`, confirm the linear coopmat kernel family dispatched (not tiled) — record `dispatch_status` per `data-model.md`
- [ ] T011 [US1] E2E capture 1B/`4w`: 2048-token prefill / 1024-token decode, pinned clocks, `ET_VK_EXECUTE_NODE_THRESHOLD=16`; record `prefill_tok_s`/`decode_tok_s` to `results/raw/`
- [ ] T012 [US1] Report 1B/`4w`'s dispatch status and e2e numbers to the user immediately (depends on T009-T011) — do not wait for any other configuration

**Checkpoint**: US1 complete — the full pipeline is proven on one configuration; safe to proceed to the remaining eight

---

## Phase 4: User Story 2 - Measure linear (4w, 8da4w) e2e for all three models (Priority: P2)

**Goal**: Extend US1's proven pipeline to the remaining five linear configurations (1B `8da4w`; 3B and 8B at both schemes), sequenced 1B → 3B → 8B per the user's explicit risk-ordering instruction, publishing each model's results as soon as they exist.

**Independent Test**: For each configuration, produce a dispatch-confirmed e2e prefill/decode tok/s pair (or an explicit blocked/failed status), independent of the other configurations in this phase.

### 1B (remaining: `8da4w`)

- [ ] T013 [P] [US2] Export 1B's `8da4w` buffer PTE: `MODEL=llama3_2_1b MAX_SEQ=3072 MAX_CTX=3072 .shared-context/scripts/export_quant.sh 8da4w 128 buffer` (depends on T005)
- [ ] T014 [US2] Stage + push 1B's `8da4w` PTE to M5 EVT1 (runner/tokenizer/prompt already staged from T008)
- [ ] T015 [US2] Coherence check + dispatch-confirm 1B/`8da4w` (same procedure as T009-T010)
- [ ] T016 [US2] E2E capture 1B/`8da4w` (same procedure as T011)
- [ ] T017 [US2] Publish `results/1b-results.md` (both `4w` from US1 and `8da4w` from T016, each compared against its `data-model.md` Prior-Finding Reference) — report to the user now

### 3B (both schemes)

- [ ] T018 [P] [US2] Export 3B's `8da4w` buffer PTE (`MODEL=llama3_2_3b ...`, depends on T005)
- [ ] T019 [US2] Stage + push 3B's `4w` (existing) and `8da4w` (new) PTEs, plus runner/tokenizer/prompt if not already on-device, to M5 EVT1
- [ ] T020 [US2] Coherence check + dispatch-confirm 3B/`4w`
- [ ] T021 [US2] Coherence check + dispatch-confirm 3B/`8da4w`
- [ ] T022 [US2] E2E capture 3B/`4w`
- [ ] T023 [US2] E2E capture 3B/`8da4w`
- [ ] T024 [US2] Publish `results/3b-results.md` — report to the user now

### 8B (both schemes — highest linear-config watchdog risk)

- [ ] T025 [P] [US2] Export 8B's `8da4w` buffer PTE (default `MODEL=llama3_1_8b`, depends on T005)
- [ ] T026 [US2] Stage + push 8B's `4w` (existing) and `8da4w` (new) PTEs, plus runner/tokenizer/prompt if not already on-device, to M5 EVT1
- [ ] T027 [US2] Coherence check + dispatch-confirm 8B/`4w`
- [ ] T028 [US2] Coherence check + dispatch-confirm 8B/`8da4w`
- [ ] T029 [US2] E2E capture 8B/`4w` at 2048-token prefill; if the GPU-watchdog issue recurs, record `blocked_reason` exactly per `data-model.md`/Edge Cases — do NOT silently retry at a shorter prefill and report that number as the 2048 result
- [ ] T030 [US2] E2E capture 8B/`8da4w`, same watchdog caveat as T029
- [ ] T031 [US2] Publish `results/8b-results.md`'s linear portion (even if one or both entries are `blocked_reason` rather than a number) — report to the user now

**Checkpoint**: US2 complete — all six linear configurations have a recorded result or an explicit blocked reason, published incrementally per model

---

## Phase 5: User Story 3 - Measure SDPA-coopmat e2e for all three models (Priority: P3)

**Goal**: Extend the existing partial M5 EVT1 SDPA-coopmat finding (1B fully measured; 8B/3B previously watchdog-blocked at 2048-prefill) to a complete set where possible, sequenced 1B → 3B → 8B, reusing each model's `4w` buffer PTE with `ET_VK_SDPA_COOPMAT=1`.

**Independent Test**: For each model, produce a dispatch-confirmed SDPA-coopmat e2e prefill/decode tok/s pair (or an explicit blocked status), independent of the linear results already captured for that model.

- [ ] T032 [US3] Dispatch-confirm 1B SDPA-coopmat: ETDump run with `ET_VK_SDPA_COOPMAT=1` + 1B's `4w` buffer PTE, confirm `sdpa_compute_attn_weights_coopmat`/`sdpa_compute_out_coopmat` dispatched (depends on T008)
- [ ] T033 [US3] E2E capture 1B SDPA-coopmat (2048-token prefill / 1024-token decode)
- [ ] T034 [US3] Append 1B's SDPA result to `results/1b-results.md` (already published in T017) — report to the user now
- [ ] T035 [US3] Dispatch-confirm 3B SDPA-coopmat (depends on T019)
- [ ] T036 [US3] E2E capture 3B SDPA-coopmat at 2048-token prefill; if the previously-observed watchdog issue recurs (per the 2026-06-23 session finding), record `blocked_reason` — do not silently substitute the 512-prefill data point from that prior session as if it were this run's 2048 result
- [ ] T037 [US3] Append 3B's SDPA result (or blocked reason) to `results/3b-results.md` — report to the user now
- [ ] T038 [US3] Dispatch-confirm 8B SDPA-coopmat (depends on T026) — highest watchdog-risk configuration in this entire feature
- [ ] T039 [US3] E2E capture 8B SDPA-coopmat at 2048-token prefill, same watchdog caveat as T036
- [ ] T040 [US3] Append 8B's SDPA result (or blocked reason) to `results/8b-results.md` — report to the user now

**Checkpoint**: US3 complete — all three SDPA-coopmat configurations have a recorded result or an explicit blocked reason, published incrementally per model

---

## Phase 6: User Story 4 - Consolidated report (Priority: P4)

**Goal**: Assemble all nine configurations' results into one document with explicit Prior-Finding Reference comparisons and no-prior-baseline flags.

**Independent Test**: Produce the consolidated report from the three already-published per-model files and confirm every comparison is traceable to a specific source document.

- [ ] T041 [US4] Assemble `results/m5-e2e-validation-report.md` from `1b-results.md`/`3b-results.md`/`8b-results.md`, cross-referencing `data-model.md`'s Prior-Finding Reference table; explicitly flag `8da4w` 3B/1B and any watchdog-blocked SDPA configuration as no-prior-baseline, never presented as reproducing a known number (depends on T017, T024, T031, T034, T037, T040)

**Checkpoint**: US4 complete — one document answers "what does this repo's current M5 EVT1 build actually deliver," per configuration, honestly scoped

---

## Phase 7: Polish & Cross-Cutting Concerns

- [ ] T042 Re-read `results/m5-e2e-validation-report.md` and confirm SC-001 through SC-004 are all satisfied: every one of the nine configurations has either a number or a stated blocked reason; every comparison is labeled directional or no-prior-baseline correctly; no number lacks a dispatch-confirmation citation

---

## Dependencies & Execution Order

- **Phase 1 (Setup)** → **Phase 2 (Foundational)**: no dependencies, run first
- **Phase 3 (US1)**: depends on Phase 2; proves the pipeline on 1B/`4w` only — the fastest path to the user's first reported result
- **Phase 4 (US2)**: depends on Phase 3 (reuses its staged runner/tokenizer/prompt and proven procedure); internally sequenced 1B → 3B → 8B per the user's risk-ordering instruction, with a publish task after each model
- **Phase 5 (US3)**: depends on the corresponding model's Phase 4 tasks completing (reuses that model's staged `4w` PTE) but does NOT depend on Phase 4 finishing entirely — 1B's SDPA work (T032-T034) can start as soon as T008 (1B staged) is done, in parallel with 3B/8B's linear work, if device time allows
- **Phase 6 (US4)**: depends on all of Phase 4 and Phase 5 completing
- **Phase 7 (Polish)**: depends on Phase 6

## Parallel Execution Examples

- T002-T005 (Phase 2) touch disjoint concerns and can run in parallel
- T013, T018, T025 (the three `8da4w` exports) touch disjoint output files and can run in parallel, ahead of when each model's on-device work actually needs them
- Once a model's linear work is staged (e.g., T008 for 1B), that model's SDPA dispatch-confirm (T032) can proceed independently of other models' linear work (T018-T031 for 3B/8B) — device-time permitting, these are not strictly sequential across models, only within a model's own linear→SDPA order is device time the real constraint (one device, one adb connection at a time)

## Implementation Strategy

**MVP = User Story 1** (T001-T012): proves the whole pipeline on the
single lowest-risk configuration (1B, `4w`) and reports that result
immediately — the fastest way to get the user their first real number and
catch any pipeline problem (stale build, driver drift, export issue)
before committing device time to the other eight configurations.

**Then, per the user's explicit instruction**: User Story 2's linear
configurations and User Story 3's SDPA configurations both proceed
1B → 3B → 8B, with a publish/report task immediately after each model's
numbers exist (T017, T024, T031 for linear; T034, T037, T040 for SDPA) --
never held back until User Story 4's final consolidated report. 8B (the
highest-risk model for both linear and SDPA at 2048-token prefill) is
deliberately tackled last in both stories, so a watchdog recurrence there
doesn't block the 1B/3B results the user has already received.
