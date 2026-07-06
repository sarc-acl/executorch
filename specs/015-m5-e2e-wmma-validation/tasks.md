---

description: "Task list for M5 EVT1 End-to-End WMMA Validation"
---

# Tasks: M5 EVT1 End-to-End WMMA Validation (Linear 4w/8da4w + SDPA)

**Input**: Design documents from `/specs/015-m5-e2e-wmma-validation/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md (all present; no `contracts/`, per plan.md's Project Structure)

**Revision note**: This task list was regenerated after `/speckit-analyze`
found one CRITICAL gap (no clock-pin verification task, contradicting
constitution Principle VII) and one HIGH gap (single-run e2e capture,
contradicting this workstream's established 3-run-mean/CoV methodology
per `.shared-context/report-for-human/e2e-spec.md`). Both are fixed below
(`research.md` Decision 5). Task granularity is also now uniform
(coherence-check and dispatch-confirm are always separate tasks, matching
the original US1 pattern that a first draft inconsistently collapsed for
US2/US3).

**Tests**: Not a separate automated suite — this feature's correctness
signal is dispatch confirmation via ETDump (Principle VI), matching how
prior e2e-measurement specs in this workstream (`009`, `011`) validated
inline rather than via a new test phase.

**Organization**: Tasks are grouped by user story per spec.md (US1 =
dispatch-confirm mechanism proof, US2 = linear e2e, US3 = SDPA e2e, US4 =
consolidated report). **US1's MVP scope is narrower than spec.md's own
Acceptance Scenario 2 might suggest**: US1 proves the pipeline on ONE
configuration (1B/`4w`) only; the "each [scheme/SDPA] independently
confirms dispatch" behavior spec.md's AS2 describes is fulfilled
*cumulatively* across US1 (1B/`4w`) + US2 (the other five linear
configurations) + US3 (all three SDPA configurations) — every
configuration still gets its own independent dispatch check before its
timing is trusted, just not all within the phase literally labeled "US1."
**Per explicit user instruction, execution within US2/US3 is sequenced
1B → 3B → 8B** (lowest GPU-watchdog risk first), with a report/publish
task immediately after each model's numbers exist — never batched until
the end.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files/resources, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4)
- Paths are relative to the repository root

## Path Conventions

- `.pte_out/` — shared export dir; the three `4w` **Buffer** PTEs (one per model) already exist and are reused; matching `Texture3D` exports also exist but are unused by this feature; the three `8da4w` Buffer PTEs are new
- `cmake-out-android-vk/examples/models/llama/llama_main` — this repo's own runner (NOT `_origcm`, `research.md` Decision 2)
- `cmake-out-android-vk-etdump/` — new build dir for the ETDump-enabled runner variant
- `specs/015-m5-e2e-wmma-validation/results/` — per-model result files, raw capture logs, and the final consolidated report

---

## Phase 1: Setup

- [X] T001 Create `specs/015-m5-e2e-wmma-validation/results/raw/` directory

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Confirm every mechanism this feature depends on is actually present, current, and *verified* (not just commanded) before spending device time on any configuration

**⚠️ CRITICAL**: No user story work can begin until this phase is complete — in particular, T007 (clock-pin verification) gates every e2e capture task in Phases 3-5

- [X] T002 [P] Re-verify M5 EVT1 driver identity (`research.md` Decision 4): `ssh yanwen.xu@sj1-dmckee-d01`, `adb -s 0000088f8e579c33 shell md5sum /vendor/lib64/hw/vulkan.samsung.so`, confirm it matches known-good `f14c51b6f8` (or `c0d117aaf2`) per `.shared-context/instruction-for-ai/flash-sumd-driver.md` — do not assume `specs/014`'s end-of-session state still holds (Principle VIII) — **DONE**: md5 `c9861e9906d03fa2c7d48b804e1a1c80`, exact match for `f14c51b6f8`, no drift since `specs/014`
- [X] T003 [P] Confirm (or build via `./build_etdump_android.sh`) this repo's ETDump-enabled Android runner; verify it's current relative to `HEAD` (`98549f93c` or later) — **DONE, with a real catch**: `build_etdump_android.sh` hardcodes `cd /local/yanwen.xu/workspace/quant-dev/executorch` (line 3) -- running it as-is would have silently built the ETDump runner in the WRONG worktree, exactly the `_origcm`-style trap `research.md` Decision 2 warns about, just baked into a script instead of a doc example. Built manually instead, in this repo, at `cmake-out-android-vk-etdump/examples/models/llama/llama_main` (aarch64, confirmed via `file`)
- [X] T004 [P] Confirm this repo's `cmake-out-android-vk/examples/models/llama/llama_main` is current relative to `HEAD`; rebuild (`cmake --build cmake-out-android-vk --target install` then the `examples/models/llama` sub-build per `build.md`) if stale — **DONE, found genuinely stale**: `llama_main`'s mtime (2026-06-30) predated `libvulkan_backend.a`'s last reinstall (2026-07-05, during `specs/014`'s T009) -- the static binary did not contain current shader code. Rebuilt; `llama_main` mtime now postdates the library
- [X] T005 [P] Confirm this repo's `.venv` is active and `executorch.extension.llm.export.export_llm` imports cleanly (`research.md` Decision 1) — **DONE**, imports cleanly
- [X] T006 Confirm the three existing `.pte_out/llama3_{1_8b,2_1b,2_3b}_4w_buffer_ctx3072.pte` files are present and readable (content validity is confirmed later via each configuration's own coherence check; the matching `Texture3D` exports are not checked here since this feature never uses them) — **DONE, all three present (5.98GB/1.69GB/3.24GB), but presence was NOT sufficient**: US1's dispatch-confirm step (T011) found all of them were internally broken -- see `research.md` Decision 6. "Present and readable" is not the same as "correctly Buffer-storage"; all three needed re-export
- [X] T007 **[Principle VII, CRITICAL per `/speckit-analyze` D1]** Pin GPU/MIF/INT clocks via `pin_freqs.sh` (509/2730/663 MHz) on the adb host, THEN verify the pin actually bound by cross-checking GFLOP/s (or e2e tok/s) against an equivalently-pinned reference measurement (e.g. `test_coopmat_linear_bench`'s own perf numbers) — per constitution Principle VII and the Q10 precedent (a ~980MHz DVFS-boost number was once mistaken for a 509MHz pin). Do NOT proceed to any task in Phase 3-5 until this is confirmed. Clocks are not persistent across reboots — if the device reboots at any point in this feature's work, repeat this task before resuming any capture. — **DONE**: `pin_freqs.sh` commanded 509/2730/663 MHz, sysfs readback confirmed min=max=509000/2730000/663000. Cross-check: `test_coopmat_linear_bench_spec014`'s `linear_q4gsw` coopmat GFLOP/s dropped from 424.6/434.1 (K=2048/4096, measured unpinned at end of `specs/014`) to 228.6/229.6 now pinned -- a ~1.86x reduction closely matching the expected ~1.93x ratio between the board's ~980MHz boost ceiling and the 509MHz pin. This is real, quantitative confirmation the pin took effect, not just a sysfs write that silently no-op'd
- [X] T008 If `/data/vendor/gpu/amdPalSettings.cfg` is present and active on the device, ask the user for explicit approval before moving it aside (`.shared-context/instruction-for-ai/commands.md` §10) — do not do this unilaterally — **DONE**: file does not exist on this device, nothing to move

**Checkpoint**: Foundation ready — driver verified, runners built, export environment confirmed, existing exports confirmed present, clock pin verified

---

## Phase 3: User Story 1 - Prove WMMA dispatches on M5 EVT1 from this repo's own build (Priority: P1) 🎯 MVP

**Goal**: Prove the entire pipeline (deploy → coherence → dispatch-confirm → 3-run e2e capture) works end-to-end on one representative, lowest-risk configuration before scaling to the other eight.

**Independent Test**: Push 1B's existing `4w` buffer PTE, confirm coherent output, confirm via a separate ETDump run that the linear coopmat kernel actually dispatched, then capture a 3-run e2e prefill/decode mean — all independent of the other eight configurations.

- [X] T009 [US1] Stage 1B's `4w` buffer PTE, this repo's `llama_main` + ETDump runner, `tokenizer.model`, and `p2048_exact.txt` to the NFS run-kit, then push all to `$D` on M5 EVT1 (depends on T002-T006) — **DONE**, plus **T009a (new, found during this task)**: `build_etdump_android.sh` hardcodes `cd` into the `quant-dev` worktree (same class of bug as `research.md` Decision 2) -- built the ETDump runner manually in this repo instead; `llama_main` (non-ETDump) was also found stale (linked before `libvulkan_backend.a`'s last reinstall) and rebuilt
- [X] T010 [US1] Coherence check: run 1B/`4w` with a short prompt (`"The capital of France is"`), confirm coherent output before proceeding — **DONE**, coherent ("Paris...")
- [X] T011 [US1] Dispatch-confirm 1B/`4w`: separate ETDump run (`--max_new_tokens=4`), pull the trace, run `analyze_etdump_shaders.py`, confirm the linear coopmat kernel family dispatched (not tiled) — record `dispatch_status` per `data-model.md` — **DONE, and this is where the real finding happened**: first attempt showed `linear_q4gsw_tiled_texture3d_texture2d_half` dispatching 112/112 times and the whole main graph in `_texture3d_half` -- coopmat was NOT dispatching despite using the "buffer" PTE. Root-caused to two stacked issues (`research.md` Decision 6): (1) this repo's venv was non-editable (stale 2026-06-30 site-packages snapshot, missing AOT fixes) -- fixed via `pip install -e . --no-build-isolation`; (2) `ET_VK_FORCE_BUFFER` (what `export_quant.sh`/`export-pte.md` use) does not exist anywhere in this repo's source -- it's a `quant-dev`-only mechanism. This repo's real mechanism is `backend.vulkan.storage_override: buffer` in `config.yaml`. Re-exported 1B's `4w` buffer PTE with the correct mechanism; re-ran dispatch-confirm: `linear_q4gsw_coopmat_buffer_texture2d_half` now dispatches 112/112 times, whole graph is `_buffer_half`, `dispatch_status = confirmed`
- [X] T012 [US1] E2E capture 1B/`4w` (depends on T007, T011): **3 repeated runs**, each 2048-token prefill / 1024-token decode, `ET_VK_EXECUTE_NODE_THRESHOLD=16`; record `prefill_tok_s`/`decode_tok_s` per run to `results/raw/`, then compute and record the mean + CoV per `research.md` Decision 5 — **DONE**: prefill 585.31/583.642/582.149 (mean 583.70, CoV 0.271%), decode 14.2374/14.2493/14.3332 (mean 14.273, CoV 0.366%) -- very tight, consistent with a genuinely pinned clock
- [X] T013 [US1] Report 1B/`4w`'s dispatch status and e2e mean/CoV to the user immediately (depends on T010-T012) — do not wait for any other configuration — **DONE**

**Checkpoint**: US1 complete — the full pipeline (including clock-pin verification and 3-run capture) is proven on one configuration; safe to proceed to the remaining eight

---

## Phase 4: User Story 2 - Measure linear (4w, 8da4w) e2e for all three models (Priority: P2)

**Goal**: Extend US1's proven pipeline to the remaining five linear configurations (1B `8da4w`; 3B and 8B at both schemes), sequenced 1B → 3B → 8B per the user's explicit risk-ordering instruction, publishing each model's results as soon as they exist.

**Independent Test**: For each configuration, produce a dispatch-confirmed, 3-run e2e prefill/decode mean+CoV (or an explicit blocked/failed status), independent of the other configurations in this phase.

### 1B (remaining: `8da4w`)

- [ ] T014 [P] [US2] Export 1B's `8da4w` buffer PTE: `MODEL=llama3_2_1b MAX_SEQ=3072 MAX_CTX=3072 .shared-context/scripts/export_quant.sh 8da4w 128 buffer` (depends on T005)
- [ ] T015 [US2] Stage + push 1B's `8da4w` PTE to M5 EVT1 (runner/tokenizer/prompt already staged from T009)
- [ ] T016 [US2] Coherence check 1B/`8da4w` (same procedure as T010)
- [ ] T017 [US2] Dispatch-confirm 1B/`8da4w` (same procedure as T011)
- [ ] T018 [US2] E2E capture 1B/`8da4w`: 3 repeated runs, mean + CoV (same procedure as T012)
- [ ] T019 [US2] Publish `results/1b-results.md` (both `4w` from US1 and `8da4w` from T018, each compared against its `data-model.md` Prior-Finding Reference) — report to the user now

### 3B (both schemes)

- [ ] T020 [P] [US2] Export 3B's `8da4w` buffer PTE (`MODEL=llama3_2_3b ...`, depends on T005) -- use `backend.vulkan.storage_override: buffer` in `config.yaml` (`research.md` Decision 6), NOT `export_quant.sh`'s `ET_VK_FORCE_BUFFER` (a no-op in this repo)
- [ ] T020a [P] [US2] **Re-export** 3B's `4w` buffer PTE the same way (the existing `.pte_out/llama3_2_3b_4w_buffer_ctx3072.pte`, dated 2026-06-17, was produced with the broken `ET_VK_FORCE_BUFFER` mechanism per `research.md` Decision 6 and is internally Texture3D despite its name -- do not reuse it as-is)
- [ ] T021 [US2] Stage + push 3B's `4w` (re-exported, T020a) and `8da4w` (new, T020) PTEs, plus runner/tokenizer/prompt if not already on-device, to M5 EVT1
- [ ] T022 [US2] Coherence check 3B/`4w`
- [ ] T023 [US2] Dispatch-confirm 3B/`4w`
- [ ] T024 [US2] Coherence check 3B/`8da4w`
- [ ] T025 [US2] Dispatch-confirm 3B/`8da4w`
- [ ] T026 [US2] E2E capture 3B/`4w`: 3 repeated runs, mean + CoV
- [ ] T027 [US2] E2E capture 3B/`8da4w`: 3 repeated runs, mean + CoV
- [ ] T028 [US2] Publish `results/3b-results.md` — report to the user now

### 8B (both schemes — highest linear-config watchdog risk)

- [ ] T029 [P] [US2] Export 8B's `8da4w` buffer PTE (default `MODEL=llama3_1_8b`, depends on T005) -- use `backend.vulkan.storage_override: buffer` in `config.yaml` (`research.md` Decision 6), NOT `export_quant.sh`'s `ET_VK_FORCE_BUFFER` (a no-op in this repo)
- [ ] T029a [P] [US2] **Re-export** 8B's `4w` buffer PTE the same way (the existing `.pte_out/llama3_1_8b_4w_buffer_ctx3072.pte`, dated 2026-06-22, was produced with the broken `ET_VK_FORCE_BUFFER` mechanism per `research.md` Decision 6 and is internally Texture3D despite its name -- do not reuse it as-is)
- [ ] T030 [US2] Stage + push 8B's `4w` (re-exported, T029a) and `8da4w` (new, T029) PTEs, plus runner/tokenizer/prompt if not already on-device, to M5 EVT1
- [ ] T031 [US2] Coherence check 8B/`4w`
- [ ] T032 [US2] Dispatch-confirm 8B/`4w`
- [ ] T033 [US2] Coherence check 8B/`8da4w`
- [ ] T034 [US2] Dispatch-confirm 8B/`8da4w`
- [ ] T035 [US2] E2E capture 8B/`4w`: 3 repeated runs at 2048-token prefill, mean + CoV; if the GPU-watchdog issue recurs on any of the 3 runs, record `blocked_reason` exactly per `data-model.md`/Edge Cases and report however many of the 3 runs completed — do NOT silently retry at a shorter prefill and report that number as the 2048 result
- [ ] T036 [US2] E2E capture 8B/`8da4w`: same 3-run procedure and watchdog caveat as T035
- [ ] T037 [US2] Publish `results/8b-results.md`'s linear portion (even if one or both entries are `blocked_reason` rather than a number) — report to the user now

**Checkpoint**: US2 complete — all six linear configurations have a recorded 3-run mean+CoV result or an explicit blocked reason, published incrementally per model

---

## Phase 5: User Story 3 - Measure SDPA-coopmat e2e for all three models (Priority: P3)

**Goal**: Extend the existing partial M5 EVT1 SDPA-coopmat finding (1B fully measured; 8B/3B previously watchdog-blocked at 2048-prefill) to a complete set where possible, sequenced 1B → 3B → 8B, reusing each model's `4w` buffer PTE with `ET_VK_SDPA_COOPMAT=1`.

**Independent Test**: For each model, produce a dispatch-confirmed, 3-run SDPA-coopmat e2e prefill/decode mean+CoV (or an explicit blocked status), independent of the linear results already captured for that model.

- [ ] T038 [US3] Dispatch-confirm 1B SDPA-coopmat: ETDump run with `ET_VK_SDPA_COOPMAT=1` + 1B's `4w` buffer PTE, confirm `sdpa_compute_attn_weights_coopmat`/`sdpa_compute_out_coopmat` dispatched (depends on T009)
- [ ] T039 [US3] E2E capture 1B SDPA-coopmat: 3 repeated runs (2048-token prefill / 1024-token decode), mean + CoV
- [ ] T040 [US3] Append 1B's SDPA result to `results/1b-results.md` (already published in T019) — report to the user now
- [ ] T041 [US3] Dispatch-confirm 3B SDPA-coopmat (depends on T021)
- [ ] T042 [US3] E2E capture 3B SDPA-coopmat: 3 repeated runs at 2048-token prefill, mean + CoV; if the previously-observed watchdog issue recurs (per the 2026-06-23 session finding), record `blocked_reason` — do not silently substitute the 512-prefill data point from that prior session as if it were this run's 2048 result
- [ ] T043 [US3] Append 3B's SDPA result (or blocked reason) to `results/3b-results.md` — report to the user now
- [ ] T044 [US3] Dispatch-confirm 8B SDPA-coopmat (depends on T030) — highest watchdog-risk configuration in this entire feature
- [ ] T045 [US3] E2E capture 8B SDPA-coopmat: 3 repeated runs at 2048-token prefill, mean + CoV; same watchdog caveat as T042
- [ ] T046 [US3] Append 8B's SDPA result (or blocked reason) to `results/8b-results.md` — report to the user now

**Checkpoint**: US3 complete — all three SDPA-coopmat configurations have a recorded 3-run mean+CoV result or an explicit blocked reason, published incrementally per model

---

## Phase 6: User Story 4 - Consolidated report (Priority: P4)

**Goal**: Assemble all nine configurations' results into one document with explicit Prior-Finding Reference comparisons and no-prior-baseline flags.

**Independent Test**: Produce the consolidated report from the three already-published per-model files and confirm every comparison is traceable to a specific source document.

- [ ] T047 [US4] Assemble `results/m5-e2e-validation-report.md` from `1b-results.md`/`3b-results.md`/`8b-results.md`, cross-referencing `data-model.md`'s Prior-Finding Reference table; explicitly flag `8da4w` 3B/1B and any watchdog-blocked SDPA configuration as no-prior-baseline, never presented as reproducing a known number (depends on T019, T028, T037, T040, T043, T046)

**Checkpoint**: US4 complete — one document answers "what does this repo's current M5 EVT1 build actually deliver," per configuration, honestly scoped

---

## Phase 7: Polish & Cross-Cutting Concerns

- [ ] T048 Re-read `results/m5-e2e-validation-report.md` and confirm SC-001 through SC-004 are all satisfied: every one of the nine configurations has either a 3-run mean+CoV number or a stated blocked reason; every comparison is labeled directional or no-prior-baseline correctly; no number lacks a dispatch-confirmation citation, a verified clock pin, or a CoV

---

## Dependencies & Execution Order

- **Phase 1 (Setup)** → **Phase 2 (Foundational)**: no dependencies, run first
- **Phase 3 (US1)**: depends on Phase 2 (including T007's clock-pin verification); proves the pipeline on 1B/`4w` only — the fastest path to the user's first reported result
- **Phase 4 (US2)**: depends on Phase 3 (reuses its staged runner/tokenizer/prompt and proven procedure); internally sequenced 1B → 3B → 8B per the user's risk-ordering instruction, with a publish task after each model
- **Phase 5 (US3)**: depends on the corresponding model's Phase 4 staging task completing (reuses that model's staged `4w` PTE) but does NOT depend on Phase 4 finishing entirely — 1B's SDPA work (T038-T040) can start as soon as T009 (1B staged) is done, in parallel with 3B/8B's linear work, if device time allows
- **Phase 6 (US4)**: depends on all of Phase 4 and Phase 5 completing
- **Phase 7 (Polish)**: depends on Phase 6

## Parallel Execution Examples

- T002-T006 (Phase 2) touch disjoint concerns and can run in parallel; T007 (clock pin) and T008 (profiler check) can also run in parallel with them, but T007 must complete before any task in Phase 3 onward
- T014, T020, T029 (the three `8da4w` exports) touch disjoint output files and can run in parallel, ahead of when each model's on-device work actually needs them
- Once a model's linear work is staged (e.g., T009 for 1B), that model's SDPA dispatch-confirm (T038) can proceed independently of other models' linear work (T020-T037 for 3B/8B) — device-time permitting, these are not strictly sequential across models, only within a model's own linear→SDPA order is device time the real constraint (one device, one adb connection at a time)

## Implementation Strategy

**MVP = User Story 1** (T001-T013): proves the whole pipeline — including
the clock-pin verification and 3-run capture methodology corrected during
`/speckit-analyze` — on the single lowest-risk configuration (1B, `4w`)
and reports that result immediately. This is the fastest way to get the
user their first real, trustworthy number and catch any pipeline problem
(stale build, driver drift, an unverified clock pin, export issue) before
committing device time to the other eight configurations.

**Then, per the user's explicit instruction**: User Story 2's linear
configurations and User Story 3's SDPA configurations both proceed
1B → 3B → 8B, with a publish/report task immediately after each model's
numbers exist (T019, T028, T037 for linear; T040, T043, T046 for SDPA) --
never held back until User Story 4's final consolidated report. 8B (the
highest-risk model for both linear and SDPA at 2048-token prefill) is
deliberately tackled last in both stories, so a watchdog recurrence there
doesn't block the 1B/3B results the user has already received.
