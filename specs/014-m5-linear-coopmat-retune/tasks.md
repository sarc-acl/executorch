---

description: "Task list for M5 EVT1 4w Linear Coopmat Retune"
---

# Tasks: M5 EVT1 `4w` Linear Coopmat Retune (fp16 Accumulate, Loop Flattening, Vectorized Dequant)

**Input**: Design documents from `/specs/014-m5-linear-coopmat-retune/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md (all present; no `contracts/`, per plan.md's Project Structure)

**Tests**: Not a separate automated suite — this feature's correctness
signal reuses (and, per FR-008, extends) the existing
`test_coopmat_linear_bench.cpp` coopmat correctness harness
(`research.md` Decision 2, revised), matching how prior specs in this
workstream validated shader changes inline rather than via a new test
phase.

**Organization**: Tasks are grouped by user story. **All phases are now
DONE.** Phase 3 (US1) committed the four changes; Phase 4 (T006-T009)
closed the FR-008 correctness-harness gap, along the way fixing a
device-access misconception (M5 EVT1 is on `sj1-dmckee-d01` via `ssh`, not
local `adb`), an unrecognized 5th driver build (backed up, reflashed to
known-good `f14c51b6f8`), and a stale-`vulkan_backend` link failure (per
`.shared-context/instruction-for-ai/build.md`'s two-step recipe). Running
T009 on the real device produced this feature's actual result: **all three
shader changes PASS correctness at production K=2048/4096 on verified-good
hardware+driver** (T011, T014). The performance A/B (T010, T012, T015) was
explicitly decided against (user, 2026-07-05) — no throughput claim is
made for any of the three changes, so FR-004's gate (correctness before
*reporting* a number) is satisfied without producing one. All three
changes' final disposition is `keep`.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files/resources, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Paths are relative to the repository root

## Path Conventions

- `backends/vulkan/runtime/graph/ops/glsl/linear_qw_coopmat.glsl` — the three code changes
- `backends/vulkan/runtime/graph/ops/glsl/linear_dq8ca_qw_coopmat.glsl`, `backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp` — the documentation-only change
- `backends/vulkan/test/custom_ops/test_coopmat_linear_bench.cpp` — FR-008's correctness-harness extension (new `kCorrectnessShapes`/`kRank3CorrectnessShapes` entries at production K)
- `specs/014-m5-linear-coopmat-retune/results/` — validation logs and disposition summary

---

## Phase 1: Setup

- [X] T001 Create `specs/014-m5-linear-coopmat-retune/results/` directory

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Confirm the working-tree diff is fully and only accounted for by the four described changes before committing anything

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T002 Diff `backends/vulkan/runtime/graph/ops/glsl/linear_qw_coopmat.glsl`, `backends/vulkan/runtime/graph/ops/glsl/linear_dq8ca_qw_coopmat.glsl`, and `backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp` against `HEAD` (`01fb136d6`) and confirm every hunk maps to exactly one of: fp16 accumulator, loop-shape flattening, vectorized dequant, or the documentation-only note — confirmed: `linear_qw_coopmat.glsl`'s hunks are the three interleaved shader changes (accumulator type at init/store sites, flattened `last`-conditioned loop replacing separate prologue/epilogue, vectorized `dequant_block`, plus the `bias_sh`/bias-tile type follow-through required by the accumulator change and stale-comment cleanup); `linear_dq8ca_qw_coopmat.glsl` and `QuantizedLinear.cpp` are comment-only
- [X] T003 [P] Confirm the unrelated `backends/cadence/utils/FACTO` submodule dirty state is out of scope for this feature (per spec Assumptions) and left untouched — confirmed not staged, not modified by any task below

**Checkpoint**: Foundation ready — every uncommitted hunk is attributed; nothing unexplained remains

---

## Phase 3: User Story 1 - Preserve and correctly attribute the existing uncommitted work (Priority: P1) 🎯 MVP

**Goal**: Commit the four changes with accurate, honestly-labeled per-change status so the real work is no longer at risk of being lost to an accidental `git stash`/`clean`/branch-switch.

**Independent Test**: `git show` the resulting commit(s) and confirm every hunk is explained by this spec's description of the four changes, with each change's current validation status stated in the commit message or an accompanying results file.

- [X] T004 [US1] Stage and commit `linear_qw_coopmat.glsl`, `linear_dq8ca_qw_coopmat.glsl`, and `QuantizedLinear.cpp` together (interleaved per `research.md` Decision 3), with a commit message naming all four changes and their not-yet-validated status for the three shader changes (depends on T002, T003)
- [X] T005 [US1] Create `specs/014-m5-linear-coopmat-retune/results/disposition-summary.md` seeded from `data-model.md`'s Retuned Shader Change table (all three shader changes `disposition: pending`, documentation change `disposition: keep`) (depends on T001)

**Checkpoint**: US1 complete — nothing from the original uncommitted working-tree state remains at risk; every change is committed and its status is recorded. **DONE** (commits `133044739`, `59a8e62df`).

---

## Phase 4: Correctness Harness Extension (FR-008) — shared prerequisite for User Stories 2 and 3

**Goal**: Close the gap `/speckit-clarify` found: `test_coopmat_linear_bench.cpp`'s existing `kCorrectnessShapes`/`kRank3CorrectnessShapes` only cover up to K=256, short of FR-003/FR-004's production-K (2048/4096+) requirement. Neither US2 nor US3 can honestly claim "passed the correctness check at production shapes" until this closes.

**Independent Test**: Build `test_coopmat_linear_bench.cpp` with the new cases added; the harness compiles and its correctness-only cases (`COOPMAT_BENCH_CORRECTNESS_ONLY=1`) run and report pass/fail for the new K=2048/4096 shapes, independent of whether any of US2/US3's shader changes have been evaluated yet.

**Status**: DONE — T006-T009 all complete. Device access, driver
verification, the stale-library build fix, and a `bench_reference` size-cap
fix were all found and resolved along the way (see each task's notes).

- [X] T006 [P] Author new `kCorrectnessShapes` entries in `backends/vulkan/test/custom_ops/test_coopmat_linear_bench.cpp` at production K (2048 and 4096 at minimum), coopmat-eligible (`M%64==0`, `N%64==0`, `K%32==0`), reusing the existing `make_deterministic_correctness_case` well-conditioned positive-data generation and `abs=0.5`/`rel=0.05` tolerance unchanged (data-model.md's Correctness Harness Extension entity; no device access required to author this) — done: added `{128, 2048, 128, 128, ""}` and `{128, 4096, 128, 128, ""}` (group_size=128, matching the real per-model group size used elsewhere in this file's `kGroup`)
- [X] T007 [P] Author a matching new `kRank3CorrectnessShapes` entry at one of T006's production-K shapes (batch=1), same data/tolerance strategy, consistent with the existing rank-3 coverage added for specs `009` (no device access required to author this) — done: added `{128, 4096, 128, 128, "", batch=1}` (the larger/higher-risk of the two new K values)
- [X] T008 Obtain M5 EVT1 device access per `.shared-context/instruction-for-ai/devices-and-access.md`; re-verify driver identity per constitution Principle VIII (depends on: none — can run before or in parallel with T006/T007) — **DONE, both halves**: device-access corrected (an earlier session wrongly concluded "no device reachable" from running `adb devices` on this workstation directly; the M5 EVT1 is on `sj1-dmckee-d01`, reached via `ssh yanwen.xu@sj1-dmckee-d01` then `adb -s 0000088f8e579c33`). Driver identity: found the flashed driver (47,671,472 B, md5 `993d49a9…`) matched none of the four documented builds — backed it up first (`/sarc-c/gpusw/users/yanwen.xu/vulkan.samsung.so.device-unknown-993d49a9-backup-2026-07-05`), then flashed the documented known-good `f14c51b6f8` (`/sarc-c/gpusw/users/yanwen.xu/vulkan.samsung.so`, per `flash-sumd-driver.md`'s push procedure, with explicit user confirmation for the `setenforce 0` step). Post-flash on-device md5 = `c9861e9906d03fa2c7d48b804e1a1c80`, an exact match for `f14c51b6f8`. Verified further by pushing the prebuilt NFS `test_coopmat_linear_bench` and running `COOPMAT_BENCH_CORRECTNESS_ONLY=1`: **16/16 Buffer-storage (coopmat) correctness cases PASSED** (the 10 unrelated FAILs were all Texture3D/tiled-path `linear_dq8ca_q4gsw`, not coopmat) — matches the documented known-good signature. M5 EVT1 is now on a verified-good driver.
- [X] T009 Build `test_coopmat_linear_bench` with T006/T007's new cases and run it with `COOPMAT_BENCH_CORRECTNESS_ONLY=1` on M5 EVT1 against the **pre-change** (`HEAD`-only, per `research.md` Decision 1) shader; confirm the new production-K cases compile, dispatch the coopmat kernel, and pass — this validates the harness extension itself, independent of any of this feature's three shader changes (depends on T006, T007, T008). **DONE**: fixed the stale-`vulkan_backend` link blocker per `.shared-context/instruction-for-ai/build.md`'s documented two-step recipe — re-ran `cmake --build cmake-out-android-vk --target install` (19s, mostly cache-hit; reinstalled a fresh `libvulkan_backend.a`), then `test_coopmat_linear_bench` linked cleanly. **Also found and fixed a second issue**: `bench_reference()`'s hardcoded `M>256||K>256||N>256` guard was silently throwing for the new K=2048/4096 cases, marking them `SKIPPED` (zero actual validation) instead of running them. Raised to `M>256||N>256||K>4096` (M/N caps unchanged — the unrelated M=1024/N=14336 perf-sweep shapes still correctly skip the O(M·N·K) CPU reference). Re-ran: all 10 new production-K cases **PASSED** (both `linear_q4gsw` and `linear_dq8ca_q4gsw`, Buffer+Texture3D, rank2+rank3) on the verified `f14c51b6f8` driver. Full detail in `results/disposition-summary.md`.

**Checkpoint**: **DONE** — the harness reports PASS for all production-K shapes against the verified known-good driver; US2/US3's correctness verdicts below are now real, tool-confirmed evidence, not assumptions.

---

## Phase 5: User Story 2 - Validate the two low-risk, same-math changes on M5 EVT1 (Priority: P2)

**Goal**: Confirm the loop-shape flattening and vectorized dequant are correctness-safe and measure their performance on the actual target device.

**Independent Test**: Build the post-change shader, pass the extended (Phase 4) INT4 coopmat correctness check at production shapes, and produce a kernel-dispatch-confirmed tier-1 timing compared against a fresh pre-change M5 EVT1 baseline.

**Status**: DONE. Correctness (T011) passed; performance A/B (T010, T012) was deliberately not pursued — see below.

- [X] ~~T010~~ [US2] Build and run the pre-change (`HEAD`-only) `linear_qw_coopmat.glsl` tier-1 coopmat microbench on M5 EVT1 per `quickstart.md` step 1; record as the baseline in `results/us2-loop-vectorized-dequant-validation.md` (depends on T008). **DECIDED SKIP (user, 2026-07-05)**: this only requires a local `git stash`/rebuild/run/`git stash pop` cycle (never touches git history), but per spec Clarifications, a same-math code-shape change may be kept without a measured win — no perf claim is being made, so the formal A/B isn't required to close this out.
- [X] T011 [US2] Run the Phase-4-extended INT4 coopmat correctness check against the post-change shader at production K=2048/4096 (depends on Phase 4 (T009)) — **DONE as part of T009's run**: since the three shader changes are already committed at `HEAD` (interleaved, `research.md` Decision 3), T009's correctness run against `HEAD` *is* this check. All `linear_q4gsw` K=2048/4096 cases (Buffer+Texture3D, rank2+rank3) PASSED.
- [X] ~~T012~~ [US2] Run the tier-1 coopmat microbench against the post-change shader; confirm kernel dispatch and `spirv-dis`-verified `OpCooperativeMatrix*KHR` presence (`research.md` Decision 4); compare against T010's baseline (depends on T010, T011). **DECIDED SKIP** (depends on T010, also skipped). Kernel dispatch already confirmed (T009: `linear_q4gsw_coopmat_buffer_texture2d_half` observed) — that part of Principle VI's requirement is satisfied; the SPIR-V/comparison part is skipped along with T010 since no perf number is being reported.
- [X] T013 [US2] Update `results/disposition-summary.md`'s `loop_flattening` and `vectorized_dequant` rows with correctness/perf outcomes and final disposition (depends on T012) — **DONE**: disposition = `keep` (correctness PASS; no performance claim made, none required per Clarifications).

**Checkpoint**: DONE. Correctness passed and disposition recorded (`keep`) for both same-math changes. Performance A/B intentionally not pursued (user decision, not a gap).

---

## Phase 6: User Story 3 - Validate the fp16-accumulate experiment's correctness before any perf claim (Priority: P3)

**Goal**: Confirm or refute the fp16-accumulate change's numerical safety at real production K-dimensions before trusting any throughput number for it.

**Independent Test**: Run the fp16-accumulate variant against the Phase-4-extended correctness check at K=2048/4096; pass within the stated `abs=0.5`/`rel=0.05` tolerance, or fail explicitly and revert.

**Status**: DONE. Correctness (T014) PASSED — the precision risk this whole feature was gated on is resolved. T016 (revert) is moot. Performance A/B (T015) was deliberately not pursued, same reasoning as Phase 5's T010/T012.

- [X] T014 [US3] Run the Phase-4-extended INT4 coopmat correctness check against the fp16-accumulate variant at production K=2048 and K=4096; record numerical divergence against the fp32-accumulate reference within the `abs=0.5`/`rel=0.05` tolerance stated in `data-model.md`'s `numerical_tolerance` field (depends on Phase 4 (T009)) — **DONE, PASSED**: same T009 run (the fp16-accumulate change is part of the same committed, interleaved diff). All `linear_q4gsw` K=2048/4096 cases (Buffer+Texture3D, rank2+rank3) PASSED within `abs=0.5`/`rel=0.05` against the fp32 CPU reference. This is the first real hardware evidence the precision risk flagged in-code does not manifest at production K.
- [X] ~~T015~~ [US3] If T014 passes: run the tier-1 coopmat microbench for the fp16-accumulate variant, confirm kernel dispatch + SPIR-V accumulator-type verification (`research.md` Decision 4), compare against T010's baseline (depends on T014, T010). **DECIDED SKIP (user, 2026-07-05)**: FR-004 only requires correctness before *reporting* a throughput number — it does not require producing one. No perf claim is made for `fp16_accumulate`, so this is skipped along with T010/T012.
- [X] ~~T016~~ [US3] MOOT — T014 passed, so no revert is needed.
- [X] T017 [US3] Update `results/disposition-summary.md`'s `fp16_accumulate` row with the final outcome and disposition (`keep` if T015 completes, `revert` if T016 executes) (depends on T015 or T016) — **DONE**: disposition = `keep` (correctness PASS on real M5 EVT1 hardware; no performance claim made, none required).

**Checkpoint**: DONE. Correctness PASSED — the highest-risk item this feature exists to resolve is resolved. Disposition recorded (`keep`). Performance A/B intentionally not pursued (user decision, not a gap).

---

## Phase 7: Polish & Cross-Cutting Concerns

- [X] T018 Once all of US2/US3 are unblocked and complete, re-read `results/disposition-summary.md` as a whole and confirm every one of the three shader changes has a non-`pending` disposition with a stated reason (spec SC-002, SC-004) (depends on T013, T017) — **DONE**: all three shader changes have a final, non-`pending` disposition (`keep` x3), each with a stated reason (correctness PASS on real hardware; performance comparison explicitly and deliberately not pursued, per spec Clarifications allowing same-math/precision-safe changes to ship without a measured win — not a silently-missing gap).

---

## Dependencies & Execution Order

- **Phase 1 (Setup)** → **Phase 2 (Foundational)**: no dependencies, run first
- **Phase 3 (US1)**: depends on Phase 2; fully deliverable without hardware — **DONE**
- **Phase 4 (FR-008 harness extension)**: T006/T007 (authoring) have no hardware dependency and can start immediately; T008 (device access) can run in parallel; T009 (running the new cases) depends on all three. Phase 4 as a whole does NOT depend on Phase 3 (it doesn't touch the three shader changes), but blocks Phases 5 and 6's correctness steps.
- **Phase 5 (US2)** and **Phase 6 (US3)**: both depend on Phase 3 (the changes must be committed before validating them) and on Phase 4 completing (a correctness verdict against an unextended harness doesn't satisfy FR-003/FR-004) — but are otherwise **independent of each other** per spec Clarifications — neither blocks the other's completion or disposition
- **Phase 7 (Polish)**: depends on both Phase 5 and Phase 6 completing

## Parallel Execution Examples

- T002 and T003 (Phase 2) touch disjoint files and can run in parallel
- T006, T007, and T008 (Phase 4) touch disjoint concerns (test-shape authoring vs. device access) and can run in parallel; T009 waits for all three
- Once Phase 4 is complete, Phase 5 (US2) and Phase 6 (US3) can proceed on independent schedules — a correctness failure in US3 (T014) never blocks US2's own tasks

## Implementation Strategy

**All phases DONE.** User Story 1 committed the four changes. Phase 4
closed the FR-008 correctness-harness gap, along the way resolving device
access, driver verification/reflash, a stale-`vulkan_backend` build fix,
and a `bench_reference` size-cap fix. Running the fixed harness produced
this feature's real headline result: **all three shader changes pass
correctness at production K=2048/4096 on verified-good M5 EVT1 hardware**
(T011, T014) — the precision risk `fp16_accumulate` was gated on is
resolved. Final disposition for all three: `keep`.

**Performance A/B (T010, T012, T015) explicitly decided against** (user,
2026-07-05): no throughput claim is made for any of the three changes, so
there is nothing requiring the formal pre-change baseline comparison
(`research.md` Decision 1). If a future session wants an actual measured
speedup/regression figure, `quickstart.md` steps 1-4 document the method
(a local `git stash`/rebuild/run cycle — never touches git history).
