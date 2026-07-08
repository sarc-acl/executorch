---

description: "Task list for M5 EVT1 Linear + SDPA Coopmat Microbenchmark Validation"
---

# Tasks: M5 EVT1 Linear + SDPA Coopmat Microbenchmark Validation

**Input**: Design documents from `/specs/016-m5-linear-sdpa-microbench/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md (all present)

**Tests**: Not requested as a separate automated suite -- each harness's
own kernel-name dispatch check + SPIR-V instruction-presence check +
existing correctness-shape coverage IS the verification, matching how
`specs/007`/`010` validated their own work (constitution Principle I/VI).

**Organization**: Real device work on M5 EVT1 (not MiniPC). Two small,
additive C++ changes (Foundational phase) are required before either
harness can produce the full case set this feature needs -- see
`research.md` Decisions 1/2 for why.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files/resources, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2)
- Paths are relative to the repository root

## Path Conventions

- `backends/vulkan/test/custom_ops/test_coopmat_linear_bench.cpp` -- `kShapes` extended, otherwise unmodified
- `backends/vulkan/test/custom_ops/test_sdpa_coopmat_bench.cpp` -- unmodified, newly built
- `backends/vulkan/test/custom_ops/CMakeLists.txt` -- new build target added
- `specs/016-m5-linear-sdpa-microbench/results/` -- raw logs, SPIR-V dumps, and the two reports

---

## Phase 1: Setup

- [X] T001 Create `specs/016-m5-linear-sdpa-microbench/results/raw/` and `specs/016-m5-linear-sdpa-microbench/results/spirv/` directories -- **DONE**

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Verify the M5 EVT1 session state this feature reuses, and make
the two additive C++ changes both user stories depend on.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T002 Re-verify clock pin bound via GFLOP/s cross-check (constitution Principle VII) -- reuse `specs/015`'s established procedure; re-pin via `pin_freqs.sh` first if the device rebooted since -- **DONE**: `/sys/kernel/gpu/{min,max}_freq` both 509000, unchanged since `specs/015`
- [X] T003 Re-verify on-device driver identity against `.shared-context/ACTIVE-STATUS.md` (constitution Principle VIII), e.g. `adb -s $S shell logcat -d | grep SUMD` -- **DONE**: `md5sum /vendor/lib64/hw/vulkan.samsung.so` = `c9861e9906d03fa2c7d48b804e1a1c80`, exact match for `f14c51b6f8`
- [X] T004 [P] Extend `kShapes` in `backends/vulkan/test/custom_ops/test_coopmat_linear_bench.cpp` with 1B (`dim=2048`/`ffn=8192`) and 3B (`dim=3072`/`ffn=8192`) K/N pairs, each tagged with a model label, alongside the existing 8B pairs (`research.md` Decision 1) -- **DONE**: `kShapes` is now a `ShapeEntry{model,K,N}` vector, 12 entries (4 shapes x 3 models); `generate_cases()` and the SUMMARY block updated accordingly
- [X] T005 [P] Add a new `test_sdpa_coopmat_bench` executable target to `backends/vulkan/test/custom_ops/CMakeLists.txt`, mirroring the existing `test_coopmat_linear_bench` target's pattern (`research.md` Decision 2) -- **CORRECTION: already done**. `add_operator_prototype(test_sdpa_coopmat_bench)` was added in commit `b19116260` (already in current HEAD) -- the Clarifications-session grep that concluded it wasn't wired had a filter bug (piped through a `grep -i "^#include\|BUILD\|CMakeLists"` prefilter that excluded the very `add_operator_prototype(...)` line it needed to find). No CMake change needed; proceeding straight to the Android rebuild.
- [X] T006 Rebuild the Android cross-build tree per `.shared-context/instruction-for-ai/build.md`'s two-step recipe: core runtime + `--target install`, then the `custom_ops` sub-build (both steps, not just the sub-build -- this session's own stale-library lesson) -- **DONE**: both steps succeeded; `test_coopmat_linear_bench` and `test_sdpa_coopmat_bench` both built cleanly (one benign implicit-conversion warning, unrelated to this feature's edits)
- [X] T007 Stage + push both rebuilt binaries (`test_coopmat_linear_bench`, `test_sdpa_coopmat_bench`) to M5 EVT1 via the NFS run-kit, `chmod 755` on-device -- **DONE**

**Checkpoint**: Both binaries run on-device without crashing on startup; T004/T005's changes compile cleanly.

---

## Phase 3: User Story 1 - Linear coopmat microbenchmark on M5 EVT1 (Priority: P1) 🎯 MVP

**Goal**: Per-op, per-model, per-scheme tiled-vs-coopmat timing on M5 EVT1,
dispatch- and correctness-confirmed, in `specs/007`'s exact report format.

**Independent Test**: Run the extended `test_coopmat_linear_bench` and
produce `results/linear-coopmat-microbench-report.md` with all 42 cases
(or explicit exclusions) -- verifiable without touching User Story 2 at all.

- [X] T008 [US1] Run `test_coopmat_linear_bench` on M5 EVT1 with `ET_VK_FORCE_TILED_LINEAR=1` set (tiled capture); save output to `specs/016-m5-linear-sdpa-microbench/results/raw/linear-tiled-m5evt1.log` -- **DONE**: the harness measures both storage types (Texture3D=tiled, Buffer=coopmat) per case in one invocation; no separate `ET_VK_FORCE_TILED_LINEAR` run needed for this harness (that env var applies to the full LLaMA graph path, not this standalone op harness, which already A/Bs storage type directly)
- [X] T009 [US1] Run `test_coopmat_linear_bench` on M5 EVT1 with no env override (default/coopmat capture); save output to `specs/016-m5-linear-sdpa-microbench/results/raw/linear-coopmat-m5evt1.log` -- **DONE**: raw output saved to `results/raw/linear-m5evt1.log` (re-run once after adding stdev to the SUMMARY block for FR-002 compliance)
- [X] T010 [US1] For each distinct coopmat kernel name observed in T009 (e.g. `linear_q4gsw_coopmat_buffer_*_half`, `linear_dq8ca_q4gsw_coopmat_buffer_*_half`), run `spirv-dis` against its compiled `.spv` and save to `specs/016-m5-linear-sdpa-microbench/results/spirv/<kernel_name>.dis.txt`; confirm `OpCooperativeMatrixLoadKHR`/`OpCooperativeMatrixMulAddKHR` presence (skip re-capture if byte-identical via `md5sum` to `specs/007`'s existing citation) -- **DONE**: `results/spirv/linear_q4gsw_coopmat_buffer_texture2d_half.dis.txt` (22 `OpCooperativeMatrix*KHR`), `results/spirv/linear_dq8ca_q4gsw_coopmat_buffer_texture2d_half.dis.txt` (48, with int8 `Matrix*SignedComponentsKHR` flags) -- freshly captured, not byte-identical to `specs/007`'s MiniPC citation (different hardware/build)
- [X] T011 [US1] Compute per-row `speedup_pct`, `significance` (non-overlapping `mean +/- 2*stdev` band), `dispatch_status`, `correctness_verified` for all 42 (model, scheme, op) cases per `data-model.md` -- **DONE**: all 42 rows `real_effect` (61.8%-77.8% speedup, far outside any noise band), `dispatch_status=confirmed`, `correctness_verified=true`
- [X] T012 [US1] Compute the time-weighted overall `4w` and `8da4w` speedup figures per `research.md`/`specs/007`'s method -- **DONE**: `4w` overall +67.0%, `8da4w` overall +75.8% (per-model weighted average, then averaged across the 3 models, matching `specs/007`'s method)
- [X] T013 [US1] Write `specs/016-m5-linear-sdpa-microbench/results/linear-coopmat-microbench-report.md` per `contracts/microbench-report-schema.md` -- overall figures, full 42-row table, Excluded section (even if empty), explicit M5 EVT1 label, one-line comparison against `specs/007`'s MiniPC figures -- **DONE**: `results/linear-coopmat-microbench-report.md` published
- [X] T014 [US1] Report the linear report's headline figures to the user -- **DONE**: reported to user -- headline: 4w +67.0%/8da4w +75.8% on M5 EVT1; notably 8da4w is a **win** here vs MiniPC's -15.2% regression, a real platform difference

**Checkpoint**: User Story 1 is independently complete and deliverable.

---

## Phase 4: User Story 2 - SDPA coopmat microbenchmark on M5 EVT1 (Priority: P2)

**Goal**: Per-model SDPA `sdpa_compute_attn_weights`/`sdpa_compute_out`
tiled-vs-coopmat timing on M5 EVT1, in `specs/010`'s exact report format.

**Independent Test**: Run the newly-built `test_sdpa_coopmat_bench` and
produce `results/sdpa-coopmat-microbench-report.md` with all 3 model cases
(or explicit blocked reasons) -- verifiable independently of User Story 1.

- [X] T015 [US2] Run `test_sdpa_coopmat_bench` on M5 EVT1 for each of the 3 target models, once with `ET_VK_SDPA_COOPMAT` unset (tiled) and once with it set to `1` (coopmat); save output to `specs/016-m5-linear-sdpa-microbench/results/raw/sdpa-m5evt1.log`. If a model's run crashes or fails to build, record the exact error text and mark that model `blocked` (spec Edge Cases) -- continue with the remaining models, do not retry with a reduced shape as a substitute. -- **DONE, and cleanly**: all 3 models ran without any build failure or crash, all `dispatch=confirmed`, no `blocked` cases. 8B +81.5%, 3B +81.8%, 1B +75.2% (tiled vs coopmat)
- [X] T016 [US2] For each of `sdpa_compute_attn_weights_coopmat`/`sdpa_compute_out_coopmat` observed dispatching in T015, run `spirv-dis` against its compiled `.spv` and save to `specs/016-m5-linear-sdpa-microbench/results/spirv/<kernel_name>.dis.txt`; confirm `OpCooperativeMatrix*KHR` presence (skip re-capture if byte-identical to `specs/010`'s existing citation) -- **DONE**: `results/spirv/sdpa_compute_attn_weights_coopmat_buffer_buffer_half.dis.txt` (36 `OpCooperativeMatrix*KHR`), `results/spirv/sdpa_compute_out_coopmat_buffer_buffer_half.dis.txt` (20) -- instruction counts match `specs/010`'s MiniPC citation exactly
- [X] T017 [US2] Compute per-model `speedup_pct`, `significance`, `dispatch_status` for all non-blocked models per `data-model.md` -- **DONE**: all 3 models `real_effect` (75.2%-81.8% speedup, stdevs <0.4% relative)
- [X] T018 [US2] Compute the overall average speedup figure across the models that produced a valid (non-blocked) measurement, per `specs/010`'s method -- **DONE**: overall average +79.5% across all 3 (non-blocked) models
- [X] T019 [US2] Write `specs/016-m5-linear-sdpa-microbench/results/sdpa-coopmat-microbench-report.md` per `contracts/microbench-report-schema.md` -- dispatch/correctness summary first, overall figure, per-model table, Excluded/Blocked section (even if empty), explicit M5 EVT1 label, one-line comparison against `specs/010`'s MiniPC figure -- **DONE**: `results/sdpa-coopmat-microbench-report.md` published
- [X] T020 [US2] Report the SDPA report's headline figures (or blocked-model status) to the user -- **DONE**: reported to user -- headline: SDPA coopmat +79.5% on M5 EVT1 vs MiniPC's +66.8%, same direction, no blocked models

**Checkpoint**: User Story 2 is independently complete and deliverable.

---

## Phase 5: Polish & Cross-Cutting Concerns

- [X] T021 Add a short side-by-side MiniPC-vs-M5-EVT1 comparison note (one paragraph, both reports' headline figures against `specs/007`'s/`specs/010`'s) -- either as a shared addendum or inline in each report, per SC-004 -- **DONE**: inline "Comparison against ... MiniPC figures (SC-004)" section in both reports, plus a shared `results/minipc-vs-m5evt1-comparison.md` addendum table
- [X] T022 Re-read both published reports and confirm SC-001 through SC-004 are satisfied: every case has a number or an explicit excluded/blocked reason, every number carries iteration count + stdev, dispatch and correctness are confirmed per row, and each report states a MiniPC-comparable overall figure -- **DONE**: SC-001 (42/42 linear cases, no exclusions needed), SC-002 (3/3 SDPA models, no blocked), SC-003 (every row has mean ± stdev from 5 timed runs), SC-004 (both reports + the shared addendum state MiniPC-comparable figures) all verified directly against the published report files

---

## Dependencies & Execution Order

- **Setup (Phase 1)**: No dependencies -- start immediately.
- **Foundational (Phase 2)**: Depends on Setup. T004 and T005 are independent file edits (`[P]`); T006 depends on both being complete; T007 depends on T006. **BLOCKS all user story work.**
- **User Story 1 (Phase 3)**: Depends on Foundational completion (needs the extended `kShapes` binary on-device). Independent of User Story 2.
- **User Story 2 (Phase 4)**: Depends on Foundational completion (needs the newly-built SDPA binary on-device). Independent of User Story 1 -- may run before, after, or interleaved with Phase 3.
- **Polish (Phase 5)**: Depends on both User Story 1 and User Story 2 being complete.

## Parallel Execution Examples

- T004 and T005 (Phase 2) touch different files and can be done in parallel.
- Once Phase 2 completes, Phase 3 (US1) and Phase 4 (US2) can run fully in parallel -- different binaries, different device sessions if desired, different report files -- since neither reads the other's output.

## Implementation Strategy

**MVP = User Story 1 only** (Phase 1 + 2 + 3): delivers the linear
microbenchmark report, the higher-priority and higher-impact half of this
feature (per spec.md's stated priorities), independently of whether the
SDPA harness ever gets wired into the build. User Story 2 (Phase 4) is
additive on top and does not block or get blocked by User Story 1's
completion or publication.
