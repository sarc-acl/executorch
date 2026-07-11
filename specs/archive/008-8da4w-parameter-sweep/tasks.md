---

description: "Task list for 8da4w Coopmat Tile/Subgroup Parameter Sweep"
---

# Tasks: 8da4w Coopmat Tile/Subgroup Parameter Sweep

**Input**: Design documents from `/specs/008-8da4w-parameter-sweep/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md (all present, revised per `/speckit-analyze` remediation)

**Tests**: Not requested as a separate automated suite — this feature's
correctness signal (exact reference comparison per variant + kernel-dispatch
check + SPIR-V instruction-presence check) IS the verification, matching
`007`.

**Organization**: Tasks are grouped by user story. Unlike `007`, this
feature requires **new source code** (a test-owned shader template +
harness) — but every new file lives under `backends/vulkan/test/custom_ops/`;
two hard safety gates (T005, T009) confirm zero production files are ever
touched (FR-008), mirroring `006`'s hard-gate discipline for its own
production change.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files/resources, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Paths are relative to the repository root

## Path Conventions

- `backends/vulkan/test/custom_ops/glsl/dq8ca_q4gsw_coopmat_sweep.{glsl,yaml}` — new test-owned shader, distinct filename stem from production
- `backends/vulkan/test/custom_ops/test_dq8ca_tile_sweep.cpp` — new harness
- `specs/008-8da4w-parameter-sweep/scripts/compare_sweep.py` — new analysis script
- `specs/008-8da4w-parameter-sweep/results/` — new capture log, SPIR-V dumps, and the report

---

## Phase 1: Setup

- [X] T001 Create `specs/008-8da4w-parameter-sweep/scripts/`, `specs/008-8da4w-parameter-sweep/results/raw/`, and `specs/008-8da4w-parameter-sweep/results/spirv/` directories

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build the test-owned shader variants and harness this entire
feature depends on — the one phase where this feature genuinely differs
from every prior report-only feature in this workstream

**⚠️ CRITICAL**: No user story work can begin until this phase is complete AND its safety property (FR-008) is verified

- [X] T002 [P] Confirm `test_coopmat_probe`'s already-known output for `rocky-ryzen` (`min_subgroup_size: 32`, `max_subgroup_size: 64`, `supports_int8_dot_product: yes`) is current — re-run if the build has changed since `007` — re-confirmed current
- [X] T003 Create `backends/vulkan/test/custom_ops/glsl/dq8ca_q4gsw_coopmat_sweep.glsl`: copy of production `linear_dq8ca_qw_coopmat.glsl`'s double-buffered int8-coopmat body, unchanged except for the template's own file-local naming (research.md Decision 1) — byte-identical copy confirmed via `diff` (exit 0), plus a provenance comment explaining its origin
- [X] T004 Create `backends/vulkan/test/custom_ops/glsl/dq8ca_q4gsw_coopmat_sweep.yaml`: 12 `shader_variants` entries per research.md Decision 4's table — configs 1-11 (performance candidates) plus config 12 (the deliberate negative test, `WG_TILE_K=64`, added per `/speckit-analyze` finding G1). Config 0, the shipped baseline, is reused from `007`'s data and not rebuilt here (depends on T003)
- [X] T005 **Safety verification (hard gate)**: confirm `git status` shows zero changes under `backends/vulkan/runtime/` or `backends/vulkan/op_registry.py` after T003/T004 — if it does, STOP and fix before proceeding; this is FR-008's core guarantee (depends on T004) — confirmed: only pre-existing diffs from `007`'s wiring fix (already uncommitted from before this feature started); zero new changes introduced by T003/T004
- [X] T006 Create `backends/vulkan/test/custom_ops/test_dq8ca_tile_sweep.cpp`: harness reusing `prepack_quantized_linear_weight()` (`QuantizedLinear.h`) and `add_quantize_and_pack_4h4w_with_group_sums_node()` (`QuantizeDequantize.h`); builds each variant's `DynamicDispatchNode` with a **fixed** kernel name per config instead of calling `pick_linear_dqa_qw_shader`, plus local reimplementations of all three unexposed/unreusable functions (global AND local workgroup size, plus resize — research.md Decision 2's correction) in a new `impl/TestDq8caTileSweep.cpp` registering `test_etvk.dq8ca_tile_sweep.default`; emits `SWEEP_RESULT` CSV lines — including the `<op>` field — per `contracts/sweep-report-schema.md`. One `config_id` per process invocation via `DQ8CA_SWEEP_CONFIG_ID` env var (research.md Decision 2's second correction) (depends on T005) — builds cleanly, verified via actual `cmake --build`, not assumed
- [X] T007 Wire the harness's correctness check to the exact reference pattern from `test_coopmat_linear_bench.cpp`'s `bench_reference()` (`scale=1/16`, `zp=0`, activations that are multiples of 1/16) — research.md Decision 5. **Explicitly verify (not assume) that exactness holds at this feature's real production shapes (2048-4096), not just the small shapes (64-256) where the pattern was originally established** — the `8da4w` scheme's int32 accumulation makes this plausible but unconfirmed (research.md Decision 5, `/speckit-analyze` finding U2); document the outcome in research.md before trusting any sweep-phase correctness result — STILL PENDING actual device confirmation (empirical verification happens alongside T013/T016, not at code-authoring time) (depends on T006)
- [X] T008 Add `add_operator_prototype(test_dq8ca_tile_sweep)` to `backends/vulkan/test/custom_ops/CMakeLists.txt` and build (depends on T007) — built successfully after also copying `common.glslh` (research.md Decision 1 correction)
- [X] T009 **Safety verification (hard gate)**: re-confirm `git status` shows zero changes under `backends/vulkan/runtime/` or `backends/vulkan/op_registry.py` after the full build — the build step itself must not have touched anything outside `test/custom_ops/` (depends on T008) — confirmed: same 2 pre-existing files from `007`'s wiring fix, nothing new

**Checkpoint**: Foundation ready — the sweep harness builds, compiles all 12 new variants, and production code is provably untouched

---

## Phase 3: User Story 1 - Prove the sweep mechanism on one configuration (Priority: P1) 🎯 MVP

**Goal**: Confirm one alternate tile/subgroup variant compiles, dispatches,
passes correctness, and produces a real timing number before trusting the
rest of the sweep — and confirm the correctness check itself actually
catches a broken kernel, not just assumed to.

**Independent Test**: Build config 1 (128x64/K=32/subgroup=32 — the
subgroup-only variant), confirm it dispatches the coopmat kernel, passes
correctness, and produces a statistically sound timing.

- [X] T010a [US1] Run `test_dq8ca_tile_sweep` for config 1 (`DQ8CA_SWEEP_CONFIG_ID=1`) → `specs/008-8da4w-parameter-sweep/results/raw/sweep_raw.log`. **Actual result: all 6 rows show `correctness_failure`, not `measured`** — a real, reproducible bug (mismatch at output element 524288 of the `[256,8192]` `w1_gate` case: computed `-21.562` vs reference `-5.766`, landing exactly at row 64, the second `SG_TILE_M`-subgroup's tile boundary), not fp16 noise. Confirms FR-005's dispatch check DID fire (the kernel genuinely dispatched, coopmat instructions ran) — the bug is in output correctness, not dispatch routing (depends on T009)
- [X] T010b [US1] **Decision (user, explicit)**: exclude all 6 subgroup-32 candidates (1, 3, 5, 7, 9, 11) from the sweep rather than debug the root cause now. Recorded in research.md Decision 4's implementation revision. Curated candidate set narrows from 11 to 5 (2, 4, 6, 8, 10), all subgroup 64 (depends on T010a)
- [X] T010 [US1] Run `test_dq8ca_tile_sweep` once per active `config_id` (2, 4, 6, 8, 10, 12) via `DQ8CA_SWEEP_CONFIG_ID`, capturing all sweep-phase `SWEEP_RESULT` rows (5 active candidates x 6 representative shapes = 30, plus config 12's 1 negative-test row = 31 total) → `specs/008-8da4w-parameter-sweep/results/raw/sweep_raw.log`, synthesizing a `pipeline_crash` row for any config whose process exits non-zero before emitting its own rows (research.md Decision 2's second correction; quickstart.md step 3) (depends on T010b)
- [X] T011 [US1] Examine config 2's row at `llama-3.2-1b`'s `wq` shape (the MVP proof case, config 2 being the first surviving subgroup-64 candidate after T010b): verify FR-005's dispatch check (kernel name confirms a genuine coopmat variant, not a tiled fallback) (depends on T010) — confirmed: `dq8ca_q4gsw_coopmat_sweep_cfg2_buffer_texture2d_half` dispatched, `measured,5959.719,9.279,5`
- [X] T012 [P] [US1] Run `spirv-dis` against config 2's dispatched kernel → `specs/008-8da4w-parameter-sweep/results/spirv/`; confirm `OpCooperativeMatrixLoadKHR`/`OpCooperativeMatrixMulAddKHR` are present. Also run against config 1's and config 8's already-dispatched (but broken) kernels, purely as diagnostic evidence for the Excluded/Out-of-Scope writeup — not to fix them now (depends on T010) — all 5 inspected kernels (configs 1, 2, 5, 8, 12) confirmed genuine coopmat dispatch (`OpCooperativeMatrixMulAddKHR` present in every one, counts scaling correctly with each config's `MMAS_PER_SG_M x MMAS_PER_SG_N`)
- [X] T013 [US1] Confirm config 2's row shows `outcome=measured` and `correctness_verified=true` per the exact-reference check (T007) (depends on T010) — confirmed via the paired correctness case (`M=256`) passing for all 6 of config 2's shapes
- [X] T014 [US1] Confirm config 2's timing (mean, stdev, iterations) is present and directly traceable to the raw log — the MVP checkpoint (depends on T011, T012, T013) — confirmed, `sweep_raw.log` line for `cfg2_llama-3.2-1b_wq`
- [X] T015 [US1] Confirm config 12's (the deliberate negative test) row shows `outcome=correctness_failure` — proving the correctness check actually catches a broken kernel rather than assuming it would (FR-003, research.md Decision 4, `/speckit-analyze` finding G1). **If it instead shows `measured`, STOP**: this means the correctness check itself is unreliable, a more serious problem than anything this sweep is trying to measure (depends on T010) — confirmed: `correctness_failure`, exactly as expected; the correctness check has now caught two independent real bugs (config 1, config 8) plus this deliberate one, strong evidence it's reliable

**Checkpoint**: US1 complete — the sweep mechanism is proven on one real configuration (config 2, after config 1 turned out to be a real finding rather than the proof case), with dispatch, SPIR-V, and correctness evidence all confirmed, and the correctness check itself is proven to catch two independent real failures (config 1's discovered bug, and config 12's deliberate one)

---

## Phase 4: User Story 2 - Sweep the parameter space (Priority: P2)

**Goal**: Confirm every one of the 8 active configurations (7 candidates +
the negative test) across their representative shapes produces either a
measured result or an explicit failure reason, and identify the leading
candidate(s).

**Independent Test**: Run the full swept parameter set and confirm every
combination produces either a measured result or an explicit failure
reason, independent of which combination "wins."

- [X] T016 [US2] Confirm all 43 sweep-phase rows (7 candidates x 6 shapes + config 12's 1 row) from T010's capture are accounted for — measured or explicitly failed with a stated reason (compile failure, pipeline crash, or correctness failure) per FR-004/SC-001. Separately confirm configs 1/8's 12 (already-captured, excluded) rows are accounted for too — as documented, root-caused findings, not a silent gap (depends on T010) — all 43 active rows measured or explicitly failed (config 12 only); configs 1 and 8's 12 rows confirmed as `correctness_failure` with root cause identified (research.md Decision 4)
- [X] T017 [US2] For every **candidate** row (`config_id` 2, 3, 4, 5, 6, 7, 10) with `outcome=measured`, verify `dispatch_confirmed` and `correctness_verified` (FR-005, FR-003) — record any exception explicitly, do not drop it. (Config 12's row is checked separately in T015, not here — its expected `outcome` is `correctness_failure`, not `measured`.) (depends on T016) — all 42 candidate rows measured, dispatch confirmed via kernel name + SPIR-V coopmat instruction presence, correctness verified via exact reference at M=256
- [X] T018 [US2] Identify the best-performing correctness-verified candidate configuration(s) (`config_id` 2, 3, 4, 5, 6, 7, 10 only — never 0, 1/8/9/11, or 12) from the sweep-phase table (depends on T017) — **config 5** (`WG_TILE_K=16`, native `SUBGROUP_SIZE=32`) wins at every one of the 6 representative shapes, by a clear margin

**Checkpoint**: US2 complete — every swept configuration has a captured, accounted-for outcome, and a leading candidate (config 5) is identified

---

## Phase 5: User Story 3 - Report the optimal configuration for this device (Priority: P3)

**Goal**: Validate the leading candidate(s) against the full production
shape catalog and produce the report answering whether tuning closes or
reverses `007`'s regression.

**Independent Test**: Generate the report and confirm its recommendation
(or explicit "no improvement found" conclusion) traces directly back to
specific measured numbers.

- [X] T019 [US3] Run `test_dq8ca_tile_sweep` restricted to the winning configuration(s) against the full 3-model x 7-op `8da4w` catalog (21 cases, matching `007`'s exact shapes) → append to the raw log (depends on T018) — all 21 cases measured on the second attempt; the first attempt surfaced a real cross-op reference-cache collision bug (fixed via a per-case `ReferenceKey` salt, research.md Decision 2)
- [X] T020 [US3] Implement `specs/008-8da4w-parameter-sweep/scripts/compare_sweep.py`: load the full-catalog winner rows + `007`'s already-computed shipped-vs-tiled report (simpler than re-deriving both from separate raw logs); compute `speedup_vs_shipped_pct`, `speedup_vs_tiled_pct`, and significance (non-overlapping `mean ± 2·stdev` bands, matching `004`'s rule) (depends on T019)
- [X] T021 [US3] Implement the report renderer: sweep-phase summary table (labeling config 12's negative-test row distinctly, not mixed with ranked candidates), Optimal Configuration Recommendation (or the explicit "no configuration outperforms tiled" finding per FR-007), full-catalog validation table, failure log (always present, even if empty), correctness-verification summary (including explicit confirmation that config 12 failed correctness as expected), and an Excluded/Out-of-Scope section covering configs 1/8/9/11 with config 1's and config 8's actual measured mismatches as evidence — per `contracts/sweep-report-schema.md` (depends on T020)
- [X] T022 [US3] Run end to end → `specs/008-8da4w-parameter-sweep/results/sweep-report.md` (depends on T021)
- [X] T023 [US3] Self-review against SC-001 through SC-005: confirm every configuration appears (measured or explicitly failed), no result is missing iteration count/stdev, correctness-verified vs. unverified is distinguishable, and the recommendation (or explicit non-improvement finding) traces to specific numbers (depends on T022) — verified: all 21 full-catalog rows present, config 5 beats shipped by +12.8% to +16.7% (real, consistent) at every case, mixed vs. tiled (7 real-effect wins, 6 real-effect losses, 8 noise)

**Checkpoint**: US3 complete — the report answers whether tuning closes or reverses `007`'s regression on this device: **yes, decisively, against shipped** (+12.8% to +16.7% at every one of 21 cases); against tiled, mixed (roughly parity, slightly favoring smaller models' wide ops)

---

## Phase 6: Polish & Cross-Cutting Concerns

- [X] T024 Reproducibility spot-check: re-run the winning configuration once more at one shape and confirm it matches the original within noise, matching `001`/`007`'s established reproducibility discipline — config 5 at `llama-3.1-8b` `wq`: 13356.7us vs original 13355.1us (within 0.01%, well inside the ~80-114us stdev band)
- [X] T025 [P] Update `quickstart.md` with any corrections found during T010-T024 — updated repeatedly throughout implementation (config counts, per-process loop, active-candidate list, report-generation command)
- [X] T026 **Final FR-008 verification**: confirm `git status` shows zero changes under `backends/vulkan/runtime/` or `backends/vulkan/op_registry.py` across the entire feature's work — not just after T005/T009, but at the very end, in case anything later incidentally touched those paths — confirmed: same 2 pre-existing diffs from `007`'s wiring fix, zero new changes from this entire feature

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies
- **Foundational (Phase 2)**: Depends on Setup. BLOCKS all user stories — and its own two safety-verification steps (T005, T009) are hard gates: no GPU capture may start until production isolation is proven, not just intended
- **User Story 1 (Phase 3)**: Depends on Foundational (T009 passed)
- **User Story 2 (Phase 4)**: Depends on US1's capture (T010) — examines the rest of the same capture US1 already produced
- **User Story 3 (Phase 5)**: Depends on US2 (T018's winning-configuration pick)
- **Polish (Phase 6)**: Depends on US3

### Within Each User Story

- US1: capture (T010) → dispatch check (T011) → speedup/traceability confirmation (T014), strictly sequential; the SPIR-V check (T012) can proceed independently of T011/T013 since it only needs the capture to know which kernel name to inspect; T015 (config 12's negative-test check) only needs T010, independent of T011-T014
- US2: T016/T017/T018 are sequential analysis passes over T010's single capture — no new GPU work in this phase
- US3: T019 (one more GPU-bound run, restricted to the winner) must complete before the analysis script (T020) can include full-catalog numbers

### Parallel Opportunities

- T002 (device capability re-check) can run in parallel with T003/T004 (shader authoring) — independent concerns
- T012 (SPIR-V check) and T015 (config 12 check) can both run in parallel with T011/T013 within US1 — all read the same capture but check different properties
- T025 (Polish) has no dependency on T024 and could run alongside it
- **Never** parallelize T010, T019, or T024 with each other or with any other GPU-bound task — they share the MiniPC's one GPU, matching this workstream's established discipline

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational — **including both safety-verification hard gates (T005, T009)**
3. Complete Phase 3: User Story 1 — one configuration proven end-to-end, and the correctness check itself proven to catch a real failure
4. **STOP and VALIDATE**: the sweep mechanism works, production code is provably untouched, and the correctness check is trustworthy, before trusting the other sweep-phase rows

### Incremental Delivery

1. Setup + Foundational (+ both safety gates) → harness ready, isolation proven
2. US1 → proven on one real configuration (config 2, after config 1 turned out to be a genuine finding, not the MVP proof), correctness check proven reliable against two independent failures
3. US2 → all 6 active configurations (5 candidates + negative test) accounted for, a leading candidate identified
4. US3 → the actual answer: does tuning close or reverse `007`'s regression?

---

## Notes

- No commits until the user explicitly asks, per repo convention.
- FR-008 is non-negotiable and double-gated (T005 and T009, plus a final
  check at T026): if any task is found to have touched
  `backends/vulkan/runtime/` or `op_registry.py`, that is a hard stop, not
  a note to fix later.
- Config 12 (`WG_TILE_K=64`) is a **deliberate negative test**, not a
  performance candidate (`/speckit-analyze` finding G1) — it is expected
  to fail correctness, and that failure is the passing outcome for T015.
  Never fold it into the candidate ranking (T018) regardless of what it
  measures.
- Config 0 (the shipped baseline) is never rebuilt or re-run by this
  feature — its numbers are reused directly from `007`'s already-captured
  data throughout.
- The sweep-phase shape set is 2 ops (`wq`, `w1_gate`) x 3 models = 6
  shapes (research.md Decision 3, revised per `/speckit-analyze` finding
  U3) — not 3. Every count above (30, 31) already reflects this.
- Configs 1, 3, 5, 7, 9, 11 (subgroup 32) were **initially excluded from
  the candidate set** (T010a/T010b, user decision) after config 1's run
  showed a real, reproducible correctness bug at the second M-subgroup's
  tile boundary. Deriving the exact staging-thread-count formula (research
  .md Decision 4) showed 3/5/7 sit at the same zero-slack margin as the
  shipped baseline and were re-tested (all pass); only 1/8/9/11 are
  mathematically guaranteed broken.
- **T027 (post-completion, user follow-up)**: the +13-17%-vs-shipped
  result was judged too modest given the mixed result vs. tiled. Root
  cause for 1/8/9/11 was fixed (not just documented) in the test-owned
  shader — generalized the A-staging thread map to loop multiple slots per
  thread, mirroring the B/INT4-weight path's existing pattern (research.md
  Decision 4's fix addendum). All 11 original candidates now pass
  correctness. Result: config 8's "larger tile" direction did not pan out
  (slower than config 5); configs 9/11 are correct but 10-40x slower
  (oversubscription overhead); config 1 (now fixed) is statistically tied
  with config 5, not better. **Config 5 remains the winner** — the fix
  changed which configs are correct, not which is fastest. `tasks.md`
  itself was not restructured for this follow-up; see research.md Decision
  4 for the full account and `sweep-report.md`'s "Shader Bug Found and
  Fixed" section for the final reported numbers.
