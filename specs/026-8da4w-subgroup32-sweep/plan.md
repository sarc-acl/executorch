# Implementation Plan: Re-Open SUBGROUP_SIZE=32 in the 8da4w CoopMat Tile/Subgroup Sweep on M5 EVT1

**Branch**: `026-8da4w-subgroup32-sweep` | **Date**: 2026-07-11 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/026-8da4w-subgroup32-sweep/spec.md`

## Summary

Redo `specs/025-8da4w-parameter-sweep`'s `8da4w` (`linear_dq8ca_q4gsw_coopmat`) tile/subgroup
search, this time treating `SUBGROUP_SIZE ∈ {32, 64}` as a real search axis instead of a
constant fixed at 64. `025` fixed it at 64 solely on the shipped shader's header comment (a
documented Xclipse PAL `vkCreateComputePipelines` crash for int8 WMMA at subgroup 32); its own
one-shot re-check (T014) found that crash does not reproduce at one tile shape/one
correctness shape, but this session's independent re-run of the same probe on a second M5
EVT1 board found the fuller picture T014 couldn't see from one data point: correctness fails
at additional shapes (`M=256` variants) that a single-shape check misses, and the probe's
performance (~1095–1169 GFLOP/s at M=2048) trails both the shipped subgroup=64 baseline
(~1688 GFLOP/s) and `025`'s actual winner (1736 GFLOP/s). This feature re-derives the legal
space with subgroup_size open, correctness-gates every surviving candidate across the full
representative multi-shape set (not one shape), and runs the same staged
analytical-pruning → on-device search → validated-winner methodology `025` used, ending in
one of two outcomes: a new winner (possibly subgroup=32), or an evidence-backed confirmation
that `025`'s winner stands and the axis is properly closed this time.

## Technical Context

**Language/Version**: Python 3 (legal-space enumeration + analytical scorer + staged-search
orchestration, extending `025`'s `scripts/enumerate_configs.py` / `score_and_shortlist.py` /
`tile_constraints.py` to carry `SUBGROUP_SIZE` as a swept field rather than a constant); GLSL
450 / `GL_KHR_cooperative_matrix` + C++17 (existing `linear_dq8ca_q4gsw_coopmat_tsweep` shader
template and its `QuantizedLinear.cpp` dispatch code — new work is a `SUBGROUP_SIZE` template
parameter and shader_variants entries, not new shader logic)

**Primary Dependencies**: `025`'s `dbuf2`-loop-structure `linear_dq8ca_q4gsw_coopmat_tsweep`
template and its `ET_VK_DQ8CA_COOPMAT_VARIANT=tsweep_t<M>x<N>k<K>g<SGX><SGY>s<sub>` dispatch
token (this feature extends the token's existing `s<sub>` field, which `025` always
instantiated at `64`, to also take `32`); the ad-hoc `sg32test` shader/binding added this
session in the `dbuf-int8-sweep` worktree (`023-8da4w-int8-dbuf-sweep-impl` branch) as a
reference implementation to fold into the extended template, then remove (spec FR-012); the
existing `test_coopmat_linear_bench` harness (`COOPMAT_BENCH_CORRECTNESS_ONLY=1` gate,
`COOPMAT_BENCH_M=2048` perf pass); the existing Android cross-build pipeline (NDK, `glslc`,
ccache) and its `cmake-out-android-vk` / `cmake-out-android-vk/bench` two-stage configure
(install the backend, then configure+build the bench subproject against it — this session
re-derived this exact sequence since it wasn't written down anywhere in
`.shared-context/instruction-for-ai/`, see research.md Decision 4); `adb` access to M5 EVT1
(primary and/or secondary board); clock-pinning script. No new external libraries.

**Storage**: N/A — file-based. Enumerated legal-space data (both subgroup sizes),
analytical scores, per-shape correctness matrices, and per-round measurement results are
JSON/CSV/Markdown under this feature's `specs/026-8da4w-subgroup32-sweep/results/`, not a
database.

**Testing**: the existing small-shape, fp32-reference `dq8ca_q4gsw` correctness check
(`COOPMAT_BENCH_CORRECTNESS_ONLY=1`), reused as the pass/fail gate (Constitution Principle
I) — but run against the **full multi-shape representative set** for every subgroup=32
candidate (spec FR-003), not the single `M=K=N=128` shape `025`'s T014 and this session's
prior probe each used. This broadened correctness scope is this feature's core methodology
change, not a new correctness mechanism.

**Target Platform**: Samsung M5 EVT1 (Exynos 2500 / Xclipse 970), Android, pinned clocks
(Principle VII); driver identity re-verified before every measurement round (Principle
VIII), on whichever of the two shared M5 EVT1 boards is used — the report states which board
produced which result if both are used (spec Assumptions), since the two boards' driver
state can independently drift.

**Project Type**: Single project — a bounded, internal research/automation addition to the
existing Vulkan backend, following `022`/`025`'s precedent exactly.

**Performance Goals**: identify whether any fully-correct (all representative shapes)
subgroup=32 `8da4w` configuration exceeds `025`'s standing winner's FLOP-weighted throughput
across the 6-shape set (spec SC-003); if none does, confirm this with per-shape evidence
rather than assumption (spec FR-008).

**Constraints**: loop structure held fixed at `025`'s confirmed `dbuf2` winner for the
entire search (spec Assumptions, same axis-separability reasoning `025` used relative to
`023`); search budget ≤15% of the re-derived legal space, hard-capped at 30 on-device
measurements (spec FR-009); no performance number counts until its configuration passes
correctness at **every** representative shape, not just one (spec FR-004 — the specific gap
this feature closes relative to `025`'s T014); driver identity and device availability
re-verified before every measurement round; default production `8da4w` dispatch behavior
MUST remain unchanged — all variants stay opt-in via the existing
`ET_VK_DQ8CA_COOPMAT_VARIANT` token, never on by default; the ad-hoc `sg32test` probe must be
superseded and removed by this feature's own extended-`tsweep` variants (spec FR-012), not
left as parallel, undocumented infrastructure.

**Scale/Scope**: the re-derived `8da4w` legal space with `SUBGROUP_SIZE` open — expected
larger than `025`'s 542-candidate space (roughly up to ~2x before eligibility pruning, since
subgroup_size was previously a fixed value and is now a second value at every tile/grid
point), with the exact count a Phase 0 research output. Correctness gating in this feature
is broader in per-candidate depth than `025`'s (multi-shape, not single-shape) but the
on-device measurement budget cap (≤15%, ≤30) is unchanged from `025`'s convention — the
budget is spent primarily on comparing survivors, and the broadened correctness check itself
is cheap per shape (small-shape harness runs), so it does not by itself blow the budget.
Tier-1 (shader microbenchmark) only — no e2e `.pte`/tok-s validation is in scope (spec
Assumptions, matching `025`).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Applicability | Status |
|---|---|---|
| I. Correctness Before Performance | Every candidate's throughput is gated on the existing `dq8ca_q4gsw` fp32-reference correctness check, now applied across the full representative shape set (not one shape) before any candidate is ranked or reported (spec FR-003/FR-004). This is a strengthening of, not a deviation from, this principle relative to `025`. | PASS |
| II. M5 EVT1 Is the Only Active Target | All measurement happens on M5 EVT1 (primary and/or secondary board) exclusively. | PASS |
| III. Explicit Eligibility Gating, Safe Fallback Always | New variants extend the existing opt-in `ET_VK_DQ8CA_COOPMAT_VARIANT`-style dispatch token catalog; production `can_use_q4gsw_coopmat`-equivalent gating for `8da4w` is not modified. Productionizing any winner (subgroup=32 or otherwise) is explicit follow-on work, not this feature's scope, matching `025`'s own deferred-shipping precedent. | N/A (documented) |
| IV. Two-Tier, Statistically Sound Benchmarking | Scoped to Tier-1 (shader microbenchmark) only, per spec Assumptions. The final winner's report includes iteration count and stddev/CoV (3-run mean, CoV<5%), matching `025`'s bar. Tier-2 e2e is explicitly out of scope. | PASS (scoped) |
| V. Document Every Driver Workaround at the Point of Use | Directly applicable: this feature's core subject is re-litigating a documented driver-crash workaround (the `SUBGROUP_SIZE=64`-only comment in `linear_dq8ca_qw_coopmat.yaml`/`.glsl`). Whatever this feature's Phase 0 research finds (crash still reproduces at some shapes / fully gone / shape-dependent) MUST be written back into that shader's point-of-use comment, not just this feature's own docs — otherwise the next reader repeats `025`'s exclusion-by-stale-assumption a third time. | PASS (tracked as a deliverable, see research.md) |
| VI. Verify With Tools, Never Assume | Throughput measured via the harness's existing GPU timestamp queries. Coopmat dispatch confirmed via kernel-name capture. Any subgroup=32 winner's compiled SPIR-V is inspected (`spirv-dis`) to confirm genuine int8 cooperative-matrix instructions, matching `025`'s winner-validation step. | PASS |
| VII. Clock Discipline | Clocks pinned and pin-verified before every measurement round. | PASS |
| VIII. Verify the Driver Before Every Coopmat Measurement | Driver hash and device availability re-checked before every measurement round on whichever board is in use (spec FR-010); if both boards are used, each round's own board and hash are recorded. | PASS |
| IX. Never Disclose Samsung-Internal Specifics Upstream | This work stays entirely within this internal workstream on `origin` (`sarc-acl/executorch`); nothing here is proposed upstream. | N/A |
| X. Consult `.shared-context/instruction-for-ai` Before Acting | The Android bench-subproject build sequence (install backend → configure+build `cmake-out-android-vk/bench`) this session had to re-derive from `.artifacts/cmd-log-*.sh` because it isn't documented in `.shared-context/instruction-for-ai/setup/README.md` is written up as part of this feature's research.md output, closing that gap for future readers (see research.md Decision 4) — consistent with this principle's intent even though the gap itself pre-dated this feature. | PASS (gap closed as part of this feature) |

No violations requiring justification — Complexity Tracking is not needed.

## Project Structure

### Documentation (this feature)

```text
specs/026-8da4w-subgroup32-sweep/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md         # Phase 1 output
├── quickstart.md         # Phase 1 output
├── contracts/
│   └── sweep-report-schema.md   # Shape of the final ranked-candidate report,
│                                  # extending 025's schema with a per-shape
│                                  # correctness matrix and a subgroup_size field
├── tasks.md              # Phase 2 output (/speckit-tasks)
└── results/               # Phase 3+ output: legal-space enumeration (both subgroup
                            # sizes), shortlist, multi-shape correctness matrices,
                            # per-round measurement logs, final report,
                            # shader-comment update diff (Principle V deliverable)
```

### Source Code (repository root)

This feature does not introduce a new src/tests tree — it extends the existing `8da4w`
tile/subgroup-sweep shader-variant catalog `025` created and adds a small extension to its
analysis/orchestration scripts. Paths below are relative to the execution worktree (see
Structure Decision) — not this `specs/`-authoring worktree.

```text
backends/vulkan/runtime/graph/ops/glsl/
├── linear_dq8ca_qw_coopmat.glsl                     # production shader -- NOT modified by
│                                                       # this feature's search itself; its
│                                                       # header comment IS updated at the end
│                                                       # per Principle V once Phase 0/3 findings
│                                                       # are known (crash status is now
│                                                       # shape-dependent, not a blanket "crashes")
├── linear_dq8ca_qw_coopmat.yaml                     # production dispatch -- NOT modified
├── linear_dq8ca_q4gsw_coopmat_dbuf{1..4}.glsl/.yaml # existing, from specs/023 -- unchanged,
│                                                       # dbuf2 reused as-is per Assumptions
└── linear_dq8ca_q4gsw_coopmat_tsweep.glsl/.yaml     # EXTENDED (not new): 025's template
                                                        # already threads SUBGROUP_SIZE through
                                                        # as a per-variant yaml parameter; this
                                                        # feature adds shader_variants entries
                                                        # with SUBGROUP_SIZE: 32 (folding in and
                                                        # then retiring the session's ad-hoc
                                                        # sg32test entry, spec FR-012) instead of
                                                        # leaving it fixed at 64 for every entry

backends/vulkan/runtime/graph/ops/impl/
└── QuantizedLinear.cpp     # EXTENDED: dq8ca_coopmat_variant()'s tsweep_t<M>x<N>k<K>g<SGX><SGY>s<sub>
                             # token parser already carries a subgroup field (`s<sub>`) that
                             # 025 always instantiated at 64 -- this feature is the first to
                             # populate it with 32-valued tokens; the session's temporary
                             # `sg32test` literal allow-list entry is removed once superseded

backends/vulkan/test/custom_ops/
└── test_coopmat_linear_bench.cpp   # reused as-is; already shape/variant-generic per 023/025
                                     # precedent -- this feature's new correctness-matrix
                                     # breadth (multiple shapes per candidate) is achieved by
                                     # invoking the existing binary with more shapes, not by
                                     # changing the binary

# Analysis/orchestration tooling (lives with the spec-kit docs in THIS repo,
# not the execution worktree, following specs/022's and specs/025's precedent):
specs/026-8da4w-subgroup32-sweep/scripts/
├── enumerate_configs.py     # Phase 1: re-derive the legal 8da4w tile/subgroup/subgroup-size
│                             # space with SUBGROUP_SIZE ∈ {32, 64} as a swept field, starting
│                             # from 025's enumerate_configs.py / tile_constraints.py and
│                             # removing the SUBGROUP_SIZE=64-only assumption
├── score_and_shortlist.py   # Phase 0/1: analytical cost model + shortlist across both
│                             # subgroup sizes, reusing 025's calibration approach
└── staged_search.py         # Phase 2+: orchestrates per-candidate adb runs including the
                              # broadened multi-shape correctness pass, parses results,
                              # applies the 15%/30-measurement budget cap, halts on
                              # driver/device drift
```

**Structure Decision**: Single project, mirroring `022`/`025`. Spec/plan/tasks documents and
analysis/orchestration scripts live in this repo's `specs/026-8da4w-subgroup32-sweep/` (this
feature's spec-kit home). Actual shader-variant edits, Android build, and on-device
measurement happen in the **existing** `dbuf-int8-sweep` worktree
(`023-8da4w-int8-dbuf-sweep-impl` branch) — a deliberate deviation from `025`'s own
precedent of branching a brand-new experiment worktree off `dev`. Reason: this session
already has working, uncommitted infrastructure sitting in that exact worktree (the `025`
`tsweep` shader/yaml and `QuantizedLinear.cpp` dispatch extension, plus the session's ad-hoc
`sg32test` probe and a functioning two-stage Android build tree at
`cmake-out-android-vk`/`cmake-out-android-vk/bench`), and this feature's entire purpose is to
extend and then retire that exact probe (spec FR-012) — branching a fresh worktree would
duplicate the build-tree bootstrap for no benefit and would leave the ad-hoc probe orphaned
in a worktree this feature never touches, undermining FR-012/SC-007. `dev/executorch` itself
is never checked out onto this feature's working branch, preserving the "never repoint an
existing worktree" rule — the deviation is *which* pre-existing worktree hosts execution, not
a repointing of `dev/`. Results produced there are copied back into this feature's `results/`
directory so the record lives with the spec, per `022`/`025`'s convention.

## Post-Design Constitution Re-Check

Re-evaluated after Phase 1 (data-model.md, contracts/, quickstart.md): no new violations
introduced. The file-based, script-orchestrated design keeps every measurement traceable to
a driver hash, board identity, and pin-verification state
(`MeasurementResult.driver_hash`/`board`/`clocks_pinned` in data-model.md); the correctness
gate is mandatory and now explicitly multi-shape
(`CorrectnessResult.per_shape_results` in data-model.md) before any `MeasurementResult`
counts; the report contract carries the Principle IV stddev/CoV field, the FR-007 head-to-head
comparison against `025`'s winner, and the FR-012/SC-007 probe-disposition statement.
Constitution Check table above still holds: PASS on all applicable principles, N/A on the
rest (documented), no Complexity Tracking entries needed.
