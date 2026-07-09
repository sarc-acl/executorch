# Implementation Plan: 8da4w (dq8ca/q4gsw) CoopMat Tile/Subgroup Parameter Sweep on M5 EVT1

**Branch**: `025-8da4w-parameter-sweep` | **Date**: 2026-07-09 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/025-8da4w-parameter-sweep/spec.md`

## Summary

Re-confirm on M5 EVT1 that `dbuf2` is the fastest loop structure for the currently-shipped
`8da4w` (`linear_dq8ca_q4gsw_coopmat`) tile/subgroup geometry (User Story 1), then — holding
that loop structure fixed — apply `022`'s smart-autotune methodology (analytical pruning →
staged on-device search → validated winner) to the shader's tile-shape × subgroup-grid ×
subgroup-size space, re-derived for `8da4w`'s int8-MMA register/shared-memory constraints
rather than reusing `4w`'s 642-candidate enumeration. Search budget: ≤15% of the legal
`8da4w` space, hard-capped at 30 real on-device measurements. Representative shapes: the
same 6-shape set (`wq` + `w1_gate` × {1B, 3B, 8B}) `022` and `023` used. Final winner is
validated at 3-run-mean/CoV<5% rigor and reported against both the currently-shipped
`8da4w` baseline and `4w`'s 128×64/K16/2×2/s32 winner.

## Technical Context

**Language/Version**: Python 3 (legal-space enumeration + analytical scorer + staged-search
orchestration script, following `022`'s `scripts/enumerate_configs.py` /
`score_and_shortlist.py` / `staged_search.py` pattern); GLSL 450 / `GL_KHR_cooperative_matrix`
+ C++17 (existing `linear_dq8ca_q4gsw_coopmat` shader template and its dispatch code —
new work is new template parameter instantiations and dispatch-table entries, not new
shader logic)

**Primary Dependencies**: the existing `linear_dq8ca_q4gsw_coopmat_dbuf{1..4}` shader family
and `ET_VK_DQ8CA_COOPMAT_VARIANT` dispatch token from `specs/023-8da4w-int8-dbuf-sweep`
(User Story 1 reuses this directly); a new tile/subgroup-parameterized template for the
`8da4w` shader analogous to `linear_q4gsw_coopmat_tsweep.{glsl,yaml}` (`022`), instantiated
at the fixed winning `dbuf` loop shape; the existing `test_coopmat_linear_bench` harness
(`COOPMAT_BENCH_CORRECTNESS_ONLY=1` correctness gate, production-shape perf pass); the
existing Android cross-build pipeline (NDK, `glslc`, ccache); `adb` access to M5 EVT1;
clock-pinning script. No new external libraries.

**Storage**: N/A — file-based. Enumerated legal-space data, analytical scores, and
per-round measurement results are JSON/CSV/Markdown under this feature's
`specs/025-8da4w-parameter-sweep/results/`, not a database.

**Testing**: the existing small-shape, fp32-reference `dq8ca_q4gsw` correctness check
(`COOPMAT_BENCH_CORRECTNESS_ONLY=1`), reused as-is as the pass/fail gate (Constitution
Principle I). No new correctness methodology.

**Target Platform**: Samsung M5 EVT1 (Exynos 2500 / Xclipse 970), Android, pinned clocks
(Constitution Principle VII); driver identity re-verified before every measurement round
(Principle VIII).

**Project Type**: Single project — a bounded, internal research/automation addition to the
existing Vulkan backend, following `022`'s precedent exactly.

**Performance Goals**: identify an `8da4w` tile/subgroup configuration whose FLOP-weighted
throughput across the 6-shape set is at least as good as the currently-shipped configuration
(spec SC-002), reported alongside the loop-structure re-confirmation result (SC-001) and the
`4w` cross-shader comparison (SC-004).

**Constraints**: loop structure held fixed at the User-Story-1-confirmed winner for the
entire tile/subgroup sweep (spec Assumptions — treated as a limitation, not silently
absorbed, if User Story 1 finds loop-structure/geometry interaction); search budget ≤15% of
the legal `8da4w` space, hard-capped at 30 on-device measurements (spec FR-007, Clarified
2026-07-09); no performance number counts until its configuration passes the correctness
gate (Principle I); driver identity and device availability re-verified before every
measurement round, not just once (Principles VII/VIII); default production `8da4w` dispatch
behavior MUST remain unchanged — all new variants are opt-in via env-var-selected
dispatch tokens, never on by default.

**Scale/Scope**: the `8da4w`-specific legal tile/subgroup space (expected smaller than `4w`'s
642 due to int8-MMA's different accumulator/shared-memory footprint — exact count is a
Phase 0 research output, not assumed here). Target shortlist after analytical pruning:
proportional to `022`'s ~24-32-config shortlist, scaled to the re-derived space size, and
never exceeding the 30-measurement hard cap end to end. Tier-1 (shader microbenchmark) only
— no e2e `.pte`/tok-s validation is in scope (spec Assumptions).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Applicability | Status |
|---|---|---|
| I. Correctness Before Performance | Every candidate's throughput is gated on the existing `dq8ca_q4gsw` fp32-reference correctness check before it is ranked or reported (spec FR-005). | PASS |
| II. M5 EVT1 Is the Only Active Target | All measurement — User Story 1's loop re-confirmation and the tile/subgroup search — happens on M5 EVT1 exclusively. | PASS |
| III. Explicit Eligibility Gating, Safe Fallback Always | New variants extend the existing opt-in `ET_VK_DQ8CA_COOPMAT_VARIANT`-style dispatch token catalog; production `can_use_q4gsw_coopmat`-equivalent gating for `8da4w` is not modified. Productionizing a winner is explicit follow-on work, not this feature's scope. | N/A (documented) |
| IV. Two-Tier, Statistically Sound Benchmarking | Scoped to Tier-1 (shader microbenchmark) only, per spec Assumptions. The final winner's report includes iteration count and stddev/CoV (3-run mean, CoV<5%, spec Clarified 2026-07-09), not a single-run average. Tier-2 e2e is explicitly out of scope. | PASS (scoped) |
| V. Document Every Driver Workaround at the Point of Use | Applies only if the tile/subgroup search surfaces a new compile/driver failure (as `022`'s 128×64/K16/4×4 attempt did once). Any such finding gets an inline comment per this principle. | Conditional — will apply if triggered |
| VI. Verify With Tools, Never Assume | Throughput is measured via the harness's existing GPU timestamp queries, never estimated. Coopmat dispatch is confirmed via kernel-name capture (not assumed from eligibility-gate logic). The final winner's compiled SPIR-V is inspected (`spirv-dis` or equivalent) to confirm genuine int8 cooperative-matrix instructions are present. | PASS |
| VII. Clock Discipline | Clocks pinned and pin-verified before every measurement round in both User Story 1 and the tile/subgroup search, not assumed to persist across rounds. | PASS |
| VIII. Verify the Driver Before Every Coopmat Measurement | Driver hash and device availability re-checked before every measurement round (spec FR-008), not only once. | PASS |
| IX. Never Disclose Samsung-Internal Specifics Upstream | This work stays entirely within this internal workstream on `origin` (`sarc-acl/executorch`); nothing here is proposed upstream. | N/A |
| X. Consult `.shared-context/instruction-for-ai` Before Acting | Build, device-access, and clock-pinning steps reuse the already-established docs/scripts (per `022`/`023` precedent) rather than re-deriving them. | PASS |

No violations requiring justification — Complexity Tracking is not needed.

## Project Structure

### Documentation (this feature)

```text
specs/025-8da4w-parameter-sweep/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md         # Phase 1 output
├── contracts/
│   └── sweep-report-schema.md   # Shape of the final ranked-candidate report
├── tasks.md              # Phase 2 output (/speckit-tasks)
└── results/               # Phase 3+ output: legal-space enumeration, shortlist,
                            # dbuf re-confirmation log, per-round measurement logs,
                            # final report
```

### Source Code (repository root)

This feature does not introduce a new src/tests tree — it extends the existing
`8da4w` shader-variant catalog and adds a small standalone automation script,
following `022`'s and `023`'s established pattern. Paths below are relative to
the dedicated experiment worktree this feature's code changes and on-device
measurements execute in (see Structure Decision) — not this `specs/`-authoring
worktree.

```text
backends/vulkan/runtime/graph/ops/glsl/
├── linear_dq8ca_qw_coopmat.glsl                    # production shader -- NOT modified
├── linear_dq8ca_qw_coopmat.yaml                    # production dispatch -- NOT modified
├── linear_dq8ca_q4gsw_coopmat_dbuf{1..4}.glsl/.yaml # existing, from specs/023 -- reused
│                                                     # as-is for User Story 1's re-confirmation
└── linear_dq8ca_q4gsw_coopmat_tsweep.glsl/.yaml     # new: tile/subgroup-parameterized
                                                      # template, analogous to specs/022's
                                                      # linear_q4gsw_coopmat_tsweep, built on
                                                      # top of the User-Story-1-winning dbuf
                                                      # loop shape; one shader_variants entry
                                                      # per shortlisted candidate

backends/vulkan/runtime/graph/ops/impl/
└── QuantizedLinear.cpp     # extended: one token in a new dq8ca_q4gsw coopmat_variant_tile()
                             # + kTokens[] (mirroring 022's pattern for the fp16 shader),
                             # additive to the existing ET_VK_DQ8CA_COOPMAT_VARIANT dbuf
                             # selection from specs/023; default (both env vars unset)
                             # dispatch behavior unchanged

backends/vulkan/test/custom_ops/
└── test_coopmat_linear_bench.cpp   # extended (or reused if already shape/variant-generic)
                                     # to time dq8ca_q4gsw_coopmat tile/subgroup variants,
                                     # one variant per process (isolates pipeline-creation
                                     # crashes, per specs/023 precedent)

# Analysis/orchestration tooling (lives with the spec-kit docs in THIS repo,
# not the execution worktree, following specs/022's precedent):
specs/025-8da4w-parameter-sweep/scripts/
├── enumerate_configs.py    # Phase 1: derive + validate the legal 8da4w tile/subgroup
│                            # configuration space under int8-MMA register/shared-memory
│                            # constraints (NOT a reuse of 022's 642-candidate 4w
│                            # enumeration -- see research.md Decision 1)
├── score_and_shortlist.py  # Phase 0/1: analytical cost model + shortlist, calibrated
│                            # against User Story 1's dbuf re-confirmation measurements
└── staged_search.py        # Phase 2+: orchestrates per-candidate adb runs, parses
                             # results, applies the 15%/30-measurement budget cap,
                             # halts on driver/device drift
```

**Structure Decision**: Single project, mirroring `022`/`023` exactly. Spec/plan/tasks
documents and analysis/orchestration scripts live in this repo's
`specs/025-8da4w-parameter-sweep/` (this feature's spec-kit home). Actual shader-variant
edits, Android build, and on-device measurement happen in a dedicated new git worktree
branched from the tip of `yanwen/dev-1.3` (this feature's spec/plan/tasks are committed to
`dev` first so the new worktree inherits them), per this workspace's "create a new worktree,
never repoint an existing one" rule — `dev/executorch` itself is never checked out onto this
feature's working branch. The new worktree is bootstrapped per constitution "Environment &
Build Bootstrap" before any build is attempted (`./install_executorch.sh --minimal` — this
is a fresh worktree, no pre-existing venv). Results produced there are copied back into this
feature's `results/` directory so the record lives with the spec.

## Post-Design Constitution Re-Check

Re-evaluated after Phase 1 (data-model.md, contracts/, quickstart.md): no new violations
introduced. The file-based, script-orchestrated design keeps every measurement traceable to
a driver hash and pin-verification state (`MeasurementResult.driver_hash`/`clocks_pinned` in
data-model.md), keeps the correctness gate mandatory before any `MeasurementResult` counts,
and the report contract carries the Principle IV stddev/CoV field plus the dbuf
re-confirmation result required by spec SC-001. Constitution Check table above still holds:
PASS on all applicable principles, N/A on the rest (documented), no Complexity Tracking
entries needed.
