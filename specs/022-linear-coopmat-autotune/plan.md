# Implementation Plan: Smart Autotuning for q4gsw CoopMat Tile Configuration on M5 EVT1

**Branch**: `022-linear-coopmat-autotune` | **Date**: 2026-07-07 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/022-linear-coopmat-autotune/spec.md`

## Summary

Find the best-performing `linear_q4gsw_coopmat` tile configuration among the
642 valid, buffer-storage-only tile geometries without measuring all of them
on the shared M5 EVT1 device. Approach: (1) score all 642 candidates offline
with a zero-device-time analytical cost model (occupancy proxy from
shared-memory footprint + thread count, register-pressure proxy from
per-subgroup accumulator count), calibrated against the 10 real measurements
already collected this session; (2) take the top-ranked candidates plus the
two known anchors (dbuf1, the 128×64/K16/2×2/s32 sweep winner) as a shortlist
of roughly 24-32 configs; (3) run a successive-halving-style staged search on
the shortlist (cheap correctness+single-shape pass → full FLOP-weighted
multi-shape pass on survivors → statistically-rigorous repeated measurement
on the final few) so weak candidates are dropped before they consume real
device time; (4) validate the winner's correctness and report it against the
production baseline and prior sweep winner.

## Technical Context

**Language/Version**: Python 3 (analytical scorer + on-device search
orchestration script); GLSL 450 / C++17 (existing shader template and
dispatch code, unchanged in kind — only new template parameter instantiations
and dispatch-table entries, following the pattern already used for the
`dbuf1-4` and `tsweep_*` variants).

**Primary Dependencies**: the existing `linear_q4gsw_coopmat_tsweep.{glsl,yaml}`
template and its `ET_VK_Q4GSW_COOPMAT_VARIANT` token dispatch in
`QuantizedLinear.cpp`; the existing `test_coopmat_linear_bench` harness
(`COOPMAT_BENCH_CORRECTNESS_ONLY=1` for the correctness gate,
`COOPMAT_BENCH_M` for the production-shape perf pass); the existing Android
cross-build pipeline (NDK r29, `glslc`, ccache); `adb` access to the M5 EVT1;
`pin_freqs.sh` for clock pinning. No new external libraries.

**Storage**: N/A — file-based. The 642-candidate enumeration, analytical
scores, and per-round measurement results are plain JSON/CSV/Markdown under
this feature's `specs/022-linear-coopmat-autotune/` directory (input data and
results), not a database.

**Testing**: the existing small-shape, fp32-reference correctness check
(`COOPMAT_BENCH_CORRECTNESS_ONLY=1`) already used throughout this workstream,
reused as-is as the pass/fail gate (Constitution Principle I). No new
correctness methodology is introduced.

**Target Platform**: Samsung M5 EVT1 (Exynos 2500 / Xclipse 970), Android,
pinned 509/2730/663 MHz GPU/MIF/INT clocks (Constitution Principle VII).

**Project Type**: Single project — a bounded, internal research/automation
addition layered on the existing Vulkan backend repo. No new application,
service, or public interface.

**Performance Goals**: identify a configuration whose FLOP-weighted GFLOP/s
(the same 12-13 production Llama prefill shape set, M=2048, already used by
`jira-tile-sweep.md` and this session's follow-up tests) is at least as good
as the current 128×64/K16/2×2/s32 sweep winner — see spec Success Criteria
SC-001.

**Constraints**: buffer weight storage only (per the standing instruction
that scoped this feature to WMMA/coopmat-relevant storage); loop structure
held fixed at the dbuf1 ("prefetch-first") shape; no more than ~96
configurations (15% of 642) may receive any real on-device measurement (spec
SC-001); total on-device measurement time must be at least 5x less than an
exhaustive, fully-rigorous run of all 642 configs would take (spec SC-002);
driver identity and device availability MUST be re-verified before every
measurement round, not just once at the start (Constitution Principles
VII/VIII); no performance number counts until its configuration passes the
correctness gate (Constitution Principle I).

**Scale/Scope**: 642 valid buffer-storage tile geometries is the full search
universe (enumerated and cross-validated against 10 real on-device results
earlier this session). Target shortlist after analytical pruning: ~24-32
configs. This feature is scoped to the **Tier-1 shader-microbenchmark** level
only (Constitution Principle IV) — it does not export a `.pte` or measure
end-to-end tokens/sec; Tier-2 e2e validation of a chosen winner, if it is
later productionized, is explicit follow-on work outside this feature's
scope (see spec Assumptions).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Applicability | Status |
|---|---|---|
| I. Correctness Before Performance | Every candidate's GFLOP/s number is gated on the existing fp32-reference correctness check before it is ranked or reported (FR-004). | PASS |
| II. M5 EVT1 Is the Only Active Target | All measurement happens on the M5 EVT1; no other device is used as a validation platform. | PASS |
| III. Explicit Eligibility Gating, Safe Fallback Always | Not applicable in this feature's scope: this work extends the existing experimental `ET_VK_Q4GSW_COOPMAT_VARIANT` variant catalog (already an opt-in, non-production toggle), it does not modify the production `can_use_q4gsw_coopmat` dispatch gate. Productionizing a winner would need to satisfy this principle as separate follow-on work. | N/A (documented) |
| IV. Two-Tier, Statistically Sound Benchmarking | Scoped to Tier-1 (shader microbenchmark) only, per Assumptions in spec.md. The final winner's report must include iteration count and stddev (`get_avg_time_us()`/`get_std_dev_us()`), not just a single-run average, to satisfy Tier-1's own rigor bar. Tier-2 e2e is explicitly out of scope. | PASS (scoped) |
| V. Document Every Driver Workaround at the Point of Use | Applies only if the search surfaces a new compile/driver failure (as the 128×64/K16/4×4 attempt already did once this session). Any such finding gets an inline comment per this principle. | Conditional — will apply if triggered |
| VI. Verify With Tools, Never Assume | GFLOP/s is measured via the harness's existing GPU timestamp queries, not estimated. The final winner's compiled SPIR-V is inspected (`spirv-dis` or equivalent) to confirm the expected cooperative-matrix instructions are present, per this principle. | PASS |
| VII. Clock Discipline | Clocks are pinned and pin-verified (via the GFLOP/s cross-check already established this session) before every measurement round, not assumed to persist from a prior round. | PASS |
| VIII. Verify the Driver Before Every Coopmat Measurement | Driver hash and device availability are re-checked before every measurement round (not only once), per FR-007. | PASS |
| IX. Never Disclose Samsung-Internal Specifics Upstream | This work stays entirely within the internal `quant-perf-optimization` workstream; nothing here is proposed upstream. | N/A |
| X. Consult `.shared-context/instruction-for-ai` Before Acting | Build, device-access, and clock-pinning steps reuse the already-established docs/scripts from this session (`build.md`, `devices-and-access.md`, `pin_freqs.sh`) rather than re-deriving them. | PASS |

No violations requiring justification — Complexity Tracking is not needed.

## Project Structure

### Documentation (this feature)

```text
specs/022-linear-coopmat-autotune/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md         # Phase 1 output
├── quickstart.md         # Phase 1 output
├── contracts/
│   └── autotune-report-schema.md   # Shape of the final ranked-candidate report
├── tasks.md              # Phase 2 output (/speckit-tasks)
└── results/               # Phase 3+ output: enumeration data, shortlist,
                            # per-round measurement logs, final report
```

### Source Code (repository root)

This feature does **not** introduce a new src/tests tree — it extends an
existing shader-variant catalog and adds a small standalone automation
script, following this workstream's existing pattern (capability probe →
prototype → benchmark → generalize).

```text
# Execution worktree note (see research.md "Where this executes"): the
# tsweep shader family this feature extends exists only as uncommitted work
# in the isolated experiment worktree created earlier this session
# (.artifacts/tsweep-256x256-smoketest/executorch, branch
# exp/tsweep-256x256-4x4-smoketest — NOT this quant-perf-optimization
# worktree). Paths below are relative to that worktree's repo root.

backends/vulkan/runtime/graph/ops/glsl/
├── linear_q4gsw_coopmat_tsweep.glsl   # unchanged (template already
│                                       # supports arbitrary tile params)
└── linear_q4gsw_coopmat_tsweep.yaml   # extended: one shader_variants
                                        # entry per shortlisted candidate

backends/vulkan/runtime/graph/ops/impl/
└── QuantizedLinear.cpp                # extended: one token in
                                        # coopmat_variant_tile() +
                                        # kTokens[] per shortlisted candidate

backends/vulkan/test/custom_ops/
└── test_coopmat_linear_bench.cpp      # unchanged (existing
                                        # ET_VK_Q4GSW_COOPMAT_VARIANT env-var
                                        # selection + COOPMAT_BENCH_M /
                                        # COOPMAT_BENCH_CORRECTNESS_ONLY
                                        # toggles are sufficient as-is)

# New, this feature (lives with the spec-kit docs in THIS repo, not the
# execution worktree, since it's analysis tooling, not product shader code):
specs/022-linear-coopmat-autotune/scripts/
├── enumerate_configs.py    # Phase 1: regenerate + validate the 642-config
│                            # universe from the known constraint model
├── score_and_shortlist.py  # Phase 0/1: analytical cost model + shortlist
└── staged_search.py        # Phase 2+: orchestrates per-candidate adb runs,
                             # parses results, applies the successive-halving
                             # budget, halts on driver/device drift
```

**Structure Decision**: Single project. Documentation and analysis/orchestration
scripts live in this repo's `specs/022-linear-coopmat-autotune/` (this is
where the constitution-governed spec-kit workflow for this workstream lives).
The actual shader-variant edits, Android build, and on-device measurement
execute in the existing isolated experiment worktree
(`.artifacts/tsweep-256x256-smoketest/executorch`) created earlier this
session, which already carries the uncommitted `tsweep`/`dbuf1-4` shader
family and has a warm, working Android build — reusing it avoids forking yet
another worktree and rebuilding from scratch (see research.md). Results
produced there are copied back into this feature's `results/` directory so
the record lives with the spec, per this workstream's existing convention
(e.g. spec 013's handoff-report pattern).

## Post-Design Constitution Re-Check

Re-evaluated after Phase 1 (data-model.md, contracts/, quickstart.md):
no new violations introduced. The file-based, script-orchestrated design
keeps every measurement traceable to a driver hash and pin-verification
state (`MeasurementResult.driver_hash`/`clocks_pinned` in data-model.md),
keeps the correctness gate mandatory before any `MeasurementResult` counts,
and the report contract mandates the Principle IV stddev/iteration-count
fields for the final winner. Constitution Check table above still holds:
PASS on all applicable principles, N/A on the rest (documented), no
Complexity Tracking entries needed.
