# Implementation Plan: End-to-End tok/s Report — Texture, Buffer, and WMMA Across 4w/8da4w

**Branch**: `009-e2e-tokrate-report` | **Date**: 2026-07-04 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/009-e2e-tokrate-report/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Consolidate this workstream's `4w`/`8da4w` coopmat investigation into one
real end-to-end tok/s report: reuse `006`'s already-captured `Texture3D`/
`Buffer` e2e numbers, and add a new WMMA-arm e2e capture for every
configuration whose actual coopmat dispatch is confirmed via ETDump
(constitution Principle VI) -- not assumed from eligibility-gate logic alone.
Producing a WMMA-eligible export requires resolving the rank-3 output
blocker `003` found (both prefill linear activations and outputs are
rank-3 `[1, M, K]`, batch always 1, never squeezed): research (Decision 1)
grounds a narrow, verified-safe relaxation of `can_use_q4gsw_coopmat()`'s
guard as the mechanism, to be proposed here and applied only with explicit
user authorization (FR-009), alongside `007`'s already-authorized but still
uncommitted `linear_q4gsw` registration fix for `4w`. The final report states
two per-scheme verdicts (never one blended number, per `007`'s own
precedent for this exact data shape) on whether WMMA actually helps this
device's real token generation rate.

## Technical Context

**Language/Version**: C++ (the guard relaxation in `QuantizedLinear.cpp`,
plus a new small rank-3 correctness check) for the production-code half;
Python 3.10+ (`uv`-managed `.venv`) for export CLI usage and the
comparison/report script, matching every prior feature in this workstream.

**Primary Dependencies**:
- `can_use_q4gsw_coopmat()` (`QuantizedLinear.cpp:161-216`) -- the
  eligibility gate whose rank check is relaxed (research.md Decision 1).
- `007`'s `linear_q4gsw` registration fix (`Q4gswLinear.cpp`/
  `QuantizedLinear.cpp`, currently uncommitted) -- required for `4w`'s
  coopmat path to be reachable at all.
- `006`'s `--vulkan-storage-override buffer` CLI flag (already committed) --
  required for `Buffer` storage, the coopmat path's other hard precondition.
- `002`'s existing ETDump capture pipeline (`--etdump_path`,
  `EXECUTORCH_BUILD_DEVTOOLS`) and per-op `kernel_name` extraction -- reused
  verbatim for FR-003's dispatch confirmation.
- `006`'s, `007`'s, and `008`'s already-published reports, read as
  read-only inputs (Texture3D/Buffer e2e numbers, microbenchmark-level
  speedup findings, shipped-config tuning finding).

**Storage**: Flat files -- new `Buffer`-storage WMMA-eligible `.pte` exports
under this feature's own `results/pte/`, ETDump captures under
`results/etdump/`, e2e capture logs under `results/raw/`, and
`specs/009-e2e-tokrate-report/results/e2e-tokrate-report.md`.

**Testing**: A new small rank-3 (batch=1) correctness check against the CPU/
tiled reference for both coopmat shaders (research.md Decision 2 -- existing
2D-only tests do not cover this shape class), plus `006`'s coherent/
non-degenerate e2e smoke-check, plus the ETDump dispatch-confirmation check
(FR-003) -- three layers, none a substitute for another.

**Target Platform**: `rocky-ryzen` MiniPC -- real device work (build,
export, ETDump capture, e2e capture), matching every prior tier-2 feature.

**Project Type**: A small, targeted production dispatch-code change (the
guard relaxation, pending explicit authorization) plus reuse of
`001`/`002`/`006`'s existing export/ETDump/e2e-capture tooling and
conventions.

**Performance Goals**: N/A -- this feature measures and consolidates
performance; it does not carry its own target (that's `005`'s role).

**Constraints**:
- The guard relaxation MUST NOT change behavior for any already-passing
  rank-2 or genuine-batch (>1) case -- verified safety property, mirroring
  `006`'s "default behavior provably unchanged" bar.
- No WMMA tok/s number is reported for a configuration unless its
  ETDump-captured kernel names are confirmed `_coopmat` (FR-003) --
  eligibility-gate logic passing is not sufficient evidence (constitution
  Principle VI).
- The WMMA arm measures only the shipped/default tile configuration
  (`WG_TILE_M=128, WG_TILE_N=64, WG_TILE_K=32, SUBGROUP_SIZE=64`, FR-008) --
  `008`'s config 5 finding is unreachable through `can_use_q4gsw_coopmat()`'s
  hard `subgroup_size() == 64` requirement and is out of scope here.
- Prefill comparisons against `006`'s reused numbers inherit `006`'s own
  documented cross-session variance caveat (research.md Decision 5) -- this
  MUST be stated alongside any prefill divergence, not silently dropped.
- No concurrent GPU load during any capture (established workstream
  discipline).

**Scale/Scope**: 3 models x 2 int4 schemes = 6 configurations (constitution
default scope), each needing a new WMMA-eligible `Buffer`-storage export, an
ETDump dispatch check, and (if confirmed) an e2e capture -- compared against
`006`'s already-captured `Texture3D`/`Buffer` numbers for that same
configuration.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Checked against `.specify/memory/constitution.md` (v1.4.0):

- **I. Correctness Before Performance (NON-NEGOTIABLE)**: PASS, conditioned
  on implementation adding the new rank-3 correctness check (research.md
  Decision 2) -- existing 2D-only tests do not cover the shape class the
  guard relaxation newly allows through, so citation alone is not enough
  here (unlike `007`'s fix, which only changed registration for
  already-covered shapes).
- **II. Samsung RDNA3 iGPU Is the Target, Not a Fallback**: PASS with scope
  note -- `rocky-ryzen` MiniPC only, consistent with every prior tier-2
  feature; Samsung/Xclipse validation of these numbers is a future feature,
  not silently skipped.
- **III. Explicit Eligibility Gating, Safe Fallback Always**: PASS. The
  relaxation narrows what `can_use_q4gsw_coopmat()` rejects but keeps it an
  explicit, testable check with the same safe tiled fallback whenever it
  doesn't hold (a genuine batch > 1 still falls back, exactly as today).
- **IV. Two-Tier, Statistically Sound Benchmarking**: PASS, tier-2
  (model-level) -- reuses `001`/`006`'s exact e2e methodology and JSON
  shape; every reported tok/s number carries its run count (FR-004).
- **V. Document Every Driver Workaround at the Point of Use**: Applies to
  the guard relaxation itself -- the inline comment at the change site MUST
  name what was relaxed (size-1 leading dim vs. genuine batch) and why it's
  safe, matching `007`'s wiring-fix comment style.
- **VI. Verify With Tools, Never Assume**: PASS, central to this feature --
  FR-003's ETDump-based dispatch confirmation is exactly this principle's
  model-level clause, applied for the first time in this workstream at
  tier-2 (prior features applied it at tier-1 via harness kernel-name/
  SPIR-V checks).

No violations identified. Complexity Tracking is not needed -- the one
production change (the guard relaxation) is a small, well-precedented,
single-function change in the same spirit as `006`'s dead-code restoration
and `007`'s registration fix, not new architecture.

*Post-Phase-1 re-check*: Phase 1's data model keeps `export_status`,
`smoke_check_status`, `dispatch_status`, and `correctness_status` as
separate, explicitly-reported fields per WMMA-Eligible Export (never folded
into a single pass/fail or into the timing number itself), so Principles I
and VI stay enforced by the data structure, not just by convention -- same
discipline `006`'s data model already established for its own smoke-check.

## Project Structure

### Documentation (this feature)

```text
specs/009-e2e-tokrate-report/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   └── e2e-tokrate-report-schema.md
└── tasks.md              # Phase 2 output (/speckit-tasks, not this command)
```

### Source Code (repository root)

One small production change, plus reuse of existing export/ETDump/e2e
tooling:

```text
backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp   # MODIFIED (pending
  # explicit authorization): can_use_q4gsw_coopmat()'s rank guard relaxed
  # per research.md Decision 1; linear_q4gsw() registration fix (007,
  # already authorized, currently uncommitted) also lands as part of this
  # feature's build
backends/vulkan/runtime/graph/ops/impl/Q4gswLinear.cpp        # 007's fix
  # (uncommitted): et_vk.linear_q4gsw.default registration moved out of
  # this file

backends/vulkan/test/custom_ops/  # NEW: small rank-3 (batch=1) correctness
  # check for both coopmat shaders (research.md Decision 2) -- exact file
  # placement decided at /speckit-tasks time, following this directory's
  # existing test-ownership conventions (008's precedent)

specs/009-e2e-tokrate-report/
├── scripts/
│   └── compare_e2e_tokrate.py   # new: reads 006's report (Texture3D/Buffer),
│                                  # 007's report (microbench finding), 008's
│                                  # report (tuning finding), this feature's
│                                  # new WMMA e2e capture, renders the report
└── results/
    ├── pte/       # new WMMA-eligible Buffer-storage .pte exports (up to 6)
    ├── etdump/    # new ETDump captures, one per export (FR-003)
    ├── raw/       # new WMMA-arm e2e capture logs/JSON
    └── e2e-tokrate-report.md

# Read-only references (not modified):
specs/006-e2e-storage-comparison/results/e2e-storage-comparison-report.md
specs/007-wmma-improvement-microbench/results/wmma-improvement-report.md
specs/008-8da4w-parameter-sweep/results/sweep-report.md
```

**Structure Decision**: Same lightweight documentation/analysis structure as
`006`/`007`/`008`: one new Python comparison/report script under this
feature's own `scripts/`, reading prior features' published reports as
read-only inputs. The one substantive difference from `006`/`007`/`008`:
this feature requires a small, explicitly-authorized change to production
dispatch code (`QuantizedLinear.cpp`'s eligibility gate) plus new test
coverage for the shape class it newly allows through -- both scoped and
justified above, not new architecture.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations -- this section is intentionally empty.
