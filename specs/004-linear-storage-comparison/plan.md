# Implementation Plan: Linear Shader Storage-Type Baseline Study (Texture3D vs. Buffer)

**Branch**: `004-linear-storage-comparison` | **Date**: 2026-07-04 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/004-linear-storage-comparison/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Extend `001`'s existing no-WMMA microbenchmark (`test_llama_baseline_bench.cpp`) with a second storage-type axis (`Texture3D` vs. `Buffer`), covering the same 96 (model, scheme, regime, op) cases now at both storage types (192 total), always run with `ET_VK_FORCE_TILED_LINEAR=1` set so both storage types measure the same tiled algorithm. Verified directly in the harness source before writing this plan: the microbenchmark constructs 2D `{M, K}` tensors (no batch dim) and every prefill shape already satisfies coopmat's tile alignment (`M%64==0`, `N%64==0`, `K%32==0` — confirmed against `GemmCoopmat.h`'s constants for all 8 ops × 3 models) — meaning, unlike the real e2e model (blocked by rank-3 output per `003`), Buffer storage alone **would** make every prefill case coopmat-eligible in this harness. The forcing toggle is therefore not a formality here; it is strictly required for every prefill case, confirmed by reading the actual eligibility checks in order rather than assumed.

## Technical Context

**Language/Version**: C++ (matching the existing prototyping benchmark harness, built via the project's CMake presets) for the harness extension; Python 3.10+ (`uv`-managed `.venv`, matching every prior tool in this workstream) for capture parsing and the comparison/report script.

**Primary Dependencies**: The existing `backends/vulkan/test/custom_ops/test_llama_baseline_bench.cpp` and its shared prototyping utilities (`utils.h`'s `TestCase`/`ValueSpec`/`execute_test_cases`); `001`'s `ET_VK_FORCE_TILED_LINEAR` env-var toggle (`QuantizedLinear.cpp`); `001`'s exact build recipe (`EXECUTORCH_BUILD_TESTS=ON` etc.).

**Storage**: Flat files — the modified harness's `RESULT,...` CSV lines (now with a `storage` column) captured to a log file, and `specs/004-linear-storage-comparison/results/storage-comparison-report.md` as the consolidated deliverable.

**Testing**: Statistical significance via non-overlapping `mean ± 2·stdev` intervals between a case's `Texture3D` and `Buffer` measurements — a transparent, stdev-based heuristic consistent with how this workstream already reasons about noise (e.g. `005`'s target noise band), not a new ad hoc method. Cross-validated against `001`'s already-published `Texture3D` numbers for the same shapes as a sanity check that the harness extension didn't regress the existing baseline.

**Target Platform**: `rocky-ryzen` MiniPC — this is a **real device** feature (like `001`), unlike `002`/`003`/`005`'s pure analysis/tooling. Requires an actual build and GPU capture, under the same resource-contention discipline (no concurrent CPU/GPU-heavy processes) established since `001`'s mid-implementation correction.

**Project Type**: Extends an existing internal C++ benchmark suite; adds one Python analysis script.

**Performance Goals**: N/A — this feature measures performance, it does not have its own throughput/latency target.

**Constraints**: MUST run every case (both storage types) with `ET_VK_FORCE_TILED_LINEAR=1` set, for the entire process — not per-case — since it is a single env-var check at dispatch time and every case in this study wants the tiled comparison; MUST NOT let a `Buffer`-storage prefill case silently dispatch coopmat; MUST verify the captured kernel name for every case is the tiled/coop family, never a `*_coopmat` name, as a hard check, not an assumption.

**Scale/Scope**: 3 models × 2 schemes × 2 regimes × 8 ops × 2 storage types = 192 cases, double `001`'s existing 96-case microbenchmark.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Checked against `.specify/memory/constitution.md` (Vulkan Cooperative-Matrix (WMMA) GEMM Constitution, v1.1.0):

- **I. Correctness Before Performance** — PASS. Only a benchmark harness and analysis script are touched; no runtime dispatch logic changes.
- **II. Samsung RDNA3 iGPU Is the Target, Not a Fallback** — PASS. Same `rocky-ryzen` proxy hardware already validated throughout this workstream.
- **III. Explicit Eligibility Gating, Safe Fallback Always** — PASS, and directly exercised: this feature deliberately forces past the coopmat eligibility gate using the same already-documented toggle (not a new, undocumented hack) to produce a controlled ablation.
- **IV. Two-Tier, Statistically Sound Benchmarking** — PASS. This is squarely Tier 1 (microbenchmark) work, reusing the existing harness's mean/stdev-over-N-iterations discipline; the significance test (non-overlapping stdev bands) keeps this consistent with the constitution's statistical rigor requirement rather than eyeballing raw numbers.
- **V. Document Every Driver Workaround at the Point of Use** — Directly applies: the harness's header comment (currently stating "every case uses Texture3D+Half output storage") MUST be updated to document the dual-storage coverage and the `ET_VK_FORCE_TILED_LINEAR` requirement at the point where a future reader would need it, not just in this spec/plan.

No violations identified. Complexity Tracking is not needed.

*Post-Phase-1 re-check*: Phase 1's data model and contract keep the significance determination formula and the coopmat-eligibility verification (kernel-name check) explicit and machine-checkable, so Principle III/IV are enforced by the tooling itself, not left to manual review.

## Project Structure

### Documentation (this feature)

```text
specs/004-linear-storage-comparison/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md        # Phase 1 output (/speckit-plan command)
├── quickstart.md        # Phase 1 output (/speckit-plan command)
└── contracts/           # Phase 1 output (/speckit-plan command)
```

### Source Code (repository root)

```text
# Modifies one existing benchmark file; adds one new analysis script.

backends/vulkan/test/custom_ops/test_llama_baseline_bench.cpp   # MODIFIED:
  # add a storage-type axis (Texture3D, Buffer) to the existing cross-product
  # (3 models x 2 schemes x 2 regimes x 8 ops), doubling 96 -> 192 cases;
  # add a `storage` column to the RESULT CSV line; update the header comment
  # to document dual-storage coverage and the ET_VK_FORCE_TILED_LINEAR
  # requirement (Constitution Principle V)

specs/004-linear-storage-comparison/
├── scripts/
│   └── compare_storage.py   # parses captured RESULT lines, pairs
│                              # Texture3D/Buffer per case, computes relative
│                              # difference + significance, verifies kernel
│                              # names are tiled/coop (never *_coopmat),
│                              # renders the consolidated report
└── results/
    ├── raw/storage_bench_raw.log     # captured RESULT lines (192 rows)
    └── storage-comparison-report.md  # US3's deliverable

# Read-only references (not modified):
specs/001-minipc-baseline-benchmarks/results/raw/*.json   # cross-check source
```

**Structure Decision**: Modify the existing microbenchmark in place rather
than duplicating it into a new file — it already owns the per-model shape
catalog (`kModels`) and case-generation pattern (`generate_cases`/`make_case`);
duplicating that ~140-line catalog into a second file would create two
sources of truth that could silently drift. The new Python script lives
alongside this feature's other artifacts, matching `001`'s convention of
keeping analysis tooling under `specs/<feature>/scripts/`.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations — this section is intentionally empty.
