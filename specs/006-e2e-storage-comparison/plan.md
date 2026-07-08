# Implementation Plan: End-to-End Texture3D vs. Buffer Storage Comparison

**Branch**: `006-e2e-storage-comparison` | **Date**: 2026-07-04 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/006-e2e-storage-comparison/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Extend `004`'s microbenchmark-level "storage switch is basically free" finding to the real end-to-end model, across all six configurations. This requires actually producing a `Buffer`-storage export of each model — which, per dedicated research before this plan was written, requires a small, low-risk fix to dead code in the Vulkan backend: a `storage_type_override` mechanism (`CompileSpec` → `VulkanPartitioner` → `TagMemoryMetaPass.default_storage`) is already fully wired end-to-end but has had zero effect since a 2025-07-31 pass rewrite (PR #12927) dropped the code that consulted it (`backends/vulkan/utils.py`'s `make_tensor_repr()` now unconditionally prefers `TEXTURE_3D`). Restoring that check is safe: the mechanism's default value is already `TEXTURE_3D` and nothing today has any path to request anything else, so this changes nothing for existing callers — it only enables the new, explicitly-opt-in `Buffer` path this feature needs.

## Technical Context

**Language/Version**: Python (matching `export_llama_lib.py`'s existing codebase) for the export-side fix and new CLI flag; Python 3.10+ (`uv`-managed `.venv`) for capture/comparison tooling, matching every prior feature.

**Primary Dependencies**:
- `backends/vulkan/utils.py`'s `TensorRepSet.make_tensor_repr()` (`utils.py:964-987`) and `backends/vulkan/_passes/tag_memory_meta_pass.py`'s `TagMemoryMetaPass` (`default_storage` at line 150/155) — the dead-code path being restored.
- `backends/vulkan/partitioner/vulkan_partitioner.py`'s `VulkanPartitioner` (already accepts `storage_type_override` as a compile option, `parse_compile_options`/`parse_compile_spec`) and `extension/llm/export/partitioner_lib.py`'s `get_vulkan_partitioner()` — where the new CLI flag gets forwarded, mirroring the existing `--vulkan-force-fp16`/`force_fp16` pattern exactly.
- `001`'s exact e2e capture methodology and fixed workload (now codified in the constitution's "Default Scope for Every Benchmark").

**Storage**: Flat files — new `Buffer`-storage `.pte` exports under this feature's own `results/pte/` (kept separate from `001`'s existing `Texture3D` `.pte`s, never overwriting them), e2e capture logs, and `specs/006-e2e-storage-comparison/results/e2e-storage-comparison-report.md`.

**Testing**: A basic smoke-check (the model runs without crashing and produces coherent, non-garbage output for a fixed prompt) before trusting any timing — per the Clarifications session, this is not a numerical-equivalence re-verification (that's assumed, an existing ExecuTorch/Vulkan-backend guarantee), just a check that this feature's own export/config didn't break something.

**Target Platform**: `rocky-ryzen` MiniPC — real device work (build, export, e2e capture), matching `001`.

**Project Type**: A small, real (non-hack) fix to existing Vulkan-backend export code, plus reuse of `001`'s e2e capture/reporting conventions.

**Performance Goals**: N/A — this feature measures performance, it does not have its own target.

**Constraints**: The `make_tensor_repr()`/`TagMemoryMetaPass` fix and the new CLI flag MUST NOT change default export behavior for anyone not passing the new flag (verified: `default_storage_type` already defaults to `TEXTURE_3D`, matching today's hardcoded behavior exactly). Buffer storage's `128MiB` default limit (`DEFAULT_BUFFER_LIMIT`, `utils.py:591`) has no active enforcement today (`within_buffer_limit()` is dead code, never called) — a large tensor (e.g. `lm_head`'s `N=128256` at prefill `M=2048` is a multi-hundred-MB fp16 buffer) could plausibly exceed real Vulkan buffer limits at runtime with no clean pre-check; this MUST be watched for and reported as a blocked/failed configuration (FR-006) if it occurs, not silently worked around.

**Scale/Scope**: 3 models × 2 schemes = 6 configurations, per the constitution's default scope, each needing a new `Buffer`-storage `.pte` export plus an e2e capture, compared against `001`'s already-existing `Texture3D` e2e numbers.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Checked against `.specify/memory/constitution.md` (Vulkan Cooperative-Matrix (WMMA) GEMM Constitution, v1.2.0):

- **I. Correctness Before Performance** — PASS, directly enforced by FR-002/FR-006: no timing number is reported for a configuration that fails its smoke-check.
- **II. Samsung RDNA3 iGPU Is the Target, Not a Fallback** — PASS. Same `rocky-ryzen` proxy hardware.
- **III. Explicit Eligibility Gating, Safe Fallback Always** — PASS. The new `Buffer`-storage path is strictly opt-in via an explicit new flag; default export behavior is provably unchanged (verified default value matches today's hardcoded behavior).
- **IV. Two-Tier, Statistically Sound Benchmarking** — PASS. Reuses `001`'s exact tier-2 (model-level) methodology and the constitution's now-explicit fixed workload/scope default.
- **V. Document Every Driver Workaround at the Point of Use** — Applies to the dead-code fix itself: the restored check and the reason it was dead (which PR dropped it, why the default value makes this safe) MUST be documented as an inline comment at the fix site, not just in this plan.

No violations identified. Complexity Tracking is not needed — this is a small, well-precedented, low-risk change (adding one more forwarded CLI option, exactly matching the existing `--vulkan-force-fp16` pattern), not new architecture.

*Post-Phase-1 re-check*: Phase 1's data model and contract keep the smoke-check and the buffer-limit risk as explicit, separately-reported fields (never silently folded into a timing number), so Principle I stays enforced by the data structure itself, not just by convention.

## Project Structure

### Documentation (this feature)

```text
specs/006-e2e-storage-comparison/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md        # Phase 1 output (/speckit-plan command)
├── quickstart.md        # Phase 1 output (/speckit-plan command)
└── contracts/           # Phase 1 output (/speckit-plan command)
```

### Source Code (repository root)

```text
# Real product code changes (small, precedented) + reuse of 001's e2e tooling.

backends/vulkan/_passes/tag_memory_meta_pass.py   # MODIFIED: restore the
  # dropped check that consults self.default_storage for ambiguous repsets
backends/vulkan/utils.py                           # MODIFIED: make_tensor_repr()
  # (or its caller) honors the storage preference instead of unconditionally
  # preferring TEXTURE_3D
extension/llm/export/partitioner_lib.py             # MODIFIED: get_vulkan_partitioner()
  # forwards a new storage-override option, mirroring force_fp16
examples/models/llama/export_llama_lib.py           # MODIFIED: new CLI flag
  # (e.g. --vulkan-storage-override buffer), mirroring --vulkan-force-fp16

specs/006-e2e-storage-comparison/
├── scripts/
│   └── compare_e2e_storage.py   # parses e2e capture stats logs for both
│                                  # storage variants, compares against 004's
│                                  # finding, renders the consolidated report
└── results/
    ├── pte/                      # new Buffer-storage .pte exports (6 configs)
    ├── raw/                      # e2e capture logs/stats for the Buffer variant
    └── e2e-storage-comparison-report.md

# Read-only references (not modified):
specs/001-minipc-baseline-benchmarks/results/raw/*.json   # Texture3D e2e baseline
specs/004-linear-storage-comparison/results/storage-comparison-report.md  # microbenchmark-level finding
```

**Structure Decision**: Small, targeted fixes to existing Vulkan-backend
export code (restoring dead code + adding one new forwarded CLI option,
exactly matching the already-existing `--vulkan-force-fp16` pattern) plus a
tooling layer under this feature's own `specs/` directory, matching `001`'s
convention. No new project/architecture.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations — this section is intentionally empty.
