# MiniPC RDNA3 Handoff Report

This closes out the `rocky-ryzen` MiniPC phase of the Vulkan
cooperative-matrix (WMMA) workstream (specs `001`-`012`) before moving to
Samsung/Xclipse hardware. It consolidates every finding into one document,
states the repo's actual current state, and gives a starting runbook for
the next machine.

## Consolidated Findings

| Spec | Headline | Tier | Source |
|---|---|---|---|
| `001-minipc-baseline-benchmarks` | Baseline e2e/microbench numbers established, coopmat/WMMA excluded (default ExecuTorch behavior) -- the comparison floor for every feature after | n/a (baseline) | `specs/001-minipc-baseline-benchmarks/results/baseline-report.md` |
| `002-etdump-shader-profiling` | ETDump-based dispatch-confirmation methodology established -- the tier-1/tier-2 verification tooling this whole workstream reuses | n/a (tooling) | `specs/002-etdump-shader-profiling/results/profiling-report.md` |
| `003-wmma-shader-candidates` | WMMA candidates identified: prefill linear GEMM blocked by a rank-3-output + `TEXTURE_3D`-storage bug (not missing code); SDPA had no WMMA implementation at the time; decode linear GEMV has no WMMA-capable kernel (structural) | n/a (candidate survey) | `specs/003-wmma-shader-candidates/results/wmma-candidates-report.md` |
| `004-linear-storage-comparison` | `Buffer` vs. `TEXTURE_3D` storage is effectively free for the large majority of cases (46/48 prefill, 35/48 decode) | microbenchmark | `specs/004-linear-storage-comparison/results/storage-comparison-report.md` |
| `005-e2e-speedup-target` | E2E speedup target set -- a goal-setting feature, no measured result of its own | n/a (target-setting, no results file) | N/A |
| `006-e2e-storage-comparison` | `004`'s finding confirmed at e2e scale once cross-session prefill variance (a real hardware property, not a storage effect) is controlled for | e2e | `specs/006-e2e-storage-comparison/results/e2e-storage-comparison-report.md` |
| `007-wmma-improvement-microbench` | Linear coopmat microbench: `4w` **+60.6% faster**; `8da4w` **-15.2% slower** (regression) | microbenchmark | `specs/007-wmma-improvement-microbench/results/wmma-improvement-report.md` |
| `008-8da4w-parameter-sweep` | Best-found `8da4w` tuning config averages **+18.2%** vs. shipped across representative shapes; a real shader bug found and fixed along the way | microbenchmark | `specs/008-8da4w-parameter-sweep/results/sweep-report.md` |
| `009-e2e-tokrate-report` | E2E tok/s: `4w` **77.8% faster** (consistent with `007`); `8da4w` **3.2% slower** (consistent direction, smaller e2e magnitude). Found and fixed the `force_fp16`/storage-override conflict that had silently defeated `Buffer` storage workstream-wide until this feature | e2e | `specs/009-e2e-tokrate-report/results/e2e-tokrate-report.md` |
| `010-sdpa-coopmat-microbench` | SDPA coopmat microbench: **+66.8% average**, all 3 models real-effect, dispatch + correctness + SPIR-V confirmed | microbenchmark | `specs/010-sdpa-coopmat-microbench/results/sdpa-coopmat-microbench-report.md` |
| `011-sdpa-coopmat-e2e` | SDPA coopmat e2e: **+27.3% average** across 6/6 configurations, all consistent in direction with `010` | e2e | `specs/011-sdpa-coopmat-e2e/results/sdpa-coopmat-e2e-report.md` |
| `012-decode-wmma-feasibility` | Decode linear GEMV roofline analysis: **memory-bandwidth-bound by a 12-50x margin** -- not worth building a WMMA shader for. Two alternative directions named (more aggressive quantization; batching/speculative decoding for a real `M>1`) | n/a (analytical, not measured) | `specs/012-decode-wmma-feasibility/results/decode-wmma-feasibility-report.md` |

**In one sentence**: prefill linear coopmat is a strong, real win for `4w`
and a real (smaller) regression for `8da4w`; SDPA coopmat is a strong, real
win for both schemes at prefill; decode is not a WMMA opportunity on this
hardware, full stop.

## Open Items

Explicitly not resolved by any prior feature:

1. **`8da4w` production-gating decision** -- `007`/`009` both confirm a
   real regression for `8da4w` linear coopmat (microbench and e2e). No
   decision has been made on whether to gate coopmat off for `8da4w` in
   production, accept the regression, or pursue further tuning beyond
   `008`'s sweep. **Deferred by the user -- the project owner has no plan
   yet.**
2. **Decode SDPA's two GEMV kernels** (`sdpa_compute_attn_weights_coop`,
   `sdpa_compute_out_coop`) -- `012` only analyzed decode's *linear* GEMV.
   Decode SDPA shares the same `M=1` structural property (confirmed via
   ETDump: both dispatch once per decode step at `M=1`), so the same
   bandwidth-bound conclusion is expected, but this was never itself
   measured or roofline-analyzed.
3. **Two directions `012` named but did not pursue**: more aggressive
   weight quantization (a separate quantization-research effort with real
   accuracy-risk tradeoffs, not a shader change -- explicitly agreed not
   worth chasing further here), and batching/speculative decoding
   (batching doesn't fit the on-device single-user target at all;
   speculative decoding could apply but is a model-serving/algorithm
   feature orthogonal to this workstream's Vulkan-backend scope).
4. **Samsung/Xclipse validation itself** -- every finding above is
   `rocky-ryzen` MiniPC-only. None of it has been confirmed on the actual
   target device (constitution Principle II). This is precisely what the
   next phase (starting with the Runbook below) exists to close.

## Repo Handoff State

- **Current branch**: `quant-perf-optimization`
- **Last actual commit**: `d8800fb02e` ("Amend constitution: add Verify
  With Tools, Never Assume")
- **Uncommitted**: 71 files changed (working tree + staged, not committed)
  -- this spans **every spec from `007` through `013`** and their
  underlying production-code fixes:
  - `backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp` (rank-3
    coopmat-guard relaxation)
  - `backends/vulkan/runtime/graph/ops/impl/Q4gswLinear.cpp` (dead
    registration fix)
  - `backends/vulkan/_passes/tag_memory_meta_pass.py` (the
    `force_fp16`/storage-override conflict fix)
  - `backends/vulkan/runtime/graph/ops/impl/SDPA.cpp` and the SDPA coopmat
    shader files (imported from a sibling branch this session)
  - `backends/vulkan/runtime/graph/ops/impl/GemmCoopmat.cpp`/`.h` (a
    restored Xclipse spec-constant workaround)
  - New test files across `backends/vulkan/test/op_tests/` and
    `backends/vulkan/test/custom_ops/`
  - All of `specs/007-*` through `specs/013-*`

**Consequence**: a fresh `git clone` of this repo's current remote branch,
today, would contain **only specs `001`-`006`** -- none of the actual
WMMA/SDPA coopmat findings, fixes, or this report itself.

**Required prerequisite for a useful clone elsewhere**: commit and push
these changes first. This report does not perform that commit/push --
per this repo's own convention, commits happen only when explicitly
asked. Confirm explicitly before doing so.

(The `.pte` model export files are a separate matter: they are gitignored
and were never part of this concern. Re-export them independently on the
Samsung/Xclipse machine -- see the Runbook below.)

## Samsung/Xclipse Runbook

A starting checklist, not a tested pipeline -- this session's environment
has no `adb`/device access to develop or validate an Android build. Status
per methodology step:

| Step | Status | Notes |
|---|---|---|
| **Methodology itself** (two-tier benchmarking discipline, ETDump dispatch-confirmation habit, SPIR-V inspection habit) | **Carries over unchanged** | These are the constitution's own Principles IV/VI, not `rocky-ryzen`-specific mechanics. Apply them exactly as this workstream has throughout. |
| **Export** (`.pte` generation) | Needs adaptation | Re-export independently for the Android/Xclipse target -- likely a different export configuration than this workstream's x86 desktop builds (per Clarifications; see `/export` reference docs). |
| **Build** | Needs adaptation | This workstream's `cmake --preset "linux"` recipe (constitution's Reference Build Recipe) is x86-desktop-specific. An Android/ARM build with the Vulkan delegate needs its own preset/toolchain -- not worked out here. |
| **Deploy & run** | Newly established | This workstream has only ever run `llama_main` directly, locally. Getting a build, export, and prompt file onto the device and a result back via `adb push`/`adb shell`/`adb pull` is new -- no existing recipe to point to. |
| **Dispatch-confirm (ETDump)** | Carries over unchanged (mechanism), needs adaptation (delivery) | The `Inspector`-based kernel-name extraction itself doesn't change; getting the `.etdump` file off-device via `adb pull` before parsing it does. |
| **Benchmark (tier-1/tier-2)** | Carries over unchanged (methodology), needs adaptation (invocation) | Same iteration-count/variance discipline; commands need an `adb shell`-wrapped equivalent. |
| **Report** | Carries over unchanged | Same report shape this workstream has used throughout (findings + citations + explicit open items). |

**First thing to check on the new device, before trusting any of the above
carries over**: confirm Xclipse's cooperative-matrix support and tile
dimensions via `test_coopmat_probe.cpp` (constitution's own Reference
Hardware Inventory already flags this as unverified -- the attached
Samsung `SM-S926B` is *believed* not to expose cooperative-matrix support
at all; a second, WMMA-capable Samsung device exists but wasn't connected
as of the constitution's last update). Every RDNA3 finding in this report
assumes a 16×16×16 Subgroup-scope cooperative-matrix configuration
matching `rocky-ryzen`'s -- do not assume Xclipse matches until the probe
confirms it.

## Overall

Twelve features' worth of work is complete and consistent: prefill linear
and SDPA coopmat both show large, real, tool-confirmed wins on this
device (with `8da4w` linear's regression as the one open exception);
decode is conclusively not a WMMA target here. None of it is committed to
git yet, and none of it has been validated on the actual target hardware
(Samsung/Xclipse) -- both are the explicit next steps, not gaps in this
phase's own completeness.
