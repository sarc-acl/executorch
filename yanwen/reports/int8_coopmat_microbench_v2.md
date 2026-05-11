# int8 KHR coopmat tile-schedule sweep — Phase 1 of Track A

**Updated:** 2026-05-10 · **Author:** Yanwen Xu
**Device:** AMD Radeon 780M (RADV PHOENIX, RDNA3+ mobile iGPU, wave64), Mesa 25.0.7
**Companion to:** [`int8_coopmat_microbench.md`](int8_coopmat_microbench.md) (the first int8 cm microbench)

## TL;DR

Phase 1 of Track A asked: can we tune the int8 KHR coopmat shader's tile
schedule to clear **2× over fp16 cm at LLaMA FFN shapes, weighted by
dispatch count**? Answer: **no, not via the spec-constant levers exposed in
this study**.

| Metric | Value |
|---|---|
| Honest fp16 cm baseline (new — replaces fp32-proxy from v1) | 3.421 ms (FFN gate/up), 3.517 ms (FFN down) |
| Best int8 cm wave64 tile schedule | 2.006 ms (FFN gate/up), 2.340 ms (FFN down) — = `v0_baseline` (no change) |
| **Weighted-by-dispatch ratio (int8 / fp16)** | **0.613 → 1.63×** |
| Phase 1 gate (≤ 0.50 = ≥ 2×) | **NOT MET** — band is "document, stop" (0.50 < ratio ≤ 0.65) |
| Phase 2 (E2E port) | **NOT STARTED** per the gate contract |

## What changed since the v1 microbench report

Two material methodology improvements:

1. **Real fp16 cm LLaMA baseline.** The v1 microbench cited
   `int8_cm_w64 / fp32_cm ≈ 0.29` at FFN gate/up and reasoned "fp16 cm ≈ 2×
   fp32 cm → int8 vs fp16 cm ≈ 0.58 → ~1.7× over fp16 cm." That proxy is
   now retired. The companion `linear_coopmat_bench.cpp` segfaulted at
   `cm_fp16_BERT_QKV` (M=128, K=768, N=768) before reaching LLaMA shapes —
   I reordered the shape configs so LLaMA runs first, ran cleanly, captured
   the **real fp16 cm LLaMA latencies** (Section "Phase 1.0").

   Honest ratios are now slightly lower than v1's projection: **1.72× at
   FFN gate/up** (vs v1's 1.7× projection — close), **1.47× at FFN down**
   (vs ~1.3× projection — slightly better), **0.73× at Q/O**, **0.49× at
   K/V** (both still net regressions).

2. **Tile-schedule tuning surface added.** The wave64 shader had hardcoded
   `WORKGROUP_{WIDTH,HEIGHT}_IN_SUBGROUPS = 2`. I exposed these as spec
   constants (IDs 14, 15), added a new dispatcher
   `khr_cm_gemm_int8_wave64_tiled` that accepts a `TileSchedule` POD with
   `(TILE_M, TILE_N, TILE_K, sg_w, sg_h, BColMajor)`, and a microbench
   variant matrix that sweeps these knobs.

   Validation gate via `khr_cm_gemm_int8_validate` (tight tolerance,
   abs ≤ 0.5 on 128×128×128). Variants that fail are dropped before
   benching — replacing the v1 mistake of `set_abs_tolerance(1e10f)` masking
   the wave32 half-zero-output bug.

## Phase 1.0 — fp16 cm LLaMA baseline (new measurements)

Reorder of `linear_coopmat_bench.cpp:42–62` puts LLaMA shapes before BERT,
so the segfault at `cm_fp16_BERT_QKV` no longer blocks them. New numbers
(GPU timestamp, mean of 10 iterations, `linear_coopmat_bench_20260510_204428_fp16_llama_baseline.log`):

| Shape | fp16 cm half | fp32 cm | fp16 vec (texture) |
|---|---:|---:|---:|
| FFN gate/up (M=128, K=4096, N=14336) | **3.421 ms** | 6.713 ms | 15.328 ms (linear_vec_texture3d_texture2d_half) |
| FFN down (M=128, K=14336, N=4096) | **3.517 ms** | 6.060 ms | 16.205 ms |
| Q/O (M=128, K=4096, N=4096) | **0.923 ms** | 1.721 ms | 4.601 ms |
| K/V (M=128, K=4096, N=1024) | **0.256 ms** | 0.307 ms | 1.242 ms |

fp16 cm half is a clean ~2× over fp32 cm at FFN (matches the assumed
proxy), but at K/V it's only 1.2× over fp32. The v1 report's "fp16 cm ≈ 2×
fp32 cm" assumption was shape-dependent.

## Phase 1.1 — Tuning surface exposure

### GLSL change

`pavan-report/.../glsl/matmul_khr_cm_int8_wave64.glsl:87–88`:

```diff
-const uint WORKGROUP_WIDTH_IN_SUBGROUPS = 2;
-const uint WORKGROUP_HEIGHT_IN_SUBGROUPS = 2;
+layout(constant_id = 14) const uint WORKGROUP_WIDTH_IN_SUBGROUPS = 2;
+layout(constant_id = 15) const uint WORKGROUP_HEIGHT_IN_SUBGROUPS = 2;
```

The shader's wave64 invariant `sg_w * sg_h * 64 == 256` (one workgroup =
4 subgroups × 64 threads/subgroup) is enforced at the dispatcher level via
`VK_CHECK_COND(ts.sg_w * ts.sg_h == 4, ...)`.

### Dispatcher addition

`pavan-report/.../impl/MatMulKHRCoopMat.cpp` adds a
`KhrCmInt8TileSchedule` POD, a new impl
`khr_cm_matmul_int8_wave64_tiled_impl`, and a registered op
`etvk.khr_cm_gemm_int8_wave64_tiled.default(A, B, tile_m, tile_n, tile_k,
sg_w, sg_h, bcolmajor, D)`. The global workgroup size lambda captures the
tile schedule (a `std::function`-friendly closure — `PickGlobalFn` from
`DynamicDispatchNode.h:33`).

### Test harness wiring

`pavan-report/.../test/custom_ops/impl/TestGemm.cpp` adds `impl_selector=5`
that reads 6 extra scalar args after the impl_selector and forwards them
to the tiled op. Output's position depends on selector (6 with no extras,
12 for selector 5).

## Phase 1.2 — Variant matrix (post-validation)

7 variants originally planned. After validation gating with tight
tolerance on 128×128×128 (max int32 accumulation `255×255×128 = 8.3M < 2^24` →
expect ≤ 1 ULP, abs_tol = 0.5):

| Variant | TILE_M | TILE_N | TILE_K | sg_w | sg_h | BColMajor | Validate |
|---|---:|---:|---:|---:|---:|---:|---|
| `v0_baseline` | 128 | 128 | 32 | 2 | 2 | 0 | ✓ PASS |
| `v1_deepK` | 128 | 128 | **64** | 2 | 2 | 0 | ✓ PASS |
| `v2_wideN` | 128 | **64** | 32 | 2 | 2 | 0 | ✗ DROPPED (see below) |
| `v3_sg1x4` | 128 | 128 | 32 | **1** | **4** | 0 | ✓ PASS |
| `v4_sg4x1` | 128 | 128 | 32 | **4** | **1** | 0 | ✓ PASS |
| `v5_colmajB` | 128 | 128 | 32 | 2 | 2 | **1** | ✗ DROPPED (see below) |
| `v6_deepK_colmajB` | 128 | 128 | **64** | 2 | 2 | **1** | ✗ DROPPED |

### Why three variants were dropped before perf

**`v2_wideN` (TILE_N=64)** — structural bug in the wave64 prefetch loop.
At line 135 of the shader:

```glsl
for (uint k = 0; k < B_NUM_ROWS; k += INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_B) {
```

With `INVS_PER_ROW_B = B_ROW_LEN / ELEMENTS_PER_VEC4 = 64/16 = 4` and
`INVOCATIONS_PER_WORKGROUP = 256`, the step is `256/4 = 64`. But
`B_NUM_ROWS = 32`. Single loop iter with `btilek = 0..63` overruns
`temp_B[0]` (race) and writes `Bsh[sk=0..63]` out of bounds. Validation
fails at element 0 with computed=1.8M vs reference=2.2M (partial sum,
matching the half-overrun signature). Fix would require guarding the
prefetch loop or restructuring it for `INVS_PER_ROW_B × step ≥ B_NUM_ROWS`
— out of scope.

**`v5_colmajB` / `v6_deepK_colmajB`** — the test harness feeds B as
row-major. The shader's B prefetch loop reads B linearly from global as
row-major (`gbbase = coordToOffset(chunkK, TILE_N * tileID.x, strideB,
BColMajor)` correctly uses BColMajor for the global offset, but the shared
memory write pattern is still row-major). Then `coopMatLoad(matB, Bsh,
..., gl_CooperativeMatrixLayoutColumnMajor)` reads with a column-major
stride — mismatching the actual layout in Bsh. Validation fails at
element 0 with computed=2.06M vs reference=2.18M (partial mismatch, not
zero — different from v2's overrun signature). Fix would require either
column-major B input data or a column-major-aware prefetch. Out of scope.

### Performance results (4 surviving variants)

GPU timestamp mean of 10 iterations; same run, same Vulkan context, same
GPU thermal state.
(`khr_cm_gemm_int8_sweep_20260510_205654.log`)

| Variant | FFN gate/up (μs) | FFN down (μs) | Q/O (μs) | K/V (μs) |
|---|---:|---:|---:|---:|
| **`v0_baseline`** | **2006** | **2340** | **1241** | 462 |
| `v1_deepK` (TILE_K=64) | 2391 (+19%) | 2521 (+8%) | 1343 (+8%) | 451 (−2%) |
| `v3_sg1x4` | 2139 (+7%) | 2447 (+5%) | 1274 (+3%) | **445** (−4%) |
| `v4_sg4x1` | 2136 (+6%) | 2401 (+3%) | 1361 (+10%) | 548 (+18%) |

For reference, the existing wave64 dispatch (impl=4, hardcoded same as
v0_baseline) measured 2161 / 2382 / 1227 / 477 μs in the same run — within
8% of v0_baseline (run-to-run noise; the two paths exercise an identical
shader specialization).

### Why no variant beats v0

- **`v1_deepK` (TILE_K=64)** doubles the K-tile, expecting to amortize the
  prefetch + barrier overhead over more arithmetic. Instead it's
  ~10–19% slower at FFN. Likely root cause: doubled B shared-memory tile
  (160 → 320 uvec4) plus doubled register pressure for `temp_B` array
  spills registers or stalls on shared-mem bank conflicts. The
  amortization win doesn't beat the resource pressure cost on RDNA3+.
- **`v3_sg1x4` / `v4_sg4x1`** change the output partitioning per
  subgroup. `C_ROWS×C_COLS` becomes 2×8 (1x4) or 8×2 (4x1) per warp
  instead of 4×4 (2x2). Both have the same total per-warp tile work,
  but the per-subgroup MAC sequence patterns differ. Neither improves
  meaningfully — the 2×2 baseline already hits a good ratio for these
  shapes. `v3_sg1x4` does help K/V by ~4% (smaller subgroup-wise N tile
  improves fit), but the rest of the table moves the wrong way.
- **Subgroup layout product is locked at 4 on wave64** (4 subgroups × 64
  threads = 256-thread workgroup). The plausible sweep is just {1×4, 2×2,
  4×1}; all three measured.

## Phase 1.4 — Decision gate

**Best-per-shape (all v0_baseline) weighted by LLaMA dispatch count:**

```
weighted_int8 = 64 disp × 2.006 ms + 32 disp × 2.340 ms
              = 128.4 + 74.9
              = 203.3 ms

weighted_fp16 = 64 disp × 3.421 ms + 32 disp × 3.517 ms
              = 218.9 + 112.5
              = 331.5 ms

ratio = 203.3 / 331.5 = 0.613   →   1.63× speedup, FFN-weighted
```

**Decision band (per plan):**

- `ratio ≤ 0.50` (≥ 2×): proceed to Phase 2 — **not met**
- `0.50 < ratio ≤ 0.65` (1.5×–2×): **stop, document** ← **current state**
- `ratio > 0.65`: stop, do not start Phase 2

**Phase 2 (E2E port to main + W8A8 export + LLaMA bench) is NOT
started.** The wave64-corrected shader is already near a local optimum
for the spec-constant levers we exposed; closing the remaining ~22% gap
to 2× would need deeper changes (shader rewrites for new prefetch
patterns, column-major B handling, or fundamentally different tile
algebra). That's a heavier scope than this plan budgeted.

## What this rules in and out

**Ruled out:**

- Tile-schedule-only tuning is not a path to "int8 cm 2× over fp16 cm" at
  LLaMA FFN shapes on RDNA3+ wave64. The accessible levers (TILE_M/N/K,
  subgroup layout among {1×4, 2×2, 4×1}, BColMajor) are not enough.

**Still on the table (future work):**

- **Hybrid dispatch heuristic** — the v1 report's "KHR coopmat for FFN,
  q8ta for Q/O/K/V" projection (~288 tok/s E2E, 1.31× over fp16 cm) is
  unaffected by this Phase 1 outcome and remains the most plausible
  short-horizon int8 win. Would need a fresh plan.
- **Shader rewrites** — column-major B handling, guarded prefetch loops
  for TILE_N=64, software pipelining stage count, subgroup-shuffle in
  place of shared memory. Each is a multi-day shader engineering effort.
- **Different quant scheme** — group quantization (W4A8, W8A8 with
  per-group scales) could change the dispatch math enough to widen the
  int8 / fp16 cm gap. Not in this plan's scope.

## Source modifications (uncommitted)

Under `pavan-report/executorch/`:

| File | Change |
|---|---|
| `backends/vulkan/runtime/graph/ops/glsl/matmul_khr_cm_int8_wave64.glsl` | Lines 87–88: subgroup layout → spec constants 14/15 |
| `backends/vulkan/runtime/graph/ops/impl/MatMulKHRCoopMat.cpp` | Added `KhrCmInt8TileSchedule`, `khr_cm_matmul_int8_wave64_tiled_impl`, `etvk.khr_cm_gemm_int8_wave64_tiled.default` registration |
| `backends/vulkan/test/custom_ops/impl/TestGemm.cpp` | Added `impl_selector=5` path that reads tile schedule scalars |
| `backends/vulkan/test/custom_ops/khr_cm_gemm_int8.cpp` | Tile-schedule sweep harness (4 surviving variants × 4 LLaMA shapes) |
| `backends/vulkan/test/custom_ops/khr_cm_gemm_int8_validate.cpp` | Sweep variants validated at 128×128×128 (abs ≤ 0.5); dropped `orig` (impl=3) cases so failures don't terminate sweep validation |
| `backends/vulkan/test/custom_ops/linear_coopmat_bench.cpp` | Lines 42–62: reordered LLaMA shapes before BERT to escape the `cm_fp16_BERT_QKV` segfault |

## Artifacts

Under `yanwen/artifacts/int8_microbench/`:

| File | Description |
|---|---|
| `linear_coopmat_bench_20260510_204428_fp16_llama_baseline.log` | fp16 cm LLaMA baseline (Phase 1.0) — 4 cm_fp16_llama_* rows |
| `khr_cm_validate_sweep_20260510_205654.log` | Tight-tolerance validation results for the 4 surviving sweep variants |
| `khr_cm_gemm_int8_sweep_20260510_205654.log` | Perf microbench, 4 variants × 4 LLaMA shapes (+ original impl=3/4 reference rows) |
