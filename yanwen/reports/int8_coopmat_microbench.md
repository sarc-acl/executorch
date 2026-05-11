# int8 coopmat shader microbenchmark — does the 2× / 4× math hold?

**Updated:** 2026-05-10 · **Author:** Yanwen Xu
**Device:** AMD Radeon 780M (RADV PHOENIX, RDNA3+ mobile iGPU), Mesa 25.0.7
**Companion to:** [`L32_S128_int8_baseline_REPORT.md`](L32_S128_int8_baseline_REPORT.md) — Phase 1 of the int8 study, the E2E baseline. This is Phase 2 — the shader-level microbench.

> **⚠ Correctness note (2026-05-10 second pass).** An earlier version of this report cited `matmul_khr_cm_int8` timings of 1.43 ms (FFN gate/up), 1.58 ms (FFN down), 0.87 ms (Q/O), 0.37 ms (K/V) and projected ~305 tok/s E2E. **Those numbers were wrong.** The shader has a hardcoded wave32 assumption (`INVOCATIONS_PER_WORKGROUP = 32 * NUM_SUBGROUPS`) that fails on the 780M (default wave64): only 4 of the expected 8 subgroups actually run, and the second vertical half of every output tile stays zero. The bench harness used `tc.set_abs_tolerance(1e10f)` which trivially passed correctness, so the bug was invisible until I wrote a tight-tolerance validation driver (`khr_cm_gemm_int8_validate.cpp`, pavan-report `c331b9cf5`). All numbers below use the **wave64-correct** variant `matmul_khr_cm_int8_wave64.glsl`. The wave32-broken shader timings are kept in the table only as `int8 cm (orig, BROKEN)` for transparency — they are ~1.5× faster than the correct shader because they do half the work.

## Why this study

Phase 1 showed the default `vulkan_8w` int8 path runs **19% slower** than fp16 baseline at L=32 S=128 prefill (60.7 vs 72.5 tok/s). That's because `linear_qcs8w_tiled_*_half_o4x4` is per-dispatch slower than `linear_vec_buffer_texture2d_half` at large N (FFN gate/up: 20.2 ms vs 12.3 ms). The W8A16 weight-bandwidth saving is real but doesn't beat the per-element dequant overhead at these shapes.

The user's hypothesis: a *real* int8 win on this hardware requires the KHR cooperative-matrix int8 path. Specifically:

- **(H1) int8 coopmat is ~2× faster than fp16 coopmat** (int8 tensor-core throughput)
- **(H2) coopmat is ~4× faster than non-coopmat** at the same precision

This phase tests both hypotheses directly at the shader level, without exporting/running a full LLaMA forward.

## Setup

Three pavan-report custom-ops binaries, each compiled standalone:

| Binary | What it benches |
|---|---|
| `khr_cm_gemm_int8` | `matmul_khr_cm_int8.glsl` (broken, impl_selector=3) and `matmul_khr_cm_int8_wave64.glsl` (correct, impl_selector=4). Both are true `coopmat<uint8_t, ...>` WMMA shaders; pure int8×int8 → fp output. |
| `linear_coopmat_bench` | `linear_vec_*` (texture3d → `linear_vec`) vs `linear_coopmat_*` (buffer → `linear_coopmat`) — fp32 and fp16 variants (fp16 added in this study). |
| `q8csw_linear` | `linear_q8csw_tiled_*` (W8A16, non-coopmat) and `linear_q8ta_q8csw_*` (W8A8 with int8-dot-product extension, non-coopmat). |

All four shapes from LLaMA 3.1 8B prefill at S=128 were added to the config lists of each binary:

| Tag | Shape (M, K, N) | LLaMA component |
|---|---|---|
| `llama_ffn_gateup` | 128, 4096, 14336 | FFN gate / up |
| `llama_ffn_down` | 128, 14336, 4096 | FFN down |
| `llama_qo` | 128, 4096, 4096 | Q / O proj |
| `llama_kv` | 128, 4096, 1024 | K / V proj |

(decode-shape M=1 omitted — coopmat's M≥128 tile requirement excludes it.)

Each binary uses 3 warmup + 10 timed runs and reports GPU-timestamp latency per kernel.

## Headline results

Per-shape per-shader latencies at LLaMA prefill shapes (best across storage variants):

| Component | Shape | fp32 vec | fp32 cm | int8 q8csw (W8A16) | int8 q8ta (W8A8) | **int8 cm (wave64, CORRECT)** | int8 cm (orig, BROKEN) |
|---|---|---:|---:|---:|---:|---:|---:|
| `llama_ffn_gateup` | M=128, K=4096, N=14336 | 15.18 ms | 6.78 ms | 9.23 ms | 3.27 ms | **2.00 ms** | 1.35 ms |
| `llama_ffn_down` | M=128, K=14336, N=4096 | 16.76 ms | 6.16 ms | 9.62 ms | 3.19 ms | **2.40 ms** | 1.54 ms |
| `llama_qo` | M=128, K=4096, N=4096 | 4.59 ms | 1.78 ms | 2.74 ms | **0.78 ms** | 1.27 ms | 0.80 ms |
| `llama_kv` | M=128, K=4096, N=1024 | 1.24 ms | 0.31 ms | 0.82 ms | **0.32 ms** | 0.53 ms | 0.31 ms |

**The fastest int8 shader is shape-dependent.** Wave64 KHR coopmat dominates at FFN shapes (2.0–2.4 ms vs q8ta's 3.2 ms, a 1.6× lift). At Q/O and K/V, the W8A8 q8ta path is actually FASTER than wave64 KHR coopmat (0.78 vs 1.27 ms at Q/O; 0.32 vs 0.53 ms at K/V). q8ta is a scalar shader that uses the `int8_dot_product` extension's 4-way int8 fused MAC instruction; at small N it has lower launch overhead than the cooperative-matrix tile.

**Caveat: no fp16 LLaMA microbench data.** The fp16 variants in `linear_coopmat_bench` were added but a segfault in `cm_fp16_BERT_QKV` (M=128 K=768 N=768) terminated the run before fp16 LLaMA shapes executed. fp16-related ratios below substitute fp32 coopmat as a proxy and cite the E2E fp16 numbers from [`L32_S128_coopmat_REPORT.md`](L32_S128_coopmat_REPORT.md) where needed.

## Hypothesis verdicts

| Ratio (smaller = numerator faster) | Hypothesis | Per-shape values | Mean | Verdict |
|---|---:|---|---:|---:|
| **R4a: int8 cm (wave64) / int8 q8csw** (coopmat lift on int8 W8A16) | ~0.25 (4×) | [0.22, 0.25, 0.46, 0.64] | **0.39** | ✓ **HOLDS at FFN (~4×); diminishes at small N (~1.5×)** |
| **R4b: int8 cm (wave64) / int8 q8ta** (coopmat lift on int8 W8A8) | ? | [0.61, 0.75, **1.63**, **1.63**] | **1.16** | ⚠ **Coopmat WINS at FFN, LOSES at Q/O and K/V** |
| R4c: int8 cm (orig, broken) / int8 cm (wave64) | bug ~0.5 | [0.68, 0.64, 0.63, 0.59] | 0.64 | bug-quant: broken shader is ~36–40% "faster" because it does half the work |
| R2b: fp32 cm / fp32 vec (coopmat lift on fp32) | ~0.32 (3.1×) | [0.45, 0.37, 0.39, 0.25] | 0.36 | ✓ HOLDS (2.78× mean) |
| R1b: int8 cm (wave64) / fp32 cm | ~0.25 (4×) | [0.29, 0.39, 0.71, **1.69**] | 0.77 | shape-dependent; int8 cm is SLOWER than fp32 cm at K/V |
| R3: int8 q8csw / fp32 vec (W8A16 lift, no coopmat) | ? | [0.61, 0.57, 0.60, 0.66] | 0.61 | int8 weights help ~1.6× even without coopmat hardware |

### What the two main hypotheses look like in detail

**(H2) "coopmat is ~4× faster than non-coopmat" — HOLDS on int8 *vs q8csw*, FAILS *vs q8ta*.**

The cleanest single comparison is R4a (`int8_cm_w64 / int8_q8csw`):

- At FFN gate/up (the dominant LLaMA shape) the wave64 KHR coopmat is **4.55× faster** than `linear_q8csw_tiled`.
- At FFN down it's **4.00×**, at Q/O **2.17×**, at K/V **1.56×**.
- Mean across LLaMA shapes: **2.56×**.

So the hypothesis holds *in shape* — coopmat IS the path to int8 LLaMA speed — but the magnitude depends on where in the model you look. The big wins are at FFN; K/V is mostly bandwidth-bound and the coopmat win is modest.

R4b (`int8_cm_w64 / int8_q8ta`) is the more honest comparison since both shaders are int8: it shows **wave64 KHR coopmat is faster at FFN shapes only**. At Q/O and K/V, the scalar `linear_q8ta_q8csw_tiled` shader (which uses `int8_dot_product`'s 4-way packed-int8 MAC) is **1.6× FASTER than coopmat**. The coopmat tile-launch overhead doesn't amortize at small N.

This is an important nuance the earlier version of this report missed. The "right" int8 shader for LLaMA prefill on this hardware isn't a single shader — it'd be a dispatch heuristic that picks coopmat for FFN-shape dispatches and q8ta for Q/K/V/O.

**(H1) "int8 coopmat is ~2× faster than fp16 coopmat" — INDIRECT, marginal at FFN, fails elsewhere.**

We don't have fp16 LLaMA microbench numbers (crash). Using fp32 coopmat as a proxy:

- At FFN gate/up: `int8_cm_w64 / fp32_cm = 0.29` → int8 is 3.4× faster than fp32 coopmat. fp16 coopmat is empirically ~2× faster than fp32 (per `matmul_coopmat_bench`'s fp16-vs-fp32 BERT runs). So int8 vs fp16 coopmat at FFN ≈ 0.29 × 2 = 0.58 → **int8 is ~1.7× faster than fp16 coopmat at FFN**. Close to the 2× hypothesis, slightly under.
- At FFN down: ~1.3× faster than fp16 coopmat (by the same proxy).
- At Q/O: ~0.7× — int8 is no longer winning vs fp16 coopmat.
- At K/V: ~0.3× — int8 coopmat is actually 3× SLOWER than fp16 coopmat (the ratio R1b = 1.69 + the fp16/fp32 factor flip the verdict negative).

The hypothesis is *directionally* supported at the FFN shapes but not as a universal "2× int8 over fp16 cm." The 2× factor is roughly the upper bound of what we see, not the typical.

### Mapping back to the E2E LLaMA forward

Using the wave64 per-dispatch numbers + LLaMA's dispatch counts:

| Component | # disp | Shape (M, K, N) | wave64 KHR cm ms/disp | wave64 cm total | fp16 baseline total | fp16 coopmat total |
|---|---:|---|---:|---:|---:|---:|
| FFN gate + up | 64 | 128, 4096, 14336 | 2.00 | **128.0 ms** | 789.4 ms | 255.9 ms |
| FFN down | 32 | 128, 14336, 4096 | 2.40 | **76.8 ms** | (part of 646.7 ms) | (part of 203.7 ms) |
| Q + O | 64 | 128, 4096, 4096 | 1.27 | **81.3 ms** | (part of 646.7 ms) | (part of 203.7 ms) |
| K + V | 64 | 128, 4096, 1024 | 0.53 | **33.7 ms** | 78.3 ms | 20.7 ms |
| **All non-lm-head linears** | 224 | — | — | **~320 ms** | 1515.9 ms | 412.7 ms |

If `matmul_khr_cm_int8_wave64` were wired into the LLaMA linear dispatch as a `linear_khr_cm_int8` shader (deferred — see Task 20), the linear contribution drops from 412.7 ms (fp16 coopmat) to **~320 ms**. Plus ~170 ms of unchanged non-linear ops → ~490 ms forward.

```
forward_int8_coopmat (wave64) ≈ 320 + 170 ≈ 490 ms
throughput_int8_coopmat        ≈ 128 / 0.490 ≈ 261 tok/s
```

- vs fp16 coopmat (219.7 tok/s) → **1.19× speedup**
- vs fp16 baseline (72.5 tok/s) → **3.61× speedup**
- vs int8 q8csw_tiled E2E (60.7 tok/s) → **4.30× speedup**

Compare to the earlier (broken-shader) projection of ~305 tok/s and 1.4× over fp16 coopmat — those were optimistic by ~16%.

A smarter heuristic that uses **coopmat at FFN + q8ta at Q/O/K/V** could do even better. With q8ta at Q/O/K/V:

| Component | # disp | best int8 ms/disp | total |
|---|---:|---:|---:|
| FFN gate + up (coopmat) | 64 | 2.00 | 128.0 ms |
| FFN down (coopmat) | 32 | 2.40 | 76.8 ms |
| Q + O (q8ta) | 64 | 0.78 | 49.9 ms |
| K + V (q8ta) | 64 | 0.32 | 20.5 ms |
| **All non-lm-head linears** | 224 | mixed | **~275 ms** |

→ forward ≈ 275 + 170 ≈ 445 ms → **288 tok/s** (1.31× over fp16 coopmat, 3.97× over fp16 baseline). The "right" path is a dispatch heuristic, not picking one int8 shader globally.

## Why int8 coopmat doesn't win uniformly

Phase 1 found that at FFN gate/up (M=128, N=14336), W8A16 `linear_qcs8w_tiled_*_half_o4x4` is **1.65× slower** than fp16 `linear_vec_buffer_texture2d_half`. This phase shows that for the same shape:

- `int8 q8csw` (W8A16, fp16 activations, fp32 accumulate, scalar dequant + MAD): **9.23 ms**
- `int8 q8ta` (W8A8, both sides int8, int8-dot-product extension, scalar pipeline): **3.27 ms**
- `int8 KHR cm wave64` (W8A8, true cooperative-matrix WMMA, int32 accumulate): **2.00 ms**

The big jumps are q8csw → q8ta (2.8×, from int8-activation + int8-dot-product hardware) and q8ta → coopmat (1.6×, from WMMA over the dot-product extension). Both wins matter.

But at Q/O (M=128, N=4096, 4× less output than FFN), the picture inverts: q8ta runs at 0.78 ms while coopmat takes 1.27 ms. The dot-product extension's smaller granularity beats the coopmat tile launch overhead when there's less total work per dispatch. Same story at K/V.

**Mechanism (best guess):** coopmat's 128×128×32 tile commits 4 subgroups × 4×4 cooperative matrices each (in the wave64 variant), with shared-memory prefetch and barrier sync. At Q/O the entire output (128×4096) is 32 tiles × 8 subgroup-tiles, so the tile-launch overhead per FLOP is high. q8ta's per-element scalar pipeline doesn't have this fixed overhead — it scales linearly with work.

## What's next

The microbench validates the wave32/wave64 fix and gives us realistic int8 shader-level numbers for two follow-up tracks:

1. **Port `matmul_khr_cm_int8_wave64` to main tree as a `linear_khr_cm_int8` dispatch site.** Gated on M ≥ 128 like fp16's `add_linear_coopmat_node`. The lm_head's M=1 falls back to `linear_qcs8w_tiled` (the W8A16 path already in main).
2. **Compose with `q8ta_linear` for Q/K/V/O.** The partitioner pattern already exists in main tree. If we wire both paths, the dispatch heuristic should be: use KHR coopmat for `N ≥ 8192` (FFN) and q8ta for smaller N.
3. **Add an export-side W8A8 quantization recipe.** Today `--pt2e_quantize vulkan_8w` is weight-only (W8A16). Both KHR coopmat int8 and q8ta need int8 activations. The likely route is `q8ta_linear`'s existing partitioner pattern in main tree composed with `is_dynamic=True` activation quant.

Targets:
- Pure-coopmat E2E (no heuristic): ~261 tok/s (1.19× over fp16 coopmat)
- Mixed coopmat+q8ta E2E: ~288 tok/s (1.31× over fp16 coopmat) — recommended

If the realized E2E falls materially below 261 tok/s, the gap is likely export-side overhead (input quantization staging, layout transitions) rather than shader-level.

## Artifacts

Under `yanwen/artifacts/int8_microbench/`:

| File | Description |
|---|---|
| `khr_cm_gemm_int8_*.log` | Both orig (impl=3) and wave64 (impl=4) int8 KHR cm bench output, all LLaMA shapes |
| `linear_coopmat_bench_*.log` | fp32 (and partial fp16) vec/cm bench output |
| `q8csw_linear_*.log` | W8A16 + W8A8 non-coopmat bench output |

The latest run is from 2026-05-10 18:25 (`*_20260510_182522.log`) — it has the wave64 numbers. Earlier `*_20260510_164216.log` artifacts are pre-wave64 and only have the broken-shader rows.

Parsing logic is in `yanwen/scripts/int8/microbench_summarize.py`. After the wave32/wave64 discovery the classifier was updated to differentiate `int8_cm_orig` (impl=3) and `int8_cm_wave64` (impl=4) into separate buckets.

## Source modifications

Under `pavan-report/executorch/`:

| File | Change | Commit |
|---|---|---|
| `backends/vulkan/test/custom_ops/khr_cm_gemm_int8.cpp` | LLaMA shape configs, dual-variant loop (impl 3 + 4) | `4db034b7e`, `c331b9cf5` |
| `backends/vulkan/test/custom_ops/linear_coopmat_bench.cpp` | LLaMA shape configs, fp16 vec+cm variants (crashes on cm_fp16_BERT_QKV) | `4db034b7e` |
| `backends/vulkan/test/custom_ops/q8csw_linear.cpp` | LLaMA shape configs covering q8ta + q8csw | `4db034b7e` |
| `backends/vulkan/runtime/graph/ops/glsl/matmul_khr_cm_int8_wave64.glsl` | NEW: wave64-correct variant (2×2 subgroup layout, 64-thread/subgroup) | `c331b9cf5` |
| `backends/vulkan/runtime/graph/ops/glsl/matmul_khr_cm_int8_wave64.yaml` | NEW: shader registration | `c331b9cf5` |
| `backends/vulkan/runtime/graph/ops/impl/MatMulKHRCoopMat.cpp` | NEW dispatcher `khr_cm_gemm_int8_wave64` (etvk.khr_cm_gemm_int8_wave64.default) | `c331b9cf5` |
| `backends/vulkan/test/custom_ops/impl/TestGemm.cpp` | impl_selector=4 routes to wave64 variant | `c331b9cf5` |
| `backends/vulkan/test/custom_ops/khr_cm_gemm_int8_validate.cpp` | NEW: tight-tolerance correctness validation driver | `c331b9cf5` |
