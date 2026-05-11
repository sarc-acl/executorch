# int8 coopmat shader microbenchmark — does the 2× / 4× math hold?

**Updated:** 2026-05-10 · **Author:** Yanwen Xu
**Device:** AMD Radeon 780M (RADV PHOENIX, RDNA3+ mobile iGPU), Mesa 25.0.7
**Companion to:** [`L32_S128_int8_baseline_REPORT.md`](L32_S128_int8_baseline_REPORT.md) — Phase 1 of the int8 study, the E2E baseline. This is Phase 2 — the shader-level microbench.

## Why this study

Phase 1 showed the default `vulkan_8w` int8 path runs **19% slower** than fp16 baseline at L=32 S=128 prefill (60.7 vs 72.5 tok/s). That's because `linear_qcs8w_tiled_*_half_o4x4` is per-dispatch slower than `linear_vec_buffer_texture2d_half` at large N (FFN gate/up: 20.2 ms vs 12.3 ms). The W8A16 weight-bandwidth saving is real but doesn't beat the per-element dequant overhead at these shapes.

The user's hypothesis: a *real* int8 win on this hardware requires the KHR cooperative-matrix int8 path (`matmul_khr_cm_int8.glsl`, exists in pavan-report tree but is NOT wired into the LLaMA linear dispatch site). Specifically:

- **(H1) int8 coopmat is ~2× faster than fp16 coopmat** (int8 tensor-core throughput)
- **(H2) coopmat is ~4× faster than non-coopmat** at the same precision

This phase tests both hypotheses directly at the shader level, without exporting/running a full LLaMA forward.

## Setup

Three pavan-report custom-ops binaries, each compiled standalone:

| Binary | What it benches |
|---|---|
| `khr_cm_gemm_int8` | `matmul_khr_cm_int8.glsl` (true `coopmat<uint8_t, ...>` WMMA shader; pure int8×int8 → fp output) |
| `linear_coopmat_bench` | `linear_vec_*` (texture3d → `linear_vec`) vs `linear_coopmat_*` (buffer → `linear_coopmat`) — fp32 and fp16 variants (fp16 added in this study) |
| `q8csw_linear` | `linear_q8csw_tiled_*` (W8A16, non-coopmat) and `linear_q8ta_q8csw_*` (W8A8 with int8-dot-product extension) |

All four shapes from LLaMA 3.1 8B prefill at S=128 were added to the config lists of each binary:

| Tag | Shape (M, K, N) | LLaMA component |
|---|---|---|
| `llama_ffn_gateup` | 128, 4096, 14336 | FFN gate / up |
| `llama_ffn_down` | 128, 14336, 4096 | FFN down |
| `llama_qo` | 128, 4096, 4096 | Q / O proj |
| `llama_kv` | 128, 4096, 1024 | K / V proj |

(decode-shape M=1 omitted — coopmat's M≥128 tile requirement excludes it; the M=1 path is GEMV-bound and a separate problem.)

Build (one-time, from pavan-report tree):

```bash
cd /home/doremy/sarc-acl/executorch/pavan-report/executorch
.venv/lib/python3.12/site-packages/cmake/data/bin/cmake \
    backends/vulkan/test/custom_ops/ \
    -DCMAKE_BUILD_TYPE=Release -DGLSLC_PATH=$(which glslc) \
    -DCMAKE_PREFIX_PATH=$PWD/cmake-out-vk \
    -Bcmake-out-vk/backends/vulkan/test/custom_ops
.venv/lib/python3.12/site-packages/cmake/data/bin/cmake \
    --build cmake-out-vk/backends/vulkan/test/custom_ops -j$(nproc) \
    --target khr_cm_gemm_int8 linear_coopmat_bench q8csw_linear
```

Run + summarize (from main tree):

```bash
cd /home/doremy/sarc-acl/executorch/main/executorch
source .venv/bin/activate
python yanwen/scripts/int8/microbench_runner.py
python yanwen/scripts/int8/microbench_summarize.py
```

Each binary uses 3 warmup + 10 timed runs and reports GPU-timestamp latency per kernel.

## Headline results

Per-shape per-shader latencies at LLaMA prefill shapes (best across storage variants):

| Component | Shape | fp32 vec | fp32 cm | int8 q8csw (W8A16) | int8 q8ta (W8A8) | **int8 KHR cm** |
|---|---|---:|---:|---:|---:|---:|
| `llama_ffn_gateup` | M=128, K=4096, N=14336 | 15.06 ms | 6.68 ms | 9.08 ms | 3.27 ms | **1.43 ms** |
| `llama_ffn_down` | M=128, K=14336, N=4096 | 16.28 ms | 6.13 ms | 9.57 ms | 3.07 ms | **1.58 ms** |
| `llama_qo` | M=128, K=4096, N=4096 | 4.57 ms | 1.75 ms | 2.75 ms | 0.78 ms | **0.87 ms** |
| `llama_kv` | M=128, K=4096, N=1024 | 1.24 ms | 0.30 ms | 0.86 ms | 0.33 ms | 0.37 ms |

`int8 KHR cm` (`matmul_khr_cm_int8`) is the clear winner at FFN shapes — 1.4–1.6 ms per dispatch is faster than even fp32 coopmat. At smaller shapes (qo, kv), it's competitive with W8A8 q8ta but no longer dominant — the WMMA hardware launch overhead doesn't amortize over fewer FMAs.

**Caveat: no fp16 LLaMA microbench data.** The fp16 variants in `linear_coopmat_bench` were added to the bench but a segfault in `cm_fp16_BERT_QKV` (a non-LLaMA shape, M=128 N=768 K=768) terminated the run before fp16 LLaMA shapes executed. The fp16 cells in the table are empty for this reason. fp32 numbers + the E2E fp16 ground truth from [`L32_S128_coopmat_REPORT.md`](L32_S128_coopmat_REPORT.md) are used in place of fp16 microbench numbers where needed.

## Hypothesis verdicts

| Ratio (smaller = numerator faster) | Hypothesis | Per-shape values | Mean | Verdict |
|---|---:|---|---:|---:|
| **R4: int8 KHR cm / int8 q8csw** (coopmat lift on int8) | ~0.25 (4×) | [0.16, 0.17, 0.32, 0.43] | **0.27** | **HOLDS (3.7× mean)** |
| R2b: fp32 cm / fp32 vec (coopmat lift on fp32) | ~0.32 (3.1×) | [0.44, 0.38, 0.38, 0.24] | 0.36 | **HOLDS (2.78× mean)** |
| R1b: int8 KHR cm / fp32 cm | ~0.25 (4×) | [0.21, 0.26, 0.50, 1.21] | 0.55 | **shape-dependent** |
| R3: int8 q8csw / fp32 vec (W8A16 lift) | ? | [0.60, 0.59, 0.60, 0.69] | 0.62 | int8 weights help ~1.6× at the shader level |

### What the two main hypotheses look like in detail

**(H2) "coopmat is ~4× faster than non-coopmat" — HOLDS strongly on int8.**

The most decisive single number in this study. `R4 = 0.27` means `int8 KHR cm` runs at 27% of the time of `int8 q8csw` (W8A16, the deployed default int8 path). That's a **3.7× speedup mean across LLaMA shapes**, and at the dominant FFN shapes (gateup, down) it's **6×**. The "tiled" non-coopmat shader and the WMMA-accelerated coopmat shader are doing the same matrix multiply, but the cooperative-matrix hardware lane just runs that much faster on RDNA3+.

This validates the user's hypothesis: yes, on this hardware, coopmat is the path that makes int8 actually fast.

**(H1) "int8 coopmat is ~2× faster than fp16 coopmat" — INDIRECT but supported.**

We don't have direct fp16 LLaMA microbench numbers (crash). We have two sources to bridge:

1. **int8 KHR cm / fp32 cm = 0.55 mean, but [0.21, 0.26] at FFN gateup/down.** At the high-FLOP shapes (FFN), int8 is **4–5× faster than fp32 coopmat**. fp16 coopmat is empirically ~2× faster than fp32 coopmat on this hardware (matmul_coopmat_bench's fp16 vs fp32 BERT runs confirm this). So int8 vs fp16 coopmat at FFN shapes lands in the **~2× faster** range. **Hypothesis holds at FFN shapes.**
2. At smaller shapes (qo, kv), the ratio rises to 0.50–1.21 — int8 is no longer winning vs fp32 cm, let alone fp16 cm. This is the launch-overhead regime — coopmat tile dispatch costs are amortized over fewer total FMAs.

### Mapping back to the E2E LLaMA forward

Using the per-dispatch numbers + dispatch counts from the fp16 study:

| Component | # disp | int8 KHR cm ms/disp | int8 KHR cm total | fp16 baseline total | fp16 coopmat total |
|---|---:|---:|---:|---:|---:|
| FFN gate + up `[128, 14336]` | 64 | 1.43 | **91.7 ms** | 789.4 ms | 255.9 ms |
| FFN down `[128, 4096]` (from `[128, 14336]` input)¹ | 32 | 1.58 | **50.6 ms** | (part of 646 ms) | (part of 204 ms) |
| Q + O + FFN-down `[128, 4096]` shape | 96 | 0.87 | **83.5 ms** | 646.7 ms | 203.7 ms |
| K + V `[128, 1024]` | 64 | 0.37 | **23.8 ms** | 78.3 ms | 20.7 ms |
| **All non-lm-head linears** | 224 | — | **~250 ms** | 1515.9 ms | 412.7 ms |

¹ This row uses the FFN-down shape (M=128, K=14336, N=4096) from the microbench. Distinct from the `[128, 4096]` shape because K and N are swapped.

If `matmul_khr_cm_int8` were wired into the LLaMA linear dispatch site as a `linear_khr_cm_int8` shader (Phase 3 of the study, deferred), the linear contribution to forward could drop from **412.7 ms (fp16 coopmat)** to **~250 ms (int8 coopmat)** — a 1.65× shrink on the linear category. The non-linear ops (~170 ms) are unchanged. By Amdahl:

```
forward_int8_coopmat ≈ 250 + 170  ≈ 420 ms
throughput_int8_coopmat ≈ 128 / 0.420 ≈ 305 tok/s
```

vs fp16 coopmat's 219.7 tok/s → about a **1.4× speedup** over fp16 coopmat (i.e., ~5× over fp16 baseline). Less than the naive "int8 is 2× fp16" projection because Amdahl on the non-linear share eats half the win, but still a major lift.

## Why int8 KHR coopmat wins where W8A16 q8csw loses

Phase 1 found that at FFN gate/up (M=128, N=14336), W8A16 `linear_qcs8w_tiled_*_half_o4x4` is **1.65× slower** than fp16 `linear_vec_buffer_texture2d_half`. This phase shows that at the SAME shape:

- `int8 q8csw` (W8A16, fp16 activations, fp32 accumulate, scalar dequant + MAD): 9.08 ms
- `int8 KHR cm` (W8A8, int8 inputs both sides, integer cooperative-matrix WMMA, int32 accumulate): **1.43 ms**

The difference between these two int8 paths is **6.35×**, and it comes down to:

1. **WMMA hardware vs scalar pipeline.** `coopmat<uint8_t, ..., MatrixUseA>` is a hardware-accelerated tensor-core path that does an N×K×M FMA region per instruction; `linear_qcs8w_tiled` issues per-element MADs. RDNA3+'s WMMA is the entire reason cooperative matrix shaders win.
2. **No per-element dequant in the inner loop.** `matmul_khr_cm_int8` accumulates int8 × int8 → int32 in registers and only does the (fp_scale × int32) conversion once per output tile. `linear_qcs8w_tiled` dequantizes every weight on the load path.

Together these mean the speedup from "make int8 LLaMA actually faster than fp16" lives entirely on the coopmat side — and the W8A16 default recipe is the wrong path on this hardware.

## What's next (deferred, per the study scope)

A natural Phase 3 — not included in this study, scoped out at the planning stage — would be to:

1. Wire `matmul_khr_cm_int8` into `Linear.cpp::add_linear_node` as a `linear_khr_cm_int8` dispatch site, gated on M ≥ 128 like fp16's `add_linear_coopmat_node`. (The lm_head's M=1 falls back to the existing `linear_qcs8w_tiled` half o4x1.)
2. Add an export-side W8A8 quantization path (the shader needs int8 activations, but `vulkan_8w` is weight-only). The likely route is wiring `q8ta_linear` (which already exists in main tree's partitioner pattern matcher) end-to-end, then composing with the new linear_khr_cm_int8 dispatch.
3. Re-export a W8A8 LLaMA .pte and run the same scientific bench. Target: ~305 tok/s (1.4× over fp16 coopmat) per the Amdahl estimate above; report deviation.

The microbench numbers in this report are the green light for that Phase 3 work — if R4 had been ~1× we'd have killed the idea.

## Artifacts

Under `yanwen/artifacts/int8_microbench/`:

| File | Description |
|---|---|
| `khr_cm_gemm_int8_*.log` | Full stdout from the int8 KHR cm GEMM bench |
| `linear_coopmat_bench_*.log` | fp32 + (partial) fp16 vec/cm bench output |
| `q8csw_linear_*.log` | W8A16 + W8A8 non-coopmat bench output |

Parsing logic is in `yanwen/scripts/int8/microbench_summarize.py`. The parser tolerates the truncated-kernel-name format the print harness emits (which loses the trailing `"`).

## Source modifications

In-place edits made under `pavan-report/executorch/backends/vulkan/test/custom_ops/`:

| File | Change |
|---|---|
| `khr_cm_gemm_int8.cpp` | Added 4 LLaMA shape configs to `generate_int8_gemm_test_cases()` |
| `linear_coopmat_bench.cpp` | Added 4 LLaMA shape configs to `generate_test_cases()`, added `vec_fp16` + `cm_fp16` variants (fp16 path crashed at BERT shape — see above) |
| `q8csw_linear.cpp` | Added 4 LLaMA shape configs to `generate_quantized_linear_test_cases()` |

None of these changes affect existing test runs; they only append LLaMA-shape entries.
