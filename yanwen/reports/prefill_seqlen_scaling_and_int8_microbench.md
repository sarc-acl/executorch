# Prefill seq_len scaling + int8 microbench results — synthesis

**Updated:** 2026-05-11 · **Author:** Yanwen Xu
**Device:** AMD Radeon 780M (RADV PHOENIX, RDNA3+ mobile iGPU, wave64), Mesa 25.0.7
**Host:** 28.9 GiB DDR5 RAM, 24 GiB swap, RADV_GTT_PCT=80 (22.4 GiB Vulkan-accessible cap)
**Model:** LLaMA 3.1 8B, L=32, fp16 weights unless noted

This report combines two threads of the study into one place:

1. **Why prefill memory blows up with seq_len** — the shape-level diagram, the activation math, and the empirical S=128/512/1024/2048 cliff behavior measured in earlier reports.
2. **The full int8 microbench picture** — all per-shape latencies across fp16 / W8A16 / W8A8 / KHR coopmat int8, the coopmat-vs-non-coopmat splits, the tile-schedule sweep outcome (Phase 1 of Track A), and what it implies for E2E.

---

# Part 1 — S=128 vs S=2048: same model, different fate

## The whole-forward diagram (unchanged across seq_len)

Per [`yanwen/reports/REPORT.md`](REPORT.md) and the L=32 fp16 baseline ETDump, every forward at L=32 dispatches:

- **224 linear matmuls** (32 layers × 7 linears per layer)
- **64 attention BMMs** (32 layers × Q@Kᵀ and attn@V)
- **2 lm_head matmuls** (one per output position pair, depending on lowering)
- Plus reshape / view, RMSNorm, softmax, residual adds, elementwise ops, CPU↔GPU staging

**The diagram structure is identical at S=128 and S=2048.** What changes is the M dimension of every prefill matmul and the size of activation tensors.

## The 4 unique linear matmul shapes, both seq_lens

| Component | Shape (M, K, N) at S=128 | Shape (M, K, N) at S=2048 | What grew | Dispatches/forward |
|---|---|---|---|---:|
| **FFN gate / up** | `(128, 4096, 14336)` | `(2048, 4096, 14336)` | **M × 16** | 64 |
| **FFN down** | `(128, 14336, 4096)` | `(2048, 14336, 4096)` | **M × 16** | 32 |
| **Q / O proj** | `(128, 4096, 4096)` | `(2048, 4096, 4096)` | **M × 16** | 64 |
| **K / V proj** | `(128, 4096, 1024)` | `(2048, 4096, 1024)` | **M × 16** | 64 |
| **lm_head** | `(1, 4096, 128256)` | `(1, 4096, 128256)` | **unchanged** | 2 |

Per-linear FLOPs scale linearly with S → 16× more arithmetic per linear at S=2048. If everything else stayed flat you'd expect ~16× slower forward. It doesn't — because two other things grow worse than linear.

## The attention BMMs — the S² killer

These two matmuls inside each attention block use runtime tensors on both sides (not weights), and their shape comes from the sequence dimension on both sides:

| BMM | Shape at S=128 | Shape at S=2048 | What grew |
|---|---|---|---|
| **Q @ Kᵀ** | `[1, 32 heads, 128, 128]` per layer | `[1, 32, 2048, 2048]` per layer | **S² = 256×** |
| **attn @ V** | `[1, 32, 128, 128] × [1, 32, 128, 128]` | `[1, 32, 2048, 2048] × [1, 32, 2048, 128]` | **S² = 256×** |

At S=128, attention BMMs were **14.5 ms total** in the fp16 baseline — under 1% of forward. At S=2048 the same dispatches would be ~256× larger → on the order of **4 seconds of pure attention BMM compute**, becoming a meaningful chunk of any otherwise-runnable forward.

The fp16 coopmat run measured **0.93 ms total** for the 64 attention BMMs at S=128 (15.6× speedup over `matmul_vec`). The S² scaling means that 15.6× would *especially matter* at higher seq.

## Activations / intermediate tensors per layer

This is the OOM story.

| Tensor | At S=128 | At S=2048 | Growth |
|---|---|---|---|
| Hidden state `[1, S, 4096]` fp16 | 1 MB | 16 MB | 16× |
| FFN intermediate `[1, S, 14336]` fp16 | 3.5 MB | 56 MB | 16× |
| **Attention scores `[1, 32, S, S]` fp16** | **1 MB** | **268 MB** | **256×** |
| Q / K / V / O outputs combined | ~24 MB | ~384 MB | 16× |
| Per-layer sum (live working set) | **~29 MB** | **~600 MB** | ~20× |
| × 32 layers | **~0.9 GB** | **~19 GB** | — |

The Vulkan delegate keeps intermediates alive during the forward (no PyTorch-eager-style free-as-you-go), so peak working set is approximately `weights + sum-of-all-live-activations`. For LLaMA 3.1 8B fp16:

```
working_set(S) ≈ 15 GB (weights, mmap'd) + activations(S)

S=128:   15 + 0.9  = ~16 GB        ← fits the 22.4 GB GTT cap comfortably
S=512:   15 + 10   = ~25 GB        ← exceeds RAM-friendly fit; thrashes
S=1024:  15 + 13   = ~28 GB        ← exceeds 28.9 GB RAM; OOM-killed
S=2048:  15 + 19   = ~34 GB        ← far past any threshold
```

## The empirical cliff map (measured)

From [`reports/REPORT.md`](REPORT.md) on the same L=32 fp16 baseline build:

| seq | Wallclock / forward | ETDump GPU time | "Memory stall" gap | Status |
|---:|---:|---:|---:|---|
| **128** | **1.77 s** | 1.91 s | ~0 (timer noise) | ✓ **performant**, 72.5 tok/s |
| **512** | **~111 s** | **13.7 s** | **~97 s (88% of wallclock)** | ◐ completes but **~60× cliff** vs. compute scaling |
| 1024 | (OOM-killed) | — | — | ✗ OOM during calibration |
| 2048 | (OOM-killed) | — | — | ✗ OOM consistently |

**Three distinct regimes, mechanically different:**

1. **Clean regime (S=128)** — working set fits in GTT with headroom. Per-shader compute scales linearly with S (12.3 ms at S=128 → 46.6 ms at S=512 for FFN gate/up = 3.8× for 4× seq, normal).
2. **Cliff regime (S=512)** — working set exceeds usable RAM but **mmap'd weight pages can be evicted from page cache and re-faulted from disk on demand**. GPU compute is fine (13.7 s) but wallclock is 111 s. The 97-second gap is `vkQueueSubmit` blocked on host page faults. Telltale: `ETVK_COPY_INPUTS` for a 4 KB tensor balloons from 16 ms (S=128) to **3420 ms** (S=512). The shader isn't slow — the staging buffer pages are getting evicted between dispatches.
3. **Hard OOM regime (S≥1024)** — even with weight-page eviction, the activation working set alone exceeds available physical memory. The process gets killed by the OOM killer.

## Why this is hardware-bound, not shader-bound

A 2× faster linear shader would still allocate the same activation intermediates and still page when the working set exceeds RAM. The cliff is independent of any shader optimization. **The memory-wait component is not visible to ETDump** because GPU timestamps only span GPU-active time — not host-side paging.

To run L=32 at seq ≥ 512 *usably* on this hardware, the lever is **reduce working-set size**, not speed up shaders:

| Lever | Effect | Status |
|---|---|---|
| **Weight-only int4 quantization** | 15 GB → ~4 GB → frees ~11 GB | Unexplored; biggest lever |
| **int8 W8A16** | 15 GB → ~8.6 GB → frees ~6 GB | Measured; helps at S=512 (see int8 baseline report) |
| **Flash attention** | Removes `[S, S]` materialization (the 256× term) | Not available in current Vulkan delegate |
| **Per-layer activation streaming** | Activations don't have to all be live at once | Would need partitioner/runtime rework |

The int8 W8A16 result already confirmed the working-set lever: at S=512, fp16 cliffed to ~111 s (88% memory-wait) while **int8 W8A16 completed in 25.65 s (39% memory-wait, 4.33× faster)** — the int8 weights brought the working set back inside RAM, the page cache stopped evicting, and the GPU could actually run compute. See [`L32_S128_int8_baseline_REPORT.md`](L32_S128_int8_baseline_REPORT.md) addendum for the full S=512 comparison.

## Performance results summary (prefill, L=32, fp16 weights unless noted)

| Config | S=128 | S=512 | S=1024 |
|---|---:|---:|---:|
| fp16 baseline (`linear_vec`) | **1.766 ± 0.006 s** (72.5 tok/s) | ~111 s (4.6 tok/s, cliffed) | OOM |
| fp16 coopmat (`linear_coopmat`) | **0.583 ± 0.002 s** (220 tok/s, 3.03×) | — | — |
| int8 W8A16 (`linear_qcs8w_tiled`) | **2.108 ± 0.029 s** (60.7 tok/s, 0.84×) | **25.65 s** (20.0 tok/s, 4.33× over fp16) | TBD |
| int8 W8A16 at S=256 | — | **4.597 ± 0.088 s** (55.7 tok/s) | — |

**Decode (`use_kv_cache=True`, max_seq_len=1024):** real autoregressive at L=32 fp16 measures **5.0 s / step → 0.20 tok/s**, with 78% of wallclock being host memory-wait (not GPU compute). Manager-spec'd answer; see [`L32_real_decode_benchmark.md`](L32_real_decode_benchmark.md).

## Bottom line for Part 1

Shape-wise nothing exotic happens at S=2048; the model is the exact same diagram. **What kills you is memory math:** activations grow with S² in the attention scores tensor (the dominant term beyond S≈256), weights stay constant at 15 GB, and their combined footprint blows past the iGPU's 22.4 GB GTT ceiling. The cliff regime at S=512 is the in-between where mmap'd weight pages can still get re-faulted from disk; beyond S=1024 even that fails.

The microbench can still characterize S=2048 shapes — it's one matmul at a time, no full forward, no activation pressure. Add `(2048, 4096, 14336)` to the config list and it runs. But end-to-end measurement at S=2048 on the 780M requires either shrinking weights (int4) or moving to a box with more RAM.

---

# Part 2 — int8 microbench: comprehensive results

This section consolidates measurements from [`int8_coopmat_microbench.md`](int8_coopmat_microbench.md) (v1) and [`int8_coopmat_microbench_v2.md`](int8_coopmat_microbench_v2.md) (the tile-schedule sweep + real fp16 cm baseline). All numbers are GPU-timestamp, mean of 10 runs, per-dispatch.

## Setup

Pure matmul (no full model), LLaMA 3.1 8B prefill shapes at M=128:

| Tag | Shape (M, K, N) | LLaMA component | Dispatches/forward |
|---|---|---|---:|
| `llama_ffn_gateup` | 128, 4096, 14336 | FFN gate + up | 64 |
| `llama_ffn_down` | 128, 14336, 4096 | FFN down | 32 |
| `llama_qo` | 128, 4096, 4096 | Q + O proj | 64 |
| `llama_kv` | 128, 4096, 1024 | K + V proj | 64 |

Microbench binaries (`khr_cm_gemm_int8`, `linear_coopmat_bench`, `q8csw_linear`) live in `pavan-report/backends/vulkan/test/custom_ops/`. Latest logs under `yanwen/artifacts/int8_microbench/`.

## The full latency matrix (per-dispatch GPU time, ms)

| Shape | fp32 vec | fp32 cm | **fp16 vec** | **fp16 cm** | int8 q8csw (W8A16) | int8 q8ta (W8A8 scalar) | int8 cm wave64 (W8A8 WMMA) | int8 cm orig (BROKEN) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FFN gate/up | 15.18 | 6.78 | 15.33 | **3.42** | 9.23 | 3.27 | **2.00** | 1.35 |
| FFN down | 16.76 | 6.16 | 16.21 | **3.52** | 9.62 | 3.19 | **2.40** | 1.54 |
| Q/O | 4.59 | 1.78 | 4.60 | **0.92** | 2.74 | **0.78** | 1.27 | 0.80 |
| K/V | 1.24 | 0.31 | 1.24 | **0.26** | 0.82 | **0.32** | 0.53 | 0.31 |

Bold = the fastest path for that shape. The "BROKEN" column is the original wave32-assumed int8 KHR coopmat shader on RDNA3+ wave64 hardware — it timed ~36% faster than the wave64-correct version because it only computed half the output tile (silent zero in the second vertical half). Caught via tight-tolerance validation (`khr_cm_gemm_int8_validate`); replaced by `matmul_khr_cm_int8_wave64.glsl` in commit `c331b9cf5`.

## Headline relationships

### Coopmat over non-coopmat, same precision (R2 from v1 report)

| Comparison | FFN gate/up | FFN down | Q/O | K/V | Mean |
|---|---:|---:|---:|---:|---:|
| **fp16 cm / fp16 vec** | 0.22 (4.5×) | 0.22 (4.6×) | 0.20 (5×) | 0.21 (4.8×) | **4.7×** |
| **int8 cm wave64 / int8 q8csw** | 0.22 (4.6×) | 0.25 (4.0×) | 0.46 (2.2×) | 0.64 (1.6×) | **3.1×** geomean |

Coopmat decisively beats its same-precision scalar/dequant counterpart at every LLaMA shape. The fp16 cm advantage (4.7× across the board) is exactly what produces the 3.03× E2E speedup measured in the fp16 coopmat report — Amdahl + non-linear ops dilute it.

### int8 over fp16 (both coopmat) — the user's "2× hypothesis"

| Shape | int8 cm wave64 / fp16 cm | Speedup |
|---|---:|---:|
| FFN gate/up | 0.58 | **1.72×** |
| FFN down | 0.68 | **1.47×** |
| Q/O | 1.38 | 0.73× (int8 SLOWER) |
| K/V | 2.06 | 0.49× (int8 ~2× slower) |

The hoped-for "2× int8 over fp16 cm" lands at **1.72× at the best (FFN gate/up)** shape and degrades. At Q/O and K/V, int8 coopmat is actively worse than fp16 coopmat — the tile is too big for the small-N work.

## int8 coopmat vs int8 non-coopmat — the cleanest cut

There are two non-coopmat int8 paths to compare against.

### vs `q8csw_tiled` (W8A16, the int8 path currently in main)

| Shape | int8 cm wave64 / int8 q8csw | Coopmat speedup |
|---|---:|---:|
| FFN gate/up | 0.217 | **4.6×** |
| FFN down | 0.249 | **4.0×** |
| Q/O | 0.464 | **2.2×** |
| K/V | 0.646 | **1.5×** |
| **Mean** | **0.394** | **2.5×** (geomean) |

Coopmat wins at every shape vs the W8A16 scalar-dequant baseline. FFN-heavy weighted speedup ≈ 4.3×. **This is the "yes, KHR coopmat int8 is the right path" comparison** — at FFN it's 4× faster than what's in main today.

### vs `q8ta` (W8A8 scalar with `int8_dot_product` extension)

| Shape | int8 cm wave64 / int8 q8ta | Coopmat speedup |
|---|---:|---:|
| FFN gate/up | 0.612 | **1.6×** |
| FFN down | 0.752 | **1.3×** |
| Q/O | **1.628** | **0.6× (q8ta WINS 1.6×)** |
| K/V | **1.656** | **0.6× (q8ta WINS 1.7×)** |
| **Mean** | **1.04** | ~wash overall |

Mixed result. Coopmat wins at FFN shapes (N ≥ 4096). q8ta wins at attention-projection shapes (N ≤ 4096). The crossover sits at **N ≈ 8192**.

## Why coopmat loses at small N

Mechanical, not tunable. The wave64 KHR coopmat shader uses a fixed 128×128 output tile with shared-mem prefetch + barrier sync + cooperative matrix load/store overhead. Per-workgroup fixed cost is independent of N.

| Shape | N | Workgroups launched | 780M utilization | Verdict |
|---|---:|---:|---|---|
| FFN gate/up | 14336 | 112 | well saturates 12-WGP GPU | **coopmat wins 1.6×** |
| FFN down | 4096 | 32 | OK | **coopmat wins 1.3×** |
| Q/O | 4096 | 32 | OK | borderline → **coopmat LOSES 0.6×** |
| K/V | **1024** | **8** | **half the GPU is idle** | **coopmat LOSES 0.6×** |

Two compounding effects at small N:

1. **Under-utilization** — K/V launches 8 workgroups; only 8 of 12 WGPs have work; rest are idle.
2. **Fixed-cost dominance** — coopmat's per-WG tile-setup overhead doesn't shrink. q8ta's scalar pipeline using the 4-way packed-int8 MAC instruction has near-zero per-dispatch overhead.

Together: tensor cores never get to flex at small N.

## Today's tile-schedule sweep (Phase 1 of Track A)

Exposed `WORKGROUP_{WIDTH,HEIGHT}_IN_SUBGROUPS` as spec constants (IDs 14/15) on `matmul_khr_cm_int8_wave64.glsl`. Built a sweep harness that emits one test case per (shape × variant) via a new `etvk.khr_cm_gemm_int8_wave64_tiled.default` op. Validated every variant with tight tolerance (abs ≤ 0.5 on 128×128×128).

Original plan: 7 variants. 3 dropped after validation:

| Variant | Reason for drop |
|---|---|
| `v2_wideN` (TILE_N=64) | Prefetch loop overruns shared memory when `B_NUM_ROWS < INVOCATIONS_PER_WORKGROUP / (B_ROW_LEN/ELEMENTS_PER_VEC4)`. Structural shader bug, not a tuning artifact. |
| `v5_colmajB` (BColMajor=1) | Test harness feeds row-major B; column-major coopMatLoad reads with mismatching stride pattern from shared memory. |
| `v6_deepK_colmajB` | Same as v5 (colmajor combined with deeper K tile). |

4 surviving variants benched:

| Variant | TILE_M | TILE_N | TILE_K | sg_w | sg_h | FFN gate/up (ms) | FFN down (ms) | Q/O (ms) | K/V (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`v0_baseline`** | 128 | 128 | 32 | 2 | 2 | **2.006** | **2.340** | **1.241** | 0.462 |
| `v1_deepK` | 128 | 128 | **64** | 2 | 2 | 2.391 (+19%) | 2.521 (+8%) | 1.343 | 0.451 |
| `v3_sg1x4` | 128 | 128 | 32 | **1** | **4** | 2.139 (+7%) | 2.447 (+5%) | 1.274 | **0.445** |
| `v4_sg4x1` | 128 | 128 | 32 | **4** | **1** | 2.136 (+6%) | 2.401 (+3%) | 1.361 | 0.548 |

**No variant beats `v0_baseline` at the FFN shapes.** Deeper K-tile (`v1_deepK`) burned register / shared-mem pressure faster than it saved on prefetch overhead. Subgroup-layout reshuffles (`v3_sg1x4`, `v4_sg4x1`) didn't meaningfully change anything. `v3_sg1x4` marginally helped K/V (4%) — too small to matter.

### Decision gate

The Track A plan defined a weighted-by-dispatch (FFN-heavy) gate to determine whether Phase 2 (E2E port + W8A8 export) should start:

```
weighted_int8_cm = 64 × 2.006 + 32 × 2.340 = 203.3 ms   (best per shape, all v0_baseline)
weighted_fp16_cm = 64 × 3.421 + 32 × 3.517 = 331.5 ms   (today's real fp16 cm baseline)
ratio            = 203.3 / 331.5 = 0.613   →   1.63× FFN-weighted speedup
target           = 0.50   →   2× speedup
verdict          = STOP (in the 0.50–0.65 "document, do not proceed" band)
```

Phase 2 E2E port was therefore not started under this plan. The user did subsequently authorize an end-to-end study anyway; that work is in progress on a separate path (see commits after `b636b82a4` in main, `ecfd66eb3` in pavan-report).

## Weighted by LLaMA dispatch count — the practical comparison

What does each int8 path look like as a linear-time contribution to a full L=32 S=128 forward (224 linear dispatches, ignoring lm_head)?

| Path | Total linear time | Relative |
|---|---:|---:|
| fp16 baseline (`linear_vec`) | 64×12.3 + 32×6.7 + 64×6.7 + 64×1.2 = **~1450 ms** | (reference) |
| fp16 coopmat (`linear_coopmat`) | 64×3.42 + 32×3.52 + 64×0.92 + 64×0.26 = **~413 ms** | 3.5× faster than fp16 vec |
| int8 q8csw_tiled (W8A16, current main int8) | 64×9.23 + 32×9.62 + 64×2.74 + 64×0.82 = **~1138 ms** | 1.27× faster than fp16 vec, 2.75× slower than fp16 cm |
| int8 q8ta (W8A8 scalar, hypothetical) | 64×3.27 + 32×3.19 + 64×0.78 + 64×0.32 = **~380 ms** | 3.8× faster than fp16 vec |
| int8 KHR cm wave64 (W8A8 WMMA, pure coopmat) | 64×2.00 + 32×2.40 + 64×1.27 + 64×0.53 = **~320 ms** | 4.5× faster than fp16 vec |
| **Hybrid** (coopmat at FFN + q8ta at Q/O/K/V) | 64×2.00 + 32×2.40 + 64×0.78 + 64×0.32 = **~275 ms** | **5.3× faster than fp16 vec** |

Projection to E2E (add ~170 ms of unchanged non-linear ops, divide 128 tokens):

| Path | Forward (ms) | tok/s | vs fp16 baseline | vs fp16 cm |
|---|---:|---:|---:|---:|
| fp16 baseline | ~1766 (measured) | **72.5** | 1.00× | 0.33× |
| fp16 coopmat | ~583 (measured) | **220** | 3.03× | 1.00× |
| int8 q8csw (W8A16, measured E2E) | ~2108 | **60.7** | 0.84× | 0.28× |
| int8 KHR cm wave64 (projected) | ~490 | ~261 | 3.60× | 1.19× |
| Hybrid coopmat+q8ta (projected) | ~445 | ~288 | 3.97× | 1.31× |

## Headline takeaways

1. **fp16 cm is 4.5–5× faster than fp16 vec across every LLaMA shape.** The 3.03× E2E speedup comes straight from these per-shape numbers (diluted by ~10% non-linear ops via Amdahl).

2. **int8 KHR coopmat (wave64) is the fastest single shader at the FFN shapes.** 2.0 ms at FFN gate/up — beats fp16 cm by 1.72×, beats W8A16 by 4.6×.

3. **No single shader dominates everywhere.** q8ta scalar wins at small N. The best int8 path is a **hybrid heuristic**: coopmat for FFN (N ≥ 8192), q8ta for Q/O/K/V (N ≤ 4096).

4. **Pure coopmat E2E projection: ~261 tok/s (1.19× over fp16 cm).** Hybrid heuristic projection: ~288 tok/s (1.31× over fp16 cm). Both require wiring the wave64 shader into a real LLaMA dispatch site + W8A8 export recipe (not yet shipped end-to-end).

5. **Tile-schedule tuning is not the path to 2× int8 over fp16 cm.** The spec-constant levers explored (TILE_K, subgroup layout, BColMajor) don't close the gap. Closing it would need deeper shader rewrites (different prefetch pattern, guarded loops for TILE_N=64, or alternate tile algebra).

6. **The wave32 correctness bug** in `matmul_khr_cm_int8.glsl` original was caught only after writing a tight-tolerance validation driver. The original "1e10f" bench tolerance accepted half-zero outputs. This is now documented in `int8_coopmat_microbench_v2.md` and prevented going forward by the sweep-validation infrastructure committed in `e4aa21c21` / `fd0666988`.

## Practical recommendation

If asked "what's the int8 story on the 780M?", the microbench data says:

- **Today: run fp16 coopmat for prefill.** 3× over fp16 baseline, easy win, validated E2E (220 tok/s).
- **Next E2E step: wire wave64 int8 KHR coopmat for FFN + q8ta scalar for Q/O/K/V.** Projected ~288 tok/s (1.31× over fp16 cm). Requires W8A8 export recipe + dispatcher heuristic; partial unblock work was committed today across pavan-report (q8ta shaders fp16 variants, staging buffer int8 fix) and main (XNNPACK feasibility scripts + W8A8 export drivers).
- **Don't chase pure-coopmat 2×** by tile-tuning — the bottleneck at small N is structural occupancy on a 12-WGP iGPU, not tile-schedule. K/V launches only 8 workgroups regardless of how you tune the tile.
- **Memory-architecture limits** (the Part 1 cliff/OOM story) are independent and require quantization for working-set shrinkage. int4 weights remain the biggest unexplored single lever for both decode and S≥1024 prefill on this hardware.

## Connection between the two parts

The two halves of this report aren't independent:

- **Part 1** shows that memory architecture caps useful prefill at ~S=256 in fp16 on this iGPU. The cliff is independent of shader speed.
- **Part 2** shows that the linear-time fraction of the forward can be 4–5× faster with the right int8 KHR coopmat dispatch — but only IF the model can run end-to-end at the seq length you care about.

These are orthogonal levers:
- **Faster shaders** (Part 2) → more tok/s within the runnable seq range
- **Smaller weights** (int4 quant, out of scope for both this study and the current int8 work) → push the runnable seq range up

A complete optimization story for the 780M would combine both: int4 weights to unblock S≥1024 prefill and decode-without-thrashing, and an int8 coopmat hybrid heuristic to wring 3.6–4.0× over fp16 baseline on the linears that remain. Neither alone is sufficient.

## Artifacts

Under `yanwen/artifacts/int8_microbench/`:

- `khr_cm_gemm_int8_20260510_182522.log` — wave64 int8 KHR coopmat at LLaMA shapes (v1 source)
- `linear_coopmat_bench_20260510_204428_fp16_llama_baseline.log` — real fp16 cm LLaMA baseline (Phase 1.0 unblock)
- `khr_cm_gemm_int8_sweep_20260510_205654.log` — tile-schedule sweep (4 variants × 4 shapes)
- `khr_cm_validate_sweep_20260510_205654.log` — tight-tolerance validation of the 4 surviving variants
- `q8csw_linear_20260510_182522.log` — W8A16 and W8A8 q8ta non-coopmat numbers

Under `yanwen/artifacts/L32/`, `L32_coopmat/`, `L32_int8/`:

- ETDumps and memprobes for the E2E numbers cited in the performance results table (fp16 baseline 72.5 tok/s, fp16 coopmat 220 tok/s, int8 W8A16 60.7 tok/s + 25.65 s at S=512).
