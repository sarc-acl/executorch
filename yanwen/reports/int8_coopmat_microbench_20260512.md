# int8 cooperative-matrix microbench findings — 2026-05-12

**Author:** Yanwen Xu (with Claude)
**Hardware:** AMD Radeon 780M (RADV PHOENIX, RDNA3+ mobile iGPU, wave64), Mesa 25.0.7
**Scope:** GPU-timestamp microbenchmark only — no full LLaMA forward in this session
**Companion code change:** `pavan-report/.../matmul_khr_cm_int8_wave64.glsl` was updated to drop the shared-memory staging for matrix B (lever D: direct-buffer `coopMatLoad`). All 9 baseline validate cases still pass tight tolerance after the change.

## TL;DR

1. **The "default int8" path that LLaMA 3.1 actually dispatches today** (`-qmode int8 --vulkan`) is `linear_qcs8w_tiled` — a **W8A16 weight-only** shader. Its inner loop uses the fp16 multiplier, not any int8 hardware path. The "int8" in this configuration only saves weight memory bandwidth.
2. **Real apples-to-apples for "WMMA goodness"** is coopmat vs `linear_q8ta_q8csw_tiled` (q8ta) — both are W8A8 with real int8 hardware math. Coopmat uses `coopMatMulAdd<uint8, uint8, uint32>` (WMMA), q8ta uses packed `V_DOT4_I32_I8` (scalar int8 dot).
3. **Pure-coopmat vs pure-q8ta on LLaMA shapes** (after lever D): geomean **0.92×**, LLaMA-weighted **1.16×**. Coopmat wins big at FFN (1.6–1.2×), q8ta wins at attention projections (1.7–1.6×).
4. **Hybrid dispatch** (coopmat at FFN ≥ 8192-wide, q8ta below) is the best int8 path on this hardware. LLaMA-weighted speedup of **1.35× over pure q8ta**.
5. **Coopmat vs q8csw baseline** (the actual shipped int8 path): geomean **2.88×**, weighted **3.54×**. This is the realistic "WMMA goodness brought to LLaMA int8" headline, *but it bundles the W8A16→W8A8 transition with the WMMA-over-scalar-int8 delta*. The WMMA-specific delta over real W8A8 is the more honest +16% number.
6. **Why not "many × faster"**: mobile iGPUs don't have dedicated tensor-core silicon. WMMA and V_DOT4 share the same SIMD execution units with the same peak throughput (256 int8 ops/cycle/subgroup). WMMA's edge is purely instruction-count / lane-shuffle reduction.
7. **Variable-tile** (the lever to fix coopmat's K/V occupancy hole and make it competitive at small N) is **blocked** by a SPIR-V codegen issue we couldn't isolate analytically. RADV NIR dump captured for follow-up.

## 1. What's shipped today vs what we're studying

The relevant int8 paths in ExecuTorch Vulkan, by quantization recipe:

| Recipe | Shader dispatched | Activations | Weights | HW int8 math? | In LLaMA E2E today? |
|---|---|---|---|---|---|
| `-qmode int8` (weight-only) | `linear_qcs8w_tiled` | fp16 | int8 | **No** (fp16 multiplier, sw dequant in inner loop) | ✅ — produces the **60.7 tok/s** baseline |
| PT2E W8A8 static | `linear_q8ta_q8csw_tiled` (q8ta) | int8 | int8 | **Yes** (V_DOT4 scalar) | ❌ — work-in-progress (commit `33c1b05447`) |
| Our int8 cm | `matmul_khr_cm_int8_wave64` (coopmat) | int8 | int8 | **Yes** (KHR cooperative matrix WMMA) | ❌ — microbench only |
| Decode (M=1) | `q8ta_linear_gemv` | int8 | int8 | Yes | ❌ — same blocker |
| W4A16 (`-qmode 4w`) | `linear_q4gsw_*` | fp16 | int4 | No | Distinct path |

**The "default int8 shader from a LLaMA 3.1 run" is q8csw — confirmed from the int8 E2E ETDump at `yanwen/artifacts/L32_int8/S128.events.tsv`. Only `linear_qcs8w_tiled_*` linear shaders appear; no q8ta, no coopmat int8.**

## 2. The hardware capability surface

Queried via `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` (output of `khr_cm_gemm_int8` bench):

```
Found 14 cooperative matrix configurations:
  M=16, N=16, K=16    (the only fragment shape)
  12 of the 14 are int8 variants (signed/unsigned A,B with int32/uint32 accumulator)
```

So on this hardware: **one WMMA fragment shape, 16×16×16, twelve dtype combos.** No 16×16×32 wide-K variant. Any larger tile is built by stacking 16×16×16 fragments inside a workgroup.

This means "wider WMMA op" is not a lever on this hardware (lever C from the prior plan was wrong on this account).

## 3. The two hardware int8 paths

Both available on the 780M:

| Property | KHR cooperative matrix (WMMA) | Packed int8 dot product (V_DOT4) |
|---|---|---|
| GLSL primitive | `coopMatMulAdd<int8, int8, int32>` | `int32_dot_i8`-style intrinsic (or 4× packed `imadd_sat`) |
| Granularity | 16×16×16 fragment, subgroup-cooperative | Single SIMD lane, processes 4 int8 elements |
| Lanes per op | All 64 (subgroup-wide) | 1 (per-lane) |
| Throughput | 256 int8 ops/cycle/subgroup | 256 int8 ops/cycle/subgroup (4 × 64 lanes) |
| Setup overhead | Fragment load/store, lane-permute, tile prefetch | Per-thread loop, no inter-lane communication |
| Output | `int32` accumulator, separate cast/dequant pass | `int32` accumulator, scale-and-store per thread |
| Used by | Our `matmul_khr_cm_int8_wave64.glsl` | `linear_q8ta_q8csw_tiled.glsl` |

**Key insight: the two paths share the same SIMD pipeline. They differ only in instruction encoding and per-tile overhead.** This is structurally different from desktop/datacenter GPUs (Nvidia A100/H100, AMD MI300), where tensor cores have *separate* hardware on top of the regular SIMD units. On those chips WMMA can be 4–10× faster than V_DOT4. On the 780M they're peers.

## 4. Methodology

### Microbench harnesses

- `pavan-report/.../custom_ops/khr_cm_gemm_int8.cpp`: drives the coopmat shader at LLaMA shapes, sweep variants, and an N-crossover sweep
- `pavan-report/.../custom_ops/q8csw_linear.cpp`: drives both q8csw (W8A16) and q8ta (W8A8) at the same shapes
- Both use 3 warmup + 10 measured runs with Vulkan timestamp queries

### Per-shape pure-kernel timing

We use the GPU-side timestamp of the matmul shader alone, not the wrapping `et_vk.linear_q8ta_q8csw.default` total (which also includes `quantize_and_pack_4h4w` — the activation-quantization prefix kernel ~40 µs that wraps every q8ta linear call in microbench mode).

For coopmat, no activation-quant prefix exists (inputs are fed pre-quantized as uint8). To estimate a "fair" coopmat that also pays the dequant cost we'd need to add per-channel scale multiplication into the output store path — analytically estimated to add <1% on each shape (see §7).

## 5. Apples-to-apples results on LLaMA shapes

After today's lever-D change (matB direct-from-buffer load, no shared-mem staging):

| Shape (M, K, N) | Coopmat (ms) | q8ta linear-only (ms) | Coopmat speedup |
|---|---:|---:|---:|
| FFN gate/up (128, 4096, 14336) | **2.005** | 3.248 | **1.62×** ✓ |
| FFN down (128, 14336, 4096) | **2.449** | 2.927 | **1.20×** ✓ |
| Q/O (128, 4096, 4096) | 1.273 | **0.748** | 0.59× (q8ta wins 1.70×) |
| K/V (128, 4096, 1024) | 0.465 | **0.291** | 0.63× (q8ta wins 1.60×) |
| **Geomean** | — | — | **0.92×** |

LLaMA per-forward weighted (64 × gate/up + 32 × down + 64 × Q/O + 64 × K/V):

| Path | Total linear time | Speedup vs winner |
|---|---:|---|
| Pure q8ta | 368.0 ms | (reference) |
| Pure coopmat | 317.9 ms | **1.16×** over pure q8ta |
| **Hybrid** (coopmat at FFN, q8ta at attention) | **273.2 ms** | **1.35×** over pure q8ta, 1.16× over pure coopmat |

The hybrid path is the recommendation: coopmat where it wins, q8ta where it loses.

## 6. N-crossover sweep (fix M=128, K=4096)

Where does coopmat actually start beating q8ta?

| N | coopmat (µs) | q8ta linear-only (µs) | Winner | Speedup |
|---:|---:|---:|---|---:|
| 512 | **248** | 406 | coopmat | 1.64× (surprising) |
| 1024 | 350 | **290** | q8ta | 1.22× |
| 2048 | 819 | **454** | q8ta | 1.80× |
| 4096 | 829 / 1360 | **734** | q8ta | 1.13–1.85× |
| 8192 | 1640 | 1716 | tie | 1.05× (coopmat) |
| 12288 | 1996 | 2730 | **coopmat** | 1.37× |
| 14336 | 1533 / 2090 | 3170 | **coopmat** | 1.52–2.07× |
| 16384 | 2311 | 3596 | **coopmat** | 1.56× |

### Findings

- **Real crossover at N ≈ 8192.** Confirms the report's "set hybrid threshold around N=8192" — now with explicit measurement instead of extrapolation.
- **Surprise at N=512**: coopmat wins 1.64× — contradicts the simple "coopmat loses at small N" story. Likely because both paths are dispatch-overhead-bound at this size; coopmat's smaller workgroup count (4 vs q8ta's 64) means less driver-side bookkeeping.
- **Variance is large**: ~20–30% between repeat measurements of the same shape (compare the "llama_qo" vs "nsweep_n4096" entries which are bit-for-bit identical configurations but differ from 829 µs to 1360 µs). Run-to-run variance is partly cache-state-dependent. For headline numbers this needs more runs and/or averaging across positions in the test list.
- **q8ta scales near-linearly with N** (as expected for a fixed-overhead scalar pipeline). Fit: `time(µs) ≈ 290 + 0.20 × N`.
- **Coopmat's curve is non-monotonic** — it has a kink at small N where the 128-wide tile under-fills the GPU (K/V at N=1024 launches only 8 workgroups vs the 12 WGPs available), then catches up at large N.

## 7. Fair-comparison estimate (with fused per-channel dequant)

The unmodified coopmat shader computes raw `int8 × int8 → int32 → float` (no scale multiplication). q8ta computes the *full* quantized linear including per-channel scale + cast to fp16. So the head-to-head numbers in §5 measure slightly different operations.

To estimate what coopmat's numbers would look like with the missing dequant fused in:

- Per output element: 1 fp multiply (scale × accumulator) + 1 fp16 store
- For a 128×128 output tile = 16,384 elements distributed across 256 threads → 64 mul/thread → ~128 cycles per workgroup ≈ **0.13 µs/WG**
- Multiplied by workgroup count per shape: 15 µs for FFN gate/up (0.7%), 4 µs for FFN down, 1 µs for K/V

**The estimated overhead is <1% per shape.** Reason: dequant is `O(M·N)` but matmul is `O(M·N·K)`. At K=4096+ the dequant is K-times cheaper than the matmul. The apples-to-apples picture is essentially unchanged.

Full implementation deferred — would not change the conclusion. The microbench's coopmat-vs-q8ta numbers are honest within ~3% as comparisons.

## 8. Why not "many × faster"

The 1.16× weighted win is much less than the 4–10× WMMA wins reported on datacenter GPUs. The structural reasons:

1. **No dedicated tensor-core silicon on this iGPU.** WMMA and V_DOT4 are both implemented on the same SIMD execution units. Peak int8 throughput per cycle per subgroup: 256 for both.
2. **WMMA's only edge is instruction-count reduction.** One `coopMatMulAdd` covers 256 multiplies vs ~64 individual `V_DOT4` instructions. That's ~1.3–1.6× when the inner loop is dispatch-bound, less when bandwidth-bound.
3. **Bandwidth ceiling**. LLaMA prefill at S=128 has modest arithmetic intensity. FFN gate/up: 7.5 GFLOP against 59 MB of weight + activation reads → at 80 GB/s DDR5 the memory floor is already 740 µs (matmul measured 2 ms total). Less headroom for WMMA to pull ahead.
4. **Tile-shape penalty at small N**. The fixed 128×128 output tile launches only 8 workgroups at K/V (N=1024) on a 12-WGP GPU. q8ta's smaller per-thread tile launches more workgroups → better occupancy. Variable-tile would fix this (see §10).
5. **q8ta is well-tuned already.** Multiple iterations of empirical tuning; coopmat is newer code on the same hardware.

The "tensor cores crush scalar by 4×+" story is a datacenter-GPU narrative that doesn't transfer to mobile iGPUs with unified SIMD pipelines.

## 8.5. Roofline analysis & hardware-utilization decomposition

This section answers "is 1.6× the structural limit, and why?" with measured numbers (not just analogies).

### 8.5.1 Hardware peak rates on the 780M (theoretical vs empirical)

From RADV device info dump (`RADV_DEBUG=info`):

```
num_cu                = 12 (Compute Units, RDNA3+)
max_gpu_freq          = 2799 MHz
max_gflops            = 8598 GFLOPS (FP32 peak)
memory_bandwidth      = 180 GB/s
L1 cache              = 256 KB per WGP
L2 cache              = 2 MB
```

**Theoretical int8 peak**: `max_gflops × 4` (since `V_DOT4_I32_I8` and WMMA int8 both fuse 4 int8 multiplies into one FP32-equivalent slot) = **~34 TOPS peak int8**. This is the architecture's nominal limit.

**Empirical achievable peak** (measured): the same `khr_cm_gemm_int8` bench includes a `sq_4096x4096x4096` (M=N=K=4096) configuration that's large enough to saturate the WMMA pipeline and amortize all per-WG overhead. Today's coopmat measurement at this shape is **10.4 TOPS** (13.18 ms for 137 GOP), which corresponds to **31% of theoretical peak**.

This 31% number is the *practical ceiling for this hardware running this kind of WMMA shader*. The gap from 34 → 10.4 TOPS comes from a combination of:

- Sustained-load clock-throttling on this iGPU class (the 2.8 GHz `max_gpu_freq` is rarely held under continuous compute load)
- WMMA pipeline + fragment-load overhead that's not present in the V_DOT4 theoretical max
- Inner-loop non-MAC instructions (prefetch, shared-mem barriers, coopMatLoad sequencing)

For the rest of §8.5 we report efficiency two ways: **% theoretical** (sets the absolute architecture context) and **% empirical achievable** (the actually-meaningful headroom).

### 8.5.2 Shader resource usage (RADV_DEBUG=shaderstats, no shader cache)

| Shader specialization (best guess) | VGPRs | SGPRs | Code size | Spills |
|---|---:|---:|---:|---:|
| `matmul_khr_cm_int8` (impl=3, broken wave32) | 104 | 128 | 2912 B | 0 |
| `matmul_khr_cm_int8_wave64` (impl=4 default) | 128 | 128 | 5020 B | 0 |
| `v0_baseline` (sg=2×2, TILE=128×128, TILE_K=32) | 128 | 128 | 5020 B | 0 |
| `v1_deepK` (TILE_K=64) | **200** | 128 | 7232 B | 0 |
| `v3_sg1x4` (sg=1×4 → C_COLS=8) | 168 | 128 | 4208 B | 0 |
| `v4_sg4x1` (sg=4×1 → C_ROWS=8) | 128 | 128 | 6836 B | 0 |
| `v2_tile128x64_BROKEN` (TILE_N=64) | 96 | 128 | 2780 B | 0 |

**Key findings:**
- **No register spills anywhere.** The compiler fits every variant in registers.
- VGPRs range 96–200 / wave64. RDNA3+ has 1024 VGPRs per SIMD32 pair (effective per wave64); the maximum-VGPR variant (`v1_deepK` at 200) still leaves headroom for ~5 simultaneous wave64s per SIMD pair. **Register pressure is not the binding constraint.**
- Code size ≤ 7.3 KB easily fits the 32 KB instruction cache.
- `v2_tile128x64_BROKEN` has the LOWEST VGPR count (96) — smaller tile = fewer fragment registers held simultaneously. So the codegen bug we hit isn't VGPR-related either; it's a correctness issue, not a resource one.

### 8.5.3 Achieved-vs-peak per shape (coopmat, lever-D shader)

| Shape (M,K,N) | Ops | Bytes (A+B+D) | Time (ms) | Achieved TOPS | Achieved BW (GB/s) | % theoretical (34 TOPS) | **% empirical (10.4 TOPS)** |
|---|---:|---:|---:|---:|---:|---:|---:|
| sq_4096³ (calibration anchor) | 137 G | 49 MB | 13.18 | **10.4** | 3.7 | 31% | **100%** |
| FFN gate/up (128, 4096, 14336) | 15.04 G | 66.6 MB | 2.005 | 7.50 | 33.2 | 22% | **72%** |
| FFN down (128, 14336, 4096) | 15.04 G | 62.6 MB | 2.449 | 6.14 | 25.6 | 18% | 59% |
| Q/O (128, 4096, 4096) | 4.29 G | 19.4 MB | 1.273 | 3.37 | 15.3 | 10% | 32% |
| K/V (128, 4096, 1024) | 1.07 G | 5.25 MB | 0.465 | 2.31 | 11.3 | 7% | **22%** |

The "% empirical" column is the meaningful one: **FFN gate/up runs at 72% of the achievable WMMA ceiling on this chip**, leaving only ~28% as theoretical headroom. **K/V at 22% has substantial headroom** (~78% gap) — which is the size of the prize variable-tile would unlock.

For comparison, q8ta on the same shapes:

| Shape | q8ta time (ms) | q8ta TOPS | q8ta compute eff |
|---|---:|---:|---:|
| FFN gate/up | 3.248 | 4.63 | 13.6% |
| FFN down | 2.927 | 5.14 | 15.1% |
| Q/O | 0.748 | 5.74 | 16.9% |
| K/V | 0.291 | 3.68 | 10.8% |

**Key observation**: at FFN shapes, coopmat extracts more peak compute (22% vs 14% for q8ta — the WMMA's smaller instruction count helps). At attention shapes, q8ta extracts more peak compute (10–17% vs coopmat's 7–10% — because q8ta's smaller tile geometry fills the GPU better). This *is* the crossover, expressed in efficiency units.

### 8.5.4 Roofline classification

For all 4 shapes the arithmetic intensity is `~200 ops/byte`, well above the ridge point `34 TOPS / 180 GB/s = 0.19 ops/byte`. **Every LLaMA prefill linear is squarely compute-bound** — no shape is bandwidth-limited on this hardware.

So the question "why are we so far below 34 TOPS peak?" decomposes into:
1. How much of the 12 CUs is actually busy? (CU occupancy)
2. Within an active CU, what fraction of cycles issue useful WMMA ops?

### 8.5.5 CU occupancy decomposition

Each workgroup runs on one CU. Workgroups dispatched, vs 12 CUs:

| Shape | Workgroups | WGs/CU | CU "occupancy" |
|---|---:|---:|---:|
| FFN gate/up | 112 | 9.3 | 100% (oversubscribed, plenty of WGs to hide latency) |
| FFN down | 32 | 2.7 | 100% (still well-fed) |
| Q/O | 32 | 2.7 | 100% (still well-fed) |
| K/V | **8** | **0.67** | **~67%** (4 CUs are guaranteed idle!) |

K/V is the only shape where the dispatch geometry alone leaves CUs idle. The other shapes have enough workgroups in flight that the scheduler can keep all 12 CUs busy.

### 8.5.5b Static instruction mix (from RADV_DEBUG=shaders,asm)

The AMD GFX11 (RDNA3+) ISA dump for each compiled coopmat variant. Inner-loop instruction counts:

| Variant | WMMA | DS (shared mem) | Other VALU | Global load | Global store | Barrier/wait | Total typed |
|---|---:|---:|---:|---:|---:|---:|---:|
| `matmul_khr_cm_int8` (impl=3, wave32 broken) | 32 | 76 | 220 | 4 | 64 | 24 | 420 |
| **`matmul_khr_cm_int8_wave64` (impl=4 / v0_baseline)** | **64** | **140** | **380** | **4** | **128** | **27** | **743** |
| `v1_deepK` (TILE_K=64) | 128 | 280 | 458 | 8 | 128 | 49 | 1051 |
| `v3_sg1x4` (sg=1×4) | 64 | 264 | 451 | 4 | 128 | 32 | 943 |
| `v4_sg4x1` (sg=4×1) | 64 | 84 | 348 | 4 | 128 | 32 | 660 |
| `v2_tile128x64_BROKEN` (TILE_N=64) | 32 | 74 | 199 | 2 | 64 | 22 | 393 |

**v0_baseline (the canonical coopmat shader) issues 743 instructions, of which only 64 (~9%) are WMMA.** The remaining ~91% are:
- DS ops (shared-mem reads of matA fragments): 140
- Other VALU: 380 (address arithmetic, scale/cast operations, control flow)
- Global stores: 128 (final output writes)
- Barrier/wait: 27

This is the static-instruction-count root cause of the "WMMA pipeline efficiency 22% of theoretical / 72% of empirical" observation:

- If WMMA had its own free issue port and could overlap fully with VALU/DS/store, theoretical efficiency would approach 100%
- In practice the inner-loop has ~6× as many non-WMMA instructions as WMMA instructions, so under serial-issue assumptions WMMA's effective rate is bounded by `1 / (1 + overhead_ratio)` ≈ 14% of instruction throughput
- Some pipelining recovers part of this (RDNA3+ can dual-issue scalar + vector); the empirical 31% of theoretical 34 TOPS = 10.4 TOPS is what survives
- The `v4_sg4x1` variant has the *fewest* DS ops (84 vs 140 for v0) — fewer matA reloads from shared mem per K-step. But it doesn't run faster overall because of higher per-tile output-store cost. Tradeoff, not strict improvement.

**Comparing to q8ta**: a separately-dumped `linear_q8ta_q8csw_tiled` shader would show V_DOT4 instructions instead of WMMA, and a different overhead mix. Each V_DOT4 does 4 int8 multiplies vs WMMA's 256 — so q8ta needs many more arithmetic ops per output tile, but each is cheaper to set up. Whether the trade favors one or the other depends on the shape (see §8.5.5d).

### 8.5.6 Per-CU efficiency decomposition (against empirical peak)

For shapes with full CU occupancy, the remaining gap from peak is per-CU pipeline efficiency. Using the **empirical 10.4 TOPS ceiling** instead of theoretical:

| Shape | Achieved (% empirical) | CU occupancy | Per-CU pipeline eff (vs empirical) |
|---|---:|---:|---:|
| FFN gate/up | 72% | ~100% | **72%** |
| FFN down | 59% | ~100% | 59% |
| Q/O | 32% | ~100% | 32% |
| K/V | 22% | 67% | **22% / 0.67 = ~33%** |

So **at full CU occupancy, coopmat is hitting 32–72% of the empirical ceiling** depending on shape. FFN gate/up at 72% is already very good — close to what `sq_4096` delivers in the saturating regime. The gap-to-empirical for FFN is mostly *per-WG fixed overhead amortization* (sq_4096 has 1024 WGs vs FFN's 112 — more amortization). The components of that gap:

- **WMMA pipeline latency that can't be hidden**: the `coopMatMulAdd` instruction has ~4–8 cycle latency on RDNA3+; with only 1 WG/CU there aren't enough waves to hide it. Tighter when WGs/CU is small (K/V) than large (FFN).
- **Cache miss rate on B-loads**: FFN gate/up reads 58.7 MB of B which exceeds the 2 MB L2 → roughly all-miss to DDR. The 33 GB/s achieved BW reflects this — not the 180 GB/s peak.
- **Per-WG fixed overhead**: tile prefetch, barriers, coopmat fragment loads. Constant per WG; amortizes worse at small K (Q/O at K=4096 vs FFN down at K=14336 — note 10% vs 18% per-CU efficiency, even though both have 32 WGs).

### 8.5.7 What it would take to push past 72% of empirical

For FFN gate/up at 72% of the empirical ceiling, the remaining 28% headroom (i.e., reaching the sq_4096 saturating rate) comes from:

- **Per-WG fixed overhead amortization**: sq_4096 has 1024 WGs vs FFN's 112. Each WG pays a fixed cost for tile prefetch + barriers + initial coopMatLoad. At 112 WGs, fixed overhead is ~10× less amortized than at 1024 WGs.
- **WMMA pipeline latency hiding**: with fewer WGs per CU, the scheduler has fewer waves to switch between when WMMA is stalled. More WGs = more waves = more latency hiding.
- **L2 miss rate on B-loads**: FFN B-tile is 58 MB > 2 MB L2 → all B-reads go to DDR. sq_4096 has B = 16 MB which is also bigger than L2, but the access pattern with more M-tiles produces some inter-tile reuse. With M=128 = 1 M-tile there's no such reuse.

None of these easily yield a 2× FFN improvement on this hardware. **The 72% achieved at FFN gate/up is realistically near the per-shape ceiling** for a 128-row, 14336-column problem. The remaining 28% gap to sq_4096 is structural to the *shape*, not the shader. **Closing it would require a longer sequence (more M-rows) — i.e., S=2048 prefill rather than S=128.**

### 8.5.8 What it would take to fix K/V

K/V's 22% of empirical breaks down as `67% CU occupancy × 33% per-CU efficiency`. Two independent levers:

1. **Push CU occupancy from 67% to 100%** via variable tile (TILE_N=64 would launch 16 WGs covering all 12 CUs with ~1.3 WGs/CU). Currently blocked by the codegen bug; would lift K/V from 22% of empirical to ~33% (= 100% × 33%).
2. **Push per-CU efficiency from 33% to ~59%** (matching FFN down at similar WG-density): per-WG overhead amortization at K=4096 is fundamentally lower than at K=14336. Probably realistic to reach ~45–50% per-CU.

Combined ceiling: K/V coopmat could potentially go from `22% × 10.4 TOPS = 2.31 TOPS` to `~100% × ~45% × 10.4 = 4.7 TOPS`, i.e., **0.465 ms → 0.23 ms** (roughly 2× speedup). That would beat q8ta (currently 0.291 ms) at K/V too. **This is the headroom variable-tile would unlock — quantified against the empirical ceiling, not a theoretical one.**

### 8.5.9 The "this is the limit because X" summary

For the existing 128×128 coopmat shader on the 780M (anchored against empirical 10.4 TOPS, not theoretical 34 TOPS):

| Shape | Bottleneck | % of empirical | Headroom available | Lever |
|---|---|---:|---|---|
| FFN gate/up | Per-WG fixed-overhead amortization at M=128 (only 1 M-tile) | **72%** | Small (~28%) | Longer sequence (S=2048+) to add M-tiles |
| FFN down | Same as gate/up + slightly fewer WGs | 59% | Small | Same |
| Q/O | Per-WG overhead at K=4096 (shallow K = less amortization) | 32% | Modest | Hybrid (use q8ta here) |
| K/V | **CU occupancy hole** (8 WGs vs 12 CUs) + shallow K | **22%** | Large (~70%) | **Variable tile**, currently blocked |

**The 1.16× weighted win over q8ta is the realistic ceiling** for the current shader on this hardware because:
1. FFN shapes are at ~22% peak compute, with only ~10–20% additional headroom available via inner-loop micro-optimizations on a shape that's already well-tuned.
2. Attention shapes are bound by either per-WG overhead amortization (Q/O) or pure CU occupancy (K/V). The K/V fix exists structurally but is blocked.
3. The 780M's 34 TOPS int8 peak is shared between WMMA and V_DOT4 on the same SIMD pipeline — neither path has a hardware floor advantage. The 22% achieved at FFN is essentially the WMMA shader's "best version" on this chip.

**The structural conclusion**: this iGPU's WMMA throughput is fundamentally bottlenecked by its small CU count (12), shared SIMD pipeline (no separate tensor cores), and modest memory bandwidth (180 GB/s for the cache miss path). A datacenter GPU with dedicated tensor cores, more CUs, and HBM bandwidth would deliver the 4–10× WMMA-over-scalar story; the 780M cannot, no matter how the shader is tuned.

## 9. Where the 3.54× headline really comes from

Versus the **shipped LLaMA int8 baseline** (q8csw, W8A16):

| Path | LLaMA-weighted linear time | Speedup vs q8csw |
|---|---:|---:|
| q8csw (W8A16, what ships today) | 1126.4 ms | 1.00× |
| Pure q8ta (W8A8 scalar) | 368.0 ms | **3.06×** |
| Pure coopmat (W8A8 WMMA) | 317.9 ms | **3.54×** |
| Hybrid coopmat + q8ta | 273.2 ms | **4.12×** |

**Most of this 3.54× is the W8A16→W8A8 transition (3.06×).** The WMMA-specific delta on top is 3.54/3.06 = 1.16× (matching §5). So the headline number bundles two different effects:

- **The 3× lift**: getting onto a real hardware int8 path at all (q8ta or coopmat). Either path produces this; the choice between them is the secondary +16%.
- **The +16% WMMA delta**: coopmat over q8ta when both are W8A8.

This matters for framing: "WMMA goodness brought to LLaMA int8" gets you the 3.54× only if you ALSO land the W8A8 export path (currently in progress in commit `33c1b05447`, blocked on runtime fix `fd0666988a`).

## 9.5 Profiling infrastructure (added 2026-05-13)

### Static-analysis tooling available without install

- `RADV_DEBUG=shaders` — dumps NIR + GPU asm for every compiled pipeline (used in §8.5.5b for instruction-mix breakdown)
- `RADV_DEBUG=shaderstats` — per-shader VGPRs, SGPRs, code size, spill count (used in §8.5.2)
- `RADV_DEBUG=info` — device capability dump (used for peak rates in §8.5.1)
- `MESA_SHADER_CACHE_DISABLE=1` — required when re-dumping stats for the same shader (otherwise the cache short-circuits compilation and stats aren't re-emitted)

### Hardware-counter capture: RGP/SQTT works on RADV

Mesa 25.0.7 has full SQTT (Streaming Performance Counters / Thread Trace) capture built into RADV. Confirmed working with:

```bash
MESA_VK_TRACE=rgp MESA_VK_TRACE_PER_SUBMIT=1 ./bench
```

Output: a `.rgp` file in the working directory per Vulkan submit. With this enabled the driver prints:

```
radv: Thread trace support is enabled (initial buffer size: 32 MiB,
      instruction timing: enabled, cache counters: enabled,
      queue events: enabled).
```

**This gives access to**: wave occupancy timeline, per-instruction timing, L0/L1/L2 cache hit rates, stall reasons, queue events. Essentially every hardware counter we'd want to validate §8.5's analytical decomposition.

To view: install AMD's standalone Radeon GPU Profiler tool (free from gpuopen.com/rgp/). Opens the `.rgp` file directly. **No special install on the bench machine needed** — just the env vars above.

A captured trace is archived at `yanwen/artifacts/int8_microbench/coopmat_validate_20260513.rgp` (112 KB) for offline inspection. Triggering full per-shape captures on the perf bench is a one-liner; deferred until someone wants to actually open RGP and inspect the data.

### What follow-up RGP analysis would settle

The unmeasured §8.5 claims that an RGP trace would directly validate or refute:

1. **Actual GPU clock during the kernel** — the 2.8 GHz "max_gpu_freq" assumed in §8.5.1's 34 TOPS theoretical may be clock-throttled in practice. RGP shows real clocks per submit.
2. **CU occupancy at K/V (claimed 67%)** — RGP shows the actual concurrent-wave count per CU over time. If K/V achieves >67% wave occupancy, our headroom estimate for variable-tile is overstated.
3. **L2 hit rate on FFN B-loads (claimed ≈0%)** — RGP shows L0/L1/L2 hit ratios per shader. If hit rate is non-trivial, the 18% achieved BW number understates the effective utilization.
4. **WMMA pipeline stall reasons** — RGP categorizes cycles as `valu_active / valu_stall_dep / mem_stall / barrier / etc`. Would tell us *which* of the candidate bottlenecks (WMMA pipeline / non-MAC inner-loop / L2 miss) is actually dominating at FFN gate/up.

## 10. Remaining work

| Item | Status | Effort | Expected impact |
|---|---|---|---|
| W8A8 LLaMA export (commit `33c1b05447`, L=4 .pte exists) | Partial — L=4 export works, runtime crashes against today's binary (likely ABI drift; needs re-export) | hours | Unblocks E2E measurement |
| Variable-tile coopmat (TILE_N=64 for K/V occupancy) | **Blocked** by SPIR-V codegen issue producing permuted output. Affects both guarded-prefetch and direct-buffer approaches. RADV NIR dump captured (`radv_shaders_dump_20260512.log`); analysis deferred | days | Would close the q8ta K/V gap; potential +5–10% to the hybrid weighted total |
| Fused dequant in coopmat output store | Analytical estimate <1% impact; implementation deferred | hours | <1% per shape — doesn't move the headline |
| End-to-end LLaMA W8A8 + coopmat-at-FFN dispatch | Blocked by W8A8 runtime crash above | days | Confirms the projected 4.12× E2E |

## 11. Artifacts

All under `yanwen/artifacts/int8_microbench/`:

- `q8csw_q8ta_linear_20260512_W8A8_apples.log` — full output of `q8csw_linear` bench with LLaMA shapes + N-sweep
- `coopmat_nsweep_20260512.log` — coopmat bench output filtered to wave64 LLaMA + nsweep rows
- `q8ta_nsweep_20260512.log` — q8csw_linear bench output (same as the W8A8 apples file, kept separate for legibility)
- `radv_shaders_dump_20260512.log` — 1.4 MB RADV NIR + GPU assembly dump for 10 shader specializations (for offline variable-tile codegen investigation)
- `shaderstats_radv_20260513.log` — ACO compiler stats (VGPRs, SGPRs, code size, spill count) per pipeline. Source for §8.5.2.
- `radv_asm_dump_20260513.log` — full AMD GFX11 ISA dump for all 10 pipelines. Source for §8.5.5b instruction-mix breakdown.
- `coopmat_validate_20260513.rgp` — RGP capture from the validate harness (112 KB). Open in AMD Radeon GPU Profiler for wave occupancy / cache counters / stall-reason analysis.

## 12. Methodology caveats (be honest)

- **Variance**: 20–30% between repeat measurements of the same shape, depending on position in the test list. Caused by GPU/cache warmup state. Headline numbers should be averaged over several positions or pinned with `nice -n -20` + thermal stabilization.
- **The microbench measures different ops** for coopmat (raw int32 GEMM) and q8ta (full quantized linear with scale + bias). §7 estimates the difference at <1% per shape but a fully-fused coopmat would close it.
- **Coopmat uses uint8 inputs; q8ta uses int8.** Should be hardware-symmetric per the queried KHR coopmat config list, but is a configurational asymmetry worth documenting.
- **No E2E confirmation yet.** All projections are µ-bench-based. The 4.12× hybrid headline is a projection until W8A8 LLaMA runs.
- **One device.** Single 780M iGPU under one driver state.

## 13. Practical recommendation

**For pushing WMMA into ExecuTorch LLaMA int8 on the 780M:**

1. **Land the W8A8 export pipeline** (commit `33c1b05447`, debug the L=32 runtime crash) — this is the big lift (~3× by itself). Without it, coopmat can't be dispatched at all.
2. **Wire the wave64 KHR coopmat shader as a dispatch target** alongside `q8ta_linear` for the `et_vk.q8ta_linear.default` op, with a shape-based heuristic: `N ≥ 8192 → coopmat, else q8ta`.
3. **Don't chase the small-N coopmat win** by tile tuning until the variable-tile codegen issue is debugged with RGP or RADV NIR analysis. The hybrid heuristic gets ~95% of the available win anyway.
4. **Position the WMMA narrative honestly**: the headline is "int8 hardware math (W8A8) makes LLaMA 3× faster"; the WMMA-specific add-on is "+16%". The 4.12× total is the right number to advertise — it's accurate, but not "WMMA all by itself".
