# LLaMA 3.1 8B prefill at L=32 S=128 with W8A16 int8 — findings

**Updated:** 2026-05-10 · **Author:** Yanwen Xu · **Branch:** `main` · **Scope:** L=32, S=128, prefill, weight-only int8 (`--pt2e_quantize vulkan_8w`)

**Device:** AMD Radeon 780M (RADV PHOENIX, RDNA3+ mobile iGPU), Mesa 25.0.7
**Host:** 28.9 GiB DDR5 RAM, 24 GiB swap (`/swapfile`)
**Model:** LLaMA 3.1 8B with per-channel symmetric int8 weights (PT2E `VulkanQuantizer`), fp16 activations

Companion to [`REPORT.md`](REPORT.md) (the fp16 `linear_vec` baseline). Compare line-for-line.

## TL;DR

| | fp16 baseline | **int8 W8A16** | Ratio (int8 / fp16) |
|---|---:|---:|---:|
| **Steady-state forward** | 1765.6 ± 6.0 ms | **2107.8 ± 28.8 ms** | **1.19× SLOWER** |
| **Throughput** | 72.5 tok/s | **60.7 tok/s** | 0.84× |
| **.pte size** | 16.06 GB | **8.56 GB** | 0.53× |
| **peak Shmem** | 920 MB | **2 MB** | 0.002× |
| **min MemFree** | 6.2 GB | **10.0 GB** | +60% headroom |
| **Linear share of GPU time** | 80.1% | **84.1%** | (more dominated) |
| cv across reps | 0.3% | 1.4% | |
| W1 (load + iter 0) | 21.4 s | **15.6 s** | 0.73× |

**The headline finding is that vulkan_8w int8 is NOT a speed win at S=128 prefill on this hardware.** It's a memory/footprint win (~half the .pte size, ~440× less GPU-visible Shmem, +4 GB MemFree headroom) bought at a 19% wallclock regression. The cause is per-dispatch: the `linear_qcs8w_tiled_*half_o4x4` shader runs slower than fp16's `linear_vec_buffer_texture2d_half` for large-N shapes (FFN gate/up), which dominate the forward.

## What was needed to get this number

Two upstream gaps had to be patched before the benchmark would even run end-to-end:

1. **`linear_qcsnw_tiled.yaml` was missing all `_half` variants.** The `vulkan_8w` quantizer produces an int8 graph with fp16 activations, so the runtime asks for `linear_qcs8w_tiled_buffer_buffer_texture2d_texture2d_half_o4x4` — but the YAML only generated `_float` variants. Result: `Could not find ShaderInfo` throw at first dispatch. Fix: added `DTYPE: half` variants (see `backends/vulkan/runtime/graph/ops/glsl/linear_qcsnw_tiled.yaml`).

2. **`linear_qcsnw_tiled.glsl` uses `float16_t` in a local array** but the codegen helper `define_required_extensions(OUT_STORAGE="texture3d", DTYPE="half")` only emits the fp16 extension when storage is `buffer`. Fix: added an explicit `$if DTYPE == "half": #extension GL_EXT_shader_explicit_arithmetic_types_float16 : require` at the top of the GLSL source.

3. **`linear_qcsnw_coop.glsl` (M=1 path) uses `shared VEC4_T` which would need an even broader set of fp16 extensions for `shared` storage class.** Workaround: in `QuantizedLinearQCSNW.cpp::can_use_coop_impl`, return `false` for half-precision graphs so M=1 dispatches fall through to the tiled algorithm (which has my new half variants). The perf hit is negligible at the lm_head's M=1 shape.

These three changes are local edits in the main tree and are not load-bearing for fp32 int8 paths. Without them, `--pt2e_quantize vulkan_8w` does not work end-to-end with fp16 LLaMA on `main` today.

## Methodology

Same scientific bench as `REPORT.md`: 1 calibration subprocess at N=1 + 3 measurement subprocesses at N=8, algebraic subtraction. Same runner (`main/executorch/cmake-out-vk/executor_runner`, EVENT_TRACER on). Same analyzers.

Export pipeline (in `yanwen/scripts/int8/run_llama31_int8.py::export_pte`):

```
load fp16 model (16 GB weights)
  → torch.export(strict=True).module()       (22 s)
  → prepare_pt2e(graph, VulkanQuantizer.w8a16)
  → forward(dummy_input)                     (19 s — observers latch weight stats)
  → convert_pt2e(fold_quantize=True)         (41 s — int8 weights materialized)
  → torch.export(strict=False)               (2 s)
  → to_edge_transform_and_lower([VulkanPartitioner({})])  (237 s)
  → to_executorch.write_pte                  (8.56 GB)
```

The Vulkan partitioner's `QuantizedLinearMatch` pattern (`backends/vulkan/patterns/quantized_linear.py`) detects the (dequantize-per-channel → linear) pairs PT2E emits and folds them to `aten._weight_int8pack_mm.default`, which the runtime routes to `linear_qcs8w_tiled_*` via `QuantizedLinearQCSNW.cpp::weight_int8pack_mm`.

## Headline run

```
W1 (load + iter 0 + teardown):   15.56 s
rep 1 steady:                     2135.7 ms
rep 2 steady:                     2078.1 ms
rep 3 steady:                     2109.7 ms

Steady-state forward:  2107.8 ± 28.8 ms   (cv = 1.4%)
Prefill throughput:    60.7 tokens/sec
```

Run command:

```bash
source /home/doremy/sarc-acl/executorch/main/executorch/.venv/bin/activate
cd /home/doremy/sarc-acl/executorch/main/executorch
python yanwen/scripts/int8/bench_llama31_int8.py --n_layers 32 --seq_len 128
```

Log: [`artifacts/L32_int8/bench_L32_S128_*.log`](../artifacts/L32_int8/).

## Memory at L=32 S=128 (int8 W8A16)

| Quantity | int8 W8A16 | fp16 baseline | Delta |
|---|---:|---:|---:|
| peak Shmem (RADV GTT-backed) | **2 MB** | 920 MB | **−918 MB** |
| peak Cached (.pte mmap) | 9.18 GB | 9.46 GB | similar |
| min MemFree during run | **10.0 GB** | 6.2 GB | **+3.8 GB** |
| peak Swap used | 4.3 GB | ~0 (this run started with system swap already used) | n/a |
| `.pte` on disk | **8.56 GB** | 16.06 GB | −7.5 GB |

The two big takeaways:

- **Vulkan-visible Shmem dropped 440× (920 MB → 2 MB).** With int8 weights packed via texture2d and the activations going buffer-only, the int8 path doesn't materialize the huge intermediate tensors that the fp16 path's texture3d activations did. Means int8 should run S>128 much further before cliffing.
- **MemFree headroom is +3.8 GB** — this hardware would now have room to do S=256–512 at int8 where fp16 cliffed.

Neither of these benefits matters to the wallclock number at S=128 — but they unblock follow-on experiments that were OOM-walled with fp16.

## ETDump shader breakdown

Generated via the canonical analyzer:

```bash
python /home/doremy/sarc-acl/executorch/pavan-report/executorch/yanwen_plan/analyze_etdump.py \
    yanwen/artifacts/L32_int8/S128.etdp
```

### Categories side-by-side

| Category | int8 (ms) | int8 % | fp16 (ms) | fp16 % | Per-category ratio (int8/fp16) |
|---|---:|---:|---:|---:|---:|
| **linear** | **1983.4** | 84.1% | 1527.6 | 80.1% | **1.30×** |
| reshape / view | 105.2 | 4.5% | 106.7 | 5.6% | 0.99× |
| CPU↔GPU copy | 89.0 | 3.8% | 91.4 | 4.8% | 0.97× |
| CPU fallback (cat) | 77.4 | 3.3% | 76.7 | 4.0% | 1.01× |
| elementwise | 59.1 | 2.5% | 59.0 | 3.1% | 1.00× |
| bmm / matmul | 14.3 | 0.6% | 14.5 | 0.8% | 0.99× |
| softmax | 6.7 | 0.3% | 6.9 | 0.4% | 0.97× |
| other / fallbacks | ~24 | ~1.0% | ~20 | ~1.0% | similar |
| **TOTAL ETDump** | **2359** | | 1908 | | **1.24×** |

Everything except **linear** is virtually identical to fp16 — surrounding ops aren't affected by the int8 swap. The 30% regression in the linear category is the entire story.

### Linear by output shape — the actionable cut

```bash
python yanwen/scripts/linear_by_shape.py yanwen/artifacts/L32_int8/S128.etdp
```

| Output shape | # disp | int8 avg ms | fp16 avg ms | Ratio | Component | int8 share of linear |
|---|---:|---:|---:|---:|---|---:|
| `[128, 14336]` | 64 | **20.24** | 12.30 | **1.65× SLOWER** | FFN gate + up | **65.3%** |
| `[128, 4096]` | 96 | 6.67 | 6.70 | 1.00× same | Q + O + FFN down | 32.3% |
| `[128, 1024]` | 64 | **0.63** | 1.20 | **0.53× faster** | K + V | 2.0% |
| `[1, 128256]` | 2 | 3.23 | 6.60 | **0.49× faster** | lm_head | 0.3% |

All 224 LLaMA linears dispatch `linear_qcs8w_tiled_buffer_buffer_texture2d_texture2d_half_o4x4`. The 2 lm_head dispatches use `linear_qcs8w_tiled_buffer_buffer_buffer_buffer_half_o4x1` (weights too wide for texture2d → buffer fallback, same as fp16's reason).

**The lopsided per-shape behavior is the real finding:**

- At **large N (14336)** the int8 shader is significantly slower per dispatch (20.2 vs 12.3 ms — 1.65× regression). Hypothesis: at large N, the kernel is compute-bound (~big inner accumulation, scale-load overhead per output column), and weight bandwidth was not the bottleneck in fp16, so dequant adds work without saving anything.
- At **mid N (4096)** the two paths are essentially equal — bandwidth savings exactly cancel dequant overhead.
- At **small N (1024) and M=1 (128256)** int8 wins ~2×. These are the shapes where weight memory bandwidth WAS the bottleneck in fp16, so eliminating half of it is a clean win.

Aggregate: the FFN gate/up contribution is 65% of linear time, so the regression there dominates the average. K/V's 2× win is invisible at the model level because K/V was already only 2% of linear time.

### Time mapped back to model components

| Component | int8 ms | fp16 ms | Lever |
|---|---:|---:|---|
| **FFN linears** (gate + up + down) | ~1422 | ~1004 | 1.42× SLOWER — int8 hurts here |
| **Attention linears** (Q + K + V + O) | ~555 | ~510 | 1.09× — wash |
| `aten.cat` on CPU (KV concat) | 77.4 | 76.7 | unchanged (not a Vulkan op) |
| `ETVK_COPY_OUTPUTS` | ~75 | 75.4 | unchanged |
| Reshape/view + elementwise + bmm + softmax | ~190 | ~190 | unchanged |

## What's next

If the goal is `int8 < fp16 wallclock at S=128 prefill`, on this hardware:

1. **The FFN gate/up shape is the problem.** Fix the int8 shader's behavior at large N (M=128, N=14336). Two angles:
   - Re-tile (the `_o4x4` schedule may not be optimal for the M=128, N=14336 fan-out).
   - Use a coopmat-int8 variant. Pavan-report has `matmul_khr_cm_int8.glsl` (a true `coopmat<uint8_t, ...>` WMMA shader) that's not wired into the linear dispatch. The microbench follow-up (Phase 2 of this study, [`int8_coopmat_microbench.md`](int8_coopmat_microbench.md)) measures it directly to see if it gets us to the ~2× int8/fp16 the user expected.
2. **The wins at small N are real but masked.** A model with smaller K/V or a different head count would already see net int8 wins. LLaMA 3.1 8B's specific shape ratios are just unfavorable for the current int8 shader.
3. **The .pte and Shmem savings are unambiguously valuable** for any deployment that's memory-constrained. The wallclock regression is the price; whether it's worth it depends on the deployment.

## Reproduction

```bash
cd /home/doremy/sarc-acl/executorch/main/executorch
source .venv/bin/activate
sudo swapon /swapfile

# Phase 1: re-export with PT2E vulkan_8w (~5 min, ~26 GB Python RAM peak)
python yanwen/scripts/int8/setup_llama31_int8.py --n_layers 32 --seq_len 128

# Phase 2: scientific bench (~3 min, ~2.1 s/forward → 60 tok/s)
python yanwen/scripts/int8/bench_llama31_int8.py --n_layers 32 --seq_len 128

# Phase 3: ETDump capture (~30 s)
python yanwen/scripts/int8/bench_llama31_int8.py --n_layers 32 --seq_len 128 \
    --num_executions 8 --etdump-analyze
```

If the runner throws `Could not find ShaderInfo with name linear_qcs8w_tiled_*_half_*`, the three upstream patches in this report aren't applied — see `backends/vulkan/runtime/graph/ops/glsl/linear_qcsnw_tiled.{yaml,glsl}` and `backends/vulkan/runtime/graph/ops/impl/QuantizedLinearQCSNW.cpp`.

## Artifacts

Under `yanwen/artifacts/L32_int8/`:

| File | Source | Description |
|---|---|---|
| `S128.etdp` | symlink | Binary ETDump (N=8 capture) |
| `S128.events.tsv` | symlink | Inspector dataframe with per-shader GPU timings |
| `S128.memprobe.tsv` | symlink | `/proc/meminfo` samples during last run |
| `setup_L32_S128_*.log` | copy | Phase 1 export log (W8A16 quantizer + lowering) |
| `bench_L32_S128_*.log` | copy | Scientific-mode bench output (3 reps) |
| `etdump_L32_S128_*.log` | copy | ETDump capture + analyzer outputs |
| `S512.etdp`, `S512.events.tsv`, `S512.memprobe.tsv` | symlink | S=512 ETDump + memprobe |
| `setup_L32_S512_*.log`, `bench_L32_S512_*.log`, `etdump_L32_S512_*.log` | copy | S=512 export + bench + ETDump logs |

## Addendum: int8 at S=512 — int8 unblocks the fp16 memory cliff

**Wallclock:** **25.65 ± 0.30 s** steady forward (cv=1.2%) — **20.0 tok/s**.

This is the second decisive int8 result: where the **fp16 baseline cliffed to ~111 s/forward** at S=512 (88% memory-wait, swap thrashing, MemFree dropped to 226 MB), **int8 W8A16 completes at 25.65 s/forward — 4.33× faster**, with **6.25 GB MemFree headroom** and no thrash regime.

### Comparison vs fp16 at S=512

| Metric | fp16 baseline (cliff) | int8 W8A16 | int8 / fp16 |
|---|---:|---:|---:|
| **Wallclock per forward** | ~111 s | **25.65 s** | **0.23× (4.33× faster)** |
| ETDump total (GPU-active) | 13.7 s | 15.74 s | 1.15× (int8 dispatches slightly slower) |
| Memory-wait (wallclock − ETDump) | ~97 s (88% of wallclock) | ~10 s (38% of wallclock) | **0.10× (10× less wait)** |
| peak Shmem (Vulkan GTT) | 9.8–11.5 GB | **3.17 GB** | 0.30× |
| peak Cached (.pte mmap) | 9.5 GB of 16 GB | 6.51 GB of 8.56 GB | similar fraction |
| **min MemFree** | **226 MB** | **6.25 GB** | **28× more headroom** |
| peak Swap used | 12.2 GB | 7.0 GB | 0.57× |

The cliff fp16 hit at S=512 was a working-set problem, not a compute problem. fp16's working set (15 GB weights + ~10 GB Shmem) overflowed RAM, evicting weight pages from the page cache, causing every layer to re-fault from disk through swap. The GPU sat idle 88% of the time waiting on host memory.

**int8 halves the weights side of the working set** (8.56 GB on disk vs 16 GB) and the Shmem is also smaller (3.17 GB vs 10+ GB; activation+staging intermediates fit better with the int8 weight tensors as buffer-storage rather than texture3d). Net working set drops from ~26 GB to ~12-15 GB — comfortably inside the 28.9 GB RAM. No re-faulting, no thrashing. The GPU is actually running compute most of the time.

### Linear-by-shape at S=512 (int8)

```
python yanwen/scripts/linear_by_shape.py yanwen/artifacts/L32_int8/S512.etdp
```

| Output shape | # disp | sum ms | avg ms | scaling vs S=128 |
|---|---:|---:|---:|---:|
| `[512, 14336]` FFN gate/up | 64 | 5917.0 | **92.45** | 4.57× for 4× seq (slightly super-linear) |
| `[512, 4096]` Q/O/FFN-down | 96 | 3966.6 | **41.32** | 6.20× for 4× seq (super-linear) |
| `[512, 1024]` K/V | 64 | 235.7 | **3.68** | 5.84× for 4× seq |
| `[1, 128256]` lm_head | 2 | 6.4 | 3.21 | unchanged (seq-independent) |

The super-linear scaling at mid-N shapes (Q/O/FFN-down) suggests int8 still pays some cache-miss penalty at S=512 — but it's far less severe than fp16's 60× wallclock blow-up.

### Category breakdown (int8 S=512)

| Category | Time | Share | vs S=128 |
|---|---:|---:|---|
| linear | 10125.8 ms | 64.3% | 5.1× growth for 4× seq |
| **CPU↔GPU copy** | 3512.0 ms | **22.3%** | **39× growth** (ETVK_COPY_INPUTS = 2019 ms, ETVK_COPY_OUTPUTS = 1493 ms) |
| reshape / view | 622.0 ms | 4.0% | 5.9× growth |
| elementwise | 481.4 ms | 3.1% | 8.1× growth |
| bmm / matmul | 313.0 ms | 2.0% | 21.9× growth (attention BMMs grow O(seq²)) |
| CPU fallback (cat) | 294.8 ms | 1.9% | 3.8× growth |
| softmax | 247.7 ms | 1.6% | 37× growth |
| other / fallbacks | ~150 ms | ~1.0% | ~10× growth |
| **TOTAL** | **15744.1 ms** | | (vs 25650 ms wallclock = 61% GPU-active) |

The CPU↔GPU copy category jumps from 3.8% at S=128 to **22.3%** at S=512 — `ETVK_COPY_INPUTS`/`OUTPUTS` are paying for staging buffer round-trips on larger tensors. Same pattern fp16 showed but milder (fp16's S=512 cliff put `ETVK_COPY_INPUTS` at 3.4 s; int8 sees 2.0 s).

### Headline takeaway

**int8 makes S=512 runnable at usable wallclock, and pushes the cliff onset out from somewhere ≤S=128 (fp16) to somewhere between S=256 and S=512 (int8).**

| Config | Status | Forward | tok/s | Memory-wait fraction |
|---|---|---:|---:|---:|
| fp16 S=512 | ◐ runs but full cliff | ~111 s | 4.6 | 88% |
| **int8 S=128** | ✓ clean | 2.108 s | 60.7 | ~0% |
| **int8 S=256** | ✓ **clean** (no cliff) | **4.597 s** | **55.7** | **~0%** |
| **int8 S=512** | ◐ partial cliff | 25.65 s | 20.0 | **39%** |
| fp16 S=1024 | ✗ OOM-killed | — | — | — |
| int8 S=1024 | TBD (Shmem at S=512 was 3.17 GB; naive 4× scaling for 2× seq would give ~12 GB — likely fits within 24 GB GTT cap, but tight; worth a separate experiment) | | | |

### S=256 confirmation: the cliff onset is between S=256 and S=512

At S=256, int8 stays fully in the clean regime: **4.597 ± 0.088 s/forward, 55.7 tok/s**, with min MemFree at **9.47 GB** (more headroom than even S=128). ETDump sums to 5.55 s, *greater* than wallclock — the normal GPU-dispatch pipelining signature where overlap on the timeline makes per-shader timestamps over-sum slightly. No memory-wait gap. Per-shape scaling is well-behaved (FFN gate/up: 20.24 → 43.76 ms for 2× seq, ~2.16× linear).

So the empirical map of int8's regime at L=32:

```
S=128  →  clean (60.7 tok/s)
S=256  →  clean (55.7 tok/s)
S=512  →  partial cliff (20.0 tok/s, 39% memory-wait)
S=1024 →  unknown (Shmem extrapolation suggests it might fit, but untested)
```

The cliff for int8 is shallower at onset than fp16's (39% vs 88% memory-wait at the same shape), because the smaller working set means staging-buffer page faults are less catastrophic — but it's the same mechanism.

So int8's memory benefit isn't an afterthought — at S=512 it's the *whole story*. The 4× int8/fp16 speedup at S=512 isn't from the int8 shader being faster than fp16 (it's actually 15% slower at the dispatch level), but from the lower working set unblocking the page-cache regime entirely.

This complements the S=128 result cleanly:
- **At S=128**, fp16 was already fitting in RAM, so int8's memory benefit didn't show up as wallclock — only the per-dispatch regression did. int8 was 19% slower.
- **At S=512**, fp16's working set exceeded RAM and tipped into thrashing. int8's halved weights brought the working set back in budget. int8 won by 4.33×.

The cross-over point is wherever the fp16 working set first exceeds RAM. On this 28.9 GB box that happens between S=128 and S=512.
