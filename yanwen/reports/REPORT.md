# Pure LLaMA 3.1 8B prefill on AMD 780M iGPU — L=32 findings

**Updated:** 2026-05-10 · **Author:** Yanwen Xu · **Branch:** `main` · **Scope:** L=32 only

**Device:** AMD Radeon 780M (RADV PHOENIX, RDNA3+ mobile iGPU), Mesa 25.0.7
**Host:** 28.9 GiB DDR5 RAM, 24 GiB swap (`/swapfile`), 868 GiB `/home`
**Model:** stock LLaMA 3.1 8B fp16, no coopmat, no Stephen's shader, `VulkanPartitioner({})`
**Mode:** prefill only (single forward over `[1, seq_len]` input, no autoregressive decode)

## TL;DR

| Question | Answer |
|---|---|
| **L=32 S=128 forward time** | **1.77 s ± 6 ms** (cv 0.3%, 3 reps × N=8, calibration-subtracted) |
| **Throughput** | **72.5 tokens/sec prefill** at S=128 |
| **Memory at S=128** | peak Shmem 0.9 GB, MemFree 6.2 GB headroom — comfortable |
| **Where time goes** | **80% in linears** (all on tiled `linear_vec`, not coopmat); **57% FFN, 29% attention proj** |
| **Largest seq that runs** | S=512 completes but is unusable (~95 s/forward, swap thrashing) |
| **Hard OOM boundary** | S ≥ 1024 — pure memory-architecture limit, not shader-related |
| **Biggest optimization lever** | switch linears to `linear_coopmat` — **measured 3.03× whole-forward speedup → 219.7 tok/s prefill**. See [`L32_S128_coopmat_REPORT.md`](L32_S128_coopmat_REPORT.md). |

## Methodology

### Benchmark — algebraic-subtraction across subprocesses

`wallclock/N` from a single subprocess overstates steady-state forward time, because every subprocess pays a fixed `(load + iter 0 + teardown)` cost. At L=32 this fixed cost is ~21 s; even at N=16 it inflates the reported forward by ~1.3 s (~80% over the true number).

The fix:

```
W1 = wallclock at N=1   = load + iter 0 + teardown            (calibration)
WK = wallclock at N=K   = load + iter 0 + (K-1)·steady + tear (measurement)
steady_forward = (WK - W1) / (K - 1)
```

Default: 1 calibration subprocess at N=1, then 3 measurement subprocesses at N=8 each. Mean ± stdev across the 3 reps quantifies across-run variance. Implementation in `scripts/run_llama31_pure.py:bench_steady_state()`.

### Why not ETDump for the headline number

ETDump's per-iter `Method::execute` event reports CPU-side dispatch time only — the GPU work is async after `vkQueueSubmit`. For wrapper events the value diverges from wallclock (e.g. `Method::execute` mean = 1970 ms, close to true 1766 ms steady-state, but the underlying dispatch returns much faster than GPU completes; the closeness is coincidental for prefill).

ETDump **is** authoritative for **per-shader GPU time**: the Vulkan delegate writes `vkCmdWriteTimestamp` query-pool entries around each compute pipeline (gated on `ET_EVENT_TRACER_ENABLED`), and those land in `events.tsv` as kernel-name leaf events. We use those for the per-shader breakdown below.

## Headline result: L=32 S=128

```
W1 (load + iter 0 + teardown):   21.41 s
rep 1 steady:                     1.770 s
rep 2 steady:                     1.768 s
rep 3 steady:                     1.759 s

Steady-state forward:  1.766 ± 0.006 s   (cv = 0.3%)
Prefill throughput:    72.5 tokens/sec
```

Run command:

```bash
python yanwen/scripts/bench_llama31_pure.py --n_layers 32 --seq_len 128
```

Log: `yanwen/artifacts/L32/S128_bench.log`.

## Memory at L=32 S=128

| Quantity | Peak | Notes |
|---|---:|---|
| Shmem (Vulkan GTT-backed allocations) | **0.9 GB** | Activations + staging + workspace |
| Cached (.pte mmap pages resident) | 9.5 GB / 16 GB | Only touched pages cached |
| min MemFree during run | 6.2 GB | Healthy headroom |
| Swap delta during run | ~0 MB | No paging |
| GTT cap (`RADV_GTT_PCT=80`) | 24.8 GB | Hard ceiling |

L=32 S=128 is comfortably within budget. Working set ≈ 16 GB (weights + Shmem) of 22.4 GB GTT cap and 28.9 GB RAM.

## ETDump shader breakdown (L=32 S=128)

Generated via the canonical analyzer:

```bash
python /home/doremy/sarc-acl/executorch/pavan-report/executorch/yanwen_plan/analyze_etdump.py \
    yanwen/artifacts/L32/S128.etdp
```

### Categories

ETDump total = 1908 ms; matches wallclock-measured forward 1766 ms within timer noise (8% gap from CPU-fallback ops + measurement granularity).

| Category | Time | Share | # disp | Notes |
|---|---:|---:|---:|---|
| **linear** | **1527.6 ms** | **80.1%** | 226 | Q/K/V/O + FFN gate/up/down × 32 + lm_head |
| reshape / view | 106.7 ms | 5.6% | 2950 | `view_*`, `permute`, `slice`, `view_convert_*` |
| **CPU↔GPU copy** | 91.4 ms | 4.8% | 194 | `ETVK_COPY_OUTPUTS` is **75.4 ms by itself** |
| **CPU fallback (`aten.cat`)** | 76.7 ms | 4.0% | 64 | KV-cache concat — not delegated to Vulkan |
| elementwise | 59.0 ms | 3.1% | 1219 | residual add, mul, sigmoid, where |
| bmm / matmul | 14.5 ms | 0.8% | 64 | Attention `Q@Kᵀ` and `attn@V` — tiny at S=128 |
| Other CPU fallbacks | ~20 ms | 1.0% | ~193 | mask construction (`eq.Scalar`, `mul.Scalar`, `logical_not`, `scalar_tensor`), embedding |
| softmax | 6.9 ms | 0.4% | 32 | one per attention block |

### Linear dispatches by output shape

| Output shape | # disp | sum ms | avg ms | Component | Share of linear |
|---|---:|---:|---:|---|---:|
| `[128, 14336]` | 64 | **789.4** | 12.3 | FFN gate + up (`32 × 2`) | **51.7%** |
| `[128, 4096]` | 96 | 646.7 | 6.7 | Q + O + FFN down (`32 × 3`) | 42.3% |
| `[128, 1024]` | 64 | 78.3 | 1.2 | K + V (`n_kv_heads=8 × head_dim=128`, `32 × 2`) | 5.1% |
| `[1, 128256]` | 2 | 13.2 | 6.6 | lm_head | 0.9% |

**All linears use `linear_vec_buffer_texture2d_half`** — the GEMV-style (non-coopmat) shader from
`backends/vulkan/runtime/graph/ops/glsl/linear_vec.glsl`. Variant: weights as `texture2d`, activations
as `buffer`, fp16. Full per-GLSL-shader breakdown in
[`L32_S128_shader_breakdown.md`](L32_S128_shader_breakdown.md).

### Time mapped back to model components

| Component | ms | Share of forward | Optimization lever |
|---|---:|---:|---|
| **FFN linears** (gate + up + down) | ~1004 | **57%** | `linear_coopmat` shader → ~halved (per 2026-05-06 synthetic data) |
| **Attention linears** (Q + K + V + O) | ~510 | 29% | Same lever |
| `aten.cat` on CPU (KV concat) | 76.7 | 4.3% | Add Vulkan `cat` op or restructure KV cache as preallocated tensor |
| `ETVK_COPY_OUTPUTS` | 75.4 | 4.3% | 256 KB output in 75 ms ≈ 3.4 GB/s — far below DDR5 BW; investigate staging path |
| Reshape/view + elementwise + bmm + softmax + remaining | ~190 | ~10% | Already small |

## Safe seq_len at L=32 — empirical map

| seq | wallclock / forward | ETDump GPU time | "memory stall" | peak Shmem | min MemFree | peak Swap | Status |
|---:|---:|---:|---:|---:|---:|---:|---|
| **128** | **1.77 s** | 1.91 s | ~0 (timer noise) | 0.9 GB | 6.2 GB | baseline | ✓ **performant** |
| **512** | **~111 s** | **13.7 s** | **~97 s (88%)** | 9.8–11.5 GB (run-to-run) | 0.2 GB | 12.2 GB | ◐ completes but **~60× cliff**; mostly memory-fault stalls |
| 1024 | (OOM-killed) | — | — | 13.2 GB at OOM | 0.2 GB | 25.8 GB (saturated 24 GB) | ✗ OOM during calibration |
| 2048 | (OOM-killed) | — | — | — | — | — | ✗ 5/5 prior attempts (2026-05-06) too |

Per-layer Shmem scaling (super-linear in seq):

```
S=128:    917 MB / 32 ≈  29 MB/layer
S=512: 9852–11507 MB / 32 ≈ 308–360 MB/layer  →  ~12× growth for 4× seq
S=1024: 13166 MB / 32 ≈ 411 MB/layer (at OOM-time, partial)
```

### Why S=512 already cliffs — quantified

The S=512 ETDump confirms the cliff is **not compute-bound**:

```
GPU dispatches active (ETDump sum):    13.7 s
Wallclock per forward:                 111   s
"Memory-wait" (wallclock − ETDump):    ~97   s   ≈ 88% of wallclock
```

Per-shader compute scales roughly linearly with seq (e.g. `[seq, 14336]` linear avg: 12.3 ms at S=128 → 46.6 ms at S=512 = 3.8× for 4× seq). That's normal GPU compute behavior, **not** thrashing. The 60× wallclock blow-up vs. the 4× seq-ratio comes entirely from the 88% memory-wait component — page faults blocking buffer binding and `vkCmdCopy*` operations.

Working set at S=512:

```
15 GB weights (mmap'd, ~10 GB resident at peak Cached) +
9.8–11.5 GB Shmem (Vulkan intermediates) + ~1 GB other ≈ 26–28 GB
```

…against **28.9 GB system RAM**. With MemFree dropping to 226 MB, the page cache evicts weight pages, every layer re-faults from disk through swap. The shader is fine — the box can't fit the working set.

A telltale: at S=128 `ETVK_COPY_INPUTS` is 16 ms total. At S=512 it balloons to **3420 ms** — for an input tensor that's only 4 KB. The GPU isn't slow at copying; it's stalling on `vkCmdCopyBuffer` while the staging buffer page is faulted back in.

### Why this is hardware-bound, not shader-bound

A 2× faster linear shader would still fit the same intermediates and still page when working set exceeds RAM. The cliff is independent of any shader optimization. The memory-wait component is **not visible to ETDump** because GPU timestamps only span the time the GPU is actively executing — not the time the host is paging memory in.

To run L=32 at seq ≥ 512 *usably* on this hardware, the move is to **reduce working-set size**, not speed up shaders:

- **Weight-only int4 quantization**: 15 GB → ~4 GB. Largest single lever.
- **KV-cache offload** to disk/streaming: smaller benefit (KV is only ~256 MB at S=2048).
- **Per-layer activation streaming**: would require partitioner / runtime changes.

All three are out of scope for "pure original" benchmarking — but they're the only paths to running larger contexts on this iGPU.

## ETDump shader breakdown (L=32 S=512, for the curious)

Captured at N=3 in legacy mode (~5 min). The shape of the breakdown changes dramatically vs. S=128:

| Category | Time | Share at S=512 | Share at S=128 |
|---|---:|---:|---:|
| linear | 5904.9 ms | 43.2% | 80.1% |
| **CPU↔GPU copy** | **5371.4 ms** | **39.3%** | 4.8% |
| reshape / view | 718.5 ms | 5.3% | 5.6% |
| elementwise | 531.1 ms | 3.9% | 3.1% |
| bmm / matmul | 338.5 ms | 2.5% | 0.8% |
| CPU fallback (cat) | 297.2 ms | 2.2% | 4.0% |
| softmax | 269.3 ms | 2.0% | 0.4% |
| CPU fallback (embedding) | 93.7 ms | 0.7% | 0.0% |
| Other CPU fallbacks | 130 ms | 0.9% | 1.0% |
| **TOTAL** | **13675 ms** | (vs 111 000 ms wallclock) | (vs 1766 ms wallclock) |

The big mover: `ETVK_COPY_INPUTS` jumps from 16 ms (S=128) to **3420 ms** (S=512), and `ETVK_COPY_OUTPUTS` from 75 ms to 1951 ms. These are paying for memory-fault cost on staging buffers that get evicted under pressure.

Linear by shape at S=512:

| Output shape | # disp | sum ms | avg ms | Component | Share of linear |
|---|---:|---:|---:|---|---:|
| `[512, 14336]` | 64 | 2984.6 | 46.6 | FFN gate + up | 50.5% |
| `[512, 4096]` | 96 | 2549.1 | 26.6 | Q + O + FFN down | 43.2% |
| `[512, 1024]` | 64 | 357.0 | 5.6 | K + V | 6.0% |
| `[1, 128256]` | 2 | 14.3 | 7.1 | lm_head | 0.2% |

Per-shape avg scales ~linearly with seq (4× growth for 4× seq), as expected for GEMV-style shaders. **The linear shaders themselves are fine.** It's everything around them that's stalling on swap.

## Optimization recommendations (ordered)

1. **Switch linears to `linear_coopmat` shader.** 2× speedup measured on 2026-05-06 synthetic 4-layer LLaMA at S=2048 (`linear_vec` 4057 ms → `linear_coopmat` 2041 ms). Applied here, 1528 ms of linear time would drop to ~760 ms — total forward ~1140 ms, throughput **~115 tok/s** vs. current 72. The full `coopmat_previous_work.diff` is in `pavan-report/executorch/yanwen_plan/`.
2. **Move `aten.cat` (KV concat) to Vulkan** or restructure KV cache as a preallocated buffer with index writes. Recovers ~76 ms (4%) and removes a CPU↔GPU sync per layer.
3. **Investigate `ETVK_COPY_OUTPUTS` cost**: 75 ms for a 256 KB output on UMA is ~25× below DDR5 bandwidth. Likely a host-visible staging round-trip rather than a direct map. Recovers up to ~75 ms (4%).
4. **Don't optimize attention BMMs / softmax at S=128.** Combined <2%. Will matter at higher seq, but the box can't run higher seq anyway.

## Reproduction

See `INSTRUCTIONS.md`. Headline command for L=32 S=128:

```bash
cd /home/doremy/sarc-acl/executorch/main/executorch
source .venv/bin/activate
python yanwen/scripts/bench_llama31_pure.py --n_layers 32 --seq_len 128
```

ETDump capture:

```bash
python yanwen/scripts/bench_llama31_pure.py --n_layers 32 --seq_len 128 \
    --num_executions 8 --etdump-analyze
```

## Artifacts

Under `yanwen/artifacts/L32/`:

| File | Source | Description |
|---|---|---|
| `S128.etdp` | symlink | Binary ETDump (N=8 capture) |
| `S128.events.tsv` | symlink | Inspector dataframe with per-shader GPU timings |
| `S128.memprobe.tsv` | symlink | `/proc/meminfo` samples during last run |
| `S128_bench.log` | copy | Scientific-mode bench output (3 reps) |
| `S512.memprobe.tsv` | symlink | Memprobe from S=512 cliff run |
| `S512_legacy.log` | copy | N=2 wallclock run (~95 s/forward) |
| `S512.etdp` | symlink | Binary ETDump (N=3, cliff regime, 111 s/forward) |
| `S512.events.tsv` | symlink | Inspector dataframe |
| `S512_etdump.log` | copy | ETDump bench output with categories |
| `S1024.memprobe.tsv` | symlink | Truncated probe — OOM-killed during calibration |
| `S1024_oom.log` | copy | Bench log showing rc=-9 |
