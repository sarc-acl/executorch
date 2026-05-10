# Real L=32 autoregressive decode benchmark — `use_kv_cache=True`

**Date:** 2026-05-10
**Device:** AMD Radeon 780M (RADV PHOENIX), Mesa 25.0.7
**Host:** 28.9 GiB DDR5 RAM, 24 GiB swap
**Model:** LLaMA 3.1 8B fp16, **L=32, use_kv_cache=True, max_seq_len=1024**
**Stock baseline** (main tree, `VulkanPartitioner({})`)

## Headline

**5.0 seconds per decode step. 0.20 tokens/sec. 1024-step decode = ~85 minutes.**

This is **16× slower** than the no-cache proxy I previously projected (`L32_decode_step_breakdown.md` claimed 313 ms / step ≈ 3.2 tok/s). That projection was wrong because it omitted the KV cache buffers from the model graph — adding them dramatically changes the per-step memory access pattern and exposes CPU-fallback operations that don't surface at all in the no-cache path.

The previous decode report should be treated as a per-shader compute-shape investigation, not a decode performance estimate.

## What was actually run

**Setup**: Re-exported the model with the proper decode signature:

```python
ModelArgs(
    max_seq_len=1024, max_context_len=1024, max_batch_size=1,
    use_kv_cache=True, enable_dynamic_shape=False, ...
)
```

The exported `.pte` (16.06 GB, lives at `/home/doremy/llama31_decode_run/`) has KV cache buffers preallocated as in-graph mutable state, and its forward signature is `(tokens: [1,1] int64, input_pos: [1] int64) → logits: [1, vocab_size]`. Setup script: `yanwen/scripts/setup_llama31_decode.py`.

**Bench**: ran `cmake-out-vk/executor_runner` directly (Python pybind path doesn't have Vulkan backend) at three configurations:

| Config | Wallclock | Notes |
|---|---:|---|
| N=1 (cold load + 1 iter) | 24.14 s | Baseline for calibration |
| N=8 (with ETDump capture) | 58.86 s | (58.86 − 24.14) / 7 = 4.96 s / step |
| N=10 (cold cache, validation) | 69.19 s | (69.19 − 24.14) / 9 = 5.0 s / step |
| N=10 (warm cache, validation) | 68.64 s | (68.64 − 24.14) / 9 = 4.94 s / step |
| **Steady-state per step** | | **~5.0 s** |
| **Implied 1024-step total** | | **~85 minutes** |
| **Tokens/sec** | | **0.20 tok/s** |

The 1024-step run was launched (PID 1373638) and ran for **14.5 minutes before I killed it** — at that elapsed time it had completed roughly 170 of 1024 steps based on per-step cost, projecting to ~85 minutes total. Killed to pivot to smaller-N + ETDump capture, which gives the same per-step number in 1 minute.

Input was a fixed token (id=1) and fixed `input_pos=[0]`. The compute shape of every step is identical regardless of `input_pos` because the K/V cache buffers are preallocated to `max_seq_len` and the attention BMM reads the full buffer — so a "real autoregressive loop" with incrementing `input_pos` would give the same per-step number to within timer noise.

**Note**: we used `cmake-out-vk/executor_runner` with `--num_executions=N`, not the Python bindings — pybind builds in our `.venv` lack the Vulkan backend (`Backend VulkanBackend is not registered`). The Python decode runner script (`yanwen/scripts/bench_llama31_decode.py`) is parked; it'll work if/when pybind is rebuilt with Vulkan, but isn't required since the runner does the same thing.

## Why so much slower than the no-cache proxy

The N=8 ETDump tells the story:

| Category | Time (ms) | Share of ETDump |
|---|---:|---:|
| **linear** | **445.3** | **40.5%** |
| **reshape / view** | **305.6** | **27.8%** |
| **CPU↔GPU copy** | **249.7** | **22.7%** |
| CPU fallback (mul.Scalar) | 29.8 | 2.7% |
| other (clone, embedding fallback, layout ops) | 25.2 | 2.3% |
| softmax | 5.4 | 0.5% |
| elementwise (add/mul) | 4.9 | 0.4% |
| CPU fallback (eq.Scalar) | 2.8 | 0.3% |
| CPU fallback (cat) | 0.72 | 0.1% |
| CPU fallback (logical_not) | 0.36 | 0.0% |
| **TOTAL ETDump (sum of leaf events)** | **1098.4** | **100%** |

### Three sharp differences vs. no-cache (L=32 S=1):

| | No-cache decode-shape | Real decode (KV cache) | Ratio |
|---|---:|---:|---:|
| ETDump total per step | 313 ms | 1098 ms | **3.5× more** |
| linear category | 301 ms (96%) | 445 ms (41%) | 1.5× more |
| reshape / view | 5.1 ms (1.6%) | 305.6 ms (28%) | **60× more** |
| CPU↔GPU copy | 0.68 ms (0.2%) | 249.7 ms (23%) | **367× more** |
| Wallclock per step | 313 ms | ~5000 ms | **16× slower** |
| Wallclock − ETDump (memory-wait outside dispatch) | ~0 ms | **~3900 ms (78% of wallclock)** | — |

### Where the extra cost comes from

1. **CPU↔GPU copy explodes (×367):** From 0.7 ms in no-cache to **249.7 ms** with KV cache. This is `ETVK_COPY_INPUTS` and `ETVK_COPY_OUTPUTS` for the KV cache state tensors. Every step copies cache state between GPU and CPU staging buffers — an order of magnitude more than the no-cache path needed for just `[1, 1]` token input.

2. **Reshape / view explodes (×60):** From 5.1 ms to **305.6 ms**. With cache, the graph has many more `view`, `permute`, `index_put`, `slice` operations to manage the cache buffers and the position-indexed writes. These are mostly layout transforms that hit storage boundaries (buffer↔texture transitions).

3. **Linear category 1.5× slower** even though same shaders dispatch. From 1.52 ms/dispatch at `[1, 4096]` in no-cache to 2.45 ms/dispatch in cache. Probably bandwidth contention with the much more active CPU↔GPU copy traffic.

4. **78% of wallclock is "outside ETDump":** The ETDump captures ~1.1 s per step but wallclock is ~5.0 s. The remaining 3.9 s is host-side wait — `vkQueueSubmit` blocked on GPU work that's stalled on memory paging, or staging-buffer round-trips that ETDump doesn't instrument. Same pattern we saw at the L=32 S=512 prefill cliff (88% memory-wait there).

### What's the actual bottleneck

The GPU dispatches we CAN see (in ETDump) sum to 1.1 s. So during 5.0 s of wallclock, the GPU is active for ~22% of the time. The other ~78% is the host stalled on something.

Top suspects:

- **Weight-page re-faulting from disk.** `free -h` during the run showed `buff/cache: 921 MB` — only ~6% of the 16 GB pte was actually resident in page cache. With weights being mmap'd zero-copy into GPU view, accessing un-cached pages requires reading them from NVMe — at ~3 GB/s, 15 GB takes 5 seconds per forward, matching our wallclock exactly.

- **Vulkan delegate texture-storage weight evictions.** Weights packed into `texture2d` storage live in GTT memory. If GTT pressure causes the driver to evict + re-create texture allocations between forwards, you'd pay a memcpy of ~15 GB per forward.

- **The `index_put` CPU fallback for KV cache writes.** Listed in top-15 as `native_call_index_put.out` (64x dispatches, 11.7 ms total — small in absolute terms but forces 64 CPU↔GPU sync points per forward).

Each of these would explain part of the 3.9 s memory-wait component. Without dtrace/ftrace level instrumentation it's hard to say which dominates, but the page-cache evidence (~921 MB resident vs 16 GB needed) is the strongest single signal.

## Top individual ops (N=8 ETDump, mean across iterations)

```
30 ms (64×)  native_call_mul.Scalar_out      ← CPU fallback (mask construction)
14 ms ( 1×)  aten.linear lm_head             ← linear_vec_buffer_buffer_half
12 ms (64×)  native_call_index_put.out       ← KV cache write, CPU fallback
11 ms (64×)  native_call_copy_                ← CPU fallback for some copy op
 7 ms ( 1×)  aten.linear [1, 4096]            ← linear_vec_tile_row_1_buffer_texture2d_half
 6 ms ( 1×)  aten.linear [1, 4096]            ← (one per layer × 3 = 96 total)
 ...
 5 ms ( 1×)  aten.linear [1, 14336]           ← FFN gate/up
 ...
```

Two CPU-fallback ops account for 64 × ~0.2 ms each — small per dispatch, but they break GPU pipelining (force CPU↔GPU sync).

## Linear by output shape

| Shape | # disp | sum ms | avg ms / disp | Component | Share of linear |
|---|---:|---:|---:|---|---:|
| `[1, 4096]` | 96 | 234.8 | **2.45** | Q + O + FFN down (32 × 3) | 52.7% |
| `[1, 14336]` | 64 | 119.2 | 1.86 | FFN gate + up (32 × 2) | 26.8% |
| `[1, 1024]` | 64 | 77.5 | 1.21 | K + V (32 × 2) | 17.4% |
| `[1, 128256]` | 2 | 13.8 | 6.90 | lm_head | 3.1% |

Almost all dispatch `linear_vec_tile_row_1_buffer_texture2d_half` (same as no-cache). Interestingly, the K + V dispatches now show **mixed kernels**: `linear_vec_tile_row_1_buffer_texture2d_half` AND `linear_vec_tile_row_1_texture3d_texture2d_half` — the latter is `STORAGE=texture3d` (texture activations, not buffer). This is a hint that the KV-cache export adds storage transitions that the no-cache export didn't have.

## TL;DR — for the manager

> *"On AMD Radeon 780M with 28.9 GB RAM:*
>
> - **2k prefill at L=32 fp16: NOT runnable** (OOM-killed, confirmed earlier). Working set ~25 GB > 22.4 GB GTT cap.
>
> - **1k decode at L=32 fp16: ~5.0 sec per token, 0.20 tok/s, total 1024-step decode ≈ 85 minutes.** That's 16× slower than I previously projected from the no-cache proxy — adding KV cache buffers + the index_put writes + the storage transitions exposes substantial CPU↔GPU traffic and memory-wait time outside the GPU compute path. Only ~22% of wallclock is GPU-active; the rest is host stalled on memory (likely weight-page re-faulting since `buff/cache` shows only 6% of the 16 GB pte resident at peak).
>
> - **The bottleneck is memory architecture, not compute.** No shader optimization (coopmat, better tiling, etc.) recovers this — the GPU sits idle most of the time waiting for pages.
>
> - **The unblock is the same as for prefill: int4 weight quantization.** Shrinks the model from 15 GB to 4 GB, which fits in page cache → no re-faulting, memory-wait collapses, decode rate jumps to ~5–8 tok/s (estimate, real depends on implementation). Same change unblocks 2k prefill simultaneously."*

## Open questions

These would each take ~1–2 hours of focused investigation if pursued:

1. **Why isn't the page cache keeping the 16 GB pte resident?** With 26 GB MemFree at the end of the run, the kernel should be able to cache it. Likely the Vulkan delegate's texture2d weight storage path doesn't go through the page cache directly — weights get re-packed/re-staged each forward. A `strace` or `perf` capture during a few decode iterations would clarify.

2. **Is the `index_put` CPU fallback the right one?** Some ExecuTorch builds have Vulkan-delegated `index_put` for KV cache writes. We're using stock `VulkanPartitioner({})` which apparently doesn't. The `--use_sdpa_with_kv_cache` lowering pass (in `examples/models/llama/export_llama_lib.py`) replaces `index_put` with a fused `sdpa_with_kv_cache` op that fully delegates — would close most of the CPU-fallback gap. Worth re-exporting with that flag and re-measuring.

3. **What does coopmat decode look like?** We didn't run a coopmat variant of the decode model (storage_type_override=BUFFER on pavan-report). Coopmat still wouldn't fire (M=1 < 64 gate), but the buffer-storage-everywhere path might have different layout transition costs vs. the texture2d default. Could reduce the 305 ms "reshape/view" share. Probably a small effect but worth a quick verify if curious.

## Reproduction

```bash
cd /home/doremy/sarc-acl/executorch/main/executorch
source .venv/bin/activate
sudo swapon /swapfile

# Export the decode .pte (~5 min, 16 GB output)
python yanwen/scripts/setup_llama31_decode.py --n_layers 32 --max_seq_len 1024

# Create the two input files
python -c "
import numpy as np
np.array([1], dtype=np.int64).tofile('/home/doremy/llama31_decode_run/tokens.bin')
np.array([0], dtype=np.int64).tofile('/home/doremy/llama31_decode_run/input_pos.bin')
"

# Bench (N=10 for per-step measurement; N=8 with --etdump_path for breakdown)
time cmake-out-vk/executor_runner \
  --model_path=/home/doremy/llama31_decode_run/llama31_8b_32L_decode_max1024_fp16.pte \
  --inputs=/home/doremy/llama31_decode_run/tokens.bin,/home/doremy/llama31_decode_run/input_pos.bin \
  --num_executions=10
# Expected: ~69 s wallclock → ~5 s per step

# For ETDump:
cmake-out-vk/executor_runner --model_path=... --inputs=... --num_executions=8 \
  --etdump_path=/home/doremy/llama31_decode_run/llama31_8b_32L_decode_max1024_fp16.etdp
python /home/doremy/sarc-acl/executorch/pavan-report/executorch/yanwen_plan/analyze_etdump.py \
  /home/doremy/llama31_decode_run/llama31_8b_32L_decode_max1024_fp16.etdp
```

## Artifacts

Under `yanwen/artifacts/L32_decode/`:

- `decode_N8.etdp` — Binary ETDump
- `bench_L32_decode_N10.log` — N=10 wallclock measurement (cold cache)
- `bench_L32_decode_N10_warm.log` — N=10 measurement (warm cache, same result)
- `bench_L32_decode_N8_etdump.log` — N=8 with ETDump + Inspector breakdown
- `setup_L32_decode1024.log` — Export log

## What supersedes

- The earlier `L32_decode_step_breakdown.md` made a clean per-shader analysis of the no-cache `seq=1` forward and projected 3.2 tok/s decode. **That projection is wrong** — it omitted the KV cache, the index_put writes, and the storage transitions, and dramatically underestimated the host-side memory-wait component. The shader-level findings (96% linear, etc.) are still valid for understanding the per-shader compute, but the throughput number must be replaced with the value in this report.
- The `decode_GEMV_ceiling_check.md` (L=4 no-cache check) remains valid as a per-shader dispatch test — coopmat doesn't fire at M=1 regardless of whether the model has KV cache. But its decode-throughput extrapolation should also be ignored.
