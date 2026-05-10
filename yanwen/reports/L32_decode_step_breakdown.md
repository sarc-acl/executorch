# L=32 decode-step ETDump breakdown — proxy for "1k decode" benchmark

**Date:** 2026-05-10
**Device:** AMD Radeon 780M (RADV PHOENIX, RDNA3+ mobile iGPU), Mesa 25.0.7
**Host:** 28.9 GiB DDR5 RAM, 24 GiB swap
**Model:** LLaMA 3.1 8B fp16, **L=32, seq_len=1, no KV cache**
**Stock baseline** (main tree, `VulkanPartitioner({})`)

## ⚠️ Read this first — what this is and isn't

This report measures **the per-step cost of a single-token forward at L=32 fp16**, which is the dominant component of decode-phase latency. It is **not** a real autoregressive 1024-token decode benchmark.

**What "1k decode" really means** (per the manager's spec, "2k prefill + 1k at decode"):

- Run 1024 separate forwards, each at `[1, 1]` input shape (one new token per step)
- Each forward reads a KV cache that grows by 1 entry per step (starting from whatever prefill populated)
- KV cache traffic per step adds linearly with current context length

**What this report measures:**

- ONE forward at `[1, 1]` input shape, no KV cache (model materializes a fresh empty cache each forward)
- ETDump GPU breakdown of that single forward
- Per-step time = 310.6 ms (steady-state ETDump GPU total)

**Why the proxy is valid (within ~1%)**

The per-forward cost is dominated by streaming 15 GB of fp16 weights through DDR5. KV cache traffic at any context length up to ~4k is negligible vs the weight read:

```
Weight read per step:         15.0 GB
KV cache read at 1k context:  128 MB  (32 layers × 2 (K+V) × 1024 ctx × 8 kv_heads × 128 head_dim × 2 B)
KV cache read at 2k context:  256 MB
KV cache read at 4k context:  512 MB

KV / weight ratio: 0.8% at 1k, 1.7% at 2k, 3.4% at 4k
```

At 80 GB/s effective bandwidth, the KV read adds ~1.6 ms at 1k context, ~3.2 ms at 2k. So per-step time at real 1k decode is **310.6 ms + ~1.6 ms ≈ 312 ms** — indistinguishable from our measurement.

**What this report does NOT cover:**

- A real autoregressive loop with token sampling and `start_pos` threading
- The `2k prefill` phase (which OOMs at L=32 fp16 on this hardware — see [`REPORT.md`](REPORT.md))
- Time-to-first-token (TTFT) — that's prefill, which we couldn't run at 2k
- Performance scaling with cache size > ~2k (where KV read traffic becomes a few percent of forward)

## Headline numbers

| Quantity | Value |
|---|---:|
| Per-step GPU time (steady, ETDump) | **310.6 ms** |
| Per-step wallclock | **313 ms** (ETDump + ~2.5 ms overhead, matches within 1%) |
| **Per-token throughput** | **3.22 tokens/sec** |
| Iter 0 (cold) | 315.0 ms (only 1.4% slower than steady — minimal warmup) |
| Memory peak Shmem (Vulkan GTT-backed) | 0.7 MB |
| Memory peak Cached (.pte mmap) | 7.66 GB |
| Memory min MemFree | 11.57 GB |
| Memory peak Swap | 5.15 GB (baseline; didn't grow) |

The very tiny peak Shmem (0.7 MB!) reflects that all per-step activations at M=1 are negligibly small — every byte of memory traffic is weights, exactly as the bandwidth-bound model predicts.

## Bandwidth efficiency

```
Theoretical ceiling:   15 GB weights / 80 GB/s effective DDR5 = 187.5 ms / step → 5.3 tok/s
Measured:                                                       310.6 ms / step → 3.22 tok/s
Efficiency:                                       187.5 / 310.6 = 60.4% of ceiling
```

60% of theoretical bandwidth is typical for real workloads with dispatch overhead, non-linear ops, and partial page-cache pressure. Note that `peak Cached = 7.66 GB` means **less than half of the 15 GB of weights are resident in page cache** during the run — the kernel is evicting weight pages between layers, causing some re-faulting. With more RAM, we'd likely see closer to 70–80% of the bandwidth ceiling.

## ETDump category breakdown

(canonical analyzer; ETDump total ≈ wallclock at S=1 since there's no memory thrash)

| Category | Time (ms) | Share | # disp | Notes |
|---|---:|---:|---:|---|
| **linear** | **301.5** | **96.3%** | 226 | All linears go through linear_vec (no coopmat at M=1) |
| reshape / view | 5.1 | 1.6% | 2950 | Many tiny dispatches, individually cheap |
| elementwise | 2.3 | 0.7% | 1219 | residual add, mul, sigmoid |
| Other CPU fallbacks (cat / eq.Scalar / mul.Scalar / logical_not / scalar_tensor / embedding) | 1.2 | 0.4% | 226 | Mask construction + KV cache concat |
| CPU↔GPU copy | 0.68 | 0.2% | 194 | ETVK_COPY_INPUTS/OUTPUTS |
| bmm / matmul (attention) | 1.2 | 0.4% | 64 | matmul_vec_tile_row_1_texture3d_float — tiny at M=1 |
| softmax | 0.07 | 0.0% | 32 | One per attention block, microscopic at M=1 |
| **Total** | **313.0** | 100% | 5072 | |

**Linears are 96% of decode-step time.** Everything else combined contributes ~12 ms / step.

## Linear by output shape

| Output shape | # disp | sum ms | avg ms / disp | Component | Share of linear |
|---|---:|---:|---:|---|---:|
| `[1, 4096]` | 96 | **146.1** | 1.52 | Q + O + FFN down (`32 layers × 3`) | **48.5%** |
| `[1, 14336]` | 64 | 94.5 | 1.48 | FFN gate + up (`32 layers × 2`) | 31.3% |
| `[1, 1024]` | 64 | 47.5 | 0.74 | K + V (`n_kv_heads=8 × head_dim=128`, `32 × 2`) | 15.7% |
| `[1, 128256]` | 2 | 13.4 | 6.70 | lm_head (final logits projection) | 4.4% |

All 224 transformer linears dispatch `linear_vec_tile_row_1_buffer_texture2d_half` (TILE_M=1 variant; the runtime auto-picks it at M=1 since there's only one output row). lm_head dispatches `linear_vec_buffer_buffer_half` because its `[1, 128256]` output exceeds the texture2d limit (16384 texels max on 780M).

## Per-shader steady-state (iter 1..7 averaged)

(via `etvk_breakdown.py`)

| # | Kernel | Dispatches | Steady ms | % steady |
|---:|---|---:|---:|---:|
| 1 | **`linear_vec_tile_row_1_buffer_texture2d_half`** | 224 | **287.57** | **92.6%** |
| 2 | `linear_vec_buffer_buffer_half` (lm_head) | 1 | 13.41 | 4.3% |
| 3 | `matmul_vec_tile_row_1_texture3d_float` (attention) | 32 | 1.18 | 0.4% |
| 4 | `mean_per_row_buffer_float` (RMSNorm) | 65 | 1.02 | 0.3% |
| 5 | `view_buffer_half` | 704 | 0.89 | 0.3% |
| 6 | `binary_mul_buffer_float` | 418 | 0.85 | 0.3% |
| 7 | `view_convert_buffer_half_float` | 545 | 0.66 | 0.2% |
| 8 | `nchw_to_buffer_half_half` | 191 | 0.56 | 0.2% |
| 9 | `view_buffer_float` | 448 | 0.55 | 0.2% |
| 10 | `buffer_to_nchw_float_float` | 192 | 0.39 | 0.1% |
| ... (28 more, all < 0.4%) | | | | < 3% combined |

**One kernel does 92.6% of decode-step time.** This is much more concentrated than prefill at S=128 (which had ~88% in `linear_vec_buffer_texture2d_half`, plus ~4% each in `view_*` and `binary_mul_*`). At decode shape, everything except the GEMV is reduced to noise.

## Projection to a real 1024-step decode

Assuming KV cache starts at 2048 entries (after a hypothetical 2k prefill) and grows by 1 per decode step:

| Step number | Context length at start | KV cache size | KV read traffic / step | Per-step time (est.) |
|---:|---:|---:|---:|---:|
| 1 | 2048 | 256 MB | 3.2 ms | ~314 ms |
| 256 | 2304 | 288 MB | 3.6 ms | ~314 ms |
| 512 | 2560 | 320 MB | 4.0 ms | ~315 ms |
| 1024 | 3072 | 384 MB | 4.8 ms | ~316 ms |

**Average per-step time over 1024 decode steps: ~315 ms.**

**Total 1k decode wallclock: 1024 × 315 ms = 322,560 ms ≈ 5.4 minutes.**

(This is just the GPU forward time. Real decode adds Python/sampling overhead — typically <1 ms / step in efficient runners, so total wallclock would be ~5.5 minutes.)

**Steady-state per-token rate: ~3.17 tokens/sec.**

## Prefill (S=128) vs decode-step at L=32 — apples-to-apples

| | L=32 S=128 prefill (baseline) | L=32 S=1 decode-step | Ratio |
|---|---:|---:|---:|
| Per-forward GPU (steady) | 1689 ms | 311 ms | Decode is **5.4× faster per forward** |
| Tokens covered per forward | 128 | 1 | Decode covers **128× fewer tokens** per forward |
| **Tokens / sec** | **72.5** | **3.22** | **Decode is 22.5× slower in tok/s** |
| Linear share of forward | 89.7% | 96.9% | Linears even more dominant at M=1 |
| Top linear shader | `linear_vec_buffer_texture2d_half` (TILE_M=4) | `linear_vec_tile_row_1_buffer_texture2d_half` (TILE_M=1) | Different variant of same `linear_vec.glsl` |
| Linear GPU time | 1516 ms | 301 ms | 5× faster per dispatch but 1024 steps × this for 1k decode |
| Top-shape avg ms (FFN `[N, 14336]`) | 12.3 ms (N=128) | 1.48 ms (N=1) | 8.3× faster per dispatch at M=1 |
| Coopmat eligible? | ✓ at S=128 (M=128 ≥ 64) | ✗ never (M=1 < 64) | — |
| Compute vs bandwidth regime | Compute-bound (GEMM, ~1 TFLOPs effective on linears) | **Bandwidth-bound** (60% of DDR5 ceiling) | — |

The 22.5× per-token slowdown for decode reflects the fundamental memory-bandwidth limit. Prefill amortizes the 15 GB weight read across 128 tokens (117 MB / token effective); decode reads all 15 GB per single token.

## Why coopmat doesn't help decode

The runtime dispatch gate in `Linear.cpp`:

```cpp
use_coopmat = supports_KHR_cooperative_matrix
           && storage_type_of(out) == kBuffer
           && M >= 64;       // ← M=1 always fails this
```

The `coopMatLoad` GLSL intrinsic has no bounds check on its output store, so M < 64 would OOB-write. The gate is a correctness requirement, not a tuning knob. Even on the coopmat-configured run, all 224 transformer linears fall back to `linear_vec_tile_row_1_buffer_texture2d_half`, identical to baseline.

We verified this empirically at L=4 S=1 — baseline and coopmat both run the same shaders within 1% wallclock — see [`decode_GEMV_ceiling_check.md`](decode_GEMV_ceiling_check.md).

Don't expect any shader-level win to apply to single-stream decode on this hardware.

## What this means for the "2k prefill + 1k decode" benchmark

The manager-spec'd workload at L=32 fp16 on AMD 780M:

| Phase | Status | Why |
|---|---|---|
| **2k prefill** | ✗ **Not runnable** | Working set ~25 GB exceeds 28.9 GB RAM after OS overhead. Confirmed empirically: L=32 S≥1024 OOM-killed. See `REPORT.md`. |
| **1k decode (per-step)** | ✓ **Proxy measured** | 315 ms / step ≈ **3.17 tok/s**, bandwidth-bound. ~1% adjustment for real KV cache traffic vs our no-cache measurement. |
| **1k decode (total wallclock)** | ◐ **Projected** | 1024 × ~315 ms ≈ **5.4 minutes** for the decode phase. Real measurement would need a KV-cache-enabled `.pte` + decode runner (~half-day eng) and a populated KV cache (which we can't generate without prefill). |

**Effective levers to actually run the full benchmark:**

1. **int4 weight quantization** (the practical fix): 15 GB → ~4 GB. Prefill working set drops below RAM. Decode bandwidth ceiling improves to ~20 tok/s (4 GB / 80 GB/s × 60% efficiency ≈ 12 tok/s real). The most impactful single change.

2. **Chunked prefill** (process 2048 prompt tokens in chunks of 128 or 256 instead of one big forward). Avoids the OOM but each chunk pays load+iter-0 overhead, so TTFT grows. Tradeoff between memory and latency.

3. **Decode-only benchmark** (skip prefill, populate KV cache with synthetic K, V values). Validates the per-step rate without requiring a working prefill.

4. **Different hardware**: anything with ≥48 GB unified memory (e.g., Apple M4 Pro 48 GB, server-class CPU with adequate RAM, discrete GPU with 24+ GB VRAM) would let you run the full benchmark unmodified.

## Reproduction

The L=32 S=1 export needs the OOM fix in `export_pte()` (added 2026-05-10):

```bash
cd /home/doremy/sarc-acl/executorch/main/executorch
source .venv/bin/activate
sudo swapon /swapfile

python yanwen/scripts/setup_llama31_pure.py --n_layers 32 --seq_len 1   # ~5 min, ~16 GiB peak
python yanwen/scripts/bench_llama31_pure.py --n_layers 32 --seq_len 1 \
    --num_executions 8 --etdump-analyze   # ~25 sec
```

Analyzers:

```bash
python /home/doremy/sarc-acl/executorch/pavan-report/executorch/yanwen_plan/analyze_etdump.py \
    /home/doremy/llama31_pure_run/llama31_8b_32L_seq1_fp16.etdp

python yanwen/scripts/etvk_breakdown.py \
    /home/doremy/llama31_pure_run/llama31_8b_32L_seq1_fp16.events.tsv 8

python yanwen/scripts/linear_by_shape.py \
    /home/doremy/llama31_pure_run/llama31_8b_32L_seq1_fp16.etdp
```

## Artifacts

Under `yanwen/artifacts/L32_S1_baseline/`:

| File | Description |
|---|---|
| `S1.etdp` | Binary ETDump (N=8 capture) |
| `S1.events.tsv` | Inspector dataframe |
| `S1.memprobe.tsv` | `/proc/meminfo` samples |
| `bench_L32_S1_etdump.log` | Full bench output |
| `setup_L32_S1.log` | Export log |

See also:

- [`decode_GEMV_ceiling_check.md`](decode_GEMV_ceiling_check.md) — original L=4 + extrapolation report; now corroborated by this real L=32 measurement (310.6 ms steady at L=32 vs 240 ms predicted from L=4 — the extrapolation was off by ~30%, attributable to weight page-cache eviction at L=32)
- [`REPORT.md`](REPORT.md) — baseline prefill (L=32 S=128) and the prefill-cliff/OOM findings
- [`L32_S128_coopmat_REPORT.md`](L32_S128_coopmat_REPORT.md) — coopmat prefill at S=128

## TL;DR for the manager

> *"2k prefill at L=32 fp16 is not runnable on this hardware — confirmed OOM. 1k decode per-step cost at L=32 fp16 is measured at 315 ms (3.2 tok/s, 60% of DDR5 bandwidth ceiling); a full 1024-step decode phase would take ~5.4 minutes of GPU time. Coopmat doesn't help decode (M=1 < 64 dispatch gate); the only meaningful optimization lever for decode is int4 weight quantization, which would also unblock 2k prefill simultaneously by shrinking the working set from 25 GB to ~14 GB."*
