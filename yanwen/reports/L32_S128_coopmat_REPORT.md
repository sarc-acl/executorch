# LLaMA 3.1 8B prefill at L=32 S=128 with linear_coopmat — findings

**Updated:** 2026-05-10 · **Author:** Yanwen Xu
**Device:** AMD Radeon 780M (RADV PHOENIX, RDNA3+ mobile iGPU), Mesa 25.0.7
**Host:** 28.9 GiB DDR5 RAM, 24 GiB swap (`/swapfile`)
**Model:** LLaMA 3.1 8B fp16, prefill (single forward over `[1, 128]`)

Companion to [`REPORT.md`](REPORT.md) (the linear_vec baseline). Read that first if you haven't seen it.

## TL;DR

| | linear_vec (baseline) | linear_coopmat | **Speedup** |
|---|---:|---:|---:|
| **Steady-state forward** | 1765.6 ± 6.0 ms | **582.6 ± 2.4 ms** | **3.03×** |
| **Throughput** | 72.5 tok/s | **219.7 tok/s** | **3.03×** |
| Linear total (steady GPU) | 1515.94 ms | **412.65 ms** | **3.67×** |
| Per-layer FFN [128, 14336] avg | 12.3 ms | **4.00 ms** | **3.08×** |
| Per-layer Q/O/FFN-down [128, 4096] avg | 6.7 ms | **2.12 ms** | **3.16×** |
| Per-layer K/V [128, 1024] avg | 1.2 ms | **0.32 ms** | **3.75×** |
| Attention BMMs (matmul_*) | 14.5 ms | **0.93 ms** | **15.6×** |
| Peak Shmem | 0.9 GB | 0.79 GB | (slightly less) |
| W1 (load + iter 0) | 21.4 s | 19.3 s | (slightly faster) |
| cv across reps | 0.3% | 0.4% | — |

The ~2× speedup we projected (based on the 2026-05-06 synthetic 4-layer LLaMA at S=2048) **underestimated**. We measured 3× across the full forward, and ~3.7× on linear time alone. The per-shape kernel speedups (3.08–3.75×) are very close to each other, suggesting the coopmat tile schedule is near-optimal for these matmul shapes on RDNA3+ mobile.

## Methodology delta vs baseline

Same scientific bench (1 calibration N=1 + 3 measurement N=8 subprocesses, algebraic-subtraction). Two changes:

1. **Runner:** `pavan-report/executorch/cmake-out-vk/executor_runner` (built from the `pavan-report` branch with linear_coopmat / matmul_coopmat / addmm_khr_cm GLSL shaders compiled into `spv.cpp`). Required two missing-`<algorithm>`-include fixes during build (compiler-version drift on the branch).
2. **Partitioner:** `VulkanPartitioner({"storage_type_override": VkStorageType.BUFFER})` — forces linear/matmul outputs to `kBuffer` storage. The runtime then dispatches `linear_coopmat` when `M ≥ 64` (S=128 ✓), the GPU exposes `VK_KHR_cooperative_matrix` (780M ✓), and `VK_DISABLE_COOPMAT` is unset.

Re-exported `.pte` lives at `/home/doremy/llama31_pure_run_coopmat/llama31_8b_32L_seq128_fp16.pte` (16.06 GB — same size as baseline; weights dominate, storage layout doesn't affect serialized weight count).

## Verification: coopmat actually fired

The runner's debug logs show 224 `linear_coopmat` dispatches per forward and 32 `matmul_coopmat` dispatches:

```
[VK_LINEAR] Using linear_coopmat (cooperative matrix, bias=0)   ← 224× per iter
[VK_MATMUL] Using matmul_coopmat (cooperative matrix)           ←  32× per iter
```

Two `linear_vec_buffer_buffer_half` events remain — the **lm_head `[1, 128256]`** has `M=1 < 64`, so it falls back. (Same single-dispatch behavior as the 2026-05-06 prior measurement.)

## Headline result

```
W1 (load + iter 0 + teardown):  19.35 s
rep 1 steady:                    582.1 ms
rep 2 steady:                    585.1 ms
rep 3 steady:                    580.4 ms

Steady-state forward:  582.6 ± 2.4 ms   (cv = 0.4%)
Prefill throughput:    219.7 tokens/sec
```

Run command:
```bash
source /home/doremy/sarc-acl/executorch/pavan-report/executorch/.venv/bin/activate
cd /home/doremy/sarc-acl/executorch/main/executorch
python yanwen/scripts/coopmat/bench_llama31_coopmat.py --n_layers 32 --seq_len 128
```

Log: `yanwen/artifacts/L32_coopmat/S128_bench.log`.

## Memory at L=32 S=128 (coopmat)

| Quantity | coopmat | baseline | Delta |
|---|---:|---:|---:|
| peak Shmem (Vulkan GTT-backed) | 0.79 GB | 0.92 GB | −0.13 GB |
| peak Cached (.pte mmap) | 10.27 GB | 9.46 GB | +0.81 GB |
| min MemFree during run | 6.68 GB | 6.18 GB | +0.50 GB |
| peak Swap during run | 5.04 GB | ~baseline | + ~1 GB (system noise) |

Buffer-storage activations use slightly less Vulkan-GTT than the texture2d-storage baseline. The `.pte` weight pages get cached more aggressively (10.27 GB resident vs 9.46 GB) — likely because buffer-storage weights get touched in a more linear pattern. Either way, **comfortable headroom at S=128**.

## ETDump categories — side-by-side

(canonical analyzer; ETDump-reported total is summed across all leaf events, slightly above wallclock due to async overlap; relative shares are reliable)

| Category | coopmat (ms) | coopmat % | baseline (ms) | baseline % | Per-category speedup |
|---|---:|---:|---:|---:|---:|
| **linear** | **493.7** | 56.6% | 1527.6 | 80.1% | **3.09×** |
| reshape / view | 113.3 | 13.0% | 106.7 | 5.6% | 0.94× (slightly slower) |
| CPU↔GPU copy | 91.9 | 10.5% | 91.4 | 4.8% | 1.00× |
| CPU fallback (cat) | 78.0 | 8.9% | 76.7 | 4.0% | 0.98× |
| elementwise | 64.0 | 7.3% | 59.0 | 3.1% | 0.92× |
| CPU fallback (eq.Scalar) | 12.3 | 1.4% | 13.9 | 0.7% | 1.13× |
| softmax | 7.8 | 0.9% | 6.9 | 0.4% | 0.88× |
| **bmm / matmul** | **0.9** | 0.1% | 14.5 | 0.8% | **15.6×** (matmul_coopmat) |
| Other CPU fallbacks | ~5.5 | 0.6% | ~6.2 | 0.3% | — |

Two interesting effects beyond the linear win:

1. **Attention BMMs got matmul_coopmat for free.** 64 dispatches at 0.014 ms each = 0.93 ms total, vs. baseline `matmul_vec_*` = 14.5 ms total. **15.6× speedup on attention matmuls.** Tiny in absolute time at S=128, but will matter at higher seq.
2. **Non-linear ops did NOT speed up.** Reshape/view, CPU↔GPU copy, CPU fallbacks, elementwise are all unchanged (or even slightly slower from buffer-storage layout transitions). This is what allows Amdahl's law to dominate.

Amdahl back-check: linear was 89.7% of GPU time at baseline (1515.94 ms / 1689.5 ms steady-state GPU, from the per-kernel breakdown). With linear sped up 3.67× and other 10.3% unchanged:

```
predicted speedup = 1 / (0.897 / 3.67 + 0.103) = 1 / 0.347 = 2.88×
measured speedup = 3.03×
```

Within 5%. The small extra win likely comes from iter 0 / cold-cache cost dropping too (W1 went 21.4 → 19.3 s, ~10% faster init).

## Linear by output shape

| Output shape | # disp | sum ms | avg ms | Kernel | Component | Per-shape speedup |
|---|---:|---:|---:|---|---|---:|
| `[128, 14336]` | 64 | 255.9 | **4.00** | `linear_coopmat_half` | FFN gate + up | **3.08×** (vs 12.3) |
| `[128, 4096]` | 96 | 203.7 | **2.12** | `linear_coopmat_half` | Q + O + FFN down | **3.16×** (vs 6.7) |
| `[128, 1024]` | 64 | 20.7 | **0.32** | `linear_coopmat_half` | K + V | **3.75×** (vs 1.2) |
| `[1, 128256]` | 2 | 13.4 | 6.69 | `linear_vec_buffer_buffer_half` | lm_head (FALLBACK, M=1) | 1.0× (no change) |

K/V (`[128, 1024]`) shows the largest per-kernel speedup (3.75×). Q/O/FFN-down (`[128, 4096]`) and FFN gate/up (`[128, 14336]`) cluster around 3.1×. Net effect: **all three GEMM shapes that LLaMA actually uses see ~3× from the cooperative-matrix tiling**.

## Per-shader (steady-state, iter 1..7 averaged)

(via `etvk_breakdown.py` — total = 586.2 ms, matches wallclock-measured 582.6 ms within timer noise)

| # | kernel | dispatches | steady ms | % steady |
|---:|---|---:|---:|---:|
| 1 | **`linear_coopmat_half`** | 224 | **398.58** | **68.0%** |
| 2 | `binary_mul_buffer_float` | 418 | 24.10 | 4.1% |
| 3 | `view_buffer_half` | 704 | 22.62 | 3.9% |
| 4 | `view_buffer_float` | 544 | 18.86 | 3.2% |
| 5 | `linear_vec_buffer_buffer_half` (lm_head) | 1 | 13.17 | 2.2% |
| 6 | `view_convert_buffer_half_float` | 545 | 13.09 | 2.2% |
| 7 | `view_convert_buffer_float_half` | 193 | 10.85 | 1.9% |
| 8 | `buffer_to_nchw_float_float` | 224 | 9.90 | 1.7% |
| 9 | `binary_mul_buffer_half` | 97 | 9.23 | 1.6% |
| 10 | `permute_buffer_half` | 192 | 8.99 | 1.5% |
| 11 | `nchw_to_buffer_float_float` | 160 | 8.97 | 1.5% |
| 12 | `sigmoid_float_buffer` | 32 | 6.79 | 1.2% |
| 13 | `softmax_buffer_float` | 32 | 6.65 | 1.1% |
| 14 | `expand_buffer_half` | 64 | 6.11 | 1.0% |
| 24 | `matmul_coopmat_float` (attention) | 64 | 0.89 | 0.2% |

Bucketed by family (steady):

| Family | Steady ms | % steady | Baseline % |
|---|---:|---:|---:|
| **matmul / linear** | 412.65 | **70.4%** | 89.7% |
| layout / copy / slice | 124.97 | 21.3% | 6.8% |
| binary | 41.91 | 7.2% | 2.4% |
| softmax | 6.65 | 1.1% | 0.4% |

Linear dropped from 89.7% → 70.4% of forward (because everything else stays the same in absolute time but linear shrunk). Layout/copy/slice grew from 6.8% → 21.3% as a *share* (still the same ~125 ms in absolute time), now visible as the next-largest contributor.

## What's next, given this result

1. **Layout/copy is the new ceiling.** With linear at 70% and the rest unchanged, a perfect 10× linear shader would reduce forward to 415 / 10 + 173 = **214 ms**. Going below that requires touching layout/copy/binary too. **Most likely candidates: `view_buffer_*` and `view_convert_buffer_*`** (3272 dispatches between them, contributing 7-8% of forward time — fragmented, may benefit from delegate-boundary consolidation).
2. **CPU `aten.cat` is now disproportionately important.** 78 ms (8.9%) on KV concat. A Vulkan `cat` op would recover this entirely AND remove the CPU↔GPU sync per layer.
3. **`ETVK_COPY_OUTPUTS` is still 75 ms.** Same value as baseline — buffer storage didn't help. Worth a separate investigation (256 KB output in 75 ms ≈ 3.4 GB/s on UMA is an outlier).
4. **Don't bother with attention BMMs / softmax.** Combined <2% even in the coopmat regime.
5. **Memory is not the constraint at S=128 with coopmat.** 22 GB of GTT cap, 6.7 GB MemFree headroom. Buffer storage didn't blow up the working set. **S=512 with coopmat is worth a separate experiment** — buffer storage may shift the cliff differently than texture2d.

## Reproduction

### One-time setup

Build pavan-report's runner (only needed once; took ~3–5 min wallclock with `j$(nproc)`):

```bash
cd /home/doremy/sarc-acl/executorch/pavan-report/executorch
.venv/lib/python3.12/site-packages/cmake/data/bin/cmake . -Bcmake-out-vk \
    --preset linux \
    -DCMAKE_INSTALL_PREFIX=cmake-out-vk \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_VULKAN=ON \
    -DEXECUTORCH_BUILD_DEVTOOLS=ON \
    -DEXECUTORCH_ENABLE_EVENT_TRACER=ON \
    -DEXECUTORCH_ENABLE_LOGGING=OFF
.venv/lib/python3.12/site-packages/cmake/data/bin/cmake --build cmake-out-vk -j$(nproc) --target install
```

If the build fails with `std::find` / `std::rotate` errors, add `#include <algorithm>` to the offending `runtime/` `.cpp` / `.h` files. (One-off bug on the `pavan-report` branch from a now-stricter GCC; fixed in this session.)

### Run

```bash
# Activate pavan-report's venv (modified partitioner with storage forcing)
source /home/doremy/sarc-acl/executorch/pavan-report/executorch/.venv/bin/activate
sudo swapon /swapfile     # required for the 32L export's ~16 GiB Python RAM peak

cd /home/doremy/sarc-acl/executorch/main/executorch

# Phase 1: re-export .pte with storage_type_override=BUFFER (~5 min)
python yanwen/scripts/coopmat/setup_llama31_coopmat.py --n_layers 32 --seq_len 128

# Phase 2: scientific bench (~2 min, output: 582 ± 3 ms / 220 tok/s)
python yanwen/scripts/coopmat/bench_llama31_coopmat.py --n_layers 32 --seq_len 128

# Phase 3: ETDump capture for shader breakdown (~25 s)
python yanwen/scripts/coopmat/bench_llama31_coopmat.py --n_layers 32 --seq_len 128 \
    --num_executions 8 --etdump-analyze
```

### Analyzers (work cross-tree, no venv constraint)

```bash
python /home/doremy/sarc-acl/executorch/pavan-report/executorch/yanwen_plan/analyze_etdump.py \
    yanwen/artifacts/L32_coopmat/S128.etdp

python yanwen/scripts/etvk_breakdown.py yanwen/artifacts/L32_coopmat/S128.events.tsv 8

python yanwen/scripts/linear_by_shape.py yanwen/artifacts/L32_coopmat/S128.etdp
```

## Artifacts

Under `yanwen/artifacts/L32_coopmat/`:

| File | Description |
|---|---|
| `S128.etdp` | Binary ETDump (N=8 capture from coopmat run) |
| `S128.events.tsv` | Inspector dataframe with per-shader GPU timings |
| `S128.memprobe.tsv` | `/proc/meminfo` samples during last subprocess |
| `S128_bench.log` | Scientific-mode bench output (3 reps) |
| `S128_etdump.log` | ETDump capture + Inspector breakdown |
| `S128_setup.log` | Phase 1 export log |

See also: [`L32_S128_coopmat_shader_breakdown.md`](L32_S128_coopmat_shader_breakdown.md) for the full per-GLSL-shader inventory mirroring `L32_S128_shader_breakdown.md`.
