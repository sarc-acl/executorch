# Decode-shaped (M=1) check — does coopmat fire?

**Date:** 2026-05-10
**Hardware:** AMD Radeon 780M (RADV PHOENIX), 28.9 GiB RAM, fp16 prefill
**Model:** LLaMA 3.1 8B fp16, **L=4 (not L=32), seq_len=1** (M=1 in every linear)

## Why this test exists

Coopmat's dispatch gate (in `Linear.cpp`) is `M >= 64`. Decode generates one token at a time → every linear has M=1 → **coopmat never fires for decode**. This test verifies that empirically and measures the per-forward cost of the fallback path, so we can extrapolate decode throughput.

## Why L=4 instead of L=32

L=32 seq=1 export OOM-killed twice during the `et.buffer` materialization (the buffer balloon ~16 GB + Python tensor refs ~12 GB > available RAM). L=4 is enough to answer the dispatch question because the gate decision is per-linear and depends only on the per-linear shape — layer count doesn't affect which shader runs. Extrapolation to L=32 by per-shape avg ms × per-layer dispatch counts is shown at the bottom.

## Headline

**Coopmat does not fire on linears at M=1.** Both runs dispatch the identical fallback shader (`linear_vec_tile_row_1_buffer_texture2d_half`). Per-forward GPU time differs by < 1%, within timer noise.

| | Baseline (`main`, no override) | "Coopmat" (`pavan-report`, BUFFER override) |
|---|---|---|
| Per-forward GPU time (steady, ETDump) | **41.1 ms** | **40.7 ms** |
| Top linear shader (all 28 non-lm_head linears) | `linear_vec_tile_row_1_buffer_texture2d_half` | `linear_vec_tile_row_1_buffer_texture2d_half` **(same)** |
| Linear shader time (steady) | 26.72 ms | 26.57 ms |
| lm_head shader | `linear_vec_buffer_buffer_half` | `linear_vec_buffer_buffer_half` **(same)** |
| lm_head time (steady) | 13.16 ms | 13.16 ms |
| Attention BMMs shader | `matmul_vec_tile_row_1_texture3d_float` | `matmul_coopmat_float` |
| Attention BMM time (sum) | 0.15 ms | 0.02 ms (negligible) |
| `linear_coopmat` dispatches | 0 | **0** |
| `matmul_coopmat` dispatches | 0 | 8 (attention; tiny absolute cost) |

The only place coopmat fires in the "coopmat" run is the attention BMMs (`Q @ Kᵀ` and `attn @ V`), which contribute < 0.1% of forward time. **The linear path is unchanged.**

## Why the linear path is identical

The runtime dispatch gate (`Linear.cpp` in pavan-report):

```cpp
bool use_coopmat = !VK_DISABLE_COOPMAT
                && adapter.supports_cooperative_matrix()
                && storage_type_of(out) == kBuffer
                && M >= 64;                           // ← fails at M=1
```

When `use_coopmat=false`, the code falls back to `add_linear_node()` and `prepack_fp_linear_weight(force_buffer=use_coopmat)` → `force_buffer=false` → weights default to `kTexture2D` (per the prepack helper's heuristic, with the usual `max_extent` fallback to buffer for huge dims like the lm_head).

So even with `storage_type_override=BUFFER` set at partitioner time (which forces activations to buffer), the weights still end up as texture2d for normal-size linears at M=1. The runtime kernel name `linear_vec_tile_row_1_buffer_texture2d_half` is identical between the two runs.

The `_tile_row_1_` infix vs the prefill case's `_vec` (TILE_M=4): at M=1 there's only one output row, so `pick_linear_shader()` selects the TILE_M=1 variant. Sensible auto-pick.

## Per-linear breakdown at M=1 (baseline; coopmat is identical)

| Output shape | # dispatches | sum ms | avg ms | Component | Share of linear |
|---|---:|---:|---:|---|---:|
| `[1, 128256]` | 2 | **13.17** | 6.59 | lm_head (1 dispatch + 1 buffer_to_nchw layout op) | 32.7% |
| `[1, 14336]` | 8 | 11.71 | 1.46 | FFN gate + up (`4 layers × 2`) | 29.0% |
| `[1, 4096]` | 12 | 11.51 | 0.96 | Q + O + FFN down (`4 layers × 3`) | 28.5% |
| `[1, 1024]` | 8 | 3.95 | 0.49 | K + V (`4 layers × 2`) | 9.8% |

**lm_head dominates** at 33% of forward — it's a single huge `[1, 128256]` matmul against fp16 weights of size `[4096, 128256]` = 1 GB just for the lm_head weights. The cost doesn't scale with layer count (there's only one lm_head per model regardless of L).

The other 67% scales linearly with `n_layers`: per-layer non-lm_head time = `(40.33 - 13.17) / 4 = 6.79 ms/layer`.

## Extrapolation to L=32

| Component | Per-layer ms | × layers | At L=32 |
|---|---:|---:|---:|
| Per-layer linears (7 × per layer) | 6.79 ms | × 32 | **217.3 ms** |
| lm_head (constant) | — | × 1 | **13.2 ms** |
| Non-linear overhead (estimated ~4% of total at L=4, scales weakly) | | | ~10 ms |
| **L=32 forward at M=1 (decode-mimicking)** | | | **~240 ms** |
| **Implied decode throughput** | | | **~4.2 tokens/sec** |

This matches the bandwidth-bound prediction: 15 GB of fp16 weights / ~80 GB/s effective DDR5 ≈ 5.3 tok/sec theoretical ceiling; real measurement ~4.2 tok/sec is ~80% of ceiling, plausible for a real workload with KV cache traffic + dispatch overhead.

## What this means

1. **Coopmat is irrelevant for decode** on this hardware/codebase. The 3.03× prefill speedup we measured at L=32 S=128 does not carry over.
2. **Decode is memory-bandwidth-bound.** The per-forward time is dominated by streaming 15 GB of fp16 weights through DDR5 once per token. No shader change can beat the bandwidth ceiling.
3. **Real levers for faster decode:**
   - **Weight-only int4 quantization**: 15 GB → 4 GB → ~4× faster (linear in bandwidth reduction)
   - **KV-cache offload / paged attention**: only if KV gets large; at small ctx it's negligible
   - **Speculative decoding**: orthogonal but the most practically large win for "tokens-per-second" UX
   - **Batched decode**: process N independent sequences at once → M=N in linears → coopmat re-engages for N ≥ 64

Shader-level optimization (coopmat, better tiling, etc.) belongs to prefill territory, not decode.

## Caveats / what this test does NOT measure

- **No KV cache.** Real decode reuses K/V across iterations. Our test just runs M=1 self-attention with no history. The compute shape (M=1 linears) is what we wanted to characterize, but real decode has additional cache-read traffic per layer that we don't capture here.
- **No autoregressive loop.** Each iteration runs the same input; we're not threading start_pos or sampling tokens. Real decode runtime adds Python-side sampling + token feedback overhead (cheap relative to GPU time, but non-zero).
- **L=4 extrapolated to L=32.** We didn't actually run L=32 due to OOM during export. If linear time scales perfectly linearly, the extrapolation is accurate. If there's any superlinear effect (cache pressure between layers, etc.), the L=32 number could be higher.
- **The 1% baseline-vs-coopmat gap is noise.** Within run-to-run variance for sub-1-ms ops. Don't read it as a coopmat-specific effect.

## Reproduction

```bash
cd /home/doremy/sarc-acl/executorch/main/executorch

# Baseline
source .venv/bin/activate
python yanwen/scripts/setup_llama31_pure.py --n_layers 4 --seq_len 1
python yanwen/scripts/bench_llama31_pure.py --n_layers 4 --seq_len 1 \
    --num_executions 8 --etdump-analyze

# Coopmat
source /home/doremy/sarc-acl/executorch/pavan-report/executorch/.venv/bin/activate
python yanwen/scripts/coopmat/setup_llama31_coopmat.py --n_layers 4 --seq_len 1
python yanwen/scripts/coopmat/bench_llama31_coopmat.py --n_layers 4 --seq_len 1 \
    --num_executions 8 --etdump-analyze
```

Then run the three analyzers (`pavan-report/.../analyze_etdump.py`, `etvk_breakdown.py`, `linear_by_shape.py`) on both `.etdp` files for the side-by-side.

## Artifacts

`yanwen/artifacts/L4_S1_baseline/` and `yanwen/artifacts/L4_S1_coopmat/` — etdp, events.tsv, memprobe, bench + setup logs for each.
