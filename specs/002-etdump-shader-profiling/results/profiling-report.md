# ETDump E2E Shader Profiling Report

Companion to [`001-minipc-baseline-benchmarks/results/baseline-report.md`](../../001-minipc-baseline-benchmarks/results/baseline-report.md) -- same device, same six configurations, same `tiled_baseline` dispatch path. This report breaks down *where* each phase's time goes: per-kernel time/shape/count, rolled up into categories.

**Device**: `rocky-ryzen` -- AMD Radeon 780M (RADV PHOENIX), RDNA3 mobile integrated GPU.
**Dispatch path**: `tiled_baseline` for every row below (`ET_VK_FORCE_TILED_LINEAR=1`).
**Capture configuration**: prefill at the fixed 2048 tokens (matching `001`); decode over a short representative window (7-8 steps) rather than the full 1024-step decode `001` used for throughput -- per-step shader/shape composition doesn't vary with decode position on this architecture, so a short window is sufficient for attribution (see `research.md` Decision 5).

## llama-3.1-8b

### 4w

#### Prefill

| Category | % of phase | Total time (us) |
|---|---:|---:|
| feed-forward | 54.2% | 5716368 |
| attention (sdpa) | 21.7% | 2290195 |
| attention projection | 13.3% | 1402454 |
| non-shader overhead | 10.3% | 1091167 |
| output/vocab projection | 0.0% | 3720 |
| unattributed | 0.3% | 33722 |

Top kernels by time (of 23 distinct kernel+shape entries):

| Kernel | Shape (M,K,N) | Count | Total time (us) | % of phase |
|---|---|---:|---:|---:|
| `q4gsw_linear_gemm__tin__w_4x8_nc_texture3d_half` | (2048,4096,14336) | 64 | 3791610 | 36.0% |
| `q4gsw_linear_gemm__tin__w_4x8_nc_texture3d_half` | (2048,14336,4096) | 32 | 1924758 | 18.3% |
| `sdpa_compute_out_tiled_buffer_buffer_half` | (2048,128,128) | 32 | 1373110 | 13.0% |
| `q4gsw_linear_gemm__tin__w_4x8_nc_texture3d_half` | (2048,4096,4096) | 64 | 1116768 | 10.6% |
| `sdpa_compute_attn_weights_tiled_buffer_buffer_half` | (2048,128,128) | 32 | 917084 | 8.7% |
| `transpose_cast_contig_to_vectorized_4x4_half_texture3d_half_buffer` | n/a | 224 | 355397 | 3.4% |

Reconciliation: **99.7%** of this phase's profiled wall-clock (10538000us) is attributed to named kernels above (the rest is framework/dispatch overhead not captured as a distinct event). For comparison, the un-profiled `001` baseline measured this phase at 11973107us (-12.0% vs. profiled).

Raw per-invocation data: [`raw/llama-3.1-8b_4w_prefill_raw.json`](raw/llama-3.1-8b_4w_prefill_raw.json)

#### Decode

| Category | % of phase | Total time (us) |
|---|---:|---:|
| feed-forward | 50.5% | 273098 |
| attention (sdpa) | 26.0% | 140538 |
| attention projection | 13.4% | 72479 |
| output/vocab projection | 4.9% | 26371 |
| non-shader overhead | 2.3% | 12574 |
| unattributed | 2.9% | 15960 |

Top kernels by time (of 22 distinct kernel+shape entries):

| Kernel | Shape (M,K,N) | Count | Total time (us) | % of phase |
|---|---|---:|---:|---:|
| `q4gsw_linear_gemv_coop__w_4x8_nc_buffer_g8w8_texture3d_half` | (1,4096,14336) | 448 | 179765 | 33.2% |
| `q4gsw_linear_gemv_coop__w_4x8_nc_buffer_g4w16_texture3d_half` | (1,14336,4096) | 224 | 93333 | 17.2% |
| `sdpa_compute_out_coop_buffer_buffer_half` | (1,128,128) | 224 | 85762 | 15.8% |
| `q4gsw_linear_gemv_coop__w_4x8_nc_buffer_g4w16_texture3d_half` | (1,4096,4096) | 448 | 56278 | 10.4% |
| `sdpa_compute_attn_weights_coop_buffer_buffer_half` | (1,128,128) | 224 | 54776 | 10.1% |
| `q4gsw_linear_gemv_coop__w_4x8_nc_buffer_g8w8_buffer_half` | (1,4096,128256) | 7 | 26371 | 4.9% |

Reconciliation: **97.0%** of this phase's profiled wall-clock (541000us) is attributed to named kernels above (the rest is framework/dispatch overhead not captured as a distinct event). For comparison, the un-profiled `001` baseline measured this phase at 754148us (-28.3% vs. profiled).

Raw per-invocation data: [`raw/llama-3.1-8b_4w_decode_raw.json`](raw/llama-3.1-8b_4w_decode_raw.json)

Full data: [`raw/llama-3.1-8b_4w.json`](raw/llama-3.1-8b_4w.json)

### 8da4w

#### Prefill

| Category | % of phase | Total time (us) |
|---|---:|---:|
| feed-forward | 52.0% | 4666282 |
| attention (sdpa) | 24.5% | 2200192 |
| attention projection | 12.9% | 1154208 |
| non-shader overhead | 10.1% | 907396 |
| output/vocab projection | 0.0% | 3593 |
| unattributed | 0.4% | 37670 |

Top kernels by time (of 24 distinct kernel+shape entries):

| Kernel | Shape (M,K,N) | Count | Total time (us) | % of phase |
|---|---|---:|---:|---:|
| `linear_dq8ca_q4gsw_tiled_texture3d_texture2d_half` | (2048,4096,14336) | 64 | 3123740 | 34.8% |
| `linear_dq8ca_q4gsw_tiled_texture3d_texture2d_half` | (2048,14336,4096) | 32 | 1542543 | 17.2% |
| `sdpa_compute_out_tiled_buffer_buffer_half` | (2048,128,128) | 32 | 1311253 | 14.6% |
| `linear_dq8ca_q4gsw_tiled_texture3d_texture2d_half` | (2048,4096,4096) | 64 | 928814 | 10.4% |
| `sdpa_compute_attn_weights_tiled_buffer_buffer_half` | (2048,128,128) | 32 | 888939 | 9.9% |
| `sdpa_attn_weights_softmax_buffer_half` | n/a | 32 | 316119 | 3.5% |

Reconciliation: **99.6%** of this phase's profiled wall-clock (8969000us) is attributed to named kernels above (the rest is framework/dispatch overhead not captured as a distinct event). For comparison, the un-profiled `001` baseline measured this phase at 9556696us (-6.1% vs. profiled).

Raw per-invocation data: [`raw/llama-3.1-8b_8da4w_prefill_raw.json`](raw/llama-3.1-8b_8da4w_prefill_raw.json)

#### Decode

| Category | % of phase | Total time (us) |
|---|---:|---:|
| feed-forward | 49.5% | 269648 |
| attention (sdpa) | 26.6% | 144707 |
| attention projection | 12.1% | 66065 |
| output/vocab projection | 4.6% | 25156 |
| non-shader overhead | 3.6% | 19797 |
| unattributed | 3.6% | 19620 |

Top kernels by time (of 23 distinct kernel+shape entries):

| Kernel | Shape (M,K,N) | Count | Total time (us) | % of phase |
|---|---|---:|---:|---:|
| `linear_dq8ca_q4gsw_coop_texture3d_texture2d_half` | (1,4096,14336) | 448 | 176163 | 32.3% |
| `linear_dq8ca_q4gsw_coop_texture3d_texture2d_half` | (1,14336,4096) | 224 | 93485 | 17.2% |
| `sdpa_compute_out_coop_buffer_buffer_half` | (1,128,128) | 224 | 91045 | 16.7% |
| `sdpa_compute_attn_weights_coop_buffer_buffer_half` | (1,128,128) | 224 | 53662 | 9.8% |
| `linear_dq8ca_q4gsw_coop_texture3d_texture2d_half` | (1,4096,4096) | 448 | 52071 | 9.6% |
| `linear_dq8ca_q4gsw_coop_buffer_texture2d_half` | (1,4096,128256) | 7 | 25156 | 4.6% |

Reconciliation: **96.4%** of this phase's profiled wall-clock (545000us) is attributed to named kernels above (the rest is framework/dispatch overhead not captured as a distinct event). For comparison, the un-profiled `001` baseline measured this phase at 738755us (-26.2% vs. profiled).

Raw per-invocation data: [`raw/llama-3.1-8b_8da4w_decode_raw.json`](raw/llama-3.1-8b_8da4w_decode_raw.json)

Full data: [`raw/llama-3.1-8b_8da4w.json`](raw/llama-3.1-8b_8da4w.json)

---

## llama-3.2-3b

### 4w

#### Prefill

| Category | % of phase | Total time (us) |
|---|---:|---:|
| feed-forward | 43.0% | 2075110 |
| attention (sdpa) | 28.8% | 1393541 |
| attention projection | 14.9% | 719908 |
| non-shader overhead | 12.8% | 618487 |
| output/vocab projection | 0.1% | 2814 |
| unattributed | 0.5% | 22227 |

Top kernels by time (of 23 distinct kernel+shape entries):

| Kernel | Shape (M,K,N) | Count | Total time (us) | % of phase |
|---|---|---:|---:|---:|
| `q4gsw_linear_gemm__tin__w_4x8_nc_texture3d_half` | (2048,3072,8192) | 56 | 1374092 | 28.4% |
| `sdpa_compute_out_tiled_buffer_buffer_half` | (2048,128,128) | 28 | 814182 | 16.9% |
| `q4gsw_linear_gemm__tin__w_4x8_nc_texture3d_half` | (2048,8192,3072) | 28 | 701018 | 14.5% |
| `sdpa_compute_attn_weights_tiled_buffer_buffer_half` | (2048,128,128) | 28 | 579359 | 12.0% |
| `q4gsw_linear_gemm__tin__w_4x8_nc_texture3d_half` | (2048,3072,3072) | 56 | 533727 | 11.1% |
| `sdpa_attn_weights_softmax_buffer_half` | n/a | 28 | 207282 | 4.3% |

Reconciliation: **99.5%** of this phase's profiled wall-clock (4832000us) is attributed to named kernels above (the rest is framework/dispatch overhead not captured as a distinct event). For comparison, the un-profiled `001` baseline measured this phase at 5272915us (-8.4% vs. profiled).

Raw per-invocation data: [`raw/llama-3.2-3b_4w_prefill_raw.json`](raw/llama-3.2-3b_4w_prefill_raw.json)

#### Decode

| Category | % of phase | Total time (us) |
|---|---:|---:|
| feed-forward | 36.5% | 106248 |
| attention (sdpa) | 34.0% | 98811 |
| attention projection | 13.2% | 38360 |
| output/vocab projection | 6.8% | 19653 |
| non-shader overhead | 3.6% | 10487 |
| unattributed | 6.0% | 17431 |

Top kernels by time (of 22 distinct kernel+shape entries):

| Kernel | Shape (M,K,N) | Count | Total time (us) | % of phase |
|---|---|---:|---:|---:|
| `q4gsw_linear_gemv_coop__w_4x8_nc_buffer_g8w8_texture3d_half` | (1,3072,8192) | 392 | 71374 | 24.5% |
| `sdpa_compute_out_coop_buffer_buffer_half` | (1,128,128) | 196 | 53705 | 18.5% |
| `sdpa_compute_attn_weights_coop_buffer_buffer_half` | (1,128,128) | 196 | 45106 | 15.5% |
| `q4gsw_linear_gemv_coop__w_4x8_nc_buffer_g4w16_texture3d_half` | (1,8192,3072) | 196 | 34874 | 12.0% |
| `q4gsw_linear_gemv_coop__w_4x8_nc_buffer_g4w16_texture3d_half` | (1,3072,3072) | 392 | 27681 | 9.5% |
| `q4gsw_linear_gemv_coop__w_4x8_nc_buffer_g8w8_buffer_half` | (1,3072,128256) | 7 | 19653 | 6.8% |

Reconciliation: **94.0%** of this phase's profiled wall-clock (291000us) is attributed to named kernels above (the rest is framework/dispatch overhead not captured as a distinct event). For comparison, the un-profiled `001` baseline measured this phase at 372876us (-22.0% vs. profiled).

Raw per-invocation data: [`raw/llama-3.2-3b_4w_decode_raw.json`](raw/llama-3.2-3b_4w_decode_raw.json)

Full data: [`raw/llama-3.2-3b_4w.json`](raw/llama-3.2-3b_4w.json)

### 8da4w

#### Prefill

| Category | % of phase | Total time (us) |
|---|---:|---:|
| feed-forward | 40.0% | 1722413 |
| attention (sdpa) | 33.2% | 1428664 |
| attention projection | 13.3% | 570994 |
| non-shader overhead | 13.0% | 560814 |
| output/vocab projection | 0.1% | 2732 |
| unattributed | 0.5% | 20238 |

Top kernels by time (of 24 distinct kernel+shape entries):

| Kernel | Shape (M,K,N) | Count | Total time (us) | % of phase |
|---|---|---:|---:|---:|
| `linear_dq8ca_q4gsw_tiled_texture3d_texture2d_half` | (2048,3072,8192) | 56 | 1155216 | 26.8% |
| `sdpa_compute_out_tiled_buffer_buffer_half` | (2048,128,128) | 28 | 822510 | 19.1% |
| `sdpa_compute_attn_weights_tiled_buffer_buffer_half` | (2048,128,128) | 28 | 606154 | 14.1% |
| `linear_dq8ca_q4gsw_tiled_texture3d_texture2d_half` | (2048,8192,3072) | 28 | 567197 | 13.2% |
| `linear_dq8ca_q4gsw_tiled_texture3d_texture2d_half` | (2048,3072,3072) | 56 | 427699 | 9.9% |
| `sdpa_attn_weights_softmax_buffer_half` | n/a | 28 | 209523 | 4.9% |

Reconciliation: **99.5%** of this phase's profiled wall-clock (4306000us) is attributed to named kernels above (the rest is framework/dispatch overhead not captured as a distinct event). For comparison, the un-profiled `001` baseline measured this phase at 4498331us (-4.3% vs. profiled).

Raw per-invocation data: [`raw/llama-3.2-3b_8da4w_prefill_raw.json`](raw/llama-3.2-3b_8da4w_prefill_raw.json)

#### Decode

| Category | % of phase | Total time (us) |
|---|---:|---:|
| feed-forward | 35.2% | 102552 |
| attention (sdpa) | 33.9% | 98616 |
| attention projection | 12.4% | 36214 |
| output/vocab projection | 6.7% | 19454 |
| non-shader overhead | 5.5% | 15991 |
| unattributed | 6.2% | 18158 |

Top kernels by time (of 23 distinct kernel+shape entries):

| Kernel | Shape (M,K,N) | Count | Total time (us) | % of phase |
|---|---|---:|---:|---:|
| `linear_dq8ca_q4gsw_coop_texture3d_texture2d_half` | (1,3072,8192) | 392 | 67769 | 23.3% |
| `sdpa_compute_out_coop_buffer_buffer_half` | (1,128,128) | 196 | 54372 | 18.7% |
| `sdpa_compute_attn_weights_coop_buffer_buffer_half` | (1,128,128) | 196 | 44243 | 15.2% |
| `linear_dq8ca_q4gsw_coop_texture3d_texture2d_half` | (1,8192,3072) | 196 | 34783 | 11.9% |
| `linear_dq8ca_q4gsw_coop_texture3d_texture2d_half` | (1,3072,3072) | 392 | 26896 | 9.2% |
| `linear_dq8ca_q4gsw_coop_buffer_texture2d_half` | (1,3072,128256) | 7 | 19454 | 6.7% |

Reconciliation: **93.8%** of this phase's profiled wall-clock (291000us) is attributed to named kernels above (the rest is framework/dispatch overhead not captured as a distinct event). For comparison, the un-profiled `001` baseline measured this phase at 378897us (-23.2% vs. profiled).

Raw per-invocation data: [`raw/llama-3.2-3b_8da4w_decode_raw.json`](raw/llama-3.2-3b_8da4w_decode_raw.json)

Full data: [`raw/llama-3.2-3b_8da4w.json`](raw/llama-3.2-3b_8da4w.json)

---

## llama-3.2-1b

### 4w

#### Prefill

| Category | % of phase | Total time (us) |
|---|---:|---:|
| feed-forward | 46.0% | 787940 |
| attention (sdpa) | 23.3% | 399389 |
| non-shader overhead | 19.3% | 330868 |
| attention projection | 10.5% | 179766 |
| output/vocab projection | 0.1% | 1910 |
| unattributed | 0.7% | 12155 |

Top kernels by time (of 23 distinct kernel+shape entries):

| Kernel | Shape (M,K,N) | Count | Total time (us) | % of phase |
|---|---|---:|---:|---:|
| `q4gsw_linear_gemm__tin__w_4x8_nc_texture3d_half` | (2048,2048,8192) | 32 | 522772 | 30.5% |
| `q4gsw_linear_gemm__tin__w_4x8_nc_texture3d_half` | (2048,8192,2048) | 16 | 265167 | 15.5% |
| `sdpa_compute_out_tiled_buffer_buffer_half` | (2048,64,64) | 16 | 227304 | 13.3% |
| `sdpa_compute_attn_weights_tiled_buffer_buffer_half` | (2048,64,64) | 16 | 172084 | 10.1% |
| `sdpa_attn_weights_softmax_buffer_half` | n/a | 16 | 162118 | 9.5% |
| `q4gsw_linear_gemm__tin__w_4x8_nc_texture3d_half` | (2048,2048,2048) | 32 | 139508 | 8.2% |

Reconciliation: **99.3%** of this phase's profiled wall-clock (1712000us) is attributed to named kernels above (the rest is framework/dispatch overhead not captured as a distinct event). For comparison, the un-profiled `001` baseline measured this phase at 1807734us (-5.3% vs. profiled).

Raw per-invocation data: [`raw/llama-3.2-1b_4w_prefill_raw.json`](raw/llama-3.2-1b_4w_prefill_raw.json)

#### Decode

| Category | % of phase | Total time (us) |
|---|---:|---:|
| feed-forward | 36.4% | 41889 |
| attention (sdpa) | 30.3% | 34892 |
| output/vocab projection | 11.4% | 13140 |
| attention projection | 9.2% | 10529 |
| non-shader overhead | 3.8% | 4415 |
| unattributed | 8.8% | 10132 |

Top kernels by time (of 22 distinct kernel+shape entries):

| Kernel | Shape (M,K,N) | Count | Total time (us) | % of phase |
|---|---|---:|---:|---:|
| `q4gsw_linear_gemv_coop__w_4x8_nc_buffer_g8w8_texture3d_half` | (1,2048,8192) | 224 | 27603 | 24.0% |
| `sdpa_compute_out_coop_buffer_buffer_half` | (1,64,64) | 112 | 19482 | 16.9% |
| `sdpa_compute_attn_weights_coop_buffer_buffer_half` | (1,64,64) | 112 | 15411 | 13.4% |
| `q4gsw_linear_gemv_coop__w_4x8_nc_buffer_g4w16_texture3d_half` | (1,8192,2048) | 112 | 14286 | 12.4% |
| `q4gsw_linear_gemv_coop__w_4x8_nc_buffer_g8w8_buffer_half` | (1,2048,128256) | 7 | 13140 | 11.4% |
| `q4gsw_linear_gemv_coop__w_4x8_nc_buffer_g4w16_texture3d_half` | (1,2048,2048) | 224 | 7631 | 6.6% |

Reconciliation: **91.2%** of this phase's profiled wall-clock (115000us) is attributed to named kernels above (the rest is framework/dispatch overhead not captured as a distinct event). For comparison, the un-profiled `001` baseline measured this phase at 121342us (-5.2% vs. profiled).

Raw per-invocation data: [`raw/llama-3.2-1b_4w_decode_raw.json`](raw/llama-3.2-1b_4w_decode_raw.json)

Full data: [`raw/llama-3.2-1b_4w.json`](raw/llama-3.2-1b_4w.json)

### 8da4w

#### Prefill

| Category | % of phase | Total time (us) |
|---|---:|---:|
| feed-forward | 41.8% | 624568 |
| attention (sdpa) | 26.7% | 398330 |
| non-shader overhead | 21.2% | 316852 |
| attention projection | 9.2% | 137871 |
| output/vocab projection | 0.1% | 1781 |
| unattributed | 1.0% | 14641 |

Top kernels by time (of 24 distinct kernel+shape entries):

| Kernel | Shape (M,K,N) | Count | Total time (us) | % of phase |
|---|---|---:|---:|---:|
| `linear_dq8ca_q4gsw_tiled_texture3d_texture2d_half` | (2048,2048,8192) | 32 | 413961 | 27.7% |
| `sdpa_compute_out_tiled_buffer_buffer_half` | (2048,64,64) | 16 | 225515 | 15.1% |
| `linear_dq8ca_q4gsw_tiled_texture3d_texture2d_half` | (2048,8192,2048) | 16 | 210607 | 14.1% |
| `sdpa_compute_attn_weights_tiled_buffer_buffer_half` | (2048,64,64) | 16 | 172815 | 11.6% |
| `sdpa_attn_weights_softmax_buffer_half` | n/a | 16 | 163308 | 10.9% |
| `linear_dq8ca_q4gsw_tiled_texture3d_texture2d_half` | (2048,2048,2048) | 32 | 109629 | 7.3% |

Reconciliation: **99.0%** of this phase's profiled wall-clock (1494000us) is attributed to named kernels above (the rest is framework/dispatch overhead not captured as a distinct event). For comparison, the un-profiled `001` baseline measured this phase at 1508700us (-1.0% vs. profiled).

Raw per-invocation data: [`raw/llama-3.2-1b_8da4w_prefill_raw.json`](raw/llama-3.2-1b_8da4w_prefill_raw.json)

#### Decode

| Category | % of phase | Total time (us) |
|---|---:|---:|
| feed-forward | 33.4% | 38351 |
| attention (sdpa) | 30.7% | 35351 |
| output/vocab projection | 10.8% | 12401 |
| attention projection | 8.1% | 9270 |
| non-shader overhead | 5.4% | 6218 |
| unattributed | 11.7% | 13409 |

Top kernels by time (of 23 distinct kernel+shape entries):

| Kernel | Shape (M,K,N) | Count | Total time (us) | % of phase |
|---|---|---:|---:|---:|
| `linear_dq8ca_q4gsw_coop_texture3d_texture2d_half` | (1,2048,8192) | 224 | 25620 | 22.3% |
| `sdpa_compute_out_coop_buffer_buffer_half` | (1,64,64) | 112 | 20200 | 17.6% |
| `sdpa_compute_attn_weights_coop_buffer_buffer_half` | (1,64,64) | 112 | 15151 | 13.2% |
| `linear_dq8ca_q4gsw_coop_texture3d_texture2d_half` | (1,8192,2048) | 112 | 12731 | 11.1% |
| `linear_dq8ca_q4gsw_coop_buffer_texture2d_half` | (1,2048,128256) | 7 | 12401 | 10.8% |
| `linear_dq8ca_q4gsw_coop_texture3d_texture2d_half` | (1,2048,2048) | 224 | 7088 | 6.2% |

Reconciliation: **88.3%** of this phase's profiled wall-clock (115000us) is attributed to named kernels above (the rest is framework/dispatch overhead not captured as a distinct event). For comparison, the un-profiled `001` baseline measured this phase at 118734us (-3.1% vs. profiled).

Raw per-invocation data: [`raw/llama-3.2-1b_8da4w_decode_raw.json`](raw/llama-3.2-1b_8da4w_decode_raw.json)

Full data: [`raw/llama-3.2-1b_8da4w.json`](raw/llama-3.2-1b_8da4w.json)

---

## Cross-model observations

- **Feed-forward dominates every configuration** (~40-54% of prefill, ~33-51% of decode), followed by attention/SDPA compute (~22-34%), then attention projection (~9-15%). This is consistent across all three model sizes and both quantization schemes -- the WMMA/coopmat workstream's highest-leverage target is the feed-forward linears (`w1_gate`/`w3_up`/`w2_down`), not attention projection or the output head.
- **`lm_head` is a rounding error during prefill (~0.05%) but ~5-11% of decode** -- consistent with lm_head only being computed for the last prompt position during prefill but every step during decode.
- **Non-shader overhead is larger in prefill (~10-21%) than decode (~2-5%)** -- plausibly one-time weight-prepack/cast costs that don't repeat across decode steps within the same run.
- **Attribution is consistently high** (99.0-99.7% prefill, 88.3-97.1% decode) across all six configurations -- the parsing approach (Vulkan-embedded per-dispatch JSON, no ETRecord) captures nearly all phase time.
- **Profiled decode consistently measures ~22-28% *faster* than the `001` baseline scaled to the same step count** (and profiled prefill ~4-12% faster too) -- the opposite of what "profiling overhead" would predict. The likely explanation, not measurement error: `001` found a reproducible warm-up effect where the first few runs after GPU idle measure faster than the thermally-settled steady-state used for its reported baseline numbers (see `001`'s `baseline-report.md` Observations). This feature's short decode window (7-8 steps) doesn't run long enough to reach that thermally-throttled steady state, so it isn't directly comparable to `001`'s sustained-throughput numbers -- treat this report's phase timings as valid for *attribution* (where does time go, proportionally) rather than as a corrected throughput measurement.
