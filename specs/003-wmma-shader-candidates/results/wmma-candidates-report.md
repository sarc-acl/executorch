# WMMA-Optimizable Shader Candidates Report

Built entirely from already-classified data across the six `001`/`002` baseline configurations -- no new profiling. See [`research.md`](../research.md) for why each group is classified the way it is, and each config's own `results/classifications/<model>_<scheme>.json` for full detail.

**No `classification: "a"` (WMMA already in effect) entries exist in this data** -- every capture was taken under `tiled_baseline`. Nothing below should be read as "already using WMMA in production."

## Existing implementation blocked

### Prefill linear GEMM (attention projection + feed-forward + output projection) -- blocked by rank-3 output + TEXTURE_3D storage

**Existing/prospective shader(s)**: linear_dq8ca_q4gsw_coopmat (linear_dq8ca_qw_coopmat.glsl), q4gsw_linear_coopmat (linear_qw_coopmat.glsl)

**Blocking reason(s)**:
- output tensor is rank-3 ([1,M,K]); can_use_q4gsw_coopmat() rejects dim_of(output) > 2 (backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp:192-194)
- output tensor storage is TEXTURE_3D; can_use_q4gsw_coopmat() requires Buffer storage (backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp:196-197)

**Total time across all six configurations**: 19757.88 ms (19757883 us)

| Model | Scheme | Phase | Total time (us) | % of phase |
|---|---|---|---:|---:|
| llama-3.1-8b | 4w | prefill | 7118823 | 67.5% |
| llama-3.1-8b | 8da4w | prefill | 5820490 | 64.9% |
| llama-3.2-3b | 4w | prefill | 2795018 | 57.9% |
| llama-3.2-3b | 8da4w | prefill | 2293407 | 53.3% |
| llama-3.2-1b | 4w | prefill | 967706 | 56.5% |
| llama-3.2-1b | 8da4w | prefill | 762439 | 51.0% |

## No WMMA implementation exists

### SDPA (prefill + decode) -- no WMMA implementation exists

**Existing/prospective shader(s)**: none exists (see reason)

**Blocking reason(s)**:
- no WMMA implementation exists for SDPA; add_sdpa_compute_attn_weights_node/add_sdpa_compute_out_node only ever select _tiled or _coop kernel names (backends/vulkan/runtime/graph/ops/impl/SDPA.cpp); the generic add_matmul_coopmat_node/coopmat_mm.glsl path exists but is not called anywhere in SDPA.cpp

**Total time across all six configurations**: 8663.23 ms (8663226 us)

| Model | Scheme | Phase | Total time (us) | % of phase |
|---|---|---|---:|---:|
| llama-3.1-8b | 4w | prefill | 2290195 | 21.7% |
| llama-3.1-8b | 8da4w | prefill | 2200192 | 24.5% |
| llama-3.2-3b | 8da4w | prefill | 1428664 | 33.2% |
| llama-3.2-3b | 4w | prefill | 1393541 | 28.8% |
| llama-3.2-1b | 4w | prefill | 399389 | 23.3% |
| llama-3.2-1b | 8da4w | prefill | 398330 | 26.7% |
| llama-3.1-8b | 8da4w | decode | 144707 | 26.6% |
| llama-3.1-8b | 4w | decode | 140538 | 26.0% |
| llama-3.2-3b | 4w | decode | 98811 | 34.0% |
| llama-3.2-3b | 8da4w | decode | 98616 | 33.9% |
| llama-3.2-1b | 8da4w | decode | 35351 | 30.7% |
| llama-3.2-1b | 4w | decode | 34892 | 30.3% |

### Decode linear GEMV (attention projection + feed-forward + output projection) -- no WMMA-capable GEMV kernel exists

**Existing/prospective shader(s)**: linear_dq8ca_q4gsw_coopmat (linear_dq8ca_qw_coopmat.glsl), q4gsw_linear_coopmat (linear_qw_coopmat.glsl)

**Blocking reason(s)**:
- no WMMA-capable GEMV (M=1) kernel exists; is_gemv_case routes to the tiled/coop kernel choice before can_use_q4gsw_coopmat() is ever called, and the existing coopmat shaders are tiled multi-row designs not applicable at M=1 (backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp)

**Total time across all six configurations**: 1197.43 ms (1197429 us)

| Model | Scheme | Phase | Total time (us) | % of phase |
|---|---|---|---:|---:|
| llama-3.1-8b | 4w | decode | 371948 | 68.7% |
| llama-3.1-8b | 8da4w | decode | 360869 | 66.2% |
| llama-3.2-3b | 4w | decode | 164262 | 56.4% |
| llama-3.2-3b | 8da4w | decode | 158221 | 54.4% |
| llama-3.2-1b | 4w | decode | 65558 | 57.0% |
| llama-3.2-1b | 8da4w | decode | 60022 | 52.2% |
| llama-3.1-8b | 4w | prefill | 3720 | 0.0% |
| llama-3.1-8b | 8da4w | prefill | 3593 | 0.0% |
| llama-3.2-3b | 4w | prefill | 2814 | 0.1% |
| llama-3.2-3b | 8da4w | prefill | 2732 | 0.1% |
| llama-3.2-1b | 4w | prefill | 1910 | 0.1% |
| llama-3.2-1b | 8da4w | prefill | 1781 | 0.1% |
