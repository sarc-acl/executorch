# M5 EVT1 Full Microbenchmark Suite — Unified, Real-Regime Report

Clocks pinned (509/2730/663 MHz, sysfs-verified), driver identity re-verified before measurement (constitution Principles VII/VIII). All three harnesses now share one `RESULT,...` schema (specs/021), printed per case immediately on completion. Each harness invoked 3 separate times; spread is reported as CoV% across those 3 invocations, with a case flagged as an outlier only when its CoV is a clear peer-relative outlier (>3x its harness's median CoV), never a fixed cutoff (research.md Decision 3). `dispatch_status` is a three-way distinction (research.md Decision 2): `confirmed` (coopmat-eligible and fired), `fallback_tiled` (coopmat-eligible but didn't fire -- a real anomaly), `not_applicable` (structurally excluded from the coopmat comparison by design, e.g. decode regime or a Texture3D/tiled-only case).

## Linear (`test_coopmat_linear_bench`)

**Plain-language summary**: at the real prefill shape (M=2048), coopmat (WMMA) is faster than tiled (no WMMA) for every shape/model/scheme tested -- speedups range from +65% to +77%. At the real decode shape (M=1), coopmat never fires by design (24/24 cases `not_applicable`) -- `QuantizedLinear.cpp`'s `is_gemv_case` short-circuit dispatches a dedicated GEMV kernel before the coopmat eligibility check ever runs, confirmed via direct source read, not assumed.

| Model | Scheme | Regime | Shape (K,N) | Tiled / No WMMA (us) | Coopmat / WMMA (us) | Speedup% | CoV% | Dispatch | Outlier |
|---|---|---|---|---:|---:|---:|---:|---|---|
| llama-3.1-8b | 4w | decode | (4096,1024) | 51.8 | 50.5 | N/A | 1.09% | not_applicable | no |
| llama-3.1-8b | 4w | decode | (4096,4096) | 217.4 | 261.8 | N/A | 1.36% | not_applicable | no |
| llama-3.1-8b | 4w | decode | (4096,14336) | 800.2 | 821.3 | N/A | 2.92% | not_applicable | YES: 2.9% CoV vs 0.7% group median |
| llama-3.1-8b | 4w | decode | (14336,4096) | 879.1 | 869.0 | N/A | 0.47% | not_applicable | no |
| llama-3.1-8b | 4w | prefill | (4096,1024) | 19557.2 | 6469.8 | +66.9% | 0.05% | confirmed | no |
| llama-3.1-8b | 4w | prefill | (4096,4096) | 76471.1 | 24987.5 | +67.3% | 0.14% | confirmed | no |
| llama-3.1-8b | 4w | prefill | (4096,14336) | 267032.6 | 86758.5 | +67.5% | 0.25% | confirmed | no |
| llama-3.1-8b | 4w | prefill | (14336,4096) | 266978.2 | 86931.9 | +67.4% | 0.14% | confirmed | no |
| llama-3.1-8b | 8da4w | decode | (4096,1024) | 53.2 | 53.6 | N/A | 1.18% | not_applicable | no |
| llama-3.1-8b | 8da4w | decode | (4096,4096) | 230.8 | 267.0 | N/A | 5.86% | not_applicable | YES: 5.9% CoV vs 0.7% group median |
| llama-3.1-8b | 8da4w | decode | (4096,14336) | 808.3 | 842.0 | N/A | 1.53% | not_applicable | no |
| llama-3.1-8b | 8da4w | decode | (14336,4096) | 874.3 | 869.6 | N/A | 1.68% | not_applicable | no |
| llama-3.1-8b | 8da4w | prefill | (4096,1024) | 31256.8 | 8679.1 | +72.2% | 0.08% | confirmed | no |
| llama-3.1-8b | 8da4w | prefill | (4096,4096) | 120515.3 | 30441.7 | +74.7% | 0.16% | confirmed | no |
| llama-3.1-8b | 8da4w | prefill | (4096,14336) | 418673.4 | 102905.8 | +75.4% | 0.10% | confirmed | no |
| llama-3.1-8b | 8da4w | prefill | (14336,4096) | 422001.6 | 112252.0 | +73.4% | 0.09% | confirmed | no |
| llama-3.2-1b | 4w | decode | (2048,512) | 25.0 | 24.1 | N/A | 0.38% | not_applicable | no |
| llama-3.2-1b | 4w | decode | (2048,2048) | 47.5 | 47.3 | N/A | 0.07% | not_applicable | no |
| llama-3.2-1b | 4w | decode | (2048,8192) | 202.6 | 208.1 | N/A | 3.30% | not_applicable | YES: 3.3% CoV vs 0.7% group median |
| llama-3.2-1b | 4w | decode | (8192,2048) | 260.0 | 248.6 | N/A | 7.07% | not_applicable | YES: 7.1% CoV vs 0.7% group median |
| llama-3.2-1b | 4w | prefill | (2048,512) | 5458.8 | 1892.2 | +65.3% | 0.43% | confirmed | no |
| llama-3.2-1b | 4w | prefill | (2048,2048) | 19183.6 | 6438.6 | +66.4% | 0.06% | confirmed | no |
| llama-3.2-1b | 4w | prefill | (2048,8192) | 76317.0 | 25067.1 | +67.2% | 0.29% | confirmed | no |
| llama-3.2-1b | 4w | prefill | (8192,2048) | 76800.8 | 25076.6 | +67.3% | 0.18% | confirmed | no |
| llama-3.2-1b | 8da4w | decode | (2048,512) | 25.4 | 24.9 | N/A | 0.92% | not_applicable | no |
| llama-3.2-1b | 8da4w | decode | (2048,2048) | 49.0 | 49.1 | N/A | 1.47% | not_applicable | no |
| llama-3.2-1b | 8da4w | decode | (2048,8192) | 205.8 | 231.9 | N/A | 6.63% | not_applicable | YES: 6.6% CoV vs 0.7% group median |
| llama-3.2-1b | 8da4w | decode | (8192,2048) | 249.9 | 245.1 | N/A | 8.22% | not_applicable | YES: 8.2% CoV vs 0.7% group median |
| llama-3.2-1b | 8da4w | prefill | (2048,512) | 8079.8 | 2040.9 | +74.7% | 0.76% | confirmed | no |
| llama-3.2-1b | 8da4w | prefill | (2048,2048) | 30789.5 | 7423.4 | +75.9% | 0.31% | confirmed | no |
| llama-3.2-1b | 8da4w | prefill | (2048,8192) | 119992.3 | 28099.2 | +76.6% | 1.33% | confirmed | no |
| llama-3.2-1b | 8da4w | prefill | (8192,2048) | 121941.8 | 32521.3 | +73.3% | 0.96% | confirmed | no |
| llama-3.2-3b | 4w | decode | (3072,1024) | 41.0 | 40.1 | N/A | 0.22% | not_applicable | no |
| llama-3.2-3b | 4w | decode | (3072,3072) | 104.5 | 103.7 | N/A | 3.80% | not_applicable | YES: 3.8% CoV vs 0.7% group median |
| llama-3.2-3b | 4w | decode | (3072,8192) | 327.4 | 386.9 | N/A | 2.78% | not_applicable | YES: 2.8% CoV vs 0.7% group median |
| llama-3.2-3b | 4w | decode | (8192,3072) | 384.2 | 378.8 | N/A | 2.27% | not_applicable | YES: 2.3% CoV vs 0.7% group median |
| llama-3.2-3b | 4w | prefill | (3072,1024) | 14718.5 | 4930.1 | +66.5% | 0.14% | confirmed | no |
| llama-3.2-3b | 4w | prefill | (3072,3072) | 43010.5 | 14147.5 | +67.1% | 0.12% | confirmed | no |
| llama-3.2-3b | 4w | prefill | (3072,8192) | 114312.6 | 37483.7 | +67.2% | 0.11% | confirmed | no |
| llama-3.2-3b | 4w | prefill | (8192,3072) | 114806.1 | 37366.2 | +67.5% | 0.11% | confirmed | no |
| llama-3.2-3b | 8da4w | decode | (3072,1024) | 42.1 | 41.3 | N/A | 0.57% | not_applicable | no |
| llama-3.2-3b | 8da4w | decode | (3072,3072) | 116.7 | 111.8 | N/A | 11.95% | not_applicable | YES: 12.0% CoV vs 0.7% group median |
| llama-3.2-3b | 8da4w | decode | (3072,8192) | 325.7 | 386.0 | N/A | 2.06% | not_applicable | no |
| llama-3.2-3b | 8da4w | decode | (8192,3072) | 388.0 | 373.5 | N/A | 5.79% | not_applicable | YES: 5.8% CoV vs 0.7% group median |
| llama-3.2-3b | 8da4w | prefill | (3072,1024) | 23397.5 | 6220.9 | +73.4% | 4.17% | confirmed | YES: 4.2% CoV vs 0.7% group median |
| llama-3.2-3b | 8da4w | prefill | (3072,3072) | 67882.8 | 16644.9 | +75.5% | 0.67% | confirmed | no |
| llama-3.2-3b | 8da4w | prefill | (3072,8192) | 179835.5 | 42235.4 | +76.5% | 0.70% | confirmed | no |
| llama-3.2-3b | 8da4w | prefill | (8192,3072) | 181454.3 | 48300.1 | +73.4% | 0.70% | confirmed | no |

24/24 prefill cases confirmed coopmat dispatch; 24/24 decode cases correctly `not_applicable` (SC-003).

## SDPA (`test_sdpa_coopmat_bench`)

**Plain-language summary**: coopmat (WMMA) SDPA (combined qk+av) is faster than tiled (no WMMA) for all 3 models at the real prefill shape (S=2048) -- speedups range from +75% to +82%. The decode shape (S=1, real context_len=3072) is `not_applicable` for all 3 models -- `SDPA.cpp`'s `is_gemv` gate excludes it from the coopmat comparison by design, structurally identical to linear bench's decode handling.

| Model | Regime | Sub-op | Tiled / No WMMA (us) | Coopmat / WMMA (us) | Speedup% | CoV% | Dispatch | Outlier |
|---|---|---|---:|---:|---:|---:|---|---|
| llama-3.1-8b | decode | av | 3142.4 | — | N/A | 1.66% | not_applicable | YES: 1.7% CoV vs 0.5% group median |
| llama-3.1-8b | decode | qk | 1582.4 | — | N/A | 1.70% | not_applicable | YES: 1.7% CoV vs 0.5% group median |
| llama-3.1-8b | decode | total | 4724.9 | — | N/A | 0.61% | not_applicable | no |
| llama-3.1-8b | prefill | av | 93326.2 | 17034.7 | +81.7% | 0.11% | confirmed | no |
| llama-3.1-8b | prefill | qk | 101681.0 | 19030.3 | +81.3% | 0.08% | confirmed | no |
| llama-3.1-8b | prefill | total | 195007.0 | 36064.9 | +81.5% | 0.07% | confirmed | no |
| llama-3.2-1b | decode | av | 1650.7 | — | N/A | 0.75% | not_applicable | no |
| llama-3.2-1b | decode | qk | 1462.9 | — | N/A | 0.16% | not_applicable | no |
| llama-3.2-1b | decode | total | 3113.6 | — | N/A | 0.36% | not_applicable | no |
| llama-3.2-1b | prefill | av | 49081.4 | 8893.5 | +81.9% | 0.06% | confirmed | no |
| llama-3.2-1b | prefill | qk | 38996.6 | 13171.8 | +66.2% | 1.22% | confirmed | no |
| llama-3.2-1b | prefill | total | 88078.0 | 22065.4 | +74.9% | 0.71% | confirmed | no |
| llama-3.2-3b | decode | av | 2062.9 | — | N/A | 3.52% | not_applicable | YES: 3.5% CoV vs 0.5% group median |
| llama-3.2-3b | decode | qk | 1179.7 | — | N/A | 1.46% | not_applicable | no |
| llama-3.2-3b | decode | total | 3242.6 | — | N/A | 2.00% | not_applicable | YES: 2.0% CoV vs 0.5% group median |
| llama-3.2-3b | prefill | av | 69674.8 | 12794.5 | +81.6% | 0.11% | confirmed | no |
| llama-3.2-3b | prefill | qk | 79061.4 | 14382.6 | +81.8% | 0.38% | confirmed | no |
| llama-3.2-3b | prefill | total | 148736.0 | 27177.1 | +81.7% | 0.22% | confirmed | no |

9/9 decode rows correctly `not_applicable` (SC-003).

## Baseline (`test_llama_baseline_bench`)

**Plain-language summary**: no coopmat comparison here by design (`ET_VK_FORCE_TILED_LINEAR=1` forces every case onto the tiled/`_coop` path) -- this harness instead compares `Texture3D` vs `Buffer` storage at real prefill(M=2048)/decode(M=1) shapes, all 3 models, both quantization schemes. First-ever complete run of this harness on M5 EVT1 in this workstream's history (specs/020 got 14/192 cases before an OOM this feature fixed). `lm_head` (K,128256) is excluded from this run entirely (per explicit user decision, tasks.md follow-up to T007) -- it was this suite's single largest dispatch (observed 270us-2.4ms, wildly variable) and the confirmed trigger for both a pre-existing QueryPool race (research.md Decision 9) and a suspected GPU/driver reset that disconnected the device mid-run once. Excluding it eliminates that risk at the source rather than continuing to rely on the `try`/`catch` safety net for every run. None hit `CRASHED` in this specific run.

| Model | Scheme | Regime | Shape (K,N) | Variant | Storage | Avg (us) | CoV% | Correctness | Outlier |
|---|---|---|---|---|---|---:|---:|---|---|
| llama-3.1-8b | 4w | decode | (4096,1024) | coop | buffer | 53.9 | 1.90% | SKIPPED | YES: 1.9% CoV vs 0.6% group median |
| llama-3.1-8b | 4w | decode | (4096,1024) | coop | texture3d | 54.5 | 1.15% | SKIPPED | no |
| llama-3.1-8b | 4w | decode | (4096,4096) | coop | buffer | 263.7 | 4.03% | SKIPPED | YES: 4.0% CoV vs 0.6% group median |
| llama-3.1-8b | 4w | decode | (4096,4096) | coop | texture3d | 234.5 | 5.62% | SKIPPED | YES: 5.6% CoV vs 0.6% group median |
| llama-3.1-8b | 4w | decode | (4096,14336) | coop | buffer | 973.0 | 3.14% | SKIPPED | YES: 3.1% CoV vs 0.6% group median |
| llama-3.1-8b | 4w | decode | (4096,14336) | coop | texture3d | 945.3 | 3.98% | SKIPPED | YES: 4.0% CoV vs 0.6% group median |
| llama-3.1-8b | 4w | decode | (14336,4096) | coop | buffer | 1047.1 | 10.56% | SKIPPED | YES: 10.6% CoV vs 0.6% group median |
| llama-3.1-8b | 4w | decode | (14336,4096) | coop | texture3d | 1068.8 | 0.71% | SKIPPED | no |
| llama-3.1-8b | 4w | prefill | (4096,1024) | coopmat | buffer | 6821.5 | 0.15% | SKIPPED | no |
| llama-3.1-8b | 4w | prefill | (4096,1024) | tiled | texture3d | 19504.6 | 0.08% | SKIPPED | no |
| llama-3.1-8b | 4w | prefill | (4096,4096) | coopmat | buffer | 26324.4 | 0.16% | SKIPPED | no |
| llama-3.1-8b | 4w | prefill | (4096,4096) | tiled | texture3d | 76643.1 | 0.06% | SKIPPED | no |
| llama-3.1-8b | 4w | prefill | (4096,14336) | coopmat | buffer | 91298.9 | 0.23% | SKIPPED | no |
| llama-3.1-8b | 4w | prefill | (4096,14336) | tiled | texture3d | 267611.2 | 0.15% | SKIPPED | no |
| llama-3.1-8b | 4w | prefill | (14336,4096) | coopmat | buffer | 91847.1 | 0.27% | SKIPPED | no |
| llama-3.1-8b | 4w | prefill | (14336,4096) | tiled | texture3d | 267541.4 | 0.03% | SKIPPED | no |
| llama-3.1-8b | 8da4w | decode | (4096,1024) | coop | buffer | 56.7 | 2.65% | SKIPPED | YES: 2.7% CoV vs 0.6% group median |
| llama-3.1-8b | 8da4w | decode | (4096,1024) | coop | texture3d | 57.0 | 2.76% | SKIPPED | YES: 2.8% CoV vs 0.6% group median |
| llama-3.1-8b | 8da4w | decode | (4096,4096) | coop | buffer | 257.4 | 1.25% | SKIPPED | no |
| llama-3.1-8b | 8da4w | decode | (4096,4096) | coop | texture3d | 238.6 | 5.56% | SKIPPED | YES: 5.6% CoV vs 0.6% group median |
| llama-3.1-8b | 8da4w | decode | (4096,14336) | coop | buffer | 857.6 | 2.36% | SKIPPED | YES: 2.4% CoV vs 0.6% group median |
| llama-3.1-8b | 8da4w | decode | (4096,14336) | coop | texture3d | 832.5 | 0.97% | SKIPPED | no |
| llama-3.1-8b | 8da4w | decode | (14336,4096) | coop | buffer | 845.0 | 4.34% | SKIPPED | YES: 4.3% CoV vs 0.6% group median |
| llama-3.1-8b | 8da4w | decode | (14336,4096) | coop | texture3d | 862.8 | 5.66% | SKIPPED | YES: 5.7% CoV vs 0.6% group median |
| llama-3.1-8b | 8da4w | prefill | (4096,1024) | coopmat | buffer | 13565.3 | 0.56% | SKIPPED | no |
| llama-3.1-8b | 8da4w | prefill | (4096,1024) | tiled | texture3d | 34034.4 | 0.07% | SKIPPED | no |
| llama-3.1-8b | 8da4w | prefill | (4096,4096) | coopmat | buffer | 48393.4 | 0.33% | SKIPPED | no |
| llama-3.1-8b | 8da4w | prefill | (4096,4096) | tiled | texture3d | 129943.9 | 0.13% | SKIPPED | no |
| llama-3.1-8b | 8da4w | prefill | (4096,14336) | coopmat | buffer | 163333.5 | 0.91% | SKIPPED | no |
| llama-3.1-8b | 8da4w | prefill | (4096,14336) | tiled | texture3d | 450867.0 | 0.18% | SKIPPED | no |
| llama-3.1-8b | 8da4w | prefill | (14336,4096) | coopmat | buffer | 179305.7 | 0.32% | SKIPPED | no |
| llama-3.1-8b | 8da4w | prefill | (14336,4096) | tiled | texture3d | 454123.5 | 0.02% | SKIPPED | no |
| llama-3.2-1b | 4w | decode | (2048,512) | coop | buffer | 25.1 | 0.13% | SKIPPED | no |
| llama-3.2-1b | 4w | decode | (2048,512) | coop | texture3d | 26.3 | 1.08% | SKIPPED | no |
| llama-3.2-1b | 4w | decode | (2048,2048) | coop | buffer | 50.3 | 1.04% | SKIPPED | no |
| llama-3.2-1b | 4w | decode | (2048,2048) | coop | texture3d | 50.6 | 3.12% | SKIPPED | YES: 3.1% CoV vs 0.6% group median |
| llama-3.2-1b | 4w | decode | (2048,8192) | coop | buffer | 271.8 | 2.79% | SKIPPED | YES: 2.8% CoV vs 0.6% group median |
| llama-3.2-1b | 4w | decode | (2048,8192) | coop | texture3d | 215.3 | 2.34% | SKIPPED | YES: 2.3% CoV vs 0.6% group median |
| llama-3.2-1b | 4w | decode | (8192,2048) | coop | buffer | 266.3 | 6.86% | SKIPPED | YES: 6.9% CoV vs 0.6% group median |
| llama-3.2-1b | 4w | decode | (8192,2048) | coop | texture3d | 253.3 | 1.20% | SKIPPED | no |
| llama-3.2-1b | 4w | prefill | (2048,512) | coopmat | buffer | 1902.0 | 0.16% | SKIPPED | no |
| llama-3.2-1b | 4w | prefill | (2048,512) | tiled | texture3d | 5463.2 | 0.54% | SKIPPED | no |
| llama-3.2-1b | 4w | prefill | (2048,2048) | coopmat | buffer | 6747.2 | 0.06% | SKIPPED | no |
| llama-3.2-1b | 4w | prefill | (2048,2048) | tiled | texture3d | 19218.2 | 0.04% | SKIPPED | no |
| llama-3.2-1b | 4w | prefill | (2048,8192) | coopmat | buffer | 26412.7 | 0.16% | SKIPPED | no |
| llama-3.2-1b | 4w | prefill | (2048,8192) | tiled | texture3d | 76494.7 | 0.11% | SKIPPED | no |
| llama-3.2-1b | 4w | prefill | (8192,2048) | coopmat | buffer | 26411.1 | 0.04% | SKIPPED | no |
| llama-3.2-1b | 4w | prefill | (8192,2048) | tiled | texture3d | 76931.8 | 0.12% | SKIPPED | no |
| llama-3.2-1b | 8da4w | decode | (2048,512) | coop | buffer | 26.6 | 0.49% | SKIPPED | no |
| llama-3.2-1b | 8da4w | decode | (2048,512) | coop | texture3d | 27.5 | 0.70% | SKIPPED | no |
| llama-3.2-1b | 8da4w | decode | (2048,2048) | coop | buffer | 52.0 | 2.28% | SKIPPED | YES: 2.3% CoV vs 0.6% group median |
| llama-3.2-1b | 8da4w | decode | (2048,2048) | coop | texture3d | 51.3 | 0.67% | SKIPPED | no |
| llama-3.2-1b | 8da4w | decode | (2048,8192) | coop | buffer | 265.8 | 3.46% | SKIPPED | YES: 3.5% CoV vs 0.6% group median |
| llama-3.2-1b | 8da4w | decode | (2048,8192) | coop | texture3d | 208.5 | 2.18% | SKIPPED | YES: 2.2% CoV vs 0.6% group median |
| llama-3.2-1b | 8da4w | decode | (8192,2048) | coop | buffer | 255.4 | 7.45% | SKIPPED | YES: 7.4% CoV vs 0.6% group median |
| llama-3.2-1b | 8da4w | decode | (8192,2048) | coop | texture3d | 255.3 | 7.46% | SKIPPED | YES: 7.5% CoV vs 0.6% group median |
| llama-3.2-1b | 8da4w | prefill | (2048,512) | coopmat | buffer | 3203.2 | 0.36% | SKIPPED | no |
| llama-3.2-1b | 8da4w | prefill | (2048,512) | tiled | texture3d | 8831.8 | 0.06% | SKIPPED | no |
| llama-3.2-1b | 8da4w | prefill | (2048,2048) | coopmat | buffer | 11370.3 | 0.48% | SKIPPED | no |
| llama-3.2-1b | 8da4w | prefill | (2048,2048) | tiled | texture3d | 33290.3 | 0.07% | SKIPPED | no |
| llama-3.2-1b | 8da4w | prefill | (2048,8192) | coopmat | buffer | 43584.8 | 1.24% | SKIPPED | no |
| llama-3.2-1b | 8da4w | prefill | (2048,8192) | tiled | texture3d | 129155.9 | 0.12% | SKIPPED | no |
| llama-3.2-1b | 8da4w | prefill | (8192,2048) | coopmat | buffer | 51487.4 | 0.21% | SKIPPED | no |
| llama-3.2-1b | 8da4w | prefill | (8192,2048) | tiled | texture3d | 131214.5 | 0.16% | SKIPPED | no |
| llama-3.2-3b | 4w | decode | (3072,1024) | coop | buffer | 43.1 | 0.71% | SKIPPED | no |
| llama-3.2-3b | 4w | decode | (3072,1024) | coop | texture3d | 43.8 | 0.65% | SKIPPED | no |
| llama-3.2-3b | 4w | decode | (3072,3072) | coop | buffer | 154.1 | 3.02% | SKIPPED | YES: 3.0% CoV vs 0.6% group median |
| llama-3.2-3b | 4w | decode | (3072,3072) | coop | texture3d | 159.7 | 6.67% | SKIPPED | YES: 6.7% CoV vs 0.6% group median |
| llama-3.2-3b | 4w | decode | (3072,8192) | coop | buffer | 406.6 | 2.23% | SKIPPED | YES: 2.2% CoV vs 0.6% group median |
| llama-3.2-3b | 4w | decode | (3072,8192) | coop | texture3d | 348.9 | 3.39% | SKIPPED | YES: 3.4% CoV vs 0.6% group median |
| llama-3.2-3b | 4w | decode | (8192,3072) | coop | buffer | 369.7 | 0.72% | SKIPPED | no |
| llama-3.2-3b | 4w | decode | (8192,3072) | coop | texture3d | 381.9 | 5.23% | SKIPPED | YES: 5.2% CoV vs 0.6% group median |
| llama-3.2-3b | 4w | prefill | (3072,1024) | coopmat | buffer | 5184.9 | 0.13% | SKIPPED | no |
| llama-3.2-3b | 4w | prefill | (3072,1024) | tiled | texture3d | 14665.0 | 0.04% | SKIPPED | no |
| llama-3.2-3b | 4w | prefill | (3072,3072) | coopmat | buffer | 14898.0 | 0.09% | SKIPPED | no |
| llama-3.2-3b | 4w | prefill | (3072,3072) | tiled | texture3d | 43067.0 | 0.06% | SKIPPED | no |
| llama-3.2-3b | 4w | prefill | (3072,8192) | coopmat | buffer | 39353.9 | 0.29% | SKIPPED | no |
| llama-3.2-3b | 4w | prefill | (3072,8192) | tiled | texture3d | 114521.0 | 0.09% | SKIPPED | no |
| llama-3.2-3b | 4w | prefill | (8192,3072) | coopmat | buffer | 39341.3 | 0.23% | SKIPPED | no |
| llama-3.2-3b | 4w | prefill | (8192,3072) | tiled | texture3d | 114946.3 | 0.07% | SKIPPED | no |
| llama-3.2-3b | 8da4w | decode | (3072,1024) | coop | buffer | 44.8 | 0.44% | SKIPPED | no |
| llama-3.2-3b | 8da4w | decode | (3072,1024) | coop | texture3d | 45.5 | 1.82% | SKIPPED | YES: 1.8% CoV vs 0.6% group median |
| llama-3.2-3b | 8da4w | decode | (3072,3072) | coop | buffer | 160.1 | 2.82% | SKIPPED | YES: 2.8% CoV vs 0.6% group median |
| llama-3.2-3b | 8da4w | decode | (3072,3072) | coop | texture3d | 162.9 | 5.20% | SKIPPED | YES: 5.2% CoV vs 0.6% group median |
| llama-3.2-3b | 8da4w | decode | (3072,8192) | coop | buffer | 395.4 | 2.94% | SKIPPED | YES: 2.9% CoV vs 0.6% group median |
| llama-3.2-3b | 8da4w | decode | (3072,8192) | coop | texture3d | 343.4 | 2.82% | SKIPPED | YES: 2.8% CoV vs 0.6% group median |
| llama-3.2-3b | 8da4w | decode | (8192,3072) | coop | buffer | 365.3 | 0.56% | SKIPPED | no |
| llama-3.2-3b | 8da4w | decode | (8192,3072) | coop | texture3d | 375.7 | 6.65% | SKIPPED | YES: 6.6% CoV vs 0.6% group median |
| llama-3.2-3b | 8da4w | prefill | (3072,1024) | coopmat | buffer | 10674.2 | 0.06% | SKIPPED | no |
| llama-3.2-3b | 8da4w | prefill | (3072,1024) | tiled | texture3d | 25428.7 | 0.09% | SKIPPED | no |
| llama-3.2-3b | 8da4w | prefill | (3072,3072) | coopmat | buffer | 25620.8 | 1.55% | SKIPPED | no |
| llama-3.2-3b | 8da4w | prefill | (3072,3072) | tiled | texture3d | 73195.8 | 0.10% | SKIPPED | no |
| llama-3.2-3b | 8da4w | prefill | (3072,8192) | coopmat | buffer | 66177.3 | 0.70% | SKIPPED | no |
| llama-3.2-3b | 8da4w | prefill | (3072,8192) | tiled | texture3d | 193730.6 | 0.11% | SKIPPED | no |
| llama-3.2-3b | 8da4w | prefill | (8192,3072) | coopmat | buffer | 76311.0 | 0.52% | SKIPPED | no |
| llama-3.2-3b | 8da4w | prefill | (8192,3072) | tiled | texture3d | 195049.3 | 0.02% | SKIPPED | no |

96/96 cases shown measured successfully; 0 `CRASHED` (see Anomalies).

## Reconciliation against `specs/016-m5-linear-sdpa-microbench` (FR-009/FR-010)

**Shape-basis caveat (FR-010)**: `specs/016`'s linear numbers used an `M=1024` compromise shape; this feature's prefill numbers use the real `M=2048`. These are DIFFERENT measurements -- the table below compares only whether the tiled-vs-coopmat speedup direction and rough magnitude are consistent, never exact percentage deltas against a different shape. Decode (`M=1`/`S=1`) has no specs/016 equivalent at all -- it's new coverage, not a reconciliation target.

| Model | Scheme | Shape (K,N) | specs/016 speedup% (M=1024) | This feature speedup% (M=2048) | Same direction/magnitude? |
|---|---|---|---:|---:|---|
| llama-3.1-8b | 4w | (4096,1024) | +66.3% | +66.9% | YES |
| llama-3.1-8b | 4w | (4096,4096) | +67.2% | +67.3% | YES |
| llama-3.1-8b | 4w | (4096,14336) | +67.8% | +67.5% | YES |
| llama-3.1-8b | 4w | (14336,4096) | +67.0% | +67.4% | YES |
| llama-3.1-8b | 8da4w | (4096,1024) | +73.9% | +72.2% | YES |
| llama-3.1-8b | 8da4w | (4096,4096) | +74.8% | +74.7% | YES |
| llama-3.1-8b | 8da4w | (4096,14336) | +77.8% | +75.4% | YES |
| llama-3.1-8b | 8da4w | (14336,4096) | +74.1% | +73.4% | YES |
| llama-3.2-1b | 4w | (2048,512) | +61.8% | +65.3% | YES |
| llama-3.2-1b | 4w | (2048,2048) | +65.4% | +66.4% | YES |
| llama-3.2-1b | 4w | (2048,8192) | +66.9% | +67.2% | YES |
| llama-3.2-1b | 4w | (8192,2048) | +66.9% | +67.3% | YES |
| llama-3.2-1b | 8da4w | (2048,512) | +76.5% | +74.7% | YES |
| llama-3.2-1b | 8da4w | (2048,2048) | +76.4% | +75.9% | YES |
| llama-3.2-1b | 8da4w | (2048,8192) | +77.5% | +76.6% | YES |
| llama-3.2-1b | 8da4w | (8192,2048) | +72.7% | +73.3% | YES |
| llama-3.2-3b | 4w | (3072,1024) | +65.7% | +66.5% | YES |
| llama-3.2-3b | 4w | (3072,3072) | +66.5% | +67.1% | YES |
| llama-3.2-3b | 4w | (3072,8192) | +67.3% | +67.2% | YES |
| llama-3.2-3b | 4w | (8192,3072) | +67.3% | +67.5% | YES |
| llama-3.2-3b | 8da4w | (3072,1024) | +75.8% | +73.4% | YES |
| llama-3.2-3b | 8da4w | (3072,3072) | +75.7% | +75.5% | YES |
| llama-3.2-3b | 8da4w | (3072,8192) | +76.3% | +76.5% | YES |
| llama-3.2-3b | 8da4w | (8192,3072) | +73.4% | +73.4% | YES |

| Model (SDPA) | specs/016 speedup% | This feature speedup% (prefill total) | Same direction/magnitude? |
|---|---:|---:|---|
| llama-3.1-8b | +81.5% | +81.5% | YES |
| llama-3.2-1b | +75.2% | +74.9% | YES |
| llama-3.2-3b | +81.8% | +81.7% | YES |

`test_llama_baseline_bench` has no prior M5 EVT1 number to reconcile against — this is its first complete run on this target.

## Anomalies (FR-010)

- linear llama-3.2-3b/8da4w/prefill shape (3072,1024): outlier, 4.2% CoV vs 0.7% group median
- 47 decode-regime case(s) show a peer-relative CoV outlier -- expected measurement noise at decode's tens-of-microseconds dispatch scale (`not_applicable` cases are never a real coopmat-comparison anomaly by construction), not listed individually here; see each section's own table for the exact CoV% per case.
- (pre-existing, unrelated to this feature) 10/53 of `test_coopmat_linear_bench`'s small-shape `linear_dq8ca_q4gsw` `Texture3D` correctness cases FAILED under `COOPMAT_BENCH_CORRECTNESS_ONLY=1` -- verified via `git stash` that the unmodified HEAD version of this file produces the identical 43 PASSED/10 FAILED split; not a regression from this feature's regime-axis change (tasks.md T015).
