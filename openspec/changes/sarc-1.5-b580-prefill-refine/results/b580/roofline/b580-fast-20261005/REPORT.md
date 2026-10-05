# Intel(R) Arc(tm) B580 Graphics (BMG G21) roofline report

Device: host AMD Ryzen 5 9600X 6-Core Processor (SoC AMD Ryzen 5 9600X 6-Core Processor), Fedora Linux 44 (Workstation Edition), driver 109060099, subgroup 32.
Plan(s): fast. GPU clock **DVFS-governed** (not pinned); results depend on the governor and thermal state.
Code: None, runner 810e098c8abb2dfb. Rows from other runner or shader builds excluded: 0.

Validation policy: **pre_and_post**; checks bracket continuous sampling and do not guarantee detection of transient errors between checks. Historical pre-only rows are diagnostic only.
Excluded rows: 55; reasons are in all-configurations.csv.
Sustained roofs require three distinct stable batches per current confirmed configuration and duration

Short-run columns are the best validated configuration: median and best (minimum time, the STREAM/BabelStream convention) of the samples, and the differential rate (paired L vs L/2 runs, removing fixed per-dispatch cost). Sustained is the median of the last 60 s of each sustained run (durations are recorded in sustained-runs.csv). Controls never define a roof. `spill` marks variants whose driver statistics report register spilling: spills only slow a kernel, so the value is still an achievable lower bound.

| roof | short median | short best | differential | sustained | unit |
|---|---:|---:|---:|---:|---|
| alu_fp16 | 27.295 | 27.297 | 27.348 (fixed 0%) | 27.294 (1×) | TFLOP/s |
| alu_fp32 | 12.912 | 12.913 | 12.949 (fixed 0%) | — | TFLOP/s |
| cache_read_effective | 6020.517 | 6026.907 | 6027.406 (fixed 0%) | — | GB/s |
| dot_int8 | 32.060 | 32.064 | 32.149 (fixed 0%) | — | TOP/s |
| global_copy | 408.141 | 413.180 | 413.773 (fixed 1%) | — | GB/s |
| global_read | 465.197 | 466.931 | 471.107 (fixed 1%) | 465.340 (1×) | GB/s |
| global_triad | 393.197 | 395.444 | 394.354 (fixed 0%) | — | GB/s |
| global_write | 403.182 | 412.174 | 415.173 (fixed 3%) | — | GB/s |
| matrix_fp16 | 112.918 | 112.979 | 115.901 (fixed 3%) | — | TFLOP/s |
| matrix_fp16_feed_cache | 106.961 | 107.707 | 109.138 (fixed 2%) | — | TFLOP/s |
| matrix_fp16_feed_dram | 444.238 | 444.667 | 452.050 (fixed 2%) | — | GB/s |
| matrix_fp16_feed_shared | 109.728 | 109.853 | 113.745 (fixed 4%) | — | TFLOP/s |
| matrix_fp16_fp32 | 115.682 | 115.689 | 115.952 (fixed 0%) | 115.682 (1×) | TFLOP/s |
| matrix_fp16_fp32_feed_cache | 105.521 | 105.993 | 106.235 (fixed 1%) | — | TFLOP/s |
| matrix_fp16_fp32_feed_dram | 440.008 | 440.969 | 445.225 (fixed 1%) | — | GB/s |
| matrix_fp16_fp32_feed_shared | 106.353 | 106.469 | 106.855 (fixed 0%) | — | TFLOP/s |
| matrix_int8 | 231.363 | 231.383 | 231.909 (fixed 0%) | — | TOP/s |
| matrix_int8_feed_cache | 198.040 | 198.926 | 198.185 (fixed 0%) | — | TOP/s |
| matrix_int8_feed_dram | 436.588 | 437.833 | 442.754 (fixed 1%) | — | GB/s |
| matrix_int8_feed_shared | 206.388 | 206.616 | 212.244 (fixed 3%) | — | TOP/s |
| shared_fp16_read | 3447.914 | 3448.363 | 3650.177 (fixed 6%) | — | GB/s |
| shared_fp16_write | 4102.190 | 4102.580 | 4183.288 (fixed 2%) | — | GB/s |
| shared_fp32_read | 7258.691 | 7260.768 | 7267.753 (fixed 0%) | — | GB/s |
| shared_fp32_write | 6147.160 | 6147.531 | 6151.263 (fixed 0%) | — | GB/s |
| texture_rgba16f_buffer_cache | 1519.657 | 1520.154 | 1519.947 (fixed 0%) | — | GB/s |
| texture_rgba16f_buffer_dram | 421.223 | 421.601 | 421.685 (fixed 0%) | — | GB/s |
| texture_rgba16f_tex2d_cache | 1353.622 | 1357.882 | 1354.810 (fixed 0%) | — | GB/s |
| texture_rgba16f_tex2d_dram | 260.520 | 262.826 | 259.900 (fixed -0%) | — | GB/s |
| texture_rgba16f_tex3d_cache | 1340.619 | 1342.911 | 1341.153 (fixed 0%) | — | GB/s |
| texture_rgba16f_tex3d_dram | 441.185 | 442.405 | 441.584 (fixed 0%) | — | GB/s |
| texture_rgba32f_buffer_cache | 1547.258 | 1547.572 | 1547.627 (fixed 0%) | — | GB/s |
| texture_rgba32f_buffer_dram | 422.396 | 422.613 | 422.603 (fixed 0%) | — | GB/s |
| texture_rgba32f_tex2d_cache | 1440.759 | 1444.030 | 1441.984 (fixed 0%) | — | GB/s |
| texture_rgba32f_tex2d_dram | 403.259 | 404.713 | 404.223 (fixed 0%) | — | GB/s |
| texture_rgba32f_tex3d_cache | 1378.054 | 1380.541 | 1377.630 (fixed -0%) | — | GB/s |
| texture_rgba32f_tex3d_dram | 573.090 | 576.816 | 575.148 (fixed 0%) | — | GB/s |

## Roof confirmation

Each roof's top candidates were re-measured in fresh processes, round-robin with alternating order. The roof is the median of the best candidate's repeats; the sweep maximum (a single run) is shown for comparison. Roofs without a quality-passing candidate (standard error of the median <= 3 %, not short, fixed cost <= 10 %) are marked unconfirmed.

| roof | confirmed median | repeat range | repeats | sweep max (unconfirmed) |
|---|---:|---:|---:|---:|
| alu_fp16 | 27.295 | 27.295–27.295 (0.0%) | 3 | 27.295 |
| alu_fp32 | 12.912 | 12.911–12.912 (0.0%) | 3 | 12.912 |
| cache_read_effective | 6020.517 | 6019.646–6020.656 (0.0%) | 3 | 6020.051 |
| dot_int8 | 32.060 | 32.046–32.062 (0.1%) | 3 | 32.060 |
| global_copy | 408.141 | 407.553–408.486 (0.2%) | 3 | 408.940 |
| global_read | 465.197 | 465.138–465.331 (0.0%) | 3 | 465.440 |
| global_triad | 393.197 | 392.939–393.248 (0.1%) | 3 | 393.353 |
| global_write | 403.182 | 402.164–404.398 (0.6%) | 3 | 403.007 |
| matrix_fp16 | 112.918 | 112.898–112.919 (0.0%) | 3 | 112.920 |
| matrix_fp16_feed_cache | 106.961 | 106.928–107.015 (0.1%) | 3 | 107.050 |
| matrix_fp16_feed_dram | 444.238 | 444.223–444.323 (0.0%) | 3 | 444.235 |
| matrix_fp16_feed_shared | 109.728 | 109.726–109.734 (0.0%) | 3 | 109.741 |
| matrix_fp16_fp32 | 115.682 | 115.681–115.683 (0.0%) | 3 | 115.682 |
| matrix_fp16_fp32_feed_cache | 105.521 | 105.455–105.625 (0.2%) | 3 | 105.197 |
| matrix_fp16_fp32_feed_dram | 440.008 | 439.854–440.440 (0.1%) | 3 | 439.927 |
| matrix_fp16_fp32_feed_shared | 106.353 | 106.345–106.359 (0.0%) | 3 | 106.329 |
| matrix_int8 | 231.363 | 231.352–231.365 (0.0%) | 3 | 231.369 |
| matrix_int8_feed_cache | 198.040 | 196.934–198.071 (0.6%) | 3 | 198.298 |
| matrix_int8_feed_dram | 436.588 | 434.179–436.793 (0.6%) | 3 | 436.858 |
| matrix_int8_feed_shared | 206.388 | 206.372–206.392 (0.0%) | 3 | 206.285 |
| shared_fp16_read | 3447.914 | 3447.839–3447.960 (0.0%) | 3 | 3447.613 |
| shared_fp16_write | 4102.190 | 4102.145–4102.217 (0.0%) | 3 | 4102.181 |
| shared_fp32_read | 7258.691 | 7258.265–7258.797 (0.0%) | 3 | 7258.957 |
| shared_fp32_write | 6147.160 | 6147.160–6147.210 (0.0%) | 3 | 6146.901 |
| texture_rgba16f_buffer_cache | 1519.657 | 1519.644–1519.765 (0.0%) | 3 | 1519.696 |
| texture_rgba16f_buffer_dram | 421.223 | 421.172–421.255 (0.0%) | 3 | 421.248 |
| texture_rgba16f_tex2d_cache | 1353.622 | 1353.231–1353.721 (0.0%) | 3 | 1406.992 |
| texture_rgba16f_tex2d_dram | 260.520 | 260.186–260.597 (0.2%) | 3 | 259.998 |
| texture_rgba16f_tex3d_cache | 1340.619 | 1340.584–1341.125 (0.0%) | 3 | 1341.025 |
| texture_rgba16f_tex3d_dram | 441.185 | 441.118–441.370 (0.1%) | 3 | 441.474 |
| texture_rgba32f_buffer_cache | 1547.258 | 1547.255–1547.258 (0.0%) | 3 | 1547.295 |
| texture_rgba32f_buffer_dram | 422.396 | 422.389–422.436 (0.0%) | 3 | 422.385 |
| texture_rgba32f_tex2d_cache | 1440.759 | 1440.166–1441.182 (0.1%) | 3 | 1440.677 |
| texture_rgba32f_tex2d_dram | 403.259 | 402.972–403.376 (0.1%) | 3 | 402.838 |
| texture_rgba32f_tex3d_cache | 1378.054 | 1377.917–1378.205 (0.0%) | 3 | 1377.934 |
| texture_rgba32f_tex3d_dram | 573.090 | 572.522–574.298 (0.3%) | 3 | 572.858 |

## Cooperative matrix fed from memory, by reuse

Best validated median per source and CHAINS (multiply-adds per loaded A/B tile pair). `load GB/s` counts the A/B tile bytes loaded. DRAM-fed rates grow with reuse until the matrix unit limits them, so the `matrix_*_feed_dram` roof above is a bandwidth; look a kernel's ops per loaded byte up here instead. `gates` lists quality gates the row failed (such rows never define a roof).

| dtype | source | CHAINS | ops / loaded byte | rate | load GB/s | gates |
|---|---|---:|---:|---:|---:|---|
| fp16 | shared | 8 | 42.7 | 109.741 TFLOP/s | 2572.1 | — |
| fp16 | cache | 1 | 5.3 | 17.989 TFLOP/s | 3372.9 | — |
| fp16 | cache | 2 | 10.7 | 35.350 TFLOP/s | 3314.0 | — |
| fp16 | cache | 4 | 21.3 | 67.699 TFLOP/s | 3173.4 | — |
| fp16 | cache | 8 | 42.7 | 107.050 TFLOP/s | 2509.0 | — |
| fp16 | dram | 1 | 5.3 | 2.321 TFLOP/s | 435.3 | — |
| fp16 | dram | 2 | 10.7 | 4.565 TFLOP/s | 428.0 | — |
| fp16 | dram | 4 | 21.3 | 8.893 TFLOP/s | 416.9 | — |
| fp16 | dram | 8 | 42.7 | 17.149 TFLOP/s | 401.9 | — |
| fp16_fp32 | shared | 1 | 5.3 | 27.519 TFLOP/s | 5159.7 | — |
| fp16_fp32 | shared | 2 | 10.7 | 53.544 TFLOP/s | 5019.7 | — |
| fp16_fp32 | shared | 4 | 21.3 | 101.937 TFLOP/s | 4778.3 | — |
| fp16_fp32 | shared | 8 | 42.7 | 106.359 TFLOP/s | 2492.8 | — |
| fp16_fp32 | cache | 1 | 5.3 | 18.562 TFLOP/s | 3480.3 | — |
| fp16_fp32 | cache | 2 | 10.7 | 36.574 TFLOP/s | 3428.8 | — |
| fp16_fp32 | cache | 4 | 21.3 | 70.162 TFLOP/s | 3288.8 | — |
| fp16_fp32 | cache | 8 | 42.7 | 105.625 TFLOP/s | 2475.6 | — |
| fp16_fp32 | dram | 1 | 5.3 | 2.330 TFLOP/s | 436.8 | — |
| fp16_fp32 | dram | 2 | 10.7 | 4.603 TFLOP/s | 431.5 | — |
| fp16_fp32 | dram | 4 | 21.3 | 9.029 TFLOP/s | 423.2 | — |
| fp16_fp32 | dram | 8 | 42.7 | 17.696 TFLOP/s | 414.7 | — |
| int8 | shared | 1 | 10.7 | 34.375 TOP/s | 3222.6 | — |
| int8 | shared | 2 | 21.3 | 72.886 TOP/s | 3416.5 | — |
| int8 | shared | 4 | 42.7 | 136.072 TOP/s | 3189.2 | — |
| int8 | shared | 8 | 85.3 | 206.392 TOP/s | 2418.7 | — |
| int8 | cache | 1 | 10.7 | 24.153 TOP/s | 2264.3 | — |
| int8 | cache | 2 | 21.3 | 49.053 TOP/s | 2299.4 | — |
| int8 | cache | 4 | 42.7 | 99.686 TOP/s | 2336.4 | — |
| int8 | cache | 8 | 85.3 | 198.298 TOP/s | 2323.8 | — |
| int8 | dram | 1 | 10.7 | 4.622 TOP/s | 433.3 | — |
| int8 | dram | 2 | 21.3 | 9.140 TOP/s | 428.4 | — |
| int8 | dram | 4 | 42.7 | 17.946 TOP/s | 420.6 | — |
| int8 | dram | 8 | 85.3 | 35.335 TOP/s | 414.1 | — |

## Device-state sentinel

`alu_fp32_v4_c16 wg256 groups512` measured before and after every stage and every 20 configurations. Values below 85 % of the median reading (11.679 TFLOP/s) mark a stage that ran on a throttled or otherwise degraded device; re-measure those stages.

| UTC | label | TFLOP/s | state |
|---|---|---:|---|
| 2026-10-05T07:40:08 | validate_start | 11.679 | ok |
| 2026-10-05T07:40:21 | validate_end | 11.679 | ok |
| 2026-10-05T07:40:58 | cache_end | 11.679 | ok |
| 2026-10-05T07:41:07 | sweep-memory_20 | 11.679 | ok |
| 2026-10-05T07:41:19 | memory_end | 11.679 | ok |
| 2026-10-05T07:41:42 | sweep-compute_40 | 11.679 | ok |
| 2026-10-05T07:42:14 | sweep-compute_60 | 11.679 | ok |
| 2026-10-05T07:42:46 | sweep-compute_80 | 11.679 | ok |
| 2026-10-05T07:43:19 | sweep-compute_100 | 11.679 | ok |
| 2026-10-05T07:43:51 | sweep-compute_120 | 11.679 | ok |
| 2026-10-05T07:44:25 | sweep-compute_140 | 11.679 | ok |
| 2026-10-05T07:44:58 | sweep-compute_160 | 11.679 | ok |
| 2026-10-05T07:45:22 | compute_end | 11.679 | ok |
| 2026-10-05T07:45:32 | sweep-matrix-feed_180 | 11.679 | ok |
| 2026-10-05T07:46:06 | sweep-matrix-feed_200 | 11.679 | ok |
| 2026-10-05T07:46:23 | matrix_feed_end | 11.679 | ok |
| 2026-10-05T07:46:42 | sweep-texture_220 | 11.680 | ok |
| 2026-10-05T07:46:46 | texture_end | 11.679 | ok |
| 2026-10-05T07:47:18 | sweep-shared_240 | 11.679 | ok |
| 2026-10-05T07:47:50 | sweep-shared_260 | 11.679 | ok |
| 2026-10-05T07:48:19 | shared_end | 11.679 | ok |
| 2026-10-05T07:48:23 | latency-capacity_280 | 11.679 | ok |
| 2026-10-05T07:48:58 | latency_end | 11.679 | ok |
| 2026-10-05T07:49:19 | confirm_300 | 11.679 | ok |
| 2026-10-05T07:49:54 | confirm_320 | 11.679 | ok |
| 2026-10-05T07:50:33 | confirm_340 | 11.679 | ok |
| 2026-10-05T07:51:07 | confirm_360 | 11.679 | ok |
| 2026-10-05T07:51:46 | confirm_380 | 11.679 | ok |
| 2026-10-05T07:52:20 | confirm_400 | 11.679 | ok |
| 2026-10-05T07:52:53 | confirm_420 | 11.679 | ok |
| 2026-10-05T07:53:32 | confirm_440 | 11.679 | ok |
| 2026-10-05T07:54:07 | confirm_460 | 11.679 | ok |
| 2026-10-05T07:54:12 | confirm_end | 11.679 | ok |
| 2026-10-05T07:54:13 | sustain_0 | 11.679 | ok |

## Ridge points (short-run roofs)

Arithmetic intensity (ops per byte of that level) at which each compute roof meets each memory roof.

| compute roof | global | cache | shared_fp32 | shared_fp16 |
|---|---:|---:|---:|---:|
| alu_fp16 | 58.67 | 4.53 | 3.76 | 6.65 |
| alu_fp32 | 27.76 | 2.14 | 1.78 | 3.15 |
| dot_int8 | 68.92 | 5.33 | 4.42 | 7.82 |
| matrix_fp16 | 242.73 | 18.76 | 15.56 | 27.53 |
| matrix_fp16_feed_cache | 229.93 | 17.77 | 14.74 | 26.07 |
| matrix_fp16_feed_shared | 235.87 | 18.23 | 15.12 | 26.75 |
| matrix_fp16_fp32 | 248.67 | 19.21 | 15.94 | 28.20 |
| matrix_fp16_fp32_feed_cache | 226.83 | 17.53 | 14.54 | 25.72 |
| matrix_fp16_fp32_feed_shared | 228.62 | 17.67 | 14.65 | 25.93 |
| matrix_int8 | 497.34 | 38.43 | 31.87 | 56.40 |
| matrix_int8_feed_cache | 425.71 | 32.89 | 27.28 | 48.28 |
| matrix_int8_feed_shared | 443.66 | 34.28 | 28.43 | 50.31 |

## Figures

![roofline](roofline.png)

![working set](working-set.png)

![shared stride](shared-stride.png)

![sustained](sustained-trends.png)

See also [TUNING.md](TUNING.md), [SUPPLEMENT.md](SUPPLEMENT.md) (latency, TLB, line size, ERT, memory type) and [ISA-CHECK.md](ISA-CHECK.md).

## Limits

- Bandwidths are shader-logical bytes; physical DRAM/L2 traffic is not measurable without `VK_KHR_performance_query` (exposed: False).
- No vendor theoretical peaks are used; values are achievable rates on this device and driver.
- Cache knees move with workgroup size and are not cache capacities; see the pointer-chase results.
- Sustained values need 3 batches (plan `gold`) before they replace short-run roofs.
