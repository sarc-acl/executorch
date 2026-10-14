# Intel(R) Arc(tm) B580 Graphics (BMG G21) roofline report

Device: host AMD Ryzen 5 9600X 6-Core Processor (SoC AMD Ryzen 5 9600X 6-Core Processor), Fedora Linux 44 (Workstation Edition), driver 109060099, subgroup 32.
Plan(s): fast. GPU clock **DVFS-governed** (not pinned); results depend on the governor and thermal state.
Code: None, runner 810e098c8abb2dfb. Rows from other runner or shader builds excluded: 0.

Validation policy: **pre_and_post**; checks bracket continuous sampling and do not guarantee detection of transient errors between checks. Historical pre-only rows are diagnostic only.
Excluded rows: 56; reasons are in all-configurations.csv.
Sustained roofs require three distinct stable batches per current confirmed configuration and duration

Short-run columns are the best validated configuration: median and best (minimum time, the STREAM/BabelStream convention) of the samples, and the differential rate (paired L vs L/2 runs, removing fixed per-dispatch cost). Sustained is the median of the last 60 s of each sustained run (durations are recorded in sustained-runs.csv). Controls never define a roof. `spill` marks variants whose driver statistics report register spilling: spills only slow a kernel, so the value is still an achievable lower bound.

| roof | short median | short best | differential | sustained | unit |
|---|---:|---:|---:|---:|---|
| alu_fp16 | 27.295 | 27.297 | 27.348 (fixed 0%) | 27.295 (1×) | TFLOP/s |
| alu_fp32 | 12.912 | 12.913 | 12.949 (fixed 0%) | — | TFLOP/s |
| cache_read_effective | 6020.256 | 6025.731 | 6031.735 (fixed 0%) | — | GB/s |
| dot_int8 | 32.061 | 32.063 | 32.149 (fixed 0%) | — | TOP/s |
| global_copy | 406.895 | 412.368 | 411.546 (fixed 1%) | — | GB/s |
| global_read | 465.013 | 466.889 | 471.279 (fixed 1%) | 465.130 (1×) | GB/s |
| global_triad | 393.773 | 397.129 | 394.414 (fixed 0%) | — | GB/s |
| global_write | 402.220 | 408.270 | 416.071 (fixed 3%) | — | GB/s |
| matrix_fp16 | 112.860 | 112.974 | 115.850 (fixed 3%) | — | TFLOP/s |
| matrix_fp16_feed_cache | 107.069 | 107.678 | 109.056 (fixed 2%) | — | TFLOP/s |
| matrix_fp16_feed_dram | 443.810 | 444.678 | 450.753 (fixed 2%) | — | GB/s |
| matrix_fp16_feed_shared | 109.723 | 109.857 | 113.678 (fixed 3%) | — | TFLOP/s |
| matrix_fp16_fp32 | 115.678 | 115.687 | 115.954 (fixed 0%) | 115.677 (1×) | TFLOP/s |
| matrix_fp16_fp32_feed_cache | 105.483 | 105.941 | 106.363 (fixed 1%) | — | TFLOP/s |
| matrix_fp16_fp32_feed_dram | 439.999 | 441.444 | 444.499 (fixed 1%) | — | GB/s |
| matrix_fp16_fp32_feed_shared | 106.404 | 106.531 | 106.863 (fixed 0%) | — | TFLOP/s |
| matrix_int8 | 231.363 | 231.383 | 231.905 (fixed 0%) | — | TOP/s |
| matrix_int8_feed_cache | 198.045 | 199.607 | 198.902 (fixed 0%) | — | TOP/s |
| matrix_int8_feed_dram | 436.123 | 437.353 | 439.837 (fixed 1%) | — | GB/s |
| matrix_int8_feed_shared | 206.453 | 206.768 | 211.897 (fixed 3%) | — | TOP/s |
| shared_fp16_read | 3450.718 | 3451.283 | 3649.095 (fixed 5%) | — | GB/s |
| shared_fp16_write | 4102.145 | 4102.509 | 4183.119 (fixed 2%) | — | GB/s |
| shared_fp32_read | 7258.691 | 7261.088 | 7267.326 (fixed 0%) | — | GB/s |
| shared_fp32_write | 6147.242 | 6147.456 | 6151.248 (fixed 0%) | — | GB/s |
| texture_rgba16f_buffer_cache | 1482.422 | 1482.853 | 1482.799 (fixed 0%) | — | GB/s |
| texture_rgba16f_buffer_dram | 421.198 | 421.558 | 421.399 (fixed 0%) | — | GB/s |
| texture_rgba16f_tex2d_cache | 1309.048 | 1310.085 | 1309.552 (fixed 0%) | — | GB/s |
| texture_rgba16f_tex2d_dram | 260.315 | 262.841 | 260.629 (fixed 0%) | — | GB/s |
| texture_rgba16f_tex3d_cache | 1348.535 | 1349.405 | 1349.007 (fixed 0%) | — | GB/s |
| texture_rgba16f_tex3d_dram | 440.962 | 442.748 | 440.873 (fixed -0%) | — | GB/s |
| texture_rgba32f_buffer_cache | 1493.371 | 1493.838 | 1493.814 (fixed 0%) | — | GB/s |
| texture_rgba32f_buffer_dram | 422.697 | 422.958 | 422.864 (fixed 0%) | — | GB/s |
| texture_rgba32f_tex2d_cache | 1406.565 | 1407.724 | 1406.943 (fixed 0%) | — | GB/s |
| texture_rgba32f_tex2d_dram | 402.354 | 404.042 | 402.493 (fixed 0%) | — | GB/s |
| texture_rgba32f_tex3d_cache | 1373.209 | 1373.995 | 1373.307 (fixed 0%) | — | GB/s |
| texture_rgba32f_tex3d_dram | 572.628 | 575.987 | 571.923 (fixed -0%) | — | GB/s |

## Roof confirmation

Each roof's top candidates were re-measured in fresh processes, round-robin with alternating order. The roof is the median of the best candidate's repeats; the sweep maximum (a single run) is shown for comparison. Roofs without a quality-passing candidate (standard error of the median <= 3 %, not short, fixed cost <= 10 %) are marked unconfirmed.

| roof | confirmed median | repeat range | repeats | sweep max (unconfirmed) |
|---|---:|---:|---:|---:|
| alu_fp16 | 27.295 | 27.295–27.295 (0.0%) | 3 | 27.288 |
| alu_fp32 | 12.912 | 12.911–12.912 (0.0%) | 3 | 12.911 |
| cache_read_effective | 6020.256 | 6018.879–6020.505 (0.0%) | 3 | 6018.675 |
| dot_int8 | 32.061 | 32.053–32.062 (0.0%) | 3 | 32.064 |
| global_copy | 406.895 | 406.090–407.109 (0.3%) | 3 | 407.528 |
| global_read | 465.013 | 464.992–465.210 (0.0%) | 3 | 464.950 |
| global_triad | 393.773 | 393.467–393.830 (0.1%) | 3 | 393.695 |
| global_write | 402.220 | 401.427–402.259 (0.2%) | 3 | 402.053 |
| matrix_fp16 | 112.860 | 112.851–112.884 (0.0%) | 3 | 112.862 |
| matrix_fp16_feed_cache | 107.069 | 107.024–107.137 (0.1%) | 3 | 107.091 |
| matrix_fp16_feed_dram | 443.810 | 443.803–443.857 (0.0%) | 3 | 443.616 |
| matrix_fp16_feed_shared | 109.723 | 109.720–109.738 (0.0%) | 3 | 109.740 |
| matrix_fp16_fp32 | 115.678 | 115.676–115.680 (0.0%) | 3 | 115.680 |
| matrix_fp16_fp32_feed_cache | 105.483 | 105.385–105.583 (0.2%) | 3 | 105.219 |
| matrix_fp16_fp32_feed_dram | 439.999 | 439.888–440.046 (0.0%) | 3 | 440.011 |
| matrix_fp16_fp32_feed_shared | 106.404 | 106.402–106.419 (0.0%) | 3 | 106.416 |
| matrix_int8 | 231.363 | 231.345–231.363 (0.0%) | 3 | 231.357 |
| matrix_int8_feed_cache | 198.045 | 197.185–198.289 (0.6%) | 3 | 196.813 |
| matrix_int8_feed_dram | 436.123 | 434.163–436.257 (0.5%) | 3 | 436.108 |
| matrix_int8_feed_shared | 206.453 | 206.337–206.521 (0.1%) | 3 | 206.446 |
| shared_fp16_read | 3450.718 | 3450.658–3450.718 (0.0%) | 3 | 3450.598 |
| shared_fp16_write | 4102.145 | 4102.083–4102.154 (0.0%) | 3 | 4102.154 |
| shared_fp32_read | 7258.691 | 7258.584–7258.904 (0.0%) | 3 | 7258.531 |
| shared_fp32_write | 6147.242 | 6147.232–6147.242 (0.0%) | 3 | 6147.242 |
| texture_rgba16f_buffer_cache | 1482.422 | 1482.396–1482.487 (0.0%) | 3 | 1482.422 |
| texture_rgba16f_buffer_dram | 421.198 | 421.151–421.230 (0.0%) | 3 | 421.198 |
| texture_rgba16f_tex2d_cache | 1309.048 | 1308.769–1309.069 (0.0%) | 3 | 1309.358 |
| texture_rgba16f_tex2d_dram | 260.315 | 259.951–260.404 (0.2%) | 3 | 260.038 |
| texture_rgba16f_tex3d_cache | 1348.535 | 1348.488–1348.656 (0.0%) | 3 | 1348.783 |
| texture_rgba16f_tex3d_dram | 440.962 | 440.957–441.051 (0.0%) | 3 | 441.379 |
| texture_rgba32f_buffer_cache | 1493.371 | 1493.307–1493.410 (0.0%) | 3 | 1493.480 |
| texture_rgba32f_buffer_dram | 422.697 | 422.686–422.700 (0.0%) | 3 | 422.740 |
| texture_rgba32f_tex2d_cache | 1406.565 | 1406.365–1406.638 (0.0%) | 3 | 1407.004 |
| texture_rgba32f_tex2d_dram | 402.354 | 402.283–402.574 (0.1%) | 3 | 402.875 |
| texture_rgba32f_tex3d_cache | 1373.209 | 1372.992–1373.340 (0.0%) | 3 | 1372.614 |
| texture_rgba32f_tex3d_dram | 572.628 | 572.549–572.863 (0.1%) | 3 | 572.356 |

## Cooperative matrix fed from memory, by reuse

Best validated median per source and CHAINS (multiply-adds per loaded A/B tile pair). `load GB/s` counts the A/B tile bytes loaded. DRAM-fed rates grow with reuse until the matrix unit limits them, so the `matrix_*_feed_dram` roof above is a bandwidth; look a kernel's ops per loaded byte up here instead. `gates` lists quality gates the row failed (such rows never define a roof).

| dtype | source | CHAINS | ops / loaded byte | rate | load GB/s | gates |
|---|---|---:|---:|---:|---:|---|
| fp16 | shared | 8 | 42.7 | 109.740 TFLOP/s | 2572.0 | — |
| fp16 | cache | 1 | 5.3 | 17.942 TFLOP/s | 3364.2 | — |
| fp16 | cache | 2 | 10.7 | 35.362 TFLOP/s | 3315.2 | — |
| fp16 | cache | 4 | 21.3 | 67.699 TFLOP/s | 3173.4 | — |
| fp16 | cache | 8 | 42.7 | 107.137 TFLOP/s | 2511.0 | — |
| fp16 | dram | 1 | 5.3 | 2.319 TFLOP/s | 434.8 | — |
| fp16 | dram | 2 | 10.7 | 4.559 TFLOP/s | 427.4 | — |
| fp16 | dram | 4 | 21.3 | 8.890 TFLOP/s | 416.7 | — |
| fp16 | dram | 8 | 42.7 | 17.125 TFLOP/s | 401.4 | — |
| fp16_fp32 | shared | 1 | 5.3 | 27.507 TFLOP/s | 5157.6 | — |
| fp16_fp32 | shared | 2 | 10.7 | 53.510 TFLOP/s | 5016.5 | — |
| fp16_fp32 | shared | 4 | 21.3 | 101.885 TFLOP/s | 4775.9 | — |
| fp16_fp32 | shared | 8 | 42.7 | 106.419 TFLOP/s | 2494.2 | — |
| fp16_fp32 | cache | 1 | 5.3 | 18.516 TFLOP/s | 3471.7 | — |
| fp16_fp32 | cache | 2 | 10.7 | 36.342 TFLOP/s | 3407.1 | — |
| fp16_fp32 | cache | 4 | 21.3 | 70.199 TFLOP/s | 3290.6 | — |
| fp16_fp32 | cache | 8 | 42.7 | 105.583 TFLOP/s | 2474.6 | — |
| fp16_fp32 | dram | 1 | 5.3 | 2.328 TFLOP/s | 436.4 | — |
| fp16_fp32 | dram | 2 | 10.7 | 4.603 TFLOP/s | 431.6 | — |
| fp16_fp32 | dram | 4 | 21.3 | 8.982 TFLOP/s | 421.0 | — |
| fp16_fp32 | dram | 8 | 42.7 | 17.652 TFLOP/s | 413.7 | — |
| int8 | shared | 1 | 10.7 | 34.209 TOP/s | 3207.1 | — |
| int8 | shared | 2 | 21.3 | 72.844 TOP/s | 3414.6 | — |
| int8 | shared | 4 | 42.7 | 135.855 TOP/s | 3184.1 | — |
| int8 | shared | 8 | 85.3 | 206.521 TOP/s | 2420.2 | — |
| int8 | cache | 1 | 10.7 | 24.736 TOP/s | 2319.0 | — |
| int8 | cache | 2 | 21.3 | 48.795 TOP/s | 2287.3 | — |
| int8 | cache | 4 | 42.7 | 99.558 TOP/s | 2333.4 | — |
| int8 | cache | 8 | 85.3 | 198.289 TOP/s | 2323.7 | — |
| int8 | dram | 1 | 10.7 | 4.616 TOP/s | 432.7 | — |
| int8 | dram | 2 | 21.3 | 9.129 TOP/s | 427.9 | — |
| int8 | dram | 4 | 42.7 | 17.830 TOP/s | 417.9 | — |
| int8 | dram | 8 | 85.3 | 35.651 TOP/s | 417.8 | — |

## Device-state sentinel

`alu_fp32_v4_c16 wg256 groups512` measured before and after every stage and every 20 configurations. Values below 85 % of the median reading (11.680 TFLOP/s) mark a stage that ran on a throttled or otherwise degraded device; re-measure those stages.

| UTC | label | TFLOP/s | state |
|---|---|---:|---|
| 2026-10-09T08:49:13 | validate_start | 11.680 | ok |
| 2026-10-09T08:49:21 | validate_end | 11.681 | ok |
| 2026-10-09T08:49:57 | cache_end | 11.680 | ok |
| 2026-10-09T08:50:06 | sweep-memory_20 | 11.680 | ok |
| 2026-10-09T08:50:18 | memory_end | 11.680 | ok |
| 2026-10-09T08:50:41 | sweep-compute_40 | 11.680 | ok |
| 2026-10-09T08:51:13 | sweep-compute_60 | 11.680 | ok |
| 2026-10-09T08:51:46 | sweep-compute_80 | 11.681 | ok |
| 2026-10-09T08:52:18 | sweep-compute_100 | 11.681 | ok |
| 2026-10-09T08:52:51 | sweep-compute_120 | 11.680 | ok |
| 2026-10-09T08:53:24 | sweep-compute_140 | 11.681 | ok |
| 2026-10-09T08:53:58 | sweep-compute_160 | 11.680 | ok |
| 2026-10-09T08:54:21 | compute_end | 11.681 | ok |
| 2026-10-09T08:54:32 | sweep-matrix-feed_180 | 11.680 | ok |
| 2026-10-09T08:55:05 | sweep-matrix-feed_200 | 11.680 | ok |
| 2026-10-09T08:55:23 | matrix_feed_end | 11.680 | ok |
| 2026-10-09T08:55:41 | sweep-texture_220 | 11.680 | ok |
| 2026-10-09T08:55:45 | texture_end | 11.680 | ok |
| 2026-10-09T08:56:16 | sweep-shared_240 | 11.680 | ok |
| 2026-10-09T08:56:49 | sweep-shared_260 | 11.680 | ok |
| 2026-10-09T08:57:18 | shared_end | 11.680 | ok |
| 2026-10-09T08:57:22 | latency-capacity_280 | 11.681 | ok |
| 2026-10-09T08:57:58 | latency_end | 11.680 | ok |
| 2026-10-09T08:58:18 | confirm_300 | 11.680 | ok |
| 2026-10-09T08:58:53 | confirm_320 | 11.680 | ok |
| 2026-10-09T08:59:32 | confirm_340 | 11.680 | ok |
| 2026-10-09T09:00:06 | confirm_360 | 11.680 | ok |
| 2026-10-09T09:00:46 | confirm_380 | 11.680 | ok |
| 2026-10-09T09:01:19 | confirm_400 | 11.681 | ok |
| 2026-10-09T09:01:53 | confirm_420 | 11.680 | ok |
| 2026-10-09T09:02:31 | confirm_440 | 11.681 | ok |
| 2026-10-09T09:03:07 | confirm_460 | 11.680 | ok |
| 2026-10-09T09:03:11 | confirm_end | 11.681 | ok |
| 2026-10-09T09:03:13 | sustain_0 | 11.680 | ok |

## Ridge points (short-run roofs)

Arithmetic intensity (ops per byte of that level) at which each compute roof meets each memory roof.

| compute roof | global | cache | shared_fp32 | shared_fp16 |
|---|---:|---:|---:|---:|
| alu_fp16 | 58.70 | 4.53 | 3.76 | 6.65 |
| alu_fp32 | 27.77 | 2.14 | 1.78 | 3.15 |
| dot_int8 | 68.95 | 5.33 | 4.42 | 7.82 |
| matrix_fp16 | 242.70 | 18.75 | 15.55 | 27.51 |
| matrix_fp16_feed_cache | 230.25 | 17.78 | 14.75 | 26.10 |
| matrix_fp16_feed_shared | 235.96 | 18.23 | 15.12 | 26.75 |
| matrix_fp16_fp32 | 248.76 | 19.21 | 15.94 | 28.20 |
| matrix_fp16_fp32_feed_cache | 226.84 | 17.52 | 14.53 | 25.71 |
| matrix_fp16_fp32_feed_shared | 228.82 | 17.67 | 14.66 | 25.94 |
| matrix_int8 | 497.54 | 38.43 | 31.87 | 56.40 |
| matrix_int8_feed_cache | 425.89 | 32.90 | 27.28 | 48.28 |
| matrix_int8_feed_shared | 443.97 | 34.29 | 28.44 | 50.33 |

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
