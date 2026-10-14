# Radeon RX 7900 XTX roofline report

Device: <gpu-host>, Ubuntu 25.04, driver 8388957, subgroup 64.
Plan(s): fast, quick. GPU clock **DVFS-governed** (not pinned); results depend on the governor and thermal state.
Code: None, runner 1aead0d970b265f7. Rows from other runner or shader builds excluded: 0.

Validation policy: **pre_and_post**; checks bracket continuous sampling and do not guarantee detection of transient errors between checks. Historical pre-only rows are diagnostic only.
Excluded rows: 127; reasons are in all-configurations.csv.
matrix_int8_feed_dram: repeat_unstable
shared_fp32_0: insufficient_quality_repeats, repeat_unstable
matrix_fp16_fp32_feed_dram: repeat_unstable
matrix_fp16_fp32_feed_dram: insufficient_quality_repeats
Sustained roofs require three distinct stable batches per current confirmed configuration and duration

Short-run columns are the best validated configuration: median and best (minimum time, the STREAM/BabelStream convention) of the samples, and the differential rate (paired L vs L/2 runs, removing fixed per-dispatch cost). Sustained is the median of the last 60 s of each sustained run (durations are recorded in sustained-runs.csv). Controls never define a roof. `spill` marks variants whose driver statistics report register spilling: spills only slow a kernel, so the value is still an achievable lower bound.

| roof | short median | short best | differential | sustained | unit |
|---|---:|---:|---:|---:|---|
| alu_fp16 | 65.044 | 65.734 | 65.135 (fixed 0%) | 65.676 (1×) | TFLOP/s |
| alu_fp32 | 46.842 | 47.723 | 47.539 (fixed 1%) | — | TFLOP/s |
| cache_read_effective | 25731.493 | 26037.217 | 26707.308 (fixed 4%) | — | GB/s |
| dot_int8 | 70.489 | 71.566 | 71.508 (fixed 1%) | — | TOP/s |
| global_add | 777.615 | 785.805 | 782.936 (fixed 1%) | — | GB/s |
| global_copy | 909.205 | 919.703 | 907.379 (fixed -0%) | — | GB/s |
| global_dot | 1588.869 | 1619.007 | 1625.453 (fixed 2%) | — | GB/s |
| global_read | 1109.441 | 1116.676 | 1129.342 (fixed 2%) | 1103.969 (1×) | GB/s |
| global_scale | 909.642 | 918.073 | 909.667 (fixed 0%) | — | GB/s |
| global_triad | 778.216 | 781.367 | 784.080 (fixed 1%) | — | GB/s |
| global_write | 1594.629 | 1682.656 | 1619.937 (fixed 2%) | — | GB/s |
| matrix_fp16 | 139.599 | 140.870 | 144.361 (fixed 3%) | — | TFLOP/s |
| matrix_fp16_feed_cache | 79.393 | 81.099 | 81.595 (fixed 3%) | — | TFLOP/s |
| matrix_fp16_feed_dram | 878.423 | 881.665 | 924.042 (fixed 5%) | — | GB/s |
| matrix_fp16_feed_shared | 129.947 | 131.909 | 140.784 (fixed 8%) | — | TFLOP/s |
| matrix_fp16_fp32 | 140.834 | 143.365 | 140.191 (fixed -0%) | 139.270 (1×) | TFLOP/s |
| matrix_fp16_fp32_feed_cache | 77.463 | 79.017 | 79.204 (fixed 2%) | — | TFLOP/s |
| matrix_fp16_fp32_feed_dram | 1109.759 | 1117.039 | 1154.914 (fixed 4%) | — | GB/s |
| matrix_fp16_fp32_feed_shared | 137.155 | 140.114 | 138.799 (fixed 1%) | — | TFLOP/s |
| matrix_int8 | 141.597 | 143.144 | 142.233 (fixed 0%) | — | TOP/s |
| matrix_int8_feed_cache | 86.450 | 90.738 | 87.526 (fixed 1%) | — | TOP/s |
| matrix_int8_feed_dram | 1110.535 | 1164.022 | 1181.578 (fixed 6%) | — | GB/s |
| matrix_int8_feed_shared | 114.714 | 116.328 | 115.999 (fixed 1%) | — | TOP/s |
| shared_fp16_write | 17627.702 | 17915.217 | 19190.401 (fixed 8%) | — | GB/s |
| shared_fp32_read | 21562.234 | 22216.624 | 22075.509 (fixed 2%) | — | GB/s |
| shared_fp32_write | 26129.079 | 26569.133 | 26356.566 (fixed 1%) | — | GB/s |
| texture_rgba16f_buffer_cache | 1420.639 | 1423.196 | 1439.222 (fixed 1%) | — | GB/s |
| texture_rgba16f_buffer_dram | 866.486 | 869.617 | 866.947 (fixed 0%) | — | GB/s |
| texture_rgba16f_tex2d_cache | 1409.523 | 1414.698 | 1428.156 (fixed 1%) | — | GB/s |
| texture_rgba16f_tex2d_dram | 815.383 | 841.577 | 841.397 (fixed 3%) | — | GB/s |
| texture_rgba16f_tex3d_cache | 1410.419 | 1413.240 | 1429.780 (fixed 1%) | — | GB/s |
| texture_rgba16f_tex3d_dram | 706.672 | 719.201 | 719.694 (fixed 2%) | — | GB/s |
| texture_rgba32f_buffer_cache | 1435.652 | 1437.054 | 1455.132 (fixed 1%) | — | GB/s |
| texture_rgba32f_buffer_dram | 885.444 | 888.153 | 885.190 (fixed -0%) | — | GB/s |
| texture_rgba32f_tex2d_cache | 1427.775 | 1428.999 | 1446.038 (fixed 1%) | — | GB/s |
| texture_rgba32f_tex2d_dram | 805.407 | 825.979 | 825.985 (fixed 2%) | — | GB/s |
| texture_rgba32f_tex3d_cache | 1427.900 | 1430.048 | 1447.290 (fixed 1%) | — | GB/s |
| texture_rgba32f_tex3d_dram | 812.989 | 818.994 | 827.219 (fixed 2%) | — | GB/s |
| global_copy_hostcoherent (control) | 909.353 | 917.571 | 908.854 (fixed -0%) | — | GB/s |
| global_read_hostcoherent (control) | 1107.301 | 1116.296 | 1125.557 (fixed 2%) | — | GB/s |
| global_write_hostcoherent (control) | 1583.098 | 1656.600 | 1602.812 (fixed 1%) | — | GB/s |

## Roof confirmation

Each roof's top candidates were re-measured in fresh processes, round-robin with alternating order. The roof is the median of the best candidate's repeats; the sweep maximum (a single run) is shown for comparison. Roofs without a quality-passing candidate (standard error of the median <= 3 %, not short, fixed cost <= 10 %) are marked unconfirmed.

| roof | confirmed median | repeat range | repeats | sweep max (unconfirmed) |
|---|---:|---:|---:|---:|
| alu_fp16 | 65.044 | 64.723–65.162 (0.7%) | 3 | 65.143 |
| alu_fp32 | 46.842 | 46.657–46.949 (0.6%) | 3 | 46.677 |
| cache_read_effective | 25731.493 | 25444.315–25766.533 (1.3%) | 3 | 25475.252 |
| dot_int8 | 70.489 | 70.353–70.553 (0.3%) | 3 | 70.484 |
| global_add | 777.615 | 777.525–778.261 (0.1%) | 3 | 778.131 |
| global_copy | 909.205 | 909.174–909.451 (0.0%) | 3 | 909.932 |
| global_dot | 1588.869 | 1579.524–1590.528 (0.7%) | 3 | 1580.736 |
| global_read | 1109.441 | 1105.779–1109.800 (0.4%) | 3 | 1107.755 |
| global_scale | 909.642 | 909.273–910.000 (0.1%) | 3 | 910.154 |
| global_triad | 778.216 | 777.410–778.382 (0.1%) | 3 | 777.765 |
| global_write | 1594.629 | 1591.410–1644.531 (3.3%) | 3 | 1613.648 |
| matrix_fp16 | 139.599 | 138.018–139.953 (1.4%) | 3 | 139.154 |
| matrix_fp16_feed_cache | 79.393 | 79.313–79.677 (0.5%) | 3 | 78.906 |
| matrix_fp16_feed_dram | 878.423 | 878.103–878.435 (0.0%) | 3 | 877.997 |
| matrix_fp16_feed_shared | 129.947 | 128.897–130.196 (1.0%) | 3 | 129.085 |
| matrix_fp16_fp32 | 140.834 | 140.354–141.105 (0.5%) | 3 | 141.721 |
| matrix_fp16_fp32_feed_cache | 77.463 | 77.224–77.626 (0.5%) | 3 | 76.555 |
| matrix_fp16_fp32_feed_dram | unconfirmed | — | — | 1109.759 |
| matrix_fp16_fp32_feed_shared | 137.155 | 137.104–138.106 (0.7%) | 3 | 136.380 |
| matrix_int8 | 141.597 | 140.897–141.638 (0.5%) | 3 | 141.914 |
| matrix_int8_feed_cache | 86.450 | 85.721–88.716 (3.5%) | 3 | 85.264 |
| matrix_int8_feed_dram | 1110.535 | 1105.435–1151.084 (4.1%) | 3 | 1109.008 |
| matrix_int8_feed_shared | 114.714 | 114.687–115.109 (0.4%) | 3 | 114.036 |
| shared_fp16_write | 17627.702 | 17530.620–17631.324 (0.6%) | 3 | 17609.791 |
| shared_fp32_read | 21562.234 | 21376.704–21620.771 (1.1%) | 3 | 21890.244 |
| shared_fp32_write | 26129.079 | 26085.637–26237.420 (0.6%) | 3 | 26177.851 |
| texture_rgba16f_buffer_cache | 1420.639 | 1419.621–1420.927 (0.1%) | 3 | 1419.745 |
| texture_rgba16f_buffer_dram | 866.486 | 853.043–866.565 (1.6%) | 3 | 867.040 |
| texture_rgba16f_tex2d_cache | 1409.523 | 1409.431–1410.042 (0.0%) | 3 | 1409.608 |
| texture_rgba16f_tex2d_dram | 815.383 | 812.630–815.762 (0.4%) | 3 | 813.186 |
| texture_rgba16f_tex3d_cache | 1410.419 | 1409.722–1410.533 (0.1%) | 3 | 1410.071 |
| texture_rgba16f_tex3d_dram | 706.672 | 706.392–707.664 (0.2%) | 3 | 707.147 |
| texture_rgba32f_buffer_cache | 1435.652 | 1435.633–1435.815 (0.0%) | 3 | 1435.633 |
| texture_rgba32f_buffer_dram | 885.444 | 885.403–885.472 (0.0%) | 3 | 885.066 |
| texture_rgba32f_tex2d_cache | 1427.775 | 1427.298–1427.804 (0.0%) | 3 | 1427.995 |
| texture_rgba32f_tex2d_dram | 805.407 | 803.705–806.215 (0.3%) | 3 | 804.941 |
| texture_rgba32f_tex3d_cache | 1427.900 | 1427.186–1428.015 (0.1%) | 3 | 1427.967 |
| texture_rgba32f_tex3d_dram | 812.989 | 812.664–813.783 (0.1%) | 3 | 813.226 |

## Cooperative matrix fed from memory, by reuse

Best validated median per source and CHAINS (multiply-adds per loaded A/B tile pair). `load GB/s` counts the A/B tile bytes loaded. DRAM-fed rates grow with reuse until the matrix unit limits them, so the `matrix_*_feed_dram` roof above is a bandwidth; look a kernel's ops per loaded byte up here instead. `gates` lists quality gates the row failed (such rows never define a roof).

| dtype | source | CHAINS | ops / loaded byte | rate | load GB/s | gates |
|---|---|---:|---:|---:|---:|---|
| fp16 | shared | 1 | 8.0 | 26.823 TFLOP/s | 3352.9 | — |
| fp16 | shared | 2 | 16.0 | 53.649 TFLOP/s | 3353.1 | — |
| fp16 | shared | 4 | 32.0 | 102.633 TFLOP/s | 3207.3 | — |
| fp16 | shared | 8 | 64.0 | 130.196 TFLOP/s | 2034.3 | — |
| fp16 | cache | 1 | 8.0 | 10.535 TFLOP/s | 1316.9 | — |
| fp16 | cache | 2 | 16.0 | 21.076 TFLOP/s | 1317.3 | — |
| fp16 | cache | 4 | 32.0 | 41.273 TFLOP/s | 1289.8 | — |
| fp16 | cache | 8 | 64.0 | 79.677 TFLOP/s | 1244.9 | — |
| fp16 | dram | 1 | 8.0 | 6.867 TFLOP/s | 858.3 | — |
| fp16 | dram | 2 | 16.0 | 13.310 TFLOP/s | 831.9 | — |
| fp16_fp32 | shared | 1 | 8.0 | 28.689 TFLOP/s | 3586.2 | — |
| fp16_fp32 | shared | 2 | 16.0 | 57.238 TFLOP/s | 3577.4 | — |
| fp16_fp32 | shared | 4 | 32.0 | 110.829 TFLOP/s | 3463.4 | — |
| fp16_fp32 | shared | 8 | 64.0 | 138.106 TFLOP/s | 2157.9 | — |
| fp16_fp32 | cache | 1 | 8.0 | 10.729 TFLOP/s | 1341.1 | — |
| fp16_fp32 | cache | 2 | 16.0 | 21.334 TFLOP/s | 1333.4 | — |
| fp16_fp32 | cache | 4 | 32.0 | 41.421 TFLOP/s | 1294.4 | — |
| fp16_fp32 | cache | 8 | 64.0 | 77.626 TFLOP/s | 1212.9 | — |
| fp16_fp32 | dram | 1 | 8.0 | 9.226 TFLOP/s | 1153.2 | — |
| fp16_fp32 | dram | 2 | 16.0 | 15.116 TFLOP/s | 944.8 | — |
| fp16_fp32 | dram | 4 | 32.0 | 28.241 TFLOP/s | 882.5 | — |
| fp16_fp32 | dram | 8 | 64.0 | 47.824 TFLOP/s | 747.3 | — |
| int8 | shared | 1 | 16.0 | 48.866 TOP/s | 3054.1 | — |
| int8 | shared | 2 | 32.0 | 73.305 TOP/s | 2290.8 | — |
| int8 | shared | 4 | 64.0 | 96.937 TOP/s | 1514.6 | — |
| int8 | shared | 8 | 128.0 | 115.109 TOP/s | 899.3 | — |
| int8 | cache | 1 | 16.0 | 22.262 TOP/s | 1391.4 | — |
| int8 | cache | 2 | 32.0 | 43.406 TOP/s | 1356.4 | — |
| int8 | cache | 4 | 64.0 | 69.615 TOP/s | 1087.7 | — |
| int8 | cache | 8 | 128.0 | 88.716 TOP/s | 693.1 | — |
| int8 | dram | 1 | 16.0 | 18.408 TOP/s | 1150.5 | — |
| int8 | dram | 2 | 32.0 | 36.639 TOP/s | 1145.0 | — |
| int8 | dram | 4 | 64.0 | 56.569 TOP/s | 883.9 | — |
| int8 | dram | 8 | 128.0 | 58.019 TOP/s | 453.3 | — |

## Device-state sentinel

`alu_fp32_v4_c16 wg256 groups512` measured before and after every stage and every 20 configurations. Values below 85 % of the median reading (23.927 TFLOP/s) mark a stage that ran on a throttled or otherwise degraded device; re-measure those stages.

| UTC | label | TFLOP/s | state |
|---|---|---:|---|
| 2026-10-09T02:59:21 | validate_start | 23.730 | ok |
| 2026-10-09T02:59:26 | validate_end | 23.929 | ok |
| 2026-10-09T02:59:55 | first-look_20 | 23.780 | ok |
| 2026-10-09T03:00:15 | first_look_end | 23.684 | ok |
| 2026-10-09T03:00:24 | sweep-cache_40 | 23.813 | ok |
| 2026-10-09T03:00:38 | cache_end | 23.620 | ok |
| 2026-10-09T03:00:55 | sweep-memory_60 | 23.906 | ok |
| 2026-10-09T03:01:05 | memory_end | 23.764 | ok |
| 2026-10-09T03:01:22 | sweep-compute_80 | 23.476 | ok |
| 2026-10-09T03:01:44 | sweep-compute_100 | 23.433 | ok |
| 2026-10-09T03:02:05 | sweep-compute_120 | 23.467 | ok |
| 2026-10-09T03:02:08 | compute_end | 23.412 | ok |
| 2026-10-09T03:02:28 | sweep-matrix-feed_140 | 23.374 | ok |
| 2026-10-09T03:02:51 | sweep-matrix-feed_160 | 23.308 | ok |
| 2026-10-09T03:03:14 | sweep-matrix-feed_180 | 23.255 | ok |
| 2026-10-09T03:03:30 | matrix_feed_end | 23.253 | ok |
| 2026-10-09T03:03:39 | sweep-texture_200 | 23.297 | ok |
| 2026-10-09T03:03:46 | texture_end | 23.439 | ok |
| 2026-10-09T03:04:44 | sweep-shared_220 | 23.900 | ok |
| 2026-10-09T03:04:57 | shared_end | 23.617 | ok |
| 2026-10-09T03:05:03 | latency-capacity_240 | 23.925 | ok |
| 2026-10-09T03:05:37 | latency_end | 23.913 | ok |
| 2026-10-09T03:05:55 | ert_260 | 23.363 | ok |
| 2026-10-09T03:06:23 | ert_280 | 23.157 | ok |
| 2026-10-09T03:06:30 | ert_end | 23.327 | ok |
| 2026-10-09T03:06:43 | memory_type_end | 23.279 | ok |
| 2026-10-09T03:06:52 | confirm_300 | 23.137 | ok |
| 2026-10-09T03:07:23 | confirm_320 | 23.485 | ok |
| 2026-10-09T03:07:47 | confirm_340 | 23.331 | ok |
| 2026-10-09T03:08:19 | confirm_360 | 23.363 | ok |
| 2026-10-09T03:08:41 | confirm_380 | 23.281 | ok |
| 2026-10-09T03:09:12 | confirm_400 | 23.638 | ok |
| 2026-10-09T03:09:20 | confirm_end | 23.413 | ok |
| 2026-10-09T03:09:24 | validate_start | 24.393 | ok |
| 2026-10-09T03:09:25 | validate_end | 24.482 | ok |
| 2026-10-09T03:09:57 | cache_end | 24.496 | ok |
| 2026-10-09T03:10:05 | sweep-memory_20 | 24.510 | ok |
| 2026-10-09T03:10:18 | memory_end | 24.482 | ok |
| 2026-10-09T03:10:40 | sweep-compute_40 | 24.268 | ok |
| 2026-10-09T03:11:11 | sweep-compute_60 | 24.406 | ok |
| 2026-10-09T03:11:42 | sweep-compute_80 | 24.277 | ok |
| 2026-10-09T03:12:15 | sweep-compute_100 | 24.632 | ok |
| 2026-10-09T03:12:47 | sweep-compute_120 | 24.115 | ok |
| 2026-10-09T03:13:20 | sweep-compute_140 | 24.265 | ok |
| 2026-10-09T03:13:53 | sweep-compute_160 | 24.326 | ok |
| 2026-10-09T03:14:16 | compute_end | 24.168 | ok |
| 2026-10-09T03:14:27 | sweep-matrix-feed_180 | 24.384 | ok |
| 2026-10-09T03:15:00 | sweep-matrix-feed_200 | 24.310 | ok |
| 2026-10-09T03:15:18 | matrix_feed_end | 24.202 | ok |
| 2026-10-09T03:15:45 | sweep-texture_220 | 24.376 | ok |
| 2026-10-09T03:15:50 | texture_end | 24.460 | ok |
| 2026-10-09T03:16:53 | sweep-shared_240 | 24.883 | ok |
| 2026-10-09T03:17:40 | sweep-shared_260 | 24.848 | ok |
| 2026-10-09T03:18:27 | shared_end | 24.955 | ok |
| 2026-10-09T03:18:31 | latency-capacity_280 | 25.107 | ok |
| 2026-10-09T03:19:19 | latency_end | 24.563 | ok |
| 2026-10-09T03:19:39 | confirm_300 | 24.164 | ok |
| 2026-10-09T03:20:23 | confirm_320 | 24.785 | ok |
| 2026-10-09T03:20:58 | confirm_340 | 24.292 | ok |
| 2026-10-09T03:21:42 | confirm_360 | 24.644 | ok |
| 2026-10-09T03:22:13 | confirm_380 | 24.291 | ok |
| 2026-10-09T03:22:47 | confirm_400 | 24.558 | ok |
| 2026-10-09T03:23:27 | confirm_end | 24.549 | ok |
| 2026-10-09T03:23:29 | sustain_0 | 24.322 | ok |

## Ridge points (short-run roofs)

Arithmetic intensity (ops per byte of that level) at which each compute roof meets each memory roof.

| compute roof | global | cache | shared_fp32 | shared_fp16 |
|---|---:|---:|---:|---:|
| alu_fp16 | 40.79 | 2.53 | 2.49 | 3.69 |
| alu_fp32 | 29.37 | 1.82 | 1.79 | 2.66 |
| dot_int8 | 44.20 | 2.74 | 2.70 | 4.00 |
| matrix_fp16 | 87.54 | 5.43 | 5.34 | 7.92 |
| matrix_fp16_feed_cache | 49.79 | 3.09 | 3.04 | 4.50 |
| matrix_fp16_feed_shared | 81.49 | 5.05 | 4.97 | 7.37 |
| matrix_fp16_fp32 | 88.32 | 5.47 | 5.39 | 7.99 |
| matrix_fp16_fp32_feed_cache | 48.58 | 3.01 | 2.96 | 4.39 |
| matrix_fp16_fp32_feed_shared | 86.01 | 5.33 | 5.25 | 7.78 |
| matrix_int8 | 88.80 | 5.50 | 5.42 | 8.03 |
| matrix_int8_feed_cache | 54.21 | 3.36 | 3.31 | 4.90 |
| matrix_int8_feed_shared | 71.94 | 4.46 | 4.39 | 6.51 |

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
