# Radeon RX 7900 XTX roofline report

Device: host 13th Gen Intel(R) Core(TM) i9-13900KS (SoC 13th Gen Intel(R) Core(TM) i9-13900KS), Ubuntu 25.04, driver 8388957, subgroup 64.
Plan(s): fast, quick. GPU clock **DVFS-governed** (not pinned); results depend on the governor and thermal state.
Code: None, runner 1aead0d970b265f7. Rows from other runner or shader builds excluded: 0.

Validation policy: **pre_and_post**; checks bracket continuous sampling and do not guarantee detection of transient errors between checks. Historical pre-only rows are diagnostic only.
Excluded rows: 134; reasons are in all-configurations.csv.
matrix_int8_feed_dram: repeat_unstable
matrix_fp16_fp32_feed_dram: insufficient_quality_repeats
matrix_fp16_fp32_feed_dram: repeat_unstable
Sustained roofs require three distinct stable batches per current confirmed configuration and duration

Short-run columns are the best validated configuration: median and best (minimum time, the STREAM/BabelStream convention) of the samples, and the differential rate (paired L vs L/2 runs, removing fixed per-dispatch cost). Sustained is the median of the last 60 s of each sustained run (durations are recorded in sustained-runs.csv). Controls never define a roof. `spill` marks variants whose driver statistics report register spilling: spills only slow a kernel, so the value is still an achievable lower bound.

| roof | short median | short best | differential | sustained | unit |
|---|---:|---:|---:|---:|---|
| alu_fp16 | 63.447 | 65.971 | 63.841 (fixed 1%) | 64.622 (1×) | TFLOP/s |
| alu_fp32 | 45.562 | 47.321 | 45.698 (fixed 0%) | — | TFLOP/s |
| cache_read_effective | 24989.117 | 26394.822 | 25886.104 (fixed 3%) | — | GB/s |
| dot_int8 | 69.243 | 70.959 | 69.205 (fixed -0%) | — | TOP/s |
| global_add | 778.136 | 787.721 | 783.448 (fixed 1%) | — | GB/s |
| global_copy | 909.494 | 922.059 | 907.281 (fixed -0%) | — | GB/s |
| global_dot | 1626.389 | 1654.787 | 1659.519 (fixed 2%) | — | GB/s |
| global_read | 1108.158 | 1113.810 | 1123.125 (fixed 1%) | 1104.483 (1×) | GB/s |
| global_scale | 909.759 | 922.116 | 907.232 (fixed -0%) | — | GB/s |
| global_triad | 778.587 | 785.401 | 783.357 (fixed 1%) | — | GB/s |
| global_write | 1615.850 | 1689.997 | 1652.114 (fixed 2%) | — | GB/s |
| matrix_fp16 | 136.403 | 139.591 | 141.270 (fixed 3%) | — | TFLOP/s |
| matrix_fp16_feed_cache | 78.219 | 80.009 | 80.603 (fixed 3%) | — | TFLOP/s |
| matrix_fp16_feed_dram | 879.095 | 882.600 | 925.156 (fixed 5%) | — | GB/s |
| matrix_fp16_feed_shared | 127.499 | 129.684 | 137.325 (fixed 8%) | — | TFLOP/s |
| matrix_fp16_fp32 | 141.909 | 143.800 | 142.367 (fixed 0%) | 138.669 (1×) | TFLOP/s |
| matrix_fp16_fp32_feed_cache | 76.150 | 78.078 | 76.796 (fixed 1%) | — | TFLOP/s |
| matrix_fp16_fp32_feed_dram | 986.666 | 989.966 | 1052.005 (fixed 6%) | — | GB/s |
| matrix_fp16_fp32_feed_shared | 136.098 | 138.012 | 136.509 (fixed 0%) | — | TFLOP/s |
| matrix_int8 | 142.621 | 143.998 | 142.520 (fixed -0%) | — | TOP/s |
| matrix_int8_feed_cache | 84.835 | 86.361 | 85.596 (fixed 1%) | — | TOP/s |
| matrix_int8_feed_dram | 1098.904 | 1127.411 | 1057.630 (fixed -4%) | — | GB/s |
| matrix_int8_feed_shared | 113.856 | 115.506 | 114.562 (fixed 1%) | — | TOP/s |
| shared_fp16_write | 17385.473 | 17818.890 | 18915.528 (fixed 8%) | — | GB/s |
| shared_fp32_read | 21191.973 | 22011.850 | 21218.801 (fixed 0%) | — | GB/s |
| shared_fp32_write | 26098.515 | 26527.698 | 26059.917 (fixed -0%) | — | GB/s |
| texture_rgba16f_buffer_cache | 1423.280 | 1428.553 | 1441.435 (fixed 1%) | — | GB/s |
| texture_rgba16f_buffer_dram | 868.042 | 870.819 | 871.051 (fixed 0%) | — | GB/s |
| texture_rgba16f_tex2d_cache | 1407.660 | 1412.839 | 1425.322 (fixed 1%) | — | GB/s |
| texture_rgba16f_tex2d_dram | 813.975 | 835.978 | 839.527 (fixed 3%) | — | GB/s |
| texture_rgba16f_tex3d_cache | 1412.088 | 1417.803 | 1429.954 (fixed 1%) | — | GB/s |
| texture_rgba16f_tex3d_dram | 708.965 | 713.292 | 720.244 (fixed 2%) | — | GB/s |
| texture_rgba32f_buffer_cache | 1430.345 | 1436.273 | 1447.114 (fixed 1%) | — | GB/s |
| texture_rgba32f_buffer_dram | 892.943 | 894.231 | 894.280 (fixed 0%) | — | GB/s |
| texture_rgba32f_tex2d_cache | 1425.592 | 1433.721 | 1443.910 (fixed 1%) | — | GB/s |
| texture_rgba32f_tex2d_dram | 809.229 | 832.847 | 832.811 (fixed 3%) | — | GB/s |
| texture_rgba32f_tex3d_cache | 1424.745 | 1517.613 | 1444.203 (fixed 1%) | — | GB/s |
| texture_rgba32f_tex3d_dram | 816.936 | 823.113 | 834.163 (fixed 2%) | — | GB/s |
| global_copy_hostcoherent (control) | 908.940 | 916.969 | 906.595 (fixed -0%) | — | GB/s |
| global_read_hostcoherent (control) | 1108.433 | 1117.215 | 1122.264 (fixed 1%) | — | GB/s |
| global_write_hostcoherent (control) | 1608.190 | 1692.425 | 1660.791 (fixed 3%) | — | GB/s |

## Roof confirmation

Each roof's top candidates were re-measured in fresh processes, round-robin with alternating order. The roof is the median of the best candidate's repeats; the sweep maximum (a single run) is shown for comparison. Roofs without a quality-passing candidate (standard error of the median <= 3 %, not short, fixed cost <= 10 %) are marked unconfirmed.

| roof | confirmed median | repeat range | repeats | sweep max (unconfirmed) |
|---|---:|---:|---:|---:|
| alu_fp16 | 63.447 | 63.152–64.730 (2.5%) | 3 | 63.303 |
| alu_fp32 | 45.562 | 45.429–46.277 (1.9%) | 3 | 45.480 |
| cache_read_effective | 24989.117 | 24925.746–25721.785 (3.2%) | 3 | 24902.624 |
| dot_int8 | 69.243 | 68.811–69.850 (1.5%) | 3 | 68.766 |
| global_add | 778.136 | 777.735–778.437 (0.1%) | 3 | 778.161 |
| global_copy | 909.494 | 909.156–909.612 (0.1%) | 3 | 909.364 |
| global_dot | 1626.389 | 1622.567–1627.584 (0.3%) | 3 | 1630.211 |
| global_read | 1108.158 | 1107.915–1108.822 (0.1%) | 3 | 1109.241 |
| global_scale | 909.759 | 909.041–910.474 (0.2%) | 3 | 910.051 |
| global_triad | 778.587 | 778.246–778.688 (0.1%) | 3 | 778.196 |
| global_write | 1615.850 | 1595.366–1629.715 (2.1%) | 3 | 1657.164 |
| matrix_fp16 | 136.403 | 136.355–137.762 (1.0%) | 3 | 136.560 |
| matrix_fp16_feed_cache | 78.219 | 78.154–78.655 (0.6%) | 3 | 77.684 |
| matrix_fp16_feed_dram | 879.095 | 878.676–879.515 (0.1%) | 2 | 878.873 |
| matrix_fp16_feed_shared | 127.499 | 126.766–128.233 (1.2%) | 2 | 126.548 |
| matrix_fp16_fp32 | 141.909 | 141.874–142.223 (0.2%) | 3 | 141.931 |
| matrix_fp16_fp32_feed_cache | 76.150 | 76.002–76.161 (0.2%) | 3 | 75.535 |
| matrix_fp16_fp32_feed_dram | unconfirmed | — | — | 986.666 |
| matrix_fp16_fp32_feed_shared | 136.098 | 135.774–136.248 (0.3%) | 3 | 135.215 |
| matrix_int8 | 142.621 | 141.757–142.628 (0.6%) | 3 | 141.871 |
| matrix_int8_feed_cache | 84.835 | 84.339–85.282 (1.1%) | 3 | 83.361 |
| matrix_int8_feed_dram | 1098.904 | 1083.043–1116.093 (3.0%) | 3 | 1119.978 |
| matrix_int8_feed_shared | 113.856 | 113.243–114.187 (0.8%) | 3 | 112.547 |
| shared_fp16_write | 17385.473 | 17294.942–17476.925 (1.0%) | 3 | 17382.105 |
| shared_fp32_read | 21191.973 | 21104.450–21332.556 (1.1%) | 3 | 21468.555 |
| shared_fp32_write | 26098.515 | 25952.446–26153.343 (0.8%) | 3 | 26214.800 |
| texture_rgba16f_buffer_cache | 1423.280 | 1422.920–1423.795 (0.1%) | 3 | 1471.433 |
| texture_rgba16f_buffer_dram | 868.042 | 867.613–868.141 (0.1%) | 3 | 868.241 |
| texture_rgba16f_tex2d_cache | 1407.660 | 1407.259–1408.064 (0.1%) | 3 | 1407.716 |
| texture_rgba16f_tex2d_dram | 813.975 | 813.818–815.849 (0.2%) | 3 | 814.463 |
| texture_rgba16f_tex3d_cache | 1412.088 | 1410.909–1412.328 (0.1%) | 3 | 1412.329 |
| texture_rgba16f_tex3d_dram | 708.965 | 707.834–709.121 (0.2%) | 3 | 709.207 |
| texture_rgba32f_buffer_cache | 1430.345 | 1429.396–1430.388 (0.1%) | 3 | 1430.809 |
| texture_rgba32f_buffer_dram | 892.943 | 892.845–893.397 (0.1%) | 3 | 892.670 |
| texture_rgba32f_tex2d_cache | 1425.592 | 1424.498–1425.830 (0.1%) | 3 | 1426.383 |
| texture_rgba32f_tex2d_dram | 809.229 | 807.974–809.257 (0.2%) | 3 | 809.453 |
| texture_rgba32f_tex3d_cache | 1424.745 | 1424.633–1425.357 (0.1%) | 3 | 1423.292 |
| texture_rgba32f_tex3d_dram | 816.936 | 816.100–817.152 (0.1%) | 3 | 817.650 |

## Cooperative matrix fed from memory, by reuse

Best validated median per source and CHAINS (multiply-adds per loaded A/B tile pair). `load GB/s` counts the A/B tile bytes loaded. DRAM-fed rates grow with reuse until the matrix unit limits them, so the `matrix_*_feed_dram` roof above is a bandwidth; look a kernel's ops per loaded byte up here instead. `gates` lists quality gates the row failed (such rows never define a roof).

| dtype | source | CHAINS | ops / loaded byte | rate | load GB/s | gates |
|---|---|---:|---:|---:|---:|---|
| fp16 | shared | 1 | 8.0 | 26.724 TFLOP/s | 3340.5 | — |
| fp16 | shared | 2 | 16.0 | 53.682 TFLOP/s | 3355.1 | — |
| fp16 | shared | 4 | 32.0 | 101.443 TFLOP/s | 3170.1 | — |
| fp16 | shared | 8 | 64.0 | 128.233 TFLOP/s | 2003.6 | — |
| fp16 | cache | 1 | 8.0 | 10.597 TFLOP/s | 1324.7 | — |
| fp16 | cache | 2 | 16.0 | 20.999 TFLOP/s | 1312.4 | — |
| fp16 | cache | 4 | 32.0 | 40.758 TFLOP/s | 1273.7 | — |
| fp16 | cache | 8 | 64.0 | 78.655 TFLOP/s | 1229.0 | — |
| fp16 | dram | 1 | 8.0 | 6.875 TFLOP/s | 859.4 | — |
| fp16 | dram | 2 | 16.0 | 13.299 TFLOP/s | 831.2 | — |
| fp16_fp32 | shared | 1 | 8.0 | 28.921 TFLOP/s | 3615.1 | — |
| fp16_fp32 | shared | 2 | 16.0 | 57.636 TFLOP/s | 3602.2 | — |
| fp16_fp32 | shared | 4 | 32.0 | 109.316 TFLOP/s | 3416.1 | — |
| fp16_fp32 | shared | 8 | 64.0 | 136.248 TFLOP/s | 2128.9 | — |
| fp16_fp32 | cache | 1 | 8.0 | 10.779 TFLOP/s | 1347.4 | — |
| fp16_fp32 | cache | 2 | 16.0 | 21.140 TFLOP/s | 1321.2 | — |
| fp16_fp32 | cache | 4 | 32.0 | 40.957 TFLOP/s | 1279.9 | — |
| fp16_fp32 | cache | 8 | 64.0 | 76.161 TFLOP/s | 1190.0 | — |
| fp16_fp32 | dram | 1 | 8.0 | 8.764 TFLOP/s | 1095.5 | — |
| fp16_fp32 | dram | 2 | 16.0 | 15.766 TFLOP/s | 985.4 | — |
| fp16_fp32 | dram | 4 | 32.0 | 24.897 TFLOP/s | 778.0 | — |
| fp16_fp32 | dram | 8 | 64.0 | 47.324 TFLOP/s | 739.4 | — |
| int8 | shared | 1 | 16.0 | 47.641 TOP/s | 2977.6 | — |
| int8 | shared | 2 | 32.0 | 72.552 TOP/s | 2267.2 | — |
| int8 | shared | 4 | 64.0 | 95.243 TOP/s | 1488.2 | — |
| int8 | shared | 8 | 128.0 | 114.187 TOP/s | 892.1 | — |
| int8 | cache | 1 | 16.0 | 21.751 TOP/s | 1359.5 | — |
| int8 | cache | 2 | 32.0 | 42.394 TOP/s | 1324.8 | — |
| int8 | cache | 4 | 64.0 | 68.907 TOP/s | 1076.7 | — |
| int8 | cache | 8 | 128.0 | 85.282 TOP/s | 666.3 | — |
| int8 | dram | 1 | 16.0 | 17.840 TOP/s | 1115.0 | — |
| int8 | dram | 2 | 32.0 | 36.445 TOP/s | 1138.9 | — |
| int8 | dram | 4 | 64.0 | 56.721 TOP/s | 886.3 | — |
| int8 | dram | 8 | 128.0 | 58.025 TOP/s | 453.3 | — |

## Device-state sentinel

`alu_fp32_v4_c16 wg256 groups512` measured before and after every stage and every 20 configurations. Values below 85 % of the median reading (23.621 TFLOP/s) mark a stage that ran on a throttled or otherwise degraded device; re-measure those stages.

| UTC | label | TFLOP/s | state |
|---|---|---:|---|
| 2026-09-27T17:09:31 | validate_start | 24.695 | ok |
| 2026-09-27T17:09:37 | validate_end | 24.598 | ok |
| 2026-09-27T17:10:07 | first-look_20 | 23.858 | ok |
| 2026-09-27T17:10:28 | first_look_end | 23.966 | ok |
| 2026-09-27T17:10:37 | sweep-cache_40 | 23.500 | ok |
| 2026-09-27T17:10:52 | cache_end | 23.342 | ok |
| 2026-09-27T17:11:09 | sweep-memory_60 | 23.378 | ok |
| 2026-09-27T17:11:20 | memory_end | 23.352 | ok |
| 2026-09-27T17:11:37 | sweep-compute_80 | 23.295 | ok |
| 2026-09-27T17:12:00 | sweep-compute_100 | 23.149 | ok |
| 2026-09-27T17:12:22 | sweep-compute_120 | 23.258 | ok |
| 2026-09-27T17:12:25 | compute_end | 23.229 | ok |
| 2026-09-27T17:12:45 | sweep-matrix-feed_140 | 23.049 | ok |
| 2026-09-27T17:13:08 | sweep-matrix-feed_160 | 23.086 | ok |
| 2026-09-27T17:13:32 | sweep-matrix-feed_180 | 23.018 | ok |
| 2026-09-27T17:13:49 | matrix_feed_end | 23.029 | ok |
| 2026-09-27T17:13:57 | sweep-texture_200 | 23.603 | ok |
| 2026-09-27T17:14:05 | texture_end | 22.938 | ok |
| 2026-09-27T17:15:02 | sweep-shared_220 | 23.535 | ok |
| 2026-09-27T17:15:16 | shared_end | 23.541 | ok |
| 2026-09-27T17:15:21 | latency-capacity_240 | 23.605 | ok |
| 2026-09-27T17:15:54 | latency_end | 23.494 | ok |
| 2026-09-27T17:16:13 | ert_260 | 22.932 | ok |
| 2026-09-27T17:16:40 | ert_280 | 22.803 | ok |
| 2026-09-27T17:16:48 | ert_end | 22.852 | ok |
| 2026-09-27T17:17:01 | memory_type_end | 22.878 | ok |
| 2026-09-27T17:17:11 | confirm_300 | 22.853 | ok |
| 2026-09-27T17:17:41 | confirm_320 | 22.947 | ok |
| 2026-09-27T17:18:05 | confirm_340 | 23.250 | ok |
| 2026-09-27T17:18:36 | confirm_360 | 23.108 | ok |
| 2026-09-27T17:18:59 | confirm_380 | 22.879 | ok |
| 2026-09-27T17:19:30 | confirm_400 | 23.141 | ok |
| 2026-09-27T17:19:38 | confirm_end | 22.922 | ok |
| 2026-09-27T17:19:41 | validate_start | 24.070 | ok |
| 2026-09-27T17:19:43 | validate_end | 23.776 | ok |
| 2026-09-27T17:20:15 | cache_end | 24.138 | ok |
| 2026-09-27T17:20:23 | sweep-memory_20 | 23.999 | ok |
| 2026-09-27T17:20:36 | memory_end | 23.872 | ok |
| 2026-09-27T17:20:59 | sweep-compute_40 | 23.660 | ok |
| 2026-09-27T17:21:31 | sweep-compute_60 | 23.964 | ok |
| 2026-09-27T17:22:03 | sweep-compute_80 | 23.629 | ok |
| 2026-09-27T17:22:37 | sweep-compute_100 | 23.900 | ok |
| 2026-09-27T17:23:09 | sweep-compute_120 | 23.743 | ok |
| 2026-09-27T17:23:43 | sweep-compute_140 | 23.531 | ok |
| 2026-09-27T17:24:17 | sweep-compute_160 | 24.038 | ok |
| 2026-09-27T17:24:41 | compute_end | 23.716 | ok |
| 2026-09-27T17:24:51 | sweep-matrix-feed_180 | 23.732 | ok |
| 2026-09-27T17:25:24 | sweep-matrix-feed_200 | 23.708 | ok |
| 2026-09-27T17:25:43 | matrix_feed_end | 23.536 | ok |
| 2026-09-27T17:26:11 | sweep-texture_220 | 24.055 | ok |
| 2026-09-27T17:26:15 | texture_end | 23.612 | ok |
| 2026-09-27T17:27:12 | sweep-shared_240 | 24.563 | ok |
| 2026-09-27T17:28:01 | sweep-shared_260 | 24.408 | ok |
| 2026-09-27T17:28:48 | shared_end | 24.301 | ok |
| 2026-09-27T17:28:52 | latency-capacity_280 | 24.437 | ok |
| 2026-09-27T17:29:39 | latency_end | 24.349 | ok |
| 2026-09-27T17:30:00 | confirm_300 | 23.910 | ok |
| 2026-09-27T17:30:40 | confirm_320 | 24.226 | ok |
| 2026-09-27T17:31:15 | confirm_340 | 23.703 | ok |
| 2026-09-27T17:32:06 | confirm_360 | 23.924 | ok |
| 2026-09-27T17:32:39 | confirm_380 | 23.811 | ok |
| 2026-09-27T17:33:13 | confirm_400 | 23.692 | ok |
| 2026-09-27T17:33:49 | confirm_end | 23.859 | ok |
| 2026-09-27T17:33:50 | sustain_0 | 23.757 | ok |

## Ridge points (short-run roofs)

Arithmetic intensity (ops per byte of that level) at which each compute roof meets each memory roof.

| compute roof | global | cache | shared_fp32 | shared_fp16 |
|---|---:|---:|---:|---:|
| alu_fp16 | 39.01 | 2.54 | 2.43 | 3.65 |
| alu_fp32 | 28.01 | 1.82 | 1.75 | 2.62 |
| dot_int8 | 42.57 | 2.77 | 2.65 | 3.98 |
| matrix_fp16 | 83.87 | 5.46 | 5.23 | 7.85 |
| matrix_fp16_feed_cache | 48.09 | 3.13 | 3.00 | 4.50 |
| matrix_fp16_feed_shared | 78.39 | 5.10 | 4.89 | 7.33 |
| matrix_fp16_fp32 | 87.25 | 5.68 | 5.44 | 8.16 |
| matrix_fp16_fp32_feed_cache | 46.82 | 3.05 | 2.92 | 4.38 |
| matrix_fp16_fp32_feed_shared | 83.68 | 5.45 | 5.21 | 7.83 |
| matrix_int8 | 87.69 | 5.71 | 5.46 | 8.20 |
| matrix_int8_feed_cache | 52.16 | 3.39 | 3.25 | 4.88 |
| matrix_int8_feed_shared | 70.01 | 4.56 | 4.36 | 6.55 |

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
