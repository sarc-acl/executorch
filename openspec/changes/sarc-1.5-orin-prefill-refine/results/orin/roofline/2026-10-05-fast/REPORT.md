# NVIDIA Tegra Orin (nvgpu) roofline report

Device: host  (SoC ), Ubuntu 24.04.5 LTS, driver 2496888832, subgroup 32.
Plan(s): fast. GPU clock state **unavailable** (frequency and pinning not verified).
Code: None, runner c7fba81beb1ecb83. Rows from other runner or shader builds excluded: 0.

Validation policy: **pre_and_post**; checks bracket continuous sampling and do not guarantee detection of transient errors between checks. Historical pre-only rows are diagnostic only.
Excluded rows: 86; reasons are in all-configurations.csv.
Sustained roofs require three distinct stable batches per current confirmed configuration and duration

Short-run columns are the best validated configuration: median and best (minimum time, the STREAM/BabelStream convention) of the samples, and the differential rate (paired L vs L/2 runs, removing fixed per-dispatch cost). Sustained is the median of the last 60 s of each sustained run (durations are recorded in sustained-runs.csv). Controls never define a roof. `spill` marks variants whose driver statistics report register spilling: spills only slow a kernel, so the value is still an achievable lower bound.

| roof | short median | short best | differential | sustained | unit |
|---|---:|---:|---:|---:|---|
| alu_fp16 | 1.762 | 1.769 | 1.771 (fixed 1%) | 1.762 (1×) | TFLOP/s |
| alu_fp32 | 1.201 | 1.207 | 1.229 (fixed 2%) | — | TFLOP/s |
| cache_read_effective | 203.928 | 204.135 | 205.352 (fixed 1%) | — | GB/s |
| dot_int8 | 2.090 | 2.099 | 2.106 (fixed 1%) | — | TOP/s |
| global_copy | 64.074 | 64.153 | 64.103 (fixed 0%) | — | GB/s |
| global_read | 62.107 | 62.489 | 65.099 (fixed 5%) | 62.112 (1×) | GB/s |
| global_triad | 64.302 | 64.411 | 65.507 (fixed 2%) | — | GB/s |
| global_write | 58.015 | 58.229 | 57.983 (fixed -0%) | — | GB/s |
| matrix_fp16 | 9.716 | 9.768 | 9.910 (fixed 2%) | — | TFLOP/s |
| matrix_fp16_feed_cache | 4.232 | 4.275 | 4.516 (fixed 6%) | — | TFLOP/s |
| matrix_fp16_feed_dram | 60.334 | 60.482 | 65.962 (fixed 9%) | — | GB/s |
| matrix_fp16_feed_shared | 9.511 | 9.560 | 9.870 (fixed 4%) | — | TFLOP/s |
| matrix_fp16_fp32 | 9.722 | 9.776 | 9.821 (fixed 1%) | 9.723 (1×) | TFLOP/s |
| matrix_fp16_fp32_feed_cache | 5.628 | 5.679 | 5.906 (fixed 5%) | — | TFLOP/s |
| matrix_fp16_fp32_feed_dram | 61.240 | 61.504 | 63.428 (fixed 3%) | — | GB/s |
| matrix_fp16_fp32_feed_shared | 8.771 | 8.776 | 9.354 (fixed 6%) | — | TFLOP/s |
| matrix_int8 | 19.482 | 19.507 | 19.638 (fixed 1%) | — | TOP/s |
| matrix_int8_feed_cache | 8.438 | 8.483 | 8.782 (fixed 4%) | — | TOP/s |
| matrix_int8_feed_dram | 59.272 | 59.437 | 64.704 (fixed 8%) | — | GB/s |
| matrix_int8_feed_shared | 17.818 | 17.912 | 18.671 (fixed 5%) | — | TOP/s |
| shared_fp16_read | 2061.882 | 2070.717 | 2227.908 (fixed 7%) | — | GB/s |
| shared_fp16_write | 611.343 | 614.861 | 623.675 (fixed 2%) | — | GB/s |
| shared_fp32_read | 8371.545 | 8377.086 | 8413.866 (fixed 1%) | — | GB/s |
| shared_fp32_write | 4278.042 | 4302.924 | 4284.430 (fixed 0%) | — | GB/s |
| texture_rgba16f_buffer_cache | 48.959 | 49.243 | 49.369 (fixed 1%) | — | GB/s |
| texture_rgba16f_tex2d_cache | 46.244 | 46.528 | 46.681 (fixed 1%) | — | GB/s |
| texture_rgba16f_tex2d_dram | 20.089 | 20.168 | 20.105 (fixed 0%) | — | GB/s |
| texture_rgba16f_tex3d_cache | 41.038 | 41.328 | 41.357 (fixed 1%) | — | GB/s |
| texture_rgba16f_tex3d_dram | 40.141 | 40.315 | 40.914 (fixed 2%) | — | GB/s |
| texture_rgba32f_buffer_cache | 38.147 | 38.387 | 38.428 (fixed 1%) | — | GB/s |
| texture_rgba32f_buffer_dram | 61.989 | 63.047 | 65.292 (fixed 5%) | — | GB/s |
| texture_rgba32f_tex2d_cache | 44.876 | 45.164 | 45.333 (fixed 1%) | — | GB/s |
| texture_rgba32f_tex2d_dram | 20.135 | 20.172 | 20.152 (fixed 0%) | — | GB/s |
| texture_rgba32f_tex3d_cache | 39.652 | 39.919 | 39.957 (fixed 1%) | — | GB/s |
| texture_rgba32f_tex3d_dram | 40.841 | 41.155 | 41.733 (fixed 2%) | — | GB/s |

## Roof confirmation

Each roof's top candidates were re-measured in fresh processes, round-robin with alternating order. The roof is the median of the best candidate's repeats; the sweep maximum (a single run) is shown for comparison. Roofs without a quality-passing candidate (standard error of the median <= 3 %, not short, fixed cost <= 10 %) are marked unconfirmed.

| roof | confirmed median | repeat range | repeats | sweep max (unconfirmed) |
|---|---:|---:|---:|---:|
| alu_fp16 | 1.762 | 1.762–1.763 (0.1%) | 3 | 1.762 |
| alu_fp32 | 1.201 | 1.201–1.202 (0.1%) | 3 | 1.202 |
| cache_read_effective | 203.928 | 203.883–204.046 (0.1%) | 3 | 204.053 |
| dot_int8 | 2.090 | 2.089–2.090 (0.0%) | 3 | 2.090 |
| global_copy | 64.074 | 64.060–64.080 (0.0%) | 3 | 64.083 |
| global_read | 62.107 | 62.098–62.145 (0.1%) | 3 | 62.058 |
| global_triad | 64.302 | 64.279–64.332 (0.1%) | 3 | 64.361 |
| global_write | 58.015 | 58.011–58.031 (0.0%) | 3 | 58.091 |
| matrix_fp16 | 9.716 | 9.716–9.718 (0.0%) | 3 | 9.681 |
| matrix_fp16_feed_cache | 4.232 | 4.200–4.258 (1.4%) | 3 | 6.915 |
| matrix_fp16_feed_dram | 60.334 | 60.282–60.391 (0.2%) | 3 | 60.425 |
| matrix_fp16_feed_shared | 9.511 | 9.509–9.511 (0.0%) | 3 | 9.414 |
| matrix_fp16_fp32 | 9.722 | 9.719–9.724 (0.0%) | 3 | 9.721 |
| matrix_fp16_fp32_feed_cache | 5.628 | 5.600–5.656 (1.0%) | 2 | 5.663 |
| matrix_fp16_fp32_feed_dram | 61.240 | 61.228–61.297 (0.1%) | 3 | 61.188 |
| matrix_fp16_fp32_feed_shared | 8.771 | 8.712–8.773 (0.7%) | 3 | 8.777 |
| matrix_int8 | 19.482 | 19.473–19.495 (0.1%) | 3 | 19.489 |
| matrix_int8_feed_cache | 8.438 | 8.425–8.450 (0.3%) | 2 | 8.453 |
| matrix_int8_feed_dram | 59.272 | 59.222–59.381 (0.3%) | 3 | 59.225 |
| matrix_int8_feed_shared | 17.818 | 17.800–17.823 (0.1%) | 3 | 17.810 |
| shared_fp16_read | 2061.882 | 2060.733–2069.941 (0.4%) | 3 | 2061.186 |
| shared_fp16_write | 611.343 | 611.296–611.467 (0.0%) | 3 | 611.687 |
| shared_fp32_read | 8371.545 | 8369.670–8371.545 (0.0%) | 3 | 8375.529 |
| shared_fp32_write | 4278.042 | 4276.160–4283.537 (0.2%) | 3 | 4284.906 |
| texture_rgba16f_buffer_cache | 48.959 | 48.950–48.970 (0.0%) | 3 | 48.983 |
| texture_rgba16f_tex2d_cache | 46.244 | 46.244–46.253 (0.0%) | 3 | 46.505 |
| texture_rgba16f_tex2d_dram | 20.089 | 20.081–20.095 (0.1%) | 3 | 20.099 |
| texture_rgba16f_tex3d_cache | 41.038 | 41.032–41.057 (0.1%) | 3 | 41.049 |
| texture_rgba16f_tex3d_dram | 40.141 | 40.110–40.193 (0.2%) | 3 | 40.142 |
| texture_rgba32f_buffer_cache | 38.147 | 38.146–38.201 (0.1%) | 3 | 38.154 |
| texture_rgba32f_buffer_dram | 61.989 | 61.895–62.321 (0.7%) | 3 | 61.807 |
| texture_rgba32f_tex2d_cache | 44.876 | 44.872–44.886 (0.0%) | 3 | 44.875 |
| texture_rgba32f_tex2d_dram | 20.135 | 20.132–20.143 (0.1%) | 3 | 20.145 |
| texture_rgba32f_tex3d_cache | 39.652 | 39.651–39.680 (0.1%) | 3 | 39.675 |
| texture_rgba32f_tex3d_dram | 40.841 | 40.834–40.855 (0.1%) | 3 | 40.724 |

## Cooperative matrix fed from memory, by reuse

Best validated median per source and CHAINS (multiply-adds per loaded A/B tile pair). `load GB/s` counts the A/B tile bytes loaded. DRAM-fed rates grow with reuse until the matrix unit limits them, so the `matrix_*_feed_dram` roof above is a bandwidth; look a kernel's ops per loaded byte up here instead. `gates` lists quality gates the row failed (such rows never define a roof).

| dtype | source | CHAINS | ops / loaded byte | rate | load GB/s | gates |
|---|---|---:|---:|---:|---:|---|
| fp16 | shared | 8 | 64.0 | 9.511 TFLOP/s | 148.6 | — |
| fp16 | cache | 1 | 8.0 | 1.233 TFLOP/s | 154.1 | — |
| fp16 | cache | 2 | 16.0 | 2.312 TFLOP/s | 144.5 | — |
| fp16 | cache | 4 | 32.0 | 4.258 TFLOP/s | 133.1 | — |
| fp16 | cache | 8 | 64.0 | 6.936 TFLOP/s | 108.4 | — |
| fp16 | dram | 1 | 8.0 | 0.446 TFLOP/s | 55.8 | — |
| fp16 | dram | 2 | 16.0 | 0.865 TFLOP/s | 54.1 | — |
| fp16 | dram | 4 | 32.0 | 1.681 TFLOP/s | 52.5 | — |
| fp16 | dram | 8 | 64.0 | 3.298 TFLOP/s | 51.5 | — |
| fp16_fp32 | shared | 1 | 8.0 | 4.534 TFLOP/s | 566.8 | — |
| fp16_fp32 | shared | 2 | 16.0 | 8.242 TFLOP/s | 515.1 | — |
| fp16_fp32 | shared | 4 | 32.0 | 8.777 TFLOP/s | 274.3 | — |
| fp16_fp32 | cache | 1 | 8.0 | 1.201 TFLOP/s | 150.1 | — |
| fp16_fp32 | cache | 2 | 16.0 | 2.223 TFLOP/s | 139.0 | — |
| fp16_fp32 | cache | 8 | 64.0 | 5.663 TFLOP/s | 88.5 | — |
| fp16_fp32 | dram | 1 | 8.0 | 0.448 TFLOP/s | 56.0 | — |
| fp16_fp32 | dram | 4 | 32.0 | 1.644 TFLOP/s | 51.4 | — |
| fp16_fp32 | dram | 8 | 64.0 | 2.810 TFLOP/s | 43.9 | — |
| int8 | shared | 1 | 16.0 | 4.658 TOP/s | 291.1 | — |
| int8 | shared | 8 | 128.0 | 17.823 TOP/s | 139.2 | — |
| int8 | cache | 1 | 16.0 | 1.502 TOP/s | 93.9 | — |
| int8 | cache | 2 | 32.0 | 2.901 TOP/s | 90.7 | — |
| int8 | cache | 4 | 64.0 | 5.217 TOP/s | 81.5 | — |
| int8 | cache | 8 | 128.0 | 8.453 TOP/s | 66.0 | — |
| int8 | dram | 1 | 16.0 | 0.755 TOP/s | 47.2 | — |
| int8 | dram | 4 | 64.0 | 3.369 TOP/s | 52.6 | — |

## Device-state sentinel

`alu_fp32_v4_c16 wg256 groups512` measured before and after every stage and every 20 configurations. Values below 85 % of the median reading (0.730 TFLOP/s) mark a stage that ran on a throttled or otherwise degraded device; re-measure those stages.

| UTC | label | TFLOP/s | state |
|---|---|---:|---|
| 2026-10-05T02:52:56 | validate_start | 0.730 | ok |
| 2026-10-05T02:53:10 | validate_end | 0.730 | ok |
| 2026-10-05T02:53:55 | cache_end | 0.730 | ok |
| 2026-10-05T02:54:13 | sweep-memory_20 | 0.730 | ok |
| 2026-10-05T02:54:38 | memory_end | 0.730 | ok |
| 2026-10-05T02:55:08 | sweep-compute_40 | 0.730 | ok |
| 2026-10-05T02:56:08 | sweep-compute_60 | 0.730 | ok |
| 2026-10-05T02:57:13 | sweep-compute_80 | 0.730 | ok |
| 2026-10-05T02:59:33 | sweep-compute_100 | 0.730 | ok |
| 2026-10-05T03:01:42 | sweep-compute_120 | 0.730 | ok |
| 2026-10-05T03:05:52 | sweep-compute_140 | 0.730 | ok |
| 2026-10-05T03:08:13 | sweep-compute_160 | 0.730 | ok |
| 2026-10-05T03:08:43 | compute_end | 0.730 | ok |
| 2026-10-05T03:09:00 | sweep-matrix-feed_180 | 0.731 | ok |
| 2026-10-05T03:09:52 | sweep-matrix-feed_200 | 0.730 | ok |
| 2026-10-05T03:10:26 | matrix_feed_end | 0.730 | ok |
| 2026-10-05T03:10:57 | sweep-texture_220 | 0.730 | ok |
| 2026-10-05T03:11:08 | texture_end | 0.730 | ok |
| 2026-10-05T03:11:57 | sweep-shared_240 | 0.730 | ok |
| 2026-10-05T03:12:38 | sweep-shared_260 | 0.730 | ok |
| 2026-10-05T03:13:24 | sweep-shared_280 | 0.731 | ok |
| 2026-10-05T03:14:08 | sweep-shared_300 | 0.730 | ok |
| 2026-10-05T03:14:47 | shared_end | 0.730 | ok |
| 2026-10-05T03:14:55 | latency-capacity_320 | 0.730 | ok |
| 2026-10-05T03:16:12 | latency_end | 0.730 | ok |
| 2026-10-05T03:16:48 | confirm_340 | 0.730 | ok |
| 2026-10-05T03:17:42 | confirm_360 | 0.730 | ok |
| 2026-10-05T03:18:41 | confirm_380 | 0.730 | ok |
| 2026-10-05T03:19:36 | confirm_400 | 0.730 | ok |
| 2026-10-05T03:20:34 | confirm_420 | 0.730 | ok |
| 2026-10-05T03:21:27 | confirm_440 | 0.730 | ok |
| 2026-10-05T03:22:19 | confirm_460 | 0.730 | ok |
| 2026-10-05T03:23:18 | confirm_480 | 0.730 | ok |
| 2026-10-05T03:24:04 | confirm_end | 0.730 | ok |
| 2026-10-05T03:24:10 | sustain_0 | 0.730 | ok |

## Ridge points (short-run roofs)

Arithmetic intensity (ops per byte of that level) at which each compute roof meets each memory roof.

| compute roof | global | cache | shared_fp32 | shared_fp16 |
|---|---:|---:|---:|---:|
| alu_fp16 | 27.40 | 8.64 | 0.21 | 0.85 |
| alu_fp32 | 18.68 | 5.89 | 0.14 | 0.58 |
| dot_int8 | 32.50 | 10.25 | 0.25 | 1.01 |
| matrix_fp16 | 151.10 | 47.65 | 1.16 | 4.71 |
| matrix_fp16_feed_cache | 65.81 | 20.75 | 0.51 | 2.05 |
| matrix_fp16_feed_shared | 147.92 | 46.64 | 1.14 | 4.61 |
| matrix_fp16_fp32 | 151.19 | 47.67 | 1.16 | 4.71 |
| matrix_fp16_fp32_feed_cache | 87.52 | 27.60 | 0.67 | 2.73 |
| matrix_fp16_fp32_feed_shared | 136.40 | 43.01 | 1.05 | 4.25 |
| matrix_int8 | 302.97 | 95.53 | 2.33 | 9.45 |
| matrix_int8_feed_cache | 131.22 | 41.38 | 1.01 | 4.09 |
| matrix_int8_feed_shared | 277.09 | 87.37 | 2.13 | 8.64 |

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
