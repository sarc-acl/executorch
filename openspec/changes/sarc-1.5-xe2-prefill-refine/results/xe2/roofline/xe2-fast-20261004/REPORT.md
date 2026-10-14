# Intel(R) Graphics (BMG G31) roofline report

Device: host Intel(R) Core(TM) i9-14900K (SoC Intel(R) Core(TM) i9-14900K), Fedora Linux 44 (Cloud Edition), driver 109060099, subgroup 32.
Plan(s): fast. GPU clock state **unavailable** (frequency and pinning not verified).
Code: None, runner 810e098c8abb2dfb. Rows from other runner or shader builds excluded: 0.

Validation policy: **pre_and_post**; checks bracket continuous sampling and do not guarantee detection of transient errors between checks. Historical pre-only rows are diagnostic only.
Excluded rows: 62; reasons are in all-configurations.csv.
Sustained roofs require three distinct stable batches per current confirmed configuration and duration

Short-run columns are the best validated configuration: median and best (minimum time, the STREAM/BabelStream convention) of the samples, and the differential rate (paired L vs L/2 runs, removing fixed per-dispatch cost). Sustained is the median of the last 60 s of each sustained run (durations are recorded in sustained-runs.csv). Controls never define a roof. `spill` marks variants whose driver statistics report register spilling: spills only slow a kernel, so the value is still an achievable lower bound.

| roof | short median | short best | differential | sustained | unit |
|---|---:|---:|---:|---:|---|
| alu_fp16 | 42.853 | 42.872 | 42.932 (fixed 0%) | 42.598 (1×) | TFLOP/s |
| alu_fp32 | 20.289 | 20.301 | 20.358 (fixed 0%) | — | TFLOP/s |
| cache_read_effective | 9978.406 | 9981.048 | 9994.656 (fixed 0%) | — | GB/s |
| dot_int8 | 50.365 | 50.367 | 50.418 (fixed 0%) | — | TOP/s |
| global_copy | 532.292 | 564.993 | 534.902 (fixed 0%) | — | GB/s |
| global_read | 603.304 | 605.384 | 606.934 (fixed 1%) | 603.728 (1×) | GB/s |
| global_triad | 483.571 | 496.315 | 472.023 (fixed -2%) | — | GB/s |
| global_write | 509.188 | 532.539 | 511.720 (fixed 0%) | — | GB/s |
| matrix_fp16 | 173.324 | 173.335 | 180.314 (fixed 4%) | — | TFLOP/s |
| matrix_fp16_feed_cache | 158.531 | 161.638 | 162.866 (fixed 3%) | — | TFLOP/s |
| matrix_fp16_feed_dram | 598.359 | 598.923 | 607.685 (fixed 2%) | — | GB/s |
| matrix_fp16_feed_shared | 168.427 | 168.662 | 177.512 (fixed 5%) | — | TFLOP/s |
| matrix_fp16_fp32 | 179.942 | 179.952 | 180.443 (fixed 0%) | 179.939 (1×) | TFLOP/s |
| matrix_fp16_fp32_feed_cache | 151.969 | 153.293 | 153.091 (fixed 1%) | — | TFLOP/s |
| matrix_fp16_fp32_feed_dram | 590.254 | 591.940 | 594.942 (fixed 1%) | — | GB/s |
| matrix_fp16_fp32_feed_shared | 166.594 | 166.816 | 167.420 (fixed 0%) | — | TFLOP/s |
| matrix_int8 | 359.875 | 359.910 | 360.879 (fixed 0%) | — | TOP/s |
| matrix_int8_feed_cache | 278.213 | 281.661 | 279.554 (fixed 0%) | — | TOP/s |
| matrix_int8_feed_dram | 610.155 | 614.457 | 652.412 (fixed 6%) | — | GB/s |
| matrix_int8_feed_shared | 323.307 | 323.853 | 327.908 (fixed 1%) | — | TOP/s |
| shared_fp16_read | 3331.016 | 3331.221 | 3493.216 (fixed 5%) | — | GB/s |
| shared_fp16_write | 6455.425 | 6457.201 | 6606.948 (fixed 2%) | — | GB/s |
| shared_fp32_read | 10018.256 | 10020.615 | 10034.637 (fixed 0%) | — | GB/s |
| shared_fp32_write | 10501.253 | 10501.891 | 10508.126 (fixed 0%) | — | GB/s |
| texture_rgba16f_buffer_cache | 2272.611 | 2272.946 | 2273.617 (fixed 0%) | — | GB/s |
| texture_rgba16f_buffer_dram | 564.075 | 564.918 | 564.525 (fixed 0%) | — | GB/s |
| texture_rgba16f_tex2d_cache | 2014.636 | 2015.189 | 2014.939 (fixed 0%) | — | GB/s |
| texture_rgba16f_tex2d_dram | 536.332 | 541.820 | 534.443 (fixed -0%) | — | GB/s |
| texture_rgba16f_tex3d_cache | 1970.497 | 1971.044 | 1971.044 (fixed 0%) | — | GB/s |
| texture_rgba16f_tex3d_dram | 559.279 | 566.628 | 558.575 (fixed -0%) | — | GB/s |
| texture_rgba32f_buffer_cache | 2283.923 | 2284.494 | 2285.267 (fixed 0%) | — | GB/s |
| texture_rgba32f_buffer_dram | 567.298 | 567.700 | 567.499 (fixed 0%) | — | GB/s |
| texture_rgba32f_tex2d_cache | 2060.169 | 2060.938 | 2060.634 (fixed 0%) | — | GB/s |
| texture_rgba32f_tex2d_dram | 720.517 | 739.555 | 715.294 (fixed -1%) | — | GB/s |
| texture_rgba32f_tex3d_cache | 1993.258 | 1994.131 | 1993.547 (fixed 0%) | — | GB/s |
| texture_rgba32f_tex3d_dram | 764.590 | 771.215 | 763.963 (fixed -0%) | — | GB/s |

## Roof confirmation

Each roof's top candidates were re-measured in fresh processes, round-robin with alternating order. The roof is the median of the best candidate's repeats; the sweep maximum (a single run) is shown for comparison. Roofs without a quality-passing candidate (standard error of the median <= 3 %, not short, fixed cost <= 10 %) are marked unconfirmed.

| roof | confirmed median | repeat range | repeats | sweep max (unconfirmed) |
|---|---:|---:|---:|---:|
| alu_fp16 | 42.853 | 42.731–42.869 (0.3%) | 3 | 42.869 |
| alu_fp32 | 20.289 | 20.287–20.300 (0.1%) | 3 | 20.287 |
| cache_read_effective | 9978.406 | 9977.727–9978.632 (0.0%) | 3 | 9978.230 |
| dot_int8 | 50.365 | 49.594–50.365 (1.5%) | 3 | 50.365 |
| global_copy | 532.292 | 531.926–535.435 (0.7%) | 3 | 533.992 |
| global_read | 603.304 | 603.266–603.484 (0.0%) | 3 | 603.989 |
| global_triad | 483.571 | 482.428–484.154 (0.4%) | 3 | 482.680 |
| global_write | 509.188 | 509.092–513.264 (0.8%) | 3 | 510.307 |
| matrix_fp16 | 173.324 | 173.323–173.327 (0.0%) | 3 | 173.326 |
| matrix_fp16_feed_cache | 158.531 | 158.493–160.622 (1.3%) | 3 | 158.489 |
| matrix_fp16_feed_dram | 598.359 | 598.269–598.423 (0.0%) | 3 | 598.341 |
| matrix_fp16_feed_shared | 168.427 | 168.406–168.466 (0.0%) | 3 | 168.373 |
| matrix_fp16_fp32 | 179.942 | 179.940–179.943 (0.0%) | 3 | 179.931 |
| matrix_fp16_fp32_feed_cache | 151.969 | 151.945–152.169 (0.1%) | 3 | 152.626 |
| matrix_fp16_fp32_feed_dram | 590.254 | 588.726–590.661 (0.3%) | 3 | 604.199 |
| matrix_fp16_fp32_feed_shared | 166.594 | 166.448–166.620 (0.1%) | 3 | 166.451 |
| matrix_int8 | 359.875 | 359.874–359.894 (0.0%) | 3 | 359.877 |
| matrix_int8_feed_cache | 278.213 | 278.190–279.435 (0.4%) | 3 | 280.081 |
| matrix_int8_feed_dram | 610.155 | 607.369–612.304 (0.8%) | 3 | 609.927 |
| matrix_int8_feed_shared | 323.307 | 322.304–323.333 (0.3%) | 3 | 319.719 |
| shared_fp16_read | 3331.016 | 3330.989–3331.046 (0.0%) | 3 | 3331.046 |
| shared_fp16_write | 6455.425 | 6455.046–6455.485 (0.0%) | 3 | 6455.158 |
| shared_fp32_read | 10018.256 | 10017.318–10018.408 (0.0%) | 3 | 10018.865 |
| shared_fp32_write | 10501.253 | 10500.977–10501.435 (0.0%) | 3 | 10501.474 |
| texture_rgba16f_buffer_cache | 2272.611 | 2272.572–2272.631 (0.0%) | 3 | 2272.611 |
| texture_rgba16f_buffer_dram | 564.075 | 564.065–564.210 (0.0%) | 3 | 564.312 |
| texture_rgba16f_tex2d_cache | 2014.636 | 2014.575–2014.682 (0.0%) | 3 | 2014.627 |
| texture_rgba16f_tex2d_dram | 536.332 | 535.519–536.928 (0.3%) | 3 | 536.056 |
| texture_rgba16f_tex3d_cache | 1970.497 | 1970.463–1970.585 (0.0%) | 3 | 1970.353 |
| texture_rgba16f_tex3d_dram | 559.279 | 558.851–559.846 (0.2%) | 3 | 559.588 |
| texture_rgba32f_buffer_cache | 2283.923 | 2283.879–2284.038 (0.0%) | 3 | 2284.078 |
| texture_rgba32f_buffer_dram | 567.298 | 567.284–567.323 (0.0%) | 3 | 567.308 |
| texture_rgba32f_tex2d_cache | 2060.169 | 2060.029–2060.366 (0.0%) | 3 | 2060.223 |
| texture_rgba32f_tex2d_dram | 720.517 | 719.118–722.558 (0.5%) | 3 | 718.116 |
| texture_rgba32f_tex3d_cache | 1993.258 | 1993.189–1993.303 (0.0%) | 3 | 1993.390 |
| texture_rgba32f_tex3d_dram | 764.590 | 763.098–765.476 (0.3%) | 3 | 764.703 |

## Cooperative matrix fed from memory, by reuse

Best validated median per source and CHAINS (multiply-adds per loaded A/B tile pair). `load GB/s` counts the A/B tile bytes loaded. DRAM-fed rates grow with reuse until the matrix unit limits them, so the `matrix_*_feed_dram` roof above is a bandwidth; look a kernel's ops per loaded byte up here instead. `gates` lists quality gates the row failed (such rows never define a roof).

| dtype | source | CHAINS | ops / loaded byte | rate | load GB/s | gates |
|---|---|---:|---:|---:|---:|---|
| fp16 | shared | 8 | 42.7 | 168.466 TFLOP/s | 3948.4 | — |
| fp16 | cache | 1 | 5.3 | 29.569 TFLOP/s | 5544.2 | — |
| fp16 | cache | 2 | 10.7 | 57.282 TFLOP/s | 5370.2 | — |
| fp16 | cache | 4 | 21.3 | 107.091 TFLOP/s | 5019.9 | — |
| fp16 | cache | 8 | 42.7 | 160.622 TFLOP/s | 3764.6 | — |
| fp16 | dram | 1 | 5.3 | 3.126 TFLOP/s | 586.2 | — |
| fp16 | dram | 2 | 10.7 | 6.147 TFLOP/s | 576.3 | — |
| fp16 | dram | 4 | 21.3 | 11.968 TFLOP/s | 561.0 | — |
| fp16 | dram | 8 | 42.7 | 23.022 TFLOP/s | 539.6 | — |
| fp16_fp32 | shared | 1 | 5.3 | 42.372 TFLOP/s | 7944.8 | — |
| fp16_fp32 | shared | 2 | 10.7 | 82.705 TFLOP/s | 7753.6 | — |
| fp16_fp32 | shared | 4 | 21.3 | 155.550 TFLOP/s | 7291.4 | — |
| fp16_fp32 | shared | 8 | 42.7 | 166.620 TFLOP/s | 3905.2 | — |
| fp16_fp32 | cache | 1 | 5.3 | 27.137 TFLOP/s | 5088.2 | — |
| fp16_fp32 | cache | 2 | 10.7 | 52.490 TFLOP/s | 4921.0 | — |
| fp16_fp32 | cache | 4 | 21.3 | 101.150 TFLOP/s | 4741.4 | — |
| fp16_fp32 | cache | 8 | 42.7 | 152.626 TFLOP/s | 3577.2 | — |
| fp16_fp32 | dram | 1 | 5.3 | 3.131 TFLOP/s | 587.1 | — |
| fp16_fp32 | dram | 2 | 10.7 | 6.205 TFLOP/s | 581.7 | — |
| fp16_fp32 | dram | 4 | 21.3 | 12.260 TFLOP/s | 574.7 | — |
| fp16_fp32 | dram | 8 | 42.7 | 25.193 TFLOP/s | 590.5 | — |
| int8 | shared | 1 | 10.7 | 53.959 TOP/s | 5058.7 | — |
| int8 | shared | 2 | 21.3 | 113.915 TOP/s | 5339.8 | — |
| int8 | shared | 4 | 42.7 | 210.809 TOP/s | 4940.8 | — |
| int8 | shared | 8 | 85.3 | 323.333 TOP/s | 3789.1 | — |
| int8 | cache | 1 | 10.7 | 36.438 TOP/s | 3416.1 | — |
| int8 | cache | 2 | 21.3 | 73.365 TOP/s | 3439.0 | — |
| int8 | cache | 4 | 42.7 | 140.580 TOP/s | 3294.8 | — |
| int8 | cache | 8 | 85.3 | 280.081 TOP/s | 3282.2 | — |
| int8 | dram | 1 | 10.7 | 6.491 TOP/s | 608.6 | — |
| int8 | dram | 2 | 21.3 | 12.999 TOP/s | 609.3 | — |
| int8 | dram | 4 | 42.7 | 24.832 TOP/s | 582.0 | — |
| int8 | dram | 8 | 85.3 | 47.424 TOP/s | 555.8 | — |

## Device-state sentinel

`alu_fp32_v4_c16 wg256 groups512` measured before and after every stage and every 20 configurations. Values below 85 % of the median reading (18.350 TFLOP/s) mark a stage that ran on a throttled or otherwise degraded device; re-measure those stages.

| UTC | label | TFLOP/s | state |
|---|---|---:|---|
| 2026-10-04T21:54:17 | validate_start | 18.350 | ok |
| 2026-10-04T21:54:22 | validate_end | 18.350 | ok |
| 2026-10-04T21:54:55 | cache_end | 18.351 | ok |
| 2026-10-04T21:55:03 | sweep-memory_20 | 18.350 | ok |
| 2026-10-04T21:55:15 | memory_end | 18.350 | ok |
| 2026-10-04T21:55:37 | sweep-compute_40 | 18.350 | ok |
| 2026-10-04T21:56:08 | sweep-compute_60 | 18.350 | ok |
| 2026-10-04T21:56:40 | sweep-compute_80 | 18.349 | ok |
| 2026-10-04T21:57:12 | sweep-compute_100 | 18.351 | ok |
| 2026-10-04T21:57:44 | sweep-compute_120 | 18.350 | ok |
| 2026-10-04T21:58:16 | sweep-compute_140 | 18.351 | ok |
| 2026-10-04T21:58:50 | sweep-compute_160 | 18.349 | ok |
| 2026-10-04T21:59:12 | compute_end | 18.350 | ok |
| 2026-10-04T21:59:22 | sweep-matrix-feed_180 | 18.350 | ok |
| 2026-10-04T21:59:55 | sweep-matrix-feed_200 | 18.350 | ok |
| 2026-10-04T22:00:12 | matrix_feed_end | 18.350 | ok |
| 2026-10-04T22:00:30 | sweep-texture_220 | 18.350 | ok |
| 2026-10-04T22:00:35 | texture_end | 18.351 | ok |
| 2026-10-04T22:01:08 | sweep-shared_240 | 18.350 | ok |
| 2026-10-04T22:01:41 | sweep-shared_260 | 18.351 | ok |
| 2026-10-04T22:02:10 | shared_end | 18.350 | ok |
| 2026-10-04T22:02:14 | latency-capacity_280 | 18.350 | ok |
| 2026-10-04T22:02:46 | latency_end | 18.350 | ok |
| 2026-10-04T22:03:06 | confirm_300 | 18.350 | ok |
| 2026-10-04T22:03:40 | confirm_320 | 18.350 | ok |
| 2026-10-04T22:04:19 | confirm_340 | 18.351 | ok |
| 2026-10-04T22:04:52 | confirm_360 | 18.350 | ok |
| 2026-10-04T22:05:32 | confirm_380 | 18.351 | ok |
| 2026-10-04T22:06:05 | confirm_400 | 18.350 | ok |
| 2026-10-04T22:06:37 | confirm_420 | 18.350 | ok |
| 2026-10-04T22:07:15 | confirm_440 | 18.351 | ok |
| 2026-10-04T22:07:50 | confirm_460 | 18.350 | ok |
| 2026-10-04T22:07:55 | confirm_end | 18.350 | ok |
| 2026-10-04T22:07:56 | sustain_0 | 18.350 | ok |

## Ridge points (short-run roofs)

Arithmetic intensity (ops per byte of that level) at which each compute roof meets each memory roof.

| compute roof | global | cache | shared_fp32 | shared_fp16 |
|---|---:|---:|---:|---:|
| alu_fp16 | 71.03 | 4.29 | 4.08 | 6.64 |
| alu_fp32 | 33.63 | 2.03 | 1.93 | 3.14 |
| dot_int8 | 83.48 | 5.05 | 4.80 | 7.80 |
| matrix_fp16 | 287.29 | 17.37 | 16.51 | 26.85 |
| matrix_fp16_feed_cache | 262.77 | 15.89 | 15.10 | 24.56 |
| matrix_fp16_feed_shared | 279.17 | 16.88 | 16.04 | 26.09 |
| matrix_fp16_fp32 | 298.26 | 18.03 | 17.14 | 27.87 |
| matrix_fp16_fp32_feed_cache | 251.89 | 15.23 | 14.47 | 23.54 |
| matrix_fp16_fp32_feed_shared | 276.14 | 16.70 | 15.86 | 25.81 |
| matrix_int8 | 596.51 | 36.07 | 34.27 | 55.75 |
| matrix_int8_feed_cache | 461.15 | 27.88 | 26.49 | 43.10 |
| matrix_int8_feed_shared | 535.89 | 32.40 | 30.79 | 50.08 |

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
