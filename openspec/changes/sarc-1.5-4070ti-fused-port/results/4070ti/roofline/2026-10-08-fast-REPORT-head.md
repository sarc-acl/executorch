# NVIDIA GeForce RTX 4070 Ti SUPER roofline report

Device: host AMD EPYC 4464P 12-Core Processor (SoC AMD EPYC 4464P 12-Core Processor), Ubuntu 26.04.1 LTS, driver 2580660800, subgroup 32.
Plan(s): fast. GPU clock **DVFS-governed** (not pinned); results depend on the governor and thermal state.
Code: None, runner 5edad6896f44f511. Rows from other runner or shader builds excluded: 0.

Validation policy: **pre_and_post**; checks bracket continuous sampling and do not guarantee detection of transient errors between checks. Historical pre-only rows are diagnostic only.
Excluded rows: 66; reasons are in all-configurations.csv.
matrix_fp16_fp32_feed_dram: insufficient_quality_repeats, repeat_unstable
Sustained roofs require three distinct stable batches per current confirmed configuration and duration

Short-run columns are the best validated configuration: median and best (minimum time, the STREAM/BabelStream convention) of the samples, and the differential rate (paired L vs L/2 runs, removing fixed per-dispatch cost). Sustained is the median of the last 60 s of each sustained run (durations are recorded in sustained-runs.csv). Controls never define a roof. `spill` marks variants whose driver statistics report register spilling: spills only slow a kernel, so the value is still an achievable lower bound.

| roof | short median | short best | differential | sustained | unit |
|---|---:|---:|---:|---:|---|
| alu_fp16 | 44.823 | 44.971 | 44.864 (fixed 0%) | 44.826 (1×) | TFLOP/s |
| alu_fp32 | 44.170 | 44.425 | 44.273 (fixed 0%) | — | TFLOP/s |
| cache_read_effective | 7115.214 | 7130.909 | 7122.857 (fixed 0%) | — | GB/s |
| dot_int8 | 77.802 | 77.823 | 77.909 (fixed 0%) | — | TOP/s |
| global_copy | 645.990 | 647.006 | 648.107 (fixed 0%) | — | GB/s |
| global_read | 713.999 | 720.126 | 736.884 (fixed 3%) | 2655.931 (1×) | GB/s |
| global_triad | 631.771 | 633.321 | 629.885 (fixed -0%) | — | GB/s |
| global_write | 641.011 | 644.653 | 640.505 (fixed -0%) | — | GB/s |
| matrix_fp16 | 182.838 | 182.914 | 184.476 (fixed 1%) | — | TFLOP/s |
| matrix_fp16_feed_cache | 157.916 | 158.872 | 159.712 (fixed 1%) | — | TFLOP/s |
| matrix_fp16_feed_dram | 650.167 | 650.481 | 659.203 (fixed 1%) | — | GB/s |
| matrix_fp16_feed_shared | 177.762 | 177.865 | 184.365 (fixed 4%) | — | TFLOP/s |
| matrix_fp16_fp32 | 92.312 | 92.323 | 92.376 (fixed 0%) | 92.807 (1×) | TFLOP/s |
| matrix_fp16_fp32_feed_cache | 89.596 | 89.615 | 89.695 (fixed 0%) | — | TFLOP/s |
| matrix_fp16_fp32_feed_dram | 669.443 | 669.960 | 708.096 (fixed 5%) | — | GB/s |
| matrix_fp16_fp32_feed_shared | 92.181 | 92.190 | 92.374 (fixed 0%) | — | TFLOP/s |
| matrix_int8 | 369.157 | 369.265 | 369.455 (fixed 0%) | — | TOP/s |
| matrix_int8_feed_cache | 278.570 | 280.190 | 279.454 (fixed 0%) | — | TOP/s |
| matrix_int8_feed_dram | 698.948 | 700.099 | 776.100 (fixed 10%) | — | GB/s |
| matrix_int8_feed_shared | 367.763 | 367.923 | 369.327 (fixed 0%) | — | TOP/s |
| shared_fp16_read | 1344.248 | 1344.360 | 1454.119 (fixed 8%) | — | GB/s |
| shared_fp16_write | 11803.168 | 11809.283 | 12130.809 (fixed 3%) | — | GB/s |
| shared_fp32_read | 21071.883 | 21113.626 | 21115.702 (fixed 0%) | — | GB/s |
| shared_fp32_write | 19155.300 | 19209.623 | 19159.450 (fixed 0%) | — | GB/s |
| texture_rgba16f_buffer_cache | 1670.859 | 1671.394 | 1670.484 (fixed -0%) | — | GB/s |
| texture_rgba16f_buffer_dram | 636.145 | 636.408 | 636.103 (fixed -0%) | — | GB/s |
| texture_rgba16f_tex2d_cache | 1569.356 | 1570.219 | 1569.741 (fixed 0%) | — | GB/s |
| texture_rgba16f_tex2d_dram | 509.610 | 512.500 | 508.814 (fixed -0%) | — | GB/s |
| texture_rgba16f_tex3d_cache | 1626.410 | 1627.447 | 1626.822 (fixed 0%) | — | GB/s |
| texture_rgba16f_tex3d_dram | 629.896 | 630.778 | 630.084 (fixed 0%) | — | GB/s |
| texture_rgba32f_buffer_cache | 1677.182 | 1677.835 | 1677.130 (fixed -0%) | — | GB/s |
| texture_rgba32f_buffer_dram | 634.686 | 635.048 | 634.705 (fixed 0%) | — | GB/s |
| texture_rgba32f_tex2d_cache | 1613.441 | 1614.241 | 1613.692 (fixed 0%) | — | GB/s |
| texture_rgba32f_tex2d_dram | 493.469 | 495.439 | 492.952 (fixed -0%) | — | GB/s |
| texture_rgba32f_tex3d_cache | 1647.945 | 1648.473 | 1648.217 (fixed 0%) | — | GB/s |
| texture_rgba32f_tex3d_dram | 630.632 | 631.065 | 630.542 (fixed -0%) | — | GB/s |

## Roof confirmation

Each roof's top candidates were re-measured in fresh processes, round-robin with alternating order. The roof is the median of the best candidate's repeats; the sweep maximum (a single run) is shown for comparison. Roofs without a quality-passing candidate (standard error of the median <= 3 %, not short, fixed cost <= 10 %) are marked unconfirmed.

| roof | confirmed median | repeat range | repeats | sweep max (unconfirmed) |
|---|---:|---:|---:|---:|
| alu_fp16 | 44.823 | 44.822–44.965 (0.3%) | 3 | 44.824 |
| alu_fp32 | 44.170 | 43.920–44.181 (0.6%) | 3 | 44.175 |
| cache_read_effective | 7115.214 | 7115.168–7128.858 (0.2%) | 3 | 7114.308 |
| dot_int8 | 77.802 | 77.797–77.804 (0.0%) | 3 | 77.803 |
| global_copy | 645.990 | 645.897–646.122 (0.0%) | 3 | 646.049 |
| global_read | 713.999 | 713.118–714.891 (0.2%) | 3 | 715.707 |
| global_triad | 631.771 | 631.542–632.396 (0.1%) | 3 | 631.628 |
| global_write | 641.011 | 640.795–642.146 (0.2%) | 3 | 641.308 |
| matrix_fp16 | 182.838 | 182.777–182.894 (0.1%) | 3 | 182.828 |
| matrix_fp16_feed_cache | 157.916 | 157.891–158.785 (0.6%) | 3 | 158.730 |
| matrix_fp16_feed_dram | 650.167 | 650.164–650.244 (0.0%) | 3 | 650.235 |
| matrix_fp16_feed_shared | 177.762 | 177.750–177.850 (0.1%) | 3 | 177.752 |
| matrix_fp16_fp32 | 92.312 | 92.307–92.316 (0.0%) | 3 | 92.314 |
| matrix_fp16_fp32_feed_cache | 89.596 | 89.587–89.598 (0.0%) | 3 | 89.599 |
| matrix_fp16_fp32_feed_dram | 669.443 | 669.360–669.469 (0.0%) | 3 | 698.229 |
| matrix_fp16_fp32_feed_shared | 92.181 | 92.180–92.181 (0.0%) | 3 | 92.183 |
| matrix_int8 | 369.157 | 369.149–369.214 (0.0%) | 3 | 369.210 |
| matrix_int8_feed_cache | 278.570 | 278.567–278.612 (0.0%) | 3 | 278.489 |
| matrix_int8_feed_dram | 698.948 | 698.838–699.006 (0.0%) | 3 | 698.810 |
| matrix_int8_feed_shared | 367.763 | 367.493–367.781 (0.1%) | 3 | 367.410 |
| shared_fp16_read | 1344.248 | 1344.241–1344.269 (0.0%) | 3 | 1344.233 |
| shared_fp16_write | 11803.168 | 11777.091–11805.164 (0.2%) | 3 | 11778.084 |
| shared_fp32_read | 21071.883 | 20945.187–21078.915 (0.6%) | 3 | 20935.488 |
| shared_fp32_write | 19155.300 | 19128.341–19158.407 (0.2%) | 3 | 19158.902 |
| texture_rgba16f_buffer_cache | 1670.859 | 1670.720–1670.903 (0.0%) | 3 | 1667.592 |
| texture_rgba16f_buffer_dram | 636.145 | 636.106–636.155 (0.0%) | 3 | 636.142 |
| texture_rgba16f_tex2d_cache | 1569.356 | 1569.356–1569.398 (0.0%) | 3 | 1569.507 |
| texture_rgba16f_tex2d_dram | 509.610 | 509.443–509.851 (0.1%) | 3 | 508.561 |
| texture_rgba16f_tex3d_cache | 1626.410 | 1626.363–1626.918 (0.0%) | 3 | 1626.580 |
| texture_rgba16f_tex3d_dram | 629.896 | 629.775–629.938 (0.0%) | 3 | 629.839 |
| texture_rgba32f_buffer_cache | 1677.182 | 1677.039–1677.199 (0.0%) | 3 | 1677.370 |
| texture_rgba32f_buffer_dram | 634.686 | 634.654–634.709 (0.0%) | 3 | 634.831 |
