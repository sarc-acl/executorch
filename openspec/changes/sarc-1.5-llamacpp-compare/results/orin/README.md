# Jetson Orin Nano: arms and results (llama.cpp Vulkan and CUDA)

Session `s1`, 2026-10-06, on the primary device after the tuning campaign had closed. 14 arms, `SESSION_OK`:
five valid runs in every cell, no invalid run. Exact lines in `arms.tsv`. The session ran on the device
(15 W mode, governor `nvhost_podgov`, 612 MHz ceiling; no `jetson_clocks`, no `nvpmodel`, no sudo). Every timed
run sat at 612 MHz.

- ExecuTorch: `stock` = upstream `release/1.5` at `985c1ceccc` plus the compile-only backport, cross-built for
  aarch64 for this change with the campaign's recipe (the campaign had no stock build); `sarc` = `6a7cc8cc6`;
  `tuned` = `fd44f8011` with `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=orin-refine5
  ET_VK_SARC_SOFTMAX_VARIANT=orin_g64` (the campaign's final stack; unmerged development branch).
- llama.cpp `b11430`: Vulkan cross-built in the campaign's image with glslc 2026.1 (cooperative matrix and
  integer dot product enabled; the device reports `matrix cores: NV_coopmat2`), CUDA built natively on the
  second Orin (`-DGGML_CUDA=ON -DGGML_NATIVE=ON`, CUDA 13.2) and copied over.
- Settings screen (1B, Q4_0, llama-bench tok/s, `screen.csv`): Vulkan default 1515, `-ub 1024` flash attention on
  **1544** / off 975, `-ub 2048` on 1517; CUDA default 1663, `-ub 1024` on **1694**, `-ub 2048` on 1662.

## Result (tok/s, median of 5; llama.cpp by llama-bench at its best setting, Q4_0)

| model | stock 4w | SARC 4w | SARC 8da4w | tuned 4w | tuned 8da4w | llama.cpp Vulkan | llama.cpp CUDA | tuned 4w / Vulkan | tuned 4w / CUDA | tuned 8da4w / CUDA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1B | 229.4 | 890.4 | 823.2 | 1491.6 | 1381.0 | 1538.1 | 1689.3 | 0.97 | 0.88 | 0.82 |
| 3B | 82.5 | 360.4 | 320.3 | 629.2 | 570.3 | 557.2 | 630.2 | 1.13 | 1.00 | 0.90 |
| 8B | 35.4 | 189.7 | 170.4 | 295.3 | 268.9 | 244.3 | 286.4 | 1.21 | 1.03 | 0.94 |

The ExecuTorch arms reproduce the campaign's final session and the September table within 0.2 %.
llama.cpp's Q4_K_M files run faster than Q4_0 on Vulkan here (1695.9 / 645.8 / 287.3), which puts llama.cpp
Vulkan with Q4_K_M ahead of tuned 4w on 1B and 3B and 3 % behind on 8B; on CUDA Q4_K_M is the slower file
(1483.0 / 555.9 / 246.1). llama.cpp's two timers agree within 3 % on this device. Text check: all arms
continue the real-text prompt fluently.
