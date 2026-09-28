# ExecuTorch kernels on the Radeon RX 7900 XTX roofline

Source: `results/newdev-20260927/7900xtx-mb/base-linear1.json` (ExecuTorch device: radeon rx 7900 xtx, group size 128).

Ops = 2·M·N·K (integer ops for 8da4w). Bytes = compulsory traffic (int4 weights + fp16 scales + inputs + fp16 output, each once), so intensity is an upper bound. Attainable = min(compute roof, intensity × DRAM read roof). Only confirmed roofs with repeat spread ≤ 5 % are used.

## Roofs used

| roof | confirmed median | unit | repeat spread |
|---|---:|---|---:|
| global_read | 1108.158 | GB/s | 0.1% |
| matrix_fp16 | 136.403 | TFLOP/s | 1.0% |
| matrix_fp16_feed_shared | 127.499 | TFLOP/s | 1.2% |
| matrix_int8 | 142.621 | TOP/s | 0.6% |
| matrix_int8_feed_shared | 113.856 | TOP/s | 0.8% |

## Kernels

| model | layer | scheme | regime | storage | suite | kernel | M×K×N | µs | TOP/s | ops/B | bound | % attainable | % compute roof | % shared-fed MMA |
|---|---|---|---|---|---|---|---|---:|---:|---:|---|---:|---:|---:|
| llama-3.1-8b | wq_wo | 4w | prefill | texture3d | linear | `linear_q4gsw_coopmat_tsweep_dbuf4_t128x128k32g42s32_texture3d_texture2d_half` | 2048×4096×4096 | 1250.6 | 54.95 | 1628 | compute | 40% | 40% | 43% |
| llama-3.1-8b | wk_wv | 4w | prefill | texture3d | linear | `linear_q4gsw_coopmat_tsweep_dbuf4_t128x128k32g42s32_texture3d_texture2d_half` | 2048×4096×1024 | 397.8 | 43.19 | 743 | compute | 32% | 32% | 34% |
| llama-3.1-8b | w1_w3 | 4w | prefill | texture3d | linear | `linear_q4gsw_coopmat_tsweep_dbuf4_t128x128k32g42s32_texture3d_texture2d_half` | 2048×4096×14336 | 3836.9 | 62.69 | 2274 | compute | 46% | 46% | 49% |
| llama-3.1-8b | w2 | 4w | prefill | texture3d | linear | `linear_q4gsw_coopmat_tsweep_dbuf4_t128x128k32g42s32_texture3d_texture2d_half` | 2048×14336×4096 | 4379.9 | 54.91 | 2274 | compute | 40% | 40% | 43% |
| llama-3.2-3b | wq_wo | 4w | prefill | texture3d | linear | `linear_q4gsw_coopmat_tsweep_dbuf4_t128x128k32g42s32_texture3d_texture2d_half` | 2048×3072×3072 | 662.6 | 58.34 | 1287 | compute | 43% | 43% | 46% |
| llama-3.2-3b | wk_wv | 4w | prefill | texture3d | linear | `linear_q4gsw_coopmat_tsweep_dbuf4_t128x128k32g42s32_texture3d_texture2d_half` | 2048×3072×1024 | 306.2 | 42.08 | 700 | compute | 31% | 31% | 33% |
| llama-3.2-3b | w1_w3 | 4w | prefill | texture3d | linear | `linear_q4gsw_coopmat_tsweep_dbuf4_t128x128k32g42s32_texture3d_texture2d_half` | 2048×3072×8192 | 1778.9 | 57.95 | 1744 | compute | 42% | 42% | 45% |
| llama-3.2-3b | w2 | 4w | prefill | texture3d | linear | `linear_q4gsw_coopmat_tsweep_dbuf4_t128x128k32g42s32_texture3d_texture2d_half` | 2048×8192×3072 | 1724.2 | 59.78 | 1744 | compute | 44% | 44% | 47% |
| llama-3.2-1b | wq_wo | 4w | prefill | texture3d | linear | `linear_q4gsw_coopmat_tsweep_dbuf4_t128x128k32g42s32_texture3d_texture2d_half` | 2048×2048×2048 | 330.6 | 51.97 | 907 | compute | 38% | 38% | 41% |
| llama-3.2-1b | wk_wv | 4w | prefill | texture3d | linear | `linear_q4gsw_coopmat_tsweep_dbuf4_t128x128k32g42s32_texture3d_texture2d_half` | 2048×2048×512 | 112.2 | 38.29 | 390 | compute | 28% | 28% | 30% |
| llama-3.2-1b | w1_w3 | 4w | prefill | texture3d | linear | `linear_q4gsw_coopmat_tsweep_dbuf4_t128x128k32g42s32_texture3d_texture2d_half` | 2048×2048×8192 | 1202.3 | 57.15 | 1358 | compute | 42% | 42% | 45% |
| llama-3.2-1b | w2 | 4w | prefill | texture3d | linear | `linear_q4gsw_coopmat_tsweep_dbuf4_t128x128k32g42s32_texture3d_texture2d_half` | 2048×8192×2048 | 1239.2 | 55.46 | 1358 | compute | 41% | 41% | 43% |
| llama-3.1-8b | wq_wo | 8da4w | prefill | texture3d | linear | `linear_dq8ca_q4gsw_coopmat_tsweep_dbuf4zpg_t128x64k32g42s32_texture3d_texture2d_half` | 2048×4096×4096 | 858.0 | 80.09 | 2030 | compute | 56% | 56% | 70% |
| llama-3.1-8b | wk_wv | 8da4w | prefill | texture3d | linear | `linear_dq8ca_q4gsw_coopmat_tsweep_dbuf4zpg_t128x64k32g42s32_texture3d_texture2d_half` | 2048×4096×1024 | 232.8 | 73.80 | 1163 | compute | 52% | 52% | 65% |
| llama-3.1-8b | w1_w3 | 8da4w | prefill | texture3d | linear | `linear_dq8ca_q4gsw_coopmat_tsweep_dbuf4zpg_t128x64k32g42s32_texture3d_texture2d_half` | 2048×4096×14336 | 2841.7 | 84.64 | 2468 | compute | 59% | 59% | 74% |
| llama-3.1-8b | w2 | 8da4w | prefill | texture3d | linear | `linear_dq8ca_q4gsw_coopmat_tsweep_dbuf4zpg_t128x64k32g42s32_texture3d_texture2d_half` | 2048×14336×4096 | 2862.1 | 84.04 | 3146 | compute | 59% | 59% | 74% |
| llama-3.2-3b | wq_wo | 8da4w | prefill | texture3d | linear | `linear_dq8ca_q4gsw_coopmat_tsweep_dbuf4zpg_t128x64k32g42s32_texture3d_texture2d_half` | 2048×3072×3072 | 467.2 | 82.74 | 1626 | compute | 58% | 58% | 73% |
| llama-3.2-3b | wk_wv | 8da4w | prefill | texture3d | linear | `linear_dq8ca_q4gsw_coopmat_tsweep_dbuf4zpg_t128x64k32g42s32_texture3d_texture2d_half` | 2048×3072×1024 | 177.0 | 72.80 | 1062 | compute | 51% | 51% | 64% |
| llama-3.2-3b | w1_w3 | 8da4w | prefill | texture3d | linear | `linear_dq8ca_q4gsw_coopmat_tsweep_dbuf4zpg_t128x64k32g42s32_texture3d_texture2d_half` | 2048×3072×8192 | 1267.0 | 81.35 | 1950 | compute | 57% | 57% | 71% |
| llama-3.2-3b | w2 | 8da4w | prefill | texture3d | linear | `linear_dq8ca_q4gsw_coopmat_tsweep_dbuf4zpg_t128x64k32g42s32_texture3d_texture2d_half` | 2048×8192×3072 | 1261.4 | 81.72 | 2433 | compute | 57% | 57% | 72% |
| llama-3.2-1b | wq_wo | 8da4w | prefill | texture3d | linear | `linear_dq8ca_q4gsw_coopmat_tsweep_dbuf4zpg_t128x64k32g42s32_texture3d_texture2d_half` | 2048×2048×2048 | 231.6 | 74.19 | 1163 | compute | 52% | 52% | 65% |
| llama-3.2-1b | wk_wv | 8da4w | prefill | texture3d | linear | `linear_dq8ca_q4gsw_coopmat_tsweep_dbuf4zpg_t128x64k32g42s32_texture3d_texture2d_half` | 2048×2048×512 | 79.6 | 53.93 | 627 | compute | 38% | 38% | 47% |
| llama-3.2-1b | w1_w3 | 8da4w | prefill | texture3d | linear | `linear_dq8ca_q4gsw_coopmat_tsweep_dbuf4zpg_t128x64k32g42s32_texture3d_texture2d_half` | 2048×2048×8192 | 856.3 | 80.25 | 1479 | compute | 56% | 56% | 70% |
| llama-3.2-1b | w2 | 8da4w | prefill | texture3d | linear | `linear_dq8ca_q4gsw_coopmat_tsweep_dbuf4zpg_t128x64k32g42s32_texture3d_texture2d_half` | 2048×8192×2048 | 919.2 | 74.76 | 2031 | compute | 52% | 52% | 66% |
