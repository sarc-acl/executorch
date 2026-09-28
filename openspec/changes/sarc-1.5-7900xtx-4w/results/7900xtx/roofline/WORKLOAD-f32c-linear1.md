# ExecuTorch kernels on the Radeon RX 7900 XTX roofline

Source: `results/newdev-20260927/7900xtx-mb/f32c-linear1.json` (ExecuTorch device: radeon rx 7900 xtx, group size 128).

Ops = 2·M·N·K (integer ops for 8da4w). Bytes = compulsory traffic (int4 weights + fp16 scales + inputs + fp16 output, each once), so intensity is an upper bound. Attainable = min(compute roof, intensity × DRAM read roof). Only confirmed roofs with repeat spread ≤ 5 % are used.

## Roofs used

| roof | confirmed median | unit | repeat spread |
|---|---:|---|---:|
| global_read | 1108.158 | GB/s | 0.1% |
| matrix_fp16_feed_shared | 127.499 | TFLOP/s | 1.2% |
| matrix_fp16_fp32 | 141.909 | TFLOP/s | 0.2% |

## Kernels

| model | layer | scheme | regime | storage | suite | kernel | M×K×N | µs | TOP/s | ops/B | bound | % attainable | % compute roof | % shared-fed MMA |
|---|---|---|---|---|---|---|---|---:|---:|---:|---|---:|---:|---:|
| llama-3.1-8b | wq_wo | 4w | prefill | texture3d | linear | `linear_q4gsw_coopmat_tsweep_dbuf4_t128x128k32g42s32f32c_texture3d_texture2d_half` | 2048×4096×4096 | 1147.1 | 59.91 | 1628 | compute | 42% | 42% | 47% |
| llama-3.1-8b | wk_wv | 4w | prefill | texture3d | linear | `linear_q4gsw_coopmat_tsweep_dbuf4_t128x128k32g42s32f32c_texture3d_texture2d_half` | 2048×4096×1024 | 363.3 | 47.29 | 743 | compute | 33% | 33% | 37% |
| llama-3.1-8b | w1_w3 | 4w | prefill | texture3d | linear | `linear_q4gsw_coopmat_tsweep_dbuf4_t128x128k32g42s32f32c_texture3d_texture2d_half` | 2048×4096×14336 | 3506.8 | 68.59 | 2274 | compute | 48% | 48% | 54% |
| llama-3.1-8b | w2 | 4w | prefill | texture3d | linear | `linear_q4gsw_coopmat_tsweep_dbuf4_t128x128k32g42s32f32c_texture3d_texture2d_half` | 2048×14336×4096 | 3739.9 | 64.31 | 2274 | compute | 45% | 45% | 50% |
| llama-3.2-3b | wq_wo | 4w | prefill | texture3d | linear | `linear_q4gsw_coopmat_tsweep_dbuf4_t128x128k32g42s32f32c_texture3d_texture2d_half` | 2048×3072×3072 | 612.9 | 63.07 | 1287 | compute | 44% | 44% | 49% |
| llama-3.2-3b | wk_wv | 4w | prefill | texture3d | linear | `linear_q4gsw_coopmat_tsweep_dbuf4_t128x128k32g42s32f32c_texture3d_texture2d_half` | 2048×3072×1024 | 280.7 | 45.91 | 700 | compute | 32% | 32% | 36% |
| llama-3.2-3b | w1_w3 | 4w | prefill | texture3d | linear | `linear_q4gsw_coopmat_tsweep_dbuf4_t128x128k32g42s32f32c_texture3d_texture2d_half` | 2048×3072×8192 | 1625.7 | 63.41 | 1744 | compute | 45% | 45% | 50% |
| llama-3.2-3b | w2 | 4w | prefill | texture3d | linear | `linear_q4gsw_coopmat_tsweep_dbuf4_t128x128k32g42s32f32c_texture3d_texture2d_half` | 2048×8192×3072 | 1606.3 | 64.17 | 1744 | compute | 45% | 45% | 50% |
| llama-3.2-1b | wq_wo | 4w | prefill | texture3d | linear | `linear_q4gsw_coopmat_tsweep_dbuf4_t128x128k32g42s32f32c_texture3d_texture2d_half` | 2048×2048×2048 | 302.5 | 56.79 | 907 | compute | 40% | 40% | 45% |
| llama-3.2-1b | wk_wv | 4w | prefill | texture3d | linear | `linear_q4gsw_coopmat_tsweep_dbuf4_t128x128k32g42s32f32c_texture3d_texture2d_half` | 2048×2048×512 | 103.3 | 41.59 | 390 | compute | 29% | 29% | 33% |
| llama-3.2-1b | w1_w3 | 4w | prefill | texture3d | linear | `linear_q4gsw_coopmat_tsweep_dbuf4_t128x128k32g42s32f32c_texture3d_texture2d_half` | 2048×2048×8192 | 1100.8 | 62.42 | 1358 | compute | 44% | 44% | 49% |
| llama-3.2-1b | w2 | 4w | prefill | texture3d | linear | `linear_q4gsw_coopmat_tsweep_dbuf4_t128x128k32g42s32f32c_texture3d_texture2d_half` | 2048×8192×2048 | 1163.4 | 59.07 | 1358 | compute | 42% | 42% | 46% |
