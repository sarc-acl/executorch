# Linear Shader Storage-Type Comparison Report (Texture3D vs. Buffer)

## Prefill verdict: Buffer storage is effectively free for the large majority of cases (46/48), with 2 isolated exception(s): llama-3.1-8b/4w/wv (+3.5%), llama-3.2-1b/4w/wk (+4.8%)

## Decode verdict: Buffer storage is effectively free for the large majority of cases (35/48), with 13 isolated exception(s): llama-3.1-8b/4w/wk (-16.5%), llama-3.1-8b/8da4w/wq (+4.0%), llama-3.1-8b/8da4w/wv (+21.8%), llama-3.1-8b/8da4w/w2_down (+2.3%), llama-3.2-3b/4w/w2_down (+3.1%), llama-3.2-3b/8da4w/wv (+51.0%), llama-3.2-3b/8da4w/lm_head (+2.5%), llama-3.2-1b/4w/wk (-14.2%), llama-3.2-1b/4w/lm_head (-3.0%), llama-3.2-1b/8da4w/wq (+7.5%), llama-3.2-1b/8da4w/wk (+26.0%), llama-3.2-1b/8da4w/wv (+25.8%), llama-3.2-1b/8da4w/lm_head (+3.0%)

## Full case table

| Model | Scheme | Regime | Op | Texture3D (us) | Buffer (us) | Diff % | Significance |
|---|---|---|---|---:|---:|---:|---|
| llama-3.1-8b | 4w | decode | lm_head | 3835.5 | 3758.2 | -2.0% | noise |
| llama-3.1-8b | 4w | decode | w1_gate | 395.7 | 398.0 | +0.6% | noise |
| llama-3.1-8b | 4w | decode | w2_down | 418.6 | 423.1 | +1.1% | noise |
| llama-3.1-8b | 4w | decode | w3_up | 396.5 | 398.7 | +0.6% | noise |
| llama-3.1-8b | 4w | decode | wk | 42.7 | 35.7 | -16.5% | real_effect |
| llama-3.1-8b | 4w | decode | wo | 119.9 | 120.3 | +0.3% | noise |
| llama-3.1-8b | 4w | decode | wq | 121.2 | 120.4 | -0.7% | noise |
| llama-3.1-8b | 4w | decode | wv | 36.0 | 34.8 | -3.2% | noise |
| llama-3.1-8b | 4w | prefill | lm_head | 535664.6 | 530404.1 | -1.0% | noise |
| llama-3.1-8b | 4w | prefill | w1_gate | 59020.5 | 59377.8 | +0.6% | noise |
| llama-3.1-8b | 4w | prefill | w2_down | 63123.1 | 64324.8 | +1.9% | noise |
| llama-3.1-8b | 4w | prefill | w3_up | 60071.9 | 59384.8 | -1.1% | noise |
| llama-3.1-8b | 4w | prefill | wk | 5172.0 | 5311.0 | +2.7% | noise |
| llama-3.1-8b | 4w | prefill | wo | 17721.8 | 17687.6 | -0.2% | noise |
| llama-3.1-8b | 4w | prefill | wq | 17634.6 | 17685.5 | +0.3% | noise |
| llama-3.1-8b | 4w | prefill | wv | 5117.4 | 5294.6 | +3.5% | real_effect |
| llama-3.1-8b | 8da4w | decode | lm_head | 3543.1 | 3598.9 | +1.6% | noise |
| llama-3.1-8b | 8da4w | decode | w1_gate | 388.8 | 390.8 | +0.5% | noise |
| llama-3.1-8b | 8da4w | decode | w2_down | 418.9 | 428.4 | +2.3% | real_effect |
| llama-3.1-8b | 8da4w | decode | w3_up | 391.9 | 392.4 | +0.1% | noise |
| llama-3.1-8b | 8da4w | decode | wk | 29.0 | 29.8 | +2.8% | noise |
| llama-3.1-8b | 8da4w | decode | wo | 112.8 | 115.8 | +2.7% | noise |
| llama-3.1-8b | 8da4w | decode | wq | 111.4 | 116.0 | +4.0% | real_effect |
| llama-3.1-8b | 8da4w | decode | wv | 28.1 | 34.3 | +21.8% | real_effect |
| llama-3.1-8b | 8da4w | prefill | lm_head | 408394.8 | 412389.2 | +1.0% | noise |
| llama-3.1-8b | 8da4w | prefill | w1_gate | 46488.4 | 46339.1 | -0.3% | noise |
| llama-3.1-8b | 8da4w | prefill | w2_down | 46991.0 | 46035.6 | -2.0% | noise |
| llama-3.1-8b | 8da4w | prefill | w3_up | 46247.4 | 46082.3 | -0.4% | noise |
| llama-3.1-8b | 8da4w | prefill | wk | 3619.5 | 3501.3 | -3.3% | noise |
| llama-3.1-8b | 8da4w | prefill | wo | 13526.7 | 13292.1 | -1.7% | noise |
| llama-3.1-8b | 8da4w | prefill | wq | 13695.8 | 13289.5 | -3.0% | noise |
| llama-3.1-8b | 8da4w | prefill | wv | 3534.7 | 3529.9 | -0.1% | noise |
| llama-3.2-1b | 4w | decode | lm_head | 1948.3 | 1890.4 | -3.0% | real_effect |
| llama-3.2-1b | 4w | decode | w1_gate | 122.0 | 123.5 | +1.2% | noise |
| llama-3.2-1b | 4w | decode | w2_down | 129.3 | 129.1 | -0.1% | noise |
| llama-3.2-1b | 4w | decode | w3_up | 123.3 | 121.9 | -1.1% | noise |
| llama-3.2-1b | 4w | decode | wk | 15.4 | 13.2 | -14.2% | real_effect |
| llama-3.2-1b | 4w | decode | wo | 34.1 | 34.1 | +0.1% | noise |
| llama-3.2-1b | 4w | decode | wq | 33.7 | 35.8 | +6.0% | noise |
| llama-3.2-1b | 4w | decode | wv | 13.3 | 13.3 | -0.5% | noise |
| llama-3.2-1b | 4w | prefill | lm_head | 298497.5 | 300334.8 | +0.6% | noise |
| llama-3.2-1b | 4w | prefill | w1_gate | 18976.5 | 19814.0 | +4.4% | noise |
| llama-3.2-1b | 4w | prefill | w2_down | 20003.7 | 20610.0 | +3.0% | noise |
| llama-3.2-1b | 4w | prefill | w3_up | 19471.4 | 19551.4 | +0.4% | noise |
| llama-3.2-1b | 4w | prefill | wk | 1402.9 | 1469.5 | +4.8% | real_effect |
| llama-3.2-1b | 4w | prefill | wo | 4930.5 | 4895.0 | -0.7% | noise |
| llama-3.2-1b | 4w | prefill | wq | 4917.1 | 4929.1 | +0.2% | noise |
| llama-3.2-1b | 4w | prefill | wv | 1440.0 | 1490.1 | +3.5% | noise |
| llama-3.2-1b | 8da4w | decode | lm_head | 1745.3 | 1797.2 | +3.0% | real_effect |
| llama-3.2-1b | 8da4w | decode | w1_gate | 112.4 | 112.2 | -0.2% | noise |
| llama-3.2-1b | 8da4w | decode | w2_down | 112.8 | 114.8 | +1.8% | noise |
| llama-3.2-1b | 8da4w | decode | w3_up | 112.1 | 113.2 | +0.9% | noise |
| llama-3.2-1b | 8da4w | decode | wk | 7.4 | 9.3 | +26.0% | real_effect |
| llama-3.2-1b | 8da4w | decode | wo | 28.9 | 30.9 | +6.8% | noise |
| llama-3.2-1b | 8da4w | decode | wq | 28.7 | 30.9 | +7.5% | real_effect |
| llama-3.2-1b | 8da4w | decode | wv | 7.4 | 9.3 | +25.8% | real_effect |
| llama-3.2-1b | 8da4w | prefill | lm_head | 214388.7 | 212570.5 | -0.8% | noise |
| llama-3.2-1b | 8da4w | prefill | w1_gate | 13977.9 | 13929.7 | -0.3% | noise |
| llama-3.2-1b | 8da4w | prefill | w2_down | 14435.0 | 14348.9 | -0.6% | noise |
| llama-3.2-1b | 8da4w | prefill | w3_up | 13969.2 | 14238.7 | +1.9% | noise |
| llama-3.2-1b | 8da4w | prefill | wk | 984.4 | 1012.2 | +2.8% | noise |
| llama-3.2-1b | 8da4w | prefill | wo | 3601.7 | 3543.0 | -1.6% | noise |
| llama-3.2-1b | 8da4w | prefill | wq | 3575.7 | 3597.1 | +0.6% | noise |
| llama-3.2-1b | 8da4w | prefill | wv | 997.5 | 1013.6 | +1.6% | noise |
| llama-3.2-3b | 4w | decode | lm_head | 2829.5 | 2842.1 | +0.4% | noise |
| llama-3.2-3b | 4w | decode | w1_gate | 183.0 | 183.4 | +0.2% | noise |
| llama-3.2-3b | 4w | decode | w2_down | 184.0 | 189.7 | +3.1% | real_effect |
| llama-3.2-3b | 4w | decode | w3_up | 183.3 | 181.5 | -0.9% | noise |
| llama-3.2-3b | 4w | decode | wk | 27.3 | 27.1 | -0.7% | noise |
| llama-3.2-3b | 4w | decode | wo | 71.8 | 72.5 | +0.9% | noise |
| llama-3.2-3b | 4w | decode | wq | 70.6 | 70.8 | +0.3% | noise |
| llama-3.2-3b | 4w | decode | wv | 26.6 | 26.8 | +0.9% | noise |
| llama-3.2-3b | 4w | prefill | lm_head | 435026.9 | 440656.0 | +1.3% | noise |
| llama-3.2-3b | 4w | prefill | w1_gate | 27953.8 | 27654.3 | -1.1% | noise |
| llama-3.2-3b | 4w | prefill | w2_down | 28735.9 | 30125.1 | +4.8% | noise |
| llama-3.2-3b | 4w | prefill | w3_up | 27444.4 | 27629.5 | +0.7% | noise |
| llama-3.2-3b | 4w | prefill | wk | 3867.3 | 3922.4 | +1.4% | noise |
| llama-3.2-3b | 4w | prefill | wo | 10376.8 | 10693.1 | +3.0% | noise |
| llama-3.2-3b | 4w | prefill | wq | 10599.3 | 10575.2 | -0.2% | noise |
| llama-3.2-3b | 4w | prefill | wv | 3864.8 | 3999.8 | +3.5% | noise |
| llama-3.2-3b | 8da4w | decode | lm_head | 2712.3 | 2780.7 | +2.5% | real_effect |
| llama-3.2-3b | 8da4w | decode | w1_gate | 169.9 | 171.5 | +0.9% | noise |
| llama-3.2-3b | 8da4w | decode | w2_down | 175.0 | 177.9 | +1.7% | noise |
| llama-3.2-3b | 8da4w | decode | w3_up | 169.7 | 170.1 | +0.2% | noise |
| llama-3.2-3b | 8da4w | decode | wk | 19.5 | 22.9 | +17.8% | noise |
| llama-3.2-3b | 8da4w | decode | wo | 66.3 | 64.6 | -2.6% | noise |
| llama-3.2-3b | 8da4w | decode | wq | 67.0 | 65.0 | -2.9% | noise |
| llama-3.2-3b | 8da4w | decode | wv | 15.4 | 23.3 | +51.0% | real_effect |
| llama-3.2-3b | 8da4w | prefill | lm_head | 319457.4 | 315685.2 | -1.2% | noise |
| llama-3.2-3b | 8da4w | prefill | w1_gate | 20178.0 | 20548.4 | +1.8% | noise |
| llama-3.2-3b | 8da4w | prefill | w2_down | 21015.8 | 20939.5 | -0.4% | noise |
| llama-3.2-3b | 8da4w | prefill | w3_up | 20401.0 | 20647.1 | +1.2% | noise |
| llama-3.2-3b | 8da4w | prefill | wk | 2655.0 | 2654.4 | -0.0% | noise |
| llama-3.2-3b | 8da4w | prefill | wo | 7767.8 | 7700.5 | -0.9% | noise |
| llama-3.2-3b | 8da4w | prefill | wq | 7839.5 | 7719.3 | -1.5% | noise |
| llama-3.2-3b | 8da4w | prefill | wv | 2776.9 | 2657.2 | -4.3% | noise |

## Infeasible / contaminated cases

none

## Cross-check against 001's published Texture3D numbers

consistent across all 96 checked cases (within the same significance band)
