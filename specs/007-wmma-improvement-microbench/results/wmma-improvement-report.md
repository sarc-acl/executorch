# WMMA Coopmat Improvement Microbenchmark Report

**By scheme (time-weighted across each scheme's 21 measured ops, weighted by each op's own share of its configuration's total tiled-baseline time -- see research.md Decision 6 addendum):**
- `4w`: WMMA is **+60.6% faster** than tiled
- `8da4w`: WMMA is **-15.2% slower** than tiled

**Blended overall (both schemes combined, equal weight per configuration): +22.7%.** This single number is provided for completeness but should not be read alone -- it averages a large, consistent `4w` win against a consistent `8da4w` regression (see table below); neither scheme's result is noise (every row in both schemes shows the same-direction effect).

**Statistical basis (FR-003)**: every `Tiled`/`WMMA` value below is a mean ± standard deviation over 5 timed runs, confirmed uniform across every one of the 42 rows in both the tiled-baseline (`004`) and WMMA captures -- no result here is a single untimed sample.

## Full case table

| Model | Scheme | Op | Tiled (us) | WMMA (us) | Speedup % | Significance | Dispatch | Correctness |
|---|---|---|---:|---:|---:|---|---|---|
| llama-3.1-8b | 4w | w1_gate | 59377.8 ± 755.2 | 24174.2 ± 281.0 | +59.3% | real_effect | confirmed | verified |
| llama-3.1-8b | 4w | w2_down | 64324.8 ± 1501.5 | 27958.7 ± 87.3 | +56.5% | real_effect | confirmed | verified |
| llama-3.1-8b | 4w | w3_up | 59384.8 ± 438.9 | 24570.2 ± 613.4 | +58.6% | real_effect | confirmed | verified |
| llama-3.1-8b | 4w | wk | 5311.0 ± 32.4 | 1890.6 ± 12.3 | +64.4% | real_effect | confirmed | verified |
| llama-3.1-8b | 4w | wo | 17687.6 ± 110.2 | 7142.0 ± 62.0 | +59.6% | real_effect | confirmed | verified |
| llama-3.1-8b | 4w | wq | 17685.5 ± 101.3 | 7148.3 ± 28.3 | +59.6% | real_effect | confirmed | verified |
| llama-3.1-8b | 4w | wv | 5294.6 ± 24.9 | 1886.6 ± 11.3 | +64.4% | real_effect | confirmed | verified |
| llama-3.1-8b | 8da4w | w1_gate | 46339.1 ± 1071.4 | 53428.4 ± 54.9 | -15.3% | real_effect | confirmed | verified |
| llama-3.1-8b | 8da4w | w2_down | 46035.6 ± 506.2 | 56174.5 ± 436.7 | -22.0% | real_effect | confirmed | verified |
| llama-3.1-8b | 8da4w | w3_up | 46082.3 ± 647.5 | 53293.2 ± 79.5 | -15.7% | real_effect | confirmed | verified |
| llama-3.1-8b | 8da4w | wk | 3501.3 ± 15.8 | 4244.4 ± 14.5 | -21.2% | real_effect | confirmed | verified |
| llama-3.1-8b | 8da4w | wo | 13292.1 ± 83.8 | 15593.6 ± 63.4 | -17.3% | real_effect | confirmed | verified |
| llama-3.1-8b | 8da4w | wq | 13289.5 ± 70.7 | 15495.6 ± 8.0 | -16.6% | real_effect | confirmed | verified |
| llama-3.1-8b | 8da4w | wv | 3529.9 ± 40.6 | 4255.3 ± 21.1 | -20.6% | real_effect | confirmed | verified |
| llama-3.2-1b | 4w | w1_gate | 19814.0 ± 1035.2 | 7199.9 ± 32.1 | +63.7% | real_effect | confirmed | verified |
| llama-3.2-1b | 4w | w2_down | 20610.0 ± 342.5 | 8357.0 ± 177.1 | +59.5% | real_effect | confirmed | verified |
| llama-3.2-1b | 4w | w3_up | 19551.4 ± 1062.4 | 7304.1 ± 136.8 | +62.6% | real_effect | confirmed | verified |
| llama-3.2-1b | 4w | wk | 1469.5 ± 7.4 | 512.7 ± 9.9 | +65.1% | real_effect | confirmed | verified |
| llama-3.2-1b | 4w | wo | 4895.0 ± 97.0 | 1860.1 ± 30.6 | +62.0% | real_effect | confirmed | verified |
| llama-3.2-1b | 4w | wq | 4929.1 ± 122.4 | 1938.0 ± 72.5 | +60.7% | real_effect | confirmed | verified |
| llama-3.2-1b | 4w | wv | 1490.1 ± 23.5 | 512.7 ± 16.5 | +65.6% | real_effect | confirmed | verified |
| llama-3.2-1b | 8da4w | w1_gate | 13929.7 ± 239.7 | 15608.6 ± 72.6 | -12.1% | real_effect | confirmed | verified |
| llama-3.2-1b | 8da4w | w2_down | 14348.9 ± 253.1 | 16798.5 ± 66.3 | -17.1% | real_effect | confirmed | verified |
| llama-3.2-1b | 8da4w | w3_up | 14238.7 ± 514.5 | 15735.4 ± 335.6 | -10.5% | noise | confirmed | verified |
| llama-3.2-1b | 8da4w | wk | 1012.2 ± 29.2 | 1178.6 ± 16.3 | -16.4% | real_effect | confirmed | verified |
| llama-3.2-1b | 8da4w | wo | 3543.0 ± 65.9 | 4059.7 ± 4.8 | -14.6% | real_effect | confirmed | verified |
| llama-3.2-1b | 8da4w | wq | 3597.1 ± 79.2 | 4061.1 ± 19.3 | -12.9% | real_effect | confirmed | verified |
| llama-3.2-1b | 8da4w | wv | 1013.6 ± 36.8 | 1177.2 ± 5.0 | -16.1% | real_effect | confirmed | verified |
| llama-3.2-3b | 4w | w1_gate | 27654.3 ± 359.9 | 10653.6 ± 61.1 | +61.5% | real_effect | confirmed | verified |
| llama-3.2-3b | 4w | w2_down | 30125.1 ± 526.1 | 12129.5 ± 239.1 | +59.7% | real_effect | confirmed | verified |
| llama-3.2-3b | 4w | w3_up | 27629.5 ± 596.7 | 10823.6 ± 381.6 | +60.8% | real_effect | confirmed | verified |
| llama-3.2-3b | 4w | wk | 3922.4 ± 35.3 | 1424.6 ± 29.6 | +63.7% | real_effect | confirmed | verified |
| llama-3.2-3b | 4w | wo | 10693.1 ± 253.5 | 4035.9 ± 110.7 | +62.3% | real_effect | confirmed | verified |
| llama-3.2-3b | 4w | wq | 10575.2 ± 274.9 | 4016.4 ± 94.1 | +62.0% | real_effect | confirmed | verified |
| llama-3.2-3b | 4w | wv | 3999.8 ± 98.1 | 1429.0 ± 27.5 | +64.3% | real_effect | confirmed | verified |
| llama-3.2-3b | 8da4w | w1_gate | 20548.4 ± 287.6 | 23075.2 ± 30.0 | -12.3% | real_effect | confirmed | verified |
| llama-3.2-3b | 8da4w | w2_down | 20939.5 ± 645.9 | 24538.5 ± 279.7 | -17.2% | real_effect | confirmed | verified |
| llama-3.2-3b | 8da4w | w3_up | 20647.1 ± 257.9 | 23068.2 ± 8.8 | -11.7% | real_effect | confirmed | verified |
| llama-3.2-3b | 8da4w | wk | 2654.4 ± 36.0 | 3199.5 ± 22.3 | -20.5% | real_effect | confirmed | verified |
| llama-3.2-3b | 8da4w | wo | 7700.5 ± 25.2 | 8897.0 ± 106.6 | -15.5% | real_effect | confirmed | verified |
| llama-3.2-3b | 8da4w | wq | 7719.3 ± 92.1 | 8920.5 ± 158.8 | -15.6% | real_effect | confirmed | verified |
| llama-3.2-3b | 8da4w | wv | 2657.2 ± 12.7 | 3194.5 ± 12.8 | -20.2% | real_effect | confirmed | verified |

## Excluded / Out-of-Scope

- `lm_head`, all 6 configurations: excluded -- the harness's synthetic M=2048 'prefill' case for this op has no production analogue; the real model's lm_head/vocab projection is always M=1 (a GEMV) regardless of phase (research.md Decision 3).
- Decode-regime linear ops, all configurations: excluded -- no WMMA-capable GEMV (M=1) kernel exists today (003's classification 'c', FR-006).
- No other exclusions.

## Correctness-verification summary

- `linear_q4gsw_coopmat`: SPIR-V inspection confirmed genuine cooperative-matrix instructions (`OpCooperativeMatrixLoadKHR`/`OpCooperativeMatrixMulAddKHR`); existing correctness coverage via `test_coopmat_linear_bench.cpp`'s `kCorrectnessShapes` confirmed (research.md Decision 7).
- `linear_dq8ca_q4gsw_coopmat`: SPIR-V inspection confirmed genuine cooperative-matrix instructions (`OpCooperativeMatrixLoadKHR`/`OpCooperativeMatrixMulAddKHR`); existing correctness coverage via `test_coopmat_linear_bench.cpp`'s `kCorrectnessShapes` confirmed (research.md Decision 7).
