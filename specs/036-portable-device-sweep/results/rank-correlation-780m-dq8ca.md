# Microbench vs e2e rank correlation -- dq8ca (780M, specs/035 round-1 data)

Tokens joined: 36

| ranking signal | spearman rho | e2e-top5 in signal-top10 |
|---|---|---|
| linear_dq8ca_q4gsw_M1024_K14336_N4096_Buffer | 0.997 | 1.00 |
| linear_dq8ca_q4gsw_M1024_K4096_N1024_Buffer | 0.997 | 1.00 |
| linear_dq8ca_q4gsw_M1024_K4096_N14336_Buffer | 0.998 | 1.00 |
| linear_dq8ca_q4gsw_M1024_K4096_N4096_Buffer | 0.999 | 1.00 |
| mean_gflops | 0.999 | 1.00 |

Best signal: `linear_dq8ca_q4gsw_M1024_K4096_N4096_Buffer` (rho=0.999, recall=1.00)

Verdict: **Usable as prefilter**: rank new-device candidates by `linear_dq8ca_q4gsw_M1024_K4096_N4096_Buffer` before spending e2e time.
