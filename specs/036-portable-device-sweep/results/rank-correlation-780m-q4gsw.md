# Microbench vs e2e rank correlation -- q4gsw (780M, specs/035 round-1 data)

Tokens joined: 16

| ranking signal | spearman rho | e2e-top5 in signal-top10 |
|---|---|---|
| linear_q4gsw_M1024_K14336_N4096_Buffer | 0.876 | 1.00 |
| linear_q4gsw_M1024_K4096_N1024_Buffer | 0.794 | 0.80 |
| linear_q4gsw_M1024_K4096_N14336_Buffer | 0.912 | 1.00 |
| linear_q4gsw_M1024_K4096_N4096_Buffer | 0.835 | 1.00 |
| mean_gflops | 0.865 | 1.00 |

Best signal: `linear_q4gsw_M1024_K4096_N14336_Buffer` (rho=0.912, recall=1.00)

Verdict: **Usable as prefilter**: rank new-device candidates by `linear_q4gsw_M1024_K4096_N14336_Buffer` before spending e2e time.
