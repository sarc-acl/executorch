# Arc Pro B70: arms and results

Session `s1`, 2026-10-06, card b70-0, taken while the tuning campaign was held on both cards. 14 arms; exact
lines in `arms.tsv`. Validity: the campaign's calibrated clock floor (2505 MHz median in the prefill window)
and no other GPU workload; this adapter has no foreign-busy ceiling (dedicated card, campaign held).

- ExecuTorch: `stock` (the `release/1.5` binary built for the B580 session), `sarc` = `6a7cc8cc6` (the
  campaign's pristine parent), `tuned` = `8666b6531` with `ET_VK_SARC_UNVERIFIED=1
  ET_VK_SARC_DEV_PROFILE=xe2-refine2` (unmerged development branch; accepted by the reference-error rule).
- llama.cpp `b11430`, built for this host's CPU (`-march=raptorlake`; the VM has no AVX-512). Vulkan
  (`-DGGML_VULKAN=ON`) and SYCL (oneAPI DPC++ 2026.1.1, `-DGGML_SYCL=ON -DGGML_SYCL_F16=ON`, Level Zero paths on,
  run as a host process with `ZE_AFFINITY_MASK=0`).
- `best` settings from the 1B screen (`screen.csv`, llama-bench tok/s): Vulkan default 8551, `-ub 1024` flash
  attention off **9731**, `-ub 2048` off 8261 / on 8600, `-ub 512` off 8567; SYCL default 17853, `-ub 2048` on
  **21514**, `-ub 1024` on 20775, `-ub 2048` off 8677.

## Result (tok/s, median of 5; llama.cpp by llama-bench at its best setting, Q4_0): session `s2`

Session `s2`, 2026-10-06 19:38 to 20:01 UTC, after the tuning campaign had finished and pushed. Same 14 arms,
script and validity rule as `s1`; the only change is the `tuned` arm, now the campaign's final build
`ff29c08ef` with `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=xe2-refine5` (the binary of its final session).
`cells.csv`, `runs.csv` and `checks.csv` are `s2`; the first session is kept as `*-s1.csv`.

| model | stock 4w | stock 8da4w | SARC 4w | SARC 8da4w | tuned 4w | tuned 8da4w | llama.cpp Vulkan | llama.cpp SYCL | tuned 4w / Vulkan | tuned 4w / SYCL | tuned 8da4w / SYCL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1B | 4592 | 8359 | 11703 | 12412 | 17965 | 20687 | 9697 | 21547 | 1.85 | 0.83 | 0.96 |
| 3B | 1705 | 3266 | 4842 | 5211 | 7502 | 9526 | 5203 | 9300 | 1.44 | 0.81 | 1.02 |
| 8B | 781 | 1381 | 2435 | 2731 | 3380 | 4491 | 2743 | 4530 | 1.23 | 0.75 | 0.99 |

Five valid runs in every cell of this table. Against `s1`: every arm other than tuned `4w` is within 1.8 % (the
llama.cpp arms within 0.2 % on SYCL and 1.8 % on Vulkan); tuned `4w` is 2.6 / 1.8 / 2.9 % higher, the gain of the
final profile's `4w` tiles, and reproduces the campaign's final session (17965 / 7529 / 3380) within 0.4 %.
The session reads `SESSION_INCOMPLETE` for one reason only, as in `s1`: llama.cpp's fresh-process timer on SYCL
is rejected by the clock rule on 1B and 3B (the cold-start cost of the SYCL runtime); on 8B it reads 2037 tok/s
against 4530 warm. Text check: all arms continue the real-text prompt fluently.

## Result of the first session `s1` (tuned = `8666b6531`, profile `xe2-refine2`)

| model | stock 4w | stock 8da4w | SARC 4w | SARC 8da4w | tuned 4w | tuned 8da4w | llama.cpp Vulkan | llama.cpp SYCL | tuned 4w / Vulkan | tuned 4w / SYCL | tuned 8da4w / SYCL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1B | 4582 | 8325 | 11703 | 12412 | 17504 | 20687 | 9704 | 21571 | 1.80 | 0.81 | 0.96 |
| 3B | 1702 | 3256 | 4830 | 5265 | 7367 | 9570 | 5232 | 9296 | 1.41 | 0.79 | 1.03 |
| 8B | 778 | 1375 | 2421 | 2734 | 3285* | 4481 | 2695 | 4521 | 1.22 | 0.73 | 0.99 |

\* four valid runs: two of six read a median clock of 2500 MHz, 5 MHz under the floor, at the same speed
(3277 and 3272 tok/s); the other four read 2517 MHz.

The ExecuTorch arms reproduce the campaign's final session within 0.3 % and the September table within 1 %.
Q4_K_M runs within 4 % of Q4_0 on both backends. llama.cpp's fresh-process timer on SYCL reads 2.1 to 4.8 times
lower than its warm timer and its runs are rejected by the clock rule on 1B and 3B, as on the Arc B580: the
cold-start cost of the SYCL runtime, not GPU work. Text check: all arms continue the real-text prompt fluently.
