# Status of the llama.cpp comparison on the company side's devices (COMPANY-SIDE.md section 10)

Updated 2026-10-13 01:07 UTC (10.2, HIP: 7900 XTX rows appended; RX 7600 HIP measured in session `rx7600-hip2` with every arm, rows appended). Nothing is pushed. 10.1 and 10.3 are as published (`94a6de3e5`).

| device | 10.1 state | where |
|---|---|---|
| RX 7900 XTX | done: session `rx7900xtx-1` (AMDVLK 2025.Q2.1, all 14 arms, three models, both timers, both quantizations); ARMS.md committed before it; rows in `cells.csv`; extras (real-text prompt variant, decode with and without the final configuration) done | `7900xtx/` |
| RX 7600 | done: round 2 final as tuned arm, session `rx7600-r2` (RADV Mesa 26.2.3, all 14 arms, three models); ARMS.md committed before it; rows replaced in `cells.csv`; the 2026-10-08 files are kept in `rx7600/history-2026-10-08/` | `rx7600/` |
| M51 | done (10.3, with the owner decision of 2026-10-10 06:39 UTC): stock Android arm built and measured; sessions m51-4 (1B, 3B) and m51-5 (8B) with stock, SARC, tuned (`4w`, `8da4w`) and the correct-output llama.cpp arm, all timed runs valid; relative speedups committed in `m51/ratios.csv` (tuned / llama.cpp Q4_0 and Q4_K_M by `lc` and `lb`, tuned / SARC, tuned / stock); llama.cpp does not run correctly on that driver as shipped. No absolute figure committed; nothing in `cells.csv` | `m51/README.md`, `m51/ratios.csv` |

Extras of 10.4 (real-text prompt variant, decode with and without the final configuration) are in `7900xtx/` and `rx7600/`; not done for M51.

## 10.2, llama.cpp HIP (ROCm), owner decisions 2026-10-12 18:03 and 18:16 UTC

**Owner decision 2026-10-12 (owner side): the RX 7900 XTX HIP session's rows are accepted and appended to `results/cells.csv` (48 rows, source `7900xtx/hip/cells.csv`), with the baseline check recorded as not met on one cell (2.20 % against 2 %). The table below is as the company side wrote it before that decision.**

| device | state | where |
|---|---|---|
| RX 7900 XTX | **done:** HIP (ROCm 7.2.1, built in a container image, host untouched) screened (flash attention on/off x `-ub` {512,1024,2048}, both quantizations) and timed in one interleaved session `hip1` with stock, SARC and tuned ExecuTorch and the Vulkan reference arms (all three models, both timers, default and best, both quantizations; text and GPU-use checks passed). Vulkan baseline before / after: 1B Vulkan `lb` Q4_0 `best` -1.65 %, 1B tuned `4w` -2.20 %; the second is 0.2 points over the 2 % bound and was **accepted by the owner (2026-10-12 20:21 UTC)**: its prefill is about 89 ms against the runner's 1 ms timer, so 23011 -> 22506 tok/s is two timer steps; the `lb` cell is inside 2 %; nothing was installed between the two measurements. The 24 `hip-` rows are appended to `results/cells.csv` (`kind` = `lc` / `lb` with the `hip-` arm prefix is accepted, `aggregate.py` unchanged) | `7900xtx/hip/` |
| RX 7600 | **done:** HIP (ROCm 7.2.4 user space, native `gfx1102`, no `HSA_OVERRIDE_GFX_VERSION`) screened (kit list, both quantizations; `best` = `-ub 512 -fa on` for Q4_0, `-ub 1024 -fa on` for Q4_K_M) and timed in one interleaved session `rx7600-hip2` (2026-10-12 22:46 to 2026-10-13 00:56 UTC) with all 22 arms: stock, SARC and tuned ExecuTorch, llama.cpp Vulkan (RADV) and HIP, `lc` and `lb`, Q4_0 and Q4_K_M, default and best, 1B / 3B / 8B; GPU use and check text passed; the 66 cells are appended to `results/cells.csv` under session `rx7600-hip2` (earlier rows kept). **The +9 % baseline of 2026-10-12 21:28 UTC is not in the session:** the cell was measured 15 minutes after the workstation's restart (`last -x`: up 2026-10-12 21:12:52 UTC) and its ExecuTorch and `lb` values (+9.5 % / +8.8 %) are absent from the screen anchor (22:20 UTC) and the whole session, which equal the 2026-10-10 session within 0.7 % (ExecuTorch, Vulkan `lb`) and 2.8 % (Vulkan `lc`); cause UNVERIFIED; the 2026-10-10 ratios and absolute numbers stand. Four cells have only counted (`clock_low`) runs: stock `4w` at 3B and 8B, HIP `lc` Q4_0 at 8B (default, best) | `rx7600/hip/README.md`, `rx7600/hip/` |

HIP on the RX 7600 in one paragraph: tuned ExecuTorch `4w` is ahead of llama.cpp HIP in every cell (by `lb` at `best` Q4_0 1.75 / 1.68 / 1.55, Q4_K_M 1.87 / 1.76 / 1.59 for 1B / 3B / 8B; by `lc` 1.56 to 2.24), unlike the 7900 XTX at 1B. HIP is faster than the llama.cpp Vulkan arm by 4 to 9 % (Q4_0) and 19 % (Q4_K_M) by `lb` at 1B and 3B (8B compares with the Vulkan `lc`, because the Vulkan `lb` is the named slow path there: Q4_0 1.05 to 1.06, Q4_K_M 1.22 to 1.27); HIP `llama-bench` has no slow path at 8B. The HIP `lc` Q4_0 cells at 8B and stock `4w` at 3B and 8B rest on counted `clock_low` runs (strict: none); the `lb` ratios do not. Same-session tuned / stock `4w` 4.10 / 4.39* / 4.03*, tuned / SARC `4w` 1.42 / 1.30 / 1.24, as 2026-10-10.

HIP on the 7900 XTX in one paragraph: HIP is faster than llama.cpp's Vulkan backend at every size (Q4_0 `lb` best 1.47 / 1.32 / 1.27, Q4_K_M 1.69 / 1.55 / 1.51 for 1B / 3B / 8B). **Tuned ExecuTorch `4w` is no longer ahead of every llama.cpp arm:** at 1B HIP Q4_0 by `lb` is ahead of tuned (tuned / HIP 0.93 at best) and HIP Q4_K_M by `lb` is level (1.00); at 3B and 8B tuned is ahead of HIP by `lb` by 1.19 to 1.26 (best), and by `lc` (fresh process) in every cell (1.20 to 2.01), but all 96 HIP `lc` runs are `clock_low`: under the rule as it stands they have no valid cell (`cells-strict.csv` has none), so those `lc` ratios exist only under the counting rule that still awaits owner confirmation. The report's sentence "ahead of llama.cpp at every size" needs this qualification for the 7900 XTX (the baseline gate is accepted, see above). Details, deviations, clock notes: `7900xtx/hip/README.md`.

## RX 7900 XTX in one paragraph

Tuned `4w` is ahead of llama.cpp Vulkan Q4_0 (1.36 to 1.75 over all eight arms; 1.36 to 1.56 by `lb` at `best`) and Q4_K_M (1.73 to 2.09; 1.73 to 1.95 by `lb` at `best`) at every model size, by either timer and at both settings; `lc`
pays the card's clock ramp (its `clock_low` runs are counted and shown); stock, parent and tuned were measured in the same session, so the review's
cross-session objection no longer applies (tuned / stock `4w` 3.28 / 4.12 / 3.89; the 3B and 8B stock `4w` values rest on `clock_low` runs only, all eight under the 2670 MHz floor, so they have no strict value; 1B has five valid runs of seven).

**Owner decision 2026-10-12: the counting rule is confirmed for the RX 7900 XTX and the RX 7600.** A run whose only invalidity reason is `clock_low` is counted on these cards as on the 780M (the load itself holds the clock under the floor; no foreign workload is involved), every counted cell stays marked (`n_accepted`, `*` in the tables), and `cells-strict.csv` remains the record of the rule without it. This covers the Vulkan sessions of 2026-10-10 and the HIP session `hip1`, including its 96 `lc` runs. The request below is kept as written.

**Owner decision needed:** `aggregate.py --accept clock_low` was written for the 780M; this task counts `clock_low`-only runs on the 7900 XTX (stock `4w` at 3B and 8B, `lc` Q4_0 and 1B Q4_K_M) and on the RX 7600 (8B stock `4w`) as the 780M and the pushed 2026-10-08 RX 7600 result did. No owner decision extends it to these cards; `cells-strict.csv` holds the rule as it stands. Please confirm or reject the counting rule before the figures are used. Details, deviations and audit: `7900xtx/README.md`.

## RX 7600 in one paragraph

With the round 2 final as the tuned arm, tuned `4w` is ahead of llama.cpp Vulkan Q4_0 by 1.89 / 1.74 / 1.63 (1B / 3B / 8B) and of Q4_K_M by 2.21 / 2.09 / 2.01; the
`llama-bench` slow path at 8B (prompt-only test at 2048 tokens) is shown and named, the 8B llama.cpp number is `lc`. Details: `rx7600/README.md`.

## M51 in one paragraph

Tuned ExecuTorch `4w` is ahead of llama.cpp Vulkan Q4_0 by 2.2 to 2.9 (by `lb` 2.2 / 2.5 / 2.4 at 1B / 3B / 8B) and of Q4_K_M by 2.5 to 3.3, ahead of SARC by 1.12 to 1.30 and of stock by 2.5 to 2.8. llama.cpp's own defaults abort on that driver and its cooperative-matrix path computes wrong output; the reference is the one configuration that matches llama.cpp's CPU backend. Method and intervals: `m51/README.md`, `m51/ratios.csv`.
