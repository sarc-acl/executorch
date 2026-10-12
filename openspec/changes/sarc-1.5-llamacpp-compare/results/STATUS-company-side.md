# Status of the llama.cpp comparison on the company side's devices (COMPANY-SIDE.md section 10)

Updated 2026-10-12 20:13 UTC (10.2, HIP, added). Nothing is pushed. 10.1 and 10.3 are as published (`94a6de3e5`).

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
| RX 7900 XTX | **measured, one gate not met:** HIP (ROCm 7.2.1, built in a container image, host untouched) screened (flash attention on/off x `-ub` {512,1024,2048}, both quantizations) and timed in one interleaved session `hip1` with stock, SARC and tuned ExecuTorch and the Vulkan reference arms (all three models, both timers, default and best, both quantizations; text and GPU-use checks passed). The Vulkan baseline before / after: 1B Vulkan `lb` Q4_0 `best` -1.65 %, 1B tuned `4w` **-2.20 %** against the 2 % bound, so by the request's rule the check is not met; stopped and recorded, no repeat to obtain a pass, **rows not yet appended to `results/cells.csv`** (owner decision) | `7900xtx/hip/` |
| RX 7600 | **built, not run; blocked:** ROCm 7.2.4 is installed and the native `gfx1102` target builds, but the session that did this work has no group `render` (`/dev/kfd` `Permission denied`) and `sg render` fails because the tool harness runs the process with `no_new_privs`; no workaround tried. Needs a process that has the group; everything else is ready (see its README) | `rx7600/README.md` (HIP section) |

HIP on the 7900 XTX in one paragraph: HIP is faster than llama.cpp's Vulkan backend at every size (Q4_0 `lb` best 1.47 / 1.32 / 1.27, Q4_K_M 1.69 / 1.55 / 1.51 for 1B / 3B / 8B). **Tuned ExecuTorch `4w` is no longer ahead of every llama.cpp arm:** at 1B HIP Q4_0 by `lb` is ahead of tuned (tuned / HIP 0.93 at best) and HIP Q4_K_M by `lb` is level (1.00); at 3B and 8B tuned is ahead of HIP by `lb` by 1.19 to 1.26 (best), and by `lc` (fresh process) in every cell (1.20 to 2.01), but all 96 HIP `lc` runs are `clock_low`: under the rule as it stands they have no valid cell (`cells-strict.csv` has none), so those `lc` ratios exist only under the counting rule that still awaits owner confirmation. The report's sentence "ahead of llama.cpp at every size" needs this qualification for the 7900 XTX, subject to the baseline gate above. Details, deviations, clock notes: `7900xtx/hip/README.md`.

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
