# Radeon RX 7900 XTX: llama.cpp comparison, one interleaved session

Session `rx7900xtx-1`, 2026-10-10 01:06 to 02:03 UTC, on `<gpu-host>` (the card is the host's display adapter, idle), 14 arms
(`arms.tsv`, `ARMS.md`, committed before the session) on 1B, 3B and 8B, after the tuning campaign had finished. **Driver for every
arm, ExecuTorch and llama.cpp alike: AMDVLK 2025.Q2.1 (LLPC)**, `VK_ICD_FILENAMES` of the campaign; no RADV set. Validity: the
campaign's rule as `row.py` applies it, through the new adapter `kit/hosts/7900xtx/host.sh` (clock floor 2670 MHz, thermal bit
36 masked as the campaign's owner decision says, card idle before each run, no foreign GPU user or build seen; see `ARMS.md`).
The session started from a cool card (edge and junction at 27 C) under the campaign's gpu-lab lock; the screen that fixed `best`
(`screen.csv`, `ARMS.md`) was run the hour before with the same adapter.

Files: `runs.csv` (every run, valid or not, with its reason: 113 timed runs valid, 82 not, 12 discarded `lc` repetitions 0, 30 text
checks), `cells.csv` (the table's source, counts `clock_low`-only runs as the 780M and the RX 7600 do, see below),
`cells-strict.csv` (the rule as it stands), `checks.csv`, `screen.csv`, `recompute.txt` (the medians of both cells files recomputed
from `runs.csv` independently of `aggregate.py`: 42 cells, no difference).

## Result (tok/s, median; llama.cpp = the higher of its two timers at its better setting)

| model | stock 4w | stock 8da4w | SARC 4w | SARC 8da4w | tuned 4w | tuned 8da4w | llama.cpp Q4_0 | llama.cpp Q4_K_M | tuned 4w / Q4_0 | tuned 4w / Q4_K_M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1B | 6781* | 10240 | 19884 | 22506 | 22261 | 24675 | 16409 | 12877 | 1.36 | 1.73 |
| 3B | 2614* | 4104 | 10240 | 10557 | 10779 | 11636 | 6907 | 5529 | 1.56 | 1.95 |
| 8B | 1259* | 2248 | 4763 | 4983 | 4900 | 5319 | 3241 | 2581 | 1.51 | 1.90 |

llama.cpp, all eight arms (tok/s):

| model | lc Q4_0 default | lc Q4_0 best | lb Q4_0 default | lb Q4_0 best | lc Q4_K_M default | lc Q4_K_M best | lb Q4_K_M default | lb Q4_K_M best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1B | 14749* | 14995* | 15950 | 16409 | 12068* | 12465* | 12392 | 12877 |
| 3B | 6171* | 6406* | 6562 | 6907 | 5162 | 5264* | 5388 | 5529 |
| 8B | 3136* | 3171* | 3176 | 3241 | 2467 | 2499 | 2520 | 2581 |

Ratios tuned 4w / llama.cpp, by timer and setting:

| model | Q4_0 lc default | Q4_0 lc best | Q4_0 lb default | Q4_0 lb best | Q4_K_M lc default | Q4_K_M lc best | Q4_K_M lb default | Q4_K_M lb best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1B | 1.51 | 1.48 | 1.40 | 1.36 | 1.84 | 1.79 | 1.80 | 1.73 |
| 3B | 1.75 | 1.68 | 1.64 | 1.56 | 2.09 | 2.05 | 2.00 | 1.95 |
| 8B | 1.56 | 1.54 | 1.54 | 1.51 | 1.99 | 1.96 | 1.94 | 1.90 |

Valid runs per cell under the rule as it stands / counted (`*` = counted with `clock_low` as the only reason):

| arm | 1B | 3B | 8B |
|---|---:|---:|---:|
| `stock-4w` | 5 / 7 | 0 / 8 | 0 / 8 |
| `stock-8da4w` | 5 / 5 | 5 / 5 | 5 / 5 |
| `sarc-4w` | 5 / 5 | 5 / 5 | 5 / 5 |
| `sarc-8da4w` | 5 / 5 | 5 / 5 | 5 / 5 |
| `tuned-4w` | 5 / 5 | 5 / 5 | 5 / 5 |
| `tuned-8da4w` | 5 / 5 | 5 / 5 | 5 / 5 |
| `lc-q4_0-default` | 0 / 8 | 0 / 8 | 0 / 8 |
| `lc-q4_0-best` | 0 / 8 | 0 / 8 | 0 / 8 |
| `lc-q4_k_m-default` | 1 / 8 | 5 / 5 | 5 / 5 |
| `lc-q4_k_m-best` | 0 / 8 | 5 / 6 | 5 / 5 |
| `lb-q4_0-default` | 5 / 5 | 5 / 5 | 5 / 5 |
| `lb-q4_0-best` | 5 / 5 | 5 / 5 | 5 / 5 |
| `lb-q4_k_m-default` | 5 / 5 | 5 / 5 | 5 / 5 |
| `lb-q4_k_m-best` | 5 / 5 | 5 / 5 | 5 / 5 |
\* counted under the clock note below. Five valid runs in every other ExecuTorch and `lc` cell (the strict count is in the last table);
`lb` = one process of five repetitions. The ExecuTorch prefill timer has 1 ms resolution: a 1B prefill of about 90 ms moves by 1 % per
millisecond.

Same-session ratios over the stock arm (the cross-session objection of the review): tuned `4w` / stock `4w` 3.28 / 4.12 / 3.89
(1B / 3B / 8B), tuned `8da4w` / stock `8da4w` 2.41 / 2.84 / 2.37; SARC `4w` / stock `4w` 2.93 / 3.92 / 3.78, SARC `8da4w` / stock
`8da4w` 2.20 / 2.57 / 2.22; tuned / SARC `4w` 1.12 / 1.05 / 1.03, `8da4w` 1.10 / 1.10 / 1.07 (the campaign's final session:
1.12 / 1.05 / 1.03 and 1.11 / 1.12 / 1.07). Stock `4w` is counted under the clock note (3B and 8B: all eight runs).

Both timers, both quantizations: tuned `4w` is ahead of llama.cpp Vulkan Q4_0 by 1.36 to 1.56 at its better setting (1.40 to 1.64 at
the default; 1.48 to 1.68 by `lc` alone) and ahead of Q4_K_M by 1.73 to 1.95 (1.80 to 2.09 at the default). Q4_K_M is 20 to 22 %
slower than Q4_0. `best` (`-b 2048 -ub 1024 -fa on`) is 2 to 5 % above `default`. `lb` is 2 to 9 % above `lc` for the same setting, the
`lc` number being the one that pays the clock ramp of a fresh process (below); no llama-bench slow path above 1024 prompt tokens
was seen on this card (8B `lb` and `lc` within 3.5 %).

## Clock notes: `clock_low` is the only reason of every invalid timed run (82 of 195), and it is counted

- **`lc`, fresh process.** `llama-completion` has no full-prompt warm-up in its process, so the card, idle during the cooling
  wait before each run, ramps its shader clock up inside the prompt window (one 1B run: 23 samples, median 2534 MHz, minimum 1169 MHz).
  The median over the window is 2450 to 2670 MHz for Q4_0 and 1B Q4_K_M, below the campaign's 2670 MHz floor,
  so `row.py` marks every such run `clock_low`, replaces it (three extra rounds) and the cell ends with no valid run
  (`cells-strict.csv` leaves it empty). The runs are as steady as the valid ones (spread 0.3 to 2.6 %, 2 to 9 % under their `lb`
  counterpart). 3B and 8B Q4_K_M `lc` reach the floor (their prompt windows are longer) and are valid. ExecuTorch runs and `lb`
  have a warm-up inside the process. This is a property of the timer, not a disturbance.
- **Stock `4w` on 3B and 8B** runs at a median of 2523 to 2662 MHz (all eight runs under the floor, at the same speed: spread 0.4 to
  2.4 %), the workload-limited clock of the 780M and the RX 7600 at 8B; 1B reaches the floor in five runs of seven.
- `cells.csv` counts the `clock_low`-only runs (`n_accepted`, `clk_med_mhz` give how many and the cell's median clock), as the
  780M and the 2026-10-08 RX 7600 result do; `cells-strict.csv` leaves the 10 cells with fewer than five valid runs as they are (the
  strict value is empty where no run is valid). In a cell where every run is counted, the median is over eight runs. The headline
  ratios use `lb`, which has no such cells, for llama.cpp (the higher timer is `lb` in every cell), so they do not depend on this.

## Text check

`checks.csv`: every arm continues the real-text prompt fluently. The ExecuTorch arms agree with each other on 1B and 3B; on 8B the
two `8da4w` SARC arms (`sarc`, `tuned`) give a different later word from the `4w` arms and stock, as on the RX 7600. llama.cpp
gives a different, equally plausible continuation; its tokenizer splits the check prompt differently (1980 against 1972 tokens).

## Audit notes

- Thermal: `row.py` applies its thermal test only to runs that passed the clock test, so for the 82 counted runs it was not applied.
  It was applied afterwards to the adapter's samples of every timed run (ExecuTorch prefill windows, `lc` whole processes, `lb` whole
  processes): no sample carries a temperature throttle bit other than the masked bit 36. Bit 36 is set in about 30 % of the samples
  of a tuned `4w` run (449 of 1472 in the r1 files) and rejects nothing.
- The adapter read the card's `gpu_busy_percent` (ten samples of 50 ms) before each of the 237 runs: 0 % in 236, 5 % (the bound) in one; no
  wait was needed. No host build, no process holding the card, no foreign GPU user in any run (`<run>.others`,
  `<run>.builds`, the one-second watcher of the session).
- Against the campaign's final session (same binaries, environment and `.pte`): tuned `4w` 22261 / 10779 / 4900 against 23011 / 10667 /
  4900, SARC `4w` 19884 / 10240 / 4763 against 20480 / 10139 / 4763, tuned `8da4w` 24675 / 11636 / 5319 against 24381 / 11770 / 5375,
  SARC `8da4w` 22506 / 10557 / 4983 against 22022 / 10503 / 5020: within 1.2 % on 3B and 8B; 1B differs by up to 3.3 % (one or two
  milliseconds of a 90 to 100 ms prefill).
- Medians of `cells.csv` and `cells-strict.csv` were recomputed from `runs.csv` independently of `aggregate.py` (`recompute.txt`): no difference.

## Deviations from the kit

- **New adapter** `kit/hosts/7900xtx/host.sh` (one file, as the kit asks). It writes row.py's 6-column format so that `row.py` judges the
  campaign's thermal rule; it waits for builds and a busy card instead of aborting; `session.sh`, `row.py` and `aggregate.py` are
  unchanged. The session script runs on the GPU host (it only runs staged binaries; everything was built on the workstation).
- **llama.cpp built without `-DGGML_NATIVE=ON`:** the native build of the workstation (AVX-512) stops with an illegal instruction on the
  GPU host's CPU. `b11430` was built again with `-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON`; the Vulkan
  backend is the same. The first attempts of the dry run (the illegal instruction, and a bug of the adapter's first version, a missing
  `shift`, both before any timed run) are not part of the results.
- **`.pte` files:** the campaign's exports at context 3072, not the 2560-context shared exports of `MODELS.md`; llama.cpp runs with
  `-c 2560` as fixed.
- **GGUF files regenerated** from the HuggingFace revisions of `MODELS.md` exactly as for the RX 7600 (see its history README): Q4_0
  with `b10229` and the August options, Q4_K_M with `kit/make-q4km.sh`; 256 bytes smaller than `MODELS.md`, same parameter counts.
- **Stock arm** built natively from an export of `985c1ceccc` plus the backport (podman does not run on the build workstation, so
  `build-stock.sh` was not used); the same binary as the RX 7600 session. It ran on AMDVLK for the first time here.
- **Counting rule** for `clock_low`: above.
- **HIP not done:** not part of this task.
