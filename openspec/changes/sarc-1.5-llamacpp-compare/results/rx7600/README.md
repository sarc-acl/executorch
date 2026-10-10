# Radeon RX 7600: llama.cpp comparison, round 2 final as the tuned arm

Session `rx7600-r2`, 2026-10-10 01:47 to 03:23 UTC, on the workstation's own card (idle: no process holding the card, no build seen
by the one-second watcher), 14 arms (`arms.tsv`, `ARMS.md`, committed before the session) on 1B, 3B and 8B, after the tuning campaign's
round 2 had finished. **This replaces the result of 2026-10-08** (round 1 as the tuned arm), whose files are kept in
`history-2026-10-08/` with its README. **Driver for every arm, ExecuTorch and llama.cpp alike: user-space RADV Mesa 26.2.3**
(`VK_ICD_FILENAMES`), Vulkan only. Validity: the campaign's rule as `row.py` applies it, clock floor 2420 MHz, sampled every 10 ms
(`kit/hosts/rx7600/host.sh`, unchanged since 2026-10-08); no foreign-busy ceiling (no per-client engine accounting on this card).
The session started from a cool card (core temperature 40 C, waited 441 s) under the campaign's gpu-lab lock; a first start of the same
session at 01:37 UTC was stopped by the actor after three minutes because the card had not cooled (idle reference 50 C); its files
are not used.

Files: `runs.csv` (every run, with its reason: 157 timed runs valid, 8 not, 12 discarded `lc` repetitions 0, 30 text checks), `cells.csv`
(the table's source, counts `clock_low`-only runs as the 780M and the 2026-10-08 RX 7600 result do, see below), `cells-strict.csv`
(the rule as it stands), `checks.csv`, `screen.csv`, `recompute.txt` (medians of both cells files recomputed from `runs.csv`
independently of `aggregate.py`: 42 cells, no difference), `lb-slowpath-8b.txt`.

## Result (tok/s, median; llama.cpp = the higher of its two timers at its better setting)

| model | stock 4w | stock 8da4w | SARC 4w | SARC 8da4w | tuned 4w | tuned 8da4w | llama.cpp Q4_0 | llama.cpp Q4_K_M | tuned 4w / Q4_0 | tuned 4w / Q4_K_M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1B | 2702 | 3793 | 7817 | 7314 | 11070 | 10611 | 5849 | 5016 | 1.89 | 2.21 |
| 3B | 974 | 1386 | 3287 | 3080 | 4267 | 4096 | 2451 | 2042 | 1.74 | 2.09 |
| 8B | 471* | 730 | 1519 | 1404 | 1888 | 1814 | 1157 | 941 | 1.63 | 2.01 |

llama.cpp, all eight arms (tok/s):

| model | lc Q4_0 default | lc Q4_0 best | lb Q4_0 default | lb Q4_0 best | lc Q4_K_M default | lc Q4_K_M best | lb Q4_K_M default | lb Q4_K_M best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1B | 5366 | 5338 | 5849 | 5833 | 4695 | 4678 | 5010 | 5016 |
| 3B | 2363 | 2360 | 2451 | 2448 | 2024 | 1982 | 2042 | 2042 |
| 8B | 1157 | 1157 | 96† | 96† | 941 | 940 | 94† | 94† |

Ratios tuned 4w / llama.cpp, by timer and setting:

| model | Q4_0 lc default | Q4_0 lc best | Q4_0 lb default | Q4_0 lb best | Q4_K_M lc default | Q4_K_M lc best | Q4_K_M lb default | Q4_K_M lb best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1B | 2.06 | 2.07 | 1.89 | 1.90 | 2.36 | 2.37 | 2.21 | 2.21 |
| 3B | 1.81 | 1.81 | 1.74 | 1.74 | 2.11 | 2.15 | 2.09 | 2.09 |
| 8B | 1.63 | 1.63 | n/a† | n/a† | 2.01 | 2.01 | n/a† | n/a† |

Valid runs per cell under the rule as it stands / counted (`*` = counted with `clock_low` as the only reason):

| arm | 1B | 3B | 8B |
|---|---:|---:|---:|
| `stock-4w` | 5 / 5 | 5 / 5 | 0 / 8 |
| `stock-8da4w` | 5 / 5 | 5 / 5 | 5 / 5 |
| `sarc-4w` | 5 / 5 | 5 / 5 | 5 / 5 |
| `sarc-8da4w` | 5 / 5 | 5 / 5 | 5 / 5 |
| `tuned-4w` | 5 / 5 | 5 / 5 | 5 / 5 |
| `tuned-8da4w` | 5 / 5 | 5 / 5 | 5 / 5 |
| `lc-q4_0-default` | 5 / 5 | 5 / 5 | 5 / 5 |
| `lc-q4_0-best` | 5 / 5 | 5 / 5 | 5 / 5 |
| `lc-q4_k_m-default` | 5 / 5 | 5 / 5 | 5 / 5 |
| `lc-q4_k_m-best` | 5 / 5 | 5 / 5 | 5 / 5 |
| `lb-q4_0-default` | 5 / 5 | 5 / 5 | 5 / 5 |
| `lb-q4_0-best` | 5 / 5 | 5 / 5 | 5 / 5 |
| `lb-q4_k_m-default` | 5 / 5 | 5 / 5 | 5 / 5 |
| `lb-q4_k_m-best` | 5 / 5 | 5 / 5 | 5 / 5 |
\* 8B stock `4w` counted under the clock note. † `llama-bench` at 8B, see the slow path below; the 8B llama.cpp number is the `lc` timer, and
no `lb` ratio is given. Five valid runs in every other ExecuTorch and `lc` cell; `lb` = one process of five repetitions.

Same-session ratios over the stock arm: tuned `4w` / stock `4w` 4.10 / 4.38 / 4.01 (1B / 3B / 8B), tuned `8da4w` / stock `8da4w` 2.80 / 2.96 / 2.49;
tuned / SARC `4w` 1.42 / 1.30 / 1.24, `8da4w` 1.45 / 1.33 / 1.29 (the campaign's round 2 final session: 1.41 / 1.30 / 1.24 and 1.45 / 1.33 / 1.29).

Both timers, both quantizations: tuned `4w` is ahead of llama.cpp Vulkan Q4_0 by 1.89 / 1.74 / 1.63 (1B / 3B / 8B; against `lc` at its better setting 2.07 /
1.81 / 1.63) and ahead of Q4_K_M by 2.21 / 2.09 / 2.01. Q4_K_M is 12 to 19 % slower than Q4_0 (`lc`, best setting). `lc` is 1 to 9 % below `lb` where
`lb` is not slow (1B and 3B). Against the 2026-10-08 session the tuned `4w` arm gained 5.4 / 6.9 / 7.8 % (round 2 against round 1), the SARC and stock arms and
the `default` llama.cpp arms did not move (0.4 % for the ExecuTorch arms, 1.6 % at most for llama.cpp `lc` default, `lb` 0.1 %).

## The `llama-bench` slow path at 8B (named and shown)

At 8B every `llama-bench` arm reads 94 to 96 tok/s (Q4_0 and Q4_K_M, default and `best`) against 941 to 1157 by `llama-completion` on the same files and flags.
Diagnostic after the session (`lb-slowpath-8b.txt`, one repetition each, Q4_0): `-p 512` / `1024` / `1536` read 1252 / 1228 / 1197 tok/s with flash attention on
(1178 / 1111 / 1093 off), `-p 2048` reads **96** (on) and **517** (off), while `-pg 2048,1` (the same prompt followed by one generated token) reads 1139. So it is
the prompt-only test at 2048 tokens that hits a slow GPU path on this card at 8B; the GPU is busy at full power during it. The cause is UNVERIFIED (on
2026-10-08 the slow readings started at 1536 tokens: the path moves with the compute-buffer sizes, not with a fixed length). 1B and 3B `lb` agree with `lc` within 1 to 9 %
(`lb` higher). The ratios of `lc` at 8B stand; the `lb` column of the table for 8B is not a measure of llama.cpp.

## Clock note: stock `4w` at 8B

All eight runs of stock `4w` on 8B ran at a median of 2323 to 2329 MHz, under the 2420 MHz floor, at the same speed (spread 0.9 %), so `row.py` rejects all of them
(`clock_low`) and the session ends `SESSION_INCOMPLETE` for this one cell. It is the longest prefill of the session (4.3 s) and the same pattern as on 2026-10-08
and on the 780M: the clock is a property of that workload. `cells.csv` counts those runs (`n_accepted` = 8, `clk_med_mhz` = 2328, the median over eight
runs), `cells-strict.csv` leaves the cell empty. Every other arm reached the floor in every run.

## Text check

`checks.csv`: every arm continues the real-text prompt fluently. The ExecuTorch arms agree with each other on 1B and 3B; on 8B the two `8da4w` SARC arms (`sarc`, `tuned`)
give a different later word from the `4w` arms and stock, as on 2026-10-08 and on the 7900 XTX. llama.cpp gives a different, equally plausible continuation (its tokenizer
splits the check prompt into 1980 tokens, ExecuTorch's into 1972).

## Audit notes

- Thermal: the adapter keeps the campaign's `indep_throttle_status` beside each run (`<run>.clk.full`, not committed); `row.py` does not judge it on this card. Temperature bits
  (32 to 47) appear in the prefill windows of ExecuTorch runs only for stock `4w`: two samples of 58 and 60 in two valid 1B runs (junction 72 to 78 C) and 1 to 7 samples of
  about 340 in five of the eight counted 8B runs (up to 84 C); over the whole `lc` and `lb` processes bits appear in every run (72 runs), a few samples of 250 to 420 each, around model load and the end.
  Not judged, as on 2026-10-08. The thermal bit of the 8B stock cell does not change its median measurably (the runs are within 0.9 %).
- Against the campaign's final session (same binaries, environment and `.pte`): tuned `4w` 11070 / 4267 / 1888 against 11070 / 4267 / 1886, SARC `4w` 7817 / 3287 / 1519 against
  7847 / 3293 / 1517, tuned `8da4w` 10611 / 4096 / 1814 against 10667 / 4104 / 1814, SARC `8da4w` 7314 / 3080 / 1404 against 7341 / 3084 / 1403: within 0.5 %.
- Medians of `cells.csv` and `cells-strict.csv` were recomputed from `runs.csv` independently of `aggregate.py` (`recompute.txt`): no difference.
- Other GPU users: none in any run; desktop idle. Builds and other jobs on the workstation: none seen by the watcher during the session (the Android build of the M51 work,
  01:40 to 01:43 UTC, ended before the session started; the M51 session and file copies ran as processes beside it, no compilers).

## Deviations from the kit

- **Settings screen and `best`:** `best` is `-b 2048 -ub 512 -fa on` (not the 2026-10-08 `-ub 1024`), from the new screen of `ARMS.md`: no setting beats the default beyond noise
  on this card, so `best` is the highest-ranked non-default setting.
- **`.pte` files:** the campaign's exports at context 3072, not the 2560-context shared exports of `MODELS.md`; llama.cpp runs with `-c 2560` as fixed.
- **GGUF files regenerated** from the HuggingFace revisions of `MODELS.md` (see `history-2026-10-08/README.md`): Q4_0 with `b10229` and the August options, Q4_K_M with
  `kit/make-q4km.sh`; every file 256 bytes smaller than `MODELS.md`, same parameter counts. The same files as on 2026-10-08.
- **Stock arm** built natively from an export of `985c1ceccc` plus the backport (podman does not run on this host, so `build-stock.sh` was not used); the same binary as on 2026-10-08.
- **Counting rule** for `clock_low`: above.
- **HIP not done:** not part of this task.
