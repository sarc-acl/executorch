# Radeon RX 7600: llama.cpp HIP (ROCm 7.2.4) in the kit's protocol (request 10.2), one session with every arm

Session `rx7600-hip2`, 2026-10-12 22:46 to 2026-10-13 00:56 UTC, on the workstation's own card (idle: no process holding the card, no host build seen by the one-second
watcher or in any run's `before` / `after` record), **22 arms** (`arms.tsv`, `ARMS.md`, committed before the session) on 1B, 3B and 8B: stock, SARC and tuned ExecuTorch (`4w`, `8da4w`),
llama.cpp Vulkan (`lc` / `lb`, Q4_0 / Q4_K_M, `default` / `best`) and llama.cpp HIP (the same eight arms with the `hip-` prefix). This is the session the owner decision of
2026-10-12 21:44 UTC asks for: every ratio below comes from this one session. **Drivers: ExecuTorch and the Vulkan llama.cpp arms on user-space RADV Mesa 26.2.3
(`VK_ICD_FILENAMES`); the HIP arms on the ROCm 7.2.4 user-space runtime** (`-dev ROCm0`, card `gfx1102`, native target, **no `HSA_OVERRIDE_GFX_VERSION`, no `gfx1100` build**). Validity: the campaign's rule through
`kit/hosts/rx7600/host.sh`, unchanged (clock floor 2420 MHz, at least five clock samples, 2048 prompt tokens, no foreign GPU user). The session started from a cool card (core temperature 40 C, waited 80 s).
Status `SESSION_INCOMPLETE` (`done.txt` of the session): four cells have fewer than five valid runs, all `clock_low`-only (below).

Files here: `runs.csv` (every run: 214 timed runs valid, 32 not, 24 discarded `lc` repetitions 0, 42 text checks; every invalid run has `clock_low` as its only reason), `cells.csv` (counts the `clock_low`-only
runs as `../cells.csv` and `../../7900xtx/hip/cells.csv` do; 66 cells), `cells-strict.csv` (the rule as it stands), `checks.csv`, `recompute.txt` (the medians of both cells files recomputed from `runs.csv`
independently of `aggregate.py`: 66 cells each, no difference), `screen.csv` and `screen-runs.csv` (settings screen, in `ARMS.md`), `gpu-check.txt`, `baseline-before-runs.csv` (the stopped round's baseline cell, below).
All 66 cells are appended to `../../cells.csv` under session `rx7600-hip2` (source column `rx7600/hip/cells.csv`); the rows of the 2026-10-10 session `rx7600-r2` are kept. **`kind` = `lc` / `lb` with the `hip-` arm prefix is accepted** by the owner
(`aggregate.py` unchanged); the HIP rows are those whose arm starts with `hip-`.

## The Vulkan baseline cell of 2026-10-12 21:28 UTC (+9 %) and what the session shows

Before this session a Vulkan baseline cell (1B tuned `4w`, 1B `lb` Q4_0 `best`, RADV; five repetitions, `baseline-before-runs.csv`) read **+9.5 %** (12118 against 11070 tok/s) and **+8.8 %**
(6346 against 5833) against the 2026-10-10 session `rx7600-r2`, with the card's median clock unchanged (2520 against 2516 MHz; 2444 against 2446). By the stop condition (within 2 %, or stop and record) that round stopped; the owner then
decided (21:44 UTC) to compare nothing across days and to measure every arm in one session. Facts, from `last -x` and the session:

- The workstation was shut down at 2026-10-12 21:05:48 UTC and booted at **21:12:52 UTC** (`last -x`; the previous boot was 2026-09-18 21:28:39 UTC, so the 2026-10-10 session ran in that previous boot).
  The baseline cell was measured 15 minutes after this boot (21:28 UTC). Kernel (`6.8.0-107`), Mesa (26.2.3, `31e9a6b2e9`), the ICD file and the binaries did not change between the two sessions; nothing was installed by this task.
- **The +9 % is not in this session.** The tuned `4w` anchor of the HIP screen (22:20 UTC, 67 minutes after the boot) read 11070 / 4267 tok/s at 1B / 3B, the 2026-10-10 values, and in the timed session (from 22:46 UTC) the ExecuTorch arms and
  the Vulkan `lb` arms agree with the 2026-10-10 session within 0.7 % in every cell (below), the Vulkan `lc` arms within 2.8 % (1B and 3B: +0.8 to +2.8 %, one cell -1.8 %; 8B: -0.3 to -0.5 %).
- Reading, UNVERIFIED (no run separated the causes): the +9 % was present in the first minutes after the restart and gone about an hour later; what differed (for example the card's or the system's state shortly after boot, the clock and power
  management, background activity) was not examined. The 21:28 UTC measurement is therefore not a reference of the environment of the session and is not used in any ratio.
- **The 2026-10-10 session's ratios stand** (its arms shared one session), **and so do its absolute numbers**: this session reproduces them (table below), so the "may be about 9 % low" caution of the owner decision does not apply to the figures
  of `../README.md`.

Same arms, 2026-10-10 session `rx7600-r2` against this session (tok/s, median; counted cells included):

| model | stock 4w | stock 8da4w | SARC 4w | SARC 8da4w | tuned 4w | tuned 8da4w |
|---|---:|---:|---:|---:|---:|---:|
| 1B | 2702 / 2702 | 3793 / 3793 | 7817 / 7817 | 7314 / 7314 | 11070 / 11070 | 10611 / 10611 |
| 3B | 974 / 971* | 1386 / 1382 | 3287 / 3287 | 3080 / 3075 | 4267 / 4258 | 4096 / 4096 |
| 8B | 471* / 468* | 730 / 731 | 1519 / 1518 | 1404 / 1402 | 1888 / 1886 | 1814 / 1812 |

Vulkan `lb` `best` Q4_0 / Q4_K_M: 1B 5833 / 5016 against 5805 / 4991 (-0.5 %), 3B 2448 / 2042 against 2444 / 2036 (-0.2 / -0.3 %), 8B (slow path, below) 96 / 94 against 96 / 94.
Vulkan `lc` `best` Q4_0 / Q4_K_M: 1B 5338 / 4678 against 5477 / 4733 (+2.6 / +1.2 %), 3B 2360 / 1982 against 2402 / 2021 (+1.8 / +1.9 %), 8B 1157 / 940 against 1152 / 936 (-0.4 / -0.5 %).

## Result (tok/s, median; `*` = counted `clock_low`-only runs)

| model | stock 4w | stock 8da4w | SARC 4w | SARC 8da4w | tuned 4w | tuned 8da4w | Vulkan Q4_0 (`lb` best) | Vulkan Q4_K_M (`lb` best) | HIP Q4_0 (`lb` best) | HIP Q4_K_M (`lb` best) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1B | 2702 | 3793 | 7817 | 7314 | 11070 | 10611 | 5805 | 4991 | 6328 | 5931 |
| 3B | 971* | 1382 | 3287 | 3075 | 4258 | 4096 | 2444 | 2036 | 2540 | 2418 |
| 8B | 468* | 731 | 1518 | 1402 | 1886 | 1812 | 96† | 94† | 1216 | 1184 |

† Vulkan `llama-bench` at 8B takes the slow path named in `../README.md` (prompt-only test at 2048 tokens; the 8B Vulkan llama.cpp number is its `lc`, 1152 / 936 at `best`); the 96 / 94 are not a measure of llama.cpp. HIP `llama-bench` does not take it:
its 8B `lb` and `lc` agree within 0.4 % (Q4_0) and 4 % (Q4_K_M).

HIP, all eight arms (tok/s):

| model | `lc` Q4_0 default | `lc` Q4_0 best | `lb` Q4_0 default | `lb` Q4_0 best | `lc` Q4_K_M default | `lc` Q4_K_M best | `lb` Q4_K_M default | `lb` Q4_K_M best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1B | 6130 | 6123 | 6331 | 6328 | 4952 | 4954 | 5934 | 5931 |
| 3B | 2511 | 2510 | 2541 | 2540 | 2173 | 2238 | 2338 | 2418 |
| 8B | 1212* | 1212* | 1217 | 1216 | 1106 | 1143 | 1145 | 1184 |

Ratios tuned `4w` / HIP, by timer and setting (above 1.0: tuned ahead; below: HIP ahead):

| model | Q4_0 `lc` default | Q4_0 `lc` best | Q4_0 `lb` default | Q4_0 `lb` best | Q4_K_M `lc` default | Q4_K_M `lc` best | Q4_K_M `lb` default | Q4_K_M `lb` best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1B | 1.81 | 1.81 | 1.75 | 1.75 | 2.24 | 2.23 | 1.87 | 1.87 |
| 3B | 1.70 | 1.70 | 1.68 | 1.68 | 1.96 | 1.90 | 1.82 | 1.76 |
| 8B | 1.56* | 1.56* | 1.55 | 1.55 | 1.71 | 1.65 | 1.65 | 1.59 |

HIP over Vulkan, same quantization, same session. `lb` at `best` for 1B and 3B: Q4_0 1.09 / 1.04, Q4_K_M 1.19 / 1.19. At 8B the Vulkan `lb` is the slow path, so the comparison uses the Vulkan `lc` at `best`
(HIP `lb` / Vulkan `lc` 1.06 Q4_0, 1.27 Q4_K_M; HIP `lc` / Vulkan `lc` 1.05 Q4_0, 1.22 Q4_K_M). By `lc` at `best`, all sizes: Q4_0 1.12 / 1.05 / 1.05, Q4_K_M 1.05 / 1.11 / 1.22 (1B / 3B / 8B).

HIP `lc` / `lb` at the same setting, the cost of a fresh process: Q4_0 0.97 / 0.99 / 1.00 (`default`; `best` the same), Q4_K_M `default` 0.83 / 0.93 / 0.97, `best` 0.84 / 0.93 / 0.97 (1B / 3B / 8B).

Same-session ratios over stock and SARC: tuned `4w` / stock `4w` 4.10 / 4.39* / 4.03* (the 3B and 8B stock `4w` values rest on counted `clock_low` runs only), tuned `8da4w` / stock `8da4w` 2.80 / 2.96 / 2.48; tuned / SARC `4w` 1.42 / 1.30 / 1.24, `8da4w` 1.45 / 1.33 / 1.29
(2026-10-10 session: 4.10 / 4.38 / 4.01, 2.80 / 2.96 / 2.49, 1.42 / 1.30 / 1.24, 1.45 / 1.33 / 1.29).

Valid runs per cell under the rule as it stands / counted (`cells-strict.csv` / `cells.csv`; every other cell of the 22 arms has 5 / 5):

| arm | 1B | 3B | 8B |
|---|---:|---:|---:|
| `stock-4w` | 5 / 5 | 0 / 8 | 0 / 8 |
| `hip-lc-q4_0-default` | 5 / 5 | 5 / 5 | 0 / 8 |
| `hip-lc-q4_0-best` | 5 / 5 | 5 / 5 | 0 / 8 |

(`lb` cells: one process, five repetitions, the repetitions are the sample.) The counted `clock_low` runs: stock `4w` at 3B median clock 2391 to 2417 MHz (2428 in the 2026-10-10 session, where the cell was valid), at 8B 2305 to 2311 MHz
(2323 to 2329 MHz on 2026-10-10); HIP `lc` Q4_0 at 8B 2388 to 2395 MHz, 1.0 to 1.3 % under the 2420 MHz floor, spread of the eight runs 0.8 / 1.7 %. The cause of the low clock was not examined (earlier notes of this card call it a property of the workload); the runs are as reproducible as the valid ones, and `row.py` correctly rejects them; counted as the 780M and the 2026-10-08 and 2026-10-10 results of this card did, strict values absent. The ratios that rest on them are marked `*`.

## What it says

- **Tuned ExecuTorch `4w` is ahead of llama.cpp HIP in every cell** on this card: 1.55 to 1.87 against HIP Q4_0 / Q4_K_M by `lb` at `best` (Q4_0 1.75 / 1.68 / 1.55, Q4_K_M 1.87 / 1.76 / 1.59) and 1.56 to 2.24 by `lc`.
  This is unlike the RX 7900 XTX (`../../7900xtx/hip/README.md`), where HIP is level with or ahead of tuned at 1B.
- **HIP is faster than llama.cpp's Vulkan backend here, but by much less than on the 7900 XTX:** 4 to 9 % for Q4_0 and 19 % for Q4_K_M by `lb` at 1B and 3B; by `lc` 5 to 12 % (Q4_0) and 5 to 22 % (Q4_K_M, 1.22 at 8B). The Vulkan Q4_K_M arm loses more to Q4_0 (14 to 19 % by `lc`)
  than HIP does (HIP `lb`: -6.3 / -4.8 / -2.7 % at 1B / 3B / 8B; HIP `lc`: -19 / -11 / -6 %).
- **`best` against `default`:** HIP Q4_0 `best` (`-ub 512 -fa on`) is the default (0.0 to 0.1 % apart, as the screen said); HIP Q4_K_M `best` (`-ub 1024 -fa on`) is 0.0 / 3.4 / 3.4 % above `default` by `lb` at 1B / 3B / 8B (`lc` +0.0 / +3.0 / +3.4 %).
- **The two timers differ little for Q4_0** (HIP `lc` 0.97 to 1.00 of `lb`) and more for Q4_K_M at 1B (0.83); the first evaluation of the HIP backend (loading of its kernels) being part of that gap is a reading, UNVERIFIED.
  Both timers are reported for every cell; the higher is `lb`.
- **Q4_0 and Q4_K_M are separate rows**, as are default and best. Tuned `4w` / HIP Q4_K_M by `lb` at `best` 1.87 / 1.76 / 1.59 is the figure beside the Q4_0 1.75 / 1.68 / 1.55.

## GPU use and text check

GPU use is checked, not assumed (`gpu-check.txt`; 1B Q4_0, verbose `llama-completion`): `backends: ROCm`, device `ROCm0` (`gfx1102`), `offloaded 17/17 layers to GPU`, the card at 100 % busy, VRAM growing from 33 to 5393 MiB; the
JSON of every HIP `llama-bench` log says `backends: ROCm`. The HIP `lc` windows are GPU windows (6100 tok/s at 1B is not a CPU rate; the median clock in the `best` Q4_0 windows is 2528 / 2424 / 2392 MHz at 1B / 3B / 8B).
`checks.csv`: every arm continues the real-text prompt fluently. Each HIP arm continues exactly like the Vulkan arm of the same quantization on the same model, in all 12 pairs (`default` and `best` alike: the same first 60 characters),
differently from the ExecuTorch arms (a different tokenizer split, 1980 tokens against 1972), which agree with each other on 1B and 3B; on 8B the two `8da4w` SARC arms continue with a different later word, as on 2026-10-10.

## Audit notes

- Thermal: temperature bits (as in `../README.md`) are not judged by `row.py` on this card; the highest temperature of any timed run is 85 C (8B; the Vulkan `lb` slow-path runs and stock `4w`). Other GPU users: none in any of the 312 runs (`others` empty); host builds: none (`before` / `after` records empty, buildwatch empty).
- Medians of `cells.csv` and `cells-strict.csv` were recomputed from `runs.csv` independently of `aggregate.py` (`recompute.txt`): 66 cells each, no difference.
- ExecuTorch binaries, `.pte`, environments, the Vulkan llama.cpp build, the GGUF files and the ICD: those of `../ARMS.md`, not rebuilt. The HIP binaries: those of the stopped round (sha256 in `ARMS.md`), built before any measurement.

## Deviations from the kit and from the request

- **Native `gfx1102`:** the request allows a `gfx1100` build with `HSA_OVERRIDE_GFX_VERSION=11.0.0` as the fallback; it was not needed and not used.
- **Arm names** `hip-{lc,lb}-{q4_0,q4_k_m}-{default,best}` instead of `hip-q4_0`, `hip-q4km`; `kind` stays the timer (`lc` / `lb`), as on the 7900 XTX.
- **`best` per quantization** (screen, `ARMS.md`): `-ub 512 -fa on` for Q4_0, `-ub 1024 -fa on` for Q4_K_M, because the rule of the screen is per quantization and the two differ here (on the 7900 XTX one string served both).
- **Vulkan baseline cell of the request (before / after, within 2 %):** not met on 2026-10-12 21:28 UTC (+9.5 / +8.8 %, above); replaced by the owner decision of 21:44 UTC with one session that measures every arm; the session reproduces the 2026-10-10 values (above).
  The ROCm 7.2.4 installation predates the 2026-10-10 session (installed 2026-10-09, not by this task), so there is no "before the install" cell.
- **Build and run on the workstation**, no container; the build ran in a shell with group `render`, as the owner decisions of 2026-10-12 describe; no package, group, udev rule or device permission was changed.
- **Counting rule** for `clock_low` as in `../README.md` (the owner's confirmation of it is still outstanding): it applies to stock `4w` at 3B and 8B and to HIP `lc` Q4_0 at 8B (strict count 0 for these; `cells-strict.csv`).
  The headline ratios use `lb`, which has no counted run, except where `*` marks a counted cell.
- **Not repeated:** the extras of request 10.4 (decode and the real-text prompt) for the HIP arms.
- **Not tested:** the 2 % bound applied to the HIP cells themselves (the HIP `lb` cells have one process each).
