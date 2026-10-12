# Radeon RX 7900 XTX: llama.cpp HIP (ROCm 7.2.1) in the kit's protocol (request 10.2)

Session `hip1`, 2026-10-12 18:57 to 19:54 UTC, on `<gpu-host>` (the card is the host's display adapter, idle), 16 arms (`arms.tsv`, `ARMS.md`,
committed before the session) on 1B, 3B and 8B. Same card, host, binaries, `.pte` files, validity rule and counting rule as the
2026-10-10 session in `../README.md` (read that first). **Drivers: ExecuTorch and the Vulkan reference arms on AMDVLK 2025.Q2.1;
llama.cpp HIP arms on ROCm 7.2.1** (kernel driver `amdgpu` 6.16.13, `gfx1100`, device `ROCm0`). Files: `runs.csv` (every run: 98 timed runs
valid, 112 not, 12 discarded `lc` repetitions 0, 30 text checks; every invalid run has `clock_low` as its only reason), `cells.csv`
(counts `clock_low`-only runs, as `../cells.csv`), `cells-strict.csv` (the rule as it stands), `checks.csv`, `recompute.txt` (the medians of
both cells files recomputed from `runs.csv` independently of `aggregate.py`: 48 cells, no difference), `screen.csv` and `screen-runs.csv`
(settings screen, in `ARMS.md`), `gpu-check.txt`, `baseline.csv` with `baseline-before-runs.csv` and `baseline-after-runs.csv`.

## The Vulkan baseline check did not pass on one of its two cells: read this first

The request requires the Vulkan baseline (1B tuned `4w`, `lb` Q4_0 `best`, both AMDVLK) measured before and after the HIP work and
agreeing within 2 %, and says to stop and record when it moves more. Measured with the same adapter, five repetitions each, 18:38 UTC
(before) and 19:56 UTC (after the timed session):

| cell | 2026-10-10 session | before | timed session | after | after / before |
|---|---:|---:|---:|---:|---:|
| 1B tuned `4w` (tok/s) | 22261 | 23011 | 22261 | 22506 | **-2.20 %** |
| 1B Vulkan `lb` Q4_0 `best` (tok/s) | 16409 | 16294 | 16254 | 16025 | -1.65 % |

The `lb` cell agrees (1.65 %); the tuned `4w` cell moves 2.20 %, 0.2 points over the bound. Both moved the same way. What can be said:
the ExecuTorch prefill timer has a resolution of 1 ms, 89 ms against 91 ms is 2.2 % by itself, and the five values of the before
set spread over 9 % (22022 to 24094) while the after set spread over 2.2 %; the four measurements of each cell lie within 3.4 % and 2.4 %
(the 2026-10-10 session, before, timed, after), and the timed session's own value sits between before and after for the `lb` cell. That is
an argument that the environment did not change, not a proof, and a small common drift of about 2 % cannot be excluded. By the rule as written the check is
**not met**, the actor did not repeat the measurement to obtain a passing value, and **the rows of this session are not appended to
`results/cells.csv`** until the owner decides (to append them: the lines of `cells.csv` here without the header, with the source column
`7900xtx/hip/cells.csv` as the other rows have it). Nothing else was disturbed: no reboot, no kernel module, no package or driver change on the host
(see `ARMS.md` for what was done), no foreign GPU user or host build in any run (`others` empty in all 252 rows).

## Result (tok/s, median; `*` = counted `clock_low`-only runs, as in `../README.md`)

| model | stock 4w | stock 8da4w | SARC 4w | SARC 8da4w | tuned 4w | tuned 8da4w | Vulkan Q4_0 (`lb` best) | Vulkan Q4_K_M (`lb` best) | HIP Q4_0 (`lb` best) | HIP Q4_K_M (`lb` best) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1B | 6759 | 10292 | 20078 | 22022 | 22261 | 24381 | 16254 | 13157 | 23855 | 22217 |
| 3B | 2609* | 4112 | 10039 | 10557 | 10722 | 11378 | 6845 | 5554 | 9031 | 8606 |
| 8B | 1260* | 2236 | 4774 | 4971 | 4911 | 5319 | 3243 | 2582 | 4111 | 3892 |

HIP, all eight arms (tok/s):

| model | `lc` Q4_0 default | `lc` Q4_0 best | `lb` Q4_0 default | `lb` Q4_0 best | `lc` Q4_K_M default | `lc` Q4_K_M best | `lb` Q4_K_M default | `lb` Q4_K_M best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1B | 17677* | 18566* | 22875 | 23855 | 11073* | 11288* | 21775 | 22217 |
| 3B | 7550* | 8088* | 8421 | 9031 | 5905* | 6198* | 8110 | 8606 |
| 8B | 3765* | 3986* | 3888 | 4111 | 3217* | 3378* | 3688 | 3892 |

Ratios tuned `4w` / HIP, by timer and setting (above 1.0: tuned ahead; below: HIP ahead):

| model | Q4_0 `lc` default | Q4_0 `lc` best | Q4_0 `lb` default | Q4_0 `lb` best | Q4_K_M `lc` default | Q4_K_M `lc` best | Q4_K_M `lb` default | Q4_K_M `lb` best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1B | 1.26 | 1.20 | **0.97** | **0.93** | 2.01 | 1.97 | 1.02 | 1.00 |
| 3B | 1.42 | 1.33 | 1.27 | 1.19 | 1.82 | 1.73 | 1.32 | 1.25 |
| 8B | 1.30 | 1.23 | 1.26 | 1.19 | 1.53 | 1.45 | 1.33 | 1.26 |

HIP over Vulkan, same quantization, `lb` at `best`, same session: Q4_0 1.47 / 1.32 / 1.27, Q4_K_M 1.69 / 1.55 / 1.51 (1B / 3B / 8B).

Same-session ratios over stock and SARC (tuned `4w` / stock `4w` 3.29 / 4.11 / 3.90; tuned `8da4w` / stock `8da4w` 2.37 / 2.77 / 2.38; tuned / SARC
`4w` 1.11 / 1.07 / 1.03, `8da4w` 1.11 / 1.08 / 1.07) repeat the 2026-10-10 session within 0.07 (3.28 / 4.12 / 3.89, 2.41 / 2.84 / 2.37, 1.12 / 1.05 / 1.03,
1.10 / 1.10 / 1.07). The Vulkan reference arms repeat the 2026-10-10 `lb` `best` cells within 0.9 %, except the 1B Q4_K_M cell (2.2 % higher) (16254 / 6845 / 3243 against 16409 / 6907 / 3241 for
Q4_0, 13157 / 5554 / 2582 against 12877 / 5529 / 2581 for Q4_K_M).

## What it says

- **The HIP backend is faster than llama.cpp's Vulkan backend on this card at every size**, by 27 to 47 % (Q4_0) and 51 to 69 % (Q4_K_M) with `lb` at `best`.
  Q4_K_M costs HIP 5 to 7 % against Q4_0 by `lb` (Vulkan: 19 to 20 %).
- **Tuned ExecuTorch `4w` is no longer ahead of every llama.cpp arm.** At 1B, HIP Q4_0 by `lb` is ahead of tuned `4w` (23855 against 22261; tuned / HIP 0.93 at `best`,
  0.97 at default) and HIP Q4_K_M by `lb` is level with it (1.00 at `best`, 1.02 at default, inside the 2 % to 3 % between-session movement of the tuned 1B cell).
  At 3B and 8B tuned `4w` is ahead of HIP by 19 to 27 % (Q4_0) and 25 to 33 % (Q4_K_M) by `lb`. By `lc`, the fresh-process timer, tuned `4w` is ahead in every cell (1.20 to 1.42 Q4_0,
  1.45 to 2.01 Q4_K_M). Tuned `8da4w` is ahead of tuned `4w` by 6 to 10 % and has no llama.cpp counterpart.
- **`best` is 2.0 to 7.2 % above `default`** for HIP by `lb` (Q4_0 4.3 / 7.2 / 5.7 %, Q4_K_M 2.0 / 6.1 / 5.5 %), as the screen said.
- **The two timers differ more on HIP than on Vulkan.** `lc` is 0.77 to 0.97 of `lb` for Q4_0 and 0.51 to 0.87 for Q4_K_M (smaller models lose more: 1B Q4_K_M `lc` is half of `lb`).
  `lc` is the fresh process of the kit and pays the card's clock ramp (median clock inside the 1B windows 970 to 1340 MHz against the 2670 MHz floor, so
  all 96 `lc` runs are `clock_low`, counted and shown) and whatever a process pays on its first evaluation; that the first evaluation of the HIP backend (loading of its
  kernels and libraries) is a large part of the 1B Q4_K_M gap is a reading, UNVERIFIED: no run separated the two causes. Both are reported; the higher of the two is `lb`.
  The first repetition inside a 1B `lb` process is up to 20 % slower than the other four (the same ramp); the median over five absorbs it, the spread column of `cells.csv` shows it (16 to 20 % for the HIP 1B cells).

## Text check

`checks.csv`: every HIP arm continues the real-text prompt fluently, and differently from the ExecuTorch arms (a different tokenizer split, 1980 tokens against 1972), as the Vulkan
arms do. The 1B Q4_0 `default` and `best` HIP runs continue differently from each other (flash attention and the larger micro-batch change the floating-point accumulation order);
both are plausible text. GPU use: `gpu-check.txt` (all 17 layers offloaded to `ROCm0`, the card at 100 % busy, VRAM growing by 1.4 GiB, a CPU fallback would show none of this) and the JSON of every HIP
`llama-bench` log (`backends: ROCm`, 12 of 12); the HIP `lc` windows are GPU windows (18000 tok/s at 1B Q4_0 is not a CPU rate).

## Deviations from the kit and from the request

- **Build in a container, host untouched.** The host has the ROCm 7.2.1 runtime packages but no compiler driver, no HIP headers and no BLAS libraries, and its distribution release
  (25.04) is not the one ROCm is built for (24.04). Following the owner's fallback: build in the public image `rocm/dev-ubuntu-24.04:7.2.1` (digest in `ARMS.md`), run on the host with
  copies of the runtime libraries next to the binaries. The container installed `cmake`, `ninja-build`, `hipblas-dev` and `rocblas-dev` inside itself, through the host's CA
  bundle mounted read-only (the company web gateway's CA is not in the image; verification stayed on). No container ran during the session; no `--device` flag was needed. Three attempts failed
  before the build finished (the package repository's certificate was not trusted by the container; a package of the image could not be replaced under the bundle mounted over its path; and my own mistake of excluding `build*` from the source copy, which dropped `cmake/build-info.cmake`); none of them reached a timed run. The failed attempts left root-owned files in my own scratch directory.
- **`-DGGML_NATIVE=OFF` with AVX2** for the CPU part, as the Vulkan build of this host.
- **Arm names** `hip-{lc,lb}-{q4_0,q4_k_m}-{default,best}` instead of the request's `hip-q4_0`, `hip-q4km`, and `kind` stays the timer (`lc` / `lb`) because `aggregate.py` is unchanged; the HIP rows are those whose
  arm starts with `hip-` (the precedent is the `sy-` rows of the SYCL devices).
- **Baseline check** not met on one cell: above. **`results/cells.csv` not appended** for that reason.
- **Counting rule** for `clock_low` as in `../README.md` (still awaiting the owner's confirmation there): it applies to all 96 `lc` runs of the HIP arms (strict count 0, `cells-strict.csv`) and to stock `4w` at 3B and 8B (8 counted runs each, as on 2026-10-10).
  The headline ratios here use `lb`, which has no counted run, for HIP and Vulkan.
- **Not repeated:** the extras of the request's 10.4 (decode and the real-text prompt, `../README.md`); no HIP number exists for them.
- **Not tested:** the 2 % bound applied to the HIP cells themselves (no repeat session); the session's HIP `lb` cells have no second process.
