# M51: llama.cpp Vulkan and stock ExecuTorch against tuned ExecuTorch (relative speedups only)

Measured 2026-10-10 on one M51 board with the kit's protocol (the M51 campaign's board rules: canonical driver checked before and after every run, clocks
pinned and read back, profiler configuration aside, sampler on the board, cool start and cooling between runs, arms interleaved in one session per model
group, five valid runs per ExecuTorch and `llama-completion` cell, `llama-bench` one process of five repetitions, one text check per arm). **Only relative
figures are committed** (owner rule of 2026-10-08, confirmed 2026-10-10): ratios of one arm to another, with their interval. The raw `runs.csv` and `cells.csv`
(absolute rates and times) stay in a local directory; this file has no rate, no time and no identifier of the board, its host or its driver.

## The statement

"Ahead of" means outside the campaign's 2 % noise band, for both llama.cpp timers (`lc`, `lb`) and for every llama.cpp setting that produces correct output;
the observed gaps are far outside the band.

| model | tuned ExecuTorch `4w` against llama.cpp Vulkan Q4_0 | against llama.cpp Vulkan Q4_K_M |
|---|---|---|
| 1B | ahead of | ahead of |
| 3B | ahead of | ahead of |
| 8B | ahead of | ahead of |

## The ratios (`ratios.csv`)

Each cell is the ratio of medians, with the range between brackets (the lowest run of the numerator over the highest run of the denominator, and the highest
over the lowest; five runs each, or the five repetitions of one `llama-bench` process). llama.cpp = the correct-output configuration below at its best setting.
Sessions: the fourth (1B and 3B) and the fifth (8B), each with stock, SARC (the campaign's parent) and tuned (the campaign's final configuration, profile `c8`),
`4w` and `8da4w`, and the llama.cpp arms interleaved; every one of their 126 timed runs was valid under the rules above.

| model | tuned `4w` / Q4_0 by `lb` | tuned `4w` / Q4_0 by `lc` | tuned `4w` / Q4_K_M by `lb` | tuned `4w` / Q4_K_M by `lc` | tuned / SARC `4w` | tuned / stock `4w` |
|---|---:|---:|---:|---:|---:|---:|
| 1B | 2.18 (2.16 to 2.20) | 2.92 (2.63 to 2.94) | 2.55 (2.52 to 2.57) | 3.14 (3.12 to 3.45) | 1.30 (1.26 to 1.31) | 2.76 (2.72 to 2.79) |
| 3B | 2.54 (2.53 to 2.55) | 2.80 (2.78 to 2.84) | 2.95 (2.94 to 2.96) | 3.27 (3.24 to 3.29) | 1.15 (1.15 to 1.17) | 2.71 (2.69 to 2.85) |
| 8B | 2.43 (2.43 to 2.44) | 2.53 (2.52 to 2.55) | 2.88 (2.87 to 2.88) | 3.02 (2.99 to 3.02) | 1.12 (1.12 to 1.13) | 2.48 (2.47 to 2.48) |

| model | tuned `8da4w` / Q4_0 by `lb` | tuned `8da4w` / Q4_0 by `lc` | tuned `8da4w` / Q4_K_M by `lb` | tuned `8da4w` / Q4_K_M by `lc` | tuned / SARC `8da4w` | tuned / stock `8da4w` |
|---|---:|---:|---:|---:|---:|---:|
| 1B | 2.70 (2.68 to 2.71) | 3.61 (3.27 to 3.63) | 3.16 (3.14 to 3.17) | 3.90 (3.88 to 4.26) | 1.37 (1.35 to 1.39) | 3.36 (3.34 to 3.38) |
| 3B | 3.22 (3.20 to 3.23) | 3.56 (3.51 to 3.59) | 3.74 (3.71 to 3.75) | 4.15 (4.10 to 4.17) | 1.20 (1.19 to 1.22) | 3.35 (3.32 to 3.36) |
| 8B | 3.17 (3.16 to 3.19) | 3.30 (3.28 to 3.34) | 3.75 (3.73 to 3.77) | 3.93 (3.89 to 3.95) | 1.17 (1.16 to 1.18) | 3.12 (3.10 to 3.16) |

How `ratios.csv` is computed (columns `model, scheme, arm_pair, ratio, low, high, n`): per arm the population is the valid timed runs of the ExecuTorch and `lc` arms
(one rate each) or the five repetitions of the `lb` arm; `ratio` = median of the numerator arm over median of the denominator arm; `low` = lowest numerator value
over highest denominator value; `high` = highest over lowest; `n` = the smaller of the two population sizes; `scheme` is the ExecuTorch scheme of the numerator.
Arm pairs: `tuned-S/lc-Q-best` and `tuned-S/lb-Q-best` (llama.cpp Q4_0 `q4_0`, Q4_K_M `q4_k_m`), `tuned-S/sarc-S`, `tuned-S/stock-S`. Nothing from M51 is in
`results/cells.csv`, which holds absolute medians.

Reading: tuned `4w` is ahead of llama.cpp by a factor of 2.2 to 3.3 (`8da4w` has no llama.cpp counterpart; its rows show the same arms for completeness, 2.7 to 4.2);
`lb`, the more favourable timer, gives the smaller factor. Tuned is ahead of SARC and of stock in every cell.

## Does llama.cpp run correctly on that driver? Not as shipped, and only in one configuration

- **Default settings: it does not run.** Flash attention is selected automatically and its cooperative-matrix pipeline cannot be created by the driver;
  llama.cpp aborts at start-up (`-fa on` the same). Recorded locally for every model and quantization, two attempts each.
- **With flash attention off the default path runs, but its output is wrong.** The cooperative-matrix matrix-multiplication path computes garbage: the check
  prompt's continuation is nonsense tokens, against the sensible continuation of every ExecuTorch arm and of llama.cpp's own CPU backend on the same
  board. This was checked on 1B (both quantizations) and 3B against the CPU reference, token for token.
- **With the cooperative-matrix path disabled (`GGML_VK_DISABLE_COOPMAT=1`) it runs and its output equals the CPU backend's** (1B and 3B, both
  quantizations, the eight-token continuation identical; 8B: the same continuation as on the other devices). This configuration, at its screened best setting
  (`-b 2048 -ub 1024 -fa on`: `llama-bench` is flat across the screened settings and this is the highest, within the noise band of the default setting;
  `llama-completion` is faster at it than at the default), is the llama.cpp arm of every ratio above. It does not trip the watchdog at 8B.
- **Side note, the wrong-output path.** The faster cooperative-matrix path with flash attention off (wrong output, 1B and 3B only, measured in the first session) is behind tuned ExecuTorch too:
  1B: tuned `4w` / Q4_0 1.73 by `lb`, 2.83 by `lc`; / Q4_K_M 1.98 by `lb`, 3.60 by `lc`; 3B: tuned `4w` / Q4_0 1.55 by `lb`, 1.93 by `lc`; / Q4_K_M 1.83 by `lb`, 2.39 by `lc`. These ratios are in `ratios.csv` labelled `SIDE-NOTE-wrong-output`; they are not a result for llama.cpp, whose output is wrong there.
- Two variants that change the integer-dot path were screened and not used (one changes the check text on Q4_0, so it is not eligible as "correct").
- **OpenCL:** the llama.cpp OpenCL backend builds for arm64 against the board's OpenCL library, but at start-up it rejects the GPU as unsupported (its
  target is Adreno) and drops the device, so every layer would run on the CPU: it does not run on this GPU. No OpenCL arm was measured.
- `lc` (a fresh process) is far below `lb` for llama.cpp on this driver (cause not established; the process's first evaluation is part of the `lc` number, as the
  kit says). Both timers are reported and either gives the same statement.

## The stock arm

Stock ExecuTorch built for Android arm64: upstream `release/1.5` at `985c1ceccc` plus the kit's compile-only backport (`stock-backport-03f41d2031`), the NDK r29, the workspace build tool the M51
campaign's own is a copy of, with its flags (`sarc-build-native.sh --llama --no-tests`, Release, Vulkan, `android-28`, the Vulkan SDK's `glslc`), as the x86_64 stock arm. For the 8B session the same tree
carries the opt-in `ET_VK_EXECUTE_NODE_THRESHOLD` block of the workspace's `m51-execute-node-threshold.patch` (behaviour unchanged unless the variable is set), and 8B runs set it to the
campaign's value in every ExecuTorch arm; 1B and 3B use the unpatched build and never set it. Stock builds and runs on this board, and its text check equals the SARC and tuned arms'.

## Deviations from the kit and what was not done

- The kit's `session.sh` runs binaries on the machine it runs on; the board session script is a local-only variant of it with the board rules of the
  M51 campaign's `e2e_m51.sh` (not committed, because it names the board). `row.py`, `session.sh` and `aggregate.py` are unchanged, but they did not judge the M51 runs:
  the validity of every M51 run was judged by that local variant, with the M51 campaign's rules (not by `row.py`).
- llama.cpp `b11430`, built with the NDK r29 (Android 34, arm64-v8a, `-DGGML_VULKAN=ON -DGGML_NATIVE=OFF`, static libraries, the NDK's OpenMP runtime pushed beside the
  binaries). GGUF files as on the other devices. The same prompt, context (`-c 2560`) and `--override-kv tokenizer.ggml.add_bos_token=bool:false` as the kit.
- Earlier local sessions (the first with the cooperative-matrix path, the second and third with the correct-output llama.cpp configuration but without a stock arm) agree with the
  fourth and fifth within their noise (one noisy `lc` cell excepted) and are kept locally; the ratios here come from the fourth and fifth only, except the labelled side note.
