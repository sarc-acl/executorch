# M51: llama.cpp Vulkan against tuned ExecuTorch (relative statement only)

Measured 2026-10-10 on one M51 board with the kit's protocol (the M51 campaign's board rules: canonical driver checked before and after every
run, clocks pinned and read back, profiler configuration aside, sampler on the board, cool start and cooling between runs, interleaved
arms, five valid runs per ExecuTorch and `llama-completion` cell, `llama-bench` one process of five repetitions, one text check per arm). **Every
number stays local** (owner rule of 2026-10-08): this file has no rate, no time, no ratio and no identifier of the board, its host or its driver.
"Ahead of" means outside the campaign's 2 % noise band, for both llama.cpp timers (`lc`, `lb`) and for every llama.cpp setting that produces
correct output; the observed gaps are far outside the band, not borderline.

## The statement

| model | tuned ExecuTorch `4w` against llama.cpp Vulkan Q4_0 | against llama.cpp Vulkan Q4_K_M |
|---|---|---|
| 1B | ahead of | ahead of |
| 3B | ahead of | ahead of |
| 8B | ahead of | ahead of |

(Tuned ExecuTorch `8da4w` is ahead of the same llama.cpp arms by more; `8da4w` has no llama.cpp counterpart.) The ExecuTorch arms are the M51
campaign's final configuration (profile `c8`, 8B with the node threshold setting the campaign requires); the parent (`sarc`) arms were measured in the
same sessions and are ahead of llama.cpp too.

## Does llama.cpp run correctly on that driver? Not as shipped, and only in one configuration

- **Default settings: it does not run.** Flash attention is selected automatically and its cooperative-matrix pipeline cannot be created by the driver;
  llama.cpp aborts at start-up (`-fa on` the same). Recorded in `runs.csv` (local) for every model and quantization, two attempts each.
- **With flash attention off the default path runs, but its output is wrong.** The cooperative-matrix matrix-multiplication path computes garbage: the check
  prompt's continuation is nonsense tokens, against the sensible continuation of every ExecuTorch arm and of llama.cpp's own CPU backend on the same
  board. This was checked on 1B (both quantizations) and 3B against the CPU reference, token for token.
- **With the cooperative-matrix path disabled (`GGML_VK_DISABLE_COOPMAT=1`) it runs and its output equals the CPU backend's** (1B and 3B, both
  quantizations, token for token; 8B: the same continuation as on the other devices). This configuration, at its screened best setting, is the llama.cpp
  arm of the statement above (screen of 1B: `llama-bench` is flat across the screened settings, `-b 2048 -ub 1024 -fa on` is the highest, within the noise band of the default setting; `llama-completion` is faster at it than at the default). It does not trip the watchdog at 8B.
- The faster, wrong-output path (cooperative matrices, flash attention off) was also measured on 1B and 3B; tuned ExecuTorch is ahead of it too, so the
  statement does not depend on which of the two llama.cpp configurations is taken. Its speeds are not used for anything else.
- Two variants that change the integer-dot path were screened and not used (one changes the check text on Q4_0, so it is not eligible as "correct").
- **OpenCL:** the llama.cpp OpenCL backend builds for arm64 against the board's OpenCL library, but at start-up it rejects the GPU as unsupported (its
  target is Adreno) and drops the device, so every layer would run on the CPU: it does not run on this GPU. No OpenCL arm was measured.
- `lc` (a fresh process) is far below `lb` for llama.cpp on this driver (cause not established; the process's first evaluation is part of the `lc` number, as the
  kit says). Both timers are reported locally and either gives the same statement.

## Deviations from the kit and what was not done

- No `stock` ExecuTorch arm on M51: an Android build of the stock release was not made (the statement above does not need it).
- The kit's `session.sh` runs binaries on the machine it runs on; the board session script is a local-only variant of it with the board rules of the
  M51 campaign's `e2e_m51.sh` (not committed, because it names the board). `row.py`, `session.sh` and `aggregate.py` are unchanged, but they did not judge the M51 runs: the validity of every M51 run was judged by that local variant, with the M51 campaign's rules (not by `row.py`).
- llama.cpp `b11430`, built with the NDK r29 (Android 34, arm64-v8a, `-DGGML_VULKAN=ON -DGGML_NATIVE=OFF`, static libraries, the NDK's OpenMP runtime pushed beside the
  binaries). GGUF files as on the other devices. The same prompt, context (`-c 2560`) and `--override-kv tokenizer.ggml.add_bos_token=bool:false` as the kit.
- A first timed session (coopmat path, flash attention off) is kept locally as the record of the wrong-output path; the sessions behind the statement are the
  second (1B, 3B) and third (8B) one, with the correct-output configuration.
