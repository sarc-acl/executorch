# Kit of the llama.cpp comparison

| file | what it does |
|---|---|
| `PRIOR-WORK.md` | what the August 2026 comparison contains and what is taken from it |
| `MODELS.md`, `VERSIONS.md` | model files with bits per weight; pinned versions; the prompt's token count |
| `make-q4km.sh` | Q4_K_M from the F16 GGUF files, default quantization, with a note per file |
| `build-stock.sh` | the stock ExecuTorch arm: `release/1.5` at `985c1ceccc` plus the compile-only backport |
| `session.sh` | one interleaved session of all arms on one device |
| `row.py` | judges one run (valid or not, and why) |
| `aggregate.py` | medians per (model, arm) from one or more sessions |

## Running a device

1. Build llama.cpp at the pinned commit outside the repository, tools only
   (`-DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_NATIVE=ON`, targets `llama-completion llama-bench`). It
   needs `vulkan.hpp`, SPIRV-Headers and a `glslc` that knows the cooperative-matrix GLSL extension.
2. Build or reuse the ExecuTorch arms and stage them: one directory per arm with `llama_main`,
   `libllama_runner.so`, `env` (one `KEY=VALUE` per line, may be empty) and `COMMIT`; `prompt_2048.txt` and
   `prompt_check.txt` of `sarc-1.5-e2e-benchmark` beside them.
3. Write `arms.tsv` in the stage directory and the same list, with commits, in `results/<device>/ARMS.md`.
   Commit `ARMS.md` before the timed session.
4. `session.sh --tools <the device's campaign tools> --stage <stage dir> --out <name>`.
5. `aggregate.py <device> <stage dir>/<name> --out results/<device>/cells.csv`.

## What a valid timed run is

The device's tuning campaign rules, unchanged: exit status 0; exactly 2048 prompt tokens; no foreign GPU
workload before, during or after; at least five clock samples inside the prefill window; the median clock in
the window at or above the campaign's calibrated floor; no thermal throttle reason; foreign engine time in the
window at or below the campaign's calibrated ceiling. An invalid run stays in `runs.csv` with its reason and
is replaced, at most three times per arm.

## How the two llama.cpp timers differ

- `lc`, `llama-completion` on the real prompt: a fresh process per run, as for ExecuTorch. It has no
  full-prompt warm-up inside the process (ExecuTorch's `--warmup` runs the whole prompt once before the timed
  pass), so repetition 0 of each arm is run and discarded; what a later process still pays on its first
  evaluation is part of its number.
- `lb`, `llama-bench`: one process, its own warm-up, five repetitions, synthetic tokens. The most favourable
  way to time llama.cpp and the one others will reproduce. No per-repetition window exists, so the clock is
  recorded over the whole process and not judged.

Both are reported. Where they differ, the difference is stated, and the higher is llama.cpp's number.

## Differences between the runtimes that go with every table

Quantization format and bits per weight (`MODELS.md`); block 32 against group 128; scale search (round to
nearest against HQQ); tied against untied output layer for 1B and 3B (storage only); `8da4w` has no llama.cpp
counterpart; fp16 compute on the ExecuTorch Vulkan side; the check prompt tokenizes to 1972 tokens in
ExecuTorch and 1980 in llama.cpp (the timed prompt is 2048 in both); model load is excluded on both sides.
