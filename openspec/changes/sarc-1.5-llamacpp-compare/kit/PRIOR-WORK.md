# Prior work: the August 2026 ExecuTorch vs llama.cpp comparison

Read on 2026-10-05 from the 780M host, read-only. The earlier work is a spec of the 1.3-era repository
(`specs/042-et-vs-llamacpp-vulkan`: `PROTOCOL.md`, `report.md`, `scripts/`, `logs/`), a llama.cpp checkout with
a `VERSION.txt`, and GGUF files with a note each. None of it is in this repository. This file records what it
contains and what this change takes from it.

## What it compared

ExecuTorch Vulkan `release/1.3` (no cooperative-matrix kernels at all) against llama.cpp Vulkan `b10229`
(commit `c745be2a2c`), Llama 3.2 1B, 3.2 3B and 3.1 8B, `4w` with 4-bit embeddings against uniform Q4_0, on the
Radeon 780M and two Android phones. Workload: 2048-token prefill plus 1023 decode steps, greedy.

## Results on the Radeon 780M (tok/s, median of 3 timed runs after one discarded run, 2026-08-02)

| model | ExecuTorch 1.3 prefill | llama.cpp prefill | ratio | ExecuTorch decode | llama.cpp decode |
|---|---:|---:|---|---:|---:|
| 1B | 704.0 | 2166.2 | llama.cpp 3.08x | 63.30 | 92.17 |
| 3B | 249.9 | 753.3 | llama.cpp 3.01x | 19.73 | 35.40 |
| 8B | 105.4 | 342.5 | llama.cpp 3.25x | 10.07 | 17.62 |

llama.cpp ablation on the 780M, 1B prefill: default 2166.2; integer-dot path off 2161.7; cooperative matrix off
2080.4; both off 1395.8. The two fast paths are redundant, not additive.

These are old numbers on an old ExecuTorch and an old llama.cpp. They are context for this change, not a
result of it. The phone results are out of scope here.

## What this change takes over unchanged

- **Model files.** F16 and Q4_0 GGUF of all three models, converted from the same upstream revisions as the
  ExecuTorch exports (`Llama-3.2-1B@4e20de36`, `Llama-3.2-3B@13afe512`, `Llama-3.1-8B@d04e592b`). Q4_0 was
  made with `llama-quantize --pure --output-tensor-type q4_0 --token-embedding-type q4_0`, because the default
  silently promotes the embedding / output tensor to Q6_K. Verified there: Q4_0 and F32 norms only.
- **Bits per weight, measured:** ExecuTorch `4w` with 4-bit embeddings 4.213 / 4.184 / 4.158 (1B / 3B / 8B);
  Q4_0 4.552 / 4.521 / 4.509. llama.cpp carries 7.8 to 8.4 % more bits (block 32 against group 128).
- **Other differences it lists, all to be disclosed again:** scale search (HQQ against round-to-nearest
  max-abs); tied embeddings in the 1B and 3B GGUF where ExecuTorch stores an untied output layer (same
  compute, different storage); model load (eager against mmap), excluded from both rates.
- **The llama.cpp invocation**, as the starting point of the `aligned` and `best` tiers:
  `llama-completion -m <gguf> -f <prompt> -n <N> --temp 0 --top-k 1 -c <ctx> -ngl 99 -t <cores> -fit off
  -b 2048 -ub 2048 -fa on -ctk f16 -ctv f16 -no-cnv --no-warmup --ignore-eos -s 1234 --perf`, with the time
  read from the `prompt eval time = ... ms / 2048 tokens` line.
- **Reasons behind three of those flags:** `-fit off`, because `-fit` is on by default and adjusts unset
  arguments to the memory free at launch; `-fa on` set explicitly, because `auto` makes the configuration
  device dependent; the first process of a cell is discarded as the warm-up instead of an in-process warm-up.
- **Three things it verified by experiment:** llama.cpp adds the beginning-of-sequence token itself; `-n N`
  yields N-1 decode steps in both runtimes; a regular expression for `eval time` must be anchored or it also
  matches `prompt eval time`.
- **Validity rules:** wrong prompt token count is invalid; a missing timing line is a crash row, not a missing
  row; every run carries a check of the generated text, because a backend can be fast and numerically wrong (it
  was, on one phone, only for long prompts: a short-prompt smoke test does not catch it).
- **Capability readout:** `llama-bench --list-devices` prints whether llama.cpp sees cooperative matrix and
  integer dot product on a device. Record it per device.
- **Ablation switches:** `GGML_VK_DISABLE_COOPMAT`, `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT`,
  `GGML_VK_DISABLE_F16`. Useful to explain a result; not arms of this change.
- **Build options for Vulkan:** `-DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_NATIVE=ON`, tools only.
  It needed a Vulkan SDK with `vulkan.hpp`, SPIRV-Headers and a recent `glslc` (cooperative-matrix GLSL).

## What differs now, and how this change handles it

| point | August | this change |
|---|---|---|
| ExecuTorch | `release/1.3`, one arm | `release/1.5` stock, SARC, tuned: three arms |
| ExecuTorch model files | own exports, context 3072, `4w` with 4-bit embeddings only | the shared exports used by `sarc-1.5-e2e-benchmark` and the tuning campaigns, `4w` and `8da4w` (context and embedding quantization recorded in `MODELS.md`) |
| prompt | real text, 2047 tokens plus the beginning-of-sequence token | the e2e-benchmark prompt, 2048 tokens without one; see "Prompt" below |
| workload | prefill and 1023 decode steps | prefill only (one new token) |
| llama.cpp | `b10229` | one current commit, pinned in `VERSIONS.md`; `b10229` kept as a fallback arm |
| quantization | Q4_0 | Q4_0 and default Q4_K_M |
| tiers | one (batch 2048, flash attention on) | default, aligned, best |
| repeats | 3 after one discard | 5 valid after one discard |
| timer cross-check | `llama-bench` rejected as primary because it decodes at depth 0 | prefill only, so that objection does not apply; still secondary because it feeds synthetic tokens |
| thermal rule | cool-down floor plus return to within 3 C of a settled baseline | the tuning campaigns' per-device validity rule |
| backends | Vulkan | Vulkan, CUDA, SYCL |

## Prompt

The August prompt file holds 2047 text tokens and relies on each runtime adding the beginning-of-sequence
token. The ExecuTorch 1.5 measurements this change must stay comparable with (`cells.csv` and the campaigns)
use a 2048-token prompt and add none. llama.cpp adds one by default, which would make its prompt 2049 tokens.
This change keeps the ExecuTorch prompt, turns the automatic token off on the llama.cpp side, and requires
`2048 tokens` in every llama.cpp timing line. The flag that does this at the pinned commit is recorded in
`VERSIONS.md` together with the token count it produced.

## Not found

No CUDA or SYCL build, no Q4_K_M file, no measurement on Intel or NVIDIA hardware, no `llama-bench`
cross-check results (the script the run script refers to is not in the directory).
