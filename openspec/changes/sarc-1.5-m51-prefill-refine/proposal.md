# sarc-1.5-m51-prefill-refine

Dev zone only. Branch `topic/m51-prefill-refine`, forked from `topic/780m-prefill-refine` at `f5f1bf10c` (the
parent of this campaign). Device: the Samsung Xclipse (M51) board, reached over adb from a build workstation.

**This directory holds tooling and number-free prose only.** By the device owner's rule, no figure of this device
(throughput, ratios, percentages, kernel times, clock values, register counts) and no host, board or driver
identifier is committed or pushed. Every table, the running `STATUS.md`, the thresholds and the raw data are kept
in the campaign's local artifact directory (`m51-LOCAL-ONLY`). Commit messages say what changed, never how much.

## Why

The xclipse rows of `impl/sarc/table_amd.cpp` (4w, 8da4w, attention QK^T and attention x V, all `kUnverified`)
already run the Llama prefill on this device with the SARC kernels. The device is in the Radeon 780M's position
(attention rows exist, RDNA-derived), so this change ports the 780M's second-layer results (owner decision N1)
and gates each one on this device: the fused attention kernel, the fp32 softmax without the zero tail, the linear
kernel chosen per layer shape, and the whole-texel 8da4w weight staging. No sampled parameter search.

## Thresholds

Fixed before the first measurement, in the local file `THRESHOLDS.md` of the artifact directory, sha256
`fcd8de6833afd9488a1e21ee95a58e9f16684b624be7b4ae2432f0ccc6fa564d`. They are the shared campaign rules (R4.4
baseline tolerance, R6 noise band, repeat rule and clock-floor rule, R8 kernel-screen margin, R11 stop rule, owner
decisions D1 and D3) applied to this device, plus the validity checks that pinned clocks allow (no active time at
any other GPU frequency, GPU cooling state zero, no GPU reset). Values derived from the A/A session go into a
separate local file, by the rule written in the frozen one.

## Method (adaptations to a board reached over adb)

- Builds on the workstation: `tools/build_tag.sh <tag> <commit>` exports exactly one commit and every submodule at
  its pinned commit from the git object stores (`tools/export_commit.sh`), then cross-compiles with
  `tools/build-native.sh` (NDK, Vulkan SDK glslc). The pinned build container is not available, so the shipped
  SPIR-V cannot match `sarc/golden/spirv.json`: the golden check is reported as pending, and the shipped variants
  of every candidate build are compared byte for byte with the parent build made by the same toolchain instead.
- The board has no bash, so `sarc/tools/verify.sh` runs unmodified on the workstation against a stage directory
  whose `llama_main` and `test_llama_microbench` are wrappers around `tools/adbshim.sh`. The shim runs the staged
  binaries on the board with the caller's `ET_VK_*` environment, maps the model and tokenizer paths and the
  `--json-out` file, and returns the device process's output and exit status. Every run is preceded by the
  device-state guard (driver identity, profiler configuration aside, clock pins read back) and logged.
- Timed sessions: `tools/e2e_m51.sh`, the 780M's `e2e5.sh` protocol on the board (fresh `llama_main` per run,
  `--warmup`, one new token, arms interleaved, cooling before each run), with a sampler on the board for the GPU
  clock, busy, temperature and thermal state, the per-frequency active time and the GPU reset count around each
  run, and the model load time of each run (whether the load was slow). `tools/summarize.py` summarises.
- Coordinator hold: every build, `verify.sh` call and timed session is one unit of `tools/hold.sh`.
- Board, serial and driver identity live in a local settings file read by `tools/dev.sh`, never in this tree.

### Tools taken from the 780M campaign and the device workspace

| tool | origin | what changed |
|---|---|---|
| `hold.sh` | 780M `tools/hold.sh` | artifact directory |
| `e2e_m51.sh` | 780M `tools/e2e5.sh` | runs on the board over adb; the sampler reads the board's GPU nodes; validity rules for pinned clocks; cooling also ends when the temperature stops falling; a second check prompt (unaligned real text) |
| `summarize.py` | 780M `tools/summarize.py` | column names; repeat count from the environment; load time; three next-token items |
| `stage.sh` | 780M `tools/stage.sh` | pushes the binaries and prompts to the board; adds the `verify.sh` wrappers and the flat model links |
| `build-native.sh` | workspace `tools/sarc-build-native.sh` | venv, glslc and NDK paths from the environment; lower default parallelism (another campaign measures on the workstation) |
| `build_tag.sh`, `export_commit.sh`, `adbshim.sh`, `dev.sh`, `push_models.sh` | new | |
| `r1329.txt` | new | an unaligned real-text prompt (the first part of the kit's `prompt_real_2048.txt`) for the `r*.txt` item of `verify.sh` |

## Status

Kept in the local `STATUS.md`.
