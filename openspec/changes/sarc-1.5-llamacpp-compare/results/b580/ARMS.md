# Arc B580: arms of the comparison (written before the timed session, 2026-10-05)

Device: Arc B580 (Mesa ANV 26.2.3), the owner's desktop GPU. Models and prompt as in `kit/MODELS.md` and
`kit/VERSIONS.md`; context 2560; 2048 prompt tokens in every arm.

## ExecuTorch Vulkan (each with `4w` and `8da4w`)

| arm | build | selection |
|---|---|---|
| `stock` | upstream `release/1.5` at `985c1ceccc` plus the compile-only backport (built for this change, `kit/build-stock.sh`) | none |
| `sarc` | `6a7cc8cc6`, the pristine parent build of `sarc-1.5-b580-prefill-refine` (its baseline agreed with `cells.csv`) | none |
| `tuned` | `2a52dfd2b`, build `topic3` of the same campaign: an unmerged development branch | `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=b580-refine3` |

The tuned profile was accepted by the campaign's reference-error rule, not by a next-token-identical gate: in
8B `8da4w` its next token differs from the parent's on two prompts.

## llama.cpp `b11430` (`8345f3339`), Vulkan, device Vulkan0

Quantizations `q4_0` (uniform 4-bit, August files) and `q4_k_m` (default quantization, made for this change:
5.18 / 5.01 / 4.89 bits per weight for 1B / 3B / 8B by llama-quantize's own count). Exact lines: `arms.tsv`.

| tier | settings beyond full offload |
|---|---|
| `default` | none (micro-batch 512, flash attention `auto`) |
| `aligned` | `-b 2048 -ub 2048`: the whole prompt in one batch, as ExecuTorch runs it |
| `best` | `-b 2048 -ub 1024 -fa off`, the fastest of the screen below |

`lc` arms time the real prompt with `llama-completion` (all three tiers); `lb` arms are `llama-bench`
(`default` and `best`). 16 arms in all.

## The screen that fixed the `best` tier (1B, `q4_0`; `screen.csv`)

| settings | llama-completion tok/s | llama-bench tok/s |
|---|---:|---:|
| default | 5284 | 5547 |
| `-ub 2048`, flash attention on | 5413 | 6018 |
| `-ub 2048`, flash attention off | 5536 / 5491 | 6257 / 6241 |
| `-ub 1024`, flash attention on | 5749 / 5717 | 6144 |
| `-ub 512`, flash attention off | 6617 | 7082 |
| **`-ub 1024`, flash attention off** | **6857** | **7532** |

Medians of 3 valid runs (llama-completion) or 5 in-process repetitions (llama-bench); two values where a
setting was in both screens. Not screened: other micro-batch sizes, thread counts, other models.

## Thresholds fixed here

- Validity of a run: `kit/README.md` (the campaign's calibrated clock floor 2699 MHz and foreign-busy
  ceiling 5 %).
- `llama-bench` against `llama-completion` on the same settings: the screen shows llama-bench 5 to 11 % higher.
  A cell where they differ by more than 15 % is looked into before it is reported.
- The ExecuTorch `sarc` arm against `sarc-1.5-e2e-benchmark/results/cells.csv`: within 3 %, or explained.

## Second session: llama.cpp SYCL (written before it ran, 2026-10-05)

llama.cpp `b11430` built with `-DGGML_SYCL=ON -DGGML_SYCL_F16=ON` (icx / icpx), oneAPI DPC++ compiler 2026.1.1,
oneMKL 2026.1.0, oneDNN 2026.0.2, Level Zero headers present (`GGML_SYCL_SUPPORT_LEVEL_ZERO_API` on), in a
container; run as a host process with the runtime libraries of that image and the host's Level Zero GPU driver
26.35.39758.11. Environment as in the owner's production SYCL services:
`UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 ZES_ENABLE_SYSMAN=1`. Device `SYCL0`. Exact lines: `arms-sycl.tsv`.

Reference arms repeated in this session so that the two sessions can be joined: ExecuTorch `sarc` 4w, `tuned`
4w and 8da4w, and llama.cpp Vulkan `lb` best on Q4_0.

Settings screen (1B, Q4_0; `screen-sycl.csv`), llama-bench tok/s: default 11967; `-ub 2048` flash attention on
**14856**, off 5914; `-ub 1024` on 13689, off 6489; `-ub 512` off 6356. `best` = `-b 2048 -ub 2048 -fa on`.
The opposite of the Vulkan backend, where flash attention off was faster.

**The two timers disagree by a factor of 3.5 on SYCL with flash attention on**, far beyond the 15 % threshold,
and this was looked into before the session: `llama-completion` in a fresh process reads 4196 tok/s (488 ms)
where `llama-bench` in a warm process reads 14856 (138 ms). During the fresh process's prompt evaluation the GPU
clock is at idle for most of the window (median 1250 to 2100 MHz against 2850 under load; the campaign's clock
rule rejects every such run as `clock_low`), so the extra time is not GPU work: it is first-use work on the
CPU side of the SYCL runtime that an in-process warm-up removes. ExecuTorch's `--warmup` removes the same kind
of cost from its number. The warm number (`lb`) is therefore the one comparable with the ExecuTorch arms; the
fresh-process number (`lc`) is kept and reported as the cold-start cost, and its runs are expected to be
rejected by the clock rule.

## Third session: 8B only (written before it ran, 2026-10-05)

Two purposes. (1) Repeat the 8B part of the SYCL session: desktop use of the card pushed foreign engine time
to 5 to 12 % there, so the ExecuTorch reference arms and three SYCL arms did not reach five valid runs (the SYCL
Q4_0 `best` arm did: 3107 tok/s). (2) Screen llama.cpp Vulkan settings on the 8B model, since its `best` tier
was chosen on 1B only: micro-batch 256, 512, 1024, 2048 with flash attention off, and 1024 with it on.
The Vulkan `best` number reported for 8B becomes the fastest of these. Exact lines: `arms-8b.tsv`.
