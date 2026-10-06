# Results of the five campaigns and of the llama.cpp comparison

What this file is for: the final numbers a new campaign is compared with, in one place, with the source of
each. Nothing here was measured for this package. The per-campaign evidence (raw runs, gates, traces) is in
`openspec/changes/sarc-1.5-<tag>-prefill-refine/` on each topic branch; the comparison's is in
`openspec/changes/sarc-1.5-llamacpp-compare/results/` on branch `topic/llamacpp-compare`.

Quantity everywhere: prefill tokens per second on a 2048-token prompt (`prompt_2048.txt`), one fresh process
per run with `--warmup`, median of 5 valid runs. Models: Llama 3.2 1B, Llama 3.2 3B, Llama 3.1 8B. Schemes:
`4w` (4-bit weights) and `8da4w` (4-bit weights, 8-bit dynamic activations).

## 1. Tuning gain per device

"September" is the SARC release of the earlier report (`sarc-1.5-e2e-benchmark/results/cells.csv`); the gains
are over it. "x stock" is the speed-up over unmodified ExecuTorch 1.5 (`release/1.5`), before and after the
campaign. All values are geometric means over the three models. Source: the owner's summary of the campaigns'
final sessions.

| device | 4w gain | 8da4w gain | both | x stock, September -> now | state on 2026-10-06 | branch |
|---|---:|---:|---:|---|---|---|
| Jetson Orin Nano | +65.7 % | +67.8 % | +66.7 % | 4.46 -> 7.43 | finished and pushed on the corrected build (reopened once by review, `LESSONS.md` L8) |
| Arc B580 | +46.0 % | +77.8 % | +61.1 % | 2.19 -> 3.53 | finished, pushed | `topic/b580-prefill-refine` |
| Arc Pro B70 | +45.3 % | +70.9 % | +57.6 % | 2.17 -> 3.42 | result stable; sampled search still running | `topic/xe2-prefill-refine` |
| RTX 4070 Ti SUPER | +43.2 % | +49.2 % | +46.2 % | 3.57 -> 5.22 | finished, pushed | `topic/4070ti-prefill-refine` |
| Radeon 780M | +30.0 % | +33.8 % | +31.9 % | 2.08 -> 2.74 | stop rule met, closing | `topic/780m-prefill-refine` |

**The Jetson Orin Nano, Arc Pro B70 and Radeon 780M rows are as of 2026-10-06 and are to be updated** when the
re-gate, the search and the closing session have ended. The Orin numbers were measured with the first form of
its last softmax kernel, before the fix of L8.

Two notes on reading the table:

- Arc B580, 8da4w: the September value of the 8B cell (1680 tok/s) was a disturbed median (`LESSONS.md` L21).
  Against the parent re-measured in the same session the campaign's gain is +57.65 % over both schemes instead
  of +61.1 %.
- The devices that gained most are the ones that still ran the stock attention kernels in September.

## 2. Where the gain came from

| source | devices | gain (geometric mean, end to end) |
|---|---|---|
| Cooperative-matrix QK^T and attention x V kernels, ported from the 780M's | B580, B70, RTX 4070 Ti SUPER, Orin (the four that ran the stock attention) | +46 to +58 % |
| Softmax that reduces in fp32 and does not write the zero tail | all | +4 % on the 780M; not reported separately for the others |
| 8da4w: reading whole texels of weights (the fetch cost more than the multiply) | Intel cards, Orin | +6 to +7 % on Intel |
| One fused attention kernel | 780M (already had the attention kernels) | +13 % |
| Linear kernel chosen per layer shape | 780M | +3 % |
| Initial parameter refinement | 780M | +8 % |

What did not pay:

| attempt | result | cost |
|---|---|---|
| Tile sweeps of the linear kernels | nothing faster, on every device | hours per sweep |
| Sampled parameter search, 2000 samples per space, B70 4w and 8da4w spaces | nothing faster | about two days |
| Softmax that reads its row once (Orin) | slower | one candidate |

## 3. Per-cell numbers

### Arc B580 (final session of the campaign; source: its `STATUS.md`)

Parent = the pristine parent commit, re-measured in the same session; tuned = profile `b580-refine3`.

| cell | parent | tuned | gain | September | vs September |
|---|---:|---:|---:|---:|---:|
| 1B 4w | 8641.35 | 12962.00 | +50.00 % | 8605.04 | +50.6 % |
| 1B 8da4w | 8827.59 | 15170.40 | +71.85 % | 8789.70 | +72.6 % |
| 3B 4w | 3436.24 | 5184.81 | +50.89 % | 3419.03 | +51.6 % |
| 3B 8da4w | 3518.90 | 6400.00 | +81.88 % | 3506.85 | +82.5 % |
| 8B 4w | 1726.81 | 2306.31 | +33.56 % | 1693.96 | +36.1 % |
| 8B 8da4w | 1845.05 | 2998.54 | +62.52 % | 1680.07 | +78.5 % (September value disturbed) |

Gated steps: attention kernels +45.95 % (accepted under the reference-error rule, not a plain pass: the 8B 8da4w
next token differs on two prompts); 8da4w whole-texel tile +7.67 %; a 4w tile -0.07 % (not adopted); a QK^T
variant +0.16 %. The last two are the two consecutive candidates under 2 % that ended the campaign.

Where the time went (warm ETDump, ms per prefill, same source):

| cell | arm | total | linear | QK^T | attention x V | softmax | other |
|---|---|---:|---:|---:|---:|---:|---:|
| 1B 4w | parent | 235.1 | 87.9 | 40.7 | 48.3 | 22.4 | 35.8 |
| 1B 4w | tuned | 156.5 | 88.2 | 7.0 | 8.0 | 17.3 | 36.0 |
| 8B 4w | parent | 1179.1 | 651.8 | 153.7 | 187.9 | 45.2 | 140.5 |
| 8B 4w | tuned | 884.4 | 665.5 | 20.1 | 21.8 | 34.8 | 142.2 |
| 8B 8da4w | parent | 1104.9 | 554.5 | 150.0 | 183.4 | 44.8 | 172.3 |
| 8B 8da4w | tuned | 680.4 | 433.2 | 19.4 | 21.0 | 34.3 | 172.6 |

("other" of the 8da4w rows includes the 8-bit activation quantize, 59.8 ms.)

**Update 2026-10-06 10:45 UTC, Jetson Orin Nano closed.** The corrected softmax kernel (only the elected lane
stores) was gated again and accepted (+0.34 % over its parent, where the first form had read +0.41 %), and the
final session of the whole stack on the corrected build reads, in tok/s (parent -> final): 1B 890.8 -> 1489.5 and
824.8 -> 1382.9; 3B 360.5 -> 629.0 and 320.3 -> 570.3; 8B 189.8 -> 295.5 and 170.4 -> 269.2 (4w and 8da4w), +66.65 %
geomean, every next-token item the same as the parent's. These replace the numbers of the first form in the table
below by at most 0.2 % per cell. The llama.cpp comparison on this device was taken with the first form.

### Jetson Orin Nano (as of 2026-10-06, to be updated; source: its `STATUS.md`, session before the re-gate)

| cell | parent | tuned | gain |
|---|---:|---:|---:|
| 1B 4w | 890.82 | 1490.54 | +67.3 % |
| 1B 8da4w | 824.14 | 1380.98 | +67.6 % |
| 3B 4w | 360.44 | 629.38 | +74.6 % |
| 3B 8da4w | 320.30 | 570.95 | +78.3 % |
| 8B 4w | 189.77 | 295.61 | +55.8 % |
| 8B 8da4w | 170.53 | 269.19 | +57.9 % |

60 timed runs, all valid, every run at 612 MHz, repeat spread at most 0.28 %. Next token equal to the parent's
in all six cells.

### Arc Pro B70, RTX 4070 Ti SUPER, Radeon 780M

Their campaigns' own final tables are in `STATUS.md` on their branches and were not copied here. The cells of
section 4 are the same builds measured again in the comparison sessions; on the B70 they reproduce the
campaign's final session within 0.3 %, on the RTX 4070 Ti SUPER the tuned 4w arm reads 29681 / 12881 / 5988
against the campaign's 29681 / 12800 / 5988.

## 4. Comparison with llama.cpp (2026-10-05 and 2026-10-06)

Source: `sarc-1.5-llamacpp-compare/results/<device>/README.md` and `results/b580/cells*.csv`. llama.cpp tag
`b11430`, Q4_0 files, context 2560, its warm timer (`llama-bench`) at its best screened setting per device.
"stock" = unmodified ExecuTorch `release/1.5`; "SARC" = the September kernels (each campaign's parent); "tuned" =
the campaign's accepted profile. The ratios in the B580 table were computed for this file from the medians
shown; the other tables are copied.

The arms are not identical configurations: quantization format and bits per weight differ (Q4_0 in blocks of
32 against groups of 128), `8da4w` has no llama.cpp counterpart, the tuned arms are unmerged development
branches, and some were accepted by the reference-error rule. Read the per-device README before quoting a
number.

### Arc B580 (llama.cpp Vulkan and SYCL)

| model | stock 4w | stock 8da4w | SARC 4w | SARC 8da4w | tuned 4w | tuned 8da4w | llama.cpp Vulkan | llama.cpp SYCL | tuned 4w / Vulkan | tuned 4w / SYCL | tuned 8da4w / SYCL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1B | 3180 | 5953 | 8641 | 8866 | 13045 | 15284 | 7529 | 15523 | 1.73 | 0.84 | 0.98 |
| 3B | 1191 | 2195 | 3442 | 3519 | 5198 | 6420 | 3675 | 6232 | 1.41 | 0.83 | 1.03 |
| 8B | 541 | 920 | 1727 | 1845 | 2301 | 3003 | 1976 | 3189 | 1.16 | 0.72 | 0.94 |

ExecuTorch and llama.cpp Vulkan from session `s1`; SYCL from session `s2-sycl` (1B, 3B) and `s3-8b` (8B).

### Arc Pro B70 (llama.cpp Vulkan and SYCL; as of 2026-10-06, to be updated)

| model | stock 4w | stock 8da4w | SARC 4w | SARC 8da4w | tuned 4w | tuned 8da4w | llama.cpp Vulkan | llama.cpp SYCL | tuned 4w / Vulkan | tuned 4w / SYCL | tuned 8da4w / SYCL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1B | 4582 | 8325 | 11703 | 12412 | 17504 | 20687 | 9704 | 21571 | 1.80 | 0.81 | 0.96 |
| 3B | 1702 | 3256 | 4830 | 5265 | 7367 | 9570 | 5232 | 9296 | 1.41 | 0.79 | 1.03 |
| 8B | 778 | 1375 | 2421 | 2734 | 3285 | 4481 | 2695 | 4521 | 1.22 | 0.73 | 0.99 |

8B tuned 4w: four valid runs (two of six read a median clock 5 MHz under the floor at the same speed).

### RTX 4070 Ti SUPER (llama.cpp Vulkan and CUDA)

| model | stock 4w | SARC 4w | SARC 8da4w | tuned 4w | tuned 8da4w | llama.cpp Vulkan | llama.cpp CUDA | tuned 4w / Vulkan | tuned 4w / CUDA | tuned 8da4w / CUDA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1B | 6804 | 19692 | 21113 | 29681 | 33032 | 31893 | 40861 | 0.93 | 0.73 | 0.81 |
| 3B | 2554 | 8715 | 9660 | 12881 | 14949 | 12471 | 15837 | 1.03 | 0.81 | 0.94 |
| 8B | 1110 | 4472 | 5007 | 5988 | 6954 | 5539 | 7418 | 1.08 | 0.81 | 0.94 |

8B tuned arms: four valid runs each. The first session on this device was discarded (`LESSONS.md` L39).

ExecuTorch's own upstream CUDA backend on the same card, 4w, its own exporter (context 3072, bf16):

| model | tok/s | against stock Vulkan 4w |
|---|---:|---:|
| 1B | 5802 | 0.85 |
| 3B | 1930 | 0.76 |
| 8B | 836 | 0.75 |

### Jetson Orin Nano (llama.cpp Vulkan and CUDA; as of 2026-10-06, to be updated)

| model | stock 4w | SARC 4w | SARC 8da4w | tuned 4w | tuned 8da4w | llama.cpp Vulkan | llama.cpp CUDA | tuned 4w / Vulkan | tuned 4w / CUDA | tuned 8da4w / CUDA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1B | 229.4 | 890.4 | 823.2 | 1491.6 | 1381.0 | 1538.1 | 1689.3 | 0.97 | 0.88 | 0.82 |
| 3B | 82.5 | 360.4 | 320.3 | 629.2 | 570.3 | 557.2 | 630.2 | 1.13 | 1.00 | 0.90 |
| 8B | 35.4 | 189.7 | 170.4 | 295.3 | 268.9 | 244.3 | 286.4 | 1.21 | 1.03 | 0.94 |

### Radeon 780M (llama.cpp Vulkan only; as of 2026-10-06, to be updated)

| model | stock 4w | stock 8da4w | SARC 4w | SARC 8da4w | tuned 4w | tuned 8da4w | llama.cpp Q4_0 | llama.cpp Q4_K_M | tuned 4w / llama.cpp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1B | 1176* | 1620 | 2681 | 2538 | 3828 | 3690 | 2880 | 2081 | 1.33 |
| 3B | 404* | 560 | 1126 | 1044 | 1450 | 1365 | 959 | 690 | 1.51 |
| 8B | 184* | 275* | 507* | 483 | 630 | 610 | 451* | 308 | 1.40 |

\* ran at a workload-dependent clock below the campaign's floor and is counted with its clock recorded
(`LESSONS.md` L17). No vendor backend of llama.cpp was measured on this device.

### What the comparison shows

- Against llama.cpp's Vulkan backend the tuned kernels are 1.16 to 1.80 times as fast on the AMD and Intel
  devices and level on the NVIDIA ones (0.93 to 1.21).
- Vendor backends of llama.cpp are faster than tuned 4w: SYCL by 19 to 39 %, CUDA by 23 to 38 % on the
  RTX 4070 Ti SUPER; on the Orin CUDA is 13 % ahead on 1B and level on 3B and 8B. Tuned 8da4w is within 6 % of
  SYCL.
- ExecuTorch's own upstream CUDA backend ran 4w at 0.75 to 0.85 times stock Vulkan on the RTX 4070 Ti SUPER.
- llama.cpp's best settings differ per device and backend, and its fresh-process timer reads several times
  lower than its warm timer on SYCL (`LESSONS.md` L42, L43).

## 5. Starting points for the next devices (already in the tree, unverified)

Source: `sarc-1.5-e2e-benchmark/contrib/{7900xtx,rx7600}/NOTES.md`, measured 2026-09-28 with unverified rows,
a native toolchain (not the pinned container) and `ET_VK_SARC_UNVERIFIED=1`. Prompt "the" x 2048, tok/s.

| device | driver then | 4w 1B / 3B / 8B | 8da4w 1B / 3B / 8B | x stock (real-text prompt, both schemes) |
|---|---|---|---|---:|
| Radeon RX 7900 XTX | AMDVLK 2025.Q2.1 | 20078 / 10089 / 4774 | 22261 / 10396 / 4971 | 2.79 |
| Radeon RX 7600 | RADV, Mesa 26.2.3 (user-space build) | 7787 / 3287 / 1517 | 7340 / 3080 / 1403 | 2.48 |

These are the "published table" a new AMD campaign compares its baseline with (3 % per cell, else explain).
They were taken on different commits and a different toolchain than a campaign will build, so a difference is
possible and must be explained, not assumed away.
