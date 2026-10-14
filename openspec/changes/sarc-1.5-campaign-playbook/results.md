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

| device | 4w gain | 8da4w gain | both | x stock, September -> now | state on 2026-10-07 | branch |
|---|---:|---:|---:|---|---|---|
| Jetson Orin Nano | +65.7 % | +67.8 % | +66.7 % | 4.46 -> 7.43 | finished and pushed on the corrected build (reopened once by review, `LESSONS.md` L8) | `topic/orin-prefill-refine` |
| Arc B580 | +46.0 % | +77.8 % | +61.1 % | 2.19 -> 3.53 | finished, pushed | `topic/b580-prefill-refine` |
| Arc Pro B70 | +48.8 % | +70.7 % | +59.4 % | 2.17 -> 3.46 | finished, pushed | `topic/xe2-prefill-refine` |
| RTX 4070 Ti SUPER | +43.2 % | +49.2 % | +46.2 % | 3.57 -> 5.22 | finished, pushed | `topic/4070ti-prefill-refine` |
| Radeon 780M | +29.8 % | +36.4 % | +33.1 % | 2.08 -> 2.76 | finished, pushed | `topic/780m-prefill-refine` |

All five rows are final: every campaign ended by the stop rule, was signed off by the independent reviewer and
is pushed (the last one, the Radeon 780M, on 2026-10-07 01:16 UTC). Measured against the parent re-measured in
the campaign's own final session the gains are +59.23 % (Arc Pro B70) and +33.82 % (Radeon 780M); the table
uses the September report's values as the base, as for the other rows.

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
| Sampled parameter search, 2000 samples per space, B70 4w and 8da4w spaces | 4w tiles 2 to 6 % faster per shape, +1.1 % end to end; nothing for 8da4w | 33 hours with the two enumerations below |
| Enumeration of the attention kernels' parameters, B70 (1102 + 2828 configurations) | 3 to 22 % at kernel level; +0.6 and +0.95 % end to end, inside the band, not adopted | hours |
| The same enumeration on the 780M (1796 configurations) | QK^T 9 to 30 %, attention x V 8 to 10 % at kernel level, on the three-kernel path that the fused kernel replaces; not gated | 11 hours |
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

### Jetson Orin Nano (session before the re-gate; the final build's numbers are in the update above; source: its `STATUS.md`)

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

### Arc Pro B70 (final session `s12-final5`; source: its `STATUS.md`)

Tuned = profile `xe2-refine5`. tok/s, 4w / 8da4w: 1B 17964.90 / 20686.90, 3B 7529.41 / 9570.09, 8B 3379.54 /
4481.40; +59.23 % geomean over the pristine parent, in two final sessions.

### Radeon 780M (final sessions; source: its `STATUS.md`)

Tuned = `ET_VK_SARC_DEV_PROFILE=780m-refine3 ET_VK_SARC_780M_PROFILE=c11`.

| cell | `dev/1.5` | tuned | gain |
|---|---:|---:|---:|
| 1B 4w | 2691.20 | 3835.21 | +42.51 % |
| 1B 8da4w | 2537.79 | 3764.71 | +48.35 % |
| 3B 4w | 1140.95 | 1458.69 | +27.85 % |
| 3B 8da4w | 1049.72 | 1403.70 | +33.72 % |
| 8B 4w | 517.43 | 638.60 | +23.42 % |
| 8B 8da4w | 487.85 | 628.03 | +28.73 % |

+33.82 % geomean over `dev/1.5`.

### RTX 4070 Ti SUPER

Its campaign's final table is in `STATUS.md` on its branch and was not copied here. The cells of section 4 are
the same build measured again in the comparison session: the tuned 4w arm reads 29681 / 12881 / 5988 against
the campaign's 29681 / 12800 / 5988.

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

### Arc Pro B70 (llama.cpp Vulkan and SYCL; second session, with the campaign's final kernels)

| model | stock 4w | stock 8da4w | SARC 4w | SARC 8da4w | tuned 4w | tuned 8da4w | llama.cpp Vulkan | llama.cpp SYCL | tuned 4w / Vulkan | tuned 4w / SYCL | tuned 8da4w / SYCL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1B | 4592 | 8359 | 11703 | 12412 | 17965 | 20687 | 9697 | 21547 | 1.85 | 0.83 | 0.96 |
| 3B | 1705 | 3266 | 4842 | 5211 | 7502 | 9526 | 5203 | 9300 | 1.44 | 0.81 | 1.02 |
| 8B | 781 | 1381 | 2435 | 2731 | 3380 | 4491 | 2743 | 4530 | 1.23 | 0.75 | 0.99 |

Five valid runs in every cell. Every arm other than tuned 4w is within 1.8 % of the first session (profile
`xe2-refine2`: tuned 4w 17504 / 7367 / 3285), which is kept beside it as `*-s1.csv`.

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

### Jetson Orin Nano (llama.cpp Vulkan and CUDA; tuned arm = the first form of the last softmax kernel, within 0.2 % of the final build)

| model | stock 4w | SARC 4w | SARC 8da4w | tuned 4w | tuned 8da4w | llama.cpp Vulkan | llama.cpp CUDA | tuned 4w / Vulkan | tuned 4w / CUDA | tuned 8da4w / CUDA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1B | 229.4 | 890.4 | 823.2 | 1491.6 | 1381.0 | 1538.1 | 1689.3 | 0.97 | 0.88 | 0.82 |
| 3B | 82.5 | 360.4 | 320.3 | 629.2 | 570.3 | 557.2 | 630.2 | 1.13 | 1.00 | 0.90 |
| 8B | 35.4 | 189.7 | 170.4 | 295.3 | 268.9 | 244.3 | 286.4 | 1.21 | 1.03 | 0.94 |

### Radeon 780M (llama.cpp Vulkan only; tuned arm one step before the final configuration, whose 8da4w cells are about 2 % higher)

| model | stock 4w | stock 8da4w | SARC 4w | SARC 8da4w | tuned 4w | tuned 8da4w | llama.cpp Q4_0 | llama.cpp Q4_K_M | tuned 4w / llama.cpp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1B | 1176* | 1620 | 2681 | 2538 | 3828 | 3690 | 2880 | 2081 | 1.33 |
| 3B | 404* | 560 | 1126 | 1044 | 1450 | 1365 | 959 | 690 | 1.51 |
| 8B | 184* | 275* | 507* | 483 | 630 | 610 | 451* | 308 | 1.40 |

\* ran at a workload-dependent clock below the campaign's floor and is counted with its clock recorded
(`LESSONS.md` L17). No vendor backend of llama.cpp was measured on this device: its HIP (ROCm) backend exists, but
this integrated GPU is outside ROCm's support list and it was not tried.

### What the comparison shows

- Against llama.cpp's Vulkan backend the tuned kernels are 1.16 to 1.85 times as fast on the AMD and Intel
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

The Radeon RX 7600 campaign started on 2026-10-06 (`topic/rx7600-prefill-refine`): its baseline reproduced the
row above within 0.4 % per cell, the 780M's softmax timed +1.48 % and the 780M's fused attention kernel +18.22 %
on top of it (+9.8 to +29.4 % per cell, every next-token item the same).
