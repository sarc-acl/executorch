# sarc-1.5-b580-prefill-refine: status

**2026-10-05 08:35 UTC — running. Accepted: candidate 0 (`b580-refine0`, SDPA kernels, +45.95 % geomean over
the parent, `ACCEPTED (reference-error rule, owner decision 2026-10-04)`, not a plain pass) and candidate 1
(`b580-refine1`, + 8da4w tile `xe2bt_t128x128k64g84s16m8`, +7.67 % geomean over candidate 0, `GATE_PASS`,
bit-identical). Candidate 2 (`b580-refine2`, 4w band-drain tile): `GATE_PASS`, bit-identical, **-0.07 %**
geomean, no gain, not adopted: the first candidate under 2 %. Candidate 3 (`b580-refine3`, QK^T `xe2c` for
head_dim 128) is being built and gated.**

Branch `topic/b580-prefill-refine`, parent `6a7cc8cc6` (no profile). Host `fedora` (the owner's desktop), Arc
B580 = PCI `0000:03:00.0`, Vulkan device 0, `ETVK_DEVICE_INDEX=0`, lock
`86800be2-0000-0000-0300-000000000000`, ANV Mesa 26.2.3. Nothing was run on the Ryzen iGPU (device 1).

## Running now

Detached chain `chain5.sh` (`.artifacts/logs/chain5.status`), one GPU job at a time. **If the machine reboots
it is gone**; restart from the first step whose status line is missing. Remaining steps: build `topic3`
(`1eec0cda0`) and its probe; `gate_sdpa.sh s5-c3` (`b580-refine3` against `b580-refine1`, both arms build
`topic3`; 12 passes of the three SDPA tiers, `verify.sh`, timing, traces); `sdpa_ref.sh` (error against the
fp32 reference); `probe.sh s5-c3` (arms P and C); `decide.py`. After it: a quiet rerun of the 4w screen and a
final session of the recommended profile against the pristine parent.

## How noisy the card was

Until about 05:40 UTC the desktop session was idle and locked (`loginctl`: `IdleHint=yes`, `LockedHint=yes`).
Nine processes hold the card (gnome-shell, Xwayland, firefox, ghostty, ...; listed in every session's
`env.txt`); their share of engine time inside the timed prefill windows was 0.00 % in all 144 timed runs of
`s1-aa` and `s2-c0` (0.3 to 0.5 % in a smoke run made while the display was on). The "about 20 % from
gnome-shell" of the campaign notes was not observed in that state. Calibration from `s1-aa`: `BUSYMAX` = 5 %
(the floor of the rule in `proposal.md`), `CLKMIN` = 2699 MHz (97 % of the lowest per-run median, 2783 MHz in
the 8B 4w cell), idle package temperature 47 C. A/A spread asks for 5 repeats, not 7 (largest A/A cell
deviation 0.08 %, largest arm spread 0.43 %).

Between about 05:40 and 07:30 UTC the owner was using the desktop (`IdleHint=no`). The 4w kernel screen ran
in that period and shows it: repeat spreads of 10 to 34 % on several tiles, and one hot / disturbed base run
that made a tile look 20 % faster than it is (`results/b580/screens/README.md`). Session `s3-c1` also ran in
it: foreign engine time 1.2 to 1.4 % per cell (median), 2.3 % at most, no run rejected, arm spreads up to
2.2 % and absolute tok/s 1.6 to 2.5 % lower than on the idle desktop, in both arms alike. By `s4-c2`
(08:04 UTC) the card was quiet again (0.00 %). Timed sessions reject a run whose foreign engine share
exceeds 5 % (none so far); screens have no such filter, so a screen result counts only if it holds in every
round.

## Parent control and baseline (re-measured here)

Builds: `build/parent` = `6a7cc8cc6`, copied by rsync from `fedora-gpu-eval` (the B70 campaign's build, same OS
and driver); the three binaries have the sha256 recorded there and its 53 shipped SPIR-V variants match
`sarc/golden/spirv.json` on this copy (`results/b580/parent.copy.txt`). `build/topic1` = `98e3eccf4`, built here
in `localhost/et-vk-build:rocky10`, golden unchanged (53 variants).

Parent control `s0-parent-verify` (unmodified `verify.sh --models 1b,3b,8b --schemes 4w,8da4w --pdiff`, no
environment): `CONTROL_RECORDED`. 12 of 12 production-diff cases ALL PASSED, 28 of 28 numeric and 4 of 4
rank-3 correctness cases PASSED, decode 31 tokens, default vs tiled SAME on the real-text prompt (both
schemes) and on the unaligned prompt (4w). The device-status lines of pristine `dev/1.5` are the same as on
the B70 and are compared line by line for every candidate: `correctness rc=1`, `linear <scheme> rc=1`,
`1b 8da4w unaligned: default vs tiled output DIFFER`, SDPA tiers on the stock kernels with 0 mismatches.

Baseline + A/A, session `s1-aa` (pristine parent against build `topic1` with no environment, arms interleaved,
median of 5 valid runs, tok/s; `results/b580/sessions/s1-aa/`):

| cell | parent | topic, no env | A/A | expected (`cells.csv`) | parent vs expected |
|---|---:|---:|---:|---:|---:|
| 1B 4w | 8641.35 | 8641.35 | 0.00 % | 8605.04 | +0.42 % |
| 1B 8da4w | 8865.80 | 8865.80 | 0.00 % | 8789.70 | +0.87 % |
| 3B 4w | 3436.24 | 3436.24 | 0.00 % | 3419.03 | +0.50 % |
| 3B 8da4w | 3524.96 | 3524.96 | 0.00 % | 3506.85 | +0.52 % |
| 8B 4w | 1725.36 | 1726.81 | +0.08 % | 1693.96 | +1.85 % |
| 8B 8da4w | 1845.05 | 1845.05 | 0.00 % | 1680.07 | **+9.82 %** |

A/A geomean +0.01 %; 60 timed runs, none rejected; next token parent vs topic SAME in all six cells on
`prompt_2048.txt`, `prompt_check.txt` and `r1304.txt`. The timer resolution is 1 ms = 0.43 % of a 1B prefill
(232 to 237 ms), which is why many readings are identical.

**8B 8da4w is 9.8 % above the expected value, outside the 5 % agreement fixed in `proposal.md`.** Cause, from
the old run table (`sarc-1.5-e2e-benchmark/results/runs_all.csv`): its five SARC runs of that cell were
1809, 1798, 1664, 1613 and 1680 tok/s, two fast and three slow, a 12 % spread, on a desktop that was in use
(the report's "Arc B580 variance" section says so); the published 1680 is the median of the slow group.
Tonight's ten runs of that cell lie between 1840 and 1847 (spread 0.3 %) with 0.00 % foreign engine time,
2 % above the old fast group, like the other five cells (+0.4 to +1.9 %). So the old median was depressed
by desktop activity, not a different kernel or configuration: the dispatched kernels in the parent control
are the shipped rows. Every comparison below is against the parent re-measured in the same session, and the
"original `dev/1.5` numbers" are quoted with this caveat.

## Per-cell numbers against the parent

### Candidate 0, `b580-refine0` (the B70's SDPA prefill kernels): ACCEPTED (reference-error rule, owner decision 2026-10-04)

Kernels: QK^T `sarc_sdpa_qk_coopmat_pk_t128x64k32g44s16m8nf`, attn*V
`sarc_sdpa_av_coopmat_xe2_t128x64k32g44s16m8` (head_dim 128) and `sarc_sdpa_av_coopmat_sweep_t64x64k32g44s16m8`
(head_dim 64), with the truncated SARC softmax: exactly the B70's `xe2-refine1`, reached through
`impl/sarc_dev/B580Sdpa.cpp` (device string `bmg g21`, `b580-*` profiles only).

Session `s2-c0`: pristine parent build (no environment) against build `topic1` with
`ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=b580-refine0`; tok/s, median of 5 valid runs per arm, arms
interleaved (`results/b580/sessions/s2-c0/`):

| cell | parent | candidate 0 | gain | B70 gain (`xe2-refine1`) | next token parent vs candidate (timed / real-text / unaligned) |
|---|---:|---:|---:|---:|---|
| 1B 4w | 8641.35 | 12962.00 | +50.00 % | +49.57 % | SAME / SAME / SAME |
| 1B 8da4w | 8865.80 | 13473.70 | +51.97 % | +49.55 % | SAME / SAME / SAME |
| 3B 4w | 3442.02 | 5171.72 | +50.25 % | +51.80 % | SAME / SAME / SAME |
| 3B 8da4w | 3524.96 | 5417.99 | +53.70 % | +55.60 % | SAME / SAME / SAME |
| 8B 4w | 1725.36 | 2306.31 | +33.67 % | +35.58 % | SAME / SAME / SAME |
| 8B 8da4w | 1843.38 | 2531.52 | +37.33 % | +40.07 % | **DIFFER / DIFFER** / SAME |

Geomean +45.95 % (B70: +46.86 %), every cell far outside the +-2 % band (A/A +0.01 %). 60 timed runs, none
rejected; foreign engine time 0.00 % in every run; median clocks 2717 to 2850 MHz (the candidate's 8B 4w runs
are power-limited at 2733 MHz against the parent's 2783).

**This is not a plain pass.** The gate (`gate_sdpa.sh`, `sessions/s2-c0/gate.txt`, 34 PASS lines) ends
`GATE_FAIL` on two lines that state one fact: the candidate's next token differs from the parent's for 8B 8da4w
on `prompt_2048.txt` and on `prompt_check.txt`. Everything else passes: SDPA correctness 12 passes x tiers
all / extended / full, 0 mismatches and `pairing=ok` in all 192 cases; unmodified `verify.sh` completed, 12 of 12
production-diff cases ALL PASSED, 28 of 28 numeric correctness cases, default vs tiled SAME on both prompts for
both schemes (the parent's `1b 8da4w unaligned DIFFER` becomes SAME), decode 31 tokens, every other status line
equal to the parent control; traces for 12 runs. The candidate replaces fp16-accumulating attention kernels by
fp32-accumulating ones, an arithmetic change, so it is decided by the owner's second decision of 2026-10-04
with the thresholds fixed there (`tools/decide.py` unchanged from the B70 campaign, `sessions/s2-c0/decision.txt`):

1. **Error against the fp32 CPU reference** (`tools/sdpa_ref.sh`, both arms with the `topic1` test binary on
   the same inputs; `results/b580/sdpa-error/c0-full.csv`): not larger than the parent's in every case.

   | case (S = 2048 unless noted) | parent rms / max | candidate rms / max |
   |---|---|---|
   | 1B head configuration | 1.27e-4 / 1.37e-3 | 2.76e-5 / 1.19e-3 |
   | 3B head configuration | 1.28e-4 / 1.33e-3 | 2.79e-5 / 1.15e-3 |
   | 8B head configuration | 1.28e-4 / 1.75e-3 | 2.76e-5 / 1.06e-3 |
   | 8B, S = 1024 at input_pos 1024 | 1.42e-4 / 1.27e-3 | 1.29e-5 / 1.24e-4 |

   The eight `extended` cases agree (`c0-extended.csv`): candidate rms 2.5e-5 to 6.6e-5 against 1.0e-4 to
   1.3e-4, maximum 2.5e-4 to 1.2e-3 against 6.9e-4 to 1.7e-3.
2. **Logits** (`tools/probe.sh`, a fresh process per prompt, four arms; `results/b580/probe/s2-c0/`). The probe
   reproduces the gate: of the 12 gate-prompt comparisons only the two 8B 8da4w ones differ.
   - `prompt_check.txt` (real text), 8B 8da4w: a three-way near-tie in the parent, logits 15.469 / 15.320 /
     15.000 for ids 45647 / 6062 / 70159 (top-2 margin 0.148, parent tiled identical); the candidate has
     15.172 / 15.172 / 15.242, so id 70159 leads by 0.070. KL(parent || candidate) = 0.042 nat.
   - `prompt_2048.txt` (2048 times " the"), 8B 8da4w: parent top-2 margin 0.336 (ids 247 / 118 at 11.664 /
     11.328); the candidate moves id 247 to 8.930 and id 53 from 8.125 to 11.328: KL 0.975 nat, largest logit
     change 6.06. Not a small move; it is on the degenerate prompt where every attention row averages 2048
     identical values and the parent's fp16 accumulation is at its worst. Reported in full, not explained away.
   - 35 real-text windows per cell, candidate default against parent default, and beside it parent tiled
     against parent default:

     | cell | top-1 differs (cand / parent-tiled) | mean KL nat (cand / parent-tiled) | max KL | max logit diff | perplexity ratio, 27 windows |
     |---|---|---|---|---|---|
     | 1B 4w | 1 / 0 of 35 | 3.0e-3 / 9.7e-4 | 0.033 / 0.010 | 0.80 / 0.75 | 1.011 / 0.994 |
     | 1B 8da4w | 1 / 0 | 7.5e-2 / 4.6e-2 | 0.91 / 0.33 | 5.08 / 5.17 | 0.913 / 0.933 |
     | 3B 4w | 1 / 0 | 7.7e-4 / 4.2e-4 | 0.008 / 0.005 | 0.83 / 0.61 | 1.018 / 1.014 |
     | 3B 8da4w | 1 / 1 | 2.5e-2 / 5.3e-2 | 0.29 / 1.14 | 3.62 / 3.10 | 0.976 / 0.995 |
     | 8B 4w | 0 / 0 | 7.1e-4 / 4.2e-4 | 0.007 / 0.006 | 0.70 / 0.61 | 0.993 / 1.005 |
     | 8B 8da4w | 0 / 1 | 1.4e-2 / 2.5e-2 | 0.25 / 0.31 | 2.73 / 2.92 | 1.011 / 0.909 |

3. **Gross-divergence check** (reject above 0.5 nat mean KL or top-1 differing on more than a third of the
   windows, in any cell): largest mean KL 0.075 nat, at most 1 of 35 windows differs. Passed.
4. Next-token items that differ, listed and not waived: 8B 8da4w on `prompt_2048.txt` and `prompt_check.txt`.
   Windows where the candidate's top-1 differs from the parent's: `w1280-gpl-384` (1B 4w), `w1536-gpl-0`
   (1B 8da4w, 3B 4w, 3B 8da4w); logits of all arms in `probe/s2-c0/differing.md`.

**Equal to the B70 to the last digit.** The reference-error table and the probe summary (top-1 counts, KL,
logit differences, perplexities; first 12 columns of `summary.csv`) are identical to the B70 campaign's
`xe2-refine1` files: on the same kernels the two cards compute the same values, for the parent arm too. The
B70's acceptance evidence therefore carries over, and was nevertheless re-measured here.

Decode (not part of the prefill gate; `sessions/s2-c0/decode/summary.csv`, 5 runs per arm, 31 tokens after a
2048-token prefill): candidate / parent 0.994 to 1.003 in the six cells, no direction (the B70 saw 0.7 to
2.0 % slower everywhere).

Where the gain comes from (warm ETDump of both arms, ms per prefill, `sessions/s2-c0/trace/families.csv`):

| cell | arm | total | linear GEMM | QK^T | attn*V | softmax | other |
|---|---|---:|---:|---:|---:|---:|---:|
| 1B 4w | parent | 235.2 | 88.0 | 40.7 | 48.2 | 22.4 | 35.9 |
| 1B 4w | candidate 0 | 156.9 | 88.3 | 7.0 | 7.9 | 17.5 | 36.2 |
| 3B 4w | parent | 593.1 | 256.2 | 102.0 | 124.0 | 29.9 | 81.0 |
| 3B 4w | candidate 0 | 393.2 | 259.7 | 14.0 | 14.7 | 22.9 | 81.9 |
| 8B 4w | parent | 1180.5 | 652.3 | 153.7 | 188.0 | 45.3 | 141.2 |
| 8B 4w | candidate 0 | 885.1 | 664.7 | 21.4 | 21.8 | 34.7 | 142.5 |
| 8B 8da4w | parent | 1104.9 | 554.4 | 150.1 | 183.5 | 44.7 | 172.2 |
| 8B 8da4w | candidate 0 | 803.6 | 554.5 | 20.8 | 21.0 | 34.4 | 172.9 |

QK^T and attn*V together go from 38 % (1B) and 29 % (8B) of the prefill to 9 % and 5 %. Kernel times per layer
(`sessions/s2-c0/sdpa-correctness/perf-*.log`, S = 2048, ms): 8B QK^T 4.69 -> 0.62, attn*V 5.73 -> 0.66,
softmax 1.40 -> 1.08; 3B 3.62 -> 0.47, 4.40 -> 0.52, 1.06 -> 0.82; 1B 2.53 -> 0.42, 3.02 -> 0.49, 1.41 -> 1.08.
The B580 runs these kernels 1.35 to 1.5 times slower than the B70 (0.41 / 0.48 / 0.80 on 8B), about its
ratio of roofs. After candidate 0 the linear kernels are 56 to 75 % of the prefill and the truncated softmax
(4 to 12 %) is larger than QK^T and attn*V together in the 1B cells.

### Candidate 1, `b580-refine1` (candidate 0 + 8da4w linear tile `xe2bt_t128x128k64g84s16m8`): GATE_PASS, bit-identical

Kernel: `sarc_dev_linear_dq8ca_coopmat_zpg_xe2bt_t128x128k64g84s16m8` (the B70 campaign's texel-wise family:
128 x 128 tile, K = 64 per chunk, 512 threads, every thread stages one A block and one packed-weight texel per
chunk) for every 8da4w linear shape, instead of the shipped `zpg_t256x64k32g48s16m8`. Same values and MMA
order as the release kernel; the 4w cells run the same kernels in both arms.

Session `s3-c1`: build `topic2` in both arms, parent arm = candidate 0 (`b580-refine0`), candidate arm
`ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=b580-refine1`; tok/s, median of 5 valid runs per arm, arms
interleaved (`results/b580/sessions/s3-c1/`):

| cell | candidate 0 | candidate 1 | gain | next token (timed / real-text / unaligned) |
|---|---:|---:|---:|---|
| 1B 4w | 12720.50 | 12720.50 | 0.00 % | SAME / SAME / SAME |
| 1B 8da4w | 13212.90 | 14840.60 | +12.32 % | SAME / SAME / SAME |
| 3B 4w | 5056.79 | 5069.31 | +0.25 % | SAME / SAME / SAME |
| 3B 8da4w | 5291.99 | 6224.92 | +17.63 % | SAME / SAME / SAME |
| 8B 4w | 2248.08 | 2245.61 | -0.11 % | SAME / SAME / SAME |
| 8B 8da4w | 2491.48 | 2934.10 | +17.77 % | SAME / SAME / SAME |

Geomean **+7.67 %**; the three 8da4w cells are far outside the +-2 % band, the three 4w cells are inside it, as
they must be. 60 timed runs, none rejected. The owner was using the desktop during this session: foreign
engine time 1.2 to 1.4 % per cell (median), 2.3 % at most, under the 5 % limit; arm spreads up to 2.2 % (0.4 %
on the idle desktop), and candidate 0 reads 1.6 to 2.5 % lower here than in `s2-c0`. Both arms were interleaved
under the same conditions.

Gate (`gate.sh`, `sessions/s3-c1/gate.txt`): `GATE_PASS`, 33 PASS lines. Unmodified `verify.sh` with the
candidate environment: 12 of 12 production-diff cases ALL PASSED (non-zero zero points for 8da4w), 28 of 28
numeric correctness cases, dispatched kernel `..._xe2bt_t128x128k64g84s16m8` for all 8da4w cases in buffer and
texture3d, default vs tiled SAME on both prompts for both schemes, decode 31 tokens, status equal to the
parent control. Next token candidate 0 vs candidate 1 SAME in all six cells on all three prompts. No SDPA
kernel changes against candidate 0, so the SDPA passes of `s2-c0` stand.

Bit-identity (`probe.sh s3-c1`, arms P and C; `results/b580/probe/s3-c1/`, `decision.txt`): the whole logits
vector of candidate 1 equals candidate 0's bit for bit on all 35 real-text windows and both gate prompts, in
all six cells. A staging change, as claimed; no arithmetic rule is needed.

Where the gain comes from (warm ETDump, ms per prefill, `sessions/s3-c1/trace/families.csv`): the linear
GEMM family and nothing else.

| cell | arm | total | linear GEMM | everything else |
|---|---|---:|---:|---:|
| 1B 8da4w | candidate 0 | 154.6 | 75.2 | 79.3 |
| 1B 8da4w | candidate 1 | 134.6 | 57.6 | 76.9 |
| 3B 8da4w | candidate 0 | 385.3 | 228.1 | 157.2 |
| 3B 8da4w | candidate 1 | 323.7 | 169.2 | 154.5 |
| 8B 8da4w | candidate 0 | 809.3 | 558.8 | 250.5 |
| 8B 8da4w | candidate 1 | 688.5 | 437.4 | 251.1 |

Linear GEMM 1.31x (1B), 1.35x (3B), 1.28x (8B) faster in the model, as in the kernel screen (1.29x, 1.36x,
1.28x). In the kernel the gain is the fetch phase: each packed-weight texel is fetched once per chunk instead
of 8 times, with one barrier per 64 K instead of per 32.

Against the B70: the same tile is the B70's best at kernel level (1.26x there); at the fork point the B70
campaign had not gated it end to end.

### Candidate 2, `b580-refine2` (candidate 1 + 4w band-drain tile): GATE_PASS, bit-identical, no gain, not adopted

Kernel: `sarc_linear_q4gsw_coopmat_sweep_t128x128k16g44s16m8flib` (the shipped 4w tile and body with the
texture3d drain staged one band at a time, 18.4 instead of 24.6 KiB of shared memory) for the 4w linear
shapes. Session `s4-c2`: build `topic2` in both arms, parent arm = candidate 1 (`b580-refine1`); median of 5
valid runs per arm (`results/b580/sessions/s4-c2/`):

| cell | candidate 1 | candidate 2 | gain |
|---|---:|---:|---:|
| 1B 4w | 13044.60 | 12962.00 | -0.63 % |
| 1B 8da4w | 15170.40 | 15170.40 | 0.00 % |
| 3B 4w | 5171.72 | 5158.69 | -0.25 % |
| 3B 8da4w | 6380.06 | 6380.06 | 0.00 % |
| 8B 4w | 2290.83 | 2301.12 | +0.45 % |
| 8B 8da4w | 2994.15 | 2994.15 | 0.00 % |

Geomean **-0.07 %**, every cell inside the +-2 % band: noise, not a gain. `GATE_PASS` (33 PASS lines; the
band-drain kernel is the one dispatched for texture3d), next token SAME everywhere, logits bit-identical to
candidate 1 on all 35 windows and both gate prompts in all six cells (`results/b580/probe/s4-c2/`). 60 timed
runs, none rejected, foreign engine time 0.00 %. This confirms end to end what the kernel screen said after
its second round, and it is the B70's result (1.004x at kernel level there). The shipped 4w tile stays;
candidate 2 is recorded as the **first candidate under 2 %**.

## Linear kernels on this card (screens and phase timing; `results/b580/screens/`, `results/b580/phases/`)

Phase timing of the shipped tiles (shader clock, 1B shapes, share of a wave), B580 against B70:

| kernel | card | barrier | fetch | MMA | LDS store | prologue + epilog | drain + write |
|---|---|---:|---:|---:|---:|---:|---:|
| 4w `t128x128k16g44s16m8fli` | B580 | 19 to 23 % | 22 to 25 % | 40 to 41 % | 13 to 14 % | 0 to 1 % | 1 % |
| | B70 | 22 to 23 % | 22 to 26 % | 37 to 40 % | 13 to 15 % | 1 % | 1 % |
| 8da4w zpg `t256x64k32g48s16m8` | B580 | 18 to 20 % | 34 to 39 % | 20 to 23 % | 14 to 16 % | 5 to 10 % | 1 to 2 % |
| | B70 | 18 to 20 % | 32 to 36 % | 22 to 23 % | 15 to 16 % | 5 to 9 % | 2 % |

The two cards spend a wave the same way; the B580 fetches slightly longer in the 8da4w kernel.

- **8da4w** (`screen2-8da4w`, quiet card): `xe2bt_t128x128k64g84s16m8` is the fastest tile on all 12 shapes in
  both rounds, 1.19x to 1.45x per shape, 1.31x weighted (B70: the same tile, 1.26x). Same choice as the B70,
  3 to 4 points more gain. It is candidate 1.
- **4w** (`screen3-4w`, desktop in use): no tile beats the shipped one in both rounds. The band-drain tiles
  are 1.00x against the undisturbed base round (B70: 1.004x); a 1.2x reading after round 1 was a slow base
  run. Same result as the B70: the shipped 4w tile stays.

## SDPA kernels on this card (`screen1-sdpa`, 59 profiles, 2 rounds agreeing within 0.5 %)

Per layer, S = 2048, us (tables in `results/b580/screens/README.md`). QK^T: `xe2c_t128x64k32g44s16m8nf` 572
(8B) / 440 (3B) against candidate 0's `pk` kernel 619 / 472, i.e. 7.6 % and 6.8 % faster; on 1B (head_dim 64)
the best kernel is 2.7 % faster than `pk`, under the 3 % rule, so `pk` stays there. attn*V: candidate 0's
kernels are the best for both head dimensions. Every ranking equals the B70's. Candidate 3 = candidate 1 with
that QK^T kernel for head_dim 128; QK^T is 2.4 to 3.1 % of the prefill on 3B / 8B, so the expected end-to-end
effect is about 0.2 %.

## Roofs (re-measured, not the old evidence)

igpu-roofline plan `fast`, device `b580`, 2026-10-05 07:40 to 08:03 UTC, driver Mesa 26.2.3 (109060099),
runner `810e098c8abb`, clocks not pinned, sentinel `ok` at all 34 checkpoints, every roof confirmed by 3
repeats (`results/b580/roofline/b580-fast-20261005/REPORT.md`; artifacts `roofline/b580-fast-20261005/`).
The tool is the fleet copy already on this host (`~/.cache/igpu-roofline/fleet-fast-20260926`), run unchanged
from a copy in the artifact directory, with the campaign's venv. The desktop was quiet during the run.

| roof | B580 (this run) | B70 (its campaign's run) | B580 / B70 |
|---|---:|---:|---:|
| matrix fp16 | 112.9 TFLOP/s | 173.3 | 0.65 |
| matrix fp16 -> fp32 | 115.7 TFLOP/s | 179.9 | 0.64 |
| matrix int8 | 231.4 TOP/s | 359.9 | 0.64 |
| matrix fp16 fed from shared memory | 109.7 TFLOP/s | 168.4 | 0.65 |
| matrix int8 fed from shared memory | 206.4 TOP/s | 323.3 | 0.64 |
| global read / write / copy | 465 / 403 / 408 GB/s | 603 / 509 / 532 | 0.77 / 0.79 / 0.77 |

Linear kernels in the model against these roofs (time-weighted over the prefill GEMMs of the warm ETDump,
`tools/roof_util.py`):

| kernel | 1B | 3B | 8B | B70 |
|---|---:|---:|---:|---:|
| 4w shipped tile | 45.1 TFLOP/s = 40.0 % | 44.5 = 39.4 % | 43.0 = 38.1 % | 36.5 to 38.9 % |
| 8da4w shipped tile (parent, candidate 0) | 54.0 TOP/s = 23.3 % | 51.6 = 22.3 % | 51.6 = 22.3 % | 22.8 to 24.4 % |
| 8da4w `xe2bt_t128x128k64g84s16m8` (candidate 1) | 69.2 TOP/s = 29.9 % | 68.2 = 29.5 % | 65.4 = 28.2 % | not gated there |

(4w and shipped 8da4w rows from `s2-c0`, the candidate 1 row from `s3-c1`.) The B580 has 0.64 of the B70's
matrix roofs but 0.77 of its memory rates; both cards sit at the same percentage of their matrix roofs.

## Next

1. Candidate 3 gate. If it is under 2 % too, the campaign is done by the rule (two consecutive candidates
   under 2 %).
2. A quiet rerun of the 4w screen.
3. Final session of the recommended profile against the pristine parent, proposal and report.

## Blocking

Nothing. The desktop is in use now; if timed runs keep being rejected for foreign engine time the sessions
will take longer or end incomplete, and this file will say so.
