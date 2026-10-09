# sarc-1.5-b70-fused-port: status

**2026-10-09 10:45 UTC (host clock) — finished; nothing is running or queued. `b70-fused1` (the B580's fused
attention kernel, unchanged, on top of `xe2-refine5`): `GATE_PASS` as a plain pass on the build of the committed
head, **+8.26 % geomean over the tuned parent** (+8.34 % on the first gate; the B580 measured +9.18 %) and
**+72.87 % geomean over the first campaign's pristine parent**. One candidate, as a confirmation closes (the B580's
candidate 2 was not adopted there, so none here). One negative finding: decode is 0 to 2.5 % slower. Summary and
tables: `proposal.md`.**

Branch `topic/b70-fused-port` (from `origin/topic/xe2-prefill-refine` at `5617714b0`). Parent of every comparison:
`5617714b0` with `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=xe2-refine5`. Host `fedora-gpu-eval`, card `b70-0`
only (guest PCI `0000:01:00.0`, `ETVK_DEVICE_INDEX=0`, lock `868023e2-0000-0000-0100-000000000000`); the second card
is not used. Artifacts `/home/doremy/hmz-sarc-b70-fused/.artifacts/`.

As found: ANV Mesa 26.2.3 (the first campaign's version), kernel 7.2.9-200.fc44 (host up since 2026-10-08 17:06
UTC; the first campaign's journal did not record its kernel version in `STATUS.md`, so a kernel change since its
`s12-final5` cannot be excluded: the baseline comparison below is the check). GT frequency policy `min_freq` 1200,
`max_freq` 2800, `rp0` 2800, `rpe` 400, `rpn` 400 MHz, `power_profile` `[base] power_saving`; nothing changed.
Device limits read with `vulkaninfo`: `maxComputeSharedMemorySize` 49152, subgroup size 16 to 32 (default 32),
`computeFullSubgroups` true, `maxComputeWorkgroupSubgroups` 64: the same as the B580's record. RAM 46 GB; the six
model files were not in the page cache after the reboot, so decision D5 is applied (each model read before its
cell, both arms alike, resident share recorded per run in `runs.csv`, column `cached_pct`).

## Running now

Nothing. `tools/chain3.sh b70-fused1 6ac44c483` ended `CHAIN3_DONE` at 10:32 UTC. No `HOLD` was set and no
foreign GPU process was seen during the campaign (`.artifacts/logs/foreign.log` does not exist; the one wait
recorded anywhere was a tool self-test waiting on this campaign's own build container at 05:47 UTC, see
`proposal.md`).

## Closing (chain 3, 08:39 to 10:32 UTC), on the build of the committed head

Build `topic2` = an export of `6ac44c483` (the branch head when the closing started; no local patch; the commits
after it touch only `openspec/`): shipped-SPIR-V golden PASS (53 variants), `SPV_IDENTITY_OK` (1525 parent
shaders unchanged, 47 fused and copy-pass shaders identical to the B580 build `topic6`).

**Full gate of the final stack, session `s3-final`** (build `parent` with the parent environment against `topic2`
with `b70-fused1`; 09:23 to 09:30 UTC for the timed part): `GATE_PASS`, 35 PASS lines, no FAIL line; SDPA tiers
`all` / `extended` / `full` 12 passes each, rc 0, 0 mismatches, `pairing=ok`, 192 of 192 case runs served by the
fused kernel; `verify.sh` against `s0-parent-verify`: `VERIFY_SAME` (34 lines); next token SAME in all six cells
on the three prompts; probe `PROBE_CHECK_OK`; `decide.py` `GATE_PASS`. Extra tiers as in `s2-c1` (`peaked` 60 of
60, `fused` 24 of 60 served, 0 mismatches). Reference error `final-ref`: identical to `c1-ref` in every digit.
Tok/s, median of 7 valid runs per arm (recomputed from `runs.csv`; 84 timed runs, all valid, foreign engine time
0.00 %, lowest median clock 2533 MHz):

| cell | parent | `b70-fused1` | gain | gain in `s2-c1` (build `topic1`) | spread parent / cand |
|---|---:|---:|---:|---:|---|
| 1B 4w | 17964.90 | 20686.90 | **+15.15 %** | +15.15 % | 1.75 / 3.00 % |
| 1B 8da4w | 20480.00 | 24381.00 | **+19.05 %** | +17.86 % | 1.01 / 5.75 % |
| 3B 4w | 7529.41 | 7968.87 | **+5.84 %** | +6.25 % | 0.73 / 2.32 % |
| 3B 8da4w | 9570.09 | 9990.24 | **+4.39 %** | +4.39 % | 2.75 / 3.30 % |
| 8B 4w | 3379.54 | 3482.99 | **+3.06 %** | +4.12 % | 0.82 / 1.20 % |
| 8B 8da4w | 4481.40 | 4623.02 | **+3.16 %** | +3.16 % | 1.09 / 1.34 % |

Geomean **+8.26 %** (`s2-c1`: +8.34 %). Decode in this session: 1B -2.2 / 0.0 %, 3B -2.3 / -1.1 %, 8B -1.1 /
-0.9 % (4w / 8da4w).

**Hook condition on the final build:** `test_sarc_select` release tables identical for the parent and `topic2`
exports (1240 checks, 31 rows; dev zone 1559 / 35 against 1561 / 37); the four executables are kept in
`.artifacts/raw/final-select/` for the reviewer (`test_sarc_select-{rel,dev}-{parent,topic2}`; run them from the
matching `.artifacts/src/<tag>/executorch` with the yaml list of `tools/select_check.sh`). `verify.sh` with no
environment on `topic2` (`s0-topic2-noenv`) against `s0-parent-noenv`: `VERIFY_SAME` (34 lines).

**Against the first campaign's pristine parent, session `s4-pristine`** (10:23 to 10:31 UTC; build `pristine` = the
binaries of its `s12-final5`, no profile, no `ET_VK_SARC_UNVERIFIED`, against `topic2` with `b70-fused1`; 7 valid
runs per arm, 84 timed runs, all valid):

| cell | pristine parent | `b70-fused1` | total gain |
|---|---:|---:|---:|
| 1B 4w | 11702.90 | 20686.90 | +76.77 % |
| 1B 8da4w | 12337.30 | 24381.00 | +97.62 % |
| 3B 4w | 4864.61 | 8000.00 | +64.45 % |
| 3B 8da4w | 5251.28 | 9990.24 | +90.24 % |
| 8B 4w | 2435.20 | 3512.86 | +44.25 % |
| 8B 8da4w | 2730.67 | 4623.02 | +69.30 % |

Geomean **+72.87 %**. The session's status is `E2E5_INCOMPLETE` for one reason: next token of 8B 8da4w against
the pristine parent DIFFERs on `prompt_2048.txt` and `prompt_check.txt` (SAME on `r1304.txt`; SAME everywhere in
the other five cells). Those are the two items the first campaign's candidate 1 was accepted with under the
reference-error rule; against the tuned parent, which is this campaign's gate, every item is SAME. This session
is a timing comparison, not a gate. Traces of 1B 4w and 8B 4w (attention, ms per prefill): pristine 80.8 / 284.3,
tuned parent 23.7 / 56.5, `b70-fused1` 7.5 / 32.1 (`stage/s4-pristine/trace/attention.csv`, `s3-final` likewise).

`sarc/tools/check.sh --no-build`: `check.sh: PASS` (output in `results/b70/d4/check-no-build.txt`, and again at
the pushed head, below).

## Candidate 1, `b70-fused1`: GATE_PASS, +8.34 % geomean (session `s2-c1`, 07:10 to 08:39 UTC)

Parent = build `parent` (`5617714b0`) with `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=xe2-refine5`; candidate
= build `topic1` (`8ba3607be`) with `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=b70-fused1`. Tok/s, median of
the first 7 valid runs per arm, arms interleaved; recomputed from `stage/s2-c1/raw/runs.csv`:

| cell | parent | candidate 1 | gain | spread parent / cand | next token (2048 / real-text / 1792-token prompt) |
|---|---:|---:|---:|---|---|
| 1B 4w | 17964.90 | 20686.90 | **+15.15 %** | 1.75 / 5.83 % | SAME / SAME / SAME |
| 1B 8da4w | 20686.90 | 24381.00 | **+17.86 %** | 1.00 / 1.20 % | SAME / SAME / SAME |
| 3B 4w | 7529.41 | 8000.00 | **+6.25 %** | 0.37 / 0.39 % | SAME / SAME / SAME |
| 3B 8da4w | 9570.09 | 9990.24 | **+4.39 %** | 1.40 / 0.49 % | SAME / SAME / SAME |
| 8B 4w | 3379.54 | 3518.90 | **+4.12 %** | 1.31 / 1.38 % | SAME / SAME / SAME |
| 8B 8da4w | 4481.40 | 4623.02 | **+3.16 %** | 0.66 / 0.68 % | SAME / SAME / SAME |

Geomean **+8.34 %**, every cell outside the +-2 % band (A/A -0.06 %). 84 timed runs, all valid: foreign engine
time 0.00 % in every run, at least one guard poll and 8 or more clock samples per run, lowest median clock 2533
MHz (`CLKMIN` 2457), no thermal throttle reason. Of the 24 untimed next-token runs two (8B 4w on
`prompt_check.txt`, both arms, 2450 and 2433 MHz) carry the reason `clock_low`; they are token comparisons, for
which the clock does not matter (`e2e5.sh toklog`), and are not in any median.

Gate (`gate_sdpa.sh`, `stage/s2-c1/gate.txt`): `GATE_PASS`, 35 PASS lines, no FAIL line. SDPA correctness 12
passes x tiers `all` / `extended` / `full` (4 / 8 / 4 cases): every pass rc 0, 0 mismatches and `pairing=ok` in all
192 case runs, all 192 served by the fused kernel alone; hashes of the test binary, runner library and runner in
`sdpa-correctness/hashes.txt`, statuses in `rc.csv`. Unmodified `verify.sh` with the candidate environment:
identical to `s0-parent-verify` line by line with the rates removed (`verify_diff.txt`: `VERIFY_SAME`, 34 lines).
Logits probe `PROBE_CHECK_OK`; `decide.py`: `GATE_PASS` (plain pass: no next-token item differs, so neither the
near-tie nor the reference-error decision is used).

Reference error (the candidate changes the arithmetic; `results/b70/sdpa-error/c1-ref/`, both arms on the staged
build, 0 mismatches everywhere); rms / maximum absolute error against the fp32 CPU reference:

| tier | case | parent's three kernels | `b70-fused1` | not larger (rms / max) |
|---|---|---|---|---|
| `full` | 1B heads, S = 2048 | 2.76e-5 / 1.19e-3 | 2.05e-5 / 7.2e-4 | yes / yes |
| `full` | 3B heads, S = 2048 | 2.79e-5 / 1.15e-3 | 2.05e-5 / 7.1e-4 | yes / yes |
| `full` | 8B heads, S = 2048 | 2.76e-5 / 1.06e-3 | 2.02e-5 / 7.9e-4 | yes / yes |
| `full` | 8B heads, S = 1024 at input_pos 1024 | 1.29e-5 / 1.24e-4 | 8.96e-6 / 6.9e-5 | yes / yes |
| `extended` | 8 cases | | | yes / yes in all 8 |
| `peaked` | `peaked_tiny_gqa_s256` | 4.06e-4 / 2.32e-3 | 3.77e-4 / **2.50e-3** | yes / **no** |
| `peaked` | the other 4 cases | | | yes / yes |

The same numbers as on the B580 (same kernel, same inputs), including the one synthetic `peaked` case whose
maximum error is 8 % above the parent's while its rms is 7 % below; `peaked` is reported, not a gate item
(`thresholds.txt`, fixed before the run). Extra tiers, 12 passes each: `peaked` 60 of 60 case runs served by the
fused kernel with 0 mismatches; `fused` 60 case runs with 0 mismatches, 24 of them served by the fused kernel
(`fused_s64`, `fused_s192_pos64`; the other three cases have S or head_dim the chosen blocks do not fit and
fail by the tier's design, as they would on the B580).

Where the gain comes from (warm ETDump of both arms, ms per 2048-token prefill, attention kernels by name,
`stage/s2-c1/trace/attention.csv`):

| cell | arm | dispatch total | QK^T | softmax | attn*V | fused kernel | K/V copy pass | attention |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1B 4w | parent | 105.9 | 4.8 | 12.7 | 6.1 | | | 23.7 |
| 1B 4w | candidate 1 | 89.8 | | | | 7.3 | 0.25 | 7.5 |
| 1B 8da4w | parent | 91.7 | 4.8 | 12.6 | 6.1 | | | 23.5 |
| 1B 8da4w | candidate 1 | 75.6 | | | | 7.1 | 0.22 | 7.3 |
| 3B 4w | parent | 261.7 | 9.3 | 17.0 | 10.9 | | | 37.2 |
| 3B 4w | candidate 1 | 246.6 | | | | 20.8 | 0.67 | 21.4 |
| 3B 8da4w | parent | 206.4 | 8.9 | 16.6 | 10.6 | | | 36.2 |
| 3B 8da4w | candidate 1 | 190.3 | | | | 19.5 | 0.64 | 20.1 |
| 8B 4w | parent | 595.6 | 14.4 | 25.9 | 16.2 | | | 56.5 |
| 8B 4w | candidate 1 | 571.7 | | | | 31.5 | 0.68 | 32.2 |
| 8B 8da4w | parent | 448.5 | 13.4 | 25.0 | 15.3 | | | 53.8 |
| 8B 8da4w | candidate 1 | 424.2 | | | | 28.9 | 0.66 | 29.6 |

Attention goes from 23.7 to 7.5 ms on 1B (-68 %), 37.2 to 21.4 ms on 3B (-42 %), 56.5 to 32.2 ms on 8B (-43 %);
the dispatch total falls by the same 16 / 15 / 24 ms, so nothing else moved.

**Negative finding, decode** (`stage/s2-c1/decode/summary.csv`, 5 runs per arm, 32 new tokens after the
2048-token prompt): the candidate decodes 0.8 to 2.5 % slower than the parent in every cell (1B 4w 100.65 ->
98.10 tok/s, -2.5 %; 1B 8da4w -1.2 %, 3B -1.3 / -1.5 %, 8B -0.8 / -1.0 %). The fused kernel does not serve a
decode call (S = 1); the three kernels of `xe2-refine5` run there as before. Not located in this campaign (no
decode trace was taken); the likely place is the per-step cost of the two extra graph nodes per layer that
dispatch nothing in decode, which is an inference, not a measurement. Decode is not a gate item and not this
campaign's target; it is reported so that a promotion does not overlook it.

## Candidate 2: none

Task section 6.4: read the B580 campaign's `STATUS.md` when candidate 1 here is gated. Read at 08:00 UTC (its
working copy, head `f613e3ed4`): "Candidate 2 (`b580-fused2` ...): `GATE_PASS` on the committed head (session
`s3b-c2`), -0.13 % geomean over candidate 1, every cell inside the +-2 % band: no gain, not adopted". Inside the
band, so there is no candidate 2 here and the campaign closes with candidate 1. (That is a citation of another
campaign, branch `topic/b580-fused-port` at `f613e3ed4`, not a number of this one. The pin for sources stays
`cea76c634`.)

## Chain 1 (05:46 to 07:09 UTC): results

Builds, each from an export of one commit with its submodules (31 trees), shipped SPIR-V golden PASS (53 variants):
`parent` = `5617714b0`, `topic1` = `8ba3607be` (hooks, test blocks, kernel files, `b70-fused` profiles).
`pristine` is a copy of the first campaign's build `parent` (`6a7cc8cc6`; `llama_main` sha256 `45e3242e3881...`,
the binary its `s12-final5` timed), not rebuilt.

**SPIR-V identity** (`tools/spv_identity.sh`, `results/b70/identity/`): `SPV_IDENTITY_OK`. All 1525 shaders of the
parent build are byte-identical in `topic1` (which has 1572); all 47 `sarc_dev_b580_sdpa_fused*` / `..._kvt*`
shaders of `topic1` are byte-identical to the B580 campaign's build `topic6` (`247d08851`, the build its
candidate 1 was gated on).

**Hook condition D4 (nothing selected, nothing changed): met.**
1. `test_sarc_select` built from the parent export and from the `topic1` export: release tables identical output
   (1240 checks, 31 rows, both); with the dev zone and `ET_VK_SARC_UNVERIFIED=1` 1559 checks / 35 rows (parent) and
   1561 / 37 (topic: the two `b70-*` base rows), 215 candidates both. Executables and outputs kept in
   `.artifacts/raw/d4/` (outputs and hashes in `results/b70/d4/`).
2. `spirv_golden.py` PASS, 53 shipped variants, on `parent` and on `topic1`.
3. Unmodified `verify.sh` with no environment on `topic1` (`s0-topic1-noenv`) against the parent's
   (`s0-parent-noenv`): `VERIFY_SAME`, 34 lines, rates removed, kernel names included
   (`stage/s0-topic1-noenv/verify_diff.txt`).

**Parent snapshot `s0-parent-verify`** (parent environment `xe2-refine5`): `CONTROL_RECORDED`, one device-status
item (`correctness rc=1` with 28 of 28 numeric and 4 of 4 rank-3 cases PASSED); SDPA tiers 4 / 8 / 4 cases with 0
mismatches on the parent's three kernels. The two no-environment snapshots show the five status items the first
campaign recorded for pristine `dev/1.5` on this card.

**Baseline and A/A, session `s1-aa`** (06:44 to 06:56 UTC): build `parent` against build `topic1`, both with the
parent environment; median of the first 5 valid runs per arm, arms interleaved; 60 timed runs, none rejected;
tok/s (recomputed from `runs.csv`):

| cell | parent | topic, same environment | A/A | expected (`s12-final5`) | parent vs expected | spread parent / topic |
|---|---:|---:|---:|---:|---:|---|
| 1B 4w | 17964.90 | 17964.90 | 0.00 % | 17964.90 | 0.00 % | 0.87 / 1.75 % |
| 1B 8da4w | 20686.90 | 20686.90 | 0.00 % | 20686.90 | 0.00 % | 7.48 / 1.00 % |
| 3B 4w | 7529.41 | 7529.41 | 0.00 % | 7529.41 | 0.00 % | 0.00 / 0.37 % |
| 3B 8da4w | 9570.09 | 9570.09 | 0.00 % | 9570.09 | 0.00 % | 2.75 / 2.30 % |
| 8B 4w | 3385.12 | 3379.54 | -0.16 % | 3379.54 | +0.17 % | 1.49 / 0.82 % |
| 8B 8da4w | 4491.23 | 4481.40 | -0.22 % | 4481.40 | +0.22 % | 0.87 / 0.44 % |

A/A geomean -0.06 %; baseline within 0.22 % (limit 3 %); next token SAME in all six cells on the three prompts.
Equal medians are equal millisecond counts: the runner's timer has a 1 ms step (114 ms for 1B 4w, 0.9 % a step).
Per run: foreign engine time 0.00 % in all 60, at least one guard poll while the runner executed, 9 to 45 clock
samples in the prefill window, model file 100 % resident before every run, throttle reasons in the samples `none`
and `pl2` only. Calibration (`tools/thresholds.txt`, dated block, committed `4172bc183` before candidate 1 was
timed): `CLKMIN` 2457 MHz, idle 58 C, **7 repeats** (the parent arm of 1B 8da4w spread 7.48 %: one run of five
read 19140 tok/s, 107 ms against 99 ms).

**The one kernel screen, `screen1-select`** (06:56 to 07:09 UTC, build `topic1`, 3 rounds, cooled before every
run; kernel time per layer at S = 2048 in us, fused kernel + copy pass, for the parent QK^T + softmax + attn*V;
each round's value; `results/b70/screens/screen1-select{,-runs}.csv`):

| head_dim | profile | 1B | 3B | 8B | vs the parent's three kernels |
|---|---|---|---|---|---|
| | parent `xe2-refine5` | 1485 / 1481 / 1481 | 1284 / 1286 / 1285 | 1685 / 1697 / 1691 | 1.00x |
| 64 | `d64_t32x32s32m8ro` (the 780M's) | 4255 / 4281 / 4142 | | | 0.35x |
| 64 | **`d64_t16x64s16m8g4roj`** (incumbent, the B580's) | **486 / 483 / 487** | | | **3.05x** |
| 64 | `d64_t16x64s16m8g4oj` | 553 / 551 / 550 | | | 2.69x |
| 128 | `d128_t16x64s32m8ro` (the 780M's) | | 9569 / 9606 / 9528 | 12577 / 12565 / 12604 | 0.13x |
| 128 | **`d128_t16x128s16m8g8oj`** (incumbent, the B580's) | | **750 / 749 / 748** | **969 / 966 / 967** | **1.72x / 1.75x** |
| 128 | `d128_t16x64s16m8g4oj` | | 824 / 813 / 819 | 1047 / 1047 / 1043 | 1.57x / 1.61x |

No screened variant is faster than the incumbent of its head_dim in any round (the nearest is 8 to 14 % slower),
so by the `kernel_screen` rule **`b70-fused1` is the B580's pair**, as committed. Same ranking as the B580's
screen 5 in every row.

## What was brought from the B580, pinned

Source: `/mnt/linux-share/hmz-campaigns/b580-fused/executorch`, branch `topic/b580-fused-port`, **pinned at
`cea76c634`** (fetched as `refs/remotes/b580/topic/b580-fused-port`; its head is not followed).

| commit here | what | source |
|---|---|---|
| `cfa31c1d2` | release-zone hook D4.1, softmax variant name | cherry-pick of `fab9606c3` (B580 branch; from `b969e8f1c2`) |
| `cbbe36e0c` | release-zone hook D4.3, entry point of the fused attention node | cherry-pick of `0ffc84a2d` (B580 branch; from `1c8861aa7e`) |
| `087d4c4a9` | `test_llama_microbench.cpp`: insert-only `4070ti-fused` blocks (98 added lines, 0 removed) | `origin/topic/4070ti-fused-port`, `ed8b5af91`, unchanged |
| `7d7877980` | `sarc_dev_b580_sdpa_fused.{glsl,yaml}`, `sarc_dev_b580_sdpa_kvt.{glsl,yaml}`, `impl/sarc_dev/b580/SdpaB580Fused.cpp` | `cea76c634`, files unchanged |
| `dfbaecca1` | `impl/sarc_dev/B70Sdpa.cpp` (base rows for `b70-*` profile names) and the `b70-fused` blocks of `Overrides.cpp` | new here |

Notes on these choices:

- The fused hook was cherry-picked alone first and conflicted in `Select.h` (its context is the `softmax_variant`
  field), so both hooks are carried exactly as the B580 branch carries them; the release zone of this branch then
  differs from the parent by the same 57 added lines as the B580's. Nothing here sets `softmax_variant`.
- The test blocks are the 4070 Ti port's, not the B580's: the B580 changed lines of the shared test file in place
  (task section 4.3 forbids that here). Consequence: the `fused` tier has five cases here (the 4070 Ti's adds
  `fused_s32`), four on the B580; that tier is reported, not a gate item.
- The selector is the B580's file, reused unchanged: it is device-neutral for Intel (it asks only for active SDPA
  rows and for the variant list of the profile; no device string). The two functions it calls
  (`sdpa_fused_variants_b580`, `register_sdpa_fused_b580`) are defined in the `b70-fused` block here. A merge with
  the B580 branch will meet two definitions of them and must join the two profile tables; that is deliberate
  (two silent registrations on one hook would be worse).
- The fused yaml has no generator (`gen_b580.py` does not mention it); nothing was generated here.
- **The task file names the pair `d64_t16x32s16m8ro` / `d128_t16x64s16m8ro`; that was the B580's first smoke
  definition. At the pinned commit `b580-fused1` is `d64_t16x64s16m8g4roj` + `d128_t16x128s16m8g8oj`** (its
  `STATUS.md`, "selected by screen 5", and `kFusedB580Pair` in its `Overrides.cpp`). `b70-fused1` is that pair.

## Shared-memory reading (R7), done before any gate

Read by me on `glsl/sarc_dev/sarc_dev_b580_sdpa_fused.glsl` as built for the two variants of `b70-fused1`
(`MULTI_SG`, `ONLINE`, `QK_J_OUTER`; head_dim 64: 16 rows, 64-column blocks, 4 subgroups, `SEGS` 4, `P_STRIDE` 9;
head_dim 128: 16 rows, 128-column blocks, 8 subgroups, `SEGS` 8, `P_STRIDE` 17). Lane = `gl_SubgroupID * 16 +
gl_SubgroupInvocationID` owns row `lane % 16` and segment `lane / 16`: one lane per (row, segment). Every barrier
is `memoryBarrierShared(); barrier()`.

| slot | writers | what orders the reads |
|---|---|---|
| `Psh` scores, tile (i, j) | one `coopMatStore` by the subgroup with `j % G == gl_SubgroupID` (one column each in both variants); tiles are disjoint | `SYNC()` after `qk_block`, then each lane reads its own segment |
| `Rsh[row * SEGS + seg]` | the one lane of (row, seg) | `SYNC()` before the row's lanes read all segments, `SYNC()` after |
| `Gsh` | `atomicOr` by any lane; plain clear by lane 0 only | `SYNC()` between the `atomicOr`s and the one read per lane; the clear follows the `SYNC()` after the divisor stores, which every lane enters after its read; a `SYNC()` follows the clear |
| `Dsh[row * 4 + j]` | the lane of the row whose segment is j (segments 0 to 3; with 8 segments the upper four write nothing) | `SYNC()` before the `coopMatLoad`s, `SYNC()` after |
| `Psh` e values | each lane its own segment | `SYNC()` before `av_block` loads the block, `SYNC()` after it and before the next block's score stores |
| `Psh[row * P_STRIDE + j]` final sums | as `Dsh` | `SYNC()` before the `coopMatLoad` of `den`; the last loads of `Psh` are a `SYNC()` earlier |
| `t_output` | one `coopMatStore` per tile by the subgroup that owns that head_dim slice; other workgroups own other rows or heads | not read |

No slot has two plain writers in a phase. Control flow around every `barrier()` is uniform for the workgroup: the
two early returns precede the first barrier and depend on `gl_WorkGroupID`, uniforms, `gl_NumSubgroups` and
`gl_SubgroupSize`; the block count is per workgroup; the rescale branch tests `Gsh`, read by every lane after the
same barrier. The values written by the lanes of one row to `Dsh` are computed from maxima already reduced over
the row, so the slot's single writer holds the row's value. The copy pass `sarc_dev_b580_sdpa_kvt` has no shared
memory. This agrees with the B580's table; its reading applies here because the limits it rests on are the same
(49152 bytes of shared memory against 3588 / 5892 used; required subgroup size 16 in both yaml entries; subgroup
range 16 to 32).

Specification sentences relied on, each found again on this host through the `vulkan-docs` server (exact-phrase
`search_docs`, 05:47 UTC):

- "... all compute shader invocations for a single workgroup must enter it before any will continue beyond it."
  (`glsl/latest/builtinfunctions.md`, https://docs.vulkan.org/glsl/latest/chapters/builtinfunctions.html)
- "Full subgroups are required when the X dimension of the workgroup size is a multiple of the reported value of
  the SubgroupSize built-in, and one of the following is true: ..." (`spec/latest/shaders.md`,
  https://docs.vulkan.org/spec/latest/chapters/shaders.html). As the B580 campaign found, none of the listed
  conditions holds for these pipelines (SPIR-V 1.3, no full-subgroups flag), so the kernel's run-time check
  (`gl_NumSubgroups == G`, `gl_SubgroupSize == 16`, NaN rows otherwise) is what the lane bijection rests on. That
  campaign's finding F1 (the cooperative-matrix pipelines of the branch, shipped ones included, lack the flag the
  specification requires) applies here unchanged and is not repeated as a question.
- "... a single implementation-dependent invocation within the instance of the matrix's scope performs a
  non-atomic store to that memory location." (`spec/latest/memorymodel.md`,
  https://docs.vulkan.org/spec/latest/appendices/memorymodel.html)
- "If the shader was created with a required subgroup size, the SubgroupSize decorated variable will match that
  value." (`refpages/latest/SubgroupSize.md`)

The remaining sentences of the B580's list (atomics, the data-race definition, uniform control flow) were read in
its `STATUS.md` at `cea76c634` and not searched again here.

## Not done, and why

- Roofs not re-measured (igpu-roofline is not set up in this campaign's artifacts; no linear kernel changed):
  no percent-of-roof is claimed.
- The decode slowdown was measured, not located.
- `tools/test_nexttoken.sh` was not run to completion on the adapted tools (stopped by hand, see `proposal.md`).
- No llama.cpp comparison (N9) and no pull request: outside this task.

## Decision needed from the owner

Not blocking; the campaign is closed on what the task asked.

1. **Decode is 0 to 2.5 % slower with the fused node present** (both sessions; see candidate 1). If that matters
   for a promotion, locating it needs a decode trace and possibly a change in the node file that came from the
   B580 unchanged; neither is in the scope of a confirmation.
2. The B580 campaign's finding F1 (full-subgroups flag) applies here unchanged.

## Blocking

Nothing.
