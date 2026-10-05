# sarc-1.5-xe2-prefill-refine: status

**2026-10-05 02:50 UTC — running. Candidate 1 (`xe2-refine1`, SDPA): +46.9 % geomean over the parent,
`ACCEPTED (reference-error rule, owner decision 2026-10-04)`, not a plain pass. Candidate 2 (`xe2-refine2`,
8da4w linear): +6.9 % geomean over candidate 1, `GATE_PASS`, logits bit-identical. Candidate 3 is in its gate.**

Branch `topic/xe2-prefill-refine`, parent `6a7cc8cc6` (head of `topic/780m-prefill-refine`). Host
`fedora-gpu-eval`, card `b70-0` only (guest PCI `0000:01:00.0`, Vulkan device 0, **`ETVK_DEVICE_INDEX=0`**,
deviceUUID = lock UUID `868023e2-0000-0000-0100-000000000000`), ANV, Mesa 26.2.3. Nothing was run on the second
B70 or on the B580.

## Running now

One GPU job at a time (`.artifacts/logs/chain11.status`): `gate_sdpa.sh s4-c3` (`xe2-refine3` against
`xe2-refine2`, build `topic7` in both arms), its logits probe and decode A/B; then, only if candidate 3 is
accepted, `gate.sh s5-c4` (`xe2-refine4` against `xe2-refine3`). Expected to finish about 05:00 UTC.

## Needs the owner's attention

- **`nvtop` (pid 1952, pts/0, started 17:01 UTC, before the campaign) holds a DRM file of both B70 cards.** It
  is not a workload: every DRM client it owns shows zero engine cycles and zero GPU memory. The campaign guard
  records it per session as an idle monitor (`env.txt`: `idle monitors ... 1952:nvtop`) and would treat it as a
  foreign GPU process, and stop, the moment either number is non-zero. If measuring beside it is not wanted,
  close it and tell me; every session so far ran with it open.
- No other GPU process has appeared. `llama-server`, `comfyui`, `vllm`, `ollama` were inactive at the start.

## Parent control and baseline (re-measured here, not copied)

Builds: `localhost/et-vk-build:rocky10` was built on this host from `tools/Containerfile` (shaderc v2023.8).
`build/parent` = `6a7cc8cc6` exported from the object store; its 53 shipped SPIR-V variants match
`sarc/golden/spirv.json`. `build/topic1` = `9c2f22564`, golden unchanged.

Parent control `s0-parent-verify` (unmodified `verify.sh --models 1b,3b,8b --schemes 4w,8da4w --pdiff`, no
environment): `CONTROL_RECORDED`. 12 of 12 production-diff cases ALL PASSED, 28 of 28 numeric correctness
cases and 4 of 4 rank-3 cases PASSED, decode 31 tokens, default vs tiled SAME on the real-text prompt (both
schemes) and on the unaligned prompt (4w). Status of pristine `dev/1.5` on this device that a 780M-style
checker would call a failure, recorded as such and compared line by line for every candidate:

| line | parent value | why |
|---|---|---|
| `correctness rc=1` | every numeric case PASSED | the rank-3 M = 128 8da4w cases cannot take the 256-row Xe2 tile and fall back to the tiled kernel |
| `1b 8da4w unaligned: default vs tiled output DIFFER` | DIFFER | also in `sarc-1.5-8da4w-port/results/{b580,b70}` |
| `linear <scheme> rc=1` | rc=1 | texture3d coopmat dispatch reported as unexpected, as on the 780M |
| SDPA tiers all / extended / full | stock kernels, 0 mismatches in 4 / 8 / 4 cases | no SDPA row on Intel, the test reports the missing coopmat dispatch |

Baseline + A/A, session `s1-aa` (pristine parent build against the topic build with no environment, arms
interleaved, median of 5 valid runs, tok/s; `results/xe2/sessions/s1-aa/`):

| cell | parent | topic, no env | A/A | expected (`cells.csv`) | parent vs expected |
|---|---:|---:|---:|---:|---:|
| 1B 4w | 11770.10 | 11770.10 | 0.00 % | 11702.9 | +0.57 % |
| 1B 8da4w | 12412.10 | 12337.30 | -0.60 % | 12412.1 | 0.00 % |
| 3B 4w | 4864.61 | 4864.61 | 0.00 % | 4864.61 | 0.00 % |
| 3B 8da4w | 5278.35 | 5264.78 | -0.26 % | 5251.28 | +0.52 % |
| 8B 4w | 2435.20 | 2438.10 | +0.12 % | 2438.10 | -0.12 % |
| 8B 8da4w | 2737.97 | 2737.97 | 0.00 % | 2737.97 | 0.00 % |

A/A geomean -0.12 %, every cell inside +-0.6 %; 60 timed runs, none rejected; next token parent vs topic SAME
in all six cells on `prompt_2048.txt`, `prompt_check.txt` and `r1304.txt`. The baseline agrees with the
device's `cells.csv` numbers within 0.6 %. The timer resolution is 1 ms, i.e. 0.57 % of a 1B prefill (174 ms).

## How this host differs from the 780M protocol (all in `tools/`, reasons in the script headers)

- **Clock and throttle.** `freq0/throttle/status` reads 1 with reason `pl2` (the card's power limit) in every
  loaded run, at 2580 to 2800 MHz; the 4w cells run power-limited (median 2583 to 2750 MHz), the 8da4w cells
  at 2800 MHz. That is the card's normal clock control, so it is counted per run, not rejected. A run is
  rejected for a thermal reason (`thermal`, `prochot`, `ratl`) in any sample, or a median clock under
  `clkmin` = 2505 MHz (97 % of the lowest per-run median of `s1-aa`). The first attempt of `s1-aa` used the
  780M rule (any throttle flag) and a 0.1 s sampler, rejected all 11 runs it made and was stopped
  (`superseded/s1-aa-attempt1-sampler-0.1s-throttle-flag/`).
- **Sampler.** 10 ms (`tools/sampler.py`); a 1B prefill is 0.17 s, about 17 samples. No effect on tok/s in a
  3 x 4 comparison of sampling periods (`raw/smoke/`).
- **Cooling.** The idle package temperature drifts between 56 and 64 C with the fan hysteresis, with nothing
  running. Waiting for idle + 5 C therefore cost 120 s per run and was not cooling anything; the wait now also
  ends when the temperature has stopped falling. Start temperatures of `s1-aa`: 57 to 67 C.
- **Gate checker.** `gate_check.py` compares the two parent-status lines above against the parent control
  instead of requiring `rc=0` / `SAME`; everything else is absolute. It also requires the next token parent vs
  candidate on the three named prompts.
- **Builds and the guard.** The guard matches a running build container (its command line names `llama_main`),
  so no build runs during a measurement, by construction.

## Xe2 SDPA (order of work, step 2)

Reached from the dev zone without a hook: `impl/sarc_dev/Xe2Sdpa.cpp` registers `kUnverified` SDPA base rows
for `bmg g21` / `bmg g31` that match only while `ET_VK_SARC_DEV_PROFILE` names an `xe2-*` profile. A candidate
environment is therefore `ET_VK_SARC_UNVERIFIED=1` + `ET_VK_SARC_DEV_PROFILE=xe2-...`; every other
configuration selects exactly what the release tables select (`test_sarc_select` unchanged: 31 release rows).
Kernels: the 780M twins built for MMA 8x16x16 (fp16 x fp16 -> fp32, which Xe2 exposes, so accumulation
precision is unchanged) and subgroup 16, plus Xe2 families with a fragment-contiguous shared-memory layout.
All from `tools/gen_xe2.py`.

Kernel-level screens (`test_llama_microbench --sdpa`, S = 2048, ms per layer, `results/xe2/screens/`):

| kernels | 8B QK^T / softmax / attn*V | 3B | 1B |
|---|---|---|---|
| stock (the parent) | 3.38 / 1.03 / 3.95 | 2.57 / 0.77 / 2.89 | 1.82 / 1.03 / 2.13 |
| base rows `xe2-sdpa0` (straight port) | 0.83 / 0.80 / 0.57 | 0.62 / 0.59 / 0.44 | 0.43 / 0.80 / 0.38 |
| **`xe2-refine1`** (candidate 1) | 0.41 / 0.80 / 0.48 | 0.31 / 0.60 / 0.38 | 0.30 / 0.80 / 0.38 |

- Screen 1 (48 profiles, 1 round) and screen 2 (15 profiles, 2 rounds, agree within 0.01 ms).
- QK^T: packed staging with a ColumnMajor K load (`pk_t128x64k32g44s16m8nf`) is twice as fast as the scalar
  fp16 staging of the release-style kernel; `NO_MASK_FILL` is worth 0.3 ms on 8B. The fragment-contiguous
  ColumnMajor variant (`xe2c`) is another 0.02 to 0.03 ms on 3B / 8B: kept for a later candidate.
- attn*V: the 128-row fragment-layout tile `xe2_t128x64k32g44s16m8` is best for head_dim 128 (0.48 / 0.38 ms
  against 0.57 / 0.44); head_dim 64 keeps the 64 x 64 tile. Subgroup tiles larger than 32 x 16 lose.
- With these kernels the truncated softmax (0.6 to 0.8 ms) is the largest of the three. Its shader name is
  fixed in the release zone (`impl/sarc/SdpaCoopmat.cpp`), so a dev variant cannot replace it without a hook.
- `xe2-sdpa0`, SDPA correctness tier `all`, one pass: 4 of 4 PASSED, 0 mismatches, `pairing=ok`.

## Linear kernels (step 1 and step 3, so far measurement only)

Phase timing of the shipped tiles (shader clock, 1B shapes, share of a wave; `results/xe2/phases/`):

| kernel | barrier | fetch | MMA | LDS store | prologue + epilog | drain + write |
|---|---:|---:|---:|---:|---:|---:|
| 4w `t128x128k16g44s16m8fli` | 22 to 23 % | 22 to 26 % | 37 to 40 % | 13 to 15 % | 1 % | 1 % |
| 8da4w zpg `t256x64k32g48s16m8` | 18 to 20 % | 32 to 36 % | 22 to 23 % | 15 to 16 % | 5 to 9 % | 2 % |

Tile screens (`screens/screen3-8da4w.csv`, `screen4-4w.csv`, kernel time, 2 rounds, all three models):

- 8da4w: every other tile shape is slower than the shipped one (0.26 to 0.88x). More than 4 x 1 MMA tiles per
  subgroup collapses (0.26 to 0.52x); the same subgroup tile with twice the weight fetches per thread is 0.80x.
  The one gain: texel-wise weight staging on a K = 64 tile, `bt_t128x64k64g44s16m8`, **1.12x** (its zpg twin
  0.68x), so fetching each packed-weight texel once instead of 8 times is what matters. The 780M `bt` body
  cannot run on the shipped 512-thread tile (128 slots); family `xe2bt` (generated, not built yet) can.
- 4w: every other tile is slower (0.14 to 0.90x), including K = 32 chunks (0.50x).

Texel-wise weight staging (`screens/screen5-8da4w.csv`, `screen6-4w.csv`): fetching each packed-weight texel
once, with only the threads that own a texel staging it, is slower than the shipped kernels on this device
(8da4w 0.56 to 0.95x, 0.92x on the shipped tile; 4w 0.80 to 0.85x). The same staging with every thread owning
exactly one texel and K = 64 per chunk is the 1.12x tile above. So the cost is the serial work of a thread
per chunk, not the number of fetches; batch 3 (in the build now running) adds balanced K = 64 / 128 tiles.

## Roofs (re-measured, not the old evidence)

igpu-roofline plan `fast`, `b70-0`, 2026-10-04 21:54 to 22:17 UTC, driver Mesa 26.2.3 (109060099), runner
`810e098c8abb`, clocks not pinned, sentinel healthy at every checkpoint, every roof confirmed by 3 repeats
(`results/xe2/roofline/xe2-fast-20261004/REPORT.md`; artifacts `roofline/xe2-fast-20261004/`): matrix fp16
173.3 TFLOP/s, matrix fp16 -> fp32 179.9 TFLOP/s, matrix int8 359.9 TOP/s; fed from shared memory 168.4 /
166.6 / 323.3; global read 603 GB/s, write 509 GB/s, copy 532 GB/s. The tool is the fleet copy that was already
on this host, run unchanged from a copy in the artifact directory. A first run was stopped after 4 minutes by
the campaign guard (its name fallback matched the text of an operator shell command, no GPU process was
involved); it is under `superseded/`, and the guard no longer matches inline shell command text.

## Per-cell numbers against the parent

### Candidate 1, `xe2-refine1` (SDPA prefill kernels): ACCEPTED (reference-error rule, owner decision 2026-10-04)

Session `s2-c1`: pristine parent build (no environment) against build `topic4` with
`ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=xe2-refine1`; tok/s, median of 5 valid runs per arm, arms
interleaved (`results/xe2/sessions/s2-c1/`):

| cell | parent | candidate 1 | gain | next token parent vs candidate (timed / real-text / unaligned prompt) |
|---|---:|---:|---:|---|
| 1B 4w | 11702.90 | 17504.30 | +49.57 % | SAME / SAME / SAME |
| 1B 8da4w | 12337.30 | 18450.50 | +49.55 % | SAME / SAME / SAME |
| 3B 4w | 4853.08 | 7366.91 | +51.80 % | SAME / SAME / SAME |
| 3B 8da4w | 5264.78 | 8192.00 | +55.60 % | SAME / SAME / SAME |
| 8B 4w | 2420.80 | 3282.05 | +35.58 % | SAME / SAME / SAME |
| 8B 8da4w | 2737.97 | 3835.21 | +40.07 % | **DIFFER / DIFFER** / SAME |

Geomean +46.86 %, every cell far outside the +-2 % band (A/A noise 0.6 %). 61 timed runs, one rejected
(`8b 4w cand r5`, `clock_low`) and replaced.

**This is not a plain pass.** The gate (`gate_sdpa.sh`, `sessions/s2-c1/gate.txt`) ends `GATE_FAIL` on two
lines that state one fact: the next token of the candidate differs from the parent's for 8B 8da4w on
`prompt_2048.txt` and on `prompt_check.txt`. Everything else in the gate passes: SDPA correctness 12 passes x
tiers all / extended / full, 0 mismatches and `pairing=ok` in all 192 cases; unmodified `verify.sh`
completed, 12 of 12 production-diff cases ALL PASSED, 28 of 28 numeric correctness cases, default vs tiled SAME
on both prompts for both schemes, decode 31 tokens, every other status line equal to the parent control;
traces for 12 runs. The candidate replaces the stock attention kernels, which accumulate in fp16, by kernels
that accumulate in fp32: an arithmetic change. It is therefore decided by the owner's second decision of
2026-10-04 (`CAMPAIGN.md`), with the thresholds fixed there (`tools/decide.py`, `sessions/s2-c1/decision.txt`):

1. **Error against the fp32 CPU reference** (`results/xe2/sdpa-error/c1-full.csv`, build `topic5`, same
   inputs for both arms): not larger than the parent's in every case.

   | case (S = 2048 unless noted) | parent rms / max | candidate rms / max |
   |---|---|---|
   | 1B head configuration | 1.27e-4 / 1.37e-3 | 2.76e-5 / 1.19e-3 |
   | 3B head configuration | 1.28e-4 / 1.33e-3 | 2.79e-5 / 1.15e-3 |
   | 8B head configuration | 1.28e-4 / 1.75e-3 | 2.76e-5 / 1.06e-3 |
   | 8B, S = 1024 at input_pos 1024 | 1.42e-4 / 1.27e-3 | 1.29e-5 / 1.24e-4 |

   The eight `extended` cases agree (`c1-extended.csv`): candidate rms 2.5e-5 to 6.6e-5 against 1.0e-4 to
   1.3e-4.
2. **Logits** (`tools/probe.sh`: a fresh process per prompt, whole logits vector kept, four arms;
   `results/xe2/probe/s2-c1/`). The probe reproduces the gate: of the 12 gate-prompt comparisons only the two
   8B 8da4w ones differ.
   - `prompt_check.txt` (real text, 1972 tokens), 8B 8da4w: a three-way near-tie in the parent, logits
     15.469 / 15.320 / 15.000 for ids 45647 / 6062 / 70159 (top-2 margin 0.148; parent tiled identical to
     parent default); the candidate has 15.172 / 15.172 / 15.242, so id 70159 leads by 0.070. KL(parent ||
     candidate) = 0.042 nat. Candidate tiled is identical to candidate default.
   - `prompt_2048.txt` (2048 times the token " the"), 8B 8da4w: parent top-2 margin 0.336 (ids 247 / 118 at
     11.664 / 11.328; parent tiled 11.516 / 11.109). The candidate moves id 247 to 8.930 and id 53 from 8.125
     to 11.328, which becomes the top token: KL 0.975 nat, the largest logit change 6.06. This is not a small
     move. It is on the degenerate prompt, where every attention row averages 2048 identical values and the
     parent's fp16 accumulation is at its worst; it is reported in full, not explained away.
   - 35 real-text windows per cell (two documents, lengths 256 to 2048 tokens), candidate default against
     parent default, and beside it parent tiled against parent default (two arms already accepted on this
     device):

     | cell | top-1 differs (cand / parent-tiled) | mean KL nat (cand / parent-tiled) | max KL | max logit diff | perplexity ratio, 27 windows |
     |---|---|---|---|---|---|
     | 1B 4w | 1 / 0 of 35 | 3.0e-3 / 9.7e-4 | 0.033 / 0.010 | 0.80 / 0.75 | 1.011 / 0.994 |
     | 1B 8da4w | 1 / 0 | 7.5e-2 / 4.6e-2 | 0.91 / 0.33 | 5.08 / 5.17 | 0.913 / 0.933 |
     | 3B 4w | 1 / 0 | 7.7e-4 / 4.2e-4 | 0.008 / 0.005 | 0.83 / 0.61 | 1.018 / 1.014 |
     | 3B 8da4w | 1 / 1 | 2.5e-2 / 5.3e-2 | 0.29 / 1.14 | 3.62 / 3.10 | 0.976 / 0.995 |
     | 8B 4w | 0 / 0 | 7.1e-4 / 4.2e-4 | 0.007 / 0.006 | 0.70 / 0.61 | 0.993 / 1.005 |
     | 8B 8da4w | 0 / 1 | 1.4e-2 / 2.5e-2 | 0.25 / 0.31 | 2.73 / 2.92 | 1.011 / 0.909 |

     (Perplexity ratio = arm / parent default on the 27 windows whose next token is known; exact values in
     `summary.csv`.)
3. **Gross-divergence check** (reject above 0.5 nat mean KL or top-1 differing on more than a third of the
   windows, in any cell): largest mean KL 0.075 nat, at most 1 of 35 windows differs. Passed.
4. Next-token items that differ, listed and not waived: 8B 8da4w on `prompt_2048.txt` and `prompt_check.txt`
   (parent vs candidate). Windows where the candidate's top-1 differs from the parent's: `w1280-gpl-384`
   (1B 4w), `w1536-gpl-0` (1B 8da4w, 3B 4w, 3B 8da4w); logits of all arms in `probe/s2-c1/differing.md`.

Decode (not part of the prefill gate, measured because the truncated softmax also runs in decode;
`sessions/s2-c1/decode/summary.csv`, 5 runs per arm, 31 tokens after a 2048-token prefill): candidate 0.7 to
2.0 % slower than the parent in all six cells (1B 4w 102.7 -> 101.3 tok/s, 8B 4w 32.2 -> 31.6), inside the
repeat range of each cell but in the same direction everywhere. The 78 tok/s seen once inside `verify.sh` was
a single noisy run. Not investigated further.

Where the gain comes from (warm ETDump of both arms, ms per prefill, `sessions/s2-c1/trace/families.csv`):

| cell | arm | total | linear GEMM | QK^T | attn*V | softmax | other |
|---|---|---:|---:|---:|---:|---:|---:|
| 1B 4w | parent | 165.5 | 59.0 | 29.9 | 34.8 | 16.5 | 25.3 |
| 1B 4w | candidate 1 | 108.2 | 59.1 | 4.8 | 6.2 | 12.7 | 25.4 |
| 3B 4w | parent | 412.3 | 172.0 | 76.2 | 85.2 | 22.0 | 56.9 |
| 3B 4w | candidate 1 | 269.1 | 174.1 | 9.4 | 11.1 | 17.0 | 57.5 |
| 8B 4w | parent | 831.2 | 443.9 | 117.1 | 134.7 | 33.6 | 101.9 |
| 8B 4w | candidate 1 | 611.1 | 451.6 | 14.6 | 16.3 | 26.0 | 102.6 |
| 8B 8da4w | parent | 737.5 | 348.2 | 108.3 | 126.2 | 32.6 | 122.2 |
| 8B 8da4w | candidate 1 | 524.3 | 348.1 | 13.4 | 15.2 | 25.0 | 122.6 |

Linear kernels in the model against the fresh roofs (both arms the same kernels): 4w 63.3 to 67.5 TFLOP/s =
36.5 to 38.9 % of the fp16 matrix roof (173.3); 8da4w 82.1 to 87.8 TOP/s = 22.8 to 24.4 % of the int8 matrix
roof (359.9).


### Candidate 2, `xe2-refine2` (8da4w linear, texel-wise weight staging on a balanced K = 64 tile): GATE_PASS

Session `s3-c2`: build `topic5` in both arms, `xe2-refine1` (candidate 1) against `xe2-refine2`; tok/s, median
of 5 valid runs per arm, arms interleaved (`results/xe2/sessions/s3-c2/`):

| cell | candidate 1 | candidate 2 | gain | next token (timed / real-text / unaligned prompt) |
|---|---:|---:|---:|---|
| 1B 4w | 17504.30 | 17504.30 | 0.00 % | SAME / SAME / SAME |
| 1B 8da4w | 18789.00 | 20480.00 | +9.00 % | SAME / SAME / SAME |
| 3B 4w | 7393.50 | 7366.91 | -0.36 % | SAME / SAME / SAME |
| 3B 8da4w | 8192.00 | 9570.09 | +16.82 % | SAME / SAME / SAME |
| 8B 4w | 3292.60 | 3297.91 | +0.16 % | SAME / SAME / SAME |
| 8B 8da4w | 3828.04 | 4491.23 | +17.32 % | SAME / SAME / SAME |

Geomean +6.88 %; the three 8da4w cells are outside the +-2 % band, the three 4w cells (whose kernels are the
same in both arms) are inside +-0.4 %. 60 timed runs, none rejected. `gate.txt`: no FAIL line; unmodified
`verify.sh` completed, 12 of 12 production-diff cases ALL PASSED, 28 of 28 numeric correctness cases, default
vs tiled SAME on both prompts for both schemes (also on the 1B 8da4w unaligned prompt, where the parent
control reads DIFFER), decode 31 tokens. Logits probe (`results/xe2/probe/s3-c2/`): bit-identical to candidate
1 on all 35 real-text windows of all six cells, so this is a staging change with no arithmetic change
(`decision.txt`: `GATE_PASS`). Decode A/B, 3 runs per arm: 0.98 to 1.00x, inside the repeat range.

Where the gain comes from (warm ETDump, ms per prefill, `sessions/s3-c2/trace/families.csv`): only the linear
GEMM family moves.

| cell | total, candidate 1 -> 2 | linear GEMM, candidate 1 -> 2 |
|---|---|---|
| 1B 8da4w | 100.9 -> 91.8 | 45.4 -> 36.4 (1.25x) |
| 3B 8da4w | 242.5 -> 206.4 | 136.3 -> 100.1 (1.36x) |
| 8B 8da4w | 524.6 -> 448.6 | 348.2 -> 272.6 (1.28x) |

Phase timing of the two tiles (`results/xe2/phases/prof-parent-8da4w.csv`, `prof-c2-8da4w.csv`, 1B wq/wo,
shader-clock cycles of one wave over the kernel): both cover the layer with 256 tiles of 16384 outputs, the
candidate with half as many K chunks (K = 64 against 32). Total 128176 cycles against 190448; weight fetch
36672 against 68640, barrier 20344 against 35104, shared-memory store 21008 against 29480, MMA 33088 against
43200. The saving is mostly in fetch and barrier.


## Next

1. Candidate 3 (`xe2-refine3`): fragment-contiguous ColumnMajor QK^T, 0.02 to 0.03 ms per layer at kernel
   level; candidate 4 (`xe2-refine4`): the 64-column K = 64 8da4w tile where N >= 4K.
2. 4w linear: tile shapes, K = 32 chunks, texel-wise staging and a smaller drain all fail to beat the shipped
   tile; the first split-staging body was 15 % slower than the release kernel on the shipped geometry
   (screen 9), the revised one is in screen 11.
3. Softmax: now the largest attention kernel (12.7 of 108 ms on 1B). The single-read variant is not faster
   (screen 10); a variant with subgroup reductions is in screen 12. Both need a release-zone hook and are
   measured through a local uncommitted patch only (`tools/hook-sdpa-softmax.patch`).
4. Stop after two consecutive gated candidates under 2 % geomean.

## Awaiting B580 confirmation

Candidate 2 (`xe2-refine2`, plain gate pass on the B70): 8da4w linear
`sarc_dev_linear_dq8ca_coopmat_zpg_xe2bt_t128x128k64g84s16m8`.

Candidate 1 (`xe2-refine1`, accepted under the reference-error rule on the B70): QK^T `sarc_sdpa_qk_coopmat_pk_t128x64k32g44s16m8nf`,
attn*V `sarc_sdpa_av_coopmat_xe2_t128x64k32g44s16m8` (head_dim 128) and
`sarc_sdpa_av_coopmat_sweep_t64x64k32g44s16m8` (head_dim 64), with the truncated SARC softmax. Every Xe2 variant so far keeps its shared memory under 46000 bytes (the B70
reports `maxComputeSharedMemorySize` 49152; the B580's value was not read here and should be confirmed), uses
workgroups of at most 1024 invocations and the 8x16x16 / subgroup-16 shapes the shipped Intel rows already use
on both cards, and does not depend on the amount of device memory.
