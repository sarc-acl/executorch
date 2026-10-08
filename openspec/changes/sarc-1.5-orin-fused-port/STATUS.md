# STATUS: sarc-1.5-orin-fused-port

**2026-10-08 21:23 UTC. Baseline and A/A done (`s1-aa`): the parent is within 0.09 % of `s10-final` in every cell,
A/A geomean -0.05 %; clock floor 593 MHz. Both controls accepted. Running: traces of `s1-aa`, the hook control
`s2n-noenv`, then candidate 1 (`chain3`). No candidate measured yet.**

All times are UTC from `date -u`.

## Running now

- Device `duck-naughty` (detached, `~/hmz-sarc-orin-fused/jobs/<job>.{status,out}`):
  - `chain2` (session part done 21:20): warm traces of both arms of `s1-aa`, then `s2n-noenv` = unmodified
    `verify.sh` on `topic1` with nothing selected against `s0n-noenv` (hook control, D4).
  - `chain3` (queued behind `chain2`): candidate 1 = profile `orin-fused1` on `topic1`: one correctness pass per
    tier (`all`, `extended`, `full`, `peaked`, `fused`; the chain stops if a case is not PASSED), error against the
    fp32 reference for stock / parent / candidate (`sdpa-error1`), kernel timing of the four forms of the fused
    kernel, 3 rounds (`sdpa-screen1`), then the gate `s3-c1` (`gate_sdpa.sh`: 12 passes of `all`, `extended`,
    `full`, 3 of `peaked` and `fused`, unmodified `verify.sh`, interleaved session, traces). About 3.5 hours.
- Workstation: nothing. `logits_dump` built for `parent` and `topic1` (20:49) and deployed.
- Coordinator hold: `tools/HOLD.md` (device: `~/hmz-sarc-orin-fused/HOLD`; builds: `.artifacts/HOLD`). None seen.
- The workstation's build lock was held by the Arc B580 campaign (exclusively for its build until 19:51, shared
  for timed sessions afterwards); my builds waited for it each time, as the task says.

## Baseline and A/A (`s1-aa`, 20:35 to 21:20 UTC, record-only clock)

Parent arm = build `parent` (`8973ced76`), candidate arm = build `topic1` (`0f14f2a1a`: hook + candidate 1
code), **both with the parent environment** (`orin-refine5` + `orin_g64`): the A/A also shows that the hook and
the linked dev code cost nothing while the fused node is not selected. Tok/s, median of 5 valid runs per arm,
arms interleaved; recomputed from `runs.csv`:

| cell | parent | `topic1`, parent environment | ratio | `s10-final` (expected) | parent vs expected | repeat spread (parent / `topic1`) |
|---|---:|---:|---:|---:|---:|---|
| 1B 4w | 1489.45 | 1488.37 | 0.9993 | 1489.45 | 0.00 % | 0.07 / 0.15 % |
| 1B 8da4w | 1382.85 | 1380.05 | 0.9980 | 1382.85 | 0.00 % | 0.34 / 0.47 % |
| 3B 4w | 629.38 | 629.19 | 0.9997 | 628.99 | +0.06 % | 0.15 / 0.12 % |
| 3B 8da4w | 570.16 | 570.16 | 1.0000 | 570.32 | -0.03 % | 0.14 / 0.08 % |
| 8B 4w | 295.27 | 295.27 | 1.0000 | 295.53 | -0.09 % | 0.19 / 0.12 % |
| 8B 8da4w | 269.05 | 269.12 | 1.0003 | 269.19 | -0.05 % | 0.08 / 0.09 % |

- Baseline: every cell within 0.09 % of the first campaign's final session (threshold 3 %).
- A/A: geomean -0.05 %, largest cell -0.20 %, repeat spread at most 0.47 %. The noise on this device is far
  inside the +-2 % band. The runner's timer has a 1 ms step: 0.07 % of a 1B prefill (1375 ms).
- 60 timed runs, all valid: rc 0, 2048 prompt tokens, 0 generated, no foreign GPU process, 13 to 73 clock
  samples per prefill window (threshold 5), median clock 612 MHz in every run. Start temperature 57 to 62 C.
  Next token parent vs `topic1`: SAME in 24 of 24 rows. `gate_check.py session --calibration --require-logs`:
  ACCEPT, 0 findings.
- Clock floor (`results/orin/clkmin.json`, `calibrate_clock.py`): floor(0.97 x 612) = **593 MHz**, device-wide; no
  run below it.
- Memory: at least 5628 MB available before every timed run; swap in use (74 and 106 MB at two looks) since the first 8B run of
  the parent control (the model file is read into the page cache before each cell, D5). Model load 1.6 to 10.8 s,
  none slow enough to abort: 0 runner aborts in the session.

## Pristine control `s0n-noenv` (20:35 UTC, `GATE_ACCEPTED`)

Unmodified `verify.sh` on the `parent` build with nothing selected: the dev/1.5 state of this device.
`gate_check.py verify`: ACCEPT, 0 findings. Its default-arm prefill runs: 1B 890.82 / 823.15, 3B 360.25 / 320.20,
8B 189.63 / 170.33 tok/s; the published `cells.csv` numbers are 890.82 / 822.82, 360.37 / 320.30, 189.74 / 170.43:
within 0.1 %. Error of the stock attention kernels against the fp32 reference on the S = 2048 cases: rms 8.4e-05
to 8.7e-05, maximum 1.41e-03 to 1.71e-03 (four times and twice the parent's).

## Builds

Cross image `localhost/et-jetson-cross:jp7.2.1` (`d182f725bb32`), shaderc v2026.1, 8 jobs, exclusive build lock;
provenance `.artifacts/build/<tag>.src.txt`.

| tag | commit | source | SPIR-V |
|---|---|---|---|
| `parent` | `8973ced76` | `git archive` + 30 pinned submodules, tree sha256 `3bbbd4cb...` | 1610 shaders, **all byte-identical to the first campaign's final build `topic14`** (`diff` of the two `spv.sha256` lists: 0 lines); `libllama_runner.so` has `topic14`'s hash |
| `topic1` | `0f14f2a1a` | hard links to `parent` + 77 changed paths, each verified by blob hash | 1620 shaders: the parent's 1610 byte-identical + the 10 new `sarc_dev_orin_sdpa_*`; `tools/shipped.py`: all 53 shipped variants byte-identical to `parent`: **UNCHANGED** |

`spirv_golden.py` reads FAIL with 14 DIFF lines on both builds, as on every build of the first campaign: the
cross image's glslc is not the one the goldens were made with; none of the 14 is a kernel the Orin rows
dispatch (`build/topic1.shipped.txt`: same set in parent and candidate, Orin kernels differing: 0).

## Parent control `s0-parent-verify` (20:14 UTC, `GATE_ACCEPTED`)

Unmodified `sarc/tools/verify.sh --models 1b,3b,8b --schemes 4w,8da4w --pdiff` on the `parent` build with the
parent environment. `gate_check.py verify` against itself: ACCEPT, 0 findings. What it prints (and every
candidate must print again): `correctness rc=1`, `linear 4w rc=1`, `linear 8da4w rc=1`, the six `pdiff ... buffer
rc=1` lines (no buffer row for this device; the same lines as in the first campaign), six `pdiff ... texture3d`
ALL PASSED, default vs tiled SAME on the check and the unaligned prompt for 1B 4w and 8da4w, decode 31 tokens.
Its single default-arm prefill runs: 1B 1489.45 / 1379.12, 3B 628.99 / 570.47, 8B 294.85 / 269.01 tok/s
(expected 1489.45 / 1382.85, 628.99 / 570.32, 295.53 / 269.19): within 0.3 %.

Error of the parent's attention kernels against the fp32 CPU reference (one pass per tier, recorded with the
control; this is what criterion 1 of the reference-error rule compares the candidate with):

| case (S = 2048) | rms | maximum |
|---|---:|---:|
| `1b_head_config_s2048` | 2.101e-05 | 9.135e-04 |
| `3b_head_config_s2048` | 2.068e-05 | 7.828e-04 |
| `8b_head_config_s2048` | 2.057e-05 | 8.911e-04 |
| `tiny_gqa_s2048` | 2.080e-05 | 6.943e-04 |
| `tiny_d128_s2048` | 2.071e-05 | 7.796e-04 |

## Parent

`8973ced76` with `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=orin-refine5 ET_VK_SARC_SOFTMAX_VARIANT=orin_g64`
(the first campaign's final stack; the softmax hook `307abb2ed` is committed, so the softmax is named by the
dev-zone variable, no local patch). Expected (`s10-final`): 1B 1489.45 / 1382.85, 3B 628.99 / 570.32,
8B 295.53 / 269.19 tok/s (4w / 8da4w). Pristine state: the same commit, no environment.

## Ceiling (computed before candidate 1, from the first campaign's `s10-final` traces)

Attention (QK^T + softmax + attn*V) in the parent, ms per 2048-token prefill, and its share of the dispatch
total (`sarc-1.5-orin-prefill-refine/proposal.md`, "Where the gain is"):

| | 1B 4w | 1B 8da4w | 3B 4w | 3B 8da4w | 8B 4w | 8B 8da4w | geomean |
|---|---:|---:|---:|---:|---:|---:|---:|
| attention / total, ms | 302 / 1359 | 302 / 1465 | 516 / 3235 | 517 / 3571 | 783 / 6913 | 784 / 7587 | |
| share | 22.2 % | 20.6 % | 16.0 % | 14.5 % | 11.3 % | 10.3 % | |
| gain if attention cost nothing (ceiling) | +28.6 % | +26.0 % | +19.0 % | +16.9 % | +12.8 % | +11.5 % | +19.0 % |
| gain if 76 % of it goes (the RX 7600's fused kernel) | +20.3 % | +18.6 % | +13.8 % | +12.4 % | +9.4 % | +8.5 % | +13.7 % |
| gain if 50 % of it goes | +12.5 % | +11.5 % | +8.7 % | +7.8 % | +6.0 % | +5.4 % | +8.6 % |

The task expects +5 to +12 % geomean. On this device attention is bound by memory traffic (the first campaign:
half of the softmax is the first read of a row) and shared memory is expensive, so the lower half is as likely
as the upper.

## Candidate 1: what was written (commit `0f14f2a1a`, build `topic1`), not measured yet

- Hook: `9d91480b2` = cherry-pick of `1c8861aa7e`, alone, same patch-id (`0027e89f...`), 49 lines in `SDPA.cpp`,
  `sarc/SdpaCoopmat.{cpp,h}`, `sarc/Select.h`. `sarc/tools/check.sh --no-build` on `0f14f2a1a`: `check.sh: PASS`;
  `test_sarc_select` on the release tables `PASS (1240 checks, 31 rows, 0 candidates, dev zone absent,
  unverified off)`, the parent's line.
- `glsl/sarc_dev/sarc_dev_orin_sdpa_fused3sb.{glsl,yaml}`: the RX 7600's `fused3sb` (`b3bb758e38`), identical from
  `#version` on (`diff`: no difference); `sarc_dev_orin_sdpa_{kvt,vt}`: the 780M's copy passes, identical from
  `#version` on. Variants: `rko` (one pass, packed: candidate 1), `ro` (one pass, unpacked), `rk`, `r` (two
  passes) for head_dim 64 (`t32x32`) and 128 (`t16x64`).
- `impl/sarc_dev/orin/SdpaOrinFused.cpp`: port of the 780M's selector; sets `Override::sdpa_fused_*` only when a
  variant is named (profile `orin-fused1` or `ET_VK_SARC_ORIN_SDPA_FUSED`), so with nothing selected both are null.
- `Overrides.cpp`, blocks `orin-fused profiles` / `orin-fused override`: profile `orin-fused1` = the preferences
  of `orin-refine5`; an `orin-fused*` profile names the softmax `orin_g64` itself. Candidate environment:
  `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=orin-fused1`.
- `test_llama_microbench.cpp`: the fused kernel is recognised (timing, `fused=` in `[sdpa-kernels]`, pairing: no
  QK^T / softmax / attn*V kernel may run beside it); the 780M's tiers `peaked` and `fused` added.

### Shared-memory reading of the kernel (R7, before any gate)

Read in the source and in the SPIR-V of all eight variants (host glslc, shaderc v2026.1; the ten binaries of
build `topic1` have the same sha256): every `OpMemoryBarrier` is followed by `OpControlBarrier` with execution scope Subgroup (7 pairs in the
`t32x32` one-pass variants, 9 in `t16x64`).

| shared object | writer | readers | ordered by |
|---|---|---|---|
| `Psh` scores | `coopMatStore` of the whole subgroup (`qk_block`) | each lane, its own row segment | barrier pair after `qk_block` |
| `Psh` e | each lane, its own segment `e_idx + [0, SEG_V8)`: row `id % M`, segment `id / M`, one lane per slot | `coopMatLoad` (`av_block`) | barrier pair before `av_block`; another after it, before the next `qk_block` store |
| `Rsh` (row maximum per block, row sum) | each lane, slot `e_row * SEGS + e_seg`: one lane per slot | the lanes of the same row | barrier pair after the store, another after the reads |
| `Dsh` rescale divisors | each lane, `e_row * 4 + j`, `j = e_seg, e_seg + SEGS, ...`: disjoint per lane | `coopMatLoad` | barrier pair after the stores, another after the load; inside `if (subgroupAny(...))`, which is uniform over the subgroup |
| `Psh` divisors (end) | each lane, `e_row * P_STRIDE + j`, disjoint per lane | `coopMatLoad` | barrier pair after the stores |

No slot has two writers (no "every lane stores the reduced value"); no read of another lane's slot without an
execution barrier after its store; no write to a slot another lane may still be reading (a barrier pair closes
every read phase). Row ownership uses `gl_SubgroupInvocationID` only, never `gl_LocalInvocationID`. What the
kernel still assumes, as every cooperative-matrix kernel on this device does: a workgroup of 32 is exactly one
full subgroup. The copy passes use no shared memory; every invocation writes its own 4 x 4 block.
Not lockstep-dependent, but worth watching in the tiers: the one-pass form starts from a row maximum of -inf
(`exp(+inf)`, `0 / inf`); the `peaked` tier exercises the rescale path.

## Next step

`chain3` (candidate 1) is queued. After its gate: the 41-prompt real-text probe of the four arms and
`ref_error_rule.py`, then the candidate-2 decision from `sdpa-screen1` by the rule in `tools/thresholds.txt`.

## Thresholds

`tools/thresholds.txt` (committed in `086651dd4`, before any measurement).

## Taken from the 4070 Ti fused port

Nothing yet: `origin/topic/4070ti-fused-port` does not exist at 19:45 UTC (`git fetch origin`).
