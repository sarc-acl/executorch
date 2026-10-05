# STATUS: 780M prefill campaign, round 2 (parameter space + beyond)

Updated 2026-10-04 19:00 PDT. Parent for this round: profile `780m-refine3` (build `topic-r1`).
Artifacts: `rocky-ryzen:~/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-04/` (new raw data; moved there from a
private `/tmp` directory on 2026-10-04) and `.../780m-prefill-refine-2026-10-03/` (earlier builds and sessions).

## Running now (detached chains; nothing needs attention)

1. `chain1b.sh`: gate of candidate 7 (softmax r3, `stage/c7-softmax-r3`) from a cool start: e2e session first,
   then the SDPA tiers (12 passes each), `verify.sh`, traces; until about 20:15 PDT.
2. `chain2.sh`: igpu-roofline `fast` plan (`~/igpu-roofline/campaigns/780m/2026-10-04-fast-prefill-refine2`), about 30 min.
3. `chain3.sh`: 4w refinement round 1 (1,002 neighbour configurations, `raw/refine1/screen.csv`), about 1 h.
4. `chain4.sh`: the 8B 8da4w cell of the candidate-7 session again, from a cool start (see below).
5. Then the QK^T / attn*V / 8da4w enumeration resumes by itself (`raw/space/results.csv`; 55 of 1,724 SDPA runs
   done, 8da4w not started; about 16 h, pausing 06:40 to 07:40).

## Candidate 7 (softmax r3, through the uncommitted hook): e2e session done, gate still running

Session `c7-softmax-r3` (`<artifacts>/stage/c7-softmax-r3`), started 18:13 PDT at 45 C after the full 30 min
wait. Both arms are the same binary (build `hook3` = the branch + `hooks/softmax-name-hook.patch` in a scratch
tree) with `ET_VK_SARC_DEV_PROFILE=780m-refine3`; the candidate arm adds `ET_VK_SARC_780M_SOFTMAX=r3`.
Tok/s, median of 5 valid runs per arm, arms interleaved:

| cell | parent (`780m-refine3`) | candidate (+ softmax r3) | gain | repeat spread parent / candidate | next token (2048 / unaligned prompt) |
|---|---:|---:|---:|---|---|
| 1B 4w | 2828.73 | 3020.65 | **+6.78 %** | 0.28 / 0.29 % | SAME / SAME |
| 1B 8da4w | 2824.83 | 3011.76 | **+6.62 %** | 0.28 / 0.15 % | SAME / SAME |
| 3B 4w | 1190.70 | 1231.51 | **+3.43 %** | 0.12 / 0.18 % | SAME / SAME |
| 3B 8da4w | 1170.29 | 1206.12 | **+3.06 %** | 0.40 / 0.35 % | SAME / SAME |
| 8B 4w | 541.23 | 554.41 | **+2.44 %** | 0.71 / 0.11 % | SAME / SAME |
| 8B 8da4w | (548.03, 549.06, ...) | (562.48, 562.02, ...) | INCOMPLETE | 4 valid runs per arm | SAME / SAME |

Geomean of the five complete cells +4.45 %, each outside the +-2 % band and far above the A/A floor (+-0.23 %).
The parent arm agrees with `t3-aa` within 0.5 % (2828.73 / 2828.73, 1192.08 / 1172.97, 544.10 / 549.50).

**The 8B 8da4w cell is incomplete and will be measured again** (`chain4.sh`, cool start, sweep paused): its
runs 4 to 8 (eight runs) are marked `other_gpu_process`. The process the session saw was pid 576199,
`/bin/bash`: an interactive monitoring shell of this campaign whose command line contained the name of the
runner binary as text. `e2e5.sh` looks for other GPU processes with `pgrep -af`, which matches full command lines,
so the shell was counted for the 6 minutes it was alive. No GPU process ran (the lock holder was the session; the
flagged runs read 548.0 to 549.4 and 561.9 to 562.5 tok/s, the same as the valid ones). The flag is not removed
and the tool is not changed; the cell is repeated instead. Pitfall for anyone watching a session: do not put the
runner or microbench binary names in a shell command while `e2e5.sh` is running.

Still running for this candidate: SDPA correctness tiers (tier `all`: 12 of 12 passes, 4 of 4 cases, pairing ok),
`verify.sh`, traces. Still owed: the bitwise comparison of the SDPA output against the parent.

## Can the sweep slot in between two timed runs of a session? No (checked 2026-10-04 18:44 PDT, during `c7`)

Asked by the owner after seeing `sweep_space.py` alive and the GPU at 79 C during the candidate-7 session.

- `e2e5.sh` takes the gpu-lab lock once, at the start of the script (`exec 9>>lock; flock 9`), and holds it until
  it exits: the lock is per session, not per run. `lslocks` during the session shows `e2e5.sh` (pid 565053) as the
  holder of `~/.cache/gpu-lab/lock-00000000-c400-...`. `sweep_space.py` takes the same lock for each of its own
  runs, so it cannot run while a session holds it.
- Independently of the lock, the sweep waits while `<artifacts>/PAUSE` exists; `gate2.sh` creates it before the
  cool-down wait and removes it when the whole gate is done. `PAUSE` has existed since 17:42 PDT.
- The enumeration process seen alive (pid 537928) is that paused sweep. Its last result row is stamped
  21:15:48 UTC (14:15 PDT), four hours before the session's first run (01:13:58 UTC); `raw/space/results.csv` and
  `sweep.out` have not been written since. No microbench process ran during the session.
- 79 C is the in-run temperature of `llama_main` itself (the runs of this session end at 60 to 73 C and peak
  higher; the earlier sessions peaked at 88 to 95 C). The session started at 45 C after the full 30 min wait
  (`prestart.txt`: the device did not get below 45 C this evening; `t3-aa` started at 42 C).

Conclusion: the session is not affected and is not repeated. The one way a sweep job could run next to a
session would be a session script that locks per run; none of the session tools does.

## Owner decisions in force (read from `CAMPAIGN.md`, section "Owner decisions", 2026-10-04)

- Next-token near-ties: a `DIFFER` no longer rejects by itself if the logits of all four arms, a real-text
  comparison on at least 32 prompts and the parent-tiled vs parent-default floor are measured; recorded as
  `ACCEPTED (near-tie, owner decision 2026-10-04)`. Not used so far: no `DIFFER` has occurred in this round.
- Arithmetic changes are judged against the fp32 reference (rms and maximum error not larger than the parent's
  on the production shapes), with a gross-divergence check; recorded as `ACCEPTED (reference-error rule, ...)`.
  A change meant to be bit-identical does not use this rule and must show bit-identical output. Candidate 7
  (softmax r3) is meant to be bit-identical on every element that is read; the bitwise comparison of the SDPA
  output against the parent is still owed (the test dump is written, the build waits for a quiet moment).
- Large parameter spaces: the method below.

## Owner decision, 2026-10-04 (relayed): how the 4w family is searched

The full 4w space (26.8 days) is not wanted and the staged plan (geometries first, flags afterwards) is dropped
before any of its rows were measured. Instead:

1. cost per configuration broken down, and a cheap screening mode validated against the full measurement by rank
   correlation on at least 50 configurations;
2. a uniform random sample of 2,500 of the 125,712 survivors, seed 20261004
   (`tools/sample_space.py`; survivor list sha256 `5b0cd644...605d75`, `enum_space.py --list 4w`), screened;
3. parameter importance and pair interactions from that sample, and a check of the staged plan's independence
   assumption against it;
4. local refinement around the best 20 sampled configurations (one or two parameters at a time);
5. the best 10 per shape get the full measurement (5 repeats, 12 correctness passes).

8da4w, QK^T and attn*V stay full enumerations.

### Cost of one 4w configuration (`raw/cost/cost-1.csv`, one configuration, texture3d, correctness off)

| mode | shapes | runs per shape | wall |
|---|---:|---|---:|
| no case (process start, Vulkan device, exit) | 0 | | 0.05 s |
| full measurement | 12 | 3 warm-up + 5 timed | 16.3 s |
| setup only | 12 | 0 + 1 | 12.1 s |
| `w1_w3` of each model | 3 | 3 + 5 | 5.3 s |
| `w1_w3` of each model | 3 | 1 + 2 | 4.6 s |
| `wq_wo` of each model | 3 | 1 + 2 | 2.2 s |

About 74 % of the full measurement is per-shape setup in the test harness (host-side generation, packing and
upload of up to 58.7 million 4-bit weights per shape, graph build, pipeline creation): 0.26 to 2.2 s per shape,
growing with N x K. The 84 extra runs cost 4.2 s. Process start is negligible and one shader module is compiled
per process, so neither is worth optimising. A run without warm-up reads 1.5 to 2.2 times too slow, so the
screening mode keeps one warm-up run.

### Screening mode against the full measurement (`results/780m/space/validate/`)

The first 64 configurations of the random sample (a prefix of the draw, so itself a uniform sample), each
measured in full (12 shapes, 3 + 5 runs; 24.6 s a configuration on this sample) and in five candidate modes.
Spearman rank correlation of the mode's kernel time with the model's linear time per layer from the full
measurement (2 `wq_wo` + 2 `wk_wv` + 2 `w1_w3` + `w2`), per model (8B / 1B / 3B), and how many of the full top 10
are in the mode's top 10 and top 20:

| mode | s per configuration | rho, 8B / 1B / 3B | top 10 in top 10 | top 10 in top 20 |
|---|---:|---|---|---|
| `wq_wo`, 1 + 1 runs | 2.4 | 0.996 / 0.995 / 0.990 | 10 / 10 / 9 | 10 / 10 / 10 |
| **`wq_wo`, 1 + 2 runs** | 2.7 | 0.995 / 0.996 / 0.997 | 10 / 9 / 9 | 10 / 10 / 10 |
| `wq_wo`, 1 + 3 runs | 3.1 | 0.995 / 0.997 / 0.997 | 10 / 10 / 9 | 10 / 10 / 10 |
| `w1_w3`, 1 + 2 runs | 6.5 | 0.995 / 0.998 / 0.998 | 10 / 9 / 9 | 10 / 10 / 10 |
| `wk_wv`, 1 + 2 runs | 1.5 | 0.992 / 0.992 / 0.996 | 10 / 10 / 10 | 10 / 10 / 10 |

Against every single shape of the full measurement the screening shape has rho 0.972 to 1.000 (lowest for `w2`,
the K = 8192 / 14336 shape). All 64 configurations dispatched their own kernel on all 12 shapes; the kernel-time
coefficient of variation over the 5 timed runs has median 0.5 % and 90th percentile 2.2 %.

Chosen: `--op=wq_wo --runs=1,2` (N = K = 2048 / 3072 / 4096). `wk_wv` is cheaper and ranks as well here, but its
N = 512 / 1024 is the one shape where the wide tiles were measured to lose in round 1, so it is not used alone.
What the agreement does not show: the sample spans a 10x range of kernel times, so a high rho is expected; the
top-10 overlap is the sharper check, and the confirmation stage measures the leaders on all 12 shapes anyway.


## Done in this round

| step | session | result |
|---|---|---|
| baseline re-check, cool start (40 C) | `t1-recheck` | `dev/1.5` 2698.29 / 2544.10 (1B 4w / 8da4w), 1147.98 / 1043.30 (3B), 523.65 / 488.32 (8B) tok/s: within 0.8 % of `sarc-1.5-e2e-benchmark/results/cells.csv` (2698.29 / 2544.1, 1151.21 / 1051.33, 525.80 / 489.13). `780m-refine3` over `dev/1.5`: +4.98 / +11.19, +3.84 / +11.98, +3.99 / +12.44 %, geomean +8.00 %. The flagged 8B 4w cell, re-measured cool: **+3.99 %** (523.65 -> 544.54). Binaries of 2026-10-03, not rebuilt. |
| A/A, warm start | `t2-aa` (superseded) | started 90 s after the re-check (idle reference 44 C, run starts at 49 C): a warm-up, not used as the noise floor. Geomean +0.01 %, cells within +-0.14 %. |
| A/A, cool start (42 C, the idle temperature after 30 min) | `t3-aa` | **noise floor**: geomean +0.01 %, cells -0.23 to +0.19 %, repeat spread at most 0.52 %, clock 2797 to 2800 MHz, next token SAME in all cells on both prompts. Per cell (tok/s, 4w / 8da4w): 1B 2828.73 / 2828.73, 3B 1192.08 / 1172.97, 8B 544.10 / 549.50. |

### The random sample: 2,500 4w configurations screened (`results/780m/space/rand/`)

Seed 20261004, `wq_wo` of each model with 1 + 2 runs, 15:29 to 17:40 PDT. After the first 136 configurations the
cooling limit between runs was raised from 55 to 62 C (it cost 4.5 s a configuration); the 64 validation
configurations re-screened at 62 C differ from their 55 C values by a median of +0.08 % (5th to 95th percentile
-2.9 to +3.3 %, which is the repeatability of this mode). 2,444 configurations were timed; no process failed and
no Vulkan context was lost. The other 56 (2.2 %) kept the table kernel on all shapes: the selector's shared-memory
model (`impl/sarc/Select.cpp`, release zone) refuses them although the shader fits (it does not know the band
drain or `FRAG_LAYOUT`); reaching them needs a hook.

The space is mostly bad and sharply peaked: the median configuration is 2.23x slower than the sample's fastest,
the slowest 248x; 0.1 % are within 2 % of the fastest, 0.3 % within 10 %, 10 % within 1.5x.

Fastest of the sample (geomean kernel time of the three `wq_wo` shapes): `bx_t128x256k32g42s32f32xpiw` 3612 us,
`t128x128k32g42s32f32ciw` 3659 us (the shipped tile + `IMG_W`), `bx_t128x64k16g22s32f32xpiw` 3687 us,
`t256x128k32g42s32f32bbt` 3718 us. The shipped tile and the refine3 wide tile themselves are measured in the same
mode in refinement round 1 (they are not in the sample).

Parameter importance (`importance/importance.csv`, `levels.csv`; preliminary, the local effects around the optimum
come from the refinement). "Range" is between the geomean times of the best and the worst level over the whole
sample; "unique R^2" is what the additive model of log time loses without the parameter; "best-of-level spread"
is how much slower the fastest sampled configuration of the worst level is than the fastest of the best level
(what the parameter still costs when everything else is chosen well; it is a minimum over a sample, so about 3 %
of it is noise):

| parameter | range over the sample | unique R^2 | best-of-level spread | fastest level / levels of the fastest 5 % |
|---|---:|---:|---:|---|
| `WG_TILE_M` | 380 % | 0.254 | 47 % | 128 (63 % of the fastest 5 %), then 64, 256; 32 never |
| `WG_TILE_N` | 272 % | 0.154 | 35 % | 128, 64, 256 all present; 32 never |
| `SG_GRID_Y` | 118 % | 0.108 | 23 % | 2, then 4, 1 |
| `SG_GRID_X` | 29 % | 0.100 | 16 % | 2 and 4 |
| ACC (accumulator) | 129 % | 0.080 | 52 % | fp32: 90 % of the fastest 5 %, best in all 21 geometry strata |
| `WG_TILE_K` | 144 % | 0.028 | 27 % | 32 and 16; 64 is 27 % behind at best |
| `SUBGROUP_SIZE` | 0.6 % | 0.013 | 19 % | 32 (84 % of the fastest 5 %): no effect on average, large near the optimum |
| CSH (drain) | 46 % | 0.002 | 38 % (full without pool) | in Ash, band and pooled full within 3 % of each other |
| `IMG_A` | 4.2 % | 0.000 | 6.9 % | off |
| `SH_F16V4` | 2.3 % | 0.000 | 6.5 % | off |
| `FRAG_LAYOUT` | 6.0 % | 0.000 | 4.4 % | off |
| `B_COLMAJOR` | 9.5 % | 0.000 | 2.9 % | either |
| `IMG_W` | 4.2 % | 0.000 | 2.9 % | either |
| texel-wise staging (`bx`) | 1.2 % | 0.000 | 1.3 % | either: does not matter |
| derived: MMAs per subgroup, (M/Y/16) x (N/X/16) | 4485 % | (eta^2 0.605 alone) | | 8 and 16 (4 to 32 usable); 64 and more are 11 to 25x slower on average |
| derived: threads per workgroup | 340 % | (eta^2 0.062) | 46 % | 256, then 128 |

One derived quantity explains more of the variance (60 %) than all 14 yaml parameters additively (63 %): how
many 16 x 16 MMA tiles one subgroup owns. That is why M, N and the grid look important and interact.

Pair interactions (`importance/pairs.csv`; gain in R^2 over the additive model, noise level about 0.001):
tile M x ACC 0.038, tile N x ACC 0.034, tile K x ACC 0.032, M x grid Y 0.021, M x N 0.019, M x K 0.016,
N x K 0.014, K x CSH 0.007, N x grid X 0.006. No pair of two boolean options is above 0.003.

The staged plan's independence assumption, checked (`importance/independence.txt`): it does not hold. The
additive model explains 63.1 % of the variance of log time; geometry x geometry pairs add 14.4 points,
geometry x option pairs add 12.3 points, option x option pairs 1.2 points (all pairs: 87.9 %). The options do not
interact with each other, but they do interact with the tile geometry about as strongly as the geometry
parameters interact among themselves. For the accumulator the interaction changes the size of the effect, not
its sign (fp32 is best in every stratum). For the drain mode and the boolean options the best level changes
with the geometry (drain: a different best level in 10 of 21 strata; `FRAG_LAYOUT` 5, `IMG_A` 5, `IMG_W` 8), so
"best geometry first, then best flags on it" could have missed combinations; at the size of those effects
(1 to 10 %) that is exactly the margin this campaign is looking for.

## Part 1: parameter space (static count, `tools/enum_space.py`)

| family | combinations | device | flags | geometry | shared memory | shape | survivors | tile geometries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4w | 75,497,472 | 64,487,424 | 10,515,456 | 292,692 | 76,188 | 0 | 125,712 | 590 |
| 8da4w zpg | 165,888 | 133,632 | 7,632 | 22,002 | 384 | 0 | 2,238 | 594 |
| SDPA QK^T | 9,216 | 3,840 | 0 | 2,832 | 820 | 0 | 1,724 | 628 |
| SDPA attn*V | 4,608 | 1,920 | 0 | 2,023 | 28 | 167 | 470 | 446 |

Columns "device" to "shape" count the combinations removed by each static rule, in that order.
The rules replay without a rejection on the 64 existing variants this device runs (`enum_space.py --check`).

Plan (`tools/plan_space.py`; `results/780m/space/sweep-manifest.csv`): 8da4w (2,238), QK^T (1,724) and attn*V
(470) in full, batches b00 to b07. The 4w rows of that manifest (the staged plan, batches b07 to b10) are not run.

Timing sample, 30 configurations drawn at random (`results/780m/space/sample30-results.csv`, 20:21 to 20:33 UTC):

| family | sample | s per configuration | planned | projected |
|---|---:|---:|---:|---:|
| SDPA (one QK^T and one attn*V token per run: 8 correctness cases, then timing) | 10 runs, 20 tokens | 12.5 per run | 1,724 runs | 6.0 h |
| 8da4w | 10 | 15.8 | 2,238 | 9.8 h |
| 4w, staged (dropped, see the owner decision) | 10 | 18.4 (15.3 to 26.4) | 1,642 | 8.4 h |
| total without 4w | | | | 15.8 h, plus cooling waits and the 06:40 to 07:40 pause |
| 4w, all 125,712 survivors | | 18.4 | 125,712 | **642 h (26.8 days): over the 48 h limit, not started** |

The first sample is in `<artifacts>/superseded/sample30-kernel-name-not-recognised/`: the linear family names
lacked `linear_`, which the microbench needs to report the kernel time.

Found by the sample: `CSH_BAND` 4w tiles whose `SG_GRID_Y * 16 * WG_TILE_N * 2` reaches 64 KiB are refused by
the selector's shared-memory check (`impl/sarc/Select.cpp`, release zone; it does not know the band drain), so
they keep the table kernel and are recorded with `dispatched=0`.

Linear configurations are timed with the microbench's correctness gate off (the gate only knows the table
kernels' small shapes); correctness is checked in the repeat stage. SDPA configurations are checked in every run.

## Part 2: softmax (measured in the microbench; `raw/softmax/`, build `hook2`)

`sarc_sdpa_attn_weights_softmax` op time per layer at S = 2048, profile `780m-refine3` for QK^T and attn*V, two
runs each; all variants reached through `hooks/softmax-name-hook.patch` applied in a scratch tree:

| variant | what | 1B (32 heads) | 3B (24 heads) | 8B (32 heads) | extended correctness tier |
|---|---|---:|---:|---:|---|
| release | three loads of the row prefix, exp twice | 7.19 / 7.21 ms | 5.42 / 5.41 ms | 7.21 / 7.22 ms | 8 of 8 |
| `r1` | one load kept in registers, exp once | 5.94 / 5.95 ms | 4.46 / 4.46 ms | 5.94 / 5.93 ms | 8 of 8 |
| `r2` (dropped) | r1 + in-wave tree reductions, no shared memory | 5.95 ms | 4.45 ms | 5.93 ms | 8 of 8 |
| `r3` | r1 + zero fill bounded to the K-chunks the SARC attn*V kernels read | 4.41 / 4.43 ms | 3.32 / 3.30 ms | 4.45 / 4.42 ms | 8 of 8 |
| `m1` (measurement only, wrong results) | r3 without exp | 4.42 / 4.42 ms | 3.32 / 3.31 ms | 4.42 / 4.45 ms | 0 of 8, as intended |

- r3 is -38.6 % on the kernel (r1 -17.6 %). Removing exp entirely changes nothing (m1 = r3), and neither do the
  barriers (r2 = r1): the kernel is bound by memory traffic, about 284 MB per layer at about 64 GB/s with r3.
- r3 depends on its neighbour: the release softmax comment says attn*V stages every chunk below `context_len`,
  but the SARC attn*V kernel already stops at the chunk holding column `a + M - 1 + input_pos` of its row tile
  (`useful_chunks`), so zeros past that chunk are never read. r3 keeps the full fill unless S and `input_pos` are
  multiples of 256 and head_dim a multiple of 64. With an attn*V that reads whole rows (upstream kernel,
  `ET_VK_DISABLE_COOPMAT`) r3 would be wrong; the microbench pairing check now fails such a pairing.
- Expected end to end (16 / 28 / 32 layers): about -44 ms (1B), -59 ms (3B), -89 ms (8B).

## Next

1. Validation result -> screening mode; build the other four sample batches; screen the 2,500 configurations.
2. Gate candidate 7 (softmax r3) from a cool start: e2e session first, then the SDPA tiers, `verify.sh`, traces.
3. Resume the SDPA and 8da4w enumerations; parameter importance, refinement and confirmation for 4w.

## Blocking

Nothing.
