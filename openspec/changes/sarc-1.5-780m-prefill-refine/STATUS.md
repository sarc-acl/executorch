# STATUS: 780M prefill campaign, round 2 (parameter space + beyond)

Updated 2026-10-04 14:25 PDT. Parent for this round: profile `780m-refine3` (build `topic-r1`).
Artifacts: `rocky-ryzen:~/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-04/` (new raw data; moved there from a
private `/tmp` directory on 2026-10-04) and `.../780m-prefill-refine-2026-10-03/` (earlier builds and sessions).

## Running now (one detached chain, `chain1.sh`; nothing needs attention)

1. Building the other four batches of the random 4w sample (CPU only).
2. Screening the 2,500 sampled 4w configurations (`raw/rand/screen.csv`), about 3.2 h.
3. Gate of candidate 7 (softmax r3, `stage/c7-softmax-r3`) from a cool start: e2e session first, then the SDPA
   tiers (12 passes each), `verify.sh`, traces; about 2.5 h.
4. Then the QK^T / attn*V / 8da4w enumeration resumes by itself (`raw/space/results.csv`, 55 of 1,724 SDPA runs
   done, 8da4w not started; about 16 h, pausing 06:40 to 07:40).

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
