# STATUS: 780M prefill campaign, round 2 (parameter space + beyond)

Updated 2026-10-04 23:00 PDT (2026-10-05 06:00 UTC). Parent for this round: profile `780m-refine3` (build `topic-r1`).
Artifacts: `rocky-ryzen:~/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-04/` (new raw data) and
`.../780m-prefill-refine-2026-10-03/` (earlier builds and sessions).

## State

| item | state |
|---|---|
| candidate 7 (softmax r3) | **complete**: gate passed, bit-identical to its parent, six cells +2.39 to +6.78 %, geomean **+4.10 %** over `780m-refine3` |
| candidate 8 (fused SDPA kernel, Part 2) | **e2e session done: +7.84 to +20.14 %, geomean +13.16 % over candidate 7**, all 60 runs valid; the rest of the gate is running (`chain7.sh`); next token differs in 1 of 12 checks, so it is judged by the reference-error rule, whose evidence is being collected. Not accepted yet |
| igpu-roofline `fast` plan | finished 20:15 PDT; matrix roofs 14.766 TFLOP/s (fp16 -> fp32) and 14.379 TOP/s (int8) |
| 4w refinement round 1 | finished 21:41 PDT: 991 neighbours screened; nothing beats the sample's best by more than the screening scatter; the confirmation with the full measurement is queued |
| 8da4w screen, 4w / 8da4w confirmation, QK^T / attn*V enumeration | queued behind the candidate-8 gate (`chain6d.sh`) |

Chained over the three sessions (`t1-recheck`, `c7-softmax-r3`, `c8-fused3`; not a direct measurement, that comes
at the end): candidate 8 is about +27 % geomean over `dev/1.5` (1B 4w 2698 -> 3625 tok/s, 8B 8da4w 488 -> 606).

## Running now (detached chains; nothing needs attention)

1. `chain7.sh`, the gate of candidate 8. Done: e2e session; SDPA tiers `all` and `extended`, 12 passes each, 0
   mismatches. Running: tier `full` (6 of 12 passes done, about 3 min each). Then `verify.sh`, traces, tier
   `peaked` (12 passes), the error against the fp64 reference for both arms on the `extended`, `peaked` and `full`
   tiers, steady-clock kernel timing, and the real-text logits probe (32 prompts, four arms, six cells). Until
   about 00:30 PDT.
2. `chain6d.sh` (replaces `chain6.sh` .. `chain6c.sh`, which were only waiting), after that:
   - the SDPA sweep binaries again, so that the SDPA suite can be timed at a steady clock (below);
   - 4w confirmation (`tools/confirm.sh`): every configuration within 8 % of the fastest of the random sample and
     of refinement round 1 (96 configurations), full measurement; the 10 fastest per shape 5 times; 12
     production-diff passes each. About 2.5 h;
   - 8da4w: validation of the screening mode on 64 configurations, then all 2,238 survivors screened (`wq_wo`,
     3 + 3 runs), then the same confirmation. This changes how each 8da4w configuration is timed, not which
     ones: say so if the full measurement of all 2,238 is wanted instead (9.8 h);
   - the QK^T / attn*V enumeration (1,724 runs, 20 warm-up + 8 timed runs each, about 7 h, pausing 06:40 to
     07:40), into `raw/space/results-steady.csv`.

### What the clock does to the microbench (found 2026-10-04, affects how the sweeps are read)

The GPU clock idles at 800 MHz and needs sustained load to reach 2800 MHz. In `llama_main` it is at 2800 MHz
throughout; in the microbench it is not always:

- SDPA suite, 3 warm-up + 5 timed runs of 3 to 15 ms each: the timed runs are still on the rising clock. The
  coefficient of variation over the 5 runs is 7 to 9 % (median of the 330 enumeration rows measured so far) and a
  compute-bound kernel reads up to 1.7 times slower than at 40 + 10 runs, while a memory-bound one barely moves.
  `ET_VK_SDPA_PERF_RUNS` (added) sets the run counts; the enumeration will use 20 + 8. The 55 runs done with 3 + 5
  stay in `raw/space/results.csv` and are not used for ranking.
- Linear suite, full measurement (3 + 5): steady, coefficient of variation 0.5 to 1 %. Longer runs only heat the
  device (300 + 5: +5 %).
- Linear suite, screening mode (1 + 2): the first timed run is still rising (12 to 15 % between the two timed
  runs). On 14 configurations across the ranking the screening value is 3 to 13 % above the steady one, by a
  different amount per configuration; the order is kept overall (Spearman 0.987) but not among the first five
  (`results/780m/space/clock-check.csv`). The random sample and refinement round 1 were screened this way, so their
  single values scatter by about +-5 %: fine for the parameter-importance table (averages over many
  configurations), not for choosing among near-ties. Therefore the confirmation takes everything within 8 % of the
  fastest (not the 10 best by rank), and the 8da4w screen uses 3 + 3 runs.

### Pauses and overlaps (for whoever audits the timings)

- Builds are CPU work and the screening sweep does not look at `PAUSE`, so the sweep process was stopped
  (`SIGSTOP`, between two runs) for each build and continued afterwards: 03:33:14 to 03:34:33, 03:41:31 to
  03:41:33, 03:41:53 to 03:43:11, 03:49:09 to 04:05:35 and 04:13:42 to 04:13:45 UTC (`logs/sweep-pauses.txt`).
- Candidate-8 correctness and kernel-timing runs took the gpu-lab lock between sweep runs from 03:28 to 04:16 UTC.
  They cannot overlap a sweep run, but they warm the device, and the sweep waits until it is back under 62 C
  before each run: that is why round 1 is slow, not a change in what it measures.
- During the candidate-8 gate: a build (05:29 to 05:34 UTC) and correctness / kernel-timing runs of the one-pass
  variant and the four-arm logits run (05:35 to 05:57 UTC) were slotted in while the gate ran its SDPA
  correctness tiers, after the timed session had ended (05:28 UTC). Nothing ran next to the session.
- One build of mine started while the roofline was still running and was stopped after about 15 s (03:05:28 to
  03:05:42 UTC, `-j6`). It overlapped the confirm runs of `mem_write`, `mem_copy`, `mem_triad` and
  `sharedbw_fp16`. The matrix roofs used below were confirmed earlier in the run and repeat within 0.1 %; the
  shared-memory fp16 read / write roofs show a 15 % repeat range and are not used.

## Roofs, re-measured (igpu-roofline `fast` plan, 2026-10-04 19:46 to 20:15 PDT)

Run `~/igpu-roofline/campaigns/780m/2026-10-04-fast-prefill-refine2` (code `dbdd193e`, RADV PHOENIX, Mesa 25.2.7,
GPU clock DVFS-governed, not pinned). Confirmed medians (3 repeats each):

| roof | value | repeat range | earlier value (`sarc-1.5-e2e-benchmark/evidence/roofline.md`) |
|---|---:|---:|---:|
| `matrix_fp16_fp32` (4w linear, SDPA) | 14.766 TFLOP/s | 0.1 % | 14.772 |
| `matrix_int8` (8da4w linear) | 14.379 TOP/s | 0.1 % | 14.393 |
| `global_read` / `global_write` / `global_copy` | 86.6 / 77.6 / 71.5 GB/s | 0.0 to 0.1 % | |
| MMA fed from shared memory, one tile pair per 1 / 2 / 4 / 8 multiply-adds | 2.42 / 4.82 / 9.65 / 14.73 TFLOP/s | | |
| MMA fed from a cache-resident buffer, same | 4.01 / 7.89 / 10.75 / 12.80 TFLOP/s | | |

The last two rows are what shaped candidate 8: operand loads, not the matrix unit, limit a kernel that loads a
tile pair for fewer than about 8 multiply-adds, and contiguous tiles in a cache-resident buffer load faster than
tiles in shared memory.

## Candidate 7 (softmax r3, through the uncommitted hook): ACCEPTED, bit-identical, +4.10 % geomean

Session `c7-softmax-r3` (`results/780m/sessions/c7-softmax-r3/`, raw data `<artifacts>/stage/c7-softmax-r3`),
started 18:13 PDT at 45 C after the full 30 min wait. Both arms are the same binary (build `hook3` = the branch +
`hooks/softmax-name-hook.patch` in a scratch tree) with `ET_VK_SARC_DEV_PROFILE=780m-refine3`; the candidate arm
adds `ET_VK_SARC_780M_SOFTMAX=r3`. Tok/s, median of 5 valid runs per arm, arms interleaved:

| cell | parent (`780m-refine3`) | candidate (+ softmax r3) | gain | repeat spread parent / candidate | next token (2048 / unaligned prompt) |
|---|---:|---:|---:|---|---|
| 1B 4w | 2828.73 | 3020.65 | **+6.78 %** | 0.28 / 0.29 % | SAME / SAME |
| 1B 8da4w | 2824.83 | 3011.76 | **+6.62 %** | 0.28 / 0.15 % | SAME / SAME |
| 3B 4w | 1190.70 | 1231.51 | **+3.43 %** | 0.12 / 0.18 % | SAME / SAME |
| 3B 8da4w | 1170.29 | 1206.12 | **+3.06 %** | 0.40 / 0.35 % | SAME / SAME |
| 8B 4w | 541.23 | 554.41 | **+2.44 %** | 0.71 / 0.11 % | SAME / SAME |
| 8B 8da4w (repeat, 21:46 PDT, start 44 C) | 548.47 | 561.56 | **+2.39 %** | 0.56 / 0.36 % | SAME / SAME |

Geomean of the six cells **+4.10 %**, each outside the +-2 % band and far above the A/A floor (+-0.23 %).
The parent arm agrees with `t3-aa` within 0.5 % (2828.73 / 2828.73, 1192.08 / 1172.97, 544.10 / 549.50).
The 8B 8da4w cell is from the repeat (`sessions/c7-softmax-r3/repeat-8b8da4w/`, both arms interleaved, 5 valid
runs each, cool start); in the first session 8 of its runs carry `other_gpu_process` (below) and its 4 valid runs
per arm read 548.0 to 549.1 against 562.0 to 562.5 tok/s, the same gain.

Gate items (all with the candidate's environment):

| item | result |
|---|---|
| `verify.sh --models 1b,3b,8b --schemes 4w,8da4w --pdiff` | correctness rc = 0; 12 of 12 production-diff cases ALL PASSED; default vs tiled SAME on the real-text and the unaligned prompt; decode 31 tokens; `linear <scheme> rc=1` as on the parent. Status lines identical to the parent's (`s8-r3final/verify.out`) |
| next token parent vs candidate | SAME in all six cells on both prompts |
| SDPA tiers, 12 passes each | `all` 4 of 4, `extended` 8 of 8, `full` 4 of 4 in every pass, 0 mismatches, `pairing=ok` in every case; control with the table kernels 1 pass each, same |
| traces (one warm run per arm) | softmax 115.1 -> 70.5 ms (1B), 150.2 -> 93.3 ms (3B), 229.1 -> 142.3 ms (8B); QK^T and attn*V unchanged |

Where the time is after candidate 7 (`sessions/c7-softmax-r3/trace/families.csv`, candidate arm, ms; 4w / 8da4w):

| family | 1B | 3B | 8B |
|---|---|---|---|
| linear GEMM | 373 / 351 | 1081 / 1030 | 2799 / 2625 |
| softmax | 70 / 70 | 93 / 93 | 142 / 142 |
| QK^T | 47 / 46 | 129 / 127 | 191 / 187 |
| attn*V | 42 / 42 | 77 / 78 | 119 / 119 |
| elementwise (`mul`, `sigmoid`, `add`; upstream) | 68 / 68 | 129 / 130 | 244 / 243 |
| copy / view / other (upstream) | 61 / 36 | 135 / 86 | 229 / 137 |
| 8-bit activation quantize (upstream) | - / 43 | - / 101 | - / 167 |
| RMSNorm + RoPE (upstream) | 14 / 14 | 40 / 41 | 60 / 61 |
| total dispatch | 680 / 675 | 1694 / 1695 | 3797 / 3693 |

Single traced runs: the linear rows of the two arms differ by up to 3 % (8B 4w 2716 against 2799 ms) without any
linear kernel having changed, so differences of that size between sessions are not resolved by a trace.

Why the cell was repeated: in the first session its runs 4 to 8 (eight runs) are marked `other_gpu_process`. The process the session saw was pid 576199,
`/bin/bash`: an interactive monitoring shell of this campaign whose command line contained the name of the
runner binary as text. `e2e5.sh` looks for other GPU processes with `pgrep -af`, which matches full command lines,
so the shell was counted for the 6 minutes it was alive. No GPU process ran (the lock holder was the session; the
flagged runs read 548.0 to 549.4 and 561.9 to 562.5 tok/s, the same as the valid ones). The flag is not removed
and the tool is not changed; the cell is repeated instead. Pitfall for anyone watching a session: do not put the
runner or microbench binary names in a shell command while `e2e5.sh` is running.

Bitwise comparison against the parent (`results/780m/softmax/bitwise-c7.txt`): the raw fp16 SDPA output of
every correctness case of the tiers `all`, `extended`, `peaked` and `full` (21 cases, up to 16.8 MB each, the three
production head configurations at S = 2048 among them) is **byte-identical** between the parent kernels and the
parent kernels with softmax r3. Candidate 7 therefore does not use the near-tie or the reference-error rule.

## Candidate 8 (Part 2): fused SDPA kernel `sarc_dev_780m_sdpa_fused3`, session done, gate running

After candidate 7, QK^T + softmax + attn*V are still 160 of 680 ms on 1B, 299 of 1694 ms on 3B and 452 of
3797 ms on 8B, and all three kernels are bound by the traffic of the S x S attention matrix (QK^T writes it,
the softmax reads and rewrites it, attn*V reads it; about 550 MB per layer on 1B by count of bytes), not by
arithmetic. The fused kernel never writes that matrix. For a block of query rows of one head it walks the context
in blocks, twice: pass A computes the scores (fp32 accumulate, scaled, rounded to fp16, exactly as the QK^T kernel
does) and keeps the row maxima; pass B computes the scores again, e = exp(score - max) in fp16, the row sums in
fp32 and `acc += e V` in fp32; the output is `acc / sum`.

### Gate so far (session `c8-fused3`, `results/780m/sessions/c8-fused3/`)

Started 21:59 PDT at 44 C. Both arms are the same binary (build `fused5` = the branch + both hook patches in a
scratch tree) with `ET_VK_SARC_DEV_PROFILE=780m-refine3 ET_VK_SARC_780M_SOFTMAX=r3` (= candidate 7); the candidate
arm adds `ET_VK_SARC_780M_SDPA_FUSED=fused3_d64_t32x32g11s32rk,fused3_d128_t16x64g11s32rk`. Tok/s, median of 5
valid runs per arm, arms interleaved; recomputed from `runs.csv`:

| cell | parent (candidate 7) | candidate 8 | gain | repeat spread parent / candidate | next token (2048 / unaligned prompt) |
|---|---:|---:|---:|---|---|
| 1B 4w | 3020.65 | 3624.78 | **+20.00 %** | 0.15 / 0.18 % | SAME / SAME |
| 1B 8da4w | 3011.76 | 3618.37 | **+20.14 %** | 0.15 / 0.18 % | SAME / SAME |
| 3B 4w | 1232.25 | 1376.34 | **+11.69 %** | 0.06 / 0.20 % | SAME / SAME |
| 3B 8da4w | 1201.88 | 1344.71 | **+11.88 %** | 1.24 / 0.79 % | SAME / SAME |
| 8B 4w | 556.07 | 601.12 | **+8.10 %** | 0.46 / 0.59 % | SAME / SAME |
| 8B 8da4w | 561.71 | 605.74 | **+7.84 %** | 0.19 / 0.21 % | **DIFFER** / SAME |

Geomean **+13.16 %**; 60 timed runs, none rejected; median clock 2783 to 2800 MHz; start temperature 44 to 50 C,
peak 93 C. The unaligned prompt (1972 tokens) is not served by the fused kernel, so its checks compare the same
kernels in both arms.

The one differing item, 8B 8da4w on `prompt_2048.txt` (2048 times the token " the"): the model is undecided there.
Next-token logits of the top candidates in the four arms (`results/780m/probe/c8-differ-8b-8da4w-prompt_2048.csv`):

| token | parent tiled | parent default | candidate tiled | candidate default |
|---|---:|---:|---:|---:|
| 118 (a byte token) | 11.461 (p 0.0126) | 11.500 (p 0.0129) | 11.125 | 11.156 |
| 53 `V` | 11.297 | 11.234 (p 0.0099) | 11.422 (p 0.0119) | 11.367 (p 0.0113) |
| 5531 `cal` | 11.047 | 11.180 | 11.102 | 11.016 |
| top-2 margin | 0.164 | 0.266 | 0.297 | 0.211 |

The parent's top token has probability 1.3 % and leads by 0.27 logit (0.16 in its tiled arm); the candidate moves
token 118 by -0.34 and token 53 by +0.13. Largest logit difference over the vocabulary: candidate default against
parent default 0.66, parent tiled against parent default 0.83; KL 8.2e-3 against 4.6e-3 nat. This is item 1 of
the evidence; the candidate is not accepted on it alone (the error against the reference on the production shapes
and the real-text comparison decide, both still running).

Done so far in the gate: SDPA tiers `all` (4 of 4) and `extended` (8 of 8), 12 passes each, 0 mismatches, the
fused kernel the only SDPA kernel dispatched, `pairing=ok`; controls with the table kernels 1 pass each; tier
`full` 6 of 12 passes, 4 of 4 each.

### Structures

Three structures were built and measured; the third is the candidate.

| structure | what a workgroup is | kernel time per layer, 1B / 8B head configuration |
|---|---|---|
| three kernels (candidate 7) | | 9.97 / 13.52 ms |
| `fused`: K and V staged in shared memory, 4 to 6 barriers a block | 4 to 8 subgroups, 64 or 128 rows | 3.84 / 11.7 ms |
| `fused2`: staging double-buffered (1 barrier a block), score rows private to a subgroup, V transposed | 2 to 8 subgroups | not faster than `fused` (first 5 runs: 6.2 against 5.9 ms on 1B) |
| `fused3`: one subgroup, no staging, no barrier; Q tiles in registers, K from the cache buffer, V from a transposed copy | 1 subgroup, 16 or 32 rows | 2.55 / 9.95 ms |
| `fused3`, packed (`...rk`): K and V read from tile-packed copies in which a 16 x 16 operand tile is 512 contiguous bytes | same | **2.80 / 5.57 ms, copy pass included** (2.46 / 4.84 ms without it) |

(Steady GPU clock: 40 warm-up and 10 timed runs, median; `results/780m/fused/kernel-time-steady.csv`. With the
default 3 + 5 runs of the microbench the clock is still rising and the fused kernels read up to 1.7 times slower:
`kernel-time-development.csv`, where only values in the same column compare.)

What the measurements say about why:

- The disassembly shows two 128-bit shared-memory loads per lane for every operand tile; `fused` and `fused2` load
  1.0 to 1.5 tiles per multiply-add, which the roofline table above puts at a small fraction of the matrix roof.
  Removing the row padding from the shared tiles made `fused2` 2.6 times slower (15.5 against 5.9 ms): shared
  memory loads were the limit, not the barriers.
- Keeping the Q tiles in registers and reading K straight from the buffer halves the loads per multiply-add.
  Reloading Q from the buffer for every block instead costs 2.2 to 3.0 times (1B 10.2 against 4.6 ms, 8B 44.7
  against 15.0 ms, first-5-run values): loads from the cache buffers are only cheap when a tile is contiguous. In
  the caches a 16 x 16 tile is 16 runs a row stride apart; the packed copies make it one run, which is what
  brought head_dim 128 from 9.95 to 4.96 ms (16 x 16 tile, steady clock).
- The packed `fused3` runs 3.1 M (1B) and 6.3 M (8B) multiply-adds a layer in 2.46 and 4.84 ms: about 71 % of the
  matrix roof. Without pass A the same kernel takes 1.58 / 3.12 ms (measurement-only variant `...rkm1`): a
  one-pass (running-maximum) form would save at most that, about 2 % end to end; not attempted.
- The copy pass (`sarc_dev_780m_sdpa_kvt`, 4 to 16 MB a layer) costs 0.35 (1B) to 0.7 ms (8B), three times what
  the bytes cost at the copy roof; a 16 x 4 workgroup instead of 8 x 8 is slower (2.36 against 2.27 ms on 1B);
  not tuned further.

### One-pass variant (`...rko`, built and measured, a later candidate)

`ONLINE` keeps a running row maximum instead of pass A: when a block raises the maximum of a row, that row's
accumulator and sum are scaled by exp(old - new) (an element-wise divide of the accumulator tiles by a divisor
tile from shared memory, only in subgroups where some row moved). Build `fused6`: correct on tiers `all`,
`extended`, `peaked` and `fused` (one pass each, 0 mismatches), error against the fp64 reference equal to or
slightly below the two-pass kernel's. Kernel time per layer, copy pass included, steady clock: 1B 2.80 -> 2.27 ms,
3B 4.33 -> 3.22 ms, 8B 5.54 -> 4.04 ms. End to end that is about +1.5 to +2 %, so it is not gated alone: it goes
into the next candidate together with the Part 1 winners per shape.

Variants chosen (`ET_VK_SARC_780M_SDPA_FUSED=fused3_d64_t32x32g11s32rk,fused3_d128_t16x64g11s32rk`): for head_dim
64, 32 rows x 32-column blocks (2.80 ms; 16 x 32 reads 2.77 to 2.79 ms, a tie); for head_dim 128, 16 rows x 64
(8B 5.56 to 5.58 ms, 3B 4.32 ms; 16 x 16, 16 x 32 and 32 x 32 are 1.5 to 4 % behind on both models, twice).

- Files: `glsl/sarc_dev/sarc_dev_780m_sdpa_{fused,fused2,fused3,vt,kvt}.{glsl,yaml}`,
  `impl/sarc_dev/Sdpa780mFused.cpp`, the 780m block of `Overrides.cpp`.
- It needs `hooks/sdpa-fused-hook.patch` (about 70 added lines in the release zone and `SDPA.cpp`, not applied on
  this branch; described in `hooks/README.md`): nodes appended after the three SDPA nodes, and an empty dispatch
  for those three when the fused node serves the call. It serves prefill calls whose S and `input_pos` are
  multiples of the row tile and the context block; decode, unaligned prompts and `ET_VK_DISABLE_COOPMAT` keep the
  three kernels. It also serves aligned lengths the 128-row QK^T tile does not (64, 192, ...), which the stock path
  runs on the tiled kernels.
- It changes the arithmetic (the three kernels round e / sum to fp16 before attn*V; this kernel rounds e and
  divides the fp32 accumulator), so it is judged by the reference-error rule of the second owner decision.
- Test additions (`test/sarc_dev/test_llama_microbench.cpp`, no tolerance and no existing case changed): the fused
  kernel is recognised by the SDPA cases (it must then be the only SDPA kernel dispatched);
  `ET_VK_SDPA_ERROR_REPORT=1` prints rms and maximum error against an fp64 reference computed from the fp16
  inputs; tier `peaked` (Q scaled by 8, so a few context positions carry most of a row's weight; the existing
  tiers have near-uniform attention); tier `fused` (S = 32, 64, 192, 320, shapes only the fused kernel takes);
  `ET_VK_SDPA_PERF_RUNS` for the steady-clock timing. `test/sarc_dev/probe/logits_probe` writes the next-token
  logits of token prompts, for the real-text comparison (`tools/probe_{prompts,run,compare}.*`).

Results so far (one pass each unless said; `<artifacts>/fx/`, `<artifacts>/stage/c8-fused3/`):

- Correctness, chosen variants: tiers `all` 4 of 4, `extended` 8 of 8, `peaked` 5 of 5; tier `fused` 12 passes,
  5 of 5 each; 0 mismatches, the fused kernel the only SDPA kernel dispatched.
- Error against the fp64 reference, three kernels -> fused: rms 1.9e-5 to 4.6e-5 -> 0.95e-5 to 2.4e-5 on the 8
  `extended` cases (about half), maximum lower in all 8; on the 5 `peaked` cases rms 2.6e-4 to 2.7e-4 -> 2.2e-4 to
  2.4e-4, maximum lower in 4 and higher in 1 (`peaked_tiny_gqa_s256`: 1.74e-3 -> 1.98e-3; both arms round the
  same fp16 scores there). The `full` tier, which holds the production shapes the rule names, runs in the gate.
- End to end, one run per arm, not a session: 1B 4w 3020.65 -> 3624.78 tok/s (+20.0 %). Projected from the
  kernel times and the candidate-7 traces: about +20 % (1B), +12 % (3B), +8 % (8B).

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
  output against the parent is still owed. Candidate 8 (fused SDPA) changes the arithmetic and uses this rule.
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

### Refinement round 1 (`results/780m/space/refine1/`, 20:21 to 21:41 PDT, screening mode)

1,002 configurations that differ from one of the 22 best sampled ones in one parameter (and in two for the best
five); 991 dispatched their kernel on the three screened shapes. `tools/refine_summary.py`:

- Fastest by geomean of the three shapes: `t256x128k32g24s32f32bbt` 3533 us, `t256x128k32g18s32f32bbt` 3535,
  `bx_t128x256k32g42s32f32xpi` 3578, `t128x256k32g42s32f32cbt` 3598; the sample's best was 3612 and the
  refine3 tile `t128x256k32g42s32f32c` reads 3623 (+2.5 %). All of that is inside the screening scatter (+-5 %):
  the round found no geometry or option that clearly beats what the sample already had. The full measurement
  decides (queued).
- Fastest alternative level of each parameter, relative to its centre (median over the centres; below 1 = faster):
  accumulator 12.0 (fp32 everywhere), subgroup size 1.49, tile K 1.20, grid X 1.19, grid Y 1.17, tile M 1.14,
  tile N 1.14, `FRAG_LAYOUT` 1.06, `B_COLMAJOR` 1.02, texel-wise staging 1.01, drain mode 1.01, `IMG_W` 1.00,
  `IMG_A` 1.00, `SH_F16V4` 0.96 (4 centres only). So around the best configurations every geometry parameter is at
  a local optimum (the nearest alternative costs 14 to 49 %), and the options are flat within the scatter.

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

1. Finish the candidate-8 gate and record it under the reference-error rule (or reject it, with the numbers).
2. 4w and 8da4w confirmation -> the best configuration per shape, the local effect of each parameter around it.
3. Candidate 9: candidate 8 + the one-pass fused variant + the Part 1 winners per shape, gated against candidate 8.
4. QK^T / attn*V enumeration and their repeat stage (response surface; with the fused kernel these two kernels
   only serve the calls it does not take).
5. A direct session of the final configuration against `dev/1.5` and against `780m-refine3`.

## Blocking

Nothing.
