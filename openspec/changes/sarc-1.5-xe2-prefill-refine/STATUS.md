# sarc-1.5-xe2-prefill-refine: status

**2026-10-06 06:20 UTC. The sampled parameter search (review follow-up) is running detached on both B70 cards.
The host was rebooted at 05:57 UTC by someone else (not by this campaign); both queues died with it and were
restarted at 06:11 UTC; no measured row was lost (see "The reboot of 05:57 UTC"). Done: 4w sample and two
refinement rounds; 8da4w sample and refinement (nothing beats the candidate 2 tile). Running: 4w round c. Then
the 8da4w incumbents (requeued after an id collision of mine), attn*V and QK^T. Projected end of the search:
2026-10-06 21:30 UTC at the earliest, 2026-10-07 05:00 UTC if attn*V and QK^T run both further rounds. Since
the reboot the `llm-api-*` services are active again; none has taken a card (see "Needs the owner's
attention"). A coordinator hold is in place. Earlier results are unchanged: winner `xe2-refine2` (candidates 1
and 2), +57.5 % geomean over the parent (`s6-final`); candidate 1 is `ACCEPTED (reference-error rule, owner
decision 2026-10-04)`, not a plain pass; candidates 3 and 4 passed their gates with no measurable gain. Nothing
found by the search is adopted or gated yet.**

Branch `topic/xe2-prefill-refine`, parent `6a7cc8cc6` (head of `topic/780m-prefill-refine`). Host
`fedora-gpu-eval`. Card `b70-0` (guest PCI `0000:01:00.0`, Vulkan device 0, **`ETVK_DEVICE_INDEX=0`**,
deviceUUID = lock UUID `868023e2-0000-0000-0100-000000000000`) for every selecting measurement, session, gate
and reported number; the second B70 (guest PCI `0000:02:00.0`, Vulkan device 1, `ETVK_DEVICE_INDEX=1`, lock
`868023e2-0000-0000-0200-000000000000`) for cheap-mode screens only, since 20:34 UTC (owner decision
2026-10-05). ANV, Mesa 26.2.3. Nothing was run on the B580.

## What is running, and how to follow or restart it (for an actor with no memory of this session)

The reviewer of the first report asked for (1) the sampled parameter search of the owner decision "how large
parameter spaces are searched" with its parameter-importance table, (2) a resumable linear sweep, (3) a
corrected run count for `s2-c1`. Items 2 and 3 are done (`tools/screen.sh`, `screen13-resume-test`; the count
is in the candidate 1 section). Item 1 is running. When it has ended: gate what it found (below), update
`proposal.md` ("Not done" still says the search was not run) and this file, run `bash sarc/tools/check.sh
--no-build`, commit, `git push origin topic/xe2-prefill-refine`. No PR.

Two queues, `tools/sweep_queue.sh`, one per card, both started with `nohup setsid` (they survive the session):
`.artifacts/queue/` (card 0, every stage) and `.artifacts/queue1/` (card 1; it only receives the odd half of a
cheap screen from `tools/sweep_screen.sh`). Each runs the scripts in its `pending/` one at a time in name order
and moves them to `done/`. Check: `cat ~/hmz-sarc-xe2/.artifacts/queue{,1}/status`, `tail
.artifacts/queue/log/<job>.out`. Never start a GPU job, a build or `tools/test_guard.sh` beside them: the guard
of the running job treats it as a foreign GPU process and stops the queue (rc 76); put new work into
`queue/pending/<NNN>-<name>.sh`. Restart a dead queue with `nohup setsid bash tools/sweep_queue.sh >
.artifacts/queue/queue.out 2>&1 &` (card 1: the same with `XE2_CARD=1` and `queue1/`). Every runner is
resumable: it skips rows already in `raw/<name>/results.csv` and moves an interrupted attempt to
`raw/<name>/superseded/`. If the second card's queue is not running, `sweep_screen.sh` measures everything on
card 0. To end a queue when nothing is pending: `touch .artifacts/queue/STOP`. After a stage:
`bash tools/sweep_collect.sh <space>` copies the evidence into `results/xe2/sweep/`.

**The stop at 23:00 UTC and the restart.** `022-stage1-8da4w` had finished the validation (60 configurations
cheap and full x 2) and had begun the first split screen when card 0's runner ended with `SWEEP_ABORTED rc=76 at
200094`: its guard reported `1174050:gl.sh` as a foreign GPU process. That was this campaign's own `gl.sh` on
card 1, seen in the moment between its start and its registration in `run/card1.job` (its command line names
the test binary; the card test had not met this because both of its runners had one common parent). No foreign
process was involved, and no service came back. Fix (`27594fe8f`): each queue registers itself
(`run/queue<N>.top`), so everything a queue starts is known before it runs. The queue was restarted at 23:04 and
the stage re-entered as `023-stage1-8da4w`. **Kept:** everything measured, i.e. the sample, the sweep build
`sw1-8da4w`, all validation rows and the 47 screen rows of `022`; the runner skips them. **Discarded:**
nothing measured; the refused launch of `200094` left a log with no result, which is under
`raw/sw1-8da4w/superseded/interrupted-2026-10-05T23:04:50Z/`, and the configuration was measured after the
restart. Card 1 kept screening through the stop. One known waste: card 1 re-screens the 30 odd-position
configurations of the validation set, which card 0 already had (about 9 minutes; `sweep_merge.py` keeps card
0's rows).

**The reboot of 05:57 UTC.** `last -x` shows a system boot at 05:58; the journal of the previous boot ends at
05:55 with user sessions logging out from 05:52. This campaign did not cause or request it (no `sudo` beyond
`systemctl is-active`, no kernel setting, no profiler). The control session lost the host at the same time and
got it back at 06:01. Both queues and their runners were gone. State found: last rows of `sw2c-4w` at 05:56:52
(card 0) and of `sw2c-4w-c1` a few seconds earlier; pending jobs intact; `HOLD` absent; Vulkan device order and
UUIDs unchanged (checked with `vulkaninfo --summary`); models mounted; build image present. Stale
`run/*.job` / `run/*.top` files removed, queue 1 and then queue 0 restarted at 06:11 with the documented
command; `027-rounds-4w` and `sw2c-4w-c1` re-entered and continued after the last recorded configuration.
**Kept:** every recorded row and every build. **Lost:** the two configurations that were being measured when
the host went down (measured again; their partial logs, if any, are under `raw/<name>/superseded/`), and 14
minutes.

**Queues at 06:20 UTC, 2026-10-06** (seed 20261005 everywhere):

| job | what | state |
|---|---|---|
| `010`, `016`, `017` (4w) | sample 3000 drawn -> 2000 legal, validation, cheap screen; round 1 (326 neighbours, 23 finalists full x 2); round b (319 neighbours, 26 finalists full x 2) | done 2026-10-05 19:59 |
| `018`, `021` card test | `018` did not run the test (a timing assertion of the guard test itself); `021`: `test_guard.sh` 41 assertions pass, `card_test.sh` `CARD_SPLIT_OK` | done 20:34 |
| `019-rounds-4w` | rule check for round c on the cheap numbers only: not run; rule extended, see `027` | done |
| `020`, `022`, `023` stage 1 8da4w | sample, build (`020`, stopped by me after the build so that the card test runs first); validation and the stop of 23:00 (`022`); the split screen of the 2000 (`023` + `sw1-8da4w-c1`) | done 04:18 |
| `025-stage2-8da4w` (+ `sw2-8da4w-c1`) | correctness of the top, 323 neighbours cheap on both cards, 19 finalists full x 2 | done 05:33 |
| `0255-incumbent-8da4w`, `026-rounds-8da4w` | incumbents and neighbours with colliding ids: **superseded** (`results/xe2/superseded/sw2i-8da4w-id-collision/`); the round-b check that used them: no round | done 05:39, replaced by `028`, `029` |
| `027-rounds-4w` (+ `sw2c-4w-c1`) | round c for 4w (the full confirmation of round b moved 8B w2 by 3 %): 320 neighbours cheap on both cards, then correctness and finalists full x 2 | RUNNING (started 05:39, re-entered 06:11 after the reboot); about 07:30 |
| `028-incumbent-8da4w`, `029-rounds-8da4w` | the three incumbent 8da4w tiles and their 26 unmeasured legal neighbours with ids from 290000; then the round rule and the confirmation tables | pending, about 15 min + up to 4 h if a round runs |
| `030-stage1-av`, `035-stage2-av`, `036-rounds-av` | every legal attn*V configuration (enumerated) | pending, about 1.5 h + 1.5 h + up to 2.5 h |
| `040-stage1-qk`, `045-stage2-qk`, `046-rounds-qk` | every legal QK^T configuration (enumerated) | pending, about 5 h + 2 h + up to 3 h |

**Rate with two cards:** 8da4w screen 18.5 s per configuration on card 0 and 18.0 s on card 1, i.e. 9.1 s for
the pair (12 of the 2000 ran into the 240 s timeout); 4w 13.5 s per card. 8da4w stage 2 took 1 h 15 min.
**Projection:** 4w round c until about 07:30; 8da4w incumbents 07:45; attn*V 11:00 to 13:30; QK^T 20:30 to
23:30 without further rounds, up to 2026-10-07 05:00 UTC with both. The search started 2026-10-05 05:33 UTC, so
the 48-hour mark is 2026-10-07 05:33 UTC: a refinement round that would start so late that it cannot end
before that mark is not started; it is reported instead with its count and projection.

**Then (not started):** every kernel that beats the incumbent by more than 2 % in a full x 2 confirmation with
correctness `ok` goes, per shape, into one profile `xe2-refine5` in `tools/gen_xe2.py` (kernel + row + shape
predicates), then commit, `build-both.sh`, `stage.sh`, `gate.sh` (`gate_sdpa.sh` if an SDPA kernel changes),
`probe.sh`, `decide.py --bit-identical` for staging-only changes, and a timed session against `xe2-refine2`,
as `chain11.sh` in `.artifacts/logs/` did for candidate 4. All of it on `b70-0` with the second card idle
(`pair_lock excl` in the tools enforces it).

Owner instructions of 2026-10-05 and how they are met: status kept current with planned / done per family,
rate, projected end and interim findings (this file); no driver-level profiler tracing (nothing here uses any;
ETDump, shader-clock phase timing, microbench timing, igpu-roofline and sensors only); the second card
(below); the release-zone hooks decision is not needed (no accepted candidate needs a hook: the SDPA rows are
registered from the dev zone, the two hook-only softmax variants were not faster and are not candidates; if a
softmax variant is ever adopted, use `softmax-variant-hook.patch` beside `CAMPAIGN.md` unchanged and drop
`tools/hook-sdpa-softmax.patch`).

## Coordinator hold

In place since 2026-10-06 01:40 UTC (owner decision 2026-10-06 00:35 UTC), on both cards, without a queue
restart and without losing a running job (every unit sources `tools/host.sh` afresh).

- Files: **`/home/doremy/hmz-sarc-xe2/.artifacts/HOLD`** (created and removed by the coordinator only; this
  campaign never creates or removes it) and **`/home/doremy/hmz-sarc-xe2/.artifacts/HELD`** (written by this
  campaign).
- While `HOLD` exists nothing of this campaign starts on either card: no GPU process, no session, no gate, no
  build. What is running is finished normally, not killed. Each unit that would start next appends one line
  `HELD <UTC time> card<N> <what it would start next>` to `HELD` and polls once a minute. **`HELD` is written
  only at a moment when no measurement and no build of this campaign is running on either card** (the writer
  must get `run/pair.lock` exclusively for that moment), so it never appears while the other card still works:
  its existence means both cards are idle. When `HOLD` is gone the waiting units remove `HELD` and continue
  exactly where they were (the same configuration, the same queue job; nothing is skipped or repeated).
- Smallest unit = one GPU process of `tools/gl.sh`, i.e. during the search one configuration: cheap screen
  10 to 20 s normally and at most 240 s (its timeout); full measurement about 50 s, at most 600 s; correctness
  at most 300 s. A sweep build (compile check + build, at most 5 minutes so far) is also one unit. So after
  `HOLD` appears, `HELD` follows within 4 minutes during a screen and within 10 minutes otherwise. Later, while
  a candidate is gated, the units that cannot be interrupted are longer: unmodified `verify.sh` (about 8
  minutes) and one timed session (about 12 minutes); `HELD` then follows within about 12 minutes.
- Where the check is: `gpu_begin` in `host.sh`, called by `gl.sh` before every GPU process, by the session,
  gate, trace, parent-control and roofline tools and by `build-sweep.sh` before they start, and by
  `sweep_queue.sh` before each job (this last one from the next queue start; it is not needed for the hold to
  work). The check is made with the pair lock held, so a job cannot slip in between the check and its start.
- Tested without the coordinator: `tools/test_hold.sh` (a dry run of the same functions in a temporary artifact
  directory, CPU only, so it can run beside the queues): 11 assertions, `TEST_HOLD_PASS` at 01:40 UTC. It
  covers: no `HELD` without `HOLD`; a running job is not killed; no `HELD` while a job still runs on the other
  card; nothing new starts; one `HELD` line per waiting unit of both cards once idle; a unit arriving later and
  a unit inside a tool that already holds the pair lock wait too; after `HOLD` is removed every unit runs and
  `HELD` is gone.
- Holds obeyed: from 2026-10-06 03:10 UTC on each is recorded in `.artifacts/hold.log` (the `HELD` line and
  the release time). Before that there is one that the result rows show but no log records: both cards
  started nothing between 02:32 UTC (card 1) / 02:36 UTC (card 0, after its running configuration ended) and
  02:52:12 / 02:52:16 UTC, when both continued with the next configuration of their lists. I did not create
  `HOLD`; I take this to be the coordinator's use or test of it. No row was lost or repeated. The
  configuration that was running on card 0 when it began (`202040`) ended in its 240 s timeout (rc 124) at
  02:35:58, before `HELD` could be written; it is one of 10 timeouts of this screen and is not re-measured.
- Not done on purpose: the guard does not know the coordinator's measurement and was not taught to. CPU-only
  analysis steps between two units (`sweep_analyze.py`, seconds to a minute) are not GPU jobs and can still
  finish during a hold; no GPU job and no build is started by hand while `HELD` exists.

## The second B70 (owner decision 2026-10-05; `results/xe2/sweep/cardtest/`)

Tools: `XE2_CARD` (0 or 1) selects lock, PCI device, sensors and `ETVK_DEVICE_INDEX` in `host.sh`. The
foreign-process guard looks at the DRM clients of both cards (a Vulkan process opens both when it enumerates)
and accepts, besides its own tree, only the job that a registered, living `gl.sh` of this campaign runs on the
other card (`run/card<N>.job`: pid and start time); anything else on either card stops the job (`test_guard.sh`
has the cases). A cheap screen holds `run/pair.lock` shared; every other measurement and every build holds it
exclusively, so a full measurement, a session, a gate and a build always run with the other card idle.
`sweep_run.py` writes a `card` column and refuses any mode but `cheap` on card 1. `sweep_screen.sh` splits a
cheap screen by row position (odd rows to card 1, `raw/<build>-c1/`); afterwards the second card's times are
multiplied per shape by the ratio of the two cards' `base` arms (measured on each card every 50
configurations) and appended to the first card's `results.csv` with `card` = 1 (`sweep_merge.py`, factors in
`card1-scale.csv`); the second card's own file is kept as measured. Correctness, the validation and every
full x 2 confirmation stay on `b70-0`.

Acceptance was fixed before the first run (commit `19708af85`): one identical batch (the first 30
configurations of `sweep/sw1-4w/checked.csv` plus the shipped kernel = 31 arms, cheap mode) screened (a) on
`b70-0` alone, (b) on the second card alone, (c) on both at once; Spearman rank correlation of the
layer-weighted score >= 0.95 and of every shape class >= 0.90 for card-to-card and for alone against together
on each card; wall time of (c) at most 1.5 times the longer of (a) and (b). Result: `CARD_SPLIT_OK`.

| comparison | Spearman, score | Spearman, shape classes | median time ratio | single configurations around it (90th percentile / worst) |
|---|---:|---|---|---|
| card 1 against card 0, each alone | 0.9996 | 1.0000 (all four) | 0.9997 to 1.0002 | 0.15 to 0.76 % / 15.7 % (one wk/wv time of run (a); otherwise at most 3.0 %) |
| card 0, together against alone | 1.0000 | 1.0000 | 0.9998 to 1.0002 | 0.18 to 0.44 % / 15.1 % (the same run (a) value; otherwise at most 2.0 %) |
| card 1, together against alone | 1.0000 | 0.9996 to 1.0000 | 0.9999 to 1.0005 | 0.12 to 0.42 % / 2.1 % |
| card 1 against card 0, together (not gated) | 0.9996 | 0.9996 to 1.0000 | 0.9997 to 1.0002 | 0.22 to 0.52 % / 1.9 % |

Wall time: (a) 428 s, (b) 429 s, (c) 460 s for both batches, i.e. 1.07 times one card alone and 1.86 times the
throughput. The engine counters of the running test process (`drm-a.txt`, `drm-b.txt`) show work on
`0000:01:00.0` only in (a) and on `0000:02:00.0` only in (b): `ETVK_DEVICE_INDEX` 0 / 1 select the intended
cards. Package temperature at most 75 C (card 0) and 73 C (card 1), alone and together (`sensors.csv`, every
2 s, with the clock of both cards). The rule "together disturbs the ranking -> alternate or do not use" did not
apply.

## The sampled parameter search (review follow-up)

Tools: `tools/sweep.py` (spaces, static pruning, seeded sample, neighbours, overlay), `build-sweep.sh` (compile
check, exact shared memory from the SPIR-V, sweep build), `sweep_run.py` (resumable runner: modes cheap / full
/ corr), `sweep_screen.sh`, `sweep_merge.py`, `sweep_analyze.py`, `sweep_confirm.py`, `sweep_stage1.sh`,
`sweep_stage2.sh`, `sweep_refine.sh`, `sweep_rounds.sh`, `sweep_queue.sh`, `sweep_collect.sh`. The sweep
variants exist only in the sweep builds (`build/sw*`: an export of HEAD plus the generated overlay), never in
the working copy.

Legal space = after the analytic pruning (`sweep.py count`): workgroup <= 1024 invocations, whole 8 x 16 MMA
tiles per subgroup, tile sizes dividing every production shape, the flag exclusions and the known staging
rules of each body. Then every drawn configuration is compiled with the build's glslc and its exact shared
memory is read from the SPIR-V (limit 46000 bytes; a tile over the device limit can hang the GPU).

| kernel family (space) | parameters | analytically legal | drawn / compile / fit shared memory | planned | done | measured rate |
|---|---|---:|---|---:|---:|---|
| 4w linear | body (release, split staging `xe2s`, texel-wise `xe2bx`), M, N, K, subgroup grid, subgroup size, layout, IMG_A, IMG_W, drain, accumulator | 338448 | 3000 / 2851 / 2067 | 2000 + neighbours | 2000 cheap (60 also full x 2); 326 + 319 neighbours cheap; 23 + 26 finalists full x 2; round c pending | 13.5 s per configuration (cheap), 50 s (full), one card |
| 8da4w linear | body (zpg, bt, xe2bt, zpgtr), M, N, K, grid, subgroup size, zpgtr flags | 44670 | 3000 / 2333 / 2093 | 2000 + neighbours | 2000 cheap (60 also full x 2; 970 on card 1); 323 neighbours cheap; 19 finalists full x 2; incumbents pending | 18.3 s per configuration and card (cheap), 9.1 s for the two cards |
| attn*V | family (sweep, ml, xe2), M, N, K, grid, subgroup size | 1131 | all drawn; checked when its build runs | every legal one (about 850) | 0 | about 9 s expected |
| QK^T | family (sweep, pk, xe2, xe2c), M, N, K, grid, subgroup size, NO_MASK_FILL | 4536 | all drawn; checked when its build runs | every legal one (about 3400) | 0 | about 9 s expected |

Sample size: 2000 per linear space (the lower end of the owner's 2000 to 3000): a full enumeration with the
full measurement would take about 4 months (4w) and 3 weeks (8da4w). The two SDPA spaces can be enumerated
inside a day, so they are not sampled: every legal configuration is screened in the cheap mode.

Refinement rounds, by rule (`tools/sweep_rounds.sh`, numbers used in `rounds.txt`): round b runs if round 1
moved the best cheap score or the best of a shape class by more than 2 % over the sample; round c if round b
did so over round 1, or if the full confirmation of round b has a (model, shape) whose best configuration is
more than 2 % faster than the best of round 1's confirmation; no round d.

### 4w (`results/xe2/sweep/4w/`)

- **Validation of the cheap mode** (`validation.csv`; 60 configurations, cheap once against the median of two
  full runs): Spearman rank correlation 0.963 to 1.000 over the twelve (model, shape) pairs and 0.995 to 1.000
  for the layer-weighted score; the two full repeats differ by 0.05 to 0.11 % (median). The threshold was
  not written down before this measurement; 0.9 was the working figure.
- **Drift** (`drift.csv`): the table kernel repeated every 50 configurations (54 times) spreads under 0.5 % per
  shape.
- **Sample** (1993 of 2000 with all four shapes dispatched; `analysis-sample.txt`): none ahead of the shipped
  tile on the layer-weighted score (best 0.951x); 2 of 1993 are faster on wk/wv (best 1.043x), one each by
  under 0.5 % on wq/wo and w2, none on w1/w3. The median configuration is about 8 times slower.
- **Parameter importance** (`importance-sample.csv`: the uniform sample alone; share of the variance of log
  layer-weighted time explained by each parameter by itself, its best level, and the median time of that level
  relative to the overall median):

  | parameter | variance share | best level | level medians (relative) |
  |---|---:|---|---|
  | subgroup grid, y (`sy`) | 31.7 % | 16 | 1: 2.88, 2: 1.03, 4: 0.61, 8: 0.42, 16: 0.41 |
  | subgroup grid, x (`sx`) | 14.7 % | 16 | 1: 2.00, 2: 0.90, 4: 0.66, 8: 0.50, 16: 0.36 |
  | body (`fam`) | 7.7 % | split staging `xe2s` | release 1.66, `xe2bx` 1.18, `xe2s` 0.56 |
  | accumulator | 6.4 % | fp16 | fp16 0.66, fp32 1.42, grouped 1.34 |
  | tile M | 5.2 % | 64 | 32: 0.89, 64: 0.80, 128: 1.28, 256: 2.64 |
  | tile N | 1.4 % | 64 | 32: 0.85, 64: 0.85, 128: 1.06, 256: 1.78 |
  | shared-memory layout | 1.0 % | frag | col 1.49, f16 0.94, frag 0.93, pad 0.97 |
  | K per chunk | 0.9 % | 64 | 16: 1.00, 32: 0.98, 64: 0.97, 128: 2.40 |
  | subgroup size | 0.7 % | 16 | 16: 0.82, 32: 1.12 |
  | IMG_A, drain, IMG_W | 0.3 %, 0.2 %, 0.0 % | 0, pool, 0 | within 0.90 to 1.06 |

  The per-class tables are in the same file. The main effects describe the bulk of the space (how to avoid a
  slow kernel), not its top: the best configurations all have M = 128, K = 16 or 32 and 512 threads, which the
  level medians alone would not pick. **Pair interactions** (`interactions-sample.csv`: joint share minus the
  additive share of the two main effects): the largest are body x N 3.6 %, body x K 1.7 %, `sx` x `sy` 1.7 %,
  M x `sy` 1.6 %, N x layout 1.6 %; everything else under 1 %. `importance.csv` / `interactions.csv` are the
  same analysis over the sample plus all refinement neighbours (biased towards the best region; kept for
  reference).
- **Refinement** (cheap, one-parameter neighbours of the 20 best + 5 per class): round 1 `sw2-4w` 326 legal
  neighbours, round b `sw2b-4w` 319. **Confirmation** (full x 2 against the shipped kernel, the 10 best per
  class; `confirm.csv`, `confirm-b.csv`; repeat spread at most 0.44 %; correctness `ok` for each):

  | configuration | differs from the shipped `t128x128k16g44s16m8fli` by | 1B wq/wo, wk/wv, w1/w3, w2 | 3B | 8B |
  |---|---|---|---|---|
  | `150312` (release body), round 1 | subgroup grid 8 x 2 instead of 4 x 4, band drain | 1.034, 0.966, 1.019, 1.041 | 1.030, 1.060, 1.021, 1.029 | 1.024, 1.063, 1.018, 1.049 |
  | `160001` (`xe2s`), round b | the same on the split-staging body | 1.034, 0.980, 1.021, 1.044 | 1.029, 1.060, 1.026, 1.029 | 1.023, 1.063, 1.019, 1.050 |
  | `160054` (`xe2s`), round b | tile 128 x 256, grid 16 x 2, band drain | 1.046, 0.907, 1.004, 1.055 | 0.688, 0.881, 0.848, 0.863 | 1.019, 1.074, 1.004, 1.082 |
  | `160178` (`xe2s`), round b | K = 32, grid 8 x 4 (512 threads), IMG_W, default drain | 0.869, 1.171, 0.847, 0.808 | 0.646, 0.710, 0.625, 0.820 | 0.778, 0.887, 0.788, 0.787 |

  (Speed relative to the shipped kernel in the round-b confirmation, x > 1 = faster; every arm in the csv.)
  Shapes are independent: the best per shape is `160178` for 1B wk/wv, `160054` for 1B wq/wo and w2 and for 8B
  wk/wv and w2, `160001` / `150312` elsewhere. Round b moved no cheap (1B) class best by more than 1.3 %, but
  its confirmation moved 8B w2 from 1.050x to 1.082x, so round c is queued (`027`). Per layer this is 2 to 5 %
  less 4w linear time, i.e. about 1.5 to 3 % of a 4w cell: at the edge of the noise band. It is a candidate
  for `xe2-refine5` after round c; nothing is gated yet.

### 8da4w (`results/xe2/sweep/8da4w/`; sample and refinement done, incumbents pending)

- **Validation of the cheap mode** (`validation.csv`; 59 configurations with all shapes, cheap once against
  the median of two full runs; threshold 0.9, as for 4w, this time fixed beforehand): Spearman rank correlation
  0.984 to 1.000 over the twelve (model, shape) pairs and 0.998 to 1.000 for the layer-weighted score; the two
  full repeats differ by 0.08 to 0.29 % (median).
- **Drift** (`drift.csv`, card 0; card 1's own base rows are in `sw1-results-card1.csv`): the table kernel
  repeated every 50 configurations (21 times) spreads 0.06 to 1.5 % per shape. **Card 1** against card 0 on the
  base arm: factors 0.9969 to 1.0030 per shape (`sw1-card1-scale.csv`), applied to its 970 configurations.
- **Sample** (1973 of 2000 with all four shapes dispatched, 12 timeouts; `analysis-sample.txt`): none ahead of
  the shipped tile (best 0.966x on the layer-weighted score, a `zpgtr` 128 x 64 tile), so none near the
  candidate 2 tile, which this screen measures at 1.203x of the shipped one (1.257 / 1.152 / 1.172 / 1.245x for
  wq/wo, wk/wv, w1/w3, w2; the candidate 4 tile 1.107x, with 1.190x on w1/w3).
- **Parameter importance** (`importance-sample.csv`, the uniform sample alone, layer-weighted score):

  | parameter | variance share | best level | level medians (relative) |
  |---|---:|---|---|
  | subgroup grid, y (`sy`) | 37.0 % | 16 | 1: 2.11, 2: 1.16, 4: 0.41, 8: 0.15, 16: 0.12 |
  | subgroup grid, x (`sx`) | 12.3 % | 16 | 1: 1.92, 2: 0.84, 4: 0.60, 8: 0.42, 16: 0.22 |
  | tile N | 5.9 % | 64 | 32: 0.69, 64: 0.64, 128: 1.35, 256: 1.88 |
  | `zpgtr` flags (DRAIN_UNROLL, B_PAIR, B_SEL_EARLY_N, A_RAW, CSH_IN_ASH) | 1.3 %, 1.2 %, 0.6 %, 0.6 %, 0.5 % | mostly "not `zpgtr`" | off / on within 0.83 to 1.20 |
  | K per chunk | 0.9 % | 32 | 32: 0.98, 64: 0.99, 128: 1.16 |
  | body (`fam`) | 0.7 % | `xe2bt` | bt 0.64, xe2bt 0.26, zpg 0.40, zpgtr 1.03 |
  | tile M | 0.6 % | 32 | 32: 0.80, 64: 0.87, 128: 1.11, 256: 1.18 |
  | subgroup size | 0.0 % | 32 | 16: 1.02, 32: 0.99 |

  The body has a small variance share only because 95 % of the legal space, and so of a uniform sample, is the
  `zpgtr` body with its five flags (42240 of 44670); the three other bodies are 2.5 to 4 times faster at the
  median. **Pair interactions** (`interactions-sample.csv`): M x `sy` 2.6 %, K x `sy` 2.2 %, M x N 2.1 %,
  M x B_PAIR 1.9 %, M x DRAIN_UNROLL 1.7 %, N x A_RAW 1.6 %; the rest smaller.
- **Refinement and confirmation** (`sw2-8da4w`: 323 legal one-parameter neighbours of the best 20 + 5 per
  class, cheap on both cards; `sw3-8da4w`: 19 finalists full x 2 against the candidate 2 tile, `confirm.csv`):
  the best neighbour is 1.012x of the shipped tile on the layer-weighted score (`250192`, a `zpgtr` 128 x 128
  K = 64 tile). In the full confirmation the best configuration per (model, shape) is 0.78 to 0.90x of the
  candidate 2 tile; the shipped tile is 0.72 to 0.88x of it and the candidate 4 tile 0.71 to 1.03x (1.033x
  only on 1B w1/w3, the shape candidate 4 was gated for, without an end-to-end gain). The sample and its
  refinement therefore contain no 8da4w candidate.
- **The incumbents in the search** (`tools/sweep_incumbent.sh`, job `028`; the first attempt, `0255`, numbered its configurations in the id range of `sw2-8da4w` and is superseded): because the kernels already in use
  beat the whole sample, a refinement that starts from the sample's best 20 never visits their neighbourhood.
  The candidate 2 tile, the candidate 4 tile and the shipped tile are therefore added as configurations
  together with their unmeasured one-parameter neighbours (build `sw2i-8da4w`), and the rule-based rounds then
  refine from the best 20 of everything. This adds to the owner's procedure; the sample and its own refinement
  are unchanged.

## Needs the owner's attention

- **Since the reboot of 2026-10-06 05:57 UTC eight `llm-api-*` services are `active (running)`**
  (`llm-api-coder`, `-flash`, `-qwen36`, `-qwen36vlm`, `-qwen38`, `-qwen38fp8`, `-qwen38heretic`,
  `-qwen38uncensored`: "Bounded request admission for ..."). They came up with the boot; this campaign did not
  start them and does not stop them. At 06:10 UTC no process held a DRM file of either B70 and
  `llama-server`, `comfyui`, `vllm`, `ollama` were inactive, so no card is taken and the search continues. If a
  request makes one of them load a model on a card, the guard stops the queue of the job that sees it (rc 76,
  `QUEUE_STOPPED`), and I stop using that card and report, as the campaign section says. If these services are
  meant to stay off for the campaign, they need to be stopped again by the owner.
- `nvtop` did not come back after the reboot; the paragraph below describes the sessions before it.
- **`nvtop` (pid 1952, pts/0, started 17:01 UTC, before the campaign) holds a DRM file of both B70 cards.** It
  is not a workload: every DRM client it owns shows zero engine cycles and zero GPU memory. The campaign guard
  records it per session as an idle monitor (`env.txt`: `idle monitors ... 1952:nvtop`) and would treat it as a
  foreign GPU process, and stop, the moment either number is non-zero. If measuring beside it is not wanted,
  close it before any re-measurement; every session of this campaign ran with it open.
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

## Linear kernels (step 1 and step 3): phase timing and screens

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
per chunk, not the number of fetches. Screen 7 (balanced K = 64 / 128 tiles) gave candidate 2's tile
`xe2bt_t128x128k64g84s16m8` at 1.26x; screens 8, 9 and 11 (4w band drain and split staging) found no 4w tile
faster than the shipped one (best 1.002x); screens 10 and 12 (hook-only softmax variants) no faster softmax.
All are listed in `results/xe2/screens/README.md`.

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

Geomean +46.86 %, every cell far outside the +-2 % band (A/A noise 0.6 %). 62 timed runs: 61 valid and one
rejected (`8b 4w cand r5`, `clock_low`), replaced by the next valid run of that arm.

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

### Candidate 3, `xe2-refine3` (fragment-contiguous ColumnMajor QK^T): GATE_PASS, no measurable gain, not adopted

Session `s4-c3`, build `topic7` in both arms, `xe2-refine2` against `xe2-refine3` (`gate_sdpa.sh`): geomean
+0.32 %, cells -0.16 to +1.01 %, all inside the +-2 % band. No FAIL line: SDPA tiers all / extended / full 12
passes each, 0 mismatches, `pairing=ok`; `verify.sh` status as candidate 2; next token SAME in all six cells
on the three prompts; logits bit-identical on every probe window (`probe/s4-c3/`). The trace does show the
kernel-level gain (QK^T per prefill 9.3 -> 8.8 ms on 3B 4w, 14.5 -> 13.6 ms on 8B 4w, 4.8 -> 4.8 ms on 1B),
which is 0.1 to 0.2 % of a prefill.

### Candidate 4, `xe2-refine4` (8da4w 64-column K = 64 tile where N >= 4K): GATE_PASS, no gain, not adopted

Session `s5-c4`, build `topic7` in both arms, `xe2-refine3` against `xe2-refine4` (`gate.sh`): geomean -0.11 %,
cells -1.98 to +0.66 %, all inside the band. No FAIL line; next token SAME everywhere; logits bit-identical
(`probe/s5-c4/`). The predicate matches only the 1B w1 / w3 layers (N = 8192, K = 2048); there the tile is
slower in the model than the candidate 2 tile (1B 8da4w linear GEMM 36.7 -> 37.8 ms per prefill, cell
-1.98 %). In screen 7 it was 1.11x of the shipped tile over all 1B layers, the candidate 2 tile 1.19x.

### Winner against the parent, measured directly (session `s6-final`)

Pristine parent build, no environment, against build `topic7` (`8666b6531`) with `ET_VK_SARC_UNVERIFIED=1
ET_VK_SARC_DEV_PROFILE=xe2-refine2`; tok/s, median of 5 valid runs per arm, arms interleaved; the last column
is the device's SARC number in `sarc-1.5-e2e-benchmark/results/cells.csv`:

| cell | parent | winner | gain | `cells.csv` | winner vs `cells.csv` |
|---|---:|---:|---:|---:|---:|
| 1B 4w | 11770.10 | 17504.30 | +48.72 % | 11702.9 | +49.57 % |
| 1B 8da4w | 12412.10 | 20686.90 | +66.67 % | 12412.1 | +66.67 % |
| 3B 4w | 4864.61 | 7393.50 | +51.99 % | 4864.61 | +51.99 % |
| 3B 8da4w | 5264.78 | 9615.02 | +82.63 % | 5251.28 | +83.10 % |
| 8B 4w | 2435.20 | 3292.60 | +35.21 % | 2438.10 | +35.05 % |
| 8B 8da4w | 2734.31 | 4481.40 | +63.90 % | 2737.97 | +63.68 % |

Geomean +57.47 % over the parent. 60 timed runs, none rejected. The session ends `E2E5_INCOMPLETE` on the two
next-token items of candidate 1 (8B 8da4w on `prompt_2048.txt` and `prompt_check.txt`), the same items as in
`s2-c1`; the other 16 comparisons are SAME. This session has timing and traces only: the gate of the winner's
kernels is `s2-c1` (SDPA) and `s3-c2` (8da4w linear).

Linear kernels of the winner in the model against the fresh roofs (`sessions/s6-final/trace/gemm.csv`): 4w
63.2 to 67.6 TFLOP/s = 36.5 to 39.0 % of the fp16 matrix roof (173.3), unchanged; 8da4w 104.8 to 115.3 TOP/s
= 29.1 to 32.0 % of the int8 matrix roof (359.9), from 82.1 to 87.7 (22.8 to 24.4 %).


## Next

See "What is running" at the top: the search, then the gate of what it found (`xe2-refine5`), then the final
update of `proposal.md` and the push.

## Awaiting B580 confirmation

The winner `xe2-refine2` is candidates 1 and 2; candidates 3 and 4 are not adopted and need no confirmation.

Candidate 2 (`xe2-refine2`, plain gate pass on the B70): 8da4w linear
`sarc_dev_linear_dq8ca_coopmat_zpg_xe2bt_t128x128k64g84s16m8`.

Candidate 1 (`xe2-refine1`, accepted under the reference-error rule on the B70): QK^T `sarc_sdpa_qk_coopmat_pk_t128x64k32g44s16m8nf`,
attn*V `sarc_sdpa_av_coopmat_xe2_t128x64k32g44s16m8` (head_dim 128) and
`sarc_sdpa_av_coopmat_sweep_t64x64k32g44s16m8` (head_dim 64), with the truncated SARC softmax. Every Xe2 variant keeps its shared memory under 46000 bytes (the B70
reports `maxComputeSharedMemorySize` 49152; the B580's value was not read here and should be confirmed), uses
workgroups of at most 1024 invocations and the 8x16x16 / subgroup-16 shapes the shipped Intel rows already use
on both cards, and does not depend on the amount of device memory.
