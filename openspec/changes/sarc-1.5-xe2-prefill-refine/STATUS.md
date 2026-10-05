# sarc-1.5-xe2-prefill-refine: status

**2026-10-05 18:05 UTC — CHECKPOINT for a restart of the control session; the sampled parameter search keeps
running detached. Read "Handoff" first. Earlier results are unchanged: winner `xe2-refine2` (candidates 1 and
2), +57.5 % geomean over the parent (`s6-final`); candidate 1 is `ACCEPTED (reference-error rule, owner
decision 2026-10-04)`, not a plain pass; candidates 3 and 4 passed their gates with no measurable gain.**

## Handoff (written 2026-10-05 18:05 UTC, for an actor with no memory of this session)

**Task in progress.** The reviewer of the first report asked for three things: (1) the sampled parameter search
of the owner decision "how large parameter spaces are searched" with its parameter-importance table, (2) a
resumable linear sweep, (3) a corrected run count for `s2-c1`. Items 2 and 3 are done and committed
(`tools/screen.sh`, `tools/sweep_run.py`; `screen13-resume-test` shows interrupt and resume). Item 1 is running.
When it is finished: update `proposal.md` (its "Not done" section still says the search was not run) and this
file, run `bash sarc/tools/check.sh --no-build`, commit, `git push origin topic/xe2-prefill-refine`. No PR.

**What is running detached, and how to check it.** One queue, `tools/sweep_queue.sh` (started with `nohup
setsid`; it survives the session). It runs the scripts in `.artifacts/queue/pending/` one at a time in name
order and moves each to `queue/done/`. Check: `cat ~/hmz-sarc-xe2/.artifacts/queue/status` (start / end and rc
per job), `tail .artifacts/queue/log/<job>.out`, `pgrep -af sweep_queue`. Never start a second GPU job or a
build beside it on `b70-0`: the guard of the running job treats it as a foreign GPU process and stops (rc 76);
put new work into `queue/pending/<NNN>-<name>.sh` instead. If the queue has died, restart it with `nohup
setsid bash tools/sweep_queue.sh > .artifacts/queue/queue.out 2>&1 &`; every runner is resumable (it skips rows
already in `raw/<name>/results.csv` and moves an interrupted attempt to `raw/<name>/superseded/`). To end the
queue when nothing is pending: `touch .artifacts/queue/STOP`.

**Queue at 18:05 UTC** (seed 20261005 everywhere):

| job | what | state |
|---|---|---|
| `010-stage1-4w` | sample 3000 -> 2000 legal, validation, cheap screen | done 15:44 |
| `015-screen-resume-test` | `screen.sh` interrupt / resume | done |
| `016-stage2-4w` | correctness of the top, 326 neighbours (`sw2-4w`), full x 2 of 23 finalists (`sw3-4w`) | done 17:53 |
| `017-refine-b-4w` | second refinement round (`tools/sweep_refine.sh 4w b`): neighbours of the new best, then finalists `sw3b-4w` | RUNNING since 17:53, about 2.5 h |
| `020-stage1-8da4w`, `025-stage2-8da4w` | 3000 drawn -> 2000 legal; refs shipped, candidate 2 tile, candidate 4 tile | pending, about 10 h + 4.5 h |
| `030-stage1-av`, `035-stage2-av` | every legal attn*V configuration | pending, about 3 h + 2 h |
| `040-stage1-qk`, `045-stage2-qk` | every legal QK^T configuration (enumerated, not sampled) | pending, about 9 h + 2.5 h |

Projected end on one card: about 2026-10-07 06:00 UTC, which is past the 48-hour mark of the search (started
2026-10-05 05:33 UTC) by half an hour because of the second 4w round. That is the reason to do the split onto
the second card (below) before the 8da4w stage starts, or else to report before starting QK^T.

**Instructions received today that the files did not yet reflect.**

1. Owner, 14:00 UTC: keep this file current with planned / done per family, rate, projected end, interim
   findings. Done here; repeat after every stage.
2. Owner, 16:50 UTC, shared rule now in `CAMPAIGN.md` "Not allowed": no driver-level profiler tracing
   (`MESA_VK_TRACE*`, `RADV_*TRACE*`, `INTEL_MEASURE`, Perfetto GPU sources, capture tools, kernel settings).
   Nothing in this campaign uses any of them; ETDump, shader-clock phase timing, microbench timing,
   igpu-roofline and sensors stay allowed.
3. Owner, about 17:55 UTC, `CAMPAIGN.md` section "Owner decision, 2026-10-05: the second B70 may be used for the
   search" (read it with `cat`): the second Arc Pro B70 (guest PCI `0000:02:00.0`, Vulkan device **1**,
   deviceUUID and lock `868023e2-0000-0000-0200-000000000000`, sensors under
   `/sys/bus/pci/devices/0000:02:00.0/{hwmon/hwmon1,tile0/gt0/freq0}`; index and UUID read with `vulkaninfo
   --summary` at 17:58, pin the index, do not assume) may run cheap-mode SCREENS only. Every selecting
   measurement (full x 2 confirmation), session, gate and reported number stays on `b70-0` alone with the
   second card idle. **NOT STARTED: nothing has run on the second card and no tool knows about it yet.**
   Required before splitting, in this order:
   - teach the tools a card parameter: `host.sh` (PDEV, LOCK, `ETVK_DEVICE_INDEX`, sensors) and
     `sweep_run.py` (the hwmon path is hard-coded to `0000:01:00.0`; add a `card` column to every row);
     `gpu_others` must not count this campaign's own job on the other card (its ancestry rule only knows
     `XE2_TOP`) and must still catch anything else on either card; one queue per card, each with its own lock;
     builds stay out of screening windows on both cards;
   - fix the acceptance threshold BEFORE looking, as for the cheap mode. Intended: Spearman rank correlation
     of the layer-weighted score >= 0.95 and of every shape class >= 0.90, both for card-to-card and for
     alone-versus-together, on one identical batch of about 30 configurations of the space in progress (the
     first 30 of `sweep/sw1-8da4w/checked.csv` once that build exists, or of `sw1-4w`); also report the time
     ratios. Screen it (a) on `b70-0` alone, (b) on the second card alone, (c) on both at once;
   - if together disturbs the ranking: alternate the cards or do not use the second one, and say so;
   - then split the remaining cheap screens (8da4w, attn*V, QK^T) in two halves by row parity of
     `checked.csv`, keep refinement confirmation and `corr` on `b70-0`, and update the projection here.
   If an `llm-api-*` service or ComfyUI comes back and takes a card: stop using that card and report.
4. Owner decision of 2026-10-05 in `CAMPAIGN.md`, "the release-zone hooks the accepted candidates need may be
   committed": not needed here. No accepted candidate of this campaign needs a hook (the SDPA rows are
   registered from the dev zone; the two hook-only softmax variants were not faster and are not candidates).
   If a softmax variant is ever adopted, use `softmax-variant-hook.patch` beside `CAMPAIGN.md` unchanged and
   drop `tools/hook-sdpa-softmax.patch`.
5. Relayed, 18:00 UTC: the control session is being restarted; do not start a new long step before it.

**Next concrete steps.**

1. When `017-refine-b-4w` ends: `python3 tools/sweep_confirm.py 4w .artifacts/sweep/sw3b-4w/checked.csv
   .artifacts/raw/sw3b-4w/results.csv .artifacts/sweep/an-4w/confirm-b.csv`. If round b moved the best score or
   a class best by more than 2 % over round 1, queue round c (`sweep_refine.sh 4w c ref=shipped:`); otherwise
   the 4w search is finished. Copy `sweep/an-4w/*.csv`, the `checked.csv` and `results.csv` of every 4w build
   into `results/xe2/sweep/4w/` (the copies there now are the state after round 1).
2. Decide the 4w candidate (below), do the second-card work of item 3, then let 8da4w, attn*V and QK^T run.
3. For each space after its stage 2: `sweep_confirm.py`, the 2 % rule for a further round, collect, update
   this file.
4. Any kernel that beats the incumbent by more than 2 % in the full x 2 confirmation with correctness `ok`
   becomes a candidate the usual way: add it to `tools/gen_xe2.py` (kernel + row + a profile `xe2-refine5`
   with shape predicates), commit, `build-both.sh`, `stage.sh`, `gate.sh` (or `gate_sdpa.sh`), `probe.sh`,
   `decide.py --bit-identical` if it only changes staging, as `chain11.sh` in `.artifacts/logs/` did.

**The 4w finding so far (round 1, `results/xe2/sweep/4w/`).** The uniform sample of 2000 found nothing faster
than the shipped tile (best 0.951x on the layer-weighted score). The neighbours of its best 20 did:

| configuration | what differs from the shipped `t128x128k16g44s16m8fli` | full x 2 confirmation against the shipped kernel (`sw3-4w`) |
|---|---|---|
| `150312` (release body) | subgroup grid 8 x 2 instead of 4 x 4, band drain (`CSH_BAND`) | 1B wq/wo 1.034x, w1/w3 1.019x, w2 1.040x; 3B 1.029x / 1.022x / 1.030x and wk/wv 1.060x; 8B 1.020x / 1.015x / 1.051x and wk/wv 1.063x; repeat spread at most 0.44 % |
| `150202` (`xe2s` body) | K = 32, grid 8 x 4 (512 threads), no IMG_A, default drain | 1B wk/wv 1.166x (79.2 against 92.4 us); slower than shipped on the other shapes (0.80 to 0.85x) |

Both have correctness `ok` (numeric cases that dispatched them). Per layer of 1B this is about 3 % less linear
time, i.e. about 2 % end to end on a 4w cell: at the edge of the noise band, so it is a candidate only if
round b or c finds more. Neither is a candidate yet; nothing was gated.

Branch `topic/xe2-prefill-refine`, parent `6a7cc8cc6` (head of `topic/780m-prefill-refine`). Host
`fedora-gpu-eval`, card `b70-0` only (guest PCI `0000:01:00.0`, Vulkan device 0, **`ETVK_DEVICE_INDEX=0`**,
deviceUUID = lock UUID `868023e2-0000-0000-0100-000000000000`), ANV, Mesa 26.2.3. Nothing was run on the second
B70 or on the B580.

## The sampled parameter search (review follow-up; state at 18:05 UTC, see "Handoff" for what runs)

Tools: `tools/sweep.py` (spaces, static pruning, seeded sample, neighbours, overlay), `build-sweep.sh` (compile
check, exact shared memory from the SPIR-V, sweep build), `sweep_run.py` (resumable runner: modes cheap / full
/ corr), `sweep_analyze.py`, `sweep_confirm.py`, `sweep_stage1.sh`, `sweep_stage2.sh`, `sweep_refine.sh`,
`sweep_queue.sh`. The sweep variants exist only in the sweep builds (`build/sw*`: an export of HEAD plus the
generated overlay), never in the working copy.

Legal space = after the analytic pruning (`sweep.py count`): workgroup <= 1024 invocations, whole 8 x 16 MMA
tiles per subgroup, tile sizes dividing every production shape, the flag exclusions and the known staging
rules of each body. Then every drawn configuration is compiled with the build's glslc and its exact shared
memory is read from the SPIR-V (limit 46000 bytes; a tile over the device limit can hang the GPU).

| kernel family (space) | parameters | analytically legal | drawn / compile / fit shared memory | planned | done | measured rate |
|---|---|---:|---|---:|---:|---|
| 4w linear | body (release, split staging `xe2s`, texel-wise `xe2bx`), M, N, K, subgroup grid, subgroup size, layout, IMG_A, IMG_W, drain, accumulator | 338448 | 3000 / 2851 / 2067 | 2000 + neighbours | 2000 cheap (60 also full x 2), 326 neighbours cheap, 23 finalists full x 2; round b running | 15.0 s per configuration (cheap), 50 s (full) |
| 8da4w linear | body (zpg, bt, xe2bt, zpgtr), M, N, K, grid, subgroup size, zpgtr flags | 44670 | 3000 drawn; checked when its build runs | 2000 + neighbours | 0 | expected as 4w |
| attn*V | family (sweep, ml, xe2), M, N, K, grid, subgroup size | 1131 | all drawn; checked when its build runs | every legal one (about 850) | 0 | about 9 s expected |
| QK^T | family (sweep, pk, xe2, xe2c), M, N, K, grid, subgroup size, NO_MASK_FILL | 4536 | all drawn; checked when its build runs | every legal one (about 3400) | 0 | about 9 s expected |

Sample size: 2000 per linear space (the lower end of the owner's 2000 to 3000): a full enumeration with the
full measurement would take about 4 months (4w) and 3 weeks (8da4w). The two SDPA spaces can be enumerated
inside a day, so they are not sampled: every legal configuration is screened in the cheap mode.

### 4w, stage 1 and round 1 of stage 2 (`results/xe2/sweep/4w/`)

- **Validation of the cheap mode** (`validation.csv`; 60 configurations, cheap once against the median of two
  full runs): Spearman rank correlation 0.963 to 1.000 over the twelve (model, shape) pairs and 0.995 to 1.000
  for the layer-weighted score; the two full repeats differ by 0.05 to 0.11 % (median). The threshold was
  not written down before this measurement; 0.9 was the working figure.
- **Drift** (`drift.csv`): the shipped kernel repeated every 50 configurations spreads under 0.5 % per shape.
- **Sample** (1993 of 2000 with all four shapes dispatched): none ahead of the shipped tile on the score, best
  0.951x, the median configuration about 8 times slower.
- **Parameter importance** (`importance.csv`, share of the variance of log time explained by each parameter
  alone, sample + round-1 neighbours; the final table follows the last round): see the file; in the sample
  alone the subgroup grid dominates (`sy` 31 %, `sx` 16 %), then body 8 %, accumulator 6 %, tile M 5 %; N,
  layout, K, subgroup size, IMG_A, drain, IMG_W 1 % or less each. Pair interactions in `interactions.csv`.
- **Refinement round 1** (`sw2-4w`: 370 one-parameter neighbours of the 20 best + 5 per class, 326 legal) and
  **confirmation** (`sw3-4w`, `confirm.csv`): the two configurations in "Handoff".

## Needs the owner's attention

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

Nothing is queued. What a further campaign could try, and what was not done here, is in `proposal.md`
("Limits" and "Not done").

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
