# sarc-1.5-4070ti-fused-port: the fused attention kernel on the RTX 4070 Ti SUPER

Second campaign on this card. The first (`sarc-1.5-4070ti-prefill-refine`) ended with the cooperative-matrix
attention kernels and the fp32 no-tail softmax (`4070ti-refine1`), +46.35 % geomean over `dev/1.5` from the
committed branch. This one ports one more thing on top: the fused attention kernel of the Radeon 780M
(`fused3`, in the RX 7600's `fused3sb` form with subgroup barriers), one-pass variants. It is a port: no sampled
search, no enumeration, no tile sweep; two candidates at most.

- Device: NVIDIA GeForce RTX 4070 Ti SUPER, driver 615.71.09 (as found 2026-10-08), the only GPU of a KVM guest;
  subgroup size 32 (minimum = maximum), `computeFullSubgroups` and `subgroupSizeControl` available,
  `maxComputeSharedMemorySize` 49152 bytes, at most 1024 invocations per workgroup. Clock policy as found:
  P0, 3120 MHz maximum, 285 W limit; not changed.
- Parent of every comparison: commit `6050b1287` with
  `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=4070ti-refine1`. Pristine arm of the closing session: the same
  build with no environment.
- Candidate profile names: `4070ti-fusedN`, selected by `ET_VK_SARC_DEV_PROFILE` alone.

## Thresholds (fixed before the first measurement; `tools/thresholds.txt`)

| what | value |
|---|---|
| valid timed run | rc 0, 2048 prompt tokens, 0 generated tokens, no GPU process of another owner before, during or after, at least 2 clock samples in the prefill window (sampler 20 ms, a 1B prefill is 60 to 70 ms), median clock at or above the floor |
| clock floor | floor(0.97 x the lowest per-run median clock of the usable A/A runs), one device-wide value; a run whose median is below 1500 MHz (still ramping from the 210 MHz idle clock) is listed and not used for the floor |
| repeats | median of 5 valid runs per arm and cell; 7 if any A/A cell is further than 1.0 % from 1 or any arm's repeat spread exceeds 3.0 % |
| noise band | a difference inside +-2 % is noise |
| baseline | each parent cell within 3 % of the committed-build numbers of the first campaign (1B 29681 / 33032, 3B 12881 / 14949, 8B 5988 / 6954 tok/s, 4w / 8da4w), else explained first |
| A/A | geomean within 1 % of 1 and every cell within +-2 %, else nothing is optimised |
| kernel screen | a kernel replaces the incumbent only when at least 3 % faster in every round |
| reference-error rule (D3) | per production-shape case the candidate's rms and maximum error against the fp32 reference not larger than the parent's; 0 mismatches in every tier; no cell with mean KL above 0.5 nat or top-1 differing on more than one third of at least 32 real-text prompts |
| stop | two consecutive gated candidates under 2 % geomean, or the port list of the task exhausted (task section 6.4), or candidate 1 not correct within 24 hours of device time |

The timer of the runner has 1 ms steps: one step is 1.4 to 1.6 % of a 1B prefill at the parent's rate. 1B cells
are reported with the ETDump dispatch total beside tok/s.

## Tools

Copied from `sarc-1.5-4070ti-prefill-refine/tools/` (same host, same card). What changed:

| file | change | why |
|---|---|---|
| `common.sh` | campaign root, artifact directory (`<campaign-root>/.artifacts`), change directory, process tag `4070ti-fused-port`; `PARENT_COMMIT=6050b1287` and `PARENT_ENV`; `TMPDIR` under the artifact directory; `hold_point` | paths of this campaign; the parent carries a profile; L23; coordinator hold (D6) |
| `gatelib.sh`, `e2e5.sh`, `trace.sh`, `build-both.sh` | `hold_point` before every gate step, every cell of a timed session, every traced run, every build | D6: the smallest unit is one cell, one `verify.sh`, one SDPA pass, one traced run or one build |
| `parent_verify.sh` | takes a session name and an environment (default: the parent's) | two snapshots: `s0-parent-verify` (parent with its profile) and `s0-pristine-verify` (no environment, reference of the hook control) |
| `hook_control.sh` | takes the reference snapshot; the staged environment must equal the reference's | the hook is shown inert with nothing selected and under the parent's profile |
| `calibrate_clock.py` | the floor is taken from the lowest per-run median (rule R6), ramp runs listed; constants read from `thresholds.txt` | the shared rule replaces the first campaign's per-cell-median rule |
| `collect.sh` | paths | |
| `thresholds.txt` | new | task section 6.1 |

Changed after the copy, for this campaign's candidate and rules (each change has regression tests in
`tools/test_gate_check.py`, 48 tests; the sibling's file had 40):

| file | change | why |
|---|---|---|
| `gate_check.py sdpa` | a case may be served by the fused kernel instead of the three (then `qk=? softmax=? av=?` and `qk_coopmat=NO av_coopmat=NO` are required, a fused kernel beside one of the three is a finding); under a `4070ti-fused` profile the three production cases must be served by the fused kernel; the tier `all` is required beside `extended` and `full`, 12 passes each | the sibling's check demanded `qk_coopmat=yes av_coopmat=yes` on every case, which a fused kernel can never print; the task names three tiers (the first gate, `s2-c1`, ran only two: found in review) |
| `gate_check.py session`, `summarize.py`, `gatelib.sh` | the number of valid timed runs per arm is `reps` of `thresholds.txt` (7), not a fixed 5 | the repeat rule fixed before the A/A session |
| `gate_check.py session`, `runrow.py`, `e2e5.sh` | the sampler also reads the driver's throttle reasons every 20 ms (`clocks_event_reasons.sw_thermal_slowdown`, `.hw_thermal_slowdown`, `.hw_slowdown`, the raw mask); a timed run is valid only with at least 2 such samples and no thermal reason in any sample of the run; four columns appended to `runs.csv` | R6 requires "no thermal throttle reason"; the sibling's tools and the first three sessions of this campaign did not record it (found in review) |
| `gate_sdpa.sh` | tiers `all extended full`; the test binary's and runner library's hashes and the environment in `sdpa-correctness/IDENTITY.txt`, every pass's exit status in `exit-status.txt` | as above |
| `quick_e2e.sh`, `sdpa_screen.sh` | `hold_point`, D5 warming | D6, D5 |
| `attention_families.py` | new: the fused kernel and its copy pass as families of their own | the kit's analysis files both under copy/view/other |

Not carried over: the generators (`gen_4070ti_*.py`, `devzone.py`), the linear screens and phase-timing tools
(`screen.sh`, `prof.sh`, ...), the runner probes (`exit_probe.sh`, `first_use_probe.sh`), the local hook patches.
This campaign generates nothing and sweeps nothing. Unchanged from the sibling: `nexttoken.py`,
`llama_main_rc.sh`, `warm_file.py`, `stage.sh`, `gate.sh`, `gate_rest.sh`, `session.sh`, `gl.sh`, `mktree.sh`,
`ref_error_rule.py`, the probe tools.

Builds: `tools/build-both.sh <tag> <commit>` exports the commit and its pinned submodules from the git object
stores, runs the tree's own `sarc/tools/build.sh --llama` (and `--traced`) in `localhost/et-vk-build:rocky10`
through the docker shim, and fails unless `sarc/tools/spirv_golden.py` reports the shipped variants unchanged.

## Release-zone hook (owner decision 2026-10-05)

Commit `35e3728c9` is the 780M campaign's `1c8861aa7e` cherry-picked unchanged (hook D4.3, the entry point of
the fused attention node): 49 lines in `impl/SDPA.cpp` (7), `impl/sarc/SdpaCoopmat.cpp` (21),
`impl/sarc/SdpaCoopmat.h` (8), `impl/sarc/Select.h` (13). `Override::sdpa_fused_add` and
`Override::sdpa_fused_serves` are null by default; `sarc::add_sdpa_fused` is called once after the three SDPA
nodes of an LLM-mode op and does nothing without the first; `sarc::sdpa_fused_skip` gives the three nodes an
empty launch geometry only for a call the second says the fused node serves. **It is larger than a switch and
stays subject to the owner's review before any promotion.** On this branch the dev zone sets the two fields
only while `ET_VK_SARC_DEV_PROFILE` names a `4070ti-fused` profile, so every other configuration of a dev build
has them null as well. The evidence for the D4 conditions is in "Hook control" below.

## The port (candidate 1, profile `4070ti-fused1`)

What runs: for an LLM-mode attention call with fp16 buffers whose shape fits (S a multiple of the row tile and
of the column tile, `input_pos` a multiple of the column tile, head_dim 64 or 128), the node
`impl/sarc_dev/4070ti/Sdpa4070tiFused.cpp` dispatches `sarc_dev_4070ti_sdpa_kvt` (tile-packed copies of K and V
for the visible context) and then `sarc_dev_4070ti_sdpa_fused3sb_<variant>`, and the QK^T, softmax and
attention x V nodes dispatch nothing. Every other call (decode, an unaligned prompt) runs the three kernels of
`4070ti-refine1` with the `4070ti_nzf` softmax, exactly as the parent does.

| head_dim | variant | rows x columns per workgroup | workgroup | models |
|---|---|---|---|---|
| 64 | `fused3sb_d64_t32x32g11s32rko` | 32 x 32 | 32 invocations = 1 subgroup | 1B |
| 128 | `fused3sb_d128_t16x64g11s32rko` | 16 x 64 | 32 invocations = 1 subgroup | 3B, 8B |

These are the two variants the Radeon 780M and the RX 7600 run (`r` Q tiles kept in registers, `k` tile-packed
K and V, `o` one pass with a running row maximum). No other variant is built and none was screened.

What changed against the source kernels:

| item | 780M / RX 7600 | here | why |
|---|---|---|---|
| shader text from `#version` on | `sarc_dev_780m_sdpa_fused3sb.glsl` (RX 7600 commit `b3bb758e38`) | the same plus four lines: `if (gl_NumSubgroups != 1u \|\| gl_SubgroupSize != SUBGROUP_SIZE) return;` after the row-range check | the kernel's shared memory is private to one subgroup only if the workgroup is one subgroup; the pipeline asks for subgroup size 32 but nothing asks for full subgroups, so the kernel checks it itself and writes nothing otherwise (the correctness tiers then fail on stale output instead of racing silently) |
| constants (`MMA`, tiles, `SUBGROUP_SIZE`, `P_STRIDE`, spec constants) | | unchanged | 16 x 16 x 16 fp16 and fp32-accumulator matrices and subgroup 32 exist on this card |
| name prefix | `sarc_dev_780m_sdpa_` | `sarc_dev_4070ti_sdpa_` | separable content; `Sdpa780mFused.cpp` is not edited |
| copy pass | `sarc_dev_780m_sdpa_kvt.glsl` | the same text from `#version` on, workgroup 8 x 8 fixed | |
| the unpacked path (`vt`) and the two-pass and unpacked variants | built | not built | not what the AMD cards ship |
| selection | `ET_VK_SARC_780M_SDPA_FUSED` or the 780M profile variable | `ET_VK_SARC_DEV_PROFILE=4070ti-fused1` alone | one name per configuration |

Shared memory of one workgroup (limit on this card 49152 bytes):

| array | elements | d64 `t32x32` | d128 `t16x64` | holds |
|---|---|---:|---:|---|
| `Psh` | `WG_TILE_M x max(WG_TILE_N / 8 + 1, 4)` uvec4 | 32 x 5 x 16 = 2560 B | 16 x 9 x 16 = 2304 B | the block's scores (fp16), then e (fp16); at the end the row sums as 16 fp32 per row |
| `Rsh` | `WG_TILE_M x SEGS` float | 32 x 1 x 4 = 128 B | 16 x 2 x 4 = 128 B | per-invocation block maximum, at the end the per-invocation row sum |
| `Dsh` | `WG_TILE_M x 4` vec4 | 32 x 4 x 16 = 2048 B | 16 x 4 x 16 = 1024 B | per-row rescale divisor exp(new max - old max), 16 fp32 per row |
| total | | 4736 B | 3456 B | |

Barrier reasoning for NVIDIA. Invocation `l` of the subgroup owns row `l % WG_TILE_M` and column segment
`l / WG_TILE_M` (one segment of 32 columns for d64; two segments of 32 for d128). Who writes each slot and
what orders each read by another invocation:

| step (per column block unless noted) | write | writer | read by | ordered by |
|---|---|---|---|---|
| scores | `Psh` rows x columns of the block | the subgroup's cooperative-matrix store (`coopMatStore`, subgroup scope) | each invocation, its own segment | `memoryBarrierShared(); subgroupBarrier();` after the store |
| block maximum (d128 only, `SEGS > 1`) | `Rsh[row x SEGS + segment]` | the owning invocation, one slot each | both invocations of the row | barriers after the write; barriers again after the reads, before the slot is written in the next block |
| rescale divisor (only when `subgroupAny` saw a maximum rise: the branch is uniform over the subgroup) | `Dsh[row x 4 + j]`, `j` = segment, segment + SEGS, ... | the owning invocation(s) of the row; disjoint slots, equal values (both segments hold the same running maximum) | the subgroup's `coopMatLoad` of the divisor tile | barriers after the writes; barriers after the loads |
| e = exp(score - maximum) | `Psh`, own segment | the owning invocation | the subgroup's `coopMatLoad` in attention x V | barriers after the writes; barriers after attention x V, before the next block's score store |
| row sum (once, at the end) | `Rsh[row x SEGS + segment]` | the owning invocation | both invocations of the row | barriers after the write (the last reads of `Rsh` were followed by barriers) |
| denominator (once) | `Psh[row x P_STRIDE + j]`, `j` as for `Dsh` | the owning invocation(s); disjoint slots | the subgroup's `coopMatLoad` | barriers after the writes (the last loads of `Psh` were followed by barriers) |

No slot is written by two invocations, no value is reduced into one shared slot by several lanes (the lesson
of L8 does not arise in this kernel), and no invocation reads a slot another invocation wrote without an
execution barrier between the write and the read. Every barrier is in control flow that is uniform over the
subgroup: the block loop bound and both early returns depend on the workgroup only, and the one conditional
block is entered on `subgroupAny`. The 780M's `fused3` has the same structure with `memoryBarrierShared()`
alone, which orders nothing between invocations that are scheduled independently; that form is not built here.

Where the one-subgroup assumption is enforced: (1) the local size is the variant's `g11s32` = 32 invocations
(`pick_fused_gwg`, `pick_required_lwg`); (2) the pipeline is created with required subgroup size 32 (the
`SUBGROUP_SIZE` parameter of the yaml becomes `// REQUIRED_SUBGROUP_SIZE = 32`, read back by
`resolve_required_subgroup_size`; this card's minimum and maximum subgroup size are both 32); (3) the kernel
returns unless `gl_NumSubgroups == 1` and `gl_SubgroupSize == 32`. The copy pass `kvt` has no shared memory and
writes each element of the copies from exactly one invocation.

Arithmetic: scores and the attention x V accumulation are fp32 cooperative-matrix accumulators as in the
parent's kernels; the score is rounded to fp16 once (store to `Psh`), e is computed in fp32 against the running
maximum and rounded to fp16 for the matrix multiply, the row sum and the final division are fp32. The summation
order differs from the three-kernel path, so the candidate is judged by the reference-error rule (D3).

The shared dev-zone test `backends/vulkan/test/sarc_dev/test_llama_microbench.cpp` changes by insertion only:
eight blocks delimited by `// >>> 4070ti-fused <id>` and `// <<< 4070ti-fused <id>`, 99 added lines, no line of
the parent's file removed or modified (`git diff 6050b1287 HEAD` of the file has no `-` line). A case that
dispatched a fused kernel is judged in the block `verdict`, placed before the existing verdict: none of the
three kernels may have run, the numeric comparison and its tolerances are the existing ones, and the two report
lines have the existing form with `fused=<kernel>` added. A case without a fused kernel never enters a block
and prints exactly what the parent's test prints. The other blocks add the fused kernel and its copy pass to
the perf suite's total, print the timed runs, confirm a perf case served by a fused kernel, and add the 780M's
`peaked` and `fused` tiers (a per-case Q scale, applied after the existing generation loop).

The first version of this candidate (commit `6217da0a9`, build `topic2`) edited nine places of that file in
place. That broke R3 (append only, in delimited blocks) and was found in review; the insert-only form replaces
it (build `topic3`) and everything was rebuilt, gated and timed again. The sessions on `topic2` are kept as
evidence and are not the reported result.

## Hook control (D4 conditions)

| condition | evidence |
|---|---|
| with nothing selected every dispatch is as before | `s1-ctl-noenv`: unmodified `verify.sh` on `topic1` (`35e3728c9`) with no environment equals `s0-pristine-verify` line by line, rates aside (34 lines, `control.diff` empty, verify-check ACCEPT) |
| the same under the parent's own profile | `s1-ctl-parent`: `verify.sh` with `4070ti-refine1` equals `s0-parent-verify` line by line (`control.diff` empty); A/A `s1-aa` geomean 1.0039 |
| `test_sarc_select` unchanged | 1240 checks, 31 rows (release tables); 1536 checks, 33 rows (dev zone): the counts of the first campaign's final tree |
| `spirv_golden.py` unchanged | PASS, 53 shipped variants, on `parent`, `topic1`, `topic2` and `topic3`; beyond the golden, all 1513 SPIR-V files of the parent build are byte-identical in `topic1`, `topic2` and `topic3` (the last two add three, the same bytes in both: the two fused variants and the copy pass). The 8da4w kernel shared with the Orin is among them |
| reproducible from the committed branch, no local patch | `topic3` is commit `ed8b5af91` of this branch, built from its export; the hook controls `s4-ctl-noenv` and `s4-ctl-parent` (CONTROL_SAME), the A/A re-check `s4-aa` (geomean 1.0002), the gate `s4-c1` and the closing session `s5-pristine` are on that build. The branch head differs from it only under `openspec/` (`git diff --name-status ed8b5af91 HEAD -- . ':!openspec'` is empty). `topic2` (`6217da0a9`) carried the first gate `s2-c1`, superseded after the review |
| `sarc/tools/check.sh` | PASS. It does not report the hook: `impl/sarc/` is inside the zones it checks and `impl/SDPA.cpp` is listed in `sarc/HOOKS`; the four files are named here under the owner decision of 2026-10-05 |

## Result

One candidate, accepted: **+11.49 % geomean over the parent** (the first campaign's stack), inside the band the
task predicted (+8 to +12 %), **+63.28 % over the pristine `dev/1.5` state** (the first campaign ended at
+46.35 %). The numbers are those of build `topic3` (`ed8b5af91`), gate `s4-c1` and closing session
`s5-pristine`, the sessions that record the driver's throttle reasons per run. tok/s, median of 7 valid
interleaved runs per arm:

| cell | pristine `dev/1.5` (`s5-pristine`) | published (`cells.csv`) | parent, `4070ti-refine1` (`s4-c1`) | `4070ti-fused1` (`s4-c1` / `s5-pristine`) | gain over the parent | gain over pristine | gain over published |
|---|---:|---:|---:|---|---:|---:|---:|
| 1B 4w | 19883.5 | 19692.3 | 29681.2 | 35310.3 / 35929.8 | +18.97 % | +80.70 % | +82.46 % |
| 1B 8da4w | 21113.4 | 20898.0 | 32507.9 | 39384.6 / 39384.6 | +21.15 % | +86.54 % | +88.46 % |
| 3B 4w | 8752.1 | 8752.1 | 12962.0 | 14027.4 / 14027.4 | +8.22 % | +60.27 % | +60.27 % |
| 3B 8da4w | 9752.4 | 9660.4 | 14948.9 | 16254.0 / 16254.0 | +8.73 % | +66.67 % | +68.25 % |
| 8B 4w | 4511.0 | 4491.2 | 6023.5 | 6380.1 / 6400.0 | +5.92 % | +41.88 % | +42.50 % |
| 8B 8da4w | 5056.8 | 5031.9 | 6989.8 | 7474.5 / 7501.8 | +6.93 % | +48.35 % | +49.09 % |
| geomean | | | | | **+11.49 %** | **+63.28 %** | +64.34 % |

- `s4-c1` (the gate, 2026-10-09 01:09 to 02:38 UTC): GATE_ACCEPTED, plain pass: 36 of 36 SDPA passes (12 each of
  `all`, `extended`, `full`; 192 cases) with 0 mismatches and `pairing=ok`, `verify.out` equal to the parent
  snapshot line by line with rates removed, next token SAME in 24 of 24 rows, 84 timed runs all valid, 12
  traces. `s5-pristine` (02:40 to 02:54 UTC): 84 timed runs all valid, `gate_check.py session` ACCEPT with 0
  findings, next token SAME in 24 of 24 rows against the stock attention kernels too. In both, every timed run
  has at least 48 samples of the driver's throttle reasons and no thermal reason in any of them. The pristine
  arm is within 1.03 % of the published numbers in every cell.
- History: the first gate (`s2-c1`, build `topic2`) read +11.64 % and the first closing session (`s3-pristine`)
  +63.53 %. Review round 1 found that they had not recorded throttle reasons, that the gate lacked the 12 `all`
  passes, and that the shared test file had been edited in place. They are kept as evidence with that
  limitation; the absence of throttling in them is not inferred. The candidate's medians agree with the new
  sessions within two steps of the 1 ms timer in every cell (at most 1.6 %, on 3B 8da4w).
- 1B cells are whole milliseconds (69 -> 58 ms, 63 -> 52 ms); one timer step is 1.7 to 1.9 % of the candidate's
  time. ETDump dispatch totals: 67.6 -> 56.3 ms and 61.2 -> 50.0 ms.

Where the gain comes from: the attention block alone (warm ETDump, ms per prefill, parent -> candidate;
`results/4070ti/sessions/s4-c1/trace/attention.csv`):

| cell | QK^T + softmax + attention x V (parent) | fused kernel + K / V copy (candidate) | attention removed | linear GEMM | everything else | attention share of the prefill |
|---|---|---|---:|---|---|---|
| 1B 4w | 4.18 + 7.47 + 5.42 = 17.07 | 5.71 + 0.11 = 5.82 | 66 % | 34.30 -> 34.21 | 16.19 -> 16.29 | 25 % -> 10 % |
| 1B 8da4w | 4.15 + 7.46 + 5.41 = 17.02 | 5.70 + 0.11 = 5.81 | 66 % | 27.66 -> 27.67 | 16.50 -> 16.56 | 28 % -> 12 % |
| 3B 4w | 8.44 + 9.80 + 8.17 = 26.40 | 13.58 + 0.34 = 13.92 | 47 % | 96.58 -> 95.74 | 33.06 -> 32.94 | 17 % -> 10 % |
| 3B 8da4w | 8.35 + 9.70 + 8.09 = 26.14 | 13.34 + 0.33 = 13.67 | 48 % | 74.76 -> 74.38 | 35.11 -> 35.01 | 19 % -> 11 % |
| 8B 4w | 13.01 + 14.92 + 11.63 = 39.56 | 20.29 + 0.44 = 20.73 | 48 % | 239.58 -> 238.70 | 60.96 -> 60.79 | 12 % -> 6 % |
| 8B 8da4w | 12.51 + 14.83 + 11.49 = 38.83 | 19.40 + 0.42 = 19.82 | 49 % | 181.16 -> 181.27 | 71.21 -> 73.08 | 13 % -> 7 % |

Kernel level, for comparison with the other devices (`test_llama_microbench --sdpa`, us per layer at S = 2048,
three interleaved rounds on `topic3`, `results/4070ti/screens/sdpa-screen2.csv`): fused kernel plus copy pass
against QK^T + softmax + attention x V of the parent: 1B 369 to 381 against 1059 to 1065 (0.35 to 0.36x), 3B 488
to 510 against 935 to 942 (0.52 to 0.55x), 8B 619 to 654 against 1211 to 1215 (0.51 to 0.54x); the stock
kernels of `dev/1.5` take about 3220, 3545 and 4705. The d64 variant gains more than the d128 one, as the 1B
cells show end to end. (`sdpa-screen1.csv` is the same screen on `topic2`: the same ratios within 0.02.)

Percent of the freshly measured roofs (igpu-roofline `fast`, run `roofline/2026-10-08-fast`, driver 615.71.09,
clocks as found; `results/4070ti/roofline/`, rates from `s4-c1`'s traces): matrix fp16 182.8 TFLOP/s, fp16 with
fp32 accumulator 92.3 TFLOP/s, int8 369.2 TOP/s, DRAM read 714 / write 641 / copy 646 GB/s, all confirmed with
3 repeats within 0.6 % and equal to the run of 2026-10-04 within 0.3 %.

| kernel | rate in the prefill | roof | percent |
|---|---|---|---|
| fused attention, d64 (1B) | 48.9 to 49.0 TFLOP/s of executed matrix work (98.5 % of it below the causal diagonal) | 92.3 (fp16 -> fp32 matrix) | 53 % |
| fused attention, d128 (3B, 8B) | 54.8 to 58.5 TFLOP/s (97 % causal) | 92.3 | 59 to 63 % |
| 4w linear (`ga` tiles), unchanged | 116 to 121 TFLOP/s | 182.8 (fp16 matrix) | 64 to 66 % |
| 8da4w linear (zpgtr), unchanged | 144 to 158 TOP/s | 369.2 (int8 matrix) | 39 to 43 % |

Negative results and things that did not go as written:

- No candidate was rejected and no gate failed, but the first gate was incomplete and the campaign was reopened
  once after review (see History above). No runner abort occurred in the 920 `llama_main` calls of this
  campaign (176 in eight `verify.sh` runs, 696 in six sessions, 24 traced, 24 in the quick look): all rc 0, with the model file warmed before every call (D5).
  One slow load was recorded among the 696 session runs: `prefill-8b-8da4w-parent-r1` of `s4-c1` (3.0 s before
  the first loaded clock sample, `results/4070ti/sessions/s4-c1/loads.csv`); it ended rc 0, is valid under the
  fixed criteria (clock, samples, no thermal reason) and is kept. The other five sessions record none.
- Outside the production shapes the candidate's error against the reference is larger than the parent's in one
  metric on 3 of 9 test shapes, by 0.2 to 5 %. Criterion 1 of D3 names the production shapes, where it is not
  larger; the numbers are all in `results/4070ti/probe/fused1-topic3/reference-error-rule.txt` (and, identical,
  in `probe/fused1/` for `topic2`).
- Measured against the first decision's test (twice the distance of the parent's two linear arms), the three 4w
  cells are outside (`compare.csv`, last line), as the first campaign's candidates were; D3 replaced that test
  for arithmetic changes, and no next-token item differed, so neither rule was needed for the verdict.
- The kernel text is not unchanged from the 780M's, as the task expected: one guard was added (see the table
  above). It costs nothing measurable and turns a silent assumption into a checked one.
- The repeat count became 7 instead of 5 by the rule fixed before the A/A session; both triggers were one step
  of the 1 ms timer on 1B 8da4w.
- Only the two variants the AMD cards ship were built. Whether another tile is faster on this card is not known
  and was not asked.

What limits further progress on this card:

- Attention is now 6 to 12 % of the prefill and the fused kernel runs at 53 to 63 % of the fp16 -> fp32 matrix
  roof; even at the roof it would return about 5 % on 1B, 4 % on 3B and 2.5 % on 8B.
- The linear kernels are 55 to 75 % of the prefill (34 of 56 ms on 1B 4w, 239 of 321 ms on 8B 4w) at 64 to
  66 % (4w) and 39 to 43 % (8da4w) of their matrix roofs; the first campaign swept their tiles and staging and
  found nothing above the noise band. In 8da4w a wave still spends more time fetching weights than multiplying.
- "Everything else" (copies and views, elementwise kernels, the 8-bit quantisation of 8da4w, RMSNorm) is 16 to
  73 ms per prefill, 19 to 33 % of it, in kernels this campaign does not own.

## Final verification

The final stack is `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=4070ti-fused1` on the committed branch
(build `topic3` = `ed8b5af91`, no local patch; later commits change only this directory). Its gate `s4-c1` is
the verification of everything together: unmodified `verify.sh` with the candidate environment on the timed
binaries, the attention tiers (12 passes each of `all`, `extended` and `full`; one pass each of `fused` and
`peaked` before it), the golden check on the build, the reference-error evidence on the same build, and the
timed session against the parent; `s5-pristine` is the timed session against the pristine state. Every timed
run of both records the driver's throttle reasons and has no thermal one. `sarc/tools/check.sh --no-build`:
PASS (output in `STATUS.md`). `tools/test_gate_check.py`: 48 tests pass.
