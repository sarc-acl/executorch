# STATUS: sarc-1.5-orin-fused-port

**2026-10-09 01:45 UTC. Candidate 1 (fused attention kernel, profile `orin-fused1`) passed its gate `s3-c1` on
build `topic1`: +6.01 % geomean over the tuned parent (1B +10.9 / +10.0 %, 3B +4.8 / +4.4 %, 8B +3.3 / +3.0 %), no
next-token item differs, reference-error criterion 1 met. NOT CLOSED: reading the 4070 Ti fused port's review
showed two things my first sessions lack (no thermal-throttle record in the timed runs; the shared test file
edited in place), so both are put right and candidate 1 is gated again on build `topic3` before anything is
reported as final. Candidate 2 (the faster form of the kernel per head_dim, `orin-fused2`) follows, with the
owner's agreement of 01:00 UTC; the closing chain is queued behind it. Nothing measured on `topic3` exists yet.**

All times are UTC from `date -u`.

## Running now

- Device `duck-naughty` (detached, `~/hmz-sarc-orin-fused/jobs/<job>.{status,out}`), three chains in a row:
  - `chain4` (waiting since 22:07, measuring since 00:35): 41-prompt real-text logits of the four arms (parent /
    candidate 1 x default / tiled, builds `parent` and `topic1`), then the logits at the gate's unaligned position,
    the comparison and `ref_error_rule.py` (`probe/c1-fused/`), and the peaked-tier error of both arms. One arm
    takes 42 minutes (parent-default 00:35 to 01:17), so it ends about 03:30, not 02:10 as written before.
  - `chain6` (queued 00:55 behind `chain4`), about 8.5 hours, until about 12:00: `s4-aa2` (A/A re-check, parent
    build against `topic3`, both with the parent environment, committed clock floor, with the throttle record);
    `sdpa-error2` (stock / parent / candidate 1 / candidate 2, `topic3`'s test binary); `s5-c1` (candidate 1 gated
    again: parent build against `topic3` with `orin-fused1`); `c2-pre` and `s6-c2` (candidate 2, `orin-fused2`,
    against candidate 1, both on `topic3`).
  - `chain7` (queued behind `chain6`), the closing chain, 4.5 to 7.5 hours: `s7n-noenv` (hook control on `topic3`);
    `s7-final` (only if the final stack is `orin-fused2`: its full gate against the tuned parent); `s8-pristine`
    (final stack against the pristine state); the real-text probe of the final stack on `topic3`
    (`probe/final-fused/`); `mem1` (memory probe of the K / V copies); `roof-final` (fresh roofs, igpu-roofline
    `fast`, 41 minutes in the first campaign).
  - `chain5` (the first form of the candidate-2 chain, on `topic2`) was ended at 00:41 before it started a job.
- Workstation: nothing. `topic3` (`ca62778e6`) was built 00:42 to 00:55 and is deployed with its `logits_dump`.
- Coordinator hold: `tools/HOLD.md` (device: `~/hmz-sarc-orin-fused/HOLD`; builds: `.artifacts/HOLD`). None seen.

## Final stack: the rule, fixed 01:40 UTC before `s5-c1` and `s6-c2` have a number

`chain7.sh` applies it without me: the final stack is `orin-fused2` only if `s6-c2` is `GATE_ACCEPTED` and its
geomean gain over candidate 1 is at least 2 % (outside the noise band); in every other case it is `orin-fused1`,
provided `s5-c1` is accepted (otherwise the chain stops and nothing is final). This is the owner's sentence of
01:00 UTC ("if `s6-c2` is inside the band, candidate 1 alone is the final stack") read on the geomean, the
quantity R11 stops on. If the two 1B cells alone come out above 2 % while the geomean stays under it, candidate 2
is still not adopted and the cells are reported as measured. Either way the campaign stops after candidate 2
(`thresholds.txt`, "stop").

The build that is measured as final is `topic3` = `ca62778e6`. Later commits change only this change directory
(tools, evidence, text): `git diff ca62778e6 HEAD` outside `openspec/changes/sarc-1.5-orin-fused-port/` is empty
and is checked again at closing, so `topic3` is the build of the branch head's code.

## Decision needed from the owner

**Candidate 2 goes beyond the clause I fixed for it.** `tools/thresholds.txt` (committed before any measurement)
says candidate 2 exists only if the unpacked one-pass form is at least 3 % faster at kernel level, and otherwise
there is none. That clause is answered: the unpacked forms are 1.5 to 3.4 times slower (`sdpa-screen1`), so by
it there is no candidate 2. The same screen, which measured all four forms of the ported kernel, showed
something the clause did not anticipate: for head_dim 64 the **two-pass** packed form is 17 % faster than the
one-pass form in every round (9.62 against 11.59 ms per 1B layer); for head_dim 128 the one-pass form stays
faster (13.9 against 19.2 ms, 18.3 against 25.3). The task's own candidate-2 list names the one-pass / two-pass
choice (in the other direction), R8 says "choose the best kernel per shape" with the 3 % in every round margin,
and no new kernel, tile or search is involved: both forms are variants of the one ported shader, already built.
So I gate it as candidate 2 (`orin-fused2` = two passes for head_dim 64, one pass for 128), expecting about
+2 % on the two 1B cells and nothing elsewhere (under +1 % geomean), and I say here that this is my reading, not
the letter of my own pre-registered clause, which I have not edited.

**Answered by the owner, 2026-10-09 01:00 UTC (task file, "candidate 2 as you read it"):** agreed; gate
`orin-fused2` as candidate 2; the pre-registered clause stays unedited and this note stays. If `s6-c2` is inside
the band, candidate 1 alone is the final stack and the campaign closes by N3 with candidate 2 as the first
sub-threshold candidate. Nothing is open under this heading now.


## Candidate 1 (`orin-fused1`): gate `s3-c1`, `GATE_ACCEPTED 2026-10-09T00:34:54Z all steps passed`

Parent arm: build `parent` (`8973ced76`) with `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=orin-refine5
ET_VK_SARC_SOFTMAX_VARIANT=orin_g64`. Candidate arm: build `topic1` (`0f14f2a1a`) with `ET_VK_SARC_UNVERIFIED=1
ET_VK_SARC_DEV_PROFILE=orin-fused1`. Tok/s, median of 5 valid interleaved runs per arm; every number below
recomputed from `runs.csv` (`results/orin/sessions/s3-c1/`):

| cell | parent | candidate 1 | gain | repeat spread (parent / candidate) | next token (4 prompts) |
|---|---:|---:|---:|---|---|
| 1B 4w | 1489.45 | 1651.61 | +10.89 % | 0.07 / 0.08 % | SAME |
| 1B 8da4w | 1383.78 | 1521.55 | +9.96 % | 0.47 / 0.22 % | SAME |
| 3B 4w | 628.99 | 659.37 | +4.83 % | 0.22 / 0.23 % | SAME |
| 3B 8da4w | 570.16 | 595.00 | +4.36 % | 0.06 / 0.06 % | SAME |
| 8B 4w | 295.40 | 305.12 | +3.29 % | 0.10 / 0.10 % | SAME |
| 8B 8da4w | 269.08 | 277.21 | +3.02 % | 0.05 / 0.08 % | SAME |

Geomean **+6.01 %**, every cell outside the +-2 % band. Inside the task's expected +5 to +12 %, at its lower end.

- 60 timed runs, all valid by the rules of that session: rc 0, 2048 prompt tokens, 0 generated, no foreign GPU
  process, 12 to 73 clock samples per window, median clock 612 MHz in every run (floor 593), start temperature
  57 to 62 C. **Limitation: no thermal-throttle record exists for these runs** (below); `s5-c1` repeats the
  gate with it.
- SDPA correctness, 12 passes of `all`, `extended`, `full` and 3 of `peaked`, `fused` (42 logs, 222 cases): 0
  mismatches, every case served by the fused kernel alone (`qk=? softmax=? av=?`, `pairing=ok`);
  `gate_check.py sdpa`: ACCEPT, 0 findings.
- Unmodified `verify.sh` with the candidate environment on the timed binaries: `gate_check.py verify` against
  `s0-parent-verify` ACCEPT, 0 findings; `verify.out` equals the parent snapshot line by line with the rates
  removed (0 differing lines); all four default-vs-tiled items SAME; decode 31 tokens (18.9 / 10.8 tok/s, the
  parent's 19.1 / 10.9: decode is not served by the fused node).
- `gate_check.py session`: ACCEPT, 0 findings; next token parent vs candidate SAME in 24 of 24 rows;
  `env-check`: ACCEPT. Shipped SPIR-V of `topic1`: UNCHANGED (53 of 53).
- **How it is recorded:** the gate wrote `all steps passed` because no next-token item differs. The candidate
  replaces the three attention kernels, an arithmetic change, so it is recorded under the reference-error rule
  and not as a plain pass, once `chain4` has produced the real-text evidence. Criterion 1 as fixed in
  `thresholds.txt` (candidate not larger than this campaign's parent in rms and in maximum on all five S = 2048
  cases) is met (`results/orin/sdpa-error1/`, one binary, same inputs):

  | case (S = 2048) | rms: stock / parent / candidate 1 | maximum: stock / parent / candidate 1 |
  |---|---|---|
  | `1b_head_config_s2048` | 8.547e-05 / 2.101e-05 / 2.049e-05 | 1.713e-03 / 9.135e-04 / 7.227e-04 |
  | `3b_head_config_s2048` | 8.686e-05 / 2.068e-05 / 2.053e-05 | 1.408e-03 / 7.828e-04 / 7.061e-04 |
  | `8b_head_config_s2048` | 8.696e-05 / 2.057e-05 / 2.022e-05 | 1.587e-03 / 8.911e-04 / 7.911e-04 |
  | `tiny_gqa_s2048` | 8.357e-05 / 2.080e-05 / 2.050e-05 | 1.498e-03 / 6.943e-04 / 6.078e-04 |
  | `tiny_d128_s2048` | 8.384e-05 / 2.071e-05 / 2.034e-05 | 1.467e-03 / 7.796e-04 / 5.595e-04 |

  Outside the criterion's cases, on record: `8b_head_config_s1024_pos1024` maximum 6.942e-05 -> 7.226e-05
  (candidate 4 % larger), rms 9.018e-06 -> 8.980e-06. The 4070 Ti port measured the same numbers to three digits
  with the same kernels (its STATUS: 1B 2.101e-5 / 2.049e-5, 9.14e-4 / 7.23e-4): the two NVIDIA drivers agree.

Where the gain comes from (warm ETDump of both arms of `s3-c1`, ms per 2048-token prefill, parent -> candidate
1; `results/orin/sessions/s3-c1/trace/attention.csv`, `tools/trace_kernels.py`):

| cell | QK^T | softmax | attn*V | fused kernel | K / V copy | attention total | linear GEMM | dispatch total |
|---|---|---|---|---:|---:|---|---|---|
| 1B 4w | 107.6 -> 0 | 133.1 -> 0 | 61.5 -> 0 | 172.4 | 3.1 | 302.2 -> 175.5 | 655.5 -> 654.6 | 1360.0 -> 1225.1 |
| 1B 8da4w | 107.6 -> 0 | 133.0 -> 0 | 61.5 -> 0 | 172.2 | 3.1 | 302.1 -> 175.3 | 778.1 -> 774.3 | 1468.5 -> 1334.5 |
| 3B 4w | 198.5 -> 0 | 174.1 -> 0 | 143.8 -> 0 | 371.4 | 11.5 | 516.4 -> 382.9 | 1862.4 -> 1861.0 | 3236.9 -> 3088.4 |
| 3B 8da4w | 198.9 -> 0 | 174.3 -> 0 | 143.7 -> 0 | 371.1 | 11.2 | 516.9 -> 382.4 | 2171.0 -> 2165.4 | 3571.3 -> 3421.4 |
| 8B 4w | 301.3 -> 0 | 266.0 -> 0 | 216.3 -> 0 | 561.6 | 13.0 | 783.5 -> 574.6 | 4657.4 -> 4656.6 | 6914.1 -> 6690.8 |
| 8B 8da4w | 301.2 -> 0 | 266.0 -> 0 | 216.1 -> 0 | 561.0 | 12.9 | 783.3 -> 573.9 | 5305.1 -> 5302.1 | 7591.0 -> 7368.3 |

All of the gain is attention: -42 % on 1B, -26 % on 3B, -27 % on 8B (the RX 7600 saw -76 %, the 4070 Ti -66 %
and -47 to -49 %). The copy pass is small: 3 to 13 ms per prefill, 0.19 ms a layer on 1B and 0.41 ms on 3B / 8B.
Per layer the fused kernel takes 10.8 / 13.3 / 17.5 ms. The same 17.2 GFLOP of a 1B layer would take 1.8 ms at
this device's fp16 -> fp32 matrix roof (9.7 TFLOP/s): the kernel runs at 16 % of it, where the 780M ran at 71 %
and the 4070 Ti at 53 to 63 %. The first campaign measured why: this device feeds its matrix unit from DRAM at
2.8 TFLOP/s against 8.8 from shared memory, and this kernel loads K and V tiles straight from DRAM (its design:
no shared staging). That is what limits it here (and what a staged structure would address; not in this
campaign's scope).

## Kernel level: the four forms of the ported kernel (`sdpa-screen1`, 22:12 to 22:25 UTC, build `topic1`)

`test_llama_microbench --sdpa`, ms per layer at S = 2048 (copy pass + fused kernel; the parent: QK^T + softmax +
attn*V), 3 rounds interleaved, every round listed (`results/orin/screens/sdpa-screen1.csv`):

| form | 1B (head_dim 64, `t32x32`) | 3B (head_dim 128, `t16x64`) | 8B (head_dim 128, `t16x64`) |
|---|---|---|---|
| parent, three kernels | 18.87 / 18.89 / 18.90 | 18.45 / 18.46 / 18.45 | 24.45 / 24.48 / 24.48 |
| `rko` one pass, packed (**candidate 1**) | 11.62 / 11.53 / 11.59 | 13.84 / 13.90 / 13.90 | 18.25 / 18.27 / 18.25 |
| `rk` two passes, packed | **9.61 / 9.62 / 9.64** | 19.19 / 19.20 / 19.26 | 25.26 / 25.27 / 25.33 |
| `ro` one pass, unpacked | 17.21 / 17.26 / 17.29 | 47.11 / 47.08 / 47.19 | 62.27 / 62.28 / 62.47 |
| `r` two passes, unpacked | 23.56 / 23.43 / 23.47 | 66.46 / 66.51 / 66.35 | 88.28 / 88.35 / 88.42 |

- The copy passes save far more than they cost: unpacked is 1.5 times (head_dim 64) and 3.4 times (128) slower.
  By the clause of `thresholds.txt` there is no "unpacked" candidate.
- One pass against two: for head_dim 128 the one-pass form is 28 % faster, as on the 780M; for head_dim 64 it is
  17 % slower. On the 780M the one-pass form won for both. The two-pass form computes the scores twice, but it
  declares 2 KB less shared memory per workgroup (no rescale divisors) and has no rescale branch; the first
  campaign measured that a workgroup's time on this device grows with the shared memory it declares. That is an
  observation that fits, not a measured cause.

## What I took from the 4070 Ti fused port (`origin/topic/4070ti-fused-port`, read 00:38 UTC at `ed8b5af91`)

Its campaign closed at +11.64 % and was reopened by its review for three things. Checked against this campaign:

| its finding | here | what I did |
|---|---|---|
| the gate ran no 12 passes of tier `all` | not affected: `gate_sdpa.sh` here runs 12 of `all`, `extended` and `full` since the first gate | nothing |
| no timed run recorded a thermal-throttle reason, so R6's "no thermal throttle reason" was never evaluated | **affected**: the sampler recorded clock, load, power and the GPU temperature, no throttle state. `s1-aa` and `s3-c1` are kept as measured WITH THAT LIMITATION (GPU temperature at most 66 C against trip points of 70 C (alert) and 99 C (throttle), and a 612 MHz clock in every run; that is context, not a record) | every clock sample now carries the state of the 12 thermal cooling devices of the module (`cpufreq-cpu0/4`, `devfreq-17000000.gpu`, the `*-throttle-alert` devices, `hot-surface-alert`; the fan is left out); a timed run with a nonzero state or without the record is invalid (`runrow.py`, 6 new unit tests, 54 in all). A/A re-check `s4-aa2` and the gate again (`s5-c1`) under it; thresholds unchanged |
| the shared test `test_llama_microbench.cpp` was edited in place (R3) | **affected**: my first form changed 9 existing lines | rewritten as seven insert-only blocks delimited by `// >>> orin-fused <id>` (`ca62778e6`; `git diff 8973ced76 -- backends`: 0 deleted lines in all 13 files); same cases, same seeded inputs. New build `topic3`; everything reported as final is gated on it |

Also taken: its numbers for comparison. Same kernel, same architecture family: +11.64 % there (1B +19 / +21 %,
3B +8 / +10 %, 8B +6 / +7 %) against +6.0 % here; its fused kernel removes 66 % / 47 to 49 % of attention time,
here 42 % / 26 to 27 %; its copy pass 0.1 to 0.4 ms per prefill, here 3 to 13 ms. Its reference errors equal
mine to three digits.

## Specification text the shader reading rests on (owner note 2026-10-09; R7)

From the `vulkan-docs` index (built 2026-10-08T23:04:58Z from docs.vulkan.org). The MCP server was not attached
to this session, so I called it over stdio from the shell (`search_docs`, `get_page`, `spirv_opcode`); the pages
are `spec/latest/memorymodel.md` and `glslext/latest/GL_KHR_shader_subgroup.md`.

- What must not happen (memory model, "Data Race"): "Let X and Y be operations that access overlapping sets of
  memory locations M, where X != Y, and at least one of X and Y is a write, and X and Y are not mutually-ordered
  atomic operations. If there does not exist a location-ordered relation between X and Y for each location in M,
  then there is a data race. Applications must ensure that no data races occur during the execution of their
  application."
- What `subgroupBarrier()` is (GL_KHR_shader_subgroup): "subgroupBarrier() -> OpControlBarrier( /*Execution*/
  Subgroup, /*Memory*/Subgroup, /*Semantics*/AcquireRelease | UniformMemory | WorkgroupMemory | ImageMemory)",
  and "For each active invocation within a subgroup that reaches the same dynamic instance of a subgroup
  built-in function, all active invocations within a subgroup must execute the dynamic instance of the function
  before any invocation can proceed."
- Why a store before it is ordered before another lane's load after it (memory model): "If A is a release
  barrier, B is an acquire barrier, and C is a control barrier (where A can equal C, and B can equal C), then A
  synchronizes-with B if all of the following are true: A is program-ordered before (or equals) C; C is
  program-ordered before (or equals) B; A and B are in the instance of each other's memory scopes; A and B are
  in the instance of C's execution scope."

In the kernel every exchange through `Psh`, `Rsh` and `Dsh` has such a barrier between the store and the other
lane's load (in the SPIR-V: `OpControlBarrier %uint_3 %uint_3 %uint_3400`, execution and memory scope Subgroup,
semantics AcquireRelease | UniformMemory | WorkgroupMemory | ImageMemory, one after each `OpMemoryBarrier`; 7
pairs in `d64 rko`, 9 in `d128 rko` and in `d64 rk`), and no location has two writers between two barriers (the
table under "Shared-memory reading" below). `memoryBarrierShared()` alone, which the 780M's `fused3` uses, is a
memory barrier and no control barrier: it does not make the other lanes arrive, which is why this port starts
from `fused3sb`. A workgroup is one subgroup here, so the subgroup barrier is the only one needed.

## Hook control `s2n-noenv` (21:35 to 21:52 UTC): `GATE_ACCEPTED`, owner decision D4

Unmodified `verify.sh` on build `topic1` (`0f14f2a1a`: the hook `9d91480b2` + the whole dev zone of this
campaign) with **nothing selected** (no environment), against `s0n-noenv` (the parent build, nothing selected):

- `verify.out` equal line by line with the rates removed: **0 differing lines** (`verify-lines.txt`);
- `gate_check.py verify` against `s0n-noenv`: ACCEPT, 0 findings (every correctness case, production-diff shape
  and linear dispatch state equal); 22 of 22 runner calls rc 0; no `[sarc_dev]` banner in any log;
- default-arm prefill runs 890.44 / 821.83, 360.37 / 320.15, 189.67 / 170.50 tok/s (`s0n-noenv`: 890.82 / 823.15,
  360.25 / 320.20, 189.63 / 170.33);
- `test_sarc_select` on the release tables: `PASS (1240 checks, 31 rows, 0 candidates, dev zone absent,
  unverified off)`, the parent's line; shipped SPIR-V: all 53 variants byte-identical to the parent build;
- with the parent environment selected instead, `topic1` times like the parent build: the A/A below.

So with nothing selected the branch dispatches what the parent dispatches. The entry point stays subject to the
owner's review before any promotion (hook D4.3).

## Candidate 1 pre-check (`raw/c1-pre/`, 21:54 to 22:05 UTC, one pass per tier, build `topic1`)

`ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=orin-fused1`. Banners: `softmax variant: orin_g64`, `profile active:
orin-fused1`, `orin fused attention: ...d64_t32x32g11s32rko ...d128_t16x64g11s32rko`.

| tier | cases | PASSED, 0 mismatches | kernels |
|---|---:|---:|---|
| `all` | 4 | 4 | fused only (`qk=? softmax=? av=?`, `pairing=ok`) |
| `extended` | 8 | 8 | fused only |
| `full` | 4 | 4 | fused only |
| `peaked` (sharp rows: the rescale path) | 5 | 5 | fused only |
| `fused` (S = 32, 64, 192, 320: shapes no three-kernel tile fits) | 5 | 5 | fused only |

Error against the fp32 CPU reference on the S = 2048 cases, parent (from `s0-parent-verify`) -> candidate
(this pass); the formal comparison on one binary is `sdpa-error1`:

| case | rms parent -> candidate | maximum parent -> candidate |
|---|---|---|
| `1b_head_config_s2048` | 2.101e-05 -> 2.049e-05 | 9.135e-04 -> 7.227e-04 |
| `3b_head_config_s2048` | 2.068e-05 -> 2.053e-05 | 7.828e-04 -> 7.061e-04 |
| `8b_head_config_s2048` | 2.057e-05 -> 2.022e-05 | 8.911e-04 -> 7.911e-04 |
| `tiny_gqa_s2048` | 2.080e-05 -> 2.050e-05 | 6.943e-04 -> 6.078e-04 |
| `tiny_d128_s2048` | 2.071e-05 -> 2.034e-05 | 7.796e-04 -> 5.595e-04 |

Not an S = 2048 case and so outside the criterion as fixed in `thresholds.txt`, but on record:
`8b_head_config_s1024_pos1024` reads rms 9.018e-06 -> 8.980e-06 and maximum 6.942e-05 -> 7.226e-05 (the
candidate's maximum is 4 % larger there).

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
| `topic2` | `c6be297f1` (profile `orin-fused2` added) | hard links to `topic1` + changed paths | 1620 shaders, all byte-identical to `topic1`; shipped: **UNCHANGED**. Deployed, used for nothing (superseded by `topic3` before any job ran on it) |
| `topic3` | `ca62778e6` (test support as insert-only blocks; the last commit that changes code) | hard links to `topic2` + 104 changed paths | 1620 shaders; shipped: **UNCHANGED** (53 of 53 equal to `parent`; `build/topic3.shipped.txt`); `llama_main` `d37d44dd...`, `test_llama_microbench` `c1cfe15b...` |

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

Wait for `chain4`, `chain6`, `chain7`. After each: `pull.sh`, `collect.sh`, recompute from `runs.csv`, STATUS, commit.
At the end: trace analysis and percent of the fresh roofs, `check.sh --no-build`, `proposal.md`, push.

## Thresholds

`tools/thresholds.txt` (committed in `086651dd4`, before any measurement).
