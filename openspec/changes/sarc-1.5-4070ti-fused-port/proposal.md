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

Not carried over: the generators (`gen_4070ti_*.py`, `devzone.py`), the linear screens and phase-timing tools
(`screen.sh`, `prof.sh`, ...), the runner probes (`exit_probe.sh`, `first_use_probe.sh`), the local hook patches.
This campaign generates nothing and sweeps nothing. `gate_check.py`, `runrow.py`, `nexttoken.py`, `summarize.py`,
`llama_main_rc.sh`, `warm_file.py`, `stage.sh`, `gate*.sh`, `session.sh`, `gl.sh`, `mktree.sh` are unchanged
(`tools/test_gate_check.py`: 40 tests pass).

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
