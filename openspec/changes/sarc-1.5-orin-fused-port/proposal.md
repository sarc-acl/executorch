# sarc-1.5-orin-fused-port

Second campaign on the Jetson Orin Nano (`duck-naughty`). Branch `topic/orin-fused-port`, forked from
`topic/orin-prefill-refine` at `8973ced76`. It ports one thing on top of the first campaign's final stack: the
fused attention kernel of the Radeon 780M (third structure, with the RX 7600's subgroup barriers). A port, by
owner decision N1: no sampled search, no enumeration, no tile sweep; two candidates at most.

Parent of every comparison: `8973ced76` with
`ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=orin-refine5 ET_VK_SARC_SOFTMAX_VARIANT=orin_g64`
(the first campaign's final stack, `sarc-1.5-orin-prefill-refine/proposal.md`, "Outcome"). The pristine state is
the same commit with no environment.

Day-to-day state is in `STATUS.md`. This file is completed as the campaign proceeds.

## Thresholds (fixed before the first measurement)

`tools/thresholds.txt`, committed before any run of this campaign:

- baseline: every cell of the parent within 3 % of `s10-final` (1B 1489.45 / 1382.85, 3B 628.99 / 570.32,
  8B 295.53 / 269.19 tok/s, 4w / 8da4w);
- noise band +-2 %; median of 5 valid interleaved runs per arm and cell;
- clock floor: floor(0.97 x the lowest per-cell median clock of this campaign's own A/A session), device-wide;
- a run needs at least 5 clock samples in its prefill window (0.1 s sampler; the shortest prefill is 1.2 s);
- kernel screen margin 3 % in every round;
- reference-error rule (D3): "the parent" is this campaign's parent, the tuned stack's arithmetic; the stock
  kernels' error is reported beside it, not judged;
- candidate 2 only if the unpacked one-pass form is at least 3 % faster at kernel level in every round.

## Tools: what differs from the sibling (`sarc-1.5-orin-prefill-refine/tools`)

Copied, originals untouched. Not carried over: the `chain*.sh` of the first campaign, its generators
(`gen_orin_*.py`, `orin_refine.py`), the linear screen, phase-timing, decode and second-device tools and the old
local hook patch. Byte-identical to the sibling's and not listed: 37 files (`gl.sh`, `stage.sh`, `gate.sh`,
`timed.sh`, `session.sh`, `noenv_verify.sh`, `probe_*.py`, `probe_run.sh`, `ref_error_rule.py`, `sdpa_err.sh`,
`sdpa_screen.sh`, `shipped.py`, `summarize.py`, `nexttoken.py`, `calibrate_clock.py`, `roof.sh`, `roof_fast.py`,
`roof_util.py`, `mktree*.sh`, `build-extra.sh`, `wsrun.sh`, ...).

| tool | change |
|---|---|
| `common.sh` | device directory `~/hmz-sarc-orin-fused`, change directory `sarc-1.5-orin-fused-port`, workstation root `/mnt/linux-share/hmz-campaigns/jetson-fused` with the artifact directory `.artifacts` itself, `PARENT_COMMIT=8973ced76`, campaign tag `orin-fused-port`; the second Orin is removed (not available); `hold_wait` (coordinator hold, D6) called by `take_lock`; `throttle_read` and a sixth field in every clock sample: the state of the module's 12 thermal cooling devices (the fan left out) |
| `runrow.py`, `e2e5.sh` | a timed run needs at least 5 clock samples (was 2; `thresholds.txt`), and is invalid with a nonzero throttle state in its window (`thermal_throttle`) or without the record (`no_throttle_record`); the model file is read into the page cache before each cell (D5) and the load time is recorded; an 8B run is not started with less than 5000 MB available; the session log lists the cooling devices and trip points |
| `gate_sdpa.sh`, `gate_check.py`, `test_gate_check.py` | the SDPA step runs 12 passes of the tier `all` as well as `extended` and `full`, and under an `orin-fused` profile 3 passes of the 780M's tiers `peaked` and `fused`; a case the fused node serves must name that fused kernel and no QK^T / softmax / attention x V kernel, every other case the three kernels and the profile's softmax; the fused-attention banner must match the environment; 54 unit tests |
| `gatelib.sh`, `parent_verify.sh` | `PARENT_CTL_NAME` selects the control (`s0-parent-verify`: the tuned parent; `s0n-noenv`: nothing selected); the parent control takes an environment, because this campaign's parent is a tuned stack; `hold_wait` before `verify.sh` |
| `trace_analyze.sh`, `trace_kernels.py` (new) | the ETDump analysis uses the first campaign's devtools venv read-only; `trace_kernels.py` adds `attention.csv` and `kernels.csv` (the fused kernel and its copy pass by name) |
| `sdpa_screen_summary.py`, `collect.sh` | under a fused profile the per-layer total is the copy pass + the fused kernel, every round listed; the pre-checks, the probes and the error files of this campaign are collected |
| `pull.sh`, `build-orin.sh`, `deploy.sh`, `drun.sh`, `trace.sh` | paths and comments; the second Orin's mirror removed; D5 in `trace.sh` |
| new | `thresholds.txt`; `HOLD.md`; `buildq.sh`, `extraq.sh` (build queue under the desktop build lock); `memprobe.sh` (memory of the K / V copies); `chain1.sh` to `chain9.sh` (the detached device queues as they ran; `chain5`, `chain6`, `chain7` were ended before they started a job) |

## Release-zone hook (owner decision 2026-10-05)

Commit `9d91480b2` is the 780M campaign's `1c8861aa7e` cherry-picked unchanged (same patch-id; hook D4.3, the
entry point of the fused attention node): 49 added lines, none removed, in `impl/SDPA.cpp` (7),
`impl/sarc/SdpaCoopmat.cpp` (21), `impl/sarc/SdpaCoopmat.h` (8), `impl/sarc/Select.h` (13); the diff is
`git show 9d91480b2`. `Override::sdpa_fused_add` and `Override::sdpa_fused_serves` are null by default;
`sarc::add_sdpa_fused` is called once after the three SDPA nodes of an LLM-mode op and does nothing without the
first; `sarc::sdpa_fused_skip` gives the three nodes an empty launch geometry only for a call the second says the
fused node serves. **It is larger than a switch and stays subject to the owner's review before any promotion.**
On this branch the dev zone sets the two fields only when a fused variant is named (profile `orin-fused1` or
`orin-fused2`, or the kernel-timing variable `ET_VK_SARC_ORIN_SDPA_FUSED`), so every other configuration of a dev
build has them null as well. No other release-zone file is changed by this campaign
(`git diff --name-status 8973ced76 HEAD`: the four files above; everything else is dev zone or this change
directory).

## The port

What runs: for an LLM-mode attention call with fp16 buffers whose shape fits (S a multiple of the row tile and of
the column tile, `input_pos` a multiple of the column tile, head_dim 64 or 128), the node
`impl/sarc_dev/orin/SdpaOrinFused.cpp` dispatches `sarc_dev_orin_sdpa_kvt` (tile-packed copies of K and V for the
visible context) and then `sarc_dev_orin_sdpa_fused3sb_<variant>`, and the QK^T, softmax and attention x V nodes
dispatch nothing. Every other call (decode, an unaligned prompt) runs the parent's three kernels with the softmax
`orin_g64`, exactly as the parent does: an `orin-fused*` profile carries the preferences of `orin-refine5` and
names that softmax itself, so one variable selects the whole stack.

| head_dim | models | rows x columns per workgroup | `orin-fused1` (candidate 1) | `orin-fused2` (candidate 2) |
|---|---|---|---|---|
| 64 | 1B | 32 x 32, 32 invocations | `fused3sb_d64_t32x32g11s32rko` (one pass) | `fused3sb_d64_t32x32g11s32rk` (two passes) |
| 128 | 3B, 8B | 16 x 64, 32 invocations | `fused3sb_d128_t16x64g11s32rko` (one pass) | the same |

`r`: Q tiles kept in registers; `k`: tile-packed K and V; `o`: one pass with a running row maximum. The yaml also
builds the unpacked forms (`ro`, `r`), reachable only through `ET_VK_SARC_ORIN_SDPA_FUSED` for kernel timing.

What changed against the source kernels:

| item | source | here | why |
|---|---|---|---|
| fused kernel from `#version` on | `sarc_dev_780m_sdpa_fused3sb.glsl` of the RX 7600 branch (`b3bb758e38`): the 780M's third structure with `subgroupBarrier()` after every `memoryBarrierShared()` | the same text plus five lines after the row-range check: `if (gl_NumSubgroups != 1u \|\| gl_SubgroupSize != SUBGROUP_SIZE) return;` (commit `0bed38090`, as the 4070 Ti port has it) | the shared memory is private to one subgroup only if the workgroup is one full subgroup. The pipeline requires subgroup size 32 but not full subgroups (release zone), so the kernel checks it and writes nothing otherwise: the correctness tiers then fail instead of the kernel racing silently |
| constants (`MMA`, tiles, `SUBGROUP_SIZE`, `P_STRIDE`, spec constants) | | unchanged | fp16 16 x 16 x 16 matrices with fp32 result and subgroup 32 exist on this device (`results/orin/vk-caps-first-campaign.txt`) |
| copy passes | `sarc_dev_780m_sdpa_{kvt,vt}.glsl` (780M) | the same text from `#version` on; workgroup 8 x 8 | |
| names | `sarc_dev_780m_sdpa_` | `sarc_dev_orin_sdpa_` | separable content; `Sdpa780mFused.cpp` is not edited |
| selector | `impl/sarc_dev/780m/Sdpa780mFused.cpp` (`8518659efb`) | `impl/sarc_dev/orin/SdpaOrinFused.cpp`: node, launch geometry and fit rule unchanged; variants named by profile | one name per configuration |
| `impl/sarc_dev/Overrides.cpp` | | two appended blocks, `orin-fused profiles` and `orin-fused override` (16 lines) | R3 |
| `backends/vulkan/test/sarc_dev/test_llama_microbench.cpp` | | seven insert-only blocks delimited by `// >>> orin-fused <id>` (`ca62778e6`); no line of the parent's file removed or modified | R3. The first form (`0f14f2a1a`, builds `topic1`, `topic2`) changed 9 existing lines; found by reading the 4070 Ti port's review |

Shared memory of one workgroup:

| array | elements | d64 `t32x32` | d128 `t16x64` | holds |
|---|---|---:|---:|---|
| `Psh` | `WG_TILE_M x max(WG_TILE_N / 8 + 1, 4)` uvec4 | 32 x 5 x 16 = 2560 B | 16 x 9 x 16 = 2304 B | the block's scores (fp16), then e (fp16); at the end the row sums as 16 fp32 per row |
| `Rsh` | `WG_TILE_M x SEGS` float | 128 B | 128 B | per-invocation block or row maximum, at the end the per-invocation row sum |
| `Dsh` (one-pass forms only) | `WG_TILE_M x 4` vec4 | 2048 B | 1024 B | per-row rescale divisor exp(new maximum - old maximum), 16 fp32 per row |
| total, one pass / two passes | | 4736 / 2688 B | 3456 / 2432 B | |

The reading of the kernel for unsynchronised shared writes (who writes each slot, what orders each read by
another invocation), the specification sentences it rests on and the SPIR-V barrier counts are in `STATUS.md`
("Shared-memory reading of the kernel", "Specification text the shader reading rests on", and the row on the
one-full-subgroup check under "What I took from the 4070 Ti fused port"). In short: every slot of `Psh`, `Rsh`
and `Dsh` has exactly one writing invocation or is written by a cooperative-matrix store of the whole subgroup;
every read of a slot another invocation wrote comes after `memoryBarrierShared(); subgroupBarrier();`; every
read phase is closed by the same pair before the slot is written again; every barrier is in control flow that
is uniform over the subgroup (the block loop bound and both early returns depend on the workgroup only, the one
conditional block is entered on `subgroupAny`). The copy passes use no shared memory.

Arithmetic: scores and the attention x V accumulation are fp32 cooperative-matrix accumulators as in the
parent's kernels; the score is rounded to fp16 once (store to `Psh`), e is computed in fp32 against the row
maximum and rounded to fp16 for the matrix multiply, the row sum and the final division are fp32. The summation
order differs from the three-kernel path, so both candidates are judged by the reference-error rule (D3).

## Result, hook control, final verification

Written at closing. Until then the measured state is in `STATUS.md`.
