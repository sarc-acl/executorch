# M2a: `fused3sb`, subgroupBarrier() after every memoryBarrierShared() in the fused attention kernel; session `m2a-sgbarrier`, 2026-10-08 04:36 to 07:50 UTC

Owner decision 2026-10-07 23:08 UTC: its own gated candidate, reported either way. The kernel is a copy of
`sarc_dev_780m_sdpa_fused3.glsl` with 13 `subgroupBarrier();` lines added (`glsl/sarc_dev/sarc_dev_780m_sdpa_fused3sb.glsl`; diff
it against the 780M file), variants `fused3sb_d64_t32x32g11s32rko` and `fused3sb_d128_t16x64g11s32rko`. Parent arm = candidate 3
(build `c3`, `rx7600-refine2`, `fused3` variants). Candidate arm = build `c4` (commit `129cea7ac`) with the same profile and
`ET_VK_SARC_780M_SDPA_FUSED=fused3sb_d64_...,fused3sb_d128_...`.

Shader read (R7, by this actor): every added barrier is in subgroup-uniform control flow (straight-line code, the loops over
blocks, `if (SEGS > 1u)` constant, `if (subgroupAny(...))`); the workgroup is one subgroup (32 invocations, no `barrier()`), so
the execution barrier plus the memory barrier now order every cross-lane read of `Rsh`, `Dsh`, `Psh` after the writes, instead
of relying on lockstep. No shared location has two writers (unchanged from the 780M file).

| check | result |
|---|---|
| timed session (`runs.csv`, 60 timed runs, all valid; `summary.csv`) | geomean **+0.00 %**; cells 0.00 / 0.00 / 0.00 / -0.19 / 0.00 / +0.17 %; next token SAME in all cells |
| SDPA tiers `all` / `extended` / `full` with the candidate env, 12 passes each | 0 failed, 0 mismatches, `pairing=ok` in all 36 (`sdpa-correctness/summary.txt`); the `fused3sb` kernels dispatched; the table control pass of each tier too |
| SDPA output vs candidate 3 (`sdpa-error/bitwise.txt`) | **21 of 21 byte-identical** (`all`, `extended`, `peaked`, `full`) |
| error against the fp64 reference (`sdpa-error/error.csv`) | 17 of 17 rows `yes`, rms and max ratio exactly 1.000 (identical outputs) |
| `verify.sh` | identical to the candidate-3 gate's `verify.out` (rates removed); against the snapshot `s0` only the two kernel-name lines of the linear kernels differ (candidate 3's) |

Adopted under the rule written beforehand in `proposal.md` (gate passes, byte-identical output, change not worse than -2 %): the final stack
uses `fused3sb`. It gains nothing in speed (as expected) and removes the formal race of the 780M kernel.
