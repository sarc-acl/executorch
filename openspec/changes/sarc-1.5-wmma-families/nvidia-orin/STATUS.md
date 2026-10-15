# NVIDIA Orin WMMA matrix status

Updated: 2026-10-16 00:05 UTC. Actor: Claude (claude-opus-5-5), lean protocol of the owner decision of
2026-10-15 23:30 UTC. Deadline 2026-10-17 18:00 UTC.

## Running now

Matrix queue `orin-wmma-m1-20261015` (state:
`/mnt/linux-share/hmz-campaigns/wmma-orin/.artifacts/jobs/orin-wmma-m1-20261015/state.json`, evidence mirrored
to `.artifacts/target-evidence/orin-wmma-m1-20261015/`). Steps, in order, on one binary:

1. `verify-control`: unmodified `verify.sh --pdiff`, models 1b,3b,8b, control `orin-fused1` (the snapshot rerun).
2. `session-aa`: six-cell A/A, control in both arms, 7 valid runs per arm, record-only clock; then `floor.json`.
3. Per candidate `orin-fused-wmma-01` .. `-11`: `verify-<profile>` -> `screen.py` -> `session-<profile>` (control
   vs candidate, interleaved, 7 valid runs per arm) if the production diff passes and the entry is texture3d.

Projection: about 12 x 20 min verify plus 8 x 45 min sessions, roughly 10 to 11 hours, well under 48 hours.

## Matrix

Ratio = control median tok/s / candidate median tok/s (elapsed-time ratio; > 1 means the candidate is slower).
Limits: geomean <= 1.03 and every cell <= 1.08.

| candidate | linear kernel | correctness | 1B 4w | 1B 8da4w | 3B 4w | 3B 8da4w | 8B 4w | 8B 8da4w | geomean |
|---|---|---|---|---|---|---|---|---|---|
| -01 | q4gsw t256x128k16g42s32ga tex3d | pending | | | | | | | |
| -02 | q4gsw t128x128k16g24s32ga tex3d | pending | | | | | | | |
| -03 | q4gsw t128x256k16g42s32ga buf/tex2d | pending | not timed (buffer IO) | | | | | | |
| -04 | q4gsw t128x128k16g42s32ga buf/tex2d | pending | not timed (buffer IO) | | | | | | |
| -05 | q4gsw t128x128k16g42s32ga buf/buf | pending | not timed (buffer IO) | | | | | | |
| -06 | dq8ca zpgtr t128x128k64g44s32mk32ra tex3d | pending | | | | | | | |
| -07 | dq8ca zpgtr t128x128k64g44s32mk32ra buf/tex2d | pending | not timed (buffer IO) | | | | | | |
| -08 | q4gsw bx t128x128k32g42s32f32c tex3d | pending | | | | | | | |
| -09 | q4gsw orin t256x128k16g42s32bt tex3d | pending | | | | | | | |
| -10 | dq8ca zpgtr orin_bf t128x128k64g24s32mk32ra tex3d | pending | | | | | | | |
| -11 | q4gsw t128x128k16g22s32 tex3d | pending | | | | | | | |

No timed cell, A/A or control gate exists yet. The matrix is incomplete.

## Fixed before data

Rules are in `proposal.md`, section "Lean matrix protocol". The snapshot job `orin-wmma-s0-20261015-1` failed on
a tool bug (`/proc/<pid>/task/<pid>/children` is absent on the Orin kernel). It is kept as a failed record and is
not evidence. The fix (PPid scan) is commit `aff996d876`.

## Provenance

Inputs: INPUT.md snapshot `f393c3ff7ff4fc41e4651a5109a8509097c501f6`; archive SHA256
`65d1e3ca0eb7e9dffa8e9fbbb1a100bf2cf609868720b9053de67ff8bf5247d8` and NVIDIA manifest SHA256
`ef5d13471421898c8515e9a8356caff64a20aa97432175cbcab7bf3a5cd748b1`, both reconfirmed on 2026-10-15 23:25 UTC.
Original base `617897184529ff180af491ee0d5d7de778cbb6c6`. Changed base for both arms (memory-model runtime change
of the owner decision of 2026-10-15 03:57 UTC): `cbb7df9b6a7a605d28192fbc2a8b62c6f3611d9c`, build
`orin-wmma-mm-20261015-1`, golden PASS on 53 shipped variants. Control environment `ET_VK_SARC_UNVERIFIED=1
ET_VK_SARC_DEV_PROFILE=orin-fused1 ET_VK_SARC_ORIN_SDPA_FUSED=fused3sb_d64_t32x32g11s32rko,fused3sb_d128_t16x64g11s32rko`.
Each candidate changes only `ET_VK_SARC_DEV_PROFILE` to `orin-fused-wmma-NN`; attention stays identical.
Earlier preparation and evidence: git history of this file and `proposal.md`.

## Question for the coordinator

(none)
