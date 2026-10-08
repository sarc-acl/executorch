# STATUS: sarc-1.5-4070ti-fused-port

**2026-10-08 19:50 UTC. Step 1 of 6 (baseline, A/A, parent snapshot). Running now: the build of the parent
`6050b1287` (detached, `<artifact-dir>/queue/q01-build-parent.sh`). No GPU job has run yet.**

Next: commit the release-zone hook (`1c8861aa7e`, cherry-pick), build it, the two parent snapshots, the hook
control, the A/A session, calibration.

Blocking: nothing.

## Per-cell numbers against the parent

None yet.

## Coordinator hold

`<artifact-dir>/HOLD` (coordinator) and `<artifact-dir>/HELD` (this campaign), `tools/common.sh: hold_point`.
Smallest unit: one cell of a timed session (about 8 minutes on 8B with cooling), one `verify.sh` (about 15
minutes), one SDPA correctness pass, one traced run, one build (about 6 minutes). While held, a timed session
releases the device lock and takes it again afterwards. Test: pending.
