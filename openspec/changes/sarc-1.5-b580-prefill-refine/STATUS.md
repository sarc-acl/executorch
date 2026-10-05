# sarc-1.5-b580-prefill-refine: status

**2026-10-05 — started. Nothing measured yet.**

Branch `topic/b580-prefill-refine`, parent `6a7cc8cc6`. Host `fedora` (the owner's desktop), Arc B580 =
Vulkan device 0, `ETVK_DEVICE_INDEX=0`, lock `86800be2-0000-0000-0300-000000000000`.

## Running now

- Copy of the parent build from `fedora-gpu-eval` (rsync, `.artifacts/logs/rsync-parent.status`).
- Build `topic1` (this commit) in the build container, under the exclusive desktop-build lock
  (`.artifacts/logs/build-topic1.out`).

## Next

Parent control `s0-parent-verify`, then the baseline + A/A session `s1-aa` (calibration of the clock and
foreign-busy thresholds, fixed in `proposal.md` before the session).

## Blocking

Nothing.
