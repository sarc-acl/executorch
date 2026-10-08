# STATUS: sarc-1.5-orin-fused-port

**2026-10-08 20:20 UTC. Started. Nothing measured yet.**

All times are UTC from `date -u`.

## Running now

- Workstation: `build-parent` (detached, `.artifacts/jobs/build-parent.{status,out}`): cross-build of the parent
  `8973ced76` (tag `parent`), source export in progress, then the container build under the desktop build lock.
- Device `duck-naughty`: nothing of this campaign. Found idle at 19:37 UTC (no runner, no Actions job, 6.5 GB
  available, swap unused, 53 C, power mode 15 W).

## Parent

`8973ced76` with `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=orin-refine5 ET_VK_SARC_SOFTMAX_VARIANT=orin_g64`
(the first campaign's final stack; the softmax hook `307abb2ed` is committed, so the softmax is named by the
dev-zone variable, no local patch). Expected (`s10-final`): 1B 1489.45 / 1382.85, 3B 628.99 / 570.32,
8B 295.53 / 269.19 tok/s (4w / 8da4w). Pristine state: the same commit, no environment.

## Next step

Deploy the parent build, parent control (`s0-parent-verify`, unmodified `verify.sh` with the parent
environment; `s0n-noenv` with nothing selected), baseline + A/A session, clock floor. In parallel on the
workstation: the hook commit and the candidate-1 build.

## Thresholds

`tools/thresholds.txt` (committed before the first measurement).
