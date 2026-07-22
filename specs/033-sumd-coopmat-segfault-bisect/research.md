# Research: SUMD Driver Bisect for the Coopmat-Dispatch Segfault Regression

**Feature**: `033-sumd-coopmat-segfault-bisect` | **Date**: 2026-07-21

No `[NEEDS CLARIFICATION]` markers remained in the spec — this document resolves the remaining
*technical* unknowns needed to execute the bisect, not spec-level ambiguity.

## 1. Range endpoints (already fixed, not calendar-derived)

**Decision**: Use the exact two SHAs already measured this session, rather than resolving from a
calendar date (unlike `specs/032`, which had to pick nearest-to-date endpoints):

- **Good**: `f14c51b6f8` (`f14c51b6f850dbe6d1becfccef8e264e435c373b`, 2026-06-15, "SC: Fix src size
  for variable_input_width in SSA") — confirmed `good` via `COOPMAT_BENCH_CORRECTNESS_ONLY=1
  test_coopmat_linear_bench_origcm` on M5 EVT1 with driver hash `c9861e9906d03fa2c7d48b804e1a1c80`:
  all 16 test cases `PASSED`.
- **Bad**: `7bb715f7cc` (2026-07-21, "xgl: use DeferCompileOptimizedPipeline") — confirmed `bad`:
  first test case passes, second test case (coopmat dispatch) segfaults, tombstone confirms
  `SIGSEGV`/`SEGV_MAPERR` inside `vulkan.samsung.so`, driver hash
  `1ebb7318b5dd8cd3fb2449d7b0b8b6ce`.

**Rationale**: Both endpoints were already built fresh and measured under conditions close to
(though not identical in tooling to) the bisect harness itself — User Story 1 re-confirms them
*under the harness*, not from scratch, since the SHAs and verdicts are already known with high
confidence.

**Alternatives considered**: Re-deriving endpoints from a calendar range (e.g. "everything since
`f14c51b6f8` landed") — rejected; we already have a tighter, empirically-confirmed disagreeing
pair, so there's no reason to widen the range and add unnecessary bisect steps.

## 2. Range size and bisect step budget

**Finding**: `f14c51b6f8..7bb715f7cc` on SUMD `main` spans **303 commits** (confirmed via `git
rev-list --count`). `git bisect` converges in ⌈log2(N)⌉ steps for a monotonic range, so **~9
interior steps** are expected, plus the 2 endpoint re-confirmations from User Story 1 — call it
**~11 build+flash+test cycles** nominal, more if any commits are `skip`ped.

**Implication for planning**: each cycle is a full SUMD Android release build (observed this
session to take on the order of 10-20 minutes) + NFS stage + adb flash + one correctness-bench
invocation, plus a mandatory pre-flash backup and post-step restore of whatever driver was
already on the shared device (see §6). This is a multi-hour investigation.

## 3. Driving `git bisect` against a hardware-in-the-loop test

**Decision**: Drive `git bisect` interactively (`git bisect start/good/bad`, one step at a time),
same as `specs/032`.

**Rationale**: Same reasoning as `specs/032` §3, and more strongly warranted here: this session
observed the M5 EVT1 board drift to at least three different undocumented driver hashes within a
few hours (consistent with a teammate actively flashing their own builds concurrently) — an
unattended `git bisect run` would have no way to detect and back up a concurrent drift before
overwriting it. Step-by-step driving lets each step check-and-back-up the device's current state
immediately before flashing.

**Alternatives considered**: `git bisect run scripts/bisect-test.sh` fully unattended — rejected
for the same shared-device-drift reason, even more so than in `specs/032` given how frequently
this specific board drifted this session.

## 4. Clock pinning (workspace default, not a device-specific override)

**Finding**: Unlike `specs/032` (which had to override to M41's own non-default maximum), M5 EVT1
*is* this workspace's default pin target — 509/2730/663 MHz (GPU/MIF/INT) is already the
documented default for this exact device.

**Decision**: Reuse `/sarc-c/gpusw/users/yanwen.xu/android-run/pin_freqs.sh` with no overrides:

```bash
S=0000088f8e579c33 GPUFREQ=509000 MIFFREQ=2730000 INTFREQ=663000 \
  /sarc-c/gpusw/users/yanwen.xu/android-run/pin_freqs.sh
```

**Rationale**: Consistency with every other measurement taken on this device this session. Per
spec Assumptions, the crash/no-crash predicate is not expected to be clock-sensitive — this is a
safety/consistency default, not a load-bearing part of the bisect logic itself.

## 5. Build/deploy mechanics (already documented, reused as-is)

**Decision**: Follow `/local/yanwen.xu/sumd/CLAUDE.md` verbatim for every bisect step: `uv run
scripts/run.py --os android --build --build-type release` (with `vulkan-sdk` stripped from
`LD_LIBRARY_PATH`, per the documented GpuRt "Too many users" build-failure workaround), stage the
resulting `.so` to a per-worktree NFS subdir with a `cmp` verify, then the standard adb push
sequence to M5 EVT1 (`sj1-dmckee-d01`, serial `0000088f8e579c33`). Building happens inside a fresh
`git worktree add ../<short-sha> <sha>` per candidate commit — this session already used this
exact pattern twice (`f14c51b6f8-revert-69e887`/`main`'s own worktree, and `7bb715f7cc`).

**Rationale**: Already validated multiple times this session; no reason to deviate.

## 6. Driver backup/restore protocol (new relative to `specs/032`)

**Decision**: Immediately before flashing any candidate commit's driver, run `md5sum
/vendor/lib64/hw/vulkan.samsung.so` on-device. If that hash is not one this study itself already
has a backup of (i.e. it's either the device's pre-bisect state or a fresh drift), `adb pull` it
and copy to a dated NFS backup path (`vulkan.samsung.so.<short-context>-backup-<date>`) *before*
overwriting it. After the step's verdict is captured (bench completed, or crash/tombstone
captured), flash that same backed-up driver back.

**Rationale**: This session's own experience (multiple undocumented drifts on this exact board
within hours, one of them turning out to be a teammate's active work) makes this a hard
requirement, not a nice-to-have — losing someone else's in-progress build to an unattended bisect
step would be a real cost to a real person, not a hypothetical.

**Alternatives considered**: Skipping backup/restore and just re-flashing the team's documented
default (`f14c51b6f8`) at the end — rejected, because that would silently discard whatever
non-default build was actually on the device (e.g. a teammate's `T159889`-tagged work), which per
this session's incident is exactly the kind of loss to avoid.

## 7. Driver identity capture

**Decision**: Capture `adb shell md5sum /vendor/lib64/hw/vulkan.samsung.so` after every flash (to
confirm the flash took) and again immediately before running the test (to catch any drift that
happened in between) — same reliable identity check `specs/032` converged on (`logcat | grep
SUMD` is documented as unreliable for this, per that spec's research.md §6 finding — still true
here, not re-derived).

## 8. Crash-detection mechanics (new relative to `specs/032`)

**Decision**: The bisect-test script runs `COOPMAT_BENCH_CORRECTNESS_ONLY=1
./test_coopmat_linear_bench_origcm` and classifies the result by:
- **`good`**: process exits 0 and stdout contains `Completed 16 test cases`.
- **`bad`**: process exits with a signal-indicating code (segfault → 139) or stdout stops mid-run
  without reaching the completion line. Immediately after, check `adb shell "ls -lat
  /data/tombstones/ | head -3"` and pull the newest tombstone if its timestamp is after the test
  started — that becomes this step's crash evidence (FR-008). If no tombstone exists (process
  killed before one was written), fall back to the exit code and last-printed console line as the
  crash signature (spec Edge Cases).
- **`skip`**: build failure, flash failure, or a hang exceeding a bounded timeout with the device
  otherwise still responsive after recovery.

**Rationale**: Directly implements the spec's FR-007/FR-008 predicate and evidence-capture
requirement using tools already proven this session (the two tombstones already pulled and read
for the teammate's report and for this session's own fresh `7bb715f7cc` build).

## 9. Source-reading the culprit (new relative to `specs/032`)

**Decision**: Once `git bisect` converges, read the first-bad commit's diff (`git show
<first-bad-sha>`) and summarize what changed and why it plausibly causes a coopmat-dispatch
segfault.

**Rationale**: Per this workspace's memory (`flash-sumd-driver` — "07-17: Rule 0 lifted
workspace-wide, reading source now allowed too"), SUMD source-reading is no longer off-limits.
`specs/032` had to stop at the SHA/metadata for exactly this reason; this study can go one step
further and turn "which commit" into "why," making the eventual driver-team report more
actionable.
