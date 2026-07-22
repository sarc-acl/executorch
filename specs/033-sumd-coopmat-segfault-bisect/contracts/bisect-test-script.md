# Contract: Bisect Test Procedure

**Feature**: `033-sumd-coopmat-segfault-bisect` | **Date**: 2026-07-21

This is the one "interface" this feature defines: the procedure/script invoked once per commit
under test (`scripts/bisect-test.sh`, authored during implementation), driven manually one step
at a time (per `research.md` §3) rather than via unattended `git bisect run` — the exit-code
contract below is still the natural `git bisect`-compatible convention regardless of who invokes
it.

## Invocation

```bash
scripts/bisect-test.sh <sumd-worktree-dir> [bisect_role]
```

Run with cwd anywhere; `<sumd-worktree-dir>` is the already-checked-out SHA-named SUMD worktree
(e.g. `/local/yanwen.xu/sumd/<short-sha>/`) for the commit currently under test. This script does
**not** create the worktree or check out the commit — that's the caller's responsibility (`git
worktree add`), keeping this script's job narrowly "build this checkout, back up whatever's on
the device, flash, test, restore, verdict it." `bisect_role` is one of `endpoint-good` /
`endpoint-bad` / `bisect-step` / `skip-adjacent-probe` (data-model.md) and is recorded verbatim
into the appended trace row; defaults to `bisect-step` if omitted.

## Preconditions

- The target worktree exists and is checked out at the commit under test (detached HEAD), with
  submodules initialized.
- M5 EVT1 (serial `0000088f8e579c33`) is reachable via `ssh yanwen.xu@sj1-dmckee-d01` and
  responsive.
- `test_coopmat_linear_bench_origcm` is already staged at `$D` on-device (this script does not
  push it — it's reused unmodified across every bisect step, per Assumptions).
- Clocks are pinned to the workspace default for this device (509/2730/663 MHz) — this script
  re-verifies the pin via sysfs readback rather than assuming a prior step's pin still holds.

## Side effects

1. **Backup**: `md5sum /vendor/lib64/hw/vulkan.samsung.so` on-device. If that hash is not already
   a driver this study flashed, `adb pull` it and copy to a dated NFS backup path before
   proceeding (FR-003) — recorded as this step's Driver Backup Record.
2. Builds the SUMD driver from the worktree's current HEAD (`uv run scripts/run.py --os android
   --build --build-type release`, with `vulkan-sdk` stripped from `LD_LIBRARY_PATH`).
3. Stages the resulting `.so` to NFS (`cmp`-verified) and flashes it to M5 EVT1.
4. Captures `driver_hash_post_flash` (`adb shell md5sum /vendor/lib64/hw/vulkan.samsung.so`).
5. Re-pins via `pin_freqs.sh` (`S=0000088f8e579c33 GPUFREQ=509000 MIFFREQ=2730000
   INTFREQ=663000`, per research.md §4) — cheap and idempotent, run every step rather than
   trusting a prior step's pin.
6. Captures `driver_hash_pre_test` (same md5sum command) immediately before running the bench —
   if this disagrees with `driver_hash_post_flash`, the step is invalid (drift happened mid-step)
   and must be re-run, not verdicted.
7. Runs `COOPMAT_BENCH_CORRECTNESS_ONLY=1 ./test_coopmat_linear_bench_origcm` with a bounded
   timeout.
8. On crash (non-zero/signal exit code, or stdout stops before `Completed 16 test cases`): checks
   for a fresh tombstone (`adb shell "ls -lat /data/tombstones/ | head -3"`, matched by timestamp
   after the test started), pulls it to `results/tombstones/<short-sha>.txt` if found, and records
   `crash_evidence` (tombstone fields if present, else just `exit_code` + `last_console_line`).
9. **Restore**: re-flashes whatever driver was backed up in step 1 (skipped only if the device was
   already on a driver this study itself flashed, i.e. no backup was needed for this step) and
   sets `restored_after_step = true`.
10. Appends one row to the bisect-trace log (per `data-model.md`'s Bisect Step schema) —
    irrespective of outcome, including failures.

**Source-reading**: this script itself never opens SUMD source — every step's verdict comes
purely from build success/failure and on-device behavior, same discipline as `specs/032`. The
difference from `specs/032` is only that, once `git bisect` converges (outside this script, as a
separate one-time step per `research.md` §9), the *first-bad commit's* diff is read to populate
`Culprit Commit.diff_summary` — Rule 0 no longer forbids this, but nothing in this per-step
script's own loop depends on it.

## Outputs (exit code contract — matches `git bisect run`'s convention)

| Exit code | Meaning | Condition |
|---|---|---|
| `0` | good | Build succeeded, driver hash was stable pre/post-test, and the bench printed `Completed 16 test cases` with no crash |
| `1` | bad | Build succeeded, driver hash was stable pre/post-test, and the bench crashed or stopped before completing |
| `125` | skip | Build failed (even after the documented `LD_LIBRARY_PATH` retry), the flash failed `cmp`-verify, the driver hash drifted mid-step, or the bench hung past the bounded timeout while the device remained otherwise responsive |

Stdout/stderr: human-readable log of every side-effect step above, plus the final one-line
verdict summary (`commit=<sha> driver_hash="<hash>" verdict=<good|bad|skip> [reason=<...>]
[tombstone=<path>]`) — this line is what gets appended to the bisect-trace log
(`data-model.md`).

## Postconditions

- Exactly one row is appended to the bisect-trace log for this commit, regardless of outcome.
- The device is left on whatever driver was on it before this step ran (restored in side effect
  9) — a bisect step never leaves the shared board on a study driver after it completes.
- The SHA-named worktree is left in place afterward (never removed without explicit instruction).
- For every `bad` verdict, a tombstone file exists under `results/tombstones/` unless the crash
  genuinely produced none (fallback case, noted in `crash_evidence`).
