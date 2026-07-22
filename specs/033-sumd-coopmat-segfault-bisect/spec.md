# Feature Specification: SUMD Driver Bisect for the Coopmat-Dispatch Segfault Regression

**Feature Branch**: `033-sumd-coopmat-segfault-bisect`

**Created**: 2026-07-21

**Status**: Draft

**Input**: User description: "We need to do a bisect on the SUMD to find the regression, the
process is similar to the last bisect spec"

## Context

A teammate reported `test_coopmat_linear_bench_origcm` (run with
`COOPMAT_BENCH_CORRECTNESS_ONLY=1`) segfaulting on M5 EVT1 partway through the correctness
matrix — the first (tiled) test case passes, then the process crashes with `SIGSEGV` on the very
next (coopmat) dispatch. The crash is inside `vulkan.samsung.so` itself (confirmed via
`/data/tombstones/`), not in ExecuTorch/test-harness code.

Two endpoints were already measured this session, on M5 EVT1 (serial `0000088f8e579c33`, host
`sj1-dmckee-d01`), same binary, same correctness bench:

- **`f14c51b6f8`** (the team's documented known-good driver, `main` @ 2026-06-15) — all 16 test
  cases `PASSED`, no crash.
- **`origin/main` tip at investigation time, `7bb715f7cc`** (`main` @ 2026-07-21, built fresh from
  source for this check) — segfaults on the second test case, tombstone confirms `SIGSEGV`
  (`SEGV_MAPERR`, read) inside `vulkan.samsung.so`, same call chain both times
  (`ComputeGraph::execute()` → `DispatchNode::encode()` → `register_shader_dispatch()` →
  `CommandBuffer::dispatch()` → crash inside the driver).

Since the model/harness/binary are held constant and only the SUMD build differs, this points at
one of the 303 commits between `f14c51b6f8` and `7bb715f7cc` on SUMD `main` that introduced a
coopmat-dispatch crash. This spec bisects that range to find the first-bad commit, following the
same overall process as `specs/032-sumd-driver-bisect` (that investigation's bisect-test.sh
harness, report format, and skip/re-narrow handling are the template), adapted for a different
device, predicate, and range.

### What's different from `specs/032-sumd-driver-bisect`

- **Predicate is a crash, not a throughput ordering.** `specs/032` compared two tok/s numbers;
  this bisect just needs the correctness bench to finish (`good`) or segfault/crash before
  finishing (`bad`) — no measurement noise, no near-parity ambiguity.
- **Device is M5 EVT1** (`0000088f8e579c33` / `sj1-dmckee-d01`), not M41 — because the crash was
  found and both endpoints were confirmed there. M5 EVT1 is a heavily shared board this session
  (multiple driver drifts observed within a few hours, from at least one teammate actively
  flashing their own builds) — every bisect step MUST back up whatever driver is on the device
  immediately before flashing over it, and restore it after the step's verdict is recorded, so
  the bisect doesn't clobber someone else's in-progress work.
- **Rule 0 (SUMD source off-limits) no longer applies.** Per this workspace's memory
  (`flash-sumd-driver` / "07-17: Rule 0 lifted workspace-wide, reading source now allowed too"),
  reading SUMD source is permitted. Once the first-bad commit is found, this study MAY read its
  diff to characterize the regression (unlike `specs/032`, which stopped at the SHA/metadata
  because source-reading was off-limits at the time).

## Clarifications

None — the predicate (crash vs. no crash), device, and range are all already pinned down by this
session's own measurements; no ambiguous decision point met the bar for a clarification question.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Re-confirm both range endpoints under the bisect harness (Priority: P1)

Before spending build/flash/test cycles walking the interior of the range, an engineer needs to
re-confirm, under the exact bisect harness (fresh build from each endpoint SHA, staged/flashed the
same way every subsequent step will be), that `f14c51b6f8` still passes cleanly and `7bb715f7cc`
still crashes — the ad-hoc checks earlier this session used slightly different build/flash
sequences and should not be assumed to transfer without re-verification under the harness itself.

**Why this priority**: If the harness itself can't reproduce what was already found, nothing
downstream is trustworthy — this is the cheapest possible check that gates everything else.

**Independent Test**: Can be fully tested by running the bisect harness against only the two
endpoint SHAs and confirming their verdicts still disagree.

**Acceptance Scenarios**:

1. **Given** `f14c51b6f8` and `7bb715f7cc` on SUMD `main`, **When** each is built via the bisect
   harness, flashed to M5 EVT1, and run through
   `COOPMAT_BENCH_CORRECTNESS_ONLY=1 test_coopmat_linear_bench_origcm`, **Then** `f14c51b6f8`
   completes all 16 test cases with no crash (`good`) and `7bb715f7cc` segfaults before completing
   (`bad`).
2. **Given** both endpoints instead agree under the harness, **When** this is discovered, **Then**
   the study stops before bisecting and reports the disagreement (with both endpoints' full
   console output and any tombstone) rather than proceeding with a range that has no isolatable
   culprit.

---

### User Story 2 - Bisect to the first-bad commit (Priority: P1)

Once the range is confirmed to bracket the crash, an engineer needs `git bisect` walked to
completion so a single first-bad SUMD commit is identified — the commit that introduced the
coopmat-dispatch segfault.

**Why this priority**: This is the actual goal of the task — everything else is in service of this
outcome.

**Independent Test**: Can be fully tested by running the full bisect sequence and confirming `git
bisect` converges on exactly one commit, with every intermediate step's build/flash/run result
recorded.

**Acceptance Scenarios**:

1. **Given** a commit `git bisect` selects for testing, **When** it is built, flashed to M5 EVT1
   (with the previously-active driver backed up first and restored after), and run through the
   correctness bench, **Then** it is marked `good` (all 16 cases pass) or `bad` (crash/segfault
   before completion) per the same predicate as the endpoints, and `git bisect` is advanced with
   that verdict.
2. **Given** all bisect steps have been evaluated, **When** `git bisect` terminates, **Then**
   exactly one commit is identified as the first-bad commit, with its SHA, author, date, and
   subject line recorded.

---

### User Story 3 - Document driver provenance, crash evidence, and the full bisect trace (Priority: P2)

An engineer revisiting this investigation later (or handing it to the driver team) needs to know
exactly which driver build was on the device for every step, the crash evidence (tombstone
backtrace) at each `bad` verdict, and the full sequence of verdicts that led to the culprit — not
just the final answer.

**Why this priority**: Without this, the bisect result is unverifiable, and a crash report without
a tombstone/backtrace is much less actionable for the driver team than one with it.

**Independent Test**: Can be fully tested by reading the resulting document and confirming every
tested commit has a recorded driver hash, build outcome, verdict, and — for every `bad` verdict —
a captured tombstone or crash signature.

**Acceptance Scenarios**:

1. **Given** the bisect has completed, **When** the study document is produced, **Then** it lists
   every tested commit in the order tested, each with SHA, on-device driver identity (md5sum of
   `vendor/lib64/hw/vulkan.samsung.so` after flashing), build outcome, verdict
   (`good`/`bad`/`skip`), and — for any `bad` verdict — the captured tombstone/backtrace.
2. **Given** the first-bad commit is identified, **When** the document is finalized, **Then** it
   states the last-good commit and first-bad commit side by side (SHA, date, subject, driver
   hash), and — since Rule 0 no longer blocks source-reading — includes a read of the first-bad
   commit's diff describing what it changed and why that plausibly causes the crash.

---

### Edge Cases

- What happens when a candidate commit fails to build? → Retry once with the documented
  `LD_LIBRARY_PATH` workaround (strip `vulkan-sdk`, per the known GpuRt "Too many users" gotcha);
  if it still fails, mark that commit `skip` in `git bisect skip` and record the build failure.
- What happens when a commit builds but the resulting driver hangs (rather than cleanly
  segfaulting) on-device? → Treat a hang the same as a crash for bisect purposes only if a
  bounded timeout is exceeded and the device remains otherwise responsive after
  reconnect/recovery; record it as `bad` with "hang" as the signature (distinct from a tombstoned
  segfault) rather than guessing a `good` verdict. If the device itself becomes unresponsive and
  needs a manual recovery step, mark that commit `skip` instead and record what recovery was
  needed.
- What happens if enough commits are `skip`ped that `git bisect` can't narrow to a single commit?
  → Manually build/test the nearest testable commit(s) adjacent to the skipped span (walking
  outward as needed) and feed each verdict back into `git bisect` until it converges on exactly
  one commit — do not accept an ambiguous multi-commit range as final.
- What happens if the M5 EVT1 board is found on an unrecognized driver hash mid-bisect (i.e.
  someone else flashed something between bisect steps)? → Back it up first (never overwrite an
  unknown build without saving it), note the hash in the bisect log, then proceed with the
  step's flash — the bisect does not stop for drift, but every drift encountered must be logged
  and the pre-existing build restored once the step's verdict is captured.
- What happens if a bisect step's crash doesn't produce a tombstone (e.g. the process is killed
  before a tombstone is written)? → Fall back to the process exit code (segfault → 128+11=139)
  and console output (last line printed before the crash) as the crash signature; still record
  the verdict as `bad`, just note the missing tombstone.
- What happens if flashing partially fails (NFS staging share full/truncated `.so`)? → `cmp`-verify
  the staged artifact against the local build before pushing; a failed verify blocks that step
  rather than flashing a possibly-corrupt driver.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The study MUST first re-confirm, under the bisect harness itself, that `f14c51b6f8`
  is `good` (all 16 correctness-bench cases pass) and `7bb715f7cc` is `bad` (crashes before
  completing) before bisecting the interior of the range; if they agree under the harness, the
  study MUST stop and report the disagreement instead of bisecting.
- **FR-002**: Every measurement in this study MUST run on the M5 EVT1 device at serial
  `0000088f8e579c33` via `ssh yanwen.xu@sj1-dmckee-d01` — no other device may be substituted at
  any step.
- **FR-003**: Before flashing any candidate driver, the study MUST back up whatever driver is
  currently on the device (md5sum + copy to NFS) if it is not already a driver this study itself
  flashed, and MUST restore that backed-up driver after the step's verdict is captured — the
  device is shared and other people's in-progress work must not be lost or left overwritten.
- **FR-004**: For every commit under test, the study MUST build the SUMD driver from that commit
  (Android release build, per this workspace's documented `uv run scripts/run.py --os android
  --build --build-type release` procedure) and flash it to the M5 EVT1 device before testing.
- **FR-005**: For every commit under test, the study MUST record the on-device driver identity
  (`md5sum /vendor/lib64/hw/vulkan.samsung.so`) alongside that commit's SHA, both immediately
  after flashing (to confirm the flash took) and before running the test (to catch any
  intervening drift).
- **FR-006**: Every measurement MUST run `COOPMAT_BENCH_CORRECTNESS_ONLY=1
  ./test_coopmat_linear_bench_origcm` (the exact binary and invocation that reproduced the crash)
  against the freshly-flashed driver.
- **FR-007**: The good/bad predicate for a commit MUST be: `good` if the correctness bench
  completes all 16 test cases (prints "Completed 16 test cases" with no crash); `bad` if the
  process crashes (segfault or other fatal signal) before completing.
- **FR-008**: For every `bad` verdict, the study MUST capture the crash evidence — the on-device
  tombstone (`/data/tombstones/`) if one was written, plus the console output up to the crash —
  and record it alongside that commit's entry.
- **FR-009**: The study MUST drive `git bisect` (or an equivalent manual bisection over the
  confirmed-disagreeing range) to convergence on exactly one first-bad commit. If `skip`ped
  commits leave `git bisect` unable to narrow further, the study MUST manually build/test
  commit(s) adjacent to the skipped span and feed those verdicts back in, rather than reporting an
  ambiguous multi-commit range as final.
- **FR-010**: The study MUST record, for every tested commit in bisect order: SHA, build outcome,
  on-device driver hash, verdict (`good`/`bad`/`skip` with reason), and crash evidence where
  applicable — not just the final culprit.
- **FR-011**: Once the first-bad commit is identified, the study MUST read its diff (source-reading
  is permitted per Rule 0 being lifted) and record what it changed, so the report explains *why*
  the change plausibly causes the observed crash, not just *which* commit it is.
- **FR-012**: The study MUST report the identified first-bad commit's SHA, author, date, and
  subject line, alongside the last-good commit's SHA/date, sufficient for both to be opened
  directly in Gerrit.

### Key Entities

- **Bisect Step**: One SUMD commit under test — carries commit SHA, build outcome, on-device
  driver hash (post-flash and pre-test), verdict (`good`/`bad`/`skip`), and — for `bad` — the
  captured tombstone/crash signature.
- **Culprit Commit**: The `git bisect`-identified first-bad commit — carries SHA, author, date,
  subject, diff summary (what changed and why it plausibly causes the crash), and the driver
  hashes observed immediately before (last-good) and at (first-bad) this commit.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Both range endpoints (`f14c51b6f8`, `7bb715f7cc`) are re-confirmed under the bisect
  harness to produce different verdicts before any interior bisect step is measured.
- **SC-002**: `git bisect` converges to exactly one first-bad commit within the confirmed range.
- **SC-003**: A single document records every tested commit's SHA, driver hash, build outcome,
  verdict, and crash evidence (where applicable), in the order tested — reconstructable without
  replaying this session.
- **SC-004**: 100% of measurements in the study were taken on M5 EVT1 serial `0000088f8e579c33` —
  no step used a substitute device.
- **SC-005**: 100% of steps that overwrote a non-study driver on the shared board restored it
  afterward — zero net driver-state loss for other users of the board.
- **SC-006**: The final report states the culprit commit (SHA, author, date, subject, diff
  summary) and the driver hashes bracketing it (last-good vs. first-bad), sufficient for the
  commit to be opened directly in Gerrit and handed to the driver team as a repro + root-cause
  report.

## Assumptions

- "SUMD main branch" means the commit history of the `main` branch in `/local/yanwen.xu/sumd/`
  (worktree `main`, tracking `origin/main`), bounded by `f14c51b6f8` (last-known-good, already
  confirmed) and `7bb715f7cc` (first-known-bad at investigation start, already confirmed) — 303
  commits apart at the time this spec was written. If `origin/main` has moved further by the time
  bisecting starts, `7bb715f7cc` (not a newer tip) remains the fixed "bad" endpoint, so the range
  doesn't shift mid-study.
- Building and flashing SUMD for this study is permitted; per Rule 0 being lifted workspace-wide
  (2026-07-17), reading SUMD source is now also permitted and is used once the culprit is found
  (a deliberate difference from `specs/032`, where source-reading was off-limits).
- A single run per commit is sufficient for this predicate (no averaging/reps needed, unlike a
  throughput bisect) — a crash is deterministic enough in this bench (reproduced identically twice
  already, on both the teammate's build and this session's fresh `7bb715f7cc` build) that a single
  pass/fail run per commit is trusted as the verdict.
- Clocks are pinned to this workspace's M5 EVT1 default (509/2730/663 MHz) for consistency with
  other work on this device, though the predicate (crash vs. no crash) is not expected to be
  clock-sensitive — this is a safety/consistency default, not a load-bearing requirement of the
  bisect itself.
- The M5 EVT1 board's active-use pattern observed this session (multiple undocumented driver
  hashes appearing between checks, consistent with at least one teammate actively flashing their
  own builds) is expected to continue during this bisect; the backup/restore protocol (FR-003) is
  the mitigation, not a one-time precaution.
- `test_coopmat_linear_bench_origcm` (the existing NFS-staged binary that reproduced the crash) is
  reused as-is for every bisect step — it is not rebuilt per-commit, since the regression is in the
  driver, not the test harness, and rebuilding the harness itself would add a second variable to
  control for.
