# Feature Specification: SUMD Driver Bisect for the 8da4w-Slower-Than-4w Regression

**Feature Branch**: `032-sumd-driver-bisect`

**Created**: 2026-07-16

**Status**: Draft

**Input**: User description: "Perform a bisect on SUMD main branch from Nov 2024 to March 2026
(check these two end first, make sure one is good and one is bad). 'good' means '8da4w is faster
than 4w', 'bad' is the other way around. Do on M41 device on this ssh xgpusw-debug07 export
ANDROID_SERIAL=00000a34cdd4abd3 . Must be this device. Flush the driver, take note the versions,
and pin the freq to the max ... 23400000.sgpu (GPU) 980 MHz, 17000010.devfreq_mif (DRAM) 5333 MHz,
17000020.devfreq_int 800 MHz ... And run the Release 1.3 vanilla, use 1B 4w and 8da4w for test, 1
rep is enough, but still need 2048 prefill. The goal of this task is to find the blame, which
commit that caused regression."

## Context

`specs/024-8da4w-slower-than-4w` already explains a *shader-level* mechanism (extra dequant
bookkeeping) for why 8da4w's tiled kernel can be slower than 4w's tiled kernel on M5 EVT1 — that
finding is about the ExecuTorch/shader side and is treated as expected there, not a regression.

This spec is about a *different, driver-level* signal collected 2026-07-16: on M41, running the
identical release/1.3 vanilla workload, the 4w-vs-8da4w ordering flips depending on which SUMD
build is on the device:

- M41's own native/factory driver → 8da4w prefill is ~30% **faster** than 4w (777.2 vs 599.1
  tok/s floating; 411.6 vs 316.4 tok/s pinned at 509/2730/663) — the expected ordering.
- M5 EVT1's known-good driver `f14c51b6f8` cross-flashed onto M41, pinned to M41's own max
  (980/5333/800) → 8da4w prefill is ~30% **slower** than 4w (425.3 vs 605.2 tok/s) — inverted.

Since the model, workload, and (M41) hardware are held constant and only the SUMD build differs,
this points at a SUMD commit between whatever produced the native driver and `f14c51b6f8` that
changed the relative cost of the 8da4w vs 4w dispatch path. The goal of this feature is to find
that commit via `git bisect` over SUMD's `main` branch.

## Clarifications

### Session 2026-07-16

- Q: How should a near-parity (close-margin) 4w-vs-8da4w single-rep result be classified? → A:
  Strict comparison, no tie-break zone — whichever quant mode has the higher prefill tok/s is the
  verdict for that commit, however small the margin.
- Q: If enough commits are marked `skip` (build/run failures) that `git bisect` can't narrow to a
  single commit, what should happen? → A: Manually build/test the nearest testable commit(s)
  adjacent to the skipped range to force further narrowing, rather than accepting an ambiguous
  multi-commit range.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Confirm the bisect range actually brackets the regression (Priority: P1)

Before spending build/flash/measure cycles on a full bisect, an engineer needs to confirm the
two range endpoints (the oldest commit in scope, ~Nov 2024, and the newest, ~Mar 2026) actually
disagree — one measures "good" (8da4w prefill faster than 4w) and the other "bad" (8da4w prefill
slower than 4w). If both endpoints agree, `git bisect` has nothing to find and the range or the
predicate needs to be revisited before proceeding.

**Why this priority**: Every subsequent bisect step is wasted effort if the range doesn't actually
bracket a flip. This is the cheapest possible check (2 measurements) that gates everything else.

**Independent Test**: Can be fully tested by building, flashing, and measuring only the two
endpoint commits and confirming their good/bad verdicts differ.

**Acceptance Scenarios**:

1. **Given** the Nov 2024 and Mar 2026 boundary commits on SUMD `main`, **When** each is built,
   flashed to M41 (serial `00000a34cdd4abd3`, host `xgpusw-debug07`), and measured with the
   release/1.3 vanilla 1B runner at 2048-token prefill (1 rep, 4w and 8da4w), **Then** exactly one
   endpoint's 8da4w prefill tok/s exceeds its 4w prefill tok/s ("good") and the other's does not
   ("bad").
2. **Given** both endpoints instead return the same verdict, **When** this is discovered, **Then**
   the study stops before bisecting and reports the disagreement (with both endpoints' numbers)
   rather than proceeding with a range that has no isolatable culprit.

---

### User Story 2 - Bisect to the first bad commit (Priority: P1)

Once the range is confirmed to bracket a flip, an engineer needs `git bisect` walked to
completion so a single first-bad SUMD commit is identified — the commit that changed the 4w/8da4w
prefill ordering.

**Why this priority**: This is the actual goal of the task ("find the blame") — everything else is
in service of this outcome.

**Independent Test**: Can be fully tested by running the full bisect sequence and confirming `git
bisect` reports converging on exactly one commit, with every intermediate step's measurement
recorded.

**Acceptance Scenarios**:

1. **Given** a commit `git bisect` selects for testing, **When** it is built, flashed to M41, and
   measured (1B, 4w and 8da4w, 2048-token prefill, 1 rep, clocks pinned to M41's max 980/5333/800),
   **Then** it is marked `good` or `bad` per the same predicate as the endpoints, and `git bisect`
   is advanced with that verdict.
2. **Given** all bisect steps have been evaluated, **When** `git bisect` terminates, **Then**
   exactly one commit is identified as the first bad commit, with its SHA, author, date, and
   subject line recorded.

---

### User Story 3 - Document driver provenance and the full bisect trace (Priority: P2)

An engineer revisiting this investigation later (or reporting it) needs to know exactly which
driver build (commit + version string) was on the device for every measurement, and the full
sequence of good/bad verdicts that led to the culprit — not just the final answer.

**Why this priority**: Without this, the bisect result is unverifiable and undebuggable if the
verdict is ever questioned; it's also what turns this into a citable artifact instead of chat
history.

**Independent Test**: Can be fully tested by reading the resulting document and confirming every
tested commit has a recorded driver version string, build outcome, 4w/8da4w prefill numbers, and
verdict, in bisect order.

**Acceptance Scenarios**:

1. **Given** the bisect has completed, **When** the study document is produced, **Then** it lists
   every tested commit in the order tested, each with SHA, driver identity (md5sum of
   `vendor/lib64/hw/vulkan.samsung.so` after flashing), 4w prefill tok/s, 8da4w prefill tok/s, and verdict
   (`good`/`bad`/`skip`).
2. **Given** the first-bad commit is identified, **When** the document is finalized, **Then** it
   states the last-good commit's driver version and the first-bad commit's driver version
   side-by-side, so the version-string delta is visible even without reading source.

---

### Edge Cases

- What happens when a candidate commit fails to build (e.g. the documented `LD_LIBRARY_PATH`
  GpuRt shader "Too many users" issue)? → Retry once with the documented workaround (strip
  `vulkan-sdk` from `LD_LIBRARY_PATH`); if it still fails to build, mark that commit `skip` in
  `git bisect skip` and record the build failure, rather than guessing a verdict.
- What happens when a commit builds but the resulting driver crashes or hangs on-device? →
  Mark `skip` and record the crash/hang signature; do not count it as either verdict.
- What happens if enough commits are `skip`ped that `git bisect` can no longer narrow to a single
  commit (a surviving range of untestable commits)? → Do not accept the ambiguous range as the
  final answer: manually build and test the nearest testable commit(s) immediately adjacent to the
  skipped span (walking outward one commit at a time if needed) to force further narrowing, and
  feed each result back into `git bisect` until it converges on exactly one commit.
- What happens when a single-rep measurement lands close to parity (4w and 8da4w prefill within
  noise of each other)? → No special handling: the verdict is a strict comparison (whichever
  quant mode's prefill tok/s is higher wins), even for a narrow margin — there is no near-parity
  re-run or tie-break zone (see Clarifications).
- What happens if the M41 device (shared board) becomes unresponsive or is in use by someone else
  mid-bisect? → Reconnect/verify liveness before continuing; do not switch to a different device
  to keep the bisect moving — this device is a hard requirement, not a convenience default.
- What happens if flashing partially fails (e.g. NFS staging share full/truncated `.so`, per the
  known `/sarc-c/gpusw` capacity gotcha)? → `cmp`-verify the staged artifact against the local
  build before pushing to device; a failed verify blocks that step rather than flashing a
  possibly-corrupt driver.
- What happens if the confirmed pin values (980/5333/800) don't stick or drift between
  measurements? → Verify via sysfs readback before each measurement; a pin that didn't take is
  treated as an invalid measurement for that step, not silently reported as pinned.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The study MUST identify concrete SHAs on SUMD `main` closest to Nov 2024 and Mar
  2026 to serve as the initial bisect range endpoints, before any build/flash/measure work begins.
- **FR-002**: The study MUST measure both range endpoints first and confirm they produce different
  verdicts (one `good`, one `bad`) before proceeding to bisect the interior of the range; if they
  agree, the study MUST stop and report the disagreement instead of bisecting.
- **FR-003**: Every measurement in this study MUST run on the M41 device at serial
  `00000a34cdd4abd3` via `ssh xgpusw-debug07` — no other device may be substituted at any step.
- **FR-004**: For every commit under test, the study MUST build the SUMD driver from that commit
  and flash it to the M41 device before measuring, without reading or reviewing SUMD driver source
  code (per `/local/yanwen.xu/sumd/CLAUDE.md` Rule 0 — building/deploying is in scope, reading
  source is not).
- **FR-005**: For every commit under test, the study MUST record the on-device driver identity
  (`md5sum /vendor/lib64/hw/vulkan.samsung.so` — never `logcat | grep SUMD`, which is documented
  as unreliable for identifying which build is active, since it dumps build-ancestry commit
  hashes rather than the active build's own) alongside that commit's SHA.
- **FR-006**: Before every measurement, the study MUST pin and sysfs-verify M41's own maximum
  clocks: `23400000.sgpu` (GPU) = 980 MHz, `17000010.devfreq_mif` (DRAM) = 5333 MHz,
  `17000020.devfreq_int` = 800 MHz — not the workspace-wide default pin (509/2730/663), which is a
  different, lower-clocked target.
- **FR-007**: Every measurement MUST use the release/1.3 vanilla runner (plain `release/1.3`
  ExecuTorch, no node-threshold patch), Llama 3.2 1B, at 2048-token prefill, for both the 4w and
  8da4w quantization modes, 1 rep each.
- **FR-008**: The good/bad predicate for a commit MUST be: `good` if that commit's 8da4w prefill
  tok/s > its 4w prefill tok/s; `bad` otherwise.
- **FR-009**: The study MUST drive `git bisect` (or an equivalent manual bisection over the
  confirmed-disagreeing range) to convergence on exactly one first-bad commit. If `skip`ped
  commits leave `git bisect` unable to narrow further, the study MUST manually build/test
  commit(s) adjacent to the skipped span (walking outward as needed) and feed those verdicts back
  in, rather than reporting an ambiguous multi-commit range as final.
- **FR-010**: The study MUST record, for every tested commit in bisect order: SHA, build outcome,
  driver version string, 4w prefill tok/s, 8da4w prefill tok/s, and verdict (`good`/`bad`/`skip`
  with reason) — not just the final culprit.
- **FR-011**: The study MUST report the identified first-bad commit's SHA, author, date, and
  subject line, alongside the driver version strings of the last-good and first-bad commits.

### Key Entities

- **Bisect Step**: One SUMD commit under test — carries commit SHA, build outcome, on-device
  driver version string, 4w prefill tok/s, 8da4w prefill tok/s, and verdict (`good`/`bad`/`skip`).
- **Culprit Commit**: The `git bisect`-identified first-bad commit — carries SHA, author, date,
  subject, and the driver version strings observed immediately before (last-good) and at
  (first-bad) this commit.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Both range endpoints (~Nov 2024, ~Mar 2026) are measured and confirmed to produce
  different verdicts before any interior bisect step is measured.
- **SC-002**: `git bisect` converges to exactly one first-bad commit within the confirmed range.
- **SC-003**: A single document records every tested commit's SHA, driver version, build outcome,
  4w/8da4w prefill numbers, and verdict, in the order tested — reconstructable without replaying
  this session.
- **SC-004**: 100% of measurements in the study were taken on M41 serial `00000a34cdd4abd3` — no
  step used a substitute device.
- **SC-005**: The final report states the culprit commit (SHA, author, date, subject) and the
  driver version strings bracketing it (last-good vs first-bad), sufficient for the commit to be
  opened directly in Gerrit for source-level follow-up (outside this study's scope, since driver
  source reading is off-limits here).

## Assumptions

- "SUMD main branch from Nov 2024 to March 2026" means the commit history of the `main` branch in
  `/local/yanwen.xu/sumd/` (worktree `main`, tracking `origin/main`), bounded by the commits
  closest to those two calendar dates — exact boundary SHAs are resolved and confirmed (per FR-001)
  before the bisect proper starts, not assumed in advance.
- Building and flashing SUMD for this study is permitted; reading, reviewing, or modifying SUMD
  driver source is not (per `/local/yanwen.xu/sumd/CLAUDE.md` Rule 0) — bisect verdicts are
  produced purely from build success/failure and on-device measurement, never from source
  inspection.
- Decode throughput is not part of the good/bad predicate (only prefill tok/s ordering matters);
  it may be recorded if trivially available from the same run but is not required.
- A single rep per commit/quant-mode is accepted for bisect speed (an explicit, deliberate
  reduction from this workspace's usual 3-rep default) — accepting some noise risk, with verdicts
  decided by strict comparison and no near-parity re-run (see Clarifications).
- Clocks are pinned to M41's own maximum (980/5333/800 MHz) for this study specifically, not the
  workspace-wide default pin (509/2730/663 MHz) used elsewhere — a deliberate deviation to remove
  DVFS/thermal variance as a confound while isolating the driver commit.
- This bisect uses M41 exclusively as the vehicle for isolating the SUMD commit, per explicit
  instruction ("Must be this device") — it does not aim to produce M5 EVT1-comparable headline
  numbers, and is not superseded by that device's usual "secondary reference" framing elsewhere in
  this workspace, since here the driver-under-test is the subject, not the device's own baseline
  perf.
- The 4w/8da4w PTEs (Llama 3.2 1B, ctx supporting 2048-token prefill) already exist or can be
  staged from this workspace's existing NFS/`.pte_out` locations without a fresh export per
  bisect step.
