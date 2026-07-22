---

description: "Task list for SUMD Driver Bisect for the Coopmat-Dispatch Segfault Regression"
---

# Tasks: SUMD Driver Bisect for the Coopmat-Dispatch Segfault Regression

**Input**: Design documents from `specs/033-sumd-coopmat-segfault-bisect/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/bisect-test-script.md, quickstart.md

**Tests**: Not requested — this is a hardware bisect investigation; verification is the
endpoint-disagreement gate (US1), the driver-hash pre/post-flash stability check, and the
quickstart SC checklist below, not a code test suite.

**Organization**: Tasks are grouped by user story (US1=re-confirm range endpoints, US2=bisect to
the first-bad commit, US3=document driver provenance + crash evidence + full trace, per spec.md's
priorities P1/P1/P2 — US1 and US2 are both P1 since US2 cannot start without US1's gate, but
they're sequenced as separate phases here since US1 is independently completable/testable on its
own, same reasoning `specs/032` used).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to
- Device: M5 EVT1, serial `0000088f8e579c33`, via `ssh yanwen.xu@sj1-dmckee-d01` (hard-locked,
  FR-002 — every task below runs there, not restated per-task)
- SUMD worktrees live under `/local/yanwen.xu/sumd/<short-sha>/`; this feature's own docs/scripts
  live under `specs/033-sumd-coopmat-segfault-bisect/`
- Driver backup/restore (FR-003) and crash-evidence capture (FR-008) are wired into
  `scripts/bisect-test.sh` itself (Phase 2), so every subsequent per-commit task inherits them
  automatically rather than needing a separate backup/restore/capture task per step

---

## Phase 1: Setup

**Purpose**: Confirm the device, staged assets, and known-good/known-bad SHAs before spending any
build/flash time.

- [X] T001 Confirm M5 EVT1 (`0000088f8e579c33` via `ssh yanwen.xu@sj1-dmckee-d01`) is reachable and
  responsive (`adb devices`, `adb shell getprop ro.build.fingerprint`). — Confirmed:
  `Exynos/full_erd9975_c/erd9975:17/CP2A.260605.016/eng.abis:userdebug/20260720.134218,...`.
- [X] T002 [P] Confirm `test_coopmat_linear_bench_origcm` is already staged at
  `$D=/data/local/tmp/llama_vk` on-device (per quickstart.md Prerequisites) — this binary is
  reused as-is for every step, never rebuilt. — Confirmed present.
- [X] T003 [P] `cd /local/yanwen.xu/sumd/main && git fetch origin` and re-confirm
  `f14c51b6f850dbe6d1becfccef8e264e435c373b` (good) / `7bb715f7cc` (bad) are still the intended
  range endpoints and that the range is still ~303 commits (`git rev-list --count`) — record any
  drift from research.md §1/§2 if `origin/main` has moved. — Confirmed unchanged: still 303
  commits, despite `origin/main` advancing to `5ce9e559e9` (2026-07-21 15:06) — per spec
  Assumptions, `7bb715f7cc` stays the fixed bad endpoint.
- [X] T004 [P] Record current on-device driver hash (`adb shell md5sum
  /vendor/lib64/hw/vulkan.samsung.so`) as the pre-bisect baseline — this is the first potential
  "non-study driver" the backup/restore protocol (FR-003) may need to preserve. — Baseline hash
  `5abb6aa6dfd01ba6b32be72fbdf6ef0e`, unrecognized (not any documented hash) — a real drift, per
  spec context about this board's active sharing this session.
- [X] T005 [P] Create `specs/033-sumd-coopmat-segfault-bisect/results/bisect-report.md` skeleton
  (header, empty per-commit trace table per data-model.md's Bisect Step columns, empty Culprit
  Commit section) and an empty `specs/033-sumd-coopmat-segfault-bisect/results/tombstones/` dir. — Created.

**Checkpoint**: Device/assets confirmed, endpoint SHAs re-verified, report skeleton and tombstone
dir exist.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build the one shared artifact every user story depends on — the per-commit
backup→build→flash→test→restore→verdict procedure — per `contracts/bisect-test-script.md`.

**⚠️ CRITICAL**: No bisect step (US1, US2, or US3) can run until this phase is complete.

- [X] T006 Author `specs/033-sumd-coopmat-segfault-bisect/scripts/bisect-test.sh` per
  `contracts/bisect-test-script.md` invocation/preconditions: takes `<sumd-worktree-dir>
  [bisect_role]`, builds (`uv run scripts/run.py --os android --build --build-type release`,
  `vulkan-sdk` stripped from `LD_LIBRARY_PATH`, retry once on the documented GpuRt "Too many
  users" failure), stages+`cmp`-verifies the `.so` to NFS. — Authored.
- [X] T007 Extend `bisect-test.sh` (T006) with the driver backup/restore side effect (contract side
  effects 1 and 9 / FR-003): before flashing, `md5sum` the on-device driver; if it's not already a
  hash this study flashed, `adb pull` it and copy to a dated NFS backup path
  (`vulkan.samsung.so.<context>-backup-<date>`) before overwriting; after the step's verdict is
  captured, re-flash that backed-up driver and set `restored_after_step=true`. — Implemented and
  verified working every step; one **manual out-of-band flash during T011 debugging bypassed this
  protocol** (see results/bisect-report.md's Driver Backup Log note) — found and fixed at the end
  of the session (device manually restored to its original pre-bisect hash).
- [X] T008 Extend `bisect-test.sh` (T006/T007) with driver-identity capture (contract side effects
  4 and 6 / FR-005): `driver_hash_post_flash` immediately after flashing, re-pin clocks
  (`pin_freqs.sh` with `S=0000088f8e579c33 GPUFREQ=509000 MIFFREQ=2730000 INTFREQ=663000`, per
  research.md §4), then `driver_hash_pre_test` immediately before running the bench — exit `125`
  (skip) if the two hashes disagree (mid-step drift). — Implemented; hashes matched on every one
  of the 11 real runs, no drift-induced skip occurred.
- [X] T009 Extend `bisect-test.sh` (T006-T008) with the measurement + crash-detection step
  (contract side effects 7-8 / FR-006-FR-008): run `COOPMAT_BENCH_CORRECTNESS_ONLY=1
  ./test_coopmat_linear_bench_origcm` under a bounded timeout; classify `good` (exit 0 + stdout
  contains "Completed 16 test cases"), `bad` (signal-indicating exit code or stdout stops before
  completion — on `bad`, check `adb shell "ls -lat /data/tombstones/ | head -3"` for a
  post-test-start tombstone, pull it to `results/tombstones/<short-sha>.txt` if found, else fall
  back to exit code + last console line), or `skip` (build/flash failure, hash drift, or hang past
  timeout with device still responsive) — emit the exit-code contract (`0`/`1`/`125`) and the
  one-line verdict summary format from `contracts/bisect-test-script.md`'s Outputs section. —
  Implemented; first version omitted `COOPMAT_BENCH_CORRECTNESS_ONLY=1` from the invocation
  (caught and fixed in T011).
- [X] T010 Wire `bisect-test.sh` (T006-T009) to append one row per invocation to
  `results/bisect-report.md`'s trace table (data-model.md's Bisect Step schema, including
  `bisect_role`, driver hashes, verdict, `crash_evidence`/`pre_step_backup` sub-objects) —
  irrespective of outcome, including failures (contract postcondition 1). — Implemented; also logs
  every backup/restore event to `results/.driver-backup-log.tsv`.
- [X] T011 Dry-run `bisect-test.sh` once against the already-existing
  `/local/yanwen.xu/sumd/7bb715f7cc/` worktree (already built this session, per quickstart.md
  Prerequisites) to confirm the full backup→build→flash→test→restore→verdict→log loop works
  end-to-end before trusting it inside US1/US2 — this is expected to produce a `bad` verdict with
  a captured tombstone; fix any script bugs found before proceeding. — **Found and fixed a real
  bug**: the run command was missing `COOPMAT_BENCH_CORRECTNESS_ONLY=1`, so the binary ran its
  default (non-correctness) mode and never printed "Completed 16 test cases" — this silently
  misclassified the known-good `f14c51b6f8` as `bad`. Backup/restore/tombstone machinery itself
  worked correctly on this same dry run. Fixed; the dry run's row was discarded from the report
  (see report's Trace section note) and both endpoints were re-run cleanly under US1.

**Checkpoint**: `bisect-test.sh` is proven to backup/flash/measure/restore/verdict correctly —
endpoint re-confirmation (US1) can begin.

---

## Phase 3: User Story 1 - Re-confirm both range endpoints under the bisect harness (Priority: P1) 🎯 MVP

**Goal**: Both range endpoints are measured under `bisect-test.sh` itself and shown to still
produce different verdicts (`f14c51b6f8`=good, `7bb715f7cc`=bad) before any interior bisect step
runs.

**Independent Test**: Running the bisect harness against only the two endpoint SHAs produces one
`good` and one `bad` verdict.

**Depends on**: Phase 2 (needs a working `bisect-test.sh`).

### Implementation for User Story 1

- [X] T012 [US1] `cd /local/yanwen.xu/sumd/main && git worktree add ../f14c51b6f8
  f14c51b6f850dbe6d1becfccef8e264e435c373b` (good endpoint — needs a fresh worktree per
  quickstart.md Step 1, the earlier session's `f14c51b6f8-revert-69e887` worktree was for an
  unrelated experiment and no longer exists). — Found already existing, checked out at exactly
  this SHA (contrary to quickstart's note that it had been removed) — reused as-is, no new
  worktree needed.
- [X] T013 [US1] Run `scripts/bisect-test.sh ../f14c51b6f8 endpoint-good`; confirm verdict=`good`
  (exit 0, "Completed 16 test cases") — row is appended to `results/bisect-report.md`
  automatically (T010). — Confirmed `good` (after T011's fix).
- [X] T014 [US1] Run `scripts/bisect-test.sh ../7bb715f7cc endpoint-bad` (reusing the existing
  worktree, per T011's dry run — re-run cleanly under the `endpoint-bad` role rather than reusing
  T011's dry-run row if T011 already produced one, so the report's `endpoint-bad` row is
  unambiguous); confirm verdict=`bad` with a captured tombstone. — Confirmed `bad`, tombstone
  captured (SIGSEGV/SEGV_MAPERR), crash on the dispatch immediately after the first test case.
- [X] T015 [US1] Compare verdicts (FR-001 / Acceptance Scenario 1 vs. 2). If they disagree as
  expected, proceed to Phase 4. If the harness instead finds both endpoints agree, STOP — do not
  bisect a range with no isolatable culprit — and report the disagreement-from-expectation (both
  endpoints' full console output and any tombstone) instead. — Disagreed as expected (SC-001 met)
  — proceeded to Phase 4.

**Checkpoint**: Range confirmed to bracket the crash under the harness itself (SC-001) — this
alone is a valid, deliverable increment even if the full bisect doesn't complete.

---

## Phase 4: User Story 2 - Bisect to the first-bad commit (Priority: P1)

**Goal**: `git bisect` converges on exactly one first-bad SUMD commit, with every intermediate
step's build/flash/run result recorded.

**Independent Test**: Running the full bisect sequence produces a `git bisect` convergence on
exactly one commit, with every tested commit's row present in `results/bisect-report.md`.

**Depends on**: Phase 3 (US1) confirming the endpoints disagree under the harness.

### Implementation for User Story 2

- [X] T016 [US2] `cd /local/yanwen.xu/sumd/main && git bisect start 7bb715f7cc
  f14c51b6f850dbe6d1becfccef8e264e435c373b` (bad first, good second, per quickstart.md Step 2). —
  Started; 303 revisions, ~9 estimated steps.
- [X] T017 [US2] Bisect loop: for each commit `git bisect` selects, `SHA=$(git rev-parse --short
  HEAD)`, `git worktree add ../$SHA HEAD`, run `scripts/bisect-test.sh ../$SHA`, feed the verdict
  back via `git bisect good`/`bad`/`skip` per the script's exit code. Repeat until `git bisect`
  reports the first-bad commit (~9 iterations expected, research.md §2). If the M5 EVT1 board is
  found on an unrecognized driver hash at the start of any step, `bisect-test.sh`'s backup logic
  (T007) handles it automatically — do not bypass it even if it slows the loop down (spec Edge
  Cases). — Ran 9 interior steps, all clean (no skips): e85497828f=bad, f8d24657a4=bad,
  674d11d8ec=bad, c4153769bb=good, 02c6a42337=bad, 63bcd7824a=good, c259018f96=good,
  308a98df10=bad, 805609f0da=bad (converged).
- [X] T018 [US2] If a run of `skip`s stalls `git bisect`'s convergence (FR-009 / spec Edge Cases),
  manually build/test commit(s) adjacent to the skipped span (walking outward as needed) and feed
  those verdicts back in via the same loop, until `git bisect` narrows to exactly one commit — do
  not accept an ambiguous multi-commit range as final. — Not needed: zero skips occurred in this
  bisect, every one of the 9 interior steps produced a clean good/bad verdict.
- [X] T019 [US2] Once `git bisect` converges: `git bisect log >
  specs/033-sumd-coopmat-segfault-bisect/results/bisect-log-raw.txt`; record the identified
  first-bad commit's SHA, author, date, and subject line (`git show --no-patch --format='%H%n%an%n%ci%n%s'
  <first-bad-sha>`); `git bisect reset` to restore `main` to its normal branch (confirm not left
  detached). — Converged: **first-bad `805609f0dabbbbe4f1b1687adf1d35a0b1e8a6f9`** ("spal: Skip
  GL2 WB+Inv WA on clean global barriers", Aaron Zhong, 2026-06-10). Log captured; `git bisect
  reset` confirmed `main` back on branch `main`, not detached.

**Checkpoint**: First-bad commit identified; every step's row is in the trace table — the core
"find the culprit" goal (SC-002) is met.

---

## Phase 5: User Story 3 - Document driver provenance, crash evidence, and the full bisect trace (Priority: P2)

**Goal**: `results/bisect-report.md` is a complete, self-contained record: every tested commit's
SHA/driver-hash/build-outcome/verdict/crash-evidence in order, plus the Culprit Commit section with
its diff-derived explanation of *why* it plausibly causes the crash.

**Independent Test**: Reading `results/bisect-report.md` alone (no session replay) shows every
tested commit's full row (with tombstone/crash evidence for every `bad` row) and the culprit's
SHA/author/date/subject/diff-summary with last-good/first-bad driver hashes side by side.

**Depends on**: Phase 4 (US2) — needs the completed trace and identified culprit commit.

### Implementation for User Story 3

- [X] T020 [US3] Verify `results/bisect-report.md`'s trace table has one row per commit actually
  tested (endpoints + bisect-loop steps + any skip-adjacent probes from T018) — no gaps, and every
  `bad` row has a non-null `crash_evidence` (FR-008/FR-010). — Verified: 11 rows (2 endpoints + 9
  interior steps), all 7 `bad` rows have a captured tombstone reference, no gaps.
- [X] T021 [US3] Read the first-bad commit's diff (`git show <first-bad-sha>` in
  `/local/yanwen.xu/sumd/main`, permitted per Rule 0 being lifted — research.md §9) and write
  `diff_summary`: what the commit changed and why that plausibly causes the observed
  coopmat-dispatch `SIGSEGV` (FR-011). — Read; mechanism: gates an always-safe GL2 WB+Inv cache
  flush behind a resource-tracker "clean barrier" heuristic — if the tracker misclassifies the
  coopmat dispatch's writes, the flush is skipped and the next dispatch reads stale/invalid L2
  data, matching the fault-address pattern in every tombstone.
- [X] T022 [US3] Write the Culprit Commit section of `results/bisect-report.md`: first-bad SHA
  /author/date/subject, last-good SHA/date, both commits' on-device driver hashes side by side,
  and T021's `diff_summary` — sufficient for both commits to be opened directly in Gerrit
  (FR-012/SC-006). — Written.
- [X] T023 [US3] Add a self-contained summary paragraph at the top of `results/bisect-report.md`
  (device, predicate, method, headline culprit finding) — readable without this session's chat
  context, same convention `specs/032`'s report used. — Written.

**Checkpoint**: `results/bisect-report.md` is the complete, standalone deliverable.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final validation that the report satisfies all six spec Success Criteria before
calling this done.

- [X] T024 Run through `quickstart.md`'s "Validating the deliverable" checklist (SC-001 through
  SC-006) against the finished `results/bisect-report.md` and record the outcome of each check. —
  **SC-001**: met, endpoints disagreed under the harness before any interior step. **SC-002**: met,
  exactly one first-bad commit (`805609f0da`). **SC-003**: met, single document with all 11 rows +
  crash evidence. **SC-004**: met, see T025. **SC-005**: met with one documented exception (the
  T011 manual-flash gap), fixed by end of session — see T026. **SC-006**: met, Culprit Commit
  section has SHA/author/date/subject/diff-summary + bracketing driver hashes.
- [X] T025 Confirm every trace row's device is M5 EVT1 `0000088f8e579c33` by construction
  (`bisect-test.sh` hardcodes the serial) — spot-check a few rows' `md5sum` captures (SC-004). —
  Confirmed by construction (hardcoded `SERIAL` var); spot-checked hashes match remote output.
- [X] T026 Confirm every row that required a pre-step backup (T007) shows
  `restored_after_step=true`, and spot-check `/sarc-c/gpusw/users/yanwen.xu/` for the backup files
  themselves — zero net driver-state loss for other users of the shared board (SC-005). — All 11
  `.driver-backup-log.tsv` rows show `restored_after_step=1`. **One exception found**: a manual
  out-of-band flash during T011 debugging (outside `bisect-test.sh`) left the device on
  `c9861e9906…` instead of the true pre-session baseline `5abb6aa6dfd01ba6b32be72fbdf6ef0e` — the
  backup file existed on NFS (nothing lost), but it wasn't auto-restored since it bypassed the
  script. Manually restored at end of session; confirmed device now back on
  `5abb6aa6dfd01ba6b32be72fbdf6ef0e`. Documented in the report's Driver Backup Log section.
- [X] T027 Confirm `git worktree list` in `/local/yanwen.xu/sumd/main` shows every SHA-named
  worktree created during this bisect still present (left in place, per `sumd/CLAUDE.md`
  convention), and that `main` itself is back on branch `main` (not detached) after `git bisect
  reset`. — Confirmed: all 9 new worktrees (`e85497828f`, `f8d24657a4`, `674d11d8ec`,
  `c4153769bb`, `02c6a42337`, `63bcd7824a`, `c259018f96`, `308a98df10`, `805609f0da`) present
  alongside the reused `f14c51b6f8`/`7bb715f7cc`; `main` confirmed on branch `main`, not detached.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately.
- **Foundational (Phase 2)**: Depends on Setup (needs confirmed device/assets/endpoints) — blocks
  ALL user stories (there is no bisect step without a working `bisect-test.sh`).
- **User Story 1 (Phase 3)**: Depends on Foundational — first gate, must pass (endpoints disagree)
  before Phase 4.
- **User Story 2 (Phase 4)**: Depends on User Story 1's endpoints disagreeing — cannot start
  otherwise (spec Acceptance Scenario 2).
- **User Story 3 (Phase 5)**: Depends on User Story 2's completed trace and identified culprit.
- **Polish (Phase 6)**: Depends on all three user stories being complete.

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational. Blocks User Story 2.
- **User Story 2 (P1)**: Can start only after User Story 1 confirms disagreement. Blocks User
  Story 3.
- **User Story 3 (P2)**: Can start only after User Story 2 converges on a culprit commit.

This feature's stories are **strictly sequential**, unlike the template's general case — each
gates the next by design (US2 needs US1's confirmed-disagreeing range; US3 documents US2's
result), so there is no parallel-team opportunity across stories here.

### Parallel Opportunities

- Setup tasks marked [P] (T002-T005) can run in parallel.
- Nothing in Phase 2 is marked [P] — `bisect-test.sh` is authored/extended incrementally in one
  file, each task building on the last.
- Within Phase 4's bisect loop (T017), the loop itself is inherently sequential (each step's
  verdict determines `git bisect`'s next commit) — no parallelism available.

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup.
2. Complete Phase 2: Foundational (CRITICAL — blocks all stories).
3. Complete Phase 3: User Story 1 — re-confirm both endpoints disagree under the harness.
4. **STOP and VALIDATE**: If the endpoints don't disagree, the investigation itself is inconclusive
   — report that finding rather than proceeding.

### Incremental Delivery

1. Complete Setup + Foundational → harness ready.
2. Add User Story 1 → endpoints confirmed to disagree (MVP gate).
3. Add User Story 2 → first-bad commit identified.
4. Add User Story 3 → full trace + crash evidence + diff-explained culprit documented.
5. Polish → all six Success Criteria checked off.

---

## Notes

- [P] tasks = different files/no dependencies; most of this feature's tasks are NOT [P] since
  almost everything either extends the single `bisect-test.sh` file or is a sequential bisect-loop
  step that depends on the previous step's verdict.
- [Story] label maps task to specific user story for traceability.
- This is a hardware-in-the-loop investigation, not conventional software — "commit after each
  task" doesn't apply; instead, append a `results/bisect-report.md` row after every
  `bisect-test.sh` invocation (wired automatically in T010) and update the Culprit Commit section
  once US2/US3 complete.
- Avoid: skipping the backup/restore step to save time (FR-003 is a hard requirement on this
  actively-shared board, not an optimization target); treating an ambiguous multi-commit
  `git bisect` result as final (FR-009); silently discarding a build/flash failure instead of
  recording it as `skip` with a reason.
