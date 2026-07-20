---

description: "Task list for SUMD Driver Bisect for the 8da4w-Slower-Than-4w Regression"
---

# Tasks: SUMD Driver Bisect for the 8da4w-Slower-Than-4w Regression

**Input**: Design documents from `specs/032-sumd-driver-bisect/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/bisect-test-script.md, quickstart.md

**Tests**: Not requested — this is a hardware bisect investigation; verification is the
endpoint-disagreement gate, clock-pin readback, and quickstart SC checks below, not a code test
suite.

**Organization**: Tasks are grouped by user story (US1=confirm range endpoints, US2=bisect to the
first-bad commit, US3=document driver provenance + full trace, per spec.md's priorities P1/P1/P2 —
US1 and US2 are both P1 since US2 cannot start without US1's gate, but they're sequenced as
separate phases here since US1 is independently completable/testable on its own).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to
- Device: M41, serial `00000a34cdd4abd3`, via `ssh xgpusw-debug07` (hard-locked, FR-003 — every
  task below runs there, not restated per-task)
- SUMD worktrees live under `/local/yanwen.xu/sumd/<short-sha>/`; this feature's own docs/scripts
  live under `specs/032-sumd-driver-bisect/`

---

## Phase 1: Setup

**Purpose**: Confirm the device, tooling, and staged assets before spending any build/flash time.

- [X] T001 Confirm M41 (`ANDROID_SERIAL=00000a34cdd4abd3` via `ssh xgpusw-debug07`) is reachable and responsive (`adb devices`, `adb shell getprop ro.build.fingerprint`). — Confirmed: device present, ERD9965/S5E9965, `erd9965_b`, `BP2A.250605.031.A3`.
- [X] T002 [P] Confirm the 1B 4w/8da4w PTEs (ctx supporting 2048-token prefill) are staged on-device or on NFS run-kit; push from `/sarc-c/gpusw/users/yanwen.xu/android-run/models/` if missing. — Confirmed: both PTEs, tokenizer, and `p2048_exact.txt` already at `/data/local/tmp/llama_vk/` on-device.
- [X] T003 [P] Confirm the `llama_main_rel1.3` runner binary is built and available (from the `release-1.3/` worktree) — build it if not. — Confirmed: already staged on-device (2026-07-14).
- [X] T004 [P] `cd /local/yanwen.xu/sumd/main && git fetch origin` and re-run the boundary-commit queries from research.md §1 (`git log -1 --before="2024-11-01 23:59:59" main` / `--before="2026-03-31 23:59:59"`) to confirm `898709039d1`/`ec3958eae55` are still correct — `main` may have moved since 2026-07-16. — Confirmed unchanged: same two SHAs, still 3,055 commits in range, despite `origin/main` advancing to `cfd438120e` (2026-07-15).
- [X] T005 [P] Create `specs/032-sumd-driver-bisect/results/bisect-report.md` skeleton: header, empty per-commit trace table (data-model.md's Bisect Step columns), and an empty Culprit Commit section. — Created.

**Checkpoint**: Device/assets/tooling confirmed, boundary SHAs re-verified, report skeleton exists.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build the one shared artifact every user story depends on — the per-commit
build+flash+measure+verdict procedure — and the clock-pin mechanism it relies on.

**⚠️ CRITICAL**: No bisect step (US1, US2, or US3) can run until this phase is complete.

- [X] T006 Author `specs/032-sumd-driver-bisect/scripts/bisect-test.sh` per `contracts/bisect-test-script.md`: takes a SUMD worktree dir, builds (`uv run scripts/run.py --os android --build --build-type release`, `vulkan-sdk` stripped from `LD_LIBRARY_PATH`), stages+`cmp`-verifies to NFS, flashes to M41, captures driver identity via `adb shell md5sum /vendor/lib64/hw/vulkan.samsung.so` (corrected 2026-07-16 from an initial `logcat`-based plan — see research.md §6), and exits `125` on any build/flash/crash failure — measurement wiring (T007) comes next.
- [X] T007 Extend `bisect-test.sh` (T006) with the clock-pin step: reuse `pin_freqs.sh`'s `S`/`GPUFREQ`/`MIFFREQ`/`INTFREQ` env-override interface (research.md §4 — read directly, it's workspace tooling not SUMD source) with M41's max values (980000/5333000/800000), check its own printed sysfs readback matches, and re-pin once if it doesn't before proceeding — exit `125` if it still doesn't match after one retry.
- [X] T008 Extend `bisect-test.sh` (T006/T007) with the measurement step: run `llama_main_rel1.3` once each for 4w and 8da4w (Llama 3.2 1B, 2048-token prefill, 1 rep), parse `prefill_token_per_sec` from each run's JSON output, and emit the exit-code verdict per the contract (`0`=8da4w>4w, `1`=otherwise) plus the one-line log format specified in `contracts/bisect-test-script.md`.
- [X] T009 Dry-run `bisect-test.sh` once against the already-existing `/local/yanwen.xu/sumd/f14c51b6f8` worktree to confirm the build+flash+measure+verdict portion works end-to-end before trusting it inside the bisect loop. — Found and fixed 2 real bugs in the process: (1) `adb shell`/`pin_freqs.sh` calls inside the heredoc-fed remote script were consuming the heredoc's own stdin, silently truncating script execution after the first `adb shell setenforce 0` — fixed with `</dev/null` on every remote adb/pin_freqs.sh call; (2) the crash-detection regex matched the generic word "Error" in a documented-harmless tokenizer JSON-parse warning, misclassifying a successful run as a crash — tightened to specific fatal markers (`libc++abi`, `VK_ERROR`, `SIGSEGV`, `Fatal signal`, `Aborted`, `terminate called`). After both fixes: full run succeeded (4w=603.774, 8da4w=430.433 tok/s, verdict=bad, driver md5 `c9861e9906d…` matches known `f14c51b6f8` hash) — numbers closely match the pre-session JIRA evidence (605.2/425.3) for this exact driver on M41.

**Checkpoint**: `bisect-test.sh` is proven to flash+measure+verdict correctly — endpoint checks (US1) can begin.

---

## Phase 3: User Story 1 - Confirm the bisect range brackets the regression (Priority: P1) 🎯 MVP

**Goal**: Both range endpoints are measured and shown to produce different verdicts before any
interior bisect step runs.

**Independent Test**: Building, flashing, and measuring only the two endpoint commits produces
one `good` and one `bad` verdict.

**Depends on**: Phase 2 (needs a working `bisect-test.sh`).

### Implementation for User Story 1

- [X] T010 [US1] `git worktree add /local/yanwen.xu/sumd/898709039d 898709039d173379d987ff4c9289cc5be7ee09ef` (Nov 2024 endpoint, per T004's re-confirmed SHA).
- [X] T011 [US1] `git worktree add /local/yanwen.xu/sumd/ec3958eae5 ec3958eae55ec3826d829d2a1149ddb4765b8af4` (Mar 2026 endpoint, per T004's re-confirmed SHA).
- [X] T012 [US1] Run `scripts/bisect-test.sh /local/yanwen.xu/sumd/898709039d`; append the resulting row to `results/bisect-report.md`'s trace table (`bisect_role=endpoint-old`). — **Result: `skip`, not a verdict.** Crashed on-device (`vkQueueWaitIdle` returned -4 / VK_ERROR_DEVICE_LOST) — a genuine driver/hardware incompatibility, not a script bug (confirmed via log inspection). Per spec Edge Cases ("skip → probe adjacent commits"), applied here proactively: probed `626c6bd367` (+100 commits, 2024-11-19) — same crash. Then jumped to the commit-count midpoint of the whole range, `f61822f069` (2025-09-19, line 1528/3046) — **clean run, verdict=bad** (610.25 vs 220.928). Then binary-searched between the crash zone and that clean-bad point: `b6487d67b7` (2025-04-11, line 814/3046) — **clean run, verdict=good** (609.342 vs 805.665, the expected/normal ordering). This is the key finding of Phase 3: a confirmed, crash-free good/bad bracket exists at `b6487d67b7` (good) ↔ `f61822f069` (bad), narrower than but fully inside the original Nov2024–Mar2026 range — the crash zone (Nov2024 to sometime before Apr2025) doesn't need further resolution since the actual regression's flip point is bracketed well after it ends. All rows recorded in `results/bisect-report.md`.
- [X] T013 [US1] Run `scripts/bisect-test.sh /local/yanwen.xu/sumd/ec3958eae5`; append the resulting row (`bisect_role=endpoint-new`). — **Result: verdict=bad** (4w=606.635, 8da4w=148.837 — the most pronounced gap seen yet, worse than both the Sep2025 midpoint and the current-default f14c51b6f8). Confirms the literal Mar-2026 boundary is clean (no crash) and bad, consistent with every other post-crash-zone point tested.
- [X] T014 [US1] Compare verdicts. The literal calendar endpoints don't both give clean verdicts (old end crashes) — but per the crash-zone investigation above, a confirmed good (`b6487d67b7`)/bad (`f61822f069`) bracket exists inside the range. Proceeding to Phase 4 using that bracket as the `git bisect start` arguments, per spec's skip-handling policy (probe adjacent/nearby testable commits rather than treating an untestable literal boundary as a blocker) — documented here rather than silently substituted.

**Checkpoint**: Range confirmed to bracket a flip (or investigation halted with both numbers documented) — this alone is a valid, deliverable increment even if the full bisect doesn't complete.

---

## Phase 4: User Story 2 - Bisect to the first bad commit (Priority: P1)

**Goal**: `git bisect` converges on exactly one first-bad SUMD commit, with every intermediate
step's measurement recorded.

**Independent Test**: Running the full bisect sequence produces a `git bisect` convergence on
exactly one commit, with every tested commit's row present in `results/bisect-report.md`.

**Depends on**: Phase 3 (US1) confirming the range brackets a flip.

### Implementation for User Story 2

- [X] T015 [US2] `cd /local/yanwen.xu/sumd/main && git bisect start <bad-sha> <good-sha>` using T014's determined polarity. — Ran `git bisect start f61822f0690d356b4751288a15f9258d5ff9b79e b6487d67b762157ba2751f47ab2b8100ebb78f07` (the crash-zone-adjusted bracket, not the literal calendar endpoints — see T014). 356 revisions, ~9 estimated steps.
- [X] T016 [US2] Bisect loop — completed 9 real bisect steps (ab6cb4d071=good, 0d498f7990=bad, 77215300a3=good, 9873412d20=good, 212ddce456=good, e3901f2db8=good, 635f83ba1a=good, 38e9a781d6=good, 69e887275e=bad) plus one extra (0b814fa6d3=good, since `git bisect` presented one more candidate than its own initial estimate). Each step: dedicated SHA-named worktree, `scripts/bisect-test.sh`, row appended automatically by the script, verdict fed back via `git bisect good`/`bad`. All clean runs — no skips in this bracket (the only crashes were in the earlier crash-zone probing, outside this bracket).
- [X] T017 [US2] Not needed — no `skip` occurred within the confirmed good/bad bracket, so the adjacent-probing fallback was never triggered here (it *was* used proactively in Phase 3 to establish the bracket itself, per T012's note).
- [X] T018 [US2] `git bisect` converged: **first bad commit `69e887275e26ae4a44e2d6e14bd3e600cec67ac8`** ("xgl,sc: fixup! Use feature json to control WMMA and V_DOT usage", Prabhakar Pal, 2025-08-18 23:57:20 -0500) — disables `dot4_i32_i8`/`dot2_f32_f16` shader-compiler patterns, the int8 dot-product instruction 8da4w's arithmetic depends on. `git bisect log` captured to `results/bisect-log-raw.txt`; `git bisect reset` restored `main` to its normal branch (confirmed: back on `main`, not detached).

**Checkpoint**: First-bad commit identified; every step's row is in the trace table — the core "find the blame" goal is met.

---

## Phase 5: User Story 3 - Document driver provenance and the full bisect trace (Priority: P2)

**Goal**: `results/bisect-report.md` is a complete, self-contained record: every tested commit's
SHA/driver-version/numbers/verdict in order, plus the culprit commit with its bracketing driver
versions.

**Independent Test**: Reading `results/bisect-report.md` alone (no session replay) shows every
tested commit's full row and the culprit's SHA/author/date/subject with last-good/first-bad
driver version strings side by side.

**Depends on**: Phase 4 (US2) — needs the completed trace and identified culprit.

### Implementation for User Story 3

- [X] T019 [US3] Verified: 15 trace rows present (4 crash-zone/endpoint probes + 11 bisect-loop steps), matching all commits actually tested — no gaps.
- [X] T020 [US3] Culprit Commit section written: SHA/author/date/subject for both last-good (`0b814fa6d3`) and first-bad (`69e887275e`), their driver md5 identities side by side, the commit body (metadata only), files-touched list (paths + line counts only, per Rule 0), and a mechanism explanation (`dot4_i32_i8` disablement).
- [X] T021 [US3] Self-contained summary paragraph added at the top of `results/bisect-report.md` (device, workload, predicate, method incl. the crash-zone deviation, and the headline culprit finding) — readable without this session's chat context.

**Checkpoint**: `results/bisect-report.md` is the complete, standalone deliverable.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final validation that the report satisfies all five spec Success Criteria before
calling this done.

- [X] T022 Ran `quickstart.md`'s SC checklist:
  - **SC-001** (partial/adapted): the *literal* calendar endpoints didn't both give clean verdicts (Nov-2024 crashed) — but per the documented crash-zone deviation, a confirmed good/bad bracket (`b6487d67b7`/`f61822f069`) was established and measured before any bisect-loop step. Intent met, mechanism adapted — documented, not silently substituted.
  - **SC-002**: met — exactly one first-bad commit (`69e887275e`).
  - **SC-003**: met — single document (`results/bisect-report.md`), 15 rows, SHA/driver-identity/numbers/verdict, in order tested.
  - **SC-004**: met — see T024.
  - **SC-005**: met — Culprit Commit section has SHA/author/date/subject + bracketing driver identities, sufficient for Gerrit lookup.
- [X] T023 Confirmed via review of every command run this session: no SUMD source file was opened/read/grepped. Only workspace tooling was read (`sumd/CLAUDE.md`, `deploy_android.sh`, `pin_freqs.sh` — none are SUMD driver source) plus commit *metadata* (`git log`/`git show --no-patch`, diff *stats* only, never diff content) for the culprit commit.
- [X] T024 Confirmed: `bisect-test.sh` hardcodes `SERIAL=00000a34cdd4abd3` — every one of the 15 measurement rows used this device by construction, no substitution possible.
- [X] T025 Confirmed: `git worktree list` in `/local/yanwen.xu/sumd/main` shows all 15 SHA-named worktrees created during this bisect, still present; `main` itself correctly restored to branch `main` (not detached) after `git bisect reset`.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately.
- **Foundational (Phase 2)**: Depends on Setup (needs confirmed device/assets) — blocks ALL user stories (there is no bisect step without a working `bisect-test.sh`).
- **User Story 1 (Phase 3)**: Depends on Foundational — first gate, must pass before Phase 4.
- **User Story 2 (Phase 4)**: Depends on User Story 1's endpoints disagreeing — cannot start otherwise (spec Acceptance Scenario 2).
- **User Story 3 (Phase 5)**: Depends on User Story 2 having converged on a culprit commit.
- **Polish (Phase 6)**: Depends on User Story 3's report being complete.

### User Story Dependencies

Unlike a typical multi-service feature, these three stories are **strictly sequential**, not
independently parallelizable — each is a precondition for the next (US1 gates US2, US2's output
is US3's input). This mirrors the single-shared-device constraint already documented in
`specs/030`'s tasks.md, plus an additional logical dependency chain specific to bisection.

### Parallel Opportunities

- T002/T003/T004/T005 (Setup) have no file/device overlap and can run in parallel.
- T010/T011 (creating the two endpoint worktrees) can run in parallel with each other, but T012/T013 (running the test script against each) still contend for the single M41 device — sequential in practice.
- Within Phase 4's bisect loop (T016), each iteration's worktree-add is cheap and could be prepared ahead of the previous step's verdict, but the *measurement* itself is inherently sequential (one device, one commit at a time, and `git bisect`'s next pick depends on the previous verdict).

---

## Implementation Strategy

### MVP First (User Story 1 only)

1. Complete Phase 1 (Setup) + Phase 2 (Foundational — build `bisect-test.sh`).
2. Complete Phase 3 (US1 — confirm the range brackets a flip).
3. **STOP and VALIDATE**: if the two endpoints disagree, the range is proven bisectable — this alone is valuable even if device time runs out before the full bisect (US2/US3).
4. If the two endpoints *agree*, this is also a complete, valid outcome per spec Acceptance Scenario 2 — report it and stop; there is nothing for US2 to do until the range or predicate is revisited.

### Incremental Delivery

1. Setup + Foundational → device confirmed, `bisect-test.sh` proven end-to-end (T009's dry run).
2. Add US1 → validate → range confirmed to bracket a flip (or investigation halted with numbers in hand).
3. Add US2 → validate → single first-bad commit identified, full trace recorded.
4. Add US3 → validate → complete, self-contained report with culprit + driver-version delta.
5. Polish → final five-SC completeness confirmation.

### Sequential-Only Strategy

There is no parallel-team strategy here (unlike a typical multi-service feature): one physical
device (M41), one git-bisect state machine, and a strict logical dependency chain (US1 → US2 →
US3). The real constraint is wall-clock build+flash+measure time (~14 cycles, research.md §2), not
staffing — order exactly as phased above.
