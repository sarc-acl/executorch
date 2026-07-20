# Quickstart: SUMD Driver Bisect for the 8da4w-Slower-Than-4w Regression

**Feature**: `032-sumd-driver-bisect` | **Date**: 2026-07-16

This validates the bisect procedure end-to-end. Full mechanics are in `research.md` and
`contracts/bisect-test-script.md`; this is the runnable sequence, not a restatement of them.

## Prerequisites

- `/local/yanwen.xu/sumd/main` worktree up to date (`git fetch origin` if the boundary commits
  below aren't yet local).
- M41 reachable: `ssh xgpusw-debug07`, then `export ANDROID_SERIAL=00000a34cdd4abd3`.
- 1B 4w/8da4w PTEs (ctx supporting 2048-token prefill) staged on M41 or on the NFS run-kit.
- `llama_main_rel1.3` runner binary available (built from `release-1.3/` worktree).

## Step 1 — Verify the range brackets a flip (User Story 1 / FR-001, FR-002)

```bash
cd /local/yanwen.xu/sumd/main
git worktree add ../898709039d 898709039d173379d987ff4c9289cc5be7ee09ef   # Nov 2024 endpoint
git worktree add ../ec3958eae5 ec3958eae55ec3826d829d2a1149ddb4765b8af4   # Mar 2026 endpoint
scripts/bisect-test.sh ../898709039d
scripts/bisect-test.sh ../ec3958eae5
```

**Expected outcome**: the two verdicts differ (one `good`, one `bad`). If not, stop — do not
proceed to Step 2 (spec User Story 1, Acceptance Scenario 2).

## Step 2 — Start the bisect (User Story 2 / FR-009)

```bash
cd /local/yanwen.xu/sumd/main
git bisect start ec3958eae55ec3826d829d2a1149ddb4765b8af4 898709039d173379d987ff4c9289cc5be7ee09ef
```

(`bad` commit first, `good` commit second — reverse the two SHAs above if Step 1 found the
opposite polarity.)

For each commit `git bisect` checks out next:

```bash
SHA=$(git rev-parse --short HEAD)
git worktree add ../$SHA HEAD    # or: cd into a fresh detached checkout of $SHA
scripts/bisect-test.sh ../$SHA
git bisect good    # or `bad` / `skip`, per the script's exit code / your own read of its output
```

Repeat until `git bisect` reports the first bad commit (expect ~12 iterations, per
`research.md` §2). If a run of `skip`s stalls convergence, manually test commit(s) adjacent to
the skipped span (spec Edge Cases) and feed those verdicts in the same way.

## Step 3 — Confirm and record (User Story 3 / FR-010, FR-011)

```bash
git bisect log > /local/yanwen.xu/workspace/dev/executorch/specs/032-sumd-driver-bisect/results/bisect-log-raw.txt
git bisect visualize --format='%H %ci %s'   # the identified first-bad commit
git show --no-patch --format='%H%n%an%n%ci%n%s' <first-bad-sha>
git bisect reset
```

Then write `results/bisect-report.md`: one row per tested commit (`data-model.md`'s Bisect Step
schema) in the order tested, plus the Culprit Commit summary with last-good/first-bad driver
version strings side by side.

## Validating the deliverable

- **SC-001**: both endpoint rows are present and disagree, dated before any interior-step row.
- **SC-002**: exactly one commit is reported as first-bad; `git bisect log`'s replay confirms it.
- **SC-003**: every row in `results/bisect-report.md` has a verdict — none silently dropped.
- **SC-004**: every row's device is M41 `00000a34cdd4abd3` (spot-check a few rows' `md5sum` capture).
- **SC-005**: the report states the culprit's SHA/author/date/subject and the two bracketing
  driver version strings.
