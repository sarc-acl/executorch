# Quickstart: SUMD Driver Bisect for the Coopmat-Dispatch Segfault Regression

**Feature**: `033-sumd-coopmat-segfault-bisect` | **Date**: 2026-07-21

This validates the bisect procedure end-to-end. Full mechanics are in `research.md` and
`contracts/bisect-test-script.md`; this is the runnable sequence, not a restatement of them.

## Prerequisites

- `/local/yanwen.xu/sumd/main` worktree up to date (`git fetch origin`).
- M5 EVT1 reachable: `ssh yanwen.xu@sj1-dmckee-d01`, then `S=0000088f8e579c33`.
- `test_coopmat_linear_bench_origcm` already staged at `$D=/data/local/tmp/llama_vk` on-device.
- `/local/yanwen.xu/sumd/7bb715f7cc/` already exists (built this session to confirm the bad
  endpoint) and can be reused directly. `f14c51b6f8` needs a fresh SHA-named worktree (Step 1)
  — the worktree used earlier this session to test it (`f14c51b6f8-revert-69e887`) was created for
  an unrelated experiment and has already been removed; don't assume it's still there.

## Step 1 — Re-confirm the range under the harness (User Story 1 / FR-001)

```bash
cd /local/yanwen.xu/sumd/main
git worktree add ../f14c51b6f8 f14c51b6f850dbe6d1becfccef8e264e435c373b   # good endpoint
# ../7bb715f7cc already exists from this session
scripts/bisect-test.sh ../f14c51b6f8 endpoint-good
scripts/bisect-test.sh ../7bb715f7cc endpoint-bad
```

**Expected outcome**: `f14c51b6f8` verdicts `good` (exit 0), `7bb715f7cc` verdicts `bad` (exit 1,
tombstone captured). If not — if the harness itself disagrees with this session's earlier ad hoc
checks — stop and report the disagreement (spec User Story 1, Acceptance Scenario 2) before
proceeding to Step 2.

## Step 2 — Start the bisect (User Story 2 / FR-009)

```bash
cd /local/yanwen.xu/sumd/main
git bisect start 7bb715f7cc f14c51b6f850dbe6d1becfccef8e264e435c373b
```

(`bad` commit first, `good` commit second.)

For each commit `git bisect` checks out next:

```bash
SHA=$(git rev-parse --short HEAD)
git worktree add ../$SHA HEAD
scripts/bisect-test.sh ../$SHA
git bisect good    # or `bad` / `skip`, per the script's exit code
```

Repeat until `git bisect` reports the first-bad commit (expect ~9 iterations, per `research.md`
§2). If a run of `skip`s stalls convergence, manually test commit(s) adjacent to the skipped span
(spec Edge Cases) and feed those verdicts in the same way. If the device is found on an
unrecognized hash at the start of any step, back it up before proceeding (contract side effect 1)
— do not skip this even if it slows the loop down.

## Step 3 — Confirm, read the culprit's diff, and record (User Story 3 / FR-010, FR-011, FR-012)

```bash
git bisect log > /local/yanwen.xu/workspace/dev/executorch/specs/033-sumd-coopmat-segfault-bisect/results/bisect-log-raw.txt
git bisect visualize --format='%H %ci %s'   # the identified first-bad commit
git show --no-patch --format='%H%n%an%n%ci%n%s' <first-bad-sha>
git show <first-bad-sha>                     # read the diff — permitted, Rule 0 lifted (research.md §9)
git bisect reset
```

Then write `results/bisect-report.md`: one row per tested commit (`data-model.md`'s Bisect Step
schema, including `crash_evidence` for every `bad` row) in the order tested, plus the Culprit
Commit summary — SHA/author/date/subject, last-good/first-bad driver hashes side by side, and
`diff_summary` explaining what the commit changed and why it plausibly causes the segfault.

## Validating the deliverable

- **SC-001**: both endpoint rows are present and disagree, dated before any interior-step row.
- **SC-002**: exactly one commit is reported as first-bad; `git bisect log`'s replay confirms it.
- **SC-003**: every row in `results/bisect-report.md` has a verdict and, for `bad` rows, crash
  evidence — none silently dropped.
- **SC-004**: every row's device is M5 EVT1 `0000088f8e579c33` (spot-check a few rows' `md5sum`
  capture).
- **SC-005**: every row that required a pre-step backup shows `restored_after_step = true` —
  spot-check `/sarc-c/gpusw/users/yanwen.xu/` for the backup files themselves.
- **SC-006**: the report states the culprit's SHA/author/date/subject/diff summary and the two
  bracketing driver hashes.
