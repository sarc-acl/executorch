# Quickstart: MiniPC RDNA3 Handoff Report

A documentation feature -- no device capture, no new code. Validation is
reading the report and checking its numbers, not running anything.

## 1. Produce the report

Write `specs/013-minipc-handoff-report/results/handoff-report.md` covering:
- Consolidated Findings table (research.md Decision 1)
- Open items (research.md Decision 2)
- Repo Handoff State (research.md Decision 3)
- Samsung/Xclipse Runbook (high-level, per Clarifications)

## 2. Validate

- Spot-check a few headline numbers against their cited source file --
  should match exactly (FR-002).
- Confirm the open items section names all four items from research.md
  Decision 2, not just some.
- Confirm the repo state section matches a fresh `git status`/`git log`.
- Confirm the runbook doesn't assert specific untested `adb`/Android
  commands as proven -- pointers and carries-over/needs-adaptation/
  newly-established status per step is enough (Clarifications).

## 3. Before actually cloning elsewhere

Commit and push the working-tree changes -- this is a prerequisite the
report names but does not perform (FR-004). Confirm explicitly with the
user before doing this, per this repo's own commit-only-when-asked
convention.
