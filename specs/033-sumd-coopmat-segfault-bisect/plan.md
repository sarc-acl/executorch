# Implementation Plan: SUMD Driver Bisect for the Coopmat-Dispatch Segfault Regression

**Branch**: `033-sumd-coopmat-segfault-bisect` | **Date**: 2026-07-21 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/033-sumd-coopmat-segfault-bisect/spec.md`

## Summary

A teammate reported `test_coopmat_linear_bench_origcm` (`COOPMAT_BENCH_CORRECTNESS_ONLY=1`)
segfaulting on M5 EVT1 partway through the correctness matrix — the crash is inside
`vulkan.samsung.so` itself (tombstone-confirmed `SIGSEGV`/`SEGV_MAPERR`), on the second (coopmat)
test case, right after the first (tiled) case passes. This session already confirmed the range
endpoints: `f14c51b6f8` (`main` @ 2026-06-15, the team's documented known-good driver) is `good`
(16/16 PASS), and `7bb715f7cc` (`main` tip @ 2026-07-21, built fresh for this check) is `bad`
(reproduces the identical crash). This feature git-bisects the 303 commits between them to find
the single first-bad commit, on M5 EVT1 (hard-locked device, serial `0000088f8e579c33`), with a
mandatory backup/restore of whatever driver is on the device before/after every step (the board
drifted to at least three undocumented builds within hours this session — a real, not
hypothetical, risk of clobbering someone else's work). Deliverable is a report with every tested
commit's driver hash, verdict, and crash evidence, plus the identified culprit commit's diff and
why it plausibly causes the crash (source-reading is permitted here, unlike `specs/032`, since
Rule 0 was lifted 2026-07-17).

## Technical Context

**Language/Version**: N/A for new source — this feature writes no ExecuTorch or SUMD source. The
only new artifacts this repo (`dev/executorch`) gains are this spec's own docs/scripts/results;
SUMD itself is built via its own existing tooling. Unlike `specs/032`, SUMD source *is* read once
the culprit commit is found (Rule 0 lifted), but never edited.

**Primary Dependencies**: SUMD build tooling (`uv run scripts/run.py --os android --build
--build-type release` in `/local/yanwen.xu/sumd/<sha-worktree>/`, per `sumd/CLAUDE.md`); the
already-built, already-crash-reproducing `test_coopmat_linear_bench_origcm` binary (reused as-is,
not rebuilt per commit); `git worktree`/`git bisect` for commit management; `adb`/tombstone
tooling for crash-evidence capture.

**Storage**: No new `.pte` files (this bench doesn't need one). New (non-source) artifacts: one
SHA-named SUMD worktree per tested commit under `/local/yanwen.xu/sumd/<short-sha>/` (left in
place afterward, per `sumd/CLAUDE.md` convention); this feature's own `results/bisect-report.md`,
raw `git bisect log`, and pulled tombstones for every `bad` verdict; dated driver backups under
`/sarc-c/gpusw/users/yanwen.xu/` for every non-study driver found on the shared device.

**Testing**: No automated test suite in the conventional sense — the "test" *is* the measurement
procedure (`contracts/bisect-test-script.md`), and its own correctness is checked by: the FR-001
endpoint-disagreement re-confirmation before bisecting, the FR-005 driver-hash verification
(post-flash and pre-test) before trusting any verdict, and the `cmp` staging verify before
trusting any flash.

**Target Platform**: M5 EVT1 (Exynos-family Samsung device, `sj1-dmckee-d01`, serial
`0000088f8e579c33`) — hard-locked per FR-002, no substitution permitted at any step.

**Project Type**: Measurement/bisect investigation only — no ExecuTorch source changes, no SUMD
source changes (only builds of pre-existing commits, never edits).

**Performance Goals**: N/A — this feature localizes a crash to a specific commit; it is not a
throughput study and reports no tok/s number.

**Constraints**: Must run exclusively on M5 EVT1 `0000088f8e579c33` (FR-002); must back up and
restore any non-study driver found on the device before/after every step (FR-003); must capture
crash evidence (tombstone or fallback signature) for every `bad` verdict (FR-008); single run per
commit is sufficient since the predicate is a deterministic crash/no-crash outcome, not a noisy
throughput number (spec Assumptions) — this is not a Principle IV statistical-rigor deviation,
since Principle IV's reps/stddev requirement governs reportable *numbers*, and this study reports
none.

**Scale/Scope**: 303 commits in range; ~9 expected interior bisect steps + 2 endpoint
re-confirmations (`research.md` §2) = ~11 full build+flash+test cycles nominal, more if commits
are `skip`ped or drift-recovery adds steps.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Check | Status |
|---|---|---|
| I. Correctness Before Performance | This entire study *is* a correctness investigation (a crash is the most severe correctness failure) — it doesn't just avoid conflicting with this principle, it directly serves it. No new/modified shader or dispatch code; every commit under test is an unmodified historical SUMD build. | PASS |
| II. Samsung M5 EVT1 Sole Target | This study runs entirely on M5 EVT1 — the constitution's named sole active target — unlike `specs/032` which had to justify running on the secondary M41 device. No justification needed. | PASS |
| III. Explicit Eligibility Gating, Safe Fallback | N/A — no new gating code; the crash occurs *inside* the driver's own coopmat dispatch, past ExecuTorch's own eligibility gate, which is unmodified across every commit under test. | PASS (N/A) |
| IV. Two-Tier, Statistically Sound Benchmarking | N/A — this study makes no throughput claim and reports no tok/s or GFLOP/s number; its predicate is a binary crash/no-crash outcome from re-running the existing tier-1 correctness harness (`test_coopmat_linear_bench_origcm`) named in this principle's own text, not a new benchmarking mechanism. | PASS (N/A) |
| V. Document Every Driver Workaround | N/A — no new driver workaround is authored by this feature. | PASS (N/A) |
| VI. Verify With Tools, Never Assume | Directly implemented, more strongly than most features under this constitution: every verdict comes from an actual on-device crash/tombstone or clean-completion signal (FR-007/008), never from source-level inference — and per Rule 0 being lifted, source is read only *after* the tool-verified culprit is found, to explain the finding, not to produce it. | PASS |
| VII. Clock Discipline | Implemented with the workspace's own default pin for this exact device (509/2730/663 MHz, sysfs-verified) — no override needed, unlike `specs/032`. | PASS |
| VIII. Verify Driver Before Every Coopmat Measurement | This study implements the strongest version of this principle in the workspace to date: driver hash is captured *and* backed up/restored at every single step (FR-003/FR-005), not just spot-checked — directly motivated by this session's own repeated drift incidents on this exact board. | PASS |
| IX. Never Disclose Samsung-Internal Specifics Upstream | This feature's entire output (device serial, hostname, driver SHAs/hashes, tombstones) is Samsung-internal and stays under `specs/033-sumd-coopmat-segfault-bisect/` — never proposed upstream; no upstream PR is in scope. | PASS (N/A for upstream) |
| X. Consult `instruction-for-ai` Before Acting | Build/deploy/flash mechanics are taken directly from `/local/yanwen.xu/sumd/CLAUDE.md` and this workspace's `.shared-context/instruction-for-ai/` device-access conventions, already used repeatedly this session, not re-derived. | PASS |

No violations, documented or otherwise — every principle either passes directly or is cleanly
N/A. No Complexity Tracking entries needed.

**Post-Phase-1 re-check**: `data-model.md` (Bisect Step / Culprit Commit) and
`contracts/bisect-test-script.md` (exit-code contract, backup/restore + crash-evidence-capture
clauses) introduce no new gate risk — both stay within what this Constitution Check already
covers. Still PASSES across all ten principles, zero deviations.

## Project Structure

### Documentation (this feature)

```text
specs/033-sumd-coopmat-segfault-bisect/
├── spec.md
├── plan.md                        # This file
├── research.md                    # Phase 0 output
├── data-model.md                  # Phase 1 output
├── contracts/
│   └── bisect-test-script.md      # Phase 1 output — the one CLI contract this feature defines
├── quickstart.md                  # Phase 1 output
├── checklists/
│   └── requirements.md
├── scripts/
│   └── bisect-test.sh             # NEW — implementation phase, per contracts/bisect-test-script.md
└── results/
    ├── bisect-log-raw.txt         # NEW — raw `git bisect log` output
    ├── bisect-report.md           # NEW — the deliverable (spec SC-003/SC-006)
    └── tombstones/                # NEW — one file per `bad` verdict's captured crash evidence
```

### Source Code (repository root)

No changes anywhere in `dev/executorch`'s own source tree — no shader/dispatch edits, no `.pte`
exports, no runner rebuild (the existing `test_coopmat_linear_bench_origcm` binary is reused
as-is for every step).

Outside this repo, in `/local/yanwen.xu/sumd/` (a separate, non-`dev/executorch` git checkout):

```text
/local/yanwen.xu/sumd/
├── main/                           # existing — source of the commit range, untouched otherwise
├── f14c51b6f8*/                    # existing — already-confirmed good endpoint
├── 7bb715f7cc/                     # existing — already-confirmed bad endpoint
├── <short-sha-1>/                  # NEW per interior bisect step — detached worktree, build-only
├── <short-sha-2>/                  # ...
└── ...                             # left in place afterward, per sumd/CLAUDE.md convention
```

Also outside this repo, on the M5 EVT1 adb host / NFS (driver backup/restore, per FR-003):

```text
/sarc-c/gpusw/users/yanwen.xu/
├── vulkan.samsung.so                              # documented team default (unchanged)
├── vulkan.samsung.so.<context>-backup-<date>       # NEW per non-study driver found on-device
└── sumd-deploy/<short-sha>/vulkan.samsung.so        # NEW per tested commit, staged before flash
```

**Structure Decision**: Follows this workstream's established measurement-study convention
(`specs/025`, `specs/030`, `specs/032`): `scripts/` for the one procedural script this feature
introduces, `contracts/` for that script's interface (schema-as-doc, matching `032`'s
`bisect-test-script.md` precedent), `results/` for the deliverable — plus a `results/tombstones/`
subfolder this feature adds beyond `032`'s layout, since crash evidence (not just a verdict) is
part of this study's required output (FR-008). No `src/`/`tests/` tree — there is no application
code here, only an investigation procedure and its report.
