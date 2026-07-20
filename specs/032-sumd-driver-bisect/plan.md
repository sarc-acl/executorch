# Implementation Plan: SUMD Driver Bisect for the 8da4w-Slower-Than-4w Regression

**Branch**: `032-sumd-driver-bisect` | **Date**: 2026-07-16 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/032-sumd-driver-bisect/spec.md`

## Summary

On M41, swapping M5 EVT1's known-good SUMD driver (`f14c51b6f8`) in for M41's own native driver
flips the release/1.3-vanilla 4w-vs-8da4w prefill ordering from the expected "8da4w faster" to an
inverted "8da4w ~30% slower" (spec Context). This feature git-bisects SUMD's `main` branch between
the commit nearest 2024-11-01 (`898709039d1`) and the commit nearest 2026-03-31 (`ec3958eae55`) —
3,055 commits, ~12 expected bisect steps — to find the single commit that changed this ordering.
Each step: build the SUMD driver from that commit (no source reading, per `sumd/CLAUDE.md` Rule
0), flash to M41 (hard-locked device, serial `00000a34cdd4abd3`), pin clocks to M41's own max
(980/5333/800 MHz, sysfs-verified), and run the release/1.3 vanilla `llama_main_rel1.3` runner
(Llama 3.2 1B, 2048-token prefill, 1 rep, 4w and 8da4w) — verdict is a strict tok/s comparison, no
tie-break (per Clarifications). Deliverable is a single report with every tested commit's driver
version, numbers, and verdict, plus the identified culprit commit.

## Technical Context

**Language/Version**: N/A for new source — this feature writes no ExecuTorch or SUMD source. The
only new artifact this repo (`dev/executorch`) gains is this spec's own docs/scripts/results; SUMD
itself is built via its own existing tooling and never read/edited (Rule 0).

**Primary Dependencies**: SUMD build tooling (`uv run scripts/run.py --os android --build
--build-type release` in `/local/yanwen.xu/sumd/<sha-worktree>/`, per `sumd/CLAUDE.md`); the
already-built `llama_main_rel1.3` runner (plain release/1.3, no node-threshold); the NFS run-kit's
already-staged 1B 4w/8da4w PTEs; `git worktree`/`git bisect` for commit management.

**Storage**: No new `.pte` files. New (non-source) artifacts: one SHA-named SUMD worktree per
tested commit under `/local/yanwen.xu/sumd/<short-sha>/` (left in place afterward, per
`sumd/CLAUDE.md` convention — never cleaned up automatically), and this feature's own
`results/bisect-report.md` + raw `git bisect log`.

**Testing**: No automated test suite in the conventional sense — the "test" *is* the measurement
procedure (`contracts/bisect-test-script.md`), and its own correctness is checked by: the
FR-002 endpoint-disagreement precondition before bisecting, the FR-006 sysfs clock-pin readback
before trusting any measurement, and the `cmp` staging verify before trusting any flash.

**Target Platform**: M41 (Exynos-family Samsung device, `xgpusw-debug07`, serial
`00000a34cdd4abd3`) — hard-locked per FR-003, no substitution permitted at any step.

**Project Type**: Measurement/bisect investigation only — no ExecuTorch source changes, no SUMD
source changes (only builds of pre-existing commits, never edits).

**Performance Goals**: N/A — this feature localizes a *direction change* (which quant mode is
faster) to a specific commit; it does not target a throughput number of its own, and per spec
Assumptions is not compared against Samsung M5 EVT1 headline numbers.

**Constraints**: Must never read/review/modify SUMD driver source (`sumd/CLAUDE.md` Rule 0); must
run exclusively on M41 `00000a34cdd4abd3` (FR-003); must pin to M41's own max clocks, not the
workspace default (FR-006); 1 rep per commit/quant-mode with a strict, no-tie-break verdict
(Clarifications) — a deliberate reduction from this workstream's usual 3-rep/CoV rigor, justified
below (Complexity Tracking) as a screening bisect, not a final performance report.

**Scale/Scope**: 3,055 commits in range; ~12 expected interior bisect steps + 2 endpoint checks
(`research.md` §2) = ~14 full build+flash+measure cycles nominal, more if commits are `skip`ped.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Check | Status |
|---|---|---|
| I. Correctness Before Performance | N/A — no new/modified shader or dispatch code; every commit under test is an unmodified historical SUMD build. | PASS (N/A) |
| II. Samsung M5 EVT1 Sole Target | This study runs entirely on M41 by explicit instruction (FR-003), not M5 EVT1. Constitution names M41 as legitimate for "fast non-target-critical iteration" — this goes further than casual iteration, but the goal (isolating a driver commit via cross-flash evidence, spec Context) is exactly the kind of secondary-device investigation that framing anticipates; the culprit's real-world impact on M5 EVT1 itself is explicitly out of scope/follow-up (spec Success Criteria don't claim M5 EVT1 relevance). | PASS |
| III. Explicit Eligibility Gating, Safe Fallback | N/A — release/1.3 vanilla predates any coopmat eligibility gate; no new gating code. | PASS (N/A) |
| IV. Two-Tier, Statistically Sound Benchmarking | Deviation: 1 rep, no stddev/CoV, no shader-microbench tier at any bisect step. Deliberate and spec-directed (Clarifications: strict comparison, no tie-break), justified by bisect step count (~14 cycles, each a full driver build+flash+2 runs) making 3-rep-per-step impractical; this is a screening tool to localize a candidate commit, not a final headline number — see Complexity Tracking. | PASS (documented deviation) |
| V. Document Every Driver Workaround | N/A — no new driver workaround is authored by this feature; any workaround already present at a given historical commit is inherited unmodified, not written here. | PASS (N/A) |
| VI. Verify With Tools, Never Assume | Directly implemented: every verdict comes from an actual on-device runner measurement (FR-007/008), never from source-level inference (which is off-limits anyway per Rule 0) or eligibility-check reasoning. | PASS |
| VII. Clock Discipline | Implemented with a spec-directed deviation from the *workspace default* pin target: M41's own max (980/5333/800) instead of 509/2730/663, sysfs-verified before every measurement (FR-006) — same "verified bound" discipline this principle requires, applied to a different, explicitly-labeled pin target. | PASS |
| VIII. Verify Driver Before Every Coopmat Measurement | Workload is release/1.3 vanilla (no coopmat) — technically N/A — but this feature implements an even stronger version of the same discipline: driver version is captured and recorded for *every* step (FR-005), not just spot-checked. | PASS (N/A, exceeded) |
| IX. Never Disclose Samsung-Internal Specifics Upstream | This feature's entire output (device serial, hostname, driver SHAs/version strings) is Samsung-internal and stays under `specs/032-sumd-driver-bisect/` — never proposed upstream; no upstream PR is in scope for this feature. | PASS (N/A for upstream) |
| X. Consult `instruction-for-ai` Before Acting | Build/deploy mechanics are taken directly from `/local/yanwen.xu/sumd/CLAUDE.md` (already read) and this workspace's `.shared-context/instruction-for-ai/` device-access conventions, not re-derived. | PASS |

No unjustified violations; one documented deviation (Principle IV) — see Complexity Tracking.

**Post-Phase-1 re-check**: `data-model.md` (Bisect Step / Culprit Commit) and
`contracts/bisect-test-script.md` (exit-code contract, Rule-0 "never performed" clause) introduce
no new gate risk — both stay within what this Constitution Check already justified. Still PASSES
across all ten principles, same single documented deviation.

## Project Structure

### Documentation (this feature)

```text
specs/032-sumd-driver-bisect/
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
    └── bisect-report.md           # NEW — the deliverable (spec SC-003/SC-005)
```

### Source Code (repository root)

No changes anywhere in `dev/executorch`'s own source tree — no shader/dispatch edits, no new
`.pte` exports (existing 1B 4w/8da4w PTEs are reused), no runner rebuild beyond what already
exists for release/1.3.

Outside this repo, in `/local/yanwen.xu/sumd/` (a separate, non-`dev/executorch` git checkout):

```text
/local/yanwen.xu/sumd/
├── main/                           # existing — source of the commit range, untouched otherwise
├── <short-sha-1>/                  # NEW per tested commit — detached worktree, build-only
├── <short-sha-2>/                  # ...
└── ...                             # left in place afterward, per sumd/CLAUDE.md convention
```

No SUMD source file is opened, edited, or reviewed in any of these worktrees — only built and
flashed (Rule 0).

**Structure Decision**: Follows this workstream's established measurement-study convention
(`specs/025`, `specs/030`): `scripts/` for the one procedural script this feature introduces,
`contracts/` for that script's interface (schema-as-doc, matching `025`'s `sweep-report-schema.md`
precedent), `results/` for the deliverable. No `src/`/`tests/` tree — there is no application code
here, only an investigation procedure and its report.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|---|---|---|
| 1 rep per commit/quant-mode, no CoV/stddev, no shader-microbench tier (deviates from Principle IV) | A full 3-rep+CoV protocol at every one of ~14 bisect steps (each already a multi-minute driver build + flash + 2 runs) would multiply this investigation's wall-clock cost several-fold, almost entirely in service of steps that exist only to *localize* the culprit commit, not to report a final number | `specs/024-8da4w-slower-than-4w` already established precedent for this workstream: a single-rep result is sufficient to establish *direction* (which quant mode is faster), with full statistical rigor explicitly deferred as follow-up (that spec's SC-003 is marked "not yet met... out of scope for this pass"). This feature follows the same precedent, additionally mitigated by the strict/no-tie-break comparison rule adopted in this spec's Clarifications (removes the one place a single-rep noise band could otherwise flip a verdict silently) |
