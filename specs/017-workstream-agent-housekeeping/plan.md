# Implementation Plan: Workstream Agent Housekeeping

**Branch**: `017-workstream-agent-housekeeping` | **Date**: 2026-07-06 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/017-workstream-agent-housekeeping/spec.md`

## Summary

Close the gap the user identified: a fresh agent session in this folder has
no path to this workstream's real operating knowledge until *after* a
`/speckit-*` command loads the constitution -- and even then, ten
expensive, already-root-caused operational gotchas from this session stay
scattered across `specs/014-016`'s `research.md` files with nothing
pointing a new session toward them. This is a documentation-only feature:
(1) add a short pointer block near the top of this folder's `CLAUDE.md`
naming the constitution, the M5 EVT1 target, and `.shared-context/`; (2)
create `.specify/memory/gotchas.md`, a living, append-as-you-go document
consolidating the ten gotchas (each with symptom/root cause/fix/citation)
plus a header instructing future sessions how to add to it; (3) amend
constitution Principles VI and X to cross-reference the new doc. No
production code, shader, build script, or `.shared-context/` file changes.

## Technical Context

**Language/Version**: Markdown documentation only -- no code changes. The
constitution amendment follows its own existing versioning convention
(semantic-versioning-style Sync Impact Report, per its Governance section).

**Primary Dependencies**:
- `./CLAUDE.md` (this folder's root) -- edited in place, minimal addition
  near the top; the rest of the stock upstream content is left untouched
  since it is still valid generic ExecuTorch guidance.
- `.specify/memory/constitution.md` -- amended (Principles VI, X;
  Governance's "ten principles" count and Sync Impact Report), version
  bumped per its own semantic-versioning rule (additive/clarifying =
  MINOR, per Assumptions).
- `.specify/memory/gotchas.md` -- new file.
- Source citations for the ten gotchas: `specs/014-m5-linear-coopmat-retune`,
  `specs/015-m5-e2e-wmma-validation/research.md` (Decisions 6, 7 and its
  reversal), `specs/016-m5-linear-sdpa-microbench` (clarify-session
  findings), this session's own build-workaround narrative (no dedicated
  spec, captured directly from this conversation).

**Storage**: Flat files -- no database, no schema. `.specify/memory/gotchas.md`
is a single markdown file with numbered entries (`G1`, `G2`, ... mirroring
`open-questions.md`'s `Q`-numbering, per Clarifications).

**Testing**: No automated test suite -- verification is the SC-001..SC-004
manual read-through checks already defined in the spec (open `CLAUDE.md`
cold; read the new doc once; grep the constitution for the two new
cross-references; `git diff --stat` shows only the three touched files).

**Target Platform**: N/A (documentation, not a runtime artifact). The
*content* of the new doc concerns the M5 EVT1 Android target, but the
feature itself has no target platform of its own.

**Project Type**: Documentation/housekeeping -- no source tree changes.

**Performance Goals**: N/A.

**Constraints**: FR-006 -- must not touch production code, shaders, build
scripts, or `.shared-context/` (shared across worktrees, out of this
feature's ownership per workspace-root `CLAUDE.md`'s branch discipline).

**Scale/Scope**: Three files touched (`CLAUDE.md`, `constitution.md`, new
`gotchas.md`); ten initial gotcha entries plus the append-convention header.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Check | Status |
|---|---|---|
| I. Correctness Before Performance | N/A -- no shader/code correctness claim made by this feature. | PASS (N/A) |
| II. Samsung M5 EVT1 Sole Target | Not altered -- the new doc documents M5 EVT1-specific gotchas, doesn't change the target. | PASS |
| III. Explicit Eligibility Gating | N/A -- no new gating code. | PASS (N/A) |
| IV. Two-Tier Benchmarking | N/A -- no benchmark produced by this feature. | PASS (N/A) |
| V. Document Driver Workarounds | This feature's gotchas doc is itself an extension of this principle's spirit (document workarounds at the point future agents will look) -- complementary, not conflicting. | PASS |
| VI. Verify With Tools, Never Assume | Amended (FR-005) to cite the ETDump-attribution finding as a concrete example -- strengthens this principle's teeth rather than weakening it. | PASS |
| VII. Clock Discipline | Not touched. | PASS (N/A) |
| VIII. Verify Driver Before Every Coopmat Measurement | Not touched. | PASS (N/A) |
| IX. Never Disclose Samsung-Internal Specifics Upstream | N/A -- `gotchas.md` lives in `.specify/memory/`, internal-only and never upstream-bound, same as `constitution.md` itself. `CLAUDE.md`'s new pointer block also names "M5 EVT1" but, like `specs/`/`.specify/`, is dev-tooling excluded from upstream PR curation by convention (Repository & Distribution Scope) -- not itself a file ever cherry-picked into a `pytorch/executorch` PR. IX only constrains content actually proposed for the public repo. | PASS (N/A) |
| X. Consult `instruction-for-ai` Before Acting | Amended (FR-005) to add the caveat that a cited `.shared-context/instruction-for-ai/` mechanism can be wrong for this repo specifically -- strengthens, doesn't contradict, the "read it first" rule. | PASS |

No violations; Complexity Tracking not needed.

**Post-Phase-1 re-check**: `data-model.md`/`quickstart.md` introduced no
new gate risk -- the Gotcha Entry schema and the two constitution
cross-reference points stayed within the scope Phase 0 already justified.
Principle IX's "never disclose upstream" rule doesn't constrain this
feature at all: `gotchas.md` lives in `.specify/memory/`, the same
internal-only, never-upstream-bound location as `constitution.md` itself
(Repository & Distribution Scope already treats this whole class of
artifact as safe for the `sarc-acl` fork without curation) -- so freely
naming "M5 EVT1" in a gotcha entry (as the constitution itself already
does throughout) raises no IX concern; IX only binds content actually
proposed for the public `pytorch/executorch` repository, which this
feature never touches. Constitution Check still PASSES across all ten
principles.

## Project Structure

### Documentation (this feature)

```text
specs/017-workstream-agent-housekeeping/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md         # Phase 1 output
├── quickstart.md         # Phase 1 output
└── tasks.md              # Phase 2 output (/speckit-tasks, not this command)
```

No `contracts/` -- this feature has no external interface (API, CLI, data
format) to document a contract for; it edits three markdown files directly.

### Source Code (repository root)

```text
CLAUDE.md                        # MODIFIED: pointer block added near the top
.specify/memory/
├── constitution.md              # MODIFIED: Principles VI, X amended; version bumped; Sync Impact Report prepended
└── gotchas.md                   # NEW: living gotchas doc, 10 initial entries + append-convention header
```

**Structure Decision**: No new directories. Two existing files edited in
place (`CLAUDE.md`, `constitution.md`), one new file created
(`.specify/memory/gotchas.md`), consistent with FR-004's requirement that
the new doc be discoverable from both.

## Complexity Tracking

*No violations -- table not needed.*
