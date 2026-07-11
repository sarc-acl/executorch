# Feature Specification: MiniPC RDNA3 Handoff Report

**Feature Branch**: `013-minipc-handoff-report`

**Created**: 2026-07-05

**Status**: Draft

**Input**: User description: "prepare a full report for handoff, we concluded our study on this minipc RDNA3 iGPU, from now on i will move the Samsung phones, so prepare the full report. And then I will clone this repo on the system that is capable to run adb with the target samsung phone"

## Clarifications

### Session 2026-07-05

- Q: Does the handoff need to transfer the actual (multi-GB, gitignored) `.pte` export files, or just the export recipe used to produce them? → A: Neither, in the sense this feature needs to handle -- the user will independently re-export `.pte` files on the Samsung/Xclipse machine themselves. The report only needs to note this is expected, not detail export mechanics or arrange file transfer.
- Q: (general guidance, applies across the whole report) → A: The report and runbook should be high-level guidance, not an exhaustive, command-by-command script -- reasonable defaults and pointers are sufficient; over-specifying untested Android/`adb` commands would overstate confidence this feature doesn't have anyway (no device access to verify them).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - One consolidated findings report, not twelve spec folders (Priority: P1) 🎯 MVP

As the contributor closing out the `rocky-ryzen` MiniPC phase of this
workstream and about to move to Samsung/Xclipse hardware, I need one
document that synthesizes everything concluded across every prior feature
(`001` through `012`) -- what was tried, what won, what regressed, what
was ruled out and why -- so that neither I nor anyone else has to re-read
twelve separate spec folders to understand the current state of
knowledge before starting on the new device.

**Why this priority**: Without this, the single most valuable output of
this entire MiniPC phase (a coherent, load-bearing understanding of what
WMMA does and doesn't help, and why) stays scattered across a dozen
folders and this conversation's history -- exactly the kind of knowledge
loss a handoff exists to prevent.

**Independent Test**: Can be fully tested by having someone who has not
read any of the twelve prior specs read only this report and correctly
answer: which quantization schemes benefit from linear coopmat and by how
much, whether SDPA coopmat helps at the e2e level, whether decode is worth
accelerating with WMMA, and what open decisions remain.

**Acceptance Scenarios**:

1. **Given** the twelve prior features' spec folders and results, **When**
   the consolidated report is produced, **Then** it states, for each major
   finding (linear coopmat by scheme, SDPA coopmat prefill, decode
   feasibility), the headline number, which feature produced it, and
   whether it was tier-1 (microbenchmark) or tier-2 (e2e) validated.
2. **Given** the report, **When** a reader checks any headline number
   against its cited source feature's own results file, **Then** the
   numbers match exactly -- no re-derivation, no rounding drift.
3. **Given** the report, **When** a reader looks for what's still
   undecided, **Then** the deferred `8da4w` production-gating decision and
   the not-yet-attempted decode-SDPA/batching/speculative-decoding
   follow-ons are named explicitly, not omitted because they're
   unresolved.

---

### User Story 2 - Repo state and prerequisites, so a clone elsewhere isn't a surprise (Priority: P2)

As the contributor who will clone this repo on a different machine (one
with `adb` access to the target Samsung phone, since this MiniPC session's
environment does not have that), I need to know exactly what this repo's
current state is -- what's committed, what's only in the working tree, and
what has to happen before a clone elsewhere would actually contain this
work -- so the move to the new machine doesn't silently lose anything.

**Why this priority**: Every finding in this workstream lives in
uncommitted working-tree changes (per this repository's own convention:
commit only when explicitly asked). A clone of the current remote branch,
today, would not contain any of it. This must be surfaced as an explicit,
named decision point, not silently assumed one way or the other.

**Independent Test**: Can be fully tested by running `git status` and
`git log` and confirming the report's description of the repo's state
matches exactly, and that the report names committing/pushing as a
prerequisite decision the user must explicitly authorize, not something
this feature performs on its own.

**Acceptance Scenarios**:

1. **Given** the repo's actual current state (branch, uncommitted files,
   commit history), **When** the report is produced, **Then** it
   accurately describes what would and would not be present in a fresh
   clone today.
2. **Given** that state, **When** the report is read, **Then** it states
   committing/pushing the working-tree changes as a required prerequisite
   for a useful clone, without performing that commit/push itself (per
   this repo's own commit-only-when-asked convention).

---

### User Story 3 - A starting runbook for the Samsung/Xclipse machine (Priority: P3)

As the contributor about to work on a Samsung/Xclipse device via `adb`
instead of direct local execution, I need a starting checklist of what
differs from this workstream's `rocky-ryzen`-based methodology (build
target, deployment mechanism, and the first things to check on genuinely
new hardware) -- not a fully worked-out Android build pipeline (this
feature has no access to the target device to develop and test one), but
enough to start from instead of a blank page.

**Why this priority**: Every prior feature's build/run/capture commands in
this workstream assume a local Linux desktop process this session can
execute directly. None of that transfers as-is to an `adb`-deployed
Android target. Without at least a starting checklist, the next session
has to rediscover from scratch which parts of the existing methodology
(the two-tier benchmarking discipline itself, the ETDump dispatch
-confirmation habit, the ETDump JSON schema) still apply unchanged, and
which parts (the exact build commands, how a capture's output gets back
off-device) need adapting.

**Independent Test**: Can be fully tested by confirming the runbook names,
for each step of this workstream's established methodology (export,
build, dispatch-confirm, benchmark, report), whether it's expected to
carry over unchanged, need adaptation, or be newly established on the
Samsung/Xclipse device -- without asserting exact untested commands as if
already proven to work on that hardware.

**Acceptance Scenarios**:

1. **Given** this workstream's established two-tier benchmarking
   discipline (constitution Principle IV) and dispatch-confirmation habit
   (Principle VI), **When** the runbook is produced, **Then** it states
   both carry over unchanged to Samsung/Xclipse -- these are methodology,
   not `rocky-ryzen`-specific mechanics.
2. **Given** this workstream's `rocky-ryzen`-specific build/deploy
   commands, **When** the runbook is produced, **Then** it names each one
   that needs adaptation for an `adb`-deployed Android target, without
   asserting a specific untested replacement command as proven.
3. **Given** the runbook, **When** it lists what to check first on the new
   device, **Then** it includes confirming Xclipse's cooperative-matrix
   support and tile dimensions before assuming this workstream's RDNA3
   findings transfer unchanged (constitution Principle II: Samsung/Xclipse
   is the real target, not assumed identical to the MiniPC proxy).

### Edge Cases

- What happens if a headline number in the consolidated report (User Story
  1) doesn't match its source feature's results file when double-checked?
  This is a defect in the report, to be corrected before the report is
  considered complete -- not a discrepancy to silently average away.
- What happens if the user decides NOT to commit/push the working-tree
  changes before cloning elsewhere? Then the report's own User Story 2
  section already states plainly that a clone today would not contain this
  work -- the decision is the user's, this feature only makes the
  consequence visible in advance.
- What happens if Xclipse turns out not to support cooperative matrix at
  all, or with different tile dimensions than RDNA3's 16x16x16? The
  runbook's own User Story 3 already names this as the first thing to
  check, precisely because this workstream's RDNA3 findings are not
  assumed to transfer automatically.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST produce one consolidated report synthesizing
  every major finding from specs `001` through `012`, citing which feature
  produced each headline number and whether it was tier-1 or tier-2
  validated.
- **FR-002**: Every headline number in the consolidated report MUST match
  its cited source feature's own results file exactly.
- **FR-003**: The report MUST explicitly name the currently-open items:
  the deferred `8da4w` production-gating decision, decode SDPA's
  unverified-but-expected-identical conclusion, the not-attempted
  batching/speculative-decoding and further-quantization directions, and
  Samsung/Xclipse validation itself as still-pending (every finding to
  date is `rocky-ryzen` MiniPC-only).
- **FR-004**: The report MUST accurately describe the repo's current state
  (branch, what's committed vs. working-tree-only) and state that
  committing/pushing is a required prerequisite for a clone elsewhere to
  contain this work -- without performing that commit/push itself. The
  `.pte` export files themselves are out of scope for this prerequisite
  (Clarifications): they are gitignored, and the user will re-export them
  independently on the Samsung/Xclipse machine -- the report only needs to
  note this expectation, not detail export mechanics or arrange transfer.
- **FR-005**: The report MUST include a runbook section for the
  Samsung/Xclipse handoff that states, for each step of this workstream's
  established methodology, whether it carries over unchanged, needs
  adaptation, or must be newly established on the new device. Per
  Clarifications, this is high-level guidance and pointers, not an
  exhaustive command-by-command script -- untested Android/`adb`-specific
  commands are not asserted as proven.
- **FR-006**: The runbook MUST name confirming Xclipse's cooperative
  -matrix support and tile dimensions as the first thing to check on the
  new device, ahead of assuming any RDNA3 finding transfers unchanged.

### Key Entities

- **Consolidated Finding**: one entry per major result (e.g. "linear
  coopmat, `4w`, e2e"), its headline number, source feature, and tier
  (microbenchmark vs. e2e).
- **Repo Handoff State**: the branch name, the list of files with
  uncommitted changes, and the explicit prerequisite (commit/push) needed
  before a clone elsewhere would contain this work.
- **Samsung/Xclipse Runbook Item**: one step of the established
  methodology (export, build, dispatch-confirm, benchmark, report), its
  carries-over/needs-adaptation/newly-established status, and what to
  check first if newly established.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A reader unfamiliar with all twelve prior features can
  correctly state, from the report alone, which quantization schemes
  benefit from linear coopmat, whether SDPA coopmat helps at the e2e
  level, and whether decode is worth accelerating with WMMA.
- **SC-002**: Every headline number in the report is verified to match its
  cited source feature's results file exactly.
- **SC-003**: The report states plainly whether a clone of this repo,
  today, would contain this workstream's findings, and what must happen
  first if not.
- **SC-004**: A reader can determine, for every step of the established
  methodology, whether it's expected to carry over to Samsung/Xclipse
  unchanged, need adaptation, or be newly established -- without treating
  any of it as already proven on that hardware.

## Assumptions

- Scope is a report and runbook only -- no actual Android/`adb` build,
  deployment, or Samsung/Xclipse device testing is performed in this
  feature, since this session's environment has no access to that device
  (User Story 3's own framing: a starting checklist, not a worked
  -out-and-tested pipeline).
- Whether to commit and push the current working-tree changes is treated
  as an explicit, named prerequisite decision for the user to make
  (FR-004), not something this feature performs on its own -- consistent
  with this repository's own "commit only when explicitly asked"
  convention.
- The report consolidates specs `001` through `012` (everything completed
  in this workstream to date); it does not re-open or re-validate any of
  their individual findings, only cites them (FR-002).
- The `.pte` export files are gitignored and not part of this feature's
  handoff concern (Clarifications) -- the user will re-export them
  independently on the Samsung/Xclipse machine, likely with a different
  export configuration anyway (mobile/Android target vs. this
  workstream's x86 desktop builds), so the existing files wouldn't
  transfer usefully even if included.
- The report and runbook are high-level guidance, not an exhaustive,
  command-by-command script (Clarifications) -- reasonable defaults and
  pointers to this workstream's existing quickstart.md files are
  sufficient; this feature does not have Android/`adb` device access to
  develop or verify exact replacement commands, so it doesn't assert any.
- Per this workstream's own precedent (`Repository & Distribution Scope`
  discussion earlier in this session), this report and runbook belong in
  the `sarc-acl/executorch` fork's `specs/` tree -- no separate decision
  about what, if anything, goes upstream to `pytorch/executorch` is made
  by this feature.
