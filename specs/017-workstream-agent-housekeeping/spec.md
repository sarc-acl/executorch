# Feature Specification: Workstream Agent Housekeeping (constitution/CLAUDE.md/gotchas consolidation)

**Feature Branch**: `017-workstream-agent-housekeeping`

**Created**: 2026-07-06

**Status**: Draft

**Input**: User description: "Before we switching to speckit, my agents were following ../claude.md and ../.shared_context , and the development has beeing super smooth. but now, in this sub folder (./) the agents that follow the speckit, is constantly hitting issues. We need to do some refactor or house keeping to make new agent in this folder more smart"

## Clarifications

### Session 2026-07-06

- Q: Should the new gotchas doc (`.specify/memory/gotchas.md`) be a one-time
  historical snapshot (just the 10 items found this session) or designed as
  a living document with an explicit append discipline for future sessions
  (mirroring `open-questions.md`'s Q-numbering pattern)? → A: living
  document -- add a maintenance discipline (a short header instruction plus
  a numbered-entry convention) so future sessions append new gotchas rather
  than letting them go unrecorded again, matching the user's own framing
  ("make new agents in this folder more smart" going forward, not just a
  one-time cleanup).

## Context (root cause, found before writing this spec)

Read directly: `./CLAUDE.md` (this folder's own file, loaded unconditionally at
the start of every session here) is **the stock, generic upstream ExecuTorch
`CLAUDE.md`** -- skills list (`/setup`, `/export`, `/building`, ...), the
`.wiki/` tribal-knowledge pointer, naming/commit/code-style conventions. It
contains **zero mention** of: `.specify/memory/constitution.md` (this
workstream's actual governing document), the M5 EVT1 / Samsung Xclipse target,
`.shared-context/` at the workspace root, or that this folder is a
speckit-driven workstream at all.

This matches the user's framing exactly: in the *other* worktrees the user
references (`../CLAUDE.md` = the workspace-root `CLAUDE.md`, `../.shared_context`),
an agent's very first file read already points at the workstream's real
operating knowledge. In *this* folder, an agent's very first file read
(`CLAUDE.md`) points at none of it -- the constitution only gets loaded
because the speckit skills themselves explicitly load it ("IF EXISTS: Load
`.specify/memory/constitution.md`"), which happens only *after* a `/speckit-*`
command runs, not before. And even once loaded, the constitution itself does
not centralize several expensive, repeatedly-rediscovered operational
gotchas from this session -- they are scattered across `specs/014`,
`specs/015`, and `specs/016`'s individual `research.md` files, which nothing
proactively points a new agent toward.

This feature is documentation/housekeeping only -- it changes no production
code, shader, or build logic. It closes the gap between "the knowledge
exists somewhere in this repo's git history" and "a new agent session finds
it before repeating the mistake."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - `CLAUDE.md` points a fresh agent at this workstream's real operating knowledge (Priority: P1)

As an agent (or human contributor) starting a task in this folder with no
prior context, the very first file I'm handed (`CLAUDE.md`) should tell me:
this folder is governed by `.specify/memory/constitution.md`, this workstream
targets Samsung M5 EVT1 specifically (not upstream ExecuTorch's general
scope), and `.shared-context/` at the workspace root holds this workspace's
build/device/driver tooling -- before I take any action based on generic
ExecuTorch assumptions.

**Why this priority**: every other fix in this feature is reachable *from*
the constitution or the new gotchas doc, but only if something points there
first. This is the single highest-leverage change -- it is the fix for the
exact failure mode the user described (a fresh agent has no idea any of this
context exists).

**Independent Test**: open `CLAUDE.md` in this folder with no other context
loaded; confirm it names the constitution, the M5 EVT1 target, and
`.shared-context/`, and tells the reader in which order/priority to consult
them, within the first few lines.

**Acceptance Scenarios**:

1. **Given** a new agent session with only `CLAUDE.md` auto-loaded, **When**
   the agent is asked to do any build/export/benchmark/dispatch-confirmation
   task in this folder, **Then** `CLAUDE.md` itself (not a file the agent has
   to discover independently) directs them to read the constitution first.
2. **Given** the same fresh session, **When** the agent needs to touch
   Android build, PTE export, or device access, **Then** `CLAUDE.md` or the
   constitution it points to names `.shared-context/instruction-for-ai/` as
   the how-to reference -- with the explicit caveat (User Story 2) that not
   every mechanism documented there exists in this repo's own source.

---

### User Story 2 - A single, authoritative doc consolidates this session's hard-won, repo-specific gotchas (Priority: P1)

As an agent picking up work in this folder, I need one place that lists the
concrete, expensive-to-rediscover mistakes already made in this repo's
history -- not scattered across `specs/014-016`'s `research.md` files, which
nothing points a new session toward -- so I don't burn hours rediscovering
the same failure mode a previous session already root-caused.

**Why this priority**: tied with User Story 1 for P1 -- the pointer from
User Story 1 is useless if the destination doc doesn't actually contain the
lessons. Each gotcha below cost real, multi-hour investigation time this
session; several were hit more than once.

**Independent Test**: for each gotcha listed in the Key Entities section
below, confirm the new doc states the symptom, the root cause, and the fix
or workaround, each citing the `specs/NNN` research.md it was originally
found in (for full narrative, not duplicated in full); separately, confirm
the doc's own header tells a future session how to append a new entry
(FR-004a) -- this is a living document, not a frozen snapshot.

**Acceptance Scenarios**:

1. **Given** the new doc exists, **When** an agent is about to run
   `cmake --build ... --target install` for the Android tree after changing
   any `backends/vulkan/` source, **Then** the doc already warns them that
   an unrelated target (`executor_runner`, stale host-arch `flatccrt.a`) can
   fail the whole `install` step silently, leaving `lib/libvulkan_backend.a`
   stale for any downstream sub-build -- before they waste a build cycle
   discovering this themselves.
2. **Given** the new doc exists, **When** an agent is about to export a
   `.pte` with buffer storage, **Then** the doc already states that
   `ET_VK_FORCE_BUFFER` (documented in `.shared-context/instruction-for-ai/export-pte.md`)
   does not exist in this repo's source, and that
   `backend.vulkan.storage_override: buffer` in `config.yaml` is the real
   mechanism -- before they silently produce a broken PTE.
3. **Given** the new doc exists, **When** an agent needs to confirm whether
   coopmat/WMMA genuinely dispatched for an e2e run, **Then** the doc already
   states that ETDump's per-event `kernel_name` field is unreliable in the
   full LLaMA graph context, and names the two independent cross-checks that
   do work (a direct wall-clock A/B against `ET_VK_FORCE_TILED_LINEAR`, and
   an isolated shader microbenchmark with its own kernel-name capture) --
   before they trust ETDump alone and reach a wrong conclusion.
4. **Given** the new doc exists, **When** an agent needs to run the SDPA
   coopmat benchmark, **Then** the doc already names `test_sdpa_coopmat_bench.cpp`
   as the correct harness and `test_coopmat_attention_bench.cpp` as the
   similarly-named but wrong one (different, unrelated shader family) --
   before they grab the wrong one by name similarity.

---

### User Story 3 - The constitution cross-references the new gotchas doc from the principles it's load-bearing for (Priority: P2)

As an agent already following the constitution's existing verification
discipline (Principle VI) or "consult docs first" discipline (Principle X),
I should be pointed at the new gotchas doc from exactly those principles,
so the connection is discoverable through governance, not just through
`CLAUDE.md`.

**Why this priority**: lower than User Stories 1-2 because the constitution
is already loaded by every speckit command (unlike `CLAUDE.md`, which is the
gap User Story 1 closes) -- this is a smaller, incremental improvement to an
already-working discovery path, not a new one.

**Independent Test**: open the constitution, confirm Principle VI (Verify
With Tools, Never Assume) and Principle X (Consult `instruction-for-ai`
Before Acting) each link to the new gotchas doc where relevant.

**Acceptance Scenarios**:

1. **Given** the constitution is loaded, **When** an agent reads Principle
   VI, **Then** it names the ETDump-per-event-kernel-name-unreliable finding
   as a concrete instance of "verify with tools, never assume" and points to
   the gotchas doc for the full list of similar findings.
2. **Given** the constitution is loaded, **When** an agent reads Principle
   X, **Then** it now explicitly warns that a `.shared-context/instruction-for-ai/`
   doc's literal command/env-var can be actively wrong for this repo
   specifically (citing the `ET_VK_FORCE_BUFFER` example), not just that the
   doc should be read first.

---

### Edge Cases

- What happens when a gotcha's underlying issue gets genuinely fixed in
  code later (e.g., someone finally root-causes the ETDump attribution bug,
  or fixes the stale-`executor_runner`-link build issue)? -- The gotchas
  doc entry MUST be updated to say "RESOLVED as of `<commit/spec>`" rather
  than deleted outright, so the historical record of what was wrong and why
  is not lost, matching how the constitution's own Sync Impact Reports
  retain superseded content instead of erasing it.
- What happens if a future gotcha is repo-specific but time-sensitive (e.g.,
  tied to a specific driver hash that will eventually be superseded)? --
  Time-sensitive/volatile facts (current driver hash, which clock values are
  pinned) stay in `.shared-context/ACTIVE-STATUS.md`/`README.md`
  §Conventions as before (Principle X); the new gotchas doc holds only
  *mechanism-level* lessons (a code path's actual behavior, a mechanism that
  doesn't exist, a build trap) that don't go stale on their own.
- What happens if `CLAUDE.md` is later overwritten by a future
  `install_executorch.sh`/tooling update that regenerates the stock
  upstream template? -- Out of scope for this feature to prevent (that is a
  tooling/process question, not a documentation-content question); flagged
  as a known risk in the new doc's own header so a future session notices
  if it recurs.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `CLAUDE.md` (this folder's root) MUST, within its first few
  lines, name `.specify/memory/constitution.md` as this folder's actual
  governing document and instruct the reader to consult it before any
  coopmat/WMMA, build, export, or device-related work.
- **FR-002**: `CLAUDE.md` MUST name the M5 EVT1 / Samsung Xclipse target and
  the workspace-root `.shared-context/` directory, consistent with what the
  constitution already says, so a reader does not need to find the
  constitution first to learn these exist.
- **FR-003**: A single new document MUST consolidate, at minimum, the
  following gotchas found during this session (each with symptom, root
  cause, fix/workaround, and a citation to the originating `specs/NNN`
  research.md for full detail):
  1. Android `cmake --build --target install` can fail on an unrelated
     target (`executor_runner`, stale host-arch `flatccrt.a`) while still
     leaving the real target (`vulkan_backend`) built but not copied to
     `lib/` -- silently staling any downstream sub-build.
  2. `ET_VK_FORCE_BUFFER` (from `.shared-context/instruction-for-ai/export-pte.md`)
     does not exist in this repo's source; the real buffer-storage-override
     mechanism is `backend.vulkan.storage_override: buffer` in `config.yaml`.
  3. A non-editable `.venv` install silently no-ops AOT/export Python code
     changes; `pip install -e . --no-build-isolation` is required.
  4. Exported `.pte` files must land directly in
     `/local/yanwen.xu/workspace/.pte_out`, never `/tmp` or a scratch dir
     (already in the constitution as of v2.3.0; cross-referenced here, not
     duplicated).
  5. `/tmp` is a small (20GB), easily-exhausted filesystem, and this
     environment's permission system denies `rm -rf` on it even for one's
     own scratch files -- use `mv` to relocate instead of deleting, or write
     scratch directly to `.artifacts/` or the job's own scratch dir from the
     start.
  6. ETDump's per-event `kernel_name` field is not reliable evidence of
     coopmat-vs-tiled dispatch in the full LLaMA graph context (confirmed
     wrong via a direct wall-clock A/B against `ET_VK_FORCE_TILED_LINEAR`
     and an independent shader microbenchmark) -- cross-check any
     ETDump-based dispatch claim before trusting it.
  7. Two similarly-named but functionally different SDPA benchmark harnesses
     exist (`test_coopmat_attention_bench.cpp` vs `test_sdpa_coopmat_bench.cpp`)
     -- only the latter exercises `SDPA.cpp`'s actual coopmat shaders.
  8. Do not conclude a CMake target "isn't wired into the build" from a
     grep that pipes through a prefilter pattern -- grep the raw file
     directly for the exact symbol first.
  9. The current production linear-coopmat shader (128x64 retune, fp16
     accumulate, flattened loop, commit `133044739`) was committed with its
     own message stating it was not yet hardware-validated; the historical
     comparison-baseline numbers (`jira-tile-sweep.md`'s 110.6/213.9/565.3)
     were measured via a different dispatch mechanism
     (`.tmp-origcm`'s `ET_VK_Q4GSW_COOPMAT_VARIANT` toggle) than this repo's
     actual production code path -- "directional" comparisons against that
     baseline are not apples-to-apples reproductions.
  10. M5 EVT1 is a shared device; a teammate may be actively using it --
      confirm with the user before assuming it is free for adb/build/flash
      work, rather than assuming a prior session's uninterrupted access
      still holds.
- **FR-004**: The new document MUST live in a location discoverable from
  both `CLAUDE.md` and the constitution (not nested inside a single
  `specs/NNN/` feature folder, since it is not scoped to one feature).
- **FR-004a**: The new document MUST be a living document, not a one-time
  snapshot: its header MUST instruct future sessions to append a new,
  numbered entry (same symptom/root-cause/fix/citation format as FR-003's
  ten) whenever a multi-hour or repeat-mistake operational issue is
  root-caused, mirroring `open-questions.md`'s Q-numbering append
  convention -- so this consolidation does not itself go stale the way the
  scattered `research.md` citations did.
- **FR-005**: The constitution's Principle VI (Verify With Tools, Never
  Assume) MUST reference the ETDump-attribution finding (FR-003 item 6) as
  a concrete example, and Principle X (Consult `instruction-for-ai` Before
  Acting) MUST be amended to warn that a cited mechanism/command from
  `.shared-context/instruction-for-ai/` can be actively wrong for this repo
  specifically (citing FR-003 item 2 as the example), per constitution
  Governance's amendment process (Sync Impact Report, version bump).
- **FR-006**: This feature MUST NOT modify any production code, shader,
  build script, or `.shared-context/` content (that directory is shared
  across worktrees and out of this feature's ownership) -- documentation
  only.

### Key Entities

- **Gotcha Entry**: one consolidated lesson. Fields: title/symptom, root
  cause, fix or workaround, originating `specs/NNN` citation, status
  (`open` / `resolved as of <ref>`).
- **CLAUDE.md Pointer Block**: the new lines added near the top of this
  folder's `CLAUDE.md`, naming the constitution, the M5 EVT1 target, and
  `.shared-context/`.
- **Constitution Cross-Reference**: the amended text within Principles VI
  and X that links to the new gotchas doc.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A reader who opens only `CLAUDE.md` (no other file) can state,
  within 30 seconds of reading, which document governs this folder's actual
  workstream rules and where this workspace's Android/device tooling lives.
- **SC-002**: All 10 gotchas listed in FR-003 are present in the new
  document, each with symptom, root cause, fix, and a citation -- verifiable
  by reading the document once, without cross-referencing `specs/014-016`.
  The document's header states the append convention for future entries
  (FR-004a), verifiable by reading the header alone.
- **SC-003**: Principle VI and Principle X of the constitution each contain
  at least one sentence pointing to the new gotchas document.
- **SC-004**: Zero production code, shader, build script, or
  `.shared-context/` files are modified by this feature (verifiable via
  `git diff --stat` showing only `CLAUDE.md`, the constitution, and the new
  doc).

## Assumptions

- The new consolidated document lives at
  `.specify/memory/gotchas.md`, alongside `constitution.md`, since it is
  workstream-wide governance-adjacent knowledge, not a single feature's
  artifact, and `.specify/memory/` is already the established home for that
  class of document in this repo.
- This feature does not attempt to fix any of the underlying issues
  themselves (the `executor_runner` stale-library build error, the ETDump
  attribution bug, etc.) -- those remain separately tracked (`open-questions.md`
  Q11, and this session's own build-workaround notes). This feature only
  ensures the *knowledge* of each issue and its workaround is easy to find
  for the next agent, not that the issue is closed.
- `.shared-context/` itself is out of scope to edit (per workspace-root
  `CLAUDE.md`'s branch/worktree discipline, it is shared across worktrees);
  this feature only adds a caveat *about* it from within this repo's own
  docs, it does not correct `.shared-context/instruction-for-ai/export-pte.md`
  itself.
- Constitution amendments in this feature are additive/clarifying (PATCH or
  MINOR per the existing semantic-versioning rule in Governance), not a
  principle redefinition -- no MAJOR version bump expected.
