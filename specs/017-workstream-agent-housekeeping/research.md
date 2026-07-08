# Research: Workstream Agent Housekeeping

## Decision 1: New gotchas doc lives at `.specify/memory/gotchas.md`, sibling to `constitution.md`

**Decision**: `.specify/memory/gotchas.md`.

**Rationale**: `.specify/memory/` is already this repo's established home
for workstream-wide (not single-feature) governance-adjacent documents --
`constitution.md` is the only other resident. A gotchas doc is exactly that
class of document: it applies across every `specs/NNN/` feature, not to
one. Placing it inside a `specs/NNN/` folder (e.g., this feature's own
`specs/017.../`) would bury it exactly the way the ten source gotchas are
already buried inside `specs/014-016`'s `research.md` files -- the
opposite of this feature's purpose.

**Alternatives considered**: A new top-level file (e.g.,
`TROUBLESHOOTING.md` at the repo root) -- rejected, adds a second
"memory"-like location alongside `.specify/memory/` for no benefit and
risks the two drifting apart on discoverability conventions. Folding the
gotchas directly into `constitution.md` as a new section -- rejected: the
constitution is a *governance* document (principles, amendment process,
Sync Impact Reports), versioned and amended deliberately; a living,
frequently-appended-to gotchas list would churn the constitution's version
number on every entry, diluting what a version bump signals. Cross-referencing
instead (FR-005) keeps each document doing one job.

## Decision 2: Numbering scheme mirrors `open-questions.md`'s `Q`-prefix -- `G1`, `G2`, ...

**Decision**: Each gotcha entry gets a stable `G<N>` identifier (`G1`
through `G10` for this feature's initial set), in the same spirit as
`.shared-context/report-for-human/open-questions.md`'s `Q1`, `Q2`, ...
convention.

**Rationale**: Reusing an already-proven, already-familiar convention from
this workspace (rather than inventing a new one) lowers the cognitive cost
for anyone who already knows how `open-questions.md` works. It also gives
each entry a stable, citable handle (`G6` = the ETDump-attribution finding)
that other docs (like this session's own spec 015 research.md) can
reference precisely, the same way `open-questions.md`'s `Q11`/`Q12` are
already cited elsewhere in this repo.

**Alternatives considered**: No numbering, just headings by title --
rejected, makes cross-referencing from other docs (constitution
Principles VI/X, future `specs/NNN` research.md files) less precise and
harder to grep for.

## Decision 3: What does NOT belong in the gotchas doc

**Decision**: The gotchas doc holds only *mechanism-level* findings (a
code path's real behavior, a build trap, a documented-but-nonexistent env
var, a naming collision between two harnesses) -- never volatile,
time-sensitive facts (current driver hash, which clocks are pinned right
now, which teammate is using the device today). Those stay in
`.shared-context/ACTIVE-STATUS.md` / `README.md` §Conventions, per
constitution Principle X, which already owns that distinction.

**Rationale**: Spec Edge Cases already draws this line explicitly. Restating
it here because it's the main risk to the doc's own long-term value (per
Clarifications' "living document" decision) -- if volatile facts leak in,
the doc rots exactly like a stale README, undermining the reason it exists.

**Alternatives considered**: One doc covering both classes of fact --
rejected, `ACTIVE-STATUS.md` already exists and owns the volatile-fact
job; duplicating that here would create two sources of truth for the same
kind of fact, the anti-pattern Principle X's "one canonical home per fact"
rule (borrowed from `.shared-context/instruction-for-ai/README.md`'s own
scope rules) already warns against.

## Decision 4: `CLAUDE.md` gets a small addition, not a rewrite

**Decision**: Add a short block near the top of `./CLAUDE.md` (after the
`# ExecuTorch` heading, before `## Skills`) naming the constitution, M5
EVT1, and `.shared-context/` -- leave the rest of the file (skills list,
`.wiki/` pointer, naming/commit/code-style conventions) untouched.

**Rationale**: The existing content is still accurate, generic ExecuTorch
guidance that applies to this checkout as much as any other -- the bug
isn't that it's wrong, it's that it's *silent* about this workstream's
specific governance. A full rewrite risks losing content another part of
this repo (or a future `install_executorch.sh` re-sync) still depends on,
and is unnecessary to close the gap FR-001/002 describe.

**Alternatives considered**: Replacing `CLAUDE.md` entirely with
workstream-specific content -- rejected, would strip out genuinely useful
generic guidance (naming conventions, code style, the `/executorch-kb`
skill) that has nothing to do with this workstream's gap.

## Decision 5: Constitution amendment scope -- Principles VI and X only, MINOR version bump

**Decision**: Amend Principle VI (add the ETDump-attribution example) and
Principle X (add the "a cited doc's mechanism can be wrong for this repo"
caveat), each with one to two added sentences, plus a `## Gotchas
Reference` pointer somewhere in Development Workflow linking to the new
doc. Bump the constitution's version by MINOR (per its own Governance
rule: "principle or section added/materially expanded"), with a Sync
Impact Report prepended per its established pattern.

**Rationale**: These are the two principles the spec's User Story 3
identified as load-bearing for the gap (verification discipline; consult-docs-first
discipline). Touching every principle would be scope creep beyond what
FR-005 asks for.

**Alternatives considered**: A brand-new Principle XI ("Consult the
gotchas doc first") -- rejected as redundant with Principle X, which
already establishes the "consult the right doc before acting" pattern;
extending X is more consistent than duplicating it in a new principle.
