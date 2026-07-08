# Quickstart: Workstream Agent Housekeeping

No device, no build, no code execution -- this is a documentation feature.
Validation is reading the three touched/created files and checking the
concrete things below.

## 1. `CLAUDE.md` cold-read check (SC-001)

```bash
head -20 CLAUDE.md
```

Confirm, without reading any other file: it names
`.specify/memory/constitution.md` as this folder's governing document, the
M5 EVT1 / Samsung Xclipse target, and `.shared-context/` at the workspace
root -- and (per Decision 4) does not otherwise disturb the existing
skills/naming/commit/code-style content below it.

## 2. Gotchas doc completeness check (SC-002)

```bash
grep -c '^### G' .specify/memory/gotchas.md   # expect 10
grep -n '^## ' .specify/memory/gotchas.md | head -3   # header section present first
```

Open the file once; confirm each `G1`-`G10` entry has symptom, root cause,
fix/workaround, and a citation (per `data-model.md`'s Gotcha Entry schema),
and that the header explains the append convention for future entries
(FR-004a) -- readable without opening `specs/014-016`.

## 3. Constitution cross-reference check (SC-003)

```bash
grep -n "gotchas.md" .specify/memory/constitution.md
```

Expect at least two hits: one inside Principle VI's text, one inside
Principle X's text (per `data-model.md`'s Constitution Cross-Reference
table) -- plus the Sync Impact Report entry documenting this amendment and
the version bump.

## 4. Scope check -- nothing else touched (SC-004)

```bash
git status --short
git diff --stat
```

Expect exactly three files: `CLAUDE.md` (modified), `.specify/memory/constitution.md`
(modified), `.specify/memory/gotchas.md` (new) -- plus this feature's own
`specs/017-workstream-agent-housekeeping/` artifacts. No production code,
shader, build script, or `.shared-context/` file appears in the diff.

## Expected outcome

A reader who has never seen this workstream before, starting from
`CLAUDE.md` alone, can reach the constitution and the gotchas doc within
two hops, and the ten gotchas save them from repeating any of this
session's ten most expensive mistakes.
