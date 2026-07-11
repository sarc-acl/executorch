# Phase 0 Research: Re-Open SUBGROUP_SIZE=32 in the 8da4w CoopMat Sweep

No `NEEDS CLARIFICATION` items remain from the plan's Technical Context — this feature
reuses `025`'s already-proven autotune methodology and `dbuf2` result, and this session's
own on-device re-verification (not a hypothetical) resolves the questions that would
otherwise be open.

## Decision 1: SUBGROUP_SIZE is re-opened as a real axis, not re-excluded by assumption

**Decision**: Re-derive the legal `8da4w` tile/subgroup/subgroup-size space with
`SUBGROUP_SIZE ∈ {32, 64}` as a swept variable, rather than inheriting `025`'s
`SUBGROUP_SIZE: 64`-only assumption.

**Evidence this session already gathered** (not re-derived from scratch — Principle VI,
verify with tools):

- The shipped shader's exclusion of `SUBGROUP_SIZE=32` rests on a header comment describing
  a `vkCreateComputePipelines` crash. `025`'s own T014 probe (one tile shape, one correctness
  shape: `M=K=N=128`) already found that crash does not reproduce on the current driver
  (`c9861e9906d03fa2c7d48b804e1a1c80` / `f14c51b6f8`).
- This session independently rebuilt and re-ran that exact probe on a **second** M5 EVT1
  board (`xgpusw-debug08`, distinct from whichever board `025` used) and found: (a) the
  pipeline still does not crash at the same tile shape, confirming T014 was not a one-off;
  (b) correctness now fails at 3 additional shapes T014 never tested
  (`M=256,K=256,N=256` Buffer; `M=256,K=128,N=128` Buffer; `M=256,K=128,N=64` Buffer — all
  pass at subgroup=64 on the identical binary); (c) at `M=2048`, the subgroup=32 probe
  measured ~1095–1169 GFLOP/s, below both the shipped subgroup=64 baseline (~1688 GFLOP/s)
  and `025`'s actual winner (1736 GFLOP/s).

**Rationale**: The crash-based exclusion is stale evidence (T014 already showed this), but
`025` deliberately did not act on that by re-opening its own already-computed search — the
right response is a dedicated feature, which is this one. This session's fuller probe result
shows the correct action is not "flip 32 on and re-run 025's process unchanged" either: a
single additional data point still isn't a swept search, and it already surfaces exactly the
failure mode a single-shape check misses (shape-dependent silent miscompute). The only
defensible path is to actually re-derive the space and correctness-gate it broadly (Decision
2), then let the performance numbers speak (User Story 3) — not to assume the outcome in
either direction from two probes.

**Alternatives considered**:
- *Treat this session's probe as sufficient and simply document "32 is worse, close the
  axis"*: rejected — one tile shape (the shipped `128×64/K32/2×2`) is not the whole space;
  `025` itself found the tile-shape optimum for subgroup=64 is *not* the shipped shape
  (`128×32/K16/1×2` won instead). A different tile shape at subgroup=32 could plausibly avoid
  whatever causes the `M=256` correctness failures — that has to be checked, not assumed
  either way.
- *Re-run `025`'s exact 542-candidate space at subgroup=64 unchanged plus a parallel
  542-candidate space at subgroup=32*: rejected as unnecessarily expensive — Decision 2
  folds subgroup_size into the same enumeration/pruning pass instead of doubling the process
  wholesale, and `025`'s winner is reused as a fixed subgroup=64 anchor rather than re-derived.

## Decision 2: Correctness gate broadens from one shape to the full representative set

**Decision**: Every subgroup=32 candidate that compiles is correctness-checked against the
same multi-shape representative set already used elsewhere in this workstream for
`8da4w`/`4w` sweeps (the small-shape harness matrix `test_coopmat_linear_bench`'s
`COOPMAT_BENCH_CORRECTNESS_ONLY=1` mode already runs, spanning multiple `M`/`K`/`N`
combinations) — not the single `M=K=N=128` shape T014 and this session's initial probe each
happened to use. A candidate's correctness verdict is reported per-shape; "passes at shape X"
and "correct" are not the same claim.

**Rationale**: This is the specific, concrete gap this feature exists to close — stated
directly in the spec's Context and User Story 2. Two independent single-shape checks (T014,
this session's probe) each missed the `M=256` failure mode by chance of which shape they
picked. A sweep that repeats that same one-shape pattern at a different tile geometry would
not actually be new evidence.

**Rationale for reusing the existing harness rather than writing a new one**: the harness
already runs a shape matrix (evidenced by this session's own log output showing
`M=64/128/256`-family cases in a single `COOPMAT_BENCH_CORRECTNESS_ONLY=1` invocation); no
new correctness methodology is needed, only running the existing one to completion and
reading its full per-shape output instead of grepping for a single line, which is what both
prior single-shape checks effectively did.

**Alternatives considered**:
- *Add a dedicated large-shape-only correctness pass just for subgroup=32 candidates*:
  rejected — the existing matrix already includes the `M=256` shapes that surfaced the
  failure; no new shapes need to be invented, only not discarded.

## Decision 3: Search budget convention is unchanged from `025`

**Decision**: Keep `025`'s proportional cap (≤15% of the legal space, hard-capped at 30
real on-device measurements) even though the legal space is now larger (subgroup_size is a
second value at every tile/grid point instead of one fixed value).

**Rationale**: This workstream's existing budget convention (`022`, `025`) is about bounding
*performance*-measurement device time, which stays the expensive step; the broadened
correctness gate (Decision 2) is cheap by comparison — it reuses an existing harness mode at
shapes it already covers, run once per surviving candidate, not a new expensive stage. No
justification exists to change the convention just because this feature adds a cheap gate,
so it does not.

**Alternatives considered**:
- *Raise the cap since the legal space roughly doubles*: rejected — the cap exists to bound
  device time on a shared board (Principle II/VIII), not to scale proportionally forever;
  `025`'s own FR-007 explicitly caps absolute measurements at 30 regardless of legal-space
  size for exactly this reason, and this feature inherits that reasoning unchanged.

## Decision 4: Closing an undocumented build-recipe gap (Android bench subproject)

**Decision**: Document, as part of this feature's `quickstart.md`, the two-stage Android
build sequence this session had to reconstruct from `.artifacts/cmd-log-*.sh` because it is
not written down in `.shared-context/instruction-for-ai/setup/README.md` (which explicitly
notes the `test_coopmat_*` microbenchmark binaries are "still `quant-dev/`-only" and does not
give the actual commands for building them from a `dev`-lineage worktree):

1. `cmake --build cmake-out-android-vk -j"$(nproc)" --target install` (the backend must be
   installed, not just built, before the bench subproject can `find_package(executorch)`
   against it) — this session's execution worktree had a stale, never-installed
   `cmake-out-android-vk`, which is why `cmake --build cmake-out-android-vk/bench --target
   test_coopmat_linear_bench` failed with "No rule to make target" on first attempt.
2. `cmake backends/vulkan/test/custom_ops -Bcmake-out-android-vk/bench
   -DCMAKE_TOOLCHAIN_FILE=... -DCMAKE_PREFIX_PATH=$(pwd)/cmake-out-android-vk ...` (configure
   the bench subproject against the just-installed tree) — only needed once, or again after
   adding a new shader/yaml that needs re-registering.
3. `cmake --build cmake-out-android-vk/bench --target test_coopmat_linear_bench -j"$(nproc)"`
   (incremental rebuild for subsequent source/shader edits).

**Rationale**: Principle X ("Consult `.shared-context/instruction-for-ai` Before Acting")
exists precisely to prevent re-deriving this kind of thing from scratch each time; this
session had to fall back to grepping `.artifacts/cmd-log-*.sh` because the canonical doc
doesn't cover it. Writing it into this feature's own `quickstart.md` at minimum keeps this
feature's own re-runs from repeating the derivation; whether it should also be promoted into
`.shared-context/instruction-for-ai/setup/README.md` itself is a documentation-maintenance
decision outside this feature's scope (that doc is explicitly owned by the `doc-maintainer`
agent, invoked only on explicit user request per its own definition) — flagged here, not
acted on unprompted.

**Alternatives considered**:
- *Silently rely on tribal memory / re-derive it again next time*: rejected — this is exactly
  the failure mode Principle X exists to prevent, and this session already paid the cost of
  deriving it once; not writing it down would waste that.

## Decision 5: Execution worktree — reuse `dbuf-int8-sweep`, do not branch a fresh one

**Decision**: Execute this feature's shader edits, build, and measurement in the existing
`dbuf-int8-sweep` worktree (`023-8da4w-int8-dbuf-sweep-impl` branch), not a newly-branched
worktree off `dev` (which is what `025`'s own research.md Decision 4 recommended, and what
`025` itself apparently did before that worktree was cleaned up).

**Rationale**: This is a direct, deliberate deviation from `025`'s precedent, justified by
this feature's own subject matter: the `dbuf-int8-sweep` worktree already has (a) `025`'s
`linear_dq8ca_q4gsw_coopmat_tsweep.{glsl,yaml}` template and its `QuantizedLinear.cpp`
dispatch extension sitting there uncommitted, (b) this session's ad-hoc `sg32test` shader
variant and allow-list entry that this feature's FR-012 requires either promoting or
removing, and (c) a working, installed `cmake-out-android-vk` + configured
`cmake-out-android-vk/bench` Android build tree (this session bootstrapped it — Decision 4).
Branching a fresh worktree per `025`'s own precedent would duplicate the ~15-20 minute
install+configure step for no benefit, and — more importantly — would leave the `sg32test`
probe orphaned in a worktree this feature never touches, which would make FR-012/SC-007
(supersede-or-document the probe) impossible to satisfy honestly. `dev/executorch` itself is
still never checked out onto this feature's branch — this deviation is about *which*
pre-existing side worktree hosts execution, not about repointing the canonical `dev/` folder,
so the workspace's "never repoint an existing worktree" rule is not violated.

**Alternatives considered**:
- *Follow `025`'s Decision 4 literally and branch a new worktree off `dev`*: rejected per
  the rationale above — it was the right call when `025` had no existing warm infrastructure
  to reuse; this feature does have that infrastructure, in a worktree already bound to a
  closely-related branch (`023-8da4w-int8-dbuf-sweep-impl`, the origin of the `dbuf1-4`
  variants this feature holds fixed).
- *Move the uncommitted `tsweep`/`sg32test` work into a brand-new worktree via `git stash` +
  apply*: rejected as unnecessary indirection — the existing worktree is already checked out
  on a branch whose whole purpose is this shader family; there's no binding-table entry that
  this deviation would violate (that branch was already the site of the relevant uncommitted
  work, not a "different branch than expected" surprise).

## Decision 6: Principle V deliverable — update the shader's point-of-use comment

**Decision**: Regardless of this feature's final performance verdict, its `results/` output
includes a proposed diff to `linear_dq8ca_qw_coopmat.glsl`/`.yaml`'s header comment, replacing
the current blanket "the Xclipse PAL compiler crashes ... at forced subgroup size 32"
statement with whatever this feature's Phase 0/3 findings actually establish (e.g., "does not
crash on driver `f14c51b6f8`+, but is shape-dependently incorrect and/or slower than
subgroup=64 at the shapes tested — see `specs/026`"), rather than leaving the stale claim in
place even if the sweep confirms subgroup=64 should still ship.

**Rationale**: Constitution Principle V requires every driver workaround to be documented at
its point of use. The current comment is itself already stale evidence (per Decision 1) sitting
uncorrected in production source; this feature is the first to have gathered enough evidence
to responsibly rewrite it. Leaving a known-stale crash claim in place — even if the practical
shipping decision doesn't change — invites a third redundant re-discovery of the same gap in
a future feature, which is precisely the failure mode `025`'s own T014 finding warned about
and this feature exists to close for good.

**Alternatives considered**:
- *Leave the comment as-is since the shipping decision (subgroup=64) may not change*:
  rejected — the practical decision and the documented evidence are different things; the
  comment's factual claim (blanket crash) is what's stale, independent of what ships.
