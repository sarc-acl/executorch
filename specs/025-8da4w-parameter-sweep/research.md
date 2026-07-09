# Phase 0 Research: 8da4w (dq8ca/q4gsw) CoopMat Tile/Subgroup Parameter Sweep

No `NEEDS CLARIFICATION` items remain from the plan's Technical Context — this feature
reuses `022`'s already-proven autotune methodology and `023`'s already-built `dbuf1-4`
variant family, and the shader source itself (read directly, not assumed) resolves the
questions that would otherwise be open.

## Decision 1: The `8da4w` legal tile/subgroup space is NOT `4w`'s 642-candidate space

**Decision**: Re-derive the legal configuration space from `linear_dq8ca_qw_coopmat.glsl`/
`.yaml`'s own constraints rather than reusing `022`'s 642-candidate `4w` enumeration. Two
concrete, source-verified differences constrain the space:

1. **`SUBGROUP_SIZE` is effectively fixed at 64, not swept.** The shipped `8da4w` shader's
   own header comment states: *"the Xclipse PAL compiler crashes in
   `vkCreateComputePipelines` when int8 WMMA is compiled at forced subgroup size 32 (fp16
   WMMA at 32 is fine; see `linear_qw_coopmat`)"* — confirmed by reading
   `linear_dq8ca_qw_coopmat.yaml` directly: `SUBGROUP_SIZE: 64` is its only value, versus
   `linear_qw_coopmat.yaml`'s `SUBGROUP_SIZE: 32` for `4w`. This collapses one entire sweep
   axis `022` had (`{32, 64}`) down to a single fixed value for `8da4w`, cutting the
   candidate count roughly in half before any other constraint is applied. Any candidate
   generator that (incorrectly) tries `SUBGROUP_SIZE=32` for this shader MUST be treated as
   illegal-by-known-driver-crash, not measured (Principle V — this exact workaround already
   carries its point-of-use comment in the shipped shader; this feature's tooling must not
   silently reintroduce the crash by ignoring that comment).
2. **Register/shared-memory footprint per candidate is larger than `4w`'s at the same tile
   shape.** `linear_dq8ca_qw_coopmat.glsl` keeps a per-subgroup `int32` MMA accumulator
   (`accum_int32[MMAS_PER_SG_M][MMAS_PER_SG_N]`) **and** a running `fp32` accumulator
   (`result[MMAS_PER_SG_M][MMAS_PER_SG_N]`) simultaneously — `4w`'s `linear_qw_coopmat.glsl`
   only carries the latter. `Ash_int8`/`Bsh_int8` are also double-buffered `shared uint`
   arrays sized off `WG_TILE_M/N * MMA_K` in bytes, and the shipped shader's own `WG_TILE_K`
   is `32` (double `4w`'s `16`), so a same-tile-shape candidate stages twice the K-depth of
   int8 data per double-buffer slot. Both raise `022`'s occupancy/register-pressure
   analytical proxies (Decision 2) at a given tile shape versus the `4w` case they were
   calibrated on.

**Rationale**: Constitution Principle VI ("Verify With Tools, Never Assume") — the source
files directly show these are real, load-bearing differences, not a hypothetical. Reusing
`4w`'s enumeration unchanged would either (a) include `SUBGROUP_SIZE=32` candidates known to
crash the driver, wasting device time and worktree-corrupting pipeline-creation-crash risk
`023` already had to isolate per-process for, or (b) mis-rank the legal `SUBGROUP_SIZE=64`
subset using an occupancy model calibrated on `4w`'s lower register/shared-memory footprint.

**Alternatives considered**:
- *Reuse `4w`'s 642 candidates, just drop `SUBGROUP_SIZE=32` entries*: rejected — this only
  fixes difference (1), not (2); the analytical scores for the remaining candidates would
  still be calibrated on the wrong footprint model.
- *Sweep `SUBGROUP_SIZE=32` anyway to double-check the crash still reproduces on the current
  driver*: rejected as in-scope-but-bounded — included as a single explicit anchor
  correctness/compile attempt (not a ranked candidate) per spec Edge Cases' "shortlisted
  candidate fails to compile" handling, to re-verify the documented workaround is still
  necessary rather than assuming a multi-version-old comment still holds. This does not
  count against the 30-measurement search budget (it is a re-verification of a known
  constraint, not a search candidate).

## Decision 2: Analytical cost model — reuse `022`'s formula, recalibrate the constant

**Decision**: Reuse `022`'s two hardware-derived proxies (occupancy proxy from
shared-memory footprint + thread count; register-pressure proxy from accumulator count per
subgroup) and its combination formula
`score = occupancy_proxy / (1 + max(0, accumulators - K) * penalty)`, but recompute the
occupancy proxy's `LDS_bytes(candidate)` term using `8da4w`'s actual shared-memory layout
(`Ash_int8`/`Bsh_int8` sized in bytes off `WG_TILE_K=32`-scaled int8 data, plus the
`izp_sh`/`ifs_sh`/`wsum_sh`/`wsc_sh`/`bias_sh` broadcast arrays this shader carries that
`4w`'s shader does not), and recompute the register-pressure term's baseline `K` and penalty
weight from User Story 1's dbuf re-confirmation measurements (the shipped 128×64/K32/2×2/s64
configuration) instead of `022`'s `4w`-calibrated `K=8`/`0.15`, since `8da4w` carries the
extra `int32` accumulator array Decision 1 identifies.

**Rationale**: The formula shape already proved effective at pruning `022`'s 642 candidates
down to a ~24-32-config shortlist without device time; only its inputs need updating for a
different shader's real memory layout. Deriving a wholly new model would be unjustified
extra design work for no evidence it prunes better.

**Alternatives considered**:
- *Skip analytical pruning, go straight to a coarse on-device sweep*: rejected — this is
  exactly the "exhaustive/guessing" failure mode `022`'s own P1 user story exists to avoid,
  and this feature's spec explicitly requires the same zero-device-time pruning stage
  (FR-003).

## Decision 3: Dispatch mechanism — additive env var, not a replacement

**Decision**: Add a new tile/subgroup-parameterized shader template
(`linear_dq8ca_q4gsw_coopmat_tsweep.{glsl,yaml}`) and a new `coopmat_variant_tile()`-style
token table in `QuantizedLinear.cpp`, analogous to `022`'s `4w` `tsweep` mechanism —
additive to, not replacing, `023`'s existing `ET_VK_DQ8CA_COOPMAT_VARIANT` dbuf-selection
env var. The tile/subgroup sweep's env var only takes effect once loop structure is fixed at
the User-Story-1-winning `dbuf` shape (i.e., the tsweep template is instantiated with that
loop structure baked in, not built as a fifth independent axis).

**Rationale**: Keeps the two axes (loop structure, tile/subgroup geometry) genuinely
separable per spec Assumptions, and matches the exact precedent `022` established for `4w`
— no new dispatch-mechanism design is needed.

**Alternatives considered**:
- *Cross loop structure and tile/subgroup geometry into one 4x-larger combined sweep*:
  rejected per spec Clarifications (2026-07-09) — duplicates `023`'s dedicated work and
  quadruples the search space for no new information, unless User Story 1 itself finds
  loop-structure/geometry interaction (tracked as a reported limitation, not silently
  absorbed, per spec Assumptions).

## Decision 4: Where this executes

**Decision**: Spec-kit documentation and analysis/orchestration scripts live in this repo
(`dev/executorch`, under `specs/025-8da4w-parameter-sweep/`), committed to `yanwen/dev-1.3`
before any execution worktree is created. Shader-variant edits, the Android build, and all
on-device measurement happen in a **new** git worktree branched from that commit — never by
checking out this feature's working branch inside the existing `dev/` worktree folder — per
this workspace's standing rule that an existing worktree's bound branch is never repointed.

**Rationale**: Matches `022`'s and `023`'s precedent, and directly follows this workspace's
`CLAUDE.md` "Critical rules" section (worktree-binding incident, 2026-07-08) and its
"Development Workflow" table (`dev/` is bound to `yanwen/dev-1.3`; new work gets a new
worktree).

**Alternatives considered**:
- *Work directly in the `dev/executorch` worktree on a feature branch, switching its
  checkout*: rejected — explicitly prohibited by workspace `CLAUDE.md`.
- *Reuse `023`'s existing worktree if it still has a warm build*: viable and preferred if
  `023`'s worktree is still present and warm at implementation time (avoids a
  submodule-init + Android cross-build, ~15-20 minutes even with ccache, per `022`'s
  research.md precedent) — deferred to Phase 2/implementation-time discovery, not decided
  here, since this feature's spec-kit authoring does not need to know that yet.

## Decision 5: Re-confirming `dbuf2` before fixing it (User Story 1)

**Decision**: Treat the user's reported "`dbuf2` wins for `8da4w`" as the starting
hypothesis, not an established fact, and re-measure all four `linear_dq8ca_q4gsw_coopmat_dbuf{1..4}`
variants (already built by `specs/023`) at the currently-shipped 128×64/K32/2×2/s64 geometry
before holding any of them fixed.

**Rationale**: Prior workstream memory records an earlier "dbuf2 wins" finding for this
exact shader that was later corrected to "dbuf1 wins" after being traced to a broken-driver
artifact. This feature does not referee that history, but per Constitution Principle VIII
(driver identity must be re-verified before every coopmat measurement) and Principle VI
(verify with tools, never assume), a claim with this specific prior false-positive history
must be re-measured under a freshly-verified driver state before ~30 more measurements are
built on top of it — an unverified foundation would put every downstream tile/subgroup
result at risk of the same artifact.

**Alternatives considered**:
- *Trust the reported `dbuf2` result and proceed straight to the tile/subgroup sweep*:
  rejected — this is precisely the risk Decision 5's rationale describes, and the spec's own
  User Story 1 (P1, MVP) already requires this re-confirmation.
