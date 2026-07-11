# Feature Specification: Re-Open SUBGROUP_SIZE=32 in the 8da4w CoopMat Tile/Subgroup Sweep on M5 EVT1

**Feature Branch**: `026-8da4w-subgroup32-sweep`

**Created**: 2026-07-11

**Status**: Draft

**Input**: User description: "now i think we should redo the specify of the 8da4w sweep, and try test 32 as well." — redo `specs/025-8da4w-parameter-sweep`'s tile/subgroup search for the `8da4w` (`dq8ca_q4gsw` int8 WMMA) coopmat shader, this time including `SUBGROUP_SIZE=32` as a real, correctness-gated search axis instead of excluding it outright.

## Context (why this feature exists now)

`specs/025-8da4w-parameter-sweep` searched the `8da4w` shader's tile-shape ×
subgroup-grid space and found a winner (`128×32/K16/1×2/s64`, 1731.0 GFLOP/s,
+2.55% over the previously-shipped config). It fixed `SUBGROUP_SIZE` at `64`
for every candidate, citing the shipped shader's own header comment: the
Xclipse PAL compiler crashes in `vkCreateComputePipelines` when int8 WMMA is
compiled at forced subgroup size 32 (fp16 WMMA at 32 is fine for the sibling
`4w` shader).

`specs/025` also ran one bounded, one-shot re-check outside its own search
budget (task T014, `results/subgroup32-reverification.md`): an ad-hoc
`sg32test` variant, same shipped `128×64/K32/2×2` tile shape, forced to
`SUBGROUP_SIZE=32`. It compiled, created its pipeline without crashing,
dispatched genuine coopmat (kernel-name confirmed), and passed the one
correctness check available at that shape (`M=K=N=128`). That result was
deliberately not acted on inside `specs/025` — re-opening the axis mid-sweep
would have invalidated the already-computed 542-candidate enumeration for no
proven benefit, since one passing shape does not establish broad viability.
It was recorded as a finding for a follow-up feature instead.

This session independently reproduced that same `sg32test` probe against a
*second* M5 EVT1 board (a different physical device than `025` used) and
found a fuller, less favorable picture than T014's single data point:

- **Correctness is shape-dependent, not uniform.** At the shipped tile shape,
  `sg32test` (subgroup=32) failed 3 additional correctness cases that pass at
  subgroup=64 on the identical binary/build: `M=256,K=256,N=256` and two
  `M=256,K=128,N={128,64}` Buffer-path shapes. `M=K=N=128` — T014's only
  tested shape — still passes. This is exactly the gap this feature exists
  to close: a single-shape probe cannot distinguish "safe everywhere" from
  "safe only at the one shape someone happened to test."
- **Performance is also worse, not better.** At the standard `M=2048`
  representative-shape measurement, `sg32test` (subgroup=32, same tile)
  measured ~1095–1169 GFLOP/s — below both the currently-shipped subgroup=64
  configuration (~1688 GFLOP/s) and `specs/025`'s actual winner (1736
  GFLOP/s, subgroup=64).

Both observations are from a single ad-hoc probe at one tile shape on one
device pairing, not a proper swept search — which is exactly the gap this
feature closes. Re-opening `SUBGROUP_SIZE` as a real search axis, with the
same rigor `025` applied to tile shape and subgroup grid, is the only way to
know whether *any* subgroup=32 candidate is both correct across the full
representative shape set and competitive on performance, or whether the
axis is correctly closed off after all — this time by evidence at proper
sweep scope, not by a stale crash report or a single passing shape.

Related prior work, not yet the answer to this question:

- `specs/025-8da4w-parameter-sweep` is the direct predecessor this feature
  redoes with one additional axis; its winner (`128×32/K16/1×2/s64`, 1736
  GFLOP/s) is this feature's baseline to beat, not a result to re-derive
  from scratch.
- `specs/022-linear-coopmat-autotune` found `4w`'s optimum
  (`128×64/K16/1×4/s32`) at subgroup=32 — establishing that subgroup=32 is
  not inherently bad on this hardware for cooperative matrix work in
  general, only previously assumed illegal for `8da4w`'s int8 path
  specifically.
- The ad-hoc `sg32test` shader variant and `QuantizedLinear.cpp` allow-list
  entry added during this session's re-verification, in the
  `dbuf-int8-sweep` worktree (`023-8da4w-int8-dbuf-sweep-impl` branch), is a
  temporary probe, not this feature's search infrastructure — see Assumptions.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Re-derive the legal search space with subgroup_size as a real axis (Priority: P1) 🎯 MVP

As the workstream engineer, I want the `8da4w` shader's legal tile-shape ×
subgroup-grid × subgroup-size space re-derived with `SUBGROUP_SIZE ∈ {32,
64}` as a real variable (not a constant fixed at 64), re-checking whether
the previously-assumed compiler-crash exclusion for 32 still holds on the
current driver, so the rest of this feature searches the space that
actually exists today instead of the narrower space `025` assumed.

**Why this priority**: Every later stage inherits this space. If the
crash-based exclusion is stale (as this session's re-verification suggests)
but the search space isn't re-derived, the feature would repeat `025`'s
same gap under a different name.

**Independent Test**: Attempt to compile and create a pipeline for a small
representative set of subgroup=32 candidates spanning several tile shapes
(not just the one shape `025`'s T014 and this session's probe happened to
test); confirm whether the documented crash reproduces on any of them on
the current driver, and record the outcome per candidate.

**Acceptance Scenarios**:

1. **Given** the `8da4w` shader's register/shared-memory constraints,
   **When** the legal space is re-derived, **Then** it explicitly states
   whether `SUBGROUP_SIZE=32` is included or excluded, and why, referencing
   real compile/pipeline-creation evidence gathered by this feature (not
   solely the shipped shader's pre-existing header comment).
2. **Given** a subgroup=32 candidate that fails to compile or crashes
   pipeline creation, **When** this occurs, **Then** it is recorded as a
   compile failure with the specific error, and the search continues with
   the remaining space — a reproduction of the historical crash is not
   treated as a bug in this feature, just a legality finding.

---

### User Story 2 - Correctness-gate subgroup=32 candidates across the full representative shape set (Priority: P1)

As the workstream engineer, I want every subgroup=32 candidate that reaches
on-device measurement checked for correctness across the same full
representative shape set used for subgroup=64 candidates — not just one
shape — so that a candidate which passes at a small shape but silently
miscomputes at a larger one (as this session found at the shipped tile
shape) cannot reach the performance ranking.

**Why this priority**: This is the specific gap this feature exists to
close. `025`'s T014 and this session's independent re-check each tested
exactly one shape; both happened to pick shapes that don't reveal the
shape-dependent failure this session found at `M=256`. Without shape-broad
correctness gating, a future reader could reasonably repeat the same
single-shape mistake a third time.

**Independent Test**: Take any subgroup=32 candidate that compiles, run it
against the full multi-shape correctness matrix (the same shapes already
used for subgroup=64 candidates in this workstream), and confirm the
feature reports a per-shape pass/fail breakdown rather than a single
pass/fail verdict.

**Acceptance Scenarios**:

1. **Given** a subgroup=32 candidate, **When** it is correctness-checked,
   **Then** the result names every shape tested and its individual
   pass/fail outcome — a candidate is not marked "correct" on the strength
   of one passing shape alone.
2. **Given** a candidate that passes at some shapes and fails at others,
   **When** this occurs, **Then** it is excluded from the performance
   ranking and reported as a shape-dependent correctness failure (naming
   which shapes failed), not silently dropped or averaged away.

---

### User Story 3 - Search for and validate a subgroup=32 (or mixed) winner against the subgroup=64 baseline (Priority: P2)

As the workstream engineer, I want the surviving (fully-correct) subgroup=32
candidates measured for performance alongside a subgroup=64 shortlist
comparable in scope to `025`'s, so I get a definitive answer to whether any
subgroup=32 configuration beats `025`'s standing winner
(`128×32/K16/1×2/s64`, 1736 GFLOP/s) — not just whether subgroup=32 is
merely legal.

**Why this priority**: Legality and correctness (User Stories 1–2) are
necessary but not sufficient — this session's own probe found a
correctly-compiling, sometimes-correct subgroup=32 configuration that was
still slower than the standing winner. The feature's actual deliverable is
a performance verdict, not just a legality/correctness map.

**Independent Test**: Run a staged, budget-capped on-device search over the
correctness-surviving subgroup=32 candidates (and a subgroup=64 shortlist
re-anchoring `025`'s winner), and confirm the feature converges on one
overall winner — new or the standing `025` winner — with a stated
percentage/factor comparison, or an explicit "no improvement found."

**Acceptance Scenarios**:

1. **Given** the correctness-surviving candidate set from User Story 2,
   **When** the staged performance search executes, **Then** every
   candidate receives at least one measurement, weaker candidates are
   eliminated early, and the top contenders receive full statistical rigor
   (3-run mean, CoV<5%, this workstream's existing bar).
2. **Given** the final overall winner, **When** it is reported, **Then**
   the report states whether it uses `SUBGROUP_SIZE=32` or `64`, and gives
   a head-to-head comparison against `025`'s standing winner (1736 GFLOP/s)
   and the pre-`025` shipped baseline.
3. **Given** no subgroup=32 candidate beats `025`'s standing winner on
   either correctness or performance, **When** the search concludes,
   **Then** the feature explicitly states this outcome — closing the axis
   with proper sweep evidence this time — rather than treating an absence
   of improvement as an inconclusive or omitted result.

---

### Edge Cases

- What happens when a subgroup=32 candidate reproduces the historical
  `vkCreateComputePipelines` crash at some tile shapes but not others? Each
  shape/candidate pair is recorded independently with its own compile
  status; a crash at one shape does not disqualify subgroup=32 candidates
  at other shapes from being attempted.
- What happens when a candidate passes correctness at every representative
  shape but the improvement over `025`'s winner is within measurement noise
  (not a clear win)? The documented tie-breaking rule from this workstream's
  existing convention applies, and the result states the margin explicitly
  rather than declaring a win on an ambiguous delta.
- What happens if re-deriving the legal space (User Story 1) finds that
  *no* subgroup=32 candidate compiles at all (full reproduction of the
  historical crash across the board)? The feature reports this as a clean,
  fully-swept confirmation that the axis is closed — a stronger and
  differently-evidenced conclusion than `025`'s exclusion-by-assumption,
  even though the practical outcome (excluded) is the same.
- What happens to the ad-hoc `sg32test` shader/binding added in the
  `dbuf-int8-sweep` worktree during this session's re-verification? It is
  superseded by this feature's own tile/subgroup-swept shader variants
  (extending the existing `tsweep` mechanism to carry `SUBGROUP_SIZE` as a
  token field) and is removed once this feature's infrastructure covers the
  same shape/tile combination it probed.
- What happens if the shared M5 EVT1 device drifts to an unexpected driver
  build mid-search? The process halts or re-verifies rather than continuing
  under unknown state, per this workstream's existing device-safety practice.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The process MUST re-derive the legal `8da4w` tile-shape ×
  subgroup-grid × subgroup-size configuration space with `SUBGROUP_SIZE`
  treated as a real variable over `{32, 64}`, rather than fixed at `64` as
  `025` assumed.
- **FR-002**: The process MUST attempt real on-device compilation/pipeline
  creation for a representative spread of subgroup=32 candidates across
  multiple tile shapes (not one shape only) to determine whether the
  historical Xclipse PAL compiler crash still reproduces, and MUST record
  the outcome per candidate/shape rather than generalizing from a single
  data point.
- **FR-003**: The process MUST correctness-check every subgroup=32
  candidate that compiles against the full multi-shape representative set
  already used for subgroup=64 candidates in this workstream — not a single
  shape — and MUST report a per-shape pass/fail breakdown.
- **FR-004**: A candidate that passes correctness at some representative
  shapes and fails at others MUST be excluded from the performance ranking
  and reported as a shape-dependent correctness failure naming the failing
  shapes, not silently dropped or treated as passing.
- **FR-005**: The process MUST include an analytical pruning/ranking stage
  before on-device measurement, covering the full re-derived space (both
  subgroup sizes), consistent with `025`'s and `022`'s existing methodology.
- **FR-006**: The process MUST measure shortlisted, correctness-surviving
  candidates using a staged approach (cheap first pass, full statistical
  rigor only for top contenders), consistent with this workstream's existing
  convention.
- **FR-007**: The process MUST report a single final recommended
  configuration with a head-to-head comparison against `025`'s standing
  winner (`128×32/K16/1×2/s64`, 1736 GFLOP/s) and the pre-`025` shipped
  baseline, explicitly stating which subgroup size the winner uses.
- **FR-008**: If no subgroup=32 candidate beats `025`'s standing winner on
  both correctness (all representative shapes) and performance, the process
  MUST report that outcome explicitly, framed as a swept confirmation that
  the axis is closed — not as an inconclusive or omitted result.
- **FR-009**: The process MUST operate within a bounded, pre-declared cap on
  total real on-device measurements, following this workstream's existing
  proportional-cap convention (no more than 15% of the full re-derived legal
  space, capped at 30 measurements in absolute terms).
- **FR-010**: The process MUST detect signs of an untrustworthy measurement
  environment (unexpected driver state, shared device unexpectedly busy) and
  halt or re-verify rather than silently continuing.
- **FR-011**: The process MUST record, for every configuration not taken to
  full on-device measurement, a documented reason it was deprioritized or
  excluded.
- **FR-012**: The process MUST address the disposition of the ad-hoc
  `sg32test` probe shader/binding added during this session's
  re-verification (in the `dbuf-int8-sweep` worktree) — either superseding
  it with this feature's own swept shader variants and removing the probe,
  or explicitly documenting why it is retained.

### Key Entities

- **Configuration Candidate**: One point in the re-derived `8da4w`
  tile/subgroup/subgroup-size search space (output tile height/width,
  K-step, subgroup grid shape, subgroup size ∈ {32, 64}), at the `dbuf2`
  loop structure `025` already confirmed; carries derived properties
  (shared-memory footprint, thread count, int32 accumulator count) and a
  compile/pipeline-creation status.
- **Shape-Correctness Result**: Per-candidate, per-representative-shape
  pass/fail outcome — the unit this feature reports at, replacing the
  single pass/fail verdict `025`'s T014 and this session's probe each used.
- **Analytical Score**: A pre-measurement estimate of a candidate's likely
  relative performance from hardware-derived proxies; used only to rank and
  prune, never reported as a final result.
- **Search Budget**: The declared cap on total real on-device measurements
  this feature's search may consume.
- **Optimal Configuration**: The final recommended `8da4w` candidate
  (tile/subgroup/subgroup-size geometry), backed by a full statistically-
  sound performance measurement and a passing correctness check at every
  representative shape.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The feature states, with per-shape evidence, whether
  `SUBGROUP_SIZE=32` is legal (compiles without crashing) across a
  representative spread of tile shapes — not one shape — explicitly
  confirming or narrowing `025`'s T014 finding.
- **SC-002**: Every subgroup=32 candidate reaching correctness checking is
  evaluated against the same full multi-shape set used for subgroup=64
  candidates, and zero shape-dependent correctness failures reach the
  performance ranking.
- **SC-003**: The process states, with numeric evidence, whether any
  fully-correct subgroup=32 candidate's measured throughput exceeds `025`'s
  standing winner (1736 GFLOP/s), and by how much or how little.
- **SC-004**: The final recommended configuration is reported with a
  3-run-mean, CoV<5% measurement and an explicit head-to-head comparison
  against both `025`'s standing winner and the pre-`025` shipped baseline.
- **SC-005**: The end-to-end process directly measures on real hardware no
  more than 15% of the full re-derived legal space, and no more than 30
  measurements in absolute terms.
- **SC-006**: For any configuration in the swept search universe, the
  workstream engineer can determine why it was or wasn't taken to on-device
  measurement without re-running the search.
- **SC-007**: The disposition of the ad-hoc `sg32test` probe (superseded and
  removed, or explicitly retained with reason) is stated in the feature's
  final report.

## Assumptions

- M5 EVT1 is the target device for all measurements in this feature, per
  this workstream's active-target constraint; device availability and
  driver identity are re-verified before use, not assumed. Either the
  primary or secondary M5 EVT1 board may be used; if both are used across
  the feature, the report states which board produced which result (the two
  boards are the same chip/build but independently drift in driver state).
- The `dbuf2` loop structure `025` confirmed for `8da4w` is reused as-is;
  this feature sweeps tile/subgroup/subgroup-size geometry only, on the same
  axis-separability reasoning `025` used relative to `023`.
- "Optimal" means highest FLOP-weighted throughput across the same 6-shape
  set (`wq` + `w1_gate` ops for the 1B/3B/8B target models) `022`/`023`/`025`
  used, measured at the standard pinned-clock configuration — this feature
  does not introduce a new shape set for the performance ranking, but does
  introduce a broader shape set specifically for the *correctness* gate
  (User Story 2), since the whole point of this feature is that a narrow
  correctness check was the gap last time.
- The ad-hoc `sg32test` shader variant and its `QuantizedLinear.cpp`
  allow-list entry, added in the `dbuf-int8-sweep` worktree
  (`023-8da4w-int8-dbuf-sweep-impl` branch) during this session, are
  temporary and not this feature's search infrastructure; this feature
  builds its own `SUBGROUP_SIZE`-carrying extension of the existing
  `tsweep` shader/dispatch mechanism (the same one `025` used) rather than
  reusing the probe as-is.
- This workstream's existing device-safety practices (driver-identity
  re-verification, halting on drift, checking shared-device availability)
  apply unchanged.
- This is an internal engineering capability for this workstream's own use;
  "user"/"engineer" throughout this spec refers to the workstream engineer
  running the sweep, not an end product user.
- A subgroup=32 winner, if found, is a drop-in alternative to `025`'s
  winner at the same shader/dispatch site (`ET_VK_DQ8CA_COOPMAT_VARIANT`);
  this feature does not itself change which configuration ships by default
  — that remains a separate decision per this workstream's existing
  Tier-1/Tier-2 validation convention (`025`'s own recommendation deferred
  shipping to a Tier-2 e2e validation).
