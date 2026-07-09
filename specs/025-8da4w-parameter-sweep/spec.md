# Feature Specification: 8da4w (dq8ca/q4gsw) CoopMat Tile/Subgroup Parameter Sweep on M5 EVT1

**Feature Branch**: `025-8da4w-parameter-sweep`

**Created**: 2026-07-09

**Status**: Draft

**Input**: User description: "I had a result showing dbuf v2 is the winner on M5 for 8da4w. Conduct a parameter sweep on 8da4w shader just like how i sweep all parameters to find the optimal config of 4w is 128x64,2x2. Find the optimal config for 8da4w"

## Context (why this feature exists now)

`specs/022-linear-coopmat-autotune` found the optimal tile/subgroup configuration for the `4w`
(fp16 weight-only) `linear_q4gsw_coopmat` shader on M5 EVT1 to be **128×64/K16/2×2/s32**, holding
loop structure fixed at dbuf1. That search was scoped to `4w` only (per its own Assumptions
section); the `8da4w` int8-activation shader (`linear_dq8ca_qw_coopmat` / `dq8ca`+`q4gsw`) was
never swept over tile/subgroup geometry with an equivalent search — only its double-buffer *loop
structure* was swept, in `specs/023-8da4w-int8-dbuf-sweep`, which the user reports found `dbuf2`
to be the winning loop shape for `8da4w` (superseding this workstream's earlier `dbuf1`-wins
finding for the same shader, which prior memory records as itself correcting a broken-driver
artifact — this feature does not need to referee that history; it takes the user's stated `dbuf2`
result as the loop-structure starting point and re-confirms it in User Story 1 before spending
further device time on top of it).

This feature closes that gap: sweep the `8da4w` shader's tile-shape/subgroup-size parameter space
on M5 EVT1, the same way `022` did for `4w`, holding loop structure fixed at the winning `8da4w`
loop shape, and report the optimal `8da4w` configuration.

Related prior work, not yet the answer to this question:
- `specs/008-8da4w-parameter-sweep` swept `8da4w` tile/subgroup parameters, but on a different
  device (`rocky-ryzen`, RDNA3 desktop iGPU) — not M5 EVT1/Xclipse, and predates the WMMA coopmat
  port landing on `dev`.
- `specs/023-8da4w-int8-dbuf-sweep` swept `8da4w`'s double-buffer loop structure on M5 EVT1, but
  held tile/subgroup geometry fixed at the currently-shipped configuration throughout — it did not
  vary tile shape or subgroup size.
- `specs/024-8da4w-slower-than-4w` is investigating a separate but related question (why `8da4w`
  underperforms `4w` end-to-end); this feature's result is an input to that investigation, not a
  substitute for it.

## Clarifications

### Session 2026-07-09

- Q: `022` used an analytical-pruning + staged-measurement search over 642 candidates because that
  was the full legal combinatorial space for `4w`. Should `8da4w`'s search reuse that same
  smart-autotune machinery, or the simpler curated-set approach `008`/`014` used for this shader
  family? → A: Reuse `022`'s smart-autotune approach (analytical pruning → staged on-device
  search → validated winner), re-deriving the `8da4w`-specific legal configuration space (tile
  shapes × subgroup grids × subgroup sizes valid for `dq8ca`/`q4gsw`'s int8-MMA shared-memory and
  register constraints, which differ from `4w`'s fp16-MMA constraints) rather than assuming `4w`'s
  642-candidate space applies unchanged. This matches the user's explicit "just like how I sweep
  ... 4w" framing and avoids re-deriving a new methodology from scratch.
- Q: Should this feature re-sweep loop structure (dbuf1-4) together with tile/subgroup geometry, or
  hold loop structure fixed at the user-reported `dbuf2` winner and sweep tile/subgroup only? → A:
  Hold loop structure fixed at `dbuf2` (re-confirmed in User Story 1) and sweep tile/subgroup
  geometry only. Loop structure was already the dedicated subject of `023`; re-sweeping it here
  would duplicate that feature's work and multiply the search space by 4x for no new information.
- Q: `022`'s search anchors included the previously-known `4w` sweep winner as a sanity check. What
  is the equivalent anchor set for `8da4w`? → A: The currently-shipped `8da4w` production
  configuration (tile/subgroup geometry as committed on `dev` today) and `4w`'s winning geometry
  (128×64/K16/2×2/s32), included as a cross-shader reference point even though it is not expected
  to be legal or optimal for `8da4w`'s different per-thread register/shared-memory footprint.
- Q: FR-007/SC-002/SC-006 require a "bounded, pre-declared cap" and "no more than 15%" of the
  legal `8da4w` configuration space, but state no concrete number. What is the actual search
  budget? → A: Proportional cap — no more than 15% of the legal `8da4w` configuration space,
  capped at 30 on-device measurements even if 15% of that space is higher (e.g., a 300-candidate
  legal space still caps at 30, not 45).
- Q: Which representative shape set should the sweep measure against? → A: The 6-shape set
  already used by `022` and `023` — the `wq` and `w1_gate` ops for each of the three target
  models (1B, 3B, 8B) — keeping this feature directly comparable to both prior results rather
  than introducing a new shape set.
- Q: What run count / CoV threshold defines the "full statistical rigor" stage for top
  contenders? → A: 3-run mean, CoV < 5% — this workstream's existing convention across
  `008`/`022`/`023`, reused as-is rather than inventing a new bar for this feature.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Re-confirm the dbuf2 loop-structure starting point (Priority: P1) 🎯 MVP

As the workstream engineer, I want to re-confirm on M5 EVT1 that `dbuf2` is
the fastest loop structure for the currently-shipped `8da4w` tile/subgroup
geometry before holding it fixed for the rest of this sweep, so that the
tile/subgroup search in User Story 2 is built on a verified foundation
rather than an unverified prior claim.

**Why this priority**: Everything downstream fixes loop structure at
`dbuf2`. If that starting point is stale, wrong, or was measured under a
different geometry than what this feature sweeps, every subsequent result
inherits the error.

**Independent Test**: Run `dbuf1`-`dbuf4` at the currently-shipped
tile/subgroup geometry, confirm each dispatches the coopmat kernel (not a
tiled fallback) and passes correctness, and confirm `dbuf2` is fastest —
or record which variant actually wins if the re-confirmation disagrees with
the prior claim.

**Acceptance Scenarios**:

1. **Given** the four double-buffer loop variants at the shipped `8da4w`
   tile/subgroup geometry, **When** each is built and measured on M5 EVT1,
   **Then** every variant produces a correctness-verified, coopmat-dispatch-
   confirmed timing number or an explicit failure reason.
2. **Given** the four measured variants, **When** compared, **Then** the
   feature records which one is actually fastest, explicitly stating
   agreement or disagreement with the user's reported `dbuf2` result, before
   that variant is held fixed for User Story 2.

---

### User Story 2 - Prune the tile/subgroup search space with zero device time (Priority: P1)

As the workstream engineer, I want the `8da4w` shader's legal tile-shape ×
subgroup-grid × subgroup-size combinations (re-derived for this shader's
int8-MMA register/shared-memory constraints, not assumed identical to
`4w`'s 642) ranked and narrowed to a small shortlist using only
analytical, hardware-derived signals, so device time is spent only on
credible candidates.

**Why this priority**: This is the step that makes the search tractable —
without it, the feature either exhaustively measures every legal
combination (expensive, shared-device time) or guesses ungrounded.

**Independent Test**: Given the enumerated legal `8da4w` configurations at
the User-Story-1-confirmed loop structure, produce a shortlist materially
smaller than the full space, with a documented inclusion/exclusion reason
per candidate, using zero on-device measurements.

**Acceptance Scenarios**:

1. **Given** the full set of legal `8da4w` tile/subgroup configurations,
   **When** pruning runs, **Then** it outputs a shortlist on the order of
   tens (not hundreds) of candidates with a ranking rationale, and zero
   device measurements have been taken.
2. **Given** the currently-shipped `8da4w` geometry and `4w`'s
   128×64/K16/2×2/s32 winner, **When** the shortlist is produced, **Then**
   both appear on it as sanity-check anchors regardless of analytical rank
   (the latter only if it is legal for `8da4w`'s constraints; if illegal,
   the feature records why instead of silently dropping it).

---

### User Story 3 - Find and validate the best-performing configuration (Priority: P2)

As the workstream engineer, I want the shortlisted `8da4w` tile/subgroup
candidates measured on M5 EVT1 using a staged approach that eliminates weak
candidates early and spends full statistical rigor only on top contenders,
ending in one validated, correctness-confirmed winner, so I have a
decision-ready optimal configuration for `8da4w`, analogous to `4w`'s
128×64/2×2 result.

**Why this priority**: This is the feature's actual deliverable — a named,
validated optimal `8da4w` configuration, not just a pruned candidate list.

**Independent Test**: Run the staged search over the User Story 2
shortlist and confirm it converges on one winning candidate that
independently passes correctness and is reported with a head-to-head
comparison against the shipped baseline and the `4w` winner.

**Acceptance Scenarios**:

1. **Given** the shortlist, **When** the staged search executes, **Then**
   every candidate receives at least one cheap measurement, only top
   performers receive full statistically-sound measurement, and candidates
   that fail to compile or fail correctness are dropped immediately.
2. **Given** a final winning candidate, **When** it is validated, **Then**
   its correctness is confirmed and its performance is reported against the
   currently-shipped `8da4w` baseline (with a percentage/factor
   improvement or explicit "no improvement found" if none beats it).
3. **Given** two or more candidates statistically indistinguishable at the
   top of the search, **When** a winner is selected, **Then** a documented
   tie-breaking rule is applied rather than an arbitrary choice.

---

### Edge Cases

- What happens when `dbuf2` re-confirmation (User Story 1) disagrees with
  the user's reported result — e.g. `dbuf1` or `dbuf3` turns out fastest at
  measurement time? The feature holds fixed whichever variant actually wins
  the re-confirmation, and explicitly states the discrepancy with the prior
  claim, rather than silently using the reported `dbuf2` regardless.
- What happens when a shortlisted candidate fails to compile or crashes
  pipeline creation? Recorded as a failure with reason, search continues.
- What happens when a candidate is mathematically incompatible with the
  quantization group size (same failure mode `008` deliberately tested for
  `8da4w`)? Caught by the correctness check, recorded as a correctness
  failure, not silently treated as passing.
- What happens if the device or driver state becomes untrustworthy
  mid-search (shared M5 EVT1 drifts to unexpected driver build, or another
  job starts running)? The process halts or re-verifies rather than
  continuing under unknown state.
- What happens if no swept configuration beats the currently-shipped
  `8da4w` baseline? The final report states this explicitly and recommends
  keeping the shipped configuration, rather than presenting a marginal
  result as a win.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The process MUST re-confirm the fastest double-buffer loop
  structure (`dbuf1`-`dbuf4`) for `8da4w` at the currently-shipped tile/
  subgroup geometry on M5 EVT1 before holding any loop structure fixed for
  the rest of the sweep, and MUST state explicitly whether the result
  matches the user-reported `dbuf2` claim.
- **FR-002**: The process MUST re-derive the legal tile-shape × subgroup-
  grid × subgroup-size configuration space specifically for the `8da4w`
  (`dq8ca`/`q4gsw` int8 cooperative-matrix) shader's own register and
  shared-memory constraints, not reuse `4w`'s 642-candidate space unchanged.
- **FR-003**: The process MUST include an analytical pruning/ranking stage
  using hardware-derived signals to reduce the full legal `8da4w`
  configuration space to a materially smaller shortlist before any
  on-device measurement.
- **FR-004**: The process MUST measure shortlisted candidates using a
  staged approach that allocates progressively more measurement effort to
  promising candidates and eliminates clearly weaker candidates early.
- **FR-005**: The process MUST confirm numerical correctness for a
  candidate before its performance number is trusted, ranked, or reported.
- **FR-006**: The process MUST report the final recommended `8da4w`
  configuration with a head-to-head comparison against the currently-shipped
  `8da4w` baseline and against `4w`'s winning configuration
  (128×64/K16/2×2/s32), including the loop-structure result from FR-001.
- **FR-007**: The process MUST operate within a bounded, pre-declared cap on
  the total number of real on-device measurements: no more than 15% of the
  full legal `8da4w` configuration space, and no more than 30 measurements
  in absolute terms even if 15% of that space is higher.
- **FR-008**: The process MUST detect signs of an untrustworthy measurement
  environment (unexpected driver state, shared device unexpectedly busy) and
  halt or re-verify rather than silently continuing.
- **FR-009**: The process MUST record, for every configuration not taken to
  full on-device measurement, a documented reason it was deprioritized or
  excluded.
- **FR-010**: If no evaluated `8da4w` configuration outperforms the current
  production baseline within the search budget, the process MUST report
  that outcome explicitly rather than present an unproven or marginal
  result as an improvement.

### Key Entities

- **Loop-Structure Variant**: One of `dbuf1`-`dbuf4`, the double-buffering
  loop shape already defined by `specs/023`; fixed at the winner found in
  User Story 1 for the remainder of this feature.
- **Configuration Candidate**: One point in the `8da4w` tile/subgroup search
  space (output tile height/width, K-step, subgroup grid shape, subgroup
  size), at the fixed loop structure; carries derived properties (shared-
  memory footprint, thread count, accumulator count for int8 MMA) and a
  validity/compile status.
- **Analytical Score**: A pre-measurement estimate of a candidate's likely
  relative performance from hardware-derived proxies; used only to rank and
  prune, never reported as a final result.
- **Measurement Result**: An on-device outcome for a candidate — correctness
  status plus one or more rounds of performance measurement at increasing
  statistical rigor.
- **Search Budget**: The declared cap on total real on-device measurements
  this feature's search may consume.
- **Optimal Configuration**: The final recommended `8da4w` candidate
  (loop structure + tile/subgroup geometry), backed by a full statistically-
  sound performance measurement and a passing correctness check.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The feature states, with numeric evidence, which loop
  structure is fastest for `8da4w` at the shipped geometry, explicitly
  confirming or refuting the user-reported `dbuf2` claim.
- **SC-002**: The process identifies an `8da4w` tile/subgroup configuration
  whose measured throughput is at least as good as the currently-shipped
  configuration, while directly measuring on real hardware no more than
  15% of the full legal `8da4w` configuration space, and no more than 30
  measurements in absolute terms (target: far fewer).
- **SC-003**: Zero configurations that fail the correctness check appear in
  the final performance ranking or report.
- **SC-004**: The final recommended `8da4w` configuration is reported with
  an explicit head-to-head comparison against both the currently-shipped
  `8da4w` baseline and `4w`'s 128×64/K16/2×2/s32 winner, backed by a
  3-run-mean, CoV<5% measurement — this workstream's existing statistical
  rigor bar.
- **SC-005**: For any configuration in the swept `8da4w` search universe,
  the workstream engineer can determine why it was or wasn't taken to
  on-device measurement without re-running the search.
- **SC-006**: The end-to-end process consumes measurably less M5 EVT1
  device time than an exhaustive, fully-rigorous measurement of the entire
  legal configuration space would require (at least a 5x reduction).

## Assumptions

- M5 EVT1 is the target device for all measurements in this feature, per
  this workstream's active-target constraint; device availability and
  driver identity are re-verified before use, not assumed.
- "Optimal" means highest FLOP-weighted throughput across the `wq` +
  `w1_gate` ops for each of the three target models (1B, 3B, 8B) — the same
  6-shape set `022` and `023` used — measured at the standard pinned-clock
  configuration, consistent with `022`'s methodology.
- The `8da4w`-specific legal configuration space is smaller than or
  different from `4w`'s 642 candidates because int8 cooperative-matrix
  accumulation has different register/shared-memory footprint per tile than
  `4w`'s fp16 accumulation; this feature re-derives that space rather than
  reusing `4w`'s enumeration (FR-002).
- Loop structure (`dbuf1`-`dbuf4`) and tile/subgroup geometry are treated as
  separable axes for this feature — User Story 1 fixes the former before
  User Story 2/3 sweep the latter — on the same reasoning `023` used to
  isolate loop structure from geometry. If the re-confirmation in User
  Story 1 finds meaningful interaction between loop structure and geometry
  (i.e., the best loop structure changes at a different geometry), that
  finding is reported explicitly as a limitation, not silently absorbed.
- The existing small-shape, fp32-reference correctness check already used
  in this workstream for `8da4w` is reused as-is; this feature does not
  define a new correctness methodology.
- This workstream's existing device-safety practices (driver-identity
  re-verification, halting on drift, checking shared-device availability)
  apply unchanged.
- This is an internal engineering capability for this workstream's own use;
  "user"/"engineer" throughout this spec refers to the workstream engineer
  running the sweep, not an end product user.
- This feature's result is an input to `specs/024-8da4w-slower-than-4w`'s
  broader investigation but does not itself close that feature — finding
  `8da4w`'s optimal tile/subgroup configuration does not, by itself,
  explain why `8da4w` underperforms `4w` if the gap persists at the new
  optimum.
