# Feature Specification: Smart Autotuning for q4gsw CoopMat Tile Configuration on M5 EVT1

**Feature Branch**: `022-linear-coopmat-autotune`

**Created**: 2026-07-07

**Status**: Complete — closed 2026-07-08, desired results obtained (18/29 tasks done; remaining tasks were report/polish work not needed)

**Input**: User description: "given the 321 combinations, cameup a smart (no not try all) way of autotune the shader. find the optimal configuration that yield the optimal result on M5 EVT1."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Prune the search space with zero device time (Priority: P1) 🎯 MVP

As the workstream engineer, I want the 642 valid buffer-storage tile
configurations for the `linear_q4gsw_coopmat` shader ranked and narrowed
down to a small shortlist using only hardware-derived analytical signals (no
on-device runs at all), so that I know which candidates are worth spending
M5 EVT1 device time on before any benchmark is executed.

**Why this priority**: This is the step that actually avoids "trying all
642" — everything downstream depends on the shortlist being both small and
defensible. Without it, the feature degenerates into either exhaustive
search (too slow, burns shared-device time) or ungrounded guessing.

**Independent Test**: Given the enumerated 642 valid buffer-storage
configurations (tile size, subgroup grid, subgroup size, all already
constraint-checked for legality on this shader), produce a ranked shortlist
whose size is a small fraction of 642, with a documented reason for every
config's inclusion or exclusion, using zero M5 EVT1 measurements.

**Acceptance Scenarios**:

1. **Given** the full list of 642 valid buffer-storage configurations,
   **When** the pruning stage runs, **Then** it outputs a shortlist that is
   materially smaller than the input set (target: on the order of tens, not
   hundreds) with a ranking rationale per candidate, and zero device
   measurements have been taken.
2. **Given** the already-known production configuration (dbuf1) and the
   previously-identified sweep winner (128×64/K16/2×2/s32), **When** the
   shortlist is produced, **Then** both of those configurations appear on it
   (as sanity-check anchors), even if the analytical ranking would not have
   surfaced them independently.

---

### User Story 2 - Find the best performer without measuring everything on the shortlist (Priority: P2)

As the workstream engineer, I want the shortlisted candidates measured on
M5 EVT1 using a staged approach that spends more measurement effort on
promising candidates and drops weak ones early, so that the total number of
real on-device benchmark runs stays small while I still end up with high
confidence in which candidate is fastest.

**Why this priority**: This is where the actual device-time savings are
realized. P1 only avoids wasting analysis effort; P2 is what avoids wasting
scarce, shared M5 EVT1 time on candidates that are clearly not competitive
after a first look.

**Independent Test**: Run the staged search over the P1 shortlist and
confirm that most candidates are eliminated after a small, cheap
measurement, while only a handful of top contenders receive the full,
statistically-sound measurement — with the total number of on-device runs
substantially below "every shortlisted candidate measured to full rigor."

**Acceptance Scenarios**:

1. **Given** the P1 shortlist, **When** the staged search executes, **Then**
   every candidate receives at least one cheap measurement, only the
   top-performing subset receives additional, more rigorous measurement, and
   candidates that fail to compile or fail correctness are dropped
   immediately without consuming further budget.
2. **Given** two or more candidates that are statistically indistinguishable
   at the end of the search, **When** the process selects a winner, **Then**
   it applies a documented, reasonable tie-breaking rule (e.g., prefer the
   smaller resource footprint) rather than picking arbitrarily.

---

### User Story 3 - Validate and report the recommended configuration (Priority: P3)

As the workstream engineer, I want the final recommended configuration
confirmed with this workstream's full correctness check and statistically
sound performance measurement, and reported side-by-side with the current
production configuration and the previously-known sweep winner, so the
recommendation is trustworthy enough to act on (productionize, or explicitly
keep the status quo).

**Why this priority**: A fast search that produces an unvalidated or
unclearly-reported answer isn't actionable. This closes the loop from
"candidate looks good" to "here is a decision-ready recommendation."

**Independent Test**: Take the winning candidate from User Story 2 and
confirm it independently passes the correctness check and produces a final
report comparing it against dbuf1 (production) and the 128×64/K16/2×2/s32
sweep winner.

**Acceptance Scenarios**:

1. **Given** a candidate selected as the search's top performer, **When**
   final validation runs, **Then** the candidate's correctness is confirmed
   and its performance is reported with a comparison table against the
   production baseline and the prior sweep winner.
2. **Given** a search where no candidate beats the existing production
   baseline within budget, **When** the final report is produced, **Then**
   it states this outcome explicitly and recommends keeping the current
   baseline, rather than presenting a marginal or unproven result as a win.

---

### Edge Cases

- What happens when a shortlisted candidate fails to compile (as happened
  with the 128×64/K16/4×4/s32 attempt during this workstream's own manual
  exploration)? The process must record the failure and move on without
  aborting the overall search.
- What happens when the device or driver state becomes untrustworthy
  mid-search (shared M5 EVT1 board drifts to an unexpected driver build, or
  another teammate's job starts running)? The process must detect this and
  halt or re-verify rather than continue producing measurements under an
  unknown state.
- What happens when the analytical pruning stage's shortlist does not
  include the already-known sweep winner? It must be added back in as a
  sanity-check anchor regardless of its analytical rank (see User Story 1,
  Acceptance Scenario 2).
- What happens when two or more candidates are statistically
  indistinguishable at the top of the search? A documented tie-breaking rule
  is applied (see User Story 2, Acceptance Scenario 2) rather than an
  arbitrary choice.
- What happens if none of the 642 valid configurations beat the current
  production configuration? The final report says so explicitly (see User
  Story 3, Acceptance Scenario 2) instead of forcing a "better" result.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The autotuning process MUST restrict its search universe to
  the buffer-storage-only, coopmat-eligible tile configuration space (the
  642 previously enumerated valid combinations) and MUST NOT take a full,
  statistically-sound on-device measurement of every one of them.
- **FR-002**: The process MUST include a pruning/ranking stage that uses
  hardware-derived analytical signals (e.g., estimated occupancy,
  shared-memory footprint, thread count, accumulator-register count per
  subgroup) to reduce the 642 valid configurations to a materially smaller
  shortlist before any on-device measurement is taken.
- **FR-003**: The process MUST measure the shortlisted candidates using a
  staged approach that allocates progressively more measurement effort to
  promising candidates and eliminates clearly weaker candidates early,
  rather than measuring every shortlisted candidate with full statistical
  rigor from the start.
- **FR-004**: The process MUST confirm numerical correctness for a
  candidate before its performance number is trusted, ranked, or reported.
- **FR-005**: The process MUST report the final recommended configuration
  together with a head-to-head comparison against the current production
  configuration (dbuf1) and the previously identified sweep winner
  (128×64/K16/2×2/s32).
- **FR-006**: The process MUST operate within a bounded, pre-declared cap on
  the total number of real on-device measurements, substantially smaller
  than the full 642-configuration space.
- **FR-007**: The process MUST detect signs of an untrustworthy measurement
  environment (unexpected driver state, shared device unexpectedly busy) and
  halt or re-verify rather than silently continue producing results under
  those conditions.
- **FR-008**: The process MUST record, for every configuration it does not
  take to full on-device measurement, a documented reason it was
  deprioritized or excluded, so the decision can be audited later without
  re-running the search.
- **FR-009**: If no evaluated configuration outperforms the current
  production baseline within the search budget, the process MUST report
  that outcome explicitly rather than present an unproven or marginal
  result as an improvement.

### Key Entities

- **Configuration Candidate**: One point in the tile-geometry search space
  (output tile height/width, K-step, subgroup grid shape, subgroup size),
  always using buffer weight storage; carries derived properties
  (shared-memory footprint, thread count, accumulator count) and a
  validity/compile status.
- **Analytical Score**: A pre-measurement estimate of a candidate's likely
  relative performance, derived from hardware-derived proxies; used only to
  rank and prune candidates, never reported as a final performance result.
- **Measurement Result**: An on-device outcome recorded for a candidate —
  correctness status, and one or more rounds of performance measurement at
  increasing statistical rigor.
- **Search Budget**: The declared cap on the total number of real,
  on-device measurements the process may consume during a run.
- **Optimal Configuration**: The final recommended candidate, backed by a
  full statistically-sound performance measurement and a passing correctness
  check.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The process identifies a configuration whose measured
  throughput is at least as good as the best previously-known configuration,
  while directly measuring on real hardware no more than 15% of the 642
  valid configurations (target: far fewer).
- **SC-002**: The end-to-end process (pruning, staged search, final
  validation) consumes measurably less M5 EVT1 device time than an
  exhaustive, fully-rigorous measurement of all 642 configurations would
  require — at least a 5x reduction in total on-device measurement time.
- **SC-003**: Zero configurations that fail the correctness check appear in
  the final performance ranking or report.
- **SC-004**: The final recommended configuration's performance claim is
  reproducible under this workstream's existing statistically-sound
  benchmarking standard and is reported with an explicit comparison against
  the current production configuration.
- **SC-005**: For any configuration in the 642-item search universe, the
  workstream engineer can determine why it was or wasn't taken to on-device
  measurement without re-running the search.

## Assumptions

- The 642 previously-enumerated, buffer-storage-only, constraint-valid tile
  geometries (output tile M/N, K-step, subgroup grid, subgroup size — all
  already checked against known thread-count, MMA-alignment,
  shared-memory-staging, and shared-memory-capacity constraints for this
  shader) constitute the full search universe for this feature. Loop
  structure (the dbuf1/2/3/4 family) is held fixed at the dbuf1
  ("prefetch-first") shape already used by this configuration space; varying
  loop structure is out of scope for this feature.
- "Optimal" means the highest FLOP-weighted throughput across the
  production Llama prefill shape set already used by this workstream's
  benchmark harness, measured at this workstream's standard pinned clock
  configuration on the M5 EVT1, consistent with existing measurement
  methodology.
- The existing small-shape, fp32-reference correctness check already used in
  this workstream is reused as-is for the correctness gate; this feature
  does not need to define a new correctness methodology.
- The already-known dbuf1 production configuration and the 128×64/K16/2×2/s32
  sweep winner serve as sanity-check anchors that the search is expected to
  at least match, not exceed by a guaranteed margin.
- This workstream's existing device-safety practices (re-verifying driver
  identity and device availability before measurement, halting on drift)
  apply unchanged; this feature does not need to invent new device-safety
  mechanisms beyond what halting/re-verification already requires.
- This is an internal engineering capability for this workstream's own use;
  "user" throughout this spec refers to the workstream engineer running the
  autotuning process, not an end product user.
