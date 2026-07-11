# Feature Specification: 8da4w Tile/Subgroup Sweep Ranked by End-to-End Throughput

**Feature Branch**: `027-e2e-tile-sweep`

**Created**: 2026-07-11

**Status**: Draft

**Input**: User description: "perform a sweep on e2e, we use e2e as the number for winner, perform a parameter sweep with e2e. (smartly)" — redo the `8da4w` (`dq8ca_q4gsw` int8 WMMA) tile/subgroup search, but rank candidates by real end-to-end model throughput instead of isolated-kernel microbenchmark GFLOP/s, since `specs/026` just found the Tier-1 (microbenchmark) winner is actually slower end-to-end.

## Context (why this feature exists now)

`specs/025` and `specs/026` each found a tile/subgroup configuration that won the isolated
`linear_dq8ca_q4gsw` microbenchmark (1731–2207 GFLOP/s, progressively "better" across the
two features). But `specs/026`'s own Tier-2 e2e validation — run on this session, on the
correct shape-matched model after an initial shape-mismatched false start — found its
microbenchmark winner is **~2.7% slower end-to-end** than the currently-shipped
configuration, which itself was never the fastest candidate on the isolated microbenchmark
either. In other words: two rounds of microbenchmark-ranked search have not yet produced a
config that is *actually* faster in the metric that matters (per this workstream's own
constitution: "e2e is the deliverable, microbench is for analysis").

This feature closes that gap directly: instead of ranking candidates by isolated-kernel
GFLOP/s and only spot-checking the winner against e2e afterward (the pattern `025`→`026`
followed, which just failed twice), rank candidates by real end-to-end throughput
throughout the search. The user's explicit ask is "(smartly)" — a full end-to-end run
(model load + 2048-token prefill) costs tens of seconds per candidate on a shared device,
and this workstream's re-derived legal tile/subgroup/subgroup-size space is estimated at
roughly 1000+ candidates (`specs/026` research.md) — an exhaustive e2e sweep over that
space is not viable. This feature must use a staged approach: cheap, already-available
microbenchmark data (from `specs/025`/`specs/026`) narrows the field first; real e2e
measurement — on the shape-matched model, the mistake `specs/026` had to catch and correct
— is spent only on a small, deliberately chosen shortlist.

Related prior work, not yet the answer to this question:

- `specs/022` (4w) and `specs/025`/`specs/026` (8da4w) all found their respective winners
  via isolated-kernel microbenchmark ranking; none of them ran a tile/subgroup sweep with
  e2e as the primary ranking signal. `specs/026`'s Tier-2 check was a single post-hoc
  validation of one candidate, not a search.
- `specs/026`'s Tier-2 validation already surfaced the exact methodology hazard this
  feature must avoid: its first e2e check used a shape-mismatched model (1B) against a
  config found on 8B-shaped microbenchmark data, producing a wrong (too pessimistic)
  result that had to be re-run on the correct model before it could be trusted. Any e2e
  measurement this feature takes must be on a model whose per-layer shapes are the ones
  the ranking is meant to represent, stated explicitly, not assumed.
- `specs/024-8da4w-slower-than-4w` is a separate, broader investigation (why `8da4w`
  underperforms `4w` end-to-end); this feature's result is an input to it, not a
  replacement — even a real e2e-ranked `8da4w` improvement would not by itself close that
  gap if `4w` remains faster in absolute terms.

## Clarifications

### Session 2026-07-11

- Q: How many top candidates should User Story 1's initial shortlist take to real e2e
  measurement? → A: 8 candidates by combined `specs/025`/`specs/026` microbenchmark rank.
- Q: What statistical bar decides an e2e "win" over baseline? → A: Adaptive/staged, not a
  flat 3-run-for-everyone rule: every candidate gets one screening run first; only a
  candidate whose single-run result is close to or ahead of the shipped baseline (within
  10% of baseline, or faster) is escalated to a 3-run confirmation. Candidates far behind
  baseline on their screening run are not re-run — this is proportional effort, not reduced
  rigor: a candidate only needs statistical confirmation once it's plausibly a real
  contender, matching the "(smartly)" instruction.

Beyond the above, scope and methodology constraints (staged/smart search, not exhaustive;
shape-matched e2e measurement; reuse of existing microbenchmark data as a pre-filter) are
directly determined by the user's request plus the concrete lessons `specs/026` already
surfaced in this same session. There is no reasonable alternative interpretation of
"(smartly)" other than "don't brute-force the full space with expensive e2e runs" —
restated as FR-002/FR-009 below.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Rank the existing microbenchmark shortlist by real e2e throughput (Priority: P1) 🎯 MVP

As the workstream engineer, I want the top 8 tile/subgroup candidates by combined
`specs/025`/`specs/026` microbenchmark rank re-ranked by actual end-to-end model throughput — not isolated GFLOP/s — on the
shape-matched model, so I can see directly whether the microbenchmark ranking predicts
the e2e ranking at all, using data that's already cheap to act on (no new shader variants
or builds needed for this story).

**Why this priority**: This is the fastest way to get a real answer with the search
infrastructure that already exists. `specs/026` already has 2 correctness-verified
candidates and `specs/025` has 25 correctness-verified candidates with microbenchmark
scores — running e2e on even a handful of these immediately tells us whether "fastest on
microbenchmark" and "fastest end-to-end" are the same ranking on this hardware, which is
the central open question after `specs/026`'s single-point finding that they disagree at
least once.

**Independent Test**: Take the top 8 candidates by microbenchmark score from
`specs/025`+`specs/026`'s combined results, run each through a full e2e prefill
measurement on the shape-matched model, and confirm the resulting e2e ranking is recorded
with enough evidence (run count, model/shape used, driver/clock state) to trust it as more
than a single anecdote.

**Acceptance Scenarios**:

1. **Given** the combined `specs/025`+`specs/026` candidate set with existing
   microbenchmark scores, **When** the top 8 are re-measured end-to-end, **Then** each
   candidate's e2e result explicitly states which model/shape it was measured on and why
   that model is shape-representative of the candidate's own microbenchmark shapes.
2. **Given** the resulting e2e ranking, **When** compared to the microbenchmark ranking,
   **Then** the feature states plainly whether the two rankings agree, partially agree, or
   disagree — not just reporting the new winner in isolation.

---

### User Story 2 - Smartly extend the search beyond the existing shortlist if the top candidates don't clearly win (Priority: P2)

As the workstream engineer, I want the search to expand to additional, previously-unmeasured
tile/subgroup candidates — chosen using the existing analytical scoring model plus what
User Story 1 learns about how well microbenchmark rank predicts e2e rank — only if User
Story 1's top candidates fail to beat the currently-shipped baseline end-to-end, so device
time is spent proportionally to how hard the answer turns out to be to find.

**Why this priority**: If User Story 1 already finds a real e2e winner, there's no need to
build and measure more shader variants — this story only fires when it's actually needed,
which is the concrete meaning of "smartly" once User Story 1's outcome is known.

**Independent Test**: Given User Story 1's outcome, confirm that no additional shader
variants are built/measured if a real e2e improvement was already found, and that if none
was found, a small, explicitly-bounded number of new candidates (informed by, not
identical to, the existing analytical ranking) are built and measured end-to-end next.

**Acceptance Scenarios**:

1. **Given** User Story 1 finds a candidate that beats the shipped baseline end-to-end with
   statistically meaningful margin, **When** this feature concludes, **Then** no new shader
   variants are built and the search stops at that winner.
2. **Given** User Story 1's top candidates all fail to beat the shipped baseline end-to-end,
   **When** the search extends, **Then** it selects new candidates using a documented
   rationale (not arbitrary), stays within a small, pre-declared additional device-time
   budget, and still measures every new candidate end-to-end (not by isolated
   microbenchmark alone) before it can be reported as a contender.

---

### User Story 3 - Report a definitive e2e-ranked answer, including "nothing beats baseline" as a valid outcome (Priority: P1)

As the workstream engineer, I want one clear, final answer to "what is the actual e2e
winner for `8da4w` tile/subgroup configuration right now" — either a specific
configuration with real e2e evidence of an improvement, or an explicit, evidence-backed
statement that the currently-shipped baseline remains the best-known e2e configuration —
so that this question (asked directly in this session) has a definitive, reusable answer
instead of remaining an open thread across `specs/025`/`specs/026`.

**Why this priority**: This is the feature's actual deliverable. Both prior sweeps ended
without a clear e2e verdict; this feature must not repeat that pattern.

**Independent Test**: Read the final report and confirm it states, unambiguously, either
(a) a specific winning token with e2e tok/s evidence and a percentage improvement over the
shipped baseline, or (b) an explicit statement that no measured candidate beat the shipped
baseline end-to-end, with the search's scope stated so the reader knows what was and
wasn't covered.

**Acceptance Scenarios**:

1. **Given** the completed search (User Stories 1–2), **When** the final report is
   produced, **Then** it names one unambiguous e2e winner — which may be the currently-
   shipped baseline itself — never leaving the question open or split across multiple
   partial results.
2. **Given** the winner is a candidate other than the shipped baseline, **When** reported,
   **Then** it includes e2e tok/s for both prefill (and decode, if measured), the model/
   shape used, run count, and the percentage improvement, with correctness already
   confirmed by the existing microbenchmark correctness gate (Constitution Principle I) —
   an e2e-fast but not-yet-correctness-verified candidate is never reported as a winner.

---

### Edge Cases

- What happens if a candidate that wins on microbenchmark also wins e2e, but only by a
  margin comparable to e2e run-to-run noise? The feature applies the adaptive statistical
  bar from Clarifications (single screening run for every candidate; 3-run confirmation
  only for candidates within 10% of, or ahead of, the shipped baseline) before calling it a
  real win, not a single-run comparison for a close result.
- What happens if the "shape-matched model" choice itself is ambiguous for a candidate
  measured across multiple shapes (the 6-shape `wq`+`w1_gate` × {1B,3B,8B} convention)?
  The feature states which model(s) each candidate was validated against and does not
  extrapolate an e2e verdict measured on one model size to claim a result for a different,
  unmeasured model size.
- What happens if extending the search (User Story 2) still finds nothing that beats the
  shipped baseline? The feature reports this explicitly as the answer (per User Story 3),
  not as an inconclusive or omitted result — "the shipped baseline is still the e2e
  winner" is a complete, valid answer to this feature's question.
- What happens if the shared M5 EVT1 device drifts to an unexpected driver build mid-search?
  The process halts or re-verifies rather than continuing under unknown state, per this
  workstream's existing device-safety practice.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The process MUST rank tile/subgroup candidates by real end-to-end model
  throughput (prefill tok/s at minimum; decode tok/s if measured) as the primary decision
  metric, not by isolated-kernel microbenchmark GFLOP/s — the latter may be used only as a
  pre-filter (FR-002), never as the reported winner-selection criterion.
- **FR-002**: The process MUST use the existing `specs/025`/`specs/026` microbenchmark
  scores and correctness results as a cheap pre-filter to select which candidates receive
  real (expensive) e2e measurement, rather than measuring every candidate in the legal
  tile/subgroup/subgroup-size space end-to-end.
- **FR-003**: Every e2e measurement MUST state which model (1B/3B/8B) and PTE it was taken
  on, and MUST use a model whose per-layer matmul shapes match the shapes the candidate's
  own microbenchmark/correctness data represents — an e2e measurement on a shape-mismatched
  model (the specific error caught and corrected in `specs/026`'s Tier-2 check) MUST NOT be
  reported as authoritative evidence for that candidate.
- **FR-004**: The process MUST NOT report an e2e winner whose correctness has not already
  been confirmed by the existing microbenchmark correctness gate (Constitution Principle I)
  at every representative shape.
- **FR-005**: The process MUST apply the adaptive statistical bar from Clarifications: every
  candidate first gets one e2e screening run; only a candidate whose screening run is within
  10% of, or faster than, the shipped baseline is escalated to a 3-run confirmation (mean
  compared against the baseline's own multi-run range, not a single baseline data point) —
  a candidate's screening-run result alone, or a candidate far behind baseline, MUST NOT be
  reported as a win or a loss requiring further measurement; only a confirmed, 3-run result
  for a close-or-ahead candidate MUST be reported as a win.
- **FR-006**: The process MUST only extend the search to new, previously-unmeasured
  tile/subgroup candidates (User Story 2) if the initial shortlist (User Story 1) fails to
  beat the shipped baseline end-to-end with the FR-005 statistical bar — the search MUST
  NOT build/measure additional shader variants once a real e2e winner is already found.
- **FR-007**: If a search extension (User Story 2) occurs, it MUST select new candidates
  using a documented rationale derived from the existing analytical scoring model and what
  User Story 1 learned about microbenchmark-vs-e2e rank agreement, and MUST stay within a
  small, pre-declared additional device-time budget.
- **FR-008**: The process MUST produce exactly one final, unambiguous answer to "what is
  the current e2e winner" — either a specific candidate with full e2e evidence, or an
  explicit statement that the currently-shipped baseline remains the best-known
  configuration end-to-end — never leaving the question split across multiple inconclusive
  results.
- **FR-009**: The process MUST operate within a bounded, pre-declared total device-time/
  measurement budget across both user stories, reflecting the user's "(smartly)"
  instruction — the process MUST justify, for every candidate not taken to e2e
  measurement, why it was skipped (e.g., "below the microbenchmark pre-filter cutoff"),
  consistent with this workstream's existing auditability convention (`specs/025`/`026`
  FR-009-equivalent).
- **FR-010**: The process MUST detect signs of an untrustworthy measurement environment
  (unexpected driver state, shared device unexpectedly busy) and halt or re-verify rather
  than silently continuing.
- **FR-011**: The process MUST explicitly compare its final e2e winner against the
  currently-shipped baseline's own e2e throughput (not just against other swept
  candidates), since this workstream's own recent finding (`specs/026`) is that the
  shipped baseline currently IS the best-known e2e configuration — any new claimed winner
  must be shown to beat that specific, real number.

### Key Entities

- **Microbenchmark Pre-Filter Score**: The existing `specs/025`/`specs/026` analytical
  score and/or measured microbenchmark GFLOP/s for a candidate, used only to decide which
  candidates are worth spending e2e device time on (FR-002) — never itself the reported
  ranking metric.
- **E2E Measurement**: A real end-to-end prefill (and optionally decode) tok/s result for
  one candidate on one specific model/PTE, carrying its stage (`screening` = 1 run,
  `confirmed` = 3-run mean triggered by a close-or-ahead screening result), run count,
  model/shape identity, driver hash, and clock-pin state — the unit this feature ranks
  candidates by.
- **Shortlist**: The initial, cheaply-selected set of candidates taken to e2e measurement
  in User Story 1 — small by construction (FR-002/FR-009), not the full legal space.
- **Search Extension Budget**: The small, pre-declared additional device-time/measurement
  allowance available to User Story 2, spent only if the initial shortlist doesn't produce
  a winner (FR-006/FR-007).
- **E2E Winner**: The feature's final answer — either a specific tile/subgroup/subgroup-
  size candidate with a confirmed, statistically-meaningful e2e improvement over the
  shipped baseline, or the shipped baseline itself, explicitly stated either way (FR-008).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The feature states one unambiguous e2e winner (a specific candidate, or the
  shipped baseline) with no open or split answer remaining.
- **SC-002**: Every e2e measurement used to support the final answer identifies its
  model/PTE and shape-representativeness explicitly; zero e2e results are reported without
  this context.
- **SC-003**: If a new winner is reported, its e2e improvement over the shipped baseline is
  backed by a 3-run confirmation (per the Clarifications adaptive bar — reached only
  because its screening run was within 10% of, or ahead of, baseline) with a stated margin
  clearly outside normal run-to-run noise — not a single-run comparison.
- **SC-004**: The total number of candidates taken to real e2e measurement is small relative
  to the full legal tile/subgroup/subgroup-size space (target: single-digit to low tens, not
  hundreds) — the process explicitly reports this count and the device-time it consumed.
- **SC-005**: The feature explicitly states, for every candidate not taken to e2e
  measurement, why it was excluded — traceable without re-running the search.
- **SC-006**: The relationship between microbenchmark rank and e2e rank (agree / partially
  agree / disagree) is stated explicitly as a finding, independent of which specific
  configuration wins — since this is itself new information this workstream has been
  missing across `specs/025`/`specs/026`.

## Assumptions

- M5 EVT1 is the target device (either board), per this workstream's active-target
  constraint; device availability and driver identity are re-verified before use.
- "Shape-matched model" means: for candidates whose microbenchmark data used the 8B-shaped
  representative shapes (K=4096/14336, this workstream's standard convention), e2e
  measurement uses the Llama 3.1 8B `8da4w` buffer PTE — the same correction `specs/026`'s
  Tier-2 validation already had to make. If a future candidate's microbenchmark data is
  keyed to a different model size, its e2e validation uses that matching model instead;
  this is stated per-candidate, not assumed globally.
- The existing `dbuf2` loop structure (confirmed by `specs/025`) and the existing
  tile/subgroup shader-variant infrastructure (`linear_dq8ca_q4gsw_coopmat_tsweep.{glsl,yaml}`,
  extended by `specs/026`) are reused as-is; this feature does not re-derive loop structure
  or build a new dispatch mechanism, only adds an e2e-measurement stage on top.
- The standard 2048-token-prefill workload (this workstream's default) is used for e2e
  measurement; decode-phase throughput may be measured opportunistically but prefill is the
  primary metric, matching `specs/026`'s Tier-2 check.
- This workstream's existing device-safety practices (driver-identity re-verification,
  halting on drift, checking shared-device availability) apply unchanged.
- This is an internal engineering capability for this workstream's own use; "user"/
  "engineer" throughout this spec refers to the workstream engineer running the sweep.
- A confirmed e2e winner, if found, does not itself get shipped/promoted to the default
  dispatch by this feature — that remains a separate follow-on decision, consistent with
  `specs/025`/`specs/026`'s own precedent of reporting Tier-1/Tier-2 evidence without
  unilaterally changing production defaults.
