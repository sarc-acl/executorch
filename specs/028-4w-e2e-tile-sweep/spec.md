# Feature Specification: 4w Tile/Subgroup Sweep Ranked by End-to-End Throughput

**Feature Branch**: `028-4w-e2e-tile-sweep`

**Created**: 2026-07-12

**Status**: Draft

**Input**: User description: "start e2e parameter sweep on 4w on dev branch, just like how last spec was did. We care about e2e speedup with all the existing full stack optimizations we have"

## Context (why this feature exists now)

`specs/027` redid the `8da4w` tile/subgroup search ranked by real end-to-end throughput
instead of isolated-kernel microbenchmark GFLOP/s, because `specs/026` found its own
microbenchmark winner was actually an e2e regression. That e2e-ranked search found a real
winner (`tsweep_t64x32k32g12s64`, +12.5% Llama 3.1 8B prefill) and it shipped to `dev` on
2026-07-12 (commit `42aabb4e0`).

`4w` (`linear_q4gsw_coopmat`) has never been through this same e2e-ranked process.
`specs/022` found `4w`'s currently-shipped tile configuration via a smart, zero-device-time-
pruned *microbenchmark* search (642 candidates → shortlist → 3 measurement rounds), and that
config (dbuf1, per `ACTIVE-STATUS.md`) has been production ever since — but, per
`ACTIVE-STATUS.md`'s own note from today, it "has never been e2e-validated either, and per
this same-session `8da4w` finding, likely shouldn't be trusted without one." This feature
closes that gap for `4w`, using the same staged, e2e-primary methodology `027` already
validated works on this hardware, and measuring against the full stack of existing
optimizations already shipped on `dev` (WMMA coopmat linear, SDPA coopmat, the node-threshold
watchdog workaround, etc.) rather than in isolation.

Related prior work, not yet the answer to this question:

- `specs/022` (4w microbenchmark-ranked autotune, shipped as today's production default) —
  this feature re-ranks its shortlist/candidates by e2e instead of taking the microbenchmark
  winner on faith.
- `specs/025`/`026`/`027` (the `8da4w` analogue of this exact question) — `027`'s staged
  methodology (microbenchmark pre-filter → screening run → 3-run confirmation only for
  close-or-ahead candidates → bounded search extension only if needed) is the directly
  reusable template for this feature; `4w`'s pre-filter data source is `022`'s results
  instead of `025`/`026`'s.
- `specs/024-8da4w-slower-than-4w` — established `4w` is currently faster than `8da4w`
  end-to-end; this feature does not change that comparison, it only asks whether `4w`
  itself can go faster than its own current shipped configuration.

## Clarifications

The following defaults are carried over directly from `specs/027`'s already-validated
methodology and this workstream's standing conventions (Assumptions below); most are not
re-litigated here because the user's request ("just like how last spec was did") explicitly
asks to reuse that approach.

### Session 2026-07-12

- Q: Should 1B/3B receive the same full staged sweep (screening + 3-run confirmation)
  independently, or should they only validate the 8B-derived winner after the fact? → A:
  Run the full staged search on 8B only (as originally scoped); once an 8B e2e winner (or
  "baseline stands") is confirmed, validate that same result end-to-end on 1B and 3B as a
  confirmation pass, not a separate search.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Rank the existing `4w` microbenchmark shortlist by real e2e throughput (Priority: P1) 🎯 MVP

As the workstream engineer, I want the top tile/subgroup candidates from `specs/022`'s
already-measured `4w` shortlist re-ranked by actual end-to-end model throughput — not
isolated GFLOP/s — on the shape-matched model, with all currently-shipped full-stack
optimizations enabled (WMMA coopmat SDPA, node-threshold watchdog workaround, etc.), so I can
see directly whether `022`'s microbenchmark ranking predicts the e2e ranking, before spending
any device time on new shader variants.

**Why this priority**: `022` already produced correctness-verified, scored candidates; running
e2e on the top handful is the fastest way to know whether the shipped `4w` config is actually
the e2e winner, mirroring exactly what `027` did for `8da4w`.

**Independent Test**: Take the top candidates by microbenchmark score from `022`'s results,
run each through a full e2e prefill measurement (2048-token prefill, Llama 3.1 8B, full stack
of existing `dev` optimizations enabled) and confirm the resulting e2e ranking is recorded with
enough evidence (run count, model/shape, driver/clock state) to trust it.

**Acceptance Scenarios**:

1. **Given** `022`'s scored candidate set, **When** the top candidates are re-measured
   end-to-end, **Then** each candidate's e2e result states which model/shape it was measured
   on and confirms that shape matches the shapes `022`'s own microbenchmark data represents.
2. **Given** the resulting e2e ranking, **When** compared to `022`'s microbenchmark ranking,
   **Then** the feature states plainly whether the two rankings agree, partially agree, or
   disagree — not just reporting the new winner in isolation.

---

### User Story 2 - Smartly extend the search beyond the existing shortlist if the top candidates don't clearly win (Priority: P2)

As the workstream engineer, I want the search to expand to additional, previously-unmeasured
`4w` tile/subgroup candidates — chosen using `022`'s analytical scoring model plus what User
Story 1 learns about microbenchmark-vs-e2e rank agreement — only if User Story 1's top
candidates fail to beat the currently-shipped baseline end-to-end, so device time is spent
proportionally to how hard the answer turns out to be to find.

**Why this priority**: Mirrors `027` User Story 2 — no new shader variants get built/measured
unless the existing shortlist already fails to produce a winner.

**Independent Test**: Given User Story 1's outcome, confirm that no additional shader variants
are built/measured if a real e2e improvement was already found, and that if none was found, a
small, explicitly-bounded number of new candidates are built and measured end-to-end next.

**Acceptance Scenarios**:

1. **Given** User Story 1 finds a candidate that beats the shipped baseline end-to-end with
   statistically meaningful margin, **When** this feature concludes, **Then** no new shader
   variants are built and the search stops at that winner.
2. **Given** User Story 1's top candidates all fail to beat the shipped baseline end-to-end,
   **When** the search extends, **Then** it selects new candidates using a documented
   rationale, stays within a small, pre-declared additional device-time budget, and still
   measures every new candidate end-to-end before it can be reported as a contender.

---

### User Story 3 - Report a definitive e2e-ranked answer for `4w`, including "nothing beats baseline" as a valid outcome (Priority: P1)

As the workstream engineer, I want one clear, final answer to "what is the actual e2e winner
for `4w` tile/subgroup configuration right now, with the full optimization stack enabled" —
either a specific configuration with real e2e evidence of an improvement, or an explicit,
evidence-backed statement that the currently-shipped baseline remains the best-known e2e
configuration — so this question has a definitive, reusable answer instead of `4w`'s shipped
default continuing to ride on unvalidated microbenchmark evidence.

**Why this priority**: This is the feature's actual deliverable, mirroring `027` User Story 3.

**Independent Test**: Read the final report and confirm it states, unambiguously, either
(a) a specific winning tile/subgroup config with e2e tok/s evidence and a percentage
improvement over the shipped baseline, measured with the full existing optimization stack
enabled, or (b) an explicit statement that no measured candidate beat the shipped baseline
end-to-end, with the search's scope stated.

**Acceptance Scenarios**:

1. **Given** the completed search (User Stories 1–2), **When** the final report is produced,
   **Then** it names one unambiguous e2e winner — which may be the currently-shipped baseline
   itself — never leaving the question open or split across multiple partial results.
2. **Given** the winner is a candidate other than the shipped baseline, **When** reported,
   **Then** it includes e2e tok/s for both prefill (and decode, if measured), the model/shape
   used, run count, and the percentage improvement over the shipped baseline, with correctness
   already confirmed by the existing microbenchmark correctness gate — an e2e-fast but
   not-yet-correctness-verified candidate is never reported as a winner.

---

### User Story 4 - Confirm the 8B-derived answer holds on 1B and 3B (Priority: P2)

As the workstream engineer, I want the final 8B e2e answer from User Story 3 (a new winning
config, or "baseline stands") re-measured end-to-end on the 1B and 3B models too, so the
result isn't silently assumed to generalize beyond the 8B shapes it was actually found on —
the same gap `specs/027` flagged as a non-blocking follow-up for `8da4w` and left open.

**Why this priority**: Confirmation, not discovery — it only runs after User Story 3 already
has a single definitive 8B answer, and reuses that answer's config rather than re-searching.

**Independent Test**: Given User Story 3's final 8B answer (a specific config or "baseline
stands"), measure that exact same config end-to-end on the 1B and 3B `4w` PTEs and report,
per model size, whether the 8B finding holds, is smaller/larger in magnitude, or reverses.

**Acceptance Scenarios**:

1. **Given** User Story 3 reports a new 8B e2e winner, **When** that winner's config is
   measured end-to-end on 1B and 3B, **Then** the report states, per model size, the e2e
   improvement (or lack thereof) over that model's own shipped baseline — not the 8B
   percentage reused as a proxy.
2. **Given** User Story 3 reports "baseline stands" on 8B, **When** the same shipped baseline
   is measured end-to-end on 1B and 3B for completeness, **Then** the report states this was
   a confirmation-only check (no alternative config was searched for 1B/3B).
3. **Given** the 1B/3B confirmation measurements, **When** any of them disagrees with the 8B
   finding's direction (e.g. the 8B winner is neutral or a regression on 1B/3B), **Then** the
   feature states this disagreement explicitly rather than reporting only the 8B result as if
   it generalized.

---

### Edge Cases

- What happens if a candidate that wins on `022`'s microbenchmark also wins e2e, but only by a
  margin comparable to e2e run-to-run noise? The feature applies the same adaptive statistical
  bar `027` used: one screening run per candidate; 3-run confirmation only for candidates
  within 10% of, or ahead of, the shipped baseline.
- What happens if `022`'s microbenchmark data is keyed to a shape that doesn't match the 8B
  e2e model used for confirmation? The feature states which model(s) each candidate was
  validated against and does not extrapolate across unmeasured shapes, matching `027`'s own
  correction of exactly this mistake.
- What happens if extending the search (User Story 2) still finds nothing that beats the
  shipped baseline? The feature reports this explicitly as the answer, not as an inconclusive
  or omitted result.
- What happens if the shared M5 EVT1 device drifts to an unexpected driver build mid-search?
  The process halts or re-verifies rather than continuing under unknown state.
- What happens if a `4w` candidate's e2e result appears to conflict with the full-stack
  optimizations already shipped (e.g. SDPA coopmat, node-threshold workaround)? The feature
  measures with those optimizations enabled as they ship on `dev` today, and does not
  isolate `4w`'s linear kernel from the rest of the stack — the deliverable is a full-stack
  e2e number, not an isolated-kernel one.
- What happens if the 8B-derived winner (User Story 3) turns out to be neutral or a
  regression on 1B and/or 3B (User Story 4)? The feature reports this disagreement
  explicitly per model size rather than defaulting to the 8B verdict, but does not
  retroactively re-open the search on 1B/3B shapes — that would be a separate follow-on
  feature, consistent with `022`'s own convention of not open-endedly expanding scope.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The process MUST rank `4w` tile/subgroup candidates by real end-to-end model
  throughput (prefill tok/s at minimum; decode tok/s if measured) as the primary decision
  metric, not by isolated-kernel microbenchmark GFLOP/s — the latter may be used only as a
  pre-filter (FR-002), never as the reported winner-selection criterion.
- **FR-002**: The process MUST use `specs/022`'s existing microbenchmark scores and
  correctness results as a cheap pre-filter to select which candidates receive real e2e
  measurement, rather than measuring every candidate in the legal tile/subgroup/subgroup-size
  space end-to-end.
- **FR-003**: Every e2e measurement MUST state which model (1B/3B/8B) and PTE it was taken on,
  and MUST use a model whose per-layer matmul shapes match the shapes the candidate's own
  microbenchmark/correctness data represents.
- **FR-004**: The process MUST NOT report an e2e winner whose correctness has not already been
  confirmed by the existing microbenchmark correctness gate at every representative shape.
- **FR-005**: The process MUST apply the adaptive statistical bar: every candidate first gets
  one e2e screening run; only a candidate whose screening run is within 10% of, or faster than,
  the shipped baseline is escalated to a 3-run confirmation (mean compared against the
  baseline's own multi-run range) — a candidate's screening-run result alone, or a candidate
  far behind baseline, MUST NOT be reported as a win.
- **FR-006**: The process MUST only extend the search to new, previously-unmeasured `4w`
  candidates (User Story 2) if the initial shortlist (User Story 1) fails to beat the shipped
  baseline end-to-end with the FR-005 statistical bar.
- **FR-007**: If a search extension (User Story 2) occurs, it MUST select new candidates using
  a documented rationale derived from `022`'s existing analytical scoring model plus what User
  Story 1 learned about microbenchmark-vs-e2e rank agreement, and MUST stay within a small,
  pre-declared additional device-time budget.
- **FR-008**: The process MUST produce exactly one final, unambiguous answer to "what is the
  current e2e winner for `4w`" — either a specific candidate with full e2e evidence, or an
  explicit statement that the currently-shipped baseline remains the best-known configuration
  end-to-end.
- **FR-009**: The process MUST operate within a bounded, pre-declared total device-time/
  measurement budget across both user stories, and MUST justify, for every candidate not
  taken to e2e measurement, why it was skipped.
- **FR-010**: The process MUST detect signs of an untrustworthy measurement environment
  (unexpected driver state, shared device unexpectedly busy) and halt or re-verify rather than
  silently continuing — including re-verifying the M5 EVT1 driver hash before any coopmat
  measurement, per this workstream's standing practice.
- **FR-011**: The process MUST explicitly compare its final e2e winner against the currently-
  shipped `4w` baseline's own e2e throughput, measured with the full stack of existing `dev`
  optimizations enabled (WMMA coopmat SDPA, node-threshold watchdog workaround, and any other
  optimization shipped by default on `dev` at the time this feature runs) — not an isolated
  `4w`-kernel-only measurement.
- **FR-012**: After User Story 3 produces a single definitive 8B answer, the process MUST
  measure that exact same config (the new winner, or the shipped baseline if none won) end-to-
  end on the 1B and 3B `4w` PTEs as a confirmation pass — not an independent staged search on
  1B/3B shapes — and MUST report, per model size, whether the 8B finding's direction (win /
  neutral / regression) holds.

### Key Entities

- **Microbenchmark Pre-Filter Score**: `specs/022`'s existing analytical score and/or measured
  microbenchmark GFLOP/s for a `4w` candidate, used only to decide which candidates are worth
  spending e2e device time on — never itself the reported ranking metric.
- **E2E Measurement**: A real end-to-end prefill (and optionally decode) tok/s result for one
  `4w` candidate on one specific model/PTE with the full existing optimization stack enabled,
  carrying its stage (`screening` = 1 run, `confirmed` = 3-run mean), run count, model/shape
  identity, driver hash, and clock-pin state.
- **Shortlist**: The initial, cheaply-selected set of `4w` candidates taken to e2e measurement
  in User Story 1 — small by construction, not the full legal space.
- **Search Extension Budget**: The small, pre-declared additional device-time/measurement
  allowance available to User Story 2, spent only if the initial shortlist doesn't produce a
  winner.
- **E2E Winner**: The feature's final answer — either a specific `4w` tile/subgroup/subgroup-
  size candidate with a confirmed, statistically-meaningful e2e improvement over the shipped
  baseline, or the shipped baseline itself, explicitly stated either way.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The feature states one unambiguous e2e winner for `4w` (a specific candidate, or
  the shipped baseline) with no open or split answer remaining.
- **SC-002**: Every e2e measurement used to support the final answer identifies its model/PTE
  and shape-representativeness explicitly; zero e2e results are reported without this context.
- **SC-003**: If a new winner is reported, its e2e improvement over the shipped baseline is
  backed by a 3-run confirmation with a stated margin clearly outside normal run-to-run noise.
- **SC-004**: The total number of candidates taken to real e2e measurement is small relative to
  the full legal tile/subgroup/subgroup-size space (target: single-digit to low tens) — the
  process explicitly reports this count and the device-time it consumed.
- **SC-005**: The feature explicitly states, for every candidate not taken to e2e measurement,
  why it was excluded — traceable without re-running the search.
- **SC-006**: The relationship between `022`'s microbenchmark rank and the new e2e rank (agree
  / partially agree / disagree) is stated explicitly as a finding, independent of which
  specific configuration wins.
- **SC-007**: The final 8B answer's config is confirmed end-to-end on both 1B and 3B, with the
  per-model-size result (holds / neutral / reverses relative to the 8B finding) stated
  explicitly — not left as an unvalidated extrapolation from 8B alone.

## Assumptions

- M5 EVT1 is the target device (either board), per this workstream's active-target constraint;
  device availability and driver identity are re-verified before use.
- "Shape-matched model" means: for candidates whose microbenchmark data used the 8B-shaped
  representative shapes, the staged search (User Stories 1–3) is run and decided on the
  Llama 3.1 8B `4w` PTE, matching `027`'s own precedent for `8da4w`. 1B and 3B are in scope
  only as a post-hoc confirmation pass (User Story 4/FR-012) on the 8B-derived answer's
  config, not as independently-searched shapes.
- "Full stack of existing optimizations" means whatever is shipped by default on `dev` at the
  time this feature runs (currently: WMMA coopmat linear for `4w`/`8da4w`, SDPA QK^T/AV
  coopmat, the `ET_VK_EXECUTE_NODE_THRESHOLD` watchdog workaround where required for 2048-token
  prefill) — this feature measures the full-stack e2e number, not an isolated-kernel one, and
  does not disable any currently-shipped optimization to isolate `4w`'s contribution.
- The existing `4w` tile/subgroup shader-variant infrastructure and dispatch mechanism from
  `specs/022` are reused as-is; this feature does not re-derive loop structure or build a new
  dispatch mechanism, only adds an e2e-measurement stage on top, mirroring `027`'s reuse of
  `025`/`026`'s infrastructure for `8da4w`.
- The standard 2048-token-prefill workload (this workstream's default) is used for e2e
  measurement; decode-phase throughput may be measured opportunistically but prefill is the
  primary metric.
- This workstream's existing device-safety practices (driver-identity re-verification, halting
  on drift, checking shared-device availability) apply unchanged.
- This is an internal engineering capability for this workstream's own use; "user"/"engineer"
  throughout this spec refers to the workstream engineer running the sweep.
- A confirmed e2e winner, if found, does not itself get shipped/promoted to the default
  dispatch by this feature — that remains a separate follow-on decision, consistent with
  `022`/`025`/`026`/`027`'s own precedent of reporting evidence without unilaterally changing
  production defaults.
