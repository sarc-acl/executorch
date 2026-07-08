# Feature Specification: M5 EVT1 8da4w T-tiled Baseline

**Feature Branch**: `018-m5-8da4w-t-tiled-baseline`

**Created**: 2026-07-06

**Status**: Draft

**Input**: User description: "ok now to fully conclude the report, we need the baseline of 8da4w."

## Context

`specs/015-m5-e2e-wmma-validation` and this week's status draft both
report real, measured `8da4w` full-stack (linear WMMA + SDPA WMMA) e2e
numbers for all three models (1B 723.00, 3B 286.31, 8B 130.05 tok/s) --
but **no `8da4w` T-tiled (stock, non-coopmat) baseline exists for any
model**, so none of those numbers can be expressed as a speedup ratio.
`4w` has this baseline already (`RESULTS-SUMMARY.md`'s trusted anchor:
1B 312.7, 3B 112.5, 8B 51.4 tok/s) and its speedup ratios are the
headline numbers in this week's report; `8da4w` is missing the exact
same thing. `ACTIVE-STATUS.md`'s own "Open / next" section already lists
"clean 8da4w T-tiled baseline @2048" as an outstanding item -- this
feature closes it. Checked: no `_texture_ctx3072.pte` exists for `8da4w`
on any of the 3 models today (8B has a stale `_texture_ctx2304.pte`, the
wrong context length; 1B/3B have no texture-storage `8da4w` PTE at all) --
new PTEs must be exported, this is not just a "run an existing file"
measurement.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 1B `8da4w` T-tiled baseline measured (Priority: P1)

As the engineer finalizing this week's speedup report, I need a real,
measured `8da4w` T-tiled prefill tok/s number for LLaMA 3.2 1B, so the
report's 1B/`8da4w` row can show a speedup ratio instead of "no baseline
yet."

**Why this priority**: 1B is the fastest and cheapest model to measure,
and proves out the export + measurement methodology (new texture-storage
PTE, correct context length, dispatch-confirmed tiled) before repeating it
on the larger, slower, more device-time-expensive models.

**Independent Test**: a `llama3_2_1b_8da4w_texture_ctx3072.pte` exists,
was measured at the standard 2048-prefill/1024-decode workload with
pinned clocks, and produces a 3-run mean + CoV prefill tok/s number with
its dispatch confirmed genuinely tiled (not coopmat).

**Acceptance Scenarios**:

1. **Given** no `8da4w` texture-storage PTE exists for 1B at `ctx3072`,
   **When** this feature exports one using the default (non-`buffer`-override)
   storage config, **Then** the resulting PTE is texture-storage internally
   (not silently buffer, per gotcha G2's caution about trusting a filename).
2. **Given** that PTE, **When** it is run at the standard workload with
   pinned, verified clocks, **Then** a 3-run mean + CoV prefill/decode
   tok/s number is produced and dispatch is confirmed tiled (via ETDump or
   equivalent), not coopmat.

---

### User Story 2 - 3B `8da4w` T-tiled baseline measured (Priority: P2)

Same as User Story 1, for LLaMA 3.2 3B.

**Why this priority**: second-cheapest model; follows the same,
now-proven methodology from User Story 1.

**Independent Test**: a `llama3_2_3b_8da4w_texture_ctx3072.pte` exists and
is measured the same way as User Story 1.

**Acceptance Scenarios**:

1. **Given** User Story 1's methodology is proven, **When** it is repeated
   for 3B, **Then** the same measured, dispatch-confirmed, 3-run-mean
   result is produced for 3B.

---

### User Story 3 - 8B `8da4w` T-tiled baseline measured (Priority: P3)

Same as User Stories 1-2, for LLaMA 3.1 8B.

**Why this priority**: 8B is the slowest and highest-device-time-cost
model, and (per this workstream's established pattern, e.g.
`specs/015` Decision 3) the highest GPU-watchdog risk at the full
2048-token prefill length -- sequenced last, after the methodology is
proven twice already.

**Independent Test**: a `llama3_1_8b_8da4w_texture_ctx3072.pte` exists
(replacing the existing, wrong-context-length `_ctx2304` texture PTE) and
is measured the same way as User Stories 1-2, including the
`ET_VK_EXECUTE_NODE_THRESHOLD=16` prefill-watchdog workaround already
established for 8B runs in this workstream.

**Acceptance Scenarios**:

1. **Given** User Stories 1-2's methodology is proven, **When** it is
   repeated for 8B (with the established watchdog workaround applied),
   **Then** the same measured, dispatch-confirmed, 3-run-mean result is
   produced for 8B.

---

### User Story 4 - Speedup table shows real ratios for all six configs (Priority: P2)

As a reader of this week's report or `specs/015`'s consolidated results,
I need the `8da4w` rows of the speedup table to show an actual "vs
baseline" ratio, the same way every `4w` row already does, so the report
is not left with three unexplained "no baseline yet" cells once this
feature's measurements exist.

**Why this priority**: this is the actual reason the baselines are being
measured -- the raw numbers alone (User Stories 1-3) don't complete the
report by themselves until they're placed into it.

**Independent Test**: open `specs/015-m5-e2e-wmma-validation/results/m5-e2e-validation-report.md`
(and the equivalent cells in this week's status draft) and confirm every
`8da4w` row has a numeric speedup ratio, not a "no baseline yet" note.

**Acceptance Scenarios**:

1. **Given** User Stories 1-3 have produced all three baselines, **When**
   the consolidated report and results files are updated, **Then** each
   `8da4w` row shows `<baseline tok/s> -> <optimized tok/s>, N.NNx` in the
   same format already used for `4w`.

---

### Edge Cases

- What if the newly-exported texture-storage PTE unexpectedly dispatches
  something other than tiled (e.g. an unintended fallback path)? -- must
  be caught by dispatch verification (User Stories 1-3's acceptance
  criteria), not silently accepted as if it were a valid T-tiled number.
- What if M5 EVT1 isn't free when this work is attempted? -- constitution
  Principle VIII / gotcha G10 discipline applies: confirm with the user
  before assuming the device is available, don't assume continuity from a
  prior session.
- What if 3B or 8B's T-tiled `8da4w` run hits the GPU prefill watchdog
  risk documented for **both** models (jira ticket #001, "2048-prefill
  GPU watchdog (8B/3B)" -- not 8B-only) at 2048 tokens? -- apply the
  already-established `ET_VK_EXECUTE_NODE_THRESHOLD=16` workaround used
  throughout this workstream's other 3B/8B measurements, per `specs/015`.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: This feature MUST export a texture-storage (T-tiled, i.e.
  the default/non-coopmat-override) `8da4w` PTE at `ctx3072` for each of
  the three target models (1B, 3B, 8B) -- none currently exist at this
  context length for this scheme.
- **FR-002**: This feature MUST measure e2e prefill and decode tok/s for
  each model's T-tiled `8da4w` PTE at the standard 2048-prefill/1024-decode
  workload (constitution Default Scope), with clocks pinned and the pin
  verified bound (Principle VII), reporting a 3-run mean with CoV for each
  (matching the existing `4w` T-tiled baseline's own methodology, so the
  two are apples-to-apples comparable).
- **FR-003**: This feature MUST re-verify the on-device driver identity
  (Principle VIII) before measuring, per this workstream's standing
  discipline -- not assume a prior session's driver state still holds.
- **FR-004**: This feature MUST confirm, via tooling (ETDump or
  equivalent), that each T-tiled baseline run genuinely dispatches the
  tiled kernel family, not coopmat -- per Principle VI ("verify with
  tools, never assume"), especially given this session's own G6/Q11
  history of ETDump attribution being unreliable in some contexts.
- **FR-005**: This feature MUST update the downstream consolidated
  report(s) -- `specs/015-m5-e2e-wmma-validation/results/m5-e2e-validation-report.md`
  and the per-model `results/*.md` files -- so every `8da4w` row shows a
  real speedup ratio, closing the gap User Story 4 describes. This
  includes the **linear-only** `8da4w` rows (already published, e.g.
  `1b-results.md`'s `8da4w` linear row currently says "None -- no prior
  M5 EVT1 `8da4w` baseline exists"), not just the full-stack
  (linear+SDPA) rows that motivated this feature -- the same T-tiled
  measurement is the correct comparison point for both, and closing only
  one of the two would leave the report's `8da4w` story half-finished.
  Updating the author's own personal status draft is out of this
  feature's scope (a personal document, not this workstream's own
  artifact) but the ratios it needs will exist once this feature
  completes.
- **FR-006**: Baseline measurements MUST use the exact same standard
  workload and clock-pinning methodology as the already-established `4w`
  T-tiled baseline (2048-prefill/1024-decode, pinned 509/2730/663 MHz) --
  this feature does not invent a new methodology, it extends the existing
  one to the missing scheme.

### Key Entities

- **T-tiled Baseline Measurement**: one per model. Fields: model, scheme
  (`8da4w`, fixed for this feature), prefill tok/s (3-run mean + CoV),
  decode tok/s (3-run mean + CoV), dispatch_status (`tiled_confirmed` --
  the only valid value; anything else is a defect in this feature's own
  measurement, not a reportable baseline), PTE export details (texture
  storage, `ctx3072`).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All three models (1B, 3B, 8B) have a genuine, measured
  `8da4w` T-tiled prefill tok/s number -- not estimated, not
  extrapolated from the `4w` ratio.
- **SC-002**: Each baseline number is backed by a 3-run mean with CoV,
  matching the rigor already applied to every other number in
  `specs/015`'s results.
- **SC-003**: The consolidated speedup table
  (`m5-e2e-validation-report.md`) shows a real numeric ratio for all six
  model x scheme combinations -- zero remaining "no baseline yet" cells.
- **SC-004**: Every baseline run's dispatch is confirmed tiled (not
  coopmat) via tooling, documented alongside the number, not assumed from
  the PTE's storage-type filename alone.

## Assumptions

- This feature reuses the already-established T-tiled baseline
  methodology and conventions from the existing `4w` baseline (same
  workload, same clock-pinning/verification discipline, same 3-run+CoV
  convention) -- it does not invent new measurement methodology.
- M5 EVT1 device access follows the same access/driver-verification
  discipline as prior specs in this workstream (constitution Principles
  VII/VIII; gotcha G10 -- confirm the device is free before assuming so).
- This is a measurement-and-reporting-only feature -- no shader or
  production dispatch-logic code changes are in scope. If the T-tiled
  export or run reveals an unexpected defect (e.g., FR-004's dispatch
  check fails), that becomes a new tracked issue
  (`open-questions.md`/gotchas), not something this feature fixes inline.
- Updating the author's own personal weekly-status draft (as distinct
  from this workstream's own `specs/015` artifacts) is out of scope for
  this feature, per FR-005's note -- that document is not owned by this
  repository's spec-kit workflow.
