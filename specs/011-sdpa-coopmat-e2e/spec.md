# Feature Specification: SDPA Coopmat E2E Validation

**Feature Branch**: `011-sdpa-coopmat-e2e`

**Created**: 2026-07-05

**Status**: Draft

**Input**: User description: "SDPA prefill e2e validation"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Confirm SDPA coopmat actually dispatches in a real exported model (Priority: P1) 🎯 MVP

As the contributor driving this workstream, I need to know that enabling
the SDPA cooperative-matrix (WMMA) prefill path in a real, fully-exported
model actually dispatches the coopmat shaders end-to-end -- not a silent
tiled fallback -- before trusting any e2e number produced with it enabled.

**Why this priority**: `010` proved SDPA coopmat is a genuine win at the
shader-microbenchmark tier (66.8% average, all three target models
real-effect), but that tier isolates one dispatch in a synthetic harness.
SDPA coopmat is also opt-in (`ET_VK_SDPA_COOPMAT`, off by default) and has
never been exercised through a real exported `.pte`. Per this workstream's
constitution (Principle IV, two-tier discipline; Principle VI, verify with
tools), a tier-1 win never substitutes for tier-2 confirmation, and an
eligibility gate passing in isolation is not evidence it fires in the real
model graph -- `009`'s own experience (the `force_fp16`/storage-override
conflict that silently defeated the linear coopmat path for months) is the
direct precedent for why this must be checked, not assumed.

**Independent Test**: Can be fully tested by running one already-exported,
`Buffer`-storage configuration with `ET_VK_SDPA_COOPMAT` set, capturing an
ETDump trace, and confirming from the actual per-op kernel names that
`sdpa_compute_attn_weights_coopmat`/`sdpa_compute_out_coopmat` dispatched
for the prefill attention op -- independent of any timing measurement.

**Acceptance Scenarios**:

1. **Given** one of `009`'s already-exported, `Buffer`-storage
   configurations, **When** it is run with `ET_VK_SDPA_COOPMAT` set and an
   ETDump trace captured, **Then** the trace confirms
   `sdpa_compute_attn_weights_coopmat` and `sdpa_compute_out_coopmat`
   actually dispatched for the prefill attention computation, not the
   `_tiled` fallback.
2. **Given** the dispatch check fails for a configuration, **When** this is
   discovered, **Then** it is reported explicitly as a blocking finding for
   that configuration -- not worked around silently, not estimated from
   other configurations.

---

### User Story 2 - Measure e2e prefill/decode tok/s with SDPA coopmat enabled, all six configurations (Priority: P2)

As the contributor driving this workstream, I need the same fixed-workload
e2e measurement (2048-token prefill, 1024-token decode) already used
throughout this workstream, captured with `ET_VK_SDPA_COOPMAT` enabled, for
every (model, scheme) configuration -- reusing `009`'s already-captured
numbers (linear coopmat enabled, SDPA still tiled) as the direct comparison
baseline, since nothing about the export changes here, only a runtime
toggle.

**Why this priority**: This is the actual measurement the report exists to
present -- does flipping on the already-proven SDPA coopmat path move the
needle at the real end-to-end level, on top of the linear coopmat gains
`009` already measured.

**Independent Test**: Can be fully tested by capturing e2e prefill/decode
tok/s for one dispatch-confirmed configuration and confirming it produces a
directly comparable pair against `009`'s existing number for that same
configuration, independent of the other five.

**Acceptance Scenarios**:

1. **Given** a configuration that passed User Story 1's dispatch check,
   **When** e2e prefill/decode tok/s is measured, **Then** it uses the
   exact same fixed workload, device, and statistically sound methodology
   (repeated runs, steady-state reporting, no resource contention) as `009`,
   so the two arms are directly comparable.
2. **Given** all six (model, scheme) configurations, **When** measurement is
   complete, **Then** each has an e2e prefill/decode tok/s pair with SDPA
   coopmat enabled, directly comparable to `009`'s existing pair for that
   configuration, or an explicit blocked/failed status in place of the
   number.

---

### User Story 3 - Report whether SDPA coopmat helps at the e2e level, per configuration (Priority: P3)

As the contributor driving this workstream, I need a consolidated report
stating, for each of the six configurations, whether enabling SDPA coopmat
changes real end-to-end tok/s relative to `009`'s baseline, and whether
that agrees with `010`'s microbenchmark-level finding (66.8% average
prefill SDPA speedup, all three models real-effect) -- so this workstream
has one place a reader can go for the real-world answer on this specific
optimization, matching how `009` closed out the equivalent question for the
linear coopmat path.

**Why this priority**: Turns six raw measurement pairs into an answer to
whether this opt-in toggle is worth turning on by default -- the natural
question this whole investigation has been building toward since `010`.

**Independent Test**: Can be fully tested by taking the completed
measurements from Stories 1 and 2 and producing a report whose
per-configuration verdicts are directly traceable to those measurements and
to `010`'s prior microbenchmark-level finding.

**Acceptance Scenarios**:

1. **Given** the measurement pairs for all six configurations, **When** the
   report is produced, **Then** each configuration states its e2e prefill/
   decode tok/s with SDPA coopmat enabled alongside `009`'s baseline, the
   relative difference, and whether that is consistent with or diverges
   from `010`'s microbenchmark-level finding.
2. **Given** a configuration where the e2e result diverges meaningfully from
   `010`'s microbenchmark-level finding, **When** the report is read, **Then**
   this divergence is called out explicitly, not averaged away or omitted.
3. **Given** the full report, **When** read end to end, **Then** it gives
   one clear, direct answer to "does enabling SDPA coopmat actually help
   this device's real token generation rate" -- not just at the isolated
   shader level.

### Edge Cases

- What happens if a configuration's dispatch check fails (SDPA coopmat
  doesn't fire even though the export is `Buffer`-storage and the toggle is
  set)? It MUST be reported as a dispatch-check failure per User Story 1,
  and no e2e number reported for it -- matching `009`'s own precedent for
  its analogous linear-coopmat dispatch check.
- What happens if the e2e result for a configuration contradicts `010`'s
  microbenchmark-level finding (e.g. no measurable prefill improvement
  despite confirmed coopmat dispatch)? This MUST be reported as a real,
  named divergence -- matching `009`'s precedent that a microbenchmark-level
  finding alone does not guarantee its e2e magnitude, only its direction is
  expected to hold, and even that must be checked, not assumed.
- What happens to decode tok/s? Expected to be materially unchanged --
  `010` established no WMMA-capable GEMV kernel exists for attention, so
  decode continues to dispatch the existing `_coop` path regardless of the
  toggle. This is a sanity-check expectation stated in Assumptions, not
  something to silently assume without measuring.
- What happens if `009`'s existing `.pte` exports turn out not to actually
  have `Buffer`+`half` storage for the SDPA-relevant tensors (Q/K/V/
  attn_weights/out), contrary to the Assumptions below? Then this feature's
  premise (no new export needed) is wrong for that configuration, and a
  fresh export is required and reported as a scope correction, not silently
  worked around.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST reuse `009`'s already-exported, `Buffer`-storage
  `.pte` files for all six (model, scheme) configurations rather than
  producing new exports, since enabling SDPA coopmat is a runtime toggle
  (`ET_VK_SDPA_COOPMAT`) with no export-time dependency.
- **FR-002**: Before any e2e number is reported for a configuration, an
  ETDump trace MUST confirm `sdpa_compute_attn_weights_coopmat` and
  `sdpa_compute_out_coopmat` actually dispatched for its prefill attention
  computation -- not assumed from the toggle being set or from `010`'s
  isolated-harness result.
- **FR-003**: The system MUST measure e2e prefill/decode tok/s, with
  `ET_VK_SDPA_COOPMAT` enabled, for every configuration that passes FR-002's
  dispatch check, using the exact same fixed workload (2048-token prefill,
  1024-token decode), device, and statistically sound methodology as `009`.
- **FR-004**: The system MUST reuse `009`'s already-captured e2e prefill/
  decode tok/s numbers (SDPA coopmat disabled, the current default) as the
  direct comparison baseline for every configuration, rather than
  re-capturing them.
- **FR-005**: The system MUST report, per configuration, the e2e prefill/
  decode tok/s with SDPA coopmat enabled alongside `009`'s baseline pair,
  with their relative difference.
- **FR-006**: The system MUST state, per configuration, whether the e2e
  result is consistent with or diverges from `010`'s microbenchmark-level
  finding for that model (66.8% average prefill SDPA speedup).
- **FR-007**: Any configuration for which FR-002's dispatch check fails MUST
  be reported explicitly as blocked/failed with a stated reason -- never
  silently omitted or estimated.
- **FR-008**: Decode tok/s MUST be reported alongside prefill for every
  configuration as a sanity check, with any unexpected decode-side change
  called out explicitly rather than silently assumed to be noise.

### Key Entities

- **SDPA-Coopmat-Enabled E2E Run**: a per-configuration run of `009`'s
  existing export with `ET_VK_SDPA_COOPMAT` set, its ETDump-confirmed
  dispatch outcome (coopmat actually fired vs. silent tiled fallback for
  the prefill attention op), and its e2e prefill/decode tok/s.
- **E2E SDPA Comparison Case**: one configuration's baseline (from `009`,
  SDPA coopmat disabled) vs. SDPA-coopmat-enabled e2e prefill/decode tok/s
  pair, their relative difference, and a consistency verdict against `010`'s
  microbenchmark-level finding.
- **SDPA Coopmat E2E Report**: the consolidated document covering all six
  configurations (or explicitly noting which are blocked), with
  per-configuration verdicts and one overall statement of whether enabling
  SDPA coopmat helps this device's real token generation rate.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For every configuration that can be measured, an ETDump trace
  confirms both SDPA coopmat shaders actually dispatched before any e2e
  number is reported for it.
- **SC-002**: Every measurable configuration has an e2e prefill/decode tok/s
  pair (SDPA coopmat enabled) captured under identical methodology,
  workload, and device to `009`'s existing baseline pair for that same
  configuration.
- **SC-003**: A reader can determine, for each of the six configurations,
  whether enabling SDPA coopmat changes real end-to-end tok/s and whether
  that agrees with `010`'s prior microbenchmark-level finding -- without
  needing to re-derive anything from raw logs.
- **SC-004**: Any configuration that could not be measured (failed dispatch
  check) is stated explicitly with its reason, never silently absent from
  the report.

## Assumptions

- `009`'s existing `Buffer`-storage `.pte` exports already have SDPA's
  relevant tensors (Q/K/V, attn_weights, output) in `Buffer`+`half` storage
  -- this was true throughout the whole graph once `009`'s `force_fp16`/
  storage-override pass fix was applied (empirically observed during that
  feature's own dispatch checks: every op's kernel name, including the
  SDPA-family ones, showed `_buffer_` after the fix). If this turns out not
  to hold for some configuration, that is reported as a scope correction
  (Edge Cases), not silently patched around.
- `009`'s existing e2e captures (SDPA coopmat disabled, the current
  production default) are reused as-is as this feature's baseline, not
  re-captured -- they were measured under the identical workload/device/
  methodology this feature also uses, so re-measuring would only add device
  time without new information. The same cross-session prefill-variance
  caveat `009` inherited from `006` applies again here for prefill
  comparisons specifically; decode is not affected.
- Scope is all three target models at both `4w`/`8da4w` schemes (six
  configurations, the constitution's default), not narrowed to one
  configuration per model the way `010`'s tier-1 microbenchmark was --
  because e2e impact depends on the surrounding scheme's own linear-coopmat
  performance profile (`009`'s own finding: `4w` and `8da4w` move in
  opposite directions), not just SDPA's own isolated dispatch, so all six
  are needed for a complete e2e picture.
- Enabling `ET_VK_SDPA_COOPMAT` is scoped to this feature's own capture
  runs only; whether to change its default (currently off/opt-in) in
  production is a separate decision this feature informs but does not make,
  matching this workstream's precedent of separating measurement features
  from production-default decisions.
- Decode tok/s is expected to be materially unchanged by the toggle (`010`:
  no WMMA-capable GEMV kernel exists for attention), but this is measured
  and reported (FR-008), not assumed without checking.
