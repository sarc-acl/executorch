# Feature Specification: End-to-End tok/s Report — Texture, Buffer, and WMMA Across 4w/8da4w

**Feature Branch**: `009-e2e-tokrate-report`

**Created**: 2026-07-04

**Status**: Draft

**Input**: User description: "Produce a full report on 4w 8da4w study, on our rocky-ryzen. Measure the e2e tok/s for 2k prefill and 1k decode, Compare Texture-baseline, Buffer-Baseline, and our WMMA shader. on all 3 models"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Confirm the WMMA-eligible e2e export actually dispatches coopmat (Priority: P1)

As the contributor driving this workstream, I need to know that at least one (model, scheme) configuration's exported artifact actually routes its linear layers through the coopmat/WMMA shader end-to-end — not silently falling back to the tiled path — before trusting any WMMA-arm timing number in the final report.

**Why this priority**: `003` found the real model's linear outputs are rank-3 (`[1, M, K]`), which fails `can_use_q4gsw_coopmat()`'s `dim_of(output) > 2` guard before its separate `storage_type_of(output) != kBuffer` check is even reached — either blocker alone prevents coopmat dispatch, independent of `006`'s existing `--vulkan-storage-override` flag (which only addresses the storage half). `007` separately found `4w`'s coopmat path was unreachable at the op level due to a routing bug (now fixed, uncommitted). Without confirming actual dispatch first, a "WMMA" e2e number could silently just be the tiled path timed twice.

**Independent Test**: Can be fully tested by producing one (model, scheme) configuration's WMMA-eligible export, running it, and confirming via ETDump (per the constitution's Principle VI) that the coopmat kernel — not the tiled fallback — is what actually executed, before any tok/s number from it is trusted.

**Acceptance Scenarios**:

1. **Given** one target model at one quantization scheme, **When** a WMMA-eligible export (Buffer storage + rank-3 resolved + `007`'s wiring fix applied) is produced and run, **Then** ETDump confirms the coopmat kernel family actually dispatched for the linear layers, not the tiled fallback.
2. **Given** the rank-3 blocker cannot be resolved for a configuration without a change whose scope or risk goes beyond this feature, **When** this is discovered, **Then** it is reported explicitly as a blocking finding for that configuration — not worked around silently, not estimated from other configurations.

---

### User Story 2 - Measure e2e prefill/decode tok/s for all three dispatch arms, both schemes, all three models (Priority: P2)

As the contributor driving this workstream, I need the same fixed-workload e2e measurement (2048-token prefill, 1024-token decode) used throughout this workstream, captured for `Texture3D`-baseline, `Buffer`-baseline, and WMMA dispatch, for every (model, scheme) configuration — reusing `006`'s already-captured `Texture3D`/`Buffer` numbers rather than re-measuring them — so the WMMA arm has a same-methodology, same-device comparison to sit alongside.

**Why this priority**: This is the actual measurement the report exists to present. `006` already produced two of the three arms; this feature's own new capture work is specifically the WMMA arm, for every configuration that passes User Story 1's dispatch check.

**Independent Test**: Can be fully tested by running the standard e2e capture procedure against one WMMA-eligible configuration and confirming it produces a directly comparable prefill/decode tok/s pair alongside `006`'s existing `Texture3D`/`Buffer` numbers for that same configuration, independent of the other five.

**Acceptance Scenarios**:

1. **Given** a WMMA-eligible export that passed User Story 1's dispatch check, **When** e2e prefill/decode tok/s is measured, **Then** it uses the exact same fixed workload, device, and statistically sound methodology (repeated runs, steady-state reporting, no resource contention) as `006`'s existing captures, so all three arms are directly comparable.
2. **Given** all six (model, scheme) configurations, **When** measurement is complete, **Then** each has a `Texture3D`, `Buffer` (both reused from `006`), and WMMA e2e prefill/decode tok/s triple recorded, or an explicit blocked/failed status in place of the WMMA number.

---

### User Story 3 - Report whether WMMA actually helps at the e2e level, per configuration (Priority: P3)

As the contributor driving this workstream, I need a consolidated report stating, for each of the six configurations, how the three dispatch arms compare at the real end-to-end tok/s level — and whether that agrees with `007`'s microbenchmark-level finding (`4w` +60.6%, `8da4w` -15.2% vs. shipped) and `008`'s tuning finding (`8da4w` config 5 closes most of that gap vs. shipped but stays roughly at parity with tiled) — so the whole `4w`/`8da4w` coopmat investigation on this device has one place a reader can go for the real-world answer.

**Why this priority**: This is what turns six raw measurement triples into an answer to the question that has motivated this entire workstream since `003`.

**Independent Test**: Can be fully tested by taking the completed measurements from Stories 1 and 2 and producing a report whose per-configuration verdicts are directly traceable to those measurements and to `007`/`008`'s prior microbenchmark-level findings.

**Acceptance Scenarios**:

1. **Given** the measurement triples for all six configurations, **When** the report is produced, **Then** each configuration states its `Texture3D`/`Buffer`/WMMA tok/s for both prefill and decode, whether WMMA's e2e result is consistent with or diverges from `007`'s microbenchmark-level finding for that scheme, and by how much.
2. **Given** a configuration where the e2e result diverges meaningfully from the microbenchmark-level finding, **When** the report is read, **Then** this divergence is called out explicitly, not averaged away or omitted.
3. **Given** the full report, **When** read end to end, **Then** it gives one clear, direct answer to "does WMMA actually help this device's real token generation rate" per scheme — not just per micro-op.

### Edge Cases

- What happens if the rank-3 blocker can be resolved for some (model, scheme) configurations but not others (e.g. a shape-specific limitation)? Those configurations MUST be reported as blocked, not silently excluded from the configuration count or estimated from the ones that did work — matching `006`'s established precedent for its own blocked configurations.
- What happens if a WMMA-eligible export runs but ETDump shows the coopmat kernel did NOT dispatch (silent tiled fallback)? That configuration MUST be reported as a dispatch-check failure, and no WMMA tok/s number reported for it — Principle VI (verify with tools, never assume) applies directly.
- What happens if the e2e WMMA result contradicts `007`'s microbenchmark-level finding or `008`'s tuning finding for a given configuration? This MUST be reported as a real, named divergence, matching `006`'s precedent — it does not invalidate the prior microbenchmark data, but means the microbenchmark-level finding alone was not sufficient to predict real-world e2e behavior for that configuration.
- What happens if resolving the rank-3 blocker requires a change to `can_use_q4gsw_coopmat()`'s eligibility gate or to how the export graph shapes linear-layer output? This is a production dispatch-code change and MUST follow this workstream's established discipline: proposed during planning, applied only with explicit user authorization (matching how `007`'s wiring fix was handled), and documented at the point it's made.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST reuse `006`'s already-captured `Texture3D` and `Buffer` e2e prefill/decode tok/s numbers for all six (model, scheme) configurations rather than re-measuring them.
- **FR-002**: The system MUST produce a WMMA-eligible e2e export for each of the six configurations, requiring: `Buffer` storage (`006`'s existing `--vulkan-storage-override` flag), the rank-3 output blocker resolved (`003`), and `007`'s coopmat wiring fix applied for `4w`.
- **FR-003**: Before any WMMA-arm tok/s number is reported for a configuration, ETDump MUST confirm the coopmat kernel family actually dispatched for its linear layers (Principle VI) — not assumed from the export succeeding alone.
- **FR-004**: The system MUST measure e2e prefill/decode tok/s for the WMMA arm of every configuration that passes FR-003's dispatch check, using the exact same fixed workload (2048-token prefill, 1024-token decode), device, and statistically sound methodology as `006`.
- **FR-005**: The system MUST report, per configuration, `Texture3D`/`Buffer`/WMMA e2e prefill/decode tok/s side by side with their relative differences.
- **FR-006**: The system MUST state, per configuration, whether the e2e WMMA result is consistent with or diverges from `007`'s microbenchmark-level finding for that scheme, and (for `8da4w`) whether it's consistent with `008`'s tuning finding.
- **FR-007**: Any configuration for which the rank-3 blocker cannot be resolved, or for which FR-003's dispatch check fails, MUST be reported explicitly as blocked/failed with a stated reason — never silently omitted or estimated.
- **FR-008**: The WMMA arm MUST use the shipped/default coopmat tile configuration (`WG_TILE_M=128, WG_TILE_N=64, WG_TILE_K=32, SUBGROUP_SIZE=64`) — `008`'s config 5 tuning finding (`SUBGROUP_SIZE=32`) is not reachable through production's `can_use_q4gsw_coopmat()` gate, which hard-requires the adapter's native `subgroup_size() == 64`, so it cannot be what "our WMMA shader" e2e-dispatches regardless of its microbenchmark-level win.
- **FR-009**: Any code change required to resolve the rank-3 blocker MUST be proposed during planning and applied only with explicit user authorization, then documented at the point it's made — matching this workstream's established discipline for production dispatch-code changes (`007`'s wiring fix).

### Key Entities

- **WMMA-Eligible Export**: a per-configuration exported artifact with `Buffer` storage, the rank-3 blocker resolved, and (for `4w`) `007`'s wiring fix applied, plus its ETDump-confirmed dispatch outcome (coopmat actually fired vs. silent tiled fallback).
- **E2E Three-Way Comparison Case**: one configuration's `Texture3D`/`Buffer`/WMMA e2e prefill/decode tok/s triple, their relative differences, and a consistency verdict against `007`'s (and, for `8da4w`, `008`'s) microbenchmark-level findings.
- **E2E tok/s Report**: the consolidated document covering all six configurations (or explicitly noting which are blocked), with per-configuration verdicts and one overall statement of whether WMMA helps this device's real token generation rate, per scheme.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For every configuration that can be measured, ETDump confirms the coopmat kernel actually dispatched before any WMMA tok/s number is reported for it.
- **SC-002**: Every measurable configuration has `Texture3D`, `Buffer`, and WMMA e2e prefill/decode tok/s captured under identical methodology, workload, and device.
- **SC-003**: A reader can determine, for each of the six configurations, how the three dispatch arms compare and whether the real end-to-end result agrees with `007`'s (and, for `8da4w`, `008`'s) prior findings — without needing to re-derive anything from raw logs.
- **SC-004**: Any configuration that could not be measured (unresolved rank-3 blocker, failed dispatch check) is stated explicitly with its reason, never silently absent from the report.

## Assumptions

- `006`'s `Texture3D`/`Buffer` e2e numbers are reused as-is, not re-captured — they were measured under the same workload/device/methodology this feature also uses, so re-measuring would only add device time without new information.
- Resolving the rank-3 output blocker is in scope for this feature (matching `006`'s own precedent of resolving its storage blocker as part of its own scope, rather than deferring it) — the exact mechanism (a graph-shape fix, or a narrower change to `can_use_q4gsw_coopmat()`'s guard) is determined during planning, not assumed here. If it turns out to require changes whose scope or risk goes beyond a contained, well-understood fix, that is reported per FR-007 rather than blocking the whole feature.
- `007`'s coopmat wiring fix (currently uncommitted in the working tree) is applied for the purpose of this feature's measurements. Whether to commit/ship that fix independently of this feature is a separate decision, not this feature's concern.
- The WMMA arm measures the shipped/default tile configuration only (FR-008) — `008`'s experimental config 5 finding is a microbenchmark-only result that production's own eligibility gate can never dispatch to (it hard-requires native `subgroup_size() == 64`), so it is out of scope for an e2e "our WMMA shader" comparison.
- Scope matches the constitution's default: all three target models, both `4w`/`8da4w` schemes, fixed 2048-token prefill / 1024-token decode workload, on `rocky-ryzen`.
- Any resolution of the rank-3 blocker that touches production dispatch code follows this workstream's established discipline: proposed during planning, applied only with explicit user authorization, documented at the point it's made (constitution Principle V and prior practice for `007`'s fix).
