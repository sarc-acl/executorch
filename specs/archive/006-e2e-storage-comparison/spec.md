# Feature Specification: End-to-End Texture3D vs. Buffer Storage Comparison

**Feature Branch**: `006-e2e-storage-comparison`

**Created**: 2026-07-04

**Status**: Draft

**Input**: User description: "given the knowledge of microbenchmark results, conduct performance study on e2e results of texture vs buffer for baseline for all 3 models, both 8da4w and 4w."

## Clarifications

### Session 2026-07-04

- Q: How should Buffer-storage correctness be verified for the full e2e model? → A: Texture3D-vs-Buffer numerical equivalence is not this feature's concern — it's assumed to already be guaranteed by ExecuTorch/the Vulkan backend's storage-type abstraction (this is a general framework property, not something specific to this workstream). The contributor's actual future correctness concern is downstream and out of scope here: once WMMA/coopmat shaders are added, *those* will need to be checked against the accepted tiled baseline — a separate, later feature. This feature only needs a basic smoke-check (does the Buffer-storage export run and produce coherent, non-garbage output?) to catch its *own* export/config mistakes, not to re-validate storage-type math equivalence.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Confirm a Buffer-storage model actually runs at all (Priority: P1)

As the contributor driving this WMMA/coopmat performance workstream, I need to know whether the real end-to-end model can actually be exported and run with its linear activations allocated as `Buffer` storage instead of today's default `Texture3D` — producing coherent (not obviously broken/garbage) output — before I trust any timing number from it.

**Why this priority**: The prior feature (`004-linear-storage-comparison`) only measured this at the isolated single-op microbenchmark level, using synthetic tensors. Whether the real model's export/graph can even be forced into `Buffer` storage at all is unverified and is a precondition for everything else in this study — a timing number from a build that silently failed to actually apply the storage change, or that crashes/produces garbage, is worse than no number at all. (Whether `Buffer` storage computes numerically the *same* result as `Texture3D` is not re-verified here — see Clarifications: that equivalence is assumed as an existing ExecuTorch/Vulkan-backend guarantee, not something this workstream owns.)

**Independent Test**: Can be fully tested by producing a `Buffer`-storage variant of one (model, scheme) configuration's exported artifact, running it, and confirming it completes and produces coherent output (a smoke-check, not a rigorous correctness comparison) before any timing is trusted.

**Acceptance Scenarios**:

1. **Given** one target model at one quantization scheme, **When** a `Buffer`-storage variant of its exported artifact is produced and run, **Then** it completes without crashing and produces coherent (non-garbage) output before any performance measurement is taken from it.
2. **Given** the `Buffer`-storage variant cannot be produced at all for a given configuration (e.g. a structural limitation in how the model is built or exported), **When** this is discovered, **Then** it is reported explicitly as a blocking finding, not worked around silently or left unexplained.

---

### User Story 2 - Measure e2e prefill/decode tok/s for both storage types, all six configurations (Priority: P2)

As the contributor driving this workstream, I need the same fixed-workload e2e measurement (2048-token prefill, 1024-token decode) already used for this workstream's baseline, run against both the `Texture3D` and `Buffer` variants, for all three target models at both quantization schemes, so I have real end-to-end numbers to compare rather than only the isolated microbenchmark evidence from `004`.

**Why this priority**: This is the actual measurement the whole feature exists to produce — extending `004`'s single-op finding to the full model, where framework overhead, memory pressure, and other real-model effects could behave differently than an isolated op does.

**Independent Test**: Can be fully tested by running the standard e2e capture procedure against both storage variants for one configuration and confirming both produce a directly comparable prefill/decode tok/s pair, independent of the other five configurations.

**Acceptance Scenarios**:

1. **Given** a `Buffer`-storage variant that passed User Story 1's smoke-check, **When** e2e prefill/decode tok/s is measured, **Then** it uses the exact same fixed workload, device, and statistically sound methodology (repeated runs, no resource contention) as this workstream's existing `Texture3D` e2e baseline (`001`), so the two are directly comparable.
2. **Given** all six configurations, **When** measurement is complete, **Then** each has both a `Texture3D` and a `Buffer` e2e prefill/decode tok/s pair recorded.

---

### User Story 3 - Report whether the microbenchmark-level finding holds at the e2e level (Priority: P3)

As the contributor driving this workstream, I need a consolidated report stating, for each configuration, whether `004`'s microbenchmark-level "storage switch is basically free" finding holds, partially holds, or does not hold once measured on the real end-to-end model — so I know whether that finding can be relied on when planning the actual coopmat-enabling fix.

**Why this priority**: This is what turns the raw e2e measurements into an answer to the question that motivated this whole feature.

**Independent Test**: Can be fully tested by taking the completed e2e measurements from Stories 1 and 2 and producing a report whose per-configuration verdicts are directly traceable to those measurements and to `004`'s prior findings.

**Acceptance Scenarios**:

1. **Given** the e2e measurements for all six configurations, **When** the report is produced, **Then** each configuration states whether its e2e result is consistent with `004`'s microbenchmark-level finding (storage switch ~free) or diverges from it, with the actual numbers shown.
2. **Given** a configuration where the e2e result diverges from the microbenchmark-level finding, **When** the report is read, **Then** this divergence is called out explicitly rather than averaged away or omitted.

### Edge Cases

- What happens if a `Buffer`-storage export cannot be produced for some configurations but can for others (e.g. a size or structural limitation specific to one model)? Those configurations MUST be reported as blocked/not-measured, not silently excluded from the configuration count or estimated from the ones that did work.
- What happens if the `Buffer`-storage variant runs but produces obviously broken/garbage output (crashes, empty output, clearly incoherent text)? That configuration MUST be reported as a smoke-check failure, and no performance number MUST be reported for it — Principle I (correctness before performance) applies to "does our export actually work," not to re-litigating storage-type numerical equivalence, which is out of scope per the Clarifications session.
- What happens if producing a `Buffer`-storage export requires a code change with its own risk or side effects (e.g. to the partitioner or memory-planning pass)? That change, and its scope, MUST be documented explicitly (Constitution Principle V's spirit — document what was needed, at the point it was needed) so a future reader understands what was modified to make this measurement possible, distinct from what is needed to actually ship the storage change in production.
- What happens if the e2e result contradicts `004`'s microbenchmark finding for a given configuration? This MUST be reported as a real, named divergence — it does not invalidate `004`'s microbenchmark data, but it does mean the microbenchmark-level finding alone was not sufficient evidence for that configuration's real-world behavior.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST produce a `Buffer`-storage variant of each target model's exported artifact, covering this workstream's default benchmark scope (all three target models, both `4w` and `8da4w` — six configurations, per the constitution's "Default Scope for Every Benchmark").
- **FR-002**: Before any performance number is measured from a `Buffer`-storage variant, it MUST pass a basic smoke-check (completes without crashing, produces coherent non-garbage output) — this catches this feature's own export/configuration mistakes; it is not a re-verification of Texture3D-vs-Buffer numerical equivalence, which is assumed as an existing ExecuTorch/Vulkan-backend guarantee (see Clarifications).
- **FR-003**: The system MUST measure e2e prefill/decode tok/s for both storage variants of every configuration using the constitution's fixed default workload (2048-token prefill, 1024-token decode) and this workstream's established statistically sound methodology (repeated runs, steady-state reporting, no resource contention during capture).
- **FR-004**: The system MUST report, per configuration, the `Texture3D` and `Buffer` e2e prefill/decode tok/s side by side, along with their relative difference.
- **FR-005**: The system MUST state, per configuration, whether the e2e result is consistent with or diverges from `004`'s microbenchmark-level "storage switch is basically free" finding.
- **FR-006**: Any configuration for which a `Buffer`-storage variant cannot be produced, or fails the smoke-check, MUST be reported explicitly as blocked/failed with a stated reason — never silently omitted or estimated.
- **FR-007**: Any code change required to produce a `Buffer`-storage export (e.g. to the partitioner or memory-planning logic) MUST be documented — what was changed and why — separately from any conclusion about whether that change is suitable for production use.
- **FR-008**: The system MUST NOT claim this feature's results say anything about coopmat eligibility — like `004`, this is purely a storage-type e2e cost/benefit measurement, independent of the WMMA/coopmat work itself.

### Key Entities

- **Buffer-Storage Export**: a per-configuration exported artifact whose linear activations are allocated as `Buffer` storage instead of the default `Texture3D`, plus its smoke-check outcome (pass/fail — does it run and produce coherent output, not a numerical-equivalence comparison).
- **E2E Storage Comparison Case**: one configuration's `Texture3D` vs. `Buffer` e2e prefill/decode tok/s pair, their relative difference, and a consistency verdict against `004`'s microbenchmark-level finding.
- **E2E Storage Comparison Report**: the consolidated document covering all six configurations (or explicitly noting which are blocked/failed), with per-configuration verdicts and an overall statement of whether `004`'s finding generalizes to the real model.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For every configuration that can be measured, a `Buffer`-storage variant passes its smoke-check (runs, coherent output) before any performance number from it is reported.
- **SC-002**: Every measurable configuration has both a `Texture3D` and a `Buffer` e2e prefill/decode tok/s pair, captured under identical methodology, workload, and device.
- **SC-003**: A reader can determine, for each configuration, whether the real end-to-end result agrees with `004`'s microbenchmark-level finding, without needing to re-derive anything from raw logs.
- **SC-004**: Any configuration that could not be measured (blocked export, failed smoke-check) is stated explicitly with its reason, never silently absent from the report.

## Assumptions

- Producing a `Buffer`-storage variant of the exported model may require a code change (likely to the Vulkan partitioner or memory-planning pass, since nothing in the current export pipeline is known to expose a flag for this) — the exact mechanism is determined during planning, not assumed here; if it turns out to be infeasible within reasonable scope for some or all configurations, that is reported per FR-006 rather than blocking the whole feature.
- **Texture3D-vs-Buffer numerical equivalence is assumed, not independently re-verified by this feature** — it is treated as an existing guarantee of ExecuTorch/the Vulkan backend's storage-type abstraction, not something specific to this workstream. Only a basic smoke-check (does it run, is the output coherent) is performed, to catch this feature's own export mistakes.
- Verifying that future WMMA/coopmat shader output matches the accepted tiled baseline is explicitly **out of scope** for this feature — it is a separate, later concern once coopmat is actually wired into the real model.
- This feature measures e2e cost/benefit of the storage switch on the real model; it does not itself attempt to fix the rank-3-output blocker `003` found or make the coopmat dispatch actually reachable — that remains separate, later work.
- Scope matches the constitution's default: all three target models, both `4w`/`8da4w` schemes, fixed 2048-token prefill / 1024-token decode workload.
- Measurements are taken on the same `rocky-ryzen` MiniPC proxy hardware already used throughout this workstream, under the same resource-contention discipline established since `001`.
