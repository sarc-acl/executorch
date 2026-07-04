# Feature Specification: Linear Shader Storage-Type Baseline Study (Texture3D vs. Buffer)

**Feature Branch**: `004-linear-storage-comparison`

**Created**: 2026-07-04

**Status**: Draft

**Input**: User description: "Perform a study on linear shaders's texture shader vs buffer shaders. Because default behaviour is executorch uses texture shader. But WMMA requires buffer storage. Thus, we need to compare the baseline at both texture and buffer. To estabilish correct understanding"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Isolate the storage-type effect on prefill linear performance (Priority: P1)

As the contributor driving the WMMA/coopmat performance workstream, I need to know how much of any future WMMA speedup would actually come from switching linear activations from today's default `Texture3D` storage to the `Buffer` storage WMMA requires — versus how much comes from the cooperative-matrix hardware path itself — so that when I eventually measure a coopmat speedup, I can attribute it correctly instead of crediting WMMA for a gain that was really just a storage-type change.

**Why this priority**: The prior feature (`003-wmma-shader-candidates`) found that today's linear activations use `Texture3D` storage, one of two reasons the existing coopmat shader never fires. Before spending effort changing that storage type, I need to know whether the switch itself is free, costly, or beneficial on its own — this is the foundational measurement everything else in this study builds on.

**Independent Test**: Can be fully tested by running the same linear operation, at the same shape and quantization scheme, through the same (non-coopmat, tiled) shader twice — once with `Texture3D`-stored activations, once with `Buffer`-stored activations — and comparing the two timings.

**Acceptance Scenarios**:

1. **Given** a prefill-shaped (M=2048) linear operation, **When** it is measured at `Texture3D` storage and at `Buffer` storage using the same tiled (non-coopmat) dispatch in both cases, **Then** both measurements are reported side by side with their relative difference, and the comparison is not contaminated by the operation silently switching to the coopmat dispatch just because `Buffer` storage happens to also be coopmat-eligible.
2. **Given** the two measurements for one case, **When** the relative difference is reported, **Then** it states whether the difference is a real, reproducible effect or within measurement noise, using the same statistically sound methodology already established for this workstream.

---

### User Story 2 - Extend the comparison to decode-regime linear performance (Priority: P2)

As the contributor driving this workstream, I need the same `Texture3D`-vs-`Buffer` comparison for the decode regime (M=1) too, so that I understand the full performance picture of a storage-type change — since a real model change would affect the same activation tensors at both prefill and decode, not just prefill.

**Why this priority**: Decode's dispatch algorithm doesn't change based on storage type (it already always uses the same GEMV path, established in prior features), but the storage type itself might still have its own performance cost at decode — a real storage-type change to the model would apply to both regimes, so leaving decode unmeasured would give an incomplete picture.

**Independent Test**: Can be fully tested by running the same decode-shaped (M=1) linear operation at both storage types and comparing the two timings, independent of Story 1.

**Acceptance Scenarios**:

1. **Given** a decode-shaped (M=1) linear operation, **When** it is measured at `Texture3D` storage and at `Buffer` storage, **Then** both measurements are reported side by side, and the report does not claim this comparison says anything about coopmat eligibility (decode never reaches that dispatch choice regardless of storage, per prior findings) — it is purely about storage-type cost at decode.

---

### User Story 3 - Produce a consolidated storage-type comparison report (Priority: P3)

As the contributor driving this workstream, I need one consolidated report covering every target model, quantization scheme, and regime, with a clear verdict on whether switching to `Buffer` storage helps, hurts, or is neutral — so I can decide whether pursuing the storage-type change (identified as a coopmat blocker) is likely to pay off before investing effort in it.

**Why this priority**: Stories 1 and 2 produce the individual measurements; this turns them into an actionable answer to the question that motivated the whole study.

**Independent Test**: Can be fully tested by taking the completed measurements from Stories 1 and 2 and producing a report with an explicit verdict, verifiable by checking that every configuration's comparison and verdict trace back to real measured numbers.

**Acceptance Scenarios**:

1. **Given** the measurements for all three target models, both quantization schemes, and both regimes, **When** the consolidated report is produced, **Then** it states, for prefill and decode separately, whether `Buffer` storage's performance is close enough to `Texture3D`'s to be considered a "free" prerequisite for enabling coopmat, or whether it carries its own measurable cost or benefit.
2. **Given** the report, **When** a reader looks for the bottom-line answer, **Then** they can find it without needing to read every individual per-shape measurement.

### Edge Cases

- What happens if, for a given shape, the `Buffer`-storage variant would also satisfy every other coopmat-eligibility condition and could silently dispatch the coopmat shader instead of the intended tiled comparison? The measurement MUST force the tiled dispatch (the same mechanism already used to produce this workstream's no-WMMA baseline) so the comparison isolates storage type alone, not storage type plus a silent algorithm change.
- What happens if a particular shape's `Buffer`-storage variant cannot be constructed or dispatched at all? That case MUST be reported explicitly as infeasible, not silently skipped or estimated.
- What happens if repeated runs show substantially different variance between the two storage types? The same statistical discard/steady-state rules already established for this workstream apply equally to both storage types — no separate standard for either.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST measure every (model, quantization scheme, regime, operation) case already covered by this workstream's existing no-WMMA baseline microbenchmark at `Buffer` storage, in addition to the `Texture3D` storage already measured, using the same tiled (non-coopmat) dispatch in both cases.
- **FR-002**: The `Buffer`-storage measurements MUST use the same mechanism already established for forcing the tiled dispatch path, to guarantee the comparison isolates storage type from any algorithm change.
- **FR-003**: The system MUST cover all three target models (Llama 3.1 8B, 3.2 3B, 3.2 1B), both target quantization schemes (`4w`, `8da4w`), and both regimes (prefill at the fixed size already used, decode at M=1) already established for this workstream.
- **FR-004**: Measurements MUST follow this workstream's already-established statistically sound methodology (repeated runs, steady-state reporting, no resource contention during capture) rather than single-shot numbers.
- **FR-005**: For every case, the system MUST report the `Texture3D` and `Buffer` timings side by side along with their relative difference.
- **FR-006**: The system MUST state, for each case, whether the observed difference is a real, reproducible effect or within measurement noise.
- **FR-007**: The system MUST NOT claim that the decode-regime comparison says anything about coopmat eligibility, since decode's dispatch path does not depend on storage type; it MUST be presented purely as a storage-type cost/benefit measurement.
- **FR-008**: The system MUST produce one consolidated report with an explicit, easy-to-find verdict — for prefill and decode separately — on whether `Buffer` storage is effectively free, costly, or beneficial relative to today's `Texture3D` default, independent of any WMMA/coopmat effect.
- **FR-009**: Any case that cannot be measured (infeasible construction/dispatch at `Buffer` storage) MUST be reported explicitly as such, not silently omitted.

### Key Entities

- **Storage Comparison Case**: one (model, scheme, regime, operation, shape) combination, carrying its `Texture3D` timing, its `Buffer` timing, their relative difference, and a noise-vs-real-effect determination.
- **Storage Comparison Report**: the consolidated document covering every case, plus the top-level prefill/decode verdicts that answer the motivating question.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For every (model, scheme, regime) combination, a reader can see the `Texture3D` and `Buffer` timings side by side and determine whether switching storage alone would help, hurt, or be neutral to performance, without needing to understand GPU driver internals.
- **SC-002**: The report explicitly distinguishes, for every case, a real reproducible difference from measurement noise, using this workstream's established statistical standard.
- **SC-003**: A reader can find the bottom-line answer to "does switching to Buffer storage cost us anything?" for both prefill and decode without reading individual per-shape numbers.
- **SC-004**: The report enables a go/no-go judgment on pursuing the storage-type change identified as a coopmat blocker, independent of whatever additional benefit WMMA itself might separately provide.

## Assumptions

- This study reuses and extends this workstream's existing no-WMMA baseline microbenchmark (a controlled, synthetic-tensor benchmark, not a full end-to-end model run) rather than building new device or export infrastructure.
- Scope matches what this workstream has already established: the same three models, the same two quantization schemes (`4w`, `8da4w`); `8w`/`8da8w` remain out of scope.
- This study measures today's tiled-dispatch performance at both storage types; it does not attempt to fix the blockers found in the prior feature or make the coopmat dispatch actually reachable in the real end-to-end model — that would be separate, later work.
- Measurements are taken on the same `rocky-ryzen` MiniPC proxy hardware already used throughout this workstream, under the same resource-contention discipline established earlier (no concurrent CPU/GPU-heavy processes during capture).
