# Feature Specification: WMMA-Optimizable Shader Candidates Report

**Feature Branch**: `003-wmma-shader-candidates`

**Created**: 2026-07-04

**Status**: Draft

**Input**: User description: "Given the ETdump report, analysis all the shaders that was used, analysis which of them could be benifit from WMMA. which shaders has already beeing using WMMA (that I wrote). Generate a report of potential WMMA-optimizable shaders. for all models, and all 4w and 8da4w path."

## Clarifications

### Session 2026-07-04

- Q: What should the ranked report sort candidates by — absolute time or relative percentage-of-phase? → A: Primary sort by absolute time (us), summed/considered across configurations — this is what determines real engineering ROI across the fleet and aligns with the constitution's existing "Llama 3.1 8B first" priority order. Relative percentage-of-phase is still shown alongside every entry for context, not dropped.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Classify every shader from the profiling report by WMMA candidacy (Priority: P1)

As the contributor driving the WMMA/coopmat performance workstream, I need every shader observed in the ETDump profiling report classified as either "already WMMA-capable and in effect," "WMMA-capable code exists but isn't in effect today (with the specific reason why)," "no WMMA implementation exists yet," or "not applicable" (not a matrix multiplication), so that I know exactly what work remains instead of re-deriving this by re-reading shader source myself.

**Why this priority**: Without this classification, "the workstream has WMMA shaders" is a vague, unverified claim — I need to know precisely which real, currently-measured shader invocations would actually benefit, and which already-written coopmat code is silently not firing (and why), before spending further effort. This is the foundational deliverable everything else refines.

**Independent Test**: Can be fully tested by taking one already-profiled configuration's shader list (from the ETDump report) and producing a classification with a stated reason for every entry, verifiable by reading the relevant dispatch code directly.

**Acceptance Scenarios**:

1. **Given** the per-shader breakdown for one configuration's prefill phase, **When** each shader is classified, **Then** every shader has exactly one of the four classifications, and every "exists but not in effect" classification names the specific blocking condition (e.g., a storage-type mismatch, a shape-alignment requirement, a device-capability gate) rather than a generic "not used" statement.
2. **Given** the same configuration's decode phase, **When** its shaders are classified, **Then** the classification accounts for whatever is different about decode's dispatch path compared to prefill's, rather than reusing prefill's reasons uncritically.
3. **Given** a shader that already has a WMMA-capable implementation written for it, **When** the report states this, **Then** it names the specific existing shader/kernel that implements it, not just "yes, possible."

---

### User Story 2 - Extend the classification to all six configurations (Priority: P2)

As the contributor driving this workstream, I need the same classification for all three target models at both the `4w` and `8da4w` quantization schemes, so that I can confirm the pattern holds everywhere (or find out where it doesn't) rather than generalizing from a single configuration.

**Why this priority**: A single configuration's classification proves the method and surfaces the big candidates; confirming it across all six configurations is what makes the resulting priority call trustworthy rather than a guess extrapolated from one model size.

**Independent Test**: Can be fully tested by repeating User Story 1's classification against the remaining five configurations' existing profiling data and confirming every shader in each has a classification.

**Acceptance Scenarios**:

1. **Given** all six configurations' profiling data, **When** classification is complete, **Then** each configuration's prefill and decode phases both have a full shader classification.
2. **Given** the six classifications, **When** compared side by side, **Then** it's possible to tell whether the same operations are blocked for the same reasons across every model size and scheme, or whether any configuration differs.

---

### User Story 3 - Produce a ranked WMMA-optimization candidates report (Priority: P3)

As the contributor driving this workstream, I need the classified shaders rolled up into a single ranked report — ordered by how much measured time each candidate represents, and separating "fix a blocking condition in already-written code" from "author a new shader" — so that I can decide what to work on next without re-deriving priority from raw data myself.

**Why this priority**: Stories 1 and 2 produce the raw classification; this is what turns it into something actually actionable — a prioritized list, not a pile of facts.

**Independent Test**: Can be fully tested by taking the completed classifications from Stories 1 and 2 and producing a ranked list, verifiable by checking that entries are ordered by their attached time-share and that each is labeled with the correct kind of follow-up work, without needing to re-run any profiling or re-classify anything.

**Acceptance Scenarios**:

1. **Given** the classifications for all six configurations, **When** the ranked report is produced, **Then** candidates are grouped into "existing WMMA implementation blocked by a fixable condition" versus "no WMMA implementation exists yet," each ordered by absolute time within its group, with relative percentage-of-phase shown alongside each entry.
2. **Given** the ranked report, **When** a reader looks at the top entries, **Then** they can identify the single largest absolute-time optimization opportunity across the whole model/scheme matrix without reading any raw kernel name or per-invocation data.

### Edge Cases

- What happens for a shader that is a matrix multiplication but whose WMMA-eligibility cannot be confidently determined from available information? It MUST be recorded as an explicit "uncertain / needs further investigation" case, not silently classified as a guess.
- What happens for decode-phase linear operations, given the profiling report already established that decode's dispatch path structurally never reaches the coopmat check regardless of any other condition? These MUST be classified with that specific structural reason, not lumped in with prefill's classification or its reasons.
- What happens if a shader already has a WMMA-capable counterpart, but that counterpart is itself never actually observed firing in any of the six configurations' captured data (because the captures were deliberately taken with WMMA disabled)? The report MUST say so explicitly, so a reader does not mistakenly conclude WMMA is already active in production today.
- What happens for a shader whose blocking condition, even if understood, might not have an obvious or low-risk fix? The report records the blocking condition and that a fix's feasibility is undetermined; it does not propose or evaluate a specific fix design.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: For each of the six baseline configurations already profiled (three models × `4w`/`8da4w`), the system MUST classify every distinct shader/kernel appearing in that configuration's prefill and decode breakdowns into exactly one of: (a) WMMA-capable and currently in effect, (b) a WMMA-capable implementation exists but is not currently in effect, (c) no WMMA-capable implementation exists yet, (d) not applicable (not a matrix multiplication).
- **FR-002**: Every shader classified as (b) MUST have its specific blocking condition named (e.g., an output-storage-type mismatch, a shape/alignment requirement not met, a device-capability gate, decode's structural GEMV exclusion) rather than a generic "not used" statement.
- **FR-003**: Every shader classified as (a) or (b) MUST name the specific existing shader/kernel that implements or would implement the WMMA path for that operation.
- **FR-004**: The classification MUST treat the prefill and decode phases of each configuration separately, since they are already known to have structurally different dispatch behavior.
- **FR-005**: Each classified shader MUST carry the time-share (percentage of phase time) it represented in the ETDump profiling report, so its potential impact is visible alongside its classification.
- **FR-006**: The system MUST produce a single consolidated report ranking classification-(b) and classification-(c) shaders primarily by absolute time (summed/considered across the six configurations), with each entry's relative percentage-of-phase shown alongside for context; (b) and (c) MUST be kept as visibly separate groups since they represent different kinds of follow-up work (fixing an existing implementation's blocking condition vs. authoring a new one).
- **FR-007**: The report MUST cover all three target models (Llama 3.1 8B, 3.2 3B, 3.2 1B) and both target quantization schemes (`4w`, `8da4w`) — six configurations total, consistent with the scope already established for this workstream.
- **FR-008**: Any shader that cannot be confidently classified MUST be recorded as an explicit "uncertain" entry rather than omitted or guessed.
- **FR-009**: The report MUST NOT claim any shader is "already using WMMA in production today" unless that shader was actually observed dispatching via its WMMA-capable kernel in the profiling data itself — since all six configurations were captured with WMMA deliberately disabled, the report is expected to state that no classification-(a) shaders were observed, rather than implying otherwise.

### Key Entities

- **Shader Classification**: one entry per distinct shader/kernel within one configuration's one phase — classification (a/b/c/d), blocking condition (if b), the existing or prospective WMMA shader name (if a or b), and the time-share carried over from the ETDump profiling report.
- **Optimization Candidate**: a Shader Classification of type (b) or (c), promoted into the ranked report with its absolute time, relative percentage-of-phase, and follow-up-work kind (fix vs. new authoring).
- **Candidates Report**: the consolidated, ranked document covering all six configurations, built entirely from already-existing Shader Classifications — no new profiling data is captured to produce it.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For at least one configuration, every shader appearing in its prefill and decode breakdowns has a stated classification and, where applicable, a specific named reason — verifiable against the dispatch code directly.
- **SC-002**: All six configurations have complete shader classifications for both phases, or an explicit "uncertain" marker for any shader that couldn't be confidently classified.
- **SC-003**: A reader can identify the single largest absolute-time optimization opportunity across the entire model/scheme matrix using the ranked report alone, without reading raw per-kernel data.
- **SC-004**: For every optimization candidate in the ranked report, a reader can tell whether the next step is fixing a specific condition in already-written code or authoring a new shader, using the report alone.
- **SC-005**: The report does not claim any shader is actively using WMMA in today's captured data unless that is literally true of the profiling data it was built from.

## Assumptions

- This feature analyzes the same six (model × scheme) configurations, same device, and same `tiled_baseline` captures already produced by the prior baseline-benchmarks and ETDump-profiling features; no new profiling captures are taken to produce this report.
- "WMMA-capable implementation exists" refers to the coopmat shaders already established in this workstream (the `linear_qw_coopmat`/`linear_dq8ca_qw_coopmat` family for `4w`/`8da4w` linears, and the generic `coopmat_mm`/matmul-coopmat path intended for future SDPA work) — this feature identifies where such code applies or is missing; it does not author any new shader or fix any blocking condition itself.
- Determining whether a named blocking condition has a feasible, low-risk fix is out of scope for this feature; the report names the blocker, not a fix design or its risk assessment.
- `8w` and `8da8w` quantization schemes remain out of scope, consistent with the on-device scope already established for this workstream.
