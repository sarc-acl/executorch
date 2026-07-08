# Feature Specification: End-to-End Speedup Target and Validation

**Feature Branch**: `005-e2e-speedup-target`

**Created**: 2026-07-04

**Status**: Draft

**Input**: User description: "Long-Term Vision, the goal of my work is to improve the performance of executorch (which does now have WMMAs, and the only WMMA shaders you saw is written by me (Yanwen Xu) before). So our end goal is, improve executorch's e2e tok/s at 2048 prefill and 1024 decode for all 3 models by X amount of speedup."

## Clarifications

### Session 2026-07-04

- Q: Decode is ~90% of total e2e wall-clock time across all three baseline models (verified: 8B 90.2%, 3B 91.2%, 1B 90.8%), and `003` found decode's GEMV dispatch has no identified WMMA fix at all — only prefill's linear GEMMs have candidates ready to pursue. Should the target be a realistic near-term combined e2e number, or scoped to prefill specifically, with combined e2e tracked but not targeted until decode has a fix? → A: Scope the primary near-term target to prefill tok/s specifically; combined e2e is measured and reported every time but is not the pass/fail bar until decode has its own identified fix. The contributor's own expectation is that prefill speedup could exceed 2x, so the prefill target is set at **at least 2x (100%) prefill tok/s improvement**, with anything beyond treated as bonus, not a ceiling.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Define an unambiguous prefill success target (Priority: P1)

As the contributor driving this WMMA/coopmat performance workstream, I need a specific, numeric target for how much faster prefill token generation should become — at the same fixed prefill (2048 tokens) workload already used for baseline measurement, across all three target models, with combined e2e (including the fixed 1024-token decode workload) tracked and reported alongside but not itself a pass/fail bar — so that "the workstream succeeded" has one unambiguous, agreed-upon meaning instead of being judged after the fact by whatever number happens to come out.

**Why this priority**: Every prior step in this workstream (the baseline numbers, the per-shader profiling, the optimization candidates, the storage-type study) exists in service of this end goal. Without a concrete target defined now, there is no way to later say whether the effort succeeded, partially succeeded, or fell short — the target must exist before implementation work proceeds, not be invented after measuring the result.

**Independent Test**: Can be fully tested by confirming a specific numeric target (or per-model targets) is written down and traceable to the already-established baseline numbers, independent of any implementation work.

**Acceptance Scenarios**:

1. **Given** the already-established baseline prefill tok/s numbers (from the baseline-benchmarks feature) for all three models, **When** the target is defined, **Then** it states a specific improvement amount (at least 2x / 100% prefill tok/s improvement, not merely "faster") expressed relative to those exact baseline numbers, with combined e2e tracked and reported alongside but not itself the pass/fail bar.
2. **Given** the target, **When** it is reviewed, **Then** it is clear that it applies per model (each of the three judged individually, not averaged), and that combined e2e improvement remains a tracked, reported number rather than a second pass/fail bar, since decode currently has no identified fix.

---

### User Story 2 - Re-measure end-to-end performance once optimization work exists (Priority: P2)

As the contributor driving this workstream, once any of the optimization work identified in this workstream's prior analysis has actually been implemented, I need to re-measure end-to-end tok/s using the exact same methodology, workload, and device as the original baseline, so the "before" and "after" numbers are directly and fairly comparable.

**Why this priority**: A target is meaningless without a trustworthy, apples-to-apples "after" measurement — this is what actually determines whether the target from Story 1 was met.

**Independent Test**: Can be fully tested by re-running the same fixed prefill/decode e2e measurement already established for the baseline, once a build with optimization changes exists, and confirming the methodology (device, workload size, statistical discipline) matches the original baseline exactly.

**Acceptance Scenarios**:

1. **Given** a build containing optimization changes from this workstream, **When** end-to-end tok/s is re-measured, **Then** it uses the same device, the same fixed prefill (2048) and decode (1024) workload, and the same statistically sound methodology (repeated runs, no resource contention) already established for the baseline.
2. **Given** the re-measurement, **When** it is compared to the baseline, **Then** the comparison is not contaminated by an unrelated confound (for example, a storage-type change measured in isolation elsewhere in this workstream) being credited as part of the observed speedup without being called out.

---

### User Story 3 - Report the outcome against the target, per model (Priority: P3)

As the contributor driving this workstream, I need a final report stating, for each of the three models, whether the prefill target from Story 1 was met, exceeded, missed, or regressed — alongside the tracked combined e2e change — so I can honestly communicate the workstream's outcome rather than leaving it to interpretation.

**Why this priority**: This is the actual deliverable that closes the loop opened by Story 1's target — without it, the target and the re-measurement exist but nothing states the verdict.

**Independent Test**: Can be fully tested by taking Story 1's target and Story 2's re-measurement and producing a report whose per-model verdicts are directly traceable to those two artifacts, without needing to re-derive anything.

**Acceptance Scenarios**:

1. **Given** the target and the re-measured numbers, **When** the report is produced, **Then** each of the three models has an explicit stated prefill outcome (met / exceeded / missed / regressed) with the actual observed prefill speedup number next to the target, plus the combined e2e change reported alongside (not judged pass/fail).
2. **Given** a model whose prefill outcome is "missed" or "regressed," **When** the report is read, **Then** it does not obscure or average away that shortfall against the other models' results.

### Edge Cases

- What happens if optimization work only ever addresses part of the pipeline (for example, prefill's linear operations) while another part (for example, decode) has no available fix at the time of re-measurement? This is the expected near-term case: the prefill outcome is judged against the FR-001 target as normal, and the report MUST also show the actual combined e2e number as measured (not extrapolated) — but MUST NOT present combined e2e as if it were expected to hit the prefill target, or as the deciding pass/fail number.
- What happens if a model's re-measured number is *worse* than baseline? This MUST be reported as a regression, explicitly, not folded into "missed target" language that could be read as merely "no improvement."
- What happens if the re-measurement cannot be produced under the same methodology as the baseline (for example, the device is unavailable, or a workload size had to change)? The comparison MUST be marked as not directly comparable rather than presented as if it were.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST define a specific, numeric primary speedup target scoped to prefill tok/s — at least 2x (100% improvement) per model, with anything beyond treated as bonus rather than a ceiling — expressed relative to the already-established baseline-benchmarks feature's numbers. Combined e2e tok/s MUST be measured and reported alongside every case but is a tracked metric, not a second pass/fail bar, until decode has its own identified fix.
- **FR-002**: The target MUST be defined before any optimization implementation begins, so it cannot be adjusted in hindsight to match whatever result is later measured.
- **FR-003**: Once optimization work exists, the system MUST re-measure both prefill tok/s and combined e2e tok/s using the identical device, the identical fixed prefill (2048 tokens) and decode (1024 tokens) workload, and the identical statistically sound methodology (repeated runs, steady-state reporting, no resource contention during capture) already established for the baseline.
- **FR-004**: The system MUST report, per model, the actual observed prefill speedup and an explicit outcome (met / exceeded / missed / regressed) against the target from FR-001, alongside the observed combined e2e change (reported, not judged pass/fail).
- **FR-005**: The system MUST NOT attribute measured speedup to the WMMA/coopmat work if it is actually explained by an unrelated confound already isolated elsewhere in this workstream (for example, a storage-type change measured independently) — any such confound's contribution MUST be called out separately.
- **FR-006**: If optimization work only addresses part of the pipeline (for example, prefill only, which is the expected near-term case since decode has no identified fix), the report MUST state the combined e2e outcome as actually measured, not a partial or extrapolated figure presented as the whole result — and MUST NOT imply combined e2e was expected to hit the prefill target.
- **FR-007**: A per-model outcome of "missed" or "regressed" MUST be stated explicitly and MUST NOT be obscured by averaging across models.
- **FR-008**: If a re-measurement cannot use the exact same methodology as the original baseline, the system MUST mark that comparison as not directly comparable rather than presenting it as equivalent.

### Key Entities

- **Speedup Target**: the numeric prefill tok/s improvement goal (from FR-001, at least 2x per model), defined once before implementation begins; combined e2e is tracked alongside but is not part of this target until decode has its own fix.
- **Re-Measurement**: the "after" prefill tok/s and combined e2e tok/s numbers for a given model, captured under the same methodology as the original baseline, once optimization work exists.
- **Outcome Report**: the final per-model comparison of Re-Measurement's prefill number against the Speedup Target, with an explicit met/exceeded/missed/regressed verdict for each model, plus the tracked (not judged) combined e2e change.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A specific, numeric prefill speedup target (at least 2x) is documented and traceable to the baseline-benchmarks feature's existing prefill numbers before any optimization implementation work begins.
- **SC-002**: Once optimization work exists, each of the three models has re-measured prefill tok/s and combined e2e tok/s numbers captured under the exact same device, workload, and statistical methodology as the original baseline.
- **SC-003**: A reader can determine, for each of the three models individually, whether the prefill target was met, exceeded, missed, or regressed, without needing to re-derive numbers from raw logs — and can see the combined e2e change reported alongside without it being mistaken for a second pass/fail bar.
- **SC-004**: The reported prefill speedup for each model is attributable specifically to the WMMA/coopmat optimization work, with any already-isolated confound (such as a storage-type change) called out separately rather than folded into the headline number.

## Assumptions

- The primary target is scoped to prefill tok/s (at least 2x / 100% improvement per model) rather than combined e2e, because decode accounts for ~90% of total e2e wall-clock time across all three baseline models (verified: 8B 90.2%, 3B 91.2%, 1B 90.8%) and currently has no identified WMMA fix at all (`003` found decode's GEMV dispatch structurally cannot reach any existing coopmat implementation) — a combined e2e target would be effectively unreachable using only the work this workstream has identified so far. Combined e2e remains a tracked, reported metric throughout.
- "The baseline" refers specifically to the numbers already captured in `001-minipc-baseline-benchmarks`, on the same `rocky-ryzen` MiniPC proxy hardware, at the same fixed 2048-token prefill / 1024-token decode workload.
- This feature defines the target and the validation methodology; it does not itself implement any optimization — the actual WMMA/coopmat fix (informed by `003`'s candidates and `004`'s storage-type study) is separate, later work.
- Scope matches this workstream's established boundaries: the same three models (Llama 3.1 8B, 3.2 3B, 3.2 1B), the same two quantization schemes (`4w`, `8da4w`) where applicable.
- Should a decode-specific WMMA fix be identified in future work, this target should be revisited to add a combined e2e pass/fail bar at that point — out of scope for this feature.
