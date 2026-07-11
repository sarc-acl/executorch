# Feature Specification: MiniPC No-WMMA Baseline Benchmarks

**Feature Branch**: `001-minipc-baseline-benchmarks`

**Created**: 2026-07-03

**Status**: Draft

**Input**: User description: "On this mini PC, we want to estabilish baseline numbers, thus, perform the e2e and get necessary baseline numbers for our comparisons. The baseline will be default behaviour with out WMMA. we will try int4 quantizations, 4w, 8da4w. need both microbenchmark results of real shapes, and e2e (tok/s) numbers on this device"

## Clarifications

### Session 2026-07-03

- Q: What does the "prefill throughput" end-to-end metric include? → A: Prefill tokens/sec and decode tokens/sec only; time-to-first-token is not part of this baseline's required metrics.
- Q: What prompt/decode lengths should the e2e runs use? → A: Fixed at 2048 prefill (prompt) tokens and 1024 decode (generated) tokens — the only sizes in scope for this feature's end-to-end tier.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Capture end-to-end token-generation baseline (Priority: P1)

As the contributor driving the WMMA/coopmat performance workstream, I need trustworthy end-to-end tokens/sec numbers for the current, non-WMMA behavior on the RDNA3 MiniPC, for each target model and int4 quantization scheme, so that any future WMMA-accelerated result can be judged against real ground truth instead of an assumed or remembered number.

**Why this priority**: Without a recorded, credible end-to-end baseline, no future "WMMA made this faster" claim can be substantiated — this is the number stakeholders and reviewers will actually ask about. It delivers value on its own even before any microbenchmark data exists.

**Independent Test**: Can be fully tested by exporting a model/scheme combination to a `.pte`, running it through the standard model runner on the MiniPC with the WMMA/coopmat dispatch path excluded, and confirming a recorded tokens/sec (prefill and decode) number with enough run metadata to reproduce it.

**Acceptance Scenarios**:

1. **Given** a `.pte` for a target model exported with a target int4 quantization scheme, **When** it is run end-to-end on the MiniPC with the coopmat/WMMA dispatch path excluded, at a fixed 2048-token prefill and 1024-token decode, **Then** a decode tokens/sec and a prefill tokens/sec number are recorded along with the model, scheme, device, and dispatch-path used.
2. **Given** all three target models and both target quantization schemes, **When** the baseline capture is complete, **Then** all six model/scheme combinations have a recorded end-to-end result (or, for any combination that could not run, an explicitly recorded reason instead of a silent gap).
3. **Given** a previously recorded baseline number, **When** the same model/scheme combination is re-run under the same conditions, **Then** the new measurement falls within the recorded run-to-run variance, confirming the number is stable enough to compare against later.

---

### User Story 2 - Capture shader-level microbenchmark baseline at real shapes (Priority: P2)

As the contributor driving this workstream, I need shader-level timing for the actual GEMM/linear shapes that occur during real prefill and decode of the target models, under the non-WMMA dispatch path, so that a future WMMA speedup can be attributed to specific shapes/operators rather than only observed as a diffuse end-to-end change.

**Why this priority**: End-to-end numbers alone can't tell us *where* a future speedup comes from or whether it's noise; per-shape timing is what makes the later comparison analyzable. It's second priority because it refines, rather than replaces, the end-to-end evidence from User Story 1.

**Independent Test**: Can be fully tested by running the existing shader microbenchmark harness against the prefill and decode GEMM shapes of a given model/scheme, on the non-WMMA dispatch path, and confirming a recorded mean time, variance, and iteration count per shape.

**Acceptance Scenarios**:

1. **Given** a target model and quantization scheme, **When** its real prefill-time and decode-time GEMM/GEMV shapes are identified, **Then** each shape is benchmarked individually on the non-WMMA dispatch path and produces a mean time with an associated variance/iteration count.
2. **Given** a recorded microbenchmark result, **When** it is inspected, **Then** it is traceable to a specific (model, scheme, shape, prefill-or-decode) combination rather than reported as an aggregate or unlabeled number.
3. **Given** the full set of target models and schemes, **When** microbenchmarking is complete, **Then** both prefill-regime shapes (at the fixed 2048-token prefill length) and decode-regime shapes (single-token, drawn from the 1024-token decode run) are covered for each of the six model/scheme combinations.

---

### User Story 3 - Produce a reusable, repeatable baseline report (Priority: P3)

As the contributor driving this workstream, I need the baseline numbers and the steps used to produce them organized into one reference report, so that this baseline can be regenerated later (e.g., after a driver or toolchain update) without re-deriving the methodology from scratch, and so future WMMA-comparison work can cite it directly.

**Why this priority**: This is a quality-of-life and durability improvement over Stories 1 and 2 — the numbers already have value without it, but a future comparison effort (or a future contributor) benefits from not having to reconstruct how the baseline was produced.

**Independent Test**: Can be fully tested by handing the report to someone unfamiliar with this effort and confirming they can identify, for any of the six model/scheme combinations, the recorded e2e and microbenchmark numbers and the conditions under which they were captured.

**Acceptance Scenarios**:

1. **Given** the completed baseline capture, **When** the report is produced, **Then** it presents e2e and microbenchmark results for all six model/scheme combinations in one place, organized by model and scheme.
2. **Given** the report, **When** someone wants to reproduce a number, **Then** the report states the device, dispatch-path configuration, and model/scheme export used for that number.

### Edge Cases

- What happens when a model/scheme combination cannot complete an end-to-end run on this device (e.g., the 8B model's memory footprint doesn't fit alongside the MiniPC's iGPU memory budget)? The gap MUST be recorded with the reason, not silently omitted from the report.
- How does the process account for run-to-run variance from thermal throttling or frequency-scaling behavior on the MiniPC during longer end-to-end runs? Repeated or extended runs MUST be used until the reported number is stable enough to be trusted as a baseline, consistent with this workstream's statistical-soundness standard.
- What happens if a shape that occurs during a real prefill/decode run doesn't align with the microbenchmark harness's existing assumptions (e.g., an odd sequence length)? The shape MUST still be benchmarked as encountered rather than rounded to a convenient nearby shape.
- How is a baseline number distinguished from a future coopmat/WMMA-enabled number for the same model/scheme, given both will eventually exist? Every recorded number MUST be labeled with which dispatch path (non-WMMA baseline vs. WMMA-enabled) produced it.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The baseline capture MUST cover all combinations of the three target models (Llama 3.1 8B, Llama 3.2 3B, Llama 3.2 1B) and the two target int4 quantization schemes (4w, 8da4w) — six combinations total — on the RDNA3 MiniPC device.
- **FR-002**: For each of the six combinations, the capture MUST produce an end-to-end result — decode tokens/sec and prefill tokens/sec, and no other e2e metric is required — measured at a fixed 2048-token prefill followed by a 1024-token decode, using a `.pte` exported for that specific model/scheme.
- **FR-003**: For each of the six combinations, the capture MUST produce shader-level microbenchmark results (mean time, variance, iteration count) for the GEMM/GEMV shapes actually exercised during that model's real prefill and decode, not synthetic or convenience shapes.
- **FR-004**: All measurements MUST be produced with the coopmat/WMMA dispatch path excluded, so they represent this workstream's "before" state; each recorded number MUST carry a label identifying it as such.
- **FR-005**: All measurements MUST follow statistically sound methodology — explicit warmup, multiple iterations, and a reported variance/spread — rather than single-shot timing, at both the microbenchmark and end-to-end tiers.
- **FR-006**: Each recorded result MUST carry enough context (device identity, model, scheme, dispatch-path label, and export used) to be unambiguous when referenced by a later comparison effort.
- **FR-007**: Any model/scheme combination that cannot be measured (export failure, out-of-memory, crash) MUST be recorded as an explicit, reasoned gap rather than omitted without explanation.
- **FR-008**: The complete set of baseline results MUST be organized into a single reusable report that a later feature can reference for delta comparisons without re-running this capture.

### Key Entities

- **Benchmark Configuration**: the tuple of (target model, quantization scheme, dispatch path) under measurement; six baseline configurations in this feature (WMMA-enabled configurations are out of scope here).
- **End-to-End Result**: a recorded (decode tokens/sec, prefill tokens/sec) pair for one Benchmark Configuration, measured at the fixed 2048-token prefill / 1024-token decode configuration, plus the run metadata needed to reproduce it.
- **Microbenchmark Result**: a recorded (shape, mean time, variance, iteration count) tuple for one GEMM/GEMV shape encountered by one Benchmark Configuration, tagged as prefill-regime or decode-regime.
- **Baseline Report**: the consolidated collection of all End-to-End Results and Microbenchmark Results for this feature, organized for reuse by future comparison work.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A decode tokens/sec and a prefill tokens/sec number, both measured at the fixed 2048-token prefill / 1024-token decode configuration, exist for all six (model × scheme) combinations on the MiniPC, or an explicit recorded reason exists for any combination that could not be measured.
- **SC-002**: Shader-level microbenchmark timing exists for the real prefill-regime and decode-regime GEMM/GEMV shapes of all six (model × scheme) combinations.
- **SC-003**: Every recorded number can be traced, without guessing, to the specific model, scheme, device, and dispatch path that produced it.
- **SC-004**: Re-measuring any single (model × scheme) combination under the same conditions reproduces a result within the originally recorded variance, confirming the baseline is stable rather than a one-off reading.
- **SC-005**: A future contributor unfamiliar with this effort can locate and correctly interpret any of the six combinations' results from the produced report alone, without asking how they were obtained.

## Assumptions

- The project's existing model-export tooling already supports producing Llama 3.1 8B, Llama 3.2 3B, and Llama 3.2 1B `.pte` files for the Vulkan backend at the 4w and 8da4w quantization schemes; this feature consumes that tooling rather than building new export capability.
- "Tokens/sec" is reported as two separate numbers, decode tokens/sec and prefill tokens/sec, at a fixed 2048-token prefill and 1024-token decode; time-to-first-token is not required for this baseline even though it appears in some existing project benchmark tables.
- Each model/scheme `.pte` export is configured to support at least a 3072-token total sequence length (2048 prefill + 1024 decode), since the fixed e2e sizes exceed this repository's more common ~2048-token default context length.
- The RDNA3 MiniPC (`rocky-ryzen`) is the only device in scope for this feature; Android/Samsung on-device measurement is explicitly out of scope and will be a separate future effort.
- Because this branch already contains merged coopmat/WMMA dispatch code, producing a genuine "no-WMMA" baseline requires a controlled way to exclude that dispatch path for the duration of these measurements (for example, measuring from a pre-coopmat build or via a build/runtime toggle); the specific mechanism is a planning/implementation decision, not a scope decision, and does not change what this feature must produce.
- fp16, 8w, and 8da8w quantization schemes are explicitly out of scope for this baseline; only 4w and 8da4w are measured, matching the schemes already validated as coopmat-eligible in this workstream.
- Measuring the coopmat/WMMA-enabled numbers themselves is out of scope for this feature — this feature produces only the "before" side of the comparison.
