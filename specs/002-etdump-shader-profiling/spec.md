# Feature Specification: ETDump E2E Shader Profiling Breakdown

**Feature Branch**: `002-etdump-shader-profiling`

**Created**: 2026-07-04

**Status**: Draft

**Input**: User description: "Use ETDump to analysis the E2E results, breakdown the shaders that was actually called, and each shader's individual time, matmul shape, portion of e2e total time, and other useful profiling information."

## Clarifications

### Session 2026-07-04

- Q: When a shader/kernel is invoked multiple times within a phase (e.g., the same linear kernel called once per transformer layer), should the breakdown report one row per call site, one aggregated row per unique kernel+shape, or both? → A: Both — an aggregated table (kernel + shape, total time, invocation count, % of phase) is the primary report; the raw per-invocation events are preserved in a companion data file for anyone who needs per-layer detail.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Break down where one configuration's e2e time actually goes (Priority: P1)

As the contributor driving the WMMA/coopmat performance workstream, I need a per-shader breakdown (name, individual time, matmul shape, and share of total time) for at least one of the baseline configurations already captured in the MiniPC baseline benchmarks, so that I know exactly which operators dominate prefill and decode time before deciding where WMMA/coopmat work will matter most.

**Why this priority**: Without this breakdown, "the baseline is 9.28 tok/s decode" is a single opaque number — nobody can tell whether that time is dominated by matmuls (where coopmat could help), by CPU↔GPU copies, or by something else entirely (where coopmat cannot help at all). This is the foundational deliverable; everything else in this feature extends it.

**Independent Test**: Can be fully tested by running one already-exported configuration with ETDump enabled, and confirming a report listing every shader/kernel invoked, its time, its matmul shape (where applicable), and its percentage of the phase's total time.

**Acceptance Scenarios**:

1. **Given** one of the six baseline `.pte` configurations and its capture conditions (device, dispatch path) from the prior baseline-benchmarks feature, **When** it is run end-to-end with ETDump enabled, **Then** an aggregated breakdown is produced listing each distinct kernel+shape invoked during prefill and during decode, each with a total time, an invocation count, and a percentage of that phase's total time — with the underlying raw per-invocation events preserved in a companion file.
2. **Given** a shader in the breakdown that performs a matrix multiplication, **When** its entry is inspected, **Then** its matmul shape (M, K, N) is present and matches the real per-layer shapes already established for that model.
3. **Given** the full breakdown for one configuration, **When** all per-shader/category times are summed per phase, **Then** the total is reconcilable against that phase's measured wall-clock time from the baseline benchmarks (documented, not silently divergent).

---

### User Story 2 - Extend the breakdown to all six baseline configurations (Priority: P2)

As the contributor driving this workstream, I need the same per-shader breakdown for all six (model × scheme) baseline configurations, so that I can compare where time goes across model sizes and quantization schemes, not just within one.

**Why this priority**: A single configuration's breakdown proves the method works; comparing across all six is what actually informs where the WMMA workstream should focus (e.g., "the FFN dominates prefill everywhere" vs. "it depends on model size").

**Independent Test**: Can be fully tested by running the same profiling procedure from User Story 1 against each of the remaining five configurations and confirming a breakdown exists for all six.

**Acceptance Scenarios**:

1. **Given** all six baseline configurations, **When** profiling is complete, **Then** each has its own per-shader breakdown (or an explicit, reasoned gap if one configuration could not be profiled).
2. **Given** the six breakdowns, **When** they are compared side by side, **Then** it is possible to tell, for each model, whether the same shaders/categories dominate time across both quantization schemes.

---

### User Story 3 - Summarize the breakdown into meaningful categories (Priority: P3)

As the contributor driving this workstream, I need the raw per-shader breakdown rolled up into a small set of meaningful categories (e.g., attention projection, feed-forward, output/vocabulary projection, non-shader overhead like CPU↔GPU copies), so that I can communicate "where the time goes" without making a reader parse dozens of individual kernel names.

**Why this priority**: This is a readability/communication improvement over Stories 1 and 2 — the raw data already has value without it, but a rolled-up summary is what actually gets shared and acted on.

**Independent Test**: Can be fully tested by taking an existing per-shader breakdown and confirming it can be grouped into named categories whose percentages sum to the phase total, without needing to re-run any profiling.

**Acceptance Scenarios**:

1. **Given** a per-shader breakdown for one configuration, **When** it is rolled up into categories, **Then** each category's percentage is shown and the categories account for the full phase time (including a category for non-shader overhead, if present).
2. **Given** the rolled-up summaries for all six configurations, **When** they are placed side by side, **Then** a reader can see at a glance which category dominates each configuration without reading raw kernel names.

### Edge Cases

- What happens when a shader/kernel does not represent a matrix multiplication (e.g., softmax, elementwise add, embedding lookup, RoPE)? It MUST still appear in the breakdown with its name, time, and percentage — its matmul shape field is simply not applicable, not omitted from the report entirely.
- What happens when a meaningful share of phase time is spent outside any shader at all (e.g., CPU↔GPU data transfer, reshape/view, CPU-fallback operators)? This MUST be captured and reported as its own category rather than silently absorbed into "other" or dropped, since prior profiling on this same hardware found such overhead to be a large share (over 50% combined) of decode time.
- What happens when enabling ETDump changes the measured performance characteristics of the run (profiling overhead)? The report MUST note the phase's total time as measured *during the profiled run* alongside the pre-existing baseline number from the prior feature, so a reader can see whether/how much profiling overhead shifted the picture, rather than conflating the two silently.
- What happens if a configuration cannot be profiled (e.g., crash, out-of-memory, missing debug symbols)? It MUST be recorded as an explicit, reasoned gap, not silently skipped, consistent with how the baseline benchmarks feature handled unmeasurable configurations.
- How is decode handled given it runs many steps? Per prior profiling on this hardware, per-step shader composition and shapes do not change with decode position (the KV-cache buffers are preallocated and read in full regardless of step index) — so profiling a short representative window of decode steps is sufficient and does not need to cover all 1024 steps from the original baseline capture.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: For each of the six baseline configurations from the MiniPC baseline-benchmarks feature, the system MUST produce a breakdown of every distinct shader/kernel invoked during prefill and, separately, during decode — not a single merged breakdown of both phases.
- **FR-002**: The primary breakdown MUST aggregate repeated invocations of the same kernel at the same shape into one entry per (kernel, shape) pair, reporting the kernel's name, total time across all its invocations in that phase, invocation count, and percentage of that phase's total time; the raw per-invocation events (individual call times) MUST be preserved in a companion data file rather than discarded.
- **FR-003**: Each aggregated entry that represents a matrix multiplication MUST report its matmul shape (M, K, N); entries that are not matrix multiplications MUST still be present in the breakdown, with the shape field marked not-applicable rather than omitted.
- **FR-004**: The breakdown MUST include non-shader time categories (e.g., CPU↔GPU copies, reshape/view, CPU-fallback operators) whenever they appear, not just GPU shader time.
- **FR-005**: For each configuration and phase, the sum of reported per-entry times MUST be reconcilable against that phase's wall-clock total as measured during the same profiled run (FR-006), and that total MUST be presented alongside the corresponding pre-existing baseline number from the prior feature for comparison.
- **FR-006**: Each profiling capture MUST record the phase's total wall-clock time as measured during the profiled run itself, since enabling profiling may add overhead relative to the un-profiled baseline measurement.
- **FR-007**: Every reported breakdown MUST be traceable to the specific model, quantization scheme, device, and dispatch path it came from, consistent with the prior feature's reporting conventions.
- **FR-008**: Decode profiling MUST cover a short representative window of decode steps rather than the full decode length used for the original baseline throughput measurement, since per-step shader/shape composition does not vary with decode position on this architecture.
- **FR-009**: Each phase's breakdown MUST also be presented as a rollup into a small set of named categories (e.g., attention projection, feed-forward, output/vocabulary projection, non-shader overhead) whose percentages account for the full phase time.
- **FR-010**: Any configuration that cannot be profiled MUST be recorded as an explicit, reasoned gap rather than omitted without explanation.

### Key Entities

- **Kernel Invocation (raw)**: one single call-site event from a profiling run — kernel name, phase (prefill or decode), matmul shape (or not-applicable), its own individual time. Preserved in a companion data file, not the primary report.
- **Aggregated Kernel Entry**: the primary breakdown row — one per unique (kernel name, shape) pair within a phase — total time summed across all its Kernel Invocations, invocation count, matmul shape (or not-applicable), and percentage of phase total.
- **Profiling Run**: one (model, scheme, phase) capture — device, dispatch path, phase wall-clock total as measured during profiling, the Aggregated Kernel Entries it produced, and the raw Kernel Invocations behind them.
- **Category Rollup**: a named grouping (e.g., attention projection, feed-forward, output projection, non-shader overhead) with an aggregated time and percentage, derived from a Profiling Run's Aggregated Kernel Entries.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For at least one baseline configuration, a complete aggregated per-phase breakdown exists showing name, total time, invocation count, matmul shape (where applicable), and percentage of phase time for every distinct kernel+shape invoked, with the raw per-invocation data available in a companion file.
- **SC-002**: For all six baseline configurations, a per-shader, per-phase breakdown exists, or an explicit recorded reason exists for any configuration that could not be profiled.
- **SC-003**: For every configuration profiled, the reported per-entry times for a phase are reconcilable against that phase's measured wall-clock total, with any material discrepancy explained rather than left unexplained.
- **SC-004**: For every configuration profiled, a category-level rollup exists that accounts for the full phase time and lets a reader identify the dominant time category without reading individual kernel names.
- **SC-005**: A reader unfamiliar with this effort can determine, for any of the six configurations, what fraction of prefill and decode time is spent in matrix multiplications versus non-shader overhead, using the produced report alone.

## Assumptions

- This feature analyzes the same six (model × scheme) configurations and the same `rocky-ryzen` MiniPC device already established in the baseline-benchmarks feature; it does not introduce new models, schemes, or hardware targets.
- Profiling targets the same `tiled_baseline` (no-WMMA) dispatch path already established; profiling a future WMMA-enabled comparison run is out of scope for this feature and would be a natural follow-on once that path exists.
- ETDump is enabled via the existing runner's built-in support (already present as an option in the same runner used for the baseline benchmarks) rather than new profiling infrastructure being built from scratch.
- Decode is profiled over a short representative window of steps rather than the full 1024-step decode length used for baseline throughput measurement, per the Edge Cases rationale; the two are expected to measure different things (throughput vs. attribution) and are not required to use identical step counts.
- A small number of repeated profiling captures per phase (fewer than the five repetitions used for baseline throughput) is sufficient to confirm the shader composition and relative time shares are representative, since this feature's goal is attribution rather than precise throughput estimation.
- The report this feature produces is a companion to, and references, the prior feature's `baseline-report.md` rather than duplicating its content.
