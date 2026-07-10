# Feature Specification: Unify M5 EVT1 Microbenchmark Structure, Shapes, and Statistics

**Feature Branch**: `021-redesign-three-m5`

**Created**: 2026-07-07

**Status**: Draft

**Input**: User description: "Redesign the three M5 EVT1 microbenchmarks (test_coopmat_linear_bench.cpp / test_sdpa_coopmat_bench.cpp / test_llama_baseline_bench.cpp) so their structure is uniform, shape coverage is real (prefill M=2048 + decode M=1 regimes), and statistics are consistent: (1) unify output to one shared RESULT,... line schema printed immediately per case (not batched at the end), so partial data survives a mid-run crash; (2) fix test_llama_baseline_bench.cpp's deterministic OOM by batching execute_test_cases() calls per model, without touching shared utils.cpp infrastructure; (3) replace linear bench's M=1024 compromise with real prefill(M=2048)/decode(M=1) regimes, honestly reporting dispatch_status=fallback_tiled when the coopmat tile-alignment gate excludes M=1; (4) split SDPA bench's combined timing into its two real sub-shapes (QK^T, attn·V) plus a total row, and add a decode(S=1) shape reported as dispatch_status=not_applicable (structurally excluded from the coopmat comparison, not a fallback); (5) update aggregate_microbench_results.py to parse the single unified format and update reconciliation methodology to compare trend/direction against prior linear-bench numbers rather than exact values, since the shape basis changed. No changes to any shader/GLSL, no changes to SDPA.cpp/QuantizedLinear.cpp dispatch logic itself, no changes to CMakeLists.txt (all three targets already registered)."

## Context (why this feature exists now)

`specs/020-run-existing-linear` ran all three existing microbenchmark
harnesses as-is and surfaced three structural problems that a follow-up
measurement pass cannot paper over:

1. **Inconsistent, not-fully-real shape coverage**: linear bench uses a
   compromise `M=1024` (neither real prefill `M=2048` nor real decode
   `M=1`); baseline bench already covers both real regimes; SDPA bench
   only covers real prefill. Source-level verification this session
   confirmed decode-shape (`M=1`/GEMV) structurally never dispatches
   coopmat in either linear (`QuantizedLinear.cpp`'s
   `can_use_q4gsw_coopmat()`: `M % tile_m != 0` when `M=1`) or SDPA
   (`SDPA.cpp`'s explicit `is_gemv` gate) — this is a real, useful fact
   to measure and report uniformly, not an edge case to work around.
2. **Three incompatible output formats**: linear prints a custom
   `SUMMARY:` table; SDPA prints clean `RESULT,...` CSV lines; baseline
   is supposed to print `RESULT,...` too but never reaches that code path
   because it is OOM-killed before `execute_test_cases()` returns —
   `specs/020`'s aggregation script needed three separate, incompatible
   parsers as a direct result.
3. **`test_llama_baseline_bench` has a deterministic, reproducible OOM**:
   `execute_test_cases()` materializes all 192 cases' tensors before
   executing any; 12 `lm_head` prefill cases (~525MB each, ~6.3GB total)
   exceed what's left of M5 EVT1's 11GB RAM. Confirmed via `dmesg`'s
   kernel oom-killer log across 3 separate invocations, always at the
   same point (14/192 cases). `specs/020` reported this as a known
   limitation rather than working around it; this feature fixes it.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - All three harnesses speak one shared result format (Priority: P1) 🎯 MVP

An engineer (or agent) analyzing microbenchmark output needs one parser
that works across all three harnesses, and needs partial results to
survive a harness crash instead of being lost entirely.

**Why this priority**: Every other story in this feature either produces
or consumes this shared format — it's the foundation the rest of the
redesign is built on, and it alone (independent of the OOM fix or shape
changes) already removes the three-separate-parsers technical debt
`specs/020` accumulated.

**Independent Test**: Can be fully tested by running each harness once
and confirming its stdout contains only `RESULT,...` lines matching one
shared schema, printed as each case completes (not batched at the end)
— verifiable by killing a harness mid-run and confirming completed
cases' `RESULT,...` lines already appear in the captured output.

**Acceptance Scenarios**:

1. **Given** any of the three harnesses runs a case to completion,
   **When** that case finishes, **Then** a `RESULT,...` line for that
   case is printed immediately, before the next case begins.
2. **Given** a harness is killed mid-run (e.g. OOM), **When** its
   captured stdout is inspected, **Then** every case that completed
   before the kill has its own complete `RESULT,...` line already
   present.
3. **Given** the three harnesses' `RESULT,...` lines, **When** parsed by
   a single shared regex, **Then** all three harnesses' output parses
   without harness-specific special-casing.

---

### User Story 2 - `test_llama_baseline_bench` completes a full run without OOM (Priority: P1)

The same engineer needs a complete, first-ever full result set from the
baseline harness on M5 EVT1 — not the 14/192-case partial data `specs/020`
was limited to.

**Why this priority**: Tied for top priority with US1: without this fix,
the baseline harness's real contribution (a genuine no-coopmat storage
comparison at real shapes, across all 3 models) remains permanently
incomplete no matter how good the output format or reporting is.

**Independent Test**: Can be fully tested by running the harness to
completion and confirming all 192 defined cases produced a `RESULT,...`
line, with no OOM kill (verified via `dmesg` showing no new oom-kill
entry for this process, and via a clean process exit code).

**Acceptance Scenarios**:

1. **Given** the harness is invoked, **When** it processes cases one at a
   time (per User Story 1's incremental-printing mechanism) rather than
   materializing all 192 at once, **Then** peak memory stays well under
   the device's available RAM (previously ~6.3GB from 12 simultaneously-
   materialized `lm_head` cases; now bounded by a single case's own
   tensors, ~525MB worst case).
2. **Given** the harness completes, **When** its output is inspected,
   **Then** all 192 cases each have a `RESULT,...` line — none missing
   due to a crash.
3. **Given** a hypothetical future failure partway through one model's
   cases, **When** that happens, **Then** every already-completed case's
   `RESULT,...` line (from this model and any prior model) is still
   present (per US1's incremental printing),
   not lost as a side effect of one model's failure.

---

### User Story 3 - Linear bench measures real prefill and decode shapes (Priority: P2)

The same engineer needs linear bench's shapes to reflect what a real e2e
run actually dispatches (prefill `M=2048`, decode `M=1`), not an
arbitrary `M=1024` compromise that doesn't correspond to any real
workload point.

**Why this priority**: Depends on US1's output format existing first
(so the new cases report through the same schema); improves shape realism
but doesn't block US1/US2's value from landing independently.

**Independent Test**: Can be fully tested by confirming linear bench's
case set includes both `regime=prefill` (`M=2048`) and `regime=decode`
(`M=1`) entries for every existing `(model, scheme, K, N)` combination,
with decode entries honestly reporting `dispatch_status=not_applicable`.

**Acceptance Scenarios**:

1. **Given** linear bench runs, **When** its prefill cases are reported,
   **Then** they use the real prefill length `M=2048`, not `M=1024`.
2. **Given** linear bench runs its decode (`M=1`) cases, **When** the
   dispatched kernel name is inspected, **Then** every decode case
   reports `dispatch_status=not_applicable` (never `confirmed` or
   `fallback_tiled`) — `QuantizedLinear.cpp`'s `is_gemv_case` check
   short-circuits decode to a dedicated `_coop` kernel before the coopmat
   eligibility check ever runs, so decode never reaches either the
   `_coopmat` or the `_tiled` code path linear bench's own tiled-vs-
   coopmat comparison is actually about — structurally identical to
   SDPA's `is_gemv` exclusion (User Story 4).
3. **Given** a reader compares this feature's linear numbers against
   `specs/016`/`specs/020`'s prior `M=1024`-based numbers, **When** they
   read the report, **Then** it explicitly states the shape basis changed
   and that exact-value comparison is not meaningful — only the
   direction/magnitude of the tiled-vs-coopmat gap is comparable.

---

### User Story 4 - SDPA bench reports its real sub-shapes and a decode shape (Priority: P2)

The same engineer needs SDPA bench's granularity to match linear bench's
(one row per real sub-operation, not one blended total), and needs the
decode-shape case to exist for regime-completeness even though it can
never exercise coopmat.

**Why this priority**: Same tier as US3 — a real-shape/structural
improvement, depends on US1's shared format, does not block US1/US2.

**Independent Test**: Can be fully tested by confirming SDPA bench's
per-model output includes 3 rows (`variant=qk`, `variant=av`,
`variant=total`) for the prefill shape, plus one decode-shape
(`S=1`, real KV-cache length) row per model reporting
`dispatch_status=not_applicable`.

**Acceptance Scenarios**:

1. **Given** SDPA bench runs a model's prefill case, **When** results are
   reported, **Then** `sdpa_compute_attn_weights_*` and
   `sdpa_compute_out_*` timings are each reported on their own row, in
   addition to (not instead of) the existing combined total.
2. **Given** SDPA bench runs a model's decode case, **When** results are
   reported, **Then** `dispatch_status=not_applicable` is used — the same
   status linear bench's own decode cases use (User Story 3), since both
   are structurally excluded from the coopmat comparison by an explicit
   `is_gemv`-style gate (`SDPA.cpp`'s `is_gemv`,
   `QuantizedLinear.cpp`'s `is_gemv_case`), not because coopmat was
   eligible and didn't fire.

---

### User Story 5 - One aggregator, one report, honest reconciliation (Priority: P3)

The same engineer needs the existing aggregation tooling updated to match
the new unified format, and needs the report's comparison against prior
numbers to be honest about what changed (shape basis) versus what didn't
(the underlying tiled-vs-coopmat conclusion).

**Why this priority**: Depends on US1-US4 all existing first — it's the
synthesis step, not new measurement.

**Independent Test**: Can be fully tested by running the updated
aggregator against fresh output from all three redesigned harnesses and
confirming it uses one shared parser (no harness-specific regex branches
remain) and its reconciliation section explicitly states the linear shape
basis changed.

**Acceptance Scenarios**:

1. **Given** the three harnesses' unified `RESULT,...` output, **When**
   the aggregator parses it, **Then** one shared regex/parser handles all
   three — no `SUMMARY:`-table or raw-dispatch-line special-casing
   remains.
2. **Given** the aggregator compares this feature's linear numbers
   against `specs/016`/`specs/020`'s prior `M=1024` numbers, **When** it
   renders the reconciliation section, **Then** it explicitly states the
   shape basis changed and compares only the tiled-vs-coopmat
   direction/magnitude trend, not exact percentage deltas.

### Edge Cases

- What happens when a decode-shape case (linear or SDPA) is evaluated for
  "coopmat dispatch confirmed"? It must never be — `not_applicable` is
  the only valid outcome for both (linear's `QuantizedLinear.cpp` and
  SDPA's `SDPA.cpp` both short-circuit decode to a dedicated `_coop`
  kernel via an explicit `is_gemv`-style check, before the coopmat
  eligibility check ever runs); a `confirmed` decode case would indicate
  a bug in the harness's own gate logic, not a benchmarking win.
- What happens if a future single model's batch (User Story 2) still
  runs out of memory? The other already-completed models' `RESULT,...`
  lines remain valid and present (per US1) — this feature does not need
  to guarantee every batch succeeds, only that a batch's failure doesn't
  destroy previously-collected data the way the current all-at-once
  design does.
- What happens to the old `M=1024` linear numbers and the old combined
  SDPA numbers? They are not deleted from prior specs' results
  directories (historical record), but this feature's own new report
  does not present them as directly comparable line items.
- What happens if the aggregation script encounters a `RESULT,...` line
  from an old, pre-redesign harness binary (mismatched format)? Treated
  as a parse failure to surface, not silently skipped — this feature
  does not need graceful backward-compatibility with the old formats.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: All three harnesses (linear, SDPA, baseline) MUST print a
  `RESULT,...` line using one shared schema (harness, model, scheme,
  regime, variant, shape dimensions, avg/stddev timing, GFLOP/s where
  applicable, dispatch_status, correctness_status) immediately after each
  case completes — never batched until the end of the run.
- **FR-002**: `dispatch_status` MUST distinguish three states:
  `confirmed` (coopmat eligible and dispatched — kernel name contains
  `_coopmat`), `fallback_tiled` (a *prefill* case whose op family is
  coopmat-eligible but this specific shape didn't satisfy the tile-
  alignment gate, so it dispatched the ordinary `_tiled` kernel), and
  `not_applicable` (this case's regime is structurally excluded from ever
  reaching either the `_coopmat` or `_tiled` code path by an explicit
  `is_gemv`-style short-circuit — both linear's decode cases and SDPA's
  decode cases dispatch a distinct `_coop` kernel instead, per direct
  source verification of `QuantizedLinear.cpp`'s `is_gemv_case` and
  `SDPA.cpp`'s `is_gemv`) — these three MUST NOT be conflated.
- **FR-003**: `test_llama_baseline_bench` MUST process its 192 cases in
  per-model batches (calling `execute_test_cases()` once per model, not
  once for all cases), without modifying shared `utils.cpp` infrastructure
  used by other benchmark binaries.
- **FR-004**: `test_llama_baseline_bench` MUST complete a full run (all
  192 cases reported) on M5 EVT1 without triggering an OOM kill.
- **FR-005**: Linear bench MUST report both a real prefill (`M=2048`) and
  real decode (`M=1`) regime for every existing `(model, scheme, K, N)`
  shape, replacing the current `M=1024` compromise.
- **FR-006**: Linear bench's decode-regime cases MUST report
  `dispatch_status=not_applicable` for every case (never `confirmed` or
  `fallback_tiled`), consistent with `QuantizedLinear.cpp`'s
  `is_gemv_case` short-circuit, which dispatches a dedicated `_coop`
  kernel for `M=1` before the coopmat eligibility check (including its
  `M % tile_m != 0` gate) ever runs — decode never reaches the `_tiled`
  code path at all, so `fallback_tiled` would misrepresent what actually
  happened.
- **FR-007**: SDPA bench MUST report `sdpa_compute_attn_weights_*` and
  `sdpa_compute_out_*` timings as separate rows (`variant=qk`,
  `variant=av`), in addition to the existing combined total
  (`variant=total`), for every model's prefill case.
- **FR-008**: SDPA bench MUST add a decode-shape (`S=1`, real KV-cache
  length) case per model, reporting `dispatch_status=not_applicable`.
- **FR-009**: `aggregate_microbench_results.py` MUST be updated to parse
  the single unified `RESULT,...` schema from all three harnesses with
  one shared parser, removing the harness-specific `SUMMARY:`-table and
  raw-dispatch-line parsing this feature's shape/format changes make
  obsolete.
- **FR-010**: The updated report's reconciliation section MUST explicitly
  state that linear bench's shape basis changed (`M=1024` → real
  `M=2048`/`M=1`) and MUST NOT present exact-percentage deltas against
  `specs/016`/`specs/020`'s prior `M=1024`-based numbers as if they were
  measuring the same thing — comparison is limited to whether the
  tiled-vs-coopmat direction and rough magnitude are consistent.
- **FR-011**: No shader/GLSL source, no `SDPA.cpp`/`QuantizedLinear.cpp`
  dispatch-gating logic, and no `CMakeLists.txt` registration may be
  modified by this feature — all changes are confined to the three
  harness `.cpp` files and the aggregation script.

### Key Entities

- **Unified result line**: one `RESULT,...` record per (harness, model,
  scheme, regime, variant) case, printed at the moment that case
  completes — the single data format all downstream tooling consumes.
- **Regime**: `prefill` (real `M`/`S`=2048) or `decode` (real `M`/`S`=1).
  Uniform across all three harnesses after this feature (baseline already
  has it; linear and SDPA gain it).
- **Variant**: the specific measured sub-operation within a case —
  `tiled`/`coopmat` for linear/baseline; `qk`/`av`/`total` for SDPA.
- **Dispatch status**: `confirmed` / `fallback_tiled` / `not_applicable`
  — a three-way distinction this feature introduces in place of the
  current two-way (fired/not-fired) flag, to separate "coopmat-eligible
  op, this shape didn't qualify" from "this regime never uses coopmat by
  design."

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All three harnesses' output can be parsed by one shared
  parser with zero harness-specific format branches.
- **SC-002**: `test_llama_baseline_bench` completes a full run (all 192
  cases reported, confirmed via `dmesg` showing no new OOM event for the
  process) on M5 EVT1, for the first time in this workstream's history.
- **SC-003**: 100% of linear-bench decode-regime cases and 100% of
  SDPA-bench decode-shape cases report a dispatch status other than
  `confirmed` — verifying the structural coopmat exclusion is uniformly
  and honestly recorded, not just assumed.
- **SC-004**: A reader of the new consolidated report can correctly state
  which regimes/models/shapes coopmat helps for, without being misled
  into comparing this feature's linear numbers against prior `M=1024`
  numbers as if they were the same measurement.

## Assumptions

- This feature does not change any shader eligibility gate, dispatch
  logic, or GLSL source — every `dispatch_status` outcome it reports is
  an observation of existing, unmodified gating behavior, not a new
  capability.
- "Real KV-cache length" for SDPA's decode case uses this workstream's
  standard `ctx3072` context length, consistent with the PTEs used
  elsewhere in this workstream's e2e measurements.
- The OOM fix for `test_llama_baseline_bench` (User Story 2) is scoped to
  that harness's own `main()`/`generate_cases()` structure and relies on
  User Story 1's per-case `execute_test_cases()` calling pattern — peak
  memory is bounded by a single case's own tensors (~525MB worst case,
  `lm_head` prefill), not by any batch size, so per-model grouping is
  organizational only, not the memory-safety mechanism (research.md
  Decision 3). If a single future case's own tensors ever approached
  the device's memory limit on their own, that would need a different
  fix outside this feature's scope (e.g. splitting one case's tensor
  across multiple dispatches).
- Historical `specs/016`/`specs/020` results files are left as-is
  (not edited/retracted) — they remain valid records of what was
  measured under the old `M=1024`/combined-SDPA-timing methodology.
