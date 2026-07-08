# Research: M5 EVT1 Linear + SDPA Coopmat Microbenchmark Validation

## Decision 1: Extend `test_coopmat_linear_bench.cpp`'s `kShapes` rather than rebuild-and-edit per model

**Decision**: Add 1B (`dim=2048, ffn=8192`) and 3B (`dim=3072, ffn=8192`)
K/N pairs directly into `kShapes` (currently only 8B's 4 pairs:
`{4096,4096}`, `{4096,1024}`, `{4096,14336}`, `{14336,4096}`), each paired
with a model-name string, so one run of the harness produces all 42 rows.

**Rationale**: The alternative (edit the 4 hardcoded numbers, rebuild,
run, repeat 3x) requires zero new code but costs 3 Android rebuild
cycles (~3-5 min each, already a well-worn action this session) and
produces 3 separate raw logs that then need manual model-labeling during
aggregation anyway. Extending the array is a small, additive, one-time
C++ change (no new logic, just more literal data) that produces a single,
self-labeled run -- less total effort and less error-prone than
model-tracking across 3 separate invocations.

**Alternatives considered**: An env-var-driven shape override (matching
the `COOPMAT_BENCH_CORRECTNESS_ONLY` pattern already in the file) --
rejected as more code for no real benefit here, since all 3 models' shapes
are known upfront and fixed, not something that needs to vary per
invocation the way a correctness-only toggle does.

## Decision 2: Wire `test_sdpa_coopmat_bench.cpp` into the CMake build as a new target

**Decision**: Add a new executable target for
`test_sdpa_coopmat_bench.cpp` in
`backends/vulkan/test/custom_ops/CMakeLists.txt`, mirroring the existing
`test_coopmat_linear_bench` target's pattern exactly (same link libraries,
same `ComputeGraph`-based structure per the file's own header comment
citing `specs/010`'s research.md Decision 8).

**Rationale**: `specs/010`'s own plan.md already documents that this file
is the *actual* harness that produced that feature's report (its header
literally says so), while `test_coopmat_attention_bench.cpp` -- which
this feature's spec originally cited before the Clarifications session --
tests an unrelated generic `matmul_coopmat`/`coopmat_mm_ref` path. Only
`test_sdpa_coopmat_bench.cpp` isolates the exact
`sdpa_compute_attn_weights_*`/`sdpa_compute_out_*` dispatches this
feature needs to measure.

**Alternatives considered**: Patching `test_coopmat_attention_bench.cpp`
to skip past its crashing sub-case and reach whatever SDPA-specific cases
it might have later in its sequence -- rejected: that file tests a
different shader family entirely (confirmed by reading its source during
`/speckit-clarify`), so even a clean run of it would not answer this
feature's question.

## Decision 3: Reuse `specs/015`'s already-verified M5 EVT1 session state (clock pin, driver identity)

**Decision**: Before either harness runs, re-verify (not re-derive from
scratch) the clock pin via the same GFLOP/s cross-check `specs/015`
already established this session, and re-confirm the driver identity via
`.shared-context/ACTIVE-STATUS.md` / `logcat | grep SUMD`, per constitution
Principles VII/VIII.

**Rationale**: This feature runs in the same uninterrupted M5 EVT1 adb
session as `specs/015`'s work (no device reboot in between, confirmed by
this session's own continuity) -- redoing the full pin-and-verify
procedure from zero would duplicate already-done work, but a quick
re-check costs nothing and guards against any drift since `specs/015`'s
last check.

**Alternatives considered**: Skipping re-verification entirely and trusting
`specs/015`'s last check -- rejected: constitution Principle VII requires
verification "before measuring," not "once per session," and a device
reboot or clock drift between features is exactly the failure mode Q10
(this workstream's own prior incident) already burned this workstream on
once.

## Decision 4: Report format mirrors `specs/007`/`specs/010` exactly, labeled M5 EVT1

**Decision**: `linear-coopmat-microbench-report.md` uses the identical
column structure as `specs/007`'s `wmma-improvement-report.md` (Model,
Scheme, Op, Tiled (us), Coopmat (us), Speedup %, Significance, Dispatch,
Correctness) and the same time-weighted overall-speedup summary-line
format; `sdpa-coopmat-microbench-report.md` mirrors `specs/010`'s columns
(Model, head_dim, num_heads, num_kv_heads, Tiled (us), Coopmat (us),
Speedup, Significance) and summary-line format. Both explicitly state
"M5 EVT1" (not MiniPC) in their title/header and link back to the MiniPC
report they mirror.

**Rationale**: Direct, at-a-glance comparability between the MiniPC and
M5 EVT1 numbers is the whole point of this feature (per the user's own
request); using a different format would defeat that.

**Alternatives considered**: None -- this was explicit in the user's
`/speckit-specify` input ("get report like ...").

## Decision 5: This feature does not attempt to resolve why `specs/015`'s ETDump dispatch-confirmation was unreliable (Q11/Q12)

**Decision**: This feature's dispatch confirmation relies entirely on
each microbenchmark harness's own kernel-name capture (a mechanism
already proven reliable via this session's direct wall-clock
`ET_VK_FORCE_TILED_LINEAR` A/B test and the correctness-checked GFLOP/s
results), not on ETDump's full-model-graph per-event kernel-name field.
Root-causing *why* that ETDump path misattributes names in the full
LLaMA graph (workspace `open-questions.md` Q11) is out of scope here.

**Rationale**: Already stated in the spec's Assumptions; restated here
because it is the single most important scoping decision separating this
feature from a "go debug Q11" feature -- this one produces an independent,
trustworthy data point via a different, already-reliable method instead.

**Alternatives considered**: Folding a Q11 root-cause investigation into
this feature's scope -- rejected, per the spec's own Assumptions section
and this session's earlier finding that Q11 needs Vulkan-API-level
instrumentation (validation layers, pipeline-creation `VK_CHECK`s) beyond
what a benchmark-and-report feature should carry.
