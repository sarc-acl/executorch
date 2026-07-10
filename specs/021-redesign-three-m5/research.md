# Research: Unify M5 EVT1 Microbenchmark Structure, Shapes, and Statistics

## Decision 1: Unified `RESULT,...` line schema

**Decision**: All three harnesses print exactly this comma-separated
schema, one line per completed case, immediately after that case's
timing/correctness check finishes:

```
RESULT,<harness>,<model>,<scheme>,<regime>,<variant>,<K>,<N>,<avg_us>,<stddev_us>,<gflops>,<dispatch_status>,<correctness_status>
```

- `harness`: `linear` | `sdpa` | `baseline`
- `regime`: `prefill` | `decode`
- `variant`: `tiled` | `coopmat` (linear/baseline); `qk` | `av` | `total`
  (SDPA)
- `dispatch_status`: `confirmed` | `fallback_tiled` | `not_applicable`
  (FR-002's three-way split)
- `correctness_status`: `PASSED` | `FAILED` | `SKIPPED` (harnesses already
  produce one of these three per case; unchanged)
- `gflops`: `-1` sentinel where not meaningful (SDPA's `qk`/`av`/`total`
  rows report raw `avg_us` as the primary metric, matching the existing
  `test_sdpa_coopmat_bench.cpp` convention of reporting time, not
  throughput, for attention ops)

**Rationale**: A single fixed-arity CSV line is trivially parsed by one
regex regardless of which harness produced it — this directly implements
spec FR-001/FR-009 and eliminates `aggregate_microbench_results.py`'s
current three separate parsers (`LINEAR_SUMMARY_RE` table-parsing,
`SDPA_RESULT_RE`, `BASELINE_RAW_RE`+`BASELINE_CASE_NAME_RE`).

**Alternatives considered**: JSON-per-line — rejected, more verbose to
hand-write in C++ `iostream` code with no existing JSON library already
linked into these prototype binaries, and CSV is already SDPA bench's
proven, working convention (`RESULT,llama-3.1-8b,128,32,8,2048,...`) —
extending it is less risky than replacing it.

**Note**: achieving true per-case (not per-batch) immediacy without
modifying `execute_test_cases()` itself requires a specific calling
pattern — see Decision 8.

## Decision 2: `dispatch_status` decision rule (three-way, FR-002)

**Decision**: Each harness computes `dispatch_status` from the same
underlying fact it already has (whether the dispatched kernel name
contains a coopmat-family substring), combined with the case's own
`regime`:

- `regime=prefill` and kernel name indicates coopmat → `confirmed`
- `regime=prefill` and kernel name does NOT indicate coopmat →
  `fallback_tiled` (an anomaly worth flagging: coopmat was structurally
  eligible for this regime but didn't fire for this specific shape)
- `regime=decode` → always `not_applicable`, regardless of which kernel
  actually dispatched. **Corrected during `/speckit-tasks` planning**:
  the exclusion mechanism is NOT `can_use_q4gsw_coopmat()`'s
  `M % tile_m != 0` tile-alignment check (that function is never even
  called for decode) — it's an earlier, explicit `is_gemv_case`
  short-circuit in `pick_linear_qw_shader`/`pick_linear_dqa_qw_shader`
  (`QuantizedLinear.cpp` lines ~250/284 and ~310/337:
  `if (weight_is_4bit && is_gemv_case) { kernel_name += "_coop"; }`),
  which dispatches a dedicated `_coop` kernel for `M=1` *before* the
  coopmat eligibility check ever runs — structurally identical to
  `SDPA.cpp`'s `is_gemv` gate. Verified by direct source read during
  task planning, not assumed from the tile-alignment logic alone
  (Principle VI) — the original Decision 2 draft cited the wrong gate.
- baseline bench: always `not_applicable` for every case, at every
  regime — it forces `ET_VK_FORCE_TILED_LINEAR=1` and has no coopmat
  toggle at all; its own comparison axis is storage type (texture3d vs
  buffer), not coopmat-vs-tiled, so `dispatch_status` is structurally
  inapplicable to its entire case set, not just its decode cases.

**Rationale**: This distinguishes "coopmat should have fired here and
didn't" (a real anomaly, `fallback_tiled`) from "coopmat can never fire
here by design" (`not_applicable`) — conflating them (as the current
two-way `fired`/`!fired` flag does) would make a structurally-expected
decode result look identical to a genuine regression.

**Alternatives considered**: Deriving `not_applicable` purely from
`M==1`/`S==1` without also checking the kernel name — rejected; the
kernel-name check is still worth keeping as a defense against a future
code change silently making decode coopmat-eligible without this
harness's own logic being updated to match (Principle VI: verify with
tools, don't just assume from the shape value).

## Decision 3: Baseline bench's OOM is actually fixed by Decision 8, not a separate batching mechanism

**Decision**: `test_llama_baseline_bench.cpp`'s `generate_cases()`
becomes `generate_cases_for_model(const ModelShapes& model)`, producing
64 cases (2 regimes × 2 storage × 2 schemes × 8 ops) instead of 192, and
`main()` loops over `kModels` (already a 3-element `std::vector`) as the
outer loop, with Decision 8's per-case `execute_test_cases()` calls as
the inner loop. **This supersedes an earlier draft of this decision**
that proposed calling `execute_test_cases()` once per model with all 64
of that model's cases at once (~2.1GB peak per call) — once Decision 8
establishes that every harness calls `execute_test_cases()` once per
individual case (to get per-case immediate printing at all), the actual
peak memory is bounded by a *single* case's tensors, not a model's worth
of them. The per-model grouping that remains is now purely organizational
(keeps `RESULT,...` output grouped by model, matches the other two
harnesses' per-model structure) — it is no longer the mechanism that
prevents the OOM.

**Rationale**: Confirmed via direct read of `utils.cpp:1704-1705` this
session that `execute_test_cases()` fully materializes its entire input
case vector before executing any case — combined with Decision 8's
one-case-per-call pattern, the worst case is now a single `lm_head`
prefill case (~525MB), not 4 of them at once (~2.1GB). This is a
strictly better bound than the original per-model-batch proposal, arrived
at only after Decision 8 was worked out in detail — the two decisions
are not independent, and this entry is left in place (rather than
deleted) specifically to record that correction rather than silently
presenting the smaller number as if it were the original plan.

**Alternatives considered**: Batching by `(model, regime)` (6 batches of
32 cases passed to `execute_test_cases()` together) for extra safety
margin over the original per-model-batch idea — moot once Decision 8
already achieves per-case granularity; no coarser batching scheme is
needed. If a future single case's own tensors (e.g. an even larger
future model's `lm_head`) ever approach the device's memory limit on
their own, that would need a different fix (e.g. splitting one case's
tensor across multiple dispatches) — out of this feature's scope, noted
in spec.md Assumptions.

## Decision 4: SDPA sub-shader split

**Decision**: `run_case()`'s single `sdpa_time_us` accumulator becomes
two accumulators (`qk_time_us`, `av_time_us`), summed from
`sdpa_compute_attn_weights_*` and `sdpa_compute_out_*` kernel timings
respectively (currently both feed the same accumulator — this is a
~4-line change to the existing per-shader-result loop). `RunResult`
gains `qk_mean_us`/`qk_stdev_us`/`av_mean_us`/`av_stdev_us` alongside the
existing combined `mean_us`/`stdev_us` (kept as `variant=total`).

**Rationale**: The GPU query-pool data needed for this split already
exists in the current loop (`shader_results` already carries each
dispatch's own `kernel_name` and duration) — this is a reporting-
granularity change, not a new measurement capability.

**Alternatives considered**: Only reporting the split, dropping the
combined total — rejected; the combined total is what answers "is the
whole attention op faster," a real, still-relevant question in its own
right (and what `specs/010`/`016`'s prior reports already used) —
additive, not a replacement (spec FR-007 says "in addition to").

## Decision 5: SDPA decode-case construction

**Decision**: New decode case per model: `batch_size=1`, query
`seq_len=1`, KV cache allocated at this workstream's standard context
length `context_len=3072` (matching the `ctx3072` PTEs used elsewhere in
this workstream), `input_pos=3071` (the last possible decode step — the
single most expensive real decode-time SDPA invocation, since it attends
over the fullest cache). Cache contents are random fill (not a real
`update_cache` walk from position 0) since only timing is measured here,
not output correctness, and GPU dispatch cost for these shaders depends
on tensor shapes/access pattern, not data values.

**Rationale**: `input_pos=3071` (not, say, an early/average decode
position) gives the single real data point that matters most for
capacity planning — the worst-case per-decode-step SDPA cost, matching
this workstream's existing "always measure/report the demanding case,
don't average it away" convention (constitution Principle VII's
per-rep-not-blended-mean rule, applied here to the case-selection choice
rather than to repeat-run reporting).

**Alternatives considered**: Sweeping multiple decode positions (e.g.
`input_pos` = 0, 1536, 3071) — rejected as out of this feature's scope;
FR-008 only requires "a decode shape," not a full decode-position sweep,
and `SDPA.cpp`'s `is_gemv` gate means the coopmat-vs-tiled question is
moot at every decode position anyway (all report `not_applicable`) — a
sweep would add cost without adding a new coopmat-relevant fact.

## Decision 6: Linear bench regime axis

**Decision**: Replace `static constexpr int64_t kM = 1024;` with
`static const std::vector<std::pair<std::string, int64_t>> kRegimes =
{{"prefill", 2048}, {"decode", 1}};`, and add a `regime` field to
`LinearConfig`. `generate_cases()`'s existing per-`(op, shape)` loop gains
an outer loop over `kRegimes`, reusing the existing `kShapes` table and
`make_case()` unchanged (M is already a `LinearConfig` field, just no
longer hardcoded to one value).

**Rationale**: `kShapes` (per-model real `K`/`N`) is already correct and
untouched by this feature — only the M dimension was ever an
approximation (spec.md Context). This is the minimal change that makes
every existing shape entry also get measured at both real regimes.

**Alternatives considered**: Keeping `M=1024` as a third regime alongside
the two real ones (for continuity with `specs/016`/`020`'s prior numbers)
— rejected per spec.md's explicit FR-010/Assumptions: the shape basis
change is accepted and documented, not hedged by keeping the old
non-real datapoint around indefinitely.

## Decision 7: Aggregation script rewrite scope

**Decision**: `aggregate_microbench_results.py`'s three parsing functions
(`parse_linear`, `parse_sdpa`, `parse_baseline`) and their three regexes
(`LINEAR_SUMMARY_RE`, `SDPA_RESULT_RE`, `BASELINE_RAW_RE`+
`BASELINE_CASE_NAME_RE`) are replaced by one `RESULT_LINE_RE` and one
`parse_result_line()` function, used identically for all three harnesses'
raw logs. The existing `aggregate()` (peer-relative CoV/outlier logic)
and `render_report()` structure are reused with field-name updates only
(e.g. `regime`/`variant` become first-class grouping keys alongside
`model`/`case_key`).

**Rationale**: This is a direct, mechanical simplification once Decision
1's unified format exists — the stability-statistics logic
(`specs/020`'s already-validated peer-relative-outlier rule) does not
need to change at all, only what it's fed.

**Alternatives considered**: Keeping the three old parsers as a fallback
for old-format logs — rejected per spec.md's Edge Cases: old-format input
is a parse failure to surface, not something this feature needs backward
compatibility for (this feature is redesigning the harnesses, not
maintaining two output formats indefinitely).

## Decision 8: How per-case immediacy is achieved without modifying `execute_test_cases()`

**Decision**: `execute_test_cases()` (`utils.cpp:1695`) is a blocking call
that only returns after every case in the vector it's given has run — it
cannot itself print incrementally without being modified, which FR-011
disallows. Each harness's `main()` is restructured to call
`execute_test_cases()` **once per case** (passing a single-element case
vector each time) inside a loop, printing that case's `RESULT,...` line
immediately after each call returns, rather than the current pattern of
one call with the full case vector followed by one final printing pass.

- For linear/baseline (whose `generate_cases()` returns a
  `std::vector<TestCase>`), the loop is: `for (auto& case : all_cases) {
  auto result = execute_test_cases([&]{ return
  std::vector<TestCase>{case}; }, ...); print_result_line(result[0]); }`.
- For baseline specifically, this loop runs *inside* each per-model batch
  (Decision 3) — i.e. nested: outer loop over 3 models, inner loop over
  that model's 64 cases, one `execute_test_cases()` call per case. This
  is what actually gives US1's crash-survives-with-partial-data property
  its real teeth: even within one model's batch, a crash on case N still
  leaves cases 1..N-1's `RESULT,...` lines already printed.
- SDPA bench already calls its own case-construction/execution logic
  directly per model (not through `execute_test_cases()` at all — it
  builds the `ComputeGraph` and calls `run_case()` itself, per
  `test_sdpa_coopmat_bench.cpp`'s existing structure) — no restructuring
  needed there beyond printing after each model's `run_case()` pair
  returns, since SDPA already only has one case per model per regime, not
  a batch to sub-divide.

**Trade-off accepted**: `execute_test_cases()`'s existing `ReferenceKey`
grouping (which shares one reference computation across cases with
identical shapes, e.g. `wq`/`wo`) no longer applies once every call
passes exactly one case — every case now computes (or skips) its own
reference independently. This is judged acceptable: the large perf-shape
cases that dominate this workstream's case count already throw
`invalid_argument` and get `SKIPPED` by the reference function itself
(`M > 256 || N > 256 || K > 4096`, per each harness's existing
`bench_reference`) before any real computation happens, so the grouping
optimization was never saving meaningful work for those cases; only the
small correctness-only shapes actually compute a real reference, and
those are cheap enough (`M,N,K <= 256` by construction) that recomputing
per case instead of once per group is not measurably slower.

**Alternatives considered**: Modifying `execute_test_cases()` to accept
an optional per-case callback invoked as each case finishes, instead of
calling it once per case from the outside — rejected because it changes
shared `utils.cpp` code used by ~15 other benchmark binaries, which
FR-011 explicitly disallows; the outside-loop approach achieves the same
observable behavior (one `RESULT,...` line per case, printed immediately)
using only the existing public `execute_test_cases()` signature.

## Decision 9: `lm_head`'s QueryPool race, found during on-device verification, handled with a case-local `try`/`catch`

**Decision**: Decision 8's per-case execution eliminated the OOM (confirmed
on-device, T010) but was the first thing in this workstream's history to
ever actually dispatch `lm_head` at its full real shape (every prior run
OOM'd before reaching it, since `lm_head` is generated last within a
model's 64-case sweep). Doing so exposed a real, pre-existing race in
shared `backends/vulkan/runtime/vk_api/QueryPool.cpp`'s `extract_results()`:
its `vkGetQueryPoolResults` call omits `VK_QUERY_RESULT_WAIT_BIT` (a
non-blocking query), and on `lm_head`'s ~270us-per-dispatch shape (the
single largest dispatch in this entire suite) it occasionally returns
`VK_NOT_READY`, which the existing `VK_CHECK` macro turns into an
uncaught `vkcompute::vkapi::Error` (inherits `std::exception`) that
crashes the whole process (`SIGABRT`) before this feature's own
per-case-crash-tolerance intent could apply.

Per explicit user decision (this is a pre-existing shared-runtime defect,
not something this feature caused or should fix — FR-011 excludes
`utils.cpp`/lower runtime changes), `test_llama_baseline_bench.cpp`'s
`main()` wraps each case's `execute_test_cases()` call in
`try { ... } catch (const std::exception& e) { ... }`. On catch: looks up
the case's `LinearConfig` via `g_case_configs` (populated by `make_case()`
before the exception, so identity survives even though the measurement
doesn't), prints a `RESULT,...` row with `correctness_status=CRASHED` and
`-1` timing sentinels, logs the exception to stderr, and continues the
loop — extending Decision 8's "partial data survives a failure" principle
from process-level crashes (OOM) to case-level exceptions.

**Verified on-device**: full 192-case run completed with exit code 0; 3
cases (`llama-3.1-8b`/`4w`/prefill/texture3d/`lm_head`,
`llama-3.1-8b`/`8da4w`/prefill/texture3d/`lm_head`,
`llama-3.2-3b`/`8da4w`/prefill/texture3d/`lm_head`) hit the race and were
recorded as `CRASHED`; the other 189 cases (including the 1B model's
`lm_head`, whose smaller `K` apparently keeps its dispatch under whatever
threshold makes the race likely, and the `buffer`-storage `lm_head`
variants) measured normally. This asymmetry (some `lm_head` cases crash,
others don't, non-deterministically related to shape/storage) is
consistent with a genuine timing race, not a logic bug in this feature's
own code.

**Alternatives considered**: Fixing the race at its source (adding
`VK_QUERY_RESULT_WAIT_BIT` to `QueryPool.cpp`, or auditing `Fence.cpp`'s
wait loop) — this would be the "real" fix, but touches shared Vulkan
runtime code used by every other benchmark binary and the production
inference path, requiring a correctness/performance validation pass far
beyond this feature's scope; explicitly deferred to a separate,
dedicated investigation, not silently bundled into this feature's diff.
