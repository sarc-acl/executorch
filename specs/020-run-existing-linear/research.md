# Research: M5 EVT1 Full Microbenchmark Suite — Stable Results Report

All items the spec might otherwise have marked `NEEDS CLARIFICATION` were
resolved during specify/clarify or by direct investigation this session
(source reads, on-device `ls`, mtime comparison) rather than left open —
recorded below for traceability.

## Decision 1: Reuse all three harnesses exactly as-is; no source changes

**Decision**: `test_coopmat_linear_bench.cpp`, `test_sdpa_coopmat_bench.cpp`,
`test_llama_baseline_bench.cpp` are used unmodified. No new shapes, no new
CMake targets, no new eligibility-gating logic.

**Rationale**: Direct source reads this session confirmed all three
already use real per-model shapes for 1B/3B/8B (linear bench's `kShapes`
covers all 3 models per weight matrix; SDPA bench's `kModels` covers all
3 models' real `head_dim`/`num_heads`/`num_kv_heads`; baseline bench's
`kModels` covers all 3 including `lm_head`, across both prefill/decode
regimes and both storage types). All three are already registered in
`backends/vulkan/test/custom_ops/CMakeLists.txt`'s `add_operator_prototype`
list — confirmed via a direct full read of the raw file, not a piped/
prefiltered grep, per `.specify/memory/gotchas.md` G8's own documented
trap (a prior session wrongly concluded a target "wasn't wired in" from a
prefiltered grep).

**Alternatives considered**: Extending shapes further (e.g. adding a
decode-phase M=1 case to the linear/SDPA coopmat benches) — rejected;
`SDPA.cpp`'s `!is_gemv` gate means coopmat structurally never dispatches
at decode (M=1) regardless of shape, so such a case would test nothing
about coopmat. This was explored and explicitly ruled out earlier in this
session (see conversation history), not silently skipped.

## Decision 2: Cross-invocation repeat count = 3

**Decision**: Each of the three binaries is invoked 3 separate times
end-to-end.

**Rationale**: Matches this workstream's established e2e repeat
convention (`specs/015`/`018`/`019` all use 3 reps), extended here to the
microbenchmark tier for the first time. 3 is the minimum that lets an
outlier be distinguished from the other 2 agreeing, without a large time
cost (each harness invocation is itself already internally repeated 3
warmup + 5 timed runs per case).

**Alternatives considered**: 2 invocations (cheaper, but a 2-way
disagreement can't identify which of the two is the outlier); 5+
invocations (more statistically robust, but no established need or
precedent in this workstream for microbenchmarks specifically, and 3x the
binary-invocation cost of the chosen option for marginal benefit given
each invocation already averages 5 internal timed runs).

## Decision 3: "Unstable" = peer-relative outlier, no fixed CoV cutoff

**Decision**: Per Clarifications (spec.md, session 2026-07-06): report
every case's 3-invocation CoV%; flag a case as unstable only when its CoV
is a clear outlier relative to its peer cases in the same run (e.g. one
case at 8% CoV when every other case in that harness/scheme is <1%),
never against an invented fixed number.

**Rationale**: Matches existing precedent exactly — `specs/015`'s
"769.35 tok/s, high CoV flagged" was called out by comparison against its
peers (every other config's CoV was far lower), not against a predefined
threshold. Inventing a number here (e.g. "flag if >5%") would be
arbitrary and either over- or under-flag relative to this hardware's
actual, already-observed noise floor.

**Alternatives considered**: Fixed 5%/10% cutoffs — rejected per the
clarification session; this workstream's pinned-clock CoVs have
historically been far below either number (0.13%-0.28% in `specs/018`),
so a fixed percentage cutoff would either be trivially never triggered or
would need re-tuning per harness/op-size with no principled way to pick
the number without more data than exists yet.

## Decision 4: Aggregation is a new, small Python script — not a reuse of `analyze_etdump_shaders.py`

**Decision**: A new `aggregate_microbench_results.py` parses each
harness's raw stdout (the `RESULT,...` / summary-table lines each harness
already prints) across its 3 invocations, computes per-case CoV, applies
Decision 3's peer-relative outlier flag, and renders the consolidated
report.

**Rationale**: `analyze_etdump_shaders.py` operates on `.etdp` ETDump
files from full e2e model runs — none of these three harnesses produce
ETDump traces; they report timing directly via `BenchmarkResult`'s own
GPU-query-pool statistics printed to stdout. There is no existing tool in
`.shared-context/scripts/` that parses this specific stdout format across
multiple invocations; `specs/007`/`010`/`016` each did their own
one-off aggregation (by hand or a throwaway script) rather than a reusable
tool — this feature's script is new but follows that same lightweight
precedent, not a novel category of tooling.

**Alternatives considered**: Hand-aggregating in the report-writing step
(no script) — rejected; 9 raw files x up to 96 cases each (baseline
bench) is too much to reconcile by eye without transcription errors, and
a script's output is directly re-runnable if a rep needs to be redone.

## Decision 5: Staging status per harness, confirmed via on-device `ls`

**Decision**: `test_coopmat_linear_bench` is already staged on M5 EVT1
as `test_coopmat_linear_bench_016` (56440208 bytes — size-identical to
the current local build, confirmed via `adb shell ls -la`); reused as-is
unless a size check at execution time shows drift. `test_sdpa_coopmat_bench`
and `test_llama_baseline_bench` are NOT currently staged on-device (built
locally, per `cmake-out-android-vk/backends/vulkan/test/custom_ops/`
mtimes newer than their sources, but never pushed) — both need a push,
neither needs a rebuild as of this session.

**Rationale**: Directly checked via `adb -s $S shell ls -la /data/local/tmp/llama_vk/`
rather than assumed from `specs/016`'s narrative (which only pushed the
linear bench, per its own plan.md's stated scope) — avoids repeating the
exact class of mistake gotcha G8 warns about, this time for on-device
staging state rather than CMake registration.

## Decision 6: Report reconciles against `specs/016`'s existing numbers, does not silently replace them

**Decision**: The consolidated report's linear/SDPA sections state
explicitly whether the new 3-invocation numbers are consistent with
`specs/016`'s single-invocation numbers (same order of magnitude, same
win/loss direction per model/scheme), and by how much they differ if at
all.

**Rationale**: FR-009 requires this explicitly. `specs/016`'s numbers are
real, already-published measurements (not synthetic) — silently
presenting new numbers with no acknowledgment of the prior measurement
would make it look like this feature found the numbers "fresh," when in
fact this feature's actual novel contribution for those two harnesses is
the *stability* evidence, not new speedup figures. `specs/016`'s own
`minipc-vs-m5evt1-comparison.md` already establishes the pattern this
feature's reconciliation follows (cite and compare against prior numbers,
don't silently overwrite them).

**Alternatives considered**: Treating this as a from-scratch measurement
with no reference to `specs/016` — rejected; it would obscure the actual
value this feature adds and risk a reader missing that the two
measurements should roughly agree.
