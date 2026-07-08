# Research: ETDump E2E Shader Profiling Breakdown

All unknowns from the plan's Technical Context are resolved below via direct
source inspection (`main.cpp`, `export_llama_lib.py`, `devtools/inspector`,
`backends/vulkan/runtime`) — see the file:line citations in each decision.
There are no remaining `NEEDS CLARIFICATION` markers.

## Decision 1: Enabling ETDump for the existing `llama_main` runner

**Decision**: Rebuild with `-DEXECUTORCH_BUILD_DEVTOOLS=ON
-DEXECUTORCH_ENABLE_EVENT_TRACER=ON` added to the constitution's Reference
Build Recipe, into a **separate** build directory
(`cmake-out-vk-profiling/`), then run `llama_main` with `--etdump_path
<file>`. No change to `main.cpp` or `examples/models/llama/CMakeLists.txt`
is expected to be *necessary*, but that file does not explicitly
`target_link_libraries(llama_main PRIVATE etdump flatccrt)` the way
`executor_runner`'s own CMakeLists does (`CMakeLists.txt:1408-1409`) — if
linking `llama_main` fails once the flags are set, the fallback is adding
that explicit link line.

**Rationale**: `main.cpp` already contains the full `ETDumpGen` code path
gated behind the `ET_EVENT_TRACER_ENABLED` preprocessor symbol
(`main.cpp:80-83` for `--etdump_path`, `main.cpp:168-244` for construction and
writing the dump) — this is existing, shipping code, not something to build.
`ET_EVENT_TRACER_ENABLED` is only defined when the CMake option
`EXECUTORCH_ENABLE_EVENT_TRACER` is on (`CMakeLists.txt:220-221`), and that
option requires `EXECUTORCH_BUILD_DEVTOOLS` to actually build the `etdump`
target it links against (`CMakeLists.txt:548-550`). A separate build
directory keeps `001`'s validated build/artifacts untouched.

**Alternatives considered**: reusing `001`'s existing `cmake-out-vk` build
directory in place — rejected; flipping on devtools/event-tracer for an
existing configured build risks an inconsistent partial reconfigure, and a
fresh directory is cheap and keeps the two features' builds cleanly separate.

**Correction found during implementation**: the flagged fallback *was*
needed, exactly as predicted — linking failed with `undefined reference to
executorch::etdump::ETDumpGen::...`. Root cause: `examples/models/llama/CMakeLists.txt`
never defines `ET_EVENT_TRACER_ENABLED` or links `etdump`/`flatccrt` itself
(unlike `executor_runner`'s own CMakeLists); it silently compiled `main.cpp`'s
`#ifdef ET_EVENT_TRACER_ENABLED` block right out. Fixed by adding an
`if(EXECUTORCH_ENABLE_EVENT_TRACER)` block to that CMakeLists (mirroring the
root project's own `add_definitions(-DET_EVENT_TRACER_ENABLED)` plus
`executor_runner`'s `list(APPEND ... etdump flatccrt)` pattern), then passing
`-DEXECUTORCH_ENABLE_EVENT_TRACER=ON` to the `examples/models/llama` cmake
invocation too (it doesn't inherit this from the root build's own configure).
Verified with a smoke-test run that `--etdump_path` produces a real,
non-empty file only after this fix.

## Decision 2: Shape attribution — skip ETRecord, use the Vulkan delegate's embedded shape JSON

**Decision**: Do **not** use `export_llama`'s `--generate_etrecord` flag or
the ETRecord+Inspector `op_graph_dict` correlation workflow. Instead, rely on
the fact that the Vulkan delegate already embeds each dispatch's tensor
shapes directly into the ETDump event name as a JSON blob, whenever the
event tracer is enabled — independent of ETRecord entirely.

**Rationale**: Direct inspection shows two facts that together make ETRecord
unnecessary here:
1. `devtools/inspector/_inspector.py`'s `Event`/`to_dataframe()` (fields at
   lines 339-358) has no shape/args fields. Shapes only exist in the
   *separate* `op_graph_dict` structure built from an ETRecord
   (`_inspector.py:1104`, `devtools/debug_format/et_schema.py:242-306`), and
   `_associate_with_op_graph_nodes` (`_inspector.py:667-701`) explicitly does
   **not** copy `output_shapes` onto `Event` objects — getting shapes this
   way requires writing the same amount of custom cross-referencing code
   that the next option makes unnecessary.
2. The Vulkan delegate independently captures shapes today, with no
   ETRecord involved: `backends/vulkan/runtime/graph/Logging.cpp:34-97`
   (`make_arg_json`/`make_operator_json`) emits each tensor arg's `"sizes"`
   (plus dtype/storage); `backends/vulkan/runtime/graph/ops/DispatchNode.cpp:63-80`
   embeds this as the dispatch's `event_name` JSON; `VulkanBackend.cpp:799-812`
   forwards that name into `event_tracer_log_profiling_delegate(...)`. So a
   plain `--etdump_path` capture (no ETRecord) already carries per-dispatch
   shape data in `Event.name` as a JSON string.

This also means `001`'s six `.pte` files can be reused completely unmodified
— `--generate_etrecord` doesn't change the exported `.pte` bytes either
(`export_llama_lib.py:1264-1287` shows both branches call the identical
`to_backend`/`to_executorch`), but since we're not using it at all, there's
no export-side dependency of any kind for this feature.

**Alternatives considered**: `--generate_etrecord` + `Inspector(etrecord=...,
etdump_path=...)` — rejected; per point 1 above, the built-in
Event↔op_graph_dict association doesn't carry shapes anyway, so this path
would need the same custom JSON-parsing effort as Decision 2's approach
*plus* an unnecessary re-export step, for no benefit.

**Two corrections found during implementation** (both against real captured
`.etdump` data, not assumed):

1. *M is not safe to read from the embedded tensor `sizes`.* For this
   dynamic-shape export, a linear op's input/output tensor JSON reports the
   Vulkan tensor's **static allocation bound**, not the actual active M for
   that specific dispatch — a decode-step (M=1) event showed the identical
   `sizes` as its prefill counterpart. K (input's last dim) and N (output's
   last dim) are unaffected (those are fixed feature/hidden dimensions, never
   dynamic) and remain safe to read directly. M must instead be inferred from
   *which kernel* was dispatched: `QuantizedLinear.cpp`'s `is_gemv_case`
   branch names the kernel `..._gemv_coop_...`/`..._coop_...` for M=1 and
   `..._gemm_...`/`..._tiled_...` for the real M (this feature's fixed 2048
   prefill) — so the kernel-name markers already used for Decision 5's
   block classification (see below) are also the correct source for M,
   confirmed empirically to reproduce every op's exact `(K, N)` from `001`'s
   `results/shapes.json` and the model's real per-layer invocation counts.
2. *A decode-window `.etdump` is not decode-only.* One decode-window capture
   contains the one prefill call needed to seed the KV-cache before decoding,
   in its own `EventBlock`, alongside the N decode-step blocks. Naively
   summing every block when building the "decode" phase gave 1575%
   over-attribution. Fixed by classifying each `EventBlock` before
   aggregating: a block containing a tiled/gemm-marked kernel is prefill
   (even if it *also* contains one GEMV-marked kernel — empirically, the
   lm_head/vocab projection is only computed for the last prompt position
   even during prefill, so one legitimately-GEMV-shaped dispatch appears
   inside an otherwise-tiled prefill block); a block with only GEMV/coop
   markers and no tiled/gemm marker is decode; blocks with neither (model
   load/init bookkeeping) are excluded from both phases.

## Decision 3: Parsing the event name JSON — small custom script, not the canned Inspector CLI

**Decision**: Write a small Python script that loads events via the
`Inspector` class's lower-level API (`Inspector(etdump_path=...)`, no
`etrecord=`), iterates `event_blocks`, and JSON-parses each event's `name`
field to extract kernel name, operator name, and per-arg tensor sizes
(deriving M/K/N for matmul-shaped args). Do not rely on
`devtools/inspector/inspector_cli.py`'s default tabular printer.

**Rationale**: the existing Vulkan profiling doc
(`docs/source/backends/vulkan/tutorials/etvk-profiling-tutorial.md:47-143`)
demonstrates `inspector_cli.py --etdump_path <file>` for the generic
`executor_runner`, but its sample output table shows only kernel names and
timings — no shape/args JSON visible — confirming the CLI's canned view
doesn't surface the embedded JSON either; a small custom script over the
same `Inspector` object is the direct, low-effort way to get it.
`Inspector.__init__` only parses pre-existing `.etdump`/`.etrecord` files
(no model execution, no pybind Vulkan build needed), so this script runs
safely in the project's `.venv` even though that `.venv`'s pybind build
lacks the Vulkan backend registered — this analysis step never invokes
Vulkan itself, it only reads the file `llama_main` already wrote.

**Alternatives considered**: extending `inspector_cli.py` itself to print
shapes — rejected as touching shared devtools code for a one-off analysis
need; a standalone script scoped to this feature is simpler and matches the
constitution's "don't build new when reuse suffices" spirit while still
reusing the `Inspector` class for the actual flatcc parsing.

## Decision 4: Category rollup mapping reuses `001`'s `results/shapes.json`

**Decision**: Map each aggregated (kernel, shape) entry to a category
(attention projection, feed-forward, output/vocab projection, non-shader
overhead, other) by matching its `(K, N)` shape against the per-model
`ops` table already in `001-minipc-baseline-benchmarks/results/shapes.json`
(e.g., `wq`/`wk`/`wv`/`wo` → attention projection; `w1_gate`/`w3_up`/`w2_down`
→ feed-forward; `lm_head` → output projection), rather than trying to
recover the specific named layer from the kernel name alone.

**Rationale**: kernel names alone don't disambiguate, e.g., `wq` from `wo`
for Llama 3.1 8B — both are `linear_q4gsw_tiled_*` at the identical
(4096, 4096) shape — but that ambiguity doesn't matter for category rollup,
since both fall under the same "attention projection" category regardless
of which specific one a given dispatch was. Reusing `001`'s already-computed
shape catalog avoids re-deriving per-model dimensions from `params.json`
again.

**Alternatives considered**: attempting finer per-layer attribution via
ETRecord's `module_hierarchy`/`stack_traces` (available even without shape
data, per `_inspector.py`'s `Event` fields) — rejected as unnecessary
precision for this feature's category-rollup goal (FR-009); may be worth
revisiting only if a future feature needs true per-layer (not per-category)
attribution.

## Decision 5: Decode profiling window size

**Decision**: Profile a short decode window (on the order of 8-16 steps,
exact count to be finalized against actual ETDump file size/runtime during
task execution) instead of the full 1024-step decode used for `001`'s
throughput measurement.

**Rationale**: per-step shader/shape composition does not change with
decode position on this architecture — the KV-cache buffers are
preallocated to `max_seq_len` and every step's attention/linear dispatches
read the full buffer regardless of `input_pos` (confirmed by prior profiling
on this same hardware, cited in the spec's Edge Cases). A short window
gives the same per-step breakdown at a fraction of the ETDump size and
capture time.

**Alternatives considered**: profiling the full 1024 decode steps to exactly
match `001`'s configuration — rejected as producing ~64-128x more identical
per-step data for no attribution benefit, and unnecessarily inflating
capture time and `.etdump` file size for a large model like Llama 3.1 8B.
