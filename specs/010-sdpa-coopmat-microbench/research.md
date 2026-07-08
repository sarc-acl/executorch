# Research: SDPA Coopmat Correctness + Microbenchmark

## Decision 1: `test_coopmat_attention_bench.cpp` does NOT test the shaders this feature cares about

**Finding** (constitution Principle VI -- verified by reading the source,
not assumed from its name/comments): the imported
`backends/vulkan/test/custom_ops/test_coopmat_attention_bench.cpp` builds
`test_etvk.test_mm.default` dispatches with `impl_selector="coopmat:<tile>"`
-- this routes through `add_matmul_coopmat_node()`/the generic
`matmul_coopmat` shader family (`GemmCoopmat.cpp`), decomposing attention
into two plain GEMMs at attention-*shaped* problems. It never constructs an
`sdpa_with_kv_cache`/`custom_sdpa` op, never calls
`add_sdpa_compute_attn_weights_node`/`add_sdpa_compute_out_node`, and never
dispatches `sdpa_compute_attn_weights_coopmat`/`sdpa_compute_out_coopmat` --
the two dedicated shaders `SDPA.cpp` actually gates on. It also has no
softmax, causal masking, or KV-cache semantics. Despite its name, it is a
tile-geometry sweep tool for the *unrelated* matmul tile-sweep thread (see
`010`'s spec.md Edge Cases), not a test of this feature's subject.

**Decision**: Treat `test_coopmat_attention_bench.cpp` as out of scope --
do not wire it into the build, do not extend it, do not cite it as
correctness evidence for `sdpa_compute_attn_weights_coopmat`/
`sdpa_compute_out_coopmat`. Build genuinely new coverage instead (Decisions
2-3 below).

**Alternatives considered**: Extending `test_coopmat_attention_bench.cpp` to
also exercise the real SDPA shaders. Rejected -- its entire harness
(`test_etvk.test_mm.default`, tile-variant selector, plain-GEMM reference)
is structurally about the matmul tile-sweep, not the SDPA op; retrofitting
it would mean rewriting nearly all of it, with no reuse benefit over
starting from the harness that already tests the real op (Decision 2).

## Decision 2: Correctness harness -- extend `sdpa_test.cpp`, not build from scratch

**Decision**: Add one new test case to the existing
`backends/vulkan/test/op_tests/sdpa_test.cpp` (`VulkanSDPATest` suite,
`test_vulkan_sdpa`/`test_reference_sdpa` machinery), using a large,
tile-aligned sequence length and `Buffer` storage, with
`ET_VK_SDPA_COOPMAT` set for that run.

**Rationale**: This file already has the load-bearing pieces this feature
needs and nothing else in the tree does:
- `test_reference_sdpa` computes a CPU/ATen ground truth for the exact same
  `sdpa_with_kv_cache`/`custom_sdpa` op family `SDPA.cpp` implements --
  causal masking, KV-cache, softmax, everything, not a decomposed proxy.
- It already parameterizes storage type and already has passing `Buffer`-
  storage cases (`test_sdpa_op_*`), so `Buffer` storage is proven to work
  through this exact harness already -- only the shape needs to change.
- Its existing cases all use tiny sequence lengths (`S` in the single
  digits to ~111, decode-incremental style) -- none are tile-aligned to
  `kSdpaCmQkTileM`(128)/`kSdpaCmTileM`(64)/`kSdpaCmTileN`(64)/
  `kSdpaCmTileK`(32), so none of them have ever exercised the coopmat gate.
  This is genuinely new coverage, not a citation of existing passing cases
  (mirroring `009`'s Decision 2 bar for its own rank-3 case).

**Alternatives considered**: A brand-new, dedicated correctness harness
built on the `custom_ops` `TestCase`/`ValueSpec` prototyping framework
(this workstream's usual pattern for new microbenchmarks, e.g.
`test_coopmat_linear_bench.cpp`). Rejected as the *first* correctness
signal -- constructing correct KV-cache/causal-mask/softmax semantics from
scratch in that lighter framework would duplicate `sdpa_test.cpp`'s
already-proven ATen-reference machinery for no benefit; the lighter
framework remains the right choice for Decision 3's *timed* microbenchmark,
where `sdpa_test.cpp`'s gtest structure has no iteration-count/stdev timing
support.

## Decision 3: Microbenchmark harness -- new, dedicated, timed harness required

**Decision**: Author a new test-owned microbenchmark file under
`backends/vulkan/test/custom_ops/` that dispatches the real
`sdpa_with_kv_cache`/`custom_sdpa` op (not a decomposed proxy) at each
target model's real prefill shape, toggling `ET_VK_SDPA_COOPMAT` on/off,
using this workstream's established `BenchmarkResult`/`execute_test_cases`
timing infrastructure (iteration count + stdev, matching every prior
microbenchmark in this workstream).

**Rationale**: Neither existing harness fits this tier: `sdpa_test.cpp` is
gtest-based with no timing/iteration infrastructure; `test_coopmat_attention_bench.cpp`
times the wrong shader family entirely (Decision 1). `007`'s own precedent
(reusing `test_llama_baseline_bench.cpp` for the linear-coopmat
microbenchmark) isn't directly available here either -- that harness
measures linear ops from a full exported model's real shapes, not a
standalone SDPA dispatch -- so a new, SDPA-specific harness is needed, sized
similarly to `test_coopmat_linear_bench.cpp`'s own scope (one op family, a
handful of real shapes, correctness-gated before timing is trusted).

**Alternatives considered**: Extracting timing from a full model run
(mirroring `002`'s ETDump-based per-op timing). Rejected as the *tier-1*
mechanism -- that is exactly this workstream's tier-2 (model-level)
methodology, reserved for `006`/`009`-style features; tier-1 needs an
isolated, standalone dispatch per the constitution's two-tier discipline
(Principle IV).

## Decision 4: Dispatch confirmation without built-in kernel-name introspection

**Decision**: The new microbenchmark harness (Decision 3) is built on the
`custom_ops` prototyping framework, which already exposes dispatched kernel
names via `BenchmarkResult::get_shader_timings()`/`get_kernel_name()` (the
same mechanism `test_coopmat_linear_bench.cpp` already uses) -- reuse it
directly: confirm each coopmat-toggled run's dispatched shader name contains
`_coopmat` before its timing is trusted (FR-003), exactly mirroring `009`'s
ETDump-based dispatch check but via the harness's own instrumentation
instead of a separate ETDump capture (no `.pte`/full-model export exists at
this tier to attach ETDump to).

**Alternatives considered**: A separate ETDump capture, mirroring `009`.
Rejected -- ETDump is this workstream's tier-2 (model-level) verification
mechanism (constitution Principle VI's ETDump clause applies to tier-2
studies specifically); at tier-1, the harness's own kernel-name field is the
established mechanism (`007`'s precedent).

## Decision 5: Real per-model prefill SDPA shapes are already tile-aligned

**Finding** (computed directly from each checkpoint's `params.json`, not
assumed): `head_dim = dim / n_heads` gives 128 for both `llama-3.1-8b`
(`dim=4096, n_heads=32`) and `llama-3.2-3b` (`dim=3072, n_heads=24`), and 64
for `llama-3.2-1b` (`dim=2048, n_heads=32`). At the fixed 2048-token prefill
workload (`S = context_len = 2048`, single prefill call, `input_pos=0`):
- QK^T requires `S % 128 == 0` (2048 ✓), `context_len % 64 == 0` (2048 ✓),
  `D % 32 == 0` (128 ✓, 64 ✓ for all three models).
- attn·V requires `S % 64 == 0` (2048 ✓), `D % 64 == 0` (128 ✓, 64 ✓ exactly
  one tile), `context_len % 32 == 0` (2048 ✓).

**Decision**: All three target models are expected to be coopmat-eligible
by shape at the fixed prefill workload -- FR-006's exclusion path is
retained as a safety net (per constitution Principle VI, verified via the
harness's own dispatch confirmation, not assumed from this arithmetic
alone) but no model is expected to trigger it.

**Alternatives considered**: None -- this is a direct computation from
existing, already-available data (`params.json`, the same source `001`
used for its own shape derivation), not a design choice.

## Decision 6: Prerequisite -- `009`'s storage/`force_fp16` fix is required

**Finding**: `SDPA.cpp`'s coopmat gate (`sdpa_buf_half`) requires `Buffer`+
`half` storage for Q/K/V/attn_weights/out, the same precondition the
quantized-linear coopmat gate has. `009`'s research.md Decision 8 found and
fixed a pass-level bug (`tag_memory_meta_pass.py`'s `force_fp16` branch
unconditionally forcing `ANY_TEXTURE`, defeating any `Buffer` storage
request) that would equally defeat SDPA's coopmat gate if unfixed --
`sdpa_buf_half` would never see `Buffer` storage for these tensors either.

**Decision**: This feature depends on `009`'s pass fix being present in the
build (it already is, in the current working tree, uncommitted). No new fix
is anticipated here, but the correctness harness (Decision 2) MUST verify
`Buffer` storage is actually reaching Q/K/V/attn_weights/out (via the same
kernel-name-suffix inspection this workstream already uses), not assume it
from the fix being present elsewhere.

**Alternatives considered**: None -- this is a direct dependency, not a
choice.

## Decision 7: `op_tests` build mechanics

**Finding**: `backends/vulkan/test/op_tests/CMakeLists.txt` requires
`find_package(GTest CONFIG REQUIRED)` and links against Torch/ATen (for
`test_reference_sdpa`'s ground truth) -- a heavier, separate dependency set
than the lightweight `custom_ops` prototyping framework used elsewhere in
this workstream. It is not currently configured/built in this environment's
`cmake-out-vk` tree.

**Decision**: Configure `backends/vulkan/test/op_tests` as its own CMake
sub-build (mirroring the existing `backends/vulkan/test/custom_ops`
sub-build pattern), on top of the already-installed `cmake-out-vk` tree.

**Alternatives considered**: None -- this is the only build path
`sdpa_test.cpp` supports; it is not part of the `custom_ops` target.

## Decision 8 (found during implementation, amends Decision 3): the `TestCase`/`ValueSpec` framework cannot build the microbenchmark harness either

**Finding**: `sdpa_impl` (`llama.custom_sdpa.default`, the LLM-mode op the
coopmat gate applies to) requires an `input_pos_symint` argument -- a real
`SymInt`, needed for KV-cache position tracking and the dispatch pickers'
dynamic-shape resize logic (`read_symint`/`get_or_create_int_param_buffer`).
Grepped `backends/vulkan/test/custom_ops/utils.{h,cpp}` for any
SymInt/symint support in `ValueSpec`/`TestCase`: none exists anywhere in
that framework -- it only ever constructs plain tensors, scalars, and
`is_none` placeholders (confirmed by reading every `ValueSpec` constructor
and the `TestCase` class directly, not assumed from a partial read).
Decision 3's plan to build the new microbenchmark on this framework (the
same one `test_coopmat_linear_bench.cpp` uses) is therefore not viable for
this specific op family.

**Decision**: Build the new microbenchmark harness on the same lower-level,
direct `ComputeGraph` construction `sdpa_test.cpp` already uses (proven
correct and dispatch-confirmable in User Story 1, via `graph.add_symint()`
and the query-pool mechanism) instead of the `TestCase`/`execute_test_cases`
abstraction. This does not require Torch/ATen -- only `test_reference_sdpa`
(the correctness ground truth) needs ATen; the graph-construction and
dispatch machinery (`ComputeGraph`, `VK_GET_OP_FN`, `GraphConfig`) is pure
Vulkan-backend C++, so the new harness stays in the lightweight
`custom_ops` build (no Torch/GTest dependency, unlike `op_tests`), using its
own random half-precision data generation (`float_to_half`, already
available in `utils.h`) and its own warmup/timed-iteration loop with
manual mean/stdev computation from query-pool timestamps -- the same
iteration-count-and-stdev discipline `execute_test_cases` would have
provided, just assembled directly rather than through that framework.

**Rationale**: Isolates exactly the two coopmat-relevant shader dispatches
(`sdpa_compute_attn_weights_*`, `sdpa_compute_out_*`) per `graph.execute()`
call via the same `get_shader_timestamp_data()`/`kernel_name` mechanism
already validated in User Story 1 -- excluding the softmax dispatch in
between (an unaccelerated shader, identical regardless of the coopmat
toggle, matching `007`'s precedent of excluding non-target-op overhead from
a per-op speedup figure).

**Alternatives considered**: Extending `ValueSpec`/`TestCase` to support
`SymInt` args, to keep using the shared framework. Rejected as
disproportionate -- would mean adding a new value kind to shared
infrastructure used by every other feature in this workstream, for a
capability only this one op family needs; the direct `ComputeGraph`
approach is already proven (`sdpa_test.cpp`) and requires no shared-code
changes at all.
