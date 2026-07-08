# Quickstart: SDPA Coopmat Correctness + Microbenchmark

Real device work on the `rocky-ryzen` MiniPC, like every prior
microbenchmark-tier feature in this workstream.

## Prerequisites

- The SDPA coopmat shaders/dispatch code (`sdpa_compute_attn_weights_coopmat`/
  `sdpa_compute_out_coopmat`, `SDPA.cpp`) already exist in the working tree
  (imported from `yanwen/quant-dev-active` in a prior session).
- `009`'s `tag_memory_meta_pass.py` fix (research.md Decision 8 there) is
  present -- without it, `Buffer` storage never reaches Q/K/V/attn_weights,
  and this feature's coopmat gate can never fire (research.md Decision 6).
- Nothing else CPU/GPU-heavy running before any capture.

## 1. Build `op_tests` as its own sub-build

**Prerequisite found during implementation**: the correctness binary
(actually named `vulkan_sdpa_test`, not `sdpa_test`) needs
`custom_ops_aot_lib`, gated behind `EXECUTORCH_BUILD_KERNELS_LLM_AOT`
(default `OFF`). If the main `cmake-out-vk` tree wasn't configured with it,
reconfigure and rebuild the main install first:

```bash
cmake . -Bcmake-out-vk --preset "linux" \
    -DCMAKE_INSTALL_PREFIX=cmake-out-vk -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DEXECUTORCH_PAL_DEFAULT=posix \
    -DEXECUTORCH_BUILD_VULKAN=ON -DEXECUTORCH_BUILD_TESTS=ON \
    -DEXECUTORCH_BUILD_DEVTOOLS=ON -DEXECUTORCH_BUILD_KERNELS_LLM_AOT=ON \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_FLAGS="-include algorithm"
cmake --build cmake-out-vk -j$(nproc) --target install --config Release
```

Then configure and build the `op_tests` sub-build itself:

```bash
cmake backends/vulkan/test/op_tests/ \
  -Bcmake-out-vk/backends/vulkan/test/op_tests \
  -DCMAKE_INSTALL_PREFIX=cmake-out-vk -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_ROOT=$(pwd) \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cmake --build cmake-out-vk/backends/vulkan/test/op_tests -j$(nproc)
```

A "Skip building sdpa_test because custom_ops_aot_lib is not found" message
means the first step above was skipped or failed.

## 2. Add and run the new correctness case

Per research.md Decision 2: extend `sdpa_test.cpp`'s `VulkanSDPATest` suite
with one tile-aligned, `Buffer`+`half`-storage case (`S`/`context_len`
multiples of 128/64/32 per the alignment rules in data-model.md;
`DECOMPOSED` mode, since that's `SDPAMode::LLM`, the mode the coopmat gate
applies to). **fp16 tolerance note**: `test_vulkan_sdpa`'s correctness check
had no explicit tolerance (fp64-oriented `at::allclose` defaults) -- it had
never been exercised at `at::kHalf` before. Add the same dtype-keyed
tolerance (`atol=1e-2, rtol=1e-2` for half) the sibling
`test_vulkan_general_sdpa` helper already uses; this strictly loosens (and
is provably safe for) the existing fp32 cases. Run once with
`ET_VK_SDPA_COOPMAT` unset (must still pass -- isolates the new shape from
the coopmat path) and once with it set -- toggle both the correctness
result and the dispatched-kernel-name check (via the graph's query-pool,
`GraphConfig::enable_querypool` + `graph.context()->querypool().get_shader_timestamp_data()`)
within the test itself using `setenv`/`unsetenv`, rather than two separate
process invocations:

```bash
./cmake-out-vk/backends/vulkan/test/op_tests/vulkan_sdpa_test \
  --gtest_filter='VulkanSDPATest.test_sdpa_op_coopmat_aligned*'
```

**Do not proceed to step 4 until both pass.**

## 3. SPIR-V-inspect both coopmat shaders

```bash
spirv-dis cmake-out-vk/vulkan_compute_shaders/sdpa_compute_attn_weights_coopmat_buffer_buffer_half.spv \
  > specs/010-sdpa-coopmat-microbench/results/spirv/sdpa_compute_attn_weights_coopmat_buffer_buffer_half.dis.txt
spirv-dis cmake-out-vk/vulkan_compute_shaders/sdpa_compute_out_coopmat_buffer_buffer_half.spv \
  > specs/010-sdpa-coopmat-microbench/results/spirv/sdpa_compute_out_coopmat_buffer_buffer_half.dis.txt
grep -c "OpCooperativeMatrix" specs/010-sdpa-coopmat-microbench/results/spirv/*.dis.txt
```

## 4. Build and run the new microbenchmark harness

Per research.md Decision 3 -- a new file, not an extension of
`test_coopmat_attention_bench.cpp` (confirmed to test an unrelated shader
family, research.md Decision 1). **Revised during implementation (research.md
Decision 8)**: NOT built on the `TestCase`/`ValueSpec` framework as
originally planned -- that framework has no `SymInt` support, and
`llama.custom_sdpa.default` requires one (`input_pos_symint`). Built
directly on `ComputeGraph` instead, mirroring `sdpa_test.cpp`'s proven
`DECOMPOSED`-mode construction, with a manual warmup/timed-run loop and
query-pool-derived mean/stdev (isolating `sdpa_compute_attn_weights_*`/
`sdpa_compute_out_*` durations from the cache-update/softmax dispatches in
between). Needs `float_to_half`-encoded host data -- `maybe_cast_and_copy_into_staging`
does not support a `Float`→`Half` cast (throws) despite its name suggesting
otherwise.

```bash
cmake --build cmake-out-vk/backends/vulkan/test/custom_ops -j$(nproc) \
  --target test_sdpa_coopmat_bench
./cmake-out-vk/backends/vulkan/test/custom_ops/test_sdpa_coopmat_bench \
  > specs/010-sdpa-coopmat-microbench/results/raw/sdpa_bench_raw.log
```

For each target model's real shape, this runs the tiled case
(`ET_VK_SDPA_COOPMAT` unset) then the coopmat case (set), and reports both
shaders' dispatched kernel names alongside the timing -- do not trust a
model's speedup number unless both say `_coopmat`.

## 5. Compare and generate the report

```bash
python specs/010-sdpa-coopmat-microbench/scripts/compare_sdpa_coopmat.py \
  --bench-raw specs/010-sdpa-coopmat-microbench/results/raw/sdpa_bench_raw.log \
  --out specs/010-sdpa-coopmat-microbench/results/sdpa-coopmat-microbench-report.md
```

## 6. Sanity-check

- Every one of the three target models appears -- either in the comparison
  table or the Excluded section with a specific reason.
- No model's speedup number appears unless both its QK^T and attn·V
  dispatches were confirmed `_coopmat`.
- The correctness/SPIR-V verification summary appears before the
  performance table, not after or omitted.
