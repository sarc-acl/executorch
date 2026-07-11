# Contract: SDPA Coopmat Correctness + Microbenchmark Data Formats

## Correctness harness (`sdpa_test.cpp` extension)

- New `TEST(VulkanSDPATest, ...)` case(s) using a tile-aligned `S` (and
  `context_len`), `Buffer` storage, run with `ET_VK_SDPA_COOPMAT=1` set
  (and `ET_VK_DISABLE_COOPMAT` unset).
- MUST still pass with the env var unset too (tiled path, existing
  behavior) -- this is the safety-property check that the new shape itself
  isn't the source of any failure, isolating whether a failure (if any) is
  specific to the coopmat path.
- Correctness bar: `test_reference_sdpa`'s existing ATen-ground-truth
  tolerance, unchanged.

## Microbenchmark harness (new, `backends/vulkan/test/custom_ops/`)

- Built on this workstream's existing `TestCase`/`execute_test_cases`
  timing framework (iteration count + stdev on every reported time,
  constitution Principle IV).
- One case pair (`ET_VK_SDPA_COOPMAT` unset / set) per target model, at
  that model's real `head_dim`/`num_heads`/`num_kv_heads` and the fixed
  2048-token prefill workload.
- Each case's dispatched kernel name(s) for both the QK^T and attn·V shader
  positions MUST be captured (via `BenchmarkResult::get_shader_timings()`,
  matching `test_coopmat_linear_bench.cpp`'s existing pattern) -- no timing
  is reported for a model unless both shaders' `_coopmat` dispatch is
  confirmed.

## SPIR-V inspection

Plain `spirv-dis` output for both `sdpa_compute_attn_weights_coopmat` and
`sdpa_compute_out_coopmat`'s compiled `.spv` (the storage/dtype variant
actually observed dispatching in the microbenchmark), under
`results/spirv/<kernel_name>.dis.txt` -- confirming
`OpCooperativeMatrixLoadKHR`/`OpCooperativeMatrixMulAddKHR` presence, per
`007`'s established format.

## `results/sdpa-coopmat-microbench-report.md`

Rules a consumer can depend on:

1. A correctness/dispatch verification summary first -- which shapes were
   checked, pass/fail, SPIR-V status -- before any performance table
   (constitution Principle I: no perf number substitutes for correctness).
2. The 3-row (or fewer) per-model comparison table, each row's tiled/coopmat
   timing, speedup %, and significance -- no result without its iteration
   count and stdev.
3. An Excluded section, present even if empty, for any model whose shape
   failed tile alignment or whose dispatch/correctness check failed --
   never silently dropped from the three-model count (FR-006).
4. One overall statement of whether SDPA coopmat helps real prefill
   attention on this device, at this tier -- explicitly scoped as tier-1
   (shader microbenchmark), not a claim about real end-to-end model
   performance (that would be a future tier-2 feature, out of scope here).
