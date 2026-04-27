# Custom-ops bench validator silently passes int outputs

## What was attempted

The Phase 4 int8 coopmat prototype
(`linear_coopmat_int8_bench`) computes int8×int8→int32 GEMM and produces an
`int32` output buffer. The bench harness's standard correctness path
(`ValueSpec::validate_against_reference`) was expected to compare the GPU
output against the CPU reference produced by `int8_linear_reference`.

## What happened

`backends/vulkan/test/custom_ops/utils.cpp:694–696`:

```cpp
if (dtype != vkapi::kFloat && dtype != vkapi::kHalf) {
  return true;
}
```

The validator short-circuits to `true` for any non-float dtype. Every
`PASSED` printed by the int8 bench therefore means "kernel ran without
crashing", not "output was numerically validated against the CPU
reference". The reference computation in `int8_linear_reference` runs and
fills `ref_int32_data`, but nothing reads it.

## Why it matters

For the Phase 4 go/no-go decision the throughput numbers are still
trustworthy (kernel timing is independent of the validator), but the
"PASSED" labels are not evidence of correctness. Anyone reading the
Phase 4 report should interpret correctness as **unverified**.

For any Phase 5 production work, this gap must be closed before landing
an int8 coopmat dispatch into the production runtime. The existing
production quantized linear path (`linear_q8ta_q8csw_tiled`) has its own
correctness coverage (CPU integer reference inside `q8csw_linear` bench);
that machinery is not directly reusable because it operates on
`linear_q8ta_q8csw.default` op-style inputs (with scales, sums, bias),
not raw int8 GEMM.

## Recommended next action

Extend `validate_against_reference` to handle `kInt` (and probably `kChar`
/ `kByte` while you are there). The Phase 1 large-shape methodology used
`std::isnan(reference_value)` as the sentinel-skip mark for sampled
validation; the integer analog is either:

- `std::numeric_limits<int32_t>::min()` as a sentinel value (already used
  by `int8_linear_reference` in the prototype), or
- a separate `std::vector<bool>` skip mask alongside `ref_int32_data`.

Either works; the sentinel approach is consistent with the existing fp
path and changes fewer call sites. Threshold: `abs_tolerance = 0.0f`,
`rel_tolerance = 0.0f` for int32 GEMM (it is exact unless the kernel
overflows, which itself is a correctness signal).

## Resolution

The Phase 4 work landed this fix in
`backends/vulkan/test/custom_ops/utils.cpp`: a `kInt` branch was
added to `validate_against_reference` that uses `INT32_MIN` as the
sentinel-skip mark and exact equality for checked elements. With the
validator extended, the original `linear_coopmat_int8.glsl` prototype
*failed* numerical validation on 8 of 9 random-data shapes — exposing
a separate, real shader bug
(`k_dim_unit_mismatch.md`). After fixing both the validator and the
shader, all 12 measured shapes pass under `RANDINT8` data.

This is the load-bearing example of why the validator gap mattered:
the shader had been silently producing wrong values for an unknown
period before the validator was extended.

## Exact log evidence

```
yanwen_docs/agent_results/int8_coopmat_exploration_rdna3/linear_int8_coopmat_bench.log
```

All 9 cases print `PASSED` regardless of what the int32 output actually
contains.
