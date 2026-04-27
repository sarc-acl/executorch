# linear_coopmat_bench initially bypassed linear_coopmat

## What was attempted

Ran the minimal fp16 linear cooperative-matrix benchmark from commit
`01eb65232fcff5ad0ff62b0d5e4fcc1009850364` using:

```bash
cmake-out-vk/backends/vulkan/test/custom_ops/linear_coopmat_bench
```

## What happened

The initial raw log showed `Cooperative matrix: SUPPORTED`, but all buffer
cases dispatched `linear_vec_*_buffer_texture2d_half`; none dispatched
`linear_coopmat_half`.

## Why it matters

`linear_coopmat_bench.cpp` was calling `test_etvk.test_mm.default`.
`aten.mm.default` prepacked constant `mat2`, but then directly called
`add_linear_tiled_node`; it did not route through `linear_packed_weight`, so the
coopmat dispatch predicate in `aten.linear.default` was bypassed.

## Exact evidence

Initial raw output:

```text
Cooperative matrix: SUPPORTED
linear_vec_buffer_texture2d_half ... cm_fp16_LLM_QKV_64tok ... PASSED
```

Code path:

```text
test_etvk.test_mm.default -> aten.mm.default -> matmul_tiled()
matmul_tiled() constant mat2 branch -> add_linear_tiled_node()
```

## Workaround

Use `test_etvk.test_linear.default` with a constant transposed weight shaped
`[N, K]` and a `None` bias. After this narrow harness change, eligible buffer
fp16 cases dispatch:

```text
[VK_LINEAR] Using linear_coopmat (cooperative matrix, bias=0)
linear_coopmat_half ... cm_fp16_LLM_QKV_64tok ... PASSED
```

## Recommended next action

Keep the benchmark on `aten.linear.default` for the fp16 linear sweep. If future
matmul-via-constant-weight coverage is needed, add an explicit coopmat branch to
`aten.mm.default` separately rather than assuming it shares the linear dispatch
predicate.
