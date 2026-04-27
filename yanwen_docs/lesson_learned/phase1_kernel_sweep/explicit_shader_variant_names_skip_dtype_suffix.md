# Explicit shader variant names skip dtype suffix

## What was attempted

Added explicit YAML variants for fp16-accumulator kernels:

```text
linear_coopmat_accum_fp16
matmul_coopmat_accum_fp16
```

The first dispatcher change appended the normal dtype suffix and requested
`linear_coopmat_accum_fp16_half`.

## What happened

The fp16-accumulator linear benchmark aborted at the first coopmat case:

```text
Could not find ShaderInfo with name linear_coopmat_accum_fp16_half
```

The generated registry contained `linear_coopmat_accum_fp16` and
`matmul_coopmat_accum_fp16` without `_half`.

## Why it matters

Explicit shader variant names in the YAML do not automatically follow the same
runtime suffix convention as generated dtype-specialized names.

## Fix

Select the explicit variants with `VK_KERNEL(linear_coopmat_accum_fp16)` and
`VK_KERNEL(matmul_coopmat_accum_fp16)` instead of constructing a suffixed string.
