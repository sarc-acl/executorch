# Vulkan Matmul/Linear GLSL Shader Baseline

This note summarizes the current understanding of the Vulkan GLSL matmul and
linear shaders, with Stephen Jia's mid-March implementation as the baseline for
future optimization work.

## Scope

Production Vulkan operator shaders live under:

```text
backends/vulkan/runtime/graph/ops/glsl/
```

There are other GLSL files in the repository, but they are test, tooling, or
third-party shaders:

- `backends/vulkan/test/glsl/`
- `backends/vulkan/test/custom_ops/glsl/`
- `backends/vulkan/tools/gpuinfo/glsl/`
- `backends/vulkan/third-party/VulkanMemoryAllocator/src/Shaders/`

For runtime operator work, use `backends/vulkan/runtime/graph/ops/glsl/` as the
primary source of truth.

## Stephen Jia Baseline

Stephen Jia's relevant baseline commit is:

```text
7a63aff49f6c9c269a9bb67bddfffd93232e3aca
Date: 2026-03-14
Subject: [ET-VK][matmul] Re-implement fp32/fp16 matmul and linear with tiled compute and blocked weight packing
Author: Sicheng Stephen Jia <ssjia@meta.com>
```

This commit replaced older addmm-oriented shaders and added tiled fp32/fp16
matmul and linear implementations.

Top-level shaders added by the baseline:

```text
backends/vulkan/runtime/graph/ops/glsl/matmul_scalar.glsl
backends/vulkan/runtime/graph/ops/glsl/matmul_vec.glsl
backends/vulkan/runtime/graph/ops/glsl/linear_scalar.glsl
backends/vulkan/runtime/graph/ops/glsl/linear_vec.glsl
backends/vulkan/runtime/graph/ops/glsl/pack_fp_linear_weight.glsl
```

## Baseline Data Types

Stephen's March 14 baseline shaders are floating point only:

```text
matmul_scalar.glsl
matmul_vec.glsl
linear_scalar.glsl
linear_vec.glsl
pack_fp_linear_weight.glsl
```

They generate variants for:

```text
float  // fp32
half   // fp16
```

These shaders do not implement int8, int4, or q4/q8 quantized matmul/linear.
Quantized matmul and linear paths live in separate shader families.

Int8/q8 examples:

```text
q8ta_linear.glsl
q8ta_linear_gemv.glsl
linear_q8csw_tiled.glsl
linear_q8ta_q8csw_tiled.glsl
```

Int4/q4 examples:

```text
linear_q4gsw_tiled.glsl
linear_q4gsw_coop.glsl
linear_dq8ca_q4gsw_tiled.glsl
pack_q4_linear_weight.glsl
pack_int4_linear_weight_transposed_interleaved.glsl
```

This distinction matters for optimization work: the Stephen baseline should be
treated as the fp32/fp16 tiled matmul/linear path, not the quantized LLM path.

Important helper headers added by the baseline:

```text
backends/vulkan/runtime/graph/ops/glsl/linear_fp_packed_weight_tile_load.glslh
backends/vulkan/runtime/graph/ops/glsl/matmul_fp_bias_apply.glslh
backends/vulkan/runtime/graph/ops/glsl/matmul_fp_mat1_tile_load.glslh
backends/vulkan/runtime/graph/ops/glsl/matmul_fp_mat2_tile_load.glslh
backends/vulkan/runtime/graph/ops/glsl/matmul_fp_out_tile_store.glslh
```

Relevant C++ dispatch/registration changes:

```text
backends/vulkan/runtime/graph/ops/impl/Linear.cpp
backends/vulkan/runtime/graph/ops/impl/Linear.h
backends/vulkan/runtime/graph/ops/impl/Matmul.cpp
backends/vulkan/runtime/graph/ops/impl/MatMul.h
```

## Direct Matmul vs Linear

The matmul shaders implement generic matrix multiplication:

```text
output = mat1 @ mat2
```

They read both operands as normal runtime tensors:

- `t_mat1`
- `t_mat2`
- optional `t_bias`

The linear shaders implement neural network linear layers:

```text
output = input @ weight + bias
```

The key implementation difference is that linear weights are prepacked before
execution:

- `t_mat1`: runtime input activation
- `t_weight_packed`: prepacked linear weight
- optional `t_bias`

That prepacking is performed by:

```text
backends/vulkan/runtime/graph/ops/glsl/pack_fp_linear_weight.glsl
```

The packed weights are loaded by:

```text
backends/vulkan/runtime/graph/ops/glsl/linear_fp_packed_weight_tile_load.glslh
```

## Scalar and Vec Variants

The baseline has scalar and vectorized entry shaders for both matmul and linear.

Matmul:

```text
matmul_scalar.glsl
matmul_vec.glsl
```

Linear:

```text
linear_scalar.glsl
linear_vec.glsl
```

The C++ side chooses scalar versus vec based on storage and dimension alignment.
For linear, `Linear.cpp` uses the vec path when tensor widths are aligned to 4;
buffer-backed unaligned cases use the scalar path.

## Tiled Compute Shape

The baseline shaders use tiles over the output matrix:

- `M`: rows of the input/output
- `N`: output columns / output channels
- `K`: reduction dimension

The shaders expose tile parameters such as:

```text
TILE_M
TILE_K4
TILE_N4
```

The `4` suffix means the dimension is grouped in vec4 units. For example,
`N4 = div_up_4(N)`.

The shared compute helper is:

```text
linear_fp_output_tile_fp_compute.glslh
```

It accumulates an `FPOutTile` from an `FPInputTile` and `FPWeightTile` using
vectorized FMA operations.

## Linear Weight Packing Layout

The fp linear prepacker stores weights in a blocked `4OC x 4IC` layout.

For a logical linear operation:

```text
output[M, N] = input[M, K] @ weight[K, N]
```

the packed layout groups:

- `N`: output channels / columns, grouped as `n4`
- `K`: input channels / reduction dim, grouped as `k4`

One packed record stores a `vec4` of four output-channel weights for one
input-channel lane:

```text
packed[b, k4, n4, dk] = {
  weight[k4 * 4 + dk, n4 * 4 + 0],
  weight[k4 * 4 + dk, n4 * 4 + 1],
  weight[k4 * 4 + dk, n4 * 4 + 2],
  weight[k4 * 4 + dk, n4 * 4 + 3],
}
```

So one logical `4IC x 4OC` block is stored as four `vec4`s:

```text
dk = 0: [w(k0,n0), w(k0,n1), w(k0,n2), w(k0,n3)]
dk = 1: [w(k1,n0), w(k1,n1), w(k1,n2), w(k1,n3)]
dk = 2: [w(k2,n0), w(k2,n1), w(k2,n2), w(k2,n3)]
dk = 3: [w(k3,n0), w(k3,n1), w(k3,n2), w(k3,n3)]
```

For texture storage, `pack_fp_linear_weight.glsl` writes:

```text
x = n4 * 4 + dk
y = b * K4 + k4
value = vec4(4 output-channel weights)
```

For buffer storage, the equivalent index is:

```text
((b * K4 + k4) * N4 + n4) * 4 + dk
```

The packed tensor shape is created in `Linear.cpp` as:

```text
height = B * K4
width  = N4 * 4 * 4  // scalar width; kWidthPacked means 4 scalars per texel
```

The prepacker uses texture2d storage unless the packed texture would exceed
device texture limits; then it falls back to buffer storage.

## Why the Packed Layout Helps

The packed layout matches the compute loop's vector shape. A scalar input value
from the `K` dimension can be broadcast across a `vec4` of four output-channel
weights, then accumulated into four output columns at once.

Benefits:

- contiguous/coalesced reads for groups of output-channel weights
- no runtime transpose of model weights
- better reuse for constant model weights
- natural fit for Vulkan texel/vec4 operations
- one prepack cost amortized across repeated inference calls

## Updates Since the Baseline

The four top-level baseline shaders have not materially changed on the current
branch since the March 14 implementation:

```text
matmul_scalar.glsl
matmul_vec.glsl
linear_scalar.glsl
linear_vec.glsl
```

Relevant helper/prepack updates after the baseline:

```text
1f0e737f24adf02db5717ab38497ade2bba245c8
Date: 2026-03-17
Subject: [ET-VK][qlinear] Add bias support to q4gsw and dq8ca_q4gsw quantized linear ops
```

This added `add_bias_to_out_tile()` in:

```text
linear_fp_output_tile_fp_compute.glslh
```

That helper is used by quantized linear shaders, not by the fp matmul/linear
entry shaders directly.

```text
6bd9bca8534c1750bbb93816ea33bc6260a7a8be
Date: 2026-04-01
Subject: [ET-VK] Fix pack_fp_linear_weight for devices without VK_KHR_16bit_storage (#18653)
```

This updated:

```text
pack_fp_linear_weight.glsl
pack_fp_linear_weight.yaml
Linear.cpp
```

The important change was separating source buffer dtype as `BUF_DTYPE`, allowing
fp16 weight packing to work on devices without `VK_KHR_16bit_storage`.

There was also an April 24 SDPA-related update on remote SS-JIA branches that
touched:

```text
linear_fp_output_tile_fp_compute.glslh
linear_fp_packed_weight_tile_load.glslh
```

That update was observed in remote branches, not on the current branch at the
time of this note.

## Matrix-Multiplication-Related Shader Families

Core fp matrix multiplication:

```text
matmul_scalar.glsl
matmul_vec.glsl
linear_scalar.glsl
linear_vec.glsl
pack_fp_linear_weight.glsl
```

Quantized linear/GEMM:

```text
q8ta_linear.glsl
q8ta_linear_gemv.glsl
linear_q8csw_tiled.glsl
linear_q8ta_q8csw_tiled.glsl
linear_q4gsw_tiled.glsl
linear_q4gsw_coop.glsl
linear_dq8ca_q4gsw_tiled.glsl
pack_q8_linear_weight.glsl
pack_q4_linear_weight.glsl
pack_int4_linear_weight_transposed_interleaved.glsl
```

Attention matmul kernels:

```text
sdpa_compute_attn_weights_tiled.glsl   // Q x K
sdpa_compute_attn_weights_coop.glsl    // cooperative Q x K
sdpa_compute_out_tiled.glsl            // attention weights x V
sdpa_compute_out_coop.glsl             // cooperative attention weights x V
```

Conv-as-GEMM or GEMM-adjacent paths:

```text
conv2d_pw_tiled.glsl
conv1d_pw.glsl
conv2d_*_linear_tiled.glsl
im2col.glsl
im2col_packed_int8.glsl
quantize_and_pack_im2col.glsl
```

These are not plain `aten.mm` or `aten.linear`, but they use similar tiled dot
product structure or linear/im2col-style packing.

## LLaMA 3.1-Relevant Shader Groups

For a LLaMA 3.1-style model, the relevant shader groups are mostly:

Linear projections and MLP:

```text
linear_vec.glsl
linear_scalar.glsl
pack_fp_linear_weight.glsl
linear_q4gsw_tiled.glsl
linear_q4gsw_coop.glsl
linear_dq8ca_q4gsw_tiled.glsl
linear_q8csw_tiled.glsl
linear_q8ta_q8csw_tiled.glsl
q8ta_linear.glsl
q8ta_linear_gemv.glsl
```

Attention and KV cache:

```text
sdpa_compute_attn_weights_tiled.glsl
sdpa_compute_attn_weights_coop.glsl
sdpa_attn_weights_softmax.glsl
sdpa_compute_out_tiled.glsl
sdpa_compute_out_coop.glsl
sdpa_kv_cache_update.glsl
```

Rotary embedding:

```text
rotary_embedding.glsl
rotary_embedding_hf.glsl
```

RMSNorm:

```text
rms_norm_texture.glsl
rms_norm_buffer.glsl
```

Embedding:

```text
embedding_texture.glsl
embedding_buffer.glsl
embedding_q4gsw.glsl
```

General graph glue may also involve:

```text
softmax.glsl
softmax_buffer.glsl
binary_op_*.glsl
unary_op.glsl
view_*.glsl
permute_*.glsl
gather_*.glsl
index_*.glsl
repeat_*.glsl
where.glsl
```

## Baseline Takeaways for Future Optimization

The fp linear path is the most important baseline for optimizing repeated model
inference because it pre-packs constant weights into a blocked layout. The fp
matmul path is more general, but less specialized because both operands are
treated as ordinary runtime tensors.

Optimization work should start by preserving these functional assumptions:

- support fp32 and fp16 for the Stephen baseline
- keep int8/int4/q4/q8 support in the separate quantized shader families
- support scalar and vec paths
- handle non-multiple-of-4 `K` and `N` dimensions
- preserve optional bias behavior
- keep texture2d and buffer fallback behavior for packed weights
- preserve dynamic shader selection from `Linear.cpp` and `Matmul.cpp`

The most relevant files to read before changing behavior are:

```text
backends/vulkan/runtime/graph/ops/glsl/linear_vec.glsl
backends/vulkan/runtime/graph/ops/glsl/linear_scalar.glsl
backends/vulkan/runtime/graph/ops/glsl/matmul_vec.glsl
backends/vulkan/runtime/graph/ops/glsl/matmul_scalar.glsl
backends/vulkan/runtime/graph/ops/glsl/pack_fp_linear_weight.glsl
backends/vulkan/runtime/graph/ops/glsl/linear_fp_output_tile_fp_compute.glslh
backends/vulkan/runtime/graph/ops/glsl/linear_fp_packed_weight_tile_load.glslh
backends/vulkan/runtime/graph/ops/impl/Linear.cpp
backends/vulkan/runtime/graph/ops/impl/Matmul.cpp
```
