# RDNA3 fp16 WMMA kernel sweep report

## Scope

Executed fp16 kernel-side microbenchmarks for `aten.linear.default` and
non-constant `aten.mm.default` on AMD Radeon 780M / RADV Phoenix. No real LLaMA
E2E, int8, or q4 work was run.

## Source state

- Base commit: `01eb65232fcff5ad0ff62b0d5e4fcc1009850364`
- Pre-existing local state at start: `yanwen_docs/background/coopmat_previous_work.diff` staged as added, `.codex/` untracked.
- Linear benchmark harness now calls `test_etvk.test_linear.default` with fp16
  `[N,K]` constant weights and `None` bias, so eligible buffer cases route to
  `linear_coopmat`.
- Added a buffer fp16 `matmul_coopmat` shader and `matmul_coopmat_bench` for
  non-constant `aten.mm.default`.
- Added fp16-accumulator variants selected by `VK_COOPMAT_ACCUM_FP16=1`.
- Added `VK_COOPMAT_REQUIRED_SUBGROUP_SIZE` for coopmat-only subgroup 32/64
  sweeps.
- Added `VK_COOPMAT_MACRO_TILE` for explicit fp16 macro-tile shader variants:
  `16x16`, `16x32`, `32x16`, `32x32`, `16x64`, `64x16`, `32x64`, `64x32`.
- Added `VK_COOPMAT_K_STEP=16|64` for explicit fp16 K-step shader variants
  around the default `TILE_K=32`.
- Added experimental `VK_COOPMAT_TEXTURE=1` linear texture3D input/output
  coopmat shader with buffer packed weights.
- Tightened the linear coopmat dispatch guard to full tile multiples:
  `M % 64 == 0`, `N % 64 == 0`, `K % 32 == 0`.

## Device

`/dev/dri` was visible. `vulkaninfo --summary` reported GPU0 as
`AMD Radeon 780M (RADV PHOENIX)`, RADV Mesa 25.0.7. GPU1 was llvmpipe. The
benchmarks used the AMD/RADV GPU, not llvmpipe.

Raw device logs:

- `yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/dev_dri.txt`
- `yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/vulkaninfo_summary.txt`

## Build And Run

Used the tested CMake flow from `yanwen_docs/agent_plans/0_overview.md`. After
adding new GLSL/YAML variants, the top-level install target was rebuilt to
regenerate the Vulkan shader registry.

Additional successful commands after the matmul/accumulator work:

```bash
cmake --build cmake-out-vk -j$(nproc) --target install --config Release
cmake --build cmake-out-vk/backends/vulkan/test/custom_ops -j$(nproc) --target linear_coopmat_bench matmul_coopmat_bench
cmake-out-vk/backends/vulkan/test/custom_ops/linear_coopmat_bench > yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/linear_coopmat_bench_tile_guard_rerun.log 2>&1
VK_COOPMAT_ACCUM_FP16=1 cmake-out-vk/backends/vulkan/test/custom_ops/linear_coopmat_bench > yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/linear_coopmat_bench_accum_fp16_fixed.log 2>&1
VK_COOPMAT_ACCUM_FP16=1 cmake-out-vk/backends/vulkan/test/custom_ops/matmul_coopmat_bench > yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/matmul_coopmat_bench_accum_fp16.log 2>&1
VK_COOPMAT_REQUIRED_SUBGROUP_SIZE=64 cmake-out-vk/backends/vulkan/test/custom_ops/linear_coopmat_bench > yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/linear_coopmat_bench_subgroup64.log 2>&1
VK_COOPMAT_REQUIRED_SUBGROUP_SIZE=32 cmake-out-vk/backends/vulkan/test/custom_ops/linear_coopmat_bench > yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/linear_coopmat_bench_subgroup32.log 2>&1
VK_COOPMAT_REQUIRED_SUBGROUP_SIZE=64 cmake-out-vk/backends/vulkan/test/custom_ops/matmul_coopmat_bench > yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/matmul_coopmat_bench_subgroup64.log 2>&1
VK_COOPMAT_REQUIRED_SUBGROUP_SIZE=32 cmake-out-vk/backends/vulkan/test/custom_ops/matmul_coopmat_bench > yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/matmul_coopmat_bench_subgroup32.log 2>&1
for tile in 16x16 16x32 32x16 32x32 16x64 64x16 32x64 64x32; do VK_COOPMAT_MACRO_TILE=$tile cmake-out-vk/backends/vulkan/test/custom_ops/linear_coopmat_bench > yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/linear_coopmat_bench_tile_${tile}.log 2>&1; VK_COOPMAT_MACRO_TILE=$tile cmake-out-vk/backends/vulkan/test/custom_ops/matmul_coopmat_bench > yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/matmul_coopmat_bench_tile_${tile}.log 2>&1; done
for k in 16 64; do VK_COOPMAT_K_STEP=$k cmake-out-vk/backends/vulkan/test/custom_ops/linear_coopmat_bench > yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/linear_coopmat_bench_k${k}.log 2>&1; VK_COOPMAT_K_STEP=$k cmake-out-vk/backends/vulkan/test/custom_ops/matmul_coopmat_bench > yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/matmul_coopmat_bench_k${k}.log 2>&1; done
VK_COOPMAT_TEXTURE=1 cmake-out-vk/backends/vulkan/test/custom_ops/linear_coopmat_bench > yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/linear_coopmat_bench_texture_sanity.log 2>&1
cmake --build cmake-out-vk/backends/vulkan/test/custom_ops -j$(nproc) --target linear_coopmat_bench matmul_coopmat_bench
VK_BENCH_CONTINUE_ON_CORRECTNESS_FAIL=1 cmake-out-vk/backends/vulkan/test/custom_ops/linear_coopmat_bench > yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/linear_coopmat_bench_large_sampled_correctness.log 2>&1
VK_BENCH_CONTINUE_ON_CORRECTNESS_FAIL=1 cmake-out-vk/backends/vulkan/test/custom_ops/matmul_coopmat_bench > yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/matmul_coopmat_bench_large_sampled_correctness.log 2>&1
```

Build/runtime issues were documented under
`yanwen_docs/lesson_learned/phase1_kernel_sweep/`.

## Capability And Routing

- `Cooperative matrix: SUPPORTED`
- Device exposes fp16 cooperative matrix configs:
  - `16x16x16 float16 x float16 -> float16`
  - `16x16x16 float16 x float16 -> float32`
- Eligible buffer fp16 linear cases route to `linear_coopmat_half`.
- Eligible buffer fp16 matmul cases route to `matmul_coopmat_half`.
- `M=1` and `M=32` linear cases fall back to Stephen's `linear_vec` because the
  current coopmat tile is 64 rows high.
- Explicit macro-tile runs use their requested tile guard. For example,
  `VK_COOPMAT_MACRO_TILE=32x64` lets linear `M=32` cases route to
  `linear_coopmat_tile_32x64`; `64x*` variants still fall back for `M=32`.
- With `VK_COOPMAT_TEXTURE=1`, eligible fp16/no-bias texture3D linear cases
  route to `linear_coopmat_texture3d_buffer`. Texture matmul and texture
  packed-weight coopmat variants are not implemented.

## Sweep Dimensions

Phase 1 covered the following fp16 RDNA3 WMMA microbenchmark dimensions. The
hardware WMMA tile is not a sweep parameter on this AMD/RADV GPU: the exposed
fp16 cooperative matrix hardware shape is `16x16x16`. The measured sweeps are
software choices around it.

| Dimension | Available / tested values | Current recommendation |
| --- | --- | --- |
| HW WMMA tile shape | `16x16x16` only on this GPU | Fixed by hardware |
| Input types | `fp16 x fp16` | Fixed for this phase |
| Accumulator type | `fp16`, `fp32` | Use fp32 |
| Subgroup / wave size | `32`, `64` | Default 64; tune per shape if needed |
| Software macro tile | `16x16`, `16x32`, `32x16`, `32x32`, `16x64`, `64x16`, `32x64`, `64x32`, plus default `64x64` | Default 64x64; consider 32x64 for M=32 |
| K-step / K blocking | `16`, `32`, `64` | Use 32 |
| Storage mode | buffer input/output + buffer weights; texture3D input/output + buffer weights for linear only | Buffer fastest; texture coopmat experimental |
| Op type | linear, matmul | Both buffer coopmat tested |
| Shape class | BERT-like, LLM decode M=1, LLM batch M=32/64, square stress, 4096^3 | Use coopmat only for full-tile eligible shapes |
| Dispatch tile eligibility | multiples of current tile: `M % tile_m == 0`, `N % tile_n == 0`, `K % tile_k == 0` | Conservative gate: `M % 64 == 0`, `N % 64 == 0`, `K % 32 == 0` |
| Correctness mode | full CPU reference for small shapes; sampled CPU reference for large shapes | Routed large coopmat cases passed sampled validation |

Explicit list:

```text
HW tile:
  16x16x16

Accumulator:
  fp16
  fp32

Subgroup size:
  32
  64

Macro tile MxN:
  16x16
  16x32
  32x16
  32x32
  16x64
  64x16
  32x64
  64x32
  default 64x64

K-step:
  16
  32
  64

Storage:
  buffer input/output + buffer packed weights
  texture3D input/output + buffer packed weights
  texture input/output + texture packed weights: not implemented

Ops:
  linear
  matmul

Correctness mode:
  full CPU reference for small shapes
  sampled CPU reference for large shapes
```

## Large-Shape Correctness

The original microbenchmarks skipped CPU reference generation whenever any of
`M/K/N` exceeded the reference limit, so large rows could print `PASSED`
without checking numeric output. The harness now supports bounded sampled CPU
reference validation for large outputs: unchecked elements are stored as `NaN`
in the reference vector, and the validator skips only those explicit `NaN`
slots. The sampled runs checked up to 8192 deterministic output positions per
large shape.

Latest sampled correctness logs:

- `yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/linear_coopmat_bench_large_sampled_correctness.log`
- `yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/matmul_coopmat_bench_large_sampled_correctness.log`

Results:

- `linear_coopmat_half` passed sampled CPU-reference validation for all routed
  large full-tile shapes, including `256x4096x4096` and `4096^3`.
- `matmul_coopmat_half` passed sampled CPU-reference validation for all routed
  large shapes, including `64x4096x4096`, `256x4096x4096`, and `4096^3`.
- One linear non-coopmat fallback failed under the current fp16 tolerances:
  `cm_fp16_LLM_FFN_down_1tok` routes to
  `linear_vec_tile_row_1_buffer_texture2d_half` because `M=1`, and one sampled
  output had `computed=7.465`, `reference=8.602`, `diff=1.138`,
  `abs_tolerance=1.0`, `rel_tolerance=0.1`. This is not a coopmat failure, but
  it is a correctness caveat for the existing buffer `M=1` fallback path.

## Linear Results

Latest default linear rerun after the stricter tile guard:

| Shape | M | K | N | Baseline kernel | Baseline us | Buffer path kernel | Buffer us | Speedup | GFLOP/s | Correctness |
| --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | --- |
| BERT_FFN_up | 256 | 768 | 3072 | `linear_vec_texture3d_texture2d_half` | 2474.096 | `linear_coopmat_half` | 641.904 | 3.854x | 1881.838 | PASSED, CPU ref skipped |
| BERT_FFN_down | 256 | 3072 | 768 | `linear_vec_texture3d_texture2d_half` | 1743.452 | `linear_coopmat_half` | 472.960 | 3.686x | 2554.042 | PASSED, CPU ref skipped |
| BERT_QKV | 128 | 768 | 768 | `linear_vec_texture3d_texture2d_half` | 468.264 | `linear_coopmat_half` | 110.856 | 4.224x | 1362.082 | PASSED, CPU ref |
| LLM_QKV_1tok | 1 | 4096 | 4096 | `linear_vec_tile_row_1_texture3d_texture2d_half` | 1123.936 | `linear_vec_tile_row_1_buffer_texture2d_half` | 1328.152 | 0.846x | 25.264 | PASSED, CPU ref skipped |
| LLM_FFN_up_1tok | 1 | 4096 | 11008 | `linear_vec_tile_row_1_texture3d_texture2d_half` | 1084.976 | `linear_vec_tile_row_1_buffer_texture2d_half` | 1087.388 | 0.998x | 82.930 | PASSED, CPU ref skipped |
| LLM_FFN_down_1tok | 1 | 11008 | 4096 | `linear_vec_tile_row_1_texture3d_texture2d_half` | 1475.100 | `linear_vec_tile_row_1_buffer_texture2d_half` | 1601.104 | 0.921x | 56.322 | PASSED, CPU ref skipped |
| LLM_QKV_32tok | 32 | 4096 | 4096 | `linear_vec_texture3d_texture2d_half` | 1304.544 | `linear_vec_buffer_texture2d_half` | 1269.652 | 1.027x | 845.698 | PASSED, CPU ref skipped |
| LLM_FFN_up_32tok | 32 | 4096 | 11008 | `linear_vec_texture3d_texture2d_half` | 2507.796 | `linear_vec_buffer_texture2d_half` | 1966.848 | 1.275x | 1467.160 | PASSED, CPU ref skipped |
| sq_1024 | 256 | 1024 | 1024 | `linear_vec_texture3d_texture2d_half` | 1251.672 | `linear_coopmat_half` | 372.740 | 3.357x | 1440.336 | PASSED, CPU ref |
| sq_4096 | 256 | 4096 | 4096 | `linear_vec_texture3d_texture2d_half` | 7930.007 | `linear_coopmat_half` | 1831.148 | 4.331x | 4691.010 | PASSED, CPU ref skipped |
| sq_4096_cube | 4096 | 4096 | 4096 | `linear_vec_texture3d_texture2d_half` | 100017.633 | `linear_coopmat_half` | 30104.926 | 3.322x | 4565.331 | PASSED, CPU ref skipped |

Raw CSVs:

- `yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/linear_coopmat_bench_tile_guard_rerun_summary.csv`
- `yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/linear_coopmat_bench_accum_fp16_summary.csv`

## Matmul Results

| Shape | M | K | N | Baseline kernel | Baseline us | Coopmat kernel | Coopmat us | Speedup | GFLOP/s | Correctness |
| --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | --- |
| BERT_FFN_up | 256 | 768 | 3072 | `matmul_vec_texture3d_half` | 3207.940 | `matmul_coopmat_half` | 589.460 | 5.442x | 2049.264 | PASSED, CPU ref skipped |
| BERT_FFN_down | 256 | 3072 | 768 | `matmul_vec_texture3d_half` | 2771.416 | `matmul_coopmat_half` | 558.064 | 4.966x | 2164.554 | PASSED, CPU ref skipped |
| BERT_QKV | 128 | 768 | 768 | `matmul_vec_texture3d_half` | 513.396 | `matmul_coopmat_half` | 114.652 | 4.478x | 1316.985 | PASSED, CPU ref |
| LLM_QKV_64tok | 64 | 4096 | 4096 | `matmul_vec_texture3d_half` | 5514.904 | `matmul_coopmat_half` | 556.788 | 9.904x | 3856.914 | PASSED, CPU ref skipped |
| sq_1024 | 256 | 1024 | 1024 | `matmul_vec_texture3d_half` | 2123.440 | `matmul_coopmat_half` | 343.612 | 6.180x | 1562.434 | PASSED, CPU ref |
| sq_4096 | 256 | 4096 | 4096 | `matmul_vec_texture3d_half` | 11027.916 | `matmul_coopmat_half` | 1829.216 | 6.029x | 4695.965 | PASSED, CPU ref skipped |
| sq_4096_cube | 4096 | 4096 | 4096 | `matmul_vec_texture3d_half` | 151358.359 | `matmul_coopmat_half` | 29838.912 | 5.073x | 4606.031 | PASSED, CPU ref skipped |

Raw CSVs:

- `yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/matmul_coopmat_bench_model_shapes_summary.csv`
- `yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/matmul_coopmat_bench_accum_fp16_summary.csv`

## Accumulator Sweep

Default `linear_coopmat_half` and `matmul_coopmat_half` use fp32 cooperative
accumulators and store fp16 outputs. The `*_accum_fp16` variants use fp16
cooperative accumulators.

Result: fp32 accumulation is the recommended default. fp16 accumulation did not
produce a consistent speedup on RADV Phoenix:

- Linear: fp32 won 5 of 6 coopmat-measured shapes; fp16 won only
  `sq_4096_cube` by 0.24%.
- Matmul: fp32 won 6 of 7 shapes; fp16 won only `BERT_FFN_down` by 0.34%.

## Subgroup Sweep

The runtime now supports a benchmark-only `VK_COOPMAT_REQUIRED_SUBGROUP_SIZE`
hook for shader names containing `coopmat`. Both required subgroup sizes 64 and
32 built and ran successfully on RADV Phoenix.

| Shape | Linear sg64 us | Linear sg32 us | Linear winner | Matmul sg64 us | Matmul sg32 us | Matmul winner |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| BERT_FFN_up | 600.460 | 589.084 | sg32 | 560.728 | 578.060 | sg64 |
| BERT_FFN_down | 453.488 | 454.252 | sg64 | 557.660 | 572.016 | sg64 |
| BERT_QKV | 110.872 | 111.116 | sg64 | 116.300 | 115.688 | sg32 |
| LLM_QKV_64tok | n/a | n/a | n/a | 601.648 | 552.380 | sg32 |
| sq_1024 | 373.368 | 374.096 | sg64 | 345.620 | 359.464 | sg64 |
| sq_4096 | 1869.704 | 1827.164 | sg32 | 1783.756 | 1794.736 | sg64 |
| sq_4096_cube | 30018.598 | 30033.213 | sg64 | 29654.797 | 29629.584 | sg32 |

Conclusion: subgroup 32 is not a global replacement for subgroup 64. Keep the
default at 64 unless a per-shape/kernel tuning table is used; subgroup 32 looks
interesting for matmul `M=64,K=N=4096` and linear `sq_4096`.

Raw CSVs:

- `yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/linear_coopmat_bench_subgroup_summary.csv`
- `yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/matmul_coopmat_bench_subgroup_summary.csv`

## Macro Tile Sweep

The hardware WMMA tile remained `16x16x16`; this sweep changed the software
workgroup output tile. All explicit macro-tile logs reported cooperative matrix
support and no failed correctness checks. The default unset path is still the
original `64x64x32` shader.

Best explicit macro tile per measured coopmat shape:

| Shape | Linear best explicit tile | Linear us | Linear GFLOP/s | Matmul best explicit tile | Matmul us | Matmul GFLOP/s |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| BERT_FFN_up | `32x64` | 686.408 | 1759.827 | `32x64` | 548.096 | 2203.919 |
| BERT_FFN_down | `64x32` | 536.896 | 2249.895 | `32x32` | 568.620 | 2124.371 |
| BERT_QKV | `32x64` | 127.824 | 1181.272 | `32x64` | 126.548 | 1193.183 |
| LLM_QKV_32tok | `32x64` | 426.840 | 2515.560 | n/a | n/a | n/a |
| LLM_FFN_up_32tok | `32x64` | 1117.648 | 2581.923 | n/a | n/a | n/a |
| LLM_QKV_64tok | n/a | n/a | n/a | `64x32` | 648.680 | 3310.544 |
| sq_1024 | `32x64` | 366.124 | 1466.364 | `32x64` | 306.836 | 1749.700 |
| sq_4096 | `64x32` | 1951.392 | 4401.952 | `64x32` | 2004.564 | 4285.189 |
| sq_4096_cube | `64x32` | 30728.926 | 4472.625 | `64x32` | 33387.445 | 4116.486 |

Compared with the default `64x64x32` path, the explicit smaller tiles are not a
global replacement. `32x64` helped matmul `BERT_FFN_up` and `sq_1024`, and
enabled linear `M=32` coopmat coverage. The default `64x64x32` remained faster
for several larger shapes, including linear `sq_4096` and matmul
`sq_4096_cube`.

Raw CSVs:

- `yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/linear_coopmat_bench_macro_tile_summary.csv`
- `yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/matmul_coopmat_bench_macro_tile_summary.csv`
- `yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/macro_tile_best_summary.csv`

## K-Step Sweep

K-step is the K dimension chunk consumed by one outer loop of a workgroup. The
hardware WMMA tile stayed `16x16x16`; this sweep changed the software `TILE_K`
from the default 32 to explicit 16 and 64 variants.

All K-step logs reported cooperative matrix support and no failed correctness
checks. Only coopmat-routed rows are included below; fallback `linear_vec` rows
for `M < 64` are excluded from the K-step decision.

| Shape | Linear best K | Linear us | Linear GFLOP/s | Matmul best K | Matmul us | Matmul GFLOP/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BERT_FFN_up | 32 | 641.904 | 1881.838 | 64 | 588.820 | 2051.492 |
| BERT_FFN_down | 32 | 472.960 | 2554.042 | 32 | 558.064 | 2164.554 |
| BERT_QKV | 32 | 110.856 | 1362.082 | 32 | 114.652 | 1316.985 |
| LLM_QKV_64tok | n/a | n/a | n/a | 32 | 556.788 | 3856.914 |
| sq_1024 | 64 | 369.648 | 1452.384 | 32 | 343.612 | 1562.434 |
| sq_4096 | 32 | 1831.148 | 4691.010 | 32 | 1829.216 | 4695.965 |
| sq_4096_cube | 32 | 30104.926 | 4565.331 | 32 | 29838.912 | 4606.031 |

Conclusion: keep `TILE_K=32` as the default. `TILE_K=16` was consistently
slower, likely because it doubles the number of outer-loop iterations and
shared-memory barriers. `TILE_K=64` reduced loop count but increased shared
memory footprint and only produced marginal wins on linear `sq_1024` and matmul
`BERT_FFN_up`; those gains are too small to justify a global switch.

Raw CSVs:

- `yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/coopmat_bench_k_step_summary.csv`
- `yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/coopmat_bench_k_step_with_default_summary.csv`
- `yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/coopmat_bench_k_step_comparison.csv`
- `yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/coopmat_bench_k_step_best_summary.csv`

## Texture3D Linear Coopmat

Implemented a separate experimental shader,
`linear_coopmat_texture3d_buffer`, selected only by `VK_COOPMAT_TEXTURE=1`.
The path uses texture3D input/output, buffer packed weights, fp16 inputs, fp32
accumulation, no bias, and the default `64x64x32` software tile.

Because cooperative matrix store cannot write an image directly, the shader
stores the result tile to shared memory first, then normal invocations unpack
that shared tile into texture3D `imageStore` calls.

| Shape | Stephen texture us | Texture coopmat us | Speedup vs Stephen texture | GFLOP/s | Correctness |
| --- | ---: | ---: | ---: | ---: | --- |
| BERT_FFN_up | 2474.096 | 1434.852 | 1.724x | 841.871 | PASSED, CPU ref skipped |
| BERT_FFN_down | 1743.452 | 1473.180 | 1.183x | 819.967 | PASSED, CPU ref skipped |
| BERT_QKV | 468.264 | 200.100 | 2.340x | 754.597 | PASSED, CPU ref |
| sq_1024 | 1251.672 | 673.880 | 1.857x | 796.686 | PASSED, CPU ref |
| sq_4096 | 7930.007 | 5399.820 | 1.469x | 1590.782 | PASSED, CPU ref skipped |
| sq_4096_cube | 100017.633 | 46451.664 | 2.153x | 2958.752 | PASSED, CPU ref skipped |

Conclusion: texture3D coopmat is viable and beats Stephen's texture shader for
eligible full-tile linear shapes, but it remains slower than the buffer coopmat
path. The likely cost is the required coopMatStore-to-shared plus imageStore
staging. This is still useful for avoiding broad buffer-storage changes in
future E2E experiments.

Raw CSV:

- `yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/linear_coopmat_bench_texture_summary.csv`

## Fastest Config Table

| Shape | Fastest measured linear config | Fastest measured matmul config | Dispatch note |
| --- | --- | --- | --- |
| BERT_FFN_up | buffer coopmat, fp32 accum, 64x64x32 | buffer coopmat, fp32 accum, 64x64x32 | Use coopmat when buffer path is already selected |
| BERT_FFN_down | buffer coopmat, fp32 accum, 64x64x32 | buffer coopmat, fp16 accum by 0.34% | Prefer fp32 accum due noise/correctness margin |
| BERT_QKV | buffer coopmat, fp32 accum, 64x64x32 | buffer coopmat, fp32 accum, 64x64x32 | Use coopmat |
| LLM_QKV_1tok | Stephen texture `linear_vec` | not measured in matmul bench | Keep fallback |
| LLM_FFN_up_1tok | buffer `linear_vec` in this microbench | not measured in matmul bench | Not WMMA; do not infer E2E storage policy |
| LLM_FFN_down_1tok | Stephen texture `linear_vec` | not measured in matmul bench | Keep fallback |
| LLM_QKV_32tok | explicit macro tile `32x64` | not measured in matmul bench | Macro tile can cover M=32; needs production gate/tuning |
| LLM_FFN_up_32tok | explicit macro tile `32x64` | not measured in matmul bench | Macro tile can cover M=32; storage needs E2E study |
| LLM_QKV_64tok | not in linear bench | buffer coopmat, fp32 accum, 64x64x32, subgroup 32 | Use coopmat for matmul-shaped M=64 |
| sq_1024 | buffer coopmat, fp32 accum, 64x64x32 | buffer coopmat, fp32 accum, 64x64x32 | Use coopmat |
| sq_4096 | buffer coopmat, fp32 accum, 64x64x32 | buffer coopmat, fp32 accum, 64x64x32 | Use coopmat |
| sq_4096_cube | buffer coopmat, fp16 accum by 0.24% | buffer coopmat, fp32 accum, 64x64x32 | Prefer fp32 accum due noise/correctness margin |

## Plan Fulfillment

This fp16 RDNA3 kernel-sweep phase is considered fulfilled. The benchmark
coverage now includes linear and matmul buffer coopmat paths, fp16 vs fp32
accumulation, subgroup 32 vs 64, macro tiles, K-step variants, linear
texture3D coopmat, and sampled large-shape correctness.

Non-blocking future work:

- Texture matmul coopmat and texture-packed weight variants were intentionally
  left for a later storage-focused phase.
- Explicit K-step variants were measured for `TILE_K=16` and `TILE_K=64`; no
  separate instruction-level unroll variants were implemented because `TILE_K=32`
  remained the practical default.
- Large-shape correctness is now sampled instead of skipped. The routed
  `linear_coopmat_half` and `matmul_coopmat_half` large cases passed; the one
  recorded failure is the existing non-coopmat `M=1` buffer fallback path.

## Recommended Dispatch Gate

For this phase, keep Stephen's shaders as the fallback and only enable fp16
coopmat when all of the following are true:

- cooperative matrix is supported;
- output storage is buffer;
- dtype is fp16;
- op is 2D linear or non-constant 2D matmul;
- `M % 64 == 0`, `N % 64 == 0`, and `K % 32 == 0`;
- use fp32 accumulation by default.
- use subgroup 64 by default unless a per-shape tuning table selects subgroup
  32.
- keep `64x64x32` as the production default macro tile for now; consider
  `32x64` only behind a tuning table for `M=32` linear or select matmul shapes.
- keep `TILE_K=32` as the production default K-step.
- keep texture3D coopmat behind `VK_COOPMAT_TEXTURE=1` until E2E storage
  behavior and the image-store staging cost are studied.

Do not enable coopmat for decode-style `M < 64`, texture-storage calls,
non-multiple shapes, or real LLaMA E2E until texture/storage behavior is studied
in the next phase.
