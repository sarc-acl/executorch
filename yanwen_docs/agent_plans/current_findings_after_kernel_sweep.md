# Current Findings After RDNA3 fp16 Kernel Sweep

This note consolidates the findings that should guide follow-up agents after
`yanwen_docs/agent_reports/kernel_sweep_fp16_rdna3.md`.

Phase 1, the fp16 RDNA3 WMMA kernel-sweep phase, is concluded. Follow-up work
should move to E2E storage behavior or production design rather than extending
the kernel sweep by default.

## What Is Now Known

- Buffer-backed fp16 cooperative-matrix kernels are materially faster than
  Stephen's fp16 texture shaders for eligible prefill-sized linear/matmul
  shapes on AMD Radeon 780M / RADV Phoenix.
- The hardware WMMA tile shape is fixed by this GPU at `16x16x16`; the sweep
  varied software choices around that fixed hardware tile.
- The conservative measured dispatch gate is:

```text
dtype == fp16
device supports VK_KHR_cooperative_matrix
operation is 2D linear or non-constant 2D matmul
activation/output storage is buffer
M % 64 == 0
N % 64 == 0
K % 32 == 0
fp32 cooperative accumulator
64x64x32 software macro tile
default subgroup size 64
```

- fp16 accumulation was not a consistent win. Keep fp32 accumulation as the
  default unless a later per-shape tuning table proves otherwise.
- Subgroup 32 works through the benchmark-only
  `VK_COOPMAT_REQUIRED_SUBGROUP_SIZE` hook, but it is not a global replacement
  for subgroup 64.
- Explicit smaller macro tiles, especially `32x64` and `64x32`, are useful
  tuning candidates and can cover `M=32` linear cases, but they are not a
  global replacement for the default `64x64x32` path.
- Decode-style `M=1` cases should remain on Stephen's shaders for this iGPU
  phase.
- K-step values `16`, `32`, and `64` were tested. Keep `K-step=32`.
- A linear texture3D input/output + buffer-weight coopmat prototype now works
  and beats Stephen's texture linear shader on measured eligible shapes, but it
  remains slower than buffer coopmat and is still experimental for E2E.

## Sweep Dimensions

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

Explicit parameter list:

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

Important distinction: the hardware WMMA tile is not something swept here. On
this AMD/RADV GPU, the exposed fp16 cooperative matrix hardware shape is
`16x16x16`; the sweeps are software choices around it: wave size, accumulator
type, macro tile, K blocking, and storage.

## Evidence Quality

The timing signal is strong, and routed large buffer coopmat cases now have
sampled CPU-reference evidence. `linear_coopmat_half` and
`matmul_coopmat_half` passed sampled validation for the routed large shapes,
including `4096^3`.

The remaining sampled correctness failure is not a coopmat dispatch failure:
`cm_fp16_LLM_FFN_down_1tok` routes to the existing
`linear_vec_tile_row_1_buffer_texture2d_half` fallback because `M=1` and missed
the current fp16 tolerance by one sampled element.

## Operational Lessons To Carry Forward

- `linear_coopmat_bench` must route through `aten.linear.default`; using
  `test_mm.default` with constant `mat2` bypassed the linear coopmat predicate.
- Adding a new GLSL/YAML shader requires rerunning the top-level CMake
  configure so the generated Vulkan shader registry contains the new variant.
- Explicit YAML variant names are exact. Do not append dtype suffixes to
  explicit names such as `linear_coopmat_accum_fp16`.
- The current subgroup-size control is a benchmark hook, not a production
  runtime selection design.

## Recommended Next Queue

1. Run the real LLaMA E2E storage study with the texture3D linear coopmat path
   as the main candidate and whole-graph buffer only as a regression/control
   case, not as the target architecture.
2. Investigate the non-coopmat `M=1` buffer fallback sampled correctness
   failure separately from coopmat production decisions.
3. Preserve production fallback to Stephen's shaders for unsupported storage,
   non-multiple shapes, decode, non-fp16, and devices without the required
   cooperative matrix configuration.
