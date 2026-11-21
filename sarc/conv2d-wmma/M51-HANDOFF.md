# Conv2D WMMA on Samsung M51 (Xclipse): handoff

Date: 2026-11-19. For the agent that continues this work on the internal device. Everything here was done on
other GPUs; nothing has run on the M51.

## What you are picking up

A cooperative-matrix (WMMA) FP16 1x1 convolution kernel for the ExecuTorch Vulkan backend, opt-in through
`benchmark_v2` routes, with variants for Intel Xe2, AMD and NVIDIA, and three techniques that raised it on Xe2
from 1.1-1.25x to 1.9-3.15x of the tiled kernel on MobileNet shapes. The task on the M51 is to find out what it
does there and to make it fast there.

- Branch: `topic/conv2d-wmma`, forked from upstream ExecuTorch `main` at `2efa9d0316` (not from `dev/1.5`),
  head `78226422bd`.
- Kernel: `backends/vulkan/runtime/graph/ops/glsl/convolution/conv2d/conv2d_pw_coopmat_body.glslh`,
  `conv2d_pw_coopmat.glsl`, `conv2d_pw_coopmat.yaml`, split-K merge `conv2d_pw_coopmat_sk_merge.glsl`.
- Dispatch and eligibility: `backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/Conv2dPW.cpp`, `Conv2d.h`.
- Benchmark: `backends/vulkan/test/ops/benchmark_v2/` (`etvk_ops_bench`), routes `pw_coopmat` (option `variant`)
  and `pw_tiled_4x8`, the `device_coopmat_properties` check, cases under
  `operators/convolution/conv2d/cases/pw_coopmat_*.json`.
- Requirements: `SPEC.md` beside this file (28 requirements).
- Survey tools (model export, run, ETDump aggregation): `survey-tools/` beside this file. They were written for the desktop fleet; the device table (`devices.sh`) is not included, write your own for the M51.

## Facts you should not have to rediscover

| Topic | Fact |
|---|---|
| Variant naming | `conv2d_pw_coopmat_<family>_t<M>x<N>_<f16|f32>[_g<grid>][_al|_alc][_sk<slices>]_half` |
| Xe2 | MMA 8x16x16, subgroup 16. Best: 32x64 tile at 128 threads, clamped unchecked reads, split-K 4. |
| AMD 780M | MMA 16x16x16, subgroup forced to 32 (default 64). 0.86-0.93x of tiled on MobileNet shapes: no gain yet. |
| NVIDIA | MMA 16x16x16, subgroup 32. No verdict: timings are bimodal on the 4070 Ti. |
| Galaxy S24+ (Xclipse 940), Pixel 7a | no `VK_KHR_cooperative_matrix` at all. Check the M51 before anything else. |
| M51 linear kernel | the shipped linear row is `t128x128k16g22s32f32xp`: subgroup 32, fp32 accumulation, one-pass texture3d drain. Start the convolution variants from those parameters. |
| fp16 accumulation | on RDNA3 it is slower than fp32 accumulation (matrix roof 10.96 against 14.77 TFLOP/s on the 780M). |
| Ordering | `barrier()` alone did not order shared-memory stores before `coopMatLoad` on one driver; the kernel uses `memoryBarrierShared(); barrier();`. Keep it. |
| Rejected on other GPUs | direct weight loads without staging, 16-byte weight loads, storage-image input, epilogue skip (fp16), column-major weight staging. Do not start there. |
| Profiling safety | never capture RGP or SQTT on an AMD integrated GPU: it has hung two hosts. Generated code and timing only, unless the device owner says otherwise. |
| Precision | fp16 accumulation has the tiled kernel's precision limits (documented cases in `pw_coopmat_fp16_accumulation_limits.json`); fp32 variants keep fp32 through bias and epilogue. |
| Validation layers | the runtime needs a Vulkan 1.3 instance and `maintenance4` for these shaders (SPIR-V 1.6, `LocalSizeId`); with that the layers report nothing on four desktops. |

## Steps

1. **Capability record.** Build `etvk_ops_bench` for the device (`build.sh --android` with the NDK, or the device's
   own toolchain) and run `device_coopmat_properties.json`. Record every enumerated shape, component types,
   subgroup size range, required-size support, shared-memory limit, driver version. If there is no fp16 shape,
   stop and report.
2. **Workload baseline.** Export the 10 ExecuTorch Vulkan CI models (mv2, mv3, resnet18, resnet50, ic3, ic4, edsr,
   dl3, w2l, densenet161) in FP32, FP16 and INT8 as the survey did, run each with ETDump (12 executions, first
   discarded), and aggregate with the survey's `aggregate.py`. This gives the M51's own ranking of shaders and
   shapes; do not assume the desktop ranking.
3. **Existing variants as they are.** Generate the collections (`cases/gen_pw_coopmat_cases.py`; the case files
   are not in the repository), then run `pw_coopmat_correctness.json` (`--validation_only`); the AMD-named
   variants are the ones that match a 16x16x16 device. Then time `pw_coopmat_mobilenet.json` and
   `pw_coopmat_resnet50.json` with routes `auto`, `pw_tiled_4x8` and every admitted variant.
4. **An M51 family, if needed.** If the enumerated shape, subgroup size or accumulator differs from the AMD
   variants, add a variant block set for it (one yaml block per tile; nothing else changes).
5. **The three techniques.** They exist for Xe2 only (`_alc`, `_g4x2`, `_sk`). Add them to the M51 family one at a
   time, each against its named parent, and check the generated code where the driver exposes it.
6. **If nothing helps** (as on the 780M): profile before porting more. Phase timing inside the kernel with the
   shader clock, and the instruction counts per K step, to see whether fetch, staging, synchronization or the
   multiply holds the time on this GPU.
7. **Report.** Per shape: tiled time, best variant, ratio, noise band; per technique: gain against its parent; what
   the workload baseline says is worth doing next on this device (the desktop survey says k x k convolution, not
   1x1, holds most of the time).

## Measurement rules (short form of the spec)

- Correctness before timing; a variant that fails is not timed.
- An A/A comparison of the parent first; its spread, not below 2%, is the noise band; differences inside it are
  "no change". Five passing runs, arms interleaved, GPU time over all dispatches of a route.
- Record device, driver, commit and binary checksums with every result.
- Decide rules before data when a decision hangs on a measurement.
- A result below 2x is a result. Report it.

## What to bring back

The capability record, the workload baseline CSVs and report, the per-shape table, and any new variant blocks as
commits on a branch `topic/m51-conv2d-wmma` (forked from `topic/conv2d-wmma`). The desktop work continues on
k x k convolution through im2col + GEMM; see `CAMPAIGN-PLAN.md` beside this file.
