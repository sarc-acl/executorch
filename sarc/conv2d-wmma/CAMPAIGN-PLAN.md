# WMMA for convolution in the ExecuTorch Vulkan backend: campaign plan

Date: 2026-11-19. Base: upstream ExecuTorch `main` at `2efa9d0316`; work branch `topic/conv2d-wmma`
(final commit of the last change: `78226422bd`).

## Goal

Make convolution in the ExecuTorch Vulkan backend faster with cooperative-matrix (WMMA) kernels on the GPUs that
expose `VK_KHR_cooperative_matrix`, and show it end to end on the models ExecuTorch itself tests.

## Where we are

Kernel level, FP16 1x1 convolution, ratio of the tiled shader's GPU time to the WMMA variant's:

| Device | MobileNetV2 (49,160,960) / (196,64,384) / (196,96,576) | State |
|---|---|---|
| Arc B580 | 3.15 / 1.90 / 2.21 | three changes done; reproduced independently |
| Arc Pro B70 | 2.51 / 1.52 / 1.88 | second change; fast pass not run there |
| Radeon 780M | 0.93 / 0.89 / 0.90 | no gain from any technique so far |
| RTX 4070 Ti SUPER | no verdict | timing sessions rejected for noise |
| Pixel 7a, Galaxy S24+ | not applicable | no cooperative-matrix extension |

What produced the gain on Xe2 (each measured, generated code checked):

1. No bounds checks in the K loop (reads past the last output channel clamped): 21 conditionals per K step down to 2.
2. Small tile with more threads (32x64 at 128 threads): load messages per K step 52 down to 28, more work groups.
3. Split-K (4 slices in parallel work groups, merge dispatch): shorter serial chain, fuller GPU.

Measured and rejected: direct weight loads without staging (0.89x on B580, 0.94-0.99x on 4070 Ti), 16-byte weight
loads, storage-image input, epilogue skip (fp16), column-major weights on the 780M.

## What the model survey says (10 models x 3 precisions x 6 devices, 174 of 180 ran)

Source: `.artifacts/2026-11-19-vulkan-model-survey/REPORT.md`.

- FP16 on the four desktops, summed over the models: convolutions a WMMA GEMM can apply to hold 83-91% of GPU shader
  time. Of that: 1x1 pointwise 20-26%, direct sliding-window (mostly 3x3) 41-48%, im2col + GEMM 24-29%.
- Models where candidates hold over 90%: resnet18, resnet50, ic3, ic4, dl3, w2l. edsr 89%, densenet161 75%,
  mv2 69%, mv3 23%.
- So the 1x1 kernel covers about a quarter of the opportunity. The rest is k x k convolution.

Kernel families by occurrence (FP16, 693 convolution instances; share of B580 convolution time):

| Family | Instances | Models | Time share | GEMM-equivalent K | Route today |
|---|---:|---:|---:|---|---|
| 1x1 | 331 | 8 | 23.5% | 3-2160 | pointwise tiled |
| 3x3 stride 1 | 178 | 7 | 37.3% | 27-9216 | direct 149, im2col+GEMM 28, Winograd 1 |
| 1x7 and 7x1 | 72 | 2 | 9.5% | 448-1792 | im2col+GEMM |
| 3x3 stride 2 | 21 | 7 | 3.5% | 27-4608 | im2col+GEMM 15, direct 6 |
| depthwise 3x3, 5x5 | 28 | 2 | 0.2% | 9, 25 | depthwise (not a GEMM) |
| 1x3 and 3x1 | 26 | 2 | 3.4% | 1152-1536 | im2col+GEMM |
| 3x3 dilated | 11 | 1 | 18.0% | 2304-18432 | direct (im2col refuses dilation) |
| 1D (k = 1, 7, 32, 48, 250) | 12 | 1 | 3.8% | 250-12000 | 1D |
| 7x7 stride 2 stem, 5x5 | 7 | 5 | 0.6% | 147, 1200 | direct |

Of the candidate instances: K is a multiple of 16 for 97%, N a multiple of 16 for 97% and of 32 for 82%, stride 1
for 96%. All run with texture input and output. INT8 (q8ta) has the same configurations; there the k x k
convolutions already route through im2col + GEMM (177 of 178 3x3).

## Plan

Each step is its own OpenSpec change, B580 first, with the fast-pass discipline: one option against a named parent,
generated code checked before timing, A/A noise band, five runs, one narrow review. Estimates are wall-clock for
the agent run.

### Step 1. WMMA for the GEMM stage of im2col (next; about half a day)

Upstream already lowers k x k convolution to `conv2d_im2col` + `conv2d_gemm` when Cout >= 128. The GEMM stage is a
1x1 convolution with K = Cin x kh x kw, so the existing kernel applies.

- 1a (hours, no new shader): time the existing `_alc` variants on the survey's GEMM-equivalent shapes as 1x1 cases:
  (49,512,4608), (196,256,2304), (784,128,1152), (289,224,1344), (25,384,4032). This says what the stage can gain.
- 1b: a WMMA route for `conv2d_gemm` that reads the im2col output, with split-K sized from K steps (72-288 here).
- Report stage time and whole-convolution time (im2col + GEMM) against today's route.
- Covers resnet18, resnet50, ic3, ic4 (3x3, 1x7, 7x1, 1x3, 3x1) and 1D for w2l.
- Exit: at least 2x on the whole convolution for the resnet 3x3 shapes on B580.

### Step 2. Dilated 3x3 through im2col + WMMA GEMM (a day)

`should_use_conv2d_im2col` refuses dilation, so dl3's dilated layers run direct at 2-6 ms per dispatch on B580 and
hold 51-71% of dl3. Extend the im2col gather to dilation and route these through step 1's GEMM.

### Step 3. Direct k x k WMMA convolution for large planes (two to three days)

edsr's 3x3 64 -> 64 at 224x224 (33 instances, 75% of edsr) and similar shapes where im2col would materialize a huge
matrix. Gather inside the kernel (kernel-tap loops, as LiteRT#11345 does) with our synchronization and the three
Xe2 techniques. Compare with im2col + WMMA on the same shapes before choosing.

### Step 4. Routing and end-to-end (two days)

A selection rule from the measured shapes (per family, per GPU family), behind a switch, then ETDump before and
after on resnet50, resnet18, edsr, dl3, ic3 and mv2, FP16. This is the first number that is not kernel level.

### Step 5. Other GPU families

- AMD (780M, and the M51 through the separate handoff): the Xe2 techniques did not help the 780M. Start from
  profiling (generated code, phase timing), not from porting. fp32 accumulation is the faster form on RDNA3.
- NVIDIA: find the cause of the 4070 Ti's bimodal timing first.
- B70: rerun the fast pass (first card only).

### Step 6. INT8 (q8ta)

Second-largest shader family in the survey. Needs the int8 cooperative-matrix shapes (Xe2: 8x16x32). After FP16
k x k is routed.

### Step 7. Upstreaming hygiene

The four Vulkan usage points (memory-model enabling, explicit full subgroups, `VK_NV_cooperative_matrix2`, instance
version), BUCK targets, and the remaining 1x1 refinements, as the PR is prepared.

## Not planned

Depthwise convolution (not a dense GEMM), M = 1 squeeze-excitation convolutions (a GEMV), phones without the
extension.
