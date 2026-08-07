# Results: AMD Radeon 780M (RADV PHOENIX)

**Device**: AMD Radeon 780M Graphics, Mesa RADV PHOENIX, driver 25.2.7 (driverVersion=104865799),
Vulkan apiVersion 1.4.318. Integrated RDNA3 GPU, native wave64, `KHR_cooperative_matrix` +
`EXT_subgroup_size_control` supported.

**Command**: `MBENCH_ROUNDS=7 MBENCH_ITERS=5 ./microbench --storage=texture`
(default `MBENCH_ROUNDS=7`; `MBENCH_ITERS` lowered from the default 20 -- see
"Gotcha" below). Raw output: [`780M-RADV_raw.txt`](780M-RADV_raw.txt).

## Gotcha: default MBENCH_ITERS=20 can trigger a GPU device-lost on this class of hardware

At the default settings, this run hit `Vulkan error -4` (`VK_ERROR_DEVICE_LOST`)
partway through the `q4gsw_coopmat` sweep. Cause: `run_timed_batch` submits
`itersPerRound` dispatches back-to-back in a single command buffer, then
`vkQueueWaitIdle`s. The `shipped(Xclipse)` q4gsw coopmat variant is
catastrophically slow on this GPU (~600us -> ~600ms/dispatch range depending on
shape, see below) purely from VGPR spilling -- at the largest shapes and the
default 20 iterations/batch, a single submission's GPU execution time exceeds
Linux's amdgpu driver hang-detection watchdog (commonly ~10s), and the kernel
resets the GPU context out from under the benchmark. The device recovers fine
for subsequent processes; only the microbench's own `VkDevice` is lost, and it
just exits. If you hit this, lower `MBENCH_ITERS` (5 was safely under the
watchdog for every shape tested here) rather than treating it as a driver bug
to work around some other way.

## Baseline: tiled kernels (no coopmat/WMMA)

`linear_dq8ca_q4gsw_tiled` (8da4w, `dotPacked4x8AccSatEXT` -> `v_dot4_i32_iu8`)
vs `linear_q4gsw_tiled` (4w, plain FMA dequant), texture storage:

**Geomean: 8da4w is 4.24x faster than 4w** (dispatch throughput only; excludes
8da4w's activation-quantization pre-pass, so this will differ from e2e ratios).

## q4gsw coopmat (fp16 WMMA) vs q4gsw tiled

| shape | tiled (us) | shipped(Xclipse) speedup | 780M-tuned speedup |
|---|---|---|---|
| 1B_qkvo | 9140.69 | 0.23x | 4.08x |
| 1B_mlp_up | 37231.02 | 0.23x | 4.28x |
| 1B_mlp_down | 37293.74 | 0.23x | 3.51x |
| 3B_qkvo | 20755.02 | 0.23x | 4.12x |
| 3B_mlp_up | 56147.08 | 0.23x | 4.32x |
| 3B_mlp_down | 56270.58 | 0.23x | 3.58x |
| 8B_qkvo | 37377.71 | 0.23x | 4.00x |
| 8B_mlp_up | 132417.09 | 0.23x | 4.26x |
| 8B_mlp_down | 133202.92 | 0.22x | 3.60x |
| 8B_mlp_down_M128 | 8187.81 | 0.25x | 3.50x |
| 8B_mlp_down_M512 | 33002.50 | 0.22x | 3.56x |
| 8B_mlp_down_M2048 | 133596.52 | 0.22x | 3.61x |

**Geomean across 12 aligned shapes: shipped(Xclipse) 0.23x, 780M-tuned 3.85x.**
(`8B_mlp_down_M32` skipped -- M=32 fails both variants' tile alignment.)

## dq8ca_q4gsw coopmat (int8 WMMA) vs dq8ca tiled

| shape | tiled (us) | shipped(Xclipse) speedup | 780M-tuned speedup | same-tile(128x64x32) speedup |
|---|---|---|---|---|
| 1B_qkvo | 2045.49 | 0.44x | 0.61x | 0.93x |
| 1B_mlp_up | 8995.22 | 0.48x | 0.67x | 1.00x |
| 1B_mlp_down | 8946.60 | 0.48x | 0.68x | 0.99x |
| 3B_qkvo | 5080.63 | 0.48x | 0.68x | 1.01x |
| 3B_mlp_up | 13661.63 | 0.49x | 0.68x | 1.01x |
| 3B_mlp_down | 13403.95 | 0.48x | 0.68x | 0.98x |
| 8B_qkvo | 8989.90 | 0.48x | 0.67x | 1.00x |
| 8B_mlp_up | 31389.58 | 0.48x | 0.67x | 0.99x |
| 8B_mlp_down | 31558.76 | 0.48x | 0.69x | 0.99x |
| 8B_mlp_down_M128 | 1968.25 | 0.46x | 0.68x | 1.00x |
| 8B_mlp_down_M512 | 7872.87 | 0.48x | 0.68x | 0.99x |
| 8B_mlp_down_M2048 | 31597.75 | 0.48x | 0.69x | 0.99x |

**Geomean across 12 aligned shapes: shipped(Xclipse) 0.48x, 780M-tuned 0.67x,
same-tile(128x64x32) 0.99x.** (`8B_mlp_down_M32` skipped on all three --
fails each variant's tile alignment.)

## Headline findings (full analysis + ISA evidence in the top-level README)

1. **Neither baseline tiled kernel uses matrix-core instructions at all** --
   q4gsw dequant-and-FMAs in plain fp, dq8ca uses the scalar
   `dotPacked4x8AccSatEXT` dot-product instruction. 8da4w's 4.24x edge over 4w
   in the baseline comes from that hardware dot-product plus deferring the
   expensive dequant/rescale to once per quant-group instead of every k-step.
2. **The shipped production coopmat defaults are tuned for a different GPU**
   (Samsung Xclipse 970) and badly mismatch this RDNA3 iGPU -- q4gsw's forced
   32-wide subgroup (native here is 64) drives severe VGPR spilling (427
   pre-schedule VGPRs vs. RDNA's 256 cap, 171 spilled), collapsing q4gsw
   coopmat to 4.5x *slower* than tiled despite genuinely dispatching
   `v_wmma_f16_16x16x16_f16`.
3. **Locally-tuned tile configs flip the result**: q4gsw coopmat goes from a
   4.5x loss to a **3.85x win**. dq8ca coopmat's own production-picked tile
   only reaches 0.67x, but retested on q4gsw's exact tile (`same-tile`)
   it reaches **0.99x** -- beating both of dq8ca's own production tiles,
   because comparing kernels on independently-tuned tiles conflates "which
   kernel" with "which tile."
4. **On matched tiles, dq8ca (int8 WMMA) is ~1.1x *faster* than q4gsw (fp16
   WMMA)** in absolute time, despite `v_wmma` instruction count being
   *identical* (32 vs 32) -- the entire delta is dq8ca's dequant/correction
   epilogue (zero-point subtraction, per-group rescale: 17x more `v_mul`, 21x
   more `v_fma`/`v_mad`) needed to walk an int32 accumulator back to a float
   result, which q4gsw's fp16 accumulator never needs.
5. **A follow-up attempt to shrink dq8ca's tile further** (64x64x32, half the
   accumulators) fully eliminated dq8ca's residual 32-VGPR spill but measured
   ~26% *slower* -- halving the tile area doubles the workgroup count, and
   each workgroup pays a largely fixed tax (barrier-synced shared-memory
   double-buffering, the correction epilogue) independent of tile size. That
   lost tile-reuse outweighed the spill fix; register pressure is not dq8ca's
   dominant remaining cost. Not included as a shipped variant -- see git log
   for the numbers if reproducing this line of investigation.
