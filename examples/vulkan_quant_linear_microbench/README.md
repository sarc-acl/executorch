# Quant linear (4w / 8da4w) Vulkan microbenchmark

Standalone Vulkan compute microbenchmark comparing the two BASELINE
(non-coopmat/non-WMMA) Vulkan linear kernels used by ExecuTorch's quantized
Llama path:

- `linear_q4gsw_tiled` (4-bit weight-only, "4w")
- `linear_dq8ca_q4gsw_tiled` (8-bit dynamic activation + 4-bit weight, "8da4w")

It does not link against ExecuTorch at all -- it's a single `.cpp` file using
raw Vulkan, with the two shaders' compiled SPIR-V embedded at build time. This
makes it easy to build once and copy the resulting binary (plus the two
`shaders/*.spv` blobs, or the header form baked into the binary -- see below)
to any Linux or Android device with a Vulkan driver.

## Provenance

The shaders in `shaders/*.glsl` are the byte-resolved production GLSL --
resolved via this repo's own `backends/vulkan/runtime/gen_vulkan_spv.py`
(NOT hand-transcribed), from branch `yanwen/dev-igpu` at commit
`c02c80254ab2b93d6b1bb772a6b793b011c2c43a`. The `shaders/*.spv` files are
`glslc` output from that resolved GLSL, generated with the shipped tile
defaults (`TILE_M4=1, TILE_K4=1, TILE_N8=1` -> a 4x8 output tile).

Two storage variants are included per kernel:
- `*_buffer_buffer_half.spv` -- IO tensors and weight both as SSBO
- `*_texture3d_texture2d_half.spv` -- IO as image3D/sampler3D, weight as
  isampler2D

Note: `t_weight_scales` / `t_weight_sums` / `t_bias` / `t_packed_int8_input`
and the dq8ca activation scale/zero-point tensors are ALWAYS buffer /
texture3d respectively, regardless of this flag -- that's hardcoded in the
shader source, not gated by the storage variant. See the `.glsl` files for
the exact per-binding types.

## What this measures (and doesn't)

- **Dispatch throughput only, not numerical correctness.** All buffers are
  filled with a deterministic pseudo-random pattern, not real quantized
  data.
- **The linear kernel only.** Production 8da4w also runs a separate
  activation-quantization pre-pass before this kernel; excluding it is
  intentional (shader-level comparison), but means the ratio here will
  differ from end-to-end tok/s ratios measured elsewhere in this project.
- Global/local workgroup sizing mirrors production exactly
  (`QuantizedLinear.cpp`'s `quantized_linear_global_wg_size` /
  `pick_hw_square_wg_size`), so the two kernels are dispatched identically
  save for the shader body itself.
- This was cross-checked against the project's own `test_llama_microbench
  --baseline` tool on an RX 7900 XTX: the *ratio* (4w vs 8da4w) matched
  within 1.5-8% across several shapes, though absolute per-dispatch time ran
  ~10-15% higher here (harness dispatch-loop overhead, not a correctness
  issue).

## Build

Desktop Linux (needs a C++17 compiler, `python3`, and Vulkan headers +
`libvulkan.so` -- e.g. from your distro's `vulkan-headers`/`libvulkan-dev`
package, or a Vulkan SDK install):

```bash
./build.sh
# if vulkan/vulkan.h isn't on your default include path:
VULKAN_SDK_INC=/path/to/vulkan/include ./build.sh
```

Android (aarch64), via the Android NDK:

```bash
NDK_ROOT=/path/to/android-ndk ANDROID_API=28 ./build_android.sh
adb push microbench_android /data/local/tmp/
adb shell chmod 755 /data/local/tmp/microbench_android
```

Both scripts regenerate `shaders/shader_*_spv.h` from the committed
`shaders/*.spv` files via `gen_spv_header.py`, then compile with those headers
baked in -- the resulting binary is fully self-contained (no separate `.spv`
files needed at runtime).

## Run

```bash
./microbench --storage=buffer|texture|both   # default: buffer
```

Env vars:
- `MBENCH_ROUNDS` (default 7), `MBENCH_ITERS` (default 20) -- per-shape
  interleaved (4w, 8da4w, 4w, 8da4w, ...) rounds and dispatches-per-round.
  Interleaving avoids blocked-sampling artifacts on GPUs without clock
  pinning.
- `MBENCH_VALIDATION=1` -- enable `VK_LAYER_KHRONOS_validation` if present.
- `MBENCH_DEVICE_INDEX=<n>` -- pick a specific `VkPhysicalDevice` if you have
  more than one (default: first discrete GPU, else device 0).

Output is a CSV (one row per shape) followed by a human-readable summary
table with a geomean speedup line, per storage mode.

## Shape sweep

Fixed at compile time in `microbench.cpp` (`kShapes`) -- covers Llama
1B/3B/8B q/k/v/o and mlp up/down projection shapes at prefill M=2048, plus an
M-sweep (32/128/512/2048) on the 8B down-projection shape. Edit `kShapes`
directly to add/remove shapes.

## Tile size

The shipped shaders use the production default tile
(`TILE_M4=1, TILE_K4=1, TILE_N8=1` -> 4x8 output tile, same for both
kernels). To sweep a different tile, regenerate the SPIR-V with
`gen_vulkan_spv.py --env TILE_M4=<x> TILE_K4=<y> TILE_N8=<z>` from the
`backends/vulkan/runtime/graph/ops/glsl/` shader sources, then swap the
`.spv` files here (binding layout is unaffected by tile size, only the
`#define TILE_*` macros change).
