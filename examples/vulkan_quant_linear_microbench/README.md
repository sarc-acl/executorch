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

## Results

Per-device results (raw output + a findings writeup) live under `results/`.
See [`results/780M-RADV.md`](results/780M-RADV.md) for the reference run this
tool's coopmat variants and README analysis below were validated against --
useful as a sanity baseline before trusting numbers from a new device. If you
run this on different hardware, consider adding a `results/<device>.md` in
the same format.

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

**If `./microbench` exits with `Vulkan error -4` (`VK_ERROR_DEVICE_LOST`)**:
`run_timed_batch` submits `MBENCH_ITERS` dispatches back-to-back in one
command buffer before waiting on the result. A pathologically slow shader
config (e.g. a badly-mismatched coopmat tile -- see "coopmat (WMMA)" below)
at the default 20 iterations/batch can push a single submission's GPU
execution time past the OS/driver's hang-detection watchdog (commonly ~10s
on Linux amdgpu), which resets the GPU context out from under the
benchmark. The GPU itself recovers fine for other processes; only this
process's `VkDevice` is lost. Lower `MBENCH_ITERS` (5 is a safe starting
point) rather than treating it as a driver bug.

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

## coopmat (WMMA)

`--storage=texture` (or `both`) additionally runs each baseline's real
KHR_cooperative_matrix / WMMA counterpart against it, if the device reports
`VK_KHR_cooperative_matrix` + `VK_EXT_subgroup_size_control` (otherwise both
sections are skipped with a note on stderr):

- `linear_q4gsw_coopmat` (`linear_q4gsw_coopmat_buffer_texture2d_half`) vs
  q4gsw tiled -- fp16 x fp16 -> fp16 MMA, `v_wmma_f16_16x16x16_f16` on RDNA.
- `linear_dq8ca_q4gsw_coopmat`
  (`linear_dq8ca_q4gsw_coopmat_buffer_texture2d_half`) vs dq8ca tiled --
  int8 x int8 -> int32 MMA, `v_wmma_i32_16x16x16_iu8` on RDNA.

Neither baseline tiled kernel ever uses matrix-core instructions -- q4gsw
tiled dequant-and-FMAs in fp, dq8ca tiled uses the scalar
`dotPacked4x8AccSatEXT` (-> `v_dot4_i32_iu8`) dot-product instruction, not a
matrix unit.

Both coopmat shaders force buffer storage for activation/output
(`coopMatLoad`/`coopMatStore` require linear memory, not images) --
"texture" here only means the weight tensor, matching the shipped variant
names. Tile/alignment requirements differ per kernel/variant and are NOT
checked inside the shader (misaligned shapes silently miscompute, so the
harness gates at dispatch time and prints a `skipped(MxNxK)` cell instead).

### Multiple tile variants per kernel -- shipped vs. locally-tuned vs. same-tile

Each kernel is measured against several tile configs, not just the shipped
default:

| Kernel | Variant | Tile (M x N x K) | Subgroup grid | Subgroup size |
|---|---|---|---|---|
| q4gsw coopmat | `shipped(Xclipse)` | 128x128x16 | 2x2 | forced 32 |
| q4gsw coopmat | `780M-tuned` | 128x64x32 | 2x2 | forced 32 |
| dq8ca coopmat | `shipped(Xclipse)` | 64x32x32 | 1x2 | native 64 |
| dq8ca coopmat | `780M-tuned` | 64x128x32 | 4x1 | forced 32 |
| dq8ca coopmat | `same-tile(128x64x32)` | 128x64x32 | 2x2 | forced 32 |

`shipped(Xclipse)` is `linear_q4gsw_coopmat`/`linear_dq8ca_q4gsw_coopmat`'s
current production default -- tuned via specs/036 (q4gsw) / specs/027
(dq8ca) on a Samsung Xclipse 970 (M51), a different GPU family from
whatever this microbench happens to run on. On a 780M (RDNA3, native
wave64), confirmed via
`RADV_DEBUG=shaders,shaderstats ./microbench --storage=texture 2>&1 | less`,
`shipped(Xclipse)` measures **~4.5x slower** (q4gsw) / **~2.1x slower**
(dq8ca) than tiled -- q4gsw's `REQUIRED_SUBGROUP_SIZE=32` (native is 64)
plus its 16 live accumulator matrices/thread drives 427 pre-schedule VGPRs
against RDNA's 256-VGPR cap, spilling ~171 VGPRs and collapsing occupancy to
4 subgroups/SIMD (vs. 14 for tiled); dq8ca hits zero spilling and decent
occupancy (8 vs. tiled's 10) yet is still slower, likely from the
`barrier()`-per-loop-iteration cost of coopmat's shared-memory
double-buffering, a cost the tiled kernels never pay (independent
per-thread output tile, no LDS, no cross-thread sync) -- unconfirmed
precisely without real GPU stall counters.

`780M-tuned` is `(linear_q4gsw|linear_dq8ca_q4gsw)_coopmat_tsweep`'s
`t128x64k32g22s32` / `t64x128k32g41s32` variants -- **not** something swept
in this session. They're dev-igpu branch's own specs/035 e2e sweep result,
a real correctness-gated (48/48 microbench cases pass), e2e-measured
(2048-token prefill) 3-round coordinate-descent search done specifically on
a 780M/RADV (`specs/035-dev-igpu-tile-sweep/findings.md` on
`yanwen/dev-igpu`). Notably, this branch's *current* shipped default
(128x128x16, tested above as `shipped(Xclipse)`) isn't even this branch's
own tuning -- a same-day *later* commit (specs/036) re-tuned for the
Xclipse 970 and overwrote specs/035's 780M-specific values.

With `780M-tuned`, this microbench's isolated-dispatch numbers land the same
direction as specs/035's own e2e prefill numbers: q4gsw coopmat flips from a
**4.5x loss** to a **~3.8x win** over tiled (matches the reported 1B prefill
1764->1977 tok/s), while dq8ca coopmat improves substantially (~42%,
matching the reported 1274->1800 tok/s) but still trails tiled's
already-hardware-accelerated `v_dot4_i32_iu8` path in this isolated
microbenchmark (0.67x) -- e2e ranks it #1 anyway per specs/035 (it's
competing against a different set of costs there: activation quantization
pre-pass, graph overhead, etc., none of which this harness measures).

dq8ca's story doesn't end there, though: its `780M-tuned` pick (64x128x32)
and q4gsw's `780M-tuned` pick (128x64x32) are *different* tiles, which
confounds "which kernel is faster" with "which tile is faster." Retested on
q4gsw's exact tile/subgroup config (`same-tile(128x64x32)`, since both
kernels' tsweep grids happen to include this token), dq8ca coopmat jumps to
**~1.0x -- matching tiled, and beating both of dq8ca's own production-picked
tiles**. On that same apples-to-apples tile, dq8ca (int8 WMMA) actually runs
**~1.1x faster** than q4gsw (fp16 WMMA) in absolute time, despite doing
provably more work per dispatch -- `v_wmma` instruction count is *identical*
(32 vs 32) between the two at this tile/shape, so the entire delta is in
dq8ca's dequant/correction epilogue (`v_mul` 12->208, `v_fma`/`v_mad` 4->84,
~2x more texture reads and barriers) needed to walk an int32 accumulator
back to a float result -- q4gsw's fp16 accumulator needs no such correction.
The takeaway: don't compare kernels using each one's own independently-tuned
tile; a fair kernel-vs-kernel comparison needs the same tile on both sides.

`same-tile(128x64x32)` still carries a 32-VGPR spill (`RADV_DEBUG=shaders,
shaderstats`), unlike q4gsw's spill-free `780M-tuned`. A follow-up tile
(64x64x32 -- half the output area, half the live accumulator matrices/
thread) confirmed the spill can be cleared entirely (0 VGPRs spilled, fewer
instructions, better estimated latency/occupancy in the compiler's own
stats) -- but measured **~26% *slower*** in practice (0.74x), not faster.
Halving the tile area doubles the workgroup count, and each workgroup pays
a largely tile-size-independent tax (double-buffered shared-memory
`barrier()` sync, the per-group correction epilogue) -- that lost tile-reuse
outweighed the spill fix. Register pressure is not dq8ca's dominant
remaining cost at this scale; a promising untested angle instead is
checking whether production's dynamic activation quantization is always
zero-point-symmetric, which would make the zero-point-correction terms in
that epilogue provably dead code.

To try a *different* tile from the ~30-90 correctness-probed candidates
already enumerated in each `*_coopmat_tsweep.yaml` (main tree), resolve it
the same way this file's shaders were resolved (see `gen_spv_header.py` and
the "Provenance" section above), then add a `CoopmatVariant` entry in
`main()`. Grep `git log -- specs/035-dev-igpu-tile-sweep specs/026-8da4w-subgroup32-sweep`
on `yanwen/dev-igpu` before picking a raw sweep-grid candidate blind: some
listed variants are correctness-gate failures (e.g. the raw e2e leader
`t64x64k32g21s32` crashes) kept in the yaml only as sweep history.
