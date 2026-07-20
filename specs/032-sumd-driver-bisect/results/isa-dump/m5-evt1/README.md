# M5 EVT1 ISA dump: 1B 4w/8da4w × baseline/WMMA-optimal

**Date**: 2026-07-17. **Device**: M5 EVT1 (serial `0000088f8e579c33`, host `sj1-dmckee-d01`).
**Driver**: production default `f14c51b6f8` (md5 `c9861e9906d03fa2c7d48b804e1a1c80`) — NOT a
self-built driver; these are the real shipped driver's compiled shaders.

## Method note: M5 needed a different dump mechanism than M41

The `Feature::LegacySettings::EnablePipelineDump` **property** mechanism documented in
`../../../../.shared-context/instruction-for-ai/result-and-report/README.md` §3c (validated on
M41) does **not** work on M5's production driver — confirmed via `strace`: the property is read
correctly and the driver even logs `Feature: Override: Success`, but zero `mkdir`/dump-file
syscalls ever occur, on both `f14c51b6f8` and a self-built `main`-HEAD driver alike. Ruled out:
property naming, `/data/vendor/gpu` permissions, SELinux mode, pipeline-cache-hit skipping
recompilation, runner-binary identity, property-value-transition timing, logd suppression,
disk-based feature-state caching, Vulkan layer/ICD-manifest interposition.

**What actually works on M5**: a classic PAL **settings file**, not a property —
`/data/vendor/gpu/amdPalSettings.cfg`:
```
EnablePipelineDump, 1
PipelineDumpDir, pipelineDump
DumpShadersSeparately, 1
```
```bash
adb shell mkdir -p /data/vendor/gpu && adb shell chmod 777 /data/vendor/gpu
adb push amdPalSettings.cfg /data/vendor/gpu/
adb shell setenforce 0   # required
# run the app; dump lands at /data/vendor/gpu/pipelineDump/<process-name>/PipelineCs_*.elf
```
Same `.AMDGPU.disasm`-section extraction procedure applies (`extract_isa_disasm.py`). This has
been folded into the main doc's §3c as an alternate mechanism.

**Crash caveat**: mid-session, running the buffer/coopmat PTE with this dump mechanism enabled
crashed the device to bootloader once (recovered via `fastboot reboot`; also corrupted
`amdPalSettings.cfg` to all-zero bytes across the reboot, requiring re-push). A retry of the exact
same combo immediately after succeeded cleanly with no recurrence — cause not fully understood,
treat as a real but non-deterministic risk of this dump mechanism on this board, not a guaranteed
repro.

## Results

| Combo | PTE storage | Prefill tok/s | Pipelines dumped |
|---|---|---|---|
| 4w baseline (T-tiled) | texture | 588.8 | 32 |
| 4w optimal (WMMA/coopmat) | buffer | 612.1 | 22 |
| 8da4w baseline (T-tiled) | texture | 427.1 | 37 |
| 8da4w optimal (WMMA/coopmat) | buffer | 429.3 | 27 |

8da4w < 4w in both storage modes — consistent with the `69e887275e` dot4_i32_i8-disablement
regression found in `specs/032-sumd-driver-bisect` (M5's shipped production driver carries this
regression too, not just the SUMD `main` bisect range).

## Key finding: no WMMA/coopmat hardware instruction anywhere, in any of the 4 dumps

Swept all 4 combos' disassembly for `v_wmma`, `v_mfma`, `v_smfmac`, `coopmat`, `matrix_mult`,
`v_dot8` — **zero matches in all four**, including both "optimal" (buffer-storage,
coopmat-API-dispatching) combos. Every quantized-linear kernel in every combo uses the same
scalar-emulation pattern (`v_mad_i32_i24`/`v_bfe_i32`/`v_ashrrev_i32`/`v_mul_i32_i24` instead of a
native dot-product or matrix instruction) — matching exactly the `dot4_i32_i8`-disabled fallback
identified in the bisect's Culprit Commit mechanism.

**Interpretation, corrected 2026-07-17 after inspecting the SPIR-V source (see `spirv/`
subdirectory)**: the quantized linear kernels ExecuTorch actually dispatches for this combo —
`linear_q4gsw_coop_buffer_buffer_half.spv` (4w) and `linear_dq8ca_q4gsw_coop_buffer_buffer_half.spv`
(8da4w) — **do not declare `OpCapability CooperativeMatrixKHR` at all**, confirmed via
`spirv-dis`. "coop" in their name means workgroup-cooperative tiling (a software design pattern),
unrelated to the Vulkan hardware cooperative-matrix extension. This is a *source-level* fact, not
a driver/feature-flag issue — no amount of enabling `Feature::M5::ShWmmaMatrixMultiply` would make
these particular shaders emit WMMA instructions, since they never ask for it. (Earlier framing of
this finding — "M5's WMMA feature flag is disabled" — was an incomplete explanation; corrected
here.) There IS a separate, genuine hardware-coopmat shader family that declares the capability
and uses `OpTypeCooperativeMatrixKHR` (`linear_coopmat_half.spv`, `matmul_coopmat_half.spv`,
copied for reference under `spirv/generic-true-coopmat-reference/`) — but those are generic
(non-quantized fp16/fp32) linear/matmul kernels, not part of the 4w/8da4w quantized dispatch path
tested here. Whether *those* kernels actually execute as real hardware WMMA on M5 is untested in
this investigation (would need a separate on-device run, e.g. via `test_coopmat_linear_bench`, if
that binary exercises the same kernel family — not yet confirmed).

The measured buffer-vs-texture speedup (612 vs 589 tok/s for 4w, ~0.5% for 8da4w) is real, but
comes from whatever else differs between the buffer-storage "coop" kernel and the texture-storage
"tiled" kernel (memory access pattern, tiling strategy, etc.) — not from hardware WMMA execution,
since the actual arithmetic is scalar-emulated in both.

## Files

- `4w-baseline/`, `4w-optimal-wmma/`, `8da4w-baseline/`, `8da4w-optimal-wmma/` — each has
  `llama_main_rel1.3/PipelineCs_0x*.{elf,pipe}` (raw dump) and `disasm/PipelineCs_0x*.disasm`
  (extracted human-readable ISA, via `.shared-context/scripts/extract_isa_disasm.py`).
- `spirv/4w-optimal-wmma/`, `spirv/8da4w-optimal-wmma/` — the actual dispatched quantized-linear
  SPIR-V (`.spv` binary, `.glsl` source, `.spvasm` human-readable disassembly via `spirv-dis`),
  pulled from `release-1.3/executorch/cmake-out-android-vk/vulkan_compute_shaders/` (build-time
  intermediate artifacts, not extracted from the on-device dump — the on-device `.pipe` files only
  *reference* an SPV filename, they don't embed the binary).
- `spirv/generic-true-coopmat-reference/` — the separate, genuinely `CooperativeMatrixKHR`-using
  generic (non-quantized) linear/matmul kernels, kept for contrast — not part of the tested
  4w/8da4w dispatch path.
