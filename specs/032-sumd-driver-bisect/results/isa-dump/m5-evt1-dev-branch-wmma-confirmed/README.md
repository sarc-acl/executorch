# M5 EVT1: WMMA confirmed — 1B 4w/8da4w, full-stack optimization, `dev` branch

**Date**: 2026-07-17. **Device**: M5 EVT1 (serial `0000088f8e579c33`, host `sj1-dmckee-d01`).
**Driver**: production default `f14c51b6f8` (md5 `c9861e9906d03fa2c7d48b804e1a1c80`).
**Runner**: `llama_main_dev`, built from the `dev` branch (`yanwen/dev-1.3`) — **not**
`release/1.3` vanilla. Clocks: floating (not pinned) — see caveat below.

## This supersedes an earlier, incorrect finding

An earlier session in this same investigation tested `llama_main_rel1.3` (vanilla release/1.3)
and found **zero WMMA instructions** in the "optimal/buffer" combo for both 4w and 8da4w,
concluding M5 never executes real WMMA for the quantized path. That conclusion was **correct for
the binary tested but wrong as a general claim** — `release/1.3` vanilla only has the older
`linear_q4gsw_coop_*`/`linear_dq8ca_q4gsw_coop_*` shaders, which never declare
`OpCapability CooperativeMatrixKHR` at all ("coop" there means workgroup-cooperative tiling, not
hardware coopmat). The actual WMMA-coopmat port (per top-level `CLAUDE.md`: "merged from
`yanwen/wmma-coopmat-port`") lives only on the `dev` branch, under a **different shader name**
(`linear_q4gsw_coopmat_*`, `linear_dq8ca_q4gsw_coopmat_*` — note "coopmat" not "coop"). That old
data (pushed to this same NFS path) has been cleared and replaced by this directory.

## Results

| Combo | Prefill tok/s (floating clocks) | `v_wmma` instructions found | Pipelines with WMMA |
|---|---|---|---|
| 4w optimal (dev branch, buffer PTE) | **1519.3** (vs 613 on release/1.3 vanilla — 2.5x) | `v_wmma_f16_16x16x16_f16`, `v_wmma_f32_16x16x16_f16` | 6 of 25 |
| 8da4w optimal (dev branch, buffer PTE) | **1463.9** (vs 429 on release/1.3 vanilla — 3.4x) | `v_wmma_f32_16x16x16_f16`, **`v_wmma_i32_16x16x16_iu8`** | 6 of 30 |

⚠️ **Clocks were floating, not pinned**, for these runs (this investigation prioritized getting a
correct ISA/dispatch answer over a rigorous throughput number — see the workspace default in
`../../../../../.shared-context/instruction-for-ai/result-and-report/README.md` §1, which calls for
pinned clocks for any report-grade number). The *presence of real WMMA instructions* is
clock-independent and fully confirmed; the *tok/s numbers* above should be re-measured pinned
(509/2730/663, or M5's own max 980/5333/800) before citing them as throughput results.

## WMMA confirmed at three independent levels

1. **SPIR-V source** (`spirv/*.spvasm`): `OpCapability CooperativeMatrixKHR`,
   `OpTypeCooperativeMatrixKHR`, `OpCooperativeMatrixMulAddKHR` — for 8da4w, with
   `MatrixASignedComponentsKHR|MatrixBSignedComponentsKHR|...` flags (signed int8 operands).
2. **Compiled ISA** (`disasm/*.disasm`): real hardware instructions —
   - 4w: `v_wmma_f16_16x16x16_f16` / `v_wmma_f32_16x16x16_f16` (16×16×16 fp16 matrix-multiply-accumulate)
   - 8da4w: `v_wmma_i32_16x16x16_iu8` (16×16×16 **int8×int8→int32** matrix-multiply-accumulate) +
     `v_wmma_f32_16x16x16_f16` (likely for a bias-add/epilogue or mixed-precision stage)
3. **Throughput**: 2.5-3.4x faster than the same combo run on `release/1.3` vanilla (which never
   attempts coopmat at the source level) — directionally consistent with genuine hardware
   matrix-unit acceleration, though see the floating-clocks caveat above before citing the exact
   ratio.

## Files

- `4w-optimal/llama_main_dev/PipelineCs_0x*.{elf,pipe}` — raw on-device pipeline dump (25 pipelines)
- `4w-optimal/disasm/PipelineCs_0x*.disasm` — extracted ISA text
- `4w-optimal/spirv/linear_q4gsw_coopmat_buffer_buffer_half.{spv,glsl,spvasm}` — SPIR-V binary,
  GLSL source, and human-readable SPIR-V disassembly for the actual dispatched kernel
- `8da4w-optimal/` — same structure (30 pipelines) for
  `linear_dq8ca_q4gsw_coopmat_buffer_buffer_half`

Also pushed to NFS: `/sarc-c/gpusw/users/yanwen.xu/artifacts/7-17-isa-dump/` (this directory's
contents, replacing the earlier incorrect release/1.3 data at that path).
