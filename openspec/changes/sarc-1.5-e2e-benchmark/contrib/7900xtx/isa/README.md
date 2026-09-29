# RX 7900 XTX: WMMA instructions in the compiled cooperative-matrix kernels

This is a compile check, not a performance run. It answers one question: do the SARC 4w and 8da4w release rows,
and the igpu-roofline matrix shaders behind the matrix roofs, compile to RDNA3 WMMA instructions on the RX 7900 XTX?
They do, under both drivers.

- Date: 2026-09-28.
- Host: `host-7900xtx`, Radeon RX 7900 XTX (Navi31, gfx1100), Ubuntu 25.04, kernel 6.14.0-37-generic.
- Tree: `dev/1.5` at `157c0d03a`. The 4w and 8da4w rows came in with `ff3f34ef8`, and their shader and op sources
  have not changed since.

## Drivers and tools

| Dir | Compiler | How the ISA was obtained |
|---|---|---|
| `radv/` | RADV, Mesa 25.0.7-0ubuntu0.25.04.2 (ACO; its disassembler is LLVM 20.1) | `RADV_DEBUG=shaders,shaderstats` during a real dispatch. The file keeps only the `disasm:` section and the `*** SHADER STATS ***` block. |
| `amdvlk/` | AMDVLK 2025.Q2.1 (LLPC), the default ICD. The e2e benchmark numbers came from this driver. | `VK_KHR_pipeline_executable_properties` on the dispatched pipeline (`ISA.cs`, statistics) |
| `rga-offline/` | Radeon GPU Analyzer 2.12.0.56, `vk-spv-offline` mode (bundled amdllpc, independent of the installed driver) | `rga -s vk-spv-offline -c gfx1100 --isa <out>.isa -a <out>.csv --comp <in>.spv` |

RADV was selected with `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json` and
`MESA_SHADER_CACHE_DISABLE=true`. The device then reports itself as "Radeon RX 7900 XTX (RADV NAVI31)".

### SARC kernels (real dispatches)

Commands:

```sh
ET_VK_SARC_UNVERIFIED=1 ./test_llama_microbench --linear --scheme=4w    --storage=texture3d --regime=prefill --model=llama-3.1-8b --skip-correctness
ET_VK_SARC_UNVERIFIED=1 ./test_llama_microbench --linear --scheme=8da4w --storage=texture3d --regime=prefill --model=llama-3.1-8b --skip-correctness
ET_VK_SARC_UNVERIFIED=1 ./test_llama_microbench --linear --scheme=8da4w --storage=buffer    --regime=prefill --model=llama-3.1-8b --skip-correctness
```

- **RADV runs:** a native host build (`tools/sarc-build-native.sh`), with `RADV_DEBUG=shaders,shaderstats`.
- **AMDVLK runs:** an inspection build of the same tree (`--inspect`, the `topic/vulkan-pipeline-inspect` patch
  applied uncommitted), with `ET_VK_DUMP_PIPELINE_STATS=<dir>`. The build also wrote every pipeline of a kernel
  into one file; the stock patch keeps only the last. The same inspection build under RADV gave the same opcode
  histogram as the `RADV_DEBUG` disassembly.

Each run dispatches the row's kernel for the four Llama 3.1 8B prefill shapes (M=2048), which the microbench
reports:

| Shape | K | N |
|---|---|---|
| wq_wo | 4096 | 4096 |
| wk_wv | 4096 | 1024 |
| w1_w3 | 4096 | 14336 |
| w2 | 14336 | 4096 |

- **Specialization constants:** local size 256x1x1, `K4_per_group`=32 and `num_groups`=K/128. The other two spec
  constants, `apply_bias` and `out_N`, are optimized out of the SPIR-V.
- **Pipeline sizes:** the pipelines use subgroup size 32. The ISA uses `exec_lo`/`vcc_lo`, which confirms wave32.
- **Differences between shapes:** the K=4096 pipelines are identical, except in the 8da4w buffer variant, where
  N changes some addressing. The K=14336 pipeline differs only in the loop bound (`0x1c0` instead of `0x80`
  chunks). Every pipeline has the same WMMA counts and the same resource usage.
- **Files kept:** one file for K=4096 (wq_wo, pipeline 1 of 4) and one for K=14336 (w2, pipeline 4 of 4).

### igpu-roofline matrix shaders

These are the configs that define the confirmed roofs of campaign
`results/newdev-20260927/7900xtx/host-7900xtx-gpu-000000000300`, taken from `report/summary.json`, section
`short_run`:

| Shader | Roof | SPIR-V sha256 |
|---|---|---|
| `matrix_fp16_16x16x16_c8` | `matrix_fp16` (fp16 accumulate) | `8f91cb88…` |
| `matrix_fp16_fp32_16x16x16_c8` | `matrix_fp16_fp32` (fp32 accumulate) | `1dc857e0…` |
| `matrix_int8_16x16x16_c4` | `matrix_int8` | `5ae93f24…` |

- The same `build/shaders/*.spv` files were compiled with the campaign's pipeline inspector, `build/host/inspect`
  (sha256 `d4c4590c…`), using its `pipeline-inspection/<name>.config.json`. The inspector ran under AMDVLK and
  under RADV (with `RADV_DEBUG=shaders,shaderstats`).
- These shaders use subgroup size 64 (one wave64 per workgroup of 64) under both drivers.
- Today's AMDVLK ISA is byte-identical to the campaign's `pipeline-inspection/*.ISA.cs.txt`.

### RGA offline

- **Roofline shaders:** RGA compiled the roofline `.spv` files unchanged.
- **SARC kernels:** RGA offline mode cannot take specialization constants or a required subgroup size. The SARC
  SPIR-V was therefore specialized first, with
  `spirv-opt --set-spec-const-default-value "0:256 1:1 2:1 4:32 5:{32|112}" --freeze-spec-const`, giving K=4096
  and K=14336.
- **Wave size caveat:** RGA compiled the SARC kernels as wave64 (`s_*_b64` exec masks), whereas the dispatched
  pipelines are wave32. Use `rga-offline/` only as a cross-check of the opcode counts. It does not represent the
  shipped pipeline's registers or schedule.

## Opcode counts

`count_wmma.py */*.isa.txt` regenerates `counts.csv`. In the table below, "in loop" is the number of WMMAs inside
the body of the backward-branch loop that the WMMAs sit in.

| Kernel | Driver | v_wmma_f32_16x16x16_f16 | v_wmma_f16_16x16x16_f16 | v_wmma_i32_16x16x16_iu8 | In loop | VGPRs | SGPRs | LDS (bytes) | Scratch / spills |
|---|---|---|---|---|---|---|---|---|---|
| 4w `sarc_linear_q4gsw_coopmat_t256x128k32g24s32f32cbt_texture3d_texture2d_half` | RADV | 64 | 0 | 0 | 32 | 240 | 128 (allocated) | 61440 | 0 / 0 VGPR, 0 SGPR spilled |
| (same) | AMDVLK | 64 | 0 | 0 | 32 | 256 | 37 | 61440 | 0 scratch |
| (same) | RGA offline (wave64) | 64 | 0 | 0 | 32 | 241 | 39 | 61440 | 0 / 0 |
| 8da4w `sarc_linear_dq8ca_coopmat_zpg_t128x64k32g42s32_texture3d_texture2d_half` | RADV | 0 | 0 | 8 | 8 | 144 | 128 (allocated) | 18432 | 0 / 0 VGPR, 0 SGPR spilled |
| (same) | AMDVLK | 0 | 0 | 8 | 8 | 120 | 54 | 18432 | 0 scratch |
| (same) | RGA offline (wave64) | 0 | 0 | 8 | 8 | 106 | 62 | 18432 | 0 / 0 |
| 8da4w `sarc_linear_dq8ca_coopmat_zpg_t128x64k32g42s32_buffer_texture2d_half` | RADV | 0 | 0 | 8 | 8 | 144 | 128 (allocated) | 14336 | 0 / 0 VGPR, 0 SGPR spilled |
| (same) | AMDVLK | 0 | 0 | 8 | 8 | 119 | 54 | 14336 | 0 scratch |
| (same) | RGA offline (wave64) | 0 | 0 | 8 | 8 | 53 | 54 | 14336 | 0 / 0 |
| roofline `matrix_fp16_16x16x16_c8` | RADV | 0 | 8 | 0 | 8 | 84 | 128 (allocated) | 0 | 0 / 0 |
| (same) | AMDVLK | 0 | 8 | 0 | 8 | 47 | 16 | 0 | 0 scratch |
| (same) | RGA offline | 0 | 8 | 0 | 8 | 49 | 16 | 0 | 0 / 0 |
| roofline `matrix_fp16_fp32_16x16x16_c8` | RADV | 8 | 0 | 0 | 8 | 84 | 128 (allocated) | 0 | 0 / 0 |
| (same) | AMDVLK | 8 | 0 | 0 | 8 | 53 | 16 | 0 | 0 scratch |
| (same) | RGA offline | 8 | 0 | 0 | 8 | 54 | 16 | 0 | 0 / 0 |
| roofline `matrix_int8_16x16x16_c4` | RADV | 0 | 0 | 4 | 4 | 48 | 128 (allocated) | 0 | 0 / 0 |
| (same) | AMDVLK | 0 | 0 | 4 | 4 | 44 | 16 | 0 | 0 scratch |
| (same) | RGA offline | 0 | 0 | 4 | 4 | 44 | 16 | 0 | 0 / 0 |

Notes on the table:

- Counts are per compiled pipeline and static, as they appear in the ISA. They are identical for K=4096 and
  K=14336.
- **SGPRs:** RADV reports the allocated SGPR count (128), not the number used.
- **VGPRs:**
  - AMDVLK allocates 256 VGPRs to the 4w kernel, the wave32 maximum. Its scratch use is 0, so nothing spills to
    memory. AMDVLK reports no separate spill counters, but the ISA has no `scratch_*` and no
    `v_writelane`/`v_readlane` instructions.
  - The 2026-09-27 study measured 243 VGPRs for the 1.4 `tsweep` build of the same tile under the same driver
    (`sarc-1.5-7900xtx-4w/results/7900xtx/isa/stats-bt/t256x128k32g24/`).
- **LDS:** for the ET kernels, AMDVLK's `ldsUsageSizeInBytes` is shown. `ldsSizePerLocalWorkGroup` is 65536 for
  every pipeline.
- **Accumulator types:** the 4w kernel uses only the fp32-accumulate WMMA (`ACC_FP32`), and 8da4w uses only the
  int8 WMMA. No `v_wmma_f16_16x16x16_f16` and no bf16 WMMA appear in either SARC kernel.
- **Undecoded lines in the RADV disassembly:**
  - The `radv/` SARC files contain some `(invalid instruction)` lines: 10 in 4w, 84 in 8da4w at K=4096 and 281
    at K=14336.
  - They are true16 VOP1/VOP3 encodings with `.h` operands, for example `v_cvt_f16_f32` writing a high half
    (`7f84153f` against the decodable `7e84153e`), and `v_mov_b16`. This LLVM 20.1 disassembler build does not
    decode them.
  - None of them are VOP3P encodings (`0xCC…`), the encoding class WMMA uses, so the counts above are not
    affected.

## Do the counts match the tiles?

WMMA on RDNA3 is 16x16x16: 16x16 outputs and K=16 per instruction. Each count is per subgroup:

- **4w, t256x128k32, `SG_GRID_X`=2, `SG_GRID_Y`=4:** 8 subgroups; each subgroup tile is 64x64, so 4x4 = 16
  accumulator fragments.
  - One K=32 chunk takes 2 K-steps, so 16 x 2 = **32 WMMA per chunk per subgroup**. This matches the 32 in the
    main loop body under all three compilers.
  - The body file peels the last chunk out of the loop, which adds another 32 and gives the static total of 64.
  - Per workgroup per chunk that is 8 x 32 = (256/16)·(128/16)·(32/16) = **256**.
- **8da4w, t128x64k32, `SG_GRID_X`=4, `SG_GRID_Y`=2, MMA_K=16 (int8):** 8 subgroups; each subgroup tile is
  64x16, so 4x1 = 4 fragments.
  - Two K=16 slabs per K=32 chunk give **8 WMMA per chunk per subgroup**. This matches the 8 in the inner loop.
    The loop is not peeled, so the static total is 8.
  - Per workgroup per chunk that is 8 x 8 = (128/16)·(64/16)·(32/16) = **64**.
- **Roofline shaders:** there is one WMMA per chain per loop iteration, so `c8` gives 8 and `c4` gives 4. This
  matches the `coopmat_muladd` count in the campaign's SPIR-V ledger (8, 8, 4).

The reference figures from the Radeon 780M dumps are "256 v_wmma per 4w kernel" and "64 v_wmma_i32_iu8 per 8da4w
kernel". They equal the per-workgroup-per-K=32-chunk counts of these two tiles. This README does not reproduce
the 780M dumps, so how those figures were counted is **UNVERIFIED**.

## SPIR-V provenance

- **Source:** the SPIR-V is from the native build, compiled with glslc from Vulkan SDK 1.4.350.1 rather than the
  pinned container.
- **4w:** the `texture3d` variant (sha256 `d73ec06a…`) is byte-identical to its `sarc/golden/spirv.json` entry.
  The pinned container generated that entry in `6fb32c33e`.
- **8da4w:** the `texture3d` (`f1c25dc7…`) and `buffer` (`a13974f6…`) variants differ from their golden entries
  (`deb69cb1…`, `a80a945b…`). The driver ISA here therefore comes from the native-build SPIR-V, not from the
  golden SPIR-V. The native build's SPIR-V is what the 7900 XTX e2e binaries used. This README does not dump the
  golden 8da4w SPIR-V.

## Files

- `radv/`, `amdvlk/`, `rga-offline/`: one `.isa.txt` per kernel, and per K for the SARC kernels. Each file starts
  with a header giving the driver, source, dispatch shape and statistics.
- `counts.csv`, `count_wmma.py`: the opcode counts, and the script that regenerates them.
