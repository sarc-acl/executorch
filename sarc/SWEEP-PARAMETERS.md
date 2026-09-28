# SARC coopmat shader parameters (4w and 8da4w)

This page lists every compile-time parameter of the SARC linear coopmat shaders. Each yaml variant is one
combination of these values and compiles to one SPIR-V file.
- Sweep candidates go in `glsl/sarc_dev/<family>_sweep.yaml`.
- Shipped variants go in `glsl/sarc/<family>.yaml`. The promotion checklist is in [README.md](README.md).

| Scheme | Family | Activation layout |
|---|---|---|
| 4w | `sarc_linear_q4gsw_coopmat` | fp16 |
| 8da4w | `sarc_linear_dq8ca_coopmat_zpg` | int8, 4h4w blocks |
| 8da4w | `sarc_linear_dq8ca_coopmat_zpgtr` | int8, row-major (`kPackedInt8_4W`) |

The choice between zpg and zpgtr is itself a sweep dimension. The op path (`impl/sarc/Dq8caCoopmat.cpp`)
packs the activations in the layout of the kernel that the build-time shape selects.

## Common parameters (all three families)

| Parameter | Meaning | Values used so far |
|---|---|---|
| `WG_TILE_M`, `WG_TILE_N` | Output tile per workgroup | M 64/128/256; N 64/128/256 |
| `WG_TILE_K` | K consumed per main-loop iteration | 16 / 32 / 64 |
| `SG_GRID_X`, `SG_GRID_Y` | Subgroup grid in the workgroup (X along N, Y along M); workgroup size = X·Y·`SUBGROUP_SIZE` | 2×1, 2×2, 4×2, 2×4, 4×4, 4×8 |
| `SUBGROUP_SIZE` | Required subgroup size (pipeline subgroup-size control) | 16 (Intel Xe2), 32, 64 (Adreno) |
| `MMA_M`, `MMA_N`, `MMA_K` | Shape of one coopmat instruction; it must be a shape the device exposes (igpu-roofline `docs/COOPMAT-SHAPES.md`) | 16×16×16 (default); M 8 on Intel Xe2; K 32 for int8 on Xe2, 4070 Ti and Orin; 64×32×16 on Adreno |
| `IO_STORAGE` | Input and output tensor storage | `buffer`, `texture3d` |
| `WEIGHT_STORAGE` | Packed weight storage | `texture2d`, `buffer` |
| `HAS_BIAS` | Bias epilogue | only `false` is shipped; the selector never picks SARC for linears with bias |
| `PRECISION` | GLSL default precision | `highp` |

## 4w only (`sarc_linear_q4gsw_coopmat`)

All flags default to off.

| Flag | Effect | Used by |
|---|---|---|
| `ACC_FP32` | fp32 accumulator | 780M (maps 1:1 onto `v_wmma_f32_16x16x16_f16`); Orin (large-K accuracy) |
| `ACC_GROUP_FP32` | fp16 accumulation within a quantization group, fp32 running total. Cannot be combined with `ACC_FP32` (`#error`). | 4070 Ti SUPER: full-rate fp16 MMA without the long-K error |
| `ACC_GROUP_FP32_REG` | As `ACC_GROUP_FP32`, but the fp32 running total is a per-invocation register array, filled from an LDS copy of each fp16 group sum (Adreno has no fp32 accumulator coopmat, and its compiler crashes on per-element coopmat access). Cannot be combined with `ACC_FP32`, `ACC_GROUP_FP32`, `CSH_IN_ASH`, `B_COLMAJOR` or `FRAG_LAYOUT` (`#error`). | Adreno 840 (sweep candidate) |
| `CSH_IN_ASH` | texture3d output drain staged in the dead A shared-memory region | 780M (less LDS, higher occupancy) |
| `FRAG_LAYOUT` | Fragment-contiguous shared-memory layout, no padding | Intel Xe2 |
| `IMG_A` | Storage-image loads for A; only with `IO_STORAGE: texture3d` | Intel Xe2 |
| `IMG_W` | Storage-image loads for the weights; only with `WEIGHT_STORAGE: texture2d` | available, not shipped |
| `SH_F16V4` | Shared memory stored as f16vec4. Cannot be combined with `CSH_IN_ASH` or `FRAG_LAYOUT` (`#error`). | Adreno 840 |

## 8da4w zpg only (`sarc_linear_dq8ca_coopmat_zpg`)

| Parameter | Effect | Used by |
|---|---|---|
| `WEIGHT_NBITS` | Weight bit width; 4 defines `WEIGHT_INT4` | 4 |
| `A_MAP_FULL` | Each thread stages exactly one A block, which removes the `a_active` guard from the hot loop. Set it only when the arithmetic `A_ACTIVE_THREADS == WG_SIZE` has been checked for the tile, and record that arithmetic in the yaml. | 780M, Intel Xe2 |
| `A_MULTI_BLOCK`, `A_BLOCKS` | Each thread stages `A_BLOCKS` A blocks (`tid + i * WG_SIZE`) | Intel Xe2 |

## 8da4w zpgtr only (`sarc_linear_dq8ca_coopmat_zpgtr`)

| Parameter | Effect | Used by |
|---|---|---|
| `WEIGHT_NBITS` | Weight bit width; 4 defines `WEIGHT_INT4` | 4 |
| `A_RAW` | A staged as raw uvec4 global-to-LDS copies | 4070 Ti SUPER, Orin |
| `B_PAIR` | One weight texel feeds both nibble parities | 4070 Ti SUPER, Orin |
| `CSH_IN_ASH` | texture3d output drain staged in `Ash_int8` | 4070 Ti SUPER, Orin (texture3d variants) |

## Hard preconditions

The shaders contain no shape checks. `sarc::select()` enforces these when it matches a row:
- `M % WG_TILE_M == 0`, `N % WG_TILE_N == 0`, `K % WG_TILE_K == 0`;
- `group_size % WG_TILE_K == 0`;
- no bias, batch 1, not a single-token (gemv) call;
- the storage combination is allowed by the row's `storages` mask;
- `SUBGROUP_SIZE` equals the device's subgroup size, or lies within its supported range when subgroup-size
  control is available;
- shared memory fits the device limit (`max_compute_shared_memory_size`).

The table row must also repeat the tile geometry in its `TileDims`, because launch dims come from the row and
kernel names are never parsed. `test_sarc_select` checks that every row names an existing variant.

Rows for the row-major (zpgtr) path stay on their kernel for every M. This is the unaligned-M fix:
the activation layout was fixed at build time, so the op cannot fall back to a 4h4w kernel.

## Variant naming

Names follow the pattern `<family>[_sweep]_<tile>_<io>_<weight>_half`. For example,
`sarc_linear_q4gsw_coopmat_t128x128k32g42s32f32c_texture3d_texture2d_half` has:
- `IO_STORAGE` texture3d;
- `WEIGHT_STORAGE` texture2d;
- the tile token `t128x128k32g42s32f32c`, decoded below.

| Token part | Meaning |
|---|---|
| `t<M>x<N>` | `WG_TILE_M` × `WG_TILE_N` |
| `k<K>` | `WG_TILE_K` |
| `g<X><Y>` | `SG_GRID_X`, `SG_GRID_Y` |
| `s<S>` | `SUBGROUP_SIZE` |
| `f32` | `ACC_FP32` |
| `c` | `CSH_IN_ASH` (4w) |
| `ga` | `ACC_GROUP_FP32` |
| `gr` | `ACC_GROUP_FP32_REG` |
| `m8` | `MMA_M` 8 |
| `m<M>x<N>x<K>` | full MMA shape (Adreno) |
| `fli` | `FRAG_LAYOUT` + `IMG_A` |
| `mk32` | `MMA_K` 32 |
| `ra` | `A_RAW` + `B_PAIR` (zpgtr) |

Any other shipped flag, such as zpg's `A_MAP_FULL`, is not in the name. Read the yaml entry.

## Shipped variants (release 1.5)

| Device | 4w | 8da4w |
|---|---|---|
| Radeon 780M | `t128x128k32g42s32f32c` | zpg `t128x64k32g42s32` + `A_MAP_FULL` |
| Arc B580 / Pro B70 | `t128x128k16g44s16m8fli` | zpg `t256x64k32g48s16m8` (MMA 8×16×32) + `A_MAP_FULL` + `A_MULTI_BLOCK` |
| RTX 4070 Ti SUPER | per shape: `t256x128k16g42s32ga`, `t128x128k16g24s32ga`, `t128x256k16g42s32ga`, `t128x128k16g42s32ga` | zpgtr `t128x128k64g44s32mk32ra` (+ `CSH_IN_ASH` for texture3d) |
| Jetson Orin | per shape: `t128x128k32g42s32f32`, `t256x128k16g22s32`, `t128x128k16g22s32` | zpgtr `t128x128k64g44s32mk32ra` |
| Xclipse (M51), unverified | `t128x128k16g22s32` | zpgtr `t128x64k32g42s32` (MMA 16×16×16) |
| Adreno 840, unverified | `t64x64k32g21s64m64x32x16` + `SH_F16V4` | – |

The authoritative list is the release yaml plus `impl/sarc/table_<vendor>.cpp`.

## Running a sweep candidate

**4w.** Set `ET_VK_SARC_Q4GSW_VARIANT=<tile token>` in a dev build, for example `t128x128k32g24s32f32c`.
- The token must name a variant in the sweep yaml or the release yaml.
- It must also have a candidate row in `impl/sarc_dev/Overrides.cpp` (`kQ4gswCandidates`), or a release row.
- The override also builds 4w on the SARC path on devices without rows.

**8da4w.** There is no runtime variant override yet. The sweep yaml has 2 zpg candidates (Xe2 tiles) and 4
zpgtr candidates (4070 Ti tiles). To test one:
- call it by name in `test_llama_microbench`; or
- add an `ET_VK_SARC_DQ8CA_VARIANT` override and candidate rows in `impl/sarc_dev/Overrides.cpp`, mirroring the
  4w one.

Both options stay in the dev zone.

**Other dev switches:**
- `ET_VK_SARC_UNVERIFIED=1` activates `kUnverified` rows.
- `ET_VK_FORCE_TILED_LINEAR=1` sends the SARC linears to release 1.5's tiled/coop kernels, as the baseline.
- `ET_VK_DISABLE_COOPMAT=1` sends SDPA to the upstream kernels.

## Adding a new parameter

1. Add the `#ifdef` to the family's untemplated `*_body.glslh`, default off.
2. Add the `$if` in both wrappers: `glsl/sarc/<family>.glsl` and `glsl/sarc_dev/<family>_sweep.glsl`. They must
   stay byte-identical from `#version` on (`check.sh`).
3. Add a default to both yamls.
4. Run `check.sh`.
   - The golden check must show no change for any shipped variant.
   - A change to a shipped variant's SPIR-V needs the owning device to re-verify it (see README).
