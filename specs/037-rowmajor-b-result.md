# specs/037: RowMajor B layout for int8 dq8ca coopmat — REJECTED (RADV 780M)

## Result

RowMajor B staging is **slower** than the shipped ColumnMajor on the 780M/RADV
at the current tile (128x64/K32/2x2/sg32, g128):

| layout      | 8da4w prefill 2048x2048 kernel | 1B 8da4w geomean vs tiled |
|-------------|-------------------------------|---------------------------|
| ColumnMajor | 2427 us                       | 0.95x kernel              |
| RowMajor    | 2559 us (~5% slower)          | 0.90x kernel              |

Correctness: 44/44 PASSED in both layouts.

## Why (ISA evidence)

The int8 WMMA `matB` lane layout wants MMA_K (=16) K-contiguous bytes per lane.

- **ColumnMajor** LDS stores each N-column's 16 K-bytes contiguously (one uint =
  4 int8, 4 uints = 16 int8 = a full column). `coopMatLoad` emits **one
  `ds_read_b128`** per column per lane.
- **RowMajor** LDS stores 4 N-contiguous int8 per uint. A lane's 16 K-values for
  a fixed N are then one-byte-per-row, `B_STRIDE_U32` apart. `coopMatLoad` is
  forced to gather them with **`ds_read_u8_d16` / `ds_read_u8_d16_hi` byte
  chains** (16+ per B region).

This is layout-inherent, not an alignment artifact: the elements a lane needs
are on the strided axis, so row-padding to 128-bit can't restore vectorized
loads. The reference `vk_cooperative_matrix_perf` measured RowMajor faster for
its types, but that regime differs (16-bit A/B where the native matB canonical
layout matches RowMajor); for int8 matB on this compiler the native canonical
layout is K-contiguous == ColumnMajor.

## Decision

Keep ColumnMajor. The header comment block (lines 73-78) already documented this
as the reason for ColumnMajor; specs/037 upgrades that from an untested
assumption to a measured, ISA-confirmed fact. Move optimization effort to the
tile fidelity sweep (specs/038) and B-unpack/staging slimming.
