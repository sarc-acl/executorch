# M51 conv2d-pw-coopmat results

Ratio-only record of the campaign (SPEC.md, "Publication of M51 results"): no absolute device
figures, no board, host, driver-build or system identifiers. Raw data stays in the local record.

## 2.1 Shared memory of the AMD-family variants

Computed from the descriptor table in `Conv2dPW.cpp` by the same formula as
`Conv2dPwCoopmatVariant::shared_bytes()`, for every `amd` and `nvidia` descriptor row (the
16x16x16 families, `_alc`, `_sk`, `_i2c` included). Sizes depend on the tile geometry and the
fragment layout only, so variants repeat across the rows.

| Shared memory | Variants |
|---:|---|
| 3.0 KiB | 2 |
| 4.0 KiB | 16 |
| 5.5 KiB | 8 |
| 6.0 KiB | 4 |
| 7.5 KiB | 29 |
| 8.0 KiB | 1 |
| 8.5 KiB | 2 |
| 10.0 KiB | 2 |
| 10.5 KiB | 1 |
| 11.5 KiB | 4 |
| 14.0 KiB | 2 |
| 15.0 KiB | 2 |
| 16.0 KiB | 1 |
| 19.0 KiB | 2 |
| 32.0 KiB | 20 |
| 37.0 KiB | 16 |

Largest: 37.0 KiB (`t128x128`). **Exclusions over the 64 KiB device cap: none.** Cross-checked on
the device: the dispatch eligibility refuses a variant whose shared memory exceeds the device
limit, and the first correctness run skips no `amd`/`nvidia` variant.

## 2.2 Pipeline creation of the AMD-family variants

One correctness run of `pw_coopmat_correctness.json` (`--validation_only`) on the canonical
driver: **all 78 non-`_i2c` `amd`/`nvidia` variants create their pipelines and pass every
eligible case** (0 failures, 0 skips; the `xe2` family skips, the device has no 8x16x16 matrix).
The uniform-buffer trip-count crash of the earlier driver did not reproduce.

## 2.4 Gate failure: `amd_t64x64_f32`, concurrent-workgroup defect

Rep 1 of the many-workgroup collection (a shape-level reduction is the reduced-grid collection:
same N and K, M cut to at most 64 pixels). Case `gate full residual (784,512,128)`, fused
residual add: **FAIL**, 666 of 401408 elements mismatch, max abs error 0.78 against atol 0.1, no
nonfinite values. First mismatch indices (flat NCHW over `(1,512,28,28)`): 163456, 163457,
163458, 163463, 163465, 163466, 163468, 163471 — channel 208, pixel 64 and the following pixels.

The same variant passes the same case with bias only, passes the reduced-grid twin
`(64,512,128)` with the fused residual, and passes the correctness collection: exact with few
workgroups, wrong in the full grid, only with the residual epilogue. Per the gate it is never
timed. Repetitions 2-3 did not reproduce it; the defect is intermittent.

Bounded diagnosis (the fix path is `XP`, TASKS 3.2; the 3.4 gate re-runs this exact case):

- **Drain passes: one.** `MMAS_M = 64 / 1 / 16 = 4`; the fp32 staging of one pass is
  `SG_GRID_Y * MMA_M * 4 * WG_TILE_N * 4 B = 16 KiB`, inside the 32 KiB drain budget, so
  `DRAIN_G = 4` and `DRAIN_PASSES = 1`. This is not the historical multi-pass drain defect: the
  only multi-pass variants at fp32 (`t128x128_f32`, two passes) pass this case.
- **Same pixel tile, fp32, passing:** `nvidia_t64x64_f32` (differs only in the K step, 16 against
  32), `amd_t64x32_f32` (K step 32, N tile 32) and every other fp32 variant on the case; the
  failing variant's own f16 sibling passes. The failing combination is K step 32 + fp32 + N tile
  64 + residual + full grid.
- **Tile alignment:** the mismatch region starts at pixel 64, the boundary of the second
  64-pixel workgroup tile (aligned in M), at channel 208, sixteen rows into channel tile 3
  (192..255) — one `MMA_M` block into the tile in N, not on a channel-tile boundary.

