# sarc-1.5-8da4w-port

## Why

This change ports the 8da4w (dq8ca: dynamic int8 activations, 4-bit weights) coopmat winners of the
release-1.4 SARC branches to release 1.5.

## What

**Kernel families:**
- `sarc_linear_dq8ca_coopmat_zpg` (4h4w activations): the 780M kernel, plus the Xe2 multi-block A staging
  behind `A_MULTI_BLOCK`. The LDS fences are kept for every device.
- `sarc_linear_dq8ca_coopmat_zpgtr` (row-major activations): the 4070 Ti kernel (`A_RAW`, `B_PAIR`,
  `CSH_IN_ASH`, MMA_K 32).
- `sarc_quantize_and_pack_4w_with_group_sums`: the row-major activation packer.

**Op path:** `impl/sarc/Dq8caCoopmat.cpp`, hooked into `linear_dq8ca_q4gsw`.
- The activation layout follows the kernel chosen for the build-time shape.
- A row-major op keeps its kernel for every M (the unaligned-M fix).
- A 4h4w op falls back to release 1.5's dq8ca tiled/coop kernels.

**Rows:**

| Device | Kernel |
|---|---|
| 780M | zpg t128x64k32g42s32 |
| B580, B70 | zpg t256x64k32g48s16 (MMA 8x16x32) |
| 4070 Ti SUPER | zpgtr mk32 t128x128k64g44s32 ra |
| Orin | zpgtr mk32 t128x128k64g44s32 ra, measured projections only |
| M51 | zpgtr t128x64k32g42s32, unverified; owned by the M51 agent |

The Adreno 8da4w variant is not ported (its 1.4 branch marks it broken).

## Results (2026-09-27; release 1.5, 2048-token prefill tok/s, 1B / 3B / 8B)

Logs are in `results/<gpu>`.
- Kernel ratio: microbench kernel median against the 1.4 confirmation, 24 shapes.
- "1.4 tuned" is the 1.4 branch's end-to-end result. On the 780M it includes the SARC SDPA coopmat,
  which this change does not contain.

| GPU | tiled | stock 1.5 | SARC | SARC / stock | 1.4 tuned | kernel ratio vs 1.4 |
|---|---|---|---|---|---|---|
| Radeon 780M | 1631 / 566 / 280 | 1624 / 569 / 281 | 1845 / 649 / 336 | 1.14 / 1.14 / 1.20× | 2538 / 1055 / 486 | 0.992 (0.985–1.002) |
| Arc B580 | 5919 / 2092 / 932 | 6095 / 2169 / 937 | 8790 / 3507 / 1829 | 1.44 / 1.62 / 1.95× | 8533 / 3374 / 1789 | 1.000 (0.986–1.021) |
| Arc Pro B70 | 8325 / 3282 / 1381 | 8292 / 3272 / 1375 | 12412 / 5251 / 2738 | 1.50 / 1.60 / 1.99× | 12264 / 5159 / 2713 | 1.000 (0.972–1.022) |
| RTX 4070 Ti SUPER | 6872 / 2563 / 1108 | 6896 / 2560 / 1106 | 21113 / 9660 / 5032 | 3.06 / 3.77 / 4.55× | 21558 / 9660 / 5032 | 1.000 (0.992–1.009) |
| Jetson Orin | 213 / 76 / 32 | 212 / 76 / 32 | 822 / 320 / 170 | 3.87 / 4.23 / 5.27× | 823 / – / – | 1.000 (0.998–1.006) |

**Production-diff (1B/3B/8B × buffer/texture3d, non-zero activation zero points):**
- Every case is numerically within tolerance on all five GPUs (for example, 780M max |err| 0.016–0.124
  against 0.5).
- The first run reported FAILED only because the harness saw no "coopmat" in the old
  `sarc_linear_dq8ca_zpg*` names. The kernels have been renamed, and the re-run is in `results/*/pdiff2`.

**Next token vs tiled:**
- The 1973-token real-text prompt matches on all five GPUs.
- The 1792-token prompt (`r1304.txt`) matches on 4070 Ti and Orin, which use the row-major path.
- On the 780M, B580 and B70 it differs from 1.5's tiled kernel, but SARC equals the 1.4 build on the
  780M: 1.4 tiled, 1.4 tuned and 1.5 SARC all give the same token, and only 1.5 tiled flips.
  Production-diff is within tolerance. This is a near-tie flip in 1.5's tiled path, not a port error.

**Decode:** works on all five GPUs.

**1.5 vs 1.4:** the kernels and end-to-end results match on B580, B70, 4070 Ti and Orin (within ~3 %).
The 780M end-to-end difference is SDPA (see `.artifacts/sarc-1.5/sdpa-gap/REPORT.md`), addressed by the
SDPA coopmat rows.
