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

## 2.4 The 10-repetition gate

Ten repetitions of `pw_coopmat_correctness.json`, the many-workgroup collection and the
reduced-grid collection (a shape-level reduction: same N and K, M cut to at most 64 pixels), all
`--validation_only`, on the canonical driver, all 78 non-`_i2c` `amd`/`nvidia` variants, driver
checksum and clock pins re-checked between repetitions, no crash, every stderr empty.

- **73 of the 78 variants pass 10/10 on all three collections: 35 `amd`, 38 `nvidia`.**
- **4 `_epi0` variants (2 per family) are correctness-only:** every gate case carries a bias or a
  residual, which they refuse, so the many-workgroup gate is not applicable to them; they pass
  the correctness collection 10/10.
- **1 fails: `amd_t64x64_f32` (below).**

### The failure: `amd_t64x64_f32`, concurrent-workgroup defect

8 failure events over 7 of the 10 repetitions, always full-grid fused-residual cases, always
this variant; every other variant passes everything. Each time the same case with bias only and
its reduced-grid twin pass, and the correctness collection passes 10/10.

| Rep | Case | Mismatches | Max abs err | First mismatch (channel, pixel) |
|---:|---|---:|---:|---|
| 1 | `(784,512,128)` | 666 | 0.78 | ch 208 (tile 3, +16), px 384 (tile 6, +0) |
| 4 | `(3136,128,256)` | 354 | 0.48 | ch 112 (tile 1, +48), px 1089 (tile 17, +1) |
| 5 | `(784,256,512)` | 207 | 0.27 | ch 112 (tile 1, +48), px 193 (tile 3, +1) |
| 6 | `(3136,128,256)` | 388 | 0.39 | ch 80 (tile 1, +16), px 1282 (tile 20, +2) |
| 7 | `(784,512,128)` | 522 | 0.59 | ch 272 (tile 4, +16), px 258 (tile 4, +2) |
| 7 | `(784,256,512)` | 180 | 0.23 | ch 112 (tile 1, +48), px 66 (tile 1, +2) |
| 8 | `(784,256,512)` | 212 | 0.30 | ch 240 (tile 3, +48), px 130 (tile 2, +2) |
| 10 | `(784,512,128)` | 537 | 0.55 | ch 464 (tile 7, +16), px 129 (tile 2, +1) |

First-mismatch flat indices and full mismatch lists in the local record. The failing cases all
have M in {784, 3136} (13 or 49 pixel tiles); no case
with M <= 289 ever fails. Channel offsets are always 16 or 48 (one or three `MMA_M` row blocks
into the 64-row channel tile); pixel offsets are 0 to 2. Per the gate the variant is **never
timed**. Whether any option (e.g. `XP`) removes this defect is UNVERIFIED; the variant is
excluded.

Bounded diagnosis:

- **Drain passes: one.** `MMAS_M = 64 / 1 / 16 = 4`; the fp32 staging of one pass is
  `SG_GRID_Y * MMA_M * 4 * WG_TILE_N * 4 B = 16 KiB`, inside the 32 KiB drain budget, so
  `DRAIN_G = 4` and `DRAIN_PASSES = 1`. This is not the historical multi-pass drain defect: the
  only multi-pass variants at fp32 (`t128x128_f32`, two passes) pass these cases.
- **Same pixel tile, fp32, passing:** `nvidia_t64x64_f32` (differs only in the K step, 16 against
  32), `amd_t64x32_f32` (K step 32, N tile 32) and every other fp32 variant; the failing
  variant's own f16 sibling passes. The failing combination is K step 32 + fp32 + N tile
  64 + residual + full grid.
- **Tile alignment:** the first mismatch always sits at an `MMA_M` row-block boundary inside a
  channel tile (offset 16 or 48) within three pixels of a pixel-tile boundary, but the region
  itself does not follow the workgroup tiling (offsets 0-2 in M, 16 or 48 in N).

## 2.5 fp16 accumulation limits (record run)

One run of `pw_coopmat_fp16_accumulation_limits.json` (`--validation_only`) for the record: the
collection exercises the 39 f16-accumulating variants (19 `amd`, 20 `nvidia`) on adversarial
cancellation and residual-alpha cases built to exceed fp16 accumulation range. **Every one of
the 39 fails at least one such case, and the baseline arms fail 19 of the 22 case groups as
well** — the defeats are inherent to fp16 accumulation, not variant defects. Per SPEC this
collection is a record, not a gate: 37 of the 39 passed the 2.4 gate 10/10; the two `_epi0`
members are correctness-only (above). The f32-accumulating variants are out of scope of this
collection.

## 4.0 Timed session (1x1, parent selection)

Five runs of `pw_coopmat_m51_timing.json` (21 shapes, arm order rotated per run) with
`--validation_mode=none`, plus two A/A opens of `pw_coopmat_m51_aa.json` before and after the
runs. Metric: per-case p50 of `gpu_timestamp_route_us`; per-cell arm time is the median over
the five runs. The observed GPU clock equals the pinned frequency in every case of every run;
all runs exit 0 with empty stderr, with the canonical driver verified after each run.

**A/A acceptance:** first open geomean 1.0015, second open 0.9959 — both pass the 1% band.
Cells outside +-2% in either open: `(196,96,576)` (+2.05%, then +3.94%), `(49,96,576)` (+2.73%),
`(25,384,4032)` (-2.76%, then -4.84%), `(784,128,512)` (-5.27%). The same cells carry the
cross-run spread of the baseline arm (22-23% for `(196,96,576)` and `(784,128,512)`, 15% for
`(25,384,4032)`); for these cells the noise band is widened to the observed spread, with this
recorded reason: position-in-sequence drift of the baseline arm on high-occupancy shapes.

**Parent selection:** best geometric mean over all 21 shapes among the 73 gated arms, at
2.148x over `pw_tiled_4x8` (and 1.861x over `auto`):

> **Parent variant: `amd_t16x32_f16_al_half`.**

The two arms with a nominally higher geomean cover only 18-19 of 21 shapes (shared-memory
overflow skips on the largest), so their geomeans are not comparable. The parent covers all
21 shapes and wins every cell beyond its noise band:

| Shape | pw_tiled_4x8 / parent | auto / parent | noise band |
|---|---:|---:|---:|
| `(196,1024,256)` | 1.11 | 1.11 | 2% |
| `(196,256,1024)` | 2.54 | 2.03 | 2% (spread) |
| `(196,256,2304)` | 2.44 | 1.72 | 2% |
| `(196,512,1024)` | 1.39 | 1.50 | 2% |
| `(196,64,384)` | 3.58 | 2.52 | 2% |
| `(196,96,576)` | 4.95 | 2.66 | 22% (spread) |
| `(25,384,4032)` | 4.96 | 3.44 | 15% (spread) |
| `(289,224,1344)` | 1.90 | 1.46 | 2% |
| `(3136,128,256)` | 1.04 | 1.05 | 2% |
| `(3136,256,64)` | 1.39 | 1.39 | 2% |
| `(3136,64,256)` | 1.22 | 1.22 | 2% |
| `(3136,64,64)` | 1.45 | 1.46 | 2% |
| `(49,160,960)` | 4.87 | 4.12 | 2% |
| `(49,2048,512)` | 1.21 | 1.70 | 2% |
| `(49,512,2048)` | 4.30 | 2.99 | 3% (spread) |
| `(49,512,4608)` | 6.45 | 4.80 | 3% (spread) |
| `(49,96,576)` | 4.95 | 3.02 | 2% |
| `(784,128,1152)` | 1.44 | 1.51 | 2% |
| `(784,128,512)` | 1.51 | 1.52 | 23% (spread) |
| `(784,256,512)` | 1.06 | 1.06 | 2% |
| `(784,512,128)` | 1.24 | 1.24 | 2% |

Geomeans over shapes: parent vs `pw_tiled_4x8` **2.148x**, parent vs `auto` **1.861x** — both
above the 1.15x campaign success threshold. `auto` itself is 0.866x of `pw_tiled_4x8`
(`auto` resolves to `pointwise` on all 21 shapes; where `auto` beats `pw_tiled_4x8` the
parent's ratio over `auto` is the larger number). The next full-coverage arms:
`amd_t16x32_f32_al_half` (2.122x), `amd_t16x32_f16_alc_half` (2.110x),
`nvidia_t16x32_f32_alc_half` (2.094x), `amd_t16x32_f32_alc_half` (2.079x),
`nvidia_t16x32_f16_alc_half` (2.068x).

## 7.2 FP16 workload baseline (ten CI models)

Twelve traced executions per model (first dropped), ETDump per model, aggregated with
`survey-tools/aggregate.py` (the desktop survey's families; the `m51` device entry added there).
ic4 is exported with random weights (weights do not affect time shares); the deviation is recorded
with 7.1.

**Category shares** of the GPU shader time summed over the ten models: convolution **92.1%**
(mean per-model share 88.6%), linear/matmul 1.1%, elementwise 3.3%, layout/data movement 3.0%,
pooling 0.5%, other 0.1%.

**Shader ranking** (share of the summed GPU shader time):

| rank | shader | share |
|---:|---|---:|
| 1 | `conv2d_gemm_texture2d_half` | 33.8% |
| 2 | `conv2d_2x2x8_border_half` | 25.8% |
| 3 | `conv2d_pw_tiled_bufw_half` | 9.5% |
| 4 | `conv2d_1x1x4_half` | 6.7% |
| 5 | `conv2d_2x2x4_border_half` | 5.5% |
| 6 | `conv2d_pw_tiled_4x8_bufw_half` | 4.7% |
| 7 | `conv1d_clamp_half` | 4.2% |
| 8 | `binary_op_texture3d_half` | 2.7% |
| 9 | `conv2d_im2col_texture2d_half` | 1.3% |

**Convolution-family shares** (routing the device dispatched, share of the convolutions' GPU
shader time): direct 41.2%, im2col+gemm 38.1%, pointwise 15.4%, 1d 4.6%, 1d-pointwise 0.5%,
depthwise 0.2%, winograd 0.1%.

**Kind x K class** (share of the summed GPU shader time):

| kind | K<256 | 256-1k | 1k-4k | 4k-8k | >=8k |
|---|---:|---:|---:|---:|---:|
| pointwise | 1.4% | 4.9% | 9.1% | - | - |
| im2col+gemm | - | 1.3% | 31.0% | 4.1% | 1.7% |
| direct | 0.8% | 27.0% | 7.4% | 3.5% | 2.5% |
| 1d | 0.0% | - | 1.8% | 1.3% | 1.4% |
| 1d-pointwise | - | - | 0.5% | - | - |
| depthwise | 0.2% | - | - | - | - |
| winograd | - | - | 0.1% | - | - |

**WMMA-candidate share:** configurations meeting the candidate criteria (SPEC.md) hold **89.7%**
of the summed GPU shader time. Per model: resnet18 95.3%, resnet50 94.6%, ic3 95.9%, ic4 96.6%,
edsr 83.1%, dl3 97.1%, w2l 96.9%, densenet161 76.9%, mv2 64.8%, mv3 24.1% (mv3's squeeze-excitation
convolutions run at M=1; depthwise convolutions are not dense GEMMs).

Two workload facts shape the rest of the campaign:

- dl3's dilated 3x3 convolutions hold 51% of dl3's GPU shader time. They are candidates by the
  arithmetic but the im2col routing rejects dilation != 1, so they run `direct`; lowering them is
  out of scope of the k-k gate (9.2) and is recorded as future work.
- The largest convolution configuration on the device is edsr's 3x3 64->64 at 224x224 output
  (GEMM M=50176, N=64, K=576), 74.3% of edsr's GPU shader time, followed by 3x3 64->256 at the
  same output plane (M=50176, N=256, K=576). Neither is among the eight desktop k-k gate shapes;
  the 9.2 gate and timing shape sets are augmented with the M51-heavy shapes (9.2 section).
