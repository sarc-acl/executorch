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

Five runs of `pw_coopmat_m51_timing.json` (21 shapes) with `--validation_mode=none`, plus two
A/A opens of `pw_coopmat_m51_aa.json` before and after the runs. Metric: per-case p50 of
`gpu_timestamp_route_us`; per-cell arm time is the median over the five runs. Within each run
every shape executes its 75 arms in one fixed order; the order alternates across runs - run r
executes the arms as `arm[(r + i) mod 75]`, a rotation of the whole list, so no arm keeps the
same position in two consecutive runs. The observed GPU clock equals the pinned frequency in
every case of every run; all runs exit 0 with empty stderr, with the canonical driver verified
after each run.

**A/A acceptance:** the opening A/A geomean is 1.0015; the closing A/A, which decides validity,
is **0.9959** - both inside the 1% band. Cells outside +-2% in either open: `(196,96,576)`
(+2.05%, then +3.94%), `(49,96,576)` (+2.73%), `(25,384,4032)` (-2.76%, then -4.84%),
`(784,128,512)` (-5.27%). The same cells carry the cross-run spread of the baseline arm
(22-23% for `(196,96,576)` and `(784,128,512)`, 15% for `(25,384,4032)`); for these cells the
noise band is widened to the observed spread, with this recorded reason: position-in-sequence
drift of the baseline arm on high-occupancy shapes.

Headline ratios are against `pw_tiled_4x8` (the THRESHOLDS baseline) and against `auto`
(which resolves to `pointwise` on all 21 shapes). `auto` holds 1.15x over `pw_tiled_4x8`.
Only full-coverage variants (21 of 21 shapes) are eligible as parent or for a headline
geometric mean; partial-coverage variants are listed with their coverage.

**Parent selection (SPEC correctness gate: fp32-accumulation):**

> **Parent variant: `amd_t16x32_f32_al_half`.** Geomean **2.122x** over `pw_tiled_4x8`,
> **1.838x** over `auto`, covering all 21 shapes, winning every cell beyond its noise band.

**fp32-accumulation variants** (geomean of `pw_tiled_4x8` / variant over covered shapes):

| variant | geomean | coverage |
|---|---:|---:|
| `amd_t16x32_f32_al_half` (parent) | 2.122 | 21/21 |
| `nvidia_t16x32_f32_alc_half` | 2.094 | 21/21 |
| `amd_t16x32_f32_alc_half` | 2.079 | 21/21 |
| `amd_t16x32_f32_alc_sk4_half` | 2.073 | 18/21 |
| `amd_t16x32_f32_alc_sk2_half` | 2.037 | 19/21 |
| `amd_t128x128_f32_alc_sk4_half` | 2.000 | 18/21 |
| `nvidia_t32x32_f32_al_half` | 1.990 | 21/21 |
| `nvidia_t16x32_f32_alc_sk4_half` | 1.955 | 19/21 |

**fp16-accumulation variants** - best `amd_t16x32_f16_al_half`, geomean 2.148x over
`pw_tiled_4x8`, 1.861x over `auto`, 21/21 shapes; about 1% above the parent, inside the 2%
band. Every fp16-accumulation variant, this one included, fails the adversarial fp16-limits
cases of the 2.5 record; the parent is fp32 by the SPEC correctness gate. Next fp16 arms:
`amd_t128x128_f16_alc_sk4_half` (2.210, 18/21), `amd_t16x32_f16_alc_half` (2.110, 21/21),
`nvidia_t16x32_f16_alc_half` (2.068, 21/21).

**Per-shape oracle** (best gated variant per shape; an oracle over the variants, not a single
variant):

| shape | pw_tiled_4x8 / best | auto / best | best variant |
|---|---:|---:|---|
| `(196,1024,256)` | 1.45 | 1.46 | `amd_t128x128_f16_alc_half` |
| `(196,256,1024)` | 3.02 | 2.42 | `amd_t128x128_f16_alc_sk4_half` |
| `(196,256,2304)` | 3.74 | 2.64 | `amd_t128x128_f16_alc_sk4_half` |
| `(196,512,1024)` | 1.88 | 2.03 | `amd_t128x128_f16_alc_sk2_half` |
| `(196,64,384)` | 3.58 | 2.52 | `amd_t16x32_f16_al_half` |
| `(196,96,576)` | 4.95 | 2.66 | `amd_t16x32_f16_al_half` |
| `(25,384,4032)` | 9.35 | 6.49 | `amd_t16x32_f32_alc_sk4_half` |
| `(289,224,1344)` | 2.47 | 1.90 | `amd_t128x128_f16_alc_sk4_half` |
| `(3136,128,256)` | 1.41 | 1.41 | `nvidia_t128x128_f32_alc_half` |
| `(3136,256,64)` | 1.77 | 1.77 | `nvidia_t128x128_f32_alc_half` |
| `(3136,64,256)` | 1.43 | 1.43 | `nvidia_t32x32_f32_al_half` |
| `(3136,64,64)` | 1.72 | 1.72 | `nvidia_t16x32_f32_alc_half` |
| `(49,160,960)` | 6.63 | 5.61 | `amd_t16x32_f32_alc_sk4_half` |
| `(49,2048,512)` | 1.71 | 2.41 | `nvidia_t32x32_f32_al_half` |
| `(49,512,2048)` | 4.30 | 2.99 | `amd_t16x32_f16_al_half` |
| `(49,512,4608)` | 6.75 | 5.02 | `nvidia_t16x32_f32_alc_sk2_half` |
| `(49,96,576)` | 6.22 | 3.80 | `amd_t16x32_f32_alc_sk4_half` |
| `(784,128,1152)` | 2.03 | 2.14 | `amd_t128x128_f16_alc_sk2_half` |
| `(784,128,512)` | 1.77 | 1.78 | `nvidia_t32x32_f32_al_half` |
| `(784,256,512)` | 1.44 | 1.45 | `amd_t128x128_f16_alc_half` |
| `(784,512,128)` | 1.59 | 1.58 | `nvidia_t128x128_f32_alc_half` |

Per-shape ratios of the parent (`amd_t16x32_f32_al_half`), with the noise band (widened to the
observed spread where the A/A and cross-run spread exceeded 2%); the parent wins every cell:

| shape | pw_tiled_4x8 / parent | auto / parent | band |
|---|---:|---:|---:|
| `(196,1024,256)` | 1.11 | 1.11 | 2% |
| `(196,256,1024)` | 2.53 | 2.03 | 2% |
| `(196,256,2304)` | 2.43 | 1.71 | 2% |
| `(196,512,1024)` | 1.38 | 1.49 | 2% |
| `(196,64,384)` | 3.55 | 2.50 | 2% |
| `(196,96,576)` | 4.81 | 2.59 | 22% |
| `(25,384,4032)` | 4.85 | 3.37 | 15% |
| `(289,224,1344)` | 1.87 | 1.43 | 2% |
| `(3136,128,256)` | 1.04 | 1.04 | 2% |
| `(3136,256,64)` | 1.41 | 1.41 | 2% |
| `(3136,64,256)` | 1.22 | 1.22 | 2% |
| `(3136,64,64)` | 1.48 | 1.48 | 2% |
| `(49,160,960)` | 4.75 | 4.02 | 2% |
| `(49,2048,512)` | 1.21 | 1.69 | 2% |
| `(49,512,2048)` | 4.22 | 2.94 | 3% |
| `(49,512,4608)` | 5.52 | 4.11 | 3% |
| `(49,96,576)` | 4.86 | 2.97 | 2% |
| `(784,128,1152)` | 1.48 | 1.56 | 2% |
| `(784,128,512)` | 1.50 | 1.51 | 23% |
| `(784,256,512)` | 1.06 | 1.06 | 2% |
| `(784,512,128)` | 1.24 | 1.24 | 2% |

Both parent geomeans (2.122x over `pw_tiled_4x8`, 1.838x over `auto`) exceed the 1.15x campaign
success threshold.

## 7.2 FP16 workload baseline (ten CI models)

Twelve traced executions per model (first dropped), ETDump per model, aggregated with
`survey-tools/aggregate.py` (the desktop survey's families; an `m51` device entry added there).
ic4 is exported with random weights (weights do not affect time shares); the deviation is recorded
with 7.1. The desktop reference is the owner's four-GPU FP16 survey (mean over 10 models x 4
desktops); the M51 primary metric is the same shape - the **mean of per-model shares** - with the
share of summed GPU shader time as the second column.

**Category shares** (primary: mean of per-model shares; secondary: share of summed time;
desktop: mean of per-model shares):

| category | M51 mean | M51 summed | desktop mean |
|---|---:|---:|---:|
| convolution | **88.6%** | 92.1% | 87.3% |
| linear/matmul | 4.8% | 1.1% | 3.3% |
| layout/data movement | 2.9% | 3.0% | 4.9% |
| elementwise | 2.7% | 3.3% | 3.4% |
| pooling | 0.7% | 0.5% | 1.1% (pooling+other) |
| other | 0.2% | 0.1% | (with pooling) |

**Top shaders** (share of a model's GPU shader time, averaged over the ten models - absent
models count zero; `n` = models dispatching it; desktop reference in the last columns):

| shader | M51 mean (n) | M51 summed | desktop mean (n) |
|---|---:|---:|---:|
| `conv2d_gemm_texture2d_half` | **32.5%** (5) | 33.8% | 26.6% (5) |
| `conv2d_pw_tiled_bufw_half` | 16.3% (8) | 9.5% | 20.4% (8) |
| `conv2d_2x2x8_border_half` | 12.0% (7) | 25.8% | 18.5% (9) |
| `conv1d_clamp_half` | 8.7% (1) | 4.2% | 8.7% (1) |
| `conv2d_pw_tiled_4x8_bufw_half` | 5.4% (8) | 4.7% | 4.9% (8) |
| `conv2d_1x1x4_half` | 5.2% (4) | 6.7% | 3.0% (4) |
| `linear_vec_bias_tile_row_1...` | 4.8% (7) | 1.1% | 3.3% (7) |
| `conv2d_2x2x4_border_half` | 4.0% (8) | 5.5% | - |

**Kernel-kind shares of convolution time** (summed over the models; classification: dims-1
convolutions are 1D, groups = Cin are depthwise, then kernel x stride x dilation; desktop
reference = one desktop GPU of the owner's survey):

| kind | M51 | desktop |
|---|---:|---:|
| 3x3 s1 | **45.1%** | 37.3% |
| 1x1 | 15.4% | 23.5% |
| 1x7/7x1 | 14.4% | 9.5% |
| 3x3 dilated | 7.4% | 18.0% |
| 3x3 s2 | 5.9% | 3.5% |
| 1x3/3x1 | 5.7% | 3.4% |
| 1D | 5.1% | 3.8% |
| depthwise | 0.2% | 0.2% |
| other | 0.9% | 0.9% |

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

## 9.2 The im2col GEMM stage (k x k gate and timed session)

Gate collections: the desktop layer table's eight shapes plus the six M51-heavy shapes of 7.2
(fourteen shapes, bias and fused-residual epilogue each), full grid and reduced grid, ten
repetitions, plus one run of the correctness collection. Driver canonical and clocks pinned
throughout; every stderr empty.

**Gate verdict: all 32 `amd`/`nvidia` `_i2c` variants pass 10/10 on both grids** — zero failing
dispatches across the twenty runs (896 coopmat dispatches per run). Every variant creates its
pipeline on the device (no shared-memory overflow skips), and every K of the fourteen shapes is
divisible by both families' K steps.

**Production-path accuracy finding.** The failures in these runs are not coopmat failures: the
`direct` route — which `auto` selects on four of the shapes — fails validation (atol 0.1,
rtol 0.01) on `3x3 64 56x56` (one of the desktop shapes), `3x3 64 224x224`, `3x3 64-256 224x224`
(both epilogues) and `3x3 192-48 28x28`, with maximum absolute errors of 0.14-0.53 against the
0.1 tolerance (1.04 on the correctness collection's gather-grid-limit case). The `im2col` route
passes everywhere. Mismatch counts scale with the output pixel count, which is why the reduced
grid passes; this is the same fp16-accumulation class as the 2.5 record. The incumbent fast path
on the K=576-class shapes is out of tolerance on this device; the tolerance-holding incumbent
route there is `im2col`.

**Timed session:** five runs with rotated arm order (auto, direct, im2col, the 32 variants),
per-case p50 of `gpu_timestamp_route_us`, per-cell median over the five runs, A/A opens of the
fixed `im2col` route before and after. A/A geomeans 0.9996 and 1.0021 (one cell at +2.1% in the
second open); observed clocks match the pin in every case. One cell (`3x3 192-48 28x28`) carries
a 6% cross-run spread of the baseline arm; its noise band is widened accordingly.

Best arm over the fourteen shapes, covering all of them:

> **`amd_t128x128_f16_alc_sk4_i2c_half` — geomean 3.69x over `auto`, 5.66x over `im2col`.**

| shape | auto / best | im2col / best | band | win |
|---|---:|---:|---:|---|
| `1x7 192-224 17x17` | 5.62 | 5.62 | 2% | yes |
| `1x7 224-256 17x17` | 5.70 | 5.75 | 2% | yes |
| `3x3 1024-256 28x28` | 4.13 | 4.13 | 2% | yes |
| `3x3 128 28x28` | 4.52 | 4.52 | 2% | yes |
| `3x3 192-48 14x14` | 2.16 | 10.09 | 2% | yes |
| `3x3 192-48 28x28` | 1.62 | 4.47 | 6% | yes |
| `3x3 256 14x14` | 10.11 | 10.11 | 2% | yes |
| `3x3 448-384 5x5` | 13.99 | 14.02 | 2% | yes |
| `3x3 512 7x7` | 12.22 | 12.22 | 2% | yes |
| `3x3 64 224x224` | 0.40 | 1.60 | 2% | no |
| `3x3 64 56x56` | 0.53 | 1.51 | 2% | no |
| `3x3 64-256 224x224` | 0.87 | 2.38 | 2% | no |
| `3x3 s2 256-512 14x14` | 10.49 | 10.43 | 2% | yes |
| `3x3 s2 512 14x14` | 12.20 | 12.19 | 2% | yes |

The best arm wins 11 of 14 cells against `auto` beyond the noise band. The three losses are
exactly the shapes where `auto` routes `direct` — the route that fails tolerance there (above);
against the tolerance-holding incumbent route (`im2col`) the best arm wins all 14 cells, by
1.5x-14x. The next arms: `nvidia_t128x128_f16_alc_sk4_i2c_half` (3.50x over `auto`),
`amd_t128x128_f16_alc_sk2_i2c_half` (3.48x), `amd_t16x32_f16_alc_sk2_i2c_half` (3.43x),
`nvidia_t16x32_f16_alc_sk4_i2c_half` (3.41x).

For reference, `auto` holds 2.03x over `direct` (where `direct` runs) and 0.65x against
`im2col` (i.e. `im2col` is 1.5x slower than `auto` on the shapes `auto` routes elsewhere); the
timing is the whole convolution, gather included.
