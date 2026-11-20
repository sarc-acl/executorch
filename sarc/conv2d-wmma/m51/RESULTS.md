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

## 5.2 Phase timing of the 1x1 parent (ablation method)

SPEC asks for in-kernel shader-clock phase timing. The device's capability record reports
api_version 1.4.354; the shader-clock extension's support on the device was not checked, and
ablation was used for the time budget: five debug builds of the same bench, each compiled with
one phase of the parent
template disabled (`--env PHASE_TIMING=1 PHASE_ABLATE=<phase>`, a debug-only template path that
the default build does not take).

What each ablation removes - stated exactly, because the ablations overlap (verified by
generating the parent variant per phase with the bench's own generator flags and counting
opcodes in the resulting SPIR-V):

- FETCH replaces the input and the weight loads with constants; the staging and the MMA still
  run (OpImageFetch 11 -> 3 in the generated SPIR-V; the three remaining are drain-side).
- STAGE skips the shared-memory stores - and, because the fetched fragments become dead code,
  the compile-time optimizer also removes the input/weight fetches (OpImageFetch 11 -> 3,
  shared OpStore 12 -> 0, already at SPIR-V level). The STAGE share therefore covers staging
  plus fetch; subtracting the fetch-only share bounds the staging store at roughly 34%.
- SYNC skips only the K-loop and post-loop `memoryBarrierShared(); barrier();` pairs
  (OpMemoryBarrier 4 -> 2, OpControlBarrier 4 -> 2; the two remaining pairs are in the drain
  passes); fetch, staging and MMA all still run.
- MMA skips the whole `mma_step`: the cooperative-matrix loads from shared memory and the
  multiplies (OpCooperativeMatrixLoadKHR 12 -> 0, OpCooperativeMatrixMulAddKHR 8 -> 0). The
  fetch and the shared stores still run, so the MMA share covers the shared-memory reads plus
  the multiply.
- DRAIN skips the drain passes and keeps one accumulator store plus a shared read to keep the
  MMA live (OpCooperativeMatrixStoreKHR 2 -> 1, three epilogue fetches and one barrier gone).
  The parent's second accumulator (`MMAS_PER_SG_N` = 2) is dead in this build, but its multiply
  chain is NOT eliminated in the generated SPIR-V (8 OpCooperativeMatrixMulAddKHR remain);
  whether the driver's ISA compiler removes it is unverified.

Each phase ran its own device session: two interleaved rounds of the ablated binary and the
canonical binary (un-ablated parent) on the same phase-timing collection (21 shapes, two arms),
`--validation_mode=none`, pinned clocks, canonical driver verified after every phase. The phase
share is (canonical parent - ablated parent) / canonical parent, both medians from the same
session. Ablation shares, two interleaved rounds per phase, locate only.

Protocol deviations from the session rules, disclosed: two rounds per phase instead of three
(note 40's reduction was accepted to keep the board schedule); no 60 s idle before a session and
no 10 s between runs (the loop script has no sleeps); the ablated binary always ran first within
each round, the order never alternated; no MIF/INT pin readback at session time - the during-run
clock evidence is the observed GPU clock, at the pinned maximum in every entry of all runs, and
a post-session readback (GPU fully pinned, MIF/INT maxes pinned) is recorded in the local
artifacts. The runs are recorded in `task5-2/phase_runs.csv` (local) with the
session/valid/binary/driver columns.

Build verification: the five ablated benches have five distinct digests, and the default-env
rebuild at committed HEAD reproduces the canonical bench digest byte for byte - the
instrumentation changes nothing in the shipped shaders. The `pw_tiled_4x8` control arm is not
instrumented in either binary and agrees between the two binaries of each session (mean
deviation 0.8-1.4% per phase; worst single cell 15.3%, a small-time cell, consistent with the
two-run median spread). An earlier independent pass that compared the ablated binaries against
the 4.0 session medians (cross-session) reproduced every phase share within 0.6 percentage
points.

Phase cost as a share of the parent route time (per-shape values in
`phase_shares_xs.csv`, local artifact; positive = ablation made the kernel faster):

| ablation | mean | median | what its removal covers |
|---|---:|---:|---|
| STAGE | 40.7% | 42.9% | shared stores + the fetches (dead-code eliminated with them); 24.6-51.2% across shapes |
| MMA | 37.4% | 37.7% | shared reads + multiply; 22.4-55.5% across shapes |
| DRAIN | 18.5% | 22.6% | epilogue; concentrated on the 56x56-plane cells (23-42%), near zero on several 196/289/784-plane cells |
| FETCH | 6.2% | 5.4% | input/weight loads; 13.6% at most |
| SYNC | 1.4% | 1.3% | K-loop barriers; noise-level, several cells negative |

The five shares are independent measurements against the full kernel and do not add to 100%
(their per-shape sum averages 104%, range 76-122%: removing one phase also hides stalls that
another phase would cause).

Reading, with the confounds stated: STAGE and MMA are entangled with their neighbours - the
stage share includes the fetch (~6%), the mma share includes the shared-memory reads - so the
measurements do not cleanly separate "staging" from "multiply", and the earlier reading
"the bottleneck is staging, not the matrix multiply" is withdrawn as unsupported. What the
measurements do establish: the K-loop barrier is free (SYNC ~1%), the epilogue costs a fifth on
small-plane shapes, and the remaining kernel time is split between the fetch+staging-store path
(~41% together) and the shared-read+multiply path (~37% together) without a clean split between
them; where the true bottleneck lies inside that data path is UNVERIFIED, and candidate choice
from these numbers alone would be premature. The debug-only ablation code contains deliberate
races and is never shipped: ABLATE_DRAIN stores an accumulator to shared without a barrier after
the last multiply's reads (write-after-read) and, in multi-subgroup variants, every subgroup
stores the same elements; its `imageStore` guard is a many-writer same-value race; ABLATE_SYNC
removes the barriers by design.

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

**Across precisions** (same ten models, same twelve-execution protocol; INT8 is nine models -
w2l has no quantized export recipe; mean of per-model shares):

| | FP16 | FP32 | INT8 |
|---|---:|---:|---:|
| convolution | 88.6% | 89.0% | 78.8% |
| top convolution shader | `gemm_texture2d` 32.5% | `gemm_texture2d` 31.1% | `q8ta_conv2d_pw` 65.2% |
| second | `pw_tiled_bufw` 16.3% | `pw_tiled_4x8_bufw` 25.8% | `q8ta_im2col` 8.1% |
| third | `2x2x8_border` 12.0% | `2x2x8_border` 11.2% | `q8ta_dequantize` 5.0% |

The convolution share is stable across precisions (FP16 88.6%, FP32 89.0%), so the campaign's
opportunity is not an FP16-routing artifact. The FP32 mix shifts toward the baseline pointwise
shader: `pw_tiled_4x8_bufw` holds 25.8% of FP32 shader time (5.4% in FP16) because the
buffer-write pointwise variant is FP16-only - in FP32 deployments the pointwise convolutions
run the THRESHOLDS baseline, which the 4.0 parent replaces. INT8 1x1 convolution
(`q8ta_conv2d_pw`, 65.2%) is out of scope by the SPEC stage table; the share is recorded to
quantify what a future INT8 stage would address.

## 7.4 End-to-end projection (1x1 and kxk parents)

Projection, lower bound. Each measured ratio is applied to every ETDump layer whose shape exactly
matches a measured case; every other layer keeps its ETDump time. The reduction is of the whole
model's GPU shader time, not only of convolution. This is a projection computed from the local
ETDump aggregation and the two run records; it is not a model run. Three columns:

- **1x1 only**: the 4.0 parent ratio (`pw_tiled_4x8` / parent, per-cell median of
  `gpu_timestamp_route_us`) on the 21 measured 1x1 shapes, matched by `(hout*wout, Cin, Cout)`
  at stride 1 (stride-2 1x1 layers do not match).
- **1x1 + kxk**: additionally the fp32 kxk parent (`amd_t16x32_f32_alc_sk2_i2c`) ratio vs `auto`
  on the 14 measured kxk shapes, matched by kernel, stride, Cin, Cout and input/output planes.
- **1x1 + kxk vs best stock**: same matches, but each kxk layer is credited
  min(auto, direct, im2col) / parent. This separates the cooperative-matrix gain from what a
  routing fix alone (today's `auto` picks the slow `im2col` route on M51) would deliver.

Shapes where the parent loses stay on the stock route, as a routing rule would: vs `auto` the
three 3x3 64-channel large-plane shapes (0.45-0.67), vs best stock additionally
`3x3 1024-256 28x28` (0.83 of `direct`; dl3's non-dilated layer).

| model | 1x1 matched | 1x1 reduction | kxk matched | 1x1 + kxk | 1x1 + kxk vs best stock |
|---|---:|---:|---:|---:|---:|
| resnet18 | 0 | 0.00% | 4 | 73.36% | 54.27% |
| resnet50 | 9 | 4.55% | 4 | 60.45% | 45.18% |
| mv2 | 3 | 6.77% | 0 | 6.77% | 6.77% |
| mv3 | 1 | 1.80% | 0 | 1.80% | 1.80% |
| ic3 | 0 | 0.00% | 1 | 11.39% | 9.07% |
| ic4 | 0 | 0.00% | 2 | 16.47% | 9.99% |
| densenet161 | 0 | 0.00% | 2 | 13.26% | 13.25% |
| dl3 | 5 | 0.68% | 2 | 9.71% | 1.47% |

The 1x1-only bound is small because exact-shape matching is strict: only shapes present in the
21-case timing session count, and several models (resnet18: no 1x1 layers; densenet161: 1x1
layers all outside the measured set) match none. The kxk column changes the picture: resnet18/50
match their im2col-routed 3x3 stack - 128/256/512 channels at 28/14/7 planes, ratios 3.7-11.4x
vs `auto` - plus one stride-2 layer each (resnet18 256->512, resnet50 512->512); the remaining
stride-2 convolutions have no measured case. mv2/mv3 gain nothing from kxk because their 3x3
convolutions are depthwise (out of scope by the SPEC stage table). The vs-best-stock column shows
the split the coordinator asked for: roughly a third to a half of the resnet18/50 headline is a
routing fix (switching `auto`'s im2col picks to `direct`), the rest is the cooperative-matrix
gain over the best stock route (1.2-5.9x on the matched layers). dl3's kxk gain shrinks to 1.5%
because its matched 1024->256 layer loses to `direct` and only the 128->128 layer (1.2x vs
`direct`) is credited. The projection bounds what routing the measured shapes into these models
would save today; a session covering each model's own shape census (939 conv configs in the
ETDump aggregation) would raise the 1x1 column toward the 7.2 opportunity (1x1 is 15.4% of FP16
conv GPU time). Recomputation: `tools/e2e_projection.py` over `survey-conv-shapes.csv`,
`survey-runs.csv`, `timing/runs.csv` and `kk-gate/kk_runs.csv` (local artifacts).

## 9.2 The im2col GEMM stage (k x k gate and timed session)

Gate collections: the desktop layer table's eight shapes plus the six M51-heavy shapes of 7.2
(fourteen shapes, bias and fused-residual epilogue each; `3x3 1024-256 28x28` is an actual dl3
layer - the non-dilated 3x3 1024->256 at a 28x28 plane, im2col-routed in the 7.2 census - not
only the GEMM-equivalent of dl3's dilated 2048->256 layers), full grid and reduced grid, ten
repetitions, plus one run of the correctness collection. Driver canonical and clocks pinned
throughout; every stderr empty.

**Gate verdict: all 32 `amd`/`nvidia` `_i2c` variants pass 10/10 on both grids** — zero failing
dispatches across the twenty runs (896 coopmat dispatches per run). Every variant creates its
pipeline on the device (no shared-memory overflow skips), and every K of the fourteen shapes is
divisible by both families' K steps.

**Production-path accuracy finding.** The failures in these runs are not coopmat failures: the
`direct` route — which `auto` selects on five of the shapes — fails validation (atol 0.1,
rtol 0.01) on four of those five: `3x3 64 56x56` (one of the desktop shapes), `3x3 64 224x224`,
`3x3 64-256 224x224` (both epilogues) and `3x3 192-48 28x28`, with maximum absolute errors of
0.14-0.53 against the 0.1 tolerance (1.04 on the correctness collection's gather-grid-limit
case). The `im2col` route passes everywhere. Mismatch counts scale with the output pixel count,
which is why the reduced grid passes. The pattern is consistent with an fp16-accumulation
overshoot of the direct shader (the 2.5 record shows this class exceeding fp16 accumulation
range), but the cause on this device is **UNVERIFIED**. The baseline of this section is the
routed path as it is today, this accuracy issue included: on the K=576-class shapes the
incumbent fast path is out of tolerance on this device, and the tolerance-holding incumbent
route there is `im2col`.

**Timed session:** five runs with rotated arm order (auto, direct, im2col, the 32 variants),
per-case p50 of `gpu_timestamp_route_us`, per-cell median over the five runs, A/A opens of the
fixed `im2col` route before and after. A/A geomeans 0.9996 and 1.0021 (one cell, `3x3 s2
256-512 14x14`, at +2.1% in the second open); observed clocks match the pin in every case. One
cell (`3x3 192-48 28x28`) carries a 6% cross-run spread of the baseline arm; its noise band is
widened accordingly. The run records carry the full SPEC columns (session, run order, validity
with reason, and the binary and driver digests, the driver verified canonical after every run).
The timing is the whole convolution, gather included.

**Baselines.** `auto` is the headline baseline (today's route, and the THRESHOLDS baseline), but
it is not the best stock route: the geometric mean of time(auto)/time(direct) over the fourteen
shapes is 2.03 — **`direct` is about 2x faster than `auto`**. On the nine shapes where `auto`
picks `im2col`, `direct` is 1.9-5.5x faster (`3x3 448-384 5x5` 5.5x, `3x3 s2 256-512 14x14`
5.0x, `3x3 512 7x7` 1.9x); on the five where `auto` already resolves to `direct`, the ratio is
1.00. The best stock route is `direct` on every shape, so a routing fix alone would already
double the k-k convolution time; every coopmat ratio against `auto` below overstates the gain
over that routing fix by about 2x. Ratios are therefore given against `auto` and against the
**best stock route per shape** (min of auto, direct, im2col - `direct` everywhere).

**Parent selection (fp32 accumulation, by the 1x1 rule):**

> **k x k parent: `amd_t16x32_f32_alc_sk2_i2c_half`.** Geomean **3.39x over `auto`**,
> **1.67x over the best stock route**, covering all fourteen shapes.

**fp16-accumulation best, reported separately:** `amd_t128x128_f16_alc_sk4_i2c_half`, geomean
**3.69x over `auto`**, **1.82x over the best stock route**, 5.66x over `im2col`, covering all
fourteen shapes. Like every fp16-accumulation arm it fails the adversarial fp16-limits cases of
the 2.5 record, which is why the parent is the fp32 arm.

**Per-shape table** (time(baseline)/time(variant); the parent column and the fp16 best column;
`best stock` is `direct` on every shape; band widened to 6% on `3x3 192-48 28x28`):

| shape | auto / parent | best stock / parent | best stock / fp16 best | band |
|---|---:|---:|---:|---:|
| `1x7 192-224 17x17` | 4.62 | 1.88 | 2.28 | 2% |
| `1x7 224-256 17x17` | 4.26 | 1.92 | 2.56 | 2% |
| `3x3 1024-256 28x28` (dl3 layer, non-dilated) | 1.97 | 0.83 | 1.73 | 2% |
| `3x3 128 28x28` | 3.65 | 1.21 | 1.50 | 2% |
| `3x3 192-48 14x14` | 2.35 | 2.35 | 2.16 | 2% |
| `3x3 192-48 28x28` | 1.80 | 1.80 | 1.62 | 6% |
| `3x3 256 14x14` | 7.28 | 2.58 | 3.58 | 2% |
| `3x3 448-384 5x5` | 23.32 | 4.21 | 2.53 | 2% |
| `3x3 512 7x7` | 11.42 | 5.94 | 6.36 | 2% |
| `3x3 64 224x224` | 0.45 | 0.45 | 0.40 | 2% |
| `3x3 64 56x56` | 0.61 | 0.61 | 0.53 | 2% |
| `3x3 64-256 224x224` | 0.67 | 0.67 | 0.87 | 2% |
| `3x3 s2 256-512 14x14` | 10.84 | 2.19 | 2.12 | 2% |
| `3x3 s2 512 14x14` | 11.43 | 3.38 | 3.60 | 2% |

The parent wins ten of fourteen cells against the best stock route beyond the noise band, and
the fp16 best eleven of fourteen. The losses are the Cin=64 large-plane shapes: `3x3 64
224x224`, `3x3 64 56x56` and `3x3 64-256 224x224` run at **0.45-0.67x of stock** under the
parent (0.40-0.87x under the fp16 best) — at Cin=64 the K=576 GEMM is small relative to the
output plane and the im2col gather dominates; the parent additionally loses `3x3 1024-256
28x28` (0.83x), the one shape where the fp16 best still wins (1.73x). Against the
tolerance-holding incumbent route (`im2col`) both variants win all fourteen cells, by
1.5x-14x.

**Routing implication.** A routing rule that adopts the coopmat variants must keep the
Cin=64-large-plane shapes on `direct` - there the coopmat route is slower than stock. But the
`direct` precision finding above applies to exactly those shapes: the route the rule must keep
is out of tolerance there today. Adoption on those shapes is therefore blocked on resolving the
`direct` accuracy issue; until then the choice there is `im2col` (in tolerance, 1.5-2.4x slower
than the coopmat route) or the out-of-tolerance fast path.

The next arms: fp32 `amd_t128x128_f32_alc_sk4_i2c_half` (3.38x over `auto`, 1.67x over best
stock) and `nvidia_t128x128_f32_alc_sk4_i2c_half` (3.38x / 1.67x); fp16
`nvidia_t128x128_f16_alc_sk4_i2c_half` (3.50x / 1.73x), `amd_t128x128_f16_alc_sk2_i2c_half`
(3.48x / 1.71x).
