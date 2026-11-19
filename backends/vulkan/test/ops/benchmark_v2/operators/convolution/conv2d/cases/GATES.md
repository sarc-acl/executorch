# Decision gates of conv2d-pw-coopmat-k-latency

Fixed before any diagnostic or probe result of this change exists (design D2, spec "Decision gates are fixed
before data"). This file is committed in the worktree as
`backends/vulkan/test/ops/benchmark_v2/operators/convolution/conv2d/cases/GATES.md` and is not edited afterwards;
a rule that turns out to be unusable is recorded as that gate's outcome. Every outcome is computed by
`OUT/tools/gates.py` from the saved session summaries and written with its inputs to `OUT/gates-outcome.json`.

Names: `<v>` stands for `conv2d_pw_coopmat_<v>_half`. Shapes are (M, N, K) = (output pixels, output channels,
input channels). The four MobileNet shapes are S1 (49,160,960), S2 (196,64,384), S3 (196,96,576), S4 (49,96,576).

## 1. Parent variant per device

The first change's best variant on (196,96,576) in `PREV/<device>/summary.json` (tag `b7a93e9cba`, task 1.2,
`OUT/parents.json`):

| Device | Parent | K step |
|---|---|---:|
| Arc B580 (`b580`) | `xe2_t128x128_f16` | 16 |
| Arc Pro B70 (`b70`) | `xe2_t128x128_f32` | 16 |
| Radeon 780M (`rocky-780m`) | `amd_t16x32_f32` | 32 |
| RTX 4070 Ti SUPER (`4070ti`) | `amd_t16x32_f16` | 32 |

## 2. Measurement of every session

- A session is one `OUT/tools/session.sh` run on one device: deployment check, capability record, correctness
  collection and admission, then five runs of the session's performance file(s).
- Arms of a shape are listed in this fixed order: the parent (A), `auto`, `pw_tiled_4x8`, the session's other
  variants in registry order, the parent again (A'). With n arms (0-based), run r (r = 1..5) executes the arms of
  every shape in the order arm[(r + i) mod n], i = 0..n-1. Shapes follow in file order.
- Time of an arm on a shape: per run, the p50 of `gpu_timestamp_route_us` (start of the first dispatch to the end of
  the last, per invocation); the arm's time t is the median of its five per-run values. An arm is usable only with
  five passing runs of its expected shader sequence (`summarize.judge(..., runs=5, route=True)`); every other arm is
  listed with its reason and enters no gate.
- Difference of two arms X, Y on a shape: d(X, Y) = 2 |t_X - t_Y| / (t_X + t_Y); X is faster than Y when t_X < t_Y.

### Noise band and session rejection

- A/A values of a MobileNet shape: for each run r, 2 |A_r - A'_r| / (A_r + A'_r) with A_r, A'_r the per-run values
  of the two parent arms. The band of the shape is the largest of the five values, and not less than 0.02.
- A session whose band exceeds 0.10 on any MobileNet shape is rejected: its results move to
  `superseded/<session>-rejected-<n>/` and the session is repeated. After three rejected sessions of the same kind on
  a device, the device has outcome **not passed** for that session kind (`OUT/<device>/NOT-PASSED.md`), and every
  gate that needs that session is **not evaluable** on that device.
- A session with no usable A/A pair on a MobileNet shape has no band on it; its ratios on that shape enter no gate.
- Two arms with d < band of the shape (strictly less) are reported as **no change** on that shape.

## 3. Cases

- K sweep (`CASES/pw_coopmat_ksweep.json`, task 3.1): K in {32, 64, 128, 256, 576, 960} at the four (M,N) pairs
  P1 (49,96), P2 (49,160), P3 (196,64), P4 (196,96), routes `auto`, the device's parent, and its `_diag_once`
  variant; bias as in the MobileNet file. With the A/A pair of the parent on the four MobileNet shapes.
- MobileNet shapes S1..S4 with bias (`pw_coopmat_mobilenet.json` definitions), and for `_epi0` the same four shapes
  without bias (`nobias`).

## 4. G1: split-K screen (B580 and 4070 Ti)

Inputs: the diagnostic session of the device (`OUT/<device>/diag/session.json`).

1. For each pair P (M,N), fit t = a + b s by least squares over the six K-sweep points of the parent arm, with
   s = ceil(K / k_step) (k_step of the device's parent). R^2 = 1 - SS_res / SS_tot. P is **usable** when R^2 >= 0.95.
   A pair with fewer than six usable K points is not usable.
2. Each MobileNet shape belongs to the pair of its (M,N): S1 -> P2, S2 -> P3, S3 -> P4, S4 -> P1. For a usable pair
   the step share of its shape is b s / (a + b s) at that shape's K.
3. The device **passes** when at least three pairs are usable and the median of the step shares of the usable
   pairs (the mean of the two middle values when four) is >= 0.50. It **fails** when at least three pairs are usable
   and the median is < 0.50. Fewer than three usable pairs is **inconclusive**. A device whose diagnostic session was
   not passed (section 2) is **not evaluable**.
4. The split-K prototype (tasks 5.1-5.4) is built when B580 or the 4070 Ti passes. Fail, inconclusive and not
   evaluable do not build it; the report states each device's outcome.

## 5. G2: staging share (reported only)

For each usable pair P of a device whose `_diag_once` arm also fits with R^2 >= 0.95 over the same six points:
share_P = 1 - b_roof / b_parent. The device's staging share is the median over those pairs (none: not available).
A share >= 1/3 is reported as support for a later direct-load change. G2 decides nothing in this change.

## 6. G3: probe carry rule

Each probe variant is compared with its one named parent on its deciding device(s), in the probe session of that
device (`OUT/<device>/probe/session.json`), on the four MobileNet shapes (`_epi0`: their `nobias` versions). On a
shape, the probe **is faster** when t_probe < t_parent and d(probe, parent) > band of that shape (strictly
greater; on a `nobias` shape the band of the shape with bias). A shape where the probe or the parent has no usable
time, or where the session has no band, counts as not faster. A probe variant is **carried** to the fleet run when it
is faster on at least two of the four shapes on at least one of its deciding devices. Otherwise it is listed with its
ratios t_parent / t_probe and marked not measured on the other devices.

The accumulator decided on is the device parent's (B580, 4070 Ti: f16; 780M: f32); the other accumulator's probe
variants are timed and reported, and are carried exactly when their decided-accumulator sibling is.

| Probe | Probe variant(s) | Named parent | Deciding device(s) |
|---|---|---|---|
| Aligned-only, N tile 32 | `xe2_t16x32_f16_al` | `xe2_t16x32_f16` | B580 |
| Aligned-only, N tile 64 | `xe2_t64x64_f16_al` | `xe2_t64x64_f16` | B580 |
| Aligned-only, N tile 32 | `amd_t16x32_f16_al`, `nvidia_t32x32_f16_al` | `amd_t16x32_f16`, `nvidia_t32x32_f16` | 4070 Ti |
| Aligned-only, N tile 64 | `amd_t16x64_f16_al`, `nvidia_t16x64_f16_al` | `amd_t16x64_f16`, `nvidia_t16x64_f16` | 4070 Ti |
| Thread grid, Xe2 | `xe2_t64x64_f16_g4x2` | `xe2_t64x64_f16` | B580 |
| Thread grid, Xe2 | `xe2_t32x64_f16_g4x2` | `xe2_t32x64_f16` | B580 |
| Small tile, NVIDIA | `nvidia_t16x16_f16` | `nvidia_t16x32_f16` | 4070 Ti |
| Wide weight load | best aligned-only + `_w16` | best aligned-only | B580, 4070 Ti |
| Storage-image input | `xe2_t16x32_f16_al_img` | `xe2_t16x32_f16_al` | B580 |
| Epilogue skip | best aligned-only + `_epi0` | best aligned-only | B580, 4070 Ti |
| Column-major weights | `amd_t16x32_f32_bcm` | `amd_t16x32_f32` | 780M |

- An N tile of 64 is aligned only on S2 (N = 64), so those probes cannot be carried by this rule (at most one
  shape); they are timed and reported.
- **Best aligned-only** of a device: among its admitted N-tile-32 `_al` variants with the decided accumulator
  (B580: `xe2_t16x32_f16_al`; 4070 Ti: `amd_t16x32_f16_al` and `nvidia_t32x32_f16_al`), the one with the lowest t
  on S3 in the device's probe session. `_w16` and `_epi0` exist for each of these, so the comparison is fixed once
  the probe session has run.
- A probe variant that is not admitted on a deciding device (failed, not offered, or removed after a validation
  message) is not carried from that device; the reason is listed.
- K step 16 against 32 (design D4) and the direct-load raw-shader probe (task 4.8) are reported, not gated.

## 7. Split-K (only if G1 builds it)

- Bases, fixed now as the best plain variant per family on (196,96,576) in the first change's summaries: Xe2
  `xe2_t128x128` (B580), NVIDIA family `nvidia_t128x128` (4070 Ti), AMD family `amd_t16x32` (780M); both
  accumulators; slice counts 2 and 4 (`_sk2`, `_sk4`).
- Deciding devices: B580 for Xe2, 4070 Ti for the NVIDIA family. A split-K variant is carried by the rule of
  section 6 against its unsplit base, in the split-K session of that device (`OUT/<device>/splitk/session.json`,
  MobileNet shapes; the ResNet-50 file is timed and reported).
- The AMD-family slice count (task 5.4) is the one whose NVIDIA-family variant with the decided accumulator has the
  lower geometric mean of t over S1..S4 on the 4070 Ti; a tie (equal to 3 significant digits) takes 2.

## 8. Fleet session (task 6.3)

Arms per device: `auto`, `pw_tiled_4x8`, the parent (A, A'), the first change's best variant of every shape of the
three files on that device (`PREV/<device>/summary.json`), every carried variant with its named parent and both
accumulators of each, and on the 4070 Ti and the 780M the AMD- and NVIDIA-named 16x32 pair in both accumulators
(design D4). The band rule of section 2 applies to the fleet session.

## 9. Two-times verdict

Per device and shape of the fleet session: ratio = t_auto / t_best, with t_best the lowest t over every admitted
and timed cooperative-matrix arm (diagnostic arms excluded). Ratio >= 2.0: candidate; otherwise keep tiled. A shape
whose |ratio - 2.0| / 2.0 < band of the shape (MobileNet shapes; elsewhere the device's largest MobileNet band) is
flagged within noise of the bar and classified by its exact value. A device on which none of S1, S2, S3 reaches 2.0
is not ready for end-to-end work.
