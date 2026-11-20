# conv2d cooperative-matrix route: M51 stage-1 results for the desktop side

Sanitized note (TASKS 8.2): ratios and shapes only - no device, board, host or driver
identifiers, no absolute times. Full record: `RESULTS.md` in this directory.

## What to adopt

Two variants, both AMD-family cooperative-matrix (`coopmatKHR`), fp16 in/out with fp32
accumulation, selected by the campaign's correctness gate (fp32 accumulation; every fp16-
accumulation variant fails the adversarial fp16-limits record cases):

- **1x1 convolutions: `conv2d_pw_coopmat_amd_t16x32_f32_al_half`** - tile 16x32, async loads.
  Geomean **2.10x** over the pointwise baseline (`pw_tiled_4x8`) and **1.85x** over the
  current auto routing on the 21-shape selection set, winning every shape of that set; the
  ten heaviest workload shapes measured later (7.3) add six wins and four dense-plane
  exceptions - see the routing rule.
- **k x k convolutions: `conv2d_pw_coopmat_amd_t16x32_f32_alc_sk2_i2c_half`** - tile 16x32,
  async loads, K-split 2, im2col input staging. Geomean **3.39x** over auto and **1.67x** over
  the best stock route per shape, covering 14/14 measured k x k shapes.

An fp16-accumulation k x k variant (`amd_t128x128_f16_alc_sk4_i2c_half`) is faster still
(3.69x / 1.82x) and wins one shape the fp32 parent loses, but is reported separately for the
accumulation reason above; adopt it only where fp16 accumulation is acceptable.

Every gated K (channel count x kernel area) is divisible by both families' K steps, and the
largest descriptor's shared memory (37 KiB, `t128x128`) fits the 64 KiB cap, so no shape needs
a fallback on shared memory.

## 1x1 ratios (baseline / variant; 31 shapes)

Selection set (21 shapes):

| shape ( plane pixels, Cout, Cin ) | pointwise baseline | auto |
|---|---:|---:|
| `49,160,960` | 4.74 | 4.03 |
| `49,2048,512` | 1.20 | 1.70 |
| `49,512,2048` | 4.15 | 2.93 |
| `49,512,4608` | 5.50 | 4.09 |
| `49,96,576` | 4.84 | 2.97 |
| `196,1024,256` | 1.11 | 1.11 |
| `196,256,1024` | 2.53 | 2.03 |
| `196,256,2304` | 2.39 | 1.69 |
| `196,512,1024` | 1.40 | 1.51 |
| `196,64,384` | 3.53 | 2.48 |
| `196,96,576` | 4.16 | 2.62 |
| `25,384,4032` | 4.81 | 3.76 |
| `289,224,1344` | 1.87 | 1.44 |
| `784,128,1152` | 1.47 | 1.56 |
| `784,128,512` | 1.50 | 1.52 |
| `784,256,512` | 1.05 | 1.06 |
| `784,512,128` | 1.25 | 1.24 |
| `3136,128,256` | 1.04 | 1.04 |
| `3136,256,64` | 1.41 | 1.41 |
| `3136,64,256` | 1.22 | 1.22 |
| `3136,64,64` | 1.48 | 1.48 |

Workload top-up, ten heaviest uncovered shapes (7.3):

| shape ( plane pixels, Cout, Cin ) | pointwise baseline | auto |
|---|---:|---:|
| `64,256,1536` | 4.60 | 3.29 |
| `64,384,1536` | 4.32 | 3.11 |
| `144,192,768` | 3.76 | 2.31 |
| `289,192,1024` | 2.40 | 1.97 |
| `289,384,1024` | 1.31 | 1.60 |
| `784,256,1024` | 0.99 | 0.99 |
| `784,512,2048` | 0.77 | 0.76 |
| `784,1024,256` | 1.06 | 1.06 |
| `784,2048,512` | 0.84 | 0.84 |
| `784,2048,1024` | 0.79 | 0.79 |

## k x k ratios (baseline / variant; 14 shapes)

`auto` resolves to `im2col` on nine of the shapes and to `direct` on five; `best stock` is the
fastest of auto / direct / im2col per shape.

| shape | auto / k x k variant | best stock / k x k variant | best stock / fp16 best |
|---|---:|---:|---:|
| `1x7 192-224 17x17` | 4.62 | 1.89 | 2.29 |
| `1x7 224-256 17x17` | 4.28 | 1.91 | 2.56 |
| `3x3 1024-256 28x28` | 1.96 | 0.82 | 1.73 |
| `3x3 128 28x28` | 3.65 | 1.21 | 1.51 |
| `3x3 192-48 14x14` | 2.35 | 2.35 | 2.16 |
| `3x3 192-48 28x28` | 1.80 | 1.78 | 1.60 |
| `3x3 256 14x14` | 7.27 | 2.58 | 3.59 |
| `3x3 448-384 5x5` | 23.36 | 4.19 | 2.51 |
| `3x3 512 7x7` | 11.47 | 5.94 | 6.33 |
| `3x3 64 224x224` | 0.45 | 0.45 | 0.40 |
| `3x3 64 56x56` | 0.61 | 0.61 | 0.53 |
| `3x3 64-256 224x224` | 0.67 | 0.67 | 0.87 |
| `3x3 s2 256-512 14x14` | 10.91 | 2.25 | 2.16 |
| `3x3 s2 512 14x14` | 11.42 | 3.37 | 3.59 |

## Routing rule

- 1x1: adopt the coopmat variant except the **dense 784-plane class** where both channel
  counts are large (`784,2048,512`, `784,2048,1024`, `784,512,2048` run at 0.77-0.84x of the
  pointwise baseline, and `784,256,1024` is a borderline tie at 0.99x) - there the baseline
  pointwise shader, which the auto routing already selects, is the faster route; keep those on
  stock. Every other measured 1x1 shape wins (1.06-4.81x).
- k x k: adopt the coopmat variant except **Cin = 64 with large output planes** (`3x3 64
  224x224`, `3x3 64 56x56`, `3x3 64-256 224x224` run at 0.45-0.67x of stock under the
  variant) - there the K = 576 GEMM is small relative to the output plane and the im2col
  gather dominates. Keep those on the stock route.
- The fp32 parent also loses `3x3 1024-256 28x28` (0.82x); the fp16 best wins it (1.73x).

One caveat for the fallback arm: on the measurement device the `direct` route exceeded the
validation tolerance on four K = 576-class shapes (max errors of 1.4-5.3x the 0.1 absolute
tolerance), while `im2col` held tolerance everywhere. Verify the tolerance of whichever stock
route the rule falls back to on your targets before shipping it.

## Where the time is (for choosing what to build next)

Ranked by share of convolution time in the ten-model workload: 3x3 s1 45.1%, 1x1 15.4%,
1x7/7x1 14.4%, 3x3 dilated 7.4%, 3x3 s2 5.9%, 1x3/3x1 5.7%, 1D 5.1%. The two adopted variants
cover the 1x1 and non-dilated k x k families; the open families are the Cin=64 large-plane
direct class, dilated 3x3 (the im2col routing rejects dilation != 1), and the 1x3/3x1 class.
