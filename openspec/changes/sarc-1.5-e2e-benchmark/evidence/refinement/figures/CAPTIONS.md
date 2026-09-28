# WMMA refinement figures: captions and sources

Regenerate everything with `./make_all.sh`, which uses `uv run --with matplotlib --with pandas --with numpy`.

- Kernel data: `../refinement.csv`. It is read-only; `../refinement.md` explains what it means.
- Measured end-to-end data (r2_e2e_measured, r4): `report/refine/cells.csv` (one row per GPU × model × scheme) and `report/refine/runs_all.csv` (every run). In these files:
  - `stock_*` = the previous best WMMA kernels (release-1.4 commits, run with the opt-in env they needed; 780M = default);
  - `sarc_*` = the re-tuned kernels (defaults).
- Step values: `igpu-roofline/docs/*-WMMA-LESSONS.md`, cited by line below.
- All kernel numbers are for Llama 3.2 1B, 3.2 3B and 3.1 8B prefill GEMMs at M = 2048, on the texture3d model path, on the release-1.4 branches.
- GPU colours (colour-blind-safe set checked with the palette validator; same in every figure):
  - 780M: vermillion;
  - B580: blue;
  - B70: violet;
  - 4070 Ti SUPER: green;
  - Orin: amber.
- Grey always means the previous kernel.

## r1_kernel_gain (print, 7.2 in) and r1_kernel_gain_slide (16:9)

**Additional prefill-GEMM speedup from the two-day re-tuning, over the previous best WMMA kernels.**

How to read it:
- Each bar is the geometric mean of the tuned/previous kernel-time ratio over the 12 projection shapes (4 per model).
  - Orin 4w uses 11 shapes: the original 8B `w2` result is numerically invalid.
- Every cell is the median of 3 repeats.
- Grey marks the previous kernel (1×). Colour is the added gain.
- Markers give the per-model time-weighted speedup, using per-layer call counts 2·wq_wo + 2·wk_wv + 2·w1_w3 + 1·w2.
- The thin line is the per-shape min–max.

| GPU | 4w geomean (1B / 3B / 8B) [range] | 8da4w geomean (1B / 3B / 8B) [range] |
|---|---|---|
| Radeon 780M | 1.30 (1.30 / 1.31 / 1.30) [1.27–1.32] | 1.00 (0.99 / 1.00 / 1.00) [0.99–1.00], unchanged kernel |
| Arc B580 †‡ | 2.49 (2.23 / 2.87 / 2.39) [2.04–3.74] | 1.58 (1.60 / 1.58 / 1.69) [1.38–1.77] |
| Arc Pro B70 † | 2.54 (2.21 / 2.91 / 2.27) [2.04–4.30] | 1.69 (1.75 / 1.64 / 1.60) [1.54–1.92] |
| RTX 4070 Ti SUPER † | **0.96** (0.96 / 0.92 / 0.89) [0.87–1.11] | 1.34 (1.31 / 1.37 / 1.38) [1.26–1.47] |
| Jetson Orin Nano † | 1.07 (1.06 / 1.07 / 1.07) [1.06–1.07] | 2.32 (2.23 / 2.27 / 2.42) [2.12–2.49] |

The 4070 Ti 4w bar is a regression, drawn hatched:
- The old fp16-accumulate kernel failed the 8B `w2` check (K = 14336): max |err| 1.49 against a 0.5 tolerance.
- The new tiles alone gave 1.036×. Per-group fp32 accumulation (`ACC_GROUP_FP32`) then cost 0.924×.

The 780M 8da4w kernel is identical before and after. The int8 matrix roof is only 1.23× the int8 dot roof on RDNA3.

Footnotes:
- † The previous WMMA kernel was opt-in (`ET_VK_TEXTURE_COOPMAT=1`, `ET_VK_COOPMAT_ANY_DEVICE=1`, or explicit enabling on Orin), so the default path ran the tiled kernel. The re-tuning also made the tuned kernels the default.
- ‡ The B580 "before" is the B70 branch's Xe2 tiles.

Source: `refinement.csv`, rows `derived` "geomean over N shapes" and "per-layer call counts". These are cross-checked in code against the per-shape `microbench kernel` rows.

## r2_e2e_measured (print) and r2_e2e_measured_slide (16:9)

**Measured end-to-end prefill speedup from the re-tuning, per GPU, scheme and model.**

How to read it:
- Bars show the measured speedup: median re-tuned tok/s ÷ median previous-best tok/s.
- Each build was run 5 times on a real-text 2048-token prompt. The two builds' runs were interleaved, co-tenant services were stopped, and each run had a cool-down.
- Error bars are the approximate 95 % paired bootstrap CI (`speedup_ci_lo/hi` in `cells.csv`).
- Hollow markers are the earlier Amdahl projection from kernel times (`r2_e2e_projection`). There is no projection for Orin 8B.
- **The projection agreed with the measurement within 3.1 % in every cell** (worst: B70 8B 4w, 1.644 projected vs 1.594 measured). That validates the kernel-level numbers in r1.
- Hatched bars: 4070 Ti 4w, the accuracy fix. The 780M 8da4w kernel is unchanged.
- † On B580, B70, 4070 Ti and Orin, the previous kernels were opt-in, and they were measured here with their opt-in env.
  - The env really switched the model path: in `refine-e2e-2026-09-28/raw/<gpu>/wrap.out`, the 1B 4w previous build ran with vs without the env at:
    - B580: 5802 vs 2576 tok/s;
    - B70: 8192 vs 3670;
    - 4070 Ti: 20078 vs 5596;
    - Orin: 876 vs 172;
    - 780M: identical (2330), because WMMA was already its default.

Measured speedup [paired 95 % CI], 1B / 3B / 8B:

| GPU | 4w | 8da4w |
|---|---|---|
| 780M | 1.16 [1.15, 1.16] / 1.18 [1.18, 1.18] / 1.21 [1.20, 1.21] | 1.00 / 1.00 / 1.00 (unchanged) |
| B580 | 1.45 [1.45, 1.45] / 1.78 [1.78, 1.80] / 1.71 [1.71, 1.72] | 1.19 / 1.21 / 1.34 |
| B70 | 1.38 [1.36, 1.42] / 1.73 [1.72, 1.74] / 1.59 [1.58, 1.60] | 1.18 / 1.19 / 1.28 |
| 4070 Ti SUPER | 1.00 [0.98, 1.00] / 0.97 [0.96, 0.97] / 0.95 [0.94, 0.95] (accuracy fix) | 1.09 / 1.14 / 1.17 |
| Orin | 1.02 / 1.02 / 1.00 | 1.43 / 1.51 / 1.71 |

Caveats:
- Prefill times are logged with 1 ms resolution; tok/s = 2048 / ms. At about 100 ms (4070 Ti 1B) one step is about 1 %, so several CIs and spreads in those cells are limited by resolution. The 4070 Ti 1B 4w medians are identical (102 ms).
- Two B70 1B cells have a repeat spread above 3 %: 4w 3.7 % / 3.4 %, and 8da4w 3.0 % stock.
- One 4070 Ti 8B 8da4w stock run was rejected (rc = 134) and replaced, leaving 4 pairs for its CI.

## r2_e2e_projection (superseded by r2_e2e_measured; kept for the record)

**Projected end-to-end prefill speedup (2048-token prompt) from the re-tuning, per GPU, scheme and model.**

How the projection works:
- Bars are Amdahl projections: before ≈ measured tuned e2e prefill + layers × Σ calls × (before − after) kernel time.
- Tuned e2e comes from the release-1.4 e2e study (3 repeats).
- Black diamonds are **measured** Orin e2e speedups (original → tuned, 3 repeats each, same PTE and prompt).
- The measured Orin values agree with the projections within 0.4 %, which validates the method:

  | Orin | measured | projected |
  |---|---:|---:|
  | 1B 4w | 1.016 | 1.018 |
  | 3B 4w | 1.020 | 1.021 |
  | 1B 8da4w | 1.426 | 1.424 |
  | 3B 8da4w | 1.513 | 1.507 |

- Orin 8B e2e was stopped by the memory guard. This is not an OOM finding.

Projected values (1B / 3B / 8B):

| GPU | 4w | 8da4w |
|---|---|---|
| 780M | 1.15 / 1.19 / 1.20 | 1.00 / 1.00 / 1.00 |
| B580 | 1.44 / 1.78 / 1.73 | 1.18 / 1.21 / 1.33 |
| B70 | 1.41 / 1.76 / 1.64 | 1.20 / 1.22 / 1.28 |
| 4070 Ti SUPER | 0.99 / 0.97 / 0.94 (regression) | 1.09 / 1.13 / 1.17 |
| Orin | 1.02 / 1.02 / — | 1.42 / 1.51 / — |

Caveats:
- Projections are not measurements, except on Orin.
- The earlier e2e study found that Amdahl *under*-predicts Xe2 4w (B70 8B tiled→tuned: 4.81× measured against 3.31× predicted), so the Xe2 4w bars are the least certain.
- For the † GPUs the deployed default path changed from tiled to tuned. The measured tiled → tuned e2e gains on release 1.4 were:
  - 4070 Ti: 3.11–5.11×;
  - B70: 1.48–4.81×;
  - B580: 1.43–4.73×;
  - 780M: 1.18–4.52×.

  Source: `WMMA-STUDY-TAKEAWAYS.md` L35–40. These are not plotted.

Source: `refinement.csv`, rows `derived (e2e projection)` and `e2e` (2048-token rows).

## r4_tok_s_before_after

**Measured prefill throughput before → after the re-tuning (tokens/s, log scale).**

- One panel per model; one row per GPU × scheme.
- Grey dot: previous best WMMA kernel. Coloured dot: re-tuned. Labels give the median tok/s before → after (median of 5 interleaved runs, 2048-token real-text prompt).
- Source: `report/refine/cells.csv`, columns `stock_median` and `sarc_median`.

8B examples:
- B580 4w: 980 → 1680 tok/s;
- B70 4w: 1488 → 2373;
- 780M 4w: 426 → 516;
- 4070 Ti 8da4w: 4267 → 4971;
- Orin 8da4w: 99 → 170;
- 4070 Ti 4w: 4708 → 4452 (accuracy fix).

## r3_steps

**The work: each profiler- or roofline-guided change, the evidence that motivated it, and its measured effect.**

How to read it:
- Each change is one row. Its evidence is in italics; the numbers quoted from the doc are on the right.
- The bold bar in GPU colour is the confirmed net from `refinement.csv` (texture3d, 3 repeats, geomean of shapes).
- The x-axis is log scale.

Bar types:

| Bar | Meaning |
|---|---|
| Filled, light | Chained step. Measured on the same base, so the steps multiply to the net: 780M 4w 1.09 × 1.19 = 1.30; 4070 Ti 4w 1.036 × 0.924 = 0.957. |
| Outlined | Factor measured on its own base (single-run 1B screen, control configuration or single shape). These do not multiply to the net. |
| Dashed | The doc gives only a range. |
| Hatched | Accuracy fix: slower but correct. |
| Text only | The doc gives no speedup for that step. |

### Step values and sources

`docs/` = `/home/doremy/Desktop/igpu-roofline/docs/`.

**Radeon 780M**

| Step | Value | Source |
|---|---|---|
| ACC_FP32 | texture3d ×1.09; VALU 637 → 330 | `780M-WMMA-LESSONS.md` L69 |
| CSH_IN_ASH | texture3d ×1.19; 4 → 6 waves/SIMD | L70 |
| One default tile | 3B shapes 8.2 → 10.8 TFLOP/s; no aggregate number | L71 |
| 8da4w not changed | int8 roof 1.23× the dot roof | L107–109 |
| Net 1.30 / 1.00 | | L26, L28 |

**Arc B580**

| Step | Value | Source |
|---|---|---|
| s16 + 32×32 subgroup tiles | 1B 19.2 → 36.9 TFLOP/s (×1.92) | `XE2-WMMA-LESSONS.md` L48–49 |
| WG_TILE_M = 256 | 1B 32.4 → 52.7 TOP/s (×1.63) | L50 |
| FRAG_LAYOUT | +12 %, buffer only | L51 |
| IMG_A | texture3d +6–29 % | L52 |
| Default + buffer variants | no speed number | L53–54 |
| Net | computed from the CSV (the doc gives only WMMA/tiled, L39) | |

**Arc Pro B70**

| Step | Value | Source |
|---|---|---|
| B580 tiles re-screened | 4w 36.1 → 62.5 TFLOP/s (×1.73), 8da4w 53.1 → 87.7 TOP/s (×1.65); single-run 1B | `XE2-WMMA-LESSONS.md` L103–106 |
| G31 8B-only 8da4w tile removed | old tile 1.56–1.73× slower | L106–108 |
| FRAG_LAYOUT | +0–4 % | L126–128 |
| Net 2.54 / 1.69 | | L115–116 |

**RTX 4070 Ti SUPER**

| Step | Value | Source |
|---|---|---|
| Ablation table | | `4070TI-WMMA-LESSONS.md` L71–80 |
| WG_TILE_K 32 → 64 | texture3d ×1.10 (buffer ×1.85) | L84 |
| A_RAW | 3B wq_wo 303 → 252 µs (×1.20) | L85 |
| B_PAIR, CSH_IN_ASH | no separate number | L86–87 |
| 4w tiles | ×1.036 | CSV `derived` step 1; the doc rounds it to 1.04× (L40, L109–115) |
| ACC_GROUP_FP32 | ×0.924 | CSV step 2; L40, L121–135 |
| Net 1.34 / 0.96 | | L38, L40 |

**Jetson Orin Nano**

| Step | Value | Source |
|---|---|---|
| Nsight Tensor Active | 9.86 % → 20.85 % | `JETSON-WMMA-LESSONS.md` L95–101 |
| raw A + paired B + LDS output reuse | 2.31×, in a 128×64/K32/g42 control | L158–160 |
| K32 → K64 | a further 1.076× (control) | L160–161 |
| M256/g22 4w | ≈1.06×; M256/g42 0.88× | L164–168 |
| fp32 K32/g42 repair for K > 8192 | 8B w2 40.784 ms (invalid) → 45.881 ms (×0.889 on that shape; excluded from the net) | L133–138, L197–201 |
| Net 1.066 / 2.324 | | L194–195 |
