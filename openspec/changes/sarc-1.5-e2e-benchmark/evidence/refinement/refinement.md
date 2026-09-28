# WMMA refinement: previous winner vs re-tuned kernel (2026-09-26/27)

Evidence for the claim: "In the last two days (2026-09-26/27), profiler-guided re-tuning of each GPU's
existing WMMA winner kernels gave additional speedup on top of the previous winners."

- Data: `refinement.csv`, with 366 rows.
- Generator: `build_refinement.py`. It is read-only over existing artifacts and was run without GPU access.
- Also written: `model_summary.json`.

## Verdict on the headline

The claim holds for the 780M (4w), the B70, the RTX 4070 Ti SUPER (8da4w) and the Orin. It needs two
exceptions and one wording caveat:

- **Regression: 4070 Ti SUPER 4w, net 0.96× (texture3d).** The new tiles gave 1.036×. Then the
  accuracy fix (`ACC_GROUP_FP32`) cost 0.924×, because the previous fp16-accumulate kernel failed 8B `w2`
  (K = 14336, max |err| 1.49 against a 0.5 tolerance). Model-weighted results are 0.96× (1B),
  0.92× (3B) and 0.89× (8B). This is a correctness trade, not a tuning gain.
- **Unchanged: 780M 8da4w, 1.00×.** The kernel is identical (`dbuf4zpg_t128x64k32g42s32`). The
  measured 0.997× is run-to-run noise.
- **Wording: "each GPU's existing winner" is not literally true for B580, B70, 4070 Ti and Orin.** On
  those GPUs the previous WMMA kernels were **not the default model path**:
  - they needed `ET_VK_TEXTURE_COOPMAT=1` (Xe2) or `ET_VK_COOPMAT_ANY_DEVICE=1` (4070 Ti), or explicit
    enabling (Orin);
  - an exported model therefore ran the tiled kernel;
  - B580's "before" was not a B580 winner either: it was the `-b70` branch's Xe2 tiles.

  Only on the 780M was the previous WMMA kernel the deployed default. A safe phrasing is "on top of
  the previous best WMMA kernels", with a footnote that on four of the five GPUs those kernels had been
  opt-in.

## Method

Per-shape values:
- Each value is the median over 3 repeats of that run's per-case GPU-timestamp kernel median, taken from
  each run's `COMPARE.json`. Source lines point at the matching row of the sibling `COMPARE.md`.
- The shapes are Llama 3.2 1B, 3.2 3B and 3.1 8B prefill at M = 2048. Each model has four projection
  shapes: `wq_wo`, `wk_wv`, `w1_w3` and `w2`.
- Both runs of a pair used the same harness (`test_llama_microbench --linear --regime=prefill`).
- Orin uses its own confirmation: `out/jetson-study/confirm/comparison.json`, with repeats and spreads
  from `summary.json`. Its original and tuned repeats were interleaved in one session on
  2026-09-27 between 06:18 and 06:23 UTC.

Time-weighted speedup per model = Σ before µs / Σ after µs over that model's shapes. Two weightings
are given:
1. **Equal weight per projection type** (as requested): `wq_wo + wk_wv + w1_w3 + w2`.
2. **Per-layer call counts**: 2·`wq_wo` + 2·`wk_wv` + 2·`w1_w3` + 1·`w2`. This is the weighting used by
   `roofline-et-study/e2e/analyze.py` (line 24) for the e2e Amdahl check.
   - The CSV also reports these sums × layers (16 / 28 / 32) as ms of linear time per 2048-token prefill.
   - The layer count cancels in the ratio.
   - **Use this one for an end-to-end projection.**

The two weightings differ by at most 0.13 (4070 Ti buffer 8B 4w) and usually by less than 0.03.

E2E projection rows (`data_kind` = "derived (e2e projection)"):
- **Method:** before ≈ measured tuned e2e prefill ms + layers × Σ count·(before − after). Tuned
  prefill times come from `roofline-et-study/e2e/e2e_summary.json` (release-1.4 branches, 3 repeats).
- **Validation on Orin:** it reproduces Orin's measured original→tuned e2e speedups at 2048 tokens:

  | Orin (2048 tokens) | projected | measured |
  |---|---:|---:|
  | 1B 4w | 1.018 | 1.016 |
  | 3B 4w | 1.021 | 1.020 |
  | 1B 8da4w | 1.424 | 1.426 |
  | 3B 8da4w | 1.507 | 1.513 |

- **Not measured for the other GPUs:** no e2e run of the *previous* WMMA kernels exists for them. The
  e2e study compared tiled with tuned only.

Control for device state: the tiled kernel is unchanged in every before/after pair. Its before/after
ratio is:
- 780M: 0.997–1.000;
- B580: 1.004–1.005;
- B70: 0.999–1.000;
- 4070 Ti: 44 of 48 cells within ±3 %. The four exceptions are all 8B `w1_w3` tiled cells, which are
  noisy in both runs (spread 28–71 %). Those cells are not WMMA cells.

WMMA repeat spread is ≤ 4.1 % in every before and after cell used.

## Per GPU

### Radeon 780M (rocky-ryzen, RADV Mesa 25.2.7): 4w 1.30× / 1.32×, 8da4w 1.00×

**Before:** `roofline-et-study/runs/780m-branch/780m`
- Captured 2026-09-26 18:57 UTC.
- Binary `tlm-780m`, sha256 `a3df49dd…`.
- `-780m` branch defaults before `9178cee44`; coopmat was already the default on the 780M.
- Kernels:
  - 4w: `t128x128k32g42s32`, plus the 3B-only texture3d override `t128x128k32g24s32`;
  - 8da4w: `dbuf4zpg_t128x64k32g42s32`.

**After:** `roofline-et-study/confirm/780m-final`
- Captured 2026-09-27 00:58 UTC.
- Binary `tlm-780m-final`.
- 4w: `t128x128k32g42s32f32c` for all shapes (`ACC_FP32` + `CSH_IN_ASH`).
- 8da4w: identical kernel.

**Results:**

| 780M | 1B | 3B | 8B | geomean (12 shapes) | doc |
|---|---:|---:|---:|---:|---|
| 4w texture3d (eq / per-layer) | 1.301 / 1.304 | 1.308 / 1.309 | 1.296 / 1.295 | 1.297 | 1.30× (L26) |
| 4w buffer | 1.325 / 1.319 | 1.325 / 1.319 | 1.323 / 1.320 | 1.317 | 1.32× (L27) |
| 8da4w texture3d | 0.996 / 0.995 | 0.997 / 0.996 | 0.997 / 0.997 | 0.997 | 1.00× (L28) |

- The doc matches the raw data.
- Projected e2e 4w, per model: 1.15× (1B), 1.19× (3B), 1.20× (8B).
- The 4w gain also removed the fp16-accumulate outliers at K = 4096 (1 in 16384 elements over tolerance).
- Doc: `780M-WMMA-LESSONS.md`.
- Clocks were automatic. No clock log exists for either run; the tiled control is within 0.3 %.

### Arc B580 (fedora, Mesa 26.2.3): 4w 2.49×, 8da4w 1.58× vs the `-b70` tiles (texture3d, opt-in before)

**Before:** `runs/b70-branch/b580`
- Captured 2026-09-26 19:14 UTC.
- Binary `tlm-b70`, sha256 `7957870e…`, which is the **`-b70` branch** build.
- Run via `run-microbench-extra.sh` with `ET_VK_TEXTURE_COOPMAT=1`, texture3d only.
- Kernels:
  - 4w: `dbuf4_xe2_t64x64k32g42s32`;
  - 8da4w: `dbuf4zpg_xe2_t128x64k64g48s32`.

**After:** `confirm/b580-default-v1`
- Captured 2026-09-26 21:18 UTC.
- Branch defaults. GT0 was at 2850 MHz according to `clocks.tsv`.
- Kernels:
  - 4w: `dbuf4_xe2_t128x128k16g44s16fli`;
  - 8da4w: `dbuf4zpg_xe2_t256x64k32g48s16`.

**Results:**

| B580 texture3d | 1B | 3B | 8B | geomean |
|---|---:|---:|---:|---:|
| 4w (eq / per-layer) | 2.244 / 2.226 | 2.750 / 2.873 | 2.416 / 2.392 | 2.489 (2.04–3.74) |
| 8da4w | 1.578 / 1.595 | 1.603 / 1.577 | 1.707 / 1.690 | 1.578 (1.38–1.77) |

- **No before/after ratio is stated in the doc.** `XE2-WMMA-LESSONS.md` L39 gives only WMMA/tiled for
  the old tiles: 2.9× and 1.55×. The raw before-run matches it: 2.89× and 1.55× (`COMPARE.md` L32/L34).
  The ratios above are new, computed from raw data.
- **Buffer has no "before".** The `-b70` branch had no buffer Xe2 variants, and buffer dispatch crashed
  (`runs/b70-branch-buffer-crash`). The CSV lists after-only buffer rows.
- **Deployment view.** Without the opt-in flag the model path ran tiled, so the realised model-path
  change is tiled → tuned: 7.16× microbench, 3.23–4.73× e2e for 4w.
- **Projected e2e vs the opt-in old tiles:**
  - 4w: 1.44× (1B), 1.78× (3B), 1.73× (8B);
  - 8da4w: 1.18×, 1.21×, 1.33×.
  - Caution: the e2e study found that Amdahl *under*-predicts Xe2 4w tiled→tuned (for example B70 8B
    measured 4.81× against 3.31× predicted), so the Xe2 4w projection is the least certain.

### Arc Pro B70 (fedora-gpu-eval, Mesa 26.2.3): 4w 2.54×, 8da4w 1.69× (texture3d, opt-in before)

**Before:** `runs/b70-branch/b70-0`
- Captured 2026-09-26 19:14 UTC.
- `ET_VK_TEXTURE_COOPMAT=1`, texture3d only.
- Kernels:
  - 4w: `xe2_t64x64k32g42s32`;
  - 8da4w: `xe2_t128x64k64g48s32`, plus the G31-only 8B override `xe2_t128x128k32g48s32`.

**After:** `confirm/b70-default-v1`
- Captured 2026-09-26 23:04 UTC.
- GT0 at 2800 MHz.
- Kernels: `xe2_t128x128k16g44s16fli` (4w) and `xe2_t256x64k32g48s16` (8da4w).

**Results:**

| B70 texture3d | 1B | 3B | 8B | geomean | doc |
|---|---:|---:|---:|---:|---|
| 4w (eq / per-layer) | 2.257 / 2.209 | 2.771 / 2.907 | 2.289 / 2.268 | 2.540 (2.04–4.30) | 2.54× (L115) |
| 8da4w | 1.752 / 1.747 | 1.657 / 1.639 | 1.588 / 1.601 | 1.694 (1.54–1.92) | 1.69× (L116) |

- The doc matches the raw data.
- The doc's round-1 figures (36.1 TFLOP/s and 53.1 TOP/s, L105–106) are single-run 1B screens and are
  not used here.
- Buffer: no before, for the same reason as the B580.
- Projected e2e:
  - 4w: 1.41×, 1.76×, 1.64×;
  - 8da4w: 1.20×, 1.22×, 1.28×.
- Same deployment caveat as the B580: the model path previously ran tiled.

### RTX 4070 Ti SUPER (gpu-dev-4004, driver 615.71.09): 8da4w 1.34× / 2.12×; 4w 0.96× (regression)

**Before:** `runs/4070ti-branch`
- Captured 2026-09-27 01:42 UTC.
- Binary `tlm-4070ti`, sha256 `d0e01e7b…`.
- Env `ET_VK_COOPMAT_ANY_DEVICE=1 ET_VK_TEXTURE_COOPMAT=1`; coopmat was not the default on this
  device.
- Kernels:
  - 8da4w: `dbuf4zpgtr_mk32_t128x128k32g44s32`;
  - 4w: `t256x128k16g22` / `t128x128k16g22` / 3B `t128x256k16g41`.

**After:** `runs/4070ti-final3`
- Captured 2026-09-27 05:05 UTC.
- Binary `tlm-4070ti-acc3`, sha256 `0cd70a43…`. Defaults, no `ET_VK_*`.
- Kernels:
  - 8da4w: `…t128x128k64g44s32ra` (k64 + `A_RAW` + `B_PAIR` + `CSH_IN_ASH`);
  - 4w: `t256x128k16g42s32ga` / `t128x128k16g24s32ga` (texture3d) and `t128x256k16g42s32ga` /
    `t128x128k16g42s32ga` (buffer). `ga` = `ACC_GROUP_FP32`.

**Results:**

| 4070 Ti | 1B | 3B | 8B | geomean | doc |
|---|---:|---:|---:|---:|---|
| 8da4w texture3d (eq / per-layer) | 1.326 / 1.306 | 1.382 / 1.366 | 1.397 / 1.378 | 1.345 (1.26–1.47) | 1.34× (L38) |
| 8da4w buffer | 2.118 / 2.095 | 2.118 / 2.103 | 2.209 / 2.192 | 2.126 (2.05–2.28) | 2.12× (L39) |
| **4w texture3d** | **0.966 / 0.963** | **0.927 / 0.923** | **0.888 / 0.886** | **0.957** (0.87–1.11) | 1.04× then 0.92× (L40) |
| 4w buffer | 0.949 / 0.931 | 0.929 / 0.922 | 1.064 / 0.973 | 0.973 (0.78–1.40) | 1.13× then 0.87× (L41) |

4w in steps, from raw data, as CSV derived rows:
- Tiles: branch → `final2` gives 1.036× (texture3d) and 1.122× (buffer).
- Accuracy fix: `final2` → `final3` gives 0.924× and 0.867×.
- The doc's buffer step "1.13×" is `final` → `final2` (1.129×). Branch → `final2` is 1.122×, which is
  equivalent.
- The net buffer result is **0.973×**. Multiplying the doc's rounded steps gives 0.98×.
- Texture3d nets to 0.957×. This is the ≈ 0.96× regression.
- The loss grows with model size (8B 0.89×), because the long-K shapes pay the most for the fp32 group
  flushes.

Other notes:
- **Projected e2e:**
  - 8da4w: 1.09× (1B), 1.13× (3B), 1.17× (8B);
  - 4w: 0.99×, 0.97×, 0.94×.
- **Contention:** the first `final3` attempt ran while ComfyUI was active. It is stored in
  `superseded/comfyui-contended/4070ti-final3-partial`. Its WMMA times equal `final3` within 0.2 %, so
  the contention affected only the tiled baseline.
- **Power:** during 4w the clock drops to 2.3–2.6 GHz at the 285 W limit (doc L101). The clock was not
  logged per run. GPU temperature at run start was not recorded for the branch run and was 62 °C for
  `final3`.

### Jetson Orin Nano 8 GB (orin-naughty, 15 W, 306–612 MHz): 4w 1.066×, 8da4w 2.324×

**Before ("original WMMA"):**
- The `-4070ti` branch kernels at `0270403ba`, **explicitly enabled on Orin; this was not the Orin
  default**.
- Kernels: `t128x128k16g22s32` (4w) and `dbuf4zpgtr_mk32_t128x128k32g44s32` (8da4w).

**After:** `0b260ffab` defaults
- 4w: `t256x128k16g22s32`, plus the fp32 `t128x128k32g42s32f32` for 8B `w2`.
- 8da4w: `…t128x128k64g44s32ra`.

**Results:**

| Orin texture3d | 1B | 3B | 8B | geomean | doc |
|---|---:|---:|---:|---:|---|
| 4w microbench (eq / per-layer) | 1.064 / 1.063 | 1.066 / 1.065 | 1.065 / 1.065 (without `w2`) | 1.066 (11 shapes) | 1.066× (L194) |
| 8da4w microbench | 2.275 / 2.226 | 2.307 / 2.272 | 2.437 / 2.424 | 2.324 | 2.324× (L195) |
| 4w e2e, 256 / 2048 prompt | 1.054 / 1.016 | 1.013 / 1.020 | not measured | | L280/L282 |
| 8da4w e2e, 256 / 2048 prompt | 1.630 / 1.426 | 1.791 / 1.513 | not measured | | L281/L283 |

- The e2e rows are measured with 3 repeats each.
- The 8B 4w `w2` original is numerically invalid: 40.78 ms "before" against 45.88 ms after with the
  fp32 repair. It is excluded from every ratio. Including it would make 8B 4w look slower (0.99×), but
  that comparison is against a wrong result.
- No e2e result exists for 8B, because the memory guard stopped it. This is not an OOM finding.

## Inconsistencies and caveats (flagged)

1. **The "before" base differs by GPU.**
   - 780M: the previous deployed default.
   - B580/B70/4070 Ti/Orin: previous kernels forced on by environment variables or explicit enabling;
     their default path was tiled.
   - B580: the "before" is the B70 branch's tiles, not a B580-specific winner.
2. **Clocks.** All runs used automatic clocks; nothing was pinned.
   - Xe2 after-runs logged GT0 at 2850/2800 MHz. The before-runs did not log clocks.
   - The 780M and 4070 Ti logged no clocks.
   - Equivalence rests on the unchanged tiled control (above). The 4070 Ti tiled control is noisy on 8B
     `w1_w3`.
3. **Timing within the window.**
   - All before/after pairs were measured on 2026-09-26 or 09-27 UTC, on the release-1.4 branches.
   - The 780M, B580 and B70 before-runs ran 2–6 h before their after-runs on the same host and binary
     family.
   - Orin interleaved original and tuned repeats in one 5-minute session. This is the best-controlled
     pair.
4. **Doc vs raw data.** The doc numbers match raw data to within rounding everywhere, except the
   4070 Ti 4w buffer net: the doc steps multiply to 0.98×, while raw branch → `final3` is 0.973×.
5. **B580 lacks a stated before/after ratio.** The doc states only WMMA/tiled values (2.9× / 1.55×), and
   those match raw data. The 2.49× / 1.58× figures here are computed, not quoted.
6. **Release 1.5.** All numbers here come from the release-1.4 branches.
   - `sarc-1.5/verify/*.json` re-measured the tuned kernels on release 1.5 (single run each; the kernels
     were renamed with a `sarc_` prefix, same tiles).
   - Time ratio 1.4-final / 1.5 = 0.996–1.008 geomean for 4070 Ti 4w/8da4w, B580 4w/8da4w, B70
     4w/8da4w and 780M 8da4w.
   - **No 780M 4w 1.5 verify file** exists in that directory. The takeaways doc (L206–207) states
     ±3 % on all five GPUs.
   - The "before" kernels were **not** re-measured on 1.5.
7. **E2E projections are Amdahl estimates, not measurements** (except Orin). They validate on Orin but
   under-predict Xe2 4w tiled→tuned in the 1.4 e2e study.

## Compact table (texture3d = model path; per-layer-weighted Σ before / Σ after; microbench, M = 2048)

| GPU × scheme | 1B | 3B | 8B | all-shape geomean | Source |
|---|---:|---:|---:|---:|---|
| 780M 4w | 1.30 | 1.31 | 1.30 | 1.30 | runs/780m-branch → confirm/780m-final |
| 780M 8da4w (unchanged) | 1.00 | 1.00 | 1.00 | 1.00 | same |
| B580 4w (vs opt-in `-b70` tiles) | 2.23 | 2.87 | 2.39 | 2.49 | runs/b70-branch/b580 → confirm/b580-default-v1 |
| B580 8da4w (vs opt-in `-b70` tiles) | 1.60 | 1.58 | 1.69 | 1.58 | same |
| B70 4w (vs opt-in) | 2.21 | 2.91 | 2.27 | 2.54 | runs/b70-branch/b70-0 → confirm/b70-default-v1 |
| B70 8da4w (vs opt-in) | 1.75 | 1.64 | 1.60 | 1.69 | same |
| 4070 Ti 4w (**regression**, accuracy fix) | 0.96 | 0.92 | 0.89 | 0.96 | runs/4070ti-branch → runs/4070ti-final3 |
| 4070 Ti 8da4w (vs opt-in) | 1.31 | 1.37 | 1.38 | 1.34 | same |
| Orin 4w (vs explicitly enabled) | 1.06 | 1.07 | 1.07* | 1.07 | out/jetson-study/confirm/comparison.json |
| Orin 8da4w (vs explicitly enabled) | 2.23 | 2.27 | 2.42 | 2.32 | same |
| Orin e2e 2048, 4w / 8da4w (measured) | 1.02 / 1.43 | 1.02 / 1.51 | — | | JETSON-WMMA-LESSONS.md L280–283 |

\* 8B 4w excludes `w2` (invalid original).
