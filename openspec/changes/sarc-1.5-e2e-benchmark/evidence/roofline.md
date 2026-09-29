# Roofline evidence for the five-GPU SARC 1.5 vs stock 1.5 prefill report

Collected 2026-09-27 (read-only; nothing was run on a GPU). Machine-readable copy: `roofline.json` (same directory).
Supporting files written here: `roofline-orin-gemm.csv` and `roofline-orin-trace.py`, both described in §6.

Path prefixes used below:

- `FF` = `/home/doremy/Desktop/igpu-roofline/results/fleet-fast-20260926`
- `JR` = `/home/doremy/Desktop/igpu-roofline/out/jetson-study/roofline-results/orin-naughty`
- `D` = `/home/doremy/Desktop/igpu-roofline/docs`
- `E` = `/home/doremy/Desktop/sarc-acl/.artifacts/e2e-1.5-2026-09-28`
- `ET` = `/home/doremy/Desktop/sarc-acl/dev/1.5/executorch/backends/vulkan/runtime/graph/ops/glsl`
- `S` = `/home/doremy/Desktop/sarc-acl/.artifacts/roofline-et-study`

## 1. Status and provenance of the roof numbers

All roofs in the tables below have the following status:

- They are **confirmed** in their campaign's `REPORT.md` "Roof confirmation" section: 3 fresh-process repeats, and
  `summary.json` has `confirmed: true, repeats: 3`.
- They come from **`fast`-plan** campaigns, not `standard` or `gold`.
- They are **short-run** values.
- The sustained runs are **one 120 s batch** of three representative roofs. That is not a confirmed sustained roof,
  which needs three batches.
- Clocks were **not pinned**.
- **No roof is ISA-verified.** Every campaign's `report/ISA-CHECK.md` has "—" in its ISA column. Only the SPIR-V
  ledger and the driver statistics are checked.

| GPU | Campaign REPORT | Plan | Measured (UTC) | Driver (Vulkan `driverVersion`) | Runner / code | Clock state (REPORT line 4) | Sentinel |
|---|---|---|---|---|---|---|---|
| Radeon 780M (rocky-ryzen) | `FF/rocky-ryzen/780m/report/REPORT.md` | fast | 2026-09-26 17:51–18:04 | RADV Mesa 25.2.7 (104865799) | runner e91952e1b236, igpu-roofline d4eb84e | DVFS, not pinned | 34/34 ok, median 3.269 TFLOP/s |
| Arc B580 (fedora) | `FF/fedora/b580/report/REPORT.md` | fast | 2026-09-26 17:51–18:04 | Mesa 26.2.3 ANV (109060099) | runner 810e098c8abb, d4eb84e | DVFS, not pinned | 34/34 ok, 11.679 |
| Arc Pro B70 (fedora-gpu-eval) | `FF/fedora-gpu-eval/b70-0/report/REPORT.md` | fast | 2026-09-26 17:51–18:05 | Mesa 26.2.3 ANV (109060099) | runner 810e098c8abb, d4eb84e | "unavailable" | 34/34 ok, 18.350 |
| RTX 4070 Ti SUPER (gpu-dev-4004) | `FF/gpu-dev-4004/4070tis/report/REPORT.md` | fast | 2026-09-27 01:16–01:32 | NVIDIA 615.71.09 (2580660800) | runner 5edad6896f44, d4eb84e | DVFS, not pinned | 34/34 ok, 25.481 |
| Jetson Orin Nano 8 GB (duck-naughty / orin-naughty) | `JR/report/REPORT.md` | fast | 2026-09-27 05:19–05:49 | NVIDIA 595.78 (2496888832) | runner 6635217988c7, repo head b65bda8 | "unavailable" (15 W mode, 306–612 MHz per `D/JETSON-WMMA-LESSONS.md:57-60`) | 34/34 ok, 0.732 |

Other provenance notes:

- 4070 Ti: ComfyUI (`zun-flux-pipeline`) was stopped for this campaign
  (`FF/control/campaign-config-4070tis.json:100`).
- Stale rows excluded: 0 in all five campaigns (REPORT line 5).
- Cross-check for the 780M: the older **standard** campaign
  `rocky-ryzen:~/igpu-roofline/campaigns/780m/2026-09-22-standard-v2/rocky-ryzen-gpu/report/REPORT.md` (runner
  48b9cbbd61c7, 5 repeats, 155/155 sentinel ok).
  - Confirmed at lines 49–65: alu_fp16 7.965, alu_fp32 5.618, dot_int8 11.440, matrix_fp16 10.936,
    matrix_fp16_fp32 14.768, matrix_int8 14.366, global_read 86.462, cache 3234.1, shared_fp16_read 2282.2 and
    shared_fp32_read 3435.8.
  - All compute and DRAM roofs agree with fast within 2.5 %, and cache within 3.4 %.
  - Shared fp32 read is bistable on this iGPU: fast 2282 vs standard 3436 GB/s. The campaigns README on rocky-ryzen
    records this.
  - That campaign is an older runner build and has no fed roofs. It is used here only as a cross-check.
- Historical data not used here:
  - 780M `2026-09-22-standard-v1` and `2026-09-23-fast-v1`, both superseded;
  - quick-fleet `results/fleet-quick-20260925`, which is a quick plan;
  - the v1 roofline data for Adreno.

## 2. Matrix (coopmat/WMMA) roofs, register-resident, confirmed

The value is the confirmed median; the parenthesis gives the repeat range; `L` is the line in that GPU's REPORT.md.

| GPU | fp16×fp16 → fp16 | fp16×fp16 → fp32 | int8×int8 → int32 |
|---|---|---|---|
| 780M | **10.958 TFLOP/s** (0.1 %) L66, 16×16×16 c8 | **14.772 TFLOP/s** (0.2 %) L70; 120 s: 14.789 | **14.393 TOP/s** (0.2 %) L74, 16×16×16 |
| B580 | **111.667** (0.0 %) L66, 8×16×16 c4 | **115.676** (0.0 %) L70; 120 s: 115.677 | **231.360** (0.0 %) L74, 8×16×32 |
| B70 | **173.324** (0.0 %) L66 | **179.942** (0.0 %) L70; 120 s: 179.939 | **359.888** (0.0 %) L74 |
| 4070 Ti SUPER | **183.729** (0.4 %) L67, 16×16×16 c8 | **92.318** (0.5 %) L71 (half rate); 120 s: 92.807 | **369.236** (0.5 %) L75, 16×16×32 |
| Orin Nano | **9.696** (0.1 %) L69 | **9.736** (0.1 %) L73; 120 s: 9.736 | **19.519** (0.0 %) L77, 16×16×32 |

### Fed variants (all confirmed)

The fed roofs use CHAINS = 8 reuse unless the config says otherwise. `_feed_dram` is a bandwidth in GB/s, measured at
low reuse. The per-reuse tables are in each REPORT's "Cooperative matrix fed from memory, by reuse" section (lines
95–135 in `FF` reports, 98–137 in the Jetson report).

| GPU | fp16 LDS / cache / DRAM | fp16→fp32 LDS / cache / DRAM | int8 LDS / cache / DRAM |
|---|---|---|---|
| 780M | 10.784 L69 / 10.295 L67 / 81.8 GB/s L68 | 14.468 L73 / 12.934 L71 / 81.3 GB/s L72 | 14.140 L77 / 12.513 L75 / 80.6 GB/s L76 |
| B580 | 109.655 L69 / 107.379 L67 / 444.3 L68 | 106.349 L73 / 105.439 L71 / 440.9 L72 | 206.393 L77 / 197.747 L75 / 436.8 L76 |
| B70 | 168.442 L69 / 160.857 L67 / 598.3 L68 | 166.629 L73 / 154.031 L71 / 590.5 L72 | 323.167 L77 / 282.576 L75 / 610.5 L76 |
| 4070 Ti | 177.972 L70 / 159.722 L68 / 650.2 L69 | 92.183 L74 / 89.689 L72 / 669.6 L73* | 369.362 L78 / 280.341 L76 / 698.9 L77 |
| Orin | 9.508 L72† / 7.080 L70 / 60.3 L71 | 8.839 L76 / 3.967 L74† / 61.2 L75 | 17.412 L80 / 8.152 L78 (4.7 % range) / 59.7 L79† |

\* The 4070 Ti REPORT line 9 lists `matrix_fp16_fp32_feed_dram: repeat_unstable` for one candidate. The roof that
was kept is confirmed at 0.0 % spread.

† The Jetson REPORT lines 9–11 list `insufficient_quality_repeats` or `repeat_unstable` for some candidates of
these roofs. The candidate that was kept is confirmed (`summary.json` `confirmed: true`). The Orin fp16→fp32
cache-fed value (3.967) is far below its sweep maximum (5.375), so treat it as a lower bound.

## 3. Scalar roofs (confirmed)

| GPU | fp16 FMA | fp32 FMA | int8 dot (dot4, 8 ops each) |
|---|---|---|---|
| 780M | **8.136 TFLOP/s** L58 (3.2 %, `alu_fp16_v2_c1`); 120 s 8.216 | **5.756** L59 | **11.717 TOP/s** L61 (`dot8_c2`) |
| B580 | **27.295** L58; 120 s 27.295 | **12.913** L59 | **32.060** L61 |
| B70 | **42.870** L58; 120 s 42.869 | **20.300** L59 | **50.365** L61 |
| 4070 Ti | **44.967** L59 (fp16 FMA ≈ fp32 FMA on this GPU) | **44.418** L60 | **78.050** L62 |
| Orin | **1.766** L61 | **1.202** L62 | **2.091** L64 |

The dot roof kernel is `/home/doremy/Desktop/igpu-roofline/shaders/dot8.comp:24`. It computes `dotPacked4x8EXT`
on **uint** operands, which is unsigned and non-saturating, followed by a separate add. The stock 8da4w kernel
instead uses the signed saturating `dotPacked4x8AccSatEXT` (§7). Nobody checked whether the two forms lower to the
same native instruction.

## 4. Memory roofs (confirmed; shader-logical bytes)

| GPU | DRAM read | DRAM write / copy | effective cache read | shared fp16 read | shared fp32 read |
|---|---|---|---|---|---|
| 780M | **86.672 GB/s** L63; 120 s 86.634 | 77.418 L65 / 71.478 L62 | 3343.0 L60 | 2157.6 L78 | 2282.2 L80 (bistable; standard-v2 3435.8) |
| B580 | **465.231** L63; 120 s 465.273 | 394.218 L65 / 408.202 L62 | 6020.2 L60 | 3450.2 L78 | 7258.9 L80 |
| B70 | **604.379** L63; 120 s 604.020 | 510.723 L65 / 532.567 L62 | 9978.8 L60 | 3331.0 L78 | 10017.5 L80 |
| 4070 Ti | **713.255** L64 | 642.461 L66 / 645.977 L63 | 7151.7 L61 | 1350.0 L79 ‡ | 21185.7 L81 |
| Orin | **62.268** L66; 120 s 62.186 | 58.162 L68 / 64.113 L65 | 204.1 L63 | 566.2 L81 | 616.2 L83 |

Anomalies to handle carefully:

- **4070 Ti sustained `global_read` = 2664.9 GB/s** (REPORT line 21; `report/sustained-runs.csv`). That is 3.7× the
  confirmed DRAM read. It cannot be DRAM, and the sustained working set was probably cache-resident. This was not
  investigated. **Do not use it.**
- ‡ **4070 Ti `shared_fp16_read` = 1350 GB/s**, against fp32 21186 GB/s and fp16 *write* 11831 GB/s. It is confirmed,
  but it is almost certainly an artifact of the fp16 read microkernel (`sharedbw_fp16_v4_op0_acc32`) on this driver,
  not an LDS limit. See `D/XE2-WMMA-LESSONS.md:173-179`: "a kernel rate above a roof means the wrong roof was
  chosen". Use the fp32-granularity value for coopmat kernels.
- On Xe2, the fp16-granularity SLM read (3.3–3.5 TB/s) is below the fp32 granularity (7.3/10.0 TB/s). The B70 4w
  kernel was measured at 4.17 TB/s SLM read by OA (`D/XE2-WMMA-LESSONS.md:126-128`).

## 5. Cooperative-matrix shapes and subgroup size (from each campaign's `capabilities.json`)

| GPU | Default subgroup (caps line) | Compute subgroup range (gpu-lab) | fp16 shapes (f16 or f32 acc) | int8 shapes | Other |
|---|---|---|---|---|---|
| 780M | 64 (`FF/rocky-ryzen/780m/capabilities.json:440`) | 32–64 | 16×16×16 | 16×16×16, s8/u8 any mix → s32/u32, also saturating | — |
| B580 | 32 (`FF/fedora/b580/capabilities.json:326`) | 16–32 | 8×16×16 | 8×16×32 s8→s32, u8→u32 | bf16 8×16×16 |
| B70 | 32 (`FF/fedora-gpu-eval/b70-0/capabilities.json:326`) | 16–32 | 8×16×16 | 8×16×32 | bf16 8×16×16 |
| 4070 Ti | 32 (`FF/gpu-dev-4004/4070tis/capabilities.json:495`) | 32 | 16×16×16, 16×8×16, 16×8×8 | 16×16×32, 16×8×32 | bf16 16×16×16 → f32; fp8 e4m3/e5m2 16×16×32 → f16/f32 |
| Orin | 32 (`JR/capabilities.json:398`) | 32 | 16×16×16, 16×8×16, 16×8×8 | 16×16×32, 16×8×32 | bf16 16×16×16 → f32 |

The shape lists start at the `matrix_shapes` line of each file: 780M line 224, B580/B70 line 218, 4070 Ti line 298,
Orin line 255. The subgroup ranges come from `/home/doremy/Desktop/gpu-lab/docs/compute-capabilities.md:43-51`.

Subgroup sizes the SARC kernels use (the `s16`/`s32` suffix and `SUBGROUP_SIZE` in the yaml):

- 780M: 32 for both 4w and 8da4w, not the default 64.
- B580 and B70: 16.
- 4070 Ti and Orin: 32.

## 6. Per-kernel achieved rates in the real model (release 1.5, M = 2048)

**Source.** ETDump per-dispatch GPU times of every prefill GEMM dispatch, taken in the actual `llama_main` runs of
this campaign. There are 532 dispatches per cell (1B + 3B + 8B).

- 780M, B580, B70 and 4070 Ti: `E/report/evidence/trace/gemm.csv`. It was produced from `E/raw/<gpu>/trace2/*.etdp`
  by `E/report/trace_analysis.py`, which uses the warm (second) execution.
- **Orin was missing from `gemm.csv`.** Its ETDumps live in `E/raw/orin/trace/`, not in `trace2/`. I parsed them
  with the same `load()`/`family()` functions (`roofline-orin-trace.py`, output `roofline-orin-gemm.csv`).
  - Orin ETDumps hold a **single execution with no warmup**.
  - Summed dispatch time equals graph time: for example, 1B 4w SARC is 2295.7 of 2299.8 ms. That graph time matches
    the timed e2e prefill of 2048/890.8 tok/s = 2.30 s.
  - The first dispatches are slow, e.g. 2.9 TFLOP/s for the first 1B 4w GEMM. Use the time-weighted aggregate, not
    the minimum.

**Metric.** The rate is 2·M·N·K divided by the dispatch time. For 8da4w these are integer "ops" (TOP/s). "tw" is
the time-weighted aggregate, Σops / Σtime.

- The GEMM time excludes the separate 8da4w activation-quantize dispatch (23–167 ms per 8B prefill, see
  `E/report/evidence/trace/families.csv`).
- It also excludes the producer of the 4w stock kernel's transposed activation input.

| GPU | Kernel | tw rate | median [min–max] per dispatch | % of matching roof | % of alternative roof |
|---|---|---:|---|---|---|
| 780M | stock 4w `q4gsw_linear_gemm__tin__w_4x8_nc_texture3d_half` | 4.160 TFLOP/s | 4.185 [3.847–4.365] | **51.1 % of fp16 FMA** (8.136) | 28.2 % of fp16→fp32 matrix |
| 780M | stock 8da4w `linear_dq8ca_q4gsw_tiled_texture3d_texture2d_half_zpint8` | 6.997 TOP/s | 7.201 [6.293–8.273] | **59.7 % of int8 dot** (11.717) | 48.6 % of int8 matrix |
| 780M | SARC 4w `…t128x128k32g42s32f32c` (fp32 acc) | 10.437 | 10.371 [9.034–10.895] | **70.7 % of fp16→fp32 matrix** (14.772) | 72.1 % of LDS-fed |
| 780M | SARC 8da4w `…zpg_t128x64k32g42s32` | 9.703 | 9.690 [8.824–10.116] | **67.4 % of int8 matrix** (14.393) | 68.6 % of LDS-fed |
| B580 | stock 4w | 8.787 | 9.093 [4.594–10.314] | **32.2 % of fp16 FMA** (27.295) | 7.9 % of fp16 matrix |
| B580 | stock 8da4w | 18.426 | 21.509 [9.363–24.345] | **57.5 % of int8 dot** (32.060) | 8.0 % of int8 matrix |
| B580 | SARC 4w `…t128x128k16g44s16m8fli` (fp16 acc) | 42.850 | 43.804 [8.569–48.089] | **38.4 % of fp16 matrix** (111.667) | 39.1 % of LDS-fed |
| B580 | SARC 8da4w `…zpg_t256x64k32g48s16m8` | 50.595 | 51.443 [22.614–56.882] | **21.9 % of int8 matrix** (231.360) | 24.5 % of LDS-fed |
| B70 | stock 4w | 13.688 | 13.879 [9.576–16.009] | **31.9 % of fp16 FMA** (42.870) | 7.9 % |
| B70 | stock 8da4w | 29.991 | 33.425 [25.611–37.977] | **59.5 % of int8 dot** (50.365) | 8.3 % |
| B70 | SARC 4w (same variant as B580) | 65.730 | 65.442 [35.941–73.995] | **37.9 % of fp16 matrix** (173.324) | 39.0 % LDS-fed |
| B70 | SARC 8da4w (same as B580) | 83.300 | 83.968 [70.065–93.065] | **23.1 % of int8 matrix** (359.888) | 25.8 % LDS-fed |
| 4070 Ti | stock 4w | 17.744 | 17.288 [12.984–19.091] | **39.5 % of fp16 FMA** (44.967) | 9.7 % of fp16 matrix |
| 4070 Ti | stock 8da4w | 17.563 | 17.332 [15.761–18.231] | **22.5 % of int8 dot** (78.050) | 4.8 % |
| 4070 Ti | SARC 4w `…t256x128k16g42s32ga` (500 dispatches, 118.44) + `…t128x128k16g24s32ga` (32, 93.60) | 118.338 | 117.273 [91.188–122.595] | **64.4 % of fp16 matrix** (183.729) | 66.5 % LDS-fed |
| 4070 Ti | SARC 8da4w `…zpgtr_t128x128k64g44s32mk32ra` | 154.193 | 152.982 [130.151–159.569] | **41.8 % of int8 matrix** (369.236) | 41.7 % LDS-fed |
| Orin | stock 4w | 0.553 | 0.551 [0.446–0.557] | **31.3 % of fp16 FMA** (1.766) | 5.7 % |
| Orin | stock 8da4w | 0.498 | 0.497 [0.398–0.499] | **23.8 % of int8 dot** (2.091) | 2.6 % |
| Orin | SARC 4w `…t256x128k16g22s32` (500 dispatches, 6.106) + `…t128x128k32g42s32f32` (32 dispatches on 8B K = 14336, 5.233) | 5.933 | 6.023 [2.868–6.217] | **61.2 % of fp16 matrix** (9.696); the f32 variant is 53.7 % of fp16→fp32 (9.736) | 62.4 % LDS-fed |
| Orin | SARC 8da4w `…zpgtr_t128x128k64g44s32mk32ra` | 4.639 | 4.637 [2.264–4.840] | **23.8 % of int8 matrix** (19.519) | 26.6 % LDS-fed |

Kernel-level SARC/stock ratios (tw) and the ratio of the stock kernels to each other:

| GPU | SARC/stock 4w | SARC/stock 8da4w | stock 8da4w ÷ stock 4w (achieved) |
|---|---:|---:|---:|
| 780M | 2.51× | 1.39× | 1.68 |
| B580 | 4.88× | 2.75× | 2.10 |
| B70 | 4.80× | 2.78× | 2.19 |
| 4070 Ti | 6.67× | 8.78× | 0.99 |
| Orin | 10.73× | 9.32× | 0.90 |

These agree with the 1.4-era microbench study on every GPU where a comparison exists:

- 780M 4w 9.7–10.8 TFLOP/s (`D/780M-WMMA-LESSONS.md:26`);
- B580 4w 33.8–47.1 (`D/XE2-WMMA-LESSONS.md:34`);
- B70 4w 46.4–69.4 (`D/XE2-WMMA-LESSONS.md:115`);
- 4070 Ti 4w 54–67 % and 8da4w 37–42 % of roof (`D/4070TI-WMMA-LESSONS.md:38-40`);
- Orin tuned 8da4w 4.39–4.86 TOP/s and 4w 5.75–6.25 TFLOP/s (`D/JETSON-WMMA-LESSONS.md:203-205`).

There is one caveat. The 1.4 study's "tiled" 4w baseline was `linear_q4gsw_tiled`
(`S/confirm/780m-final/baseline-r1.json`), a **different kernel** from the 1.5 stock `q4gsw_linear_gemm__tin`. The
8da4w tiled kernel has the same name in both releases.

## 7. What the stock 1.5 kernels compute (source, not ISA-verified)

Both stock shaders are byte-identical to upstream `release/1.5` @ `985c1ceccc` in the checkout:
`git diff 985c1ceccc` over these files and all `linear_*.glslh` shows a single change. That change is an `#ifndef`
guard in `linear_int8_input_block.glslh`, a quantize helper that the GEMM loop does not use.

**4w: `q4gsw_linear_gemm__tin__w_4x8_nc_texture3d_half`** (`ET/q4gsw_linear_gemm__tin__w_4x8.glsl`, yaml
`TILE_M: 8, TILE_N: 4` at lines 12–13).

- For `DTYPE == half`, the accumulator is `f16vec4` (lines 85–91). This is **fp16 FMA with fp16 accumulation**.
  There is no cooperative matrix and no integer dot.
- Per thread and per 4-deep K step (lines 218–293):
  - one `ivec4` weight load, of which half is used;
  - one `f16vec4` scale load (254);
  - per `k_inner`: two `f16vec4` activation loads (259–263), four int4 → int16 → fp16 conversions each multiplied
    by the scale (277–283), then **8 `f16vec4` FMAs** `acc_T[m] += B[m] * dw_vec` (288–292).
  - That is 32 fp16 FMAs per 4 dequant chains (shift, mask, subtract, convert, multiply). Dequant is amortized over
    only TILE_M = 8 rows.
- The header comment (lines 9, 82–84) says the kernel is "Adreno-optimized" and relies on 2× fp16 ALU.
- The matching roof is therefore **fp16 FMA**. The kernel reaches 31–51 % of it (§6).
- On the 4070 Ti, fp16 FMA (44.97) equals fp32 FMA (44.42), so fp16 gives no 2× there.

**8da4w: `linear_dq8ca_q4gsw_tiled_texture3d_texture2d_half_zpint8`** (`ET/linear_dq8ca_q4gsw_tiled.glsl` plus
`ET/linear_fp_output_tile_int8_int4_compute.glslh`; yaml `TILE_M4: 1, TILE_K4: 1, TILE_N8: 1` at lines 13–15, so
4 M × 8 N per thread).

- Per k4 (glsl 122–130), the thread loads 4 packed int8×4 activations and one `ivec4` of 32 int4 weights.
- It unpacks the nibbles with `& 0x0F0F0F0F` and `>> 4` (glslh 33–34).
- It then issues **32 `dotPacked4x8AccSatEXT`**: signed int8 dot4 with saturating int32 accumulation,
  `GL_EXT_integer_dot_product` (glslh 13, 39–49). **Yes, stock 8da4w uses the int8 dot product.**
- After each 128-K group there is a zero-point and sum correction plus an fp `fma` into the output tile (glsl
  132–144, glslh 56–83).
- The matching roof is **int8 dot**. The kernel reaches ≈ 58–60 % of it on AMD and Intel, but only 22–24 % on
  NVIDIA (§6).

**Why stock 8da4w beats stock 4w on the 780M and Xe2 but not on NVIDIA.** The dot-to-FMA roof ratio predicts
1.18–1.74× for stock 8-bit over stock 4-bit.

- Achieved: 1.68 on the 780M, 2.10–2.19 on Xe2, 0.99 on the 4070 Ti and 0.90 on Orin.
- On the 780M and Xe2, stock 4w sits further below its FMA roof (32–51 %) than stock 8da4w sits below its dot roof
  (≈ 58–60 %).
- On NVIDIA it is the reverse: stock 8da4w reaches only 22–24 % of the dot roof.
- No counter or ISA data exists for the stock kernels, so the *cause* of the NVIDIA shortfall is not established.
  The dot-instruction form difference noted in §3 is one untested candidate.

## 8. Counter and profiler evidence (1.4-era, SARC kernels only)

None of this evidence was collected on the 1.5 stock kernels.

**4070 Ti, Nsight** (`sudo nsys --gpu-metrics-set=ad10x`; raw
`S/remote/gpu-dev-4004/et-roofline-study/nsys/*.nsys-rep`):

- Tensor Active matched the roof fraction: ≈ 30 % for the old 8da4w and ≈ 73 % for 4w
  (`D/4070TI-WMMA-LESSONS.md:56-58`).
- Old 8da4w: TA 30 %, SM Issue 30 %, DRAM < 8 % (65–67).
- In the k64 8da4w ablation, MMA-only runs at 272 TOP/s (74 % of roof). The int8 `coopMatStore` staging was 27 % of
  time (71–80).
- 4w: TA 70–76 % on large shapes. The clock drops to 2.3–2.6 GHz at the 285 W limit (101–102).
- 4w w1_w3: MMA-only reaches 92 % of the fp16 roof. wq_wo is limited by partial waves, ≈ 73 % (104–107).
- 8da4w: 252 µs full vs 142 µs MMA-only (`D/WMMA-STUDY-TAKEAWAYS.md:157-158`).
- No SASS: **not ISA-verified** (line 52).

**Orin, Nsight** (submission-window, `D/JETSON-WMMA-LESSONS.md:95-110`):

| Kernel | Tensor Active | SM Issue | SM Active |
|---|---:|---:|---:|
| original 4w | 55.4 % | 38.5 % | 99.8 % |
| original 8da4w | 9.86 % | 21.4 % | 99.9 % |
| final 8da4w | 20.85 % | 28.82 % | not given |

- Warps in flight moved from 34.30 % to 35.65 %.
- There is no DRAM byte counter.

**Xe2, OA** (raw `S/oa/{b580,b70}/*.rec`).

B580 tuned kernels (`D/XE2-WMMA-LESSONS.md:80-93`):

| Metric | 4w | 8da4w |
|---|---:|---:|
| XVE occupancy | 95 % | 100 %+ |
| Active / stall | 52 % / 39 % | 40 % / 54 % |
| SBID stall | 47 % | 57 % |
| Barrier stall | 31 % | 20 % |
| ALU-dependency stall | 5 % | 20 % |
| SLM read | 2.86 TB/s | 2.28 TB/s |
| XMX / issued instructions | 32 % | 29 % |
| Instructions per K iteration (16 DPAS) | ~674 | ~1194 |

B70 (126–133):

- 4w: SLM read 4.17 TB/s (42 % of the 32-bit SLM roof), XVE active 63 %.
- 8da4w: SBID ~55 %, ALU dependency ~21 %, barrier ~20 %, 1.71 int32 instructions per XMX instruction.
- The 8da4w kernel keeps an int32 and an fp32 accumulator per tile, so it is capped at 4 MMAs per subgroup
  (`D/XE2-WMMA-LESSONS.md:92-93`).

**780M, in-kernel `shader_clock` phase timing** (method `D/780M-WMMA-LESSONS.md:49-63`; raw `S/prof/780m-clock/`):

- The f32c reference spends 2402 cycles per iteration in the LDS→WMMA phase and 1112 cycles in the
  store/dequant phase (78–80).
- The WMMA pipe is busy ≈ 6 waves × 32 WMMAs × ~16 cycles ≈ 3072 of ~4500 cycles per iteration, about 70 %. This
  matches 66–73 % of roof (96–97) and the 70.7 % measured in §6.
- ISA (`RADV_DEBUG=shaders`): the fp16-accumulate loop had ~350 repack moves per 32 WMMAs. `ACC_FP32` cut VALU from
  637 to 330 (69).
- The raw per-phase `.bin` dumps are in `prof/780m-clock/*/`. I found no decoded per-phase table besides the lessons
  document.

## 9. Derived ratios (confirmed short-run roofs)

The SARC 4w accumulator is taken from `ET/sarc/sarc_linear_q4gsw_coopmat.yaml`:

- **780M** (lines 38–50): `ACC_FP32: true`, so the fp16→fp32 roof applies.
- **B580 and B70** (55–73): no accumulator flag, default `ACC_FP32: false` (line 28), so fp16.
- **4070 Ti** (76–90): `ACC_GROUP_FP32: true`. The MMA accumulates in fp16 within each 128-K group; an fp32 running
  total is kept outside the MMA (comment at lines 74–75, `D/4070TI-WMMA-LESSONS.md:138-140`). The fp16 roof applies.
- **Orin: correction to the brief.** The default Orin 4w tile `t256x128k16g22s32` (116–122) has **no `ACC_FP32`**,
  so it accumulates in fp16. Only `t128x128k32g42s32f32` (112–115) uses fp32, and it is dispatched only for 8B
  K = 14336 (32 of 532 dispatches). The primary Orin ratio below therefore uses fp16. Orin's two accumulator roofs
  differ by only 0.4 %, so the ratios barely change.

The SARC 8da4w kernels are int8×int8→int32 MMA (`sarc_linear_dq8ca_coopmat_zpg.yaml`,
`sarc_linear_dq8ca_coopmat_zpgtr.yaml`).

| GPU | int8-matrix / fp16-matrix (SARC acc) | fp16-matrix / fp16-FMA | int8-matrix / int8-dot | int8-dot / fp16-FMA |
|---|---|---|---|---|
| 780M | 14.393 / 14.772 (fp32 acc) = **0.974**; vs fp16 acc 14.393 / 10.958 = 1.313 | 14.772 / 8.136 = **1.816**; fp16 acc 10.958 / 8.136 = 1.347 | 14.393 / 11.717 = **1.228** | 11.717 / 8.136 = **1.440** |
| B580 | 231.360 / 111.667 = **2.072**; vs fp32 acc 2.000 | 111.667 / 27.295 = **4.091** | 231.360 / 32.060 = **7.216** | 32.060 / 27.295 = **1.175** |
| B70 | 359.888 / 173.324 = **2.076**; vs fp32 acc 2.000 | 173.324 / 42.870 = **4.043** | 359.888 / 50.365 = **7.146** | 50.365 / 42.870 = **1.175** |
| 4070 Ti | 369.236 / 183.729 = **2.010**; vs fp32 acc 369.236 / 92.318 = 4.000 | 183.729 / 44.967 = **4.086**; fp32 acc 2.053 | 369.236 / 78.050 = **4.731** | 78.050 / 44.967 = **1.736** |
| Orin | 19.519 / 9.696 (fp16 acc, default tile) = **2.013**; fp32 acc 19.519 / 9.736 = 2.005 | 9.696 / 1.766 = **5.490**; fp32 acc 5.513 | 19.519 / 2.091 = **9.335** | 2.091 / 1.766 = **1.184** |

Reading the table:

- On the **780M**, int8 WMMA is *no faster* than the fp16→fp32 WMMA that SARC 4w uses (0.97×). It is only 1.23× the
  scalar int8 dot. This is why SARC's 8da4w gain over stock is small there: 1.39× kernel and 1.56–1.86× end to end.
- On **Xe2**, the fp16 matrix roof is 4× fp16 FMA, while int8 matrix is 7× int8 dot. The SARC 8da4w kernel,
  however, runs at only 22–23 % of the int8 roof (§6, §8). Its realized gain (2.75×) is therefore below the
  4w gain (4.8×).
- On **NVIDIA**, both headrooms are large: 4.7× on the 4070 Ti and 9.3× on Orin for int8. Stock 8da4w is also
  inefficient there (22–24 % of dot), which lets SARC 8da4w reach 8.8–9.3× at kernel level.

## 10. Compact summary

Roofs are confirmed fast-plan short-run medians (3 repeats, spread ≤ 3.2 %, sentinel healthy) and are not
ISA-verified. Kernel rates are ETDump time-weighted rates from the 1.5 e2e campaign.

| GPU | fp16 mat (fp16 acc) | fp16 mat (fp32 acc) | int8 mat | fp16 FMA | fp32 FMA | int8 dot | DRAM read | stock 4w (% FMA) | stock 8da4w (% dot) | SARC 4w (% matching mat) | SARC 8da4w (% int8 mat) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 780M | 10.958 | 14.772 | 14.393 | 8.136 | 5.756 | 11.717 | 86.7 GB/s | 4.16 (51 %) | 7.00 (60 %) | 10.44 (71 %, fp32 acc) | 9.70 (67 %) |
| B580 | 111.667 | 115.676 | 231.360 | 27.295 | 12.913 | 32.060 | 465.2 | 8.79 (32 %) | 18.43 (57 %) | 42.85 (38 %) | 50.60 (22 %) |
| B70 | 173.324 | 179.942 | 359.888 | 42.870 | 20.300 | 50.365 | 604.4 | 13.69 (32 %) | 29.99 (60 %) | 65.73 (38 %) | 83.30 (23 %) |
| 4070 Ti SUPER | 183.729 | 92.318 | 369.236 | 44.967 | 44.418 | 78.050 | 713.3 | 17.74 (39 %) | 17.56 (22 %) | 118.34 (64 %) | 154.19 (42 %) |
| Orin Nano | 9.696 | 9.736 | 19.519 | 1.766 | 1.202 | 2.091 | 62.3 | 0.553 (31 %) | 0.498 (24 %) | 5.93 (61 %) | 4.64 (24 %) |

Units: fp16 roofs in TFLOP/s; int8 roofs in TOP/s.

| GPU | int8mat / fp16mat (SARC acc) | fp16mat / fp16FMA | int8mat / int8dot | int8dot / fp16FMA |
|---|---:|---:|---:|---:|
| 780M | 0.974 | 1.816 | 1.228 | 1.440 |
| B580 | 2.072 | 4.091 | 7.216 | 1.175 |
| B70 | 2.076 | 4.043 | 7.146 | 1.175 |
| 4070 Ti | 2.010 | 4.086 | 4.731 | 1.736 |
| Orin | 2.013 | 5.490 | 9.335 | 1.184 |

## 11. Gaps (not measured or not verified)

1. **ISA verification.**
   - No roof on any of the five GPUs is ISA-verified; ISA-CHECK.md has "—" in its ISA column everywhere.
   - The stock kernels' instruction selection is not verified: whether fp16 FMA is packed or vectorized, and whether
     `dotPacked4x8AccSatEXT` lowers to native dot4 with saturation. No stock ISA or pipeline statistics were
     collected.
   - NVIDIA has no SASS route at all.
2. **Counters on stock 1.5 kernels.** None exist: no Nsight, OA or `shader_clock` data. All counter evidence covers
   the 1.4-era SARC (tsweep) kernels.
3. **Sustained roofs.**
   - Only one 120 s batch was run, for alu_fp16, global_read and matrix_fp16_fp32. No roof is a confirmed
     (3-batch) sustained roof.
   - The **4070 Ti sustained global_read (2664.9 GB/s) is invalid as a DRAM figure**.
4. **Clocks.**
   - Nothing was pinned.
   - B70 and Orin report the clock state as "unavailable". Orin's 15 W mode and 612 MHz cap come from separate
     telemetry.
   - Neither the roof campaigns nor the e2e campaign recorded the clock under load in a form comparable across GPUs.
5. **Plans.**
   - All five roof sets are `fast` plan (3 repeats).
   - Only the 780M has an independent `standard` cross-check (standard-v2, older runner). It agrees within 3.4 %
     except the bistable shared fp32 read.
6. **Suspect roofs.**
   - 4070 Ti shared_fp16_read 1350 GB/s: likely a microkernel artifact.
   - Orin fp16→fp32 cache-fed 3.967 TFLOP/s: its sweep maximum is 5.375.
   - Orin int8 cache-fed has a 4.7 % repeat range.
   - The 780M shared fp32 read is bistable.
7. **Dot-roof form mismatch.** The roof uses unsigned `dotPacked4x8EXT` plus an add. The stock kernel uses signed
   `dotPacked4x8AccSatEXT`. There is no separate roof for the saturating form.
8. **Orin kernel rates.** They come from a single un-warmed ETDump execution (`E/raw/orin/trace/`). They were not
   in the campaign's `gemm.csv`; I added them here.
9. **Physical bytes.** DRAM and L2 traffic of the kernels is not measured anywhere (`VK_KHR_performance_query` is
   not exposed; REPORT "Limits").
10. **Per-kernel data vs production shape.**
    - The ETDump rates cover M = 2048 prefill only.
    - The GEMM time excludes the 8da4w activation-quantize dispatch and the 4w input-transpose producer.
11. **Other-host 1.5 data.** On rocky-ryzen, the `2026-09-26-fast-et-study` folder is the same campaign as
    `FF/rocky-ryzen/780m`. It has no report of its own, and its report was generated locally.
