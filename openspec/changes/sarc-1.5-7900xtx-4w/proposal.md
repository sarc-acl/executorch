# sarc-1.5-7900xtx-4w

## Why

The RX 7900 XTX (RDNA3 Navi31, AMDVLK 2025.Q2.1, `host-7900xtx`) has no SARC row: the README lists it as
"no promoted 1.4 default". This change carries the 2026-09-27 roofline-guided 4w study (1.4 study branch,
tag `archive/yanwen/release14-quant-shaders-7900xtx`) onto dev/1.5 as sweep candidates plus one new
default-off shader flag, so a row can be promoted with `verify.sh` evidence once the pinned build container
is available.

## What

- `B_COLMAJOR` (tile suffix `bt`) in `sarc_linear_q4gsw_coopmat`: B staged N-major in LDS, ColumnMajor
  `coopMatLoad`. Default off; every existing variant preprocesses unchanged.
- Sweep candidates (texture3d, fp32 accumulate + `CSH_IN_ASH`): `t256x128k32g24s32f32c[bt]`,
  `t256x128k32g44s32f32c`, `t128x256k32g42s32f32c`, `t128x128k32g42s32f32cbt`, with `Overrides.cpp` rows.
- No table row and no release-yaml variant yet (needs a golden entry from the pinned glslc).

## Evidence (all in `results/7900xtx/`)

### Roofs (`roofline/REPORT.md`, igpu-roofline `fast` plan, AMDVLK, DVFS-governed)

| roof | confirmed median |
|---|---:|
| matrix fp16 (fp16 acc / fp32 acc) | 136.4 / 141.9 TFLOP/s |
| matrix int8 | 142.6 TOP/s |
| fp16 FMA / int8 dot | 63.4 TFLOP/s / 69.2 TOP/s |
| DRAM read | 1108 GB/s |
| shared-fed fp16->fp32 MMA at 16 / 32 / 64 ops per loaded LDS byte | 57.6 / 109.3 / 136.2 TFLOP/s |

Every 4w/8da4w prefill kernel is compute-side (400–2300 ops per compulsory DRAM byte vs a ridge of ~128).

### Correctness
- The fp16-accumulate 4w default (`t128x128k32g42s32`) **fails** the production diff on this card: 8B w2
  (K = 14336) 162 / 8209 sampled elements out of tolerance, K = 4096 cases 1 element at 0.506–0.56
  (`sweep-1.4/base-pdiff-4w.log`). fp32 accumulation passes; every candidate here accumulates in fp32.
- All 12 1.4 sweep tiles and all `bt` variants pass the 8B production diff (`sweep-1.4/*-pdiff.log`); the 1.5
  port of `t256x128k32g24s32f32cbt` passes 1B/3B/8B (`port-1.5/pdiff-bt*.log`).

### Kernel (microbench, 8B, sum of the 4 prefill projections, texture3d, median of 3 interleaved)

| step | 1.4 study | 1.5 port |
|---|---:|---:|
| fp16-acc default `t128x128k32g42s32` (wrong numerics) | 9765 us | – |
| `t128x128k32g42s32f32c` (780M tile) | 8798 us | – |
| `t256x128k32g24s32f32c` | 7077 us | 7054 us |
| `t256x128k32g24s32f32cbt` | 6745 us | **6647 us** |
| tiled baseline | 36484 us | 23874 us (1.5's tiled kernel is faster) |

ISA (`isa/`, AMDVLK `VK_KHR_pipeline_executable_properties`), K-loop of `t256x128k32g24`: 385 -> 277
instructions, LDS loads 144 -> 32, 128 -> 0 `ds_load_u16` gathers with `bt`; VGPR 256 -> 243, 4 waves/SIMD.
The 256-row tile halves the int4 dequant VALU per WMMA (7.3 -> 3.8) relative to 128 rows.

Negative / open results:
- `bt` with N-fastest lane mapping measured 1.27x slower (8-way LDS bank conflicts on the uvec2 stores); the
  shipped K-fastest mapping is the fix.
- `t256x128k32g44 bt` is 0.89x: 512 threads but 256 B items, half the waves do all the dequant.
- `FAST_DQ` (magic-number int4 -> fp16, `isa/stats-btf/`): halves dequant VALU and is bit-identical, but is
  1.31x slower on `t256x128k32g24` and 1.03x faster on `t128x128k32g42`. Unexplained by the static ISA; not
  ported. Needs a thread trace (RGP needs `/etc/amd` root on the host).

### End to end (`llama_main`, 2048-token prompt `p2048tok.txt`, warmup, 3 interleaved reps, spread <= 2.5%)

| model | 1.4 tiled | 1.4 tuned (all 1.4 SARC kernels) | 1.5 tiled | 1.5 stock | 1.5 + `bt` 4w |
|---|---:|---:|---:|---:|---:|
| 1B 4w | 6522 | 18963 | 6543 | 6502 | 10836 (1.66x) |
| 3B 4w | 2510 | 9352 | 2532 | 2516 | 4561 (1.80x) |
| 8B 4w | 1054 | 4472 | 1237 | 1236 | 2606 (2.11x) |

The 1.5 end-to-end gain is well below 1.4's although the 4w kernel matches: the 1.4 branch also ran the SDPA
coopmat kernels on AMD, and 1.5 has SDPA rows only for `780m`/`xclipse` (hypothesis; next step: a 7900 XTX
SDPA row/candidate).

**Dispatch trap:** a 2047-token prompt runs the tiled kernel for every 4w/8da4w linear (the coopmat gate
needs M % tile == 0), so "default" equals tiled. Use a prompt whose token count is a multiple of the tile.

## Next

1. Build with the pinned container; add the release variant, `table_amd.cpp` row `"7900 xtx"`, golden entry,
   `test_sarc_select` fixture; run `verify.sh` (next-token checks, decode) and flip to `kVerified`.
2. SDPA coopmat candidate/row for the 7900 XTX.
3. 8da4w: bigger subgroup tiles were all slower (occupancy 12 -> 7–8 waves); try deeper prefetch at 120 VGPRs.
