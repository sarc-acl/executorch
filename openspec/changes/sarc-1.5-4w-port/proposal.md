# sarc-1.5-4w-port

## Why

This change ports the 4w (q4gsw) coopmat winners of the release-1.4 per-GPU branches to release 1.5 and
checks that each device reproduces its 1.4 kernel performance:
- `yanwen/release14-quant-shaders-{b580,b70,4070ti,jetson}`;
- the 780M was done in sarc-1.5-bootstrap.

## What

- **One shader body** (`glsl/sarc/sarc_linear_q4gsw_coopmat_body.glslh`) holds every device's features
  behind default-off defines:
  - Xe2: `FRAG_LAYOUT`, `IMG_A`/`IMG_W`, MMA_M 8, subgroup 16;
  - 4070 Ti: `ACC_GROUP_FP32`;
  - 780M and Orin: `ACC_FP32`, `CSH_IN_ASH`.

  The default LDS index macros expand to the old expressions, so the verified 780M SPIR-V is unchanged
  (golden).
- **Rows:**
  - `table_intel.cpp`: BMG G21/G31;
  - `table_nvidia.cpp`: 4070 Ti SUPER per-shape tiles and Orin measured projections.

  A row now has a storage-combination mask and an optional shape predicate. The fit check also requires
  the variant's subgroup size to be supported.
- **Takeover rule:** an op is built on the SARC path only if a row fits its build-time shape. Otherwise
  it keeps the upstream path.

## Results (2026-09-27; release 1.5, 2048-token prefill tok/s)

Setup:
- Raw logs are in `results/<gpu>/{sarc,stock}`.
- Gpu-lab lock held; ComfyUI (4070 Ti) and llm-api (B70) stopped during the runs and restored.
- Default clocks.
- Kernel ratio = microbench kernel median vs the 1.4 confirmation
  (`.artifacts/roofline-et-study/confirm/*`, `runs/4070ti-final3`, the Orin 1.4 reorg build), 24 shapes
  per GPU.

| GPU | tiled 1B/3B/8B | stock 1.5 1B/3B/8B | SARC 1B/3B/8B | SARC / stock | 1.4 tuned 1B/3B/8B | kernel ratio vs 1.4 (median, range) |
|---|---|---|---|---|---|---|
| Radeon 780M | 714 / 246 / 104 | 1205 / 421 / 194 | 1930 / 690 / 352 | 1.60 / 1.64 / 1.81× | 2702 / 1142 / 516 | 0.988 (0.984–0.998) |
| Arc B580 | 2570 / 797 / 346 | 3185 / 1149 / 524 | 8498 / 3352 / 1672 | 2.67 / 2.92 / 3.19× | 8292 / 3293 / 1665 | 1.004 (0.997–1.012) |
| Arc Pro B70 | 3657 / 1149 / 503 | 4592 / 1708 / 780 | 11636 / 4842 / 2421 | 2.53 / 2.83 / 3.10× | 11636 / 4774 / 2412 | 0.999 (0.987–1.009) |
| RTX 4070 Ti SUPER | 5596 / 2046 / 881 | 6850 / 2557 / 1111 | 19692 / 8790 / 4491 | 2.87 / 3.44 / 4.04× | 20078 / 8790 / 4501 | 1.000 (0.992–1.011) |
| Jetson Orin | 172 / 61 / 26 | 229 / 83 / 35 | 890 / 361 / 190 | 3.89 / 4.37 / 5.35× | 892 / – / – | 1.000 (0.999–1.002) |

The same kernel is dispatched per shape as on 1.4 for every GPU; only the names change to `sarc_...`.

Checks:
- Next token matches tiled on the 1973-token real-text prompt and the 1304-token unaligned prompt on all
  five GPUs.
- Decode works on all five.
- Production-diff:
  - B580, B70 and 4070 Ti: 6/6 pass.
  - Orin: texture3d 3/3 pass. The 3 buffer cases fail at K = 8192 (max |err| 1.70 > 0.5), both on this
    path's tiled fallback and on stock 1.5's own `q4gsw_linear_gemm__tin` kernel. It is an upstream fp16
    buffer-path accuracy issue on the Orin, present on 1.4 as well. With the takeover rule the Orin's
    buffer ops run stock 1.5.

**1.5 vs 1.4.** The kernels match on all five GPUs. End to end, B580, B70, 4070 Ti and Orin also match
1.4 (within about 2 %; the B580 is ~2.5 % faster). The 780M is slower on 1.5 because of SDPA, not 4w:
- its 1.4 branch ran SARC's SDPA coopmat (AMD, subgroup 64 only);
- measured with ETDump: SDPA is +307 of the +309 ms per 1B prefill;
- see `.artifacts/sarc-1.5/sdpa-gap/REPORT.md`.

Porting the SDPA coopmat is a separate change.

**Not verified:**
- Rank-3 8da4w correctness cases report "not coopmat" because 8da4w is not ported yet.
- Clocks were not pinned.
