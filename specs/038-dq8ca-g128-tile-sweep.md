# specs/038: dq8ca int8 coopmat tile sweep — first trustworthy g128 sweep

First tile sweep run on the **post-restructure** (specs/036) kernel at the real
workload **group_size 128** (all prior sweeps ran at g32 and/or the pre-hoist
kernel). Microbench: `test_llama_microbench --linear`, 780M / RADV, ColumnMajor
B (specs/037). kernel_us = 8da4w prefill 2048x2048 coopmat kernel time (1B);
geomean = 8da4w kernel geomean vs tiled across all 12 shapes.

| tile        | grid | sg | wg  | gate | kernel_us | geomean | VGPR | subg/SIMD | LDS   |
|-------------|------|----|-----|------|-----------|---------|------|-----------|-------|
| 128x64 k32  | 2x2  | 32 | 128 | PASS | 2432      | 0.94    | 256  | 4         | 15872 | (prior ship)
| **128x64 k32** | **4x2** | **32** | **256** | **PASS** | **2350** | **1.00** | **128** | **8** | **15872** | **WINNER**
| 128x64 k32  | 4x1  | 32 | 128 | PASS | 2643      | 0.88    | 256  | 4         | 15872 |
| 128x64 k32  | 2x4  | 32 | 256 | PASS | 2620      | 0.91    | 128  | 8         | 15872 |
| 128x64 k32  | 2x2  | 64 | 256 | PASS | 2840      | 0.82    | 128  | 8         | 15872 |
| 128x64 k64  | 4x2  | 32 | 256 | PASS | 2423      | 0.96    | 144  | 7         | 30208 | LDS-capped
| 128x128 k32 | 4x2  | 32 | 256 | PASS | 2437      | 0.95    | 256  | 4         | 22016 | N-tile pushes VGPR back
| 64x128 k32  | 8x1  | 32 | 256 | PASS | 2690      | 0.86    | 160  | 6         | 17408 |
| 256x64 k32  | 4x2  | 32 | 256 | FAIL | (25088 LDS over budget)     | 256  | 4         | 25088 |
| 64x64 k32   | 4x2  | 32 | 256 | PASS | 2902      | 0.81    | 80   | 12        | 11264 | WG tile too small

## Finding

The limiter at the prior 2x2/sg32 ship was **VGPR pressure (256 → 4 waves/SIMD)**.
The double-accumulator (int32 accum + fp16 result, MMAS_PER_SG_M × MMAS_PER_SG_N
tiles each) is the VGPR driver. Going **4x2** splits N four ways so
MMAS_PER_SG_N drops 2 → 1, halving accumulator VGPRs (256 → 128) and **doubling
occupancy 4 → 8 waves/SIMD**. That is the whole win: 2432 → 2350 us,
0.94 → 1.00x geomean.

Guardrails the sweep proved:
- **The WG tile must stay large.** 64x64 (VGPR 80, occ 12) is *slower* (2902) —
  past ~8 waves the extra occupancy doesn't pay and the WG tile no longer
  amortizes the A/B staging.
- **Bigger N re-inflates VGPR.** 128x128/4x2 has MMAS_N back to 2 → VGPR 256 →
  occ 4 → no better than the old ship.
- **Grid orientation matters at equal VGPR.** 4x2 (tall SG tile 64x16,
  MMAS 4x1) beats 2x4 (32x32, MMAS 2x2) by ~11% — MMAS_N=1 gives one matB load
  reused across 4 matA.
- **sg64 is bad on RADV** here (2840), and **K64 blows the LDS budget** (M128 →
  30 KB, occ drops to LDS-limited 7).

## Result vs definition-of-done

- Microbench 8da4w kernel geomean (12 shapes, 1B/3B/8B): **1.00x** vs tiled ✓
- 4w geomean 4.01x, OVERALL WMMA 2.01x — 4w/SDPA unregressed (untouched shaders).
- E2e 1B 8da4w buffer prefill. Interleaved paired A/B (WMMA then tiled
  back-to-back each round, so thermal drift cancels), 5 rounds, medians:
  WMMA **~1936 tok/s** vs tiled **~1949 tok/s** — parity, tiled marginally
  (~0.7%) ahead; WMMA won one round and tied another. At session start the gap
  was ~1843 (WMMA) vs ~1957 (tiled), i.e. ~6%; the occupancy retile closes it
  to under 1%. The int8 WMMA path thus reaches parity with the tiled dot4 path
  at e2e but does not clearly beat it — the ~1957 absolute figure reflects a
  cooler prior thermal state (tiled itself measures ~1949 here). The kernel is
  at its practical limit vs tiled (microbench 1.00x); the remaining sub-1% is
  in the noise band, not a structural win still on the table (specs/037 ruled
  out RowMajor B, a SWAR unpack was perf-neutral, and A/B LDS stores are
  layout-bound to scalar b32). Per specs/039 the largest remaining e2e lever is
  not the linear at all but quantize_and_pack (11% of prefill, the 8da4w tax).
