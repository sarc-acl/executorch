# Round 2 final verification: the final stack against the pristine parent, session `r2-final`, 2026-10-09 06:17 to 09:41 UTC (gate), 09:41 to 10:16 (evidence)

Build `f2` = exported commit `73648f5bd` (the branch head at the time; no local patch; only documentation and result files were committed after it; native toolchain,
Mesa 26.2.3 user-space RADV `31e9a6b2e9`). Candidate arm: `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7
ET_VK_SARC_780M_SDPA_FUSED=fused3sb_d64_t32x32g11s32rko,fused3sb_d128_t16x64g11s32rko ET_VK_SARC_RX7600_PROFILE=rx7600-refine5` (softmax `r3`, the fused attention node with its
`subgroupBarrier()` fix, the 8da4w kernel with 24-byte A staging rows, the 4w kernel with 72 / 88-byte A / B staging rows). Parent arm: the pristine parent `f5f1bf10c`
(build `parent`, `ET_VK_SARC_UNVERIFIED=1` only), the parent of both rounds.

Tok/s, median of the first 5 valid runs per arm (`runs.csv`: 60 timed runs counted, 62 rows, 1 invalid `8b 8da4w parent r5 host_build`, replaced; recomputed from `runs.csv`):

| cell | pristine parent | final stack (round 2) | gain | repeat spread parent / final | published 2026-09-28 | final vs published | next token (2048 / real / check) |
|---|---:|---:|---:|---|---:|---:|---|
| 1B 4w | 7846.74 | 11070.30 | **+41.08 %** | 0.00 / 1.09 % | 7787 | +42.16 % | SAME / SAME / SAME |
| 1B 8da4w | 7340.50 | 10666.70 | **+45.31 %** | 0.36 / 1.04 % | 7340 | +45.32 % | SAME / SAME / SAME |
| 3B 4w | 3292.60 | 4266.67 | **+29.58 %** | 0.16 / 0.62 % | 3287 | +29.80 % | SAME / SAME / SAME |
| 3B 8da4w | 3084.34 | 4104.21 | **+33.07 %** | 0.15 / 0.40 % | 3080 | +33.25 % | SAME / SAME / SAME |
| 8B 4w | 1517.04 | 1885.82 | **+24.31 %** | 0.15 / 0.18 % | 1517 | +24.31 % | SAME / SAME / SAME |
| 8B 8da4w | 1402.74 | 1813.99 | **+29.32 %** | 0.07 / 0.18 % | 1403 | +29.29 % | SAME / SAME / SAME |

Geomean **+33.59 %** over the pristine parent (round 1's final stack: +26.90 %). Clock 2487 to 2592 MHz (floor 2420), start 42 to 54 C. The 1B cells are 196 to 270 ms
long and the runner's timer step is 1 ms (0.4 to 0.5 %): the repeat spreads of 1 % are one to two steps.

Where the gain comes from (warm ETDump of this session, ms per 2048-token prefill, pristine parent -> final; `trace-families.csv`, `trace-totals.csv`):

| cell | total | linear GEMM | attention (QK^T, softmax, AV -> fused, KV update) | quantize / elementwise / norm / rope |
|---|---:|---:|---:|---:|
| 8B 4w | 1336.1 -> 1071.3 | 1032.3 -> 914.0 | 194.6 -> 49.3 | 81.2 -> 81.3 |
| 8B 8da4w | 1447.3 -> 1115.4 | 1111.1 -> 925.1 | 194.2 -> 48.8 | 114.4 -> 115.2 |
| 1B 4w | 248.3 -> 171.5 | 142.9 -> 128.9 | 76.5 -> 13.4 | 21.5 -> 22.2 |
| 1B 8da4w | 265.3 -> 178.1 | 153.6 -> 128.9 | 76.4 -> 13.3 | 28.1 -> 29.1 |

Percent of the cited roofs (43.42 TFLOP/s fp16, 43.90 TOP/s int8, 2026-09-28, not re-measured: owner decision of 2026-10-08 23:38 UTC, answer 1; the 2 x M x N x K of the twelve
layer shapes divided by the GEMM family time above): linear GEMM at 71.2 / 70.4 % (1B 4w / 8da4w), 71.8 / 70.3 % (3B), **72.0 / 70.4 % (8B)** of the roof. The best case measured
in the 8da4w kernel with almost nothing but the WMMA loop (`round2/r2d-screen-8da4w-summary.txt`, `abl55`) is about 78 %.

## Gate (R7, R11) of the final stack, all on build `f2`

- `verify.sh` unmodified (`--models 1b,3b,8b --schemes 4w,8da4w --pdiff`) against `s0-parent-verify`: 32 / 32 lines, the two that differ are the dispatched-kernel lines of the
  linear 4w / 8da4w check (`verify-compare.txt`; the parent's own `linear <scheme> rc=1` lines and `correctness rc=1` are in the snapshot too); production-diff errors identical
  in every case; default vs tiled SAME; decode 31 tokens.
- SDPA correctness tiers `all`, `extended`, `full`: 12 passes each, 0 failed, 0 mismatches, `pairing=ok`, plus one control pass each with the parent's table kernels
  (`sdpa-correctness/summary.txt`: 39 lines, all `failed=0 mismatch_nonzero=0 pairing_not_ok=0`).
- Shipped SPIR-V: byte-identical to the native parent build (PASS, 53 variants); the 14 differences against `sarc/golden/spirv.json` are the parent's own (native glslc, `golden.txt`).
- Outputs of the 24 prefill linear shapes (4w and 8da4w, real model shapes): byte-identical to the pristine parent's (`linear-bitwise/bitwise.txt`: 24 identical, 0 differ). The first
  attempt of this comparison was cut by a full root filesystem (zero-byte dumps: 18 identical, 6 "differ"); it is kept under `<artifacts>/superseded/r2-final-enospc/` and was rerun.
- Reference-error evidence of the attention arithmetic (D3, the only arithmetic change against the parent): `sdpa-error/error.csv`, 17 rows, 16 `yes`, one `NO`
  (`peaked_tiny_gqa_s256`, S = 256, max-error ratio 1.140, rms ratio 0.812; not a production shape) -- the same rows and values as in round 1 (`../../sdpa-error/`), as expected: the
  attention kernels are byte-identical to round 1's. Raw outputs differ from the parent's in 21 of 21 cases (`sdpa-error/bitwise.txt`), by design. The real-text probe: see
  `../../round2/final-probe/`.
- Files changed since the pristine parent outside the dev zone (`backends/vulkan/runtime/graph/ops/{glsl,impl}/sarc_dev/`, `backends/vulkan/test/sarc_dev/`, `sarc/`, `openspec/`): none;
  nothing under `sarc/tools` or `sarc/golden` (`files-outside-dev-zone.txt`).
