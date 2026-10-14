# Round 2, candidate 2: 4w 256 x 128 tile with 72-byte A and 88-byte B staging rows (profile `rx7600-refine5`), session `r2-c7-q4`, 2026-10-09 05:03 to 05:53 UTC

Parent arm = candidate 1: build `c6` (commit `36c7d1cc0`) with `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7
ET_VK_SARC_780M_SDPA_FUSED=fused3sb_d64_t32x32g11s32rko,fused3sb_d128_t16x64g11s32rko ET_VK_SARC_RX7600_PROFILE=rx7600-refine4`. Candidate arm = build `c7` (native, commit
`d6d67ba78`) with the same environment and `ET_VK_SARC_RX7600_PROFILE=rx7600-refine5`. Mesa 26.2.3 (user-space RADV, `31e9a6b2e9`). Kernel:
`sarc_dev_rx7600_x_linear_q4gsw_coopmat_t256x128k32g28s32f32cbtap4bp12` on the twelve 4w prefill shapes (the 780M's 256 x 128 tile with 16 waves, fp32 accumulation,
column-major B staging) with the A staging typed uvec2 and 72 bytes between its rows and 88 bytes between the rows of the B staging (the shipped pitch is 80 bytes for both):
the 16 lanes of a `ds_read_b64` fragment load then fall on distinct LDS banks. Kernel screen: `../../round2/r2h-screen-4w-vs-picks.txt` (12 of 12 shapes at least 1.03
in both rounds against the round-1 pick, geomean of the worst-round ratios 1.067).

Tok/s, median of the first 5 valid runs per arm (`runs.csv`: 62 timed rows, 1 invalid and replaced: `8b 4w parent r2 other_gpu_process`; `summary.csv`; recomputed from `runs.csv`):

| cell | parent (candidate 1) | candidate 2 | gain | repeat spread parent / cand | next token (2048 / real / check) |
|---|---:|---:|---:|---|---|
| 1B 4w | 10449.00 | 11010.80 | **+5.38 %** | 0.51 / 1.08 % | SAME / SAME / SAME |
| 1B 8da4w | 10666.70 | 10666.70 | +0.00 % | 1.04 / 0.52 % | SAME / SAME / SAME |
| 3B 4w | 3976.70 | 4275.57 | **+7.52 %** | 0.39 / 0.42 % | SAME / SAME / SAME |
| 3B 8da4w | 4104.21 | 4104.21 | +0.00 % | 0.20 / 0.40 % | SAME / SAME / SAME |
| 8B 4w | 1754.93 | 1885.82 | **+7.46 %** | 0.26 / 0.18 % | SAME / SAME / SAME |
| 8B 8da4w | 1815.60 | 1815.60 | +0.00 % | 0.09 / 0.18 % | SAME / SAME / SAME |

Geomean **+3.33 %** (+3.334 % recomputed): above 2 %, so the stop rule does not count this candidate as one under 2 %; adopted under rule (a). Each 4w cell is outside the
+-2 % band; the 8da4w cells run the same kernels in both arms (identical medians). Clock 2489 to 2532 MHz (floor 2420), start 48 to 54 C.

Where the gain comes from (warm ETDump, `trace-totals.csv` / `trace-families.csv`, prefill GEMM family, ms per 2048-token prefill, parent -> candidate): 1B 4w 140.3 -> 128.7,
3B 4w 406.1 -> 370.6, 8B 4w 1001.0 -> 913.8 (-8.7 %); the 8da4w cells are unchanged.

Gate (R7): `verify.sh` unmodified on the candidate binaries against `s0-parent-verify`: 32 / 32 lines, the two that differ are the dispatched-kernel lines (`verify-compare.txt`);
next token SAME in all cells on the timed, the real-text and the unaligned prompt; shipped SPIR-V byte-identical to the native parent build (PASS, 53 variants; the 14 differences
against `sarc/golden/spirv.json` are the parent's own, `golden.txt`); the outputs of all 24 prefill linear shapes are byte-identical to candidate 1's
(`linear-bitwise/bitwise.txt`: 24 identical, 0 differ): same arithmetic, so D3 does not apply; no attention kernel changed. Shared-write reading: `proposal.md`.
