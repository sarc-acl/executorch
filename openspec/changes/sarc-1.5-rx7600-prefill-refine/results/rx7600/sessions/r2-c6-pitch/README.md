# Round 2, candidate 1: 8da4w A staging row pitch of 24 bytes (profile `rx7600-refine4`), session `r2-c6-pitch`, 2026-10-09 03:57 to 05:03 UTC

Parent arm = round 1's final stack: build `final` (commit `18cc0d53a`), `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7
ET_VK_SARC_780M_SDPA_FUSED=fused3sb_d64_t32x32g11s32rko,fused3sb_d128_t16x64g11s32rko ET_VK_SARC_RX7600_PROFILE=rx7600-refine2`. Candidate arm = build
`c6` (native, commit `36c7d1cc0`; `c5` of the same commit lost its traced half to a full root filesystem and is superseded) with the same environment
and `ET_VK_SARC_RX7600_PROFILE=rx7600-refine4`. Mesa 26.2.3 (user-space RADV, `31e9a6b2e9`). Kernel: `sarc_dev_rx7600_x_linear_dq8ca_coopmat_zpg_t256x64k64g48s32pa6pb4csha`
on the twelve 8da4w prefill shapes: the shipped 256 x 64 tile (K step 64, 32 waves) with the A staging rows 24 instead of 16 bytes apart in shared
memory and the texture drain tile aliased onto the A buffer. Kernel screen: `../../round2/r2g-screen-8da4w-summary.txt`.

Tok/s, median of the first 5 valid runs per arm (`runs.csv`: 60 timed runs, all valid; `summary.csv`; recomputed from `runs.csv`):

| cell | parent (round 1 final) | candidate 1 | gain | repeat spread parent / cand | next token (2048 / real / check) |
|---|---:|---:|---:|---|---|
| 1B 4w | 10449.00 | 10449.00 | +0.00 % | 0.51 / 1.02 % | SAME / SAME / SAME |
| 1B 8da4w | 10291.50 | 10666.70 | **+3.65 %** | 1.00 / 2.06 % | SAME / SAME / SAME |
| 3B 4w | 3976.70 | 3968.99 | -0.19 % | 0.19 / 0.39 % | SAME / SAME / SAME |
| 3B 8da4w | 3946.05 | 4104.21 | **+4.01 %** | 0.19 / 0.00 % | SAME / SAME / SAME |
| 8B 4w | 1748.93 | 1751.92 | +0.17 % | 0.43 / 0.34 % | SAME / SAME / SAME |
| 8B 8da4w | 1740.02 | 1813.99 | **+4.25 %** | 0.17 / 0.18 % | SAME / SAME / SAME |

Geomean **+1.96 %** (+1.961 % recomputed): under 2 % **by construction** (three cells are not touched); each 8da4w cell is outside the +-2 % noise band and
no 4w cell moves (-0.19 to +0.17 %). Adoption: rule (b) of the clarification in `proposal.md` (written before any end-to-end number); for the stop rule
this counts as **one candidate under 2 % geomean**. Clock 2491 to 2536 MHz (floor 2420), start 46 to 54 C.

Where the gain comes from (warm ETDump, `trace-totals.csv` / `trace-families.csv`, ms per 2048-token prefill, prefill GEMM family, parent -> candidate):
1B 8da4w 135.8 -> 128.8, 3B 8da4w 395.0 -> 375.1, 8B 8da4w 976.5 -> 926.9 (-5.1 %); the 4w cells are unchanged (8B 4w 1003.3 -> 999.0). The kernel-level screen
(`round2/r2g-screen-8da4w-summary.txt`) had 1.051 over the twelve shapes; the phase twins (`round2/r2e-phase-compare.txt`) show the phase that holds the LDS
fragment loads at 0.66 of the shipped tile's cycles.

Gate (R7): `verify.sh` unmodified on the candidate binaries, compared line by line with `s0-parent-verify`: 32 / 32 lines, the two that differ are the
dispatched-kernel lines of the linear 4w / 8da4w check (`verify-compare.txt`); next token SAME in all cells on the timed, the real-text and the
unaligned (1972-token) prompt; golden: shipped SPIR-V byte-identical to the native parent build (PASS, 53 variants), the 14 differences against
`sarc/golden/spirv.json` are the parent's own (native glslc, `golden.txt`); output of every one of the 24 prefill linear shapes (4w and 8da4w) byte-identical
to the parent's (`linear-bitwise/bitwise.txt`: 24 identical, 0 differ), so the arithmetic is unchanged and no reference-error evidence is needed (D3 does
not apply); no attention kernel changed (no SDPA tiers). The new shader is a copy of the release body with an address change: every shared write is a
per-thread slot of its own, every read follows `memoryBarrierShared(); barrier();` as in the release body (read for races, see `proposal.md`).
