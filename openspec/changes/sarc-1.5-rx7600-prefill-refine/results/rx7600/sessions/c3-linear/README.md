# Candidate 3: a linear kernel per layer shape (profile `rx7600-refine2`), session `c3-linear`, 2026-10-08 01:53 to 02:57 UTC

Parent arm = candidate 2 (parent binary `cfb5fb8c...`, `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7
ET_VK_SARC_780M_SDPA_FUSED=fused3_d64_t32x32g11s32rko,fused3_d128_t16x64g11s32rko`). Candidate arm = build `c3` (native, commit
`6ebf39484`, golden PASS against `golden-ref-parent.json`, the 14 differences against `sarc/golden/spirv.json` are the parent's own
native-glslc ones) with the same env plus `ET_VK_SARC_RX7600_PROFILE=rx7600-refine2`. Picks: `results/rx7600/screens/`.

Tok/s, median of 5 valid runs per arm (`runs.csv`: 60 timed runs, all valid; `summary.csv`):

| cell | parent (candidate 2) | candidate 3 | gain | next token (2048 / real / check) |
|---|---:|---:|---:|---|
| 1B 4w | 10395.90 | 10449.00 | +0.51 % | SAME / SAME / SAME |
| 1B 8da4w | 9481.48 | 10291.50 | **+8.54 %** | SAME / SAME / SAME |
| 3B 4w | 3938.46 | 3976.70 | +0.97 % | SAME / SAME / SAME |
| 3B 8da4w | 3586.69 | 3946.05 | **+10.02 %** | SAME / SAME / SAME |
| 8B 4w | 1721.01 | 1747.44 | +1.54 % | SAME / SAME / SAME |
| 8B 8da4w | 1552.69 | 1738.54 | **+11.97 %** | SAME / SAME / SAME |

Geomean **+5.49 %** (over candidate 2). The 4w cells are inside the +-2 % band (the screen picks only 6 of 12 shapes, 3 to 4 %
each); every 8da4w cell is outside it. Clock 2490 to 2564 MHz (floor 2420), start 46 to 54 C.

Gate: `verify.out` against `s0-parent-verify`: 30 of 32 lines identical; the two that differ are the `linear 4w` and
`linear 8da4w` lines, which list the dispatched kernel names (the intended change; `verify-compare.txt`). Every
`production-diff` line equals the snapshot's (the 4w `buffer` FAILED lines are the parent's own, in the release-1.5 fallback
kernels). Linear error against the sampled reference (`pdiff-error-vs-parent.txt`): identical readings in all 24 texture3d
production-diff cases. Output of the 24 real prefill linear shapes, parent env against candidate env, byte for byte:
24 of 24 IDENTICAL (`linear-bitwise/bitwise.txt`, dispatched kernels in `linear-bitwise/kernels.txt`): the candidate
does not change arithmetic (same K order and accumulation precision), so D3 is not needed. No attention kernel changed:
the SDPA tiers were not re-run for this candidate (they are part of the final verification).
Traces (`trace.out`; the ETDump files stay in `.artifacts/stage/c3-linear/trace/`): total dispatch time of the 8B 8da4w prefill 1304.2 -> 1163.6 ms (-10.8 %); `trace.out` holds only the last cells' summary lines.
