# Round 2 final session: the final stack against round 1's final stack, session `r2-final-r1`, 2026-10-09 10:16 to 11:44 UTC

Parent arm = round 1's final stack: build `final` (commit `18cc0d53a`), `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7
ET_VK_SARC_780M_SDPA_FUSED=fused3sb_d64_t32x32g11s32rko,fused3sb_d128_t16x64g11s32rko ET_VK_SARC_RX7600_PROFILE=rx7600-refine2`. Candidate arm = build `f2` (commit `73648f5bd`) with the same
environment and `ET_VK_SARC_RX7600_PROFILE=rx7600-refine5`. Timed session and next token only (the gate of the final stack is in `../r2-final/`).

| cell | round 1 final | round 2 final | gain | repeat spread parent / final | next token (2048 / real / check) |
|---|---:|---:|---:|---|---|
| 1B 4w | 10502.60 | 11130.40 | **+5.98 %** | 0.00 / 1.07 % | SAME / SAME / SAME |
| 1B 8da4w | 10291.50 | 10666.70 | **+3.65 %** | 1.99 / 9.57 % | SAME / SAME / SAME |
| 3B 4w | 3984.44 | 4266.67 | **+7.08 %** | 0.39 / 0.42 % | SAME / SAME / SAME |
| 3B 8da4w | 3953.67 | 4104.21 | **+3.81 %** | 0.19 / 0.40 % | SAME / SAME / SAME |
| 8B 4w | 1748.93 | 1887.56 | **+7.93 %** | 0.17 / 0.18 % | SAME / SAME / SAME |
| 8B 8da4w | 1740.02 | 1813.99 | **+4.25 %** | 0.17 / 0.62 % | SAME / SAME / SAME |

Geomean **+5.44 %** (+5.436 % recomputed from `runs.csv`: 60 timed runs, all valid; `summary.csv`). The 9.57 % spread of 1B 8da4w is one slow candidate run (the 1B prefill is
about 190 ms; the median is unaffected); the parent arm's numbers reproduce round 1's session of the same build within one timer step. Clock 2487 to 2551 MHz (floor 2420),
start 40 to 54 C. Every cell is outside the +-2 % band.

The two candidates in order (each against its own parent, `../r2-c6-pitch/`, `../r2-c7-q4/`): candidate 1 +1.96 % (8da4w cells +3.65 / +4.01 / +4.25 %), candidate 2 +3.33 % (4w cells
+5.38 / +7.52 / +7.46 %); together +5.44 % here (the product of the two geomeans is +5.35 %, within the session noise).
