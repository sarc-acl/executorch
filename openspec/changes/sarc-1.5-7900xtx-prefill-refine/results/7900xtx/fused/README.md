# Fused attention variant screen (2026-10-08 19:47 to 19:50 UTC)

`tools/q-screens.sh` -> `tools/fused_screen.sh`: `test_llama_microbench --sdpa` (`ET_VK_SDPA_PERF_RUNS=20,8`), the parent binary with
`ET_VK_SARC_780M_PROFILE=c7`, 3 rounds, order rotated per round, one `gl.sh` job per (round, pair); total fused kernel + copy pass at S = 2048
per model. Rule (R8): a variant replaces the incumbent (the 780M's `fused3_d64_t32x32g11s32rko`, `fused3_d128_t16x64g11s32rko`) only if it is at
least 3 % faster in every round on every model of its head dimension; among several the fastest median wins (`pick-analysis.txt`, `pick.out`).
Result: `fused3_d64_t32x32g11s32rk` and `fused3_d128_t16x64g11s32rk` (two-pass, tile-packed K / V copies, no online softmax). The online variants
(`rko`) are 1.3 to 2.6 times slower than the two-pass ones on this card with AMDVLK.

`fused-guard-attempt1-pattern-bug.log`: the first correctness pass of the incumbent pair (tier all, 4 passed, 0 failed, the fused kernel dispatched
in all 4 cases); the queue stopped on it only because the guard grepped for the wrong kernel-name prefix (`fused=fused3...` instead of
`fused=sarc_dev_780m_sdpa_fused3...`). The corrected guard ran again (`fused-guard.log`) and the queue went on.
