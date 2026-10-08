# Final verification and final session (R11), 2026-10-08 08:11 to 11:58 UTC

Build `final`: native build of commit `18cc0d53a91e42f1100cc08c7f2c8327c1c77770` (the committed head when it started; every later
commit touches only `openspec/changes/sarc-1.5-rx7600-prefill-refine/{results,tools,STATUS.md,proposal.md}`), exported from the git object
stores (`.artifacts/src/rx7600/final`, manifest beside the build), no local patch. Mesa 26.2.3 `31e9a6b2e9` for both arms.
Parent arm = pristine `f5f1bf10c` (build `parent`), env `ET_VK_SARC_UNVERIFIED=1` only.
Final stack = build `final` with
`ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7 ET_VK_SARC_780M_SDPA_FUSED=fused3sb_d64_t32x32g11s32rko,fused3sb_d128_t16x64g11s32rko ET_VK_SARC_RX7600_PROFILE=rx7600-refine2`
(softmax `r3`, fused attention kernel with the subgroup barrier, a linear kernel per layer shape).

## Timed session (`runs.csv`: 60 timed runs, all valid; `summary.csv`)

| cell | published 2026-09-28 | parent (pristine) | final stack | gain over parent | gain over published | spread parent / final | next token (2048 / real / check) |
|---|---:|---:|---:|---:|---:|---|---|
| 1B 4w | 7787 | 7846.74 | 10502.60 | **+33.85 %** | +34.9 % | 0.38 / 0.51 % | SAME / SAME / SAME |
| 1B 8da4w | 7340 | 7340.50 | 10343.40 | **+40.91 %** | +40.9 % | 0.36 / 2.46 % | SAME / SAME / SAME |
| 3B 4w | 3287 | 3297.91 | 3984.44 | **+20.82 %** | +21.2 % | 0.32 / 0.00 % | SAME / SAME / SAME |
| 3B 8da4w | 3080 | 3079.70 | 3953.67 | **+28.38 %** | +28.4 % | 0.30 / 0.00 % | SAME / SAME / SAME |
| 8B 4w | 1517 | 1517.04 | 1747.44 | **+15.19 %** | +15.2 % | 0.37 / 0.34 % | SAME / SAME / SAME |
| 8B 8da4w | 1403 | 1402.74 | 1738.54 | **+23.94 %** | +23.9 % | 0.14 / 0.17 % | SAME / SAME / SAME |

Geomean **+26.90 %** over the pristine parent (the six-cell product of the candidates' own sessions, +1.48 x +18.22 x +5.49 %, is +26.6 %).
Clock 2490 to 2597 MHz (floor 2420), start 40 to 54 C. The parent medians reproduce the A/A session's (`aa2`) within 0.4 %.
Per-cell timer step 1 ms (0.38 % at 1B 4w).

## Gate items on the committed build

| item | result |
|---|---|
| golden | `golden.txt`: PASS against `golden-ref-parent.json` (53 shipped variants, the native-glslc reference of the parent build); against `sarc/golden/spirv.json` the same 14 variants differ as in the parent build (native glslc, other devices' owners): **pending** |
| `verify.sh` unmodified | `verify.out` is identical (rates removed) to the candidate-3 and M2a gates; against `s0-parent-verify` 30 of 32 lines identical, the two that differ are the dispatched kernel names of the linear lines (`verify-compare.txt`); all texture3d and 8da4w production-diff cases ALL PASSED; the 4w `buffer` FAILED lines and `correctness rc=1` are the parent's own |
| SDPA tiers `all` / `extended` / `full`, 12 passes each | 0 failed, 0 mismatches, `pairing=ok` in all 36 (+3 table control passes) (`sdpa-correctness/summary.txt`) |
| D3.1, error against the fp64 reference (`sdpa-error/error.csv`, one coherent run, 17 rows) | production shapes (tier full, S = 2048 and 1024): 4 of 4 `yes`; `extended` 8 of 8 `yes`; `peaked` 4 of 5 `yes`; the one `NO` is `peaked_tiny_gqa_s256` (S = 256, not a production shape; max ratio 1.140, rms ratio 0.812), unchanged from `sdpa-error/README.md` of candidate 2. rms error of the final stack is 0.50 to 0.53 x the parent's on every non-peaked case |
| D3.2 / D3.3, real text (`probe/real-text-compare.csv`, 32 prompts per cell, final vs pristine, with parent-tiled vs parent-default beside) | top-1 differences 0 / 0 / 0 / 0 on 4w, 4 / 1 / 3 on 8da4w (1B / 3B / 8B) of 32; mean KL 1.3e-5 to 3.5e-2 nat; gross-divergence check ok (limits 10.7 prompts, 0.5 nat). Same numbers as candidate 2's probe: candidates 3 and M2a are bit-identical to it |
| outputs | the SDPA output differs from the pristine parent in 21 of 21 cases (different kernel and accumulation: judged by D3, `sdpa-error/bitwise.txt`); linear outputs identical (candidate 3) |

Record: **ACCEPTED (reference-error rule, owner decision 2026-10-04)** for the fused attention kernel (the only arithmetic change); evidence
`sdpa-error/`, `probe/`; no next-token item differs.

Files changed since the parent outside the dev zone: none (`files-outside-dev-zone.txt` is empty). `check-no-build.txt`: `check.sh --no-build` PASS.
