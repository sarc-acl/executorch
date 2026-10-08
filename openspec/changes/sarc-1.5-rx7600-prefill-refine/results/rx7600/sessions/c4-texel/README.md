# Candidate 4: whole-texel 8da4w weight staging on every shape (profile `rx7600-refine3`), session `c4-texel`, 2026-10-08 03:23 to 04:36 UTC

Parent arm = candidate 3 (build `c3`, commit `6ebf39484`, `rx7600-refine2`). Candidate arm = build `c4` (commit `129cea7ac`,
golden PASS against `golden-ref-parent.json`) with `ET_VK_SARC_RX7600_PROFILE=rx7600-refine3`: the same 4w picks, and on every
8da4w prefill shape `sarc_dev_linear_dq8ca_coopmat_zpg_bt_t128x64k64g42s32` (texel-wise weight staging, `gen_bt.py`) instead of
candidate 3's `t256x64k64g48s32afmb1`. The kernel screen had it at 0.977 to 1.011 against candidate 3's kernel (worst round),
so this measures the family as a whole, to complete the stop rule.

Tok/s, median of 5 valid runs per arm (`runs.csv`: 62 timed rows, 61 valid, 1 `host_build` (3B 8da4w cand r5, replaced); `summary.csv`):

| cell | parent (candidate 3) | candidate 4 | change |
|---|---:|---:|---:|
| 1B 4w | 10502.60 | 10502.60 | +0.00 % |
| 1B 8da4w | 10291.50 | 10343.40 | +0.50 % |
| 3B 4w | 3984.44 | 3984.44 | +0.00 % |
| 3B 8da4w | 3953.67 | 3953.67 | +0.00 % |
| 8B 4w | 1747.44 | 1748.93 | +0.09 % |
| 8B 8da4w | 1738.54 | 1712.37 | -1.51 % |

Geomean **-0.15 %**: inside the +-2 % band, no gain. **Not adopted**: candidate 3's kernel stays (adoption rule in `proposal.md`).
Next token SAME in every cell on the three prompts. `verify.out`: 30 of 32 lines as the snapshot, the two kernel-name lines
differ (`verify-compare.txt`); all texture3d production-diff cases ALL PASSED. This is the first gated candidate under 2 %.
