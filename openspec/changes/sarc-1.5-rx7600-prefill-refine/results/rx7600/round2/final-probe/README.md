# Real-text probe of the final stack (owner decisions D1 / D3, items 1 to 3), build `f2`, 2026-10-09 11:45 to 11:59 UTC

`tools/chain10.sh` / `tools/probe_run.sh` / `tools/probe_compare.py`: 32 real-text prompts per model (`../../probe/prompts-*.txt`), the next-token logits of four arms per cell
(pristine parent and final stack, each default and with `ET_VK_FORCE_TILED_LINEAR=1`), `logits_probe` built from the exported tree of `f2` (commit `73648f5bd`). Table: `real-text-compare.csv`.

- Gross-divergence check (D3 item 3): **ok in all six cells**: the mean KL of final-default against parent-default is at most 0.030 nat (1B 8da4w) against the limit of 0.5, the top-1 token differs
  on at most 4 of 32 prompts (limit one third).
- The final stack's rows (4w: 0 of 32 top-1 differences everywhere, KL mean 1.3e-5 to 2.9e-5; 8da4w: 4 / 1 / 3 of 32, KL mean 0.030 / 0.035 / 0.014 nat) are the numbers of the pristine parent's own
  tiled-versus-default pair for scale (parent-tiled vs parent-default: 8da4w 1 / 2 / 2 of 32, KL mean 0.049 / 0.031 / 0.035 nat).
- **The whole table (first 11 columns, every row) is identical to round 1's final-stack table** (`../../sessions/final/probe/real-text-compare.csv`), as it has to be: the linear outputs of the
  round-2 kernels are byte-identical to the pristine parent's (24 of 24 shapes) and the attention kernels are byte-identical to round 1's.
