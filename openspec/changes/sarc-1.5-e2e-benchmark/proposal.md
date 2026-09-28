# sarc-1.5-e2e-benchmark

## Why

The porting-time 1.5 numbers were single runs, and their "stock" arm was the dev binary with its rows inert. This
change re-measures the shipped release against real upstream 1.5 with repeats, so the result can be published.

## What

A fresh end-to-end prefill campaign on 2026-09-28:
- **Builds:** stock `release/1.5` (985c1ceccc + the compile-only `<algorithm>` backport) vs tag `sarc/1.5-r2`
  (fd9250c60), both built fresh.
- **Grid:** 5 GPUs × 3 Llama models × 2 schemes × 5 interleaved repeats.
- **Also collected:** next-token checks and ETDump dispatch evidence.

## Result

SARC is 1.48–5.35× faster than stock in all 30 cells. The geometric mean is 3.17× for 4w, 2.39× for 8da4w and
2.75× overall. Details are in [REPORT.md](REPORT.md).

The raw logs, ETDumps and builds stay outside the repository, under
`sarc-acl/.artifacts/e2e-1.5-2026-09-28/`. `results/` holds the aggregated CSVs, figures and scripts.
