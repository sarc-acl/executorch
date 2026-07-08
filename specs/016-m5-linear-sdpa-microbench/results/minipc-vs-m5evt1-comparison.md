# MiniPC vs M5 EVT1: Coopmat Microbenchmark Headline Comparison

One-glance summary of `specs/016`'s M5 EVT1 results against the two MiniPC
reports it mirrors (`specs/007`, `specs/010`). Full detail, per-row tables,
and dispatch/correctness citations are in each report; this file only
restates the headline figures side by side.

| Benchmark | MiniPC (`rocky-ryzen`) | M5 EVT1 | Direction |
|---|---:|---:|---|
| Linear `4w` (`specs/007` vs `linear-coopmat-microbench-report.md`) | +60.6% | **+67.0%** | Same direction, M5 EVT1 larger |
| Linear `8da4w` (`specs/007` vs `linear-coopmat-microbench-report.md`) | **-15.2%** (regression) | **+75.8%** | **Opposite direction** -- `8da4w` coopmat helps on M5 EVT1, hurts on MiniPC |
| SDPA coopmat (`specs/010` vs `sdpa-coopmat-microbench-report.md`) | +66.8% | **+79.5%** | Same direction, M5 EVT1 larger |

**Takeaway**: coopmat/WMMA is a real, `real_effect`-classified win on M5
EVT1 for every one of these three benchmarks -- including `8da4w`, where
MiniPC actually showed a regression. This is the first tool-reliable
(microbenchmark-tier, dispatch- and correctness-confirmed) evidence this
workstream has that coopmat delivers on the Samsung target, independent of
`specs/015`'s now-suspect e2e ETDump dispatch-confirmation method (see
workspace `open-questions.md` Q11).

**Not claimed here**: a model-level e2e number. All figures above are
shader-microbenchmark tier (constitution Principle IV tier 1) -- `specs/015`
remains the (currently blocked-on-Q11) source for any tier-2 e2e claim.
