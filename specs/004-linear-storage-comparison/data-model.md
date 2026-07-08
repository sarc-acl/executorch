# Data Model: Linear Shader Storage-Type Baseline Study

## Storage Comparison Case

One entry per (model, scheme, regime, op) — 96 total (matching `001`'s
existing microbenchmark catalog), each carrying both storage types'
measurements.

| Field | Type | Notes |
|---|---|---|
| `model` / `scheme` / `regime` / `op` | string | Same grain as `001`'s microbenchmark (e.g. `llama-3.2-1b`/`4w`/`prefill`/`w1_gate`) |
| `m` / `k` / `n` | int | Shape, from `001`'s `results/shapes.json` (duplicated in the harness's `kModels`) |
| `texture3d_mean_us` / `texture3d_stdev_us` | float | From the harness's `RESULT` line, `storage=texture3d` |
| `buffer_mean_us` / `buffer_stdev_us` | float | From the harness's `RESULT` line, `storage=buffer` |
| `texture3d_kernel` / `buffer_kernel` | string | The captured kernel name for each — MUST both be from the tiled/coop family (never contain `coopmat`); a `coopmat` name appearing here is a hard failure of Decision 2's forcing mechanism, not a data point |
| `relative_diff_pct` | float | `(buffer_mean_us - texture3d_mean_us) / texture3d_mean_us * 100` — positive means Buffer is slower |
| `significance` | enum | `real_effect` / `noise`, per Research Decision 3's non-overlapping-band rule |
| `baseline_cross_check` | enum or null | `consistent` / `diverged` against `001`'s published Texture3D number for this case (Decision 4); null if `001` has no matching case (shouldn't happen given identical catalogs, but not assumed) |

## Storage Comparison Report

The consolidated document (US3): a top-level verdict per regime (prefill,
decode) — "Buffer storage is effectively free" / "Buffer storage has a
measurable cost" / "Buffer storage has a measurable benefit" — derived from
how many of that regime's 48 cases show a `real_effect` significance and in
which direction, followed by the full 96-row case table. Any case whose
`texture3d_kernel`/`buffer_kernel` is not from the expected tiled/coop family,
or whose Buffer variant could not be constructed/dispatched at all, is listed
in a separate "infeasible / contaminated" section per FR-009 rather than
silently dropped from the main table.

No lifecycle/state transitions — this is a one-shot capture-and-compare,
same shape as `001`'s microbenchmark report.
