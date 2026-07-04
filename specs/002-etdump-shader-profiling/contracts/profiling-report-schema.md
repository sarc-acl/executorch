# Contract: Profiling Report Schema

This is the interface between this feature and anyone consuming its output
(a future contributor deciding where to focus WMMA/coopmat work, or a future
feature comparing profiled coopmat-enabled runs against this baseline).

## Raw record: `results/raw/<model>_<scheme>.json`

One file per configuration (six total, matching `001`'s naming:
`llama-3.1-8b_4w.json`, `llama-3.1-8b_8da4w.json`, `llama-3.2-3b_4w.json`,
`llama-3.2-3b_8da4w.json`, `llama-3.2-1b_4w.json`, `llama-3.2-1b_8da4w.json`).

```json
{
  "config": {"model": "llama-3.1-8b", "scheme": "4w", "device": "rocky-ryzen", "dispatch_path": "tiled_baseline"},
  "phases": {
    "prefill": {
      "status": "ok",
      "failure_reason": null,
      "etdump_path": "string",
      "phase_wall_clock_us_profiled": 0.0,
      "phase_wall_clock_us_baseline": 0.0,
      "attributed_pct": 0.0,
      "decode_window_steps": null,
      "aggregated": [
        {"kernel_name": "string", "shape": {"m": 2048, "k": 0, "n": 0}, "total_time_us": 0.0, "invocation_count": 0, "pct_of_phase": 0.0, "category": "string"}
      ],
      "category_rollup": [
        {"category": "string", "total_time_us": 0.0, "pct_of_phase": 0.0}
      ],
      "raw_invocations_path": "string"
    },
    "decode": {
      "status": "ok",
      "failure_reason": null,
      "etdump_path": "string",
      "phase_wall_clock_us_profiled": 0.0,
      "phase_wall_clock_us_baseline": 0.0,
      "attributed_pct": 0.0,
      "decode_window_steps": 8,
      "aggregated": [
        {"kernel_name": "string", "shape": {"m": 1, "k": 0, "n": 0}, "total_time_us": 0.0, "invocation_count": 0, "pct_of_phase": 0.0, "category": "string"}
      ],
      "category_rollup": [
        {"category": "string", "total_time_us": 0.0, "pct_of_phase": 0.0}
      ],
      "raw_invocations_path": "string"
    }
  }
}
```

Rules a consumer can depend on:

- `status` may be `"ok"`, `"failed"`, or `"pending"` (mirroring `001`'s
  contract convention). If `"failed"`, `failure_reason` is a non-empty
  string and the numeric/array fields are `null`/empty, never fabricated.
- `aggregated[].shape` is `null` (not a zero-filled object) when the kernel
  is not a matrix multiplication — never silently coerced to a numeric
  shape.
- `category_rollup` percentages plus `(1 - attributed_pct)` sum to 1.0 (an
  explicit "unattributed" remainder is expected and honest, not an error).
- `raw_invocations_path` points to a JSON file (under `results/raw/`, one per
  config+phase) holding the full list of Kernel Invocation (raw) records —
  the companion data from the Clarifications answer. It is never deleted
  even after aggregation.
- `phase_wall_clock_us_baseline` is read from `001-minipc-baseline-benchmarks/results/raw/<model>_<scheme>.json` — this feature does not re-derive or override that number, only cites it.

## Rendered summary: `results/profiling-report.md`

One document, organized by model then scheme then phase, presenting:
1. A category rollup table per (config, phase) — category | % of phase | total time.
2. The top N aggregated kernel entries by time per (config, phase) — kernel | shape | count | total time | % of phase.
3. A reconciliation line per (config, phase): "`attributed_pct`% attributed to named kernels; profiled phase total was `phase_wall_clock_us_profiled` vs. `001` baseline `phase_wall_clock_us_baseline`."

It links to `001-minipc-baseline-benchmarks/results/baseline-report.md` at
the top rather than repeating that feature's throughput numbers, and links
to each config's raw JSON for full detail.
