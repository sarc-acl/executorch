# Contract: Baseline Report Schema

This is the interface between this feature and any future feature that needs
to compute a delta against this baseline (e.g., a later "MiniPC WMMA-enabled
comparison" feature). Consumers MUST be able to rely on this shape without
re-deriving it from this feature's implementation.

## Raw record: `results/raw/<model>_<scheme>.json`

One file per Benchmark Configuration (six files total: `llama-3.1-8b_4w.json`,
`llama-3.1-8b_8da4w.json`, `llama-3.2-3b_4w.json`, `llama-3.2-3b_8da4w.json`,
`llama-3.2-1b_4w.json`, `llama-3.2-1b_8da4w.json`).

```json
{
  "config": {
    "model": "llama-3.1-8b",
    "scheme": "4w",
    "group_size": 32,
    "device": "rocky-ryzen",
    "dispatch_path": "tiled_baseline",
    "pte_path": "string"
  },
  "e2e": {
    "status": "ok",
    "failure_reason": null,
    "prefill_tokens": 2048,
    "decode_tokens": 1024,
    "prefill_tokens_per_sec": 0.0,
    "decode_tokens_per_sec": 0.0,
    "num_runs": 0,
    "variance": {
      "prefill_tokens_per_sec_stdev": 0.0,
      "decode_tokens_per_sec_stdev": 0.0
    },
    "run_metadata": {
      "git_commit": "string",
      "max_seq_len": 3072,
      "prompt_file": "string",
      "timestamp": "string"
    }
  },
  "microbench": [
    {
      "regime": "prefill",
      "op": "string",
      "shape": {"m": 2048, "n": 0, "k": 0},
      "mean_time_us": 0.0,
      "stddev_us": 0.0,
      "iterations": 0
    }
  ]
}
```

Rules a consumer can depend on:

- `e2e.status` may also be `"pending"` (not yet measured — the microbenchmark
  tier finished before the e2e tier for this configuration) in addition to
  `"ok"` and `"failed"`. A `"pending"` record's tokens/sec fields are `null`
  and `failure_reason` explains why it's pending, not why it failed.
- `config.dispatch_path` is always `"tiled_baseline"` for every record this
  feature produces — a future feature adds records with
  `dispatch_path: "coopmat_enabled"` alongside these; it MUST NOT overwrite
  or mutate these baseline records.
- If `e2e.status` is `"failed"`, `e2e.failure_reason` is a non-empty string
  and the tokens/sec fields are `null`, not `0` (FR-007: a gap is explicit,
  never a silent zero).
- `microbench` MAY be an empty array only if `e2e.status` is `"failed"` for
  the same configuration; otherwise it MUST contain at least one entry for
  `regime: "prefill"` and one for `regime: "decode"`.
- Every tokens/sec or timing number is always paired with the fields needed
  to judge its reliability (`num_runs`/`variance` for e2e, `iterations`/
  `stddev_us` for microbench) — a consumer MUST NOT treat a lone mean as
  sufficient to declare a future number "faster" or "slower".

## Rendered summary: `results/baseline-report.md`

A single markdown document, organized by model then scheme, presenting one
table per model with columns: Scheme | Prefill tok/s | Decode tok/s | # microbench
shapes covered | Status. It links to each configuration's raw JSON file
under `results/raw/` for full detail and must state, near the top, the device,
git commit, and dispatch-path convention (`tiled_baseline`) shared by every
row — matching FR-006's requirement that context travels with the numbers.
