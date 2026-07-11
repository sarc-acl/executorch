# Contract: Speedup Target / Outcome Schema

## `results/speedup-target.json`

```json
{
  "configs": [
    {
      "model": "llama-3.2-1b",
      "scheme": "4w",
      "baseline_prefill_tokens_per_sec": 1132.91,
      "baseline_prefill_stdev": 17.133,
      "target_multiplier": 2.0,
      "target_prefill_tokens_per_sec": 2265.82,
      "baseline_source": "specs/001-minipc-baseline-benchmarks/results/raw/llama-3.2-1b_4w.json"
    }
  ]
}
```

Rules a consumer can depend on:

- Exactly 6 entries, one per (model, scheme) — never collapsed across schemes (Research Decision 2).
- `target_prefill_tokens_per_sec` is always `baseline_prefill_tokens_per_sec * 2.0` — recomputable, never hand-edited.
- `baseline_source` always resolves to a real, already-existing `001` file — this script never invents a baseline number.

## Re-Measurement input (produced by future work, not this feature)

Must match `001`'s `e2e` object shape exactly (Decision 1) plus two extra
fields:

```json
{
  "model": "llama-3.2-1b",
  "scheme": "4w",
  "e2e": { "...": "identical shape to 001's e2e object" },
  "methodology_comparable": true,
  "methodology_note": null
}
```

## Outcome / report output

- `verdict` is always one of `met`, `exceeded`, `missed`, `regressed`,
  `not_comparable` — never a free-text string.
- `not_comparable` is set if and only if `methodology_comparable: false` in
  the input; when set, `observed_multiplier` is `null` and the report shows
  `methodology_note` instead of a number.
- `combined_e2e_change_pct` is always present (when comparable) and always
  rendered, but **never** feeds into `verdict` — a consumer must not infer
  pass/fail from it.
- Every report (self-test or real) carries `is_synthetic` per entry, and the
  self-test report's filename/heading states "SYNTHETIC SELF-TEST DATA —
  NOT A REAL MEASUREMENT" so it can never be mistaken for `outcome-report.md`.
