# Data Model: End-to-End Texture3D vs. Buffer Storage Comparison

## Buffer-Storage Export

One entry per (model, scheme) — 6 total.

| Field | Type | Notes |
|---|---|---|
| `model` / `scheme` | string | One of the six configurations |
| `pte_path` | string | Path to the new `Buffer`-storage `.pte`, under this feature's own `results/pte/` — never overwrites `001`'s existing `Texture3D` `.pte` |
| `export_status` | enum | `ok` / `blocked` — `blocked` if the export itself fails (e.g. a buffer-size limit, per research.md Decision 4) |
| `export_failure_reason` | string or null | Required when `export_status` is `blocked` |
| `smoke_check_status` | enum or null | `pass` / `fail` / `not_run` (if export was blocked) — per research.md Decision 3, not a numerical-equivalence check |
| `smoke_check_note` | string or null | Required when `smoke_check_status` is `fail` (e.g. crash message, degenerate output description) |

## E2E Storage Comparison Case

One entry per (model, scheme) that reached a passing smoke-check.

| Field | Type | Notes |
|---|---|---|
| `model` / `scheme` | string | |
| `texture3d_prefill_tokens_per_sec` / `texture3d_decode_tokens_per_sec` | float | From `001`'s already-published `e2e` data — not re-measured |
| `buffer_prefill_tokens_per_sec` / `buffer_decode_tokens_per_sec` | float | Newly measured, same methodology (research.md Decision 5) |
| `prefill_relative_diff_pct` / `decode_relative_diff_pct` | float | `(buffer - texture3d) / texture3d * 100` |
| `microbenchmark_consistency` | enum | `consistent` / `diverges` — whether this configuration's e2e result agrees with `004`'s microbenchmark-level finding for the same configuration |

## E2E Storage Comparison Report

The consolidated document: an overall statement of whether `004`'s
microbenchmark-level finding generalizes to the real model, then a
per-configuration section — for measurable configurations, the
`Texture3D`/`Buffer` e2e numbers and consistency verdict; for
blocked/failed configurations, the `Buffer-Storage Export`'s failure reason,
listed explicitly (never silently absent, per FR-006/SC-004).

No lifecycle/state transitions beyond: Buffer-Storage Export produced →
smoke-checked → (if passing) measured as an E2E Storage Comparison Case →
rolled into the Report.
