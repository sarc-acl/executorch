# Data Model: M5 EVT1 End-to-End WMMA Validation

## M5 EVT1 Configuration

One (model, op-family) unit -- 9 total: 3 models × {`4w`, `8da4w`} linear
(6) + 3 models × SDPA-coopmat (3).

| Field | Type | Notes |
|---|---|---|
| `model` | enum | `llama3_2_1b` / `llama3_2_3b` / `llama3_1_8b` |
| `op_family` | enum | `linear_4w` / `linear_8da4w` / `sdpa_coopmat` |
| `pte_path` | string | `.pte_out/<model>_<scheme>_buffer_ctx3072.pte`; for `sdpa_coopmat`, the same buffer PTE as `linear_4w` for that model, run with `ET_VK_SDPA_COOPMAT=1` |
| `pte_status` | enum | `reused_existing` (the 6 `4w`/pre-existing) / `newly_exported` (the 3 `8da4w`) |
| `dispatch_status` | enum | `not_yet_run` / `confirmed` (coopmat/WMMA kernel family seen in ETDump) / `fallback` (tiled kernel seen instead) / `failed` (export/run error) |
| `e2e_result` | record | `{prefill_tok_s, decode_tok_s, iteration_count}` or `not_yet_run` |
| `blocked_reason` | string\|null | e.g. "GPU watchdog at 2048-token prefill" -- populated only if `dispatch_status` or `e2e_result` couldn't complete |
| `measured_order` | int | 1-9, per Decision 3's 1B→3B→8B sequencing (within a model, linear before SDPA) |

Seeded rows (as of this feature's start):

| model | op_family | pte_status | dispatch_status | e2e_result |
|---|---|---|---|---|
| llama3_2_1b | linear_4w | reused_existing | not_yet_run | not_yet_run |
| llama3_2_1b | linear_8da4w | (needs export) | not_yet_run | not_yet_run |
| llama3_2_1b | sdpa_coopmat | reused_existing (4w PTE) | not_yet_run | not_yet_run |
| llama3_2_3b | linear_4w | reused_existing | not_yet_run | not_yet_run |
| llama3_2_3b | linear_8da4w | (needs export) | not_yet_run | not_yet_run |
| llama3_2_3b | sdpa_coopmat | reused_existing (4w PTE) | not_yet_run | not_yet_run |
| llama3_1_8b | linear_4w | reused_existing | not_yet_run | not_yet_run |
| llama3_1_8b | linear_8da4w | (needs export) | not_yet_run | not_yet_run |
| llama3_1_8b | sdpa_coopmat | reused_existing (4w PTE) | not_yet_run | not_yet_run |

## Prior-Finding Reference

A specific already-published result one Configuration's measurement can be
compared against.

| Field | Type | Notes |
|---|---|---|
| `config_key` | string | FK to M5 EVT1 Configuration (`model`+`op_family`) |
| `source_doc` | string | e.g. `.shared-context/report-for-human/4w-prefill2048-decode1024-3models.md` |
| `prior_value_tok_s` | float\|null | `null` if no prior baseline exists at all |
| `comparison_type` | enum | `directional` (always, per spec Clarifications -- this repo's shader has diverged from every prior-finding source) / `no_prior_baseline` |

Seeded reference table:

| config_key | source_doc | prior_value_tok_s (prefill) | comparison_type |
|---|---|---|---|
| llama3_1_8b/linear_4w | `report-for-human/jira-tile-sweep.md` (128x64) | 110.6 | directional |
| llama3_2_3b/linear_4w | `report-for-human/jira-tile-sweep.md` (128x64) | 213.9 | directional |
| llama3_2_1b/linear_4w | `report-for-human/jira-tile-sweep.md` (128x64) | 565.3 | directional |
| llama3_1_8b/linear_8da4w | `report-for-human/e2e-spec.md` / `RESULTS-SUMMARY.md` | 85.1 | directional |
| llama3_2_3b/linear_8da4w | -- | null | no_prior_baseline |
| llama3_2_1b/linear_8da4w | -- | null | no_prior_baseline |
| llama3_2_1b/sdpa_coopmat | `report-for-human/session-2026-06-23-sdpa-wmma-findings.md` | 763 (combined WMMA stack, not SDPA-isolated) | directional |
| llama3_2_3b/sdpa_coopmat | same doc, 512-prefill only, no exact tok/s given | null (no 2048-prefill number) | no_prior_baseline |
| llama3_1_8b/sdpa_coopmat | same doc, 512-prefill only, no exact tok/s given | null (no 2048-prefill number) | no_prior_baseline |

## M5 EVT1 E2E Validation Report

The consolidated document (User Story 4): all 9 Configurations, each with
its `e2e_result` (or `blocked_reason`) and its Prior-Finding Reference
comparison (or explicit `no_prior_baseline` flag) -- assembled last, after
`1b-results.md`/`3b-results.md`/`8b-results.md` are each already published
per Decision 3's incremental-reporting requirement.

## Lifecycle

```
not_yet_run --(export if needed)--> pte ready
  --(dispatch-confirmation run)--> dispatch_status = confirmed | fallback | failed
  confirmed --(e2e capture run)--> e2e_result populated
  fallback | failed --> blocked_reason populated, e2e_result stays not_yet_run
watchdog recurrence at 2048-prefill --> blocked_reason populated (Edge Cases),
  no e2e_result reported for that configuration
```

No other state transitions -- one-shot measure-and-report per
Configuration, sequenced per Decision 3.
