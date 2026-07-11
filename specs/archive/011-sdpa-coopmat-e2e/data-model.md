# Data Model: SDPA Coopmat E2E Validation

## SDPA-Coopmat-Enabled E2E Run

One entry per (model, scheme) -- 6 total, reusing `009`'s existing exports
(no new export entity needed).

| Field | Type | Notes |
|---|---|---|
| `model` / `scheme` | string | e.g. `llama-3.2-1b`/`4w` |
| `pte_path` | string | `009`'s existing export path, reused verbatim (research.md Decision 1) |
| `qk_dispatch_status` / `av_dispatch_status` | enum | `confirmed`/`fallback`, from ETDump kernel-name inspection with `ET_VK_SDPA_COOPMAT=1` set (research.md Decision 2). No e2e number trusted unless both are `confirmed` |
| `prefill_tok_s_mean` / `prefill_tok_s_stdev` | float | 5-rep capture, `ET_VK_SDPA_COOPMAT=1` (research.md Decision 3) |
| `decode_tok_s_mean` / `decode_tok_s_stdev` | float | Same capture, decode phase |

## E2E SDPA Comparison Case

One entry per (model, scheme, phase) -- 12 rows (6 configurations x 2
phases).

| Field | Type | Notes |
|---|---|---|
| `model` / `scheme` / `phase` | string | e.g. `llama-3.2-1b`/`4w`/`prefill` |
| `baseline_tok_s` / `baseline_stdev` | float | Parsed from `009`'s report's "WMMA" column (research.md Decision 4) -- SDPA coopmat disabled, linear coopmat enabled |
| `sdpa_coopmat_tok_s` / `sdpa_coopmat_stdev` | float | This feature's new capture (research.md Decision 3) -- absent unless the parent Run's dispatch is `confirmed` |
| `diff_pct` | float | `(sdpa_coopmat_tok_s - baseline_tok_s) / baseline_tok_s * 100` |
| `cross_session_caveat` | bool | `true` for every `prefill` row (research.md Decision 6), `false` for `decode` |
| `consistency` | enum | `consistent` / `diverges` / `not_applicable` -- direction-only comparison against `010`'s microbenchmark finding (research.md Decision 5), prefill rows only |

## SDPA Coopmat E2E Report

The consolidated document (US3): one overall statement (does enabling SDPA
coopmat help real e2e tok/s -- research.md Decision 5, no per-scheme split
needed unless the data itself diverges), the 12-row comparison table (or
fewer, with excluded configurations listed with their reason), and a
Blocked/Failed section (always present, even if empty).

No lifecycle/state transitions -- one-shot capture-and-compare, matching
`009`'s own report shape.
