# Data Model: M5 EVT1 8da4w T-tiled Baseline

## T-tiled Baseline Measurement

One per model (3 total). Mirrors the row shape already used throughout
`specs/015`'s results files, so this feature's output slots directly
into the existing tables without a schema mismatch.

| Field | Type | Notes |
|---|---|---|
| `model` | enum | `llama3_2_1b` / `llama3_2_3b` / `llama3_1_8b` |
| `scheme` | string | `8da4w`, fixed for this feature |
| `pte_path` | string | `/local/yanwen.xu/workspace/.pte_out/<model>_8da4w_texture_ctx3072.pte` |
| `pte_status` | enum | `not_yet_exported` / `exported` / `export_failed` |
| `dispatch_status` | enum | `not_yet_run` / `tiled_confirmed` (the only valid outcome for this feature) / `unexpected_coopmat` (a defect, not a reportable baseline -- see Edge Cases in spec.md) |
| `prefill_tok_s_mean` | float\|null | 3-run mean, populated once measured |
| `prefill_tok_s_cov` | float\|null | CoV across the 3 runs |
| `decode_tok_s_mean` | float\|null | 3-run mean |
| `decode_tok_s_cov` | float\|null | CoV across the 3 runs |
| `speedup_vs_optimized` | float\|null | `<optimized full-stack tok/s> / prefill_tok_s_mean` -- computed once both this baseline and the existing optimized number are available; this is the number `specs/015`'s report tables actually need |

**Final measured values (2026-07-06)**:

| model | scheme | pte_status | dispatch_status | prefill_tok_s_mean | prefill_tok_s_cov | decode_tok_s_mean | speedup_vs_optimized (full-stack) | speedup_vs_optimized (linear-only) |
|---|---|---|---|---|---|---|---|---|
| llama3_2_1b | 8da4w | exported | tiled_confirmed | 222.30 | 0.28% | 13.84 | 3.25x (723.00/222.30) | 2.40x (533.44/222.30) |
| llama3_2_3b | 8da4w | exported | tiled_confirmed | 79.83 | 0.21% | 6.84 | 3.59x (286.31/79.83) | 2.52x (200.91/79.83) |
| llama3_1_8b | 8da4w | exported | tiled_confirmed | 35.17 | 0.13% | 3.85 | 3.70x (130.05/35.17) | 2.84x (99.98/35.17) |

Dispatch confirmed via `analyze_etdump_shaders.py`: 100% `linear_dq8ca_q4gsw_tiled_texture3d_texture2d_half` for all three models (112/112, 196/196, 224/224 respectively), zero `_coopmat_` entries -- no `unexpected_coopmat` edge case triggered.

## Relationship to existing entities

- **Existing optimized full-stack numbers** (already measured, per
  `specs/015-m5-e2e-wmma-validation/data-model.md`'s seeded rows table):
  1B 723.00, 3B 286.31, 8B 130.05 tok/s. This feature's
  `speedup_vs_optimized` field divides those by this feature's own
  `prefill_tok_s_mean` -- it does not re-measure the optimized side.
- **Existing `4w` T-tiled baseline** (already established,
  `RESULTS-SUMMARY.md`'s trusted anchor: 1B 312.7, 3B 112.5, 8B 51.4
  tok/s) -- this feature's rows are the direct `8da4w` sibling of those,
  same workload, same methodology (`research.md` Decision 5), living in
  the same downstream tables once complete.

## Lifecycle

```
T-tiled Baseline Measurement created (pte_status=not_yet_exported)
  --(export with default/no storage_override, research.md Decision 1)-->
  pte_status = exported
  --(3-run timed capture + ETDump dispatch check, Decision 4/5)-->
  dispatch_status = tiled_confirmed, prefill/decode tok/s populated
  --(divide existing optimized number by this baseline)-->
  speedup_vs_optimized populated
  --(written into specs/015's results/*.md and m5-e2e-validation-report.md)-->
  done
```

If `dispatch_status` ever resolves to `unexpected_coopmat` instead of
`tiled_confirmed`, that row does NOT get a `speedup_vs_optimized` value
and does NOT get written into the downstream report as a valid baseline
-- per spec.md's Edge Cases, this becomes a new tracked issue instead.
