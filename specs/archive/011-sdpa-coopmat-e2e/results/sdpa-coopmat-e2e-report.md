# SDPA Coopmat E2E Validation Report

## Correctness + dispatch verification summary

- ETDump-confirmed for all six configurations (tasks.md T005-T009): both `sdpa_compute_attn_weights_coopmat` and `sdpa_compute_out_coopmat` dispatched with zero tiled fallback, matching each model's layer count (16/28/32 for `llama-3.2-1b`/`llama-3.2-3b`/`llama-3.1-8b`).
- No new export or rebuild was needed -- `009`'s existing `Buffer`-storage `.pte` exports already support `ET_VK_SDPA_COOPMAT` correctly (research.md Decision 1).
- Every baseline value below is cited verbatim from `009`'s already-published report (its "WMMA" column: linear coopmat enabled, SDPA still tiled), not re-measured.

## Overall: enabling SDPA coopmat improves real e2e prefill tok/s by **+27.3% on average** across 6/6 measured configurations, on top of `009`'s already-published linear-coopmat gains. This agrees in direction with `010`'s microbenchmark-level finding (66.8% average, isolated-shader tier-1 -- the smaller e2e magnitude is expected, since the whole-model number is diluted by every other op, not just SDPA's).

## Per-configuration comparison

| Model | Scheme | Phase | Baseline (009, tok/s) | SDPA coopmat (tok/s) | Diff | Consistency |
|---|---|---|---:|---:|---:|---|
| llama-3.1-8b | 4w | prefill ¹ | 316.53 ± 1.09 | 422.92 ± 10.52 | +33.6% | consistent |
| llama-3.1-8b | 4w | decode | 9.46 ± 0.01 | 9.45 ± 0.01 | -0.2% | n/a (decode unaffected by SDPA coopmat) |
| llama-3.1-8b | 8da4w | prefill ¹ | 205.75 ± 0.42 | 240.00 ± 0.52 | +16.6% | consistent |
| llama-3.1-8b | 8da4w | decode | 9.34 ± 0.01 | 9.33 ± 0.00 | -0.1% | n/a (decode unaffected by SDPA coopmat) |
| llama-3.2-3b | 4w | prefill ¹ | 649.88 ± 2.99 | 950.18 ± 14.79 | +46.2% | consistent |
| llama-3.2-3b | 4w | decode | 18.75 ± 0.01 | 18.77 ± 0.04 | +0.1% | n/a (decode unaffected by SDPA coopmat) |
| llama-3.2-3b | 8da4w | prefill ¹ | 432.68 ± 1.66 | 544.61 ± 2.79 | +25.9% | consistent |
| llama-3.2-3b | 8da4w | decode | 18.24 ± 0.01 | 18.16 ± 0.03 | -0.4% | n/a (decode unaffected by SDPA coopmat) |
| llama-3.2-1b | 4w | prefill ¹ | 1867.40 ± 33.93 | 2356.57 ± 33.15 | +26.2% | consistent |
| llama-3.2-1b | 4w | decode | 59.95 ± 0.04 | 59.97 ± 0.09 | +0.0% | n/a (decode unaffected by SDPA coopmat) |
| llama-3.2-1b | 8da4w | prefill ¹ | 1265.03 ± 9.14 | 1461.90 ± 12.85 | +15.6% | consistent |
| llama-3.2-1b | 8da4w | decode | 58.14 ± 0.06 | 58.10 ± 0.07 | -0.1% | n/a (decode unaffected by SDPA coopmat) |

¹ Prefill comparisons inherit `006`'s documented cross-session variance caveat (research.md Decision 6): captured in a different session than `009`'s baseline, on the same otherwise-idle `rocky-ryzen` MiniPC.

## Excluded / not collected

none

## Notes

- Decode tok/s is reported alongside prefill as a sanity check (FR-008) and is materially unchanged in every measured configuration, as expected -- decode has no WMMA-capable GEMV kernel for attention, so it dispatches the same path regardless of the `ET_VK_SDPA_COOPMAT` toggle.
- Scope is tier-2 (model-level e2e), `rocky-ryzen` MiniPC, 2048-token prefill / 1024-token decode -- matching every prior e2e feature in this workstream.