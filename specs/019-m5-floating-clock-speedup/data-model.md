# Data Model: M5 EVT1 Floating-Clock Speedup Table

## Floating Clock Measurement

One per (model, scheme, config_type) -- 12 total (3 models x 2 schemes x
2 config types).

| Field | Type | Notes |
|---|---|---|
| `model` | enum | `llama3_2_1b` / `llama3_2_3b` / `llama3_1_8b` |
| `scheme` | enum | `4w` / `8da4w` |
| `config_type` | enum | `t_tiled_baseline` / `full_stack_optimal` |
| `pte_path` | string | Reused verbatim from `specs/015`/`018`'s `.pte_out` entries -- no new export (research.md Decision 1) |
| `dispatch_status` | enum | Reused verbatim from the pinned measurement's own `dispatch_status` (`tiled_confirmed` or coopmat-`confirmed`) -- not re-derived under floating clocks |
| `sysfs_verified_floating` | bool | `true` only after a readback confirms `min_freq`/`max_freq` (GPU/MIF/INT) reflect the hardware's full range, not the pinned 509/2730/663 triple (research.md Decision 3) |
| `prefill_tok_s_reps` | float[3] | All 3 per-rep values, never collapsed to a mean-only field (research.md Decision 4) |
| `decode_tok_s_reps` | float[3] | Same, for decode |
| `throttle_observed` | bool | `true` if rep-to-rep spread exceeds a few percent (expected for `t_tiled_baseline` configs per Principle VII's -19%/-27% precedent; expected `false` for `full_stack_optimal`/coopmat configs, <4% precedent) |
| `speedup_vs_baseline_coldstart` | float\|null | `full_stack_optimal.prefill_tok_s_reps[0] / t_tiled_baseline.prefill_tok_s_reps[0]` for the matching (model, scheme) pair -- the primary reported ratio (research.md Decision 5) |

## Relationship to existing entities

- **Pinned T-tiled Baseline Measurement** (`specs/018-m5-8da4w-t-tiled-baseline/data-model.md`)
  and **pinned full-stack optimal numbers** (`specs/015-m5-e2e-wmma-validation/data-model.md`):
  this feature's `pte_path` and `dispatch_status` fields are copied
  directly from those entities' already-measured values -- this feature
  adds a new clock-state dimension to the same 12 (model, scheme,
  config_type) points, it does not define a 13th independent axis.

## Lifecycle

```
Floating Clock Measurement created (sysfs_verified_floating=false)
  --(write hardware min/max to min_freq/max_freq, research.md Decision 2)-->
  --(sysfs readback confirms full range, Decision 3)-->
  sysfs_verified_floating = true
  --(3 timed reps captured)-->
  prefill_tok_s_reps / decode_tok_s_reps populated
  --(compare rep spread against Principle VII's throttle precedent)-->
  throttle_observed set
  --(divide matching full_stack_optimal vs t_tiled_baseline rep-1 values)-->
  speedup_vs_baseline_coldstart populated
  --(written into this feature's own results/*.md and floating-vs-pinned-report.md)-->
  done
```

If `sysfs_verified_floating` cannot be confirmed `true` for a given
capture, that capture is discarded and re-attempted -- per spec.md's
Edge Cases, a number from an environment that isn't genuinely floating
must not be published as a floating result.
