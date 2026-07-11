# Contract: WMMA-Candidates Report Schema

## Per-config classification: `results/classifications/<model>_<scheme>.json`

One file per configuration (six total, matching `001`/`002` naming).

```json
{
  "config": {"model": "llama-3.2-1b", "scheme": "4w"},
  "phases": {
    "prefill": {
      "classifications": [
        {
          "kernel_name": "q4gsw_linear_gemm__tin__w_4x8_nc_texture3d_half",
          "shape": {"m": 2048, "k": 2048, "n": 8192},
          "category": "feed-forward",
          "classification": "b",
          "blocking_reasons": [
            "output tensor is rank-3 ([1,M,K]); can_use_q4gsw_coopmat() rejects dim_of(output) > 2 (QuantizedLinear.cpp)",
            "output tensor storage is TEXTURE_3D; can_use_q4gsw_coopmat() requires Buffer (QuantizedLinear.cpp)"
          ],
          "existing_or_prospective_shader": "linear_q4gsw_coopmat (linear_qw_coopmat.glsl)",
          "total_time_us": 0.0,
          "pct_of_phase": 0.0
        }
      ]
    },
    "decode": { "classifications": [ ] }
  }
}
```

Rules a consumer can depend on:

- `classification` is one of `"a"`, `"b"`, `"c"`, `"d"`, `"uncertain"` — never
  omitted, never a free-text value.
- `blocking_reasons` is present (non-empty) whenever `classification` is
  `"b"` or `"c"`; it MAY contain more than one entry (the prefill linear
  family has two independent reasons — do not assume exactly one).
- `existing_or_prospective_shader` is present whenever `classification` is
  `"a"`, `"b"`, or `"c"`; null for `"d"`.
- `classification: "a"` MUST NOT appear anywhere in these six files, since
  every capture was taken under `tiled_baseline` — if a consumer ever sees
  `"a"`, that is itself a signal something is wrong with either the
  classification logic or the underlying capture.
- `total_time_us`/`pct_of_phase` are carried verbatim from `002`'s
  `aggregated[]` entries — a consumer can cross-check against
  `002-etdump-shader-profiling/results/raw/<model>_<scheme>.json` directly.

## Consolidated report: `results/wmma-candidates-report.md`

Two sections, in this order:

1. **"Existing implementation blocked"** (classification `b`) — Optimization
   Candidate Groups ranked by `total_time_us_summed` descending, each with a
   per-config breakdown table and its `blocking_reasons` listed in full.
2. **"No WMMA implementation exists"** (classification `c`) — same
   structure, ranked the same way.

Each section states, at the top, that no classification-`a` entries exist in
this baseline data (per FR-009) so a reader does not assume WMMA is already
active anywhere. Links to `002`'s `profiling-report.md` and to each
config's classification JSON for full detail rather than duplicating them.
