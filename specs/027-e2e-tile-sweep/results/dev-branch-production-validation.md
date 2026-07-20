# Winner applied to `dev` (production shader), full-stack e2e validation

**Result: confirmed +12.5% on the full `dev` stack** (SDPA coopmat default-on,
`specs/026-sdpa-8da4w-defaults-e2e`), a larger improvement than the +9.32% measured
on the SDPA-less `dbuf-int8-sweep` branch used for the original sweep.

## What was changed

Applied `specs/027`'s e2e-ranked winner (`tsweep_t64x32k32g12s64`) directly to the
production shader on `dev`:

- `backends/vulkan/runtime/graph/ops/glsl/linear_dq8ca_qw_coopmat.glsl`: loop structure
  updated from "dbuf4" (store-first, prefetch-into-shared-then-barrier prologue) to
  "dbuf2" (store-first, prefetch-only-into-registers prologue) — `dev` had never
  received the dbuf2 loop-structure port from `specs/023`, only the dbuf4 original, so
  applying the tile-sweep winner required porting the loop structure too, not just the
  tile numbers.
- `backends/vulkan/runtime/graph/ops/glsl/linear_dq8ca_qw_coopmat.yaml`: tile geometry
  `WG_TILE_M/N/SG_GRID_X/Y` changed from `128/64/2/2` to `64/32/1/2` (`WG_TILE_K=32`,
  `SUBGROUP_SIZE=64` unchanged).
- `backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp`: `kDq8caQ4gswCoopmatDims`
  updated from `{128, 64, 32, 256}` to `{64, 32, 32, 128}` (must match the yaml's
  resolved `WG_SIZE`, used for eligibility-gate alignment checks and global workgroup
  size computation).

## Results (Llama 3.1 8B, `8da4w` buffer PTE, 2048-token prefill, M5 EVT1 `xgpusw-debug08`,
driver `f14c51b6f8`/`c9861e9906d03fa2c7d48b804e1a1c80`, clocks pinned 509/2730/663 MHz)

| Config | Run 1 | Run 2 | Run 3 | Mean | 
|---|---|---|---|---|
| `dev` baseline (unmodified, before this change) | 131.806 | 130.263 | 131.653 | 131.24 |
| `dev` + winner applied | 147.966 | 147.190 | 147.785 | **147.65** |

**+12.5%**, non-overlapping 3-run ranges (baseline 130.26–131.81, winner 147.19–147.97).
Coherence-checked first for both binaries (short-prompt sanity output, non-garbage).

## Correcting the user's reference number

The user's "153.3 tok/s fully-optimized" reference turns out to be the **4w** (not
8da4w) 8B prefill number — confirmed against
`specs/015-m5-e2e-wmma-validation/results/2026-07-11-dev-branch-smoke-test.md`, which
recorded both on the same `dev` commit (`573d44dac`): 8B `4w` = 153.2 tok/s, 8B `8da4w`
= 130.3 tok/s (matching this validation's own 131.24 baseline within run-to-run noise).
8da4w and 4w are different quantization schemes with different per-layer overhead
(8da4w carries dynamic per-token activation quantization `4w` doesn't) — a
lower absolute ceiling for 8da4w than 4w is expected and not itself evidence of a
problem; the relevant comparison for this sweep is 8da4w-vs-8da4w (131.24 → 147.65),
not 8da4w-vs-4w.

## Status

**Committed and shipped** — commit `42aabb4e0` on `yanwen/dev-1.3` (2026-07-12). This is
now the production default `8da4w` tile/loop configuration on `dev`.

**1B/3B follow-up validation: DONE 2026-07-13** — see
`results/1b-3b-production-validation-2026-07-13.md`. Confirmed real e2e wins on both
(+8.6% / +11.9%), same direction as this file's 8B result. No model regresses; nothing
further pending.
