# specs/035: 780M/RADV tile sweep — retuning the WMMA shaders off their Android defaults

**Date**: 2026-07-23 · **Branch**: `yanwen/dev-igpu` · **Goal metric**: e2e prefill tok/s
(2048-token protocol, 1B buffer ptes for ranking; 3B/8B for final validation).
Method: 3-round coordinate-descent sweep via the specs/023/028 `tsweep` env-toggle
machinery (ported to this branch), microbench correctness gate before every e2e rank.

## Shipped defaults (this branch)

| shader | old default (M5-tuned) | **new default (780M)** | 1B e2e |
|---|---|---|---|
| `linear_q4gsw_coopmat` (4w) | 128×64 **k16** g2×2 s32 | 128×64 **k32** g2×2 s32 | 1764 → **1977** (+12%) |
| `linear_dq8ca_q4gsw_coopmat` (8da4w) | 64×**32** k32 g1×2 **s64** | 64×**128** k32 g**4×1** s**32** | 1274 → **1800** (+41%) |

Files changed: both shader yamls' default geometry + `kQ4gswCoopmatDims`/
`kDq8caQ4gswCoopmatDims` in `QuantizedLinear.cpp` (wg_size 128 in all cases).
Default-dispatch correctness: 48/48 microbench cases pass (note the dq8ca default
is the dbuf4 loop structure at the new geometry — verified independently, and its
e2e matches the dbuf2-structured tsweep variant within noise).

## Full-model validation (new defaults, buffer ptes, optimized runner)

| model | 4w old→new | 8da4w old→new | previous overall champion |
|---|---|---|---|
| 1B | 1764 → **1977** | 1274 → **1800** | 1899 (8da4w texture) — **4w buffer now #1** |
| 3B | 710 → **~815** | 495 → **~716** | 778 (8da4w texture) — **4w buffer now #1** |
| 8B | 303 → **~363** | 214 → **~326** | 344 (8da4w texture) — **4w buffer now #1** |

vs stock release/1.3 on the same ptes: 4w buffer = **2.48× / 2.94× / 3.10×** (1B/3B/8B).
The 8da4w whole-graph-buffer config went from −33% vs its texture path to −5%.

## What the sweep learned about this GPU

1. **K-step 32 > 16 for 4w** (+13% alone) — the M5 sweep never varied K. K64 regresses
   (−14%); K128 fails the gate on dq8ca.
2. **wave32 wins on RDNA3 for the int8 shader** — every top-4 dq8ca variant is s32.
   The previous sweeps pinned s64 (specs/025 Decision 1) after specs/026 saw sg32
   miscomputes on Xclipse; on RADV, sg32 gates cleanly at the shipped shapes.
3. **Wider N-tile (128) + flat 4×1 grid** suits the dq8ca shader here — more B-reuse
   per A-fetch, all four subgroups sharing one M-row band.
4. **Gate discipline mattered**: the raw e2e leader `t64x64k32g21s32` (1855) FAILS
   correctness (bench crash, fails=1) and was disqualified — quoting it would have
   shipped a silent-miscompute config. Also gate-failed: 4w `t256x64k32g24s32`,
   dq8ca `t128x32k*`, `t64x32k64/128g12s64`, `t64x32k32g12s32`.
5. Threads-per-WG sweet spot is 128; 64-thread and 512-thread variants all regress
   (−20% to −60%).

## Reproduction

- Sweep driver: session scratchpad `igpu_sweep/gate_and_rank.sh` (env-toggle per token;
  regenerate trivially — token grammar `tsweep_t<M>x<N>k<K>g<XY>s<S>`).
- Raw rankings: round-1 16+36 variants, round-2 6+7, round-3 3 (see git history of the
  tsweep yamls for the full grids).
- Caveats: single-session numbers; decode untouched (coopmat is prefill-only);
  M5/Android keeps its own defaults — this branch's values are 780M-specific and the
  two devices genuinely want different geometries (that's the point of the branch).
