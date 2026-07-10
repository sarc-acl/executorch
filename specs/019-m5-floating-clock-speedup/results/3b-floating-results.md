# M5 EVT1 Floating-Clock Results — LLaMA 3.2 3B

**Status as of 2026-07-06. Clocks: FLOATING (unpinned) — GPU 222000-980000,
MIF 676000-5333000, INT 160000-934000 Hz, sysfs-verified per
`quickstart.md` step 2, not clamped to the pinned 509000/2730000/663000
triple.** All numbers below are **floating** and must never be read as
or substituted for the pinned headline numbers in
`specs/015-m5-e2e-wmma-validation`/`specs/018-m5-8da4w-t-tiled-baseline`.
PTEs reused verbatim from those specs — no new export, no new dispatch
confirmation (clock state doesn't affect which shader dispatches).

Per research.md Decision 4, all 3 per-rep values are shown for every
config — never collapsed into a single mean-only entry.

| Config | Rep 1 (cold-start) | Rep 2 | Rep 3 | `throttle_observed` |
|---|---|---|---|---|
| `4w` T-tiled baseline (prefill tok/s) | **190.017** | 194.271 | 194.105 | false (~2.2% spread) |
| `4w` full-stack optimal (prefill tok/s) | **473.307** | 500.244 | 499.878 | true (~5.4% spread, rep1 was the low outlier) |
| `8da4w` T-tiled baseline (prefill tok/s) | **140.111** | 139.652 | 139.32 | false (~0.6% spread) |
| `8da4w` full-stack optimal (prefill tok/s) | **352.072** | 367.42 | 359.993 | true (~4.2% spread) |

Decode tok/s per rep (same order): `4w` baseline 6.14507/6.00994/6.05885;
`4w` optimal 5.90296/5.91737/5.88283; `8da4w` baseline
6.16681/6.24374/6.24413; `8da4w` optimal 6.05358/5.97439/6.049.

## Cold-start speedup ratios (research.md Decision 5)

| Scheme | Baseline rep 1 | Optimal rep 1 | Speedup (floating) | Speedup (pinned, for reference) |
|---|---|---|---|---|
| `4w` | 190.017 | 473.307 | **2.49x** | 2.97x (112.5 -> 334.0) |
| `8da4w` | 140.111 | 352.072 | **2.51x** | 3.59x (79.83 -> 286.3) |

## Note on `4w` optimal's rep1-low-outlier pattern

Unlike 3B's other three configs (all <2.5% spread), `4w` full-stack
optimal showed rep1 (473.307) noticeably below rep2/rep3 (500.244/499.878,
~5.4% spread) -- the *opposite* direction from a warm-up-continues-to-
throttle pattern (here the GPU appears to still be ramping up its DVFS
boost during rep1, not yet throttling down). Cold-start methodology
(research.md Decision 5) still uses rep1 for the ratio regardless, per
this feature's own stated policy of not smoothing over what's actually
measured -- but this makes 3B's `4w` cold-start ratio (2.49x) a
conservative lower bound relative to what reps 2-3 would give (500.244 /
190.017 = 2.63x).

## Consistency with 1B's DVFS-boost-asymmetry finding

Both 3B schemes again show a *smaller* floating speedup (2.49x/2.51x)
than their pinned counterparts (2.97x/3.59x) -- same direction as 1B's
result (1.95x/2.07x vs 2.60x/3.25x). This continues to support the
1B results file's explanation: floating clocks lift the T-tiled baseline
proportionally more than the already compute-dense coopmat/full-stack-
optimal path (e.g. `4w`: baseline rose 190.0/112.5=1.69x while optimal
rose only 473.3/334.0=1.42x). `8da4w`'s baseline showed almost no
rep-to-rep throttle here (~0.6%), tighter than 1B's `8da4w` baseline
(~4.3%) -- consistent with 3B/8da4w being less compute-dense per rep than
1B (more tokens processed per unit time is not the driver here; this is
about DVFS settling behavior, not workload size, and is noted rather than
explained further).
