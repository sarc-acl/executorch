# M5 EVT1 Floating-Clock Results — LLaMA 3.2 1B

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
| `4w` T-tiled baseline (prefill tok/s) | **502.823** | 506.304 | 502.207 | false (<1% spread) |
| `4w` full-stack optimal (prefill tok/s) | **979.904** | 935.587 | 943.779 | true (~4.6% spread) |
| `8da4w` T-tiled baseline (prefill tok/s) | **389.132** | 372.161 | 372.635 | true (~4.3% spread) |
| `8da4w` full-stack optimal (prefill tok/s) | **806.617** | 788.299 | 944.649 | true (~19.8% spread, non-monotonic -- rep 3 rose, not fell) |

Decode tok/s per rep (same order): `4w` baseline 14.7877/15.2065/14.6528;
`4w` optimal 14.6128/14.7425/15.111; `8da4w` baseline
15.9851/15.9799/16.1545; `8da4w` optimal 15.441/15.1083/15.4637.

## Cold-start speedup ratios (research.md Decision 5)

| Scheme | Baseline rep 1 | Optimal rep 1 | Speedup (floating) | Speedup (pinned, for reference) |
|---|---|---|---|---|
| `4w` | 502.823 | 979.904 | **1.95x** | 2.60x (312.7 -> 812.6) |
| `8da4w` | 389.132 | 806.617 | **2.07x** | 3.25x (222.30 -> 723.0) |

## Note on the `8da4w` optimal config's unusual variance

Unlike every other 1B config (all <5% spread), `8da4w` full-stack
optimal showed a ~19.8% spread, and non-monotonically (rep 3 was the
*highest* value, not the lowest) -- the opposite of the tiled-throttle
pattern Principle VII documents. Not yet attributed to a specific
cause; flagged here rather than smoothed over, consistent with this
feature's own methodology (research.md Decision 4). This is the same
config whose *pinned* measurement (`specs/015`, `results/1b-results.md`
UPDATE 2) also showed an anomalously high CoV (6.87%) relative to every
other pinned config -- the two anomalies may be related, but that
connection is not established here, only noted.

## Why the floating speedup ratio is *lower* than the pinned one here

Both schemes show a *smaller* floating speedup (1.95x/2.07x) than their
pinned counterparts (2.60x/3.25x) -- the opposite of what Principle
VII's tiled-throttles-more-than-coopmat precedent would predict for the
ratio's direction. This is consistent with the T-tiled baseline getting
a *larger* proportional DVFS boost than the full-stack optimal config
did (e.g. `4w`: baseline rose 502.8/312.7=1.61x while optimal rose only
979.9/812.6=1.21x) -- floating clocks lift the (lower-power, less
memory-bandwidth-bound) tiled path further than the already
compute-dense coopmat path. This is a real, measured finding, not
un-verified speculation -- but is based on only 3 reps per config;
treat the exact ratios as directional pending the full 3B/8B data.
