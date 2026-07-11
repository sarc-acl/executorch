# Tier-2 e2e validation: t64x64k16g21s32 vs shipped baseline

**Result: DO NOT SHIP.** The Tier-1 microbenchmark winner (`tsweep_t64x64k16g21s32`,
+27.1% over `specs/025`'s winner in isolated GEMM throughput) is end-to-end SLOWER than
the currently-shipped default 8da4w dispatch on both models tested. **The correct,
shape-matched comparison (8B, see Round 2 below) shows -2.7%** — real, but much smaller
than an initial, methodologically-flawed -7.6% figure from Round 1 (kept below for the
record, with the flaw explained).

## Round 1 (FLAWED — model/shape mismatch, kept for the record)

Initial validation used the Llama 3.2 **1B** model (hidden_size=2048,
intermediate=8192). This was a methodological error caught by user review: the
microbenchmark that found `t64x64k16g21s32` as the Tier-1 winner used **8B**-shaped GEMMs
(K=4096/14336, from Llama 3.1 8B's hidden_size=4096/intermediate=14336, matching this
workstream's standard `wq`+`w1_gate` representative-shape convention) — not 1B's shapes.
Validating a config tuned/measured on 8B-shaped GEMMs against a 1B model is an
apples-to-oranges comparison, not a fair Tier-2 check.

| Config (1B model) | Run 1 | Run 2 | Run 3 | Median |
|---|---|---|---|---|
| Baseline (default dispatch) | 440.1 | 451.3 | 424.1 | 440.1 |
| `tsweep_t64x64k16g21s32` | 406.8 | 396.4 | 421.5 | 406.8 |

Delta: -7.6%. **Not trusted as the primary result** — see Round 2.

## Round 2 (CORRECTED — shape-matched, 8B model)

Setup: Llama 3.1 **8B** `8da4w` buffer PTE (matches the microbenchmark's own shape
convention), M5 EVT1 (`xgpusw-debug08`, `00000bf70c579c33`), driver `f14c51b6f8`
(`c9861e9906d03fa2c7d48b804e1a1c80`), clocks pinned 509/2730/663 MHz, 2048-token prefill
(`p2048_exact.txt`, `num_bos=1`, `ET_VK_EXECUTE_NODE_THRESHOLD=16`). Coherence-checked
first (short prompt → grammatical, if repetitive, greedy-decode output — expected at
temperature=0 on a short prompt, not a correctness failure).

| Config (8B model) | Run 1 | Run 2 | Run 3 | Median |
|---|---|---|---|---|
| Baseline (default dispatch) | 100.728 | 100.922 | 100.284 | **100.73** |
| `tsweep_t64x64k16g21s32` | 98.287 | 98.014 | 97.986 | **98.01** |

**Delta: -2.7%** (98.01 / 100.73 - 1). Distributions still don't overlap (baseline
100.28-100.92 vs new config 97.99-98.29) — smaller than Round 1's flawed figure, but still
a real, consistent regression, not noise.

## Why the Tier-1 win didn't transfer (even shape-matched)

The isolated-GEMM microbenchmark measures only the `linear_dq8ca_q4gsw` op's own kernel
time at a handful of large per-layer shapes. The real model's prefill path includes many
more ops (attention/SDPA — which has no coopmat path at all, other linears, dequant/quant
glue, inter-dispatch scheduling overhead) that the isolated benchmark doesn't capture —
per `specs/003`'s classification data, SDPA alone is ~27% of 1B's prefill phase time and
uses no coopmat. A tile/subgroup choice that's faster in isolation can still lose overall
if it interacts worse with the surrounding dispatch/scheduling pattern, occupies more
register/shared-memory pressure that starves neighboring dispatches, or the graph-level
`ET_VK_EXECUTE_NODE_THRESHOLD` command-buffer-submission behavior responds differently to
its different workgroup-size/dispatch-count profile. This is precisely why this
workstream's constitution states "e2e is the deliverable, microbench is for analysis" —
Tier-1 results are a necessary first filter, not sufficient evidence to ship, and this
finding holds even after fixing the shape-mismatch methodology error.

## Disposition

- **`specs/026`'s `sweep-report.md` `recommendation` is downgraded from
  `productionize_candidate` to `keep_shipped_baseline`.**
- The shader-comment-update diff (`results/shader-comment-update.diff`) is still valid as
  a correctness/legality record (subgroup=32 does not crash the compiler, and is
  shape-dependently correct) — that finding stands independent of this e2e result.
- The `tsweep_t64x64k16g21s32` dispatch token remains available (opt-in via
  `ET_VK_DQ8CA_COOPMAT_VARIANT`, not on by default) for any future investigation, but is
  NOT recommended as the new default 8da4w configuration.
- This result is itself useful input to `specs/024-8da4w-slower-than-4w`'s broader
  investigation: it's a second, independent data point that isolated-kernel throughput
  improvements for `8da4w` do not straightforwardly translate to e2e gains on this
  hardware.
