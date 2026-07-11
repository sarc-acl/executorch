# Tier-2 e2e validation: t64x64k16g21s32 vs shipped baseline

**Result: DO NOT SHIP.** The Tier-1 microbenchmark winner (`tsweep_t64x64k16g21s32`,
+27.1% over `specs/025`'s winner in isolated GEMM throughput) is **~7.6% SLOWER**
end-to-end than the currently-shipped default 8da4w dispatch when measured in a real
model (Llama 3.2 1B, `8da4w` buffer PTE, 2048-token prefill).

## Setup

- Device: M5 EVT1 (`xgpusw-debug08`, serial `00000bf70c579c33`)
- Driver: `f14c51b6f8` (`c9861e9906d03fa2c7d48b804e1a1c80`), clocks pinned 509/2730/663 MHz
- Model: `llama3_2_1b_8da4w_buffer_ctx3072.pte`
- Runner: `llama_main` built from the `dbuf-int8-sweep` execution worktree (this feature's
  own shader/dispatch changes), 2048-token prefill (`p2048_exact.txt`, `num_bos=1`,
  `ET_VK_EXECUTE_NODE_THRESHOLD=16`)
- Coherence check passed first (`--prompt="The capital of France is"` → coherent output,
  confirming the buffer PTE + new dispatch code produces a working model, not garbage)

## Results (prefill tok/s, 3 runs each)

| Config | Run 1 | Run 2 | Run 3 | Median |
|---|---|---|---|---|
| Baseline (default dispatch, no env var) | 440.1 | 451.3 | 424.1 | **440.1** |
| `ET_VK_DQ8CA_COOPMAT_VARIANT=tsweep_t64x64k16g21s32` | 406.8 | 396.4 | 421.5 | **406.8** |

**Delta: -7.6%** (406.8 / 440.1 - 1). The two distributions do not overlap (baseline
424.1-451.3 vs new config 396.4-421.5) — this is a real effect, not noise.

## Why the Tier-1 win didn't transfer

The isolated-GEMM microbenchmark only measures the `linear_dq8ca_q4gsw` op's own kernel
time at a few large synthetic shapes (K=4096/14336, N=1024-14336). The real model's
prefill path includes many more ops (attention, other linears, dequant/quant glue,
scheduling overhead between dispatches) and different, smaller per-layer shapes than the
microbenchmark's synthetic ones — a config that wins on the isolated large-GEMM shape can
easily lose overall if it has worse behavior at the shapes the real model actually uses,
or interacts worse with surrounding ops/scheduling. This is precisely why this workstream's
constitution states "e2e is the deliverable, microbench is for analysis" — Tier-1 results
are a necessary first filter, not sufficient evidence to ship.

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
