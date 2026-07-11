# MiniPC No-WMMA Baseline Report

**Correction (2026-07-04, via `specs/004-linear-storage-comparison`)**: the
per-op `microbench` entries in each `raw/<model>_<scheme>.json` for `wq`↔`wo`
and `wk`↔`wv` were originally swapped in every one of the 6 configs, for both
the prefill and decode regimes. Root cause: `execute_test_cases()` (the
shared prototyping harness) groups test cases by a `ReferenceKey` that
excludes storage type, reordering `results` relative to
`generate_cases()`'s original sequence; the original result-printing loop
assumed positional correspondence that grouping does not preserve, so any
two ops sharing an identical `(K, N)` shape (here, `wq`/`wo` share one shape,
`wk`/`wv` share another) silently swap. Confirmed empirically — not
assumed — by `004`'s cross-check step, then fixed at the source
(`test_llama_baseline_bench.cpp` now looks up each result by name, not
position) and re-verified reproducible across two independent re-captures.
All 6 `raw/<model>_<scheme>.json` files and `microbench_raw.log` have been
regenerated with the corrected numbers (texture3d subset of `004`'s fixed
dual-storage capture, same device/methodology). **The e2e prefill/decode
tok/s numbers below were never affected** — they come from a completely
different capture mechanism (the `llama_main` runner, not this
microbenchmark harness) — only the per-op microbenchmark shape table
(referenced from each raw JSON, not inlined in this report's summary tables)
was wrong.

**Device**: `rocky-ryzen` — AMD Radeon 780M (RADV PHOENIX), RDNA3 mobile integrated GPU, subgroup size 64, cooperative-matrix capable (14 configs; see `test_coopmat_probe`).
**Git commit**: `5426101bf440c26a5b1e7edf34867e8d5f97e317`
**Dispatch path**: `tiled_baseline` for every row below — coopmat/WMMA excluded via the `ET_VK_FORCE_TILED_LINEAR=1` runtime toggle (prefill regime) and via the architecture itself (decode/GEMV never reaches the coopmat eligibility check regardless of the toggle — see `research.md` Decision 1's correction).
**e2e configuration**: fixed 2048-token prefill, 1024-token decode, `--temperature 0`, `--warmup true`, 5 repeated runs per configuration (mean/stdev reported; see each raw JSON's `run_metadata.all_runs` for every individual run and any discarded cold-start runs).
**Methodology note**: an earlier attempt at the two Llama 3.2 1B configurations, and at the microbenchmark sweep, ran concurrently with other CPU-heavy `.pte` exports and was discarded after being caught as confounded (measured ~32% slower than the clean redo, and the system was observed swapping). All numbers below are from clean runs with no concurrent load.

## Llama 3.1 8B

| Scheme | Prefill tok/s | Decode tok/s | # microbench shapes | Status |
|---|---:|---:|---:|---|
| 4w | 171.05 ± 2.16 | 9.282 ± 0.014 | 16 | ok |
| 8da4w | 214.30 ± 0.79 | 9.475 ± 0.016 | 16 | ok |

Raw: [`raw/llama-3.1-8b_4w.json`](raw/llama-3.1-8b_4w.json), [`raw/llama-3.1-8b_8da4w.json`](raw/llama-3.1-8b_8da4w.json)

## Llama 3.2 3B

| Scheme | Prefill tok/s | Decode tok/s | # microbench shapes | Status |
|---|---:|---:|---:|---|
| 4w | 388.40 ± 3.93 | 18.773 ± 0.003 | 16 | ok |
| 8da4w | 455.28 ± 5.42 | 18.475 ± 0.011 | 16 | ok |

Raw: [`raw/llama-3.2-3b_4w.json`](raw/llama-3.2-3b_4w.json), [`raw/llama-3.2-3b_8da4w.json`](raw/llama-3.2-3b_8da4w.json)

## Llama 3.2 1B

| Scheme | Prefill tok/s | Decode tok/s | # microbench shapes | Status |
|---|---:|---:|---:|---|
| 4w | 1132.91 ± 17.13 | 57.688 ± 0.053 | 16 | ok |
| 8da4w | 1357.46 ± 12.15 | 58.955 ± 0.128 | 16 | ok |

Raw: [`raw/llama-3.2-1b_4w.json`](raw/llama-3.2-1b_4w.json), [`raw/llama-3.2-1b_8da4w.json`](raw/llama-3.2-1b_8da4w.json)

## Observations worth flagging for the future WMMA-comparison feature

- **8da4w prefill consistently beats 4w prefill** (e.g., 8B: 214.3 vs 171.1 tok/s) even though both are running the *tiled* (non-coopmat) shader here — this is int8-arithmetic tiled vs fp16-arithmetic tiled, not a coopmat effect. Worth keeping in mind when the coopmat-enabled numbers land: the interesting comparison per scheme is `tiled_baseline` → `coopmat_enabled` at the *same* scheme, not cross-scheme.
- **Decode tok/s is nearly identical between 4w and 8da4w within each model** (e.g., 1B: 57.7 vs 59.0; 8B: 9.28 vs 9.48) — consistent with decode being memory-bandwidth-bound on weight loading rather than arithmetic-bound, and with both schemes reading a similar (int4-dominated) number of weight bytes per token.
- **Decode never engages coopmat/WMMA today, independent of scheme or model** (see `research.md` Decision 1) — every decode microbench case dispatched a `*_gemv_coop_*` kernel, never a `*_coopmat` one. A future coopmat-enabled comparison should not expect a decode-tok/s change unless a GEMV-shaped hardware-coopmat path is added separately.
- **Prefill scales roughly with 1/dim² and decode roughly with 1/dim** across the three models, as expected for compute-bound prefill (GEMM, M=2048) vs bandwidth-bound decode (GEMV, M=1) on this iGPU.
- **Prefill has a warm-up effect on Llama 3.2 1B/4w** the first 1-2 runs after the GPU has been idle read noticeably faster (~1200 tok/s) than the converged steady-state (~1133 tok/s) reported above; a reproducibility spot-check (one extra run) landed at 1199.77 tok/s, consistent with this same warm-up pattern rather than contradicting the recorded steady-state. Decode reproduced tightly (57.78 vs the recorded 57.688 ± 0.053). **Anyone comparing against these numbers later should run at least 3-5 back-to-back reps and use the converged value, not a single cold run**, especially for prefill on the smaller/faster models where a handful of seconds of GPU idle is enough to reset this effect.
