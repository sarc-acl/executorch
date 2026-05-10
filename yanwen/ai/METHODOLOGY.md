# METHODOLOGY — how to bench correctly

This doc captures the bench methodology we settled on after discovering that the original 2026-05-07 sweep was systematically inflating forward times. Follow this exactly; don't invent variations without strong reason.

## The scientific bench (algebraic subtraction across subprocesses)

### Why `wallclock/N` is wrong as a headline number

A single subprocess invocation `executor_runner --num_executions=N` measures:

```
wallclock = fork + dynamic linker + Vulkan ICD init + load_method
          + prepack + iter 0 (cold) + (N-1)*steady_forward + teardown
```

If you divide that by N and report it as "per-exec time," you're including amortized `(load + iter 0 + teardown) / N` in every reported number. At L=32 the fixed cost is ~21 s, so even at N=16 you're adding ~1.3 s of contamination per "exec" — about **80% inflation** over true steady-state forward time.

This is exactly what happened in the 2026-05-07 sweep: it reported 2.89 s/exec at L=32 N=16, while true steady-state forward is 1.77 s.

### The fix: algebraic subtraction across two subprocesses

Run **two** subprocess invocations with different N values:

```
W1 = wallclock at N=1 = load + iter 0 + teardown                  (calibration)
WK = wallclock at N=K = load + iter 0 + (K-1)*steady + teardown   (measurement)

steady_forward = (WK - W1) / (K - 1)
```

The fixed cost cancels algebraically. To get a meaningful stdev, run the measurement subprocess multiple times (n_reps).

### The recipe in code

`bench_steady_state()` in `yanwen/scripts/run_llama31_pure.py` implements this. Default: **1 calibration at N=1 + 3 measurement at N=8 each**. Total wallclock ~2 min at L=32 S=128.

Output:
```
=== Calibration: N=1 ===
W1 = 21413.5 ms

=== Measurement: 3 reps × N=8 ===
rep 1: WK=33802 ms, wall/N=4225 ms, steady=(WK-W1)/7=1769.8 ms
rep 2: WK=33791 ms, wall/N=4224 ms, steady=(WK-W1)/7=1768.2 ms
rep 3: WK=33725 ms, wall/N=4216 ms, steady=(WK-W1)/7=1758.8 ms

=== Steady-state forward ===
mean ± stdev: 1765.6 ± 6.0 ms (cv=0.3%, min=1758.8, max=1769.8)
wallclock/N mean: 4221.6 ms (legacy metric; +2456 ms inflation)
```

The **headline number** is the `mean ± stdev` under "Steady-state forward". The "wallclock/N mean" is shown alongside so you can see how badly the legacy metric overstated.

### Quality gates

- **cv < 5%**: measurement is clean. If higher, re-run.
- **W1 should be roughly stable across sessions**. At L=32 it's ~19–22 s. Big drift means something changed in the environment (Mesa update, RAM contention, swap state).
- **The three per-rep steady values should be within ~10 ms of each other.** Wider spread suggests background activity.

### When to use N=1 + 3×N=8 vs. other choices

- **Default 1×N=1 + 3×N=8**: best signal-to-noise tradeoff for L=32 S=128. ~2 min wallclock.
- **Slow configs (S≥512)**: per-forward is 30s–100s; use smaller N (e.g., N=2) and fewer reps to keep total time bounded. Stdev will be looser.
- **Fast configs (L=1, S=128)**: forward is ~100 ms; N=32 is fine. Larger N improves stdev.
- **Never use N=2 calibration + N=3 measurement** (too few iters, swing variance).

## ETDump — what it captures correctly, what it misses

ETDump is the ExecuTorch event tracer. It records per-event timing including per-shader GPU dispatch time. Output: `<tag>.etdp` (binary) + `<tag>.events.tsv` (Inspector dataframe).

### What ETDump captures correctly

- **Per-shader GPU dispatch time** via `vkCmdWriteTimestamp` query-pool. These appear in `events.tsv` as kernel-name events like `{"kernel_name": "linear_coopmat_half", "operator_id": ..., "dispatch_id": ...}`.
- **CPU dispatch events** like `DELEGATE_CALL`, `ETVK_EXECUTE`, `ETVK_COMPUTE_GRAPH_EXECUTE`, `Method::execute`, `load_method`. These are wrapper events; they describe orchestration cost, not GPU compute.
- **Per-iter granularity**: each event's `perf_data.raw` is a list of length N (the `--num_executions` value).

### What ETDump misses

- **Memory-fault stalls outside the GPU dispatch path.** Example: at L=32 S=512 we observed:
  - ETDump category total: 13.7 s
  - Wallclock: 111 s
  - Difference: ~97 s = 88% of wallclock is GPU stalling on `vkCmdCopyBuffer` while staging pages fault from swap.
  - ETDump can't see this because the GPU is not "running a dispatch" during that wait.

**Implication**: at S=128 (no memory pressure), ETDump category total ≈ wallclock. In the cliff regime (S=512), ETDump is no longer a reliable proxy for forward time. **Don't draw shader-optimization conclusions from ETDump at S≥512.**

### Critical unit gotcha

**`perf_data.raw` is in milliseconds.** The legacy `analyze()` function in `run_llama31_pure.py` has a `mean_us` variable name and prints `us` in the header, but the values it sums are actually ms. This caused a unit-confusion debugging session early on. **Use the canonical analyzer** (`pavan-report/yanwen_plan/analyze_etdump.py`) which handles this correctly.

## The three analyzers — pick the right one

| Analyzer | Input | Output | When to use |
|---|---|---|---|
| `pavan-report/yanwen_plan/analyze_etdump.py` | `.etdp` | Categories (linear / matmul / softmax / etc.) + top 15 ops by total time | First-look "where does GPU time go" |
| `yanwen/scripts/linear_by_shape.py` | `.etdp` | Linear dispatches grouped by `[M, N]` output shape | "Which linear shapes dominate" — maps to Q/K/V/O/FFN per-component |
| `yanwen/scripts/etvk_breakdown.py` | `.events.tsv` (NOT `.etdp`) | Per-shader steady-state (iter 1..N-1 mean), bucketed by op family | "What's the iter-0 cold cost" and "per-shader steady time" |

All three are independent of which tree you run them from — they read the etdump/events file directly via the `Inspector` API.

**Do not use** `analyze()` from `run_llama31_pure.py`. It has two bugs (unit mislabeling + event-name filter mangles kernel events). Documented in `GOTCHAS.md`.

## What does NOT work — anti-patterns

### Don't treat `--num_executions N` results as steady-state

This is the legacy mode for back-compat with the 2026-05-07 sweep. It gives `wallclock/N`, which is inflated. Use scientific mode (`bench_steady_state()`) for headline numbers.

The `--num_executions N --etdump-analyze` form IS the correct way to capture ETDump (single subprocess so the ETDump file isn't overwritten between reps). But interpret the wallclock from that capture as "raw subprocess wallclock," not "per-exec forward time."

### Don't read `analyze()`'s "count" column as "dispatches per forward"

It's actually "distinct event positions in the graph × number of iterations probed". The count in `linear_by_shape.py`'s output IS dispatches-per-iter (across all iters), so divide by N if you want per-forward.

### Don't trust `analyze()`'s top-30 op list to surface the dominant GPU shader

The deduplication logic groups by `name.split("[")[0].split("::")[-1]`, which mangles the JSON-formatted kernel-name events. As a result, you'll see wrapper events (`DELEGATE_CALL`, `OPERATOR_CALL`, etc.) at the top but no `linear_vec_*` or `linear_coopmat_*` leaves. Use `etvk_breakdown.py` for per-shader breakdown.

### Don't grep the runner's stderr for `[VK_LINEAR]` lines and treat them as authoritative

Those are printf logs from the dispatch site in `pavan-report/.../Linear.cpp`. They confirm what was dispatched but don't carry timing. Use them only for the "did coopmat fire" verification, not for performance numbers.

### Don't run baseline bench with `pavan-report`'s venv

The partitioner code differs. You'll end up exporting a `.pte` that thinks it's baseline but has subtly different storage tags. Use `main`'s venv for baseline, `pavan-report`'s venv for coopmat.

## Cross-reference

For exact reproduction commands: see `INSTRUCTIONS.md` and `COOPMAT_WORKFLOW.md`.

For storage decisions affecting which shader is dispatched: see `SHADER_DISPATCH.md`.

For specific bugs encountered: see `GOTCHAS.md`.
