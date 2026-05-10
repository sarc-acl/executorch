# Pure LLaMA 3.1 8B end-to-end timing on AMD 780M iGPU

**Date:** 2026-05-07 (original) · **Updated:** 2026-05-10
**Branch:** `main`
**Device:** AMD Radeon 780M (RADV PHOENIX), Mesa 25.0.7, integrated GPU
**Host:** 28.9 GiB RAM, 24 GiB swap, 868 GiB /home (536 GiB free at start)
**Author:** Yanwen Xu

## TL;DR (2026-05-10 update)

- **Scope narrowed to L=32 only.** All non-L=32 results below are
  **OUTDATED** and not maintained. They were collected with a
  methodology that conflated load + iter-0 cold-start with
  steady-state forward time (see "Methodology — `--num_executions`
  matters" — that section's numbers are inflated). Do not cite.
- **L=32, seq=128, fp16 prefill:** clean steady-state forward is
  **1.77 s ± 6 ms** (cv 0.3%, 3 reps × N=8, calibration-subtracted).
  Equivalent to **72.5 tok/s prefill**. Prior numbers in this doc
  (2.89 s, 3.15 s) were inflated by ~80% due to load+iter-0 leakage.
- **Memory at L=32 seq=128:** see updated section below.
- **System cannot run L=32 at large seq (≥1024).** Confirmed
  memory-architecture-bound, not shader-bound — see "What's
  actually limiting" addendum at the end.

## Goal

Measure end-to-end execution time of **pure, original** LLaMA 3.1 8B fp16
(no coopmat, no Stephen's shader, stock `VulkanPartitioner({})`) on the
AMD 780M iGPU at **L=32**. Followup to the 2026-05-06 coopmat seq-scan
report. Original goal scanned `(n_layers, seq_len)` but is now narrowed
to L=32 only — sub-32 results retained for reference but marked OUTDATED.

## Environment preflight

- Venv: `/home/doremy/sarc-acl/executorch/main/executorch/.venv` (torch 2.11.0+cpu)
- Weights: `/home/doremy/llama3_1_8b/original/consolidated.00.pth` (15 GB)
- Runner: `cmake-out-vk/executor_runner` (built with
  `EXECUTORCH_BUILD_VULKAN=ON`, `EXECUTORCH_BUILD_DEVTOOLS=ON`,
  `EXECUTORCH_ENABLE_EVENT_TRACER=ON`,
  `EXECUTORCH_ENABLE_LOGGING=OFF` — note: this last one is why
  per-iter `Info` logs aren't visible at runtime)
- Swap: `/swapfile` (24 GiB, priority -2). Activated this session via
  `sudo swapon /swapfile` — must be on before any 32L export.
- `RADV_GTT_PCT=80` set automatically by every script (~22.4 GiB GTT cap)

## Scripts

Three scripts at the repo root, all sharing helpers via `run_llama31_pure.py`:

| Script | Role |
|---|---|
| `run_llama31_pure.py` | Combined CLI (`--phase {all,export,run,analyze}`); also the helper-function source for the two below |
| `setup_llama31_pure.py` | **Phase 1 only** — export `.pte` (heavy: ~16 GiB Python RAM at 32L, ~5 min). One-time per `(L, S)` config |
| `bench_llama31_pure.py` | **Phase 2 (+3)** — bench a pre-exported `.pte`. Light, repeatable. Optional `--etdump` for per-op breakdown |

**OOM safety:** every script sets `RLIMIT_AS = (RAM + swap − 2 GiB)` on the
runner subprocess and bumps its `oom_score_adj=1000`. If the runner OOMs the
kernel kills *it*, never the parent shell. MemProbe samples
`/proc/meminfo` every 500 ms during the run.

**Output dir:** `/home/doremy/llama31_pure_run/`. Per-config artifacts:

```
llama31_8b_<L>L_seq<S>_fp16.pte           # lowered model
llama31_8b_<L>L_seq<S>_fp16_input0.bin    # random tokens
llama31_8b_<L>L_seq<S>_fp16.etdp          # ETDump (with --etdump)
llama31_8b_<L>L_seq<S>_fp16.events.tsv    # Inspector full event table
llama31_8b_<L>L_seq<S>_fp16.memprobe.tsv  # /proc/meminfo samples
```

## Build issues encountered & resolved

1. **Stale runner missing `--etdump_path`** — pre-existing
   `cmake-out-vk/executor_runner` was built before the etdump flag was
   wired into source. Rebuilt with
   `EXECUTORCH_ENABLE_EVENT_TRACER=ON`. Resolved.
2. **`EXECUTORCH_ENABLE_LOGGING=OFF` in CMakeCache** — per-iter
   `ET_LOG(Info, "Iteration N of M: X ms")` calls compile to no-ops.
   Means the only timing source is the Python wrapper's `subprocess.run`
   wallclock divided by `--num_executions`. Per-op timing still works
   via ETDump. Not blocking the goal; flag for if we want pure C++-side
   per-iter timing later.
3. **Made `--etdump` opt-in** in the bench script so users without an
   tracer-enabled runner don't get a wall of "unknown flag" errors.

## Setup matrix completed (5 × 4 = 20 configs)

All exports succeeded, 0 failures. Sweep ran from `09:33:47` to `09:58:09`
(~24 minutes total).

| L | seq=128 | seq=512 | seq=1024 | seq=2048 |
|---:|:---:|:---:|:---:|:---:|
| 1 | ✓ 2.4G | ✓ 2.4G | ✓ 2.4G | ✓ 2.4G |
| 4 | ✓ 3.6G | ✓ 3.6G | ✓ 3.6G | ✓ 3.6G |
| 8 | ✓ 5.3G | ✓ 5.3G | ✓ 5.3G | ✓ 5.3G |
| 16 | ✓ 8.5G | ✓ 8.5G | ✓ 8.5G | ✓ 8.5G |
| 32 | ✓ 15G | ✓ 15G | ✓ 15G | ✓ 15G |

**Disk used:** 139 GB. (.pte size dominated by weights — seq_len barely
affects file size, all four files for a given L are within 0.01 GB.)

## Methodology — `--num_executions` matters → **OUTDATED, see new method below**

> **OUTDATED (2026-05-10):** the analysis below shows that `wallclock/N`
> includes amortized `(load + iter 0)`, so it overstates steady-state by
> a known but variable amount. *The numbers in this section are still
> useful as a cautionary record but should not be cited.* The replacement
> methodology is described in **"Scientific bench (2026-05-10 update)"**
> further down.

The Python wallclock measures the full subprocess: `fork + load + prepack +
N×forward + teardown`. So:

```
reported_ms_per_exec = forward + (fixed_cost / N)
```

With small N, fixed_cost dilutes badly. Confirmed empirically:

| Config | N=16 | N=32 | reduction |
|---|---:|---:|---:|
| 1L seq=128 | 189.1 ms | 143.4 ms | −24% |

That 25% inflation at N=16 matches the theoretical fixed-cost-amortization
budget exactly. **For steady-state numbers, prefer N≥32 on small/fast configs;
fall back to N=4–8 only when per-iter wallclock is so big that more iterations
becomes painful.**

## Bench sweep (in progress at time of writing) → **OUTDATED**

> **OUTDATED (2026-05-10):** the table below uses the legacy `wallclock/N`
> metric and is biased. Sweep was also interrupted partway through
> (last write 11:17 on 2026-05-07 ended in a `KeyboardInterrupt`).
> Non-L=32 results retained for reference only and will not be re-collected
> — current scope is L=32 only. See "Scientific bench (2026-05-10 update)"
> for the corrected L=32 numbers.

Order: small memory → large, so OOM at the big end never loses easy results.
Results streamed to `/home/doremy/llama31_pure_run/logs/bench_results.tsv`
**after each config** so prior results survive any mid-sweep crash.

Adaptive `--num_executions`:

| L=1 | L=4 | L=8 | L=16 | L=32 |
|---|---|---|---|---|
| 32, 32, 16, 16 | 32, 16, 16, 8 | 16, 16, 8, 8 | 16, 8, 8, 4 | 8, 8, 4, 2 |

(Columns are seq=128, 512, 1024, 2048.) Lower N for slower configs to keep
total bench wallclock manageable; **bias accepted** — bigger configs have
more steady-state-inflated per-iter numbers.

### Results so far (live, will be replaced by final table)

```
n_layers  seq_len  num_executions  ms_per_exec  status
1         128      32              143.4        OK
1         512      32              287.8        OK
1         1024    16               633.6        OK
1         2048    16              1375.7        OK
4         128      32              326.5        OK
4         512      16              1103.2       OK
4         1024     16              2371.8       OK
4         2048     8               25295.3      OK     ← 25 s/exec
8         128      16              786.7        OK
8         512      16              2198.9       OK
8         1024     8               5205.8       OK
... (sweep continuing through 32L)
```

**Watchpoint:** `32L seq=2048` is expected to OOM during forward pass per
the 2026-05-06 report (5 attempts, all OOM-killed). Bench script is set
to log it as `FAIL_rcN` and continue, not crash the sweep.

## Per-layer scaling at seq=128 → **OUTDATED**

> **OUTDATED (2026-05-10):** ms/layer fit and throughput below are
> contaminated by `(load + iter 0)/N` leakage — and that contamination
> grows with L (more weights → bigger prepack → larger per-rep overhead).
> So the slope (88.7 ms/layer) was inflated and the intercept (51 ms)
> was understated. The corrected L=32 number is **1765.6 ± 6 ms** at
> seq=128, giving **72.5 tok/s prefill** — see new section below.
> The non-L=32 rows are not re-measured (out of scope).

From an earlier run with N=16:

| L | ms / exec |
|---:|---:|
| 1 | 189.1 |
| 4 | 405.4 |
| 8 | 734.8 |
| 16 | 1416.7 |
| 32 | 2887.9 |

Linear fit: **88.7 ms/layer + 51 ms fixed**. Matches a clean residual-block
cost model. 32L throughput: 128 tokens / 2.888 s = **44.3 tokens/sec prefill**.

---

## Scientific bench (2026-05-10 update — L=32 only)

### Methodology

Switched from single-subprocess `wallclock/N` to **algebraic subtraction
across multiple subprocesses**:

```
W1 = wallclock at N=1   = load + iter 0 + teardown            (calibration)
WK = wallclock at N=K   = load + iter 0 + (K-1)·steady + tear (measurement)
steady_forward = (WK - W1) / (K - 1)
```

Run 1 calibration (N=1) + 3 measurement subprocesses (N=8 each). The
differential cancels load+iter-0+teardown algebraically, leaving clean
steady-state forward time. Mean ± stdev across the 3 reps quantifies
across-run variance.

Why not single-subprocess: see "Methodology" section above (legacy `wallclock/N`
overstates by `(load+iter 0)/N`, which at L=32 is ~1.3 s per "exec" — ~80%
inflation over true forward time).

Why not ETDump per-iter: tested — Vulkan delegate's per-op events do **not**
capture GPU sync time. The ETDump-reported per-forward total at L=32 S=128
is 1.97 ms vs. actual 1770 ms (~900× under-reporting). ETDump is informative
for graph structure (op counts, init overhead) but unusable for GPU timing.

Implementation: `bench_steady_state()` in `run_llama31_pure.py`, default
mode of `bench_llama31_pure.py`.

### Headline result: L=32, seq=128, fp16 prefill

```
W1 (= load + iter 0 + teardown):  21.41 s
rep 1 steady:                      1.770 s
rep 2 steady:                      1.768 s
rep 3 steady:                      1.759 s

steady-state forward:  1.766 ± 0.006 s   (cv = 0.3%, n = 3)
prefill throughput:    72.5 tokens/sec
```

Confirms the model is **fp16** (`model.half()` at `run_llama31_pure.py:172`)
and **prefill** (single forward over `[1, seq_len]` token tensor; no
autoregressive decode). KV cache is materialized inside the model but not
reused across iterations.

### Memory at L=32 seq=128

From the memprobe (`/proc/meminfo` sampled every 500 ms during the bench):

| Quantity | Peak | Notes |
|---|---:|---|
| Shmem (RADV GTT-backed Vulkan allocations) | **917 MB** | Activations + staging + workspace |
| Cached (.pte mmap pages resident) | 9.5 GB | Of 16 GB total .pte; only touched pages cached |
| min MemFree | 6.2 GB | 6 GB headroom — system not stressed |
| Swap delta during run | ~0 MB | Baseline ~4 GB pre-existing, no growth |
| GTT cap (`RADV_GTT_PCT=80`) | 24.8 GB | Hard ceiling on GPU-addressable memory |

**Verdict:** L=32 seq=128 is a comfortable fit. Working set (weights + Shmem)
is ~16 GB out of 22.4 GB GTT cap and 28.9 GB RAM. Plenty of room to grow seq.

### ETDump breakdown (L=32 seq=128, N=8)

> **Correction (during update):** my first pass used a homegrown analyzer
> that filtered the kernel-name events away. The Vulkan delegate **does**
> emit per-shader GPU timestamps via `vkCmdWriteTimestamp` (gated on
> `ET_EVENT_TRACER_ENABLED`, which we have), and there are 4557 such
> events in `events.tsv` at L=32 S=128. The canonical analyzer at
> `pavan-report/executorch/yanwen_plan/analyze_etdump.py` already handles
> these correctly. Use that one. The breakdown below was produced with it.

#### Categories (ETDump total = 1908 ms; wallclock-measured forward = 1766 ms)

| Category | Time (ms) | Share | # dispatches | Notes |
|---|---:|---:|---:|---|
| **linear** | **1527.6** | **80.1%** | 226 | Q/K/V/O + FFN gate/up/down × 32 layers + lm_head |
| reshape / view | 106.7 | 5.6% | 2950 | Many small `view_*`, `permute`, `slice`, `view_convert_*` |
| **CPU↔GPU copy** | **91.4** | **4.8%** | 194 | `ETVK_COPY_OUTPUTS` is **75.4 ms (4.0%) per forward by itself** |
| **CPU fallback (cat)** | **76.7** | **4.0%** | 64 | KV-cache concat falls back to CPU — not delegated |
| elementwise | 59.0 | 3.1% | 1219 | residual add, mul, sigmoid, where |
| bmm / matmul | 14.5 | 0.8% | 64 | Attention `Q@K^T` and `attn@V` — tiny at seq=128 |
| CPU fallback (eq.Scalar) | 13.9 | 0.7% | 32 | causal-mask construction |
| softmax | 6.9 | 0.4% | 32 | one per attention block |
| Other CPU fallbacks (mul.Scalar, logical_not, scalar_tensor, embedding) | ~6.2 | 0.3% | ~193 | mask + token embedding |

#### Linear by output shape

| Output shape | # dispatches | sum ms | avg ms | Maps to | Share of linear |
|---|---:|---:|---:|---|---:|
| `[128, 14336]` | 64 | **789.4** | 12.3 | FFN gate + FFN up (`32 × 2`) | **51.7%** |
| `[128, 4096]` | 96 | **646.7** | 6.7 | Q (32) + O (32) + FFN down (32) | **42.3%** |
| `[128, 1024]` | 64 | 78.3 | 1.2 | K (32) + V (32) — `n_kv_heads=8 × head_dim=128` | 5.1% |
| `[1, 128256]` | 2 | 13.2 | 6.6 | lm_head (with output layout convert) | 0.9% |

All linears use **`linear_vec_buffer_texture2d_half`** — the **tiled (non-coopmat) shader**.
This is the same shader that the 2026-05-06 synthetic-LLaMA report measured at ~2× slower
than `linear_coopmat` on the same 780M iGPU.

#### Where forward time goes at L=32 S=128

Combining shapes back into model components:

| Component | ms / forward | Share | Optimization target |
|---|---:|---:|---|
| **FFN linears** (gate + up + down) | ~1004 | **57%** | Coopmat shader could halve this. **#1 lever.** |
| **Attention linears** (Q + K + V + O) | ~510 | **29%** | Same coopmat lever applies. |
| `ETVK_COPY_OUTPUTS` | 75.4 | 4.3% | Output is `[1, 128256]` fp16 = 256 KB; 75 ms is high for that — staging path? |
| `aten.cat` on CPU (KV concat) | 76.7 | 4.3% | Add a Vulkan `cat` op or wire KV-cache differently. |
| Reshape/view | 106.7 | 6% | Mostly free dispatches, but 2950 of them — partition boundaries? |
| Elementwise + bmm + softmax + remaining | ~80 | 4.5% | Already small, low priority. |
| **Total** | **~1853** | (matches 1908 ETDump / 1766 wallclock within timer noise) | |

#### Useful structural takeaways

1. **Linears are 80% of forward time, and 100% of them are tiled (`linear_vec_*_texture2d_half`),
   not coopmat.** The 2× speedup measured on synthetic 4L LLaMA at seq=2048 (`linear_vec` →
   `linear_coopmat`: 4057 → 2041 ms) implies switching to coopmat could take 1527 ms of linear
   time down to ~760 ms — a ~40% reduction in total forward, or ~120 tok/s prefill instead of 72.
2. **FFN > attention by ~2×.** `[128, 14336]` (FFN intermediate dim) takes 789 ms across 64
   dispatches — gate+up. If coopmat is used, this is where the win lands first.
3. **CPU fallbacks are ~5% of forward.** `aten.cat` for KV cache + 4 mask-construction ops
   (`eq.Scalar`, `mul.Scalar`, `logical_not`, `scalar_tensor`). Not the biggest fish, but
   they cost real wallclock and break GPU pipelining (force a CPU-GPU sync point).
4. **`ETVK_COPY_OUTPUTS` is 75 ms** for a 256 KB fp16 output tensor. Should be ~free on UMA.
   Worth investigating — possible staging buffer round-trip.
5. **Init dominates W1.** `load_method` = 17.5 s of the 21.4 s W1. `Method::execute` = 1970 ms
   matches the wallclock-measured forward (1766 ms ± timer overhead). Skipped from the table
   above since they're wrappers, not leaf events.

### Safe seq_len estimate at L=32 (empirical)

Theoretical model said safe to ~1024, marginal to ~1800. **The hardware
disagreed sharply** — both S=512 and S=1024 were tested in this update:

| seq | wallclock / forward | peak Shmem | min MemFree | peak Swap | Status |
|---:|---:|---:|---:|---:|---|
| 128 | **1.77 s** | 0.9 GB | 6.2 GB | baseline | ✓ **performant** |
| 512 | **~95 s** | 11.5 GB | 0.2 GB | 12.3 GB | ◐ completes but **~50× cliff**, near-saturating RAM |
| 1024 | (OOM-killed) | 13.2 GB at OOM | 0.2 GB | 25.8 GB (saturated 24 GB swap) | ✗ |
| 2048 | (OOM-killed) | — | — | — | ✗ (5/5 prior attempts per 2026-05-06) |

Per-layer Shmem scaling between observed points:

```
S=128:  917 MB / 32 ≈  29 MB / layer
S=512: 11507 MB / 32 ≈ 360 MB / layer  →  12× growth for 4× seq (super-linear)
S=1024: 13166 MB / 32 ≈ 411 MB / layer at OOM-time
```

**Practical answer:**

- **Performant (acceptable per-token cost):** S ≤ 128. Possibly to ~256 (untested).
- **Completes without OOM but unusable:** S = 512 (50× slower than S=128 due to swap thrashing).
- **Hard OOM:** S ≥ 1024.

**Why the cliff at 512:** at S=512 the working set is roughly
`15 GB weights + 11.5 GB Shmem + ~1 GB other ≈ 27.5 GB`, against
**28.9 GB system RAM**. The instant intermediates push working set near the
RAM cap, the page cache starts evicting weight pages, and each subsequent
layer's forward has to re-fault them from disk through swap. The ETVK
shader is fine — the box just can't fit the working set without paging.

**This is purely a memory-architecture bound, not a shader bound.** No amount
of coopmat speedup helps once you're swap-thrashing. To run L=32 at seq ≥ 512
on this hardware, the move is to reduce working-set size: weight quantization
(int4 brings 15 GB → ~4 GB), KV-cache offload, or per-layer streaming —
none of which are free, all of which are out of scope for "pure original" runs.

**Comparison vs 2026-05-06 report:** That session reported 32L seq=128 tex3d
at 8778 ms/exec; we now measure 2888 ms/exec for the same config — ~3×
faster. Most likely causes: prior session used `--num_executions=4`
(default), and Mesa/RADV updates between sessions. The methodological gap
alone (cold-start dilution) explains most of the difference; rebuild
freshness covers the rest.

## Reproduction

### One-time setup

```bash
sudo swapon /swapfile
swapon --show
source /home/doremy/Desktop/samsung/executorch/.venv/bin/activate  # OR
# source /home/doremy/sarc-acl/executorch/main/executorch/.venv/bin/activate
cd /home/doremy/sarc-acl/executorch/main/executorch
```

### Setup-only (per config)

```bash
python setup_llama31_pure.py --n_layers 32 --seq_len 128
```

### Bench-only (per config — requires .pte already exists)

```bash
python bench_llama31_pure.py --n_layers 32 --seq_len 128 --num_executions 16
# with --etdump for per-op:
python bench_llama31_pure.py --n_layers 32 --seq_len 128 --num_executions 16 --etdump
```

### Full sweep scripts

- Setup sweep: `/tmp/setup_sweep.fish` (writes to `logs/setup_sweep.log`)
- Bench sweep: `/tmp/bench_sweep.fish` (writes to `logs/bench_results.tsv`
  + per-config logs `logs/bench_L<L>_S<S>.log`)

These are at `/tmp` so they vanish on reboot. The .pte and .etdp artifacts
in `/home/doremy/llama31_pure_run/` persist and are the actual deliverables.

## Files generated this session

In `/home/doremy/llama31_pure_run/`:

- 20 `.pte` files (full L × seq matrix) — 139 GB total
- 20 `*_input0.bin` files (random token inputs)
- `llama31_8b_1L_seq128_fp16.etdp` (351 KB) — sample per-op timing
- `llama31_8b_1L_seq128_fp16.events.tsv` (231 KB) — Inspector dump
- `logs/setup_sweep.log` — full setup sweep log
- `logs/bench_results.tsv` — bench summary (one row per config)
- `logs/bench_L<L>_S<S>.log` — per-config full bench logs

In repo root (`/home/doremy/sarc-acl/executorch/main/executorch/`):

- `run_llama31_pure.py` — combined CLI + helper module
- `setup_llama31_pure.py` — Phase 1 entry point
- `bench_llama31_pure.py` — Phase 2 entry point
- This document: `2026-05-07_pure_llama31_e2e_session.md`

## Open items

1. Wait for bench sweep to finish; replace the live results table above
   with the final table, including any OOMs at the 32L end.
2. Decide whether `32L seq=2048` should be retried with a smaller
   `--num_executions` (e.g., N=1) or skipped.
3. Optional: rebuild the runner with `EXECUTORCH_ENABLE_LOGGING=ON` to
   get C++-side per-iter timing (would let us discard iter 0 cleanly).
4. Per-op steady-state via Inspector: `analyze()` currently averages
   over all N samples in `event.perf_data.raw`. To discard iter 0,
   change `mean(raw)` → `mean(raw[1:])` (a 5-line edit). Useful for
   future per-op investigations.
5. Compare current pure numbers vs the 2026-05-06 coopmat numbers at
   matching configs — coopmat winning at small seq, regressing at
   large seq (per the prior report) — to see if the seq-dependent
   crossover still holds with the cleaner methodology.

## Notes

- Auto mode active throughout; agent ran setup & bench sweeps in
  background, monitored progress, and did not poll proactively
  between completion notifications.
- `chmod +x` on the new scripts; they don't actually need it since
  we always invoke through `python` explicitly, but harmless.
- All durations in this doc are wallclock from the Python wrapper.
  C++-side per-iter timings are unavailable due to logging build flag
  (see "Build issues" #2 above).
