# M5 EVT1 `dq8ca_q4gsw` (8da4w int8 WMMA) Double-Buffer Variant Sweep

**Feature**: `specs/023-8da4w-int8-dbuf-sweep` | **Date**: 2026-07-08 | **Target**: M5 EVT1
(Exynos 2500 / Xclipse 970), driver `f14c51b6f8` (md5 `c9861e9906d0…`), clocks pinned
509/2730/663 MHz (verified bound via sysfs readback before every run)

## Result at a glance

**dbuf2 is the fastest double-buffer loop structure for the int8 `dq8ca_q4gsw` coopmat
shader — it wins all 6/6 tested shapes, beating the shipped `dbuf4` baseline by +18.15% and
the hypothesized `dbuf3` by +7.44% on average GFLOP/s.**

The dbuf3-is-faster-for-int8 hypothesis is **REFUTED**: dbuf3 is the second-fastest variant,
not the fastest.

## SC-001: All four variants attempted

All four double-buffer variants (dbuf1, dbuf2, dbuf3, dbuf4) were ported, built, and
measured. **None failed** — every variant compiled, dispatched the int8 coopmat kernel (not
a fallback), and passed correctness at the small aligned check shape (M=128, K=128, N=128,
group_size=32). No `failure_reason` is reported for any variant.

| Variant | Compiles | Dispatches coopmat | Correctness | SPIR-V verified |
|---|---|---|---|---|
| dbuf1 | ✅ | ✅ (`..._dbuf1_buffer_texture2d_half`) | ✅ PASSED | ✅ (16 `OpCooperativeMatrixMulAddKHR`, int8 component type) |
| dbuf2 | ✅ | ✅ (`..._dbuf2_buffer_texture2d_half`) | ✅ PASSED | ✅ (16 sites) |
| dbuf3 | ✅ | ✅ (`..._dbuf3_buffer_texture2d_half`) | ✅ PASSED | ✅ (32 sites — expected, its peeled epilogue duplicates the unrolled MMA loop) |
| dbuf4 (shipped baseline) | ✅ | ✅ (`..._dbuf4_buffer_texture2d_half`) | ✅ PASSED | ✅ (16 sites) |

## SC-002: Fastest variant, per shape and overall

Measured on `test_dq8ca_dbuf_sweep_bench` (specs/023 Foundational T008), M=2048 (prefill
regime), 3-run internal mean + CoV per shape (this workstream's standard
`get_avg_time_us()`/`get_std_dev_us()` methodology). Every CoV below is under 2%, most under
0.5% — high-confidence measurements, not single untimed samples.

| Shape (K,N) | dbuf1 GFLOP/s (CoV) | dbuf2 GFLOP/s (CoV) | dbuf3 GFLOP/s (CoV) | dbuf4 GFLOP/s (CoV) | Winner |
|---|---|---|---|---|---|
| 1b_wq (2048,2048) | 1462.5 (0.17%) | **1762.5** (0.09%) | 1621.9 (0.21%) | 1499.1 (0.22%) | dbuf2 |
| 1b_w1_gate (2048,8192) | 1372.0 (0.16%) | **1783.1** (0.18%) | 1675.0 (0.09%) | 1599.5 (0.48%) | dbuf2 |
| 3b_wq (3072,3072) | 1315.5 (0.16%) | **1762.5** (0.30%) | 1661.6 (0.16%) | 1548.5 (0.76%) | dbuf2 |
| 3b_w1_gate (3072,8192) | 1307.8 (0.49%) | **1801.9** (0.23%) | 1663.7 (0.17%) | 1518.3 (1.87%) | dbuf2 |
| 8b_wq (4096,4096) | 1241.5 (0.39%) | **1788.4** (0.12%) | 1646.2 (0.16%) | 1424.3 (0.38%) | dbuf2 |
| 8b_w1_gate (4096,14336) | 1285.4 (0.16%) | **1795.1** (0.30%) | 1684.7 (0.20%) | 1461.2 (0.63%) | dbuf2 |
| **Average** | **1330.8** | **1782.3** | **1658.9** | **1508.5** | **dbuf2 (6/6 shapes)** |

**Overall winner: dbuf2** — no "varies by shape" case; dbuf2 is fastest on every single
shape tested, by a clear margin (its narrowest per-shape lead over the runner-up dbuf3 is
+8.6%, at 1b_wq: 1762.5 vs 1621.9).

Full ranking by average GFLOP/s: **dbuf2 (1782.3) > dbuf3 (1658.9) > dbuf4/shipped (1508.5)
> dbuf1 (1330.8)**.

## SC-003: dbuf3-is-faster-for-int8 hypothesis — REFUTED

The hypothesis (that dbuf3 outperforms the other three variants for int8 WMMA, by analogy
with dbuf1 winning the earlier fp16 sweep) does **not** hold:

- dbuf3 loses to dbuf2 on **6 out of 6** tested shapes.
- dbuf3's average GFLOP/s (1658.9) is **7.44% slower** than dbuf2's (1782.3).
- dbuf3 is, however, genuinely the *second*-fastest variant, and does beat the shipped
  dbuf4 baseline by +9.97% (1658.9 vs 1508.5) — so the underlying intuition that a
  different loop structure than dbuf4 could help int8 was directionally correct, just not
  that dbuf3 specifically is the winner.
- Notably, dbuf1 — the variant that won the earlier **fp16** sweep by 1.87x — is the
  **slowest** of the four variants here, 11.78% slower than the shipped int8 dbuf4 baseline.
  The fp16 result does not transfer to int8: the two shaders have materially different
  structure (int8 has a nested groups x chunks loop with a second wsum/wsc ping-pong pair
  that fp16's flat K loop doesn't; see `research.md` Decision 4), and this sweep shows their
  optimal loop structures also differ.

## SC-004: Fastest variant vs. shipped `dbuf4` baseline

**dbuf2 is +18.15% faster than the currently-shipped dbuf4 production baseline** on average
GFLOP/s (1782.3 vs 1508.5), measured in-sweep under this same harness for an
apples-to-apples comparison (not reusing a number captured a different way).

| Variant | vs. shipped dbuf4 |
|---|---|
| dbuf1 | -11.78% (slower than shipped) |
| dbuf2 | **+18.15% (fastest)** |
| dbuf3 (the hypothesis) | +9.97% |
| dbuf4 | baseline (0%) |

## SC-005: Verified vs. failed — all four are verified, none failed

Every variant in this sweep is correctness-verified and coopmat-dispatch-confirmed (see the
SC-001 table above); there is no failed or unverified variant in this dataset to be
mistaken for a valid measurement.

## Recommendation

Based purely on this Tier-1 microbenchmark evidence (per spec Clarifications, no e2e
validation is required by this feature): **dbuf2's loop structure is the strongest
candidate for the shipped `linear_dq8ca_q4gsw_coopmat` shader**, a clear ~18% win over
today's dbuf4 across all six representative shapes with no shape where it underperforms.
Whether to actually switch the shipped production shader to dbuf2 is an explicit follow-up
decision outside this feature's scope (per spec Assumptions), informed by this report.

## Methodology notes

- **Shapes**: `wq` + `w1_gate` for each of LLaMA 3.2 1B / 3.2 3B / 3.1 8B (6 shapes total,
  group_size=32), per spec Clarifications' curated set (matches `specs/008`'s sweep-phase
  convention).
- **Harness**: new sibling bench `backends/vulkan/test/custom_ops/test_dq8ca_dbuf_sweep_bench.cpp`
  (specs/023 Foundational T008) — a `test_coopmat_linear_bench.cpp` sibling rather than a
  mutation of it, since that file's shared constants are tied to other specs' historical
  numbers. Timing uses the harness's own internal 3-run mean + stddev
  (`get_avg_time_us()`/`get_std_dev_us()`), the same convention this workstream's
  microbenchmarks already use.
- **Dispatch mechanism**: `ET_VK_DQ8CA_COOPMAT_VARIANT={dbuf1,dbuf2,dbuf3,dbuf4}`, an opt-in
  env-var branch added to `QuantizedLinear.cpp`'s existing `linear_dq8ca_q4gsw_coopmat`
  kernel-name selection — mirrors the existing (uncommitted, `.tmp-origcm` worktree) fp16
  `4w` dbuf1-4 sweep's own `ET_VK_Q4GSW_COOPMAT_VARIANT` pattern, per the constitution's
  mandate to reuse that tooling. Default dispatch (env var unset) is unchanged from today's
  shipped behavior.
- **One process per variant**: each variant was measured in its own process invocation, so
  an Xclipse PAL pipeline-creation crash on one variant could not have corrupted another's
  results (moot here, since none crashed).
- **A real bug was caught and fixed during this feature's own implementation**: the
  correctness case initially reused the production M=2048 shape, which exceeds the CPU
  reference's M≤256 cap and caused a false "SKIPPED" instead of "PASSED" — fixed by giving
  the correctness case its own small M=128, after which all four variants passed cleanly.
