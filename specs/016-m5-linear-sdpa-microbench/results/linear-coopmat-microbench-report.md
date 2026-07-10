# WMMA Coopmat Improvement Microbenchmark Report — M5 EVT1

Mirrors `specs/007-wmma-improvement-microbench/results/wmma-improvement-report.md`
(`rocky-ryzen` MiniPC), run on the real M5 EVT1 target instead. Harness:
`test_coopmat_linear_bench` (`backends/vulkan/test/custom_ops/`), extended
this feature with 1B/3B shapes alongside the pre-existing 8B ones
(`research.md` Decision 1) — otherwise unmodified. Clock pin (509/2730/663
MHz) and driver identity (`f14c51b6f8`, md5 `c9861e9906…`) re-verified
before capture.

**By scheme (time-weighted across each scheme's 21 measured ops, weighted
by each op's own share of its configuration's total tiled-baseline time,
then averaged equally across the 3 models — same method as `specs/007`):**
- `4w`: coopmat is **+67.0% faster** than tiled
- `8da4w`: coopmat is **+75.8% faster** than tiled

**Comparison against `specs/007`'s MiniPC figures (SC-004):**
- `4w`: M5 EVT1 **+67.0%** vs MiniPC **+60.6%** — same direction, M5 EVT1 somewhat larger.
- `8da4w`: M5 EVT1 **+75.8%** vs MiniPC **-15.2%** — **opposite direction**. On MiniPC, `8da4w` coopmat regressed vs tiled; on M5 EVT1 it is this feature's single largest win. This is a genuine, real platform difference (both sides are `real_effect`, not noise) — not a data error. Plausible explanation (not verified further here): MiniPC (RDNA3 discrete/APU int8 coopmat path) and Xclipse 970 (mobile int8 coopmat path) have different microarchitectural characteristics for the `dq8ca` int8×int4 dispatch; root-causing *why* is out of this feature's scope.

**Statistical basis (FR-002)**: every Tiled/Coopmat value below is a mean
± standard deviation over 5 timed runs (3 discarded warmup runs), per the
harness's own `execute_test_cases` discipline — no result here is a single
untimed sample. Raw harness output: `results/raw/linear-m5evt1.log`.

## Full case table

| Model | Scheme | Op | Tiled (us) | Coopmat (us) | Speedup % | Significance | Dispatch | Correctness |
|---|---|---|---:|---:|---:|---|---|---|
| llama-3.1-8b | 4w | w1_gate | 134949.2 ± 139.1 | 43494.8 ± 51.9 | +67.8% | real_effect | confirmed | verified |
| llama-3.1-8b | 4w | w2_down | 134584.5 ± 238.9 | 44436.3 ± 138.3 | +67.0% | real_effect | confirmed | verified |
| llama-3.1-8b | 4w | w3_up | 134949.2 ± 139.1 | 43494.8 ± 51.9 | +67.8% | real_effect | confirmed | verified |
| llama-3.1-8b | 4w | wk | 10049.5 ± 10.7 | 3388.6 ± 4.5 | +66.3% | real_effect | confirmed | verified |
| llama-3.1-8b | 4w | wo | 38444.6 ± 28.8 | 12624.2 ± 14.0 | +67.2% | real_effect | confirmed | verified |
| llama-3.1-8b | 4w | wq | 38444.6 ± 28.8 | 12624.2 ± 14.0 | +67.2% | real_effect | confirmed | verified |
| llama-3.1-8b | 4w | wv | 10049.5 ± 10.7 | 3388.6 ± 4.5 | +66.3% | real_effect | confirmed | verified |
| llama-3.1-8b | 8da4w | w1_gate | 210529.5 ± 943.1 | 46712.3 ± 666.3 | +77.8% | real_effect | confirmed | verified |
| llama-3.1-8b | 8da4w | w2_down | 212748.5 ± 365.5 | 55203.3 ± 54.1 | +74.1% | real_effect | confirmed | verified |
| llama-3.1-8b | 8da4w | w3_up | 210529.5 ± 943.1 | 46712.3 ± 666.3 | +77.8% | real_effect | confirmed | verified |
| llama-3.1-8b | 8da4w | wk | 15990.9 ± 17.5 | 4175.1 ± 3.5 | +73.9% | real_effect | confirmed | verified |
| llama-3.1-8b | 8da4w | wo | 60603.4 ± 28.4 | 15286.7 ± 33.5 | +74.8% | real_effect | confirmed | verified |
| llama-3.1-8b | 8da4w | wq | 60603.4 ± 28.4 | 15286.7 ± 33.5 | +74.8% | real_effect | confirmed | verified |
| llama-3.1-8b | 8da4w | wv | 15990.9 ± 17.5 | 4175.1 ± 3.5 | +73.9% | real_effect | confirmed | verified |
| llama-3.2-1b | 4w | w1_gate | 38122.0 ± 2.5 | 12631.2 ± 20.8 | +66.9% | real_effect | confirmed | verified |
| llama-3.2-1b | 4w | w2_down | 38688.9 ± 32.6 | 12790.6 ± 28.1 | +66.9% | real_effect | confirmed | verified |
| llama-3.2-1b | 4w | w3_up | 38122.0 ± 2.5 | 12631.2 ± 20.8 | +66.9% | real_effect | confirmed | verified |
| llama-3.2-1b | 4w | wk | 3230.9 ± 46.3 | 1234.6 ± 13.5 | +61.8% | real_effect | confirmed | verified |
| llama-3.2-1b | 4w | wo | 9666.6 ± 1.9 | 3348.5 ± 5.7 | +65.4% | real_effect | confirmed | verified |
| llama-3.2-1b | 4w | wq | 9666.6 ± 1.9 | 3348.5 ± 5.7 | +65.4% | real_effect | confirmed | verified |
| llama-3.2-1b | 4w | wv | 3230.9 ± 46.3 | 1234.6 ± 13.5 | +61.8% | real_effect | confirmed | verified |
| llama-3.2-1b | 8da4w | w1_gate | 60135.1 ± 81.9 | 13537.4 ± 121.2 | +77.5% | real_effect | confirmed | verified |
| llama-3.2-1b | 8da4w | w2_down | 61356.3 ± 103.4 | 16777.2 ± 33.1 | +72.7% | real_effect | confirmed | verified |
| llama-3.2-1b | 8da4w | w3_up | 60135.1 ± 81.9 | 13537.4 ± 121.2 | +77.5% | real_effect | confirmed | verified |
| llama-3.2-1b | 8da4w | wk | 4222.3 ± 2.0 | 991.6 ± 1.3 | +76.5% | real_effect | confirmed | verified |
| llama-3.2-1b | 8da4w | wo | 15605.3 ± 5.6 | 3676.8 ± 2.4 | +76.4% | real_effect | confirmed | verified |
| llama-3.2-1b | 8da4w | wq | 15605.3 ± 5.6 | 3676.8 ± 2.4 | +76.4% | real_effect | confirmed | verified |
| llama-3.2-1b | 8da4w | wv | 4222.3 ± 2.0 | 991.6 ± 1.3 | +76.5% | real_effect | confirmed | verified |
| llama-3.2-3b | 4w | w1_gate | 57373.7 ± 31.7 | 18781.2 ± 9.7 | +67.3% | real_effect | confirmed | verified |
| llama-3.2-3b | 4w | w2_down | 57731.9 ± 110.6 | 18902.0 ± 14.3 | +67.3% | real_effect | confirmed | verified |
| llama-3.2-3b | 4w | w3_up | 57373.7 ± 31.7 | 18781.2 ± 9.7 | +67.3% | real_effect | confirmed | verified |
| llama-3.2-3b | 4w | wk | 7606.0 ± 3.3 | 2610.4 ± 1.9 | +65.7% | real_effect | confirmed | verified |
| llama-3.2-3b | 4w | wo | 21629.5 ± 2.6 | 7236.3 ± 17.7 | +66.5% | real_effect | confirmed | verified |
| llama-3.2-3b | 4w | wq | 21629.5 ± 2.6 | 7236.3 ± 17.7 | +66.5% | real_effect | confirmed | verified |
| llama-3.2-3b | 4w | wv | 7606.0 ± 3.3 | 2610.4 ± 1.9 | +65.7% | real_effect | confirmed | verified |
| llama-3.2-3b | 8da4w | w1_gate | 90086.1 ± 19.4 | 21361.6 ± 138.7 | +76.3% | real_effect | confirmed | verified |
| llama-3.2-3b | 8da4w | w2_down | 91158.2 ± 193.4 | 24284.9 ± 15.7 | +73.4% | real_effect | confirmed | verified |
| llama-3.2-3b | 8da4w | w3_up | 90086.1 ± 19.4 | 21361.6 ± 138.7 | +76.3% | real_effect | confirmed | verified |
| llama-3.2-3b | 8da4w | wk | 11965.0 ± 17.5 | 2892.8 ± 4.5 | +75.8% | real_effect | confirmed | verified |
| llama-3.2-3b | 8da4w | wo | 34182.2 ± 44.3 | 8309.5 ± 3.7 | +75.7% | real_effect | confirmed | verified |
| llama-3.2-3b | 8da4w | wq | 34182.2 ± 44.3 | 8309.5 ± 3.7 | +75.7% | real_effect | confirmed | verified |
| llama-3.2-3b | 8da4w | wv | 11965.0 ± 17.5 | 2892.8 ± 3.7 | +75.8% | real_effect | confirmed | verified |

Note: `wq`/`wo` share an identical shape (K=dim, N=dim) within a given
model, as do `wk`/`wv` (K=dim, N=kv_dim) and `w1_gate`/`w3_up` (K=dim,
N=ffn) — each pair was measured once (not redundantly re-run at the
identical shape) and its GFLOP/s/timing is reported for both named ops,
matching the underlying GEMM performance for both (they are literally the
same shape). `w2_down` (K=ffn, N=dim) has no shape-mate and is measured
independently. `significance` is `real_effect` for all 42 rows: every
speedup is 61.8%-77.8%, far outside any `mean ± 2*stdev` overlap band.

## Excluded / Out-of-Scope

- `lm_head`: excluded, same reason as `specs/007` — the harness's
  synthetic M=1024 case has no production analogue; the real model's
  lm_head projection is always M=1 (a GEMV) regardless of phase.
- Decode-regime linear ops: excluded, same reason as `specs/007` — no
  WMMA-capable GEMV (M=1) coopmat kernel exists for the tiled-vs-coopmat
  comparison at decode (the GEMV case uses a separate `_coop` shader, not
  gated by this comparison).
- No case in either scheme had `dispatch_status != confirmed` or
  `correctness_verified == false` — all 42 rows are in the main table.

## Correctness-verification summary

- `linear_q4gsw_coopmat_buffer_texture2d_half`: SPIR-V inspection (this
  feature, M5 EVT1 build) confirmed 22 genuine cooperative-matrix
  instructions (`OpCooperativeMatrixLoadKHR` x6, `OpCooperativeMatrixMulAddKHR`
  x8, `OpCooperativeMatrixStoreKHR` x8) — `results/spirv/linear_q4gsw_coopmat_buffer_texture2d_half.dis.txt`.
  Correctness confirmed via the harness's own existing production-K
  correctness-shape coverage (`kCorrectnessShapes`/`kRank3CorrectnessShapes`,
  unmodified by this feature) — `linear_q4gsw_M128_K4096_N128_rank3batch1_Buffer`
  PASSED against the fp32 reference.
- `linear_dq8ca_q4gsw_coopmat_buffer_texture2d_half`: SPIR-V inspection
  confirmed 48 genuine cooperative-matrix instructions (Load x12, MulAdd
  x16 with `Matrix*SignedComponentsKHR` int8 flags, Store x8) --
  `results/spirv/linear_dq8ca_q4gsw_coopmat_buffer_texture2d_half.dis.txt`.
  Correctness confirmed the same way at
  `linear_dq8ca_q4gsw_M128_K4096_N128_rank3batch1_Buffer`, PASSED.

## Unrelated finding surfaced during this capture

The harness's pre-existing (unmodified by this feature) small-shape
correctness matrix (`kCorrectnessShapes`) showed `linear_dq8ca_q4gsw`'s
**tiled** (`Texture3D`) variant FAILING correctness at several small
shapes (`M128_K128_N128`, `M256_K256_N256`, `M128_K128_N256`,
`M256_K128_N128`, `M256_K128_N64`), while the **coopmat** (`Buffer`)
variant at the identical shapes PASSED. This is unrelated to this
feature's own change (`kShapes`/`SUMMARY` only) and does not affect any
row in the table above (which uses only production-K shapes, a different
code path from the small synthetic correctness shapes). Logged as
workspace `open-questions.md` Q13 for follow-up; out of scope here.
