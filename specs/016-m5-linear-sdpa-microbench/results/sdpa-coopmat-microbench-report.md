# SDPA Coopmat Correctness + Microbenchmark Report — M5 EVT1

Mirrors `specs/010-sdpa-coopmat-microbench/results/sdpa-coopmat-microbench-report.md`
(`rocky-ryzen` MiniPC), run on the real M5 EVT1 target instead. Harness:
`test_sdpa_coopmat_bench` (`backends/vulkan/test/custom_ops/`) — already a
wired CMake target in current HEAD (commit `b19116260`; the Clarifications
session's conclusion that it needed wiring was based on a buggy grep, see
`research.md`/`tasks.md` T005), unmodified by this feature. Note: this is
a **different** harness from `test_coopmat_attention_bench.cpp`, which
this feature's spec originally (and incorrectly) cited before
Clarifications -- that file exercises the unrelated generic
`matmul_coopmat`/`coopmat_mm_ref` path and crashes on an unrelated-shape
assertion; it was not used here. Clock pin (509/2730/663 MHz) and driver
identity (`f14c51b6f8`, md5 `c9861e9906…`) re-verified before capture
(shared with the linear microbenchmark, same session).

## Correctness + dispatch verification summary

- `sdpa_compute_attn_weights_coopmat` and `sdpa_compute_out_coopmat`'s
  compiled SPIR-V (M5 EVT1 build) confirmed to contain genuine
  `OpCooperativeMatrix*KHR` instructions: 36 in
  `sdpa_compute_attn_weights_coopmat`, 20 in `sdpa_compute_out_coopmat` --
  identical instruction counts to `specs/010`'s MiniPC citation, confirming
  the same shader logic. See `results/spirv/sdpa_compute_attn_weights_coopmat_buffer_buffer_half.dis.txt` /
  `results/spirv/sdpa_compute_out_coopmat_buffer_buffer_half.dis.txt`.
- Every model's dispatch confirmed independently via the harness's own
  kernel-name capture (not assumed from the `ET_VK_SDPA_COOPMAT` toggle
  alone) -- all 3 report `confirmed`.
- Correctness itself is inherited from `specs/010`'s own
  `sdpa_test.cpp`/`VulkanSDPATest.test_sdpa_op_coopmat_aligned_*` coverage
  (unmodified, not re-run by this feature -- this feature is the
  microbenchmark half only, per `research.md` Decision 5's scoping).

## Overall: SDPA coopmat is **79.5% faster** than tiled on average across 3/3 measurable models (3/3 real-effect, not noise) at this tier (shader microbenchmark -- not a model-level/e2e claim)

**Comparison against `specs/010`'s MiniPC figure (SC-004):** M5 EVT1
**79.5%** vs MiniPC **66.8%** -- same direction (SDPA coopmat wins on both
platforms), M5 EVT1 noticeably larger. Unlike the linear microbenchmark's
`8da4w` result, this is not a sign-flip -- both platforms agree SDPA
coopmat is a real win, M5 EVT1's win is simply bigger.

## Per-model comparison

| Model | head_dim | num_heads | num_kv_heads | Tiled (us) | Coopmat (us) | Speedup | Significance |
|---|---:|---:|---:|---:|---:|---:|---|
| llama-3.1-8b | 128 | 32 | 8 | 194782.0 ± 328.0 | 35999.0 ± 36.0 | +81.5% | real_effect |
| llama-3.2-3b | 128 | 24 | 8 | 149019.5 ± 170.6 | 27107.2 ± 19.5 | +81.8% | real_effect |
| llama-3.2-1b | 64 | 32 | 8 | 88028.3 ± 233.6 | 21800.0 ± 40.8 | +75.2% | real_effect |

All three speedups are far outside any `mean ± 2*stdev` overlap band
(stdevs are <0.4% relative in every row) -- all `real_effect`, none `noise`.

## Excluded / Blocked models

None -- all 3 target models produced a valid, dispatch-confirmed
measurement on the first run (no build failure, no runtime crash, no
excluded shape).

## Notes

- Every mean/stdev above is computed from 5 timed runs (3 discarded
  warmup runs beforehand), matching this workstream's established
  iteration-count-and-stdev discipline -- no single untimed run is
  presented as evidence. Raw harness output (including the `RESULT,...`
  CSV lines): `results/raw/sdpa-m5evt1.log`.
- Timing isolates only the `sdpa_compute_attn_weights_*`/`sdpa_compute_out_*`
  GPU dispatches per run, excluding the KV-cache-update and softmax
  dispatches in between (unaccelerated, identical regardless of the
  coopmat toggle) -- same methodology as `specs/010`.
- Scope is tier-1 (shader microbenchmark) only, M5 EVT1, prefill
  (`S=2048`) only -- decode SDPA and any tier-2 (model-level) e2e
  measurement of this path are out of scope for this feature (that is
  `specs/015`'s territory, and `specs/015`'s own SDPA e2e attempt for 3B/8B
  crashed with `VK_ERROR_DEVICE_LOST` under `ET_VK_SDPA_COOPMAT=1` at the
  full 1024-decode length -- see workspace `open-questions.md` Q12. This
  microbenchmark's clean, crash-free result on all 3 models at the
  shader-isolation tier suggests Q12's crash is specific to the full
  e2e/decode-length context, not the coopmat shaders themselves).
- 3 configurations total (one per target model), not the constitution's
  default six -- SDPA's shape/dispatch is independent of the `4w`/`8da4w`
  quantization scheme, same as `specs/010`.
