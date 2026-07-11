# SDPA Coopmat Correctness + Microbenchmark Report

## Correctness + dispatch verification summary

- `sdpa_compute_attn_weights_coopmat` and `sdpa_compute_out_coopmat` both pass a genuinely new tile-aligned correctness check against the ATen ground truth (`backends/vulkan/test/op_tests/sdpa_test.cpp`, `VulkanSDPATest.test_sdpa_op_coopmat_aligned_*`), at `Buffer`+`half` storage, `S=128, context_len=128, head_dim=64` -- confirmed dispatched via GPU query-pool kernel-name data, not assumed from the toggle alone.
- Both shaders' compiled SPIR-V confirmed to contain genuine `OpCooperativeMatrix*KHR` instructions (36 in `sdpa_compute_attn_weights_coopmat`, 20 in `sdpa_compute_out_coopmat`).
- Every model below has its own dispatch confirmed independently (both shaders, via the microbenchmark harness's own kernel-name capture) before its speedup number is reported.

## Overall: SDPA coopmat is **66.8% faster** than tiled on average across 3/3 measurable models (3/3 real-effect, not noise) at this tier (shader microbenchmark -- not a model-level/e2e claim).

## Per-model comparison

| Model | head_dim | num_heads | num_kv_heads | Tiled (us) | Coopmat (us) | Speedup | Significance |
|---|---:|---:|---:|---:|---:|---:|---|
| llama-3.1-8b | 128 | 32 | 8 | 59666.2 ± 1755.3 | 17313.3 ± 122.6 | +71.0% | real_effect |
| llama-3.2-3b | 128 | 24 | 8 | 44078.4 ± 1958.8 | 13136.9 ± 92.5 | +70.2% | real_effect |
| llama-3.2-1b | 64 | 32 | 8 | 23622.8 ± 590.3 | 9636.6 ± 68.3 | +59.2% | real_effect |

## Excluded models

none

## Notes

- Every mean/stdev above is computed from 5 timed runs (3 discarded warmup runs beforehand), matching this workstream's established iteration-count-and-stdev discipline -- no single untimed run is presented as evidence.
- Timing isolates only the `sdpa_compute_attn_weights_*`/`sdpa_compute_out_*` GPU dispatches per run, excluding the KV-cache-update and softmax dispatches in between (unaccelerated, identical regardless of the coopmat toggle).
- Scope is tier-1 (shader microbenchmark) only, `rocky-ryzen` MiniPC, prefill (`S=2048`) only -- decode SDPA and any tier-2 (model-level) e2e measurement of this path are out of scope for this feature.
- 3 configurations total (one per target model), not the constitution's default six -- SDPA's shape/dispatch is independent of the `4w`/`8da4w` quantization scheme (spec.md Assumptions).