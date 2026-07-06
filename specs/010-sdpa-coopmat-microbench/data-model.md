# Data Model: SDPA Coopmat Correctness + Microbenchmark

## SDPA Coopmat Correctness Case

One entry per (shader, shape) pair -- at minimum 2 (one QK^T-shaped, one
attn·V-shaped), both at a small, tile-aligned shape, `Buffer`+`half`
storage, `ET_VK_SDPA_COOPMAT` set.

| Field | Type | Notes |
|---|---|---|
| `shader` | string | `sdpa_compute_attn_weights_coopmat` / `sdpa_compute_out_coopmat` |
| `shape` | string | e.g. `S=128,context_len=128,H=2,D=64` -- tile-aligned per research.md Decision 5's alignment rules |
| `dispatch_status` | enum | `confirmed` (dispatched kernel name contains `_coopmat`) / `fallback` -- read from the same kernel-name-suffix mechanism `SDPA.cpp` itself uses (`add_storage_type_suffix`/`add_dtype_suffix`), not assumed from the env toggle alone |
| `correctness_status` | enum | `passed` / `failed`, from `sdpa_test.cpp`'s existing `test_reference_sdpa` ATen-ground-truth comparison (research.md Decision 2) -- genuinely new coverage, no existing case exercises this shape |
| `spirv_verified` | bool | Whether `spirv-dis` confirmed genuine `OpCooperativeMatrix*KHR` instructions in the compiled shader (checked once per distinct shader, constitution Principle VI) |

## SDPA Prefill Comparison Case

One entry per target model -- 3 total (research.md Decision 5's scope
narrowing: SDPA's shape/dispatch is scheme-independent, so one
configuration per model, not six).

| Field | Type | Notes |
|---|---|---|
| `model` | string | `llama-3.1-8b` / `llama-3.2-3b` / `llama-3.2-1b` |
| `head_dim` / `num_heads` / `num_kv_heads` | int | From that model's `params.json`, not hardcoded |
| `tiled_mean_us` / `tiled_stdev_us` | float | `ET_VK_SDPA_COOPMAT` unset, same harness (research.md Decision 3) |
| `coopmat_mean_us` / `coopmat_stdev_us` | float | `ET_VK_SDPA_COOPMAT` set |
| `qk_dispatch_status` / `av_dispatch_status` | enum | `confirmed`/`fallback` per shader, from the harness's own kernel-name field (research.md Decision 4) |
| `speedup_pct` | float | `(tiled_mean_us - coopmat_mean_us) / tiled_mean_us * 100` -- positive means coopmat is faster |
| `significance` | enum | `real_effect` / `noise`, via this workstream's established non-overlapping `mean ± 2·stdev` band rule |
| `excluded_reason` | string, nullable | Populated (and all timing fields left empty) only if the model's real prefill shape fails a tile-alignment check or a dispatch/correctness check -- per FR-006, never silently dropped from the three-model count |

## SDPA Coopmat Microbenchmark Report

The consolidated document (US3): a correctness/dispatch verification
summary first (which shapes were checked, pass/fail, SPIR-V status) so a
reader never mistakes an unverified number for a validated one, followed by
the 3-row model comparison table (or fewer, with excluded models listed
with their reason), followed by one overall statement of whether SDPA
coopmat helps real prefill attention on this device.

No lifecycle/state transitions -- one-shot capture-and-compare, matching
`007`'s report shape for the equivalent linear-coopmat question.
