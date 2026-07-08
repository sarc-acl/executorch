# Data Model: M5 EVT1 Linear + SDPA Coopmat Microbenchmark Validation

## Linear Benchmark Case

One entry per (model, scheme, op) -- 42 total (3 models x 2 schemes x 7
ops), matching `specs/007`'s case set exactly, each carrying both the
tiled and coopmat measurements from `test_coopmat_linear_bench` at the
production shape, on M5 EVT1.

| Field | Type | Notes |
|---|---|---|
| `model` / `scheme` / `op` | string | e.g. `llama-3.2-1b`/`4w`/`w1_gate`, matching `specs/007`'s naming exactly |
| `k` / `n` | int | Shape, from each model's `params.json` (`dim`/`ffn_dim_multiplier`/`multiple_of`/`n_heads`/`n_kv_heads`) per `research.md` Decision 1's `kShapes` extension |
| `tiled_mean_us` / `tiled_stdev_us` | float | From this feature's M5 EVT1 capture, `ET_VK_FORCE_TILED_LINEAR=1` set |
| `coopmat_mean_us` / `coopmat_stdev_us` | float | From this feature's M5 EVT1 capture, default (no env override) |
| `tiled_kernel` / `coopmat_kernel` | string | Captured via the harness's own kernel-name field. `tiled_kernel` MUST be from the `_tiled`/`_coop` (gemv) family; `coopmat_kernel` MUST contain `coopmat` -- a case violating either is not a valid comparison (see `dispatch_status`) |
| `dispatch_status` | enum | `confirmed` (`coopmat_kernel` contains `coopmat`) / `fallback` (it doesn't) |
| `spirv_verified` | bool | Whether SPIR-V inspection (reused from `specs/007`'s existing citation if the shader is unchanged since, or freshly captured) confirms `OpCooperativeMatrix*KHR` instructions for this case's kernel -- checked once per distinct kernel name |
| `correctness_verified` | bool | `dispatch_status == confirmed AND spirv_verified AND` the kernel family is covered by `test_coopmat_linear_bench.cpp`'s existing correctness-shape checks |
| `speedup_pct` | float | `(tiled_mean_us - coopmat_mean_us) / tiled_mean_us * 100` -- positive means coopmat is faster |
| `significance` | enum | `real_effect` / `noise`, via the same non-overlapping `mean +/- 2*stdev` band rule `specs/007` established |
| `weight` | float | This op's share of its (model, scheme)'s 7 measured ops' total tiled-baseline time -- used for the time-weighted overall figure, same method as `specs/007` |

## SDPA Benchmark Case

One entry per model -- 3 total, matching `specs/010`'s case set exactly,
from `test_sdpa_coopmat_bench` at each model's real prefill SDPA shape, on
M5 EVT1.

| Field | Type | Notes |
|---|---|---|
| `model` | string | e.g. `llama-3.1-8b` |
| `head_dim` / `num_heads` / `num_kv_heads` | int | From each model's `params.json`, matching `specs/010`'s table columns exactly |
| `tiled_mean_us` / `tiled_stdev_us` | float | From this feature's M5 EVT1 capture, `ET_VK_SDPA_COOPMAT` unset |
| `coopmat_mean_us` / `coopmat_stdev_us` | float | From this feature's M5 EVT1 capture, `ET_VK_SDPA_COOPMAT=1` set |
| `dispatch_status` | enum | `confirmed` (harness's own kernel-name capture shows `sdpa_compute_attn_weights_coopmat`/`sdpa_compute_out_coopmat`) / `fallback` / `blocked` (build or runtime failure -- recorded with the exact error text, per spec Edge Cases) |
| `speedup_pct` | float | Same formula as the Linear case |
| `significance` | enum | `real_effect` / `noise`, same rule as `specs/010` |

## Excluded / Blocked Entries

A separate list, always rendered in each report (even if empty), covering:

- Linear: any case whose shape doesn't satisfy the coopmat tile-alignment
  precondition (`M%128==0, N%64==0, K%16==0` per `QuantizedLinear.cpp`),
  or where `dispatch_status == fallback` or `correctness_verified ==
  false` -- excluded from the main table and the time-weighted overall
  figure, listed here with the specific reason.
- SDPA: any model whose case is `blocked` (build failure wiring the new
  CMake target, or a runtime crash) -- listed with the exact error text,
  per the spec's Edge Cases; not silently dropped from the 3-model count.

## Linear Coopmat Microbenchmark Report (M5 EVT1)

The consolidated document (User Story 1): one time-weighted overall `4w`
and `8da4w` speedup figure at the top (mirroring `specs/007`'s "+60.6% /
-15.2%" format), followed by the full 42-row case table sorted by
`model`, `scheme`, `op`, followed by the Excluded section -- explicitly
labeled M5 EVT1 and linking back to `specs/007`'s MiniPC report.

## SDPA Coopmat Microbenchmark Report (M5 EVT1)

The consolidated document (User Story 2): one overall average speedup
figure across valid (non-blocked) models at the top (mirroring
`specs/010`'s "66.8% faster... 3/3 real-effect" format), followed by the
3-row (or fewer, if any are blocked) case table, followed by the
Excluded/Blocked section -- explicitly labeled M5 EVT1 and linking back to
`specs/010`'s MiniPC report.
