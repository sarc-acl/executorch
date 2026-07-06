# Data Model: Decode Shader WMMA Acceleration

## Decode Roofline Finding

The central, load-bearing entity -- gates whether anything else in this
feature executes.

| Field | Type | Notes |
|---|---|---|
| `kernel` | string | `linear_q4gsw_coop` (this feature's only in-scope kernel, spec.md Assumptions) |
| `device` | string | `rocky-ryzen` MiniPC (AMD Radeon 780M, RDNA3, Ryzen 9 7940HS) |
| `peak_compute_tflops` | float | 17.8 (FP16, research.md Decision 1) |
| `peak_bandwidth_gbs` | float | 89.6 (research.md Decision 1) |
| `machine_balance_point` | float | ~198.7 FLOPs/byte (`peak_compute_tflops * 1e12 / (peak_bandwidth_gbs * 1e9)`) |
| `kernel_arithmetic_intensity_range` | (float, float) | 4.0-16.0 FLOPs/byte (base to generous-dequant-overhead estimate, research.md Decision 2) |
| `verdict` | enum | `compute_bound` / `bandwidth_bound` / `ambiguous` -- **`bandwidth_bound`** per research.md Decision 3 (12-50x margin) |
| `recommendation` | string | Populated only when `verdict == bandwidth_bound` (FR-003) -- research.md Decision 4 |

## Decode WMMA Correctness Case *(contingent -- only populated if verdict is `compute_bound` or `ambiguous`)*

| Field | Type | Notes |
|---|---|---|
| `shape` | (M, K, N) | Small, tile-aligned to this device's cooperative-matrix dimensions |
| `output_matches_reference` | bool | vs. CPU/ATen, dtype-appropriate tolerance (`010`'s methodology) |
| `spirv_confirmed_coopmat` | bool | Genuine `OpCooperativeMatrix*KHR` instructions, not a renamed tiled kernel |

## Decode WMMA Microbenchmark Case *(contingent -- only populated if the Correctness Case passes)*

One entry per target model (3 total, if reached).

| Field | Type | Notes |
|---|---|---|
| `model` | string | One of the three target models |
| `existing_kernel_mean_us` / `stdev_us` | float | `linear_q4gsw_coop`, real per-token decode shape |
| `new_shader_mean_us` / `stdev_us` | float | New WMMA-capable shader, same shape |
| `verdict` | enum | `real_speedup` / `no_meaningful_difference` / `regression` |
| `consistent_with_roofline_prediction` | bool | Per FR-007 |

## Decode WMMA Feasibility Report

The consolidated document (US1/US2/US3): the Roofline Finding first
(always present), then -- only if reached -- the Correctness Case and the
three Microbenchmark Cases, then one overall statement of whether WMMA
acceleration is worth pursuing for decode on this device.

No lifecycle/state transitions -- one-shot analysis-and-decide, matching
this workstream's established report shape. Given research.md's finding,
this feature's actual report is expected to consist of the Roofline
Finding and its recommendation only, with the contingent sections marked
"not attempted -- roofline finding ruled this out" rather than populated.
