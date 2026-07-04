# Data Model: MiniPC No-WMMA Baseline Benchmarks

Entities are drawn directly from the spec's Key Entities section, made
concrete enough to author the Phase 2 tasks and the report schema in
`contracts/baseline-report-schema.md`. This feature is measurement/reporting
only — nothing here is a persisted application entity; all of it is flat-file
data produced by benchmark runs.

## Benchmark Configuration

The unit of work: one (model, scheme) pair measured on one device with the
coopmat/WMMA dispatch path excluded.

| Field | Type | Notes |
|---|---|---|
| `model` | enum | One of `llama-3.1-8b`, `llama-3.2-3b`, `llama-3.2-1b` |
| `scheme` | enum | One of `4w`, `8da4w` (see constitution's Quantization Scheme Matrix) |
| `group_size` | int | Chosen as a multiple of 32 (coopmat K-tile size) — see Research Decision 2 |
| `device` | string | Fixed: `rocky-ryzen` (RDNA3 iGPU MiniPC) |
| `dispatch_path` | enum | Fixed for this feature: `tiled_baseline` (coopmat excluded via the Decision 1 toggle) |
| `pte_path` | path | Path to the exported `.pte` used for this configuration |

Six configurations exist in this feature: the cross product of `model` × `scheme`.
`dispatch_path` has only one value here (`tiled_baseline`); a future feature
adds `coopmat_enabled` alongside it for the same six configurations.

## End-to-End Result

One per Benchmark Configuration.

| Field | Type | Notes |
|---|---|---|
| `config` | Benchmark Configuration ref | |
| `prefill_tokens` | int | Fixed: 2048 |
| `decode_tokens` | int | Fixed: 1024 |
| `prefill_tokens_per_sec` | float | From `Stats.prefill_token_per_sec` (`stats.h`) |
| `decode_tokens_per_sec` | float | From `Stats.decode_token_per_sec` (`stats.h`) |
| `num_runs` | int | Number of repeated runs behind this result (statistical-soundness requirement, FR-005) |
| `variance` | float or range | Spread across `num_runs` (e.g., stdev or min–max) for both tok/s numbers |
| `run_metadata` | object | Build identity (git commit/toggle state), `max_seq_len` used, prompt file path, timestamp |
| `status` | enum | `ok` or `failed` (see FR-007); if `failed`, `failure_reason` is required |

## Microbenchmark Result

One per (Benchmark Configuration, shape, regime) tuple.

| Field | Type | Notes |
|---|---|---|
| `config` | Benchmark Configuration ref | |
| `regime` | enum | `prefill` (M=2048) or `decode` (M=1) |
| `op` | string | Which linear op the shape belongs to (e.g., q/k/v/o proj, ffn gate/up/down) — for traceability, not a new abstraction |
| `shape` | (M, N, K) | Real shape as derived per Research Decision 5 |
| `mean_time_us` | float | From `BenchmarkResult::get_avg_time_us()` |
| `stddev_us` | float | From `BenchmarkResult::get_std_dev_us()` |
| `iterations` | int | From `BenchmarkResult::get_num_iterations()` (adaptive probe-then-scale) |

## Baseline Report

The consolidated, reusable artifact (FR-008, US3): one `results/baseline-report.md`
(human-readable, organized by model then scheme) plus one raw JSON file per
Benchmark Configuration under `results/raw/` holding that configuration's
End-to-End Result and all of its Microbenchmark Results. Exact shape defined
in `contracts/baseline-report-schema.md`.

No state transitions or lifecycle apply to any of these entities — each is
written once by a benchmark run and read thereafter by the report and by
future comparison work.
