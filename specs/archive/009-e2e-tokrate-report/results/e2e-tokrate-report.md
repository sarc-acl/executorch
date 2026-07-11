# End-to-End tok/s Report — Texture, Buffer, and WMMA Across 4w/8da4w

All six configurations were dispatch-confirmed (ETDump kernel-name inspection: every measured linear op's dispatched kernel contains `_coopmat`, per FR-003) and measured successfully -- no blocked/failed configurations.

**Important correction found during this feature's own verification (research.md Decision 8)**: `006`'s originally-published `Buffer` numbers (reused below for the Texture3D/Buffer columns) were captured before an unrelated pass bug (`--vulkan-force-fp16` silently defeating `--vulkan-storage-override buffer` for every per-layer linear op) was found and fixed. `006`'s own `Buffer` captures therefore never actually exercised Buffer storage or coopmat dispatch for these ops either -- they are reused here as originally published (per FR-001), but should be read as a second `Texture3D`-equivalent baseline, not a true Buffer-storage measurement. Only this feature's new `WMMA` column reflects genuine, ETDump-confirmed coopmat dispatch.

## `4w` verdict: does WMMA help?

**e2e prefill is 77.8% faster than the Buffer/tiled baseline on average across all three models (3/3 configurations consistent with 007's microbenchmark-level finding: +60.6% faster than tiled (007, time-weighted across 21 measured prefill linear ops)).**

## `8da4w` verdict: does WMMA help?

**e2e prefill is 3.2% slower than the Buffer/tiled baseline on average across all three models (3/3 configurations consistent with 007's microbenchmark-level finding: -15.2% slower than tiled (007, time-weighted across 21 measured prefill linear ops)).**

008's tuning sweep found config 5 (SUBGROUP_SIZE=32) closes most of the shipped-config gap, landing at roughly parity with tiled (-5.5% to +8% vs tiled across the full-catalog validation) -- but config 5 is not reachable through production's can_use_q4gsw_coopmat() gate (hard subgroup_size()==64 requirement), so it is context only, not part of what this feature's WMMA arm measures (FR-008).

## Per-configuration comparison

| Model | Scheme | Phase | Texture3D (tok/s) | Buffer (tok/s) | WMMA (tok/s) | WMMA vs Buffer | WMMA vs Texture3D | vs 007 finding |
|---|---|---|---:|---:|---:|---:|---:|---|
| llama-3.1-8b | 4w | prefill | 171.05 ± 2.16 | 163.46 ± 1.26 | 316.53 ± 1.09 (5 reps) | +93.6% | +85.1% | consistent |
| llama-3.1-8b | 4w | decode | 9.282 ± 0.014 | 9.299 ± 0.013 | 9.463 ± 0.008 (5 reps) | +1.8% | +1.9% | n/a (decode stays on GEMV, unaffected by coopmat) |
| llama-3.1-8b | 8da4w | prefill | 214.30 ± 0.79 | 211.21 ± 0.54 | 205.75 ± 0.42 (5 reps) | -2.6% | -4.0% | consistent |
| llama-3.1-8b | 8da4w | decode | 9.475 ± 0.015 | 9.454 ± 0.024 | 9.339 ± 0.005 (5 reps) | -1.2% | -1.4% | n/a (decode stays on GEMV, unaffected by coopmat) |
| llama-3.2-3b | 4w | prefill | 388.40 ± 3.92 | 370.81 ± 4.15 | 649.88 ± 2.99 (5 reps) | +75.3% | +67.3% | consistent |
| llama-3.2-3b | 4w | decode | 18.773 ± 0.003 | 18.746 ± 0.007 | 18.752 ± 0.014 (5 reps) | +0.0% | -0.1% | n/a (decode stays on GEMV, unaffected by coopmat) |
| llama-3.2-3b | 8da4w | prefill | 455.28 ± 5.42 | 438.00 ± 3.22 | 432.68 ± 1.66 (5 reps) | -1.2% | -5.0% | consistent |
| llama-3.2-3b | 8da4w | decode | 18.475 ± 0.011 | 18.441 ± 0.022 | 18.241 ± 0.010 (5 reps) | -1.1% | -1.3% | n/a (decode stays on GEMV, unaffected by coopmat) |
| llama-3.2-1b | 4w | prefill | 1132.91 ± 17.13 | 1135.27 ± 32.91 | 1867.40 ± 33.93 (5 reps) | +64.5% | +64.8% | consistent |
| llama-3.2-1b | 4w | decode | 57.688 ± 0.053 | 57.673 ± 0.055 | 59.952 ± 0.044 (5 reps) | +4.0% | +3.9% | n/a (decode stays on GEMV, unaffected by coopmat) |
| llama-3.2-1b | 8da4w | prefill | 1357.46 ± 12.14 | 1344.40 ± 8.42 | 1265.03 ± 9.14 (5 reps) | -5.9% | -6.8% | consistent |
| llama-3.2-1b | 8da4w | decode | 58.955 ± 0.128 | 58.900 ± 0.116 | 58.144 ± 0.057 (5 reps) | -1.3% | -1.4% | n/a (decode stays on GEMV, unaffected by coopmat) |

## Blocked / failed configurations

none

## Notes

- **Prefill cross-session comparison -- 006 documented real session-to-session prefill variance on this hardware (same .pte, mean swung from 388.4 to 355.5 tok/s, stdev from 3.9 to 22.5) unrelated to storage/dispatch type; a modest prefill delta here is not automatically a dispatch-arm effect. Decode is not affected by this and can be compared directly.**
- Decode tok/s is nearly unchanged across Texture3D/Buffer/WMMA for every configuration, as expected -- decode dispatches the GEMV/`_coop` kernel regardless of storage type or the coopmat fix (no WMMA-capable GEMV kernel exists, per `003`).
- Dispatch confirmation (ETDump kernel-name inspection) was performed once per configuration on a separate `EXECUTORCH_ENABLE_EVENT_TRACER=ON` build (mirroring `002`'s precedent, to avoid tracer overhead contaminating the timing captures above, which used the standard, non-instrumented build).