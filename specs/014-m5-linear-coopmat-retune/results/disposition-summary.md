# Disposition Summary: M5 EVT1 `4w` Linear Coopmat Retune

Status as of 2026-07-05. Schema per `../data-model.md`'s Retuned Shader
Change / Documentation Clarification records.

## Shader changes (`linear_qw_coopmat.glsl`)

| name | risk_level | correctness_result | perf_result | disposition | disposition_reason |
|---|---|---|---|---|---|
| `fp16_accumulate` | precision_risk | not_yet_run | not_yet_run | pending | Blocked on M5 EVT1 hardware access (no device reachable — `adb devices` empty) AND on Phase 4's correctness-harness extension. See `tasks.md` Phase 6 (T014-T017). |
| `loop_flattening` | same_math_code_shape | not_yet_run | not_yet_run | pending | Blocked on M5 EVT1 hardware access AND on Phase 4. See `tasks.md` Phase 5 (T010-T013). |
| `vectorized_dequant` | same_math_code_shape | not_yet_run | not_yet_run | pending | Blocked on M5 EVT1 hardware access AND on Phase 4. See `tasks.md` Phase 5 (T010-T013). |

## Documentation clarification (`linear_dq8ca_qw_coopmat.glsl` / `QuantizedLinear.cpp`)

| files | finding_date | validation_gate | disposition |
|---|---|---|---|
| `linear_dq8ca_qw_coopmat.glsl`, `QuantizedLinear.cpp` (`add_linear_dqa_qw_node`) | 2026-06-30 | None — comment-only, no runtime effect | keep |

## Correctness Harness Extension (FR-008)

| shapes_added | data_strategy | tolerance | status |
|---|---|---|---|
| `kCorrectnessShapes`: `{M:128,K:2048,N:128,group_size:128}`, `{M:128,K:4096,N:128,group_size:128}`; `kRank3CorrectnessShapes`: `{M:128,K:4096,N:128,group_size:128,batch:1}` | Reused unchanged: `make_deterministic_correctness_case`'s existing well-conditioned positive-only data | Reused unchanged: `abs=0.5`/`rel=0.05` | **written** (T006/T007 done); **not yet run** |

`test_coopmat_linear_bench.cpp` itself compiles cleanly with these additions
(confirmed via the local Android cross-build,
`cmake-out-android-vk/backends/vulkan/test/custom_ops`). Running it (T009)
is blocked by two independent prerequisites, found this session:

1. **No M5 EVT1 device reachable** (`adb devices` returned empty).
2. **New finding**: linking `test_coopmat_linear_bench` fails today with
   `undefined symbol: add_matmul_coopmat_node(...)` — reproduces identically
   at `HEAD` with none of this feature's edits applied, so it predates and
   is unrelated to this feature. Root cause: `find_package(executorch CONFIG
   REQUIRED COMPONENTS vulkan_backend)` in
   `backends/vulkan/test/custom_ops/CMakeLists.txt` links a **prebuilt,
   stale `vulkan_backend` library** that was built before `GemmCoopmat.cpp`
   (which defines `add_matmul_coopmat_node`) was restored to the source
   tree in commit `b19116260`. A full Android Vulkan backend rebuild is
   needed before T009 can even attempt to run — out of this feature's scope
   (a `/building`-skill-level prerequisite), tracked here rather than
   silently worked around.

## Next steps

Per `../tasks.md`: Phase 4 (T008-T009) needs both M5 EVT1 device access and
a fresh `vulkan_backend` build before it can complete; Phases 5 and 6
additionally depend on Phase 4. Neither Phase 5 nor Phase 6 blocks the
other; `fp16_accumulate`'s disposition does not gate
`loop_flattening`/`vectorized_dequant`'s, per this spec's Clarifications.
