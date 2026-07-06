# Disposition Summary: M5 EVT1 `4w` Linear Coopmat Retune

Status as of 2026-07-05. Schema per `../data-model.md`'s Retuned Shader
Change / Documentation Clarification records.

## Shader changes (`linear_qw_coopmat.glsl`)

| name | risk_level | correctness_result | perf_result | disposition | disposition_reason |
|---|---|---|---|---|---|
| `fp16_accumulate` | precision_risk | not_yet_run | not_yet_run | pending | Blocked on Phase 4's correctness-harness extension AND on driver-identity verification (see below). Device access itself is no longer the blocker (corrected). See `tasks.md` Phase 6 (T014-T017). |
| `loop_flattening` | same_math_code_shape | not_yet_run | not_yet_run | pending | Blocked on Phase 4 AND driver-identity verification. See `tasks.md` Phase 5 (T010-T013). |
| `vectorized_dequant` | same_math_code_shape | not_yet_run | not_yet_run | pending | Blocked on Phase 4 AND driver-identity verification. See `tasks.md` Phase 5 (T010-T013). |

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
is blocked by remaining prerequisites, found across sessions:

1. **~~No M5 EVT1 device reachable~~ CORRECTED**: an earlier session ran
   `adb devices` on this workstation (`sj1-yanwen-d01`) directly and wrongly
   concluded no device was reachable. The M5 EVT1 is attached to a
   *different* host and IS reachable: `ssh yanwen.xu@sj1-dmckee-d01` then
   `adb -s 0000088f8e579c33` (confirmed `getprop ro.soc.model` -> `s5e9975`).
   See `.shared-context/instruction-for-ai/devices-and-access.md`.
2. **NEW, found while verifying device access**: the driver currently
   flashed on this device does NOT match any known-good/known-bad hash in
   `.shared-context/instruction-for-ai/flash-sumd-driver.md`.
   `/vendor/lib64/hw/vulkan.samsung.so` is **47,671,472 B**, md5
   `993d49a9135e7c2dba74b2820da87ed1`, dated 2026-06-22 -- none of the four
   documented builds (`be1273bcbb` 45,925,296 B BAD; `c0d117aaf2` 46,081,392
   B known-good; `f14c51b6f8` 47,660,248 B known-good/current-default;
   factory 47,050,904 B) match this size or hash. This is a fifth,
   undocumented build. Per constitution Principle VIII ("never assume a
   prior session's driver is still there") and the Q9 precedent (a bad
   driver silently miscompiled coopmat with no crash, only caught by a
   small-shape correctness bench) -- **no coopmat correctness/performance
   measurement should run on this device until this driver is identified
   or a known-good one is (re)flashed.** The `logcat | grep SUMD` banner
   that would give the actual driver-identity string isn't in the current
   log buffer (only emitted when a Vulkan app initializes the driver); none
   has run recently enough on this device to produce it.
3. Linking `test_coopmat_linear_bench` fails today with `undefined symbol:
   add_matmul_coopmat_node(...)` -- reproduces identically at `HEAD` with
   none of this feature's edits applied, so it predates and is unrelated to
   this feature. Root cause: `find_package(executorch CONFIG REQUIRED
   COMPONENTS vulkan_backend)` in
   `backends/vulkan/test/custom_ops/CMakeLists.txt` links a **prebuilt,
   stale `vulkan_backend` library** that was built before `GemmCoopmat.cpp`
   (which defines `add_matmul_coopmat_node`) was restored to the source
   tree in commit `b19116260`. A full Android Vulkan backend rebuild is
   needed before T009 can even attempt to run.

## Next steps

Per `../tasks.md`: Phase 4 (T009) needs (a) the driver on the M5 EVT1
identified/resolved and (b) a fresh `vulkan_backend` build before it can
run. Device access itself (T008) is done. Phases 5 and 6 additionally
depend on Phase 4. Neither Phase 5 nor Phase 6 blocks the other;
`fp16_accumulate`'s disposition does not gate
`loop_flattening`/`vectorized_dequant`'s, per this spec's Clarifications.
