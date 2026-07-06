# Disposition Summary: M5 EVT1 `4w` Linear Coopmat Retune

Status as of 2026-07-05. Schema per `../data-model.md`'s Retuned Shader
Change / Documentation Clarification records.

## Shader changes (`linear_qw_coopmat.glsl`)

| name | risk_level | correctness_result | perf_result | disposition | disposition_reason |
|---|---|---|---|---|---|
| `fp16_accumulate` | precision_risk | not_yet_run | not_yet_run | pending | Blocked on Phase 4's correctness-harness extension (T009), which itself is blocked only on a stale `vulkan_backend` build. Device access AND driver verification are both done. See `tasks.md` Phase 6 (T014-T017). |
| `loop_flattening` | same_math_code_shape | not_yet_run | not_yet_run | pending | Same blocker as above. See `tasks.md` Phase 5 (T010-T013). |
| `vectorized_dequant` | same_math_code_shape | not_yet_run | not_yet_run | pending | Same blocker as above. See `tasks.md` Phase 5 (T010-T013). |

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
`cmake-out-android-vk/backends/vulkan/test/custom_ops`).

## Device access and driver verification (T008) — DONE

1. **Device access, corrected**: an earlier session ran `adb devices` on
   this workstation (`sj1-yanwen-d01`) directly and wrongly concluded no
   device was reachable. The M5 EVT1 is attached to a *different* host and
   IS reachable: `ssh yanwen.xu@sj1-dmckee-d01` then
   `adb -s 0000088f8e579c33` (confirmed `getprop ro.soc.model` -> `s5e9975`).
   See `.shared-context/instruction-for-ai/devices-and-access.md`.
2. **Driver identity, resolved**: found the driver flashed on the device
   (`/vendor/lib64/hw/vulkan.samsung.so`, 47,671,472 B, md5
   `993d49a9135e7c2dba74b2820da87ed1`, dated 2026-06-22) matched NONE of the
   four documented builds in
   `.shared-context/instruction-for-ai/flash-sumd-driver.md` (`be1273bcbb`
   45,925,296 B BAD; `c0d117aaf2` 46,081,392 B known-good; `f14c51b6f8`
   47,660,248 B known-good/current-default; factory 47,050,904 B) — a fifth,
   undocumented build. Per constitution Principle VIII and the Q9 precedent
   (a bad driver once silently miscompiled coopmat with no crash), this was
   resolved before any measurement, with the user's explicit approval:
   - Backed up the unknown driver: `pull` to
     `/sarc-c/gpusw/users/yanwen.xu/vulkan.samsung.so.device-unknown-993d49a9-backup-2026-07-05`
     (md5-verified as an exact copy).
   - Flashed the documented known-good `f14c51b6f8`
     (`/sarc-c/gpusw/users/yanwen.xu/vulkan.samsung.so`, md5
     `c9861e9906d03fa2c7d48b804e1a1c80`) via `flash-sumd-driver.md`'s
     documented push procedure (`root`/`remount`/`setenforce 0`/`stop`/
     push/`remount`/`start` — user explicitly confirmed the `setenforce 0`
     step).
   - Verified post-flash: on-device md5 = `c9861e9906d03fa2c7d48b804e1a1c80`
     (exact match). Pushed the prebuilt NFS `runners/test_coopmat_linear_bench`
     binary and ran `COOPMAT_BENCH_CORRECTNESS_ONLY=1`: **16/16
     Buffer-storage (coopmat) correctness cases PASSED** (the 10 unrelated
     FAILs were all Texture3D/tiled-path `linear_dq8ca_q4gsw`, not coopmat).
     `SUMD` log lines confirmed active post-run.
   - M5 EVT1 is now on a verified known-good driver.

## Remaining blocker for T009

Linking `test_coopmat_linear_bench` (with this feature's own T006/T007
production-K cases) fails today with `undefined symbol:
add_matmul_coopmat_node(...)` — reproduces identically at `HEAD` with none
of this feature's edits applied, so it predates and is unrelated to this
feature. Root cause: `find_package(executorch CONFIG REQUIRED COMPONENTS
vulkan_backend)` in `backends/vulkan/test/custom_ops/CMakeLists.txt` links
a **prebuilt, stale `vulkan_backend` library** that was built before
`GemmCoopmat.cpp` (which defines `add_matmul_coopmat_node`) was restored to
the source tree in commit `b19116260`. A full Android Vulkan backend
rebuild is needed before T009 can run. (The prebuilt NFS
`test_coopmat_linear_bench` used above to verify the driver predates this
feature's T006/T007 additions, so it doesn't exercise them — it was only
usable for driver verification.)

## Next steps

Per `../tasks.md`: Phase 4 (T009) needs a fresh `vulkan_backend` build
before it can run — device access and driver verification (T008) are both
done. Phases 5 and 6 additionally depend on Phase 4. Neither Phase 5 nor
Phase 6 blocks the other; `fp16_accumulate`'s disposition does not gate
`loop_flattening`/`vectorized_dequant`'s, per this spec's Clarifications.
