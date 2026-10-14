# Xe2 B580 bounded matrix status

Updated 2026-10-14T19:25:31.057829+00:00.
Actor: Codex. Stage: preflight; no timed or gate GPU job launched. Read-only Vulkan queries complete.
Recursive submodule preparation completed. Unmodified `check.sh --no-build` PASS
(release 1240 checks, dev 1921); initial partial-submodule check retained.
Launcher cache/environment, positive raw foreign delta/device-loss rows and named
HOLD unit-lock dry-run checks passed without GPU dispatches.
Own branch: `topic/wmma-xe2-b580`; source base `617897184529ff180af491ee0d5d7de778cbb6c6`.
Initial checkout clean. No prior run STATUS or measurement artifacts found.
Process/lock inspection found no benchmark/build/sampler job or held GPU/desktop-build lock.
Existing coordinator processes and desktop applications are left as found.

## Reviewed input

Pinned inventory `f393c3ff7ff4fc41e4651a5109a8509097c501f6`.
All 18 supplied files independently SHA-256 verified against the provenance record.
Xe2 manifest SHA-256 `5904caa4c1977f256d58b4c1c111e7fdcb2e54bc312676b40c90da6bcf048306`.
Recorded archive SHA-256 `65d1e3ca0eb7e9dffa8e9fbbb1a100bf2cf609868720b9053de67ff8bf5247d8`;
archive bytes supplied by the coordinator and independently verified to match at 19:30 UTC.
Reviewed source-only readiness does not establish queried hardware/module/model legality.

## Next step

Read final adapters and reference records, verify source hashes, validate current device/model facts,
prepare committed dev-only profiles and thresholds, then prune the finite legal manifest before jobs.
All 53 source hashes match the pinned git objects and initial checkout. All six serialized
model metadata/layout checks and queried MMA/subgroup/shared/workgroup limits pass.
Private raw evidence is under the own artifact preflight directory. 12 entries remain
PENDING exact compiled-module and format checks; no candidate has been launched.
Fresh Codex preflight review checks source/model/ABI/profile coverage. Actor implemented
its page-cache, environment, row-provenance, raw DRM, polling, device-loss and HOLD fixes.
Those orchestration changes still need exact-run validation before timing candidates.
No timings or margin claims exist. Unchanged attention control: `b580-fused1-order-p13`.
No pending owner question at initial inspection.

## Coordinator hold

HOLD/HELD live in this run's own artifact root. Units: one build, one gate, one
six-cell session. Each holds a shared .hold-busy lock and rechecks HOLD atomically;
desktop builds additionally take the existing desktop-build lock exclusively.
GPU units take it shared and use the inherited device UUID lock. Named HOLD dry-run
passed; real HOLD was neither created nor removed. Longest unit duration not measured
yet; snapshot/A/A durations will set the projection before the finite candidate queue.
