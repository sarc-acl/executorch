# Xe2 B580 bounded matrix status

Updated 2026-10-14T19:25:31.057829+00:00.
Actor: Codex. Stage: exact parent build; no timed or gate GPU job launched. Read-only Vulkan queries complete.
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

Build launch 2026-10-14T19:42:38.945102+00:00: parent-v1 at approved pristine base,
pinned image, recursive git export, eight jobs, exclusive desktop-build lock, detached
with own build log/status. Source profiles/thresholds committed at `680446bb30`.
No GPU timing or gate launched; no owner question pending.

Parent-v1 launcher PID exited before export/build creation. No compilation ran.
Empty launcher log retained; startup mechanism and inner HOLD deadlock are being
corrected and dry-run tested. No matrix restart or GPU retry occurred.

2026-10-14T19:49:33.129869+00:00: setsid dry-run survived tool-session exit.
Second fresh source review recorded six orchestration findings. Actor corrected outer
HOLD-only checks, timed clean_exec, all sensor-source checks/durable sampler failures,
refusal to overwrite existing session evidence, full runner/library/build hash checks
and stage hash manifest; removed unused historical adjudication tools. CPU timed
environment test clears ambient force-tiled. No build tag or measurement exists yet.

2026-10-14T19:50:42.797866+00:00: parent-v2 launcher PID 3507162 survives detached
under init. Unit .hold-busy READ and desktop-build WRITE locks verified. Recursive
pristine-base export in progress; source/threshold/orchestration commit `14a76d3c88`.
Model SHA-256 records for all six PTE and three tokenizers complete.
Next: parent build and unchanged shipped-SPIR-V check, compiled-module/format checks,
selector build, parent snapshot and A/A. No candidate queue until projected runtime
and baseline are checked. No owner question pending.

2026-10-14T20:09:53.088113+00:00: pristine parent-v2 completed BUILD_BOTH_OK;
unmodified shipped-SPIR-V check PASS (53), exact binary/library hashes and recursive
31-tree export manifest recorded. All 12 compiled candidate storage declarations
match the pinned inventory and fit the shared limit. Read-only format/tiling queries
identify the 8B image-extent blocker described in BLOCKER.md. No GPU model or timed
job has been launched. No queue/build currently running.

## Decision needed from the owner

Required 8B PTEs derive texture3d extents [3584,2048,1], above the matching queried
maxExtent [2048,2048,2048] in the pinned runtime's selected linear tiling. The source
check is debug-only, and historical success is not a current legality exception.
How should this required six-cell stage proceed under that allocation-limit conflict?
Please supply the owner's explicit resolution or approved conforming inputs; re-export,
release fixes and assuming an old exception are outside this stage. See BLOCKER.md
and the artifact storage/model/source evidence. This question is recorded once.

## Waiting boundary

Finish the CPU exact dispatch audit and one independent review of this explicit
blocker, commit the evidence, then wait. No model GPU job or new full build queued.
No repeated owner polling/recomputation/review; check for an answer at most every
20 minutes. All six measured cells, parent snapshot and A/A remain unstarted.
