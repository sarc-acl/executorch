# NVIDIA WMMA matrix status

Updated: 2026-10-15 04:43 UTC

Stage: REPLACEMENT BUILDS COMPLETE / FEATURE PROBES, SNAPSHOTS AND A/A PREPARED.
Running next: detached feature-control-aa-r1. Its controller state is artifact
jobs/feature-control-aa-r1/state.json. No candidate profile is queued.
The previous feature-builds-r1 job completed rc=0 at 04:39:51 UTC. No build,
runner or sampler remains active, and no campaign lock holder was present at
recovery. No owner question is pending.

## Completed replacement evidence

parent-mm-r1: 5f02679b9a20d0b43ed44ea42cb1d779e1f269ac, original approved base
plus exactly the owner-approved memory-model fix. matrix-mm-r1:
dd57ce0e4fda4d7a5c72f9ce3c13b075428eec9e, the same runtime source bytes plus
the existing eleven dev-only selectors and run evidence/tools. Both export 30
pinned submodules from git objects and use the pinned container/compiler.
All four unmodified golden checks PASS (53 shipped variants). All 2,091 SPIR-V
modules per component/build are byte-identical to the original retained parent.
The eleven fixed module shared-array audits match their source declarations.
Exact binary hashes rechecked successfully. results/feature-builds.json records
the small summary; raw build provenance, hashes and logs remain in artifacts.

The queue retired superseded matrix-r1 and new parent-mm-r1 CMakeFiles only,
after preserving configuration/link provenance and every removed file hash.
All old/new binaries, installed libraries/headers, SPIR-V, source exports,
raw logs and failed job records remain. About 30 GiB is free. There is one
active intermediate tree; no further full build or measured-binary deletion
is planned for these exact selector profiles.

New CPU-only metadata execution against parent-mm-r1 passes all nine scalar
methods on all six original models, including context/sequence length 2560.
The helper links only core runtime and mmap loader; ldd has no Vulkan dependency.
Serialized group-128 linear and packed 4-bit/group-32 embedding evidence is
unchanged. results/runtime-metadata-memory-model.json and private
metadata/parent-mm-r1/ preserve source/helper/library/image provenance and raw
outputs. No forward method or model GPU inference was used for metadata.

## Prepared queue and remaining checks

The next detached queue first executes four small feature-chain probes, linked
separately to the exact parent/matrix and llama/backend static runtime libraries.
Each probe constructs Runtime and records the real VkDeviceCreateInfo passed to
the driver, without changing it. It requires the memory-model extension enabled,
vulkanMemoryModel=1, DeviceScope=0, availability/visibility chains=0 and
vkCreateDevice success. No model, shader, dispatch or timed-run instrumentation
is involved. Compilation is complete; actual enabled-chain observations remain
PENDING until the queue runs. The first compile lacked the VMA include path;
that failed log is preserved and the corrected compile uses the existing target
include flags. No runtime build was repeated to fix the helper command.

Only after those observations pass, the queue stages the exact retained binaries
for replacement s0-parent-mm-verify and s0-topic-mm-control, using the unmodified
verify.sh --models 1b,3b,8b --schemes 4w,8da4w --pdiff. Each gate has a two-hour
cap and must retain all 37 command observations with no observer abort. The raw
control output must match the new parent line by line with rates removed. The
new parent output is also compared with the old observations to diagnose any
behavior change. Parent rc=1 dispatch-expectation labels are not relabeled PASS.
The old snapshot is historical execution evidence, not this replacement gate.

Then s1-aa-mm-r1 compares parent-mm-r1 and matrix-mm-r1 under the same final
4070ti-fused1 profile, with no WMMA candidate selected. It uses all six cells,
fresh warmed processes, odd/even interleaving and the unchanged pre-registered
5/7-repeat rule, >=5 clock samples, thermal/foreign rejection and a single
floor(0.97*lowest valid A/A median clock). Every attempt and validity decision
is retained in runs.csv. After each of the first three runs, remaining descendant
processes are recorded and must be zero. A/A has a four-hour cap; the total queue
has a nine-hour timeout including both gates/probes and export-free staging,
well below 48h. No candidate launches automatically from its outcome.

All units check HOLD and use the inherited UUID lock; A/A releases it while held.
Observer children are reaped and no polling subprocess is used. Actual HOLD is
untouched. No build runs during this queue. Gate/feature failures stop for
diagnosis. Device loss records GPU_GONE and stops without retry or reset.

After the queue: recompute raw validity, calibration, uncertainty and baseline
medians; explain any historical tuned-reference cell outside 3% before candidate
work. Finite candidate gates/sessions, alternate-buffer correctness coverage,
next-token comparisons, dispatched names, margins and independent matrix review
remain incomplete. No family-wide acceptance or measured candidate pass exists.

## Reviewed input and scope

Inventory f393c3ff7ff4fc41e4651a5109a8509097c501f6; archive SHA256
65d1e3ca0eb7e9dffa8e9fbbb1a100bf2cf609868720b9053de67ff8bf5247d8;
NVIDIA manifest SHA256
ef5d13471421898c8515e9a8356caff64a20aa97432175cbcab7bf3a5cd748b1.
All 18 input hashes remain confirmed. The immutable manifest has seven native
texture3d and four alternate-buffer entries. Queried matrix/type/subgroup,
shared memory, whole-model resources and matching formats remain recorded in
preflight/ and the earlier source summaries; aggregate peak memory is not inferred.

The 04:08 owner decision authorizes exactly Adapter.cpp, Device.cpp and Device.h
from Orin cbb7df9b6a7a605d28192fbc2a8b62c6f3611d9c. Supplied patch SHA256
5434b8e5c94ba8b18fb11e502577054bc3dea3af4e94263e6470113e316938a9.
Feature commit 8ebe66c2b and results/memory-model-fix.json preserve that scope.
Unmodified check.sh --no-build reports the two expected zone failures for
Device.cpp/Device.h; selector tests pass 1240 release / 1921 dev checks. Raw
output is checks/check-memory-model-no-build.txt. No HOOKS/tool/golden relaxation
is used and that overall check is NOT a plain PASS.

Attention remains k4070tiRefine1 fallback, 4070ti_nzf softmax, and fused3sb
variants d64_t32x32g11s32rko / d128_t16x64g11s32rko with existing K/V copy.
Candidates change only linear selection. No new shader, attention/decode/fp
linear tuning, search, release family selection, promotion, push, PR, service,
driver or host-policy change. Private paths/policy/unit facts remain in artifacts.
