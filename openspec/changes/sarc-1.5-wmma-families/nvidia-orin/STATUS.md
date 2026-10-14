# NVIDIA Orin WMMA matrix status

Updated: 2026-10-14T19:37:53.353945+00:00

Stage: launch blocked on adapter scope clarification. No build or kernel measurement launched. Read-only Vulkan properties/format queries only.
Base: `617897184529ff180af491ee0d5d7de778cbb6c6`.
Own branch: `topic/wmma-nvidia-orin`; clean base at initial inspection.
The full assembled task, including launch decision, was read before other work.
No previous local status, run queue or build artifacts exist. Initial process and
lock inspection found no build/measurement job. The target has a runner listener;
its activity still requires read-only inspection before any device work.

## Reviewed input

Inventory: `f393c3ff7ff4fc41e4651a5109a8509097c501f6`, source base above.
All 18 per-file hashes in the supplied provenance record pass. All 53 source
hashes pass against both base git objects and checkout.
NVIDIA manifest SHA256: `ef5d13471421898c8515e9a8356caff64a20aa97432175cbcab7bf3a5cd748b1`.
Archive SHA256 supplied: `65d1e3ca0eb7e9dffa8e9fbbb1a100bf2cf609868720b9053de67ff8bf5247d8`.
The archive was supplied during preflight; its SHA256 now passes. The read-only reviewed snapshot is the sole input.

## Next step

Await the existing owner scope question. Independent source preparation and
initial preflight review are complete; protocol thresholds are registered before
any measurement. Exact-module legality stays pending until a conforming build.
Historical compiler exemptions do not carry over. Both arms preserve `orin-fused1`
attention; only reviewed linear entries may change. No timing results exist.

## Decision needed from the owner

Recorded once at 2026-10-14T19:34:54.803224+00:00.

The task permits copied-tool adaptation only for host constants. Existing adapters
conflict with this task in four concrete places: A/A floor uses per-cell medians
and relaxed validity; sampler/watch shutdown does not stop child processes;
deploy scripts use ssh command strings instead of `bash -s`; timed rows omit the
required per-row build/profile/driver/candidate provenance.

May this run add narrowly scoped run-local corrections or an orchestration layer
for those four requirements, while keeping original historical adapters,
`sarc/tools`, prompts, goldens and fixed thresholds untouched? The exact source
locations and before/after behavior are in `proposal.md` under Adapter review.

No compiler exemption is requested or inferred. Parent build/golden, candidate
profiles, gates and timed matrix stay unlaunched until tooling conforms.
After independent source preparation/review finishes, wait for the owner's answer;
no repeated recomputation/review, and no polling sooner than 20 minutes.


Property inspection completed: vendor/driver signature, subgroup and compute limits
recorded privately; full KHR matrix array queried. Both required MMA types exist.
Model sizes, serialized constant methods and Vulkan delegate metadata match.
All six contexts are 2560; delegated linears use group size 128; embedding
operators encode q4/group 32. Runtime method execution stays pending.
The pinned cross image is present. Own-clone recursive submodule initialization
completed successfully; exact pinned objects are available.
Compiler and adapter protocol compatibility are under inspection; no measurement
profile has been accepted and no GPU kernel has been dispatched.

Independent preparation complete: model hashes and format/sensor queries saved
privately; no foreign known GPU workload observed at query boundaries. Seven
texture3d and four buffer entries remain in the finite manifest. Compiled-module
allocation checks and correctness evidence remain pending, not assumed.
One fresh source reviewer completed the initial preflight review (REVIEW.md).
No queue is running and no follow-up review will start while waiting on the question.
The reviewer confirms direct semantic copied-tool edits need scope clarification;
independent supplemental orchestration may satisfy the task without those edits.
The existing question is retained without repeating it.
