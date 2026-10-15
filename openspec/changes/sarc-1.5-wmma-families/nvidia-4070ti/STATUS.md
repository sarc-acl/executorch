# NVIDIA WMMA matrix status

Updated: 2026-10-15 00:36 UTC

Stage: RECOVERY / INITIAL PARENT BUILD PREPARED. The owner decision of
2026-10-15 00:32 UTC resolves the earlier metadata bootstrap question.
The initial pinned build is authorized; runtime constant methods must execute
using a CPU-only helper from that build before any model GPU workload.
No unresolved owner question remains. No model GPU workload has started.

Recovery inspection found no active build, measurement queue, sampler or device
lock holder. Existing source preparation finished rc=0; its 30 pinned recursive
submodules are reused. No exports or builds existed to reuse. Available disk
was 43 GiB; there will be one parent build root, no traced build at this step.
The pinned image and current read-only device/driver facts still match launch.
Confirmed the archive and all 18 per-file input hashes again on recovery.

Running: the next launch is the single detached parent-r1 build, using the
unmodified exported base and sarc/tools/build.sh --llama. The queue's atomic
JSON state is under artifact jobs/parent-r1/state.json, with its own launch log.
No correctness gate, calibration or candidate session is queued with it.
The build takes the inherited device lock, checks HOLD and foreign activity,
and has a two-hour timeout (not a >48-hour projection). The HOLD unit is one
build; no timing runs concurrently. A conservative 32 GiB free-space check
precedes export. No intermediate or measured binary is deleted.

Reviewed inventory: f393c3ff7ff4fc41e4651a5109a8509097c501f6.
Archive SHA256: 65d1e3ca0eb7e9dffa8e9fbbb1a100bf2cf609868720b9053de67ff8bf5247d8.
NVIDIA manifest SHA256: ef5d13471421898c8515e9a8356caff64a20aa97432175cbcab7bf3a5cd748b1.
Confirmed archive and all 18 per-file hashes from the reviewed provenance record.
Read REVIEW.md, VALIDATION.md, README.md, PROFILES.md, LEGALITY.md and inventory STATUS.md.
Exactly this immutable input will be used; 11 NVIDIA candidates remain pending
queried hardware/model legality and exact-module checks.

Scope: finite linear matrix only, with unchanged 4070ti-fused1 attention controls.
The final launch decision supersedes the historical two-small-gains stop rule.
No release selection, new bodies, attention/decode/fp tuning, push, PR, service or
hardware-policy change. Private host/model facts stay in artifact evidence.

No measured cells, accepted candidates or family-wide conclusion exists.
Independent preflight: queried all 15 cooperative-matrix type/shape records and
subgroup/full-subgroup/shared/workgroup numeric limits. All 11 manifest candidates
meet these numeric prerequisites; complete legality still awaits feature/format,
model runtime and exact-module evidence. Static graph inspection verifies all
six context lengths, groups, packed embeddings and 12 production NK pairs.
All model sizes match the kit; all six model and three tokenizer hashes recorded
privately in artifacts. Sensor latency measured at requested 5 ms: median 5.133 ms.
No poller remains. Thresholds pre-registered in proposal.md; no clock calibration
or inherited sample-count exception. Four buffer IO entries are alternate
correctness coverage, seven texture3d IO entries have native prefill coverage.

Independent preflight review: PASS for the earlier bounded preflight only;
see REVIEW.md. No repeated source-only review is requested by this recovery.
The new build adapter and queue implementation are documented in proposal.md.
Shell syntax and base-diff whitespace pass. Queue success, command failure
and missing-command checks all produce the expected terminal JSON statuses;
evidence is retained under artifact queue-checks/.

Latest per-cell numbers: none. No timing rows, candidate acceptance or family-wide
conclusion exists. The complete legal matrix is incomplete. The seven native
texture3d and four alternate buffer entries still require exact-module/format
checks, committed selectors and gates; static checks are not measured passes.

Next after build: inspect exact build/provenance and shipped-SPIR-V results,
compile/run a CPU-only runtime metadata helper against its installed runtime,
then finish legality/profile and measurement orchestration before the parent
snapshot and A/A. Runtime execution, parent snapshot, calibration, candidate gates,
all timing sessions and matrix margins remain pending. Do not launch model GPU
work until metadata and all prerequisites pass. No attention change is planned.

## Resolved owner question

The pre-build runtime-metadata circular dependency recorded on 2026-10-14 is
resolved explicitly by the 2026-10-15 00:32 UTC owner instruction. Static model
inspection can precede this initial build; CPU runtime verification remains
required before model GPU jobs. The old question is preserved in git history
and is not asked again.

## Coordinator hold

The existing HOLD dry run remains valid: separate test markers acknowledged,
lock released and reacquired, markers cleaned. Evidence remains artifact
preflight/hold-dry-run.txt. Actual HOLD was neither created nor removed.
Every future detached unit checks it; parent build is the only queued unit.
No reviewer builds/GPU jobs and no repeated preflight review are requested.
