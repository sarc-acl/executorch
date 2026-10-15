# NVIDIA WMMA matrix status

Updated: 2026-10-15 00:47 UTC

Stage: INITIAL PARENT BUILD AND CPU RUNTIME METADATA COMPLETE.
Running: nothing. The parent-r1 detached queue failed rc=3 at postprocessing,
after sarc/tools/build.sh printed SARC_BUILD_OK. The adapter counted canonical
vulkan_compute_shaders/ and its shader_cache/ as two shader output directories.
The local adapter now selects the canonical directory explicitly. No build was
repeated and no failed queue status was rewritten. Diagnosis and recovery are
preserved privately in build/parent-r1/recovery.json and jobs/parent-r1/.

The existing exact export and binaries are retained. Source commit is the
approved base; recursive manifest contains 30 pinned submodules. The original
binary hashes recheck successfully. Shipped-SPIR-V golden comparison passes
for all 53 variants in BOTH llama and backend canonical output directories.
Evidence: build/parent-r1/golden-llama.txt and golden-backend.txt. These checks
establish the golden result; no model correctness/timing pass is inferred.

CPU-only runtime helper compiled against the installed parent runtime and data
loader in the same pinned image. It links neither Vulkan nor a GPU backend,
refuses methods with inputs/backends/instructions or nonzero planned tensor
memory, and executes only the nine named scalar constant methods. All nine
methods on all six models match the saved static values, including context and
sequence length 2560. Static delegate-graph evidence supplies group-128 linear
and packed 4-bit/group-32 embedding facts, which have no corresponding runtime
constant methods in these models. No forward method or model GPU job ran.

Helper source, exact compile command/image, binary/library/source hashes, ldd,
per-model stdout/stderr and combined results are in metadata/parent-r1/.
The small nonprivate summary is results/runtime-metadata.json. The helper source
in tools/runtime_metadata.cpp matches the compiled source bytes. Remaining
launch facts are not replaced by these metadata checks.

Recovery inspection found no active jobs, samplers or device lock holder; 40 GiB
free after build. The owner decision of 2026-10-15 00:32 UTC resolved the original
metadata bootstrap question. No unresolved owner question remains. The same
archive and all 18 reviewed inventory hashes were confirmed before this build.

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
meet these numeric prerequisites; complete legality still awaits feature/format
and exact-module evidence. CPU runtime constants are now checked as above. Static graph inspection verifies all
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

Next: complete feature/descriptor-format and exact-module allocation checks,
prepare committed finite candidate profiles and run-local gate/session drivers,
then capture the unmodified parent snapshot and A/A. The initial metadata
bootstrap is complete and must not be repeated as a new preflight review.
Parent snapshot, A/A calibration, candidate gates, timing rows and full matrix
remain pending; no timing rate or margin is reported. No model GPU job is queued.

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
Every future detached unit checks it. The parent build unit has ended; no
new detached unit is queued at this checkpoint.
No reviewer builds/GPU jobs and no repeated preflight review are requested.
