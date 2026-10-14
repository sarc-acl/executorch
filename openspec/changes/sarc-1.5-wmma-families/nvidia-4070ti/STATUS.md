# NVIDIA WMMA matrix status

Updated: 2026-10-14 19:39 UTC

Stage: preflight. No build or model GPU workload started.
Running: nothing. Detached pinned submodule preparation finished rc=0;
its recursive status/log/PID are under artifact source-prep/. No model
GPU/build queue exists. All preparation stayed inside this clone.
Actor read the complete assembled CAMPAIGN.md, including the final launch decision,
then INPUT.md, existing status/artifacts, processes and locks. No existing run status,
active build/measurement queue or device lock holder was found. Checkout is clean
at source base 617897184529ff180af491ee0d5d7de778cbb6c6 on the assigned topic branch.

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

Next: finish independent source preparation and one fresh preflight review, then
wait for the owner question below. No build, gate or model GPU run while pending.
Independent reviewer must read the full task before review; no reviewer build/GPU jobs.

## Decision needed from the owner

Pre-build metadata prerequisite: the task requires model-fact validation before the
first build, and N7 specifies the ExecuTorch runtime constant method. The fresh
checkout has no built runtime or Python ExecuTorch/torch bindings. Source-schema
inspection confirms all six files have zero-instruction constant methods returning
context/sequence length 2560, linear group size 128 and 4-bit/group-32 embeddings;
this is static file evidence, not execution by the runtime.

May the initial pinned, exported-commit build proceed using that static preflight
validation, with a CPU runtime-method check required before any model GPU work?
Otherwise, please provide an approved existing runtime for the read-only metadata
check. No initial build or model GPU workload starts while this question is pending.
The question is recorded once; independent source/preflight work can finish, then
wait without recomputation or another review. No sample-count exception requested.

