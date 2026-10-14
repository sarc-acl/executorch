# NVIDIA WMMA matrix status

Updated: 2026-10-14 20:01 UTC

Stage: WAITING FOR OWNER. Preflight independently reviewed; no build or model GPU workload started.
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

Independent preflight review: PASS, fresh Codex context; see REVIEW.md.
Input/source hashes, all 11 numeric candidate checks, six static model inspections,
sensor latency and source boundaries independently verified. Review corrections:
private loss details stay in artifacts; parent comment describes invariant final
controls; trailing whitespace fixed. Shell syntax and base diff whitespace pass.
No release, shader, golden, tolerance, prompt or sarc/tools edit.

Next: await the existing owner question below. Independent preflight and source
preparation are finished; no further recomputation or review while waiting.
No owner-answer polling sooner than 20 minutes; asynchronous answers can steer.
No build, gate or model GPU run while pending. No timing rows or candidate-profile
code exist, and this is not a complete legal matrix or family acceptance.
check.sh --no-build was not run because it compiles selector binaries; the
pre-build prerequisite also blocks that compilation. Runtime metadata execution,
format/exact-module checks, parent snapshot, A/A, gates, sessions, profiles and
complete reviewed matrix remain pending launch evidence.
Independent reviewer must read the full task before review; no reviewer build/GPU jobs.

## Decision needed from the owner

Pre-build metadata prerequisite: the task requires model-fact validation before the
first build, and N7 specifies the ExecuTorch runtime constant method. The fresh
checkout has no built runtime or Python ExecuTorch/torch bindings. Source-schema
inspection confirms all six files have zero-instruction constant methods returning
context/sequence length 2560. Linear group size 128 and 4-bit/group-32 embedding
facts come from the serialized delegate graph, not those constant-method returns.
This is static file evidence, not execution by the runtime.

May the initial pinned, exported-commit build proceed using that static preflight
validation, with a CPU runtime-method check required before any model GPU work?
Otherwise, please provide an approved existing runtime for the read-only metadata
check. No initial build or model GPU workload starts while this question is pending.
The question is recorded once; independent source/preflight work can finish, then
wait without recomputation or another review. No sample-count exception requested.

## Coordinator hold

Core adapter dry run passed with separate HOLD-TEST/HELD-TEST markers: it
acknowledged the hold, released the inherited device lock, reacquired it after
release and removed test markers. Evidence: artifact preflight/hold-dry-run.txt.
The actual coordinator HOLD was neither created nor removed. Future launch units
are one build, gate or resumable session; longest-unit projection remains pending
launch-driver preparation. No build or model GPU queue exists.

## Prompt boundary correction

The strict review of 069d8d910 found a redundant prompt-path addition despite its
byte-identical contents. Removed tools/r1304.txt and changed the proposal's adapter
table to reference the existing unchanged historical prompt. Fixed thresholds and
the existing owner question are unchanged. No build or GPU workload started;
the device matrix remains unlaunched and waiting for the metadata decision.
