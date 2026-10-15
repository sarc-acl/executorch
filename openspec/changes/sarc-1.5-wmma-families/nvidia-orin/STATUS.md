# NVIDIA Orin WMMA matrix status

Updated: 2026-10-15T00:55:23.007261+00:00

Stage: successful parent build recovered after a postprocessing failure; runtime
constant-method validation complete. No build/measurement queue is running.
The legal matrix is incomplete. No model GPU job, gate or timed session has run.
Base: `617897184529ff180af491ee0d5d7de778cbb6c6`.
Own branch: `topic/wmma-nvidia-orin`.
Actor: Codex. Independent review is provided by the controller's fresh reviewer.

## Running and next step

The original queue `.artifacts/jobs/parent-20261015-pinned1/state.json` remains
`failed`, unchanged. Compilation, runner/microbench linking and the CPU helper
had all completed. The postprocessing guard falsely treated `shader_cache/` as
a second output. The local guard now checks the canonical
`build/vulkan_compute_shaders/` and `spv.cpp` directly; no rebuild was performed.
The completed export, compiler, binaries and original logs/manifests are reused.
No source, tool, shader, golden, threshold or historical adapter was changed
outside this run directory.

Recovery evidence under `.artifacts/recovery/parent-20261015-pinned1/`:
`manifest.json`, `golden.txt`, `runtime-constants.txt`, `validation.json`, and
`module-requirements/requirements.json` with exact-module disassemblies.
The recovery manifest hashes the original manifest/failed state and all six
bundle files and 2091 canonical modules. The original provenance records source
base, 30 recursive submodule pins, 50,423 export file hashes, copied recipe,
pinned compiler bytes/version and both image IDs. All recipe/compiler hashes
checked against the recorded build input. The unmodified golden check passes
all 53 shipped variants; no historical compiler exemption is used.

The CPU-only helper's target hash equals the built hash, its ELF dependencies
and link line exclude Vulkan, and nine constant methods execute successfully
on all six models. Results equal the previously validated serialized metadata
exactly; all contexts/sequence lengths are 2560. The target task copy hash also
matches, including the owner decisions. Execution was sequential, under the
inherited lock, with HOLD and foreign-worker checks; no model GPU job ran.

Every one of the 11 reviewed manifest modules passes `spirv-val --target-env
vulkan1.1`. Workgroup variable type sizes match inventory shared-memory totals
and fit the queried 49,152-byte limit. These are module requirement checks,
not GPU correctness or a final hardware/shape legality claim. Capability/type
and storage-format requirements are captured for the remaining joined checks.

Next: finish device-feature/format/shape matching, committed finite selectors
that preserve `orin-fused1` attention, compliant sampling/cleanup/provenance and
per-run A/A computation; then parent snapshot, baseline/A/A and legal matrix.
No new review was requested for the old adapter question or preflight.

## Reviewed input

Input record: external `INPUT.md` and `inputs/REVIEWED-INPUT.json`.
Inventory: `f393c3ff7ff4fc41e4651a5109a8509097c501f6`.
Archive SHA256: `65d1e3ca0eb7e9dffa8e9fbbb1a100bf2cf609868720b9053de67ff8bf5247d8`.
NVIDIA manifest SHA256: `ef5d13471421898c8515e9a8356caff64a20aa97432175cbcab7bf3a5cd748b1`.
Recovery hash checks pass: archive, all 18 per-file hashes and 53 source hashes
against pinned base git objects. Only this reviewed snapshot is used.
Seven texture3d and four buffer entries remain; alternate storage is correctness
coverage, not a silent substitution for the models' texture3d projections.
Numeric property checks and static model metadata were reviewed previously.
Exact module shared-type sizes are now checked. Final feature/format/shape
matching remains pending; runtime constant execution is complete.

## Owner question resolved

The question recorded once on 2026-10-14 about adapter scope was answered by the
owner's recovery clarification of 2026-10-15 00:32 UTC in CAMPAIGN.md. Necessary
run-local corrections are authorized; no pending scope question remains.
Historical sources, sarc/tools, goldens, prompts, tolerances and registered
thresholds remain unchanged. The earlier REVIEW.md describes the prior preflight
only; no repeat preflight review has been requested.

## Evidence limits

Parent golden and runtime constant-method validation pass on the recovered exact
build. No gate, A/A calibration, baseline or candidate timing has run. Static
group-128 linears and q4/group-32 embedding evidence remains unchanged and is
paired with successful runtime metadata validation. Candidate profiles and
measurement adapters are still pending. Both future arms preserve `orin-fused1`
attention. The matrix remains incomplete; no family-wide acceptance is claimed.
