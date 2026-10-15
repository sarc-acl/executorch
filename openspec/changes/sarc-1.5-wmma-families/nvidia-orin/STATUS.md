# NVIDIA Orin WMMA matrix status

Updated: 2026-10-15T01:13:43.450772+00:00

Stage: waiting for one owner ruling on missing logical-device feature enablement.
Independent selector/resource/helper preparation is recorded. No queue is running.
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

Independent checkpoint: all 11 manifest entries now have reproducible own dev
profiles (`results/candidate-profiles.json`). Register only the selected row,
`kUnverified`, on an explicit `orin-fused-wmma-NN` profile. Existing shape-service
and route predicates are preserved; storage-specific buffer entries do not
replace the models' texture IO. Both arms use the exact existing fused pair via
`ET_VK_SARC_ORIN_SDPA_FUSED`; softmax stays `orin_g64`. No attention implementation
or shader changed. The new selection layer preserves all unrelated choices.

CPU validation: 9,548 checks over 11 profiles, storage combinations, decode,
short/unaligned/max-context dimensions, and attention cases. Unaffected choices
and attention/softmax equal the `orin-fused1` control. The unmodified
`check.sh --no-build` passes (release 1240 checks/31 rows; dev 1921 checks/43 rows,
533 candidates with no matrix profile). Compilation took the desktop build lock.
Raw CPU dumps/check log are in `.artifacts/selector-tests/`; small summaries are
in `results/selector-validation.json`. No matrix candidate is GPU-gated.

Run-local helpers implement fully valid per-run A/A calibration without rounding
the floor, reject missing row provenance, sample sensors/process ancestry in one
childless process, preserve GPU_GONE, and quote bash-s transport arguments.
Synthetic tests reject nine invalid cases, exclude an invalid low-clock row,
preserve a fractional floor, confirm zero sampler children/SIGTERM exit, and
preserve literal shell metacharacters across transport. These are helper tests,
not real A/A data. Artifacts: `.artifacts/protocol-tests-v2/`. Full detached
measurement orchestration, exact row/log/time-window joining and real A/A/gates
remain pending after the owner ruling; the helpers are not claimed as a complete
measurement pipeline.

The coordinator's B580 extent finding was investigated for this unit, not copied
as an Orin verdict. Read-only image-format queries use the actual sampled/storage/
transfer usage (15), type and tiling. Optimal 3D formats report 16,384 per axis.
The serialized explicit images and prepack outputs of all six models fit their
matching queried limits; RoPE height 5,120 and 8B packed width 3,584 fit here.
Raw model/format records and allocation precheck are under
`.artifacts/resource-checks/`. Six default scalar tensor records and runtime-
created weight resources remain audit items; no full whole-runtime allocation
PASS is claimed. No input was changed and no limit waived.

The exact compiled-module feature join found the prerequisite below: supported
physical `vulkanMemoryModel` is absent from logical-device enablement in the
pinned base. This blocks both parent GPU snapshot and candidate measurements.
No new full review or queue will start while waiting; previous failed/build
records remain preserved. Resume on a dated CAMPAIGN.md ruling.

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
paired with successful runtime metadata validation. Candidate profiles are CPU-validated; measurement orchestration and GPU evidence
remain pending on the owner ruling. Both future arms preserve `orin-fused1`
attention. The matrix remains incomplete; no family-wide acceptance is claimed.


## Decision needed from the owner

Recorded once at 2026-10-15T01:08:03.652473+00:00.

All 11 exact manifest modules declare `OpCapability VulkanMemoryModel` and
`OpMemoryModel Logical Vulkan`. The physical-device query reports the feature
supported, but pinned `Adapter.cpp:100-251` never requests its extension or
chains `VkPhysicalDeviceVulkanMemoryModelFeatures` / Vulkan12 features at
`vkCreateDevice`; `Device.cpp` never queries that feature. This is an enabled-
feature prerequisite, independent of the successful golden and standalone
`spirv-val` checks. Vulkan's feature description explicitly requires enabling
it on the logical device; shader-module VUID `pCode-08740` applies.

May the coordinator supply a revised pinned base with the required memory-model
feature enabled, or may this stage make a narrowly specified runtime feature-
enablement change? No feature, allocation-limit or shader-compiler waiver is
requested. The existing task permits only dev-zone/run-directory changes, so
no runtime feature change or model GPU job will proceed without a dated ruling.

Evidence: `results/runtime-feature-blocker.md`, exact module requirements under
the recovery artifacts, and the unchanged base source cited there. Independent selector/resource/helper checks are recorded above. Park without
repeat review or recomputation of this question. The earlier adapter-scope question stays resolved.
