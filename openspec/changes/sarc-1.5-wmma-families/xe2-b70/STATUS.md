# Xe2 B70 finite matrix status

2026-10-15 04:45 UTC. Actor: Codex. **CPU partial-recipe audit prepared; GPU blocked.**
Read the complete updated 1064-line CAMPAIGN.md, including the 04:30 delivery note.
The shared partial RoPE recipe is now present. Archive SHA256 matches the task:
`ba113d7778555427639cea0da7444bdfa018791428fd1e218e8ebdc3f343005b`.
All 26 extracted files match the archive, all five recipe hashes match provenance,
and twelve pinned runtime source hashes agree with the approved base and preserved
topic export. Source recipe commit is `8c6c59c0ff922bd437d78fea7ec7c08dba1815fb`.
Private receipt: input-preparation/b580-rope-bundle-receipt.json.

No prior queue, GPU/build job or held device lock was found on recovery. Reviewed
inventory archive and all 18 input hashes match the pinned INPUT.md snapshot.
No duplicate export investigation, new build or repeated device query was done.

Prepared CPU-only detached job: rope-audit-01, state jobs/rope-audit-01.json.
It reuses the supplied recipe to compare local original hashes, virtual revised
file hashes, all serialized Vulkan constant records and constant-method metadata
against delivered evidence for all six originals. It writes no PTE, executes no
model method and creates no Vulkan device/resource. Its work is serialized under
the inherited host pair lock and obeys HOLD. Queue status is updated on failure.
The original files and RESOURCE_LIMIT_BLOCKED marker stay unchanged.

The bundle's complete=true is CPU preparation only. It does not resolve 8B
width, operator temporary resources, synthetic-gate legality or GPU correctness.
The queue makes no input-admission or correctness decision. On completion, inspect
its new evidence once and park for the remaining scope dependency below.
Six-cell matrix, gate and timing evidence remain absent; no measured margin or
family-wide acceptance is claimed. Independent prior selector/queue preparation
is reused. No push, PR, service, driver, policy, shader or gate change occurred.

Read the complete assembled campaign and final launch decision. Own branch starts at
`617897184529ff180af491ee0d5d7de778cbb6c6`; clean checkout at initial inspection.
No prior run status, detached queue, artifact job or held GPU lock was found.

Reviewed input is inventory commit `f393c3ff7ff4fc41e4651a5109a8509097c501f6`,
source base `617897184529ff180af491ee0d5d7de778cbb6c6`.
Archive SHA256 `65d1e3ca0eb7e9dffa8e9fbbb1a100bf2cf609868720b9053de67ff8bf5247d8`.
Xe2 manifest SHA256 `5904caa4c1977f256d58b4c1c111e7fdcb2e54bc312676b40c90da6bcf048306`.
Archive and all 18 per-file provenance hashes verified; inputs remain read-only.
Source review establishes readiness of 12 candidates, not current device legality.

No measured medians or margins exist yet. Read-only unit/property/sensor and serialized-model inspection completed.
Pinned submodule initialization and check.sh --no-build passed. Seven dev-only
profiles map the 12 input entries; buffer variants receive correctness coverage
because the original supplied model prefill IO is texture3d; revised-input
route coverage is pending the common bundle. Independent source/preflight review and focused CPU profile checks passed.
Isolated HOLD dry test passed; historical outer gate/control/trace wrappers are
inactive because their extra paths do not meet this bounded task. Next, after revised-input admission and full-gate resource legality: run
the parent snapshot and A/A, then assess baseline/calibration before candidates. No candidate is declared launch-ready yet.
The finite matrix supersedes the historical two-small-gains stop rule under the
2026-10-14 launch decision. Attention remains the final campaign configuration in
both arms. No family acceptance is inferred from this member.

## Decision needed from the owner

The partial RoPE bundle has arrived and its provenance is verified; local CPU
applicability validation is the remaining independent job. Please relay the
remaining conforming 8B/operator-temporary recipe and resolve how the full
unmodified gate's independent 8B texture3d fixtures can satisfy the image limits.
The supplied bundle explicitly leaves those needs unresolved. No reduced coverage,
illegal allocation, runtime release-zone edit or gate relaxation is authorized.
This updates the existing dependency once; after the CPU audit, park without
recomputing or requesting another review of the same unresolved scope.

## Coordinator hold

Isolated HOLD.test/HELD.test dry check passed via the inherited hold function.
Real HOLD and HELD remain coordinator-owned at the artifact root. A build is a
job boundary; no queue starts before a checked build and launch prerequisites.
Longest build/job duration remains to be measured before the matrix queue.

Original-input parent runtime metadata check completed successfully for all six
original model files; revised-input runtime metadata remains pending.
Linear group128 and 4-bit embedding group32 were read from serialized Vulkan
operator arguments. All 12 parent module capacities equal the reviewed declarations;
required format features agree with the unit query, but the matching image
extent limit now blocks 8B model launch. Exact private output and binary
hashes remain in the artifact preflight/build records.

## Per-file reviewed input hashes

| Input file | SHA256 |
|---|---|
| LEGALITY.md | 0d44a9fc58e27a5e912486baec81f2d6618195027707ad4e42b6d21c40f6bbda |
| PROFILES.md | f266487d038b46f195a5f2f42c555fd05299d479289b609148ab9c28f4a1cf4a |
| README.md | c7d2d969372a480904005d26acf58f4e6f5d1be281ec9e70c4f6e87e63c7b224 |
| REVIEW.md | a8fd4b0c68ad002a20f1b8490b86cd93e62f44e1e1f2743a8ffe2c5ebd16895f |
| STATUS.md | f642fe626ed6a8ec1b3736fc78610e3aaa73bb0023f9c37a77677d991b187b2c |
| VALIDATION.md | 0cff1dda430d49f94ab70103c3a046a21a53ae8528f36ad0705558b1eb423611 |
| generate.py | 11e6bed5304b053b6603e067dfdabb765063fdb75082de65e5be4c6ac874f97c |
| inventory.csv | f89656186b8fca434270f303c965789c515a8fe05a358910d97c7c774ed2b5a7 |
| legality-matrix.csv | b3945596c0e7ecf92538f8ea607cce755dd3591308b4add0aad0ad7a5573808c |
| manifests/nvidia-ampere-ada.json | ef5d13471421898c8515e9a8356caff64a20aa97432175cbcab7bf3a5cd748b1 |
| manifests/rdna3-radv.json | edcfa39ea7d8fac9c907a6a901754b96aa2d2894302b9c8ad623b90c2780d923 |
| manifests/xclipse.json | 2fe20db84ffba914961bda2c13f3c2a51b6857de7dbf8925f9aac2c84d0e089f |
| manifests/xe2.json | 5904caa4c1977f256d58b4c1c111e7fdcb2e54bc312676b40c90da6bcf048306 |
| packets/7900xtx-radv.md | 7f8483cc3674fd7b981131ba8332f648ddcf44ee9ef7a99e8f63328ebb36f0c5 |
| packets/COMMON.md | e6e321e80f2522ebfc68947f55a0b15f3e64404d3d528746c51ad9e5deb51cc7 |
| packets/m51-private.md | ec3f2280f5f432a5a2b55971ce27ac5ebd852782d1f0e159e0ec8ae456ad1ade |
| packets/rx7600.md | 57ea9506763c594721c1b7ea09b445cbbf130526ca9eaa0b900ab45e7ceb4789 |
| source-hashes.csv | c4a7380cf0b810473b63223a245f248528b81c4adbc6fc1fd455b8d7a6762ace |
