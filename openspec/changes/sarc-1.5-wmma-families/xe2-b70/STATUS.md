# Xe2 B70 finite matrix status

2026-10-14 19:51 UTC. Actor: Codex. Stage: prerequisite validation. Exported parent build wmma-b70-parent-01
completed (normal and traced), shipped golden PASS. No GPU gate or timing job
has started. Exported topic build wmma-b70-topic-01 is running detached from 841db81df,
before any GPU session. Parent build and runtime metadata are complete.
Preregistration/selectors committed locally at b6bce5616; no push or PR.

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
because supplied model prefill IO is texture3d. Independent source/preflight review and focused CPU profile checks passed.
Isolated HOLD dry test passed; historical outer gate/control/trace wrappers are
inactive because their extra paths do not meet this bounded task. Next: commit
preregistration, export and build parent/topic, check exact modules and
runtime metadata before any measurement. No candidate is declared launch-ready yet.
The finite matrix supersedes the historical two-small-gains stop rule under the
2026-10-14 launch decision. Attention remains the final campaign configuration in
both arms. No family acceptance is inferred from this member.

## Decision needed from the owner

None recorded at this milestone.

## Coordinator hold

Isolated HOLD.test/HELD.test dry check passed via the inherited hold function.
Real HOLD and HELD remain coordinator-owned at the artifact root. A build is a
job boundary; no queue starts before a checked build and launch prerequisites.
Longest build/job duration remains to be measured before the matrix queue.

Parent runtime metadata check completed successfully for all six model files.
Linear group128 and 4-bit embedding group32 were read from serialized Vulkan
operator arguments. All 12 parent module capacities equal the reviewed declarations;
required format/features agree with the unit query. Exact private output and binary
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
