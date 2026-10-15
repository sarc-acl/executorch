# NVIDIA Orin WMMA matrix status

Updated: 2026-10-15T04:15:05.534338+00:00

Stage: feature-enabled exact build ready for independent review.
No queue is running. No owner question is pending. The legal matrix is incomplete;
no model GPU job, correctness gate, A/A or timed session has run.

## Changed base and completed build

Original base: `617897184529ff180af491ee0d5d7de778cbb6c6`.
Own branch: `topic/wmma-nvidia-orin`. Actor: Codex.
Built source: `cbb7df9b6a7a605d28192fbc2a8b62c6f3611d9c`.
Orchestration commit: `478b2c3986eb4abfb6665b3166c0715be3c73f81`.
Both future arms must use this same changed source/binary. Control is
`orin-fused1`; candidate profiles change only their registered linear choices.
Original base/build provenance remains separate.

The owner decision of 2026-10-15 03:57 UTC authorizes the minimum Vulkan memory-
model query/device-create change. Only supported `vulkanMemoryModel` is enabled;
the two optional features remain disabled. Query and creation use the advertised
extension on the existing Vulkan 1.1 API path. Unsupported configurations retain
their behavior. Sixteen CPU mock cases pass. The fixed feature has not yet been
exercised on the physical target.

The immutable `orin-wmma-mm-20261015-1` queue completed at
2026-10-15T04:11:53.751280+00:00. It held the shared desktop build lock, used at
most eight build jobs, exported 30 pinned recursive submodules, reused the
original immutable cross/compiler images and applied no patch. It deployed
nothing and launched zero GPU jobs. The unmodified golden check passes all
53 shipped variants; no historical compiler exemption is used.

Post-build audit verifies all 50,448 recorded exported-file hashes, six bundle
hashes, four compiler-file hashes and 8,565 shader-output hashes. The three
changed runtime files and two selector source files also match their exact git
objects and have recorded compiler inputs/object outputs. All 2,091 canonical
SPIR-V modules equal the original parent bytes. Thus the prior 11 manifest-module
requirement/validation records apply by exact hash, including declared shared
sizes; this is not a final physical feature/format/shape legality verdict.

`check.sh --no-build` returns FAIL only for Device.cpp and Device.h outside its
zones, explicitly covered by the new owner decision. Adapter.cpp is already a
listed hook. Twins pass; release selector: 1,240 checks / 31 rows; default dev
selector: 1,921 checks / 43 rows / 533 candidates. No tool, golden, prompt,
tolerance, threshold or shader was modified. Do not label the whole check PASS.

Small evidence: `results/feature-enabled-build.json` and
`results/runtime-feature-enablement.md`.
Raw evidence under this run's artifact root:
- `jobs/orin-wmma-mm-20261015-1/{state.json,queue.log}`
- `build/orin-wmma-mm-20261015-1/{manifest.json,golden.txt}`
- `memory-model-tests/{fixed-build-audit.json,compiled-inputs.json,check.txt}`
- `memory-model-tests/final/result.txt`

## Preserved earlier evidence

The original `jobs/parent-20261015-pinned1/state.json` remains failed and unchanged.
Its completed build was recovered without rebuilding; original raw logs and
manifests remain preserved. `recovery/parent-20261015-pinned1/` binds those records,
golden PASS, nine CPU-only constant methods on all six models and the 11 exact
module requirement records. All runtime constants match serialized metadata,
including context/sequence length 2560. Group-128 linears and q4/group-32
embedding metadata are validated; models were not re-exported.

`results/candidate-profiles.json` maps all 11 manifest entries to committed
`orin-fused-wmma-NN` profiles. Existing CPU evidence covers 9,548 selector cases
and unchanged attention/softmax behavior; these checks were not rerun merely for
this build. Seven texture3d entries match model projection IO; four buffer entries
remain alternate-storage correctness coverage. Both arms preserve the exact
fused attention pair via `ET_VK_SARC_ORIN_SDPA_FUSED` and softmax `orin_g64`.
No matrix candidate is GPU-gated.

Run-local A/A, childless sampler and bash-s helpers have preserved synthetic
validation in `protocol-tests-v2/`. They are not a completed measurement pipeline.
`resource-checks/` validates explicitly serialized image and prepack extents
against queried format limits. Default scalar records and runtime-created weight
resources remain allocation-audit items; no full allocation PASS is claimed.

## Reviewed input

Use only INPUT.md's snapshot `f393c3ff7ff4fc41e4651a5109a8509097c501f6`.
Archive SHA256: `65d1e3ca0eb7e9dffa8e9fbbb1a100bf2cf609868720b9053de67ff8bf5247d8`.
NVIDIA manifest SHA256: `ef5d13471421898c8515e9a8356caff64a20aa97432175cbcab7bf3a5cd748b1`.
The archive, all 18 snapshot files and all 53 referenced source objects were
reconfirmed before this fixed build. Current private validation record:
`memory-model-tests/validation.json`. No mutable input is substituted.

## Review scope and next steps

This is a new runtime/build checkpoint, not another review of a pending question.
The independent reviewer should inspect the authorized three-file runtime diff,
unsupported-device behavior, finite selector/helper changes since the last review,
exact exported build provenance, golden/module identity and disclosed check result.
No reviewer build or GPU work is requested. No changed shader exists for a new
synchronization diff audit. Prior reviewed preparation should be reused.

After review: complete the physical feature/format/shape and allocation joins;
finish committed measurement orchestration with exact row/log/window provenance,
HOLD, device lock, foreign-process sampling and child cleanup; then run the
unmodified parent snapshot, six-cell control/A/A and legal finite matrix with full
gates on these exact binaries. Explain any >3% published-baseline differences
before candidates. Registered 3% geomean / 8% per-cell limits are unchanged.
Independent review and GPU gates remain required; no family-wide acceptance is
claimed. The resumed run remains under the owner's non-graceful $25 cap.

## Owner decisions resolved

Adapter scope was resolved at 2026-10-15 00:32 UTC. Runtime feature enablement was
resolved at 03:57 UTC. The earlier feature question is preserved in
`results/runtime-feature-blocker.md` and git history. No further ruling or waiver
is requested at this checkpoint.
