# NVIDIA Orin WMMA matrix status

Updated: 2026-10-15T04:27:52.355873+00:00

Stage: runtime/build review accepted; static joins complete; snapshot queue preparation committed.
No queue is running. No owner question is pending. The snapshot queue has not launched. The legal matrix is incomplete;
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


## Continuation checkpoint after runtime review

The 04:19 UTC independent review accepted the runtime/build checkpoint and
authorized continuing without another preflight review or owner question.
Current read-only target inspection found no campaign workload or held device
lock; governor remains unchanged. Fresh full properties are preserved in
`.artifacts/legality/current-vulkaninfo.txt`; unit/vendor/driver identity matches
the original query. `tools/join_legality.py` joins all 11 exact modules to
physical MMA, features, subgroup and workgroup limits and checks six models'
static resource bounds. Result: PASS for these static checks, recorded in
`.artifacts/legality/join.json`. Default scalar extents and runtime linear-weight
packing are covered; actual allocations/correctness remain GPU gate evidence.
The first join attempt reversed serialized 2D/3D enum mapping; checked
StorageUtils.h, corrected the local checker and reran successfully before launch.

New run-local `device_snapshot.py` and `snapshot_queue.py` are preparation,
not completed protocol validation. They preserve the unmodified verify.sh,
wrap its actual binaries to fill page cache and record rc/provenance, sample
sensors/foreign ancestry without children, capture memory, stop owned sampler
children, retain failures and mirror raw target evidence back to NAS.
The bridge checks NAS HOLD before the full gate boundary; device checks its
HOLD and inherited lock. No timed adoption or matrix candidate queue is included.
The copied historical parent content checker and next-token/run-row helpers
are byte-identical; its prompt-path adapter only redirects to staged prompts.
The checker is intended for parent snapshot content, not candidate acceptance.

Remaining before snapshot launch: exercise instrumentation with synthetic runner
processes (success, failure and signal cleanup), fix any findings, verify payload
provenance/paths and configure a unique detached snapshot job. Required detached
state must be written on all failures. No gate/timing evidence exists yet.
Remaining after snapshot: diagnose its results, six-cell A/A/control medians and
registered clock qualification, baseline explanation, candidate orchestration,
next-token/reference-error evidence and the legal finite matrix with full gates.

Coordinator reminder at 04:27 UTC reports about $20 of the resumed non-graceful
$25 cap spent. This commit preserves preparation and exact remaining work before
the cap can stop the run. It does not claim a queue has started or the stage is
complete. Preserve the reviewed cbb7df9b6a build; no rebuild is needed for these
orchestration-only changes.


## Snapshot launch checkpoint

Instrumentation synthetic tests now PASS for success, rc=7 failure retention,
SIGTERM propagation, sampler rc=0/no children, raw sensor/provenance records and
gate exit records. Evidence: `.artifacts/snapshot-tests/first/`. This executes a
fake CPU runner only and certifies no GPU result. Syntax and whitespace checks
pass. Gate termination now signals its owned process group and waits for cleanup.

Prepared unique snapshot-only job `orin-wmma-s0-20261015-1`, state
`.artifacts/jobs/orin-wmma-s0-20261015-1/state.json`. It reuses the reviewed exact
build and preserves unmodified verify.sh and prompts. The queue enforces HOLD at
the full-gate boundary on the workstation and target, with the inherited device
lock acquired by verify.sh; per-invocation wrappers enforce target HOLD, foreign
ancestry, sensors, cooling and memory safeguards. Both task copies include the
owner decisions. Target rc/sensor/memory/environment records and all raw outputs
are copied back on terminal completion/failure. Maximum queue projection is six
hours, below 48 hours; no candidate matrix or A/A is included.

Next action is launch, then controller parking on its JSON state. A completed
snapshot still needs content/telemetry diagnosis before A/A. If this continuation
hits its cap, preserve this job and inspect both target/NAS states before any
resume; do not start another snapshot or recompute reviewed preparation.
