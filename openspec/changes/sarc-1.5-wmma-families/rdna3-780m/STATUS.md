# rdna3-780m status

Updated 2026-10-15 04:42 UTC. Actor: Codex.
Stage: bounded RDNA3 RADV matrix; monitoring wait released by owner decision
2026-10-15 03:57 UTC. Parent build and CPU metadata complete; preparing selector build.
The complete assembled campaign and all final decisions were read on resume.
Own branch and working tree were clean; no existing campaign job, queue, sampler,
build or device-lock holder was found at that resume. The completed build below
is now preserved and reused; there is no active GPU job or build.

## Reviewed input and preserved preparation

Inventory commit: f393c3ff7ff4fc41e4651a5109a8509097c501f6.
Source base: 617897184529ff180af491ee0d5d7de778cbb6c6.
Archive SHA256: 65d1e3ca0eb7e9dffa8e9fbbb1a100bf2cf609868720b9053de67ff8bf5247d8.
RDNA3 manifest SHA256: edcfa39ea7d8fac9c907a6a901754b96aa2d2894302b9c8ad623b90c2780d923.
Archive, all 18 provenance files and all 53 pinned source hashes reconfirmed.
Inputs remain read-only. Prior preparation and its independent review are reused;
see proposal.md, REVIEW.md and results/preflight-candidates.csv.

All 19 candidates pass the partial queried declared requirements: 15 match
predicted production storage and four remain for alternate-storage correctness.
SPIR-V payloads and actual MMA operands/results now pass the exact-module checks
below. Driver LDS allocation, runtime storage and origin binding remain pending.
Six serialized model sizes/context/group/embedding records match the kit; runtime
constant-method execution now passes as recorded below. Full property,
host/compiler/sensor/model records remain privately under the artifact root.
Nineteen candidate profiles are committed for the next build; GPU gates, A/A and
per-cell results remain pending. CPU selector tests have not yet run.
Attention control remains 780m-final across arms; actual dispatch is pending.
No member or family acceptance is claimed. The matrix remains incomplete.

## Monitoring decision resolved

The owner expressly permits own-user GPU-client accounting plus device-wide busy
sampling despite inaccessible other-user client counters. This supersedes the
request for a fully observable environment; no owner question remains pending.
Every result must disclose `own_user_plus_device_busy_other_users_unobservable`
and cite owner decision 2026-10-15 03:57 UTC. Other foreign-workload, sensor, clock,
thermal, correctness and acceptance rules remain in force. No interval is claimed
fully observable; observed foreign activity still invalidates the run.

The preserved assessment is private `preflight/recovery-monitoring-20261015.json`,
SHA256 2f56d3859d276e28242b96c6deeccc89d943e80f55cbefb9898ba48e6cd99f9c.
It remains evidence of the limitation, not a monitoring pass.

## Build checkpoint and next work

Completed job: `<artifact-root>/jobs/rdna3-780m-parent-20261015a/state.json`,
rc 0 at 2026-10-15 04:25:44 UTC. Both backend and runner pass the unchanged
53-variant shipped-SPIR-V golden check. Three binary hashes were rechecked.
An after-build audit compares 50,464 exported files across the root and 30 pinned
submodule trees to their git blobs: zero mismatches. The 40 exact modules covering
19 candidates and all dispatch aliases were inspected across both shader builds.
Workgroup payload bytes equal the inventory declarations for all modules and fit
the queried limit; used matrix types/scope/shapes agree with the required fp16/
fp32 or signed-int8/int32 operations. This is not driver LDS or GPU correctness
evidence. Small evidence: results/parent-build.json. Private raw audits:
preflight/parent-export-audit.json and preflight/parent-modules.json.

The parent used tools/queue.py and tools/build_parent.sh at its recorded commit.
The builder is now generalized as tools/build_commit.sh for committed descendants.
These helpers leave the
historical adapters and sarc/tools untouched. The initial build exports the exact
unmodified base and recursively pinned submodules from object stores, uses the
existing pinned container image and unmodified build.sh --llama, and records the
image, compiler, source manifest and binary hashes. Both generated backend and
runner shader directories must pass the unmodified shipped-SPIR-V golden check.
No model forward execution was part of the parent queue. Its timeout was four
hours; the next CPU-only metadata queue has a 30-minute timeout. No work above
the 48-hour projection threshold is queued.

Foreground queue checks covered successful completion, nonzero command exit and a
missing executable; each produced the expected terminal JSON state, and failures
prevented later steps. Logs are private under queue-tests-20261015/. Shell syntax
and whitespace checks pass. These are orchestration checks, not GPU evidence.

The first CPU metadata job failed with rc 2 at 2026-10-15 04:32:12 UTC during
helper compilation, before any model execution. Its immutable state/logs and
exported helper source remain under jobs/rdna3-780m-metadata-20261015a/ and
helpers/rdna3-780m-metadata-20261015a/. Diagnosis: the installed imported targets
omit C10_USING_CUSTOM_GENERATED_MACROS, so the helper header attempted to include
the absent torch/headeronly/macros/cmake_macros.h. The parent runner's saved
flags and the pinned root CMakeLists.txt explicitly define that macro. Added the
same definition to this helper target and matched the parent's C++20 standard;
no parent source, installed header, gate or threshold was changed. No old build or
failed job is overwritten.

Metadata attempt b completed rc 0 at 2026-10-15 04:33:28 UTC. All six current
model hashes match their preserved records; nine constant methods per file match
the static records, including context/sequence length 2560. The helper ELF has
only standard C/C++ runtime dependencies and no Vulkan symbols/dependency. Small
evidence is results/runtime-metadata.json; raw evidence and helper provenance
remain in helpers/rdna3-780m-metadata-20261015b/. No forward or GPU model method
was executed. The failed attempt a remains preserved.

Next job: `<artifact-root>/jobs/rdna3-780m-matrix-20261015a/state.json`.
The queue writes queued/running/completed/failed atomically, retains logs on failure
and checks coordinator HOLD before each unit. The build takes the existing device
lock to exclude a concurrent timed job. No HOLD is created or removed by the actor.
Return waiting_job as soon as it starts; do not poll inside the actor turn.

The next queue exports the committed selector build and first runs CPU comparison
tests against the untouched parent selector sources. Only after those pass does
it build the runner/backend/tests and compare both shipped-SPIR-V goldens, then
audit the new exact export and candidate modules. An unmodified check.sh --no-build
is a subsequent CPU-only unit. No GPU model work or timing is queued.

Profiles 780m-wmma-rdna3-01 through -19 are generated only from the pinned reviewed
manifest. Their versioned environments and origin references are recorded in
results/candidate-profiles.json. All keep DEV_PROFILE=780m-final and add only the
run-local WMMA profile selector. The append-only dev block preserves control
attention/decode and checks the exact storage, declared shared-memory bound and
normal alignment. Original final-profile ordered picks and earlier table service
are replayed using target properties, including inherited/table fallbacks. Source
device names are predicate fixtures, not external hardware queries or acceptance.
Rows with no timed production coverage must be labeled accordingly. Static C++
syntax passes; behavioral selector comparisons and the new build remain pending.

After the selector build: monitored parent snapshot, runtime dispatch/storage confirmation and A/A,
then the legal matrix and full gates with invariant attention. Record and explain
baseline differences before candidates. No further preflight review is requested.
No pushes, PRs, services, privilege, driver or host-policy changes are authorized.
