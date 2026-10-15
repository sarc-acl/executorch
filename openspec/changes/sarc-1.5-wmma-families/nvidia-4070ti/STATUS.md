# NVIDIA WMMA matrix status

Updated: 2026-10-15 04:23 UTC

Stage: FEATURE FIX APPROVED / REPLACEMENT BUILDS PREPARED.
Running next: detached feature-builds-r1, two sequential builds only, bounded to
four hours of build commands plus export/check overhead. Controller state is
artifact jobs/feature-builds-r1/state.json. No model or GPU execution is queued.
The prior build/snapshot jobs are finished. No campaign lock holder or GPU job
was active at recovery. Actual HOLD was not created or removed.

The owner decision of 2026-10-15 04:08 UTC resolves the memory-model scope
question. Supplied patch SHA256
5434b8e5c94ba8b18fb11e502577054bc3dea3af4e94263e6470113e316938a9
matches. Applied exactly the three-file Orin change from
cbb7df9b6a7a605d28192fbc2a8b62c6f3611d9c, after reviewing extension/feature
query and device-chain handling. No selector or shader was imported from Orin.
Release-zone files are Adapter.cpp, Device.cpp and Device.h only. The fix queries
the advertised VK_KHR_vulkan_memory_model extension, requests it only with
supported vulkanMemoryModel, and enables only that feature. Unsupported paths
keep the feature false/absent; DeviceScope and visibility-chain flags are zero.
Source feature enablement is committed as 8ebe66c2b. Runtime enabled-chain proof
and all replacement correctness/performance evidence remain pending.

Control source is 5f02679b9a20d0b43ed44ea42cb1d779e1f269ac: original base plus
exactly the approved three files, retained by a private local git ref. Candidate
source is this branch's next committed orchestration state, with the same runtime
source bytes plus the existing dev-only selectors. Planned immutable tags are
parent-mm-r1 and matrix-mm-r1. results/memory-model-fix.json records provenance.
Inventory archive and all 18 file hashes reconfirmed unchanged before work.

One intermediate build tree at a time: the queue preserves configuration/link
provenance and object hashes, then retires only superseded CMakeFiles directories
from matrix-r1. After the new control build it does the same before building the
selector arm. Binaries, installed libraries/headers, SPIR-V, source exports,
logs and failed job records stay retained; exact binaries are rehashed before
and after retirement. Initial free space is about 36 GiB; the existing 32 GiB
pre-build check remains. No threshold or tool/golden edit is used to make space.
Each new build gets its recursive export, pinned image/compiler and binary hashes,
both unmodified golden checks, eleven-module shared-declaration audit and
byte comparison of every SPIR-V module against the retained parent.

Unmodified check.sh --no-build reports its expected zone failures for Device.cpp
and Device.h, which are covered exactly by the new owner decision; no tool or
HOOKS change masks them. Selector results and raw check output are retained
in artifact checks/check-memory-model-no-build.txt. This is not a plain PASS.

The old parent snapshot remains raw execution observations and is NOT a
conformant gate or enabled-feature proof. results/memory-model-prerequisite.json
records that old-build finding; results/memory-model-fix.json records its source
remedy. Following successful builds, execute the CPU metadata helper against the
new control runtime, obtain exact enabled-feature evidence, replacement snapshots
and A/A, then explain historical-baseline differences before candidate work.
There are no timing medians, calibrated floor, candidate passes or margins yet.
No owner question is pending. No push, PR, new shader, service or policy change.

## Retained preparation and historical observations

matrix-r1 completed rc=0 from exact exported commit
 d30add8be93f5fe814a2a6eaa4210f00d3f05a9c, with 30 pinned submodules and the
pinned image/compiler. Both unmodified golden checks pass for 53 shipped
variants. All 2091 SPIR-V files per llama/backend component are byte-identical
to the retained parent. Exact binary and library hashes are recorded in
build/matrix-r1/binaries.sha256. Source summary: results/matrix-build.json.
No source patch or new body was used, and no build is active now.

The previously prepared, unlaunched control/A/A driver names the retained
binaries; it will need fresh approved build tags after the feature fix.
Its unchanged-profile gate compares the raw outputs with rates removed. Both
A/A arms use
ET_VK_SARC_UNVERIFIED=1 and ET_VK_SARC_DEV_PROFILE=4070ti-fused1 and with no
WMMA profile. Each logical repeat interleaves arms, parent first on odd repeats
and topic first on even ones. It obtains five valid repeats per arm/cell, then
extends ALL six cells to seven if the pre-registered ratio/spread condition asks.
The one device-wide floor is floor(0.97*lowest valid A/A run median clock).
Limits are not widened. Raw calibration rows explicitly identify their purpose
and an unset clock floor; the final calibration file derives that floor.

A/A resumes from complete CSV row keys; invalid and interrupted attempts remain
recorded, with incomplete artifacts moved under superseded/. Replacement is
bounded to four attempts per logical arm/repeat; observer failures stop for
diagnosis. HOLD releases the inherited lock before acknowledgement and checks
before each resumable run. No candidate runs alongside the gate or calibration.
Cooling before the first timed row: initial <=50C/300s, then recorded idle+5C
between runs/120s; three 5-second probes with no further fall also end cooling.
The control gate has a two-hour cap, A/A a four-hour cap, total <=6h rather than
an unbounded queue. Actual data determines completion within those caps.

The new timed-run evaluator rejects fewer than five samples, failed runners,
missing stats, thermal flags, foreign clients and clocks below a supplied floor.
Nine raw-log/synthetic-invalid checks pass; power-cap alone remains allowed.
The A/A HOLD path was tested with separate HOLD-AA-TEST markers: acknowledged,
released/reacquired the lock and cleaned test files. Actual HOLD was untouched.
Evidence: artifacts checks/evaluate-run/ and checks/hold-aa.json. These checks
are protocol tests, not A/A or candidate performance results.

Parent snapshot s0-parent-verify completed rc=0 with all 37 observations:
22 model runners returned 0; correctness returned 0; all twelve production-diff
commands returned 0 and ALL PASSED; next-token comparison lines are SAME; decode
commands returned 0. Both linear benchmark commands returned 1. Their 24 cases
per scheme all have ok=true, with twelve unexpected_coopmat dispatch labels
and twelve confirmed labels each. This is parent dispatch-expectation behavior,
not a plain verification pass or acceptance of unverified starting rows.
Raw reference: artifact stage/s0-parent-verify/verify.out and verify/ logs/JSON.
Source summaries: results/parent-snapshot.json and parent-verify-normalized.txt.

Observer audit: all 22 runner observations have zero foreign-client and zero
thermal samples, fully resident model pages after warming, and no runner crash.
Default timed-prompt prefill windows contain 11/9/27/23/58/50 sensor samples
(1B 4w/8da4w, 3B 4w/8da4w, 8B 4w/8da4w). No sampler process remained.
These are snapshot observations, not an interleaved timed session. No clock
floor, calibrated repeat count, baseline median, uncertainty or gain exists yet.
Raw observation-audit.json retains every per-invocation result. Original cache
fractions can slightly exceed one because fincore includes the final page;
future observer rows cap the displayed fraction at one and retain raw byte counts.

Whole-model resource audit completed after the coordinator's B580 reminder.
That finding is not used as a verdict for this device. All six original PTEs
contain RoPE prepack images of height 5120; the 8B images include width 3584.
The matching local 3D format queries support them. The audit checks ALL serialized
nonconstant tensors, constant-consuming op classes and source-derived generated
weight, RMSNorm, cache, attention, fused-copy and int8 buffer bounds against the
matching format/type/usage and buffer limits. No allocation-limit exception or
model re-export is used. This checks per-resource limits, not aggregate peak
memory. Evidence: artifact preflight/models-resources-raw.json and
whole-model-resources.json; small summary results/whole-model-resources.json.
The earlier projection-only image check was incomplete; parent execution is
not substituted for this matching-query evidence. No further GPU work preceded
this audit after the reminder.

Reviewed inventory: f393c3ff7ff4fc41e4651a5109a8509097c501f6.
Archive SHA256: 65d1e3ca0eb7e9dffa8e9fbbb1a100bf2cf609868720b9053de67ff8bf5247d8.
NVIDIA manifest SHA256: ef5d13471421898c8515e9a8356caff64a20aa97432175cbcab7bf3a5cd748b1.
The archive and all 18 per-file input hashes were confirmed before recovery work.
The immutable input and its source references remain the candidate inventory.
All eleven fixed compiled modules match source shared-array totals/natural
alignment and are byte-identical between parent llama and backend outputs.
Both unmodified golden checks pass for all 53 shipped variants. Matrix/type,
subgroup/workgroup, fp16/int8/storage features, image formats and production
resource prerequisites are recorded privately. Candidate correctness remains
pending, with seven native texture3d and four alternate buffer entries.

CPU-only runtime metadata already completed: all nine scalar methods on each of
six models match static evidence, including context/sequence length 2560.
Group-128 linear and packed 4-bit/group-32 embedding facts come from the actual
serialized graph; those fields have no runtime constant methods in these PTEs.
The owner resolved the initial metadata bootstrap question on 2026-10-15 00:32.
Both metadata and feature-scope questions are resolved. Replacement builds/gates remain pending; no repeat preflight review is requested.

New selectors: impl/sarc_dev/NvidiaWmmaMatrix.cpp plus one appended 4070ti block
in Overrides.cpp. ET_VK_SARC_WMMA_PROFILE=4070ti-wmma-c01 through c11 requires
unverified mode and this device's pre-existing table service; each candidate
retains its recorded storage, tile, shape predicate and row-major layout. Only
the matching linear op changes. Before choices, attention controls and other
configurations stay inherited. No new shader body or family selection exists.
CPU tests of all eleven profiles passed exact kernel/storage/layout, unchanged
softmax control and decode exclusion. The original test_sarc_select is unchanged;
check.sh --no-build PASS (1240 release checks, 1921 dev checks). Evidence and the
CPU fixture binary are retained under artifact checks/.

Parent object directories were retired after preserving CMake flags/link
provenance. Installed libraries/headers, exact measured binaries, all SPIR-V,
raw logs and source export remain, and parent hashes rechecked after cleanup.
Original failed parent queue remains failed; its recovered build is reused.
The original builds remain preserved. The owner-approved replacement builds
are described above; no measured binary is deleted.

Next after control/A/A: inspect every raw validity decision and recompute the
calibration. Explain any historical tuned-reference cell outside 3% before
candidate GPU work. Finite candidate gates/sessions, alternate-storage coverage,
uncertainty, margins and independent matrix review remain incomplete. No family-
wide acceptance or candidate pass exists. Replacement evidence is pending; no owner question remains.

Scope remains finite linear matrix only. No attention/decode/fp tuning, search,
new bodies, release selection, promotion, push, PR, service or policy changes.
Private infrastructure/model paths and hardware policy stay in artifacts.

## Coordinator hold

The prior HOLD dry run remains valid; actual HOLD was not created or removed.
Prepared units are control verify.sh (<=2h), then individual resumable A/A
runs under the inherited lock after the feature-enabled replacement builds. The controller observes a JSON
state file. No >48-hour queue is launched, and reviewer performs no build/GPU work.
