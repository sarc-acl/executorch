# Orin bounded WMMA matrix

Actor: Codex. Source base `617897184529ff180af491ee0d5d7de778cbb6c6`.
Input inventory `f393c3ff7ff4fc41e4651a5109a8509097c501f6`; NVIDIA manifest
SHA256 `ef5d13471421898c8515e9a8356caff64a20aa97432175cbcab7bf3a5cd748b1`.
The archive and every supplied file/source hash were checked. No mutable inventory
or historical measurement is substituted for this reviewed snapshot.

## Stage boundaries

Complete the legal finite matrix, subject to prerequisites and the approved cap;
otherwise report an explicit blocker. No promotion, release selection, new bodies,
parameter search, push, PR, service changes or hardware policy changes.
All attention controls remain the original `orin-fused1` environment.
No measured profile or correctness acceptance is claimed by preflight.

## Threshold registration before measurements

Five valid runs per arm/cell; odd pairs control first, even pairs candidate first.
At most three additional pairs replace invalid rows; retain every invalid row.
A valid timed row has rc 0, finite positive rate, 2048 prompt tokens, zero generated
tokens, at least five clock samples inside prefill, no foreign workload before,
during or after, median clock at least the floor and no thermal throttle.
Clock floor: `0.97 * lowest valid per-run median clock in A/A`, device-wide.
Keep fractional MHz precision; do not round the threshold downward.
A/A validity before floor calibration uses all other criteria; apply the resulting
floor back to A/A. Never use lowest per-cell median as a substitute.
Noise band: +/-2%; baseline discrepancy beyond 3% per cell must be explained before
candidates. Kernel screen margin 3% every round if any screen is needed; no search.
Family limits are unchanged: for each cell define elapsed-time ratio as control
median tok/s divided by candidate median tok/s. Require its geomean <=1.03 across
all six cells and every cell <=1.08. Also report throughput ratios from the same
medians. This definition is conservative and does not widen either limit. Report
uncertainty.
Memory and swap counters are recorded for every 8B process; one process at a time.
Models enter page cache before a cell and whenever the model changes, both arms.
The unmodified full correctness gate still checks decode. Exact-build shipped
SPIR-V must pass; no historical compiler exemption carries over.
Reference-error and near-tie decisions apply only under their full criteria.

## Current preflight

The reviewed NVIDIA manifest has 11 source-distinct storage-specific entries.
Current queried MMA types, subgroup requirement, declared shared-memory sizes and
workgroup invocation counts match all 11. Exact compiled allocation and complete
format/module capability checks remain pending; this is not final legality.
Seven texture3d entries match projection IO in all six models. Four buffer entries
are retained for correctness coverage, not silently measured through texture IO.

Direct read-only flatbuffer inspection confirms 2560 context, group size 128 for
all delegated linears, q4 embedding op/group 32 and texture3d width-packed projection
IO. This is stronger evidence than the kit's historical size-based inference;
constant-method execution on this run's exported binary remains pending.

The current cross image is present and its compiler reports shaderc v2026.1.
No exact parent binary or parent golden result exists yet; no exemption is claimed.
Private device/host/model paths and raw query outputs stay outside the source tree.

## Adapter review

Copied adapters have not been changed or launched. The following existing behavior
cannot be reconciled by changing host constants alone:

| Source | Existing behavior | Required behavior |
|---|---|---|
| `sarc-1.5-orin-fused-port/tools/calibrate_clock.py:18,25,30` | minimum of per-cell medians; accepts two samples and does not filter all validity criteria | lowest valid per-run median; at least five samples; full A/A validity |
| `tools/common.sh:79,106-107` | sampler/watch stop kills only the outer shell; no child cleanup trap | samplers stop their own children and process counts are checked early |
| `tools/common.sh:31`; `tools/deploy.sh:16,25,27,29,33` | ssh executes command strings directly | every ssh script uses `bash -s` |
| `tools/e2e5.sh:85,102` | rows identify parent/cand tags but omit per-row commit, binary, profile, candidate and driver | every row carries the task's complete provenance |

Semantic corrections to copied routines exceed the host-constant-only rule.
The independent reviewer distinguishes supplemental orchestration: separate A/A
analysis, a sampler with explicit child cleanup, bash-s deployment commands and a
deterministic provenance join can implement task requirements without modifying
copied routines. The task does not expressly forbid that route. Do not describe
such new logic as constant adaptation. The owner recovery instruction of
2026-10-15 00:32 UTC resolves this question: necessary run-local orchestration
corrections are authorized. Original historical tools, gates,
prompts, goldens and fixed thresholds remain untouched.


## Recovery build checkpoint

The adapter-scope question was resolved by the recovery instruction at the end of
CAMPAIGN.md. Preparation already reviewed is reused. Inventory archive, all 18
snapshot hashes and 53 pinned source-object hashes were reconfirmed on recovery.
No existing local build or GPU queue was found; no known target workload or held
GPU lock was observed. Deployment has not occurred.

The first detached queue exports the exact unmodified base recursively from this
clone's object stores and cross-builds it under the shared desktop build lock.
It uses the approved cross image and extracts the golden's compiler and its two
SPIRV-Tools libraries from the existing workstation build image into this run's
artifacts. The copied recipe already supports an explicit compiler. No historical
compiler exemption is used: a nonzero unmodified golden check fails this queue.
Compiler/image IDs, compiler bytes/version, recursive source hashes, submodule
pins, recipe hashes, binary hashes and all compiled shader hashes are persisted
in the private build manifest. The build tag is immutable. There is no local
patch, target build, deployment or GPU execution in this queue.

| Run-local file | Adaptation and purpose |
|---|---|
| `tools/mktree.sh` | Byte-identical copy of the historical recursive commit exporter |
| `tools/common.sh` | Source-repository constant comes from private queue configuration; build-only layer |
| `tools/jetson-cross/{build.sh,aarch64.cmake,smoke.cpp,vk-caps.cpp}` | Byte-identical copies of the existing cross recipe and property/smoke helpers |
| `tools/parent_build.py` | Detached status/error handling, HOLD checks, shared desktop lock, exact export, pinned compiler, provenance and mandatory golden result; replaces historical informative-only build orchestration |
| `tools/jetson-cross/runtime_constants/` | CPU-only helper linked to the exported parent runtime; executes only listed constant methods and never loads `forward` or links the Vulkan backend |
| `tools/HOLD.md` | Describes the build job boundary and pending target-queue requirement |

Private paths and host constants are supplied in artifact configuration, outside
the source tree. The queue's JSON status is queued/running/completed/failed,
including failure before build and interruption. Syntax checks pass. A deliberately
invalid commit produced terminal `failed`, nonzero exit and zero GPU jobs; copied
export/recipe files compare byte-identically with their originals.

Still required after a successful parent build: execute the helper on all six
models and compare constants with static metadata/kit; inspect exact compiled
module capabilities and allocations; implement the authorized sampler cleanup,
bash-s deployment, complete timed-row provenance and per-run A/A calibration;
commit reproducible candidate selectors; parent snapshot, baseline/A/A, full gates,
finite legal matrix, dispatched names, medians/margins and independent review.
No parent/candidate gate, golden PASS, timing result or stage completion is claimed
before the queue produces that evidence. Attention remains `orin-fused1`.
