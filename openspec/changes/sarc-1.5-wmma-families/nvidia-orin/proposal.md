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


## Successful-build recovery, 2026-10-15

The first queue completed compilation and linking, then failed its own
postprocessing guard: `rglob("*.spv")` included the canonical output and its
`shader_cache` directory. The original failed JSON state, raw log, exported
source, compiler and manifest remain preserved. Recovered the completed build
without rebuilding. Future queues locate `build/vulkan_compute_shaders/spv.cpp`
and top-level modules directly. Separate recovery provenance binds the original
manifest/failed-state hashes, six bundle hashes and 2091 top-level module hashes.

The unmodified shipped golden comparison passes all 53 variants with the
recorded pinned compiler. The CPU-only helper's compiled target hash and copied
task hash match; its linker line/ELF dependencies exclude Vulkan. All nine
constant methods equal serialized metadata on all six models, including context
and sequence length 2560. The already verified tensor metadata establishes
linear group 128 and embedding q4/group 32. No model GPU execution occurred.

`tools/module_requirements.py` disassembles every exact manifest module, records
capabilities, image/matrix types and workgroup variable capacities, and runs
`spirv-val --target-env vulkan1.1`. All 11 validators return zero. All shared-type
capacities equal source inventory totals and fit the queried limit. Unsupported
shared types fail rather than receiving guessed sizes. Raw disassemblies and
validation are private artifacts. Final capability/format/shape matching,
reproducible candidate selectors and conforming measurement adapters remain
pending; no gate/timing/matrix-completion claim follows from these checks.

The successful-build recovery is a new evidence checkpoint. It does not reopen
or repeat the already completed source-only preflight review, request another
adapter-scope decision, or extend a historical compiler exemption.


## Finite selectors and enabled-feature prerequisite

The 11 reviewed entries map one-to-one to `orin-fused-wmma-01` through `-11` in
`results/candidate-profiles.json`. A single elected row registers only under its
own profile and with unverified rows enabled. The outer selection callback
changes only that entry's scheme/storage when its original shape rule, alignment,
parent projection boundary and complete shared-memory requirement fit. Buffer
entries support correctness coverage and do not masquerade as texture timings.
The profiles inherit `kOrinRefine5` for all unaffected choices. Both arms set
`ET_VK_SARC_ORIN_SDPA_FUSED` to the exact pair selected by `orin-fused1`; the
existing prefix behavior retains `orin_g64`. No attention shader, arithmetic,
fused-node implementation, release file or shared shader YAML changed.

New logic lives in `impl/sarc_dev/OrinWmmaLinear.cpp`. `Overrides.cpp` adds only
labelled `orin-wmma` blocks for profile aliases and a final chain installer;
existing lines are unchanged. Install order is explicit within the existing
selector translation unit. New rows are unverified and inert outside the own
profiles. CPU fixtures cover 9,548 cases across all profiles, three storage
combinations, decode and short/unaligned/maximum-context dimensions, and SDPA.
Unchanged choices and attention/softmax equal control. The unmodified no-build
check passes its zone, twin, release-selection and default dev-selection checks.
These checks do not replace exact new-build gates/dispatch evidence.

`calibrate_aa.py` implements the registered fully valid per-run minimum rule,
with exact five-row arm/cell coverage and fractional floor preserved. Its row
validator requires complete device/unit/driver/build/profile/candidate/session/
cell/log/sensor provenance and rejects missing validity records. `sampler.py`
reads sysfs and process ancestry directly, starts no children, checks all thermal
cap records, samples known foreign workloads, preserves device-loss markers and
exits cleanly on SIGTERM. `bash_s.py` sends scripts through bash-s with shlex-quoted
arguments. Synthetic tests and literal-argument target transport pass. No sampled
GPU workload or real calibration data is asserted. The first test assertion
incorrectly expected only rc rejection for a row also below the resulting floor;
corrected the assertion to expect both reasons, retaining the initial test path.
The production formula/thresholds did not change.

Read-only exact image-format queries and newly expanded serialized-resource
inspection check the B580 concern independently for Orin. Here optimal sampled/
storage/transfer 3D limits are 16,384 on every axis. All explicitly serialized
nonconstant image extents and prepack outputs fit, including RoPE height 5,120
and 8B packed width 3,584. Runtime-created resources/default scalar records remain
explicit audit items; no full model-allocation or measured correctness claim is
made. Original models remain read-only.

The enabled-feature join found a real prerequisite before any model GPU job:
all exact manifest shaders require `VulkanMemoryModel`, and the pinned runtime
never enables it at logical-device creation. Physical support, standalone
validation and golden byte identity do not establish enabled-feature legality.
`results/runtime-feature-blocker.md` gives source/specification references;
STATUS.md records the owner question once. No release-zone fix, compiler or limit
waiver is inferred. Independent source/helper/resource work above is recorded;
park without another preflight review, new build or GPU measurement until the
owner supplies a dated resolution. The matrix remains incomplete.


## Authorized runtime prerequisite, 2026-10-15

The owner's 03:57 UTC instruction resolves the feature question above and
authorizes only the necessary release-zone query/device-create change. See
`results/runtime-feature-enablement.md` for the exact three files, specification
references and 16-case CPU test evidence. The unmodified no-build check's two
zone failures are disclosed under that explicit decision. No gate, shader,
compiler, golden or threshold relaxation is used. Both arms will use the new
exact exported source commit, preserving the original base/build provenance.
No GPU correctness or timing evidence exists yet; the matrix remains incomplete.


## Feature-enabled exact build completed

Build `orin-wmma-mm-20261015-1` from `cbb7df9b6a` completed, with the
unmodified golden PASS for all 53 shipped variants. Post-build recorded-file
hashes match, and all 2,091 canonical shader modules equal the preserved
original build byte for byte. The prior 11 module requirement records are
reused by exact hash. `results/feature-enabled-build.json` records the small
summary and private evidence locations. Source/runtime review remains required.
No physical-target feature execution, GPU gate, A/A or timing has run; the
matrix remains incomplete. The original failed job/build records are unchanged.
