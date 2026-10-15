# NVIDIA linear matrix preflight

Source base is `617897184529ff180af491ee0d5d7de778cbb6c6`. The only inventory
input is the reviewed snapshot `f393c3ff7ff4fc41e4651a5109a8509097c501f6`.
Archive and all per-file hashes match; all 53 inventory source hashes match
the pinned git objects and the preflight checkout. The current selector changes
append a separate dev-only profile block; the inventoried kernel sources are unchanged. `results/input-hashes.json` preserves
the source provenance. No historical timing accepts a candidate in this stage.

This finite stage tests the legal union of Orin and Ada linear candidates on this
member. It does not choose or split the NVIDIA family. No release selection,
new shader body, parameter search, attention work, decode tuning, promotion,
upstream port, push or PR is part of the stage. The final launch approval overrides
the historical two-small-gains stop rule. The stage remains incomplete until the legal matrix and its review finish.
The recovery owner decision resolves the former metadata bootstrap question.

## Thresholds fixed before measurement

| Item | Rule |
|---|---|
| Timed run validity | rc 0; 2048 prompt tokens; 0 generated tokens; positive finite tok/s; no foreign GPU workload before/during/after; at least 5 sensor samples in prefill; median clock >= calibrated floor; no thermal throttle reason |
| Clock floor | floor(0.97 times the lowest per-run median clock of valid A/A runs), one value for this device; no inherited clock floor or arbitrary ramp exclusion |
| Thermal rejection | Any active sw_thermal_slowdown or hw_thermal_slowdown sensor flag; hw_slowdown and raw event mask retained; a power-cap flag alone does not reject |
| Repeats | 5 valid runs per arm/cell, increased to 7 if any A/A cell ratio differs from 1 by >1% or either arm's (max-min)/median spread exceeds 3%; no reduction after candidate results |
| A/A acceptance | Geomean ratio within 1% of 1; every cell within 2%; explain violations before candidates |
| Noise | +-2% is noise |
| Historical reference tolerance | Same linear/attention control within 3% per cell of the final committed campaign's closing-session values; explain any difference before candidates |
| Family margin reporting | Ratio of valid-run median candidate tok/s to same-session control; geomean ratio >=0.97 and each cell ratio >=0.92, on all six cells; one member cannot establish family acceptance |
| Kernel screen | Incumbent changes only with >=3% improvement in every round, if a screen is used; no parameter search |
| Uncertainty | Paired bootstrap of interleaved repeats, seed 407014, 20,000 draws, percentile 95% interval, ratio of arm medians; report raw spread and 1 ms timer step; confidence intervals do not relax margins |
| Arithmetic gate | Unmodified verify.sh and parent-output comparison, next-token checks on timed/real/unaligned prompts; any differing arithmetic item follows D3, or D1 for claimed identical arithmetic; never weaken tolerances |
| Stop | Finish the finite reviewed legal list within the approved cap, or report an explicit blocker; device loss stops with no retry/reset |

No measurement has occurred. Clock floor, repeat calibration, baseline medians
and uncertainty remain unset. The sensor's requested 5 ms polling measured
5.133 ms median (5.103--5.231 ms range) during read-only preflight, so there is no
request to inherit the historical two-sample exception. A future sampler must
stop and reap all its own children; counts are checked after the first jobs.

## Attention control and finite candidate coverage

Both arms use `ET_VK_SARC_UNVERIFIED=1` and the committed dev-only linear selectors
on the approved base. The control is `ET_VK_SARC_DEV_PROFILE=4070ti-fused1`:
`k4070tiRefine1` supplies its unserved attention cases; `4070ti_nzf` softmax;
fused variants `fused3sb_d64_t32x32g11s32rko` and
`fused3sb_d128_t16x64g11s32rko`, with their existing K/V copy. Linear candidate
arms must preserve those exact attention choices and alter only the required
linear selection. No other profile/variant/force/disable environment is inherited.
The initial parent snapshot is from the approved base and final control profile,
not the historical parent or upstream main.

The manifest contains 11 storage-specific shader candidates: seven texture3d
IO variants and four buffer IO variants. All meet queried numeric matrix/type,
subgroup/control, declared shared-memory and workgroup requirements. This is a
partial prerequisite result, not full hardware legality or a correctness pass.
Complete feature/descriptor-format support, actual model graph layout and exact
compiled shared allocation remain checks before dispatch.

The six files serialize width-packed texture3d prefill activations/outputs and
linear group size 128. Their quantized weights fit predicted texture2d dimensions.
Buffer IO and buffer-weight manifest entries therefore remain alternate correctness
coverage; they cannot be timed as though the fixed model used buffer IO. The
reviewed LEGALITY.md explicitly requires matching actual model storage and retains
supported alternatives for correctness. Native texture3d variants require six-cell
comparison sessions, with the other scheme unchanged and all dispatched names
recorded. Shape fit and inherited fallbacks must be recorded, never counted as
candidate dispatch. There are no candidate profiles or runs yet.

Static inspection reads the actual program's zero-instruction constant methods:
context and sequence length 2560, cache/dynamic-shape flags true, BOS 128000;
all six file sizes match kit/MODELS.md. The Vulkan graph contains embedding_q4gsw
with group 32 and a packed 4-bit weight, and linear operators with group 128.
This exceeds the kit's old inferred group/embedding evidence, but is explicitly
static inspection rather than runtime execution. The runtime constant
method check was pending at preflight. It is now complete using the initial
build's CPU-only helper, following the recovery decision of 2026-10-15 00:32 UTC;
see the recovery evidence below.

No Khronos validation layer exists in the current layer list. The unmodified
verify.sh does not enable or require one, so no exemption is inferred or asked
for. If an actual required gate item needs that layer, stop and report the item.

## Copied adapters and new source inspection tools

Only the needed core adapters were copied from the fused-port tools in this fresh
checkout; no historical checkout or artifact was executed, modified or measured.

| File | Adaptation |
|---|---|
| common.sh | Root, inherited lock and device index supplied through required WMMA environment variables so private launch constants stay in artifacts; assigned change/artifact roots; approved parent commit; final control profile; unique process tag. Policy probes unchanged; loss-report template keeps private details only in the artifact marker and appends a generic notice to source STATUS.md |
| mktree.sh | Byte-identical recursive git-object export; rejects missing pinned objects and existing destination |
| podman-shim.sh | Byte-identical artifact-local Docker adapter for the unmodified build.sh; not installed globally |
| warm_file.py | Byte-identical D5 read-only model page-cache helper |
| [Historical r1304.txt](../../sarc-1.5-4070ti-fused-port/tools/r1304.txt) | Reference the existing unchanged prompt; no prompt copy is added in this run |
| query_coop.py | New CPU-only ctypes Vulkan property enumeration; never creates a device, queue, shader or command buffer |
| inspect_models.py | New read-only mmap/schema inspection of static methods and delegate graph; no inference execution, export or Python package install |

Historical e2e/gate/build drivers were inspected, but are not launch tools here.
They retain a two-sample exception, old parent/calibration, patch option and
sampler/guard limitations. No old threshold or exception becomes current policy.
A launch driver must use exact exported commits, one intermediate build tree at
a time, preserved exact binaries, HOLD before each job, the inherited device
lock, ancestry-aware ownership and during-run foreign observations. No build
runs alongside timing. Export/image/compiler/binary manifests belong in artifacts.

The initial query and static inspection evidence, sensor rows, hashes, process
and lock checks, current clock/power policy and private model paths are under
the assigned artifact root's preflight/ directory. No private infrastructure
identifier or absolute private path is put into source evidence. No candidate
launch projection is asserted before the missing launch driver/build facts exist;
a projection over 48 hours must be reported before queueing.


## Recovery build and detached queue

The recovery instruction permits run-local orchestration corrections implementing
the existing fixed protocol. No historical adapter, sarc/tools, golden, prompt,
tolerance or post-data threshold changes. Existing preflight is reused.

| File | Purpose and adaptation |
|---|---|
| tools/build-parent.sh | New single-parent driver derived from the historical build-both sequence: exact pinned base export and recursive manifest, no patch path, one build root, no traced build at this step, image pinned by digest, four build workers for available RAM, inherited lock/HOLD/foreign guard, two-hour timeout, source/toolchain/binary hashes, golden comparison on both llama and backend outputs |
| tools/job.py | New detached command wrapper writes queued/running/completed/failed JSON atomically, records command/PIDs/return code or exception, and records failed children instead of leaving a stale running state |

The initial parent-r1 uses unmodified sarc/tools/build.sh --llama from the base
export. A conservative 32 GiB free-space check precedes export; the recovery
inspection had 43 GiB free. Source files/recursive manifest, image inspection,
compiler identity/hash, binaries and both shipped golden outputs are retained
under its private build root. This is build preparation, not correctness or
performance acceptance. Exact binaries remain available for later review.
The first build unit is limited to two hours and does not launch inference;
CPU runtime metadata execution is the next prerequisite. No >48-hour matrix
projection or candidate launch is asserted before those facts exist.


## Parent build recovery and runtime metadata

The first parent-r1 queue failed at the new adapter's directory-count check,
after the unmodified build command completed successfully. The shader cache
contains duplicate SPIR-V paths beneath the canonical output directory; it is
not a second independent compilation. The adapter now selects the canonical
vulkan_compute_shaders directory explicitly. The failed queue/log/export/build
are preserved, and no main build was repeated. Both the llama and backend
canonical outputs pass the unmodified golden comparison: 53 shipped variants.
The original binary hashes also recheck successfully.

New tools/runtime_metadata.cpp uses only the parent build's core runtime and
mmap data loader, with no Vulkan/backend link. It refuses nonconstant methods
and executes the nine named scalar metadata methods on all six input models.
Every result matches the saved static metadata; context/sequence length are
2560 in each. Group size and packed embedding metadata remain facts of the
previously inspected serialized Vulkan graph, as these models expose no runtime
constant methods for them. results/runtime-metadata.json records the scalar
results, parent source and helper hashes without private paths. Exact compile
command/image, input runtime-library hashes and raw per-model outputs remain
in the private metadata artifact directory. No forward method, GPU inference,
parent snapshot or A/A has run. This resolves bootstrap only; remaining legality,
gates, dispatched-name coverage and the measured matrix are still pending.


## Format/module prerequisites and parent snapshot driver

The new tools/query_formats.py performs read-only physical-device queries for
the three formats selected in vk_api/Types.h: fp16, signed int8 and signed int32
vectors, in 2D/3D with sampled/storage and transfer usage. All six combinations
are supported. The prepacking rule in QuantizedLinear.cpp:332-414 maps 4-bit
weights to K/4 by N/8 texels; every production NK pair fits the queried limits.
Width-packed activations/outputs also fit at full context length 2560.

New tools/module_shared.py reflects the exact eleven manifest SPIR-V modules.
Their fixed shared arrays and natural alignment match the inventoried byte
counts, and fit the queried shared-memory limit. This is compiled declaration
evidence, not driver-private allocation statistics. All eleven modules are
byte-identical in the llama and microbench backend outputs. Source fingerprints
and shape/storage restrictions still apply; no candidate is accepted by these
checks. The small evidence is results/compiled-prerequisites.json.

New tools/nvml_observer.py samples read-only clock/temperature/power/utilization,
raw event reasons and compute AND graphics clients through the installed NVML
API. Definitions were checked against the installed CUDA nvml.h. It does not
start polling children. Ownership follows the observer's process ancestry and
records starttime with cached PIDs. Foreign/unattributed clients or an NVML error
block later GPU commands; device loss writes the artifact marker. Thermal masks
0x20/0x40 implement the already fixed rules. No calibration threshold changes.

New tools/observe.py wraps exact staged binaries for the unmodified gate,
records original status/arguments/environment/binary/source and samples, applies
D5 when models change or pages are uncached, forwards signals to its own runner
and reaps it. It preserves ordinary stdout/stderr for verify.sh. Wrapper/raw
binary hashes are separate in the stage manifest. Snapshot records explicitly
state NOT_A_TIMED_SESSION. rc=0/rc=1 and ancestry/cleanup checks passed; idle NVML
query median was 37 microseconds, with no loaded freshness claim yet.

New tools/parent_snapshot.py stages the preserved parent-r1 binaries and existing
unchanged prompts, freezes the observer tool copies, clears unrelated Vulkan
environment controls and selects 4070ti-fused1. It records invariant attention,
takes no new build, checks HOLD before the gate unit, and runs the unmodified
exported verify.sh --models 1b,3b,8b --schemes 4w,8da4w --pdiff, which owns the
inherited device lock. The unit has a two-hour timeout. Expected observation
count is 15 microbench and 22 runner invocations. Snapshot completion requires
all observations and no observer abort; it is a record of parent behavior, not
acceptance of unverified starting rows. No attention tiers are added because
this stage changes no attention kernel. A/A and candidate GPU jobs are separate
future units, after the captured evidence is inspected. No >48-hour queue is
launched or projected here.


## Parent snapshot, complete resource audit and finite selectors

The unmodified parent snapshot completed with all 37 observed commands. All
22 model invocations returned 0, correctness and twelve production-diff commands
returned 0, comparison lines are SAME and decode returned 0. The two linear
benchmarks returned 1 because twelve texture cases per scheme carry the parent's
unexpected_coopmat dispatch-expectation label; all 24 cases per scheme have
ok=true. This is preserved as parent behavior, not relabeled as a plain pass.
The source text summary removes rates and trailing whitespace; exact future
line comparison uses the raw artifact reference, with rates removed.
Default prefill windows contain 9--58 NVML samples, with no foreign client or
thermal observation. No sampler subprocess remains. No snapshot rate is a
five/seven-repeat interleaved baseline or candidate gain.

The earlier twelve-projection resource check omitted whole-model resources.
Following the coordinator's reminder, tools/inspect_resources.py extends the
same schema reader to all graph tensors and operator calls; tools/audit_resources.py
checks every serialized allocation and source-derived generated-resource bound.
All original models include RoPE image height 5120, and 8B includes linear image
width 3584. These fit this device's matching queries; B580's limit is not applied
to this device. The source rules cover explicit prepack, RMSNorm/embedding/linear
weights and parameters, KV caches, attention intermediates and fused-copy/int8
buffers. Constant consumers are exhaustively classified and unknown ones fail.
Per-resource format/type/usage/extent/resource-size and buffer-range limits pass;
aggregate peak VRAM is not inferred. Raw graphs, every resource check and matching
queries stay in artifacts; results/whole-model-resources.json is the small summary.
No model re-export, release resource change or exception is needed.

NvidiaWmmaMatrix.cpp defines exactly the eleven manifest candidates under
ET_VK_SARC_WMMA_PROFILE=4070ti-wmma-cNN, alongside the unchanged
ET_VK_SARC_DEV_PROFILE=4070ti-fused1. The appended registrar is last in
Overrides.cpp, so it wraps all inherited selectors. With no matrix variable
(or without unverified mode), it installs nothing. With one selected, it
registers just its kUnverified candidate and changes only matching linear ops
when the prior table/profile serves and tile/storage/layout/shape/shared limits
fit. Imported Orin routes keep the twelve projection/group/M service condition
and their large-K or 256-tile predicates. Selection of an isolated legal kernel
can supersede a formerly earlier kernel; this is the bounded cross-device
comparison, with inherited fallback recorded wherever the selected row cannot
serve. Decode, unaligned runtime exceptions, attention and other devices keep
the prior choice. The four buffer entries still need honest alternate coverage;
fixed texture models cannot establish their dispatch. No new kernel body exists.

CPU fixture checks passed each profile's intended storage and exact kernel,
row-major layout, unchanged softmax control and decode exclusion. Existing
selector tests were left unchanged; check.sh --no-build passes. The fixture
binary and logs are in artifacts/checks. Source only changes are dev registration
and run-local tools/evidence, with no shader, release, golden, prompt, tolerance
or sarc/tools change. Source correctness is not a measured candidate gate.

The existing immutable build driver accepts a matrix tag and optional exact
commit, retaining the pinned export/image/compiler/binary/golden protocol.
Parent CMake object directories were retired, with their flags/link provenance
archived and measured binaries/libraries/SPIR-V/source/logs kept and rehashed.
The next matrix-r1 build supplies both the inert control for A/A and all finite
candidate selectors. No candidate GPU session starts before A/A calibration
and historical-baseline comparison. Displayed cache fractions now cap at one
and retain raw resident/model byte counts to account for fincore page rounding;
this changes reporting, not warming or validity criteria.
