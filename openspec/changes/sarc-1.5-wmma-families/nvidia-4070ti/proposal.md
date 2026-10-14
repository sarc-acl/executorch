# NVIDIA linear matrix preflight

Source base is `617897184529ff180af491ee0d5d7de778cbb6c6`. The only inventory
input is the reviewed snapshot `f393c3ff7ff4fc41e4651a5109a8509097c501f6`.
Archive and all per-file hashes match; all 53 inventory source hashes match
both pinned git objects and this checkout. `results/input-hashes.json` preserves
the source provenance. No historical timing accepts a candidate in this stage.

This finite stage tests the legal union of Orin and Ada linear candidates on this
member. It does not choose or split the NVIDIA family. No release selection,
new shader body, parameter search, attention work, decode tuning, promotion,
upstream port, push or PR is part of the stage. The final launch approval overrides
the historical two-small-gains stop rule. The stage remains incomplete while
its pre-build metadata question is pending in STATUS.md.

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
| Uncertainty | Fixed-seed paired bootstrap of interleaved repeats plus raw spread and 1 ms timer step; confidence intervals do not relax margins |
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
static inspection rather than runtime execution. The required runtime constant
method check remains pending; the owner question in STATUS.md resolves whether
an initial build can supply that runtime.

No Khronos validation layer exists in the current layer list. The unmodified
verify.sh does not enable or require one, so no exemption is inferred or asked
for. If an actual required gate item needs that layer, stop and report the item.

## Copied adapters and new source inspection tools

Only the needed core adapters were copied from the fused-port tools in this fresh
checkout; no historical checkout or artifact was executed, modified or measured.

| File | Adaptation |
|---|---|
| common.sh | Root, inherited lock and device index supplied through required WMMA environment variables so private launch constants stay in artifacts; assigned change/artifact roots; approved parent commit; final control profile; unique process tag. Policy probes unchanged |
| mktree.sh | Byte-identical recursive git-object export; rejects missing pinned objects and existing destination |
| podman-shim.sh | Byte-identical artifact-local Docker adapter for the unmodified build.sh; not installed globally |
| warm_file.py | Byte-identical D5 read-only model page-cache helper |
| r1304.txt | Byte-identical historical unaligned prompt, hash retained |
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
