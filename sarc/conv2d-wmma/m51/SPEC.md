# conv2d-pw-coopmat on Samsung M51 (Xclipse): Specification delta

## Purpose

Adds the requirements that hold the FP16 1×1 cooperative-matrix convolution kernel (`conv2d_pw_coopmat`) to the
Samsung M51 GPU, where nothing has run yet. Every requirement of `../SPEC.md` still applies; this file adds to it and
tightens it, and changes none of it for other devices. Decisions were taken with the device owner on 2026-11-19.

Background the requirements rest on (from the earlier M51 cooperative-matrix linear-kernel work on this fork):
- The device's only cooperative-matrix shape is 16×16×16 at subgroup scope. Its default subgroup size is 64, and 32 is
  available through required-size control.
- The device's FP16 matrix throughput is about equal to its FP16 vector-FMA throughput. A cooperative-matrix kernel can
  therefore win only through operand reuse, not through a higher peak.
- A multi-pass ("banded") texture3d drain gave wrong results when many workgroups ran at once and exact results with
  one workgroup. The root cause was not found. A one-pass full-tile drain fixed it.
- A `coopMatMulAdd` loop whose trip count comes from a uniform buffer crashed pipeline creation on one driver. A
  `coopMatStore` whose offset or stride comes from a uniform buffer was miscompiled. Specialization constants fixed both.
- FP16 accumulation failed long-K precision.
- More than 64 KiB of shared memory per workgroup hung the device.
- `barrier()` alone did not order shared-memory stores before `coopMatLoad`.
- Microbenchmark rankings and whole-model rankings often disagreed.
- Floating clocks shrank measured speedups.

Scope by convolution kernel. The order follows the desktop FP16 survey of the ten CI models, where convolution is
about 87% of GPU shader time. The M51 workload baseline re-ranks it.

| # | Kernel | Stage |
|---|---|---|
| 1 | 1×1 `conv2d_pw_coopmat` against `conv2d_pw_tiled` | 1 (this campaign run) |
| 2 | k×k: the GEMM stage of im2col on cooperative matrices | 2, after the desktop push |
| 3 | Direct k×k (`conv2d_2x2x8_border`) routed through im2col and WMMA | 2, if pushed |
| 4 | Dilated 3×3 | 2, if pushed |
| 5 | INT8 1×1 (`q8ta_conv2d_pw`), 1D convolution, INT8 quantize/dequantize | out of scope |

## ADDED Requirements

### Requirement: Fixed measurement environment
Every measured or validated result SHALL come from one named board. The driver build SHALL be identified by checksum at
the start of each session and after every reboot, recovery or driver flash, and it SHALL equal the session's recorded
canonical driver. GPU and memory clocks SHALL be pinned and read back. The vendor profiler configuration SHALL be set
aside during timing. The board SHALL be cooled before each session and between runs. These identifiers stay in the
local record and SHALL NOT appear in committed files.

#### Scenario: Driver changes during a session
- **WHEN** the driver checksum differs from the session's canonical driver at any check
- **THEN** the session stops, the numbers since the last good check are discarded, and work resumes only after the canonical driver is restored and the correctness gate is re-run

#### Scenario: Board crash
- **WHEN** the board drops to its bootloader or a run dies
- **THEN** the run's numbers are discarded, the board is recovered once, the environment checks are repeated, and a second crash in the same session stops the campaign as blocked

### Requirement: M51 capability record gates the campaign
Before any kernel runs, the `device_coopmat_properties` check SHALL record on the campaign's board:
- every enumerated cooperative-matrix property (M, N, K, A, B, C and result types, scope);
- the Vulkan API version;
- `maintenance4` support;
- the subgroup size range and required-size support for compute;
- the shared-memory limit.

The campaign SHALL stop and report when the device enumerates no 16×16×16 FP16 property with an FP16 or FP32 result,
when it does not support required subgroup size 32 for compute, or when its Vulkan version is below 1.3. The committed
summary SHALL contain only the shapes, types, scope and subgroup sizes.

#### Scenario: Expected M51 shape present
- **WHEN** the device enumerates 16×16×16 with FP16 A/B and FP32 C/result at subgroup scope, and supports required subgroup size 32
- **THEN** the AMD-family variants and the M51 family are admissible and the campaign continues

#### Scenario: No usable shape
- **WHEN** no enumerated property matches 16×16×16 FP16
- **THEN** the campaign stops before any kernel is dispatched and reports the enumerated list

### Requirement: Static safety before the first dispatch
Before a variant is first dispatched on the M51, it SHALL be shown statically that:
- its shared memory, including staging, drain and any split-K scratch, is at most 64 KiB and within the device limit;
- any new or changed shader code has been read for unsynchronized shared-memory access, and every shared-memory store
  read by `coopMatLoad` is ordered by `memoryBarrierShared()` followed by `barrier()`.

#### Scenario: Variant over the shared-memory cap
- **WHEN** a variant's computed shared-memory size exceeds 64 KiB
- **THEN** it is never dispatched on the M51 and is listed as excluded with its size

### Requirement: Loop bounds by specialization constant in the M51 family
An M51-family variant built with the option `KSPEC` SHALL take its K-loop trip count, and every other value that bounds a
cooperative-matrix loop or sets a cooperative-matrix load or store offset or stride, from specialization constants
fixed at pipeline creation, not from a uniform buffer. The option SHALL default off. With it off, the SPIR-V of every
existing variant SHALL be unchanged.

#### Scenario: Existing variants unchanged
- **WHEN** the shaders are generated with and without the `KSPEC` change applied
- **THEN** the SPIR-V of every variant that does not set `KSPEC` is byte-identical

#### Scenario: Shapes that resize
- **WHEN** a graph that uses a `KSPEC` variant is resized so that K changes
- **THEN** the pipeline is re-created with the new constant, or the variant is reported ineligible for the new shape; it never runs with a stale trip count

#### Scenario: Uniform-buffer trip count probed once
- **WHEN** the existing AMD-family variants, whose trip count comes from a uniform buffer, are first created on the canonical driver
- **THEN** whether pipeline creation succeeds is recorded, and the result does not change the `KSPEC` requirement for the M51 family

### Requirement: One-pass drain option
A variant with the option `XP` SHALL convert FP32 accumulators to FP16 before staging them in shared memory. It SHALL
drain the whole workgroup tile in one pass, with one barrier before and one after the staging. The epilogue (bias,
scale, activation, residual) SHALL still run in FP32 on values read back from the FP16 staging. This option trades the
"fp32 until the output store" guarantee of `../SPEC.md` for a single drain pass, so an `XP` variant SHALL meet the
reference-error rule of the correctness gate instead of the exact FP32-cancellation scenarios. Those scenarios SHALL be
run and their outcome recorded. The option SHALL default off.

#### Scenario: 128×128 tile in one pass
- **WHEN** the `m51_t128x128` FP32-accumulation variant with `XP` runs
- **THEN** its drain uses one pass, its shared memory is at most 64 KiB, and the generated code shows one store pass of the accumulators

#### Scenario: Cancellation cases
- **WHEN** an `XP` variant runs the fp32-accumulation cancellation cases of `../SPEC.md`
- **THEN** the results are recorded against the expected values, and the variant's error against the FP32 CPU reference on the gate shapes is not larger than the parent's

### Requirement: M51 variant family
A family `m51` SHALL be added as yaml blocks and descriptor rows only:
- MMA 16×16×16, subgroup 32, FP16 operands;
- the parent `conv2d_pw_coopmat_m51_t128x128_f32_kspec_xp_half`: workgroup tile 128×128, K step 16, subgroup grid
  2×2, FP32 accumulation, `KSPEC`, `XP`.

Every other `m51` variant SHALL differ from its named parent by exactly one option or one parameter. Variant names SHALL
follow `../SPEC.md`, with the option suffixes `_kspec` and `_xp`.

#### Scenario: Registry consistent
- **WHEN** the shader registry and the descriptor table are built
- **THEN** every `m51` yaml block has exactly one matching descriptor and the registry check reports no naming problem

### Requirement: M51 correctness gate
A variant SHALL be timed on the M51 only after it passes, in 10 consecutive repetitions:
- `pw_coopmat_correctness.json` with `--validation_only`;
- a many-workgroup collection: every timed shape of the campaign, including the long-K GEMM-equivalent shapes, with
  bias and with a fused residual;
- the same collection run with the dispatch reduced to a few workgroups.

Any mismatch in any repetition fails the variant. FP16-accumulation variants SHALL be run and recorded against the
fp16 limits collection. They SHALL be timed only if they pass every case, including long K, and are then reported
separately.

#### Scenario: Concurrent-workgroup defect
- **WHEN** a variant is exact with few workgroups but wrong in the full grid
- **THEN** it fails the gate, is never timed, and the failure is recorded with its mismatch coordinates

### Requirement: M51 timing protocol
Thresholds SHALL be committed (`THRESHOLDS.md`) before the first timed run. Each session SHALL open with an A/A run of the
parent: the A/A geometric mean must be within 1% of 1 and every cell within ±2%. Otherwise the session is repeated or
the band widened with a recorded reason. A result SHALL be the median of 5 valid runs:
- arms interleaved, alternating which runs first;
- GPU time summed over all dispatches of the route;
- the same binary for every arm;
- invalid runs kept in the record with their reason.

Committed results SHALL be ratios only.

#### Scenario: Run outside the pinned clock
- **WHEN** a run's clock readback shows time at any frequency other than the pinned one
- **THEN** the run is invalid and replaced

### Requirement: Benchmark shape set on the M51
Timing SHALL cover, in one session per comparison:
- `pw_coopmat_mobilenet.json` and `pw_coopmat_resnet50.json`;
- the GEMM-equivalent shapes (49,512,4608), (196,256,2304), (784,128,1152), (289,224,1344), (25,384,4032);
- once the workload baseline exists, the ten 1×1 shapes with the largest share of M51 GPU time not already covered.

The arms are `auto`, `pw_tiled_4x8` and every variant that passes the gate.

### Requirement: Locate before optimizing
After the parent's baseline, and before any further candidate is built, the campaign SHALL record where the parent's time
goes on the M51:
- per-pipeline register, shared-memory and scratch use and the generated instructions, from an inspection build kept
  on a separate branch that is never merged into the deliverable;
- in-kernel phase timing with the shader clock, split into fetch, staging, synchronization, multiply and drain.

Vendor performance-counter capture is allowed in sessions that are not timed. Thread-trace capture (SQTT/RGP) SHALL NOT
be used.

### Requirement: Candidate discipline and stop rule
Each candidate SHALL change one option or parameter against a named parent, chosen from the located bottleneck. The
menu, in no fixed order:
- N-major texel-wise weight staging;
- register-pressure and occupancy retiles;
- the Xe2 techniques `_alc`, thread-rich small tile and `_sk`;
- tile parameter changes.

A candidate SHALL be adopted only if it passes the gate and every cell it changes gains beyond the noise band. The
campaign SHALL stop when two consecutive gated candidates each gain under 2% in geometric mean.

### Requirement: M51 workload baseline
The ten ExecuTorch Vulkan CI models (mv2, mv3, resnet18, resnet50, ic3, ic4, edsr, dl3, w2l, densenet161) SHALL be
exported in FP16 and run on the M51 with ETDump: 12 executions, the first discarded. They SHALL be aggregated with
`../survey-tools/aggregate.py`. Committed output SHALL be time shares and rankings by shader and shape, never times. FP32
and INT8 are out of scope for this change.

### Requirement: M51 workload share by convolution family
The workload baseline SHALL also report the share of M51 GPU time per convolution family: 1×1, im2col GEMM, direct k×k,
dilated, 1D, depthwise and other. It SHALL report the share per kind × K as well. These shares SHALL rank the stage-2
kernels before stage 2 starts. Committed output SHALL be shares and rankings only.

### Requirement: End-to-end projection
For resnet18, resnet50, mv2, mv3, ic3, ic4, densenet161 and dl3, the report SHALL project the GPU-time reduction.
- Apply each measured M51 kernel ratio to each matching layer shape in the ETDump.
- Leave every other layer unchanged.

The projection SHALL be labelled "projection, lower bound". It SHALL be recomputable from the local ETDump data and
`runs.csv`, and it SHALL NOT be presented as a model run.

#### Scenario: Layer shape not measured
- **WHEN** a model layer's shape has no measured M51 ratio
- **THEN** that layer keeps its ETDump time in the projection

### Requirement: Staged scope
Stage 1 SHALL cover the 1×1 kernel, the workload baseline and the projection.

Stage 2 SHALL cover the k×k kernels (im2col GEMM, direct k×k through im2col, dilated 3×3) and measured end-to-end runs.
- It SHALL start only after the desktop side has pushed a cooperative-matrix route with im2col input to
  `origin/topic/conv2d-wmma`.
- The M51 branch SHALL be rebased onto that push, and the stage-1 parent re-gated, before stage 2 starts.
- Its kernels SHALL meet every requirement of this file: static safety, `KSPEC`, `XP`, gate, timing, locate,
  candidate discipline and publication.
- Its thresholds SHALL be committed before its first timed run.

Measured end-to-end runs SHALL use the desktop routing switch and interleaved switch-off/switch-on runs on the same
binary.

#### Scenario: Desktop route not pushed
- **WHEN** the desktop k×k route is not on `origin/topic/conv2d-wmma`
- **THEN** stage 2 does not start, and the M51 side does not implement its own version

### Requirement: M51 verdict
The report SHALL give, per shape:
- the ratio of `pw_tiled_4x8` GPU time to the best passing variant's;
- the variant's name;
- the noise band.

It SHALL also give, per technique, the gain against its parent, and say what the workload baseline indicates is worth
doing next on the M51. The campaign is a success when the geometric mean over the timed shape set is at least 1.15×
(`THRESHOLDS.md`). Otherwise it ends by the stop rule and reports the result as measured.

### Requirement: Publication of M51 results
Committed files and commit messages SHALL contain no absolute M51 figures (times, rates, register counts, clocks) and no
board, host, driver-build or system identifiers. Before every commit, the diff against `origin/topic/conv2d-wmma` and the
commit messages SHALL be checked for them. Nothing SHALL be pushed without the device owner's order.
