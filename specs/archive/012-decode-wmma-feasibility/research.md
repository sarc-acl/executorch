# Research: Decode Shader WMMA Acceleration

## Decision 1: Device peak specs -- from published sources, not assumed

**Finding**: This device (`rocky-ryzen` MiniPC) is confirmed via `vulkaninfo`
and `/proc/cpuinfo` to be an AMD Ryzen 9 7940HS ("Phoenix") APU with a
Radeon 780M iGPU (RDNA3, 12 CUs, RADV driver, `deviceID 0x15bf`). No sudo
access is available on this box to read exact installed RAM timings via
`dmidecode` (`Permission denied` on `/sys/firmware/dmi/tables/...`), so
published platform specs are used instead, per spec.md's Assumptions.

- Peak FP32 compute: ~8.9 TFLOPS (published boost-clock figure for this
  12-CU RDNA3 iGPU).
- Peak FP16 compute: ~17.8 TFLOPS (RDNA3's standard 2x FP32 rate for packed
  FP16 math).
- Peak memory bandwidth: 89.6 GB/s (dual-channel DDR5-5600, this
  platform's published maximum).

**Decision**: Use these published figures for the roofline machine balance
point (FR-001). If the actual installed RAM is slower than DDR5-5600, the
real balance point would be even lower, which only strengthens (never
weakens) a bandwidth-bound conclusion -- so this is a safe, conservative
assumption for FR-002's purposes.

**Alternatives considered**: Querying installed RAM speed via `dmidecode`/
`lshw`. Rejected -- not accessible without sudo on this box, and per the
margin found in Decision 3 below, would not change the conclusion even if
available.

**Important caveat found**: RDNA3's WMMA instructions do not use a
separate matrix-multiply unit -- they execute on the same ALUs as ordinary
vector math, reorganized to reduce instruction count and register-bank
pressure rather than to raise the peak FLOP rate itself (source: AMD
GPUOpen's own RDNA3 WMMA guide, corroborated by independent
microbenchmarking). This means "compute-bound" and "bandwidth-bound" here
refer to the same peak-FLOPs ceiling whether a kernel uses WMMA or ordinary
vectorized math -- WMMA's benefit mechanism on this specific hardware is
reduced overhead, not a higher ceiling. This doesn't change the roofline
methodology, but it is important context: even a mild bandwidth-bound
verdict wouldn't strictly guarantee zero benefit from reduced overhead --
the margin found below is large enough that this nuance doesn't change the
recommendation.

## Decision 2: Kernel's theoretical arithmetic intensity -- from the actual shader's weight format, not assumed

**Finding**: Read `backends/vulkan/runtime/graph/ops/glsl/linear_q4gsw_coop.glsl`
directly (the actual decode GEMV kernel in scope, FR-001) and its included
`linear_int4_weight_tile_load.glslh`/`linear_fp_weight_scales_load.glslh`.
Confirmed: weight is `t_packed_int4_weight` (4 bits/element, i.e. 0.5
bytes/element) plus `t_weight_scales`, one fp16 scale per `group_size`
weight elements along `K` (amortized, negligible relative to the packed
weight bytes themselves for realistic group sizes).

- FLOPs per weight element used: 2 (one multiply, one add -- the MAC that
  produces one term of the dot product).
- Bytes per weight element: 0.5 (4-bit packed).
- Base arithmetic intensity: 2 / 0.5 = **4 FLOPs/byte**.
- Even under a generous assumption of 4x extra FLOPs per element for
  dequantization overhead (subtracting zero-point, applying scale): **16
  FLOPs/byte**.

Activation (`K` elements) and output (`N` elements) reads/writes are
negligible next to the weight matrix's `K*N*0.5` bytes for realistic
decode shapes (`K`, `N` in the thousands), so they don't move this figure
meaningfully.

**Decision**: Use 4-16 FLOPs/byte as the kernel's arithmetic intensity
range (base to generous-overhead estimate) for the FR-001 roofline
comparison.

**Alternatives considered**: Deriving intensity from measured GPU counters
instead of the kernel's known format. Rejected per spec.md Clarifications
(Option A) -- the analytical approach is cheaper, and the kernel's format
is fully known from its own source, so there's no need for empirical
counter access.

## Decision 3: Roofline comparison -- decisively bandwidth-bound, not a close call

**Finding**: Machine balance point (Decision 1) is **~199 FLOPs/byte**
(17.8 TFLOPS ÷ 89.6 GB/s). The kernel's arithmetic intensity (Decision 2)
is **4-16 FLOPs/byte**. This is a **12x to 50x margin below the balance
point** -- decode's linear GEMV kernel is unambiguously memory-bandwidth
-bound on this device, not an ambiguous or borderline case.

**Decision**: Per FR-002/FR-003, this feature's conclusion is that decode's
linear layer is memory-bandwidth-bound. Per FR-003, this feature does NOT
proceed to design a new WMMA decode shader -- User Story 2 and User Story
3 are not attempted. The feature's deliverable is this finding, formally
documented (tasks.md), plus a recommendation of what would actually help
(Decision 4).

**Alternatives considered**: Proceeding to build a shader anyway "just to
check empirically." Rejected -- this would contradict spec.md's own FR-003
and Assumptions (a "don't build this" conclusion is an explicit, valid,
complete outcome, not a failure to route around), and would spend real
engineering effort investigating an outcome the roofline model already
answers with a 12-50x margin, not a few-percent one where empirical
double-checking might be warranted.

## Decision 4: What would actually help decode speed instead

**Decision**: Per FR-003, name concrete alternatives rather than leave the
"not worth it" conclusion as a dead end:

1. **More aggressive weight quantization** (fewer bytes read per token) --
   directly reduces the denominator of the kernel's own bytes-read figure,
   the actual bottleneck identified in Decision 3. This is the most direct
   lever: half the bytes read (e.g. a hypothetical 2-bit scheme) would
   roughly double decode throughput on a bandwidth-bound kernel, all else
   equal.
2. **Batching multiple concurrent requests, or speculative decoding**
   (verifying several draft tokens per step) -- both create a real `M>1`
   workload, which is the actual precondition for cooperative-matrix
   hardware to have anything to tile across. Neither is a shader change;
   both are larger serving/architecture features in their own right.

**Alternatives considered**: None further explored -- this feature's scope
(spec.md Assumptions) is the feasibility question itself, not designing
either alternative in depth; that's future work this finding points to,
not this feature's own deliverable.

## Decision 5: Contingency methodology, if a future re-check overturns this finding

**Decision**: Should new information (e.g. a much faster RAM configuration
discovered on a different unit of this device, or a materially different
GEMV kernel format in the future) reopen User Story 2/3, the methodology
already established in `010` applies unchanged: small tile-aligned shape,
CPU/ATen reference, dtype-appropriate tolerance, SPIR-V cooperative-matrix
instruction confirmation, then benchmark against the existing kernel at
each target model's real per-token shape with iteration count and
variance reported.

**Alternatives considered**: None -- no reason to invent a new methodology
for a contingency this workstream has already solved twice.
