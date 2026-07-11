# Decode Shader WMMA Feasibility Report

## Roofline Finding

**Device**: `rocky-ryzen` MiniPC -- confirmed via `vulkaninfo --summary`
(`deviceName: AMD Radeon 780M Graphics (RADV PHOENIX)`, RDNA3, 12 CUs,
`vendorID 0x1002`, `deviceID 0x15bf`) and `/proc/cpuinfo`
(`AMD Ryzen 9 7940HS w/ Radeon 780M Graphics`, "Phoenix" APU).

**Kernel in scope**: `linear_q4gsw_coop` (decode's linear GEMV path,
`M=1`, covering both `4w` and `8da4w` via its `DYNAMIC_QUANT_VARIANT`
parameter). Confirmed by reading
`backends/vulkan/runtime/graph/ops/glsl/linear_q4gsw_coop.glsl` directly:
weight is `t_packed_int4_weight` (4 bits/element) plus `t_weight_scales`,
one fp16 scale per `group_size` weight elements.

### Peak device specs (published, cited)

| Metric | Value | Source |
|---|---:|---|
| Peak FP32 compute | ~8.9 TFLOPS | Published boost-clock figure, 12-CU RDNA3 iGPU (Radeon 780M) |
| Peak FP16 compute | ~17.8 TFLOPS | RDNA3 standard 2x FP32 rate for packed FP16 math |
| Peak memory bandwidth | 89.6 GB/s | Dual-channel DDR5-5600, Ryzen 9 7940HS platform published maximum |

No sudo access is available on this box to confirm the exact installed
RAM speed via `dmidecode` (`Permission denied` on
`/sys/firmware/dmi/tables/smbios_entry_point`, reconfirmed during
implementation, T003) -- the published platform maximum is used instead.
This is a conservative choice: slower-than-assumed RAM would only lower
the real balance point further, strengthening (never weakening) a
bandwidth-bound conclusion.

### Machine balance point

```
machine_balance_point = peak_compute_flops / peak_bandwidth_bytes
                       = (17.8e12) / (89.6e9)
                       ≈ 198.7 FLOPs/byte
```

### Kernel arithmetic intensity

From the confirmed weight format (4-bit packed = 0.5 bytes/element; one
multiply-add = 2 FLOPs per weight element used in the dot product):

```
base_intensity     = 2 FLOPs / 0.5 bytes = 4.0 FLOPs/byte
generous_intensity = base_intensity × 4 (generous dequant-overhead allowance)
                    = 16.0 FLOPs/byte
```

Activation (`K` elements) and output (`N` elements) reads/writes are
negligible next to the weight matrix's `K×N×0.5` bytes for realistic
decode shapes (`K`, `N` in the thousands), so they don't move this figure
meaningfully.

### Verdict

| | Value |
|---|---:|
| Machine balance point | ~198.7 FLOPs/byte |
| Kernel intensity (base) | 4.0 FLOPs/byte |
| Kernel intensity (generous) | 16.0 FLOPs/byte |
| **Margin below balance point** | **50x (base) to 12x (generous)** |

**Verdict: `bandwidth_bound`.** This is not a close call -- even under a
generous allowance for dequantization overhead, the kernel's arithmetic
intensity sits more than an order of magnitude below this device's machine
balance point. Decode's linear GEMV kernel is unambiguously limited by
memory bandwidth (reading the weight matrix once per generated token), not
by this device's compute throughput.

**Note on RDNA3's WMMA mechanism**: RDNA3's cooperative-matrix (WMMA)
instructions do not use a separate matrix-multiply unit -- they execute on
the same ALUs as ordinary vector math, reorganized to reduce instruction
count and register-bank pressure rather than to raise the peak FLOP rate
itself (AMD GPUOpen's own RDNA3 WMMA guide; corroborated by independent
microbenchmarking). This means "compute-bound" and "bandwidth-bound" here
refer to the same peak-FLOPs ceiling whether a kernel uses WMMA or
ordinary vectorized math. This doesn't change the verdict -- the 12-50x
margin found is decisive regardless of this nuance -- but it means WMMA's
benefit mechanism on this hardware is overhead reduction, not a higher
compute ceiling, which is worth keeping in mind for any future work in
this area.

## User Story 2 (Correctness) — not attempted

Per FR-003: since the Roofline Finding's verdict is `bandwidth_bound`, this
feature does not proceed to design a new WMMA-capable decode shader. A
cooperative-matrix shader would only accelerate multiply-add throughput,
which the Roofline Finding shows is not decode's bottleneck -- building one
would not be expected to yield a measurable speedup, at a real
engineering cost that the roofline analysis lets this workstream avoid
spending.

## User Story 3 (Microbenchmark) — not attempted

Same reason as above -- gated on User Story 2 executing, which it did not.

## Recommendation

Since decode's linear GEMV kernel is memory-bandwidth-bound, the levers
that would actually help decode throughput are:

1. **More aggressive weight quantization** (fewer bytes read per token) --
   directly reduces the denominator of the kernel's own bytes-read figure,
   the actual bottleneck identified above. Roughly halving the bytes read
   per token (e.g. a hypothetical sub-4-bit scheme) would be expected to
   roughly double decode throughput on this bandwidth-bound kernel, all
   else equal.
2. **Batching multiple concurrent requests, or speculative decoding**
   (verifying several draft tokens per step) -- both create a real `M>1`
   workload, the actual precondition for cooperative-matrix hardware to
   have anything to tile across. Neither is a shader change; both are
   larger serving/architecture features in their own right, and are not
   designed further within this feature's scope.

## Overall Statement

**WMMA acceleration is not worth pursuing for decode's linear GEMV kernel
on this device.** The roofline analysis shows a decisive 12-50x margin
between the kernel's arithmetic intensity and this device's machine
balance point -- decode is memory-bandwidth-bound, and cooperative-matrix
hardware accelerates compute throughput, not memory bandwidth. This is a
different situation from every prior WMMA effort in this workstream
(prefill linear, prefill SDPA), both of which were demonstrably
compute-bound and paid off substantially (60-70%+ gains). The path to a
faster decode is more aggressive weight quantization or a real `M>1`
opportunity (batching/speculative decoding), not a new decode shader.
Decode SDPA's two GEMV kernels remain unexamined (out of scope, FR-008)
but share the same `M=1` structural property that drives this finding, so
the same conclusion is expected to hold there too, pending its own
feature.
