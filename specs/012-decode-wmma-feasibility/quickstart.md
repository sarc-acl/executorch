# Quickstart: Decode Shader WMMA Acceleration

Primarily an analytical feature, not a device-capture-heavy one -- unlike
every prior tier-1/tier-2 feature in this workstream, its main deliverable
(the roofline finding) is a calculation, not a GPU capture.

## Prerequisites

- `spec.md`'s Clarifications: roofline methodology (Option A) already
  decided -- arithmetic intensity vs. machine balance point, not GPU
  hardware performance-counter profiling.
- No sudo/dmidecode access on `rocky-ryzen` -- published platform specs
  are the source for peak compute/bandwidth (research.md Decision 1),
  already looked up during planning.

## 1. Confirm the roofline finding (US1, MVP)

Already computed during planning (research.md Decisions 1-3) -- formalize
by writing it into the report with full citations:

```text
Peak FP16 compute:     17.8 TFLOPS  (Radeon 780M, RDNA3, published boost figure)
Peak memory bandwidth: 89.6 GB/s    (Ryzen 9 7940HS platform, dual-channel DDR5-5600, published)
Machine balance point: ~198.7 FLOPs/byte

Kernel (linear_q4gsw_coop, M=1 GEMV):
  weight format: 4-bit packed (t_packed_int4_weight) + per-group fp16 scale
    (confirmed by reading backends/vulkan/runtime/graph/ops/glsl/linear_q4gsw_coop.glsl)
  arithmetic intensity: 4.0 FLOPs/byte (base) to 16.0 FLOPs/byte (generous dequant overhead)

Margin below balance point: 12x-50x -> verdict: bandwidth_bound
```

## 2. Apply the FR-002/FR-003 gate

Given `verdict == bandwidth_bound`: per FR-003, do NOT proceed to design a
new WMMA decode shader. Write the recommendation (research.md Decision 4)
into the report instead: more aggressive weight quantization, or batching/
speculative decoding to create a real `M>1` opportunity.

## 3. (Contingent, not expected to execute) User Story 2/3

Only if a future re-check overturns Decision 3's finding: follow `010`'s
exact correctness-harness and benchmark methodology (contracts/
decode-wmma-feasibility-schema.md) -- no new tooling to design.

## 4. Sanity-check

- The Roofline Finding section states its peak-spec sources and exact
  numbers, not just a bare verdict.
- User Story 2/3 sections are present in the report even though not
  attempted, explicitly marked why (never silently absent).
- One overall statement answers whether WMMA acceleration is worth
  pursuing for decode on this device, and names the recommended
  alternative.
