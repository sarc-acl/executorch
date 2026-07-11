# Contract: Decode Shader WMMA Feasibility Data Formats

## Roofline calculation (analytical, no new tooling)

A plain calculation, not a capture harness -- reproducible from the
figures in research.md Decisions 1-3:

```text
peak_compute_flops = peak_fp16_tflops * 1e12
peak_bw_bytes       = peak_bandwidth_gbs * 1e9
machine_balance_point = peak_compute_flops / peak_bw_bytes   # FLOPs/byte

kernel_intensity_base     = 2 / 0.5   # FLOPs per weight element / bytes per weight element
kernel_intensity_generous = kernel_intensity_base * 4   # dequant-overhead allowance

margin = machine_balance_point / kernel_intensity_{base,generous}
```

- `verdict = bandwidth_bound` if `margin` is large in either direction
  (kernel intensity far below the balance point) -- research.md Decision 3
  found 12-50x, decisively past any reasonable "close to it" threshold.
- `verdict = compute_bound` if kernel intensity is far *above* the balance
  point.
- `verdict = ambiguous` if within a small margin (e.g. within 2x) of the
  balance point -- not the case found here.

## `results/decode-wmma-feasibility-report.md`

Rules a consumer can depend on:

- The Roofline Finding (research.md Decisions 1-3) always appears first,
  with its cited peak-spec sources and the exact numbers used --  never
  just an assertion of the verdict without the supporting figures.
- If `verdict == bandwidth_bound`: User Story 2/3 sections are present but
  explicitly marked "not attempted -- see Roofline Finding," never
  silently omitted (mirrors this workstream's Excluded/Blocked-section
  precedent from `009`/`010`/`011`).
- If `verdict != bandwidth_bound`: the Correctness Case must appear (and
  pass) before any Microbenchmark Case timing is reported (constitution
  Principle I, non-negotiable).
- Exactly one overall statement appears: is WMMA acceleration worth
  pursuing for decode on this device, and if not, what would actually help
  instead (research.md Decision 4).

## If User Story 2/3 execute (contingent path)

Reuses `010`'s exact correctness-harness contract (small tile-aligned
shape, CPU/ATen reference, dtype-appropriate tolerance, SPIR-V
cooperative-matrix instruction confirmation) and `007`/`010`'s exact
microbenchmark contract (iteration count and variance on every timing,
dispatch confirmed before any number trusted) -- no new schema.
