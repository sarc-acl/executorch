# linear_coopmat_bench large cases skip CPU reference

## What was attempted

Verified correctness while running the small fp16 LLaMA-shaped
`linear_coopmat_bench` set.

## What happened

Only `64x128x128` generated a CPU reference. The LLaMA-shaped cases printed
messages such as:

```text
Skipping reference for large matrix (64x4096x4096)
```

The harness still reported `PASSED` because `ValueSpec::validate_against_reference`
returns true when reference data is empty.

## Why it matters

The run verifies that `linear_coopmat_half` dispatches and passes the small CPU
reference case, but the large LLaMA-shaped timings do not yet have independent
max-absolute or max-relative error measurements.

## Recommended next action

Before expanding the sweep, add baseline-output comparison against Stephen's
texture `linear_vec` path, or add bounded CPU reference cases that exercise the
same cooperative-matrix tile path without multi-billion-op CPU references.
