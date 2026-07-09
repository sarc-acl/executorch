# T014: subgroup_size=32 re-verification for `linear_dq8ca_qw_coopmat`

**Bounded, one-shot check (not counted against this feature's search budget).**

## What was tested

Built and ran a single ad-hoc variant, `linear_dq8ca_q4gsw_coopmat_sg32test`, identical to
the shipped `dbuf2` shader (loop structure) but with `SUBGROUP_SIZE=32` (instead of the
shipped `64`) at the same 128×64/K32/2×2 tile geometry, on M5 EVT1.

- Driver: `c9861e9906d03fa2c7d48b804e1a1c80` (the standard `f14c51b6f8` build)
- Clocks: pinned, verified bound (509/2730/663 MHz)
- Date: 2026-07-09

## Result: the documented crash did NOT reproduce

```
linear_dq8ca_q4gsw_coopmat_sg32test_buffer_texture2d_half ... 749.322 μs
et_vk.linear_dq8ca_q4gsw.default  linear_dq8ca_q4gsw_correctness_M128_K128_N128_Buffer
  [128x128] 1460.596 μs  2.872 GFLOP/s  PASSED
[correctness] ... -> linear_dq8ca_q4gsw_coopmat_sg32test_buffer_texture2d_half
  (coopmat dispatched), correctness=PASSED
exit code: 0
```

The shader compiled, the pipeline was created without a crash, the coopmat kernel dispatched
(confirmed by kernel-name capture, not a tiled fallback), and correctness PASSED. This
directly contradicts `linear_dq8ca_qw_coopmat.yaml`'s own header comment ("the Xclipse PAL
compiler crashes in `vkCreateComputePipelines` when int8 WMMA is compiled at forced
subgroup size 32").

## Why this is a flagged finding, not a scope change for this feature

Per research.md Decision 1's Alternatives-considered: a newly-compiling `subgroup_size=32`
result is explicitly out of scope for this feature's own search -- re-deriving the legal
space to include `subgroup_size=32` and doubling the candidate universe mid-sweep would
retroactively invalidate `enumerate_configs.py`/`score_and_shortlist.py`'s already-produced
`configs.json`/`shortlist.json` (609 candidates, budget-capped shortlist of 30) for no
proven benefit -- one passing config at one tile shape does not establish that
`subgroup_size=32` is broadly viable across this shader's full geometry space, and the
original crash report may have been driver-version- or geometry-specific.

**This is recorded here as a finding for a follow-up feature**, not acted on further in
`specs/025`. If a future feature re-opens the `subgroup_size` axis, this result (and the
specific driver hash it was observed on) is the starting evidence.

## Cleanup

The ad-hoc `linear_dq8ca_q4gsw_coopmat_sg32test.{glsl,yaml}` files and the `"sg32test"`
allow-list entry added to `QuantizedLinear.cpp`'s `dq8ca_coopmat_variant()` in the execution
worktree are temporary, single-purpose probes for this check and are removed after this
task completes (they are not part of this feature's shortlist/search artifacts and must not
be mistaken for one of the 30 budgeted candidates).
