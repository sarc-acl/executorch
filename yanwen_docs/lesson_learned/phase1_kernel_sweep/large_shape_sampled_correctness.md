# Large-shape sampled correctness

The linear and matmul coopmat microbenchmarks previously skipped CPU reference
generation for shapes above the reference limit. Because the shared validator
treats an empty reference as "no validation", those large cases could still
print `PASSED`.

For large-shape correctness runs, fill the full reference vector with `NaN` and
compute a bounded deterministic sample of output elements. The validator now
skips only explicit `NaN` reference entries and requires at least one checked
element. This keeps CPU time bounded while still catching large-shape numeric
issues.

The first sampled linear run exposed one failure in the existing non-coopmat
buffer fallback: `cm_fp16_LLM_FFN_down_1tok` uses
`linear_vec_tile_row_1_buffer_texture2d_half` because `M=1`, and missed the CPU
reference by `1.138` on one sampled element with the current fp16 tolerance.
The routed `linear_coopmat_half` and `matmul_coopmat_half` large shapes passed.
