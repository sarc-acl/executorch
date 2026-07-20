# Research: Why 8da4w Is Slower Than 4w On The Tiled (No-WMMA) Path

## Decision 1: The mechanism is dispatch count + integer-correction math, not a shader bug

**Decision**: `8da4w`'s tiled linear is slower than `4w`'s tiled linear because it does
strictly more work per output element and requires two extra whole compute-shader
dispatches before the matmul even starts — there is no coopmat/WMMA hardware available on
the tiled path to amortize that extra cost against, unlike the coopmat path where the same
bookkeeping is worthwhile.

**Evidence — shader/dispatch inventory:**

| | `4w` (`linear_q4gsw_tiled.glsl`) | `8da4w` (`linear_dq8ca_q4gsw_tiled.glsl`) |
|---|---|---|
| Activation input | Reads the fp16 input tile directly (`load_input_tile_no_checks`) | Reads a **pre-quantized int8** tile (`t_packed_int8_input`, via `load_int8_input_tile`) |
| Extra per-tile buffer reads | none | `t_int8_input_scales`, `t_int8_input_zps` (texture3d), `t_weight_sums`, `t_int8_input_sums` (buffer) |
| Weight dequant + accumulate | `fp_accumulate_with_int4_weight` — dequant int4 weight to fp16, then a **single fp16 multiply-add** | `int_accumulate_with_int4_weight` (`dotPacked4x8AccSatEXT`) — int4 weight unpacked into two 4-bit blocks, **int32 dot-product accumulation** |
| Final step | none extra | a separate dequant + **zero-point correction** pass (`accumulate_out_tile_with_int_accum_from_int4_weights`) combining input scale/zp, weight sums, and weight scale into the fp output |
| Upstream dispatches before the matmul | 0 | 2 — `choose_qparams_per_row.glsl` (per-row/per-token amax/amin → scale + zero-point, `choose_qparams.glslh:13-74`) and `quantize_and_pack_4h4w_with_group_sums.glsl` (quantizes+packs the int8 activations, computes `input_sums`) |
| C++ dispatch sites | `QuantizedLinear.cpp` `add_linear_qw_node` | `QuantizedLinear.cpp:790-872` `quantized_linear_impl` — `add_choose_qparams_per_row_node` (790s) → `add_quantize_and_pack_4h4w_with_group_sums_node` (864) → `add_linear_dqa_qw_node` (874) |

Storage types are mixed on the `8da4w` side: `input_scales`/`input_zps` are `texture3d`;
the packed int8 activations, `input_sums`, and `weight_sums` are `buffer`.

**Rationale**: the zero-point correction machinery exists because `8da4w`'s activation
quantization is *asymmetric* (has a nonzero zero-point) — a raw int8×int4 dot product is
not directly the right dequantized answer without correcting for the zero-point's cross
terms against the weight sums and the weight's own zero-point. That correction, plus the
two upstream quantize/pack dispatches, are the price of admission for later feeding
int8×int4 into hardware coopmat/WMMA dot-product instructions. **On the tiled path there is
no such hardware to amortize the cost against** — Xclipse's scalar ALUs have no throughput
advantage for int8 arithmetic over fp16, so `8da4w` pays the full bookkeeping cost with none
of the payoff. `4w`'s tiled path skips all of this because its activation is never
quantized at all (fp16 throughout).

**Why this doesn't contradict `8da4w` sometimes *beating* `4w`**: on the coopmat/WMMA path
(a different shader entirely — `linear_dq8ca_q4gsw_coopmat`/`_dbuf2`, not `_tiled`), real
int8-dot hardware instructions absorb the extra dispatches' cost and then some — GFXSW-69499
(2026-06-11) measured `8da4w` coopmat e2e (85.1 tok/s) beating `4w` coopmat e2e (79.3 tok/s)
on 8B. Both results are correct; they're different shaders answering different questions.

**Alternatives considered**:
- *A driver/compiler regression specific to `8da4w`'s tiled shader.* Rejected — the
  mechanism above fully explains a *consistent, shape-independent* slowdown without positing
  any bug; GFXSW-69499's raw table (2026-07-09) shows `8da4w` tiled slower at every one of
  12 shapes across all 3 models, which is what a structural (not incidental) cause predicts.
- *Something about texture-vs-buffer storage specifically.* Considered but not the primary
  driver — `4w`'s own T-vs-B tiled study ([[tvb-storage-cross-device]] memory; also
  `session-2026-06-22-23-findings.md` Finding "New devices" T-vs-B table) found storage-only
  effects are small (~4-14%) on this hardware, an order of magnitude below the `8da4w`-vs-`4w`
  gap (e.g. GFXSW-69499 2026-07-09: 8B (4096,4096) `4w` tiled 76,471µs vs `8da4w` tiled
  120,515µs, a ~57% gap) — storage alone doesn't explain a gap this large.

## Decision 2: Existing raw evidence is sufficient to confirm direction, not yet report-grade

The only quantitative evidence for this spec's premise today is GFXSW-69499's 2026-07-09
"personal note" microbench table (12 shapes × 2 schemes, single-shot per cell, no CoV/repeat
count — unlike this workstream's usual 3-run-mean+CoV convention used in `specs/018/022/023/025`).
The *direction* is unambiguous and consistent everywhere (`8da4w` tiled slower at all 12
cells), but a report-grade number would re-run this with the same rigor as those specs.
Not done as part of this pass — see `spec.md` SC-003.

## Decision 3: This spec's premise, once formalized, immediately unblocked a different investigation

Written up 2026-07-11 in the course of a broader session that also investigated an
unrelated `4w`-decode-throughput regression between 2026-06-17 and July measurements
([[decode-regression-june-vs-july]] memory; results doc
`specs/018-m5-8da4w-t-tiled-baseline/results/decode-regression-investigation-2026-07-11.md`).
That investigation is unrelated to this spec's mechanism (it's about `4w` decode changing
across *time*, not about `8da4w` vs `4w` on the *same* measurement) but shares the same
target device/workload family — cross-referenced here for anyone who arrives at this spec
looking for "why did my `8da4w` number change" rather than "why is `8da4w` slower than `4w`."
