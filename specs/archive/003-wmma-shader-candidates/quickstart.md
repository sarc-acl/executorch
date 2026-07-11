# Quickstart: WMMA-Optimizable Shader Candidates Report

Validates the classification pipeline for one configuration before running
the full six-configuration sweep. Unlike `001`/`002`, this feature needs no
MiniPC/GPU access, no build, and no `.venv` GPU dependencies — it only reads
already-existing JSON and source files.

## Prerequisites

- `002-etdump-shader-profiling` is complete: all six
  `specs/002-etdump-shader-profiling/results/raw/<model>_<scheme>.json`
  files exist with populated `aggregated`/`category_rollup` arrays.
- Read access to `backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp`
  and `SDPA.cpp` (for the classification script to cite; it does not need to
  compile or execute them).

## 1. Classify one configuration

```bash
python specs/003-wmma-shader-candidates/scripts/classify_shaders.py \
  --model llama-3.2-1b --scheme 4w \
  --profiling-json specs/002-etdump-shader-profiling/results/raw/llama-3.2-1b_4w.json \
  --out specs/003-wmma-shader-candidates/results/classifications/llama-3.2-1b_4w.json
```

Expected outcome: a classification JSON matching
`contracts/candidates-report-schema.md`, with every `002` aggregated entry
carrying exactly one of `a`/`b`/`c`/`d`/`uncertain`.

## 2. Verify classifications against the actual code

For every `b`/`c` entry, open the cited file:line
(`QuantizedLinear.cpp`/`SDPA.cpp`) and confirm the reason is accurate — this
is the whole point of the feature (research.md already did this once; this
step re-validates the script's output matches that manual reading, and
catches drift if the classification rules are ever extended).

- Confirm no `classification: "a"` appears anywhere (FR-009) — all captures
  used `tiled_baseline`.
- Confirm the prefill linear family (`gemm`/`_tiled` kernels in categories
  attention-projection/feed-forward/output-projection) shows **two**
  `blocking_reasons` (rank-3 output, `TEXTURE_3D` storage), not one.
- Confirm decode linears (`gemv`/`_coop` kernels, same categories) are
  classified `c`, not `b` — there is no existing GEMV-shaped coopmat kernel
  to be "blocked."
- Confirm `sdpa`-category entries (both phases) are classified `c`.

## 3. Classify the remaining five configurations

Repeat step 1 for the other five `(model, scheme)` pairs.

## 4. Generate the consolidated report

```bash
python specs/003-wmma-shader-candidates/scripts/classify_shaders.py --generate-report \
  --classifications-dir specs/003-wmma-shader-candidates/results/classifications \
  --out specs/003-wmma-shader-candidates/results/wmma-candidates-report.md
```

Expected outcome: `wmma-candidates-report.md` with two ranked sections
("existing implementation blocked" / "no WMMA implementation exists"), each
group sorted by summed absolute time (Clarifications: primary sort key),
with relative percentage-of-phase shown per-config alongside.

## 5. Sanity-check the ranking

- Confirm the top-ranked group in each section corresponds to what the raw
  `002` category rollups already showed as the dominant category
  (feed-forward, typically) — the ranking should not surface a surprising
  top entry that contradicts `002`'s own category percentages without a
  clear reason (e.g. a genuinely large absolute-time contribution from the
  8B model specifically).
- Confirm every group's `total_time_us_summed` is at least as large as its
  largest single-config contribution (a basic sum sanity check).
- Confirm each group's "existing/prospective shader(s)" line lists **every**
  distinct shader relevant to that group, not just one — a group spanning
  both `4w` and `8da4w` configs has two different coopmat shaders behind it
  (`linear_qw_coopmat.glsl` and `linear_dq8ca_qw_coopmat.glsl`); a naive
  implementation that assigns this field per-row and lets later rows
  overwrite earlier ones will silently drop one of them without erroring.
