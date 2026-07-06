# Quickstart: M5 EVT1 `4w` Linear Coopmat Retune

This feature's User Story 1 (commit the existing work) requires no device
access and is likely already done by the time you read this (see
`results/disposition-summary.md` for current status). User Stories 2/3
require Samsung M5 EVT1 device access.

## Prerequisites

- This feature's User Story 1 commit is present (`linear_qw_coopmat.glsl`,
  `linear_dq8ca_qw_coopmat.glsl`, `QuantizedLinear.cpp` carry the four
  changes described in `spec.md`).
- FR-008's correctness-harness extension (step 2 below) is written --
  requires a build toolchain but NOT device access, so it can be done ahead
  of a hardware session.
- Samsung M5 EVT1 device access (adb), per workspace
  `.shared-context/instruction-for-ai/devices-and-access.md`.
- Driver identity re-verified per constitution Principle VIII before any
  measurement (`.shared-context/ACTIVE-STATUS.md` for current expected
  hash).
- GPU/MIF/INT clocks pinned per constitution Principle VII default, unless
  a floating run is explicitly requested.
- `spirv-dis` available (Vulkan SDK).

## 1. Capture the pre-change baseline (Decision 1)

```bash
git stash push -- \
  backends/vulkan/runtime/graph/ops/glsl/linear_qw_coopmat.glsl \
  backends/vulkan/runtime/graph/ops/glsl/linear_dq8ca_qw_coopmat.glsl \
  backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp
# rebuild + push + run the existing tier-1 coopmat linear microbench on M5 EVT1
# record: mean_us, stdev_us, iterations, kernel name, per production shape
git stash pop
```

Expected outcome: a `4w` coopmat tier-1 timing for this exact pre-change
shader on M5 EVT1, with kernel-dispatch confirmation (coopmat kernel name,
not tiled fallback). This is the number US2/US3's post-change results are
diffed against -- not any number from the sibling `quant-dev` branch
(`research.md` Decision 1).

## 2. Extend the correctness harness to production K (FR-008)

```bash
# edit backends/vulkan/test/custom_ops/test_coopmat_linear_bench.cpp:
#   add entries to kCorrectnessShapes / kRank3CorrectnessShapes at K=2048 and K=4096
#   (coopmat-eligible M/N, e.g. M=128 or 256, N=128 or 256; K%32==0)
#   reuse the existing well-conditioned positive-data generation and
#   set_abs_tolerance(0.5f) / set_rel_tolerance(0.05f) -- do not invent a new strategy
# rebuild test_coopmat_linear_bench
```

Expected outcome: the harness now has correctness cases at production K,
still passing structurally sound shapes (this step only adds coverage; it
does not yet validate any of this feature's three shader changes). Skip
this step only if a later run of `git log`/`git diff` shows it's already
been done in a prior session -- check before redoing.

## 3. Validate the two same-math changes (User Story 2)

```bash
# rebuild with the working-tree changes applied (loop_flattening + vectorized_dequant + fp16_accumulate all present together, since they're interleaved -- see research.md Decision 3)
# run the extended (step 2) coopmat correctness check at production K=2048/4096
# run the tier-1 microbench; confirm kernel dispatch + spirv-dis accumulator/coopmat-instruction check
```

Expected outcome: correctness pass, a tier-1 timing comparable to step 1's
baseline, and a `spirv-dis` confirmation that `OpCooperativeMatrix*KHR`
instructions are present in the compiled shader.

**Caveat**: because `fp16_accumulate` is currently interleaved with the two
same-math changes in the same working-tree diff, this run's numbers include
`fp16_accumulate`'s effect too. If User Story 3 finds `fp16_accumulate`
must be reverted, re-run this step's tier-1 measurement once more after
that revert commit lands, to get a clean same-math-only number.

## 4. Validate the fp16-accumulate change (User Story 3)

```bash
# run the extended (step 2) coopmat correctness check specifically at production K=2048 and K=4096
# compare fp16-accumulate output against the fp32-accumulate reference within the stated tolerance (abs=0.5/rel=0.05, data-model.md's numerical_tolerance field)
```

Expected outcome: either a correctness pass (proceed to a tier-1 timing
comparison against step 1's baseline) or an explicit failure with the
divergence magnitude recorded -- in which case revert this specific change
per `research.md` Decision 3 and re-run step 3 for a clean reading of the
other two changes.

## 5. Record final disposition

Update `results/disposition-summary.md` with each of the three shader
changes' final `disposition` (`keep` / `keep_with_caveat` / `revert`) and
`disposition_reason`, per `data-model.md`'s Retuned Shader Change schema.
