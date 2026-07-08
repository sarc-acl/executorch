# Quickstart: 8da4w Int8 WMMA Double-Buffer Variant Sweep

Validation guide for this feature — proves the sweep mechanism works and produces a
trustworthy result. See `data-model.md` for the fields each step produces and
`research.md` for why each step is shaped this way.

## Prerequisites

1. Confirm M5 EVT1 is free before starting (constitution Principle VIII / project memory —
   this is shared hardware; do not assume it's free from a prior session).
2. Read `.shared-context/instruction-for-ai/README.md` first (constitution Principle X) for
   the current device/host/clock-pin conventions — do not infer them from a prior session.
3. Re-verify the on-device Vulkan driver identity (Principle VIII) before any measurement
   step below.

## Step 1: Create and bootstrap the new worktree

```bash
# from the .bare repo's parent, i.e. /local/yanwen.xu/workspace
git worktree add <new-dir> -b 023-8da4w-int8-dbuf-sweep-impl quant-perf-optimization
cd <new-dir>/executorch
uv venv .venv --seed
source .venv/bin/activate   # or activate.fish
./install_executorch.sh --minimal
```

Expected: a clean checkout of `quant-perf-optimization`'s tip (including this feature's
committed `specs/023-8da4w-int8-dbuf-sweep/` docs), with none of the original worktree's
unrelated uncommitted changes.

## Step 2: Build the four dbuf variant shaders + dispatch hook

Add `linear_dq8ca_q4gsw_coopmat_dbuf{1,2,3,4}.glsl`/`.yaml` (ported per research.md
Decision 4) and the `ET_VK_DQ8CA_COOPMAT_VARIANT` env-var branch in `QuantizedLinear.cpp`
(research.md Decision 3), then build per `.shared-context/instruction-for-ai/build.md`'s
two-step Android recipe (core runtime + `--target install`, then the dependent
`test_coopmat_linear_bench`-family target).

Expected: build succeeds for all four variants with the env var unset producing identical
behavior to today's shipped `dbuf4` dispatch (no default-behavior change).

## Step 3: Prove one variant (User Story 1 MVP)

For one variant (e.g. `dbuf1`):

```bash
ET_VK_DQ8CA_COOPMAT_VARIANT=dbuf1 adb shell <bench-binary> --correctness-only
```

Expected: process exits 0, bench harness's kernel-name log shows
`linear_dq8ca_q4gsw_coopmat_dbuf1` (not a fallback), and the existing `dq8ca_q4gsw`
correctness check passes. If the process crashes (Xclipse PAL pipeline-creation failure),
record `compiles=false` with the crash detail as `failure_reason` — do not treat this as
blocking the other three variants (research.md Decision 2).

Once one variant is proven, repeat for the remaining three (`dbuf2`, `dbuf3`, `dbuf4`) —
each in its own process invocation.

## Step 4: SPIR-V verification (once per variant)

```bash
spirv-dis <compiled-shader-for-variant>.spv | grep -i CooperativeMatrix
```

Expected: `OpCooperativeMatrixMulAddKHR` (or equivalent) present, operating on 8-bit
component types, for every variant that passed Step 3.

## Step 5: Timed sweep (User Story 2)

For each variant that passed Steps 3-4:

```bash
ET_VK_DQ8CA_COOPMAT_VARIANT=<variant> adb shell <bench-binary>
# runs all 6 shapes (wq + w1_gate x {1B,3B,8B}) within this one process,
# 3 runs per shape, pinned clocks
```

Expected per shape: a mean execution time + CoV, with the clock pin's GFLOP/s cross-check
confirming it actually bound (Principle VII).

## Step 6: Synthesize the report (User Story 3)

Populate `specs/023-8da4w-int8-dbuf-sweep/results/m5-dq8ca-dbuf-sweep-report.md` with:
- per-shape and overall fastest variant (or "varies by shape")
- the dbuf3-is-faster-for-int8 hypothesis verdict, with numbers
- the fastest variant's margin vs. the in-sweep `dbuf4` measurement
- any failed variant, with its `failure_reason`

Expected: every item in spec.md's Success Criteria (SC-001 through SC-005) is verifiable
by reading this report alone, per its own Independent Test.
