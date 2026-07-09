# Quickstart: Validate the 8da4w Tile/Subgroup Sweep End-to-End

## Prerequisites

- A dedicated execution worktree branched from `dev/executorch`'s tip (after this feature's
  spec/plan/tasks are committed to `yanwen/dev-1.3`) — never the existing `dev/` worktree
  folder itself (workspace `CLAUDE.md` worktree-binding rule; research.md Decision 4). If
  `specs/023-8da4w-int8-dbuf-sweep`'s own execution worktree is still present and warm
  (`cmake-out-android-vk/` already built with the `dbuf1-4` variants), reuse it — otherwise
  bootstrap a fresh one per `.shared-context/instruction-for-ai/` build docs
  (`./install_executorch.sh --minimal`, then the Android cross-build).
- SSH/adb access to M5 EVT1 per `.shared-context/instruction-for-ai/` §Access & Run.
- Python 3 on the build box (standard library only, matching `022`'s scripts).

## Step 0 — Re-confirm the dbuf loop-structure winner (User Story 1)

In the execution worktree, using the existing `dbuf1-4` shader family from `specs/023`:

```bash
# Pre-flight: verify driver hash + clocks pinned (Principle VII/VIII) before ANY measurement
ssh <device-host> 'adb -s <serial> shell md5sum /vendor/lib64/hw/vulkan.samsung.so'
ssh <device-host> 'bash <pin_freqs_script>'

for v in dbuf1 dbuf2 dbuf3 dbuf4; do
  ET_VK_DQ8CA_COOPMAT_VARIANT=$v \
    adb -s <serial> shell /data/local/tmp/test_coopmat_linear_bench \
    --shapes wq,w1_gate --models 1B,3B,8B --runs 3
done
```

**Expected outcome**: `dbuf_reconfirmation.json` (per
`contracts/sweep-report-schema.md` §0) is produced with 4 entries, each either measured to
completion or carrying an explicit `failure_reason`. Every measured entry has
`dispatch_confirmed: true` (kernel-name capture, not inferred). The fastest variant is
recorded — this is the loop structure held fixed for every remaining step, whichever variant
it actually is.

## Step 1 — Re-derive the legal `8da4w` tile/subgroup space

```bash
python3 specs/025-8da4w-parameter-sweep/scripts/enumerate_configs.py \
  --loop-structure <winner-from-step-0> \
  --out specs/025-8da4w-parameter-sweep/results/configs.json
```

**Expected outcome**: `configs.json` contains only `subgroup_size: 64` entries (research.md
Decision 1). Spot-check: the file must contain the currently-shipped
`tsweep_t128x64k32g22s64` entry, and must NOT contain any `...s32` token.

## Step 2 — Score and shortlist

```bash
python3 specs/025-8da4w-parameter-sweep/scripts/score_and_shortlist.py \
  --configs specs/025-8da4w-parameter-sweep/results/configs.json \
  --dbuf-reconfirmation specs/025-8da4w-parameter-sweep/results/dbuf_reconfirmation.json \
  --out specs/025-8da4w-parameter-sweep/results/shortlist.json
```

**Expected outcome**: `shortlist.json` has one entry per `configs.json` candidate (full
ranking). The shipped-config anchor is `shortlisted: true` regardless of rank. Zero on-device
measurement occurs at this step — must complete without touching adb.

## Step 3 — Pre-flight device/driver check (manual, before any round)

Same as `022`'s Step 3 — re-verify driver hash and pinned clocks immediately before Step 5,
not reused from Step 0's check (Principle VIII: re-verified before *every* coopmat
measurement round, not just once per session).

## Step 4 — Extend the shader variant catalog for the shortlist

In the execution worktree, append one `shader_variants` entry per shortlisted candidate to a
new `linear_dq8ca_q4gsw_coopmat_tsweep.yaml` (built on the Step 0 winning loop structure),
and one token branch to a new `dq8ca_coopmat_variant_tile()` + `kTokens[]` table in
`QuantizedLinear.cpp`, additive to (not replacing) `specs/023`'s existing
`ET_VK_DQ8CA_COOPMAT_VARIANT` dbuf selection (research.md Decision 3). Rebuild:

```bash
cmake --build cmake-out-android-vk -j$(nproc) --target install --config Release
cmake --build cmake-out-android-vk/bench --target test_coopmat_linear_bench -j$(nproc)
```

**Expected outcome**: build succeeds; any candidate that fails to compile is recorded with
`compile_status: compile_failed` and removed from the yaml/cpp before proceeding.

## Step 5 — Run the staged search

```bash
python3 specs/025-8da4w-parameter-sweep/scripts/staged_search.py \
  --shortlist specs/025-8da4w-parameter-sweep/results/shortlist.json \
  --bench-binary <execution-worktree>/cmake-out-android-vk/bench/test_coopmat_linear_bench \
  --ssh-host <device-host> --serial <serial> \
  --budget-cap-formula "min(round(0.15*N), 30)" \
  --out-dir specs/025-8da4w-parameter-sweep/results/
```

**Expected outcome**: `round1_results.json`, `round2_results.json`, `round3_results.json`,
and `budget.json` are produced in sequence. `budget.json`'s `configs_measured_on_hardware`
never exceeds `budget_cap` at any point. If a `halted: true` sentinel appears, stop and
re-run Step 3 before resuming.

## Step 6 — Produce the final report

```bash
python3 specs/025-8da4w-parameter-sweep/scripts/staged_search.py --report-only \
  --dbuf-reconfirmation specs/025-8da4w-parameter-sweep/results/dbuf_reconfirmation.json \
  --out-dir specs/025-8da4w-parameter-sweep/results/
```

**Expected outcome**: `sweep-report.md` exists and satisfies
`contracts/sweep-report-schema.md` §5 — states the loop-structure re-confirmation result,
names a tile/subgroup winner (or states the shipped configuration stands), includes the
shipped-baseline and `4w`-winner comparisons, states the winner's correctness result, and
reports `configs_measured_on_hardware` against the budget cap and the SC-006 5x-reduction
target.

## Success check

The feature is validated end-to-end when all of the following hold simultaneously:
- `dbuf_reconfirmation.json` has 4 entries and `sweep-report.md` states explicitly whether
  the fastest variant matches the user's reported `dbuf2` claim (SC-001).
- `configs_measured_on_hardware` in the final `budget.json` is `<= budget_cap`, and no entry
  anywhere in `configs.json`/`shortlist.json`/round files has `subgroup_size: 32`.
- The winning tile/subgroup candidate (if any) has a passing correctness result and a
  Round-3 `mean_gflops`/`stddev_gflops` pair with `run_count == 3`.
- The report's search-cost section shows ≥5x estimated device-time reduction versus the
  exhaustive-universe estimate (SC-006).
- Any candidate in `configs.json` can be traced to a `shortlist_reason` or an elimination
  round/reason without re-running anything (spot-check 3-5 arbitrary non-shortlisted
  candidates against `shortlist.json`).
