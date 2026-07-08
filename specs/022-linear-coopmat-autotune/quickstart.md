# Quickstart: Validate the Autotuning Pipeline End-to-End

## Prerequisites

- The isolated experiment worktree from earlier this session:
  `/local/yanwen.xu/workspace/.artifacts/tsweep-256x256-smoketest/executorch`
  (branch `exp/tsweep-256x256-4x4-smoketest`), with its Android build already
  present at `cmake-out-android-vk/` (`libvulkan_backend.a` +
  `cmake-out-android-vk/bench/{test_coopmat_probe,test_coopmat_linear_bench}`).
  If missing/stale, rebuild per `.shared-context/instruction-for-ai/build.md`
  §Android arm64 cross-build, then the bench sub-project step (see this
  session's own build log for the exact incremental-rebuild commands used).
- SSH/adb access to the M5 EVT1 per
  `.shared-context/instruction-for-ai/devices-and-access.md` §1
  (`ssh yanwen.xu@sj1-dmckee-d01`, serial `0000088f8e579c33`).
- Python 3 on the build box (no extra packages beyond the standard library
  are required for `enumerate_configs.py`/`score_and_shortlist.py`).

## Step 1 — Enumerate and validate the 642-config universe

```bash
python3 specs/022-linear-coopmat-autotune/scripts/enumerate_configs.py \
  --out specs/022-linear-coopmat-autotune/results/configs.json
```

**Expected outcome**: `configs.json` contains exactly 642 entries, all
`valid: true`. Spot-check: the file must contain both
`tsweep_t128x128k16g42s32` (dbuf1-equivalent) and `tsweep_t128x64k16g22s32`
(prior sweep winner), and must NOT contain `tsweep_t128x64k16g44s32` (the
config already confirmed this session to fail compilation).

## Step 2 — Score and shortlist

```bash
python3 specs/022-linear-coopmat-autotune/scripts/score_and_shortlist.py \
  --configs specs/022-linear-coopmat-autotune/results/configs.json \
  --known-measurements specs/022-linear-coopmat-autotune/results/known-measurements.json \
  --out specs/022-linear-coopmat-autotune/results/shortlist.json
```

**Expected outcome**: `shortlist.json` has 642 entries total (full ranking),
with roughly 30-40 marked `shortlisted: true` (28 top-ranked plus all 9
previously-measured, compiling known configs from `known-measurements.json`
— see research.md Decision 3's calibration-driven revision). All 9 known
anchors (including `tsweep_t128x128k16g42s32` and `tsweep_t128x64k16g22s32`)
are `shortlisted: true` with `shortlist_reason` starting `anchor:`, regardless of numeric
rank. Zero on-device measurement has occurred at this point — this step
must complete without touching adb.

## Step 3 — Pre-flight device/driver check (manual, before any round)

```bash
ssh yanwen.xu@sj1-dmckee-d01 '
S=0000088f8e579c33
adb -s $S shell md5sum /vendor/lib64/hw/vulkan.samsung.so
adb -s $S shell "ps -A | grep -iE \"llama|coopmat\""
'
ssh yanwen.xu@sj1-dmckee-d01 'bash /sarc-c/gpusw/users/yanwen.xu/android-run/pin_freqs.sh'
```

**Expected outcome**: driver hash matches the known-good value recorded in
`.shared-context/ACTIVE-STATUS.md`; no `llama`/`coopmat` process running;
clocks report pinned 509000/2730000/663000.

## Step 4 — Extend the shader variant catalog for the shortlist

In `.artifacts/tsweep-256x256-smoketest/executorch`, append one
`shader_variants` entry per shortlisted candidate to
`backends/vulkan/runtime/graph/ops/glsl/linear_q4gsw_coopmat_tsweep.yaml`,
and one token branch per candidate to `coopmat_variant_tile()` and the
`kTokens[]` array in
`backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp` — following the
exact pattern already used for every prior `tsweep_*` entry this session
(see the file's existing entries for the format). Rebuild:

```bash
cd .artifacts/tsweep-256x256-smoketest/executorch
cmake --build cmake-out-android-vk -j$(nproc) --target install --config Release
cmake --build cmake-out-android-vk/bench --target test_coopmat_linear_bench -j$(nproc)
```

**Expected outcome**: build succeeds; any candidate that fails to compile
(as happened with 128×64/K16/4×4 this session) is recorded with
`compile_status: compile_failed` and removed from the yaml/cpp before
proceeding — never left as a build-breaking entry.

## Step 5 — Run the staged search

```bash
python3 specs/022-linear-coopmat-autotune/scripts/staged_search.py \
  --shortlist specs/022-linear-coopmat-autotune/results/shortlist.json \
  --bench-binary .artifacts/tsweep-256x256-smoketest/executorch/cmake-out-android-vk/bench/test_coopmat_linear_bench \
  --ssh-host yanwen.xu@sj1-dmckee-d01 --serial 0000088f8e579c33 \
  --out-dir specs/022-linear-coopmat-autotune/results/
```

**Expected outcome**: `round1_results.json`, `round2_results.json`,
`round3_results.json`, and `budget.json` are produced in sequence.
`budget.json`'s `configs_measured_on_hardware` never exceeds 96 at any
point (check after each round, not just at the end). If a `halted: true`
sentinel appears in any round file, stop and re-run Step 3 before resuming.

## Step 6 — Produce the final report

```bash
python3 specs/022-linear-coopmat-autotune/scripts/staged_search.py --report-only \
  --out-dir specs/022-linear-coopmat-autotune/results/
```

**Expected outcome**: `autotune-report.md` exists and satisfies the
contract in `contracts/autotune-report-schema.md` — names a winner (or
explicitly states the existing winner stands), includes the dbuf1/prior-winner
comparison table, states the correctness result for the winner, and reports
`configs_measured_on_hardware` against the SC-001 (≤96) and SC-002 (≥5x
device-time reduction) targets.

## Success check

The feature is validated end-to-end when all of the following hold
simultaneously:
- `configs_measured_on_hardware` in the final `budget.json` is ≤96.
- The winning candidate (if any) in `autotune-report.md` has a passing
  correctness result and a Round-3 `mean_gflops`/`stddev_gflops` pair.
- The report's SC-002 section shows ≥5x estimated device-time reduction
  versus the exhaustive-642 estimate.
- Every one of the 642 candidates in `configs.json` can be traced to a
  `shortlist_reason` or an elimination round/reason without re-running
  anything (spot-check 3-5 arbitrary non-shortlisted candidates against
  `shortlist.json`).
