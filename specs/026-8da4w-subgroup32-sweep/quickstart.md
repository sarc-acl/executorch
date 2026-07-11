# Quickstart: Validate the 8da4w subgroup32-Reopen Sweep End-to-End

## Prerequisites

- Execution worktree: the **existing** `dbuf-int8-sweep` worktree
  (`023-8da4w-int8-dbuf-sweep-impl` branch) — reused deliberately, not a fresh worktree off
  `dev` (research.md Decision 5, a documented deviation from `025`'s own precedent). It
  already has `025`'s `linear_dq8ca_q4gsw_coopmat_tsweep.{glsl,yaml}` +
  `QuantizedLinear.cpp` dispatch extension uncommitted, plus this session's ad-hoc
  `sg32test` probe (to be superseded — Step 4).
- `cmake-out-android-vk` must be installed (not just built) before the bench subproject can
  link against it — this is the gap research.md Decision 4 documents:
  ```bash
  export ANDROID_NDK_HOME=/local/yanwen.xu/android-ndk-r29
  export ANDROID_NDK=$ANDROID_NDK_HOME
  source .venv/bin/activate
  cmake --build cmake-out-android-vk -j"$(nproc)" --target install --config Release
  ```
  Then configure the bench subproject once (skip if `cmake-out-android-vk/bench` already
  exists and is configured):
  ```bash
  GLSLC=/local/yanwen.xu/vulkan-sdk/1.4.350.1/x86_64/bin/glslc
  cmake backends/vulkan/test/custom_ops -Bcmake-out-android-vk/bench \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
    -DCMAKE_PREFIX_PATH=$(pwd)/cmake-out-android-vk -DCMAKE_BUILD_TYPE=Release \
    -DGLSLC_PATH=$GLSLC -DPYTHON_EXECUTABLE=python \
    -DCMAKE_CXX_FLAGS="-include algorithm" \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
  ```
- SSH/adb access to M5 EVT1 — either board is acceptable (spec Assumptions), record which
  one per result:
  - Primary: `ssh yanwen.xu@sj1-dmckee-d01`, serial `0000088f8e579c33`.
  - Secondary (shared, used for this session's re-verification): `ssh xgpusw-debug08`,
    `ANDROID_SERIAL=00000bf70c579c33` (mandatory env var — host shows 15 devices).
- Python 3 on the build box (standard library only, matching `022`/`025`'s scripts).

## Step 0 — Read `025`'s dbuf loop-structure winner (not re-measured)

```bash
cat specs/025-8da4w-parameter-sweep/results/... # locate the recorded dbuf2 confirmation
```

**Expected outcome**: the `dbuf2` winner and its supporting measurement are read directly
from `025`'s own results — this feature does not re-run Step 0-equivalent on-device work
(spec Assumptions: loop structure and geometry are separable axes, already settled by `025`).

## Step 1 — Probe subgroup=32 legality across multiple tile shapes (User Story 1)

Pick ≥5 tile shapes spanning small/medium/large (not just the shipped `128×64/K32/2×2`
shape T014 and this session's initial probe used). For each, add a `SUBGROUP_SIZE: 32`
variant to `linear_dq8ca_q4gsw_coopmat_tsweep.yaml` (folding in the session's existing
`sg32test` entry as one of the 5+, since it already covers the shipped shape) and rebuild:

```bash
cmake --build cmake-out-android-vk/bench --target test_coopmat_linear_bench -j"$(nproc)"
```

Push and run each, checking pipeline creation succeeds (no crash) before anything else:

```bash
ssh xgpusw-debug08 'export ANDROID_SERIAL=00000bf70c579c33; \
  adb shell md5sum /vendor/lib64/hw/vulkan.samsung.so'   # Principle VIII pre-flight
scp cmake-out-android-vk/bench/test_coopmat_linear_bench \
  xgpusw-debug08:/tmp/test_coopmat_linear_bench_sg32sweep
ssh xgpusw-debug08 'export ANDROID_SERIAL=00000bf70c579c33; \
  adb push /tmp/test_coopmat_linear_bench_sg32sweep /data/local/tmp/llama_vk/ && \
  adb shell chmod 755 /data/local/tmp/llama_vk/test_coopmat_linear_bench_sg32sweep'
for token in <tile-shape-tokens-with-s32>; do
  ssh xgpusw-debug08 "export ANDROID_SERIAL=00000bf70c579c33; \
    adb shell 'cd /data/local/tmp/llama_vk && ET_VK_DQ8CA_COOPMAT_VARIANT=$token \
    COOPMAT_BENCH_CORRECTNESS_ONLY=1 ./test_coopmat_linear_bench_sg32sweep'"
done
```

**Expected outcome**: `subgroup32_legality.json` (per `contracts/sweep-report-schema.md`
§0) with one entry per attempted shape/tile combination. Confirms or narrows this session's
finding that the historical crash does not reproduce on driver `c9861e9906…`/`f14c51b6f8` —
but this time across a spread, not one shape.

## Step 2 — Re-derive the legal space with subgroup_size open

```bash
python3 specs/026-8da4w-subgroup32-sweep/scripts/enumerate_configs.py \
  --loop-structure dbuf2 \
  --subgroup-legality specs/026-8da4w-subgroup32-sweep/results/subgroup32_legality.json \
  --out specs/026-8da4w-subgroup32-sweep/results/configs.json
```

**Expected outcome**: `configs.json` contains both `subgroup_size: 32` and `subgroup_size:
64` entries (research.md Decision 1 — the opposite of `025`'s contract). Spot-check: the
file contains `025`'s winning token (`tsweep_t128x32k16g12s64`) and its `s32` counterpart.

## Step 3 — Score and shortlist

```bash
python3 specs/026-8da4w-subgroup32-sweep/scripts/score_and_shortlist.py \
  --configs specs/026-8da4w-subgroup32-sweep/results/configs.json \
  --out specs/026-8da4w-subgroup32-sweep/results/shortlist.json
```

**Expected outcome**: `shortlist.json` has one entry per `configs.json` candidate.
`025`'s winning token is `shortlisted: true` regardless of rank
(`shortlist_reason: "anchor:025-winner"`). Zero on-device measurement occurs at this step.

## Step 4 — Broaden correctness gating across the full representative shape set

This is the feature's core new step — has no `025` equivalent (research.md Decision 2).
For every shortlisted candidate that compiles, run the existing correctness harness's full
multi-shape matrix (not a single-shape check):

```bash
ssh xgpusw-debug08 'export ANDROID_SERIAL=00000bf70c579c33; \
  adb shell md5sum /vendor/lib64/hw/vulkan.samsung.so'   # re-verify before this round too
for token in <shortlisted-tokens>; do
  ssh xgpusw-debug08 "export ANDROID_SERIAL=00000bf70c579c33; \
    adb shell 'cd /data/local/tmp/llama_vk && ET_VK_DQ8CA_COOPMAT_VARIANT=$token \
    COOPMAT_BENCH_CORRECTNESS_ONLY=1 ./test_coopmat_linear_bench_sg32sweep'"
done
python3 specs/026-8da4w-subgroup32-sweep/scripts/parse_correctness_matrix.py \
  --raw-logs specs/026-8da4w-subgroup32-sweep/results/raw/ \
  --out specs/026-8da4w-subgroup32-sweep/results/correctness_matrix.json
```

**Expected outcome**: `correctness_matrix.json` (per `contracts/sweep-report-schema.md`
§3) with a per-shape breakdown for every candidate — every candidate's shape key set is
identical. At least one previously-untested `M=256` family shape is included, so a
regression at that shape (as this session found for the shipped tile shape) cannot be
missed by a narrower shape set. Now retire the session's ad-hoc `sg32test` binding once its
one covered shape/tile combination is confirmed subsumed by this broader matrix (spec
FR-012).

## Step 5 — Run the staged performance search (correctness-surviving candidates only)

```bash
python3 specs/026-8da4w-subgroup32-sweep/scripts/staged_search.py \
  --shortlist specs/026-8da4w-subgroup32-sweep/results/shortlist.json \
  --correctness-matrix specs/026-8da4w-subgroup32-sweep/results/correctness_matrix.json \
  --bench-binary cmake-out-android-vk/bench/test_coopmat_linear_bench \
  --ssh-host xgpusw-debug08 --serial 00000bf70c579c33 \
  --budget-cap-formula "min(round(0.15*N), 30)" \
  --out-dir specs/026-8da4w-subgroup32-sweep/results/
```

**Expected outcome**: `round1_results.json`, `round2_results.json`, `round3_results.json`,
and `budget.json`. The script refuses to emit a `MeasurementResult` for any candidate whose
`correctness_matrix.json` entry has `all_shapes_pass: false` — spot-check this by confirming
none of this session's known-failing `M=256` candidates appear in any round file.
`budget.json`'s `configs_measured_on_hardware` never exceeds `budget_cap`.

## Step 6 — Produce the final report and the Principle V shader-comment diff

```bash
python3 specs/026-8da4w-subgroup32-sweep/scripts/staged_search.py --report-only \
  --out-dir specs/026-8da4w-subgroup32-sweep/results/
```

**Expected outcome**: `sweep-report.md` exists and satisfies
`contracts/sweep-report-schema.md` §6 — states `axis_disposition` up front, includes the
correctness matrix, the speedup-vs-`025`-winner table, the probe-disposition statement
(FR-012/SC-007), and a proposed diff updating `linear_dq8ca_qw_coopmat.glsl`/`.yaml`'s
header comment (research.md Decision 6) to reflect actual, shape-broad evidence instead of
the current stale blanket-crash claim.

## Success check

The feature is validated end-to-end when all of the following hold simultaneously:

- `subgroup32_legality.json` covers ≥5 tile shapes, and `sweep-report.md` states
  `axis_disposition` explicitly (win / legal-but-no-improvement / illegal-confirmed).
- `correctness_matrix.json` has an identical shape-key set for every candidate, and zero
  candidates with `all_shapes_pass: false` appear in any `round{1,2,3}_results.json`.
- `configs_measured_on_hardware` in the final `budget.json` is `<= budget_cap`.
- If a winner is reported, it has a Round-3 `mean_gflops`/`stddev_gflops` pair with
  `run_count == 3` and an explicit `subgroup_size_used` value.
- The report states the `sg32test` probe's disposition (superseded-and-removed, or retained
  with reason) and includes the proposed shader-comment diff.
- Any candidate in `configs.json` can be traced to a `shortlist_reason`, correctness
  disposition, or elimination round/reason without re-running anything.
