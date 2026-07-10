# Quickstart: M5 EVT1 Linear + SDPA Coopmat Microbenchmark Validation

Real device work on M5 EVT1 (not MiniPC) -- per constitution Principle X,
paste the env block from `.shared-context/instruction-for-ai/README.md`
§Conventions first (`$HOST $S $D $PTE $NFS $SC`), then follow
`.shared-context/instruction-for-ai/build.md`'s Android cross-build recipe
for anything not covered below.

## Prerequisites

- Clock pin verified bound (GFLOP/s cross-check) and driver identity
  verified (`logcat | grep SUMD` matches `.shared-context/ACTIVE-STATUS.md`)
  -- reuse `specs/015`'s already-established session state; re-verify if
  the device has rebooted since (research.md Decision 3).
- `test_coopmat_linear_bench` already builds and runs correctly on M5 EVT1
  in this repo's current HEAD (confirmed this session via the
  `_spec014`-tagged binary) -- rebuild fresh from current HEAD rather than
  reuse that binary, to rule out staleness.

## 1. Extend `kShapes` for 1B/3B (linear harness)

Edit `backends/vulkan/test/custom_ops/test_coopmat_linear_bench.cpp`:
add 1B (`dim=2048`, `ffn=8192`) and 3B (`dim=3072`, `ffn=8192`) K/N pairs
to `kShapes` alongside the existing 8B pairs, each tagged with a model
label (research.md Decision 1). Per-model `dim`/`ffn` derivation: read
each model's `params.json` at `/local/yanwen.xu/models/<id>/original/`
(same llama FFN-size formula already used this session for the `specs/015`
dispatch-gate diagnostics: `ffn = multiple_of * ceil(ffn_dim_multiplier *
int(2/3 * 4*dim) / multiple_of)`).

## 2. Wire `test_sdpa_coopmat_bench` into the build (SDPA harness)

Add a new executable target in
`backends/vulkan/test/custom_ops/CMakeLists.txt` for
`test_sdpa_coopmat_bench.cpp`, mirroring the existing
`test_coopmat_linear_bench` target (research.md Decision 2). Rebuild the
Android cross-build tree per `build.md`'s two-step recipe (core runtime +
`install`, then the `custom_ops` sub-build) -- both steps, not just the
sub-build, per this session's own recurring stale-library lesson.

## 3. Build for Android and push

```bash
# from repo root, venv active, per build.md's Android cross-build recipe
cmake --build cmake-out-android-vk --target install --config Release
cmake --build cmake-out-android-vk/backends/vulkan/test/custom_ops -j$(nproc)
```

Stage + push both binaries (paste `$HOST $S $D $NFS` from README
§Conventions):

```bash
cp cmake-out-android-vk/backends/vulkan/test/custom_ops/test_coopmat_linear_bench $NFS/runners/test_coopmat_linear_bench_016
cp cmake-out-android-vk/backends/vulkan/test/custom_ops/test_sdpa_coopmat_bench $NFS/runners/test_sdpa_coopmat_bench_016
ssh $HOST "adb -s $S push $NFS/runners/test_coopmat_linear_bench_016 $D/ && adb -s $S push $NFS/runners/test_sdpa_coopmat_bench_016 $D/ && adb -s $S shell chmod 755 $D/test_coopmat_linear_bench_016 $D/test_sdpa_coopmat_bench_016"
```

## 4. Run the linear microbenchmark

```bash
ssh $HOST "adb -s $S shell \"cd $D && ./test_coopmat_linear_bench_016\""
```

Capture both the correctness-matrix output (all `PASS`/`FAIL` lines) and
the tiled-vs-coopmat `SUMMARY` table into `results/raw/linear-m5evt1.log`.

## 5. Run the SDPA microbenchmark

```bash
ssh $HOST "adb -s $S shell \"cd $D && ./test_sdpa_coopmat_bench_016\""
```

Capture output into `results/raw/sdpa-m5evt1.log`. If it crashes for a
given model, record the exact error text under that model's case as
`blocked` (spec Edge Cases) and continue with the remaining models.

## 6. SPIR-V inspection

For each distinct coopmat kernel name observed dispatching in either run,
run `spirv-dis` against its compiled `.spv`
(`cmake-out-android-vk/vulkan_compute_shaders/<kernel_name>.spv`) and save
to `results/spirv/<kernel_name>.dis.txt`; note whether
`OpCooperativeMatrixLoadKHR`/`OpCooperativeMatrixMulAddKHR` are present.
Skip re-capturing a kernel whose `.spv` is byte-identical (via `md5sum`)
to `specs/007`/`010`'s already-cited SPIR-V.

## 7. Assemble both reports

Aggregate `results/raw/*.log` + `results/spirv/*.dis.txt` into
`results/linear-coopmat-microbench-report.md` and
`results/sdpa-coopmat-microbench-report.md`, per `contracts/microbench-report-schema.md`
and `data-model.md`. Each report's summary line should also state the
direct comparison against its MiniPC counterpart (`specs/007`'s "+60.6% /
-15.2%"; `specs/010`'s "66.8% faster").

## Expected outcome

Both reports exist, every row has dispatch + correctness confirmation and
an iteration-count-backed timing (SC-001/SC-002/SC-003), and each states
an overall summary figure directly comparable to its MiniPC counterpart
(SC-004).
