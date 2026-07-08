# Quickstart: WMMA Coopmat Improvement Microbenchmark

Real device work on the `rocky-ryzen` MiniPC, like `001`/`004`. Originally
planned as measurement-only with no source changes required; **that
changed during implementation** (research.md Decision 8): the `4w` scheme's
coopmat dispatch was found to be unreachable from any registered op, and
fixing it became in-scope. If you are re-running this study from a
checkout that predates that fix, you MUST rebuild the Vulkan backend first
(`QuantizedLinear.cpp`/`Q4gswLinear.cpp` changed) — a stale binary will
silently reproduce the old, wrong "0% for `4w`" result. Otherwise, this
feature reuses the already-committed `test_llama_baseline_bench.cpp`
binary and the already-captured `004` tiled baseline as-is.

## Prerequisites

- Decision 8's wiring fix is present in the checked-out source
  (`QuantizedLinear.cpp` has a `linear_q4gsw()` function registered to
  `et_vk.linear_q4gsw.default`) and the Vulkan backend has been rebuilt
  since.
- `001-minipc-baseline-benchmarks` and `004-linear-storage-comparison` are
  complete (`004`'s `results/raw/storage_bench_raw.log` is the tiled-baseline
  reference this feature diffs against — not re-captured here).
- `003-wmma-shader-candidates`'s classification data identified the
  candidate op list this feature measures (see `spec.md`'s Assumptions);
  its `pct_of_phase` values are not read at runtime by `compare_wmma.py`
  (research.md Decision 6 addendum — same-shape sibling ops like `wq`/`wo`
  can't be cleanly split from that data, so weighting uses each op's own
  measured tiled-baseline time instead).
- `test_llama_baseline_bench.cpp` is already built at
  `cmake-out-vk/backends/vulkan/test/custom_ops/test_llama_baseline_bench`
  (built for `004`; rebuild only if the build tree is stale).
- Vulkan SDK's `spirv-dis` is on `PATH` (or reference it by full path,
  e.g. `~/vulkansdk/1.4.341.1/x86_64/bin/spirv-dis`).
- Nothing else CPU/GPU-heavy running before any capture.

## 1. Capture — the same binary, WITHOUT the forcing toggle

```bash
./cmake-out-vk/backends/vulkan/test/custom_ops/test_llama_baseline_bench \
  > specs/007-wmma-improvement-microbench/results/raw/wmma_bench_raw.log 2>&1
grep "^RESULT," specs/007-wmma-improvement-microbench/results/raw/wmma_bench_raw.log | wc -l
```

Expected outcome: 192 `RESULT,...` lines (same catalog as `004`), but this
time `storage=buffer`, `regime=prefill` rows for the seven in-scope ops
(everything except `lm_head`, per `research.md` Decision 3) should show a
`kernel` field containing `coopmat` — spot-check several. A `buffer`/
`prefill` row for one of the seven in-scope ops whose kernel does **not**
contain `coopmat` is a `dispatch_status: fallback` case (FR-004) — record it,
don't discard it silently.

**Do not set `ET_VK_FORCE_TILED_LINEAR` for this capture.** That is exactly
what `004`'s already-captured data used; this capture's entire purpose is to
observe what happens when it is absent.

## 2. Inspect the dispatched kernels' compiled SPIR-V

```bash
mkdir -p specs/007-wmma-improvement-microbench/results/spirv
for k in linear_q4gsw_coopmat_buffer_buffer_half linear_dq8ca_q4gsw_coopmat_buffer_buffer_half; do
  spirv-dis "cmake-out-vk/vulkan_compute_shaders/${k}.spv" \
    > "specs/007-wmma-improvement-microbench/results/spirv/${k}.dis.txt"
  grep -c "OpCooperativeMatrixMulAddKHR" "specs/007-wmma-improvement-microbench/results/spirv/${k}.dis.txt"
done
```

Expected outcome: a nonzero count of `OpCooperativeMatrixMulAddKHR` (and
`OpCooperativeMatrixLoadKHR`) for both kernel variants — this is the
SPIR-V-level evidence required by FR-007 / constitution Principle VI. A zero
count means the compiled shader does not actually contain cooperative-matrix
instructions despite its name, which would be a serious, reportable finding
(not something to explain away).

## 3. Compare and generate the report

```bash
python specs/007-wmma-improvement-microbench/scripts/compare_wmma.py \
  --wmma-raw-log specs/007-wmma-improvement-microbench/results/raw/wmma_bench_raw.log \
  --tiled-baseline-log specs/004-linear-storage-comparison/results/raw/storage_bench_raw.log \
  --out specs/007-wmma-improvement-microbench/results/wmma-improvement-report.md
```

Expected outcome: `wmma-improvement-report.md` with the time-weighted overall
figure at the top, the 42-row case table, and the Excluded/Out-of-Scope
section per `contracts/wmma-improvement-report-schema.md`.

## 4. Sanity-check

- Confirm every one of the 42 in-scope (model, scheme, op) combinations
  appears in either the main table or the Excluded/Out-of-Scope section —
  none silently missing (FR-009, SC-001).
- Confirm `lm_head` and every decode-regime op appear in the
  Excluded/Out-of-Scope section with their stated reason, not in the main
  table.
- Confirm no row in the main table is missing its `dispatch_status`,
  `correctness_verified`, or `significance` value — a row failing any of
  these belongs in the Excluded/Out-of-Scope section instead (FR-004,
  FR-007).
- Confirm the overall figure is explicitly stated as time-weighted (not a
  bare average), per SC-005.
