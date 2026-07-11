# Quickstart: Linear Shader Storage-Type Baseline Study

Real device work, like `001` — needs an actual build and GPU capture on the
`rocky-ryzen` MiniPC. Unlike `002`/`003`/`005`, this is not pure analysis.

## Prerequisites

- `001-minipc-baseline-benchmarks` is complete (its `results/shapes.json` and
  published `microbench` numbers exist, for the Decision 4 cross-check).
- Nothing else CPU/GPU-heavy running before any capture (same discipline as
  `001`/`002` — verify with `ps aux`/`free -h` first).

## 1. Build with the modified harness

`custom_ops` is its own separate CMake project (`find_package(executorch
CONFIG REQUIRED ...)` against an already-installed build), matching the
constitution's reference recipe — it is **not** part of the root
`cmake-out-vk`/`cmake-out-vk-profiling` build trees, even though its output
lands nested under one of them:

```bash
cmake --build cmake-out-vk/backends/vulkan/test/custom_ops -j$(nproc) --target test_llama_baseline_bench
```

(Uses `001`'s original `cmake-out-vk/backends/vulkan/test/custom_ops`
sub-build, already configured per the constitution's Governance recipe. If it
doesn't exist yet, configure it first exactly as documented there.)

## 2. Capture — always with the forcing toggle set

```bash
ET_VK_FORCE_TILED_LINEAR=1 ./cmake-out-vk/backends/vulkan/test/custom_ops/test_llama_baseline_bench \
  > specs/004-linear-storage-comparison/results/raw/storage_bench_raw.log 2>&1
grep "^RESULT," specs/004-linear-storage-comparison/results/raw/storage_bench_raw.log | wc -l
```

Expected outcome: exactly 192 `RESULT,...` lines (96 cases × 2 storage
types). Every line's trailing `<kernel>` field must be a tiled/coop-family
name — spot-check a few `texture3d` AND `buffer` rows directly; if any
`buffer` row's kernel contains `coopmat`, the toggle didn't take effect
(check it's actually set in this exact shell invocation, not a different one)
and this capture must be discarded and redone before analyzing anything.

**A real bug was found and fixed during implementation, not just a
theoretical risk**: `execute_test_cases()` (in `utils.cpp`) internally groups
cases by a `ReferenceKey` that explicitly excludes `storage_type` (to reuse
reference-output computation across cases differing only in storage), and
returns `results` in group-processing order, not `generate_cases()`'s
original nested-loop order. A naive parallel loop reading `results[idx++]`
under the assumption that index order matches will silently mislabel rows —
confirmed empirically: this exact bug was already latent in `001`'s
**original, single-storage-type capture** for any pair of ops sharing an
identical `(K, N)` shape (e.g. `wq`/`wo`, `wk`/`wv` for every model), and only
became visible once this feature's storage axis made the effect large enough
to notice. The fix (already applied in the current `test_llama_baseline_bench.cpp`)
looks up each result's true identity by the name `BenchmarkResult` is seeded
with (`TestCase::name()`, via `g_case_configs`), not by position. If you ever
see two ops sharing an identical shape report suspiciously swapped-looking
numbers again, check this first before trusting the data.

## 3. Compare and generate the report

```bash
python specs/004-linear-storage-comparison/scripts/compare_storage.py \
  --raw-log specs/004-linear-storage-comparison/results/raw/storage_bench_raw.log \
  --baseline-dir specs/001-minipc-baseline-benchmarks/results/raw \
  --out specs/004-linear-storage-comparison/results/storage-comparison-report.md
```

Expected outcome: `storage-comparison-report.md` with prefill/decode verdicts
at the top, a 96-row case table, an "infeasible/contaminated" section
(ideally "none"), and a cross-check section against `001`.

## 4. Sanity-check

- The cross-check section against `001` is **expected to show divergences**
  for `wq`/`wo` and `wk`/`wv` pairs (and any other same-shape op pairs) in
  every config — these are `001`'s own pre-existing mislabeling bug (see
  step 2), not a problem with this feature's capture. The report explains
  this inline. A divergence on an op that does **not** share its shape with
  another op in the same model (`w2_down`, `lm_head`) would be unexpected and
  worth investigating.
- Confirm the "infeasible/contaminated" section is empty, or if not, that
  every listed case has a concrete stated reason (not "unknown").
- Confirm the top-line verdicts reflect the *majority* of cases, not just
  whichever direction the minority of `real_effect` cases happen to point —
  the report requires a majority of cases to show a real effect before
  characterizing the whole regime as costly/beneficial; a small number of
  isolated exceptions are listed by name instead (do not average them into
  an overall verdict).
