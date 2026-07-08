# Quickstart: 8da4w Coopmat Tile/Subgroup Parameter Sweep

Real device work on the `rocky-ryzen` MiniPC. Unlike `007`, this feature
**does** require new source -- a test-owned shader template and harness,
entirely under `backends/vulkan/test/custom_ops/` (research.md Decision 1).
No production file is touched.

## Prerequisites

- `004-linear-storage-comparison` and `007-wmma-improvement-microbench`
  are complete -- their tiled-baseline and shipped-configuration numbers
  are the comparison points this feature diffs against (not re-captured).
- `test_coopmat_probe` confirms `rocky-ryzen` reports
  `min_subgroup_size: 32` / `max_subgroup_size: 64` and int8 cooperative
  matrix support (already confirmed while investigating `007`'s
  regression).

## 1. Add the test-owned shader template

Create `backends/vulkan/test/custom_ops/glsl/dq8ca_q4gsw_coopmat_sweep.{glsl,yaml}`
per research.md Decision 1 -- copy the production double-buffered
int8-coopmat logic, then add one `shader_variants` entry per row in
research.md Decision 4's table: configs 1-11 (performance candidates) plus
config 12 (the deliberate negative test, `WG_TILE_K=64`) -- 12 new
variants total. Config 0 is the already-shipped kernel, reused from
`007`'s data, not rebuilt here.

**Verify the isolation property before anything else**: confirm
`git status` shows no changes under `backends/vulkan/runtime/` or
`backends/vulkan/op_registry.py` after this step -- if it does, stop, this
violates FR-008.

## 2. Add the test-owned harness and build

Create `backends/vulkan/test/custom_ops/test_dq8ca_tile_sweep.cpp` per
research.md Decision 2, and add its `add_operator_prototype(...)` entry to
`backends/vulkan/test/custom_ops/CMakeLists.txt`.

```bash
cmake --build cmake-out-vk/backends/vulkan/test/custom_ops -j$(nproc) --target test_dq8ca_tile_sweep
```

## 3. Run the sweep phase (US1 proves one config, US2 runs the active candidates)

One process invocation per `config_id` (env var
`DQ8CA_SWEEP_CONFIG_ID`), not one invocation covering all configs -- the
test framework only catches `ShaderNotSupportedError`, not a general
pipeline crash (research.md Decision 2's second correction), so
process-level isolation is what actually guarantees one bad config can't
silently erase every other row. Configs 1, 3, 5, 7, 9, 11 (subgroup 32)
are excluded from this loop after config 1's run surfaced a real
correctness bug (research.md Decision 4's implementation revision) --
only 2, 4, 6, 8, 10, 12 run here:

```bash
LOG=specs/008-8da4w-parameter-sweep/results/raw/sweep_raw.log
: > "$LOG"
for cfg in 2 4 6 8 10 12; do
  DQ8CA_SWEEP_CONFIG_ID=$cfg \
    ./cmake-out-vk/backends/vulkan/test/custom_ops/test_dq8ca_tile_sweep \
    >> "$LOG" 2>&1
  status=$?
  if [ $status -ne 0 ]; then
    echo "SWEEP_RESULT,$cfg,,,,,,pipeline_crash,,,,,\"process exited $status\"" >> "$LOG"
  fi
done
grep -c "^SWEEP_RESULT," "$LOG"
```

(The synthesized `pipeline_crash` line only fires if the process itself
exits non-zero/crashes before emitting its own `SWEEP_RESULT` rows for
that config -- a config that runs to completion emits its own rows
per-shape as usual, matching the CSV contract.)

Expected outcome: 31 `SWEEP_RESULT,...` lines -- 5 active candidates x 6
shapes (`wq` + `w1_gate` per model, research.md Decision 3) = 30, plus
config 12's single negative-test row = 31. Spot-check that `config_id=0`
(shipped) rows are absent -- config 0 is reused from `007`'s
already-captured data, not re-run here, to avoid redundant device time.
Spot-check config 12's single row shows `outcome=correctness_failure` --
if it instead shows `measured`, STOP: this means the correctness check
itself is broken (research.md Decision 4), a more serious problem than
anything this sweep is trying to measure. Config 1's earlier 6-row run
(already captured, showing `correctness_failure` across all 6 shapes) is
kept as a separate log entry and reported in the Excluded/Out-of-Scope
section, not re-run.

## 4. Inspect SPIR-V for any newly-dispatched kernel

```bash
for k in $(grep "^SWEEP_RESULT,.*,measured," specs/008-8da4w-parameter-sweep/results/raw/sweep_raw.log | cut -d, -f12 | sort -u); do
  spirv-dis "cmake-out-vk/vulkan_compute_shaders/${k}.spv" \
    > "specs/008-8da4w-parameter-sweep/results/spirv/${k}.dis.txt"
  grep -c "OpCooperativeMatrixMulAddKHR" "specs/008-8da4w-parameter-sweep/results/spirv/${k}.dis.txt"
done
```

(`kernel_name` is CSV field 12, after the `<op>` field added during
`/speckit-analyze` remediation -- see contracts/sweep-report-schema.md.)

Expected outcome: nonzero count for every measured variant's kernel.

## 5. Validate the winning configuration(s) at the full catalog (US3)

Once the sweep-phase table identifies the best correctness-verified
configuration(s), re-run `test_dq8ca_tile_sweep` restricted to that
configuration against all 3 models' 7 `8da4w` ops (21 cases), matching
`007`'s exact catalog.

## 6. Compare and generate the report

`007`'s already-generated `wmma-improvement-report.md` carries both the
tiled-baseline and shipped-config means/stdevs per (model, op) in one
table -- simpler to parse directly than re-deriving both from `004`'s and
`007`'s separate raw CSV logs:

```bash
python specs/008-8da4w-parameter-sweep/scripts/compare_sweep.py \
  --sweep-raw-log specs/008-8da4w-parameter-sweep/results/raw/sweep_raw.log \
  --shipped-report specs/007-wmma-improvement-microbench/results/wmma-improvement-report.md \
  --out specs/008-8da4w-parameter-sweep/results/sweep-report.md
```

## 7. Sanity-check

- Every one of the 6 active configurations (5 candidates + the negative
  test) appears in the sweep-phase table -- measured or explicitly failed
  with a reason (SC-001).
- Config 12 shows `correctness_failure`, clearly labeled as the expected
  result of a deliberate negative test, not mixed in with the ranked
  candidates or mistaken for an unexpected failure.
- Configs 1, 3, 5, 7, 9, 11 appear in the Excluded/Out-of-Scope section
  with config 1's actual failure evidence, not silently absent.
- The recommendation section states plainly whether anything beat the
  shipped configuration and whether anything beat tiled -- including the
  "nothing did" case (FR-007, SC-003).
- `git status` still shows zero changes under `backends/vulkan/runtime/`
  or `op_registry.py` at the very end (FR-008, re-verified after the full
  run, not just at step 1).
