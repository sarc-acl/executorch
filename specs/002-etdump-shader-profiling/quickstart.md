# Quickstart: ETDump E2E Shader Profiling Breakdown

Validates the profiling pipeline for one (model, scheme) configuration
before running the full six-configuration sweep. Assumes
`001-minipc-baseline-benchmarks` is already complete (six `.pte` files and
`results/shapes.json` exist), and this workstream's constitution's
Environment & Build Bootstrap has been followed (`uv`-managed `.venv`
active).

## Prerequisites

- On the `rocky-ryzen` MiniPC.
- The six `.pte` files from `001` already exist under
  `specs/001-minipc-baseline-benchmarks/results/pte/`.
- Nothing else CPU/GPU-heavy running (`ps aux` clean, `free -h` shows no
  active swap) — profiling captures are as sensitive to contention as the
  `001` e2e measurements were.

## 1. Build a separate, event-tracer-enabled runner

```bash
rm -rf cmake-out-vk-profiling
cmake . -Bcmake-out-vk-profiling --preset "linux" \
    -DCMAKE_INSTALL_PREFIX=cmake-out-vk-profiling -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DEXECUTORCH_PAL_DEFAULT=posix \
    -DEXECUTORCH_BUILD_VULKAN=ON -DEXECUTORCH_BUILD_TESTS=ON \
    -DEXECUTORCH_BUILD_DEVTOOLS=ON -DEXECUTORCH_ENABLE_EVENT_TRACER=ON \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_FLAGS="-include algorithm"
cmake --build cmake-out-vk-profiling -j$(nproc) --target install --config Release

cmake examples/models/llama -Bcmake-out-vk-profiling/examples/models/llama \
    -DCMAKE_INSTALL_PREFIX=cmake-out-vk-profiling -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DEXECUTORCH_ENABLE_EVENT_TRACER=ON
cmake --build cmake-out-vk-profiling/examples/models/llama -j$(nproc)
```

**Corrections found during implementation** (both confirmed, not just
theoretical risks):
1. `examples/models/llama`'s own cmake configure needs
   `-DEXECUTORCH_ENABLE_EVENT_TRACER=ON` passed to it explicitly — it does
   not inherit this from the root build's configure.
2. Linking `llama_main` **will** fail (`undefined reference to
   executorch::etdump::ETDumpGen::...`) without an added block in
   `examples/models/llama/CMakeLists.txt` (right after the Vulkan backend
   block, ~line 189):
   ```cmake
   if(EXECUTORCH_ENABLE_EVENT_TRACER)
     add_definitions(-DET_EVENT_TRACER_ENABLED)
     list(APPEND link_libraries etdump flatccrt)
   endif()
   ```
   This is now a permanent part of that file (not a "try this if it fails"
   step) — `examples/models/llama/CMakeLists.txt` never defined
   `ET_EVENT_TRACER_ENABLED` or linked `etdump`/`flatccrt` on its own, unlike
   `executor_runner`'s CMakeLists.

Verify with a tiny smoke test before doing real captures:
```bash
ET_VK_FORCE_TILED_LINEAR=1 ./cmake-out-vk-profiling/examples/models/llama/llama_main \
  --model_path <any .pte> --tokenizer_path <tokenizer.model> --prompt "hi" \
  --max_new_tokens 1 --etdump_path /tmp/smoke.etdump
ls -la /tmp/smoke.etdump  # must be non-empty; look for "ETDump file written to:" in stdout
```

## 2. Capture prefill ETDump (one representative config)

Same `ET_VK_FORCE_TILED_LINEAR=1`, prompt, and `--num_bos 1 --temperature 0`
convention as `001`, but with `--etdump_path` added and `--max_new_tokens 1`
(the runner needs at least one decode step; this keeps it negligible) to
isolate prefill from decode. Redirect stdout to a `.log` file too — the
parsing script reads each run's `PyTorchObserver` JSON from it for the
phase's wall-clock total:

```bash
ET_VK_FORCE_TILED_LINEAR=1 ./cmake-out-vk-profiling/examples/models/llama/llama_main \
  --model_path specs/001-minipc-baseline-benchmarks/results/pte/llama-3.2-1b_4w.pte \
  --tokenizer_path /home/doremy/checkpoints/llama3_2_1b/original/tokenizer.model \
  --prompt_file specs/001-minipc-baseline-benchmarks/results/prompts/shared_2048.txt \
  --num_bos 1 --temperature 0 \
  --max_new_tokens 1 --seq_len 3072 \
  --etdump_path specs/002-etdump-shader-profiling/results/etdumps/llama-3.2-1b_4w_prefill.etdump \
  > specs/002-etdump-shader-profiling/results/etdumps/llama-3.2-1b_4w_prefill.log 2>&1
```

## 3. Capture a short decode window (same config)

Use 7-8 steps (Research Decision 5), not the full 1024 `001` used:

```bash
ET_VK_FORCE_TILED_LINEAR=1 ./cmake-out-vk-profiling/examples/models/llama/llama_main \
  --model_path specs/001-minipc-baseline-benchmarks/results/pte/llama-3.2-1b_4w.pte \
  --tokenizer_path /home/doremy/checkpoints/llama3_2_1b/original/tokenizer.model \
  --prompt_file specs/001-minipc-baseline-benchmarks/results/prompts/shared_2048.txt \
  --num_bos 1 --temperature 0 \
  --max_new_tokens 8 --seq_len 3072 \
  --etdump_path specs/002-etdump-shader-profiling/results/etdumps/llama-3.2-1b_4w_decode.etdump \
  > specs/002-etdump-shader-profiling/results/etdumps/llama-3.2-1b_4w_decode.log 2>&1
```

Expected outcome: `ETDump file written to: ...` logged for each run (check
the `.log` file), and a `PyTorchObserver {...}` JSON line whose
`generated_tokens` may be one less than requested (same off-by-one `001`
saw) — fine for a "short window".

## 4. Parse and aggregate

```bash
python specs/002-etdump-shader-profiling/scripts/parse_etdump.py \
  --model llama-3.2-1b --scheme 4w \
  --prefill-etdump specs/002-etdump-shader-profiling/results/etdumps/llama-3.2-1b_4w_prefill.etdump \
  --prefill-stats-log specs/002-etdump-shader-profiling/results/etdumps/llama-3.2-1b_4w_prefill.log \
  --decode-etdump specs/002-etdump-shader-profiling/results/etdumps/llama-3.2-1b_4w_decode.etdump \
  --decode-stats-log specs/002-etdump-shader-profiling/results/etdumps/llama-3.2-1b_4w_decode.log \
  --decode-window-steps 7 --prefill-tokens 2048 \
  --baseline-json specs/001-minipc-baseline-benchmarks/results/raw/llama-3.2-1b_4w.json \
  --out specs/002-etdump-shader-profiling/results/raw/llama-3.2-1b_4w.json \
  --raw-out-dir specs/002-etdump-shader-profiling/results/raw
```

This uses `Inspector(etdump_path=...)` with no `etrecord=` (Research
Decision 3) and derives shape from the Vulkan delegate's embedded per-dispatch
JSON (Research Decision 2). **Two non-obvious things the script handles that
you'd otherwise get wrong** (see `research.md`'s corrections under Decisions
1-2): the decode `.etdump` also contains the seeding prefill call in its own
event block (must be excluded, not summed in); and M cannot be read from the
tensor JSON for dynamic-shape exports (it reports the static allocation
bound) — M must come from which kernel fired (`gemv`/`_coop_` ⇒ M=1,
`gemm`/`_tiled` ⇒ the real prefill M).

Expected outcome: `results/raw/llama-3.2-1b_4w.json` matching
`contracts/profiling-report-schema.md`, plus the raw per-invocation
companion files it references, and console output showing `attributed_pct`
for both phases.

## 5. Sanity-check reconciliation and shapes

- Confirm `attributed_pct` is a plausible majority (not e.g. 5%, which would
  indicate the parser is missing most events, or >>100%, which would indicate
  a different phase's events leaked in).
- Spot-check a few `aggregated[].shape` entries against
  `specs/001-minipc-baseline-benchmarks/results/shapes.json` for the same
  model — they should match the known `(K, N)` pairs for at least the
  largest-time entries (e.g. `w1_gate`/`w3_up`/`lm_head`), and
  `invocation_count` should match `n_layers × (repeats per layer) ×
  (decode steps, for the decode phase)`.

## 6. Category rollup and report

```bash
python specs/002-etdump-shader-profiling/scripts/category_rollup.py \
  --raw-json specs/002-etdump-shader-profiling/results/raw/llama-3.2-1b_4w.json \
  --shapes-json specs/001-minipc-baseline-benchmarks/results/shapes.json

python specs/002-etdump-shader-profiling/scripts/generate_report.py \
  --raw-dir specs/002-etdump-shader-profiling/results/raw \
  --out specs/002-etdump-shader-profiling/results/profiling-report.md
```

`results/profiling-report.md` should show a section for
`llama-3.2-1b` / `4w` with a category rollup and a top-kernels table, linking
back to `001`'s `baseline-report.md` and to this config's raw JSON. Category
percentages plus the `unattributed` remainder should sum to ~1.0 (the script
prints a warning if not).

Once validated on this one configuration, repeat steps 2–6 for the
remaining five, then re-run step 6's `generate_report.py` once (it reads all
raw JSON files in one pass).
