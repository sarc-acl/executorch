# Quickstart: MiniPC No-WMMA Baseline Benchmarks

This validates that the baseline-capture pipeline works end-to-end for one
(model, scheme) configuration before running the full six-configuration
sweep. It assumes the repo is already set up per this workstream's
constitution (`.specify/memory/constitution.md`) — `uv`-managed `.venv`
activated, and, if working from a fresh worktree, `install_executorch.sh --minimal`
already run.

## Prerequisites

- On the `rocky-ryzen` MiniPC (or an equivalent RDNA3 iGPU host).
- A downloaded Llama checkpoint + tokenizer for at least one target model
  (start with Llama 3.2 1B — smallest, fastest to validate with).
- This feature's `ET_VK_FORCE_TILED_LINEAR` toggle implemented in
  `QuantizedLinear.cpp` (Research Decision 1) and built in.

## 1. Build (Reference Build Recipe, per constitution)

```bash
rm -rf cmake-out-vk
cmake . -Bcmake-out-vk --preset "linux" \
    -DCMAKE_INSTALL_PREFIX=cmake-out-vk -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DEXECUTORCH_PAL_DEFAULT=posix \
    -DEXECUTORCH_BUILD_VULKAN=ON -DEXECUTORCH_BUILD_TESTS=ON \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_FLAGS="-include algorithm"
cmake --build cmake-out-vk -j$(nproc) --target install --config Release
```

## 2. Export one `.pte`

**Corrected flags (found during implementation)**: `-d fp16` is rejected by
the Vulkan partitioner (`AssertionError: Vulkan backend does not support non
fp32 dtypes at the moment`). Omit `-d` entirely (defaults to fp32 at the
export-graph level) and add `--vulkan-force-fp16` to actually run fp16 on the
Vulkan backend. The real flag names are `--max_seq_length` /
`--max_context_length`, not `--max_seq_len`.

```bash
python -m examples.models.llama.export_llama \
  --model llama3_2 \
  -c <path/to/llama-3.2-1b/original/consolidated.00.pth> \
  -p <path/to/llama-3.2-1b/original/params.json> \
  -t <path/to/llama-3.2-1b/original/tokenizer.model> \
  -kv --use_sdpa_with_kv_cache \
  -qmode 4w --group_size 32 \
  --max_seq_length 3072 --max_context_length 3072 \
  -V --vulkan-force-fp16 \
  -o specs/001-minipc-baseline-benchmarks/results/pte
```

This writes `<output-dir>/llama3_2.pte` (the export tool names the file after
`--model`, not the scheme) — rename it immediately (e.g. to
`llama-3.2-1b_4w.pte`) before exporting the next scheme into the same
directory, or use a per-config output directory, to avoid overwriting.

Confirm the `.pte` exists and record its path in
`results/raw/llama-3.2-1b_4w.json`'s `config.pte_path` (see
`contracts/baseline-report-schema.md`).

## 3. Build the tokenizer-verified 2048-token prompt

**Simplification found during implementation**: all three checkpoints ship a
byte-identical `tokenizer.model` (verified by `md5sum`), so a single shared
prompt file works for all three models — no need for one per model. Encode a
fixed source passage with the tokenizer (`bos=True`) and trim/repeat until it
encodes to exactly 2048 tokens; save it to
`specs/001-minipc-baseline-benchmarks/results/prompts/shared_2048.txt`.
Verify with the same tokenizer before using it for step 4 — token count must
be exact, not approximate (spec FR-002).

## 4. Run the e2e baseline

**Resource-contention warning (found the hard way): never run this
concurrently with a `.pte` export or another benchmark.** Even though
exports are CPU-bound, this iGPU shares memory bandwidth with the CPU, and a
concurrent export measurably slows down the GPU benchmark (~32% slower
decode tok/s was observed once, coinciding with the system swapping). Always
confirm `ps aux` shows no other export/benchmark process and `free -h` shows
no active swap growth before starting.

```bash
ET_VK_FORCE_TILED_LINEAR=1 ./cmake-out-vk/examples/models/llama/llama_main \
  --model_path specs/001-minipc-baseline-benchmarks/results/pte/llama-3.2-1b_4w.pte \
  --tokenizer_path <path/to/tokenizer.model> \
  --prompt_file specs/001-minipc-baseline-benchmarks/results/prompts/shared_2048.txt \
  --num_bos 1 --temperature 0 --warmup true \
  --max_new_tokens 1024 --seq_len 3072
```

`--num_bos 1` matters: it must match how the prompt file's token count was
verified in step 3 (`bos=True`) — the runner's default (`--num_bos 0`) does
not prepend a BOS token and would silently shift `prompt_tokens` by one.
`--warmup true` runs an internal discarded warmup pass, satisfying FR-005's
explicit-warmup requirement for this tier.

Expected outcome: a `PyTorchObserver {...}` JSON line containing
`prefill_token_per_sec`, `decode_token_per_sec`, and `prompt_tokens` (should
read exactly 2048). Repeat 5 times per FR-005 — a single run is not
sufficient evidence, and the first 1-2 runs after GPU idle may read
noticeably faster (a real, reproducible warm-up effect, not just noise; see
`baseline-report.md`'s Observations section) — report the converged
steady-state mean/stdev, not a single cold run.

## 5. Run the microbenchmark tier

```bash
cmake backends/vulkan/test/custom_ops/ -Bcmake-out-vk/backends/vulkan/test/custom_ops \
    -DCMAKE_INSTALL_PREFIX=cmake-out-vk -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DEXECUTORCH_ROOT=$(pwd) \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cmake --build cmake-out-vk/backends/vulkan/test/custom_ops -j$(nproc) --target test_llama_baseline_bench
./cmake-out-vk/backends/vulkan/test/custom_ops/test_llama_baseline_bench \
  | tee specs/001-minipc-baseline-benchmarks/results/microbench_raw.log
```

This runs all 96 cases (3 models × 2 schemes × 2 regimes × 8 ops) in one
invocation and prints one `RESULT,<model>,<scheme>,<regime>,<op>,...` line
per case. Same resource-contention warning as step 4 applies — run it alone.

## 6. Confirm the result lands in the report

`results/raw/llama-3.2-1b_4w.json` should now match the shape in
`contracts/baseline-report-schema.md`, and
`results/baseline-report.md` should show a row for `llama-3.2-1b` / `4w`.

Once this single configuration is validated end-to-end, repeat steps 2–6 for
the remaining five (model, scheme) combinations.
