# GOTCHAS — bugs, traps, and methodological mistakes we corrected

This is the "things that bit us during this session" doc. Each item: **symptom → cause → fix**. Read this before doing anything non-obvious; many of these took real session time to discover.

## 1. `analyze()` in `run_llama31_pure.py` hides kernel events and mislabels units

**Symptom**: The top-30 op list from `analyze()` shows wrappers (`load_method`, `Method::execute`, `DELEGATE_CALL`, `OPERATOR_CALL`) and operator-level events (`aten.linear.default`), but no `linear_vec_*` or `linear_coopmat_*` kernel rows. Also, the column header says `mean_us` but the values are actually in milliseconds.

**Cause**: Two compounding bugs in `analyze()` in `yanwen/scripts/run_llama31_pure.py`:
1. The deduplication key uses `name.split("[")[0].split("::")[-1]`, which mangles JSON-formatted event names like `{"kernel_name": "linear_coopmat_half", "operator_id": 5, "dispatch_id": 2}` — the split-by-`[` truncates them mid-JSON.
2. `perf_data.raw` is in milliseconds (per Inspector's own column header), but `analyze()` names its variable `mean_us` and prints "us" units while dividing by 1000 — leading to the implied "ms" value being wrong by 1000× if interpreted at face value.

**Fix**: **Don't use `analyze()` from `run_llama31_pure.py`.** Use the canonical analyzer at `/home/doremy/sarc-acl/executorch/pavan-report/executorch/yanwen_plan/analyze_etdump.py`, which correctly skips wrapper events and treats `raw` values as ms. Or use `yanwen/scripts/etvk_breakdown.py` for per-shader steady-state breakdown.

## 2. `wallclock/N` overstates steady-state by 50–80% at L=32

**Symptom**: 2026-05-07 sweep reported 2.89 s/exec at L=32 N=16, but our scientific bench measured 1.77 s steady-state forward (an 80% inflation).

**Cause**: A single subprocess's wallclock is `fork + load_method + prepack + iter 0 (cold) + (N-1)*steady + teardown`. Dividing by N folds amortized `(load + iter 0 + teardown) / N` into every "per-exec" number. At L=32 this fixed cost is ~21 s, so even at N=16 it adds ~1.3 s contamination per "exec" — about 80% of true forward.

**Fix**: Use scientific mode (`bench_steady_state()` in `run_llama31_pure.py`, default mode of `bench_llama31_pure.py` without `--num_executions`). It runs a calibration subprocess at N=1 and subtracts algebraically. See `METHODOLOGY.md`.

## 3. ETDump GPU total ≠ wallclock at S=512 (memory thrash region)

**Symptom**: At L=32 S=512, ETDump category breakdown sums to 13.7 s but wallclock is 111 s (8× under-report). At L=32 S=128, ETDump and wallclock agree within 8% (normal).

**Cause**: ETDump captures per-shader GPU dispatch time via `vkCmdWriteTimestamp` query-pool. But when the host is blocked on `vkCmdCopyBuffer` waiting for staging buffer pages to fault back from swap, the GPU is idle — there's no dispatch active, no timestamp recorded. The wait shows up only in wallclock.

**Implication**: **Don't draw shader-optimization conclusions from ETDump at S≥512** on this hardware. The bottleneck is host-side memory pressure, not GPU compute, and ETDump can't see it. For diagnostic purposes, compute `wallclock - ETDump_total`; if that's >50% of wallclock, you're in cliff regime.

**Diagnostic indicator**: at the cliff regime, `ETVK_COPY_INPUTS` reports thousands of milliseconds for a tiny tensor (e.g., 3420 ms for a 4 KB input at S=512 vs 16 ms at S=128). That's the GPU stalling on page-fault, not actually copying.

## 4. pavan-report tree fails to build with newer GCC

**Symptom**: `std::find` / `std::rotate` "no matching function" errors during build, in `runtime/graph/containers/SharedObject.cpp:16`, `runtime/graph/ops/impl/Squeeze.cpp:52`, and similar files.

**Cause**: Missing `#include <algorithm>` in 11 files. They got by on older GCC because some transitively-included header pulled in `<algorithm>`; newer GCC versions removed that transitive include. Same files in `main`'s tree have the include, but pavan-report's branch predates the fix.

**Fix**: Run the auto-fix Python script from `COOPMAT_WORKFLOW.md` step 1b. It walks `backends/vulkan/runtime/`, detects files using `std::find/sort/rotate/...` without an `<algorithm>` include, and adds it after the last existing `#include`. Idempotent.

## 5. `.pte` cannot be reused across baseline and coopmat runs

**Symptom**: Running pavan-report's runner against main's `.pte` doesn't trigger coopmat. No `[VK_LINEAR] Using linear_coopmat` stderr lines; benchmark shows baseline-like timing.

**Cause**: The partitioner serializes output tensor storage types into the `.pte` at export time. Baseline `.pte` (exported with `VulkanPartitioner({})`) has activations tagged as buffer + weights as texture2d. Coopmat dispatch requires `storage_type_of(out) == kBuffer` AND the weights also need to be buffer for the prepack to set up correctly. The texture2d-tagged weights in main's `.pte` cause the runtime to fall back to `linear_vec_buffer_texture2d_half` even when pavan-report's runner has `linear_coopmat` available.

**Fix**: Re-export the `.pte` with `VulkanPartitioner({"storage_type_override": VkStorageType.BUFFER})` for coopmat runs. Keep the two `.pte` files in separate output dirs (`/home/doremy/llama31_pure_run/` for baseline, `/home/doremy/llama31_pure_run_coopmat/` for coopmat).

## 6. Coopmat's lm_head always falls back to `linear_vec`

**Symptom**: Even in a fully-working coopmat run, `events.tsv` shows 1× `linear_vec_buffer_buffer_half` per forward, alongside the 224× `linear_coopmat_half`. ETDump dispatch counts: 2 `linear_vec` mentions per measurement subprocess (1 dispatch × ~2 sub-iterations? actually 1 unique event × multiple runs).

**Cause**: The lm_head is `[1, 128256]` — M=1 (sequence batch dim of just one token). The coopmat dispatch gate is `M >= 64` (per `Linear.cpp`'s `add_linear_coopmat_node()`). The gate is needed because the cooperative-matrix shader's store has no bounds check and would OOB-write for M < 64.

**Implication**: This is **not a bug**, it's intentional. Don't try to "fix" it — fixing it would require either bounds-checking in the coopmat shader (perf hit) or a different shader variant for tiny-M cases. The lm_head accounts for 2.2% of forward time in the coopmat run; not worth chasing.

## 7. Memprobe file gets overwritten by each subprocess

**Symptom**: After a scientific bench (1 cal + 3 measurement subprocesses), the `<tag>.memprobe.tsv` file contains data from only the **last** subprocess. Trying to compute "peak memory across all reps" from this file is misleading.

**Cause**: `MemProbe` in `run_llama31_pure.py` opens `mem_log` in write mode (`'w'`) at the start of each `run_etdump()` call. Each subprocess truncates the prior probe data.

**Fix**: If you need per-rep memory traces, modify `MemProbe` to use a per-rep filename suffix (e.g., `<tag>.memprobe.rep{i}.tsv`), or copy the file out between subprocess invocations. For headline memory numbers, the last subprocess's probe is fine — peak memory is usually consistent across reps once the warm cache is established.

## 8. Today's baseline isn't what older pavan-report measurements assumed

**Symptom**: The 2026-05-06 pavan-report synthetic-LLaMA fp32 baseline used `linear_vec` + `*_texture3d_*` shaders (texture3d activations). Our 2026-05-10 real-LLaMA fp16 baseline uses `linear_vec_buffer_texture2d_half` (buffer activations). Different shader paths, different perf characteristics.

**Cause**: `main`'s `op_registry.py` evolved to declare `aten.linear.default` with `inputs_storage=utils.CONTIGUOUS_ANY`. The `pick_representations()` logic in `utils.py` resolves this to **buffer** storage for fp16 LLaMA-shaped tensors. The older pavan-report code (and older main) defaulted to `kTexture3D` for everything.

**Implication**: The 2× speedup pavan-report measured in 2026-05-06 was against a slower baseline (texture3d activations) than today's. Today's 3× coopmat speedup is measured against a faster baseline (buffer activations) and is thus more representative of the algorithmic lift. The two speedup numbers aren't directly comparable.

If a user references "the 2× number from pavan-report's prior work," remind them that the baseline has changed and today's apples-to-apples comparison is 3×.

## 9. External cleanup can wipe scripts mid-session

**Symptom**: During this session, `yanwen/scripts/coopmat/` (which I had just created) was empty when I tried to run it. I had to recreate the three .py files. An HTML report also appeared that I didn't write.

**Cause**: Unknown — likely either the user manually editing the workspace in parallel, or some hook / external tool. Not reproducible on demand.

**Fix**: Before invoking any script, verify it exists:
```bash
ls -la /home/doremy/sarc-acl/executorch/main/executorch/yanwen/scripts/coopmat/
```
If empty: recreate from the patterns shown in `COOPMAT_WORKFLOW.md` step 2, or check git history. Consider `git add`-ing important files after creation to provide a recovery path.

## 10. Linear shape `[1, 128256]` lm_head exceeds texture2d limit

**Symptom**: In the baseline run, 224 of 226 linears dispatch `linear_vec_buffer_texture2d_half`, but 2 dispatch `linear_vec_buffer_buffer_half`. Why?

**Cause**: The lm_head has output `[1, 128256]`. The packed weight tensor has width = 128256 (or 128264 after padding). `prepack_fp_linear_weight()` falls back to `kBuffer` when `output_width / 4 > maxImageDimension2D` (16384 on 780M). For lm_head: `128256 / 4 = 32064 > 16384`, so weight storage falls back to buffer. The runtime kernel name then encodes both as buffer: `linear_vec_buffer_buffer_half`.

**Implication**: This is not specific to L=32 — it's the lm_head's huge vocabulary dimension. Even at L=1 you'd see one `linear_vec_buffer_buffer_half` dispatch. Documented for clarity.

## 11. Don't use main's venv with pavan-report's partitioner code

**Symptom**: If you accidentally activate main's venv but invoke `setup_llama31_coopmat.py`, the export might still happen but the partitioner doesn't apply the buffer-forcing pass (because main's partitioner doesn't have `force_buffer_output_ops`).

**Cause**: The modified partitioner passes (`tag_memory_meta_pass.py` with `force_buffer_output_ops`, `vulkan_preprocess.py` with `force_buffer_linear_matmul`) live inside pavan-report's installed `executorch` package, NOT main's. The Python `import executorch.backends.vulkan.partitioner...` resolves to whichever venv is active.

**Fix**: Always use pavan-report's venv for coopmat setup AND coopmat bench:
```
source /home/doremy/sarc-acl/executorch/pavan-report/executorch/.venv/bin/activate
which python
# Should print: /home/doremy/sarc-acl/executorch/pavan-report/executorch/.venv/bin/python
```

Sanity check: `python -c "from executorch.backends.vulkan._passes.tag_memory_meta_pass import TagMemoryMetaPass; import inspect; print('force_buffer_output_ops' in str(inspect.signature(TagMemoryMetaPass.__init__)))"` should print `True`.

## 12. Backgrounded builds may report failure due to truncated logs

**Symptom**: First build attempt of pavan-report's runner reported "Error 2" in the tail of the output, but the full log only showed warnings, no errors. We re-ran and got the same Error 2.

**Cause**: When piping `cmake --build ... -j$(nproc) | tail -3` in background, the first chunk of error output is lost — only the tail of the BUILD output is preserved, and the `gmake Error 2` line is just the makefile reporting that some target failed. The actual `error:` lines from the C++ compiler scrolled past.

**Fix**: When debugging build failures, tee the full output to a file:
```
cmake --build cmake-out-vk -j$(nproc) --target install 2>&1 | tee /tmp/pavan_build.log
```
Then `grep -E 'error:' /tmp/pavan_build.log` to find the real failure.

## 13. Coopmat does NOT fire on decode-shape workloads (M=1)

**Symptom**: Running the coopmat-configured path (storage_type_override=BUFFER on pavan-report tree) at seq_len=1 produces forward times within 1% of baseline. The 3.03× prefill speedup vanishes entirely.

**Cause**: The dispatch gate in `Linear.cpp` is `M >= 64`. Decode has M=1 per linear (one token at a time). All linears fall back to `linear_vec_tile_row_1_buffer_texture2d_half` on BOTH paths — identical shader, identical weight storage (texture2d wins by `prepack_fp_linear_weight()`'s default since `force_buffer=use_coopmat=false`).

**Implication**: Don't expect coopmat to help decode workloads. Decode is bandwidth-bound on this hardware (~4.2 tok/s ceiling at L=32 fp16). Real decode levers are quantization, KV-cache strategies, speculative decoding, or batching (the last re-engages coopmat at N≥64). See `reports/decode_GEMV_ceiling_check.md`.

**Side note**: `matmul_coopmat` *does* fire for the attention BMMs at M=1, but those contribute < 0.1% of forward time at S=1.

## 14. L=32 seq=1 export OOMs on 28 GiB box

**Symptom**: `setup_llama31_pure.py --n_layers 32 --seq_len 1` reports success (exit 0) but produces a 0-byte .pte and no input.bin. Kernel log shows `Out of memory: Killed process (python)` at ~28 GB anon-rss right after `[export] writing .pte ->` is logged. Same fate for the coopmat variant — though coopmat occasionally squeaks through depending on system state.

**Cause**: After `to_executorch()`, the `et.buffer` (~16 GB of serialized .pte content) lives alongside the still-alive Python-side graph/tensors (~12+ GB). Total Python anon-rss peaks at ~28 GB, which is exactly system RAM. Even with 24 GB swap, the OOM-killer fires before the buffer can be flushed to disk.

**Workaround**: Use smaller L (e.g., L=4) for any seq_len=1 dispatch / decode test. The dispatch decision is per-layer so L=4 answers the same questions. For real L=32 decode performance, extrapolate per-layer time × 32 + lm_head (which is L-independent). See `reports/decode_GEMV_ceiling_check.md` for the extrapolation pattern.

**Proper fix (if needed later)**: modify `export_pte()` to `del prog, edge` before `et.buffer` is written, and stream-write the buffer in chunks instead of `f.write(et.buffer)` all at once. ~15 line change, not done in this session.

## 15. `print()` from runner subprocess prints a giant output tensor

**Symptom**: Running `executor_runner` floods stdout with `OutputX 0: tensor(sizes=[1, 128256], [...])` followed by ~128K floats. Hard to read other output.

**Cause**: The runner unconditionally prints output tensor values via the default executor_runner. Not configurable via CLI.

**Workaround**: Pipe to a file and grep what you need:
```
python bench_llama31_pure.py ... 2>&1 | tee /tmp/run.log | grep -E '\[(env|run|bench)\]'
```

The bench scripts already do `tee`-style output to `logs/` so the tensor dump is preserved there; grep that file for `[run]`/`[bench]` lines and you'll see the relevant lines without the noise.
