# COOPMAT_WORKFLOW — exact reproduction of the coopmat experiment

This doc is the step-by-step recipe to reproduce the L=32 S=128 coopmat run from scratch. If everything's already built, skip to "Step 3" (re-export the .pte) or "Step 4" (bench).

## What you need

- `pavan-report` branch checked out at `/home/doremy/sarc-acl/executorch/pavan-report/executorch/`. Verify:
  ```
  git -C /home/doremy/sarc-acl/executorch/pavan-report/executorch branch --show-current
  # → pavan-report
  ```
- 28+ GiB RAM, 20+ GiB free swap. 32L export peaks ~16 GiB Python RAM.
- About 25–30 min total wallclock if starting cold (build runner + export + bench + ETDump).

## Step 1: build pavan-report's runner (one-time setup)

### 1a. Configure + build with EVENT_TRACER

```bash
cd /home/doremy/sarc-acl/executorch/pavan-report/executorch
.venv/lib/python3.12/site-packages/cmake/data/bin/cmake . -Bcmake-out-vk \
    --preset linux \
    -DCMAKE_INSTALL_PREFIX=cmake-out-vk \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_VULKAN=ON \
    -DEXECUTORCH_BUILD_DEVTOOLS=ON \
    -DEXECUTORCH_ENABLE_EVENT_TRACER=ON \
    -DEXECUTORCH_ENABLE_LOGGING=OFF
.venv/lib/python3.12/site-packages/cmake/data/bin/cmake --build cmake-out-vk -j$(nproc) --target install
```

**Use the venv's CMake 3.31**, not system CMake (may be 3.22, too old for some presets).

### 1b. Fix missing `<algorithm>` includes (if build fails)

`pavan-report`'s branch predates a newer-GCC fix and is missing `#include <algorithm>` in 11 files. The error will look like:

```
backends/vulkan/runtime/graph/containers/SharedObject.cpp:16:19:
  error: no matching function for call to 'find(...)'
```

Fix with this one-shot Python script (idempotent — safe to re-run):

```python
python3 -c "
import os, re
root = '/home/doremy/sarc-acl/executorch/pavan-report/executorch/backends/vulkan/runtime'
patterns = re.compile(r'std::(find|sort|rotate|reverse|transform|fill|copy|count|min_element|max_element|unique|partition|all_of|any_of|none_of|for_each|swap|max|min)\b')
includes = re.compile(r'#include\s*<algorithm>')
for d, _, files in os.walk(root):
    for fn in files:
        if not fn.endswith(('.cpp', '.h', '.hpp')): continue
        p = os.path.join(d, fn)
        try: txt = open(p).read()
        except: continue
        if patterns.search(txt) and not includes.search(txt):
            lines = txt.splitlines(keepends=True)
            last_inc = max((i for i, l in enumerate(lines) if l.startswith('#include')), default=-1)
            if last_inc >= 0:
                lines.insert(last_inc + 1, '\n#include <algorithm>\n')
                open(p, 'w').writelines(lines)
                print('FIXED:', p)
"
```

Then re-run the build command. Should now succeed.

### 1c. Verify

```bash
ls -la /home/doremy/sarc-acl/executorch/pavan-report/executorch/cmake-out-vk/executor_runner
# Expected: ~18 MB executable

grep -c 'linear_coopmat\|matmul_coopmat\|addmm_khr_cm' \
  /home/doremy/sarc-acl/executorch/pavan-report/executorch/cmake-out-vk/vulkan_compute_shaders/spv.cpp
# Expected: > 20 (coopmat SPIR-V entries present)
```

## Step 2: verify coopmat scripts exist

```bash
ls /home/doremy/sarc-acl/executorch/main/executorch/yanwen/scripts/coopmat/
# Expected:
#   run_llama31_coopmat.py
#   setup_llama31_coopmat.py
#   bench_llama31_coopmat.py
```

If missing (they vanished once during this session — see `GOTCHAS.md`), recreate them from the existing baseline scripts. The coopmat variants are thin wrappers that:
- Set `base.RUNNER = pavan-report's executor_runner path`
- Set `base.DEFAULT_OUT = /home/doremy/llama31_pure_run_coopmat/`
- Override `export_pte()` to pass `compile_options={"storage_type_override": VkStorageType.BUFFER}` to `VulkanPartitioner`

See `scripts/coopmat/run_llama31_coopmat.py` for the exact pattern.

## Step 3: re-export the .pte (heavy: ~5 min, ~16 GiB Python RAM)

**Activate pavan-report's venv first** — main's venv has a different partitioner.

```bash
source /home/doremy/sarc-acl/executorch/pavan-report/executorch/.venv/bin/activate
sudo swapon /swapfile     # required for 32L export

cd /home/doremy/sarc-acl/executorch/main/executorch
python yanwen/scripts/coopmat/setup_llama31_coopmat.py --n_layers 32 --seq_len 128
```

Output: `/home/doremy/llama31_pure_run_coopmat/llama31_8b_32L_seq128_fp16.pte` (16.06 GB). Same size as baseline; weights dominate. The difference vs baseline `.pte` is in serialized storage tags — baseline tags activations as buffer + weights as texture2d, this one tags both as buffer.

**Critical**: the baseline `.pte` at `/home/doremy/llama31_pure_run/llama31_8b_32L_seq128_fp16.pte` **cannot be reused** for coopmat. Storage tags are baked at export time. The two `.pte` files must coexist in separate directories.

## Step 4: bench (scientific mode)

```bash
source /home/doremy/sarc-acl/executorch/pavan-report/executorch/.venv/bin/activate

cd /home/doremy/sarc-acl/executorch/main/executorch
python yanwen/scripts/coopmat/bench_llama31_coopmat.py --n_layers 32 --seq_len 128
```

Default: 1 calibration (N=1) + 3 measurement (N=8). Total wallclock ~2 min.

Expected output:

```
=== Calibration: N=1 ===
W1 = 19000–22000 ms

=== Measurement: 3 reps × N=8 ===
rep 1: steady ≈ 580 ms
rep 2: steady ≈ 580 ms
rep 3: steady ≈ 580 ms

=== Steady-state forward ===
mean ± stdev: 580–585 ± <5 ms (cv < 1%)
```

If the steady value is significantly different (>5%) from the historical 582.6 ms, investigate. Possible causes: pavan-report's runner not built fresh, .pte not re-exported, system thermal throttle, RAM contention from other processes.

## Step 5: capture ETDump (for shader breakdown)

```bash
python yanwen/scripts/coopmat/bench_llama31_coopmat.py --n_layers 32 --seq_len 128 \
    --num_executions 8 --etdump-analyze
```

This uses legacy `--num_executions N` mode (single subprocess), required because scientific mode's calibration subprocess intentionally turns off ETDump capture. The wallclock here is `wallclock/N` (inflated), but the captured ETDump is what matters.

Output files in `/home/doremy/llama31_pure_run_coopmat/`:
- `llama31_8b_32L_seq128_fp16.etdp` (binary ETDump, ~20 MB)
- `llama31_8b_32L_seq128_fp16.events.tsv` (Inspector dataframe, ~7 MB)
- `llama31_8b_32L_seq128_fp16.memprobe.tsv` (memory samples)

## Step 6: verify coopmat actually fired (trust-but-verify gate)

The runner prints `[VK_LINEAR] Using linear_coopmat ...` to stderr for each dispatch. Count:

```bash
grep -c 'linear_coopmat' /home/doremy/llama31_pure_run_coopmat/logs/bench_L32_S128_*.log
# Expected: many (~224 per measurement subprocess × multiple subprocesses)

grep -c 'matmul_coopmat' /home/doremy/llama31_pure_run_coopmat/logs/bench_L32_S128_*.log
# Expected: ~32 per measurement subprocess (attention BMMs)

grep -c 'linear_vec' /home/doremy/llama31_pure_run_coopmat/logs/bench_L32_S128_*.log
# Expected: 2 per measurement subprocess (lm_head fallback only, M=1)
```

If `linear_coopmat` count is 0:
- pavan-report's venv wasn't active when running setup (check: `which python` should show inside `pavan-report/.venv`)
- The wrong runner was invoked (check `bench_llama31_coopmat.py:RUNNER` value)
- The `.pte` wasn't re-exported with the buffer override (re-run setup)
- The GPU doesn't support `VK_KHR_cooperative_matrix` (unlikely on 780M; verify with `vulkaninfo | grep -i cooperative`)
- Env var `VK_DISABLE_COOPMAT` is set (unset it)

## Step 7: analyze the ETDump

```bash
# Categories breakdown
python /home/doremy/sarc-acl/executorch/pavan-report/executorch/yanwen_plan/analyze_etdump.py \
    /home/doremy/llama31_pure_run_coopmat/llama31_8b_32L_seq128_fp16.etdp

# Linear dispatches by output shape
python yanwen/scripts/linear_by_shape.py \
    /home/doremy/llama31_pure_run_coopmat/llama31_8b_32L_seq128_fp16.etdp

# Per-shader steady-state (iter 1..N-1 averaged)
python yanwen/scripts/etvk_breakdown.py \
    /home/doremy/llama31_pure_run_coopmat/llama31_8b_32L_seq128_fp16.events.tsv 8
```

Expected category breakdown:
- linear: ~493 ms (56% of ETDump total)
- reshape/view: ~113 ms (13%)
- CPU↔GPU copy: ~92 ms (11%)
- CPU fallback (cat): ~78 ms (9%)
- elementwise: ~64 ms (7%)
- bmm/matmul: ~0.9 ms (0.1%) — coopmat handled attention too

## Step 8: compare to baseline

The baseline counterpart is `/home/doremy/llama31_pure_run/llama31_8b_32L_seq128_fp16.etdp` (run via main's runner with main's `bench_llama31_pure.py`). Run the same three analyzers on it for side-by-side comparison.

Headline speedup: 1.77 s baseline → 0.58 s coopmat = **3.03×**.

Detailed per-shape speedups in `../reports/L32_S128_coopmat_REPORT.md` (FFN ~3.1×, attention proj ~3.2×, K/V ~3.8×, attention BMMs ~15.6×).

## Total wallclock budget

Assuming pavan-report's runner is already built and scripts exist:

| Step | Time |
|---|---|
| 3. Re-export .pte (heavy) | ~5 min |
| 4. Scientific bench (3 reps × N=8) | ~2 min |
| 5. ETDump capture | ~25 s |
| 6. Verify dispatch | < 1 s |
| 7. Analyze | < 1 min |
| **Total (warm)** | **~8 min** |

Cold (build runner from scratch + fix algorithm includes): add ~15 min for the build itself.

## When to re-run

Re-run the experiment when:
- Switching ExecuTorch versions
- Switching Mesa / driver versions
- Switching kernel/system updates (could affect swap behavior)
- Validating a proposed change to `linear_coopmat.glsl` or the dispatch logic
- Confirming the documented results before publishing

Do NOT re-run just for fun — the numbers are stable to within ~0.5% across sessions.
