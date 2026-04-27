# Real LLaMA seq=2048 prefill is OOM-killed on a 28 GB host

## What was attempted

To match the previous-story baseline numbers
(`yanwen_docs/background/1_previous_story.md`: tex 5984 ms, buf 16131 ms at
seq=2048) on the current branch, I exported real LLaMA 3.1 8B with 4 layers,
fp16, seq=2048 via the helper:

```bash
timeout 1500 python3 scripts/run_real_llama_e2e_patched.py \
  --local --executor_runner cmake-out-vk-etdump/executor_runner \
  --fp16 --n_layers 4 --seq_len 2048 --runs 4 --only tex \
  --cache_dir ~/llama3_1_8b ...
```

The .pte was generated successfully
(`llama_tmp/llama31_8b_4L_seq2048_fp16_texture3d.pte`, 3.85 GB).

## What happened

`executor_runner` exited with `Killed` (exit 137 = SIGKILL by OOM killer):

```text
==== real_seq2048_tex_stephen ====
.../run_real_seq2048.sh: line 16: 1402994 Killed   env "$RUNNER" \
  --model_path="$PTE" --inputs="$IN0" --num_executions=1 \
  --etdump_path="$etdp" --print_output=none > "${stem}.etdump_run.log" 2>&1
```

Resident memory before launch: ~9 GB used / 19 GB available on a 28 GB
machine. The portable `executor_runner` reads the entire 3.85 GB .pte into
heap memory (does not mmap), then allocates planned memory regions for
activations on top. With seq=2048 the attention scores tensor is
`[1, 32, 2048, 2048]` = 256 MB per layer in fp16, plus several
permute/clone temporaries; the working set crosses the available RAM ceiling
once Vulkan staging buffers are also considered.

## Why it matters

The user's primary workload is real LLaMA fp16 seq=2048 prefill (matching the
previous-story baseline). On this 28 GB host that workload is unmeasurable
with the current portable runner. The Phase 2 study therefore could not
validate the buffer-trap regression at seq=2048 on the current branch, and
the conclusions about buffer-vs-texture storage at seq=2048 must rely on the
previous-story numbers rather than fresh measurements.

## Concrete impact on Phase 2 conclusions

- All Phase 2 wallclock numbers are at seq=256.
- At seq=256 the buffer-coopmat path actually beats the texture-coopmat path
  by ~1.22×. That is the *opposite* of the seq=2048 previous-story finding.
  The most likely explanation is that the per-op buffer-storage tax (softmax,
  ETVK_COPY, host-side dispatch gap) scales with sequence length squared
  (softmax) and per-token (copy/coherency), so at seq=256 the tax is ~64×
  smaller in attention and ~8× smaller per copy than at seq=2048.
- The Phase 2 report explicitly does **not** recommend whole-graph buffer
  storage based on the seq=256 wins, because we could not confirm the
  buffer-trap behaviour at the workload size that originally exposed it.

## Recommended next action

For a future Phase 2/3 session that needs seq=2048 numbers:

1. Run on a host with at least 48 GB system RAM (or use a discrete GPU host
   with a separate VRAM budget).
2. Or, modify `executor_runner` (or write a small alternative runner) that
   `mmap()`s the .pte instead of reading it into the heap. The portable
   runner reads via `fread`, which doubles peak RSS for large .pte files.
3. Or, partition the LLaMA forward in Python and run only the heavy linear
   subgraphs to estimate the buffer-storage tax under seq=2048 dispatch
   patterns without keeping the full graph resident.

Until one of those is in place, all real-LLaMA seq=2048 buffer-vs-texture
numbers should be cited from
`yanwen_docs/background/1_previous_story.md` and
`/home/doremy/Desktop/samsung/executorch/yanwen_plan/igpu_results/` (the
previous round's captures), not from this Phase 2 round.
