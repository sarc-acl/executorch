# Quickstart: M5 EVT1 8da4w T-tiled Baseline

Paste the env block from `.shared-context/instruction-for-ai/README.md`
§Conventions before any of this (`HOST`, `S`, `D`, `PTE`, `NFS`, `SC`).

## Prerequisites

- M5 EVT1 confirmed free (constitution Principle VIII / gotcha G10 --
  confirm with the user, don't assume continuity from a prior session).
- Driver identity re-verified (`adb -s $S shell md5sum /vendor/lib64/hw/vulkan.samsung.so`,
  expect the current documented hash in `.shared-context/ACTIVE-STATUS.md`).
- Clocks pinned (`pin_freqs.sh`) and the pin verified bound via a quick
  GFLOP/s cross-check against an already-recorded pinned baseline.
- This repo's own `llama_main` + ETDump-enabled runner already built
  (no rebuild needed for this feature -- no source changed).

## 1. Export (per model, `research.md` Decision 1 -- default config, no `storage_override`)

```bash
cd /local/yanwen.xu/workspace/quant-perf-optimization/executorch
source .venv/bin/activate
cd /local/yanwen.xu/workspace/.pte_out   # per constitution Default Scope -- export lands here directly

# Repeat per model (1B -> 3B -> 8B, research.md Decision 3), config omits storage_override entirely:
python -m executorch.examples.models.llama.export_llm \
  --config <model's existing 8da4w export config, minus any storage_override line> \
  # ... (exact invocation matches whatever this repo's export_quant.sh / export config
  #      already uses for texture-storage exports of the other schemes -- see
  #      .shared-context/instruction-for-ai/export-pte.md, with gotcha G2's caveat
  #      that ET_VK_FORCE_BUFFER doesn't exist and is irrelevant here anyway since
  #      T-tiled is the *absence* of the buffer override)
```

Confirm the result: `llama3_2_1b_8da4w_texture_ctx3072.pte` (and the 3B/8B
equivalents) exist in `.pte_out`, sized consistently with the other
texture-storage PTEs already there.

## 2. Push + run (per model, standard 2048-prefill/1024-decode workload)

```bash
adb -s $S push $PTE/<model>_8da4w_texture_ctx3072.pte $D/
adb -s $S push $NFS/assets/tokenizer.model $NFS/assets/p2048_exact.txt $D/   # if not already staged

# 3 timed reps (research.md Decision 5), add ET_VK_EXECUTE_NODE_THRESHOLD=16 for 8B (Edge Cases):
adb -s $S shell "cd $D && [ET_VK_EXECUTE_NODE_THRESHOLD=16 ]./llama_main \
  --model_path=$D/<model>_8da4w_texture_ctx3072.pte --tokenizer_path=$D/tokenizer.model \
  --prompt_file=$D/p2048_exact.txt --num_bos=1 --max_new_tokens=1024 --ignore_eos \
  --temperature=0 --warmup=true"
```
Repeat 3x per model; record prefill/decode tok/s each time, compute mean
+ CoV.

## 3. Dispatch confirmation (separate short run, `research.md` Decision 4)

```bash
adb -s $S shell "cd $D && ./llama_main_etdump \
  --model_path=$D/<model>_8da4w_texture_ctx3072.pte --tokenizer_path=$D/tokenizer.model \
  --prompt_file=$D/p2048_exact.txt --num_bos=1 --max_new_tokens=4 --warmup=false \
  --etdump_path=$D/t_tiled_8da4w_<model>.etdp"
adb -s $S pull $D/t_tiled_8da4w_<model>.etdp $NFS/etdump/
python $SC/analyze_etdump_shaders.py $NFS/etdump/t_tiled_8da4w_<model>.etdp --by kernel
```

**Expected outcome**: the linear kernel family shown is
`linear_dq8ca_q4gsw_tiled_*` for 100% of linear calls -- zero
`linear_dq8ca_q4gsw_coopmat_*` entries. If any coopmat entry appears,
stop and treat it as the "unexpected_coopmat" edge case in `spec.md`
(escalate to the stronger `ET_VK_DEBUG_ENCODE_DISPATCH` bind-time
diagnostic from `specs/015` Decision 8, don't report the number as a
baseline).

## 4. Compute the ratio and update the report

```
speedup = <existing optimized full-stack tok/s from specs/015 data-model.md> / <this feature's mean prefill tok/s>
```

Write the result into `specs/015-m5-e2e-wmma-validation/results/{1b,3b,8b}-results.md`'s
`8da4w` row and `m5-e2e-validation-report.md`'s consolidated table, in
the same `<baseline> -> <optimized>, N.NNx` format already used for
every `4w` row.

## Expected outcome

All three `8da4w` rows in `m5-e2e-validation-report.md` show a real
numeric speedup ratio; zero "no baseline yet" cells remain in the
consolidated table.
