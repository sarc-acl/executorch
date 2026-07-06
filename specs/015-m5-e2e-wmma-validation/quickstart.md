# Quickstart: M5 EVT1 End-to-End WMMA Validation

**Order: 1B first, report immediately, then 3B, then 8B** (Decision 3) --
do not batch all three models before reporting the first one.

## Prerequisites

- M5 EVT1 device access (`ssh yanwen.xu@sj1-dmckee-d01`, `adb -s
  0000088f8e579c33`) per Principle X / `devices-and-access.md`.
- Driver identity re-verified against `flash-sumd-driver.md`'s table
  (Decision 4) -- do not assume `specs/014`'s end-of-session state holds.
- GPU/MIF/INT clocks pinned via `pin_freqs.sh` (509/2730/663 MHz) unless a
  floating run is explicitly requested.
- If `/data/vendor/gpu/amdPalSettings.cfg` is present and active, confirm
  with the user before moving it aside (`commands.md` §10) -- do not do
  this unilaterally.
- This repo's own `.venv` active (`source .venv/bin/activate`) for export.
- This repo's own `llama_main` built (`cmake-out-android-vk/examples/
  models/llama/llama_main`) and an ETDump variant built via
  `./build_etdump_android.sh` if not already present.

## Per-model loop (run once for 1B, report, then repeat for 3B, then 8B)

### 1. Export `8da4w` if not already present for this model

```bash
source .venv/bin/activate
MODEL=<llama3_2_1b|llama3_2_3b|llama3_1_8b> MAX_SEQ=3072 MAX_CTX=3072 \
  .shared-context/scripts/export_quant.sh 8da4w 128 buffer
# -> .pte_out/<model>_8da4w_buffer_ctx3072.pte
```

Skip for `4w` -- those `.pte`s already exist (Decision 1).

### 2. Stage + push this model's PTEs, this repo's own runner, and tokenizer

```bash
cp .pte_out/<model>_4w_buffer_ctx3072.pte .pte_out/<model>_8da4w_buffer_ctx3072.pte $NFS/models/
ssh yanwen.xu@sj1-dmckee-d01
S=0000088f8e579c33; D=/data/local/tmp/llama_vk; NFS=/sarc-c/gpusw/users/yanwen.xu/android-run
adb -s $S shell mkdir -p $D
adb -s $S push $NFS/models/<model>_4w_buffer_ctx3072.pte $D/
adb -s $S push $NFS/models/<model>_8da4w_buffer_ctx3072.pte $D/
adb -s $S push <this repo's llama_main> $D/llama_main_015          # own build, NOT _origcm (research.md Decision 2)
adb -s $S push <this repo's ETDump llama_main> $D/llama_main_etdump_015
adb -s $S push $NFS/assets/tokenizer.model $NFS/assets/p2048_exact.txt $D/
adb -s $S shell chmod 755 $D/llama_main_015 $D/llama_main_etdump_015
```

### 3. Coherence check (both schemes) -- must say "Paris ..." before benchmarking

```bash
adb -s $S shell "cd $D && ./llama_main_015 --model_path=$D/<model>_4w_buffer_ctx3072.pte \
  --tokenizer_path=$D/tokenizer.model --prompt='The capital of France is' --seq_len=48 --temperature=0 --warmup=false"
# repeat for the 8da4w PTE
```

### 4. Dispatch confirmation (separate ETDump run, per Principle IV/VI)

```bash
adb -s $S shell "cd $D && ./llama_main_etdump_015 --model_path=$D/<model>_4w_buffer_ctx3072.pte \
  --tokenizer_path=$D/tokenizer.model --prompt_file=$D/p2048_exact.txt --num_bos=1 \
  --max_new_tokens=4 --temperature=0 --warmup=false --etdump_path=$D/linear4w.etdp"
adb -s $S pull $D/linear4w.etdp $NFS/etdump/
.venv/bin/python .shared-context/scripts/analyze_etdump_shaders.py $NFS/etdump/linear4w.etdp
# confirm linear coopmat kernel names in the per-op breakdown, not tiled
# repeat for 8da4w, and for SDPA (same 4w PTE + ET_VK_SDPA_COOPMAT=1)
```

Expected outcome: a `dispatch_status` of `confirmed` for each of this
model's three configurations (or `fallback`/`failed` with the actual
kernel name recorded, per data-model.md).

### 5. E2E capture (2048-token prefill, 1024-token decode; separate from step 4)

```bash
adb -s $S shell "cd $D && ET_VK_EXECUTE_NODE_THRESHOLD=16 ./llama_main_015 \
  --model_path=$D/<model>_4w_buffer_ctx3072.pte --tokenizer_path=$D/tokenizer.model \
  --prompt_file=$D/p2048_exact.txt --num_bos=1 --max_new_tokens=1024 --ignore_eos \
  --temperature=0 --warmup=true"
# record prefill/decode tok/s; repeat for 8da4w
# for SDPA: same 4w PTE + ET_VK_SDPA_COOPMAT=1 ./llama_main_015 ...
```

If the GPU-watchdog issue recurs at 2048-token prefill (most likely on 8B/
3B, per Decision 3's risk ordering): record `blocked_reason` exactly per
data-model.md's Edge Cases -- do not silently retry at a shorter prefill
and report that number as if it were the 2048 result.

### 6. Publish this model's results immediately

Write `results/<model>-results.md` with this model's three configurations'
`e2e_result`/`blocked_reason` and their Prior-Finding Reference comparison
(data-model.md). Report to the user now -- do not wait for the other two
models.

## After all three models: assemble the consolidated report

Once `1b-results.md`, `3b-results.md`, and `8b-results.md` all exist,
produce `results/m5-e2e-validation-report.md` (User Story 4) covering all
nine configurations plus the explicit no-prior-baseline flags for `8da4w`
3B/1B and any 2048-prefill-blocked SDPA configurations.
