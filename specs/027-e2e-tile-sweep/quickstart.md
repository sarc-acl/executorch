# Quickstart: Validate the E2E-Ranked 8da4w Sweep End-to-End

## Prerequisites

- Execution worktree: the **existing** `dbuf-int8-sweep` worktree
  (`023-8da4w-int8-dbuf-sweep-impl` branch) — already has the built `llama_main` runner
  (`cmake-out-android-vk/examples/models/llama/llama_main`) and the full
  `linear_dq8ca_q4gsw_coopmat_tsweep` shader catalog from `025`/`026`.
- Buffer-storage `8da4w` PTEs staged on the device
  (`llama3_1_8b_8da4w_buffer_ctx3072.pte` at minimum, since all 8 initial candidates are
  `shape_family: "8B"` — research.md Decision 1/2). Stage via NFS if not already on-device:
  ```bash
  ssh xgpusw-debug08 'export ANDROID_SERIAL=00000bf70c579c33; \
    adb push /sarc-c/gpusw/users/yanwen.xu/android-run/models/llama3_1_8b_8da4w_buffer_ctx3072.pte \
    /data/local/tmp/llama_vk/'
  ```
- SSH/adb access to M5 EVT1 — either board acceptable.

## Step 1 — Build the combined pre-filter ranking (zero device time)

```bash
python3 specs/027-e2e-tile-sweep/scripts/build_prefilter_ranking.py \
  --specs025-results specs/025-8da4w-parameter-sweep/results/round3_results.json \
  --specs026-results specs/026-8da4w-subgroup32-sweep/results/round3_results.json \
  --specs026-correctness specs/026-8da4w-subgroup32-sweep/results/correctness_matrix.json \
  --out specs/027-e2e-tile-sweep/results/prefilter_ranking.json
```

**Expected outcome**: `prefilter_ranking.json` per `contracts/e2e-ranking-schema.md` §0,
exactly 8 entries with `shortlisted: true`, sorted by `microbenchmark_rank`; every entry
has `correctness_all_shapes_pass: true`.

## Step 2 — Pre-flight device/driver check

```bash
ssh xgpusw-debug08 'export ANDROID_SERIAL=00000bf70c579c33; \
  adb shell md5sum /vendor/lib64/hw/vulkan.samsung.so; \
  adb shell cat /sys/kernel/gpu/min_freq /sys/kernel/gpu/max_freq'
```

**Expected outcome**: driver hash matches `.shared-context/ACTIVE-STATUS.md`'s known-good
value; clocks read 509000/509000 (pinned). Re-run `pin_freqs.sh` if not.

## Step 3 — Coherence-check each distinct model before trusting its timing

```bash
ssh xgpusw-debug08 "export ANDROID_SERIAL=00000bf70c579c33; \
  adb shell 'cd /data/local/tmp/llama_vk && ./llama_main \
  --model_path=\$PWD/llama3_1_8b_8da4w_buffer_ctx3072.pte \
  --tokenizer_path=\$PWD/tokenizer.model \
  --prompt=\"The capital of France is\" --seq_len=48 --temperature=0 --warmup=false'"
```

**Expected outcome**: coherent (if repetitive, at temperature=0) output — confirms the PTE
loads and produces valid results before any timing number from it is trusted.

## Step 4 — Screen every shortlisted candidate + baseline (1 run each)

```bash
python3 specs/027-e2e-tile-sweep/scripts/run_e2e_screen.py \
  --prefilter specs/027-e2e-tile-sweep/results/prefilter_ranking.json \
  --llama-main-binary <EXEC-WT>/cmake-out-android-vk/examples/models/llama/llama_main \
  --ssh-host xgpusw-debug08 --serial 00000bf70c579c33 \
  --out specs/027-e2e-tile-sweep/results/screen_results.json
```

Each run internally does (per candidate token, `""` for baseline):

```bash
ssh <host> "export ANDROID_SERIAL=<serial>; \
  adb shell 'cd /data/local/tmp/llama_vk && ET_VK_EXECUTE_NODE_THRESHOLD=16 \
  ET_VK_DQ8CA_COOPMAT_VARIANT=<token> ./llama_main \
  --model_path=\$PWD/<model_used> --tokenizer_path=\$PWD/tokenizer.model \
  --prompt_file=\$PWD/p2048_exact.txt --num_bos=1 --max_new_tokens=1 \
  --temperature=0 --warmup=false'" | grep PyTorchObserver
```

**Expected outcome**: `screen_results.json` per `contracts/e2e-ranking-schema.md` §1 — 9
entries (8 candidates + baseline), each `model_used` matching its `Candidate.model_used`.

## Step 5 — Compute escalation decisions

```bash
python3 specs/027-e2e-tile-sweep/scripts/run_e2e_screen.py --decide-only \
  --screen-results specs/027-e2e-tile-sweep/results/screen_results.json \
  --out specs/027-e2e-tile-sweep/results/escalation_decisions.json
```

**Expected outcome**: `escalation_decisions.json` per §2 — every shortlisted candidate has
an `escalated` bool computed from `screen_ratio >= -0.10`.

## Step 6 — Confirm escalated candidates + baseline (3 fresh runs each)

```bash
python3 specs/027-e2e-tile-sweep/scripts/run_e2e_confirm.py \
  --escalation specs/027-e2e-tile-sweep/results/escalation_decisions.json \
  --prefilter specs/027-e2e-tile-sweep/results/prefilter_ranking.json \
  --llama-main-binary <EXEC-WT>/cmake-out-android-vk/examples/models/llama/llama_main \
  --ssh-host xgpusw-debug08 --serial 00000bf70c579c33 \
  --out specs/027-e2e-tile-sweep/results/confirm_results.json \
  --summary-out specs/027-e2e-tile-sweep/results/confirmation_results.json
```

**Expected outcome**: `confirm_results.json`/`confirmation_results.json` per §3-4 — only
escalated candidates + baseline appear; each candidate's `beats_baseline` is computed per
data-model.md's non-overlapping-ranges rule, never asserted by hand.

## Step 7 — Extend the search only if nothing beats baseline (conditional)

If every `confirmation_results.json` entry has `beats_baseline: false`, select new
candidates per research.md's analytical-scoring approach and repeat Steps 4-6 for them
(writing `extension_candidates.json` per §5 first). Skip this step entirely if a winner
was already confirmed in Step 6.

## Step 8 — Produce the final report

```bash
python3 specs/027-e2e-tile-sweep/scripts/build_report.py \
  --out-dir specs/027-e2e-tile-sweep/results/
```

**Expected outcome**: `sweep-report.md` per §6 — states the `FinalAnswer` up front (a
specific winning token with confirmed improvement, or an explicit "baseline stands"
statement), the microbenchmark-vs-e2e rank agreement finding, and full screening/
confirmation tables.

## Success check

The feature is validated end-to-end when all of the following hold simultaneously:

- `prefilter_ranking.json` has exactly 8 `shortlisted: true` entries, all
  `correctness_all_shapes_pass: true`.
- Every `screen_results.json`/`confirm_results.json` entry's `model_used` matches its
  candidate's `shape_family`-derived value — spot-check this explicitly (research.md
  Decision 2 is this feature's core anti-regression target).
- `confirm_results.json` contains measurements only for escalated candidates + baseline —
  no non-escalated candidate was confirmed.
- `sweep-report.md` states exactly one `FinalAnswer` and the rank-agreement finding.
- If a winner other than baseline is reported, its `ConfirmationResult.beats_baseline` is
  `true` by the non-overlapping-ranges rule, not by improvement percentage alone.
