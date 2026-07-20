# Quickstart: Validate the E2E-Ranked 4w Sweep End-to-End

## Prerequisites

- Execution worktree: a **new** worktree cut from `dev` (`yanwen/dev-1.3`) — `022`'s own
  worktree was retired 2026-07-11, and reusing `dev` directly is against this workspace's
  standing rule (research.md Decision 4):
  ```bash
  cd /local/yanwen.xu/workspace
  git worktree add 4w-e2e-tile-sweep -b 028-4w-e2e-tile-sweep yanwen/dev-1.3
  cd 4w-e2e-tile-sweep/executorch
  ./install_executorch.sh --minimal && pip install -e . --no-build-isolation
  ```
- Buffer-storage `4w` PTEs staged on the device (all three already on NFS, no export
  needed — `llama3_1_8b_4w_buffer_ctx3072.pte` at minimum for the 8B search; `llama3_2_1b_`
  and `llama3_2_3b_4w_buffer_ctx3072.pte` for the User Story 4 confirmation pass):
  ```bash
  ssh xgpusw-debug08 'export ANDROID_SERIAL=00000bf70c579c33; \
    adb push /sarc-c/gpusw/users/yanwen.xu/android-run/models/llama3_1_8b_4w_buffer_ctx3072.pte /data/local/tmp/llama_vk/; \
    adb push /sarc-c/gpusw/users/yanwen.xu/android-run/models/llama3_2_1b_4w_buffer_ctx3072.pte /data/local/tmp/llama_vk/; \
    adb push /sarc-c/gpusw/users/yanwen.xu/android-run/models/llama3_2_3b_4w_buffer_ctx3072.pte /data/local/tmp/llama_vk/'
  ```
- SSH/adb access to M5 EVT1 — either board acceptable.

## Step 0 — Port the tsweep infra onto `dev`'s current base (research.md Decision 0)

This step has no equivalent in `027` — `4w`'s tile-sweep dispatch mechanism was never
committed anywhere and must be re-derived, not copied in. In the new worktree:

1. Read the archived reference patch (do not `git apply` it — it targets a deleted file
   and a pre-WMMA/SDPA-coopmat base):
   ```bash
   cat /local/yanwen.xu/workspace/.archived-artifacts/tmp-origcm-2026-07-08/untracked-new-files/backends/vulkan/runtime/graph/ops/glsl/linear_q4gsw_coopmat_tsweep.glsl
   cat /local/yanwen.xu/workspace/.archived-artifacts/tmp-origcm-2026-07-08/untracked-new-files/backends/vulkan/runtime/graph/ops/glsl/linear_q4gsw_coopmat_tsweep.yaml
   ```
2. Using `dev`'s current `backends/vulkan/runtime/graph/ops/glsl/linear_q4gsw_coop.glsl`
   as the structural base, add a new sibling `linear_q4gsw_coopmat_tsweep.{glsl,yaml}`
   that parameterizes tile size / subgroup grid / subgroup size, following the archived
   file's parameterization pattern.
3. In `backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp`, add an
   `ET_VK_Q4GSW_COOPMAT_VARIANT` env-var dispatch token, copying
   `ET_VK_DQ8CA_COOPMAT_VARIANT`'s existing structure in the same file (unset = today's
   fixed dispatch, unchanged).
4. Rebuild `llama_main` for Android once.
5. Re-run `022`'s existing correctness harness
   (`COOPMAT_BENCH_CORRECTNESS_ONLY=1 ./test_coopmat_linear_bench`) against every one of
   the 8 shortlisted tokens through the ported shader; record results in
   `results/port_verification.json` per `contracts/e2e-ranking-schema.md` §-1.

**Expected outcome**: `port_verification.json` shows `correctness_status: "pass"` for all
8 shortlisted tokens. Any `"fail"` halts progress on that token — it is excluded from
`prefilter_ranking.json`'s `shortlisted` set with the failure recorded as its exclusion
reason.

## Step 1 — Build the pre-filter ranking (zero device time)

```bash
python3 specs/028-4w-e2e-tile-sweep/scripts/build_prefilter_ranking.py \
  --specs022-round2 specs/022-linear-coopmat-autotune/results/round2_results.json \
  --specs022-round3 specs/022-linear-coopmat-autotune/results/round3_results.json \
  --out specs/028-4w-e2e-tile-sweep/results/prefilter_ranking.json
```

**Expected outcome**: `prefilter_ranking.json` per `contracts/e2e-ranking-schema.md` §0,
exactly 8 entries, all `shortlisted: true`, sorted by `microbenchmark_rank`.

## Step 2 — Pre-flight device/driver check

```bash
ssh xgpusw-debug08 'export ANDROID_SERIAL=00000bf70c579c33; \
  adb shell md5sum /vendor/lib64/hw/vulkan.samsung.so; \
  adb shell cat /sys/kernel/gpu/min_freq /sys/kernel/gpu/max_freq'
```

**Expected outcome**: driver hash matches the documented default
(`c9861e9906…` = `f14c51b6f8`); if not, reflash per
`.shared-context/instruction-for-ai/access-and-run/README.md` §6 before proceeding
(Constitution Principle VIII).

## Step 3 — 8B screening pass (User Story 1)

```bash
python3 specs/028-4w-e2e-tile-sweep/scripts/run_e2e_screen.py \
  --prefilter specs/028-4w-e2e-tile-sweep/results/prefilter_ranking.json \
  --port-verification specs/028-4w-e2e-tile-sweep/results/port_verification.json \
  --model-stage 8b_search \
  --out specs/028-4w-e2e-tile-sweep/results/screen_results.json
```

**Expected outcome**: 9 `E2EMeasurement` entries (8 candidates + baseline), all
`model_stage: "8b_search"`, `model_used: "llama3_1_8b_4w_buffer_ctx3072.pte"`.

## Step 4 — 3-run confirmation for close-or-ahead candidates

```bash
python3 specs/028-4w-e2e-tile-sweep/scripts/run_e2e_confirm.py \
  --screen specs/028-4w-e2e-tile-sweep/results/screen_results.json \
  --model-stage 8b_search \
  --out specs/028-4w-e2e-tile-sweep/results/confirmation_results.json
```

**Expected outcome**: `confirmation_results.json` per `contracts/e2e-ranking-schema.md`
§3 — baseline always confirmed, plus every candidate with `screen_ratio >= -0.10`.

## Step 5 — (Only if Step 4 finds no winner) Search extension (User Story 2)

Bounded, budgeted extension — see spec FR-006/FR-007/FR-009 and
`data-model.md`'s `SearchExtension` entity. Skip this step entirely if Step 4 already
confirmed a real e2e winner.

## Step 6 — Final 8B answer (User Story 3)

```bash
python3 specs/028-4w-e2e-tile-sweep/scripts/build_report.py \
  --confirmation specs/028-4w-e2e-tile-sweep/results/confirmation_results.json \
  --prefilter specs/028-4w-e2e-tile-sweep/results/prefilter_ranking.json \
  --stage final-8b \
  --out specs/028-4w-e2e-tile-sweep/results/final_8b_answer.json
```

**Expected outcome**: `final_8b_answer.json` per `contracts/e2e-ranking-schema.md` §4 —
exactly one `winner_token`, the `rank_agreement` finding (SC-006), and an
`excluded_candidates` entry for every non-confirmed candidate (SC-005).

## Step 7 — 1B/3B cross-size confirmation (User Story 4)

```bash
python3 specs/028-4w-e2e-tile-sweep/scripts/run_1b3b_confirmation.py \
  --final-8b specs/028-4w-e2e-tile-sweep/results/final_8b_answer.json \
  --out specs/028-4w-e2e-tile-sweep/results/cross_size_confirmation.json
```

**Expected outcome**: `cross_size_confirmation.json` per `contracts/e2e-ranking-schema.md`
§5 — exactly one `CrossSizeFinding` per model size (1B, 3B), each stating whether the 8B
finding's direction holds, is neutral, or reverses (SC-007).

## Step 8 — Final report

```bash
python3 specs/028-4w-e2e-tile-sweep/scripts/build_report.py \
  --final-8b specs/028-4w-e2e-tile-sweep/results/final_8b_answer.json \
  --cross-size specs/028-4w-e2e-tile-sweep/results/cross_size_confirmation.json \
  --stage full-report \
  --out specs/028-4w-e2e-tile-sweep/results/sweep-report.md
```

**Expected outcome**: `sweep-report.md` — one unambiguous answer, no open questions
remaining (spec SC-001), with the 8B evidence, rank-agreement finding, and 1B/3B
cross-size finding all stated explicitly.
