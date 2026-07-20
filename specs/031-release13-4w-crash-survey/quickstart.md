# Quickstart: Release/1.3 Crash Survey on M5 EVT1 (4w + 8da4w, Floating + Pinned)

How to run this survey (or verify it was run correctly) end to end. Assumes the env paste-block
from `.shared-context/instruction-for-ai/README.md` §Conventions (`$HOST`/`$S`/`$D`/`$NFS`), with
`llama_main_rel1.3`, `llama_main_nodethresh`, and all six (`4w`/`8da4w` × 3 models) texture
`.pte` files already staged at `$D` on the device (already true as of this session — see this
feature's `spec.md` for provenance). **Extended** (2026-07-14) beyond the original `4w`-only,
floating-only steps below: for `8da4w` and/or pinned clocks, repeat the identical procedure with
the PTE/clock swapped — see `research.md`'s extension decisions for the threshold-fallback policy
(`64` default, `32` fallback only where `64` is confirmed insufficient — currently only 8B
pinned, both quant schemes).

## Prerequisites

- `release-1.3/executorch`'s `llama_main_rel1.3` runner already built and pushed to `$D` (done,
  `specs/029`).
- `llama3_2_1b_4w_texture_ctx3072.pte`, `llama3_2_3b_4w_texture_ctx3072.pte`,
  `llama3_1_8b_4w_texture_ctx3072.pte` already pushed to `$D` (done, this session).
- `p2048_exact.txt` + `tokenizer.model` already at `$D` (done, prior sessions).

## Per-model rep sequence (repeat for 3B, then 1B, then 8B — see research.md for order rationale)

```bash
ssh $HOST
S=0000088f8e579c33
D=/data/local/tmp/llama_vk
PTE=llama3_2_3b_4w_texture_ctx3072.pte   # swap per model

# 1. Verify driver + clock BEFORE this model's first rep
adb -s $S shell md5sum /vendor/lib64/hw/vulkan.samsung.so   # expect c9861e9906d03fa2c7d48b804e1a1c80
adb -s $S shell cat /sys/class/devfreq/23400000.sgpu/min_freq /sys/class/devfreq/23400000.sgpu/max_freq
# expect 255000 / 980000 (floating); if not, set them:
adb -s $S shell "echo 255000 > /sys/class/devfreq/23400000.sgpu/min_freq"
adb -s $S shell "echo 980000 > /sys/class/devfreq/23400000.sgpu/max_freq"

# 2. Coherence check (once per model, not per rep)
adb -s $S shell "cd $D && ./llama_main_rel1.3 --model_path=$D/$PTE \
  --tokenizer_path=$D/tokenizer.model --prompt='The capital of France is' \
  --seq_len=48 --temperature=0 --warmup=false"
# expect coherent "...Paris..." output before proceeding

# 3. For rep_index in 1..3:
adb -s $S logcat -c
adb -s $S shell "cd $D && ./llama_main_rel1.3 --model_path=$D/$PTE \
  --tokenizer_path=$D/tokenizer.model --prompt_file=$D/p2048_exact.txt --num_bos=1 \
  --max_new_tokens=1024 --ignore_eos --temperature=0 --warmup=true"
#   - If the JSON stats line prints (prompt_tokens=2048, generated_tokens=1023): record
#     outcome=completed, prefill_token_per_sec, decode_token_per_sec.
#   - If the shell command errors and `adb devices` no longer lists $S: outcome=crashed.
#     Go to Recovery below, then retry this same rep_index (it does not count as a
#     4th attempt — the crashed attempt itself is rep_index; recovery doesn't consume a rep).
```

## Recovery (only on a crashed rep)

```bash
lsusb | grep 18d1                     # confirm it shows S5E9975_LK_Bootloader
fastboot devices                      # confirm $S shows up as "fastboot"
fastboot -s $S reboot                 # plain reboot, no flash/wipe
# poll until booted:
for i in $(seq 1 15); do
  BOOTED=$(adb -s $S shell getprop sys.boot_completed 2>&1)
  [ "$BOOTED" = "1" ] && break
  sleep 10
done
# then re-verify driver hash + clock range (step 1 above) before the next rep
```

If `fastboot devices` doesn't show `$S`, or `sys.boot_completed` never reaches `1` after ~2.5
minutes: **stop and escalate** — do not attempt `fastboot flash`/wipe/anything more invasive
without separate explicit authorization (per spec Edge Cases).

## Expected outcome / how to validate this feature is "done"

- `results/report.md` **on its own** (without needing to open `raw-attempts.md`) contains one
  headline table (model × prefill tok/s ± CoV × decode tok/s ± CoV × crash annotation) AND the
  raw per-attempt table backing it, matching the data shapes in `data-model.md`. `raw-attempts.md`
  remains the append-as-you-go working log; `report.md` is the self-contained deliverable.
- Every row's driver hash is recorded and matches the documented default for the reps it
  presents (or the row explicitly notes it doesn't, per spec SC-004).
- A reader with no session context can tell, from the report alone, which of {1B, 3B, 8B} is
  safe to benchmark under this exact configuration (per spec SC-003).
