# Quickstart: M41 Release/1.3 Baseline Clock & Quant-Mode Study

## Prerequisites

- M41 confirmed as the intended device: `ssh xgpusw-debug07`, `export ANDROID_SERIAL=000009b44fd4abd3`
  (this host carries several devices — the serial is mandatory, not optional).
- Driver identity re-verified (constitution Principle VIII, spec FR-001):
  `adb -s $S shell md5sum /vendor/lib64/hw/vulkan.samsung.so`. As of this session it reads
  `d5d76f1bacf404b1a07d87ec8e479bdf`, matching none of the documented M5-EVT1 known-good hashes
  (those are built for a different chip, s5e9975, not this device's s5e9965) — no flash is
  performed here; just re-confirm it hasn't drifted from that value since the driver-hash record
  was written.
- All 6 PTEs already staged on-device (`llama3_{2_1b,2_3b,1_8b}_{4w,8da4w}_texture_ctx3072.pte`,
  `~21.7GB` total in `/data/local/tmp/llama_vk/`) — confirm with `adb -s $S shell ls -la
  /data/local/tmp/llama_vk/` before starting; re-push from
  `/sarc-c/gpusw/users/yanwen.xu/android-run/models/` only if any are missing.
- `llama_main_rel1.3` already on-device and executable (`chmod 755` already applied).

## 1. One-time node-threshold probe (research.md Decision 3)

Before the real pinned sweep, run one throwaway 8B pinned rep with the env var set, to see if this
binary even recognizes it (does not count as one of the 3 reported reps):

```bash
adb -s $S shell "cd /data/local/tmp/llama_vk && ET_VK_EXECUTE_NODE_THRESHOLD=16 ./llama_main_rel1.3 \
  --model_path=/data/local/tmp/llama_vk/llama3_1_8b_4w_texture_ctx3072.pte \
  --tokenizer_path=/data/local/tmp/llama_vk/tokenizer.model \
  --prompt_file=/data/local/tmp/llama_vk/p2048_exact.txt --num_bos=1 --max_new_tokens=1024 \
  --ignore_eos --temperature=0 --warmup=true"
```

Record whether this crashes the same way as an unset run, or behaves differently (e.g. survives).
Either outcome is useful signal; do not let this probe block the real sweep — proceed regardless.

## 2. Pin clocks and verify the pin bound (spec FR-009)

```bash
S=000009b44fd4abd3 /sarc-c/gpusw/users/yanwen.xu/android-run/pin_freqs.sh
adb -s $S shell "cat /sys/class/devfreq/23400000.sgpu/min_freq /sys/class/devfreq/23400000.sgpu/max_freq \
  /sys/class/devfreq/17000010.devfreq_mif/cur_freq /sys/class/devfreq/17000020.devfreq_int/cur_freq"
```

Do not label a run "pinned" from the sysfs write alone — after each rep, check that its
`prefill_tok_s` is no more than 70% of the already-collected floating number for the same (model,
quant_mode) cell (FR-009's concrete threshold). If it exceeds that, record it as `outcome=
dvfs_artifact` / "DVFS-ARTIFACT" (FR-012) instead of a pinned result — not a crash, not a valid
pinned number, excluded from that cell's mean/CoV either way.

## 3. Run the sweep (per cell: 3 reps, continue through any crash — spec FR-006)

```bash
run() {
  MODEL_FILE=$1; LABEL=$2; REP=$3
  echo "=== ${LABEL} rep ${REP} ==="
  adb -s $S shell "cd /data/local/tmp/llama_vk && ./llama_main_rel1.3 \
    --model_path=/data/local/tmp/llama_vk/${MODEL_FILE} \
    --tokenizer_path=/data/local/tmp/llama_vk/tokenizer.model \
    --prompt_file=/data/local/tmp/llama_vk/p2048_exact.txt --num_bos=1 --max_new_tokens=1024 \
    --ignore_eos --temperature=0 --warmup=true" 2>&1 | grep -E "PyTorchObserver|Error|abi"
}
# 4w-pinned (clocks pinned from step 2):
for r in 1 2 3; do run llama3_2_1b_4w_texture_ctx3072.pte 1B-4w-pinned $r; done
for r in 1 2 3; do run llama3_2_3b_4w_texture_ctx3072.pte 3B-4w-pinned $r; done
for r in 1 2 3; do run llama3_1_8b_4w_texture_ctx3072.pte 8B-4w-pinned $r; done
# 8da4w-pinned (same pinned clocks, no re-pin needed):
for r in 1 2 3; do run llama3_2_1b_8da4w_texture_ctx3072.pte 1B-8da4w-pinned $r; done
for r in 1 2 3; do run llama3_2_3b_8da4w_texture_ctx3072.pte 3B-8da4w-pinned $r; done
for r in 1 2 3; do run llama3_1_8b_8da4w_texture_ctx3072.pte 8B-8da4w-pinned $r; done
```

Then unpin (write each devfreq node's HW min to `min_freq` and HW max to `max_freq`, per
`instruction-for-ai/access-and-run/README.md` §3 — HW ranges already probed this session:
sgpu 226000–980000, mif 676000–5333000, int 133000–800000) and repeat for `8da4w-floating`:

```bash
for r in 1 2 3; do run llama3_2_1b_8da4w_texture_ctx3072.pte 1B-8da4w-floating $r; done
for r in 1 2 3; do run llama3_2_3b_8da4w_texture_ctx3072.pte 3B-8da4w-floating $r; done
for r in 1 2 3; do run llama3_1_8b_8da4w_texture_ctx3072.pte 8B-8da4w-floating $r; done
```

If any rep's output contains `libc++abi`/`Error` instead of a `PyTorchObserver` line: that rep's
`outcome=crashed`. Immediately run the crash-attribution check (step 4) before moving to the next
rep — do not skip it and do not stop the sweep for it (FR-006).

## 4. On any crash: attribute cause before recording it (research.md Decision 4)

```bash
adb -s $S shell "dmesg | tail -50" | grep -iE "oom|killed process"
adb -s $S shell cat /proc/meminfo | grep -E "MemAvailable|MemTotal"
adb -s $S shell getprop sys.boot_completed   # confirm device still responsive (spec FR-006)
```

Record `crash_cause=host_oom` if `dmesg` shows an OOM-kill signature for the runner process;
`crash_cause=gpu_watchdog` if the device shows no OOM signature but the crash's `dmesg` context
around the crash timestamp shows a GPU reset/timeout message instead; `crash_cause=unknown` if
neither is conclusive. Do not default to "watchdog" without this check (this is exactly the gap
gotcha G11 found already cost this workstream a wrong root cause once).

## 5. Compute per-model CoV and assemble the report

For each ModelSummary with ≥2 `outcome=ok` reps (a `dvfs_artifact` or `crashed` rep does not
count): `CoV = stdev(valid prefill_tok_s) / mean(valid prefill_tok_s) × 100%` (same for decode).
Write all four tables into
`specs/030-m41-release13-baseline/results/m41-release13-baseline-report.md`, each showing all 9
rep-cells (a number, "CRASHED", or "DVFS-ARTIFACT", per FR-012), with the floating tables' means
carrying the thermal-drift caveat (FR-007) and the pinned tables' cells each confirmed via step
2's threshold cross-check before being labeled "pinned." Also record the storage type
(texture/T-tiled) and runner binary (`llama_main_rel1.3`) once, per FR-008.

## Expected outcome

A single, self-contained report (SC-004 — readable without this session's chat context) with four
complete tables (4w-pinned, 4w-floating, 8da4w-pinned, 8da4w-floating), zero omitted rep-cells,
every crash attributed via `dmesg`/`meminfo` rather than assumed, every pinned cell backed by both
a sysfs readback and a passing throughput cross-check, and an explicit statement (spec SC-006)
that these are M41 secondary/cross-device reference numbers, not Samsung M5 EVT1 headline data.
