# Quickstart: M5 EVT1 Floating-Clock Speedup Table

Paste the env block from `.shared-context/instruction-for-ai/README.md`
§Conventions before any of this (`HOST`, `S`, `D`, `PTE`, `NFS`, `SC`).

## Prerequisites

- M5 EVT1 confirmed free (constitution Principle VIII / gotcha G10).
- Driver identity re-verified (`adb -s $S shell md5sum /vendor/lib64/hw/vulkan.samsung.so`).
- All 12 PTEs (6 from `specs/018`'s T-tiled baselines, 6 from
  `specs/015`'s full-stack optimal) already staged or re-stageable from
  `.pte_out` -- no new export needed (research.md Decision 1).

## 1. Unpin clocks (research.md Decision 2)

```bash
# Read hardware min/max first -- don't assume a value, read the device's own range
adb -s $S shell cat /sys/kernel/gpu/available_frequencies   # or the devfreq equivalent

# Write hardware min -> min_freq, hardware max -> max_freq, for all three domains
adb -s $S shell "echo <hw_min> > /sys/kernel/gpu/min_freq"
adb -s $S shell "echo <hw_max> > /sys/kernel/gpu/max_freq"
adb -s $S shell "echo <hw_min> > /sys/class/devfreq/23400000.sgpu/min_freq"
adb -s $S shell "echo <hw_max> > /sys/class/devfreq/23400000.sgpu/max_freq"
adb -s $S shell "echo <mif_hw_min> > /sys/class/devfreq/17000010.devfreq_mif/scaling_devfreq_min"
adb -s $S shell "echo <mif_hw_max> > /sys/class/devfreq/17000010.devfreq_mif/scaling_devfreq_max"
adb -s $S shell "echo <int_hw_min> > /sys/class/devfreq/17000020.devfreq_int/scaling_devfreq_min"
adb -s $S shell "echo <int_hw_max> > /sys/class/devfreq/17000020.devfreq_int/scaling_devfreq_max"
```

## 2. Verify genuinely floating (research.md Decision 3 -- do not skip)

```bash
adb -s $S shell cat /sys/kernel/gpu/min_freq /sys/kernel/gpu/max_freq
adb -s $S shell cat /sys/class/devfreq/17000010.devfreq_mif/scaling_devfreq_{min,max}
adb -s $S shell cat /sys/class/devfreq/17000020.devfreq_int/scaling_devfreq_{min,max}
```
**Expected outcome**: values match the hardware's full available range,
NOT `509000`/`2730000`/`663000` (the pinned triple). If any value still
matches the pinned triple, the unpin did not take effect -- stop and
re-issue the write, do not proceed to measurement.

## 3. Run 3 reps per config (12 configs, reusing existing PTEs)

```bash
# Example: 1B / 4w / T-tiled baseline (PTE from specs/018)
for i in 1 2 3; do
  adb -s $S shell "cd $D && ET_VK_EXECUTE_NODE_THRESHOLD=16 ./llama_main_018 \
    --model_path=$D/llama3_2_1b_4w_texture_ctx3072.pte --tokenizer_path=$D/tokenizer.model \
    --prompt_file=$D/p2048_exact.txt --num_bos=1 --max_new_tokens=1024 --ignore_eos \
    --temperature=0 --warmup=true" | grep PyTorchObserver
done
```
Repeat for all 12 configs (2 config_types x 2 schemes x 3 models),
recording all 3 per-rep values each time -- do not average in place of
recording them (research.md Decision 4).

## 4. Compute the cold-start speedup ratio and publish (research.md Decision 5)

```
speedup_vs_baseline_coldstart = full_stack_optimal.rep[0] / t_tiled_baseline.rep[0]
```
for each matching (model, scheme) pair. Write per-model results to
`specs/019-m5-floating-clock-speedup/results/{1b,3b,8b}-floating-results.md`
(all 3 per-rep values visible, `throttle_observed` noted per config), and
the consolidated 6-row table to
`specs/019-m5-floating-clock-speedup/results/floating-vs-pinned-report.md`,
with a caveat paragraph stating the cold-start comparison basis.

## Expected outcome

A six-row floating-clock speedup table exists alongside (not replacing)
the pinned one, every number labeled floating, every config's 3 per-rep
values visible, and a stated comparison-basis caveat -- so a reader can
see both the floating speedup ratio and whether/how much throttle
affected either side of it.
