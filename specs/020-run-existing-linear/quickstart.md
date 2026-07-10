# Quickstart: M5 EVT1 Full Microbenchmark Suite

Paste the env block from `.shared-context/instruction-for-ai/README.md`
§Conventions before any of this (`HOST`, `S`, `D`, `SC`).

## Prerequisites

- M5 EVT1 confirmed free (constitution Principle VIII / gotcha G10).
- Driver identity re-verified (`adb -s $S shell md5sum /vendor/lib64/hw/vulkan.samsung.so`).
- Clocks re-pinned to 509/2730/663 MHz (the leftover floating state from
  the stopped `specs/019` session must be corrected first — this feature
  does not measure under floating clocks).

## 1. Confirm/refresh binaries (research.md Decision 5)

```bash
# Confirm current local build mtimes are newer than source (rebuild only if not):
stat -c "%Y %n" backends/vulkan/test/custom_ops/test_{coopmat_linear_bench,sdpa_coopmat_bench,llama_baseline_bench}.cpp
stat -c "%Y %n" cmake-out-android-vk/backends/vulkan/test/custom_ops/test_{coopmat_linear_bench,sdpa_coopmat_bench,llama_baseline_bench}

# Confirm on-device staging state (linear bench already staged as test_coopmat_linear_bench_016 per Decision 5):
adb -s $S shell ls -la $D/ | grep -E "test_coopmat_linear_bench|test_sdpa_coopmat_bench|test_llama_baseline_bench"

# Push whichever of the SDPA/baseline binaries are missing:
adb -s $S push cmake-out-android-vk/backends/vulkan/test/custom_ops/test_sdpa_coopmat_bench $D/test_sdpa_coopmat_bench_020
adb -s $S push cmake-out-android-vk/backends/vulkan/test/custom_ops/test_llama_baseline_bench $D/test_llama_baseline_bench_020
adb -s $S shell chmod 755 $D/test_sdpa_coopmat_bench_020 $D/test_llama_baseline_bench_020
```

## 2. Verify pinned clocks are genuinely bound (Principle VII)

```bash
adb -s $S shell cat /sys/kernel/gpu/min_freq /sys/kernel/gpu/max_freq
adb -s $S shell cat /sys/class/devfreq/17000010.devfreq_mif/scaling_devfreq_{min,max}
adb -s $S shell cat /sys/class/devfreq/17000020.devfreq_int/scaling_devfreq_{min,max}
```
**Expected outcome**: all values equal 509000/2730000/663000 — never the
hardware full range left over from `specs/019`. If any value doesn't
match, re-run the pin script before proceeding.

## 3. Run each harness 3 times (research.md Decision 2)

```bash
mkdir -p specs/020-run-existing-linear/results/raw
for rep in 1 2 3; do
  adb -s $S shell "cd $D && ./test_coopmat_linear_bench_016" > specs/020-run-existing-linear/results/raw/linear_rep${rep}.log
  adb -s $S shell "cd $D && ./test_sdpa_coopmat_bench_020" > specs/020-run-existing-linear/results/raw/sdpa_rep${rep}.log
  adb -s $S shell "cd $D && ./test_llama_baseline_bench_020" > specs/020-run-existing-linear/results/raw/baseline_rep${rep}.log
done
```
Each invocation is expected to exit 0 and print its own summary table
with every case's dispatch/correctness status.

## 4. Aggregate and produce the report (research.md Decision 3/4)

```bash
$SC/aggregate_microbench_results.py \
  --linear specs/020-run-existing-linear/results/raw/linear_rep{1,2,3}.log \
  --sdpa specs/020-run-existing-linear/results/raw/sdpa_rep{1,2,3}.log \
  --baseline specs/020-run-existing-linear/results/raw/baseline_rep{1,2,3}.log \
  --compare-against specs/016-m5-linear-sdpa-microbench/results/ \
  --out specs/020-run-existing-linear/results/microbenchmark-suite-report.md
```

## Expected outcome

`specs/020-run-existing-linear/results/microbenchmark-suite-report.md`
exists with one section per harness, each showing per-model/per-scheme
case results with their 3-invocation CoV, any peer-relative outliers
named explicitly, every coopmat claim backed by a `dispatch_confirmed`
flag, a reconciliation note against `specs/016`'s prior linear/SDPA
numbers, and any correctness failure or crash named in the report body —
not silently dropped from any table.
