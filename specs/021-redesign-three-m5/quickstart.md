# Quickstart: Redesigned M5 EVT1 Microbenchmark Suite

Paste the env block from `.shared-context/instruction-for-ai/README.md`
§Conventions before any of this (`HOST`, `S`, `D`, `SC`).

## Prerequisites

- M5 EVT1 confirmed free (constitution Principle VIII / gotcha G10).
- Driver identity re-verified.
- Clocks pinned to 509/2730/663 MHz, sysfs-verified (Principle VII).
- The three harnesses rebuilt from their modified source (research.md
  Decisions 3/4/5/6), following `.shared-context/instruction-for-ai/build.md`'s
  Android cross-build sequence (Principle X, gotcha G1 — don't skip the
  two-step relink).

## 1. Correctness sanity check before trusting any timing

```bash
COOPMAT_BENCH_CORRECTNESS_ONLY=1 <linear binary>
COOPMAT_BENCH_CORRECTNESS_ONLY=1 <baseline binary, if it retains this switch>
```
Confirm every small-shape case still reports `PASSED` — the regime/format
changes must not have altered any existing correctness path.

## 2. Run each harness 3 times, capturing unified output

```bash
mkdir -p specs/021-redesign-three-m5/results/raw
for rep in 1 2 3; do
  adb -s $S shell "cd $D && ./test_coopmat_linear_bench_021" > specs/021-redesign-three-m5/results/raw/linear_rep${rep}.log
  adb -s $S shell "cd $D && ./test_sdpa_coopmat_bench_021" > specs/021-redesign-three-m5/results/raw/sdpa_rep${rep}.log
  adb -s $S shell "cd $D && ./test_llama_baseline_bench_021" > specs/021-redesign-three-m5/results/raw/baseline_rep${rep}.log
done
```
**Expected outcome, contrasted with `specs/020`**: `baseline_rep*.log`
exits 0 (not 137/SIGKILL) and contains 192 `RESULT,...` lines, not 14.
All three logs contain only `RESULT,...` lines (plus harness startup
banners) — no `SUMMARY:` table, no bare per-case dispatch lines.

## 3. Verify the three-way dispatch status split

```bash
grep ",decode," specs/021-redesign-three-m5/results/raw/linear_rep1.log | grep -c ",not_applicable,"
# expected: matches the decode row count -- every linear decode case is not_applicable
# (QuantizedLinear.cpp's is_gemv_case short-circuit dispatches a dedicated
# "_coop" kernel for decode, never "_tiled" -- so this must never show as
# fallback_tiled either; confirm zero "confirmed" AND zero "fallback_tiled":
grep ",decode," specs/021-redesign-three-m5/results/raw/linear_rep1.log | grep -cE ",(confirmed|fallback_tiled),"
# expected: 0
grep ",decode," specs/021-redesign-three-m5/results/raw/sdpa_rep1.log | grep -c ",not_applicable,"
# expected: matches the decode row count -- every SDPA decode case is not_applicable
```

## 4. Aggregate and produce the report

```bash
$SC/aggregate_microbench_results.py \
  --linear specs/021-redesign-three-m5/results/raw/linear_rep{1,2,3}.log \
  --sdpa specs/021-redesign-three-m5/results/raw/sdpa_rep{1,2,3}.log \
  --baseline specs/021-redesign-three-m5/results/raw/baseline_rep{1,2,3}.log \
  --out specs/021-redesign-three-m5/results/microbenchmark-suite-report.md
```

## Expected outcome

`microbenchmark-suite-report.md` exists with: one shared table format
across all three harness sections (same columns, since they now share
one schema); baseline showing all 192 cases (not 14); a reconciliation
note stating the linear shape basis changed from `specs/020`'s `M=1024`
and that only tiled-vs-coopmat trend/direction is comparable, not exact
percentages (FR-010); zero `confirmed` dispatch statuses on any decode
row anywhere in the report (SC-003).
