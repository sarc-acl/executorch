# Quickstart: End-to-End Speedup Target and Validation

This feature has two very different modes: what runs **now** (target
recording + self-test, both real deliverables) and what runs **later**
(the real outcome report, which requires a future optimization build this
repo does not yet have). Don't skip straight to "later" — there is nothing
to compare against yet.

## Prerequisites

- `001-minipc-baseline-benchmarks` is complete: all six
  `specs/001-minipc-baseline-benchmarks/results/raw/<model>_<scheme>.json`
  files exist with a populated `e2e` object.
- No device/build access needed for steps 1-2 below.

## 1. Generate the speedup target (Story 1 — real, run now)

```bash
python specs/005-e2e-speedup-target/scripts/compute_outcome.py --generate-target \
  --baseline-dir specs/001-minipc-baseline-benchmarks/results/raw \
  --out specs/005-e2e-speedup-target/results/speedup-target.json
```

Expected outcome: `speedup-target.json` with 6 entries (per
`contracts/outcome-schema.md`), each `target_prefill_tokens_per_sec` exactly
2x its `baseline_prefill_tokens_per_sec`.

## 2. Run the self-test (validates the tool BEFORE trusting it on real data)

```bash
python specs/005-e2e-speedup-target/scripts/compute_outcome.py --selftest \
  --target specs/005-e2e-speedup-target/results/speedup-target.json \
  --out-dir specs/005-e2e-speedup-target/results/selftest
```

This constructs synthetic "after" JSON files (same schema as `001`'s `e2e`
object) designed to hit each of the five verdicts, then runs them through
the same comparison logic real data will use later.

Expected outcome: `results/selftest/selftest-outcome-report.md`, headed
"SYNTHETIC SELF-TEST DATA — NOT A REAL MEASUREMENT", showing all five
verdicts (`met`, `exceeded`, `missed`, `regressed`, `not_comparable`) each
appearing at least once and matching the scenario they were constructed for.

## 3. Sanity-check the self-test

- Confirm the `met`-scenario entry's `observed_multiplier` is within the
  baseline's own `variance.prefill_tokens_per_sec_stdev`-based noise band of
  exactly `2.0`.
- Confirm the `not_comparable`-scenario entry has `observed_multiplier: null`
  and shows its `methodology_note`, not a fabricated number.
- Confirm `combined_e2e_change_pct` is present and displayed on every
  scenario, but that no scenario's `verdict` can be explained by it alone
  (e.g. a scenario with a great combined e2e change but a below-2x prefill
  multiplier must still show `missed`).

## 4. (Future work, not part of this pass) Run against real data

Once a build containing actual optimization work exists and a real
re-measurement has been captured under `001`'s exact methodology (same
device, same fixed 2048-token prefill / 1024-token decode workload, same
5-rep statistical discipline):

```bash
python specs/005-e2e-speedup-target/scripts/compute_outcome.py \
  --target specs/005-e2e-speedup-target/results/speedup-target.json \
  --after-dir <path to the real re-measurement JSON files> \
  --out specs/005-e2e-speedup-target/results/outcome-report.md
```

This is the same tool, same code path as step 2 — only the input data
changes from synthetic to real. Nothing about the comparison logic differs.
