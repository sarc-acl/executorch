# Contract: Sweep Data Formats

## Harness output: `SWEEP_RESULT` CSV line (new format, distinct from `007`'s `RESULT` line)

```
SWEEP_RESULT,<config_id>,<model>,<op>,<m>,<k>,<n>,<outcome>,<mean_us>,<stdev_us>,<iterations>,<kernel_name>,<failure_detail>
```

- `<op>` is required (added during `/speckit-analyze` remediation): the
  sweep phase covers 2 ops per model (`wq`, `w1_gate` -- research.md
  Decision 3) and the full-catalog validation phase covers 7; `(model, m,
  k, n)` alone cannot disambiguate rows, since several `8da4w` ops share
  an identical shape within one model (`wq`/`wo`, `wk`/`wv` -- the same
  same-shape-sibling ambiguity `004`'s and `007`'s own cross-checks
  already had to account for).
- `<outcome>` is one of `measured`/`compile_failure`/`pipeline_crash`/`correctness_failure`.
  There is no separate "invalid" value -- a mathematically incompatible
  combination (e.g. `config_id=12`, research.md Decision 4) surfaces as
  `correctness_failure`, since the shader still compiles and dispatches
  but computes wrong output.
- `<mean_us>`/`<stdev_us>`/`<iterations>` are empty (not zero) when
  `outcome != measured` -- a zero would be misreadable as a real, if
  implausible, timing.
- `<failure_detail>` is empty only when `outcome == measured`; every
  non-measured row MUST carry the actual error text (FR-004) -- an empty
  detail on a failure row is a hard error in the parser, not a warning.
  For `config_id=12` (the deliberate negative test), a `failure_detail`
  confirming a numeric mismatch (not a crash) is the *expected*, correct
  outcome -- see data-model.md.
- `<kernel_name>` is always populated (even on failure, if compilation
  succeeded far enough to know the name) so `dispatch_confirmed` can still
  be checked where applicable.

## SPIR-V inspection output: `results/spirv/<kernel_name>.dis.txt`

Same format as `007`: plain `spirv-dis` output, one file per distinct
compiled variant kernel name actually measured, with the
`OpCooperativeMatrixLoadKHR`/`OpCooperativeMatrixMulAddKHR` presence check
recorded in the report's correctness-verification summary.

## `results/sweep-report.md`

Structure a consumer can rely on:

1. **Sweep-phase summary table** (5 active candidates x 6 shapes = 30
   rows + 1 negative test x 1 shape = 31 total): every cell either a
   mean±stdev timing or an explicit failure/outcome marker -- never blank
   (SC-001). The negative-test row (`config_id=12`) is clearly labeled as
   such, not mixed in with the ranked candidates. Configs 1, 3, 5, 7, 9,
   11 (subgroup 32) are NOT in this table -- excluded mid-implementation
   after a real correctness bug (research.md Decision 4's revision) --
   they appear only in the Excluded/Out-of-Scope section below, with
   config 1's actual failing data as evidence.
2. **The Optimal Configuration Recommendation** (data-model.md's entity):
   stated plainly at the top -- either a specific winning configuration
   (drawn only from `config_id` 2, 4, 6, 8, 10) with its full-catalog
   validation numbers, or the explicit "no configuration in the sweep
   outperforms tiled" finding (FR-007).
3. **Full-catalog validation table** for the winning configuration(s)
   only (21 rows, matching `007`'s catalog, with an `op` column), each row
   showing `speedup_vs_shipped_pct` and `speedup_vs_tiled_pct` with
   significance.
4. **Failure log**: every `compile_failure`/`pipeline_crash`/
   `correctness_failure` row from the sweep phase, with its actual error
   text -- present even if empty (SC-001, FR-004). Includes
   `config_id=12`'s expected correctness failure, explicitly annotated as
   the deliberate negative-test result, not an unexpected failure.
5. **Correctness-verification summary**: which kernel(s) were
   SPIR-V-inspected, what was found, confirmation the exact-reference
   check passed for every configuration appearing in the recommendation
   (SC-005), and confirmation that the negative test (`config_id=12`)
   correctly failed rather than passed.
6. **Excluded/Out-of-Scope**: configs 1, 3, 5, 7, 9, 11 (subgroup 32),
   with config 1's actual measured mismatch (element, computed vs
   reference value, and the row-64/second-subgroup-tile signature) as
   evidence, and an explicit note that the root cause is unresolved and
   was excluded by user decision rather than assumed safe.
