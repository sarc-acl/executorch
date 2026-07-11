# Contract: Storage Comparison Data Formats

## Harness output: `RESULT` CSV line (modified format)

```
RESULT,<model>,<scheme>,<regime>,<storage>,<op>,<M>,<K>,<N>,<mean_us>,<stddev_us>,<iterations>,<kernel>
```

- `<storage>` is a new field (`texture3d` or `buffer`), inserted right after
  `<regime>` — the only change to the existing `001` line format besides this
  one insertion; everything else keeps its existing meaning and order.
- `<kernel>` MUST be a tiled/coop-family name for every row in this feature's
  data (never a `*_coopmat` name) — `compare_storage.py` treats any row that
  violates this as a hard error, not a data point, per Research Decision 2.

## `results/storage-comparison-report.md`

Structure a consumer can rely on:

1. Two top-level verdict lines, one for `prefill`, one for `decode`, each
   one of: "effectively free", "measurable cost (~X% slower)", "measurable
   benefit (~X% faster)" — placed at the very top of the report, before any
   per-case table (FR-008/SC-003).
2. A full 96-row case table (`data-model.md`'s Storage Comparison Case),
   sorted by `model`, `scheme`, `regime`, `op`.
3. A separate "infeasible / contaminated" section, present even if empty
   (stating "none" explicitly) — never silently omitted (FR-009).
4. A "cross-check against `001`" section confirming this feature's own
   `Texture3D` numbers are consistent with `001`'s already-published
   microbenchmark data for the same cases (Research Decision 4).
