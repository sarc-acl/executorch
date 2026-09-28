# sarc-1.5-e2e-benchmark: map of this change

Start with [proposal.md](proposal.md) (why and what), then the reports.

## Reports

| File | Audience | Section numbering |
|---|---|---|
| [REPORT.md](REPORT.md) | Headline end-to-end results, method, limitations | Named sections, no numbers |
| [TECHNICAL-REPORT.md](TECHNICAL-REPORT.md) | Why the results look this way, and how strong the evidence is | `§N` (1-15) |
| [MANAGER-REPORT.md](MANAGER-REPORT.md) | Summary for decision makers: Part A (two-day re-tuning) and Part B (state on release 1.5) | `§AN` (Part A), `§BN` (Part B; B1-B15 mirror TECHNICAL-REPORT §1-15) |

`evidence/roofline.md` has its own numbering; a reference such as "roofline.md §11" means that file's headings, not a
report's.

[CONTRIBUTING-A-GPU.md](CONTRIBUTING-A-GPU.md) describes how to add another GPU: what to measure (§3) and the
deliverables layout (§5).

## Layout

- `results/`: aggregated CSVs, tables, figures, slides and analysis scripts (`results/scripts/`).
  `results/seven-gpu/` and `results/six-gpu/` hold the contributed-GPU aggregates.
- `evidence/`: roofline, trace, efficiency, logits and refinement data behind the technical report.
- `kit/`: the reusable measurement kit (analysis, host scripts, logits probe, patches, prompts) for contributors.
- `contrib/<gpu>/`: per-GPU contributions (`7900xtx`, `m51`, `mali`, `rx7600`, `s26`) with their own notes.
  For m51 only the device and relative speedups are published.

## Notes

- Files and figures named `*_5gpu`, `*_6gpu` and `*_6gpu_m51` are archived renders of earlier grids. They are not
  regenerated; the unsuffixed files are the current ones.
- `results/MANIFEST.json` covers the 5-GPU campaign builds only. It does not describe the contributed-GPU builds.
- Raw logs, ETDumps and builds live outside the repository (see proposal.md).
