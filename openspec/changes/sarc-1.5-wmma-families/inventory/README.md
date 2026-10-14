# WMMA families: source inventory

Pinned source: `6178971845` (`topic/wmma-inventory` starts here). This stage inventories existing
4w and 8da4w prefill kernels from all eight final profiles. It does not select a family default.
No builds, GPU queries, model access, measurements, submodule initialization or remote access were used.
Historical campaign instructions to build, time, push and stop after two small gains are superseded by
this source-only stage. Local commits and independent source review are required.

| Deliverable | Meaning |
|---|---|
| `inventory.csv` | One row per device, ordered selection route and registered storage combination; exact shader, geometry, MMA, accumulation, declared shared bytes and source references |
| `manifests/*.json` | Deduplicated candidates within each family, keyed by exact wrapper/body/parameter fingerprint; dispatch aliases and origins preserved |
| `legality-matrix.csv` | Every candidate against every member of its family, with explicit numeric requirements and `PENDING_QUERY` status |
| `PROFILES.md` | Final profile provenance, inherited paths, shadowed rows and fallback boundaries |
| `LEGALITY.md` | Static requirements, memory derivation and facts that must be queried later |
| `packets/` | Company-side work specifications for owner relay; no launch or external messaging |
| `source-hashes.csv` | SHA-256 of the source inputs at the pinned commit |
| `generate.py` | Source-only regeneration of CSVs and manifests; requires existing Python/PyYAML |

Run `python3 openspec/changes/sarc-1.5-wmma-families/inventory/generate.py` from the repository.
The generator reads source only. It contains a manually audited route list, not an implementation of the
C++ selector. Source changes require a new inventory review, not blindly reusing this route list.
All file/line references in the CSVs and manifests end in the full source commit. Line numbers refer to
that commit. The generator reads the checkout; `source-hashes.csv` must agree with the pinned objects.

The reviewed closure has 42 routes and 60 storage-specific origin rows. Family manifests contain 19 RDNA3,
12 Xe2, 11 NVIDIA and 7 Xclipse source-distinct candidates. These cover 50 exact dispatch names; one RDNA3
source alias pair is merged, with both dispatch names and all route provenance retained. Counts include
inherited fallback variants, not just the kernels reached by the twelve aligned production shapes.
A candidate's presence does not assert legality or correctness on another member. Every target needs the
queries in `LEGALITY.md`; no target was queried here. There are no measured results or winners.

Each profile route requires an earlier selector to serve the shape, then its own predicate and fit check.
Within a layer the first fitting route wins; outer profile layers replace the inherited result. A table
route is retained only where upper layers do not serve the shape. Read `PROFILES.md` with the CSV; the
predicate column alone is not an independent launch recipe. The legality matrix gives aligned build-time
requirements; row-major runtime exceptions are described separately.

Non-cooperative fallbacks are documented in `PROFILES.md` and excluded from family candidate manifests.
Unrelated sweep kernels, phase-timing twins, rejected profiles and attention/fp-linear kernels are excluded.
The supplied proposal's larger promotion and upstream-port completion conditions belong to later stages.
