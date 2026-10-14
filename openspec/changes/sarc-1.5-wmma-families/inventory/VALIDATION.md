# Source-only validation

Source baseline: `617897184529ff180af491ee0d5d7de778cbb6c6`.
Actor: Codex. Independent reviewer evidence is in `REVIEW.md`.
No benchmark numbers are asserted by this stage.

The actor ran the generator and checked these properties using Python's CSV/JSON/hash libraries and
`git show` of the pinned source objects:

| Check | Result |
|---|---|
| Pinned SHA-256 input hashes versus git objects | PASS, every entry of `source-hashes.csv` |
| Source file/line/commit reference bounds | PASS, every inventory reference resolves within the pinned file |
| Inventory origins versus expanded manifest dispatch aliases | PASS, no missing or extra dispatch name per family |
| Exact source fingerprint uniqueness within each family | PASS; one release/dev RDNA3 alias pair merged, both original names and provenance retained |
| Workgroup/subgroup/MMA tile divisibility | PASS for every candidate's declared geometry |
| A_MAP_FULL / A_MULTI_BLOCK staging coverage | PASS: workgroup size times A_BLOCKS (or one) equals the tile's 4x4 activation-block count |
| Regeneration | PASS: 42 routes, 60 origin rows; 49 source-distinct candidates covering 50 names; 110 candidate/member requirement rows |
| Privacy scan of deliverables | PASS: no private home/mount paths, user identity, IP address or supplied task-file path |
| Diff boundary | All changes confined to this run's inventory directory; no source, tool, golden, tolerance or prompt edits |
| Whitespace | `git diff --check` PASS after correcting generated CSV line endings to LF |

The initial documentation commit preceded the independent review. Review removed an unreachable M51
row-major compatibility row; actor/reviewer also removed the shadowed inherited 780M wide row. The corrected
inventory keeps reachable Orin runtime row-major fallback and actual table storage fallbacks. These were
source reachability corrections, not failed or deleted measurement runs. No measurements exist here.

Builds, `check.sh --no-build` (which still compiles selector binaries), shipped-SPIR-V comparison/golden
regeneration, model validation, timing recomputation and gate outputs are inapplicable to this explicitly
source-only stage. No new or changed shader exists for a synchronization diff audit. The existing M51 bz
component-store/compiler caveat is recorded for later exact-module review. Query-dependent legality and
company-side/private evidence remain pending as documented; they are not substituted with historical claims.

Independent review checks completeness, predicates, actual source declarations, packet requirements and
boundaries; it does not need any of the prohibited build/device checks to complete this stage.
