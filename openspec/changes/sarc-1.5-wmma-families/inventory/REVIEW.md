# Independent source review

2026-10-14 19:20 UTC. Reviewer: fresh Codex context, independently of the actor.
Source: `617897184529ff180af491ee0d5d7de778cbb6c6`. Reviewed the live tree after the
corrections below, including the draft commit `578bddd118` and its subsequent edits.

**PASS for the bounded source-only inventory.** No unresolved source-inventory
finding remains. The actor must commit the reviewed corrections and this report
locally before closing the stage. No device legality, correctness or performance
approval is implied.

The reviewer first read the complete assembled campaign, including the final launch
decision, then checked status and existing processes. No inventory build or GPU job
was active. No owner question was pending.

## Independent checks

Traced all eight final environments from campaign records and `sarc/README.md:101-108`
through `impl/sarc_dev/Overrides.cpp`, the AMD/Intel/NVIDIA tables and the build/runtime
paths in `impl/sarc/{Select,Q4gswCoopmat,Dq8caCoopmat}.cpp`. Paths abbreviated with
`impl/` or `glsl/` below are under `backends/vulkan/runtime/graph/ops/`.

Checked the ordered shape predicates, prerequisite table service, candidate-before-row
suffix lookup, outer-profile replacement, fixed int8 activation layout and runtime
row-major table reselection. Confirmed reachable inherited/table fallbacks and excluded
fully shadowed ancestors, unrelated profiles, attention, fp linear and decode kernels.
In particular, Orin's shipped row-major g44 kernel remains reachable through
`Dq8caCoopmat.cpp:94-108` after its final bf profile builds a row-major graph.

Recomputed registered storage masks and tile/subgroup geometry from the actual rows,
then checked complete YAML default/variant parameter maps against each manifest entry.
Read the wrappers and shared-array declarations in every used body. Independently
summed uvec4/uvec2/uint/int/float/fp16 capacities, double buffering, parameter arrays,
column-major B, RX A_V2/pitches, pool maxima and texture drain aliases for every
candidate. Every declared byte total agrees, including RX 7600 59,392/60,160 bytes,
780M afmb1 60,160, and M51 bz 40,960 rather than the selector's 37,888-byte estimate.
Checked subgroup-scope MMA operand/accumulator types, workgroup invocation counts,
tile divisibility and full A-map coverage. Declared requirements remain separate
from unqueried device limits and compiled allocation.

Independently verified the SHA-256 of all 53 listed source inputs against both the
pinned git objects and the checkout. Checked every machine-readable source reference's
commit and line bounds, plus exact row/YAML name references. Recomputed source-alias
fingerprints and compared alias wrapper text and parameters. Verified unique candidates,
complete origin coverage and candidate/member expansion. Running `generate.py` produced
byte-identical CSVs, manifests and hashes.

| Item | Independently verified count |
|---|---:|
| Ordered device routes | 42 |
| Storage-specific origin rows | 60 |
| RDNA3 RADV candidates / dispatch names | 19 / 20 |
| Xe2 candidates / dispatch names | 12 / 12 |
| NVIDIA Ampere/Ada candidates / dispatch names | 11 / 11 |
| Xclipse candidates / dispatch names | 7 / 7 |
| Total source-distinct candidates / dispatch names | 49 / 50 |
| Candidate/member legality rows, all PENDING_QUERY | 110 |

Read all company-side packets. They require owner relay and separately launched work,
RADV evidence before XTX matrix work, controlled attention across linear arms, six-cell
coverage and unmodified gates. M51 retains private evidence, both 8B schemes, the existing
`ET_VK_EXECUTE_NODE_THRESHOLD=32` setting for every 8B arm, historical near-tie/partial
coverage caveats and exact-compiler component-store review. No private numeric results,
run paths or infrastructure identifiers were introduced.

## Findings resolved before acceptance

The draft incorrectly included M51's old zpgtr table row as a final c8 runtime fallback.
`Dq8caCoopmat.cpp:205` starts build selection with kAny and `:238` fixes the resulting
layout. Every fitting Xclipse zpgtr row is replaced by c8's zpg_bt pick: compare
`Overrides.cpp:524-526`, `:3252-3254` and `sarc/table_amd.cpp:70-76`. The graph is therefore
4h4w and cannot enter the row-major runtime fallback. Removed that route and its two
storage candidates; `PROFILES.md` now explains the boundary.

The actor also identified the fully shadowed inherited 780M wide row during review.
Confirmed that c11's fifth pick (`Overrides.cpp:2707-2708`) has the same N>=1024 predicate,
tile alignment, storage and fit as the inherited row (`:86-87`, `:1182`). Removed the
ancestor route and candidate. Other inherited and table fallbacks remain represented.

The actor merged one exact RDNA3 source alias pair. Independently confirmed identical
wrapper text from `#version`, body and full merged parameters for the dev/release
`t256x128k32g24s32f32cbt_texture3d_texture2d_half` names. Both dispatch aliases and all
origins remain recorded. No compiled binary identity is asserted.

## Scope and inapplicable checks

Compared base-to-HEAD and base-to-working-tree file lists and inspected untracked
deliverables. Every added/changed path is inside this inventory directory. No release
code, shader, `sarc/tools`, golden, tolerance, prompt or threshold file changed.
`git diff --check 6178971845` passed after CSV output was corrected to LF.

Measured-number recomputation, exact measured-build/gate matching, SPIR-V golden checks,
timing, sensor/model verification and per-device queried legality are inapplicable to
this source-only stage. No new or modified shader requires a change-specific shared-write
review; the M51 compiler-sensitive existing component writes remain explicit future
evidence requirements. `check.sh --no-build` compiles selector tests and was not run
under the no-build boundary. No builds, GPU queries/work, model access, remote access,
submodule initialization, services, host-policy changes, pushes or PRs were performed.
Later measured family coverage, margins, minimality and any NVIDIA split remain pending.
