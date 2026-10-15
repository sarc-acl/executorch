# Xe2 B580 bounded matrix status

Updated 2026-10-15T00:43:18Z. Actor: Codex.
**BLOCKED before model GPU launch; matrix incomplete.** No active build, queue,
benchmark or sampler. No GPU model forward method, parent snapshot, A/A, gate or
candidate timing has run. No measured cell or margin is claimed.

Recovery read the complete CAMPAIGN.md including the final recovery instruction,
then inspected status, jobs, artifacts and locks. No surviving campaign job or
held device/desktop-build lock was found. Existing desktop/coordinator processes
were left unchanged. Preserved preparation and the parent build were reused.

Own branch `topic/wmma-xe2-b580`; source base
`617897184529ff180af491ee0d5d7de778cbb6c6`. Consistent outstanding work was preserved
at `449be64bc3` before extending the allocation evidence. No push or PR.

## Reviewed input

Inventory `f393c3ff7ff4fc41e4651a5109a8509097c501f6`.
Recovery independently confirmed all 18 per-file SHA-256 hashes against INPUT.md's
provenance record and verified the supplied archive bytes:
`65d1e3ca0eb7e9dffa8e9fbbb1a100bf2cf609868720b9053de67ff8bf5247d8`.
Xe2 manifest SHA-256:
`5904caa4c1977f256d58b4c1c111e7fdcb2e54bc312676b40c90da6bcf048306`.
The prior 53 pinned source checks and private queried property/model records are
preserved. Recovery additionally matched twelve allocation-path/schema sources to
both the base git objects and the parent's recursive export.

## New substantive evidence

BLOCKER.md supersedes the earlier 8B-only interpretation. The actual allocator
uses 3D RGBA16F images, linear tiling, usage 0x0f, flags zero and null pNext, with
standard width packing and default axes. Matching private queries bound every
axis to 2048. The preserved parent object code independently returns:

| Shape | Image extent | Affected cells |
|---|---|---|
| `[5120,32]` | `[8,5120,1]` | both 1B schemes |
| `[5120,64]` | `[16,5120,1]` | both 3B and both 8B schemes |
| `[1,2048,14336]` | `[3584,2048,1]` | both 8B schemes, additional conflict |

The loader constructs all value images before operator rewriting, even with
memory allocation deferred. Each model has two offending RoPE prepack images;
8B 4w/8da4w additionally have 224/192 wide images. No legal whole-model cell remains
under the supplied exports, so the recovery direction to evaluate legal cells
cannot launch any model cell. No known-invalid allocation was attempted.

CPU constant methods: all 60 outputs match serialized metadata, context 2560 in
all six models. Linear group 128 / q4 embedding group 32 remain serialized facts,
not invented constant-method outputs. No forward method was loaded by this helper.

CPU selector audit: nine profiles, 432 linear selections, 36 fallback attention
selections, all 12 inventoried shader names covered. Both identities match control.
Every arm retains default softmax and fused variants
`d64_t16x64s16m8g4roj,d128_t16x128s16m8g8oj`. Actual GPU dispatch is unobserved.
All twelve parent SPIR-V module declarations equal the inventoried shared-byte totals;
this is conditional kernel evidence, not whole-model legality or correctness.

Parent-v2's five recorded executable/library hashes still match; shipped-SPIR-V
check PASS (53). The parent was not rebuilt. No shader changed in this run.
Unmodified check.sh --no-build PASS (release 1240 / dev 1921 checks).
Full correctness, next-token/decode, sensor validity, A/A and performance evidence
remain absent/BLOCKED, never passed by inference from CPU checks.

## Evidence and review boundary

Public small evidence: BLOCKER.md, proposal.md, matrix.json,
results/cell-coverage.csv, results/module-declarations.csv,
results/selector-dispatch.csv and EVIDENCE.md.
Private raw evidence: own artifact root's preflight/recovery/, including allocation
audit with serialized value IDs/calls, exact CPU extent outputs, constant outputs,
dispatch outputs, source/library/helper hashes and commands. Earlier preflight and
parent export/build records are preserved. There are no timing rows to recompute.

One independent review is requested for the NEW allocation-path proof, the expanded
all-six-cell blocker and completed CPU dispatch evidence. No additional preflight
review loop or GPU/build request is needed to assess that evidence. After this review,
park for the owner; do not requery or recompute the unchanged blocker.

## Decision needed from the owner

This updates the single question originally recorded at 2026-10-14T20:09Z; it is
not a repeat request for an allocation-limit exception. The recovery instruction
permits legal cells only, and the expanded evidence finds none among these six.
Please supply a separately authorized and reviewed conforming model/runtime path
covering BOTH the common RoPE height and 8B linear width, or an owner-provided
environment whose exact matching image queries support those extents. New source
or inputs need provenance and revalidation before any GPU baseline or gate.
Re-exporting models, changing runtime allocation/storage, changing drivers or
waiving a queried limit is not authorized within this stage. Required 8B cells
remain BLOCKED along with the 1B/3B cells; the finite matrix is not complete.

## Next step

Independent review of EVIDENCE.md/BLOCKER.md and raw artifacts, then waiting_owner.
No detached job is queued. HOLD was neither created nor removed; host policy,
services, models, historical checkouts, release code, prompts, thresholds and
sarc/tools/goldens were left unchanged.

Recovery budget: coordinator reports $10.75 spent of the $25 recovery cap at
00:42 UTC. This is not a renewed device-run allowance. No further job is queued.
