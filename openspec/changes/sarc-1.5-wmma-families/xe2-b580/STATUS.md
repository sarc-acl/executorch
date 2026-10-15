# Xe2 B580 bounded matrix status

Recovery update 2026-10-15T04:21Z. Actor: Codex.
The complete current campaign (including 03:57 owner approval and 04:14 allocation)
was read. No surviving B580 build, model runner or sampler was found; neither GPU
nor desktop-build lock was held at inspection. Previous evidence/builds are reused;
no allocator requery, rebuild or repeated blocker review was performed.

INPUT.md's exact inventory snapshot remains pinned. This turn reconfirmed the
archive SHA-256 `65d1e3ca0eb7e9dffa8e9fbbb1a100bf2cf609868720b9053de67ff8bf5247d8`
and all 18 file hashes, including Xe2 manifest
`5904caa4c1977f256d58b4c1c111e7fdcb2e54bc312676b40c90da6bcf048306`.

Independent new work: a source-derived partial RoPE input recipe is in
prepare_rope_inputs.py and INPUT-PREPARATION.md. It changes two layout bytes per
model, preserving every other byte, and uses the existing CPU-only runtime helper
for revised-input constants. Metadata discovery found exactly two intended targets
in each of the six models. This is NOT a conforming all-six input bundle: 8B width,
operator-temporary legality and GPU correctness remain unresolved. No margin,
gate, baseline or A/A result is claimed.

A detached CPU-only preparation job is being queued with state file
`/mnt/linux-share/hmz-campaigns/wmma-b580/.artifacts/input-preparation/rope-height-v1/job-state.json`.
It takes the desktop-build lock exclusively to avoid interfering with timing,
obeys HOLD, preserves originals and failed partial files, and writes hashes,
byte-equivalence evidence and revised-input runtime-constant outputs. It performs
no build, Vulkan forward, GPU query or timing. After completion, inspect its new
evidence and finish the independent 8B resource-route investigation before parking
on the synthetic-gate question below. No new independent review is requested yet.

The following prior reviewed evidence remains historical context.


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

The old original-input question is superseded by the explicit 2026-10-15 03:57
approval. New question, recorded once on 2026-10-15: the unmodified gate constructs
8B synthetic texture3d production tensors independently of the model files. Their
3584-wide fp16 images exceed the matching 2048 limit. The --linear prefill suite
also includes those shapes. Input repair cannot make this coverage legal.
Please provide an environment supporting those exact resources, or separately
approve a concrete replacement correctness-coverage plan for these unsupported
synthetic cases. The current input-only approval grants no gate relaxation.
No synthetic GPU test has been attempted. See INPUT-PREPARATION.md for source
references. Finish independent input preparation before parking on this question.

## Next step

Finish the detached partial-input preparation, inspect its new outputs, then
finish independent 8B buffer/storage-route work within the retained allocation.
The synthetic-gate question above remains pending; no gate, measurement or new
review is queued. HOLD remains coordinator-controlled. The source recipe makes
no runtime release edit, new shader, threshold or tolerance change.

Recovery budget: coordinator reports $10.75 spent of the $25 recovery cap at
00:42 UTC. This is not a renewed device-run allowance. No further job is queued.
