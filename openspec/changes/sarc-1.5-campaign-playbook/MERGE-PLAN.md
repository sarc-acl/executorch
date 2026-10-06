# Merge plan (draft for the owner)

What it takes to bring the five campaign branches into `dev/1.5`. Nothing here has been started; nothing is
merged. This is a list of the work and its order, written on 2026-10-06 while two campaigns were still closing.
Items marked **[owner]** need a decision before they can be done.

## What is to be merged

| branch | device | state |
|---|---|---|
| `topic/b580-prefill-refine` | Arc B580 | finished, pushed |
| `topic/4070ti-prefill-refine` | RTX 4070 Ti SUPER | finished, pushed |
| `topic/orin-prefill-refine` | Jetson Orin Nano | finished, pushed |
| `topic/780m-prefill-refine` | Radeon 780M | closing |
| `topic/xe2-prefill-refine` | Arc Pro B70 | closing |

All five forked, directly or through a sibling, from one commit of `topic/780m-prefill-refine`, and all five
add to the same dev-zone files (`impl/sarc_dev/Overrides.cpp`, sweep yaml), each inside a block marked with its
device tag. The comparison (`topic/llamacpp-compare`) and this package (`topic/campaign-playbook`) add
documents and tools only.

## Tasks, in order

### M1. Change the next-token check of the gate so that a near-tie flip is not a failure **[owner]**

**Why first.** Every campaign that touched an attention kernel hit the same wall: `sarc/tools/verify.sh` and the
session tools compare the generated next token of the old and the new build, and in the 8B `8da4w` cell some
prompt positions have two candidate tokens with almost equal logits. A kernel that changes the order or the
precision of a sum (fp16 to fp32 accumulation, a fused kernel) moves the logits in the last digits; `8da4w`
re-quantizes activations to 8 bits in every layer, so a value near a rounding boundary turns a difference of
1e-5 into one of 1e-2; 8B has the most layers. The token flips, the gate fails, and the campaign then spends a
round producing reference-error evidence by hand so that the owner decision of 2026-10-04 can be applied. It
happened on the 780M, the B580 and the B70, and the September notes of two contributed devices record the same
flip. It will happen on every further device.

**What to build.** Keep the comparison, and add a second stage that runs only when it differs:

1. Next token equal: pass, as today.
2. Next token different: take the logits of both builds at that position and of the fp32 CPU reference.
   - If the margin between the two top candidates in the reference is below a fixed threshold (a near tie),
     and the new build's error against the reference (rms and maximum, over the production head
     configurations) is not larger than the old build's, report `TIE_FLIP_ACCEPTED` with the numbers.
   - Otherwise fail as today.
3. Report every accepted flip in the gate's summary; it is never silent.

**What exists already.** The campaigns wrote the pieces as dev-zone tools: a near-tie test and a next-token
reader in the RTX 4070 Ti SUPER campaign's `tools/` (`near_tie.py`, `nexttoken.py`), the reference-error
measurement and its rule checker in the Intel and Orin campaigns' `tools/` (`sdpa_ref.sh`, `sdpa_error.py`,
`decide.py`, the logits probe). M1 is to make one shared version of them under `sarc/tools/` and call it from
`verify.sh`.

**What the owner decides.** The margin threshold; whether the error criterion is "not larger than the old
build" (the rule as applied so far) or an absolute bound; and that `verify.sh` may be edited for this, since no
campaign was allowed to touch it. The thresholds are fixed before the merged gate is run on any branch.

**Check.** Replay the recorded cases: the three real flips must come out `TIE_FLIP_ACCEPTED`; the Orin
candidate that was rejected for a larger maximum error on 3B must still fail; a deliberately wrong kernel must
still fail.

### M2. Review the release-zone hooks **[owner]**

Three hooks were committed under the owner decision of 2026-10-05, each inert while nothing selects it:

- the softmax variant name (`Override::softmax_variant` in `impl/sarc/Select.h`, read in the SDPA shader-name
  function): the same patch on the Orin, the RTX 4070 Ti SUPER and the 780M branches;
- the entry point of the 780M's fused attention node, which adds a few lines to the upstream `SDPA.cpp`: the
  decision reserved this one for the owner's review before any promotion;
- on the RTX 4070 Ti SUPER the attention rows are registered from the dev zone instead of the release table,
  which is the form to keep.

Decide the final form of each and add them to `sarc/HOOKS`.

### M3. Merge the dev zone, one branch at a time

Order: 780M first (the others forked from it), then B70, B580, RTX 4070 Ti SUPER, Orin. Each as its own fork PR
into `dev/1.5`. Expected conflicts are in `Overrides.cpp` and the sweep yaml and are mechanical if the
per-device blocks were kept. After each merge: `sarc/tools/check.sh`, `test_sarc_select` (the release tables
must select exactly what they selected before), and the shipped SPIR-V golden unchanged.

Before M3 starts, confirm that the Orin branch carries the corrected softmax kernel (only the elected lane
stores the subgroup's reduced value) and not the first form.

### M4. Decide what is promoted, and promote it **[owner]**

Merging the dev zone ships nothing: every new row is `kUnverified`. Promotion is the checklist of
`sarc/README.md` (move the variant to the release yaml, flip the row, update the golden, attach the `verify.sh`
evidence), one device per PR. Candidates for promotion are each campaign's final profile. Shared shaders need
the other owner's device to re-verify: the two Intel cards share theirs, and the RTX 4070 Ti SUPER and the
Orin share the `8da4w` kernel and the packing shader.

With M1 in place the promotion evidence is produced by the gate itself instead of by hand.

### M5. Independent review

One pass over the merged result by a reviewer that did not take part: recompute each headline number from the
raw run tables kept under each change's `results/`, read every new shader for unsynchronised shared writes
(the Orin finding), and list every file outside the dev zone that changed.

### M6. Fix what is stale

`CONTRIBUTING-A-GPU.md` and the `sarc-1.5-7900xtx-4w` proposal still say the RX 7900 XTX has no table rows; it
has. `sarc/README.md`'s device table needs the new state of the five devices.

## Not part of the merge

- Decode speed, the 4w linear kernel's distance from the vendor backends, and porting the fused attention
  kernel to the other devices: new work.
- The runner's abort after a slow model load (`corrupted double-linked list`): it is in the parent too; worth
  reporting upstream, not a merge blocker.
