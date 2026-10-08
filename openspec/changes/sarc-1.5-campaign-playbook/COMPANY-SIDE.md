# Running a campaign from the other side: a coordinator without Claude, Codex, our model files or our hosts

What this file is for: the coordinator who runs the next campaigns (RX 7600, 7900 XTX, M51/Xclipse) works
under four constraints that the first five campaigns did not have. This file says what changes because of
them, and what does not. It is an addendum to `COORDINATOR.md` and `PLAYBOOK.md`, which remain the authority
for everything it does not mention. Written 2026-10-07 from the five finished campaigns and from the first day
of the RX 7600 campaign, which already ran under these constraints and is the worked example throughout;
section 6 updated 2026-10-08 after the RX 7600 campaign closed.

The four constraints:

| # | constraint | what it touches |
|---|---|---|
| C1 | the actor is an opencode agent on an internally hosted model, not Claude; the budget for a frontier model is small | how tasks are phrased, what the harness must decide on its own |
| C2 | no Codex reviewer | who reviews, where, and from what |
| C3 | the `.pte` model files were exported independently, not copied from our side | how comparability is checked |
| C4 | nothing internal (host names, users, serials, paths, device details beyond what `CLAUDE.md` already names) may reach the public remote | what the agent pushes and how it is checked |

Everything else holds: `RULES.md` word for word, `OWNER-DECISIONS.md` Parts 1 and 2, the gate, the stop rule,
the task-file mechanism, the patrol. A campaign on the other side is not a lighter campaign. It is the same
campaign with the judgment moved out of the model and into scripts and into the review.

## 1. The campaign shape: port, do not search

The gain of the five campaigns came from three things (`LESSONS.md` L1, L4, L5): the attention kernels,
the fp32 no-tail softmax, and a linear kernel chosen per layer shape. The searches did not pay (L2, L6):

| search | device | cost | end-to-end result |
|---|---|---:|---:|
| sampled parameter search, four kernel families | Arc Pro B70 | 33 h | +1.1 % |
| QK^T / attention x V enumeration, 1,796 configurations | Radeon 780M | about 20 h | 0 (the final profile does not take that path) |
| linear tile sweeps | three devices | hours each | none faster |

The RX 7600 confirmed the other half on its first day: the 780M's fused attention kernel, ported without a
change, gated at **+18.22 %** with every next-token item SAME; the softmax at +1.48 %. No search was needed to
get there.

So the default campaign on the other side is `PLAYBOOK.md` steps 4 to 13 with step 9 reduced to the port
list and nothing after it:

1. Parent build, snapshot, baseline, A/A (steps 5 to 7). Not negotiable: this is where C3 is caught (section 4).
2. Candidates, in this order, one gate each: fused attention kernel; fp32 no-tail softmax; the linear kernel per
   shape from the sibling's profile; for 8da4w the whole-texel weight staging if the device's phase timing shows
   the fetch above the multiply (L3). For a second card of a family already tuned, the sibling's final profile is
   candidate 0 (L7).
3. Stop by N3 after two consecutive gated candidates under 2 %, or when the port list is exhausted. **No sampled
   search, no enumeration, no tile sweep unless the owner orders one in the task file** (N1, D2). A model that
   is good at executing and less good at judging is at its best on a list that ends.
4. Final verification on the committed branch (step 13), push.

Budget for this shape, from the RX 7600's first day: the two attention candidates took about 14 hours of
device time including the A/A and the gates, with the actor sharing the host's CPU with another job.

### M51/Xclipse is the exception

Nothing ports from RDNA3 to Xclipse: subgroup size, cooperative-matrix shapes and the shared-memory budget
differ, and the fused kernel assumes a workgroup that is one subgroup. For this device:

- Start with what needs device hours and no judgment: the attention rows (L1, the +46 to +58 % layer, which
  this device does not have yet) through the sibling-independent path of `RULES.md` R3 (a `<Device>Sdpa.cpp`
  registration of `kUnverified` rows), then the linear tile screen as a deterministic sweep with a fixed
  configuration list, not a sampled search.
- Try the fused kernel only after the attention rows are in and gated, and only as a candidate that may fail.
- If any frontier-model budget exists, spend it here, on reading the first Xclipse shader dumps and the attention
  kernel's shape choices, and nowhere else. The RDNA3 campaigns do not need it.

## 2. Talking to an actor that is not Claude (C1)

What the RX 7600 actor did well on its first day, and which the task file should keep asking for: thresholds
committed before the first measurement; the A/A before any candidate; the sibling's tools copied and adapted;
`STATUS.md` rewritten after every step; a failed start recorded with its cause and moved to `superseded/`;
host names and paths as placeholders in everything it committed.

Where the five campaigns needed judgment, and where a weaker actor will not supply it unasked:

| situation | what Claude did | what to do instead |
|---|---|---|
| a gate fails once | found the failing run and what distinguished it before retrying (L10) | the task file says: after one failed gate, name the failing run and its cause in `STATUS.md` before any retry; a second blind retry is a stop |
| a candidate reads "inside the band" | applied N3 and D1 without being told | the session script prints `GAIN_BELOW_BAND` / `GAIN_ABOVE_BAND` itself from `thresholds.txt`; the actor copies the verdict, it does not form one |
| a new shader is written or ported | read it for shared-memory races before the gate (R7) | the task file names the three questions to answer in `STATUS.md` before the gate: who writes each shared slot, what orders each cross-lane read, does the kernel assume one subgroup per workgroup |
| an environment variable drives the candidate | noticed when it was empty (RX 7600 did, after 7 minutes) | the session script refuses to start when a variable it depends on is empty, and prints the dispatched kernel names of the first run so the actor can compare them with the expected list |
| the owner's answer is needed | wrote "Decision needed from the owner" and continued with independent work (L36) | same, and the coordinator checks on every patrol that it did continue; a weaker actor tends to wait |

Phrasing rules for the task file and every relayed message:

- Name a script and its expected output, never an outcome. "Run `tools/gate.sh c2` and paste its last 12 lines
  into `STATUS.md`", not "gate candidate 2 and record whether it passed".
- One instruction per sentence. The RX 7600 actor executed multi-step instructions correctly when they were
  numbered and lost nothing; dense paragraphs are where instructions get dropped.
- Say what is forbidden in the same place as what is asked. R10 is a list; copy the lines that apply to the
  step into the step.
- Every owner decision goes into the task file on the host, dated (L35). This is more important with a smaller
  context window: assume the actor re-reads only the task file after any interruption.

What the model's size changes about the flow file: nothing. The flow in `flow/` runs the actor and the reviewer
on the GPU host; the reviewer role is filled as section 3 says.

## 3. Review without Codex (C2)

Independent review found two real bugs in five campaigns, both after every test had passed: the Orin's
data race (L8, 6.5 hours of device time to fix after the campaign had closed) and the fused kernel's
unsynchronised cross-lane reads, found by the RX 7600 actor itself while porting (section 6). Review is not
optional under C2; what changes is who does it and from where.

**The review happens on our side, from the pushed branch.** It needs no GPU, no model file and no host access:
the reviewer recomputes the medians and geometric means from `runs.csv`, reads the diff against the parent, reads
every new shader for shared writes, and checks the gate outputs. On 2026-10-06 our coordinator reviewed the
RX 7600 branch this way in under half an hour of model time and confirmed +1.48 % and +18.22 % to the second
decimal from the raw rows.

For that to work, each push must carry what the reviewer reads:

1. `<change>/results/<tag>/sessions/<session>/runs.csv` and `checks.csv` for every timed session, complete
   (the RX 7600 had copied one file two minutes before its session ended; the review caught it).
2. The gate outputs as files, not as prose: `verify.sh` output against the snapshot, the SDPA tier counts, the
   next-token table, `pdiff.csv`.
3. `proposal.md` with the diff of any release-zone hook (D4) and the thresholds as committed before the A/A.
4. `STATUS.md` current to the push.
5. For every new or ported shader, the three shared-memory answers of section 2 in `STATUS.md`.

Cadence: push after every gated candidate, not only at the end. Findings come back as plain English in the
task file's "Owner decisions" section, dated, relayed by the coordinator; the actor treats a finding like any
owner decision. The reviewer's "done" for R11 is given the same way, after the final push.

If a second local agent is available, it may do a first pass as the flow's reviewer: a separate opencode
instance with the R12 checklist and no access to the actor's conversation finds the mechanical errors (a copied
file short by a row, a status older than the last gate). It does not replace the review on our side, because
the two bugs above needed someone to read a shader with the memory model in mind.

## 4. Model files that were not copied from us (C3)

Both sides export the same models with the same recipe, but a `.pte` is not a `.gguf`: there is no shared
checksum, and an export that differs in one setting gives numbers that look right and compare with nothing.
Three checks, all already in the kit:

1. **Metadata, before anything is built.** `openspec/changes/sarc-1.5-llamacpp-compare/kit/MODELS.md` (branch
   `topic/llamacpp-compare`) records for each of our six files its context length, group size and embedding
   quantization, read from the file. Record the same three values for each file on the other side, in the
   device's `<change>/results/<tag>/MODELS.md`, and compare. A difference in any of them stops the campaign
   before the parent build; it is an owner question.
2. **Baseline within 3 % of the published table (N7, patrol check 8).** The re-measured baseline of step 6 must
   land within 3 % of `results.md` section 1 for every cell, or the difference must be explained in `STATUS.md`
   before the first candidate. The RX 7600 passed this on all six cells. A device that is new to the table has
   no published row; then the check is the `x stock` ratio of `results.md`, which is far less sensitive, and the
   metadata check above carries the weight.
3. **Next token of the parent on the three prompts**, from `s0-parent-verify` (step 5). Our side publishes the
   parent's next tokens for the three models and two schemes with the comparison results; a different token at
   the parent, before any candidate, is an export difference, not a kernel difference.

What cannot be checked this way: tokenizer and prompt file identity. `prompt_2048.txt`, `prompt_check.txt` and
the 1,304-token unaligned prompt are in the repository; the RX 7600 host was missing the last one and lost the
unaligned check lines in every `verify.sh` output. Verify the three prompt files by `sha256sum` against the
repository copies in step 4, and treat a missing one as a stop.

## 5. What never reaches the public remote (C4)

The rule is R3's last point plus `CLAUDE.md`: the public remote carries code, results and documents; host
state, people and infrastructure stay out. What is already public and may be named: the device names and
tags of `CLAUDE.md` (M51/Xclipse, 7900 XTX, RX 7600, Adreno 840, Mali-G1), the branch names, driver and compiler
versions as version strings. What may not: host names, user names, home paths, serial numbers, IP addresses,
internal service names, the names of people, anything about the host beyond its device and its software
versions.

The RX 7600 actor got this right by hand (`<campaign-root>`, `<host>` as placeholders in `STATUS.md`). Make it
a script instead. Keep a denylist **outside the repository** (one pattern per line: host names, user names,
internal domain, anything site-specific) and run this before every push, from the working copy:

```bash
#!/usr/bin/env bash
# leak-check.sh <denylist-file> [<base-ref>]: fail if anything in the commits since <base-ref> matches the denylist.
set -euo pipefail
list="$1"; base="${2:-origin/dev/1.5}"
if git log -p --format='%H %s' "$base"..HEAD | grep -n -i -F -f "$list"; then
  echo "LEAK_CHECK_FAILED: the lines above match the denylist; do not push." >&2; exit 1
fi
if git log -p --format= "$base"..HEAD | grep -n -E '/home/[a-z]|/Users/[A-Za-z]|[0-9]{1,3}(\.[0-9]{1,3}){3}'; then
  echo "LEAK_CHECK_FAILED: a home path or an IP address is in the diff; do not push." >&2; exit 1
fi
echo LEAK_CHECK_PASSED
```

The task file says: `tools/leak-check.sh <denylist> && git push origin <branch>`, never a push without the first
half. Add the denylist's path to the task file, not its contents. Result CSVs carry paths in their `build` or
`host` columns on some devices: the sampler writes `<campaign-root>` there, not the real path (the merge plan
already lists cleaning the existing ones).

Raw artifacts (builds, traces, `.rgp` captures, model files, logs over a few MB) stay in `<artifact-dir>` outside
the working copy (R1, `CLAUDE.md`). Only `results/<tag>/` with its CSVs and the change's documents are
committed.

## 6. The fused kernel: which file to port, and what still has no name

Two items that the RX 7600 inherited from `topic/780m-prefill-refine`. The first was fixed by the RX 7600
campaign itself on 2026-10-08; the second is still open on both branches. State as of the RX 7600 push of
2026-10-08 (head `c0be2c6c27`, campaign closed at +26.90 % geomean).

1. **Port `fused3sb`, not `fused3`.** The 780M's fused attention kernel exists in three generations,
   `sarc_dev_780m_sdpa_fused{,2,3}.glsl` (same arithmetic, three structures; the 780M's final configuration
   uses the third). In all three, each lane writes its own slot of `Rsh`, `Psh` and `Dsh`, then reads the other
   lanes' slots after `memoryBarrierShared()` alone, which orders the lane's own accesses and waits for nobody.
   It is correct on RDNA3 because a wave runs in lockstep with no divergent branch between store and load, and
   every test agrees; under the Vulkan memory model it is a race (L8 is the same class). The RX 7600 actor found
   it while porting and, as merge-plan item M2a, committed `sarc_dev_780m_sdpa_fused3sb.{glsl,yaml}`: a sed copy
   of `fused3` with `subgroupBarrier()` after each of its 14 `memoryBarrierShared()` calls and nothing else
   changed ("sb" = subgroup barrier). Gated against its parent on the RX 7600: **+0.00 % geomean**, output
   byte-identical to `fused3` in 21 of 21 SDPA cases, all tiers 0 mismatches. The barrier is free on lockstep
   hardware, as expected. Variant names: `fused3sb_d64_t32x32g11s32rko` and `fused3sb_d128_t16x64g11s32rko`,
   selected through `ET_VK_SARC_780M_SDPA_FUSED` like the originals; the file keeps the `780m` prefix because
   `impl/sarc_dev/780m/Sdpa780mFused.cpp` builds the shader name from that prefix (RX 7600 owner question 3;
   default: keep the name).

   What this means for the next campaigns: port `fused3sb` and never `fused3`. On a device whose waves are not
   lockstep (M51/Xclipse) `fused3` is a port of a bug; `fused3sb` is the kernel with the assumption written
   down. One assumption is still only in the header, not enforced: the workgroup must be exactly one subgroup.
   A dispatch guard that checks the subgroup size against the workgroup size belongs in the device's selector
   before the kernel is tried on anything but RDNA3.

   What is still open on our side: `topic/780m-prefill-refine` ships `fused3`. Whether the 780M adopts
   `fused3sb` (one re-gate of candidates 8 to 11 and one timed session, by the RX 7600's numbers a no-op) is a
   merge-plan decision, not a prerequisite for the company side any more.

2. **The final configurations still have no single name.** The 780M's candidate 11 is
   `ET_VK_SARC_DEV_PROFILE=780m-refine3` plus `ET_VK_SARC_780M_PROFILE=c11`. The RX 7600's recommended
   configuration is four variables: `ET_VK_SARC_UNVERIFIED=1`, `ET_VK_SARC_780M_PROFILE=c7` (the softmax),
   `ET_VK_SARC_780M_SDPA_FUSED=<the two fused3sb variants>`, and `ET_VK_SARC_RX7600_PROFILE=rx7600-refine2` (the
   linear kernel per shape; `rx7600-refine3` is the rejected whole-texel candidate). The RX 7600's own first
   start of candidate 2 ran seven minutes without the fused kernel because one of these variables was empty
   (section 2): that is the cost of a configuration that is a composition. One dev-profile entry in
   `impl/sarc_dev/Overrides.cpp` that carries the whole stack (`780m-final`; for the RX 7600 a name that is not
   already taken, say `rx7600-final`) is what the comparison, the merge plan and the promotion PR will name; it is
   also what the empty-variable guard of section 2 compares against. A campaign that closes with a composed
   configuration should at least write the exact variable set into `proposal.md`, as the RX 7600 did.

The merge plan carries both. A campaign whose fused-kernel candidate was measured with `fused3` before the
switch should say so in its `proposal.md`, so that a re-gate on `fused3sb` is expected and not a reopening.

## 7. Patrol and reporting differences

`COORDINATOR.md`'s patrol applies with these changes:

- Check 1 (an hmz run ended): with a non-Claude actor, also check that the actor's last `STATUS.md` write and
  its last commit are within the same hour; a model that stops producing tool calls does not always end the
  run.
- Check 8 (baseline vs the published table) is a stop under C3, not a note.
- New check: `LEAK_CHECK_PASSED` is in the log of every push. A push without it is reported to the owner as an
  incident, and the owner decides whether the history is rewritten (the remote is public; a force-push of a
  topic branch is the owner's call and R3 otherwise forbids it).
- New check: the review findings from our side were appended to the task file within one patrol of arriving.

Reporting to our side: a push is the report. The push message of the branch head should say what the reviewer
should look at first ("c2 gate complete, `runs.csv` of `c2-fused`, new shader answers in STATUS.md section
R7"). Nothing else needs to travel; if something cannot be in the repository (C4), it does not travel either,
and the coordinator says so in `STATUS.md` with a placeholder ("host detail withheld: <what kind>").

## 8. Quick reference

| question | answer |
|---|---|
| which candidates | the port list of section 1, in order; no search unless the task file orders it |
| who reviews | our side, from each push; a second local agent may do a first pass |
| when to push | after every gated candidate and at the end, after `leak-check.sh` |
| what proves the models match | the three metadata values, baseline within 3 %, parent next tokens, prompt checksums |
| what the actor must never decide alone | gate pass or fail, inside or outside the band, a retry after a failure, a release-zone edit, a push without the check |
| what to do with an owner question | "Decision needed from the owner" in `STATUS.md`, then continue with independent work |
| which fused kernel to port | `fused3sb`, never `fused3`; add a subgroup-size guard before a non-RDNA3 device (section 6) |
| what still has no name | the final configurations of the 780M and the RX 7600; give the stack one dev-profile entry (section 6) |
