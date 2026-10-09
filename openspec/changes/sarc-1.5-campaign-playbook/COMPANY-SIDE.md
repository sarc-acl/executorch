# Running a campaign from the other side: a coordinator without Claude, Codex, our model files or our hosts

What this file is for: the coordinator who runs the next campaigns (RX 7600, 7900 XTX, M51/Xclipse) works
under four constraints that the first five campaigns did not have. This file says what changes because of
them, and what does not. It is an addendum to `COORDINATOR.md` and `PLAYBOOK.md`, which remain the authority
for everything it does not mention. Written 2026-10-07 from the five finished campaigns and from the first day
of the RX 7600 campaign, which already ran under these constraints and is the worked example throughout;
section 6 updated 2026-10-08 after the RX 7600 campaign closed; section 6 again and section 9 added 2026-10-09; section 10 (the llama.cpp request) added 2026-10-10
after the second round on our side (the fused kernel ported to four more devices, 780M round 3).

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

### M51/Xclipse: what the port looked like there (closed 2026-10-08; 8B still to do)

The M51 campaign (`topic/m51-prefill-refine`, 38 commits, forked from the 780M's candidate 11 like the RX 7600)
followed the port list and closed on 2026-10-08. What it taught, number-free because the device owner's rule keeps
every figure local:

- The device already had attention rows (`impl/sarc/table_amd.cpp`, xclipse, `kUnverified`), so it was in the
  780M's position, and the fused kernel ported: candidate c1 gated outside the band. Two findings: the 780M form
  with `memoryBarrierShared()` alone **gave random wrong rows at S = 2048 on this driver** (the race of section 6
  is real on non-lockstep hardware; the port uses `memoryBarrierShared()` + `barrier()` at every exchange), and
  **the one-pass form failed with either barrier form while the two-pass packed form passed**: the one-pass path
  carries another lockstep assumption. The NVIDIA ports should read this before trusting one-pass on a new driver.
- The whole-texel 8da4w staging (`zpg_bt`) on every shape: c2, outside the band. A head_dim 64 fused variant on a
  64-wide subgroup: c4, inside the band. The fp32 softmax: not applicable once the fused node serves every call.
- **Known property**: `zpg_bt` is a 4h4w-layout kernel; for prompts whose length is not a multiple of 128 the
  layout falls to the stock tiled kernel, so the 8da4w cells of the final stack are **slower than the parent at
  unaligned lengths**, and the fused attention is not dispatched there. The 2048-token headline is unaffected; the
  owner kept the stack and recorded this as a property. It must be in the promotion decision.
- **8B was not loaded on the board** (device owner's decision of 2026-10-06), so every gate is PARTIAL and c3 is
  ungated. **Lifted on 2026-10-08: `OWNER-DECISIONS.md` N10 says how 8B is done** (the setting the owner calls
  "thread hold = 32" for the 8B runs).
- Review with no numbers: our side can read the code and the prose and check the tools, but cannot recompute a
  single cell. Where the device owner's rule applies, the campaign's own `summarize.py` output and `runs.csv` stay
  local, and the reviewer on our side states that its review is of code and method only.


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

   Closed on our side 2026-10-09: `topic/780m-prefill-refine` (head `71b43d903`) adopted `fused3sb` in its final
   configuration; measured against `fused3` on the same build: -0.14 % geomean, inside the band, SDPA output
   byte-identical. Every fused port since (RTX 4070 Ti SUPER, Arc B580, Arc Pro B70, Jetson Orin Nano) carries
   the barriers. Nothing ships `fused3` any more.

2. **The RX 7600's final configuration still has no single name.** The 780M's is now one name,
   `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=780m-final` (round 3, 2026-10-09: dispatch identical to
   candidate 11, +33.90 % over `dev/1.5`); before that it was `ET_VK_SARC_DEV_PROFILE=780m-refine3` plus
   `ET_VK_SARC_780M_PROFILE=c11`. The four second-round ports were single names from the start
   (`4070ti-fused1`, `b580-fused1`, `b70-fused1`, `orin-fused1`): copy that pattern. The RX 7600's recommended
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
| what still has no name | the RX 7600's final configuration; give the stack one dev-profile entry, as `780m-final` now is (section 6) |
| what a review will reopen a campaign for | the eight items of section 9.2; do them before the first timed run |
| the full-subgroups pipeline flag (F1) | a known release-zone defect; record it, do not repair it in a campaign (section 9.3) |
| decode | measure it with and without the final profile at closing and report it (section 9.4) |
| a throttle flag in loaded runs | find out per device which bits are thermal; power and current limits are recorded, not rejected (section 9.2 item 3) |
| waiting for the owner | record the question once, do the independent work, then do nothing; no recomputation while waiting (section 9.5) |

## 9. What the second round on our side found (2026-10-09), and what to do with it

Between 2026-10-08 and 2026-10-10 the fused attention kernel was ported from the two RDNA3 devices to every
other device on our side, and the 780M was closed with `fused3sb` and one profile name. Five campaigns, five
reviews. This section is what the company-side coordinator needs from them. Branches named here are on the
public remote; read with `git show origin/<branch>:<path>`.

### 9.1 Results, and which branch is the worked example for which kind of device

| device | what was ported | over its tuned parent | over pristine `dev/1.5` | branch (change directory `sarc-1.5-<tag>-fused-port`) |
|---|---|---:|---:|---|
| RTX 4070 Ti SUPER | `fused3sb` unchanged, plus a guard: the kernel returns unless the workgroup is exactly one 32-wide subgroup | +11.49 % | +63.28 % | `topic/4070ti-fused-port` |
| Arc B580 | a multi-subgroup form: 8 x 16 x 16 matrix shape, a workgroup of 4 or 8 subgroups of 16 lanes, each subgroup owning part of the score columns | +8.82 % | about +72 % | `topic/b580-fused-port` |
| Arc Pro B70 | the B580's kernel, brought over unchanged and confirmed (5 hours, one candidate) | +8.26 % | +72.87 % | `topic/b70-fused-port` |
| Jetson Orin Nano | `fused3sb` with the same guard as the 4070 Ti | +6.13 % (cross build, item 7 of 9.2) | +76.70 % | `topic/orin-fused-port` |
| Radeon 780M | `fused3` replaced by `fused3sb`, profile `780m-final` | -0.14 % (no change) | +33.90 % | `topic/780m-prefill-refine`, "Round 3" |

Which one to read before a port:

- **A device with a 16 x 16 x 16 fp16 matrix shape and one subgroup size (RX 7900 XTX, most likely)**: the 780M
  and RX 7600 records, then `topic/4070ti-fused-port` for the guard and for what a review asks of a port.
- **A device whose matrix shape or subgroup size differs from the kernel's assumptions**: `topic/b580-fused-port`,
  `proposal.md`, "the port in two parts". The straight port compiled but spilled registers and was slower than
  the three separate kernels; the form that won splits the workgroup into several subgroups. Read the compiler
  statistics before timing anything.
- **A device where the one-pass form does not work (M51)**: the Orin record measured one-pass against two-pass
  per head dimension; two-pass was 17 % faster at kernel level for head_dim 64 and still inside the band end to
  end (+0.54 %). A two-pass kernel is a legitimate final answer.
- **A second device of a family that already has a port**: do a confirmation, not a port. The B70 task
  (`topic/b70-fused-port`, section "What to bring from the B580 and how") is the template: pin the source commit,
  bring the files unchanged, show the SPIR-V is byte-identical, repeat only the selection screen over variants
  that already exist, gate once, time once. If the RX 7900 XTX behaves like the RX 7600, this is its shape.

A second candidate was tried on three devices (fp32 no-tail softmax for the calls the fused kernel does not
take on the B580: -0.13 %; two-pass for head_dim 64 on the Orin: +0.54 %; none on the 4070 Ti) and adopted on
none. After the fused kernel, attention is a few percent of the prefill; do not expect a second attention
candidate to leave the band.

### 9.2 What the reviews reopened campaigns for: do these before the first timed run

Every item below is a rule that already existed. Each one cost a campaign between three hours and a day when a
reviewer found it afterwards.

1. **All three SDPA tiers in the gate** (`all`, `extended`, `full`), 12 passes each, with the hashes of the test
   binary and runner library and every pass's exit status recorded. The 4070 Ti gate ran two tiers and was
   reopened.
2. **R6, every predicate evidenced per run.** "No other GPU workload during the run" needs sampling WHILE the
   runner executes (the guard's process pattern and the DRM clients with their engine time), not a check before
   and after. A monitor such as `nvtop` attached to the device counts as something to record; put monitors in
   the guard's pattern. The 780M round was reopened for this.
3. **R6, the throttle record, and which reasons are thermal.** Sample the driver's throttle status in the same
   loop as the clock. Then decide BEFORE the first session, in `thresholds.txt`, which reasons reject a run. On
   the 780M the status word had bit 1 set in most loaded runs: that bit is a package power limit, set by design
   under load; the thermal bits never appeared. Owner ruling: thermal reasons reject, power and current limits
   are recorded only, unknown bits reject until read. The bit table is per chip family (for the 780M:
   `drivers/gpu/drm/amd/pm/swsmu/inc/pmfw_if/smu13_driver_if_v13_0_4.h`); a discrete RDNA3 card uses a
   different header. Do not copy the 780M's mask: read the header for the device at hand and quote it (L16).
4. **R3, the shared test file is never edited in place.** What a new kernel needs in
   `test_llama_microbench.cpp` goes in as insert-only delimited blocks; `git diff --numstat` must show 0
   removed lines. Take the blocks from `topic/4070ti-fused-port`.
5. **R5, the measured build is an export of one commit and, recursively, of its submodules from object stores.**
   A build whose submodule trees came from the working directory was rejected on the 780M and everything was
   measured again. `tools/export_recursive.sh` on `topic/780m-prefill-refine` is the worked example.
6. **Thresholds are written before the first candidate and not touched after.** A rule added to
   `thresholds.txt` after a candidate had numbers was removed by review on the B580. Creating and calibrating
   the campaign's own file before the first candidate is what R4.2 asks for and is not a "protected file"
   change (owner ruling on the B70).
7. **Know which shader compiler built what you measure.** The Orin's cross image carried a different `glslc`
   than the pinned one; `spirv_golden.py` differed on 14 shipped variants of other devices and the campaign's
   own kernels were compiled to other bytes. The reviewer reopened the campaign at closing. Owner ruling
   2026-10-09: byte identity with the golden is not required of a cross build; the numbers stand as measured
   and the compiler difference is recorded as a known limitation. What to take from it: run `spirv_golden.py`
   on the parent build BEFORE the baseline, and if it differs only because of the compiler, write that down on
   day one ("Known limitation: shader compiler") instead of discovering it in the last review. This matters
   most where the build is a cross build or runs outside the container (M51).
8. **No wait that the owner did not order.** An actor added "wait until the desktop's share of the card drops"
   after the owner had ruled "measure at once"; the sessions behind it were repeated. If runs are rejected
   because something else holds the card, report it; do not add a wait.

For the reviewer's side: keep the two `test_sarc_select` executables of the final check, because a reviewer who
may not build cannot run `check.sh --no-build` (it compiles them); the actor's output is accepted for that item
(owner ruling). Figures quoted from another campaign are citations with branch and commit, outside the numeric
audit of the campaign that quotes them.

### 9.3 F1: the full-subgroups pipeline flag is a known defect; do not repair it in a campaign

The specification requires (VUID-RuntimeSpirv-OpTypeCooperativeMatrixKHR-10770) that a pipeline whose shader
uses cooperative matrices is created with `VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT`, or that
the module is SPIR-V 1.6 or later. The runtime does neither: SPIR-V 1.3, a Vulkan 1.1 instance, stage `flags`
0 in `vk_api/Pipeline.cpp`. Every cooperative-matrix pipeline of every device has this, the shipped ones and the
RX 7600's and M51's included. Results and correctness are not affected; a validation layer reports it.

Owner decision 2026-10-09: no campaign repairs it. Record it in `proposal.md` under "Known defect: F1" (one
paragraph; the B580's `STATUS.md` has the full description and two forms of a fix) and close as measured. It is
repaired once in the release zone before any promotion (`MERGE-PLAN.md`, M2e), and every device is gated and
timed again then: the company side will be asked for one gate and one timed session per device at that point.
If a reviewer on your side raises it, this paragraph is the answer.

A device where a missing full-subgroups guarantee shows up as wrong output, not as a validation message, is a
different matter: report it at once with the failing case. The fused kernels check `gl_NumSubgroups` and
`gl_SubgroupSize` at run time and return if the workgroup is not what they assume; a port must keep that check.

### 9.4 Decode: measure it at closing

The B70 confirmation measured decode 0 to 2.5 % slower with the fused node present, and the B580's record
points the same way. Nobody traced it: prefill was the goal. It is open as `MERGE-PLAN.md` M2f. Until it is
settled, every campaign whose final profile contains the fused kernel reports, at closing, decode tok/s with
and without the profile (the same runner, 5 valid runs per arm, the three models, one scheme is enough), in
`proposal.md`. Do not tune for it and do not investigate it inside the campaign; a number is what is asked.

### 9.5 Cost: what burned budget on our side, and the rule that stopped it

The 780M closing round was planned at one day and one budget unit and took three. One unit went almost
entirely to an actor and a reviewer re-checking and recomputing while an owner question was open. With a
metered model this is the first thing to prevent:

- A question for the owner is written ONCE, under "Decision needed from the owner", with the options, what each
  costs in device time, and a recommendation. Then the independent work is finished. Then nothing: no
  recomputation, no re-review, no new commit. The task file is checked for an answer at most every 20 minutes.
- Put this sentence in the task file from the start (the B70 task has it), and the same sentence in the
  reviewer's instructions.
- If the run reaches its budget while only the review's closing statement is missing, resume with a small cap,
  not the full one.
- If a run reaches its budget while waiting for the owner, do not resume it until the answer exists.

### 9.6 What the company side is asked to do now

- **M51: the 8B model** (`OWNER-DECISIONS.md` N10), with the setting the owner calls "thread hold = 32". Unchanged
  from 2026-10-08; listed here so that this section is complete.
- **RX 7600**: one dev-profile name for the final configuration (section 6 item 2); a decode number with and
  without it (9.4); nothing else. The kernel already has the barriers.
- **RX 7900 XTX**, when it starts: a confirmation in the shape of the B70's (9.1), with items 1 to 8 of 9.2 in
  the task file from the first day.
- **Every device, later**: one gate and one timed session under the F1 repair (9.3), when our side announces it.


## 10. Request to the company side, 2026-10-10: the llama.cpp comparison on your devices

Our five devices have a llama.cpp comparison (`topic/llamacpp-compare`,
`openspec/changes/sarc-1.5-llamacpp-compare/`: the kit, the rules, `results/cells.csv`). Yours are incomplete,
and a mock conference review of the tuning overview singled that out. What is missing:

| device | what exists | what is asked |
|---|---|---|
| RX 7900 XTX | nothing: no llama.cpp number at all, and its stock ExecuTorch baseline is from the session of 2026-09-28, not from the tuning session | the full comparison (10.1), then the HIP backend (10.2) |
| RX 7600 | llama.cpp Vulkan against the round 1 configuration (your push of 2026-10-08) | the comparison again with the round 2 final configuration as the tuned arm (10.1), then the HIP backend (10.2) |
| M51 | nothing | only if the device owner allows a statement of the form "ahead of / level with / behind llama.cpp" (10.3) |

### 10.1 The comparison, with the kit (both AMD cards)

Follow `kit/README.md` on `topic/llamacpp-compare` exactly; `kit/hosts/rx7600` is your own adapter from
2026-10-08 and the model for `kit/hosts/7900xtx` (one adapter file per device, nothing else). In one interleaved
session per device, all of these arms:

- `stock` (the kit's `build-stock.sh`: `release/1.5` plus the compile-only backport), 4w and 8da4w;
- `sarc` (for your devices: the campaign's parent commit, no profile), 4w and 8da4w;
- `tuned` (the campaign's final configuration: RX 7600 round 2 final; RX 7900 XTX final stack), 4w and 8da4w;
- llama.cpp at the pinned commit `b11430`, Vulkan backend, Q4_0 and Q4_K_M (`kit/make-q4km.sh`), each at its
  screened best setting and at the default setting, with BOTH timers (`lc` = `llama-completion` in a fresh
  process, `lb` = `llama-bench`).

Why all arms in one session: the reviewers recomputed the RX 7900 XTX's "3.15x over stock" and found that its
numerator and denominator come from sessions eleven days apart. One session with stock, parent and tuned
interleaved removes that objection, and gives the llama.cpp ratio under the same conditions.

Rules that the review made non-negotiable:

1. The driver is stated per arm. The RX 7900 XTX campaign ran on AMDVLK; run llama.cpp Vulkan on the SAME driver.
   If you also run under RADV, that is a second, separately labelled set, never mixed into the first.
2. Screening is recorded: for each backend the settings tried (flash attention on and off at least; the kit's
   list), the number for each, and which one is "best". "Best setting" without the list is not accepted.
3. Both timers are reported for every llama.cpp arm, not the higher of the two. If `llama-bench` takes a slow
   path above 1024 prompt tokens on a card (it did on the RX 7600 at 8B), report both numbers and say so.
4. Q4_0 and Q4_K_M are separate rows. Our report now says "ahead of llama.cpp Vulkan Q4_0" and gives the Q4_K_M
   ratio beside it; on one of our devices the Q4_K_M ratio is below 1.0.
5. Every run in `runs.csv` with its validity and reason (the kit's `row.py`); nothing is dropped silently.
6. `results/<device>/ARMS.md` (arms and commits) is committed BEFORE the timed session.
7. Leak check before every push, as section 5. Placeholders for host and user names.

Push to `topic/llamacpp-compare`: `results/7900xtx/` (new) and `results/rx7600/` (updated), and the aggregated
rows in `results/cells.csv`. Our side then adds the RX 7900 XTX to figures 5 and 7 of the report and replaces
its cross-session stock number.

### 10.2 The AMD vendor backend: llama.cpp HIP (ROCm), both AMD cards

On Intel and NVIDIA the vendor backend (SYCL, CUDA) is faster than llama.cpp's Vulkan backend and faster than
our tuned kernels by 3 to 25 %. For AMD discrete cards that comparison does not exist. It needs ROCm.

Constraints: user-space install only (`amdgpu-install --usecase=rocm --no-dkms`); no kernel-driver, Mesa or
AMDVLK change; no reboot; a Vulkan baseline cell measured before and after the install and required to agree
within 2 % (the host is a measurement environment). If the host's distribution release is not one ROCm supports
(the RX 7900 XTX host was on a non-LTS release), do not force it: use a ROCm container image with the host
untouched, or report and wait. The RX 7600 (`gfx1102`) is not on AMD's supported list: build for `gfx1100` and
run with `HSA_OVERRIDE_GFX_VERSION=11.0.0`, and say in the results that the override was used.

Build: llama.cpp `b11430`, `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100`, with
`HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)"`. Measure it as one more llama.cpp backend inside the
kit's session (arms `hip-q4_0` and `hip-q4km`, same screening, same two timers, same validity rules), so that its
rows land in `cells.csv` with `kind` = `hip` beside the Vulkan rows. Record the ROCm version in
`kit/VERSIONS.md`.

The step-by-step instruction for the agent (checks, stop conditions, what to save) was handed to the owner on
2026-10-09; ask him for it if it has not reached you. Its stop conditions hold: unsupported release, a reboot
request, the installer touching a kernel module, `rocminfo` not listing the card, llama.cpp falling back to the
CPU, or the Vulkan baseline moving.

### 10.3 M51

Only with the device owner's agreement, and then only relative: run llama.cpp's Vulkan backend (and OpenCL if
it runs on that driver) with the kit's protocol, keep every number local, and report one line per model:
"tuned ExecuTorch 4w is ahead of / within the noise band of / behind llama.cpp Vulkan Q4_0", plus whether
llama.cpp runs correctly at all on that driver. If the owner does not agree, say so in the proposal and nothing
else is needed. No figure, no driver or device identifier in anything pushed, as before.

### 10.4 Order and size

RX 7900 XTX 10.1 first (it removes the weakest number in the report), then RX 7600 10.1, then 10.2 on whichever
card's host can take ROCm without an OS change, then 10.3. 10.1 is about two hours of device time per card with
the kit; 10.2 is mostly the install. None of this is a tuning campaign: no kernel changes, no candidates.

Two things the same review asked of every campaign, which you can fold into the same sessions at no extra cost
(section 9.4 already asks for the first): decode tok/s with and without the final configuration, and the timed
prompt's real-text variant beside the synthetic one.
