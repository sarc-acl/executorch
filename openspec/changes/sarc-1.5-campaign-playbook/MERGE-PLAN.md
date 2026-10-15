# Merge plan

What it takes to bring the campaign branches into `dev/1.5`. Written as a draft on 2026-10-06 for five
branches; all campaigns have since closed.
Items marked **[owner]** need a decision before they can be done.

## Status 2026-10-13

The dev-zone merge (M3) is done on the integration branch `topic/merge-campaigns`, which starts at `dev/1.5`
(`44e22be60f`). `dev/1.5` itself is fast-forwarded to it on the owner's word. Owner decisions of 2026-10-13
(`OWNER-DECISIONS.md` N11): merge only, nothing promoted; integration branch; the full-subgroups patch (M2e)
goes in with the merge; `topic/sarc-pinned-glslc` is left out for now.

| task | state |
|---|---|
| M1 near-tie stage of the gate | done 2026-10-15 (N12.1, amended): `sarc/tools/near_tie.py` called from `verify.sh`; margin < 0.5 logit, both tokens in the fp32 reference top 5, rms and max error not larger than the old build. Replay 19/19: the real-text flip (B580/B70) accepted, the `prompt_2048` flips and the Orin 3B case rejected. Not yet run on a GPU; no session tool yet calls it for old-vs-new builds |
| M2 review of the release-zone hooks | reviewed 2026-10-15; owner approved P1-P6 and P8 (not P7, nothing goes upstream), done on `topic/pre-promotion`, not pushed: the fused attention node is installed through one order-independent slot (`impl/sarc_dev/FusedSlot.h`, two requested nodes abort, tested in `test_sarc_select`); the 780M softmax variant no longer erases another; the softmax-grid skip only in LLM mode; `ET_VK_EXECUTE_NODE_THRESHOLD` validated and only for the default config; F1 passes only what the device enabled; `sarc/HOOKS` text. Device re-measurement of the behaviour changes is the owner's call (the change's report lists them) |
| M2a subgroup barriers | done on the campaign branches (`fused3sb`; RX 7600 +0.00 %, 780M -0.14 %, in band) and merged. The unbarriered `fused2` and `fused3` are retired on `topic/pre-promotion` (owner decision 2026-10-15): kept as files, selecting them (780M c8 to c10, `ET_VK_SARC_780M_SDPA_FUSED`) aborts at start-up |
| M2b pinned-container rebuild | not started; now tied to the compiler switch of `topic/sarc-pinned-glslc`. Native builds: RX 7600, RX 7900 XTX, M51; the Orin used the cross image's glslc |
| M2c home-directory paths | done 2026-10-15 (N12.3): `<home>/` placeholder in 330 result, evidence and prose files; tool scripts and two hash-pinned Orin probe files keep theirs |
| M2d M51 | open; the final profile is now `c8`, 8B was run under N10 with the node-threshold hook |
| M2e full-subgroups flag | merged (one commit after the branch merges). No effect on the five devices on our side (2026-10-10 / 11); the three company-side devices measure it on the merged branch |
| M2f decode with the fused node | measured 2026-10-15 (rule fixed before timing: each of six cells within ±2 %, interleaved, 7 runs per arm): all 24 cells in band on the B70 (-1.15 % six-cell geomean), 4070 Ti (-0.84 %), Orin (-0.26 %), 780M (-0.11 %). The GPU dispatches of a decode step are identical with and without the node (its two extra nodes per layer get an empty grid); the host resize pass is longer by about 3 to 18 µs per layer per step (likely the per-step pick functions and `sdpa_fused_skip` -> `variant_for`), which is the measured slowdown. Not on the B580 (host busy) nor company-side devices. Raw data outside the repo (`.artifacts/merge-campaigns/m2f/`) |
| M3 dev-zone merge | done and checked on the five devices on our side, see below |
| M4 promotion | not started |
| M5 independent review | done 2026-10-15: all eleven campaign headline numbers recompute from the committed run tables; two defects fixed on `topic/pre-promotion` (780M fused2/fused3 race: retired; microbench `[sdpa-error]` formats vs the B580 gate tool); stale playbook/README figures corrected. Open: B70 round-3 +9.45 %, Intel round-3 order results, M51 and Orin clock samples have no raw data in the repo; the QK NO_MASK_FILL / shortened softmax tail vs attn x V read range is not proven for every variant |
| M6 stale documents | `sarc/README.md` and the `sarc-1.5-7900xtx-4w` proposal updated with the merge |

Also open before promotion: the RX 7900 XTX workgroup memory above the AMDVLK limit (its `STATUS.md`, "Decision
needed from the owner").

## What is to be merged

| branch | device | state |
|---|---|---|
| `topic/b580-prefill-refine` | Arc B580 | finished, pushed |
| `topic/4070ti-prefill-refine` | RTX 4070 Ti SUPER | finished, pushed |
| `topic/orin-prefill-refine` | Jetson Orin Nano | finished, pushed |
| `topic/780m-prefill-refine` | Radeon 780M | finished, pushed |
| `topic/xe2-prefill-refine` | Arc Pro B70 | finished, pushed |

All five forked, directly or through a sibling, from one commit of `topic/780m-prefill-refine`, and all five
add to the same dev-zone files (`impl/sarc_dev/Overrides.cpp`, sweep yaml), each inside a block marked with its
device tag. The comparison (`topic/llamacpp-compare`) and this package (`topic/campaign-playbook`) add
documents and tools only.

`dev/1.5` has moved by one commit since the campaigns forked (`44e22be60f`, an `8da4w` decode fix in
`impl/sarc/Dq8caCoopmat.cpp`). It touches the decode path only and no file a campaign changed; it arrives with
the merges and needs no rework, but decode is re-checked after M3 because of it.

A further campaign has since forked from the 780M branch (`topic/rx7600-prefill-refine`, Radeon RX 7600). It
uses the 780M's fused attention kernel, so M2a below concerns it as well, and it merges after the 780M.

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

### M2a. Add the missing subgroup barriers to the fused attention kernel

Found by the RX 7600 campaign when it read the kernel it was porting: in
`glsl/sarc_dev/sarc_dev_780m_sdpa_fused3.glsl` a lane reads shared-memory slots that other lanes of its subgroup
wrote, ordered only by `memoryBarrierShared()`, with no `subgroupBarrier()`. No two invocations write the same
location, the workgroup is one subgroup, and every correctness pass on both AMD cards agrees; but under the
Vulkan memory model the read is formally unsynchronised and the kernel relies on one wave running in lockstep.
It is the same class of defect as the Orin's (`LESSONS.md` L8), found before it failed.

Do: add `subgroupBarrier()` after each such `memoryBarrierShared()`, then gate and time the kernel again on the
780M (and on the RX 7600 once its campaign has adopted it). A barrier can cost time; the result is reported
either way. Before this kernel is ported to any device whose subgroup is smaller than the workgroup, the fix is
mandatory.

### M2b. Rebuild in the pinned container where a campaign built natively

The RX 7600 host cannot run the container, so every build of that campaign used a native shader compiler and
its shipped-SPIR-V check is pending (14 of 53 release shaders differ from the golden for that reason alone).
Before that branch merges, build it once in the pinned container on a host that can, and run
`sarc/tools/check.sh` there. The five campaigns of this package built in the container.

### M2c. Remove home-directory paths from committed result files **[owner]**

Raw logs and result tables committed under `openspec/changes/*/results/` contain absolute paths with a local
user name: about 780 lines already on `dev/1.5` (the September benchmark results) and 120 to 780 more on each
campaign branch. No credential is involved. Do, before M3: replace the prefix with a placeholder in the
campaign branches' result files (one mechanical commit per branch). Whether to do the same for the lines
already on `dev/1.5` is the owner's call; removing them from history would need a rewrite of a branch that is
never force-pushed, so the practical choice is a forward commit or leaving them.

### M2d. M51: unaligned prompt lengths and the 8B gap **[owner]**

`topic/m51-prefill-refine` closed on 2026-10-08 with two things the promotion must weigh: (1) its final stack's
8da4w kernel (`zpg_bt`, a 4h4w-layout pick) falls to the stock tiled kernel for prompt lengths that are not a
multiple of 128, so those cells are slower than the parent there (recorded as a property, not fixed); a profile
promoted for M51 either excludes that pick for unaligned lengths or ships with the caveat. (2) Every M51 gate is
PARTIAL (1B and 3B) until the 8B work of `OWNER-DECISIONS.md` N10 is done; until then the M51 profile is
labelled "verified 1B and 3B only". Its numbers are local to the device owner; the review on our side is of code
and method.

### M2e. F1: create cooperative-matrix pipelines with the full-subgroups flag **[owner]**

Found by the B580 fused-port campaign (2026-10-09), confirmed there by its reviewer. The specification requires
(VUID-RuntimeSpirv-OpTypeCooperativeMatrixKHR-10770) that a pipeline whose shader uses cooperative matrices is
created with `VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT` or that the module is SPIR-V 1.6 or
later. The runtime does neither: the shaders are SPIR-V 1.3, the instance is created for Vulkan 1.1, and
`vk_api/Pipeline.cpp` creates every compute stage with `flags` 0. Every cooperative-matrix pipeline of every
device has this, the shipped ones included. Results and correctness are not affected (the fused kernels check
`gl_NumSubgroups` and `gl_SubgroupSize` at run time), but the pipelines are not valid as created.

Owner decision 2026-10-09 (option C): no campaign repairs it locally; each fused-port campaign records it as a
known defect and closes as measured. It is repaired here, once, in the release zone: set the flag wherever a
required subgroup size is in force and the device reports `computeFullSubgroups` (about 10 lines in
`Pipeline.cpp` plus the feature bit in the descriptor; with the flag the local size in X must be a multiple of
the required subgroup size, VUID-VkPipelineShaderStageCreateInfo-pNext-02757, to be checked for every shipped
variant). Then every device: `spirv_golden.py`, the full gate, one timed session of the final profile against
its own closing session. The B580 campaign's `STATUS.md` ("Decision needed from the owner", forms A and B) is
the worked description.

Status 2026-10-10: the change exists as one release-zone commit (about 50 lines in `vk_api/Adapter.cpp`,
`Pipeline.cpp`, `Pipeline.h`, `sarc/HOOKS`) and was applied unchanged and re-measured, old build against new in
one session with the verdict rule written first, on the five devices on our side. No effect on the Arc B580
(-0.04 %), Arc Pro B70 (+0.11 %), Radeon 780M (+0.02 %) and RTX 4070 Ti SUPER (-0.13 %); logits byte-identical,
SPIR-V unchanged, the validation message gone where the layer is installed; no variant on any device violates
the local-size rule. Jetson Orin Nano: both timed sessions inside the band (-0.05 %, +0.00 %), its kernel-level
step unfinished at the time of writing. The branches are local and not pushed (the owner decides). Still to do
here: the three company-side devices, and the other validation messages (a memory-model feature that is not
enabled, among others), which were recorded and not repaired.

### M2f. Decode with the fused attention node present **[owner]**

The B70 confirmation measured decode 0 to 2.5 % slower with the fused node present, and the B580 record shows
the same direction. Prefill was the goal of the port campaigns, so nobody traced it. Before promotion: one
decode trace per device family with and without the fused profile, and, if the loss is real, the fused node
must leave the decode path untouched. Owner decision 2026-10-09: not investigated now; it is done here, before
promotion, with M2e.

### M3. Merge the dev zone (done 2026-10-13, `topic/merge-campaigns`)

Ten heads carry every campaign: each fused-port branch contains its first-round branch, the RX 7600, RX 7900 XTX
and M51 branches fork from inside the 780M branch, and `topic/m51-dq8ca-decode-fix` already is `dev/1.5`. Each
was merged as a two-parent commit `[SARC] Merge topic/<name>` whose message records the conflicts and their
resolution. Order:

| # | branch | head | conflicts |
|---|---|---|---|
| 1 | `topic/campaign-playbook` | `78614e0add` | none |
| 2 | `topic/llamacpp-compare` | `9721168473` | none |
| 3 | `topic/780m-prefill-refine` | `71b43d903d` | none |
| 4 | `topic/rx7600-prefill-refine` | `c4d7451c5d` | none |
| 5 | `topic/m51-prefill-refine` | `0b42e141b0` | `Overrides.cpp` (device sections) |
| 6 | `topic/7900xtx-prefill-refine` | `82aaec9983` | `Overrides.cpp`, `fused3sb` yaml, 4w profile yaml |
| 7 | `topic/4070ti-fused-port` | `8f8c44e22e` | `Overrides.cpp`, `test_llama_microbench.cpp` |
| 8 | `topic/orin-fused-port` | `da92b4b3ce` | same two |
| 9 | `topic/b70-fused-port` | `4275735c50` | same two, five yamls |
| 10 | `topic/b580-fused-port` | `af93a30c55` | same two, `gen_xe2.py` and its generated files |

Then the full-subgroups commit (M2e) and this document update.

What was not a mechanical union:

- `test_llama_microbench.cpp`: four implementations of the fused-kernel support met (780M family unmarked,
  4070ti-fused and orin-fused as marked blocks, B580 as its own port). One of each piece is kept. The 780M
  family's kernels take that family's verdict path, every other fused kernel the 4070ti-fused block. One merge
  was textually clean and wrong: Q was scaled by `q_scale` twice (found by reading, fixed in merge 7).
- Registrar of `Overrides.cpp`: the 4070ti and B580 blocks both assigned the softmax variant unconditionally and
  the Orin block overwrote the fused entry points; the later blocks now assign only when their profile asks.
- `sdpa_fused_variants_b580()` / `register_sdpa_fused_b580()` existed on both Intel branches (the B70 uses the
  B580's node); one definition serves both.
- `gen_xe2.py`: the B70 version plus the B580's per-shape order profiles. Both generators are no-ops on the
  merged tree.
- The M51 branch seeds every SDPA correctness case from its name instead of `srand(0)` at start-up. After the
  merge this holds for every device, so the test inputs differ from those of the other campaigns' recorded runs.

Checked after every merge: `sarc/tools/check.sh --no-build` (release selection 1240 checks, 31 rows, 0
candidates, as on `dev/1.5`). Container builds at merges 9 and 10.

**Defect found after the merge and fixed (`744429583d`).** The 780M family's registrar installed its fused node
into the override unconditionally, which on the merged tree replaced another device's node depending on the order
of the static initializers. On the RTX 4070 Ti SUPER `4070ti-fused1` then ran the split SDPA path, 5.6 to 16 %
slower. The 780M node is now installed only when a 780m profile or `ET_VK_SARC_780M_SDPA_FUSED` asks for it.

**Device check of the merged head (`744429583d`), 2026-10-14.** On each device on our side, the merged build
against the campaign's own final build (the F1 build where one existed), the device's final profile in both
arms. Rule fixed before timing: every cell within +-2 %, geometric mean within +-1 %, every next token the same.
Kernel identity first (dispatched kernel per linear case, SDPA kernels, SDPA correctness verdicts), then the
campaign's own session tool (`e2e5.sh`, its lock and clock floor, 5 reps, six cells).

| device | old arm | kernels and verdicts | geometric mean | cells | next token |
|---|---|---|---|---|---|
| Arc B580 | round-3 gate build (no F1) | identical (48 linear cases, SDPA) | +0.04 % | -0.46 to +0.71 % | same (simple A/B, see note) |
| Arc Pro B70 | round-3 gate build `e8c0cb1ccf` (no F1) | identical | +0.04 % | -0.24 to +0.49 % | 18/18 same |
| Radeon 780M | F1 build `433b072c13` | identical (92 linear, 36 SDPA) | -0.07 % | -0.34 to +0.09 % | 18/18 same |
| RTX 4070 Ti SUPER | F1 build `4a3c33d727` | identical | +0.18 % | -2.00 to +2.44 % | 24/24 same |
| Jetson Orin Nano | F1 build `0f038a736` (cross) | identical; SPIR-V of all 1620 old files byte-identical | +0.02 % | 0.00 to +0.07 % | 24/24 same |

Notes: the B580 was measured with a plain interleaved llama_main A/B (5 runs per arm and cell, no clock or
foreign-load validity checks) on the build before the fix, which does not change the B580's path; a repeat on the
fixed build was discarded because the workstation was under heavy unrelated load. On the RTX 4070 Ti SUPER the
3B 8da4w cell reads +2.44 %, outside the band on the fast side; prefill times there are whole milliseconds and the
new arm's median equals the old build's median in two earlier sessions, so the old arm read low in this one.
The company-side devices (RX 7600, RX 7900 XTX, M51) have not run the check.

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
