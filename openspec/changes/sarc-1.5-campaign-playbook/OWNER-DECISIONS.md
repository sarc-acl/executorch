# Owner decisions

What this file is for: the rulings of the repository owner that override or complete single rules of
`RULES.md`. Each entry says who decided, when, what, and why. Part 1 holds the decisions made during the five
finished campaigns; they stay in force. Part 2 holds the defaults the owner set for the next campaigns. **The
owner may change any of them per device**; a per-device change is written, dated, at the end of that device's
task file, and the task file wins over this file.

All decisions are the owner's. An agent (actor, reviewer or coordinator) never extends one by analogy: a
decision authorises what it says under its conditions and nothing more.

## Part 1: decisions of the finished campaigns (in force)

### D1. 2026-10-04: near-ties on the next-token items of the gate

Applies to the next-token comparisons of the gate only (`default vs tiled` in `verify.sh`, and parent vs
candidate). Everything else in the gate stands as written. `verify.sh` and the prompts are still not edited.

Why: next-token equality is a weak correctness measure. Where the model itself is undecided, any change in
summation order can swap the top two tokens.

A `DIFFER` on such an item no longer rejects a candidate by itself, provided all of the following are
measured, recorded under `results/<tag>/probe/`, and reported:

1. At the differing position: the logits of the top candidates for every arm (parent tiled, parent default,
   candidate tiled, candidate default), showing that the parent's own top-2 margin is small and how far the
   candidate moved it.
2. A broader comparison of candidate against parent that does not rest on one position: at least 32 positions
   or prompts of real text, for all three models and both schemes, reporting top-1 agreement, mean and maximum
   KL divergence of the next-token distribution, maximum absolute logit difference, and perplexity where it can
   be computed.
3. The noise floor: the same metrics between two arms that are both already accepted as correct on this device
   (parent tiled against parent default).
4. Acceptance: the candidate-vs-parent metrics are no worse than twice the noise floor of item 3. If they are
   worse, the candidate is rejected and the numbers are reported. State the thresholds you applied.

A candidate accepted this way is recorded as `ACCEPTED (near-tie, owner decision 2026-10-04)` with the path of
the evidence, never as a plain pass, and the differing items are listed. Do not search for a kernel variant
that happens to keep the token: that is fitting to the benchmark prompt.

### D2. 2026-10-04: how a parameter space too large to enumerate is searched

No full enumeration beyond a day. Validate a cheap screening mode against the full measurement (rank
correlation on at least 50 configurations), screen a uniform random sample of 2000 to 3000 configurations with
a recorded seed, derive parameter importance and pair interactions from it, refine locally around the best 20,
confirm the best 10 per shape with the full measurement. The parameter-importance table is a required
deliverable.

Status for the next campaigns: this method applies only if a search is ordered. By N1 it is not the default.

### D3. 2026-10-04: arithmetic changes are judged against a reference, not against the parent

Replaces items 3 and 4 of D1 for any candidate that changes kernel arithmetic (a different attention kernel, a
different accumulation precision or order).

Why: "distance from the parent" measured against the spread of the parent's two linear arms is not a
meaningful floor for an attention change. Those two arms share the same attention kernels, and the stock
attention kernels accumulate in fp16, so part of the distance is the parent's own rounding error.

A next-token `DIFFER` does not reject such a candidate if all of the following hold and are recorded under
`results/<tag>/probe/` and `results/<tag>/sdpa-error/` (or the equivalent for a linear kernel):

1. Gate criterion, error against the reference. On the production-shape cases (S = 2048, every head
   configuration; for a linear kernel the production-diff shapes), the candidate's rms and maximum error against
   the fp32 CPU reference are not larger than the parent's on the same inputs. All correctness tiers still pass
   with 0 mismatches. Report both arms' numbers side by side.
2. Evidence that is always reported, in full: items 1 and 2 of D1 (logits at the differing position for the
   four arms; the real-text comparison on at least 32 prompts, all models and schemes, with top-1 differences,
   mean and maximum KL, maximum logit difference, perplexity ratio, and the parent-tiled vs parent-default
   numbers beside them for scale).
3. Gross-divergence check on that real-text comparison. This is a bug detector, not a precision bound: reject
   if, in any cell, the mean KL of candidate-default against parent-default exceeds 0.5 nat or the top-1 token
   differs on more than one third of the prompts.
4. Record the result as `ACCEPTED (reference-error rule, owner decision 2026-10-04)` with the evidence paths
   and the list of next-token items that differ. Never as a plain pass.

A candidate that claims not to change the arithmetic (a staging or layout change meant to be bit-identical)
does not use this rule: it must show bit-identical output, or stay within the parent-tiled vs parent-default
spread as in D1.

A per-device reading of this rule was asked for once and not granted: a candidate that fails criterion 1 stays
rejected.

Still forbidden: editing `verify.sh`, the prompts or tolerances; choosing thresholds after seeing a candidate's
numbers; searching for a variant that happens to keep a token.

### D4. 2026-10-05: the release-zone hooks an accepted candidate needs may be committed

The owner answered "allow all" to the question which hooks may be committed on the topic branches.

Why: a recommended configuration measured through an uncommitted local patch cannot be reproduced from the
pushed branch (`LESSONS.md` L12).

This is permission to commit a small, inert entry point. It is not a promotion, it does not relax the gate,
and it allows nothing else in the release zone (no shipped shader, no golden, no tolerance, no tool).

Allowed, each as its own commit, with the heading "Release-zone hook (owner decision 2026-10-05)" and the diff
in the campaign's `proposal.md`:

1. **Softmax variant name.** One form for every branch, so that the branches merge without a conflict: a field
   `Override::softmax_variant` (null by default) in `impl/sarc/Select.h`, read in `sdpa_softmax_shader_name` in
   `impl/sarc/SdpaCoopmat.cpp`, which appends `_<variant>` to the SARC softmax name; 8 lines. Set the field
   from the dev-zone override. If the starting branch already carries it (`git grep softmax_variant`), use it
   unchanged; drop any environment-variable form and any other local softmax patch.
2. **Attention rows for a device that has none in the release table.** First choice: register the rows from
   the dev zone (`impl/sarc_dev/<Device>Sdpa.cpp`); that needs no release-zone edit. Only if that cannot select
   the same kernels, commit the `kUnverified` rows in `impl/sarc/table_<vendor>.cpp`, and say why.
3. **Fused attention node.** The entry point may be committed, kept as small as it can be made, with the logic
   in the dev zone. It is larger than a switch, so it stays subject to the owner's review before any promotion:
   mark it so in `proposal.md`.

Conditions, all of them:

- With nothing selected (no dev profile, no variant), every device dispatches exactly what it dispatched before
  the hook. Show it: `test_sarc_select` unchanged, `spirv_golden.py` unchanged, and the parent snapshot
  (`verify.sh`) identical line by line.
- After committing, the result must be reproducible from the committed branch alone, with no local patch:
  build the branch head, confirm that the accepted candidate dispatches the same kernels as the build that was
  measured, run its gate once, and run one timed session against the parent. Report the number from the
  committed build beside the number measured through the patch.
- `sarc/tools/check.sh` may report the release-zone edits. Do not edit `check.sh`; report its output and name
  this decision.
- Reviewer: this decision authorises those entry points under those conditions and nothing more.

### D5. 2026-10-06: the model file may be read into the page cache before each cell

Why: the runner aborts after its output only in processes whose model load was slow (5 of 60 slow loads, 0 of
1261 normal ones, in both arms), which makes a gate fail at random on a host whose RAM cannot hold the models
(`LESSONS.md` L9). The disturbance has nothing to do with what is measured.

- The measuring tools may read the model file of a cell into the page cache before the first process of that
  cell, and again whenever the model changes (`cat <model> > /dev/null`), in the timed session, the next-token
  runs, the `verify.sh` staging and the trace step alike, for the parent arm and the candidate arm equally.
  Record per run whether the load was slow.
- Unchanged: the gate's criteria, `verify.sh`, `check.sh`, the goldens, the thresholds, the number of runs. A
  run that aborts is still an invalid run. No sudo, no change of kernel settings (no `drop_caches`).
- If a gate is rejected again for a runner abort in a process that was not a slow load, stop and report: that
  would be a different problem. Keep the account of the abort as a finding about the runner; do not fix the
  runner.

### D6. 2026-10-06: a campaign yields its GPU at a job boundary when the coordinator asks

Why: the owner needed each device for one to two hours for a separate measurement (the llama.cpp comparison)
while the campaigns were still running.

Every detached queue checks for the file `HOLD` in its artifact directory before each job, as described in
`tools/HOLD.md`. The coordinator creates and removes `HOLD`; the campaign never does, and it does not teach its
guard to accept the coordinator's processes.

### D7. 2026-10-05: a second unit of the same device may be used, for screening only

Decided twice (the second Arc Pro B70 card of one host; a second Jetson Orin Nano).

- A second unit is used only for work whose result is a ranking that the primary unit then confirms: cheap
  kernel and tile screens, exploratory phase timing. Every timed end-to-end session, every gate, every
  reference-error measurement and every reported number stays on the primary, alone.
- Before trusting it, run one small identical batch (20 to 30 configurations) on both, alone and, for two cards
  in one host, also together, and report rank correlation and time ratios against a threshold fixed before
  looking.
- When one configuration of 26 disagreed by 20 % (`LESSONS.md` L18), the owner ruled: retest it with at least
  three rounds per unit, keep the first verdict on record, do not edit the threshold after the fact, treat the
  configuration as device-sensitive, and use the second unit for screening anyway, resting on the rank
  agreement (0.997). If a confirmation on the primary ever contradicts a screen from the second unit, report it
  and stop using that unit.
- One GPU job at a time per unit, each under its own lock; every result row records its unit.

### D8. 2026-10-06: push a topic branch early when another campaign will fork from it

Why: the next devices of a vendor start from the finished sibling's branch (N5) on a machine that reaches only
the git remote.

The campaign pushes its topic branch at a consistent committed state without interrupting a running
measurement, and again when it closes. An ordinary push: no force, no rebase, no pull request.

### D9. 2026-10-05: no driver-level profiler tracing (superseded by N4)

Why at the time: an RGP capture hung the Radeon 780M host minutes after a capture request reached the campaign
(`LESSONS.md` L29). The owner withdrew the request and forbade driver-level tracing on every device of that
campaign. Kept here as the record; N4 reverses it for the new machines.

## Part 2: defaults for the next campaigns (recorded with this hand-over, 2026-10-06)

### N1. Devices that already have attention kernel rows get a port, not a search

The Radeon RX 7900 XTX and the Radeon RX 7600 already have (unverified) attention kernel rows. They are in the
780M's position: expect +20 to +30 %, not +50 %.

Default work: port the 780M's second-layer results: the fused attention kernel, the fp32 softmax, the linear
kernel chosen per layer shape, the whole-texel 8da4w weight staging. **No sampled parameter search by default.**

Why: on the 780M the second layer gave +31.9 %; the sampled search cost about two days on the B70 and found
nothing (`LESSONS.md` L5, L6).

### N2. No verification pass of the existing unverified rows first

Go straight to the port and verify everything together at the end.

Recommended minimal safeguard (a recommendation of this package, not a verification and not an owner
requirement): store the unmodified `verify.sh` output of the starting build once, as the snapshot
`s0-parent-verify`, so that a failure at the end can be told apart from one that was there before.

### N3. The decisions of Part 1 carry over as defaults

In particular: attention kernels that accumulate in fp32 are judged by error against the fp32 reference, not
by next-token equality with the parent (D3); the few release-zone hooks an accepted candidate needs may be
committed (D4); the model file may be read into the page cache before each cell (D5); stop when two consecutive
gated candidates gain under 2 % (`RULES.md` R11).

### N4. Profiler tracing is allowed on the new machines

This reverses D9. The factual record stays: on the Radeon 780M (RADV, Mesa 25.2.7) two RGP captures of the
three-kernel attention path worked and the third, of the fused attention kernel, hung the whole host with no
kernel message and needed a hand reboot.

Advice that goes with the permission: capture from a detached job after saving work, one capture at a time,
never during a timed session (`RULES.md` R9).

### N5. Start from the branch of the same vendor's finished device

Not from `dev/1.5`: nothing is merged yet. AMD devices start from `topic/780m-prefill-refine`. Merging the five
branches into `dev/1.5` is separate, later work.

### N6. Claude reviews Claude, with a fixed list of checks; hmz is not pinned

On the other machine Claude Code and hmz are available; Codex is not. The reviewer is therefore Claude with a
fresh context each round. Actor and reviewer models are both passed to the flow with `-a`.

Because actor and reviewer are then the same model family, the review prompt is strengthened: the reviewer
itself recomputes every reported number from the raw run files, runs the golden check, lists the files changed
outside the dev zone, reads every new shader for unsynchronised shared writes, and states what it did not
check (`flow/gpu_campaign/__init__.py`, `RULES.md` R12).

hmz is not pinned, by the owner's choice. On record: this flow was last run with hmz commit
`474add43cf42db8881741a29cb81ef3f9a8c1a87`; hmz changed its API on 2026-10-05 (`AgentView.spawn()` no longer
takes `env`; pass `env=` to `run()`); a running hmz window must be quit and reopened before `/resume` after the
flow file changes.

### N7. The other machine reaches only the git remote

Not the GPUs of the first five campaigns, not their storage. Model files there are the same models but not
byte-identical. Check each file's context length (2560, read with the ExecuTorch runtime's constant method
`get_max_context_len`) and its size against the table in `../sarc-1.5-llamacpp-compare/kit/MODELS.md`, and treat
a baseline more than 3 % from the published table as something to explain before continuing.

### N8. Several devices in parallel

One hmz workspace per device, plus one coordinating Claude Code session (`COORDINATOR.md`).

### N9. After the tuning, each device also gets the llama.cpp comparison

With the kit on branch `topic/llamacpp-compare`: add `kit/hosts/<device>/host.sh`, screen llama.cpp's settings on
the device, include the vendor's own llama.cpp backend where one exists. Results of a public device go back to
the public remote.
