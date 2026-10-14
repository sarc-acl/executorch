# sarc-1.5-campaign-playbook

## Why

Five per-device tuning campaigns (Radeon 780M, Arc B580, Arc Pro B70, RTX 4070 Ti SUPER, Jetson Orin Nano)
raised 2048-token Llama prefill speed by +32 to +67 % over the September SARC release. What they learned about
method, environment and review exists only in the task files and status files of those campaigns, on hosts
that the next machine cannot reach. The next devices (Radeon RX 7900 XTX, Radeon RX 7600) will be tuned by
another agent on another machine that reaches only the git remote.

## What

A hand-over package, documents and tools only. No kernel, no table row and no release-zone or upstream file
changes; nothing here was measured for this change.

- [PLAYBOOK.md](PLAYBOOK.md): the ordered procedure for an agent that runs a campaign on a device it has not
  seen. It refers to `sarc-1.5-e2e-benchmark/CONTRIBUTING-A-GPU.md` for how to benchmark and deliver results and
  covers what that guide does not: how to run a tuning campaign.
- [RULES.md](RULES.md), [LESSONS.md](LESSONS.md), [OWNER-DECISIONS.md](OWNER-DECISIONS.md): the shared rules,
  what happened and what it cost, and who decided what. Host names, paths and lock ids are removed.
- [COORDINATOR.md](COORDINATOR.md): what the coordinating session does, what it may do alone, its patrol
  checklist and its past mistakes.
- [results.md](results.md): final numbers of the five devices and of the llama.cpp comparison.
- [SUMMARY.md](SUMMARY.md): one page for the owner.
- `templates/TASK.md`, `templates/PROMPT.txt`: blank per-device task and launch prompt.
- `flow/gpu_campaign/__init__.py`: the hmz flow (actor and reviewer both on the GPU host), with both models as
  arguments, a review prompt that names the checks the reviewer redoes itself, and waits of 1, 5 and 15 minutes
  between failed model calls instead of ending after three.
- `tools/patrol.sh`, `tools/HOLD.md`: a read-only patrol driven by a table of campaigns, and the coordinator
  hold.

## Result

Geometric mean over Llama 3.2 1B, 3.2 3B and 3.1 8B, gain over the September release (details and sources in
[results.md](results.md)):

| device | 4w | 8da4w | both | state on 2026-10-06 |
|---|---:|---:|---:|---|
| Jetson Orin Nano | +65.7 % | +67.8 % | +66.8 % | reopened by review, re-gate running |
| Arc B580 | +46.0 % | +77.8 % | +61.1 % | finished, pushed |
| Arc Pro B70 | +45.3 % | +70.9 % | +57.6 % | result stable, search still running |
| RTX 4070 Ti SUPER | +43.2 % | +49.2 % | +46.2 % | finished, pushed |
| Radeon 780M | +30.0 % | +33.8 % | +31.9 % | stop rule met, closing |

The Orin, B70 and 780M rows are as of 2026-10-06 and are to be updated.

## Impact

- Adds one directory, `openspec/changes/sarc-1.5-campaign-playbook/`. No existing file is edited.
- The flow and the patrol script run on the control machine of a campaign, not in an ExecuTorch build.
  `flow/gpu_campaign/__init__.py` compiles and imports under hmz commit
  `474add43cf42db8881741a29cb81ef3f9a8c1a87`; the strengthened review prompt and the longer retry waits have
  not been run in a campaign. `tools/patrol.sh` was run against a local working copy only; its ssh path has not
  been run.
- The package depends on two branches that are not merged into `dev/1.5`: `topic/780m-prefill-refine` (the
  starting point of an AMD campaign) and `topic/llamacpp-compare` (the comparison kit and `kit/MODELS.md`).
- The tuned branches of the five devices stay unmerged; merging them is separate, later work.
