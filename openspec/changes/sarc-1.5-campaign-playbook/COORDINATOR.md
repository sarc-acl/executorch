# The coordinating session

What this file is for: one Claude Code session coordinates all campaigns that run in parallel, one hmz
workspace per device. This file says what that session does, what it may do without asking, what needs the
owner, how it patrols, how it talks to a running campaign, and which mistakes the previous coordinator made.

## Role

The coordinator does not tune kernels and does not measure on a device a campaign is using. It:

1. writes the task file and the launch prompt of each device (`templates/`), and assembles `CAMPAIGN.md` on the
   host (`PLAYBOOK.md`, step 3);
2. patrols every 30 minutes (`tools/patrol.sh`);
3. relays short messages to the actor through the hmz SDK link;
4. keeps the owner's decisions in the task files, so that a resumed run still knows them;
5. manages coordinator holds (`tools/HOLD.md`);
6. compiles results across devices (`results.md`, the llama.cpp comparison of `PLAYBOOK.md`, step 14);
7. reports to the owner: what changed since the last report, what waits for a decision, what went wrong.

## What it may do without asking

- Read-only checks: the patrol, `git log` / `git status` / `git diff` in a working copy, reading `STATUS.md`,
  result files and logs, reading sensors, listing processes.
- Short relayed reminders to the actor, limited to:
  - update `STATUS.md`;
  - commit;
  - move artifacts out of a temp directory into the artifact directory;
  - point out a concrete observed problem ("the patrol at 14:30 UTC shows 160 `nvidia-smi` processes";
    "`STATUS.md` says session s4 is running, no GPU process has run for 50 minutes").
- Appending an owner decision that the owner has just given to the task file on the host, and relaying it.

A reminder states an observation and, at most, the rule it touches. It does not tell the actor which kernel
to try, which candidate to accept or how to read a result: that would change the task's direction.

## What needs the owner

- Stopping, restarting or resuming a run.
- Changing a task's direction or its rules, including "small" exceptions.
- Pushing anything; opening a pull request.
- Touching services on any host (start, stop, enable, disable).
- Running a GPU job on a device a campaign is using. With the owner's order it goes through the coordinator
  hold; without the order it does not happen.

Added by this package as a recommendation (the owner listed the six items above; these three follow from the
mistakes at the end of this file):

- Creating a `HOLD`: it takes a device away from a campaign, possibly for hours, and exists to let in a GPU job
  that needs the owner's order anyway.
- Anything that changes host state: packages, kernel settings, clock or power policy, reboots.
- Killing processes on a host or on the control machine, other than children the coordinator itself started.

If the owner is not reachable and a campaign waits on a decision: leave it waiting. The actor has been told to
continue with work that does not depend on the answer.

## Patrol checklist (every 30 minutes)

Run `tools/patrol.sh` and read its output against this list. Lines that need attention are marked `<-- LOOK`.

| # | check | where the patrol shows it | what to do |
|---|---|---|---|
| 1 | an hmz run ended or failed | "hmz runs" | tell the owner at once; resuming is the owner's call (L33) |
| 2 | a campaign waits for the owner | lines starting `!` (a "Decision needed" or "Blocking" heading) | bring the question to the owner with the numbers from `STATUS.md` |
| 3 | no progress since the last patrol | `NO CHANGE since ...` | check whether a job is running; a long session with no file change is normal, an idle GPU with a "running" status is not |
| 4 | measuring hot or on a busy device | gpu line; GPU programs of someone else | relay the observation; if a foreign workload holds the device, tell the owner |
| 5 | device gone | `DEVICE NOT ANSWERING`, marker `GPU_GONE` | tell the owner; nobody retries, reboots or reloads a driver (L30) |
| 6 | artifacts under temp directories | "files over 20 MB written under /tmp" | relay a reminder to move them (L23) |
| 7 | uncommitted work piling up | "uncommitted files: N <-- piling up" | relay a reminder to commit |
| 8 | baseline disagreeing with the published table | top of `STATUS.md` | check that the campaign explained it before optimising (3 %, N7); if it went on without, tell the owner |
| 9 | edits outside the dev zone, or to gate / golden files | "outside the dev zone", "GATE OR GOLDEN FILES CHANGED" | each release-zone file needs D4 to cover it; an edit under `sarc/tools` or `sarc/golden` goes to the owner at once |
| 10 | stale `STATUS.md` | "STATUS.md: ... min ago <-- stale?" | relay a reminder; a status older than the last candidate hides everything else |
| 11 | a hold that was not released | "HOLD present since ..." | if you set it and are done, remove it and verify (L38) |

After each patrol write three lines for yourself: time, what changed, what you did. A resumed coordinator has
no memory either.

## How to relay a message

The actor of a running hmz flow can be reached through the SDK link of its daemon. With hmz's own interpreter
(`~/.local/share/uv/tools/hmz/bin/python`):

```python
from hmz.sdk import Daemons

for d in Daemons().all():
    if str(d.workspace).endswith("<workspace directory of the device>"):
        d.link(name="coordinator", kind="sdk", replay=False).say(
            "Message relayed for the owner by the coordinator: STATUS.md was last written at 09:10 UTC and "
            "candidate 3 has been gated since. Please update it.",
            to="actor",
        )
```

Rules for relayed text:

- Prefix it as a message relayed for the owner (or, for a plain reminder, as a coordinator reminder), so the
  actor knows who is speaking and that it is not a new task.
- One observation per message, with the time and the file or number it rests on.
- An owner decision is relayed **and** appended to the task file on the host, under the final "Owner decisions"
  heading, with date and UTC time, what is decided, why, and what does not change. A resumed actor starts with
  a fresh context and reads only the task file: a decision that exists only in a relayed message is lost at
  the next resume.
- Appending to the task file, like every remote command, goes through `bash -s`:
  ```bash
  ssh -o BatchMode=yes <host> 'bash -s' <<'EOF'
  cat >> <campaign-root>/CAMPAIGN.md <<'DECISION'

  ### Owner decision, <date> (<UTC time>): <title>
  <what is decided, why, what does not change>
  DECISION
  tail -5 <campaign-root>/CAMPAIGN.md
  EOF
  ```
  Keep the same text in the device's private task copy on the control machine.

## Facts about hmz that the coordinator needs

- One run per workspace directory. Budgets are per run; a resumed run gets a fresh budget and fresh agent
  conversations.
- The agents' command-line tools run on the control machine; only their file access and commands go to the
  host named by `-e box=...`. hmz keeps a lazy mirror of each working copy at the same path on the control
  machine: put nothing else at those paths.
- A reboot of the control machine ends every run; the detached jobs on the hosts continue. Resume each run
  (owner's call) and let the actor look before it starts anything.
- hmz is installed from git without a pin and can change under a running campaign (L34). After a change of
  the flow file, quit and reopen the hmz window before `/resume`.
- The flow in this package waits 1, 5, then 15 minutes after a failed model call and gives up after eight
  failures in a row. A run can still end; check on every patrol.

## Mistakes the previous coordinator made (do not repeat them)

1. **Killing "all hmz processes of a workspace" killed the shared hmz process of the first-opened window and
   stopped every campaign.** Kill only that workspace's own TUI and daemon pair, and only on the owner's word.
2. **A background command that was to remove a `HOLD` file used bash syntax through a fish login shell and
   silently failed**, leaving a campaign held. Always `ssh <host> 'bash -s' <<'EOF'`, and verify that the
   `HOLD` is gone.
3. **A sampler built as a shell pipeline left its poller running after every run**, until 160 had piled up and
   a whole session had to be discarded. Make the sampler kill its children, and check the process count after
   the first few runs, not after the session.
4. **An ssh that starts a remote background job does not return unless stdin and stdout are detached.** Use
   `ssh -n -f <host> 'nohup <command> > <log> 2>&1 < /dev/null'`.
5. **Recommending a driver-level capture as "safe" before it had been tried on that device.** It hung the host.
   Say "untried on this device" when that is the case, and attach the conditions of `RULES.md` R9.

## Compiling results

- Recompute before you report: medians and geometric means from each campaign's `runs.csv`, not from its
  `STATUS.md`. State the session each number comes from.
- Keep "over the parent re-measured in the same session" and "over the published table" apart; they differ
  when a published cell was disturbed (L21).
- Mark every number of a campaign that is not closed with its date and "to be updated".
- Results of a public device go to the public remote, on the owner's order to push.
