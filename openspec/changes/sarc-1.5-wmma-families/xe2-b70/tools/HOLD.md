# The coordinator hold

What it is for: letting the coordinator (or the owner) use a campaign's GPU for a short outside measurement
without stopping the campaign and without losing a running job. In the first five campaigns it was used to
insert the llama.cpp comparison, 20 to 75 minutes per device, between two jobs of a queue that otherwise ran
for days.

## The contract

Two files in the campaign's artifact directory (`<artifact-dir>`, the one named in the task file):

| file | written by | meaning |
|---|---|---|
| `<artifact-dir>/HOLD` | the coordinator only | "start nothing new" |
| `<artifact-dir>/HELD` | the campaign's queue only | "I have stopped at a job boundary"; one line: `HELD <UTC time> <what would start next>` |

Rules for the campaign's queue (every detached chain, sweep runner and gate script):

1. Before starting each job, check for `HOLD`. A job is the smallest restartable unit: one configuration batch,
   one session, one gate, one build.
2. While `HOLD` exists: start nothing, write `HELD`, poll once a minute.
3. A job that is running when `HOLD` appears finishes normally. It is not killed.
4. When `HOLD` is gone: remove `HELD` and continue exactly where the queue was.
5. On a host with two cards, the queues of both cards obey the same file.
6. While `HELD` exists the actor starts no GPU job and no build on that host by hand either.

Rules for who does what:

- The coordinator creates and removes `HOLD`. The campaign never does.
- The outside measurement takes the campaign's device lock while it runs, and it is a GPU workload the
  campaign's foreign-process guard does not know. That is the reason the hold exists. The campaign does not
  teach its guard to accept the coordinator's processes: a guard that learns exceptions stops guarding.
- The actor tests the hold once before it is needed (a dry run, or a differently named test file through the
  same code path) and records in `STATUS.md`, under the heading "Coordinator hold": the two paths, what the
  smallest unit is, and the longest time one unit takes. That time is how long the coordinator may have to
  wait for `HELD`.

## Reference implementation for a queue (bash)

Call `hold_point "<what starts next>"` before every job. `A` is the artifact directory.

```bash
hold_point() {
  local waited=0
  while [[ -e $A/HOLD ]]; do
    [[ $waited == 0 ]] && echo "HELD $(date -u +%FT%TZ) $1" > "$A/HELD"
    waited=1; sleep 60
  done
  [[ $waited == 1 ]] && rm -f "$A/HELD"
  return 0
}

for job in "${QUEUE[@]}"; do
  hold_point "$job"
  run_one "$job"          # under the device lock, as always
done
```

A queue whose jobs wait on the device lock without a time limit also survives an outside measurement that
simply takes the lock, but then the measurement can start in the middle of a queue's cooling wait and the
status file does not say why nothing is running. Use the hold.

## Coordinator procedure

The host's login shell may not be bash (two of the first five hosts used fish). Send every command as a
script to `bash -s`, never as a bash one-liner in the ssh argument.

```bash
# 1. Ask for the hold.
ssh -o BatchMode=yes <host> 'bash -s' <<'EOF'
touch <artifact-dir>/HOLD && ls -l <artifact-dir>/HOLD
EOF

# 2. Wait for the acknowledgement (at most the longest unit recorded in STATUS.md).
ssh -o BatchMode=yes <host> 'bash -s' <<'EOF'
for i in $(seq 1 120); do [ -s <artifact-dir>/HELD ] && { cat <artifact-dir>/HELD; exit 0; }; sleep 30; done
echo "no HELD after 60 min"; exit 1
EOF

# 3. Confirm that no GPU program of the campaign is running, then measure under the device lock.

# 4. Release, and VERIFY the release.
ssh -o BatchMode=yes <host> 'bash -s' <<'EOF'
rm -f <artifact-dir>/HOLD
[ -e <artifact-dir>/HOLD ] && { echo "HOLD STILL PRESENT"; exit 1; }
sleep 90; [ -e <artifact-dir>/HELD ] && echo "HELD still present: queue has not resumed yet" || echo "released, queue resumed"
EOF
```

Step 4 is not optional. Once, a background command that was to remove `HOLD` used bash syntax through a fish
login shell, failed without a message, and left a campaign held until the next patrol noticed it
(`LESSONS.md`, coordinator mistakes). `tools/patrol.sh` prints the hold state of every campaign for this reason.

## What the hold is not

- Not a way to stop a campaign. Stopping, restarting or resuming a run is the owner's decision
  (`COORDINATOR.md`).
- Not a lock. The device lock (`~/.cache/gpu-lab/lock-<lock-uuid>`) still serialises GPU jobs; the hold only
  decides who asks for the lock next.
- Not a pause of a running job. If the longest unit is three hours, the hold can take three hours to engage:
  ask early, or have the queue split into smaller units before the campaign starts.
