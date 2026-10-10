# guard.sh: sourced. Locks and guards of the B580 for this study (rules 5 and 9 of the task).
# gpu_job <name> <command...>: run one GPU job under the campaign lock of the card (exclusive) and the
# desktop-build lock (shared), with the desktop-share sampler beside it, and a once-a-second check for a
# foreign GPU workload (a process whose program name, argv[0], is a known compute job and that is not a
# descendant of this shell). A foreign workload is recorded in logs/<name>.foreign and makes the job invalid
# (status 76); the job itself is not killed here (the roofline tool must close its own rows).
HERE=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")"); ROOT=$(dirname "$HERE"); LOGS=${LOGS:-$ROOT/data/logs}; mkdir -p $LOGS
LAB=$HOME/.cache/gpu-lab; DEVLOCK=$LAB/lock-86800be2-0000-0000-0300-000000000000; BUILDLOCK=$LAB/lock-desktop-build
gpu_others() { local p q a0 mine
  for p in $(pgrep -f 'llama-server|ComfyUI|comfyui|ollama|vllm|llama_main|test_llama_microbench|logits_probe|roofline|custom_ops'); do
    a0=$(tr '\0' '\n' < /proc/$p/cmdline 2>/dev/null | head -1); a0=${a0##*/}
    [[ $a0 =~ ^(llama-server|ollama|vllm|llama_main|test_llama_microbench|logits_probe|roofline|roofline_sustained|inspect)$ ]] || continue
    q=$p; mine=0
    while [[ -n $q && $q -gt 1 ]]; do [[ $q == $$ ]] && { mine=1; break; }; q=$(ps -o ppid= -p $q 2>/dev/null | tr -d ' '); done
    [[ $mine == 0 && -d /proc/$p ]] && printf '%s:%s;' $p "$a0"
  done 2>/dev/null; }
busy_pct() { "$(ls -t $LAB/gpu-harness-* | head -1)" --lock-dir $LAB --list | python3 -c 'import json,sys
d=json.loads(next(l for l in sys.stdin if l.startswith("{")))
print(next(x for x in d["devices"] if x["uuid"].startswith("86800be2")).get("busy_pct",0))'; }
gpu_job() { local n=$1 o j rc s; shift
  exec 8>>"$BUILDLOCK"; flock -s -w 14400 8 || { echo "desktop-build lock busy" >&2; return 75; }
  exec 9>>"$DEVLOCK"; flock -x -w 1800 9 || { echo "campaign lock busy" >&2; return 75; }
  o=$(gpu_others); [[ -n $o ]] && { echo "before start: $o" > $LOGS/$n.foreign; return 76; }
  : > $LOGS/$n.foreign
  python3 $HERE/desktop_share.py $LOGS/$n.share.jsonl $$ 2 9>&- 8>&- & s=$!
  "$@" 9>&- & j=$!
  while kill -0 $j 2>/dev/null; do o=$(gpu_others); [[ -n $o ]] && echo "$(date -u +%FT%TZ) $o" >> $LOGS/$n.foreign; sleep 1; done
  wait $j; rc=$?; kill $s 2>/dev/null; exec 9>&- 8>&-
  [[ -s $LOGS/$n.foreign ]] && return 76; return $rc; }
# linexp-1 additions. share_mean <name>: mean foreign desktop share (percent) of the job's samples.
# gpu_job_busy <name> <cmd...>: gpu_job, then status 77 if the mean foreign share is above BUSYMAX (5.0 %, the
# tuning campaign's ceiling); the caller moves the output to superseded/busy/ and repeats once.
BUSYMAX=5.0
share_mean() { python3 -c 'import json,sys
v=[json.loads(l).get("foreign_pct") for l in open(sys.argv[1])]; v=[x for x in v if x is not None and 0 <= x <= 100]  # a share outside 0..100 is an unreadable sample
print("%.2f" % (sum(v)/len(v)) if v else "nan")' $LOGS/$1.share.jsonl 2>/dev/null || echo nan; }
gpu_job_busy() { local n=$1 rc m; gpu_job "$@"; rc=$?; m=$(share_mean $n); echo "$(date -u +%FT%TZ) $n rc=$rc foreign_mean_pct=$m" >> $LOGS/jobs.txt
  [[ $rc == 76 || $rc == 75 ]] && return $rc
  [[ $m == nan ]] && return 77   # no readable share sample: cannot be accepted
  python3 -c "import sys; sys.exit(0 if float('$m')<=$BUSYMAX else 1)" || return 77; return $rc; }
