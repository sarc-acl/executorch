#!/bin/bash
# sweep_queue.sh: runs the jobs of .artifacts/queue/pending/ one at a time, in name order (detached; one queue
# per card: XE2_CARD=1 uses .artifacts/queue1/ and takes cheap-mode screens only). A job is a bash script; add one at any time by writing <NNN>-<name>.sh into pending/. Builds go
# through the queue too: the measurement guard counts a running build container as a foreign GPU process.
# Each job's output goes to queue/log/<job>.out and its status to queue/status. A job that ends 76 or 75 (a
# foreign GPU process, or the lock busy) stops the queue (QUEUE_STOPPED): stop measuring and report. The queue
# ends when queue/STOP exists and nothing is pending. Obeys the coordinator hold (host.sh) before each job; the
# jobs themselves obey it before every GPU process and build.
. "$(dirname "$(readlink -f "$0")")/host.sh"; Q=$A/queue; [[ $XE2_CARD == 1 ]] && Q=$A/queue1; mkdir -p $Q/{pending,done,log}
exec 8>$Q/lock; flock -n 8 || { echo "a queue is already running" >&2; exit 2; }
mkdir -p $RUN; echo "$$ $(cut -d' ' -f22 /proc/$$/stat)" > $RUN/queue$XE2_CARD.top
while :; do
  j=$(ls $Q/pending/*.sh 2>/dev/null | head -1)
  if [[ -z $j ]]; then [[ -e $Q/STOP ]] && { echo "$(date -u +%FT%TZ) QUEUE_DONE" >> $Q/status; exit 0; }; sleep 20; continue; fi
  n=$(basename $j .sh); coordinator_hold "queue job $n"; echo "$(date -u +%FT%TZ) start $n" >> $Q/status
  XE2_TOP= bash $j > $Q/log/$n.out 2>&1 8>&-; rc=$?; mv $j $Q/done/
  echo "$(date -u +%FT%TZ) end $n rc=$rc" >> $Q/status
  [[ $rc == 76 || $rc == 75 ]] && { echo "$(date -u +%FT%TZ) QUEUE_STOPPED after $n rc=$rc" >> $Q/status; exit $rc; }
done
