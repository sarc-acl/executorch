#!/bin/bash
# hold.sh: the coordinator hold of this campaign's detached queue (owner decision 2026-10-06 00:35 UTC).
#   <artifacts>/HOLD   created and removed by the coordinator, never by the campaign
#   <artifacts>/HELD   written here once HOLD exists and no unit of the queue is running:
#                      one line "HELD <UTC time> <what would start next>" per waiting queue; removed when HOLD is gone
#   <artifacts> = ~/hmz-sarc/.artifacts (the artifact directory of the campaign section)
# Every unit of the queue (one sweep configuration, one microbench run, one timed session, one verify.sh, one
# trace set, one build) holds a shared lock on <artifacts>/.hold-busy while it runs and looks at HOLD after taking
# it, so HELD is only written when an exclusive lock on that file can be had: nothing is running and nothing can
# start. A unit that is running when HOLD appears finishes normally.
#   hold.sh wait "<what would start next>"        block while HOLD exists (poll 60 s), then return
#   hold.sh run "<what>" <command...>             wait, then run the command as one unit (shared lock held)
#   hold.sh watch                                 detached watcher: writes HELD when HOLD exists and the queue is
#                                                 idle (no chain at a unit boundary), removes a stale HELD
#   hold.sh vars                                  prints HOLD=..., HELD=..., BUSY=... (for gl.sh, sweep_space.py)
# Test without touching HOLD: SARC_HOLD_NAME=HOLD-TEST (then HELD-TEST), SARC_HOLD_POLL=<seconds>.
D=${SARC_HOLD_DIR:-$HOME/hmz-sarc/.artifacts}; N=${SARC_HOLD_NAME:-HOLD}; POLL=${SARC_HOLD_POLL:-60}
HOLD=$D/$N; HELD=$D/HELD${N#HOLD}; BUSY=$D/.hold-busy
held_if_idle() {
  ( flock -n -x 7 || exit 0
    [[ -e $HOLD ]] || exit 0
    grep -qsF -- " $1" "$HELD" || echo "HELD $(date -u +%FT%TZ) $1" >> "$HELD" ) 7>>"$BUSY"
}
wait_hold() {
  [[ -e $HOLD ]] || return 0
  while [[ -e $HOLD ]]; do held_if_idle "$1"; sleep "$POLL"; done
  rm -f "$HELD"
}
case ${1:-} in
  vars) echo "HOLD=$HOLD"; echo "HELD=$HELD"; echo "BUSY=$BUSY" ;;
  wait) wait_hold "${2:-next unit}" ;;
  run) what=$2; shift 2
    while :; do
      wait_hold "$what"; exec 8>>"$BUSY"; flock -s 8
      [[ -e $HOLD ]] || break
      exec 8>&-
    done
    "$@" ;;
  watch)
    while :; do
      if [[ -e $HOLD ]]; then [[ -s $HELD ]] || held_if_idle "queue idle: no unit running and none at a unit boundary"
      elif [[ -e $HELD ]]; then rm -f "$HELD"; fi
      sleep "$POLL"
    done ;;
  *) sed -n '2,18p' "$0"; exit 2 ;;
esac
