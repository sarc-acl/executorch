#!/bin/bash
# gl.sh <command...>: run one GPU job under the gpu-lab lock of the card XE2_CARD selects (one GPU job at a time
# per card). Refuses to start while a GPU process this campaign did not start is running, on either card, and
# fails (76) if one appears during the job. XE2_SHARED=1 marks a cheap-mode screen, which may run while the other
# card screens; every other job waits until the other card is idle and keeps it idle (host.sh pair_lock).
# Starts nothing while the coordinator hold is set (host.sh coordinator_hold).
. "$(dirname "$(readlink -f "$0")")/host.sh"
if [[ ${XE2_SHARED:-} == 1 ]]; then gpu_begin shared "$@"; else gpu_begin excl "$@"; fi
exec 9>>"$HOME/.cache/gpu-lab/lock-$LOCK"; flock -w 1800 9 || { echo "gpu-lab lock busy"; exit 75; }
J=$RUN/card$XE2_CARD.job; mkdir -p $RUN; echo "$$ $(cut -d' ' -f22 /proc/$$/stat)" > $J; trap 'rm -f $J' EXIT
F=$(mktemp); guarded $F "$@" 9>&- 7>&-; rc=$?; [[ -s $F ]] && cat $F >&2; rm -f $F; exit $rc
