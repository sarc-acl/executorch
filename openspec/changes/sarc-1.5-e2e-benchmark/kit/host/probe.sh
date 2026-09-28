#!/bin/bash
# Logits evidence: for every model x scheme, run logits_probe (stock and sarc builds) on
#   real_ids.txt  (2048 tokens, tile-aligned: SARC kernels engaged)
#   check_ids.txt (1972 tokens, unaligned: 4h4w/4w rows fall back to stock kernels by design)
# and write <out>/probe/<model>-<scheme>-<build>-<prompt>.json (top-10 logits + the two disputed ids).
set -uo pipefail
LOCK=""; DEV=0; MROOT=/mnt/linux-share/models; FLAT=""; OUTN=raw_real
while [[ $# -gt 0 ]]; do case $1 in
  --lock) LOCK=$2; shift ;; --device-index) DEV=$2; shift ;; --model-root) MROOT=$2; shift ;;
  --flat-models) FLAT=$2; shift ;; --out) OUTN=$2; shift ;; --gpu|--reps|--cool-max|--prompt|--models|--schemes) shift ;;
  --no-check) ;; esac; shift; done
D=$(cd "$(dirname "$0")" && pwd); cd "$D"; O=$D/$OUTN/probe; mkdir -p "$O"
exec 9>>"$HOME/.cache/gpu-lab/lock-$LOCK"; flock -w 900 9 || exit 75
export ETVK_DEVICE_INDEX=$DEV
{ date -u; hostname; command -v vulkaninfo >/dev/null && vulkaninfo --summary 2>/dev/null | grep -E 'GPU[0-9]|deviceName'; sha256sum probe-*/logits_probe; } > "$O/env.txt" 2>&1
declare -A STEM=([1b]=llama-3.2-1b:llama3_2-1b [3b]=llama-3.2-3b:llama3_2-3b [8b]=llama-3.1-8b:llama3_1-8b)
for m in 1b 3b 8b; do IFS=: read -r MD ST <<< "${STEM[$m]}"; for q in 4w 8da4w; do
  if [[ -n $FLAT ]]; then P=$FLAT/${ST}_vulkan_$q.pte; else P=$MROOT/$MD/exported/${ST}_vulkan_$q.pte; fi
  for b in stock sarc; do for pr in real check; do
    benv=(); [[ -f $D/$b/env ]] && mapfile -t benv < "$D/$b/env"   # per-build env, as in e2e.sh
    env "${benv[@]}" LD_LIBRARY_PATH=$D/$b timeout 1800 ./probe-$b/logits_probe "$P" logits_probe/${pr}_ids.txt "$O/$m-$q-$b-$pr.json" 6062 45647 \
      > "$O/$m-$q-$b-$pr.log" 2>&1 9>&-
    echo "probe $m $q $b $pr rc=$? $(tail -n1 $O/$m-$q-$b-$pr.log)"
  done; done
done; done
echo PROBE_DONE
