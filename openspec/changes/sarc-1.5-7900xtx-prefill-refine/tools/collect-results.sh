#!/bin/bash
# collect-results.sh <session> <file under stage/<session>>...: copy small evidence files into results/7900xtx/sessions/<session>/,
# host names and absolute paths replaced by placeholders (merge-plan M2c). Control workstation only.
source "$(dirname "$(readlink -f "$0")")/env.sh"
S=$1; shift; O=$T/../results/7900xtx/sessions/$S; mkdir -p "$O"
for f in "$@"; do mkdir -p "$O/$(dirname "$f")"
  sed -e "s#${GPUHOST:?GPUHOST in env.local}#<gpu-host>#g" -e "s#$GROOT#<gpu-root>#g" -e "s#$(dirname "$GROOT")#<gpu-home>#g" \
      -e "s#$A#<artifacts>#g" -e "s#$(dirname "$A")#<campaign-root>#g" "$A/stage/$S/$f" > "$O/$f"; done
