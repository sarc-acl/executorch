#!/usr/bin/env bash
# push.sh: prepare the device dir run.sh expects. Run once per device (or after a rebaseline), not per
# driver build: the binaries and models stay fixed so that only the driver changes.
#
# usage: push.sh --serial SERIAL --stock-bin FILE --sarc-bin FILE --models DIR --tokenizer FILE
#                [--device-dir /data/local/tmp/e2eb]
#   --stock-bin  llama_main built from upstream release/1.5 (985c1ceccc), static, arm64-v8a
#   --sarc-bin   llama_main built from this branch (ci/1.5), static, arm64-v8a
#   --models     host dir with llama3_2-1b / llama3_2-3b / llama3_1-8b _vulkan_{4w,8da4w}.pte
#   --tokenizer  the Llama tokenizer.model
set -euo pipefail
HERE=$(dirname "$(readlink -f "$0")")
PROMPTS=$(readlink -f "$HERE/../../../openspec/changes/sarc-1.5-e2e-benchmark/kit/prompts")
S=""; STOCK=""; SARC=""; MDIR=""; TOK=""; D=/data/local/tmp/e2eb
while [[ $# -gt 0 ]]; do
  case $1 in
    --serial) S=$2; shift ;; --stock-bin) STOCK=$2; shift ;; --sarc-bin) SARC=$2; shift ;;
    --models) MDIR=$2; shift ;; --tokenizer) TOK=$2; shift ;; --device-dir) D=$2; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac; shift
done
[[ -n $S && -n $STOCK && -n $SARC && -n $MDIR && -n $TOK ]] || { sed -n '2,12p' "$0"; exit 2; }
A() { adb -s "$S" "$@"; }
A shell "mkdir -p $D/stock $D/sarc $D/models"
A push "$STOCK" "$D/stock/llama_main"; A push "$SARC" "$D/sarc/llama_main"
A shell "chmod 755 $D/stock/llama_main $D/sarc/llama_main"
A push "$TOK" "$D/tokenizer.model"
for p in "$PROMPTS"/*.txt; do A push "$p" "$D/"; done
for stem in llama3_2-1b llama3_2-3b llama3_1-8b; do for q in 4w 8da4w; do
  f="$MDIR/${stem}_vulkan_$q.pte"
  if [[ -f $f ]]; then A push "$f" "$D/models/"; else echo "missing $f (that cell will not run)" >&2; fi
done; done
{ sha256sum "$STOCK" "$SARC" "$TOK"; (cd "$MDIR" && sha256sum *_vulkan_*.pte); } > device-files.sha256
echo "pushed to $S:$D; hashes in device-files.sha256"
