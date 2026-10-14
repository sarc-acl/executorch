#!/bin/bash
# push_models.sh: copies the six measured .pte files and the tokenizer to $DEV_ROOT/models on the board (once),
# then checks every file's sha256 on the board against MANIFEST.json. Read-only on the model share.
# PUSH_MODELS="llama3_2_1b llama3_2_3b" limits the sizes (this campaign: no 8B file on the board, owner decision 2026-10-06 19:36 UTC).
set -uo pipefail
source "$(dirname "$(readlink -f "$0")")/dev.sh"
P=${PTE_DIR:-<pte-dir>}; TOK=${TOK:-<tokenizer>}
A shell "mkdir -p $DEV_ROOT/models"
for m in ${PUSH_MODELS:-llama3_2_1b llama3_2_3b llama3_1_8b}; do for q in 4w 8da4w; do
  f=${m}_${q}_embq_ctx3072.pte; want=$(python3 -c "import json;print(json.load(open('$P/MANIFEST.json'))['files']['$f']['sha256'])")
  have=$(A shell "sha256sum $DEV_ROOT/models/$f 2>/dev/null" | cut -d' ' -f1)
  [[ $have == "$want" ]] || { A push "$P/$f" $DEV_ROOT/models/ > /dev/null && have=$(A shell "sha256sum $DEV_ROOT/models/$f" | cut -d' ' -f1); }
  echo "$f $([[ $have == "$want" ]] && echo OK || echo "MISMATCH $have") $want"
done; done
A push "$TOK" $DEV_ROOT/models/tokenizer.model > /dev/null
echo "tokenizer.model $(A shell "sha256sum $DEV_ROOT/models/tokenizer.model" | cut -d' ' -f1) local $(sha256sum "$TOK" | cut -d' ' -f1)"
echo PUSH_DONE
