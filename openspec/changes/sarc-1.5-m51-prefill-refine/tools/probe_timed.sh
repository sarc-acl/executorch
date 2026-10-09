#!/bin/bash
# probe_timed.sh <session> <model 1b|3b|8b> <scheme> [prompt file name in the stage dir = prompt_2048.txt]: the next-token logits at the last position of ONE prompt (default: the timed prompt)
# for the four arms of stage/m51-LOCAL-ONLY/<session>: parent and candidate (logits_probe of each build, env of each arm), each default and with ET_VK_FORCE_TILED_LINEAR=1
# (owner decision D1 item 1: the logits at the differing position for every arm). Needs the logits_probe binaries that probe_m51.sh pushed (lp-parent, lp-cand).
# Tokenization: probe_prompts.py's tokenizer (tiktoken, Llama 3 file), no BOS (llama_main adds none for prompt_2048.txt: its run reports 2048 prompt tokens, and so does this tokenization); PROBE_BOS=1 prepends one. The token count must equal 2048.
# Output: <stage>/probe-timed/<model>-<scheme>-<arm>-<mode>.{bin,log}, tokens-<model>.txt; analysis: probe_pos.py. One coordinator-hold unit.
set -uo pipefail
[[ -n ${SARC_HOLD_UNIT:-} ]] || exec env SARC_HOLD_UNIT=1 "$(dirname "$(readlink -f "$0")")/hold.sh" run "timed-prompt logits probe_timed.sh $*" "$0" "$@"
source "$(dirname "$(readlink -f "$0")")/dev.sh"
SES=$1; M=$2; Q=$3; PF=${4:-prompt_2048.txt}; S=$ART/stage/$LOC/$SES; DS=$DEV_ROOT/stage/$SES; O=$S/probe-timed; mkdir -p "$O"
declare -A STEM=([1b]=llama3_2_1b [3b]=llama3_2_3b [8b]=llama3_1_8b)
"$ART/venv/m51/bin/python" -I - "$S/$PF" "$O/tokens-$M-$PF" <<'PY' || exit 2
import os, sys, tiktoken
from tiktoken.load import load_tiktoken_bpe
PAT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
tok = os.environ.get("TOKENIZER", "<tokenizer>")
ranks = load_tiktoken_bpe(tok); n = len(ranks)
sp = {"<|begin_of_text|>": n, "<|end_of_text|>": n + 1}
enc = tiktoken.Encoding(name="l3", pat_str=PAT, mergeable_ranks=ranks, special_tokens=sp)
ids = ([n] if os.environ.get("PROBE_BOS") == "1" else []) + enc.encode(open(sys.argv[1]).read(), disallowed_special=())
print("tokens", len(ids), file=sys.stderr)
open(sys.argv[2], "w").write(" ".join(map(str, ids)) + "\n")
PY
A shell "mkdir -p $DS/probe" < /dev/null; A push "$O/tokens-$M-$PF" "$DS/probe/" > /dev/null
for arm in parent cand; do for mode in default tiled; do
  n=$M-$Q-$arm-$mode; st=$(device_state); [[ $st == ok ]] || { echo "board not fit before $n: $st"; exit 3; }
  e=$(tr '\n' ' ' < "$S/$arm/env"); [[ $mode == tiled ]] && e+=" ET_VK_FORCE_TILED_LINEAR=1"
  A shell "cd $DS/probe && rm -f t-$n.bin && $e timeout 1190 ./lp-$arm $DEV_ROOT/models/${STEM[$M]}_${Q}_embq_ctx3072.pte tokens-$M-$PF t-$n.bin < /dev/null > t-$n.log 2>&1; echo RC=\$? >> t-$n.log" < /dev/null > /dev/null 2>&1
  alive || { echo "board gone during timed probe $n $(date -u +%FT%TZ)" | tee -a "$ART/ABORTED"; exit 3; }
  A pull "$DS/probe/t-$n.log" "$O/$n.log" > /dev/null; A pull "$DS/probe/t-$n.bin" "$O/$n.bin" > /dev/null 2>&1
  echo "timed probe $n $(tail -1 "$O/$n.log") $(date -u +%FT%TZ)"
done; done
echo PROBE_TIMED_DONE
