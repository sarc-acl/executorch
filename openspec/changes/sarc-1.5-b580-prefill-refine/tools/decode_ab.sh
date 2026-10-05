#!/bin/bash
# decode_ab.sh <session> [reps=5] [models=1b,3b,8b] [schemes=4w,8da4w]: decode speed of the two staged arms of
# stage/<session> (parent, cand with their env files), arms interleaved. The run is the decode step of
# sarc/tools/verify.sh: prompt_2048.txt, --max_new_tokens 32, --temperature 0, no warmup, a fresh process per
# run. Prefill is this campaign's target; this checks that a candidate does not pay for it in decode
# (devices with an SDPA row use the truncated softmax in decode too). One row per run in
# stage/<session>/decode/decode.csv (model,scheme,build,rep,decode_tok_s,prefill_tok_s,generated_tokens,rc) and
# the per-cell medians in decode/summary.csv. A foreign GPU process ends it (76).
. "$(dirname "$(readlink -f "$0")")/host.sh"; S=$A/stage/${1:?session}; REPS=${2:-5}; MODELS=${3:-1b,3b,8b}; SCHEMES=${4:-4w,8da4w}
O=$S/decode; mkdir -p $O/logs || exit 2; CSV=$O/decode.csv; MROOT=/mnt/linux-share/models
[[ -f $CSV ]] || echo "model,scheme,build,rep,decode_tok_s,prefill_tok_s,generated_tokens,rc" > $CSV
declare -A STEM=([1b]=llama-3.2-1b:llama3_2-1b [3b]=llama-3.2-3b:llama3_2-3b [8b]=llama-3.1-8b:llama3_1-8b)
IFS=, read -ra MS <<< "$MODELS"; IFS=, read -ra QS <<< "$SCHEMES"
for m in "${MS[@]}"; do IFS=: read -r MD ST <<< "${STEM[$m]}"; for q in "${QS[@]}"; do for ((r = 1; r <= REPS; r++)); do
  order="parent cand"; (( r % 2 == 0 )) && order="cand parent"
  for b in $order; do
    benv=(); [[ -f $S/$b/env ]] && mapfile -t benv < $S/$b/env
    L=$O/logs/$m-$q-$b-r$r.log
    env "${benv[@]}" LD_LIBRARY_PATH=$S/$b $TOOLS/gl.sh $S/$b/llama_main --model_path $MROOT/$MD/exported/${ST}_vulkan_$q.pte \
      --tokenizer_path $MROOT/$MD/original/tokenizer.model --prompt_file $S/prompt_2048.txt --max_new_tokens 32 --temperature 0 < /dev/null > $L 2>&1; rc=$?
    g() { grep -o "\"$1\":[0-9.]*" $L | head -1 | cut -d: -f2; }
    echo "$m,$q,$b,$r,$(g decode_token_per_sec),$(g prefill_token_per_sec),$(g generated_tokens),$rc" | tee -a $CSV
    [[ $rc == 75 || $rc == 76 ]] && { echo "DECODE_ABORTED rc=$rc"; exit $rc; }
  done
done; done; done
python3 - $CSV > $O/summary.csv <<'PY'
import csv, collections, statistics as st, sys
d = collections.defaultdict(list)
for r in csv.DictReader(open(sys.argv[1])):
    if r["rc"] == "0" and r["decode_tok_s"]: d[(r["model"], r["scheme"], r["build"])].append(float(r["decode_tok_s"]))
print("model,scheme,parent_decode_tok_s,cand_decode_tok_s,ratio,n_parent,n_cand,parent_min_max,cand_min_max")
for m, q in sorted({k[:2] for k in d}):
    p, c = d[(m, q, "parent")], d[(m, q, "cand")]
    if p and c: print(f"{m},{q},{st.median(p):.2f},{st.median(c):.2f},{st.median(c) / st.median(p):.4f},{len(p)},{len(c)},{min(p):.1f}-{max(p):.1f},{min(c):.1f}-{max(c):.1f}")
PY
cat $O/summary.csv; echo DECODE_DONE
