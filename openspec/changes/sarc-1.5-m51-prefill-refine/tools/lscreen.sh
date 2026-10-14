#!/bin/bash
# lscreen.sh <mbstage name> <out csv> <4w|8da4w> <rounds> <token>...: kernel-level screen of linear kernels with
# test_llama_microbench --linear --regime=prefill --storage=texture3d --skip-correctness (all three models, the model
# path's storage). Token "base" = no override (the table's kernel); "x:<kernel base>" = exact name through
# ET_VK_SARC_780M_{Q4,DQ}; anything else = ET_VK_SARC_{Q4GSW,DQ8CA}_VARIANT=<token>. Tokens are interleaved and the
# order rotates by one every round; the board cools (G3D <= COOL_C, default 38 C, or no new minimum for 30 s, at
# most 300 s) before every run. Appends one row per (token, round, shape) to <out csv> and skips (token, round) pairs already complete
# (all shapes present), so it can be restarted. A screen, not a gate: no correctness check here.
set -uo pipefail
source "$(dirname "$(readlink -f "$0")")/dev.sh"
STG=$ART/stage/$LOC/$1; CSV=$2; Q=$3; R=$4; shift 4; TOK=("$@")
cd "$STG" || exit 2; mkdir -p lscreen
[[ -f $CSV ]] || echo "token,round,model,op,M,K,N,kernel,kernel_median_us,kernel_cov,temp_pre,utc" > "$CSV"
VAR=ET_VK_SARC_Q4GSW_VARIANT; X=ET_VK_SARC_780M_Q4; [[ $Q == 8da4w ]] && { VAR=ET_VK_SARC_DQ8CA_VARIANT; X=ET_VK_SARC_780M_DQ; }
cool() { local t0=$SECONDS t best=999 tb=$SECONDS
  while :; do t=$(gtemp); [[ $t =~ ^[0-9]+$ ]] || return; (( t < best )) && { best=$t; tb=$SECONDS; }
    (( t <= ${COOL_C:-38} || SECONDS - tb >= 30 || SECONDS - t0 >= 300 )) && return; sleep 5; done; }
n=${#TOK[@]}
for ((r = 1; r <= R; r++)); do for ((i = 0; i < n; i++)); do
  t=${TOK[$(( (i + r - 1) % n ))]}
  have=$(awk -F, -v t="$t" -v r=$r '$1 == t && $2 == r' "$CSV" | wc -l); (( have >= 12 )) && continue
  st=$(device_state); [[ $st == ok ]] || { echo "board not fit: $st"; exit 3; }
  E=(); case $t in base) ;; x:*) E=("$X=${t#x:}") ;; *) E=("$VAR=$t") ;; esac
  cool; tp=$(gtemp); j=lscreen/$Q-${t//[:\/]/_}-r$r.json
  env "${E[@]}" ET_VK_SARC_UNVERIFIED=1 timeout 1800 ./test_llama_microbench --linear --regime=prefill --scheme=$Q \
    --storage=texture3d --skip-correctness --json-out="$STG/$j" > "${j%.json}.log" 2>&1; rc=$?
  python3 - "$STG/$j" "$t" "$r" "$tp" "$(date -u +%FT%TZ)" >> "$CSV" <<'PY'
import json, sys
j, t, r, tp, utc = sys.argv[1:6]
try: d = json.load(open(j))
except Exception: sys.exit(0)
for c in d.get("cases", []):
    if c.get("suite") == "linear" and c.get("storage") == "texture3d":
        print(f'{t},{r},{c["model"]},{c["op"]},{c["M"]},{c["K"]},{c["N"]},{c["kernel"]},{c["kernel_median_us"]},{c["kernel_cov"]},{tp},{utc}')
PY
  echo "$Q $t r$r rc=$rc T=$tp rows=$(awk -F, -v t="$t" -v r=$r '$1 == t && $2 == r' "$CSV" | wc -l)"
done; done
echo LSCREEN_DONE
