#!/bin/bash
# sdpa_screen.sh <out name> <build tag> <reps> <profile...>: kernel-level SDPA screen with test_llama_microbench
# --sdpa (QK^T, softmax and attn*V times per model, GPU timestamps). Device side. Profile "table" = no profile (on
# this device the same as stock: the Orin SDPA rows need an orin-* profile); "stock" = no SARC SDPA at all (no ET_VK_SARC_UNVERIFIED: the parent's kernels). Profiles are
# interleaved; one JSON per (profile, repeat) in raw/<out name>/ and one row per (profile, repeat, model, regime,
# op, variant) appended to raw/<out name>/rows.csv. Resumable: a (profile, repeat) whose JSON exists is skipped.
# A screen, not a gate: no correctness check.
T=$(dirname "$0"); source "$T/common.sh"; O=$A/raw/$1; BD=$A/build/$2/bundle; B=$BD/test_llama_microbench; R=$3; shift 3
[[ $SIDE == device ]] || { echo "device only" >&2; exit 2; }
need $B; mkdir -p $O; { sha256sum $B; date -u; } >> $O/env.txt
[[ -f $O/rows.csv ]] || echo "profile,rep,model,regime,op,variant,mean_us,stdev_us,dispatch,kernels" > $O/rows.csv
for ((r = 1; r <= R; r++)); do for p in "$@"; do
  [[ -s $O/$p-r$r.json ]] && continue
  # "<profile>+VAR=VALUE[+VAR=VALUE]" adds environment (e.g. the softmax variant of a local-hook build)
  IFS=+ read -ra X <<< "$p"; pn=${X[0]}
  E=(ET_VK_SARC_UNVERIFIED=1); [[ $pn == stock ]] && E=(); [[ $pn != table && $pn != stock ]] && E+=("ET_VK_SARC_DEV_PROFILE=$pn"); E+=("${X[@]:1}")
  cool_start 120
  env "${E[@]}" LD_LIBRARY_PATH=$BD $T/gl.sh $B --sdpa --json-out=$O/$p-r$r.json > $O/$p-r$r.log 2>&1
  rc=$?; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "screen stopped rc=$rc (70 device lost, 75 lock busy, 76 foreign GPU process)"; exit $rc; }
  python3 - $O/$p-r$r.json $p $r >> $O/rows.csv <<'PY'
import json, sys
try: d = json.load(open(sys.argv[1]))
except Exception: sys.exit(0)
for c in (d.get("cases") or d.get("records") or (d if isinstance(d, list) else [])):
    if c.get("suite") != "sdpa": continue
    print(",".join(str(x) for x in (sys.argv[2], sys.argv[3], c.get("model"), c.get("regime"), c.get("op"), c.get("variant"), c.get("op_mean_us"), c.get("op_stdev_us"), c.get("dispatch"), c.get("kernel", ""))))
PY
  echo "$p r$r rc=$rc temp=$(gtemp)"
done; done
date -u >> $O/env.txt; echo SCREEN_DONE >> $O/env.txt
