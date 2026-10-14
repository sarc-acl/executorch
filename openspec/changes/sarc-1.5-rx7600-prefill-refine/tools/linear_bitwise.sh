#!/bin/bash
# linear_bitwise.sh <session> [schemes="4w 8da4w"]: output of every real prefill linear shape (texture3d, the model
# path; test_llama_microbench --linear, inputs from fixed seeds) under the parent env and the candidate env of
# stage/<session>, dumped with ET_VK_DUMP_OUTPUT_DIR and compared byte for byte, plus the dispatched kernel per case.
# Output stage/<session>/linear-bitwise/{<arm>-<scheme>/,<arm>-<scheme>.{log,json},bitwise.txt}
source "$(dirname "$(readlink -f "$0")")/env.sh"
S=$A/stage/$1; O=$S/linear-bitwise; mkdir -p $O
for q in ${2:-4w 8da4w}; do
  for arm in parent cand; do
    e=(); mapfile -t e < $S/$arm/env; mkdir -p $O/$arm-$q
    env "${e[@]}" ET_VK_DUMP_OUTPUT_DIR=$O/$arm-$q $T/gl.sh $S/test_llama_microbench --linear --regime=prefill --scheme=$q \
      --storage=texture3d --skip-correctness --json-out=$O/$arm-$q.json > $O/$arm-$q.log 2>&1
  done
  for f in $O/parent-$q/*.bin; do b=$(basename $f)
    if cmp -s $f $O/cand-$q/$b; then echo "$q $b IDENTICAL"; else echo "$q $b DIFFERS"; fi
  done >> $O/bitwise.txt
  /usr/bin/python3 - $O/parent-$q.json $O/cand-$q.json >> $O/kernels.txt <<'PY'
import json, sys
p, c = [{(x["model"], x["op"]): x["kernel"] for x in json.load(open(f))["cases"] if x["regime"] == "prefill"} for f in sys.argv[1:3]]
for k in sorted(p): print(k[0], k[1], p[k], "->", c.get(k))
PY
done
echo "bitwise: $(grep -c IDENTICAL $O/bitwise.txt) identical, $(grep -c DIFFERS $O/bitwise.txt) differ"
