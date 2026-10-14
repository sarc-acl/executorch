#!/bin/bash
# bitcmp.sh <out name> <build tag> <scheme> <token...>: device side. For a linear kernel that claims not to change
# the arithmetic: the raw output of every prefill case of test_llama_microbench (texture3d, all three models, the
# test's seeded inputs; ET_VK_DUMP_OUTPUT_DIR) with the kernel selected by name, compared byte for byte with the
# output of the table's kernel ("base"). base is run twice: the two base dumps must be identical themselves,
# otherwise the comparison says nothing. Output: raw/<out name>/summary.txt, one line per (token, case):
# identical, or the number of differing fp16 elements and the largest absolute difference.
T=$(dirname "$0"); source "$T/common.sh"; [[ $SIDE == device ]] || { echo "device only" >&2; exit 2; }
O=$A/raw/$1; BD=$A/build/$2/bundle; B=$BD/test_llama_microbench; Q=$3; shift 3; need $B; mkdir -p $O
VAR=ET_VK_SARC_Q4GSW_VARIANT; [[ $Q == 8da4w ]] && VAR=ET_VK_SARC_DQ8CA_VARIANT
run() { local d=$O/$1; shift; [[ -s $d/run.log ]] && return 0; mkdir -p $d; cool_start 60
  env "$@" ET_VK_DUMP_OUTPUT_DIR=$d LD_LIBRARY_PATH=$BD $T/gl.sh $B --linear --regime=prefill --scheme=$Q --storage=texture3d --skip-correctness > $d/run.log 2>&1
  local rc=$?; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && exit $rc; return 0; }
run base; run base2
for t in "$@"; do run $t $VAR=$t; done
python3 - $O base2 "$@" > $O/summary.txt <<'PY'
import glob, os, re, sys
import numpy as np
O = sys.argv[1]
def last(d):   # the last dump of every case (the timed runs repeat a case)
    r = {}
    for f in sorted(glob.glob(os.path.join(O, d, "out_*.bin")), key=lambda p: int(re.search(r"out_(\d+)_", p).group(1))):
        r[re.sub(r".*out_\d+_", "", f)[:-4]] = f
    return r
base = last("base")
for t in sys.argv[2:]:
    cur = last(t); kern = sorted(set(re.findall(r"sarc_linear_[a-z0-9_]+|linear_[a-z0-9_]*tiled[a-z0-9_]*|q4gsw_linear[a-z0-9_]*", open(os.path.join(O, t, "run.log"), errors="replace").read())))
    for case in sorted(base):
        if case not in cur: print(f"{t},{case},MISSING"); continue
        a = np.fromfile(base[case], dtype=np.float16); b = np.fromfile(cur[case], dtype=np.float16)
        if a.size != b.size: print(f"{t},{case},SIZE {a.size} vs {b.size}"); continue
        n = int((a.view(np.uint16) != b.view(np.uint16)).sum())
        d = float(np.nanmax(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if n else 0.0
        print(f"{t},{case},{'identical' if n == 0 else f'DIFFERENT elements={n}/{a.size} max_abs_diff={d:.6g}'},{';'.join(k for k in kern if 'quantize' not in k)[:140]}")
PY
echo "identical: $(grep -c ',identical,' $O/summary.txt) of $(wc -l < $O/summary.txt) (token, case) pairs"; grep -v ',identical,' $O/summary.txt | head -20
echo BITCMP_DONE
