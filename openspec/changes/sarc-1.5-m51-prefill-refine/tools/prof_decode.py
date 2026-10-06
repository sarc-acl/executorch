#!/usr/bin/env python3
"""prof_decode.py <dump dir> <tile_m> <tile_n> [M=2048] [tile_k=32]: decode the PROF kernels' phase counters.
(The 780M campaign's prof_decode.py with the K tile as a parameter: iterations = K / tile_k.)
Each out_<i>_<case>.bin is the fp16 [M][N] output of one dispatch; subgroup 0 of every workgroup wrote
[barrier, fetch, mma, lds_store]/iteration and [prologue, group_epilog, drain, write]/64 at its tile origin.
Prints, per case (median over tiles of the last dump of that case): cycles per iteration for the loop phases,
kernel-total cycles per phase (needs K from the case's list line: iterations = K/tile_k), and each phase's share."""
import collections, glob, os, re, statistics as st, sys
import numpy as np
d, tm, tn = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); M = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
TK = int(sys.argv[5]) if len(sys.argv) > 5 else 32
kmap = {}
if os.path.exists(os.path.join(d, "cases.txt")):
    for l in open(os.path.join(d, "cases.txt")):
        m = re.search(r"(\S+).*?K=(\d+)", l)
        if m: kmap[m.group(1)] = int(m.group(2))
last = {}
for f in sorted(glob.glob(os.path.join(d, "out_*.bin")), key=lambda p: int(re.search(r"out_(\d+)_", p).group(1))):
    last[re.sub(r".*out_\d+_", "", f)[:-4]] = f
names = "barrier fetch mma lds_store prologue group_epilog drain write".split()
print("case,N,tiles," + ",".join(n + "_cyc" for n in names) + ",total_cyc," + ",".join(n + "_pct" for n in names) + ",spread_mma_pct")
for case, f in last.items():
    a = np.fromfile(f, dtype=np.float16)
    if a.size % M: print(case, "size not a multiple of M"); continue
    N = a.size // M; a = a.reshape(M, N).astype(np.float64)
    t = a[0::tm, :][:, [c for n0 in range(0, N - tn + 1, tn) for c in range(n0, n0 + 8)]].reshape(-1, 8)
    mk = re.search(r"_K(\d+)_", case); K = int(mk.group(1)) if mk else None
    it = (K // TK) if K else None
    med = np.median(t, axis=0)
    tot = np.concatenate([med[:4] * (it or 1), med[4:] * 64.0])
    s = tot.sum()
    sp = (np.percentile(t[:, 2], 90) - np.percentile(t[:, 2], 10)) / med[2] * 100 if med[2] else 0
    print(f"{case},{N},{len(t)}," + ",".join(f"{x:.0f}" for x in tot) + f",{s:.0f}," + ",".join(f"{x / s * 100:.1f}" for x in tot) + f",{sp:.1f}" + ("" if it else "  (K unknown: loop phases are per iteration)"))
