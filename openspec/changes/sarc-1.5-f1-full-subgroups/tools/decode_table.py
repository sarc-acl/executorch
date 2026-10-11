#!/usr/bin/env python3
"""decode_table.py <kernel-decode dir> [<kernel dir of step 5>]: the post hoc check of f1-decode.sh. Per environment
and model, the decode-size (M = 1) "total" case of the stock tiled SDPA kernels: every process's value (us, in run
order) of each arm, the median per arm and new / old as a speed ratio (old time / new time). With the second
argument the same line from the 3 processes per arm of step 5 is printed first."""
import glob, json, os, re, statistics as st, sys
def load(D):
    v = {}
    for p in sorted(glob.glob(os.path.join(D, "*.json")), key=lambda p: int(re.search(r"-r(\d+)\.json", p)[1])):
        m = re.fullmatch(r"(fused1|noenv)-(topic4|f1)-sdpa-r(\d+)\.json", os.path.basename(p))
        if not m: continue
        for c in json.load(open(p))["cases"]:
            if c.get("M") == 1 and c.get("op") == "total" and c.get("variant") == "tiled" and c.get("ok"):
                v.setdefault((m[1], c["model"]), {}).setdefault(m[2], []).append(c.get("kernel_median_us") or c["op_mean_us"])
    return v
def show(tag, v):
    for (e, model), a in v.items():
        o, n = a.get("topic4", []), a.get("f1", [])
        r = st.median(o) / st.median(n)
        print(f"{tag} | {e} | {model} | {' '.join(f'{x:.0f}' for x in o)} | {' '.join(f'{x:.0f}' for x in n)} | {st.median(o):.0f} | {st.median(n):.0f} | {(r - 1) * 100:+.2f} % | {min(o):.0f} to {max(o):.0f} | {min(n):.0f} to {max(n):.0f}")
print("run | environment | model | old topic4, per process (us) | new f1, per process (us) | old median | new median | new / old (speed) | old range | new range")
if len(sys.argv) > 2: show("step 5 (3 processes)", load(sys.argv[2]))
show("post hoc (10 processes)", load(sys.argv[1]))
