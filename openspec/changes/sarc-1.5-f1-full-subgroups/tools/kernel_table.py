#!/usr/bin/env python3
"""kernel_table.py <raw/kernel dir> > kernel.csv: kernel-level old (topic4) against new (f1) from the JSON files
of test_llama_microbench (f1-chain.sh step 5: <env>-<build>-<suite>-r<1..3>.json, 3 processes per arm, arms
interleaved). Per case (environment, suite, model, scheme, op, M, K, N, kernel): the median over the processes
of kernel_median_us (SDPA cases report no kernel median: op_mean_us) for each arm, and the speed ratio new / old = old time / new time. A case counts only with
3 processes per arm in which it was ok and dispatched the same kernel in both arms. Summary lines per
environment and suite on stderr: geomean, extremes, cases outside +-2 %."""
import collections, glob, json, math, os, re, statistics as st, sys
D = sys.argv[1]; data = collections.defaultdict(lambda: collections.defaultdict(list)); order = []
for p in sorted(glob.glob(os.path.join(D, "*.json"))):
    m = re.fullmatch(r"(fused1|noenv)-(topic4|f1)-(linear-4w|linear-8da4w|sdpa)-r(\d)\.json", os.path.basename(p))
    if not m: continue
    try: cases = json.load(open(p))["cases"]
    except Exception as e: print(f"unreadable {p}: {e}", file=sys.stderr); continue
    for c in cases:
        if c.get("regime", "prefill") != "prefill" and m[3] != "sdpa": continue
        k = (m[1], m[3], c.get("model", ""), c.get("scheme", ""), c.get("op", ""), c.get("variant", ""), c.get("M"), c.get("K"), c.get("N"), c.get("kv_heads"))
        if k not in order: order.append(k)
        data[k][m[2]].append((c.get("kernel", ""), c.get("kernel_median_us") or c.get("op_mean_us"), c.get("ok")))
print("env,suite,model,scheme,op,variant,M,K,N,kv_heads,kernel,old_median_us,new_median_us,speed_new_over_old,difference_pct,outside_2pct,processes_old,processes_new")
groups = collections.defaultdict(list)
for k in order:
    o, n = data[k]["topic4"], data[k]["f1"]
    kern = sorted({x[0] for x in o + n})
    good = len(o) == 3 and len(n) == 3 and len(kern) == 1 and all(x[2] and x[1] for x in o + n)
    if not good:
        print(",".join(str(x) for x in k) + f",{'|'.join(kern)},,,,NOT_COUNTED,,{len(o)},{len(n)}"); groups[k[:2]].append(None); continue
    a, b = st.median(x[1] for x in o), st.median(x[1] for x in n); r = a / b; groups[k[:2]].append(r)
    print(",".join(str(x) for x in k) + f",{kern[0]},{a:.2f},{b:.2f},{r:.5f},{(r - 1) * 100:+.2f},{'yes' if abs(r - 1) > 0.02 else 'no'},3,3")
for g, rs in groups.items():
    v = [r for r in rs if r]
    if not v: print(f"{g}: no countable case of {len(rs)}", file=sys.stderr); continue
    gm = math.exp(sum(map(math.log, v)) / len(v))
    print(f"{g[0]} {g[1]}: {len(v)} of {len(rs)} cases counted, geomean {(gm - 1) * 100:+.2f} %, min {(min(v) - 1) * 100:+.2f} %, max {(max(v) - 1) * 100:+.2f} %, outside +-2 %: {sum(abs(r - 1) > 0.02 for r in v)}", file=sys.stderr)
