#!/usr/bin/env python3
"""sdpa_error_table.py <dir with parent-<tier>.log and cand-<tier>.log> [out csv]: rms and maximum error of the SDPA
output against the fp64 reference ([sdpa-error] lines of test_llama_microbench, ET_VK_SDPA_ERROR_REPORT=1), parent and
candidate side by side per case, and the gate criterion of the reference-error rule on the production shapes
(tier full, S = 2048 and the continued prefill): candidate rms and maximum not larger than the parent's."""
import re, sys, os
d = sys.argv[1]; rows = ["tier,case,elements,ref_rms,parent_rms,cand_rms,rms_ratio,parent_max,cand_max,max_ratio,cand_not_larger,parent_kernels,cand_kernels"]; ok = True
for tier in ("extended", "peaked", "full"):
    P, C, K = {}, {}, {}
    for arm, dst in (("parent", P), ("cand", C)):
        f = os.path.join(d, f"{arm}-{tier}.log")
        if not os.path.exists(f): continue
        t = open(f).read()
        for m in re.finditer(r"\[sdpa-error\] (\S+) n=(\d+) rms=(\S+) max=(\S+) ref_rms=(\S+)", t): dst[m.group(1)] = (int(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5)))
        for m in re.finditer(r"\[sdpa-kernels\] (\S+) qk=(\S+) softmax=(\S+) av=(\S+) fused=(\S+)", t):
            K[(arm, m.group(1))] = m.group(5) if m.group(5) != "-" else "+".join(x.replace("_buffer_buffer_half", "").replace("_buffer_half", "") for x in m.group(2, 3, 4))
    for c in P:
        if c not in C: continue
        n, pr, pm, rr = P[c]; _, cr, cm, _ = C[c]; good = cr <= pr and cm <= pm
        if tier == "full": ok = good if ok is None else (ok and good)
        rows.append(f"{tier},{c},{n},{rr:.4e},{pr:.4e},{cr:.4e},{cr / pr:.3f},{pm:.4e},{cm:.4e},{cm / pm:.3f},{'yes' if good else 'NO'},{K.get(('parent', c), '')},{K.get(('cand', c), '').replace('_buffer_buffer_half', '')}")
text = "\n".join(rows) + "\n"
if len(sys.argv) > 2: open(sys.argv[2], "w").write(text)
sys.stdout.write(text); print("production shapes (tier full): candidate error not larger than the parent's:", "no data" if ok is None else "yes" if ok else "NO")
if len(rows) == 1 or ok is None: sys.exit("sdpa_error_table: no complete parent/candidate rows (header only or no tier full): not a table")
