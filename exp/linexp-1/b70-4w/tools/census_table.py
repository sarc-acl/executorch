#!/usr/bin/env python3
"""census_table.py <isa counts csv from isa_counts.py> <out.csv>: per listing the figures of H2: SIMD width,
spills:fills, registers, and in the hot loop (the innermost loop with the most dpas): dpas, shared-memory load
messages (count, per dpas, by description), the widest and narrowest load message in bytes, stores, barriers,
image / sampler / buffer loads, sync.nop and its share of the loop's instructions."""
import csv, re, sys
DATA = {"d8u32": 1, "d16u32": 2, "d32": 4, "d64": 8}
def mb(desc):
    m = re.match(r"(.*)\((\d+)\)$", desc.strip()); d = m[1]; lanes = int(m[2])
    k = re.search(r"\.(d8u32|d16u32|d32|d64)(?:x(\d+))?t?(?:\.|$)", d)
    return lanes * DATA[k[1]] * int(k[2] or 1) if k else 0
rows = list(csv.DictReader(open(sys.argv[1])))
out = [["label", "listing", "copies", "simd", "spills", "fills", "grf", "instructions", "all_dpas", "loop_dpas", "loop_slm_loads", "loads_per_dpas", "loop_slm_load_bytes", "load_messages(description x count)", "max_load_msg_bytes", "min_load_msg_bytes", "loop_slm_stores", "loop_barriers", "loop_tgm_loads", "loop_sampler", "loop_ugm_loads", "loop_sync_nop", "loop_instructions", "sync_nop_share_pct", "nir_header"]]
for r in rows:
    parsed = [(d, int(c)) for d, c in re.findall(r"(\S+\(\d+\)) x(\d+)", r["loop_slm_load_messages"])]
    by = [mb(d) for d, _ in parsed] or [0]
    n = int(r["loop_dpas"] or 0); l = int(r["loop_slm_load"] or 0); li = int(r["loop_instructions"] or 0)
    out.append([r["label"], r["listing"], r["copies"], r["simd_width"], r["spills"], r["fills"], r["grf"], r["instructions_reported"], r["all_dpas"], n, l, f"{l / n:.2f}" if n else "", r["loop_slm_load_bytes"],
                r["loop_slm_load_messages"], max(by), min(by), r["loop_slm_store"], r["loop_barrier"], r["loop_tgm_load"], r["loop_smpl"], r["loop_ugm_load"], r["loop_sync_nop"], li,
                f"{100 * int(r['loop_sync_nop'] or 0) / li:.1f}" if li else "", r["nir_header"]])
csv.writer(open(sys.argv[2], "w", newline="")).writerows(out)
