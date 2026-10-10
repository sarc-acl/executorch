#!/usr/bin/env python3
"""isa_loops.py <outdir> <label=path>...: writes the listing of every loop that contains dpas (label .. while)
of each distinct dpas shader in an INTEL_DEBUG=cs dump or a roofline GEN_Assembly listing, one file per loop,
with the statistics line first. For the side-by-side reading of part C."""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isa_counts import loops  # noqa: E402
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
for arg in sys.argv[2:]:
    label, path = arg.split("=", 1)
    txt = open(path, errors="replace").read().splitlines()
    heads = [i for i, l in enumerate(txt) if l.startswith("Native code for")]
    blocks, seen = [], set()
    if heads:
        for h, e in zip(heads, heads[1:] + [len(txt)]):
            body = txt[h + 2:e]
            body = body[:next((i for i, l in enumerate(body) if l.startswith("NIR (")), len(body))]
            key = (re.search(r"src_hash (0x[0-9a-f]+)", txt[h])[1], txt[h + 1].split(" sends")[0])
            if key in seen or not any("dpas" in l for l in body): continue
            seen.add(key); blocks.append((f"{label}-{key[0]}-{len(blocks)}", txt[h] + "\n" + txt[h + 1], body))
    else:
        blocks.append((label, path, txt))
    for name, head, body in blocks:
        for n, (a, b) in enumerate(loops(body)):
            if any("dpas" in l for l in body[a:b + 1]):
                open(f"{out}/{name}.loop{n}.txt", "w").write(head + f"\n(loop lines {a + 1}-{b + 1} of the listing)\n" + "\n".join(body[a:b + 1]) + "\n")
