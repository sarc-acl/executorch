#!/usr/bin/env python3
"""isa_counts.py <out.csv> <label=path>...: counts from Intel GEN listings.
A path is either an INTEL_DEBUG=cs dump of a process (several shaders; each "Native code for ..." block with
its statistics line) or a listing returned through VK_KHR_pipeline_executable_properties by the roofline
tool (pipeline-inspection/<variant>.compute.GEN_Assembly.txt, statistics in <variant>.json).
Per shader that contains dpas: SIMD width, instructions, loops, spills:fills, sends, and per loop
(label .. `while`) the number of instructions, dpas, shared-memory (slm) loads and stores with their data
widths, global (ugm) loads/stores, sampler and typed-image sends, and sync instructions.
One dpas (repeat count 8 in the encoding, printed once) is one 8-row matrix multiply-add."""
import collections, csv, json, re, sys


def loops(lines):
    lab = {m[1]: i for i, l in enumerate(lines) if (m := re.match(r"^(L\d+):", l))}
    out = []
    for j, l in enumerate(lines):
        m = re.match(r"^\s*(?:\(W\)\s*)?while \(\d+\)\s+(L\d+)", l)
        if m and m[1] in lab:
            out.append((lab[m[1]], j))
    return out


def classify(l):
    m = re.match(r"^\s*(?:\([^)]*\)\s*)*([a-z][a-z0-9_.]*)\s*(?:\((\d+)\))?", l)
    if not m or re.match(r"^(L\d+:|START|END|\s*$)", l):
        return None
    op = m[1]
    c = l.split("//", 1)[1] if "//" in l else ""
    if op.startswith("dpas"): return "dpas"
    if op.startswith("send"):
        w = re.search(r"\.(d\d+(?:u\d+)?(?:x\d+)?v?\d*)\.", c + ".")
        w = w[1] if w else "?"
        if ".slm" in op or "slm" in c:
            return ("slm_load " if "load" in c else "slm_store " if "store" in c else "slm_other ") + w
        if "load.ugm" in c or "store.ugm" in c:
            return ("ugm_load " if "load" in c else "ugm_store ") + w
        if "sampl" in c or ".smpl" in op: return "sampler"
        if "typed" in c or ".tgm" in op: return "typed_image " + ("load" if "load" in c or "read" in c else "store")
        if ".gtwy" in op or "barrier" in c: return "gateway(barrier)"
        return "send_other"
    if op.startswith("sync"): return "sync"
    return "other"


def count(lines, a, b):
    c = collections.Counter()
    for l in lines[a:b + 1]:
        k = classify(l)
        if k: c[k] += 1; c["instructions"] += 1
    return c


def fmt(c):
    grp = lambda p: "; ".join(f"{k[len(p):]} x{v}" for k, v in sorted(c.items()) if k.startswith(p)) or "0"
    return [c["instructions"], c["dpas"], grp("slm_load "), grp("slm_store "), grp("ugm_load "), grp("ugm_store "),
            c["sampler"], grp("typed_image "), c["gateway(barrier)"], c["sync"], c["send_other"]]


if __name__ == "__main__":
    rows = [["listing", "shader", "simd", "instructions", "loops", "spills", "fills", "sends", "scope", "scope_instructions", "dpas",
             "slm_loads", "slm_stores", "ugm_loads", "ugm_stores", "sampler_sends", "typed_image_sends", "gateway_sends", "sync", "other_sends"]]
    for arg in sys.argv[2:]:
        label, path = arg.split("=", 1)
        txt = open(path, errors="replace").read().splitlines()
        blocks = []
        heads = [i for i, l in enumerate(txt) if l.startswith("Native code for")]
        if heads:
            ends = heads[1:] + [len(txt)]
            seen = set()
            for h, e in zip(heads, ends):
                st = txt[h + 1]
                body = txt[h + 2:e]
                stop = next((i for i, l in enumerate(body) if l.startswith("NIR (")), len(body))
                body = body[:stop]
                m = re.search(r"SIMD(\d+) shader: (\d+) instructions\. (\d+) loops\..*? (\d+):(\d+) spills:fills, (\d+) sends", st)
                h64 = re.search(r"src_hash (0x[0-9a-f]+)", txt[h])[1]
                key = (h64, m.groups(), sum("dpas" in l for l in body))
                if key in seen or not any("dpas" in l for l in body): continue
                seen.add(key); blocks.append((h64, m.groups(), body))
        else:
            j = json.load(open(path.replace(".compute.GEN_Assembly.txt", ".json")))
            ex = j["records"][0]["executables"][0]; s = {x["name"]: x["value"] for x in ex["statistics"]}
            simd = re.search(r"SIMD(\d+)", ex["description"])[1]
            blocks.append((j["shader"], (simd, s["Instructions"], s["Loops"], s["Spills"], s["Fills"], s["SENDs"]), txt))
        for name, st, body in blocks:
            rows.append([label, name, *st, "whole shader", *fmt(count(body, 0, len(body) - 1))])
            for n, (a, b) in enumerate(loops(body)):
                c = count(body, a, b)
                rows.append([label, name, *st, f"loop {n} (lines {a + 1}-{b + 1})", *fmt(c)])
    csv.writer(open(sys.argv[1], "w", newline="")).writerows(rows)
    for r in rows: print(",".join(str(x) for x in r))
