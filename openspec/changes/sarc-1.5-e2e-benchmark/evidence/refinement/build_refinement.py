#!/usr/bin/env python3
"""Build refinement.csv: previous WMMA winner kernels vs the 2026-09-26/27 re-tuned kernels.

Read-only over existing artifacts; no GPU access. Re-run with `python3 build_refinement.py`.

Per-shape values are medians over 3 repeats of each run's per-case kernel median (GPU timestamps),
as stored in each run's COMPARE.json (Jetson: out/jetson-study/confirm/summary.json + comparison.json).
Derived model rows: sum(before us) / sum(after us) over the model's 4 projection shapes with
(a) equal weight per projection type, and (b) per-layer call counts 2 wq_wo, 2 wk_wv, 2 w1_w3, 1 w2
(the weighting used by roofline-et-study/e2e/analyze.py; the layer count cancels in the ratio).
"""

import csv
import json
import math
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent
ART = Path("/home/doremy/Desktop/sarc-acl/.artifacts/roofline-et-study")
DOCS = Path("/home/doremy/Desktop/igpu-roofline/docs")
JET = Path("/home/doremy/Desktop/igpu-roofline/out/jetson-study")

PER_LAYER = {"wq_wo": 2, "wk_wv": 2, "w1_w3": 2, "w2": 1}
LAYERS = {"llama-3.2-1b": 16, "llama-3.2-3b": 28, "llama-3.1-8b": 32}
MODELS = ["llama-3.2-1b", "llama-3.2-3b", "llama-3.1-8b"]
OPS = ["wq_wo", "wk_wv", "w1_w3", "w2"]
E2E_GPU = {"780M": "780m", "B580": "b580", "B70": "b70-0", "4070TiS": "4070tis"}
E2E_SIZE = {"llama-3.2-1b": "1b", "llama-3.2-3b": "3b", "llama-3.1-8b": "8b"}

PAIRS = [  # gpu, before run, after run, note
    ("780M", ART / "runs/780m-branch/780m", ART / "confirm/780m-final",
     "before = -780m branch default before 9178cee44 (coopmat already default on 780M)"),
    ("B580", ART / "runs/b70-branch/b580", ART / "confirm/b580-default-v1",
     "before = -b70 branch Xe2 tiles run on B580 with ET_VK_TEXTURE_COOPMAT=1 (opt-in; model path ran tiled by default); texture3d only"),
    ("B70", ART / "runs/b70-branch/b70-0", ART / "confirm/b70-default-v1",
     "before = -b70 branch Xe2 tiles with ET_VK_TEXTURE_COOPMAT=1 (opt-in; model path ran tiled by default); texture3d only"),
    ("4070TiS", ART / "runs/4070ti-branch", ART / "runs/4070ti-final3",
     "before = -4070ti branch kernels forced on with ET_VK_COOPMAT_ANY_DEVICE=1 ET_VK_TEXTURE_COOPMAT=1 (default path was tiled); after = defaults incl. ACC_GROUP_FP32 accuracy fix"),
]

COLS = ["gpu", "scheme", "storage", "model", "shape", "before_kernel", "after_kernel", "before_value",
        "after_value", "unit", "speedup_after_over_before", "n_repeats", "source_path", "source_line",
        "data_kind", "before_spread", "after_spread", "notes"]


def rel(p):
    return str(p).replace("/home/doremy/Desktop/", "")


def md_lines(path):
    """(model, layer, scheme, storage) -> line number in a COMPARE.md table."""
    out = {}
    for i, line in enumerate(Path(path).read_text().splitlines(), 1):
        if line.startswith("| llama"):
            f = [x.strip() for x in line.split("|")]
            out[(f[1], f[2], f[3], f[4])] = i
    return out


def shapes():
    """(model, op) -> (M, K, N) from a microbench JSON."""
    s = {}
    for c in json.loads((ART / "confirm/780m-final/linear-r1.json").read_text())["cases"]:
        if c["regime"] == "prefill":
            s[(c["model"], c["op"])] = (c["M"], c["K"], c["N"])
    return s


SHAPE = shapes()


def shape_str(model, op):
    m, k, n = SHAPE[(model, op)]
    return f"{op} (M={m},N={n},K={k})"


def gm(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs))


def fmt(x, nd=2):
    return "" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


rows = []
cells = {}  # (gpu, scheme, storage, model, op) -> (before_us, after_us)


def add(**kw):
    rows.append({c: kw.get(c, "") for c in COLS})


# ---------------------------------------------------------------- per-shape microbench rows
for gpu, bdir, adir, note in PAIRS:
    B = {tuple(c["key"]): c for c in json.loads((bdir / "COMPARE.json").read_text())}
    A = {tuple(c["key"]): c for c in json.loads((adir / "COMPARE.json").read_text())}
    bl, al = md_lines(bdir / "COMPARE.md"), md_lines(adir / "COMPARE.md")
    for scheme in ("4w", "8da4w"):
        for storage in ("texture3d", "buffer"):
            for model in MODELS:
                for op in OPS:
                    k = (model, op, scheme, storage)
                    aw = (A.get(k) or {}).get("wmma") or {}
                    bw = (B.get(k) or {}).get("wmma") or {}
                    if not aw.get("us"):
                        continue
                    bu, au = bw.get("us"), aw["us"]
                    n = f"{bw.get('n', 0)}/{aw['n']}"
                    extra = note
                    if not bu:
                        extra = ("no before: the -b70 branch shipped texture3d-only Xe2 variants; buffer dispatch "
                                 "crashed ('Could not find ShaderInfo', runs/b70-branch-buffer-crash)")
                    else:
                        cells[(gpu, scheme, storage, model, op)] = (bu, au)
                    add(gpu=gpu, scheme=scheme, storage=storage, model=model, shape=shape_str(model, op),
                        before_kernel=";".join(bw.get("kernels", [])), after_kernel=";".join(aw["kernels"]),
                        before_value=fmt(bu, 1) if bu else "", after_value=fmt(au, 1), unit="us",
                        speedup_after_over_before=fmt(bu / au, 3) if bu else "", n_repeats=n,
                        source_path=f"before: {rel(bdir / 'COMPARE.md')} | after: {rel(adir / 'COMPARE.md')} (values from sibling COMPARE.json)",
                        source_line=f"before L{bl.get(k, '-')} | after L{al.get(k, '-')}",
                        data_kind="microbench kernel", before_spread=fmt(bw.get("spread"), 4) if bu else "",
                        after_spread=fmt(aw.get("spread"), 4), notes=extra)

# ---------------------------------------------------------------- Jetson per-shape rows
jcmp = json.loads((JET / "confirm/comparison.json").read_text())
jsum = json.loads((JET / "confirm/summary.json").read_text())
jtext = (JET / "confirm/comparison.json").read_text().splitlines()
jline = {}
for i, line in enumerate(jtext):
    if line.strip().startswith('"key"'):
        toks = "".join(jtext[i:i + 10])
        m = re.findall(r'"([^"]+)"', toks.split("]")[0])
        jline[(m[1], m[2], m[4])] = i + 1  # (model, scheme, op) -> line of "key"
jspread = {k: {tuple(x["key"]): x for x in jsum[k]} for k in ("original", "tuned")}
jk = {}
for r in ("original-r1", "tuned-r1"):
    for c in json.loads((JET / f"confirm/{r}/perf.json").read_text())["cases"]:
        if c.get("regime") == "prefill":
            jk[(r, c["model"], c["scheme"], c["op"])] = re.search(r'"kernel_name": "([^"]+)"', c["kernel"]).group(1) \
                if "kernel_name" in c["kernel"] else c["kernel"]
for e in jcmp:
    model, scheme, _, op, storage = e["key"][:5]
    k = tuple(e["key"])
    bu, au = e["original_us"], e["tuned_us"]
    valid = e["original_numerically_valid"]
    if valid:
        cells[("Orin", scheme, storage, model, op)] = (bu, au)
    add(gpu="Orin", scheme=scheme, storage=storage, model=model, shape=shape_str(model, op),
        before_kernel=jk[("original-r1", model, scheme, op)], after_kernel=jk[("tuned-r1", model, scheme, op)],
        before_value=fmt(bu, 1), after_value=fmt(au, 1), unit="us",
        speedup_after_over_before=fmt(bu / au, 3) if valid else "",
        n_repeats=f"{jspread['original'][k]['repeats']}/{jspread['tuned'][k]['repeats']}",
        source_path=f"{rel(JET / 'confirm/comparison.json')} (spreads: confirm/summary.json; kernels: confirm/*-r1/perf.json)",
        source_line=f"L{jline[(model, scheme, op)]}", data_kind="microbench kernel",
        before_spread=fmt(jspread["original"][k]["spread"], 4), after_spread=fmt(jspread["tuned"][k]["spread"], 4),
        notes=("original (before) is numerically invalid on this shape (fp16 accumulation fails the 8B w2 K=14336 "
               "sampled reference); tuned uses the fp32 K32/g42 repair; excluded from speedups"
               if not valid else "before = -4070ti branch (0270403ba) kernels explicitly enabled on Orin (not the Orin default); "
               f"whole-operator speedup {e['operator_speedup_vs_original']:.3f}"))

# ---------------------------------------------------------------- derived: geomean + time-weighted per model
e2e = {(r["gpu"], r["size"], r["quant"]): r for r in
       json.loads((ART / "e2e/e2e_summary.json").read_text())}
jmodel = {(r["model"], r["scheme"]): r for r in json.loads((JET / "analysis/final-model-comparison.json").read_text())}
summary = []  # for the markdown
for gpu in ["780M", "B580", "B70", "4070TiS", "Orin"]:
    for scheme in ("4w", "8da4w"):
        for storage in ("texture3d", "buffer"):
            ks = [k for k in cells if k[:3] == (gpu, scheme, storage)]
            if not ks:
                continue
            rs = [cells[k][0] / cells[k][1] for k in ks]
            g = gm(rs)
            add(gpu=gpu, scheme=scheme, storage=storage, model="all (1B/3B/8B)",
                shape=f"geomean over {len(ks)} shapes (equal weight per shape)",
                unit="ratio", speedup_after_over_before=fmt(g, 3), n_repeats="3/3",
                source_path="derived from the per-shape rows above", data_kind="derived",
                notes=f"min {min(rs):.3f} max {max(rs):.3f}")
            for model in MODELS:
                mk = [(op, cells[(gpu, scheme, storage, model, op)]) for op in OPS if (gpu, scheme, storage, model, op) in cells]
                if not mk:
                    continue
                ops = [o for o, _ in mk]
                sb = sum(v[0] for _, v in mk)
                sa = sum(v[1] for _, v in mk)
                wb = sum(PER_LAYER[o] * v[0] for o, v in mk)
                wa = sum(PER_LAYER[o] * v[1] for o, v in mk)
                miss = [o for o in OPS if o not in ops]
                mnote = (f"excludes {','.join(miss)} (invalid original)" if miss else "")
                add(gpu=gpu, scheme=scheme, storage=storage, model=model,
                    shape=f"time-weighted, equal weight per projection ({'+'.join(ops)})",
                    before_value=fmt(sb, 1), after_value=fmt(sa, 1), unit="us (sum)",
                    speedup_after_over_before=fmt(sb / sa, 3), n_repeats="3/3",
                    source_path="derived from the per-shape rows above", data_kind="derived", notes=mnote)
                L = LAYERS[model]
                add(gpu=gpu, scheme=scheme, storage=storage, model=model,
                    shape=f"time-weighted, per-layer call counts 2*wq_wo+2*wk_wv+2*w1_w3+1*w2 x {L} layers",
                    before_value=fmt(L * wb / 1000, 2), after_value=fmt(L * wa / 1000, 2), unit="ms (linear time per 2048-token prefill)",
                    speedup_after_over_before=fmt(wb / wa, 3), n_repeats="3/3",
                    source_path=f"derived; weighting as in {rel(ART / 'e2e/analyze.py')}", source_line="L24",
                    data_kind="derived", notes=mnote)
                summary.append(dict(gpu=gpu, scheme=scheme, storage=storage, model=model, eq=sb / sa, pl=wb / wa,
                                    lin_before_ms=L * wb / 1000, lin_after_ms=L * wa / 1000, miss=miss))
                # end-to-end projection (texture3d = model path)
                if storage != "texture3d" or miss:
                    continue
                if gpu in E2E_GPU:
                    r = e2e.get((E2E_GPU[gpu], E2E_SIZE[model], scheme))
                    if r:
                        tuned_ms = r["tuned"]["prefill_ms"]
                        proj_before = tuned_ms + L * (wb - wa) / 1000
                        add(gpu=gpu, scheme=scheme, storage=storage, model=model,
                            shape="e2e prefill 2048 tokens: Amdahl projection",
                            before_kernel="previous winner (projected)", after_kernel="tuned (measured e2e)",
                            before_value=fmt(proj_before, 1), after_value=fmt(float(tuned_ms), 1), unit="ms",
                            speedup_after_over_before=fmt(proj_before / tuned_ms, 3), n_repeats=f"e2e {r['n'][1]}",
                            source_path=f"{rel(ART / 'e2e/e2e_summary.json')} + per-shape rows",
                            data_kind="derived (e2e projection)",
                            notes="projected before = measured tuned e2e prefill_ms + layers*sum(count*(before-after)); "
                                  "not measured; release-1.4 branches")
                if gpu == "Orin":
                    jm = jmodel.get((model.replace("llama-", "llama").replace(".", "_"), scheme))
                    if jm:
                        p = jm["p2048"]
                        proj_before = p["tuned_ms"] + L * (wb - wa) / 1000
                        add(gpu=gpu, scheme=scheme, storage=storage, model=model,
                            shape="e2e prefill 2048 tokens: Amdahl projection (check vs measured)",
                            before_kernel="original WMMA (projected)", after_kernel="tuned (measured e2e)",
                            before_value=fmt(proj_before, 1), after_value=fmt(float(p["tuned_ms"]), 1), unit="ms",
                            speedup_after_over_before=fmt(proj_before / p["tuned_ms"], 3), n_repeats=f"e2e {p['repeats']}",
                            source_path=f"{rel(JET / 'analysis/final-model-comparison.json')} + per-shape rows",
                            data_kind="derived (e2e projection)",
                            notes=f"measured e2e speedup is {p['speedup_vs_original']:.3f}; compare")

# ---------------------------------------------------------------- Jetson measured e2e rows
jdoc = (DOCS / "JETSON-WMMA-LESSONS.md").read_text().splitlines()
for (m, s), r in jmodel.items():
    model = {"llama3_2-1b": "llama-3.2-1b", "llama3_2-3b": "llama-3.2-3b"}[m]
    ln = next(i for i, l in enumerate(jdoc, 1) if l.startswith(f"| {model[-2:].upper()} | {s} |"))
    for p, lab in (("p256", "256"), ("p2048", "2048")):
        x = r[p]
        add(gpu="Orin", scheme=s, storage="texture3d", model=model, shape=f"e2e prefill, {lab}-token prompt (warm)",
            before_kernel="original WMMA (0270403ba kernels, enabled on Orin)", after_kernel="tuned defaults (0b260ffab)",
            before_value=fmt(float(x["original_ms"]), 0), after_value=fmt(float(x["tuned_ms"]), 0), unit="ms",
            speedup_after_over_before=fmt(x["speedup_vs_original"], 3), n_repeats=f"{x['repeats']}/{x['repeats']}",
            source_path=f"{rel(DOCS / 'JETSON-WMMA-LESSONS.md')} (raw: {rel(JET / 'analysis/final-model-comparison.json')})",
            source_line=f"L{ln}", data_kind="e2e",
            notes=f"source {x['source']}; 8B not measured end to end (memory guard)")

# ---------------------------------------------------------------- doc summary rows
D = [
    ("780M", "4w", "texture3d", "780M-WMMA-LESSONS.md", 26, "1.30", "vs previous 780M default"),
    ("780M", "4w", "buffer", "780M-WMMA-LESSONS.md", 27, "1.32", "vs previous 780M default"),
    ("780M", "8da4w", "texture3d", "780M-WMMA-LESSONS.md", 28, "1.00", "unchanged kernel"),
    ("B580", "4w", "texture3d", "XE2-WMMA-LESSONS.md", 39, "", "doc gives only WMMA/tiled for the -b70 tiles on B580 (2.9x 4w, 1.55x 8da4w, opt-in); no before/after ratio stated"),
    ("B70", "4w", "texture3d", "XE2-WMMA-LESSONS.md", 115, "2.54", "vs previous -b70 WMMA"),
    ("B70", "8da4w", "texture3d", "XE2-WMMA-LESSONS.md", 116, "1.69", "vs previous -b70 WMMA"),
    ("4070TiS", "8da4w", "texture3d", "4070TI-WMMA-LESSONS.md", 38, "1.34", "range 1.26-1.47"),
    ("4070TiS", "8da4w", "buffer", "4070TI-WMMA-LESSONS.md", 39, "2.12", "range 2.04-2.28"),
    ("4070TiS", "4w", "texture3d", "4070TI-WMMA-LESSONS.md", 40, "0.96", "1.04x tiles then 0.92x accuracy fix (product 0.957)"),
    ("4070TiS", "4w", "buffer", "4070TI-WMMA-LESSONS.md", 41, "0.98", "1.13x tiles then 0.87x accuracy fix (product 0.983; raw 0.973)"),
    ("Orin", "4w", "texture3d", "JETSON-WMMA-LESSONS.md", 194, "1.066", "11 valid shapes; whole-operator 1.065"),
    ("Orin", "8da4w", "texture3d", "JETSON-WMMA-LESSONS.md", 195, "2.324", "12 shapes; whole-operator 2.235"),
]
for gpu, s, st, doc, ln, v, n in D:
    add(gpu=gpu, scheme=s, storage=st, model="all (1B/3B/8B)", shape="doc geomean over shapes", unit="ratio",
        speedup_after_over_before=v, n_repeats="3/3", source_path=f"igpu-roofline/docs/{doc}", source_line=f"L{ln}",
        data_kind="doc summary", notes=n)

# 4070 Ti intermediate steps (raw), so the 0.96x is traceable
for lab, b, a in (("step 1 tiles (branch -> final2)", "runs/4070ti-branch", "runs/4070ti-final2"),
                  ("step 2 accuracy fix (final2 -> final3)", "runs/4070ti-final2", "runs/4070ti-final3")):
    B = {tuple(c["key"]): c for c in json.loads((ART / b / "COMPARE.json").read_text())}
    A = {tuple(c["key"]): c for c in json.loads((ART / a / "COMPARE.json").read_text())}
    for st in ("texture3d", "buffer"):
        rs = [B[k]["wmma"]["us"] / A[k]["wmma"]["us"] for k in A if k[2] == "4w" and k[3] == st]
        add(gpu="4070TiS", scheme="4w", storage=st, model="all (1B/3B/8B)", shape=f"geomean over 12 shapes, {lab}",
            unit="ratio", speedup_after_over_before=fmt(gm(rs), 3), n_repeats="3/3",
            source_path=f"{rel(ART / b / 'COMPARE.json')} -> {rel(ART / a / 'COMPARE.json')}", data_kind="derived",
            notes=f"min {min(rs):.3f} max {max(rs):.3f}")

with open(OUT / "refinement.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    w.writerows(rows)
(OUT / "model_summary.json").write_text(json.dumps(summary, indent=1))
print(len(rows), "rows")
for s in summary:
    print(f"{s['gpu']:8s}{s['scheme']:6s}{s['storage']:10s}{s['model']:13s} eq {s['eq']:.3f} per-layer {s['pl']:.3f} "
          f"lin {s['lin_before_ms']:.1f}->{s['lin_after_ms']:.1f} ms {s['miss'] or ''}")
