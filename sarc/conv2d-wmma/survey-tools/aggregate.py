"""Aggregate the Vulkan model survey: raw ETDumps -> CSVs -> REPORT.md and models/MODELS.md.

Stage 1 (needs the ExecuTorch devtools, run from the workspace's `main/` directory so `executorch` is the
2efa9d0316 source tree):
    cd main && PYTHONDONTWRITEBYTECODE=1 executorch/.venv/bin/python ../.artifacts/2026-11-19-vulkan-model-survey/aggregate.py
reads OUT/models/*.json + *-export.exit + references and OUT/<device>/results/*, and writes
    models/models.csv, survey-runs.csv, survey-dispatches.csv, survey-shaders.csv, survey-conv-shapes.csv
then runs stage 2.

Stage 2 (plain Python 3, reads only the CSVs above):
    python3 OUT/aggregate.py --report-only
writes REPORT.md and models/MODELS.md. Every number in them is computed from the CSVs.
"""
import argparse
import array
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEVICES = {
    "b580": "Arc B580",
    "b70": "Arc Pro B70",
    "rocky-780m": "Radeon 780M",
    "4070ti": "RTX 4070 Ti SUPER",
    "pixel7a": "Pixel 7a (Mali-G710)",
    "galaxy-s24plus": "Galaxy S24+ (Xclipse 940)",
    "m51": "M51",
}
SHORT = {"b580": "B580", "b70": "B70", "rocky-780m": "780M", "4070ti": "4070TiS", "pixel7a": "Pixel7a",
         "galaxy-s24plus": "S24+", "m51": "M51"}
MODELS = ["mv2", "mv3", "resnet18", "resnet50", "ic3", "ic4", "edsr", "dl3", "w2l", "densenet161"]
CLASSIFIERS = {"mv2", "mv3", "resnet18", "resnet50", "ic3", "ic4", "densenet161"}
PRECISIONS = ["fp32", "fp16", "int8-q8ta"]
CATEGORIES = ["convolution", "linear/matmul", "elementwise", "layout/data movement", "pooling", "other"]
WMMA = 16  # cooperative-matrix tile edge (M, N and K step) used for the candidate arithmetic
KEPT = 11  # executions kept per combination (12 run, the first discarded)
FLAG_PCT = 50.0  # heuristic: GPU shader time below this % of the host graph time -> timestamps flagged (‡)
FLOOR_US = 1.0  # dispatch medians at or below this are counted as "near the timer floor"


# ----------------------------------------------------------------------------------------------------------
# Shader classification (by kernel name, i.e. by what was actually dispatched)
# ----------------------------------------------------------------------------------------------------------
def category(kernel):
    k = kernel
    if k.startswith(("conv", "q8ta_conv2d", "q8ta_im2col")):
        return "convolution"
    if k.startswith("upsample"):  # bilinear/nearest spatial interpolation (impl/Upsample.cpp), not a GEMM
        return "other"
    if k.startswith(("linear", "q8ta_linear")) or "matmul" in k or "addmm" in k or k.startswith(("mm_", "bmm")):
        return "linear/matmul"
    if k.startswith(("max_pool", "avg_pool", "mean2d", "adaptive")):
        return "pooling"
    if k.startswith(("binary_op", "unary_op", "q8ta_binary", "batchnorm", "clamp", "hardswish")):
        return "elementwise"
    if k.startswith(("nchw_to_", "image_to_nchw", "buffer_to_nchw", "clone", "view", "permute", "concat",
                     "set_zero", "update_concat_offset", "select", "q8ta_clone", "q8ta_quantize",
                     "q8ta_dequantize", "pixel_shuffle", "q8ta_pixel_shuffle", "copy", "slice", "repeat",
                     "pad", "transfer")):
        return "layout/data movement"
    return "other"


def routing(kernels):
    """Routing class of one convolution node from the set of kernels it dispatched."""
    ks = " ".join(kernels)
    if "winograd" in ks:
        return "winograd"
    if "im2col" in ks:
        return "im2col+gemm"
    if "transpose" in ks:
        return "transposed"
    if all(k.startswith("conv1d") for k in kernels):
        return "1d-pointwise" if all(k.startswith("conv1d_pw") for k in kernels) else "1d"
    if all("_pw" in k for k in kernels):
        return "pointwise"
    if all("_dw" in k for k in kernels):
        return "depthwise"
    return "direct"


def conv_config(op):
    """Logical convolution parameters from the ETDump operator metadata of a convolution dispatch."""
    name, args = op["name"], op["args"]
    tensors = [a for a in args if a["type"] == "TENSOR"]
    inp, out = tensors[0], tensors[-1]
    if name.startswith("et_vk.q8ta_conv2d"):
        k, s, p, d = (args[i]["values"] for i in (9, 10, 11, 12))
        g = args[13]["value"]
        tr = False
        cin, cout = inp["sizes"][1], out["sizes"][1]
        weight = "x".join(map(str, args[3]["sizes"]))  # weight TENSORREF as traced (reshaped/packed for q8ta)
    else:  # aten.convolution, et_vk.conv2d.with_epilogue, et_vk.conv_with_clamp
        w = args[1]["sizes"]
        s, p, d = (args[i]["values"] for i in (3, 4, 5))
        tr = bool(args[6]["value"])
        g = args[8]["value"]
        k = w[2:]
        cin = inp["sizes"][1]
        cout = w[1] * g if tr else w[0]
        weight = "x".join(map(str, w))
    dims = len(k)
    # logical filter dimensions [Cout, Cin/groups, k...] (transposed: [Cin, Cout/groups, k...])
    logical = ([cin, cout // g] if tr else [cout, cin // g]) + list(k)
    if dims == 1:  # conv1d: treat as H = 1
        k, s, p, d = [1] + k, [1] + s, [0] + p, [1] + d
        hin, win = 1, inp["sizes"][2]
        hout, wout = 1, out["sizes"][2]
    else:
        hin, win = inp["sizes"][2:4]
        hout, wout = out["sizes"][2:4]
    return {
        "op": name, "dims": dims, "batch": inp["sizes"][0], "cin": cin, "hin": hin, "win": win,
        "cout": cout, "kh": k[0], "kw": k[1], "stride": f"{s[0]}x{s[1]}", "padding": f"{p[0]}x{p[1]}",
        "dilation": f"{d[0]}x{d[1]}", "groups": g, "transposed": int(tr), "hout": hout, "wout": wout,
        "weight_ref_sizes": weight, "weight_logical_sizes": "x".join(map(str, logical)), "dtype": inp.get("dtype", ""), "op_in_storage": inp.get("storage", ""),
        "op_out_storage": out.get("storage", ""), "out_dtype": out.get("dtype", ""),
        "in_sizes": "x".join(map(str, inp["sizes"])), "out_sizes": "x".join(map(str, out["sizes"])),
    }


def gemm(c):
    """GEMM-equivalent shape and WMMA arithmetic of a convolution configuration (dict of ints)."""
    m = c["batch"] * c["hout"] * c["wout"]
    n = c["cout"]
    k = c["cin"] * c["kh"] * c["kw"] // c["groups"]
    macs = m * n * k
    params = n * k
    if c["transposed"]:
        verdict = "no (transposed)"
    elif c["groups"] > 1 and c["groups"] == c["cin"]:
        verdict = "no (depthwise)"
    elif c["groups"] > 1:
        verdict = "no (grouped)"
    elif c["cout"] < WMMA:
        verdict = "low-util (Cout<16)"  # runs in a padded tile; < Cout/16 of the N columns are useful
    elif m < WMMA:
        verdict = "low-util (M<16)"  # runs in a padded tile; < M/16 of the M rows are useful
    elif c["cin"] < WMMA:
        verdict = "im2col-only (Cin<16)" if k >= WMMA else "low-util (K<16)"
    else:
        verdict = "yes"
    ksteps = -(-k // WMMA)
    tiles = -(-m // WMMA) * -(-n // WMMA)
    return {"gemm_m": m, "gemm_n": n, "gemm_k": k, "macs": macs, "weight_params": params,
            "wmma": verdict, "k_steps": ksteps, "out_tiles": tiles,
            # K / (16 * K steps): useful share of all K steps; last step: (K - 16 * (K steps - 1)) / 16
            "k_fill_all_steps_pct": round(100 * k / (ksteps * WMMA), 1),
            "k_fill_last_step_pct": round(100 * (k - WMMA * (ksteps - 1)) / WMMA, 1),
            # M / (16 * ceil(M/16)) and N / (16 * ceil(N/16)): useful rows / columns of the output tiles
            "m_tile_fill_pct": round(100 * m / (WMMA * -(-m // WMMA)), 1),
            "n_tile_fill_pct": round(100 * n / (WMMA * -(-n // WMMA)), 1)}


# ----------------------------------------------------------------------------------------------------------
# Stage 1: ETDumps -> CSVs
# ----------------------------------------------------------------------------------------------------------
def read_floats(path):
    values = array.array("f")
    values.frombytes(path.read_bytes())
    return values


def p90(samples):
    return statistics.quantiles(samples, n=10, method="inclusive")[-1]


def parse_etdump(path):
    from executorch.devtools.etdump.serialize import deserialize_from_etdump_flatcc

    dump = deserialize_from_etdump_flatcc(path.read_bytes())
    runs = []
    for block in dump.run_data:
        if block.name != "Execute":
            continue
        wall, graph, dispatches = [], 0, []
        for event in block.events or []:
            prof = event.profile_event
            if not prof:
                continue
            ns = prof.end_time - prof.start_time
            label = prof.delegate_debug_id_str or ""
            if prof.name == "Method::execute":
                wall.append(ns)
            elif label == "ETVK_COMPUTE_GRAPH_EXECUTE":
                graph += ns
            elif label.startswith("{"):
                meta = json.loads(label)
                if "kernel_name" in meta:
                    dispatches.append((meta["dispatch_id"], meta["kernel_name"], ns, meta, prof.start_time,
                                       prof.end_time))
        if len(wall) != 1:
            raise ValueError(f"{path}: {len(wall)} Method::execute events in a block")
        runs.append((wall[0], graph, dispatches))
    return runs


def check_outputs(folder, name, model, precision):
    def indexed(pattern):
        files = {int(f.stem.rsplit("-", 1)[1]): f for f in pattern}
        assert sorted(files) == list(range(len(files))), sorted(files)
        return [files[i] for i in range(len(files))]

    refs = indexed((ROOT / "models").glob(f"{model}-fp32-reference-*.bin"))
    res: dict = {"outputs": len(refs)}
    outs = indexed(folder.glob(f"{name}-output-*.bin"))
    if len(outs) != len(refs):
        res["check"] = f"{len(outs)} outputs written, {len(refs)} in the reference"
        return res
    pairs = [(read_floats(o), read_floats(f)) for o, f in zip(outs, refs)]
    if any(len(x) != len(y) for x, y in pairs):
        res["check"] = "output size differs from reference"
        return res
    # max abs difference and finiteness over all outputs; argmax/top-1 checks on output 0
    res["finite"] = int(all(math.isfinite(v) for x, _ in pairs for v in x))
    res["max_abs_diff"] = max(abs(v - w) for x, y in pairs for v, w in zip(x, y))
    res["ref_max_abs"] = max(abs(w) for _, y in pairs for w in y)
    res["max_abs_diff_per_output"] = ";".join(f"{max(abs(v - w) for v, w in zip(x, y)):.6f}" for x, y in pairs)
    res["ref_min"] = min(w for _, y in pairs for w in y)
    res["ref_max"] = max(w for _, y in pairs for w in y)
    a, r = pairs[0]
    if model in CLASSIFIERS:
        top = max(range(len(a)), key=a.__getitem__)
        ref_top = max(range(len(r)), key=r.__getitem__)
        ref_top5 = sorted(range(len(r)), key=r.__getitem__)[-5:]
        res.update(top1=top, ref_top1=ref_top, top1_match=int(top == ref_top), top1_in_ref_top5=int(top in ref_top5))
        res["check"] = "top-1 match" if top == ref_top else "top-1 differs"
    elif model in ("dl3", "w2l"):
        # argmax over the class (dl3: dim 1 of NCHW) or vocabulary (w2l: dim 1 of [N, V, T]) axis
        shape = {"dl3": (1, 21, 224 * 224), "w2l": (10, 4096, 3)}[model]
        n, c, hw = shape
        agree = total = 0
        for b in range(n):
            base = b * c * hw
            for j in range(hw):
                ia = max(range(c), key=lambda ch: a[base + ch * hw + j])
                ir = max(range(c), key=lambda ch: r[base + ch * hw + j])
                agree += ia == ir
                total += 1
        res["argmax_agree_pct"] = 100 * agree / total
        res["check"] = "max-abs + argmax agreement"
    else:
        res["check"] = "max-abs"
    if precision == "int8-q8ta":
        q = indexed((ROOT / "models").glob(f"{model}-int8-q8ta-reference-*.bin"))
        qs = [read_floats(f) for f in q]
        assert len(qs) == len(pairs) and all(len(x) == len(y) for (x, _), y in zip(pairs, qs)), (name, len(qs))
        # max over all corresponding outputs (and per output, for the record)
        per = [max(abs(v - w) for v, w in zip(x, y)) for (x, _), y in zip(pairs, qs)]
        res["max_abs_diff_vs_int8_cpu"] = max(per)
        res["max_abs_diff_vs_int8_cpu_per_output"] = ";".join(f"{v:.6f}" for v in per)
        if model in CLASSIFIERS:
            res["int8_cpu_top1"] = max(range(len(qs[0])), key=qs[0].__getitem__)
    return res


def fmt(x, nd=3):
    return "" if x is None else round(x, nd)


def stage1():
    models_rows = []
    export_status = {}
    for m in MODELS:
        for p in PRECISIONS:
            info_path = ROOT / "models" / f"{m}-{p}.json"
            exit_path = ROOT / "models" / f"{m}-{p}-export.exit"
            log_path = ROOT / "models" / f"{m}-{p}-export.log"
            rc = exit_path.read_text().strip() if exit_path.exists() else "missing"
            row = {"model": m, "precision": p, "export_exit": rc}
            if rc == "0" and info_path.exists():
                info = json.loads(info_path.read_text())
                ndo = {k: v for k, v in info["non_delegated_ops"].items() if k != "getitem"}
                row.update(source=info["source"], input_shapes=";".join("x".join(map(str, s)) for s in info["input_shapes"]),
                           output_shapes=";".join("x".join(map(str, s)) for s in info["output_shapes"]),
                           parameters=info["parameters"], pte_bytes=info["pte_bytes"],
                           delegated_nodes=info["delegated_nodes"], non_delegated_nodes=info["non_delegated_nodes"],
                           non_delegated_getitem=info["non_delegated_ops"].get("getitem", 0),
                           non_delegated_other=";".join(f"{k}={v}" for k, v in ndo.items()),
                           delegated_subgraphs=info["delegated_subgraphs"], export_error="")
                export_status[m, p] = None
            else:
                err = ""
                if log_path.exists():
                    lines = [l for l in log_path.read_text().splitlines() if "Error" in l or "error" in l]
                    err = lines[0].strip() if lines else ""
                    lines = log_path.read_text().splitlines()
                    for i, l in enumerate(lines):
                        if l.startswith(("ValueError", "RuntimeError", "AssertionError", "TypeError", "KeyError")):
                            err = l.strip()
                            # innermost frame of the first traceback: where the export failed
                            frames = [x for x in lines[:i] if x.startswith('  File "')]
                            if frames:
                                f = frames[-1].split('"')
                                where = f[1].split("/executorch/", 1)[-1]
                                line_no = f[2].split(",")[1].strip().replace("line ", "")
                                func = f[2].split(" in ")[-1].strip()
                                err += f" (at {where}:{line_no}, {func})"
                            break
                row["export_error"] = err
                export_status[m, p] = f"export failed: {err}"
            models_rows.append(row)
    write_csv(ROOT / "models" / "models.csv", models_rows)
    io_counts = {(r["model"], r["precision"]): (len(r["input_shapes"].split(";")), len(r["output_shapes"].split(";")))
                 for r in models_rows if r["export_exit"] == "0"}

    runs_rows, dispatch_rows, shader_rows, conv_nodes = [], [], [], []
    for dev in DEVICES:
        folder = ROOT / dev / "results"
        for m in MODELS:
            for p in PRECISIONS:
                name = f"{m}-{p}"
                row = {"device": dev, "model": m, "precision": p, "status": "", "reason": ""}
                runs_rows.append(row)
                if export_status[m, p]:
                    row.update(status="not run", reason=export_status[m, p])
                    continue
                exit_path, dump_path = folder / f"{name}.exit", folder / f"{name}.etdp"
                if not exit_path.exists():
                    row.update(status="not run", reason="no run recorded")
                    continue
                rc = exit_path.read_text().strip()
                if rc != "0":
                    row.update(status="failed", reason=f"runner exit {rc}")
                    continue
                if not dump_path.exists():
                    row.update(status="failed", reason="no ETDump")
                    continue
                runs = parse_etdump(dump_path)
                if len(runs) != KEPT + 1:
                    row.update(status="failed", reason=f"{len(runs)} executions in ETDump")
                    continue
                kept = runs[1:]
                seq = [(d[0], d[1]) for d in kept[0][2]]
                assert all([(d[0], d[1]) for d in r[2]] == seq for r in kept), (dev, name, "dispatch sequence varies")
                assert len({i for i, _ in seq}) == len(seq), (dev, name, "duplicate dispatch ids")
                walls = [r[0] for r in kept]
                totals = [sum(d[2] for d in r[2]) for r in kept]
                spans = [max(d[5] for d in r[2]) - min(d[4] for d in r[2]) for r in kept]
                med = []
                for j, (did, kernel) in enumerate(seq):
                    samples = [r[2][j][2] for r in kept]
                    if min(samples) < 0:
                        raise ValueError(f"negative GPU timestamp delta {dev} {name} {kernel}")
                    med.append(statistics.median(samples))
                gpu_sum = sum(med)
                cats = Counter()
                shader_time, shader_count = defaultdict(float), Counter()
                nodes = defaultdict(list)
                # Boundary staging: the delegate copies each graph input into the graph before the first
                # operator (nchw_to_*) and each output out after the last (image_to_nchw_* / buffer_to_nchw_*).
                # In the ETDump the input copies carry no operator JSON and the output copies inherit the last
                # operator's JSON (VulkanBackend.cpp:678-702), so their graph operator and node are left empty
                # here; the raw labels stay in raw_etdump_operator / raw_etdump_node_id.
                n_in, n_out = io_counts[m, p]
                lead = next((i for i, (_, k) in enumerate(seq) if not k.startswith("nchw_to_")), len(seq))
                trail = next((i for i, (_, k) in enumerate(reversed(seq))
                              if not k.startswith(("image_to_nchw", "buffer_to_nchw"))), len(seq))
                assert (lead, trail) == (n_in, n_out), (dev, name, lead, trail, n_in, n_out)
                for j, (did, kernel) in enumerate(seq):
                    meta = kept[0][2][j][3]
                    stage = ("input staging" if j < n_in else
                             "output staging" if j >= len(seq) - n_out else "graph")
                    cat = category(kernel)
                    cats[cat] += med[j]
                    shader_time[kernel] += med[j]
                    shader_count[kernel] += 1
                    op = meta.get("operator") or {}
                    graph_op = stage == "graph"
                    d = {"device": dev, "model": m, "precision": p, "dispatch_id": did, "shader": kernel,
                         "category": cat, "stage": stage, "operator": op.get("name", "") if graph_op else "",
                         "node_id": op.get("delegate_node_id", "") if graph_op else "",
                         "raw_etdump_operator": op.get("name", ""),
                         "raw_etdump_node_id": op.get("delegate_node_id", ""),
                         "median_gpu_us": fmt(med[j] / 1e3),
                         "global_wg": "x".join(map(str, meta.get("global_workgroup", []))),
                         "local_wg": "x".join(map(str, meta.get("local_workgroup", [])))}
                    if cat == "convolution":
                        assert graph_op, (dev, name, did, kernel)
                        c = conv_config(op)
                        d.update({k: c[k] for k in ("in_sizes", "weight_ref_sizes", "weight_logical_sizes", "kh", "kw", "stride", "padding",
                                                     "dilation", "groups", "transposed", "out_sizes", "dtype",
                                                     "op_in_storage", "op_out_storage")})
                        nodes[op["delegate_node_id"]].append((kernel, med[j], c))
                    dispatch_rows.append(d)
                for kernel in sorted(shader_time, key=lambda k: -shader_time[k]):
                    shader_rows.append({"device": dev, "model": m, "precision": p, "shader": kernel,
                                        "category": category(kernel), "dispatches": shader_count[kernel],
                                        "total_median_gpu_us": fmt(shader_time[kernel] / 1e3),
                                        "share_pct": fmt(100 * shader_time[kernel] / gpu_sum, 3)})
                for nid, parts in nodes.items():
                    c = parts[0][2]
                    assert all(x[2] == c for x in parts), (dev, name, nid)
                    conv_nodes.append((dev, m, p, nid, c, [x[0] for x in parts], [x[1] for x in parts], gpu_sum))
                chk = check_outputs(folder, name, m, p)
                row.update(
                    status="ok", measured_runs=len(kept),
                    wall_median_ms=fmt(statistics.median(walls) / 1e6, 4), wall_p90_ms=fmt(p90(walls) / 1e6, 4),
                    vk_graph_host_median_ms=fmt(statistics.median(r[1] for r in kept) / 1e6, 4),
                    gpu_shader_ms=fmt(gpu_sum / 1e6, 4), gpu_total_median_ms=fmt(statistics.median(totals) / 1e6, 4),
                    gpu_span_median_ms=fmt(statistics.median(spans) / 1e6, 4),
                    gpu_vs_graph_pct=fmt(100 * gpu_sum / statistics.median(r[1] for r in kept), 1),
                    timestamps_flagged=int(100 * gpu_sum / statistics.median(r[1] for r in kept) < FLAG_PCT),
                    dispatches_le_1us=sum(1 for x in med if x <= FLOOR_US * 1e3),
                    dispatches=len(seq), distinct_shaders=len(shader_time),
                    conv_dispatches=sum(1 for _, k in seq if category(k) == "convolution"),
                    conv_nodes=len(nodes),
                    **{f"share_{c.split('/')[0]}_pct": fmt(100 * cats[c] / gpu_sum, 2) for c in CATEGORIES},
                    check=chk.get("check", ""), outputs=chk.get("outputs", ""), finite=chk.get("finite", ""),
                    top1=chk.get("top1", ""), ref_top1=chk.get("ref_top1", ""), top1_match=chk.get("top1_match", ""),
                    top1_in_ref_top5=chk.get("top1_in_ref_top5", ""),
                    max_abs_diff=fmt(chk.get("max_abs_diff"), 6), ref_max_abs=fmt(chk.get("ref_max_abs"), 6),
                    ref_min=fmt(chk.get("ref_min"), 6), ref_max=fmt(chk.get("ref_max"), 6),
                    argmax_agree_pct=fmt(chk.get("argmax_agree_pct"), 3),
                    int8_cpu_top1=chk.get("int8_cpu_top1", ""),
                    max_abs_diff_vs_int8_cpu=fmt(chk.get("max_abs_diff_vs_int8_cpu"), 6),
                    max_abs_diff_per_output=chk.get("max_abs_diff_per_output", ""),
                    max_abs_diff_vs_int8_cpu_per_output=chk.get("max_abs_diff_vs_int8_cpu_per_output", ""),
                )
    write_csv(ROOT / "survey-runs.csv", runs_rows)
    write_csv(ROOT / "survey-shaders.csv", shader_rows)
    flagged = {(r["device"], r["model"], r["precision"]) for r in runs_rows if r.get("timestamps_flagged") == 1}

    # Distinct convolution configurations per (model, precision), deduplicated across devices.
    shape_keys = ("dims", "batch", "cin", "hin", "win", "cout", "kh", "kw", "stride", "padding", "dilation",
                  "groups", "transposed", "hout", "wout", "dtype", "op_in_storage", "op_out_storage")
    configs = {}
    for dev, m, p, nid, c, kernels, times, gpu_sum in conv_nodes:
        key = (m, p) + tuple(c[k] for k in shape_keys)
        e = configs.setdefault(key, {"c": c, "nodes": defaultdict(set), "dev": defaultdict(list)})
        e["nodes"][dev].add(nid)
        e["dev"][dev].append((kernels, sum(times), gpu_sum))
        e.setdefault("stages", defaultdict(list))[dev].append(list(zip(kernels, times)))
        e.setdefault("node_ids", []).append((dev, m, p, nid))
        e.setdefault("ops", defaultdict(Counter))[dev][c["op"]] += 1
    conv_rows = []
    for key, e in configs.items():
        m, p = key[:2]
        c = e["c"]
        counts = {len(v) for v in e["nodes"].values()}
        assert len(counts) == 1, (key, dict(e["nodes"]))
        opv = {tuple(sorted(v.items())) for v in e["ops"].values()}
        assert len(opv) == 1, (key, dict(e["ops"]))
        row = {"model": m, "precision": p, "instances": counts.pop(),
               "op_variants": ";".join(f"{o}={n}" for o, n in opv.pop()),
               **{k: c[k] for k in ("dims", "batch", "cin", "hin", "win", "cout", "kh", "kw", "stride",
                                    "padding", "dilation", "groups", "transposed", "hout", "wout",
                                    "weight_ref_sizes", "weight_logical_sizes", "dtype", "op_in_storage", "op_out_storage")}}
        row.update(gemm(c))
        routes = {}
        for dev in DEVICES:
            if dev not in e["dev"]:
                row.update({f"{dev}_routing": "", f"{dev}_shaders": "", f"{dev}_us_per_instance": "",
                            f"{dev}_total_us": "", f"{dev}_share_pct": ""})
                continue
            items = e["dev"][dev]
            seqs = sorted({"+".join(k) for k, _, _ in items})
            routes[dev] = sorted({routing(k) for k, _, _ in items})
            total = sum(t for _, t, _ in items)
            row.update({f"{dev}_routing": "|".join(routes[dev]), f"{dev}_shaders": "|".join(seqs),
                        f"{dev}_us_per_instance": fmt(total / len(items) / 1e3),
                        f"{dev}_total_us": fmt(total / 1e3),
                        f"{dev}_share_pct": fmt(100 * total / items[0][2], 3)})
        allr = {r for v in routes.values() for r in v}
        row["routing"] = allr.pop() if len(allr) == 1 else "mixed"
        # per device, per shader stage: dispatch counts and per-dispatch medians of this configuration
        stage_rows = []
        for dev in DEVICES:
            nodes = e["stages"].get(dev)
            if not nodes:
                row[f"{dev}_stages"] = ""
                continue
            per = defaultdict(lambda: [[], []])  # shader -> [per-dispatch us], [dispatches per node]
            for node in nodes:
                cnt = Counter()
                for k, t in node:
                    per[k][0].append(t / 1e3)
                    cnt[k] += 1
                for k in cnt:
                    per[k][1].append(cnt[k])
            gsum = e["dev"][dev][0][2]
            parts = []
            for k in sorted(per, key=lambda k: -sum(per[k][0])):
                us, npn = per[k]
                assert len(npn) == len(nodes) and len(set(npn)) == 1, (key, dev, k, npn)
                parts.append(f"{k} {npn[0]}x{statistics.mean(us):.3f}us")
                stage_rows.append({
                    "device": dev, "shader": k, "instances": len(nodes), "dispatches": len(us),
                    "dispatches_per_instance": npn[0], "us_per_dispatch": fmt(statistics.mean(us)),
                    "min_us_per_dispatch": fmt(min(us)), "max_us_per_dispatch": fmt(max(us)),
                    "us_per_instance": fmt(sum(us) / len(nodes)), "total_us": fmt(sum(us)),
                    "share_pct": fmt(100 * sum(us) * 1e3 / gsum, 3),
                    "timestamps_flagged": int((dev, m, p) in flagged)})
            row[f"{dev}_stages"] = "; ".join(parts)
        row["_stages"] = stage_rows
        row["_nodes"] = e["node_ids"]
        conv_rows.append(row)
    order = {m: i for i, m in enumerate(MODELS)}
    porder = {p: i for i, p in enumerate(PRECISIONS)}
    conv_rows.sort(key=lambda r: (order[r["model"]], porder[r["precision"]], -r["macs"] * r["instances"]))
    stage_csv, node_cfg, seen = [], {}, Counter()
    for r in conv_rows:
        seen[r["model"], r["precision"]] += 1
        r["config_id"] = f"{r['model']}/{r['precision']}/{seen[r['model'], r['precision']]}"
        for x in r.pop("_stages"):
            stage_csv.append({"config_id": r["config_id"], "model": r["model"], "precision": r["precision"], **x})
        for n in r.pop("_nodes"):
            node_cfg[n] = r["config_id"]
    for d in dispatch_rows:
        if d["category"] == "convolution":
            d["config_id"] = node_cfg[d["device"], d["model"], d["precision"], d["node_id"]]
    write_csv(ROOT / "survey-dispatches.csv", dispatch_rows)
    write_csv(ROOT / "survey-conv-shape-shaders.csv", stage_csv)
    first = ["config_id", "model", "precision", "instances", "routing", "op_variants", "dims", "batch", "cin", "hin", "win", "cout",
             "kh", "kw", "stride", "padding", "dilation", "groups", "transposed", "hout", "wout", "weight_ref_sizes", "weight_logical_sizes",
             "dtype", "op_in_storage", "op_out_storage", "gemm_m", "gemm_n", "gemm_k", "macs", "weight_params",
             "wmma", "k_steps", "out_tiles", "k_fill_all_steps_pct", "k_fill_last_step_pct", "m_tile_fill_pct",
             "n_tile_fill_pct"]
    fields = first + [f"{d}_{s}" for d in DEVICES for s in ("routing", "shaders", "stages", "us_per_instance",
                                                             "total_us", "share_pct")]
    write_csv(ROOT / "survey-conv-shapes.csv", conv_rows, fields)
    print(f"stage 1: {sum(r['status'] == 'ok' for r in runs_rows)}/{len(runs_rows)} ok, {len(dispatch_rows)} "
          f"dispatch rows, {len(shader_rows)} shader rows, {len(conv_rows)} conv configurations")


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


# ----------------------------------------------------------------------------------------------------------
# Stage 2: CSVs -> REPORT.md, models/MODELS.md
# ----------------------------------------------------------------------------------------------------------
def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def num(x):
    return float(x) if x not in ("", None) else None


def f3(x):
    return f"{x:.3f}"


def pct(x):
    return f"{x:.1f}%"


def table(header, rows, align=None):
    align = align or ["---"] + ["---:"] * (len(header) - 1)
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def strip_dtype(kernel):
    for t in ("_half", "_float", "_int8"):
        kernel = kernel.replace(t, "")
    return kernel




def compress(seq):
    """'a+b+a+b' -> '(a+b) x2' for streamed im2col tiles."""
    parts = seq.split("+")
    for w in range(1, len(parts) // 2 + 1):
        if len(parts) % w == 0 and parts == parts[:w] * (len(parts) // w):
            unit = "+".join(parts[:w])
            n = len(parts) // w
            return unit if n == 1 else f"({unit}) x{n}"
    return seq


def shape_label(c):
    return (f"{c['kh']}x{c['kw']} s{c['stride']} p{c['padding']}"
            + (f" d{c['dilation']}" if c["dilation"] != "1x1" else "")
            + (f" g{c['groups']}" if c["groups"] != "1" else ""))


def stage2():
    mrows = read_csv(ROOT / "models" / "models.csv")
    runs = read_csv(ROOT / "survey-runs.csv")
    shaders = read_csv(ROOT / "survey-shaders.csv")
    convs = read_csv(ROOT / "survey-conv-shapes.csv")
    disp = read_csv(ROOT / "survey-dispatches.csv")
    write_models_md(mrows)
    run_idx = {(r["device"], r["model"], r["precision"]): r for r in runs}
    ok = [r for r in runs if r["status"] == "ok"]
    flagged = {(r["device"], r["model"], r["precision"]) for r in ok if float(r["gpu_vs_graph_pct"]) < FLAG_PCT}
    trusted = lambda d, m, p: run_idx[d, m, p]["status"] == "ok" and (d, m, p) not in flagged
    mark = lambda d, m, p: " ‡" if (d, m, p) in flagged else ""
    L = []
    L += ["# ExecuTorch Vulkan vision-model survey: where the GPU time goes on six GPUs", ""]
    L += [
        "Generated by `aggregate.py` from the raw ETDumps in `<device>/results/`: stage 1 writes the CSVs, "
        "stage 2 writes this file from the CSVs only (`python3 aggregate.py --report-only`). ExecuTorch "
        "upstream `main` at `2efa9d0316` for both export and the traced `executor_runner`. Each combination: "
        "one fixed input (the model's default example input, saved in `models/`), 12 executions, the first "
        "discarded, 11 kept. Commands are in `LOG.md`.",
        "",
        "Definitions used throughout:",
        "- *Inference time*: ETDump `Method::execute`, median and p90 over the 11 kept executions (host clock; "
        "includes input copy, command submission and the wait for the GPU).",
        "- *GPU shader time*: sum over all Vulkan dispatches of each dispatch's median GPU timestamp duration over "
        "the 11 kept executions (`survey-runs.csv:gpu_shader_ms`). All shares are of this sum, so the shares of a "
        "combination add up to 100%. `gpu_total_median_ms` (median of the per-execution sums) is also in the CSV.",
        f"- ‡ marks a combination whose GPU shader time is below {FLAG_PCT:.0f}% of the host-side Vulkan graph "
        "span (`ETVK_COMPUTE_GRAPH_EXECUTE`; `survey-runs.csv:gpu_vs_graph_pct`, column `timestamps_flagged`). "
        "The host span includes command submission and the fence wait, so a low ratio shows a gap between "
        "what the GPU timestamps account for and what the host waited, without establishing its cause (host "
        "overhead, idle GPU time between or around dispatches, or timestamp behaviour). The cutoff is a "
        "heuristic chosen for this survey. ‡ cells are included in every table, ranking and verdict; a "
        "sensitivity view that leaves them out follows the main view wherever it changes a result "
        "(sections 3, 4 and 5).",
        "- *Storage*: `op_in_storage` / `op_out_storage` in the CSVs are the operator-level metadata of the "
        "convolution's input and output tensors as the ETDump records them. The trace does not expose the "
        "storage of prepacked weights, biases or intermediates (im2col scratch, Winograd tiles); only the "
        "kernel-name suffix hints at it (`_bufw` buffer weights, `_texture2d` / `_buffer` GEMM operands). "
        "`weight_ref_sizes` is the weight TENSORREF shape as traced (for q8ta INT8 the reshaped/packed "
        "weight, e.g. `3x576`); `weight_logical_sizes` is the logical filter [Cout, Cin/groups, kh, kw] "
        "([Cout, Cin/groups, k] for conv1d).",
        "- *Shader*: the exact kernel name dispatched (with its dtype/storage suffix). *Category* is assigned "
        "from the kernel name (`aggregate.py:category`); q8ta quantize/dequantize and pixel-shuffle count as "
        "layout/data movement, batch-norm as elementwise.",
        "- *Boundary staging*: `survey-dispatches.csv:stage` marks the delegate's input copies (`input staging`, "
        "the leading `nchw_to_*` dispatches) and output copies (`output staging`, the trailing "
        "`image_to_nchw_*` / `buffer_to_nchw_*`), one per graph input/output (checked). They belong to no graph "
        "operator, so `operator` and `node_id` are empty for them; the ETDump's own labels, which for output "
        "staging repeat the last operator (`VulkanBackend.cpp:678-702`), are kept in `raw_etdump_operator` / "
        "`raw_etdump_node_id`.",
        "- *Routing class* of a convolution is taken from the kernels that node actually dispatched "
        "(`aggregate.py:routing`): `winograd` (conv2d_winograd_*), `im2col+gemm` (conv2d_im2col + conv2d_gemm, "
        "or q8ta_im2col + q8ta_conv2d_pw), `pointwise` (conv2d_pw_*, q8ta_conv2d_pw*), `depthwise` "
        "(conv2d_dw_*, q8ta_conv2d_dw), `1d`/`1d-pointwise` (conv1d*), `transposed`, otherwise `direct` "
        "(sliding window: conv2d_2x2x8_border, conv2d_1x1x4, q8ta_conv2d, ...). An im2col+gemm node streams "
        "its output rows in tiles and dispatches one (im2col, GEMM) pair per tile (`Conv2dGemm.cpp:539`, "
        "`Q8taConv2dIm2Col.cpp:441`); `(a+b) xN` below means N tiles.",
        f"- *GEMM-equivalent* of a convolution: M = batch x Hout x Wout, N = Cout, K = Cin x kh x kw / groups. "
        f"K steps = ceil(K/{WMMA}), output tiles = ceil(M/{WMMA}) x ceil(N/{WMMA}).",
        "",
    ]

    # ---------------- 1. Coverage
    L += ["## 1. Coverage", ""]
    L += [f"**{len(ok)} of {len(runs)}** (model, precision, device) combinations have results "
          f"({len(MODELS)} models x {len(PRECISIONS)} precisions x {len(DEVICES)} devices); "
          f"{len(flagged)} of them are ‡ (timestamps flagged by the {FLAG_PCT:.0f}% heuristic; evidence in the timestamp check below).", ""]
    rows = []
    for m in MODELS:
        for p in PRECISIONS:
            rows.append([m, p] + [("ok" + mark(d, m, p)) if run_idx[d, m, p]["status"] == "ok"
                                  else run_idx[d, m, p]["status"] for d in DEVICES])
    L += table(["model", "precision"] + [SHORT[d] for d in DEVICES], rows,
               ["---", "---"] + [":---:"] * len(DEVICES)) + [""]
    fails = [r for r in runs if r["status"] != "ok"]
    L += ["Combinations without results and why:", ""]
    if not fails:
        L.append("- none")
    grouped = defaultdict(list)
    for r in fails:
        grouped[(r["model"], r["precision"], r["status"], r["reason"])].append(SHORT[r["device"]])
    for (m, p, s, reason), devs in grouped.items():
        L.append(f"- `{m}` {p} on {', '.join(devs)}: {s} -- {reason}")
        if "make_q8ta_conv2d_custom_op" in reason:
            L.append("  w2l is built from Conv1d layers; the Vulkan q8ta convolution fusion pattern unpacks a 4-D "
                     "(OC, IC, H, W) weight, and a quantized Conv1d weight is 3-D, so the INT8 export fails before "
                     "a PTE exists. Making it work needs an ExecuTorch change (out of scope), so nothing ran.")
    L.append("")
    L += ["### Output check against the FP32 CPU reference", "",
          "Classifiers: device top-1 vs CPU eager FP32 top-1 on the same input, then the max abs logit "
          "difference. dl3 and w2l: max abs difference / share of pixels (dl3) or frames (w2l) whose argmax "
          "class matches. edsr: max abs pixel difference. INT8 is compared with the FP32 reference; "
          "the INT8 table below also compares it with the CPU fake-quantized model. All max abs differences are "
          "the maximum over all outputs of the model (per output in `survey-runs.csv:max_abs_diff_per_output` "
          "and `max_abs_diff_vs_int8_cpu_per_output`); output counts and sizes are checked first.", ""]
    rows = []
    for m in MODELS:
        for p in PRECISIONS:
            cells = []
            for d in DEVICES:
                r = run_idx[d, m, p]
                if r["status"] != "ok":
                    cells.append("--")
                elif m in CLASSIFIERS:
                    cells.append(("top1 ok" if r["top1_match"] == "1" else f"top1 {r['top1']}!={r['ref_top1']}")
                                 + f" ({float(r['max_abs_diff']):.3g})")
                elif r["argmax_agree_pct"]:
                    cells.append(f"{float(r['max_abs_diff']):.3g} / {float(r['argmax_agree_pct']):.1f}%")
                else:
                    cells.append(f"{float(r['max_abs_diff']):.3g}")
            rows.append([m, p] + cells)
    L += table(["model", "precision"] + [SHORT[d] for d in DEVICES], rows, ["---"] * (2 + len(DEVICES))) + [""]
    mism = [r for r in ok if r["top1_match"] == "0"]
    if mism:
        by = defaultdict(list)
        for r in mism:
            by[r["model"], r["precision"], r["top1"], r["ref_top1"], r["top1_in_ref_top5"]].append(SHORT[r["device"]])
        L.append("Top-1 mismatches: " + "; ".join(
            f"{m} {p} on {', '.join(devs)} (device {t} vs reference {rt}; device class in reference top-5: "
            f"{'yes' if t5 == '1' else 'no'})" for (m, p, t, rt, t5), devs in by.items()) + ". "
            "The same wrong class on every device points at FP16 rounding of a close race on this random "
            "input, not at a device fault.")
        L.append("")
    L += ["INT8 outputs: max abs difference to the FP32 CPU reference and to the CPU fake-quantized "
          "(`convert_pt2e`) model, as the range over the devices that ran it (per output for multi-output "
          "models: output 0 / output 1, each its range).", ""]
    rows = []
    for m in MODELS:
        rs = [run_idx[d, m, "int8-q8ta"] for d in DEVICES if run_idx[d, m, "int8-q8ta"]["status"] == "ok"]
        if not rs:
            continue
        def rng(col):
            per = [list(map(float, r[col].split(";"))) for r in rs]
            return " / ".join(f"{min(x[i] for x in per):.6f}" + (f"-{max(x[i] for x in per):.6f}"
                                                                 if min(x[i] for x in per) != max(x[i] for x in per)
                                                                 else "") for i in range(len(per[0])))
        rows.append([m, len(rs), rng("max_abs_diff_per_output"), rng("max_abs_diff_vs_int8_cpu_per_output"),
                     "/".join(sorted({r["top1_match"] for r in rs})) if m in CLASSIFIERS else "--"])
    L += table(["model", "devices", "vs FP32 CPU", "vs CPU fake-quantized", "top-1 = FP32"], rows,
               ["---", "---:", "---:", "---:", "---"]) + [""]
    w = [run_idx[d, "w2l", "fp16"] for d in DEVICES if run_idx[d, "w2l", "fp16"]["status"] == "ok"]
    if w:
        r0 = w[0]
        span = float(r0["ref_max"]) - float(r0["ref_min"])
        step = 2.0 ** (math.floor(math.log2(max(abs(float(r0["ref_min"])), abs(float(r0["ref_max"]))))) - 10)
        agree = sorted({float(x["argmax_agree_pct"]) for x in w})
        L += [f"w2l (torchaudio `Wav2Letter`, untrained, as ExecuTorch defines it): the FP32 reference "
              f"log-probabilities lie in [{float(r0['ref_min']):.6f}, {float(r0['ref_max']):.6f}], a span of "
              f"{span:.6f}, about {span / step:.1f} FP16 steps ({step:.7f} at that magnitude). FP16 max abs "
              "difference " + "-".join(sorted({f"{float(x['max_abs_diff']):.6f}" for x in w})) + " "
              f"(half an FP16 step is {step / 2:.7f}); FP16 argmax agreement "
              + "/".join(f"{a:.1f}%" for a in agree) + " on every device. FP32 argmax agreement: "
              + "/".join(f"{a:.1f}%" for a in sorted({float(run_idx[d, 'w2l', 'fp32']['argmax_agree_pct'])
                                                       for d in DEVICES if run_idx[d, 'w2l', 'fp32']['status'] == 'ok'})) + ".", ""]
    # timestamp flag evidence
    L += ["### Timestamp check (‡)", "",
          f"Cells with GPU shader time below {FLAG_PCT:.0f}% of the host graph span, and the lowest cells above "
          "the cutoff. `GPU ms` = `gpu_shader_ms` (sum of per-dispatch medians, the basis of the flag and of all "
          "shares); `per-run sum` = `gpu_total_median_ms` (median over the kept executions of each execution's "
          "shader sum); `GPU span` = median over the kept executions of last dispatch end minus first dispatch "
          "start, on the GPU timestamp clock; `host graph` is on the host clock. Last column: "
          f"dispatches whose median is at or below {FLOOR_US:.0f} us.", ""]
    srt = sorted(ok, key=lambda r: float(r["gpu_vs_graph_pct"]))
    shown = [r for r in srt if float(r["gpu_vs_graph_pct"]) < FLAG_PCT] + \
            [r for r in srt if float(r["gpu_vs_graph_pct"]) >= FLAG_PCT][:5]
    rows = [[SHORT[r["device"]], r["model"], r["precision"], f"{float(r['gpu_shader_ms']):.3f}",
             f"{float(r['gpu_total_median_ms']):.3f}", f"{float(r['gpu_span_median_ms']):.3f}",
             f"{100 * float(r['gpu_total_median_ms']) / float(r['gpu_span_median_ms']):.1f}%",
             f"{float(r['vk_graph_host_median_ms']):.3f}", f"{float(r['gpu_vs_graph_pct']):.1f}%",
             f"{r['dispatches_le_1us']}/{r['dispatches']}", "‡" if (r["device"], r["model"], r["precision"]) in flagged
             else ""] for r in shown]
    L += table(["device", "model", "precision", "GPU ms", "per-run sum ms", "GPU span ms", "per-run sum / span",
                "host graph ms", "GPU / graph", f"disp. <= {FLOOR_US:.0f} us", "flag"], rows,
               ["---"] * 3 + ["---:"] * 7 + [":---:"]) + [""]
    fl = [r for r in ok if (r["device"], r["model"], r["precision"]) in flagged]
    maj = [r for r in fl if 2 * int(r["dispatches_le_1us"]) > int(r["dispatches"])]
    mino = [r for r in fl if 2 * int(r["dispatches_le_1us"]) <= int(r["dispatches"])]
    lbl = lambda rs: ", ".join(f"{r['model']} {r['precision']}" for r in rs) or "none"
    L += [f"In {len(maj)} of the {len(fl)} flagged cells most dispatch medians are at or below {FLOOR_US:.0f} us "
          f"({lbl(maj)}); in the other {len(mino)} only a minority are ({lbl(mino)}), yet the GPU sum is still "
          "below the cutoff. In the flagged cells the median over executions of the per-execution shader sum "
          "(`gpu_total_median_ms`) is "
          f"{min(100 * float(r['gpu_total_median_ms']) / float(r['gpu_span_median_ms']) for r in fl):.1f}-"
          f"{max(100 * float(r['gpu_total_median_ms']) / float(r['gpu_span_median_ms']) for r in fl):.1f}% of the "
          "median GPU span (the sum of per-dispatch medians, `gpu_shader_ms`, used for the shares and rankings, "
          "is not a per-execution quantity and is not compared with the span), and the GPU span is "
          f"{min(100 * float(r['gpu_span_median_ms']) / float(r['vk_graph_host_median_ms']) for r in fl):.0f}-"
          f"{max(100 * float(r['gpu_span_median_ms']) / float(r['vk_graph_host_median_ms']) for r in fl):.0f}% "
          "of the host graph span. Which clock or which intervals explain the difference is not established by these "
          "traces; the flag marks the discrepancy, not a diagnosis, and it does not say that any particular "
          "dispatch is mis-timed.", "",
          f"Unflagged cells can also contain dispatch medians at or below {FLOOR_US:.0f} us, so such medians alone "
          "do not decide the flag:", ""]
    rows = []
    for d in DEVICES:
        rs = [r for r in ok if r["device"] == d and (d, r["model"], r["precision"]) not in flagged]
        if not rs:
            continue
        hit = [r for r in rs if int(r["dispatches_le_1us"]) > 0]
        rows.append([SHORT[d], len(rs), len(hit), sum(int(r["dispatches_le_1us"]) for r in rs),
                     sum(int(r["dispatches"]) for r in rs),
                     f"{min(float(r['gpu_vs_graph_pct']) for r in rs):.1f}%",
                     "; ".join(f"{r['model']} {r['precision']} {r['dispatches_le_1us']}/{r['dispatches']}"
                               for r in sorted(hit, key=lambda r: -int(r["dispatches_le_1us"]))[:3])])
    L += table(["device", "unflagged cells", f"cells with disp. <= {FLOOR_US:.0f} us", "such disp.",
                "all disp.", "lowest GPU / graph", "largest counts"], rows,
               ["---"] + ["---:"] * 5 + ["---"]) + [""]

    # ---------------- 2. Per model
    L += ["## 2. Per model and precision", "",
          "Inference = `Method::execute` median / p90 (ms). GPU = GPU shader time (ms). Disp. = Vulkan dispatches "
          "per inference. Top shaders by share of the GPU shader time (dispatch count in parentheses).", ""]
    sh_idx = defaultdict(list)
    for s in shaders:
        sh_idx[s["device"], s["model"], s["precision"]].append(s)
    for m in MODELS:
        L += [f"### {m}", ""]
        rows = []
        for p in PRECISIONS:
            for d in DEVICES:
                r = run_idx[d, m, p]
                if r["status"] != "ok":
                    rows.append([p, SHORT[d], "--", "--", "--", r["reason"]])
                    continue
                top = sorted(sh_idx[d, m, p], key=lambda s: -float(s["share_pct"]))[:3]
                tops = "; ".join(f"`{s['shader']}` {float(s['share_pct']):.1f}% (x{s['dispatches']})" for s in top)
                rows.append([p, SHORT[d], f"{float(r['wall_median_ms']):.3f} / {float(r['wall_p90_ms']):.3f}",
                             f"{float(r['gpu_shader_ms']):.3f}{mark(d, m, p)}", r["dispatches"], tops])
        L += table(["precision", "device", "inference ms", "GPU ms", "disp.", "top shaders (share, dispatches)"],
                   rows, ["---", "---", "---:", "---:", "---:", "---"]) + [""]

    # ---------------- 3. Cross-model ranking
    incl_all = lambda d, m, p: run_idx[d, m, p]["status"] == "ok"
    flag_devs = sorted({d for d, _, _ in flagged}, key=list(DEVICES).index)

    def sums(incl, d, p, key):
        tot, cnt = Counter(), Counter()
        for s_ in shaders:
            if s_["device"] == d and s_["precision"] == p and incl(d, s_["model"], p):
                tot[s_[key]] += float(s_["total_median_gpu_us"])
                cnt[s_[key]] += int(s_["dispatches"])
        return tot, cnt

    def ranking_section(incl, devs, view):
        out = []
        rows = []
        for d in devs:
            for p in PRECISIONS:
                tot, _ = sums(incl, d, p, "category")
                allt = sum(tot.values())
                if allt:
                    rows.append([SHORT[d], p, sum(incl(d, m, p) for m in MODELS), f"{allt / 1e3:.2f}"]
                                + [pct(100 * tot[c] / allt) for c in CATEGORIES])
        out += [f"Category shares of the GPU shader time summed over models ({view}):", ""]
        out += table(["device", "precision", "models", "sum GPU ms"] + CATEGORIES, rows,
                     ["---", "---"] + ["---:"] * (2 + len(CATEGORIES))) + [""]
        rows = []
        keys = [f"share_{c.split('/')[0]}_pct" for c in CATEGORIES]
        for d in devs:
            for p in PRECISIONS:
                rs = [r for r in ok if r["device"] == d and r["precision"] == p and incl(d, r["model"], p)]
                if rs:
                    rows.append([SHORT[d], p, len(rs)] + [pct(statistics.mean(float(r[k]) for r in rs)) for k in keys])
        out += [f"Mean over models of each model's category share ({view}):", ""]
        out += table(["device", "precision", "models"] + CATEGORIES, rows,
                     ["---", "---"] + ["---:"] * (1 + len(CATEGORIES))) + [""]
        for p in PRECISIONS:
            per_dev = {d: sums(incl, d, p, "shader")[0] for d in DEVICES}
            totals = {d: sum(v.values()) for d, v in per_dev.items()}
            names = set().union(*(set(v) for v in per_dev.values()))
            dv = [d for d in DEVICES if totals[d]]
            score = {n: statistics.mean(100 * per_dev[d][n] / totals[d] for d in dv) for n in names}
            rows = []
            for n in sorted(names, key=lambda n: (-score[n], n))[:15]:
                rows.append([f"`{n}`", category(n), pct(score[n])]
                            + [f"{per_dev[d][n] / 1e3:.2f} ({100 * per_dev[d][n] / totals[d]:.1f}%)"
                               if per_dev[d][n] else "--" for d in DEVICES])
            out += [f"Fleet ranking, {p} ({view}): the 15 shaders with the largest mean over devices of the "
                    "shader's share of the device's summed GPU shader time.", ""]
            out += table(["shader", "category", "mean share"] + [f"{SHORT[d]} ms (share)" for d in DEVICES], rows,
                         ["---", "---"] + ["---:"] * (1 + len(DEVICES))) + [""]
        for d in devs:
            for p in PRECISIONS:
                tot, cnt = sums(incl, d, p, "shader")
                allt = sum(tot.values())
                if not allt:
                    continue
                nm = sum(incl(d, m, p) for m in MODELS)
                out += [f"**{DEVICES[d]}, {p}** ({view}; {nm} models, {allt / 1e3:.2f} ms):", ""]
                rows = [[i + 1, f"`{n}`", category(n), f"{tot[n] / 1e3:.3f}", pct(100 * tot[n] / allt), cnt[n]]
                        for i, n in enumerate(sorted(tot, key=lambda n: (-tot[n], n))[:15])]
                out += table(["rank", "shader", "category", "ms", "share", "dispatches"], rows,
                             ["---:", "---", "---", "---:", "---:", "---:"]) + [""]
        return out

    L += ["## 3. Cross-model shader ranking", "",
          "GPU shader time summed over all models with results on the device, per precision (larger models "
          "weigh more), from all runs including ‡ cells. Category shares, the mean of the per-model category "
          "shares (each model weighs the same), a fleet ranking per precision, then per-device rankings in each "
          "device's own order (top 15 shaders by that device's summed time; share of that sum).", ""]
    L += ["### Main view: all runs", ""] + ranking_section(incl_all, list(DEVICES), "all runs")
    L += ["### Sensitivity view: ‡ cells excluded", "",
          "Only " + ", ".join(DEVICES[d] for d in flag_devs) + " has ‡ cells, so its rows and rankings are the "
          "only ones that change; the fleet rankings change through it.", ""]
    L += ranking_section(trusted, flag_devs, "‡ excluded")

    # ---------------- 4. Top convolution shaders: shape tables
    dev_total = Counter()
    conv_tot = defaultdict(Counter)
    for s_ in shaders:
        dev_total[s_["device"]] += float(s_["total_median_gpu_us"])
        if s_["category"] == "convolution":
            conv_tot[s_["shader"]][s_["device"]] += float(s_["total_median_gpu_us"])
    cscore = {n: statistics.mean(100 * v[d] / dev_total[d] for d in DEVICES) for n, v in conv_tot.items()}
    why = defaultdict(list)
    for i, n in enumerate(sorted(cscore, key=lambda n: (-cscore[n], n))[:10]):
        why[n].append(f"fleet #{i + 1}")
    TOPDEV = 3
    for view, incl in (("", incl_all), ("‡-excl. ", trusted)):
        for d in DEVICES:
            for p in PRECISIONS:
                tot, _ = sums(incl, d, p, "shader")
                allt = sum(tot.values())
                conv = [n for n in sorted(tot, key=lambda n: (-tot[n], n)) if category(n) == "convolution"]
                for i, n in enumerate(conv[:TOPDEV]):
                    tag = f"{view}{SHORT[d]} {p} conv #{i + 1} ({100 * tot[n] / allt:.1f}%)"
                    if not (view and f"{SHORT[d]} {p} conv #{i + 1} ({100 * tot[n] / allt:.1f}%)" in why[n]):
                        why[n].append(tag)
    top_conv = sorted(why, key=lambda n: (-cscore[n], n))
    L += ["## 4. Convolution shaders and the shapes they run at", "",
          f"Selected shaders: the 10 convolution shaders with the largest mean (over devices) share of the GPU "
          f"shader time summed over all models and precisions (all runs), plus the top {TOPDEV} convolution "
          "shaders of every device and precision by that device's summed time, in the main view and in the "
          f"‡-excluded view ({len(top_conv)} shaders). Each heading says why the shader is included. For each, "
          "the complete list of convolution configurations (`config_id` of `survey-conv-shapes.csv`) at which "
          "any device dispatched it, ordered by the shader's share of the model's GPU shader time summed over "
          "devices. Device cells: `us x n` = mean of this shader's dispatch medians at this configuration (us "
          "per dispatch) and its dispatches per convolution instance (several for streamed im2col tiles); `‡` = "
          "the cell's timestamps are flagged; `--` = the device did not dispatch this shader for this "
          "configuration (what it dispatched instead is in `survey-conv-shapes.csv:<device>_stages` and in "
          "section 7). `node kernels` is the kernel sequence of one instance on the devices that used this "
          "shader, so the other stages of the same convolution (e.g. the GEMM after an im2col) can be told "
          "apart; their per-dispatch times are in that shader's own table and in "
          "`survey-conv-shape-shaders.csv`, which also has the min/max dispatch median.", ""]
    stages = read_csv(ROOT / "survey-conv-shape-shaders.csv")
    cfg_by_id = {c["config_id"]: c for c in convs}
    st = defaultdict(lambda: defaultdict(dict))  # shader -> config_id -> device -> stage row
    for x in stages:
        st[x["shader"]][x["config_id"]][x["device"]] = x
    for n in top_conv:
        L += [f"### `{n}` (fleet mean share {cscore[n]:.1f}%; " + "; ".join(why[n]) + ")", ""]
        items = []
        for cid, per in st[n].items():
            c = cfg_by_id[cid]
            key_t = sum(float(x["share_pct"]) for x in per.values())
            items.append((key_t, c, per))
        items.sort(key=lambda x: (-x[0], x[1]["config_id"]))
        rows = []
        for _, c, per in items:
            seqs = sorted({compress(c[f"{d}_shaders"]) for d in per})
            cells = []
            for d in DEVICES:
                x = per.get(d)
                cells.append("--" if x is None else
                             f"{float(x['us_per_dispatch']):.3f} x{x['dispatches_per_instance']}"
                             + (" ‡" if x["timestamps_flagged"] == "1" else ""))
            rows.append([c["config_id"], shape_label(c), c["cin"], c["cout"], f"{c['hin']}x{c['win']}",
                         f"{c['hout']}x{c['wout']}", c["instances"], f"({c['gemm_m']},{c['gemm_n']},{c['gemm_k']})",
                         c["wmma"], " / ".join(f"`{q}`" for q in seqs)] + cells)
        L += table(["config", "kernel/stride/pad", "Cin", "Cout", "in HxW", "out HxW", "inst.", "GEMM (M,N,K)",
                    "WMMA", "node kernels"] + [f"{SHORT[d]} us x n" for d in DEVICES], rows,
                   ["---", "---"] + ["---:"] * 5 + ["---", "---", "---"] + ["---:"] * len(DEVICES))
        L += ["", f"{len(items)} configurations for this shader.", ""]

    # ---------------- 5. WMMA candidates
    L += ["## 5. WMMA (cooperative-matrix GEMM) candidates", "",
          f"Verdicts per configuration (`survey-conv-shapes.csv:wmma`), with {WMMA}x{WMMA} output tiles and "
          f"{WMMA}-wide K steps:",
          f"- `yes`: groups = 1, not transposed, Cin >= {WMMA}, Cout >= {WMMA}, M >= {WMMA}. The GEMM-equivalent "
          f"fills K steps from the input channels alone (pointwise directly; k x k through implicit or explicit "
          "im2col) and has at least one full output tile.",
          f"- `im2col-only (Cin<16)`: groups = 1 stems such as 3 -> 32 3x3, whose K reaches {WMMA} only across "
          "the kernel window.",
          f"- `low-util (M<16)` / `low-util (Cout<16)`: tile-utilization criteria, not impossibility. Such a "
          f"GEMM runs in padded tiles, but at most M/{WMMA} (resp. Cout/{WMMA}) of each tile's rows (columns) are "
          "useful (`m_tile_fill_pct`, `n_tile_fill_pct`). Here (FP16 configurations): "
          + "; ".join(
              f"`{v}`: " + ", ".join(
                  f"{m} {sum(1 for c in convs if c['precision'] == 'fp16' and c['wmma'] == v and c['model'] == m)}"
                  for m in MODELS if any(c["precision"] == "fp16" and c["wmma"] == v and c["model"] == m
                                         for c in convs))
              + (" (M values: " if v.endswith("(M<16)") else " (Cout values: ") + ", ".join(sorted({
                  f"{c['gemm_m']}" if v.endswith("(M<16)") else f"{c['cout']}"
                                         for c in convs if c["precision"] == "fp16" and c["wmma"] == v},
                                        key=int)) + ")"
              for v in ("low-util (M<16)", "low-util (Cout<16)")) + ".",
          "- `no (depthwise)` / `no (grouped)`: groups > 1. Depthwise has no cross-channel reduction (per output "
          "channel K = kh x kw from one input channel), so as one GEMM its weight matrix is block-diagonal; it "
          "is not a dense GEMM, and a per-group GEMM has N = 1.",
          f"- Fill metrics: `k_fill_all_steps_pct` = K / ({WMMA} x K steps), the useful share of all K steps; "
          f"`k_fill_last_step_pct` = (K - {WMMA} x (K steps - 1)) / {WMMA}, the useful share of the last step "
          "(e.g. K = 24: 75% and 50%); `m_tile_fill_pct` / `n_tile_fill_pct` likewise for the M rows and N "
          "columns of the output tiles.",
          "",
          "Every configuration is in `survey-conv-shapes.csv` and listed per model at the end of this section.",
          ""]
    share = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))  # (m,p) -> dev -> [yes, im2col-only]
    for c in convs:
        for d in DEVICES:
            v = num(c[f"{d}_share_pct"])
            if v is None:
                continue
            if c["wmma"] == "yes":
                share[c["model"], c["precision"]][d][0] += v
            elif c["wmma"].startswith("im2col-only"):
                share[c["model"], c["precision"]][d][1] += v
    L += ["### Share of each model's GPU shader time held by candidate configurations", "",
          "Cell: share in `yes` configurations (in parentheses: plus `im2col-only`).", ""]
    rows, verdict = [], {}
    for m in MODELS:
        for p in PRECISIONS:
            cells, allv, unf = [], [], []
            for d in DEVICES:
                if run_idx[d, m, p]["status"] != "ok":
                    cells.append("--")
                    continue
                y, i2c = share[m, p][d]
                allv.append(y)
                if trusted(d, m, p):
                    unf.append(y)
                cells.append(f"{y:.1f}% ({y + i2c:.1f}%){mark(d, m, p)}")
            if allv:
                rows.append([m, p] + cells)
                verdict[m, p] = (allv, unf)
    L += table(["model", "precision"] + [SHORT[d] for d in DEVICES], rows) + [""]
    dom = [(m, p, min(a), max(a)) for (m, p), (a, _) in verdict.items() if min(a) >= 50]
    ndom = [(m, p, min(a), max(a)) for (m, p), (a, _) in verdict.items() if min(a) < 50]
    L += ["**Dominated by candidates** (candidate share >= 50% on every device that ran it, all runs; range "
          "over devices): " + ", ".join(f"{m} {p} ({lo:.0f}-{hi:.0f}%)" for m, p, lo, hi in dom) + ".", "",
          "**Not dominated** (below 50% on at least one device): " +
          ", ".join(f"{m} {p} ({lo:.0f}-{hi:.0f}%)" for m, p, lo, hi in ndom) + ".", ""]
    aff = [(m, p, a, u) for (m, p), (a, u) in verdict.items() if len(u) != len(a)]
    chg = [(m, p, a, u) for m, p, a, u in aff if (min(a) >= 50) != (min(u) >= 50)]
    L += ["Sensitivity (‡ cells excluded): the verdicts of " + ", ".join(f"{m} {p}" for m, p, _, _ in aff)
          + " then rest on " + "/".join(sorted({str(len(u)) for *_, u in aff})) + " devices. "
          + ("This changes " + ", ".join(
              f"{m} {p} to {'dominated' if min(u) >= 50 else 'not dominated'} ({min(u):.0f}-{max(u):.0f}% on the "
              f"remaining devices; {min(a):.0f}% with all runs)" for m, p, a, u in chg)
             if chg else "This changes none of them")
          + ". Whether the flagged cells' shares describe the work on that GPU is not established (section 1).",
          ""]
    # computed findings (desktops: no ‡ cells)
    desk = ["b580", "b70", "rocky-780m", "4070ti"]
    find = []
    for p in ("fp16", "fp32"):
        cs, rt = [], defaultdict(list)
        for d in desk:
            total = sum(float(s["total_median_gpu_us"]) for s in shaders
                        if s["device"] == d and s["precision"] == p and incl_all(d, s["model"], p))
            acc = Counter()
            for c in convs:
                if c["precision"] == p and c["wmma"] == "yes" and incl_all(d, c["model"], p):
                    acc[c[f"{d}_routing"]] += float(c[f"{d}_total_us"])
            cs.append(100 * sum(acc.values()) / total)
            for r in ("pointwise", "direct", "im2col+gemm"):
                rt[r].append(100 * acc[r] / sum(acc.values()))
        largest = Counter(max(rt, key=lambda r: rt[r][i]) for i in range(len(desk)))
        find.append(f"{p.upper()}, summed over the models on the four desktop GPUs: candidate configurations hold "
                    f"{min(cs):.0f}-{max(cs):.0f}% of the GPU shader time; of that, "
                    + ", ".join(f"{min(v):.0f}-{max(v):.0f}% runs {r}" for r, v in rt.items())
                    + " today (table below). Largest of the three: "
                    + ", ".join(f"{r} on {n} of {len(desk)}" for r, n in largest.most_common()) + ".")
    dil = [(d, sum(num(c[f"{d}_share_pct"]) or 0 for c in convs if c["model"] == "dl3"
                   and c["precision"] == "fp16" and c["dilation"] != "1x1"))
           for d in DEVICES if incl_all(d, "dl3", "fp16")]
    dil_routes = {c[f"{d}_routing"] for c in convs for d in DEVICES
                  if c["dilation"] != "1x1" and c["precision"] != "int8-q8ta" and c[f"{d}_routing"]}
    dil_k = max(int(c["gemm_k"]) for c in convs if c["model"] == "dl3" and c["dilation"] != "1x1")
    if dil:
        find.append("dl3 FP16: the dilated 3x3 convolutions (ASPP and the dilated ResNet-50 stages) hold "
                    + ", ".join(f"{SHORT[d]} {v:.0f}%{mark(d, 'dl3', 'fp16')}" for d, v in dil)
                    + f" of the GPU shader time. They are candidates (groups = 1, K up to {dil_k}) but in FP32 "
                      f"and FP16 run {', '.join(f'`{r}`' for r in sorted(dil_routes))} on every device: "
                      "`should_use_conv2d_im2col` rejects dilation != 1 (the q8ta INT8 path does use im2col).")
    for m in ("mv2", "mv3"):
        vals = [(d, sum(num(c[f"{d}_share_pct"]) or 0 for c in convs if c["model"] == m
                        and c["precision"] == "fp16" and c["wmma"] == "no (depthwise)"),
                 sum(num(c[f"{d}_share_pct"]) or 0 for c in convs if c["model"] == m
                     and c["precision"] == "fp16" and c["wmma"] == "low-util (M<16)"))
                for d in DEVICES if incl_all(d, m, "fp16")]
        find.append(f"{m} FP16: depthwise convolutions (not a dense GEMM) hold "
                    + ", ".join(f"{SHORT[d]} {v:.0f}%{mark(d, m, 'fp16')}" for d, v, _ in vals)
                    + ("; the M = 1 squeeze-excitation convolutions, which would fill 1 of 16 rows of a "
                       "padded tile, hold " + ", ".join(f"{SHORT[d]} {w:.0f}%{mark(d, m, 'fp16')}"
                                                         for d, _, w in vals) if m == "mv3" else "")
                    + ".")
    L += ["### Findings", ""] + [f"- {x}" for x in find] + [""]
    # candidate time by the routing it takes today
    L += ["### Where candidate time runs today", "",
          "GPU time of `yes` configurations summed over models (all runs; ‡ marks a device with flagged cells "
          "in the precision), split by the routing class the device dispatched; the first column is that sum "
          "as a share of the device's summed GPU shader time for the precision.", ""]
    rclasses = ["pointwise", "direct", "im2col+gemm", "winograd", "1d", "1d-pointwise"]
    rows = []
    for p in PRECISIONS:
        for d in DEVICES:
            acc = Counter()
            for c in convs:
                if c["precision"] != p or c["wmma"] != "yes" or not incl_all(d, c["model"], p):
                    continue
                acc[c[f"{d}_routing"]] += float(c[f"{d}_total_us"])
            total = sum(float(s["total_median_gpu_us"]) for s in shaders
                        if s["device"] == d and s["precision"] == p and incl_all(d, s["model"], p))
            cand = sum(acc.values())
            if not total:
                continue
            fd = " ‡" if any((d, m, p) in flagged for m in MODELS) else ""
            extra = sorted(set(acc) - set(rclasses))
            rows.append([p, SHORT[d] + fd, pct(100 * cand / total)] +
                        [pct(100 * acc[r] / cand) if cand else "--" for r in rclasses] +
                        ["; ".join(f"{r} {100 * acc[r] / cand:.1f}%" for r in extra)])
    L += table(["precision", "device", "candidate share"] + rclasses + ["other"], rows,
               ["---", "---"] + ["---:"] * (1 + len(rclasses)) + ["---"]) + [""]
    # what holds the non-candidate time
    L += ["### What holds the rest (FP16)", "",
          "Mean over all devices (all runs) of the share of the model's GPU shader time; ‡ after the device "
          "count marks a model with a flagged FP16 cell.", ""]
    rows = []
    ckeys = ["yes", "im2col-only (Cin<16)", "no (depthwise)", "no (grouped)", "low-util (Cout<16)",
             "low-util (M<16)"]
    for m in MODELS:
        p = "fp16"
        devs = [d for d in DEVICES if incl_all(d, m, p)]
        if not devs:
            continue
        acc = Counter()
        for c in convs:
            if c["model"] == m and c["precision"] == p:
                for d in devs:
                    acc[c["wmma"]] += (num(c[f"{d}_share_pct"]) or 0.0) / len(devs)
        for d in devs:
            for cat in CATEGORIES[1:]:
                acc[cat] += float(run_idx[d, m, p][f"share_{cat.split('/')[0]}_pct"]) / len(devs)
        fd = " ‡" if any((d, m, p) in flagged for d in devs) else ""
        rows.append([m, f"{len(devs)}{fd}"] + [pct(acc[k]) for k in ckeys + CATEGORIES[1:]])
    L += table(["model", "devices", "conv yes", "conv im2col-only", "conv depthwise", "conv grouped",
                "conv low-util Cout<16", "conv low-util M<16"] + CATEGORIES[1:], rows) + [""]
    # largest candidate configurations
    L += ["### Largest candidate configurations (FP16)", "",
          "The 30 `yes` configurations with the largest FP16 GPU time summed over their instances, ranked by "
          "the mean over the four desktop GPUs of that time as a share of the model's GPU shader time. "
          "us = time per instance on each device; routing per device "
          "(pw pointwise, dir direct, i2c im2col+gemm, wg winograd).", ""]
    ab = {"pointwise": "pw", "depthwise": "dw", "direct": "dir", "im2col+gemm": "i2c", "winograd": "wg",
          "1d": "1d", "1d-pointwise": "1d-pw", "transposed": "tr"}
    desk = ["b580", "b70", "rocky-780m", "4070ti"]
    cand = [c for c in convs if c["precision"] == "fp16" and c["wmma"] == "yes"]
    cand.sort(key=lambda c: -statistics.mean(num(c[f"{d}_share_pct"]) or 0 for d in desk))
    rows = []
    for c in cand[:30]:
        route = [ab.get(c[f"{d}_routing"], c[f"{d}_routing"]) for d in DEVICES]
        rt = route[0] if len(set(route)) == 1 else "/".join(route)
        rows.append([c["model"], shape_label(c), c["cin"], c["cout"], f"{c['hout']}x{c['wout']}", c["instances"],
                     rt, f"({c['gemm_m']},{c['gemm_n']},{c['gemm_k']})", c["k_steps"], c["out_tiles"],
                     pct(statistics.mean(num(c[f"{d}_share_pct"]) or 0 for d in desk))]
                    + [f"{float(c[f'{d}_us_per_instance']):.1f}{mark(d, c['model'], 'fp16')}"
                       if c[f"{d}_us_per_instance"] else "--" for d in DEVICES])
    L += table(["model", "kernel/stride/pad", "Cin", "Cout", "out", "inst.", "routing", "GEMM (M,N,K)", "K steps",
                "tiles", "desktop share"] + [f"{SHORT[d]} us" for d in DEVICES], rows,
               ["---", "---"] + ["---:"] * 4 + ["---", "---"] + ["---:"] * (3 + len(DEVICES))) + [""]
    L += ["### Every convolution configuration", "",
          "Per model, the distinct convolution configurations. The same shapes occur in FP32, FP16 and INT8; "
          "they are listed once with the FP16 row of `survey-conv-shapes.csv`: routing per device (one value "
          "when all agree, else B580/B70/780M/4070TiS/Pixel7a/S24+), GEMM-equivalent, K steps, output tiles, "
          "candidate verdict and the FP16 share of the model's GPU shader time per device.", ""]
    for m in MODELS:
        rs = [c for c in convs if c["model"] == m and c["precision"] == "fp16"]
        if not rs:
            continue
        rs.sort(key=lambda c: -statistics.mean(num(c[f"{d}_share_pct"]) or 0 for d in DEVICES))
        L += [f"<details><summary>{m}: {len(rs)} configurations</summary>", ""]
        rows = []
        for c in rs:
            route = [ab.get(c[f"{d}_routing"], c[f"{d}_routing"]) for d in DEVICES]
            rt = route[0] if len(set(route)) == 1 else "/".join(route)
            rows.append([shape_label(c), c["cin"], c["cout"], f"{c['hin']}x{c['win']}", f"{c['hout']}x{c['wout']}",
                         c["instances"], rt, f"({c['gemm_m']},{c['gemm_n']},{c['gemm_k']})", c["k_steps"],
                         c["out_tiles"], c["wmma"]]
                        + [f"{float(c[f'{d}_share_pct']):.2f}{mark(d, m, 'fp16')}" if c[f"{d}_share_pct"] else "--"
                           for d in DEVICES])
        L += table(["kernel/stride/pad", "Cin", "Cout", "in", "out", "inst.", "routing", "GEMM (M,N,K)", "K steps",
                    "tiles", "WMMA"] + [f"{SHORT[d]} %" for d in DEVICES], rows,
                   ["---"] + ["---:"] * 5 + ["---", "---", "---:", "---:", "---"] + ["---:"] * len(DEVICES))
        L += ["", "</details>", ""]

    # ---------------- 6. Non-convolution
    L += ["## 6. Large non-convolution work", "",
          "Non-convolution shaders that hold at least 5% of a model's GPU shader time on at least one device "
          "(all runs; ‡ = flagged cell; share per device; `--` below 0.05% or not dispatched).", ""]
    big = defaultdict(dict)
    for s in shaders:
        if s["category"] != "convolution":
            big[s["model"], s["precision"], s["shader"]][s["device"]] = (float(s["share_pct"]), s["dispatches"],
                                                                         s["category"])
    rows = []
    for (m, p, n), v in sorted(big.items(), key=lambda kv: (MODELS.index(kv[0][0]), PRECISIONS.index(kv[0][1]),
                                                             -max(x[0] for x in kv[1].values()))):
        if max((x[0] for d, x in v.items()), default=0) < 5:
            continue
        cat, disp_n = next(iter(v.values()))[2], next(iter(v.values()))[1]
        rows.append([m, p, f"`{n}`", cat, disp_n] + [f"{v[d][0]:.1f}%{mark(d, m, p)}" if d in v and v[d][0] >= 0.05
                                                     else "--" for d in DEVICES])
    L += table(["model", "precision", "shader", "category", "disp."] + [SHORT[d] for d in DEVICES], rows,
               ["---"] * 4 + ["---:"] * (1 + len(DEVICES))) + [""]

    # ---------------- 7. Device differences
    L += ["## 7. Different shaders for the same layer", "",
          "Convolution configurations where the devices did not all dispatch the same kernel sequence (dtype "
          "suffix removed), grouped by pattern. Count = configurations (instances in parentheses).", ""]
    pat = defaultdict(lambda: [0, 0, []])
    for c in convs:
        seqs = {d: compress(strip_dtype(c[f"{d}_shaders"])) for d in DEVICES if c[f"{d}_shaders"]}
        if len(set(seqs.values())) <= 1:
            continue
        groups = defaultdict(list)
        for d, s in seqs.items():
            groups[s].append(SHORT[d])
        key = " vs ".join(f"{'/'.join(v)}: `{k}`" for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1])))
        e = pat[c["precision"], key]
        e[0] += 1
        e[1] += int(c["instances"])
        e[2].append(c["model"])
    rows = []
    for (p, key), (n, inst, ms) in sorted(pat.items(), key=lambda kv: (PRECISIONS.index(kv[0][0]), -kv[1][1])):
        rows.append([p, key, f"{n} ({inst})", ", ".join(sorted(set(ms), key=MODELS.index))])
    L += table(["precision", "kernels per device", "configs (inst.)", "models"], rows,
               ["---", "---", "---:", "---"]) + [""]

    # ---------------- 8. Limits
    L += ["## 8. Limits", "",
          "- One fixed input per model (the `torch.randn` example input of the model definition, seeded and "
          "saved); no accuracy evaluation. Top-1 on a random input mainly detects broken runs. `densenet161` "
          "has untrained weights and `w2l` is untrained, as ExecuTorch's export script and model zoo define them.",
          "- 11 kept executions per combination in one session per device: a survey of where time goes, not a "
          "benchmark. Phones were tested in their current state through ADB, without clock pinning or thermal "
          "control; desktops at default clocks. B580 runs took the shared gpu-lab lock between another agent's "
          "jobs. The B70's card 0 also held an idle LLM server's memory (see `b70/device.txt`, `LOG.md`).",
          "- GPU shader time is the sum of Vulkan timestamp-query intervals. Each interval is bracketed by the "
          "timestamps written around `register_shader_dispatch` (`DispatchNode.cpp:93-121`), which also records "
          "the dispatch's pipeline barrier (`Context.cpp:208`), so barrier waits fall inside the interval. Time "
          "between intervals and host overhead are not in it; they are in the inference time. ‡ marks the combinations where it falls "
          f"below {FLAG_PCT:.0f}% of the host graph time (a heuristic cutoff; {len(flagged)} cells, on "
          + ", ".join(sorted({DEVICES[d] for d, _, _ in flagged})) + "); "
          "the cause of the gap is not established. They are included in all results; the sensitivity views "
          "show what leaving them out changes.",
          "- Storage: only the convolution input/output storage is in the trace (operator-level metadata); "
          "prepacked-weight, bias and intermediate storage are not recorded.",
          "- FP16 is `force_fp16` (FP32 inputs/outputs converted at the delegate boundary). INT8 is the "
          "`int8-q8ta` recipe of the MobileNet survey (XNNPACK quantizer, static symmetric per-channel, "
          "calibrated on six random inputs): its accuracy is not meaningful, and it adds quantize/dequantize "
          "dispatches. w2l has no INT8 result (export fails, section 1).",
          "- WMMA candidacy is shape arithmetic only (groups, channel counts, M); it does not account for the "
          "layout conversions a cooperative-matrix kernel may need, nor whether the device exposes cooperative "
          "matrices (the two phones do not).",
          ""]
    (ROOT / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"stage 2: REPORT.md {len(L)} lines, models/MODELS.md")


def write_models_md(rows):
    by = defaultdict(dict)
    for r in rows:
        by[r["model"]][r["precision"]] = r
    L = ["# Exported models", "",
         "Exported by `export_models.py` (`export_all.sh`) with ExecuTorch `main` `2efa9d0316` as source tree, "
         "`torch.export(strict=True)` + `to_edge_transform_and_lower(VulkanPartitioner)` as in "
         "`examples/vulkan/export.py`; FP16 adds `force_fp16`. INT8 (`int8-q8ta`): XNNPACK quantizer, static "
         "symmetric per-channel, calibrated on the example input and five seeded random inputs. Generated from "
         "`models/models.csv` by `aggregate.py`.", "",
         "Node counts are from `get_delegation_info` on the lowered graph: delegated = edge operators inside the "
         "Vulkan delegate; non-delegated = operators left outside, which here are only the `getitem` nodes that "
         "unpack the delegate's outputs (none of the models leaves a compute operator on CPU).", ""]
    hdr = ["model", "source", "input", "output", "parameters", "PTE MB fp32 / fp16 / int8",
           "delegated nodes fp32 / fp16 / int8", "non-delegated (getitem + other)", "export problems"]
    out = []
    for m in MODELS:
        r = by[m]
        any_ok = next((r[p] for p in PRECISIONS if r[p]["export_exit"] == "0"), None)
        size = " / ".join(f"{int(r[p]['pte_bytes']) / 1e6:.1f}" if r[p]["export_exit"] == "0" else "failed"
                          for p in PRECISIONS)
        nodes = " / ".join(r[p]["delegated_nodes"] if r[p]["export_exit"] == "0" else "--" for p in PRECISIONS)
        nd = " / ".join((f"{r[p]['non_delegated_getitem']}" + (f" + {r[p]['non_delegated_other']}"
                                                               if r[p]["non_delegated_other"] else ""))
                        if r[p]["export_exit"] == "0" else "--" for p in PRECISIONS)
        probs = "; ".join(f"{p}: exit {r[p]['export_exit']}: {r[p]['export_error']}" for p in PRECISIONS
                          if r[p]["export_exit"] != "0") or "none"
        out.append([f"`{m}`", any_ok["source"], any_ok["input_shapes"], any_ok["output_shapes"],
                    f"{int(any_ok['parameters']):,}", size, nodes, nd, probs])
    L += table(hdr, out, ["---"] * 4 + ["---:"] * 4 + ["---"]) + [""]
    L += ["Files per model: `MODEL-input.bin` (float32, the input shape above), `MODEL-fp32-reference-<i>.bin` "
          "(CPU eager FP32 output i), `MODEL-int8-q8ta-reference-<i>.bin` (CPU fake-quantized output), "
          "`MODEL-PRECISION.pte`, `MODEL-PRECISION.json` (export record), `MODEL-PRECISION-export.{log,exit}`.",
          "Downloaded pretrained weights are cached in `models/cache/`."]
    (ROOT / "models" / "MODELS.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true", help="only stage 2 (CSVs -> REPORT.md, MODELS.md)")
    a = ap.parse_args()
    if not a.report_only:
        stage1()
    stage2()
