#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""INT8 cooperative-matrix convolution cases (change conv2d-all-used-kernels)
from model_sweep_configs_int8.json:

    q8ta_coopmat_performance.json  per configuration "<label> baseline" (route
                                   auto) and "<label> coopmat" (every
                                   q8ta_conv2d_pw_coopmat variant of the
                                   configuration's route: pw_coopmat on the
                                   q8ta_conv2d_pw operator for 1x1, stride 1,
                                   padding 0; im2col_coopmat with the _dg
                                   variants on q8ta_conv2d otherwise)
    q8ta_coopmat_correctness.json  small and edge configurations (odd widths
                                   and channels, batch 2, both output
                                   layouts, relu, no bias, input zero point
                                   -128 and 127) with every variant

Validation is exact (atol 0): the CPU reference is the stock requantization,
so a passing variant equals the stock kernel bit for bit where the stock kernel
passes too.

    gen_q8ta_coopmat_cases.py [--check]
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
from format_json import format_value  # noqa: E402

YAML = (
    HERE.parents[6]
    / "runtime/graph/ops/glsl/convolution/conv2d/q8ta/q8ta_conv2d_pw_coopmat.yaml"
)
CONFIGS = HERE / "model_sweep_configs_int8.json"
PERFORMANCE_TIMING = {"target_ms": 50, "warmup_runs": 3, "benchmark_runs": 5}
CORRECTNESS_TIMING = {"target_ms": 5, "warmup_runs": 1, "benchmark_runs": 1}
EXACT = {"function": "is_close", "atol": 0, "rtol": 0, "outputs": ["output"]}
INPUT_SCALE = 0.02
WEIGHT_SCALES = [0.002, 0.01]


def variants():
    names = re.findall(r"- NAME: (q8ta_conv2d_pw_coopmat_\S+)", YAML.read_text())
    return [n for n in names if not n.endswith("_merge")]


def is_pointwise(c):
    return (c["kh"], c["kw"], c["stride"], c["padding"]) == (1, 1, [1, 1], [0, 0])


def conv_args(c, out_layout="4w4c", zp=-3, bias=True, act="none", batch=1):
    k = c["kh"] * c["kw"] * c["cin"]
    k_cols = -(-k // 4) * 4
    # Output scale for a standard deviation of about 40 quantization steps.
    out_scale = INPUT_SCALE * 0.006 * 73.6 * 73.6 * math.sqrt(k) / 40
    t = lambda name, sizes, dtype, storage, layout, data: {  # noqa: E731
        "tensor": {
            "name": name,
            "sizes": sizes,
            "dtype": dtype,
            "storage": storage,
            "layout": layout,
            "data": data,
        }
    }
    cout = c["cout"]
    return [
        t(
            "input",
            [batch, c["cin"], c["h"], c["w"]],
            "int8x4",
            "buffer",
            "4w4c",
            {"rand_uniform_int": [-128, 127]},
        ),
        {"float": INPUT_SCALE},
        {"int": zp},
        t(
            "weight",
            [cout, k_cols],
            "int8",
            "constant",
            "width",
            {
                "pad_columns": {
                    "source": {"rand_uniform_int": [-127, 127]},
                    "valid_columns": k,
                }
            },
        ),
        t("weight_sums", [cout], "int32", "constant", "width", {"row_sum": "weight"}),
        t(
            "weight_scales",
            [cout],
            "float32",
            "constant",
            "width",
            {"rand_uniform_float": WEIGHT_SCALES},
        ),
        {"float": round(out_scale, 6)},
        {"int": 5},
        (
            t(
                "bias",
                [cout],
                "float32",
                "constant",
                "width",
                {"rand_uniform_float": [-1, 1]},
            )
            if bias
            else None
        ),
        {"int_list": [c["kh"], c["kw"]]},
        {"int_list": c["stride"]},
        {"int_list": c["padding"]},
        {"int_list": c["dilation"]},
        {"int": 1},
        {"string": act},
        t(
            "output",
            [batch, cout, c["hout"], c["wout"]],
            "int8x4",
            "buffer",
            out_layout,
            {"fill": 0},
        ),
    ]


def definitions(name, c, all_variants, timing, **kw):
    pw = is_pointwise(c)
    op = "et_vk.q8ta_conv2d_pw.default" if pw else "et_vk.q8ta_conv2d.default"
    args = [conv_args(c, **kw)]
    # Where stock runs the direct q8ta_conv2d shader (channels not a multiple
    # of 4, or below 32 channels on more than 4096 output pixels: the
    # single-batch rule of should_use_q8ta_conv2d_im2col), its requantization
    # can miss the CPU reference by one step; the arms repeat it, and their
    # outputs are compared with stock's by digest (tools of the change).
    stock_direct = not pw and (
        c["cin"] % 4 != 0 or (c["cin"] < 32 and c["hout"] * c["wout"] > 4096)
    )
    method = {**EXACT, "atol": 1} if stock_direct else EXACT
    common = {"operator": op, **timing, "validation_method": method}
    if pw:
        vs = [v for v in all_variants if "_dg" not in v]
    else:
        # _dgb gathers each K byte alone: only where the channels are not a
        # multiple of 4 (the stock first layers).
        tag = "_dg_" if c["cin"] % 4 == 0 else "_dgb"
        vs = [v for v in all_variants if tag in v + "_"]
    return [
        {
            "name": f"{name} baseline",
            **common,
            "route_sweep": ["auto"],
            "args_sweep": args,
        },
        {
            "name": f"{name} coopmat",
            **common,
            "route": {
                "name": "pw_coopmat" if pw else "im2col_coopmat",
                "options": {"variant": "v_VARIANT"},
            },
            "sweep_variables": {"v_VARIANT": vs},
            "args_sweep": args,
        },
    ]


def out_size(h, k, s, p, d):
    return (h + 2 * p - d * (k - 1) - 1) // s + 1


def edge_configs():
    out = []
    for kh, kw, s, p, d, cin, cout, h, w in (
        (1, 1, 1, 0, 1, 24, 40, 7, 7),
        (1, 1, 1, 0, 1, 100, 36, 9, 13),
        (1, 1, 1, 0, 1, 512, 1000, 3, 5),
        (1, 1, 1, 0, 1, 2048, 64, 6, 6),
        (3, 3, 1, 1, 1, 64, 64, 15, 17),
        (3, 3, 2, 1, 1, 32, 48, 23, 21),
        (3, 3, 1, 2, 2, 128, 72, 13, 13),
        (1, 7, 1, 0, 1, 64, 52, 9, 9),
        (7, 1, 1, 0, 1, 64, 52, 9, 9),
        (1, 1, 2, 0, 1, 64, 128, 14, 14),
        (5, 5, 1, 2, 1, 48, 16, 11, 7),
        (3, 3, 1, 12, 12, 256, 64, 28, 28),
        (7, 7, 2, 3, 1, 3, 64, 30, 30),
        (3, 3, 2, 1, 1, 3, 32, 23, 21),
        (3, 3, 1, 1, 1, 6, 20, 13, 11),
    ):
        ph, pw = (
            (0, 3) if (kh, kw) == (1, 7) else (3, 0) if (kh, kw) == (7, 1) else (p, p)
        )
        out.append(
            {
                "label": f"int8 {kh}x{kw} s{s}x{s} p{ph}x{pw} d{d}x{d} {cin}-{cout} {h}x{w}",
                "cin": cin,
                "cout": cout,
                "kh": kh,
                "kw": kw,
                "stride": [s, s],
                "padding": [ph, pw],
                "dilation": [d, d],
                "h": h,
                "w": w,
                "hout": out_size(h, kh, s, ph, d),
                "wout": out_size(w, kw, s, pw, d),
            }
        )
    return out


def repeat_checks(all_variants):
    """The _sa forms reset their tile counters for the next dispatch: five
    validated executions of one graph (full size, a shrink, full size again,
    twice more) per variant and configuration."""
    out = []
    for c in (
        {
            "label": "int8 1x1 s1x1 p0x0 d1x1 512-96 9x9",
            "cin": 512,
            "cout": 96,
            "kh": 1,
            "kw": 1,
            "stride": [1, 1],
            "padding": [0, 0],
            "dilation": [1, 1],
            "h": 9,
            "w": 9,
            "hout": 9,
            "wout": 9,
        },
        {
            "label": "int8 3x3 s1x1 p1x1 d1x1 192-40 7x7",
            "cin": 192,
            "cout": 40,
            "kh": 3,
            "kw": 3,
            "stride": [1, 1],
            "padding": [1, 1],
            "dilation": [1, 1],
            "h": 7,
            "w": 7,
            "hout": 7,
            "wout": 7,
        },
    ):
        pw = is_pointwise(c)
        for v in all_variants:
            if "_sa" not in v or ("_dg" in v) == pw:
                continue
            case = definitions(c["label"], c, [v], CORRECTNESS_TIMING)[1]
            case = {
                **{k: x for k, x in case.items() if k != "sweep_variables"},
                "name": f"{c['label']} repeat {v}",
                "route": {"name": case["route"]["name"], "options": {"variant": v}},
            }
            full = [1, c["cin"], c["h"], c["w"]]
            small = [1, c["cin"], c["h"] - 2, c["w"]]
            osz = lambda h: [1, c["cout"], h, c["wout"]]  # noqa: E731
            out.append(
                {
                    "name": f"{c['label']} repeat {v}",
                    "check": "operator_dynamic_resize",
                    "parameters": {"operator_case": case, "output_name": "output"},
                    "validation_steps": [
                        {
                            "tensor_sizes": {"input": sz},
                            "expected_output_sizes": osz(sz[2]),
                        }
                        for sz in (full, small, full, full, full)
                    ],
                }
            )
    return out


def files():
    configs = json.loads(CONFIGS.read_text())
    vs = variants()
    perf = [
        d for c in configs for d in definitions(c["label"], c, vs, PERFORMANCE_TIMING)
    ]
    corr = []
    for c in edge_configs():
        opts = [
            ("", {}),
            (" 4c1w", {"out_layout": "4c1w"}),
            (" relu nobias zp-128", {"act": "relu", "bias": False, "zp": -128}),
            (" zp127 batch2", {"zp": 127, "batch": 2}),
        ]
        for tag, kw in opts:
            corr += definitions(f"{c['label']}{tag}", c, vs, CORRECTNESS_TIMING, **kw)
    corr += repeat_checks(vs)
    return {
        HERE / "q8ta_coopmat_performance.json": perf,
        HERE / "q8ta_coopmat_correctness.json": corr,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    bad = []
    for path, value in files().items():
        text = format_value(value) + "\n"
        if args.check:
            if path.read_text() != text:
                bad.append(path.name)
        else:
            path.write_text(text)
    if bad:
        sys.exit("out of date: " + ", ".join(bad))


if __name__ == "__main__":
    main()
