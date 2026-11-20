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
    common = {"operator": op, **timing, "validation_method": EXACT}
    vs = [v for v in all_variants if ("_dg" in v) != pw]
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
