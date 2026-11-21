#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""INT8 depthwise cases of the q8ta_conv2d_dw width tiles (change
conv2d-all-used-kernels, part D) from model_sweep_configs_int8dw.json (the
INT8 depthwise layers of the model survey):

    q8ta_dw_tile_performance.json  per configuration "<label> baseline" (route
                                   auto) and "<label> dw_tile" (the _w2 and _w1
                                   variants), 4W4C input and output
    q8ta_dw_tile_correctness.json  every configuration in both layouts and edge
                                   configurations (odd sizes, channels not a
                                   multiple of 4, stride 2, dilation 2), each
                                   with every variant (stock tile 4 included)

The output must equal stock bit for bit; the CPU reference check allows one
quantization step, as the upstream depthwise cases do, and the benchmark's
output digest compares the GPU outputs.

    gen_q8ta_dw_tile_cases.py [--check]
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
from format_json import format_value  # noqa: E402

TIMING = {"target_ms": 50, "warmup_runs": 3, "benchmark_runs": 5}
TOLERANCE = {"function": "is_close", "atol": 1, "rtol": 0, "outputs": ["output"]}
ARMS = ["q8ta_conv2d_dw_w2", "q8ta_conv2d_dw_w1"]
# Edge configurations (batch 1, as the table rows): (C, H, W, k, stride,
# padding, dilation).
EDGE = [
    (30, 13, 13, 3, 1, 1, 1),
    (16, 7, 9, 5, 2, 2, 1),
    (24, 15, 11, 3, 2, 1, 1),
    (12, 14, 14, 3, 1, 2, 2),
    (8, 5, 3, 3, 1, 1, 1),
]


def tensor(name, sizes, dtype, storage, layout, data):
    return {
        "tensor": {
            "name": name,
            "sizes": sizes,
            "dtype": dtype,
            "storage": storage,
            "layout": layout,
            "data": data,
        }
    }


def dw_args(c, h, w, k, stride, pad, dil, layout="4w4c"):
    hout = (h + 2 * pad[0] - dil[0] * (k[0] - 1) - 1) // stride[0] + 1
    wout = (w + 2 * pad[1] - dil[1] * (k[1] - 1) - 1) // stride[1] + 1
    const = ("constant", "width")
    return [
        tensor(
            "input",
            [1, c, h, w],
            "int8x4",
            "buffer",
            layout,
            {"rand_uniform_int": [-128, 127]},
        ),
        {"float": 0.02},
        {"int": -3},
        tensor(
            "weight", [k[0], k[1], c], "int8", *const, {"rand_uniform_int": [-127, 127]}
        ),
        tensor("weight_sums", [c], "int32", *const, {"column_sum": "weight"}),
        tensor(
            "weight_scales",
            [c],
            "float32",
            *const,
            {"rand_uniform_float": [0.002, 0.01]},
        ),
        {"float": 0.390021},
        {"int": 5},
        tensor("bias", [c], "float32", *const, {"rand_uniform_float": [-1, 1]}),
        {"int_list": list(k)},
        {"int_list": list(stride)},
        {"int_list": list(pad)},
        {"int_list": list(dil)},
        {"int": c},
        {"string": "none"},
        tensor("output", [1, c, hout, wout], "int8x4", "buffer", layout, {"fill": 0}),
    ]


def config_args(c, layout):
    return dw_args(
        c["cin"],
        c["h"],
        c["w"],
        (c["kh"], c["kw"]),
        tuple(c["stride"]),
        tuple(c["padding"]),
        tuple(c["dilation"]),
        layout=layout,
    )


def definitions(name, args, arms, timing):
    common = {
        "operator": "et_vk.q8ta_conv2d_dw.default",
        **timing,
        "validation_method": TOLERANCE,
        "args_sweep": [args],
    }
    return [
        {"name": f"{name} baseline", **common, "route_sweep": ["auto"]},
        {
            "name": f"{name} dw_tile",
            **common,
            "route": {"name": "dw_tile", "options": {"variant": "v_VARIANT"}},
            "sweep_variables": {"v_VARIANT": arms},
        },
    ]


def files():
    configs = json.loads((HERE / "model_sweep_configs_int8dw.json").read_text())
    perf, corr = [], []
    check = {"target_ms": 1, "warmup_runs": 0, "benchmark_runs": 1}
    for c in configs:
        perf += definitions(c["label"], config_args(c, "4w4c"), ARMS, TIMING)
        for layout in ("4w4c", "4c"):
            corr += definitions(
                f"{c['label']} {layout}",
                config_args(c, layout),
                ["q8ta_conv2d_dw", *ARMS],
                check,
            )
    for cc, h, w, k, s, p, d in EDGE:
        for layout in ("4w4c", "4c"):
            name = f"edge {k}x{k} s{s} p{p} d{d} g{cc} {cc}-{cc} {h}x{w} {layout}"
            args = dw_args(cc, h, w, (k, k), (s, s), (p, p), (d, d), layout)
            corr += definitions(name, args, ["q8ta_conv2d_dw", *ARMS], check)
    return {
        HERE / "q8ta_dw_tile_performance.json": perf,
        HERE / "q8ta_dw_tile_correctness.json": corr,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    bad = []
    for path, value in files().items():
        text = format_value(value) + "\n"
        if args.check:
            if not path.exists() or path.read_text() != text:
                bad.append(path.name)
        else:
            path.write_text(text)
    if bad:
        sys.exit("out of date: " + ", ".join(bad))


if __name__ == "__main__":
    main()
