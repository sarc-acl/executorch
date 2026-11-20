#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""conv1d_simt cases (change conv2d-all-used-kernels): the conv1d and
conv1d_pw layers of the model survey's Wav2Letter (batch 10) in FP16 and FP32.

    model_sweep_configs_conv1d.json  the configurations, labelled
                                     "<fp16|fp32> 1d 1x<k> s1x<s> p0x<p>
                                     d1x<d> <cin>-<cout> 1x<L>" (the table
                                     key: kernel 1 x k on a 1 x L input)
    conv1d_simt_performance.json     per configuration "<label> baseline"
                                     (route auto) and "<label> simt" (the
                                     conv1d_simt variants of its role: _pw
                                     for kernel 1 on the height-packed
                                     layout, the others on the
                                     channels-packed layout, as the model
                                     dispatches them)
    conv1d_simt_correctness.json     the same with batch 3 and odd channels

    gen_conv1d_simt_cases.py [--check]
"""

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
from format_json import format_value  # noqa: E402

YAML = HERE.parents[6] / "runtime/graph/ops/glsl/convolution/conv1d/conv1d_simt.yaml"
# (cin, cout, kernel, stride, padding, length) of the survey's Wav2Letter.
LAYERS = [
    (250, 250, 7, 1, 3, 2),
    (250, 2000, 32, 1, 16, 2),
    (250, 250, 48, 2, 23, 4),
    (1, 250, 250, 160, 45, 700),
    (2000, 4096, 1, 1, 0, 3),
    (2000, 2000, 1, 1, 0, 3),
]
BATCH = 10
PERFORMANCE_TIMING = {"target_ms": 50, "warmup_runs": 3, "benchmark_runs": 5}
CORRECTNESS_TIMING = {"target_ms": 5, "warmup_runs": 1, "benchmark_runs": 1}
TOLERANCE = {
    "fp32": {
        "function": "is_close",
        "atol": 0.001,
        "rtol": 0.001,
        "outputs": ["output"],
    },
    "fp16": {"function": "is_close", "atol": 0.1, "rtol": 0.01, "outputs": ["output"]},
}
DTYPE = {"fp32": "float32", "fp16": "float16"}


def variants():
    names = re.findall(r"- NAME: (conv1d_simt_\S+)", YAML.read_text())
    return [n for n in names if "_merge" not in n]


def out_len(L, k, s, p):
    return (L + 2 * p - (k - 1) - 1) // s + 1


def label(prec, cin, cout, k, s, p, L):
    return f"{prec} 1d 1x{k} s1x{s} p0x{p} d1x1 {cin}-{cout} 1x{L}"


def config(prec, cin, cout, k, s, p, L):
    return {
        "label": label(prec, cin, cout, k, s, p, L),
        "cin": cin,
        "h": 1,
        "w": L,
        "cout": cout,
        "kh": 1,
        "kw": k,
        "stride": [1, s],
        "padding": [0, p],
        "dilation": [1, 1],
        "batch": BATCH,
    }


def definitions(name, prec, cin, cout, k, s, p, L, batch, timing):
    layout = "height" if k == 1 else "channels"
    dt = DTYPE[prec]
    w = float(f"{1 / (cin * k) ** 0.5:.4g}")

    def t(tname, sizes, data, storage="texture3d"):
        x = {"name": tname, "sizes": sizes, "dtype": dt, "storage": storage}
        if storage == "texture3d":
            x["layout"] = layout
        x["data"] = data
        return {"tensor": x}

    args = [
        t("input", [batch, cin, L], {"rand_uniform_float": [-1, 1]}),
        t("weight", [cout, cin, k], {"rand_uniform_float": [-w, w]}, "constant"),
        t("bias", [cout], {"rand_uniform_float": [-1, 1]}, "constant"),
        {"int_list": [s]},
        {"int_list": [p]},
        {"int_list": [1]},
        {"bool": False},
        {"int_list": [0]},
        {"int": 1},
        t("output", [batch, cout, out_len(L, k, s, p)], {"fill": 0}),
    ]
    common = {
        "operator": "aten.convolution.default",
        **timing,
        "validation_method": TOLERANCE[prec],
    }
    suffix = "_half" if prec == "fp16" else "_float"
    vs = [v + suffix for v in variants() if ("_pw" in v) == (k == 1)]
    return [
        {
            "name": f"{name} baseline",
            **common,
            "route_sweep": ["auto"],
            "args_sweep": [args],
        },
        {
            "name": f"{name} simt",
            **common,
            "route": {"name": "simt", "options": {"variant": "v_VARIANT"}},
            "sweep_variables": {"v_VARIANT": vs},
            "args_sweep": [args],
        },
    ]


def files():
    configs, perf, corr = [], [], []
    for prec in ("fp16", "fp32"):
        for layer in LAYERS:
            configs.append(config(prec, *layer))
            perf += definitions(
                label(prec, *layer), prec, *layer, BATCH, PERFORMANCE_TIMING
            )
        for cin, cout, k, s, p, L in [
            (30, 22, 7, 1, 3, 5),
            (250, 61, 32, 1, 16, 2),
            (1, 13, 25, 16, 4, 70),
            (130, 70, 1, 1, 0, 3),
        ]:
            corr += definitions(
                label(prec, cin, cout, k, s, p, L) + " batch3",
                prec,
                cin,
                cout,
                k,
                s,
                p,
                L,
                3,
                CORRECTNESS_TIMING,
            )
    return {
        HERE / "model_sweep_configs_conv1d.json": configs,
        HERE / "conv1d_simt_performance.json": perf,
        HERE / "conv1d_simt_correctness.json": corr,
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
