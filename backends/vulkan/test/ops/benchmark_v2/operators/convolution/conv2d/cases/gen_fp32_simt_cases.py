#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FP32 conv2d_simt cases (change conv2d-all-used-kernels) from
model_sweep_configs_fp32.json:

    fp32_coopmat_performance.json  per configuration "<label> baseline" (route
                                   auto) and "<label> simt" (every
                                   conv2d_simt variant of the configuration:
                                   the 1x1 forms for 1x1, stride 1, padding 0,
                                   the _dg forms otherwise)
    fp32_simt_correctness.json     edge configurations (odd sizes, input
                                   channels not a multiple of 4, stride 2,
                                   dilation, dead taps, residual and
                                   activation epilogues, no bias) with every
                                   variant

Tolerance: that of the FP32 convolution cases (atol and rtol 0.001).

    gen_fp32_simt_cases.py [--check]
"""

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE))
from format_json import format_value  # noqa: E402
from gen_im2col_coopmat_cases import conv_args  # noqa: E402

YAML = HERE.parents[6] / "runtime/graph/ops/glsl/convolution/conv2d/conv2d_simt.yaml"
CONFIGS = HERE / "model_sweep_configs_fp32.json"
PERFORMANCE_TIMING = {"target_ms": 50, "warmup_runs": 3, "benchmark_runs": 5}
CORRECTNESS_TIMING = {"target_ms": 5, "warmup_runs": 1, "benchmark_runs": 1}
TOLERANCE = {
    "function": "is_close",
    "atol": 0.001,
    "rtol": 0.001,
    "outputs": ["output"],
}


def variants():
    names = re.findall(r"- NAME: (conv2d_simt_\S+)", YAML.read_text())
    return [n for n in names if n != "conv2d_simt_merge_float"]


def fp32(value):
    if isinstance(value, dict):
        return {k: ("float32" if k == "dtype" else fp32(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [fp32(v) for v in value]
    return value


def is_pointwise(c):
    return (c["kh"], c["kw"], c["stride"], c["padding"]) == (1, 1, [1, 1], [0, 0])


def definitions(name, c, all_variants, timing, op="conv", epilogue=None, bias=True):
    args = fp32(
        conv_args(
            c["cin"],
            c["h"],
            c["w"],
            c["cout"],
            c["kh"],
            c["kw"],
            stride=tuple(c["stride"]),
            pad=tuple(c["padding"]),
            dilation=tuple(c["dilation"]),
            bias=bias,
            op=op,
            epilogue=epilogue,
        )
    )
    operator = {
        "conv": "aten.convolution.default",
        "epilogue": "et_vk.conv2d.with_epilogue",
    }[op]
    common = {"operator": operator, **timing, "validation_method": TOLERANCE}
    pw = is_pointwise(c)
    vs = [v for v in all_variants if ("_dg" in v) != pw]
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


def edge_configs():
    out = []
    for kh, kw, s, ph, pw_, d, cin, cout, h, w in (
        (1, 1, 1, 0, 0, 1, 24, 40, 7, 7),
        (1, 1, 1, 0, 0, 1, 30, 36, 9, 13),
        (1, 1, 1, 0, 0, 1, 1024, 100, 5, 3),
        (3, 3, 1, 1, 1, 1, 64, 64, 15, 17),
        (3, 3, 2, 1, 1, 1, 3, 32, 23, 21),
        (3, 3, 2, 0, 0, 1, 32, 48, 25, 25),
        (3, 3, 1, 2, 2, 2, 128, 72, 13, 13),
        (1, 7, 1, 0, 3, 1, 64, 52, 9, 9),
        (7, 1, 1, 3, 0, 1, 64, 52, 9, 9),
        (1, 1, 2, 0, 0, 1, 64, 128, 14, 14),
        (5, 5, 1, 2, 2, 1, 6, 16, 11, 7),
        (3, 3, 1, 24, 24, 24, 256, 64, 28, 28),
        (7, 7, 2, 3, 3, 1, 3, 64, 30, 30),
    ):
        ho = (h + 2 * ph - d * (kh - 1) - 1) // s + 1
        wo = (w + 2 * pw_ - d * (kw - 1) - 1) // s + 1
        out.append(
            {
                "label": f"fp32 {kh}x{kw} s{s}x{s} p{ph}x{pw_} d{d}x{d} {cin}-{cout} {h}x{w}",
                "cin": cin,
                "cout": cout,
                "kh": kh,
                "kw": kw,
                "stride": [s, s],
                "padding": [ph, pw_],
                "dilation": [d, d],
                "h": h,
                "w": w,
                "hout": ho,
                "wout": wo,
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
        corr += definitions(c["label"], c, vs, CORRECTNESS_TIMING)
        corr += definitions(
            f"{c['label']} nobias", c, vs, CORRECTNESS_TIMING, bias=False
        )
        corr += definitions(
            f"{c['label']} residual relu",
            c,
            vs,
            CORRECTNESS_TIMING,
            op="epilogue",
            epilogue=("add", 0.5, "relu"),
        )
    return {
        HERE / "fp32_coopmat_performance.json": perf,
        HERE / "fp32_simt_correctness.json": corr,
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
