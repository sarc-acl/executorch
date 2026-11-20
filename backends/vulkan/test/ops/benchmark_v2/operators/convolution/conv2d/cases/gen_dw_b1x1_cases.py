#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Depthwise sweep cases (change conv2d-all-used-kernels, part D) from
model_sweep_configs_dw.json (the FP16 and FP32 depthwise layers of the model
survey, labels prefixed with the precision):

    dw_b1x1_performance.json  per configuration "<label> baseline" with the
                              routes auto and b1x1 (the depthwise shader's 1x1
                              output batch; stride-2 layers run their own
                              shader either way)

    gen_dw_b1x1_cases.py [--check]
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE))
from format_json import format_value  # noqa: E402
from gen_fp32_simt_cases import fp32  # noqa: E402
from gen_im2col_coopmat_cases import conv_args  # noqa: E402

TIMING = {"target_ms": 50, "warmup_runs": 3, "benchmark_runs": 5}
TOLERANCE = {
    "fp32": {
        "function": "is_close",
        "atol": 0.001,
        "rtol": 0.001,
        "outputs": ["output"],
    },
    "fp16": {"function": "is_close", "atol": 0.05, "rtol": 0.01, "outputs": ["output"]},
}


def files():
    out = []
    for c in json.loads((HERE / "model_sweep_configs_dw.json").read_text()):
        prec = c["label"].split(" ", 1)[0]
        args = conv_args(
            c["cin"],
            c["h"],
            c["w"],
            c["cout"],
            c["kh"],
            c["kw"],
            stride=tuple(c["stride"]),
            pad=tuple(c["padding"]),
            dilation=tuple(c["dilation"]),
            groups=c["groups"],
        )
        if prec == "fp32":
            args = fp32(args)
        out.append(
            {
                "name": f"{c['label']} baseline",
                "operator": "aten.convolution.default",
                **TIMING,
                "validation_method": TOLERANCE[prec],
                "route_sweep": ["auto", "b1x1"],
                "args_sweep": [args],
            }
        )
    return {HERE / "dw_b1x1_performance.json": out}


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
