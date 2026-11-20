#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kernel-level sweep of the convolution layers of the model survey (study
change conv2d-coopmat-model-switch):

    model_sweep_configs.json      the swept configurations: every distinct FP16,
                                  batch-1, ungrouped, non-transposed 2D
                                  convolution of the survey that has a survey
                                  row with at least MIN_SHARE_PCT of its
                                  model's B580 GPU shader time or that an
                                  earlier change measured, with every survey
                                  row of that configuration
    model_sweep_performance.json  per configuration route auto, the short
                                  variant list on pw_coopmat (1x1) or
                                  im2col_coopmat (k x k) and its direct-gather
                                  forms on direct_coopmat (change
                                  conv2d-coopmat-direct-gather)
    model_sweep_arms.json         the short variant list per device vendor

    gen_model_sweep_cases.py --survey survey-conv-shapes.csv --measured m.json
        selects the configurations from the survey (m.json: [{"key": [cin, h,
        w, cout, kh, kw, sh, sw, ph, pw, dh, dw]}, ...]) and writes all files
    gen_model_sweep_cases.py [--check]
        rewrites (or checks) the other files from model_sweep_configs.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
from format_json import format_value  # noqa: E402
from gen_im2col_coopmat_cases import (  # noqa: E402
    conv_args,
    PERFORMANCE_TIMING,
    TOLERANCE,
)

MIN_SHARE_PCT = 0.25
# The direct shader of route auto accumulates in FP16 and misses atol 0.1 on
# deep layers (change conv2d-dilated-coopmat); the variants keep TOLERANCE.
BASELINE_TOLERANCE = {**TOLERANCE, "atol": 0.25}
CONFIGS = HERE / "model_sweep_configs.json"
# Base names of the timed variants; route pw_coopmat runs
# conv2d_pw_coopmat_<base>_half, route im2col_coopmat <base>_i2c_half, route
# direct_coopmat <base>_dg_half.
ARMS = {
    "intel": [
        "xe2_t128x128_f16_alc",
        "xe2_t128x128_f16_alc_sk2",
        "xe2_t128x128_f16_alc_sk4",
        "xe2_t32x64_f16_g4x2_alc",
        "xe2_t32x64_f16_g4x2_alc_sk2",
        "xe2_t32x64_f16_g4x2_alc_sk4",
    ],
    "amd": [
        "amd_t16x32_f16_alc",
        "amd_t16x32_f16_alc_sk2",
        "amd_t128x128_f32_alc",
        "amd_t128x128_f32_alc_sk2",
        "nvidia_t16x32_f16_alc",
    ],
    "nvidia": [
        "nvidia_t16x32_f16_alc",
        "nvidia_t16x32_f16_alc_sk2",
        "nvidia_t128x128_f32_alc",
        "nvidia_t128x128_f32_alc_sk2",
    ],
}


def pair(text):
    return [int(x) for x in text.split("x")]


def key(c):
    return [c["cin"], c["h"], c["w"], c["cout"], c["kh"], c["kw"]] + (
        c["stride"] + c["padding"] + c["dilation"]
    )


def label(c):
    s, p, d = ("x".join(map(str, c[k])) for k in ("stride", "padding", "dilation"))
    return (
        f"{c['kh']}x{c['kw']} s{s} p{p} d{d} {c['cin']}-{c['cout']} "
        f"{c['h']}x{c['w']}"
    )


def route(c):
    return "pw_coopmat" if c["kh"] == c["kw"] == 1 else "im2col_coopmat"


def variant_name(base, route_name):
    suffix = {"im2col_coopmat": "_i2c", "direct_coopmat": "_dg"}.get(route_name, "")
    return f"conv2d_pw_coopmat_{base}{suffix}_half"


def select(survey, measured):
    """The distinct eligible survey configurations that have a survey row with
    at least MIN_SHARE_PCT, or that an earlier change measured (an earlier
    benchmark shape outside the survey is not a survey configuration and is not
    swept), each with every eligible survey row of that exact configuration."""
    earlier = {tuple(m["key"]) for m in measured}
    configs = {}
    for r in csv.DictReader(open(survey)):
        if not (
            r["precision"] == "fp16"
            and r["dims"] == "2"
            and r["batch"] == "1"
            and r["groups"] == "1"
            and r["transposed"] == "0"
        ):
            continue
        c = {
            "cin": int(r["cin"]),
            "h": int(r["hin"]),
            "w": int(r["win"]),
            "cout": int(r["cout"]),
            "kh": int(r["kh"]),
            "kw": int(r["kw"]),
            "stride": pair(r["stride"]),
            "padding": pair(r["padding"]),
            "dilation": pair(r["dilation"]),
        }
        entry = configs.setdefault(label(c), {"label": label(c), **c, "layers": []})
        entry["layers"].append(
            {
                "config_id": r["config_id"],
                "instances": int(r["instances"]),
                "b580_share_pct": float(r["b580_share_pct"] or 0),
            }
        )
    selected = []
    for c in configs.values():
        if any(x["b580_share_pct"] >= MIN_SHARE_PCT for x in c["layers"]):
            c["selected_by"] = "share"
        elif tuple(key(c)) in earlier:
            c["selected_by"] = "earlier"
        else:
            continue
        selected.append(c)
    return sorted(selected, key=lambda c: c["label"])


def performance(configs):
    variants = sorted({v for arms in ARMS.values() for v in arms})
    out = []
    for c in configs:
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
        )
        base = {"operator": "aten.convolution.default", **PERFORMANCE_TIMING}
        out.append(
            {
                "name": f"sweep {c['label']} baseline",
                **base,
                "validation_method": BASELINE_TOLERANCE,
                "route_sweep": ["auto"],
                "args_sweep": [args],
            }
        )
        out.append(
            {
                "name": f"sweep {c['label']} coopmat",
                **base,
                "validation_method": TOLERANCE,
                "route": {"name": route(c), "options": {"variant": "v_VARIANT"}},
                "sweep_variables": {
                    "v_VARIANT": [variant_name(v, route(c)) for v in variants]
                },
                "args_sweep": [args],
            }
        )
        out.append(
            {
                "name": f"sweep {c['label']} direct",
                **base,
                "validation_method": TOLERANCE,
                "route": {
                    "name": "direct_coopmat",
                    "options": {"variant": "v_VARIANT"},
                },
                "sweep_variables": {
                    "v_VARIANT": [variant_name(v, "direct_coopmat") for v in variants]
                },
                "args_sweep": [args],
            }
        )
    return out


def arms():
    return {
        vendor: {
            r: [variant_name(v, r) for v in bases]
            for r in ("pw_coopmat", "im2col_coopmat", "direct_coopmat")
        }
        for vendor, bases in ARMS.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey", type=Path)
    parser.add_argument("--measured", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.survey:
        configs = select(args.survey, json.loads(args.measured.read_text()))
        CONFIGS.write_text(format_value(configs) + "\n")
    configs = json.loads(CONFIGS.read_text())
    files = {
        "model_sweep_performance.json": performance(configs),
        "model_sweep_arms.json": arms(),
    }
    stale = []
    for name, value in files.items():
        text = format_value(value) + "\n"
        if args.check:
            if (HERE / name).read_text() != text:
                stale.append(name)
        else:
            (HERE / name).write_text(text)
    print(f"{len(configs)} configurations")
    if stale:
        sys.exit("out of date: " + ", ".join(stale))


if __name__ == "__main__":
    main()
