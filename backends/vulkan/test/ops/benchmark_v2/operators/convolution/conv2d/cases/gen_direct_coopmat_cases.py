#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Writes the case files of benchmark route direct_coopmat (FP16 convolution
as one cooperative-matrix multiply that gathers its input, _dg variants;
change conv2d-coopmat-direct-gather):

    direct_coopmat_correctness.json       every _dg variant on the cases of
                                          the im2col_coopmat correctness and
                                          dilated collections and on the
                                          shapes only this route accepts
    direct_coopmat_expected_skips.json    {case: {variant: phrase}} of the
                                          variant rows that must SKIP

Run with --check to fail if a file is out of date.
"""

import argparse
import sys
from pathlib import Path

import gen_im2col_coopmat_cases as i2c
from gen_im2col_coopmat_cases import conv

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
from format_json import format_value  # noqa: E402

SKIP_PHRASE = {
    "dead": "no kernel tap reads the input",
    "groups": "direct_coopmat requires groups 1",
    "sk": "fewer than two K steps",
    "not_dg": "does not gather its input",
    "unknown": "Unknown conv2d_pw_coopmat variant",
    "int32": "computes in 32-bit integers and cannot represent the",
}

# Shapes the im2col route refuses for shape reasons (model-switch REPORT.md:
# one-pixel output, input channels or K not a multiple of the K step, 1x1
# with stride or padding), and channel counts not a multiple of four.
NEW = [
    ("1x1 64-96 one pixel", conv(64, 1, 1, 96, 1, 1)),
    ("3x3 p0 256-64 3x3 one pixel", conv(256, 3, 3, 64, 3, 3)),
    ("1x1 s2 64-128 28x28", conv(64, 28, 28, 128, 1, 1, stride=2)),
    ("1x1 s2 Cin3 15x15", conv(3, 15, 15, 24, 1, 1, stride=2)),
    ("1x1 p1 64-32 14x14", conv(64, 14, 14, 32, 1, 1, pad=(1, 1))),
    ("1x1 Cin120-40 14x14", conv(120, 14, 14, 40, 1, 1)),
    ("1x1 Cin24-144 28x28", conv(24, 28, 28, 144, 1, 1)),
    ("3x3 s2 Cin3 33x33", conv(3, 33, 33, 32, 3, 3, stride=2, pad=(1, 1))),
    ("7x7 s2 Cin3 64x64", conv(3, 64, 64, 64, 7, 7, stride=2, pad=(3, 3))),
    ("3x3 Cin24 20x20", conv(24, 20, 20, 40, 3, 3, pad=(1, 1))),
    ("3x3 Cin3-64 asymmetric padding", conv(3, 13, 17, 64, 3, 3, pad=(2, 0))),
    ("5x5 Cin48-64 25x25", conv(48, 25, 25, 64, 5, 5, pad=(2, 2))),
]


def dg_variants():
    return [n for n in i2c.yaml_names() if n.endswith("_dg_half")]


UNKNOWN = [
    "conv2d_pw_coopmat_xe2_t128x128_f16_alc_half",
    "conv2d_pw_coopmat_xe2_t999x999_f16_alc_dg_half",
]


def live_k(c):
    """K of the live taps (conv2d_direct_coopmat_plan)."""
    st, pad, dl = (
        i2c.strides(c.get("stride", 1)),
        c.get("pad", (0, 0)),
        i2c.dilations(c),
    )
    taps = 1
    for n, k, s, p, d in (
        (c["h"], c["kh"], st[0], pad[0], dl[0]),
        (c["w"], c["kw"], st[1], pad[1], dl[1]),
    ):
        taps *= i2c.live_taps(n, i2c.out_size(n, k, s, p, d), k, s, p, d)
    return taps * -(-(c["cin"] // c.get("groups", 1)) // 4) * 4


def expected_skip(variant, c):
    """The refusal phrase of a direct_coopmat row that must SKIP, or None."""
    if c.get("groups", 1) != 1:
        return SKIP_PHRASE["groups"]
    if i2c.int32_overflow(c):
        return SKIP_PHRASE["int32"]
    K = live_k(c)
    if K == 0:
        return SKIP_PHRASE["dead"]
    tokens = variant.removeprefix("conv2d_pw_coopmat_").removesuffix("_half").split("_")
    slices = next((int(t[2:]) for t in tokens if t.startswith("sk")), 1)
    if slices > 1 and not i2c.split_k_ok(K, i2c.K_STEP[tokens[0]], slices):
        return SKIP_PHRASE["sk"]
    return None


def cases(label, c, baselines, variants):
    args = i2c.conv_args(
        c["cin"], c["h"], c["w"], c["cout"], c["kh"], c["kw"], **i2c.case_kwargs(c)
    )
    base = {
        "operator": i2c.OPERATORS[c.get("op", "conv")],
        **i2c.VALIDATION_TIMING,
        "validation_method": i2c.TOLERANCE,
    }
    name = f"dg {label}"
    out = []
    if baselines:
        out.append(
            {
                "name": f"{name} baseline",
                **base,
                "route_sweep": baselines,
                "args_sweep": [args],
            }
        )
    if variants:
        out.append(
            {
                "name": f"{name} coopmat",
                **base,
                "route": {
                    "name": "direct_coopmat",
                    "options": {"variant": "v_VARIANT"},
                },
                "sweep_variables": {"v_VARIANT": variants},
                "args_sweep": [args],
            }
        )
    return out


def groups():
    """(label, case, baselines, fp32 variants only) of every case, in file
    order: the baselines of the im2col collections where they have them."""
    out = []
    for label, c in i2c.CORRECTNESS:
        if i2c.k_total(c) // 4 > i2c.MAX_3D_LAYERS or i2c.int32_overflow(c):
            baselines = []
        elif c.get("groups", 1) != 1 or i2c.gather_groups(c)[1] > i2c.MAX_GROUPS_Y:
            baselines = ["auto"]
        else:
            baselines = ["auto", "im2col"]
        out.append((label, c, baselines, False))
    for label, c in i2c.PRECISION:
        out.append((label, c, [], True))
    for label, c in i2c.DILATED + i2c.DILATED_CHUNKED + NEW:
        out.append((label, c, ["auto"], False))
    return out


def correctness(variants):
    out = []
    for label, c, baselines, fp32 in groups():
        out += cases(
            label, c, baselines, i2c.fp32_variants(variants) if fp32 else variants
        )
    label, c = i2c.CORRECTNESS[0]
    out += cases(f"{label} unknown", c, [], UNKNOWN)
    return out


def expected_skips(variants):
    out = {}
    for label, c, _, fp32 in groups():
        vs = i2c.fp32_variants(variants) if fp32 else variants
        skips = {v: p for v in vs if (p := expected_skip(v, c))}
        if skips:
            out[f"dg {label} coopmat"] = skips
    out[f"dg {i2c.CORRECTNESS[0][0]} unknown coopmat"] = {
        UNKNOWN[0]: SKIP_PHRASE["not_dg"],
        UNKNOWN[1]: SKIP_PHRASE["unknown"],
    }
    return out


FILES = {
    "direct_coopmat_correctness.json": correctness,
    "direct_coopmat_expected_skips.json": expected_skips,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="fail if a file is out of date"
    )
    args = parser.parse_args()
    variants = dg_variants()
    stale = []
    for filename, build in FILES.items():
        path = HERE / filename
        text = format_value(build(variants)) + "\n"
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(filename)
        else:
            path.write_text(text)
    print(f"{len(variants)} _dg variants")
    if stale:
        sys.exit("out of date: " + ", ".join(stale))


if __name__ == "__main__":
    main()
