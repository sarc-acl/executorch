#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Writes the case files of benchmark route im2col_coopmat (k x k FP16
convolution as the im2col gather and an _i2c cooperative-matrix variant):

    im2col_coopmat_correctness.json       every _i2c variant on the
                                          correctness cases, with the routes
                                          auto and im2col as baselines
    im2col_coopmat_expected_skips.json    {case: {variant: phrase}} of the
                                          variant rows that must SKIP
    im2col_coopmat_performance.json       the surveyed k x k layers on auto,
                                          direct, im2col and the measured
                                          _i2c variants

Run with --check to fail if a file is out of date.
"""

import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
from format_json import format_value  # noqa: E402

YAML = (
    HERE.parents[6] / "runtime/graph/ops/glsl/convolution/conv2d/conv2d_pw_coopmat.yaml"
)
TOLERANCE = {"function": "is_close", "atol": 0.1, "rtol": 0.01, "outputs": ["output"]}
VALIDATION_TIMING = {"target_ms": 5, "warmup_runs": 1, "benchmark_runs": 1}
PERFORMANCE_TIMING = {"target_ms": 50, "warmup_runs": 3, "benchmark_runs": 5}
K_STEP = {"xe2": 16, "amd": 32, "nvidia": 16}
OPERATORS = {
    "conv": "aten.convolution.default",
    "clamp": "et_vk.conv_with_clamp.default",
    "epilogue": "et_vk.conv2d.with_epilogue",
}
# Phrases of the construction-time refusals (Conv2dRoutes.cpp, Conv2dGemm.cpp).
SKIP_PHRASE = {
    "dilation": "dilation is not 1",
    "groups": "im2col_coopmat requires groups 1",
    "alc": "not a multiple of the variant's K step",
    "sk": "fewer than two K steps",
    "not_i2c": "does not read an im2col matrix",
    "unknown": "Unknown conv2d_pw_coopmat variant",
    "limit": "maximum 3D image dimension",
    "gather": "maxComputeWorkGroupCount",
}
# maxComputeWorkGroupCount.y of every device here; the gather runs 16 x 4 work
# groups over (k4, tile pixel), and a tile holds the output rows that fit a
# 16 MiB FP16 scratch (conv2d_gemm_impl).
MAX_GROUPS_Y = 65535
SCRATCH_BUDGET = 16 * 1024 * 1024
# K / 4 texel layers above this exceed maxImageDimension3D of every device here.
MAX_3D_LAYERS = 16384


def i2c_variants():
    shader = yaml.safe_load(YAML.read_text())["conv2d_pw_coopmat"]
    return [v["NAME"] for v in shader["shader_variants"] if "_i2c" in v["NAME"]]


def tensor(name, sizes, data, storage="texture3d"):
    t = {"name": name, "sizes": sizes, "dtype": "float16", "storage": storage}
    if storage == "texture3d":
        t["layout"] = "channels"
    t["data"] = data
    return {"tensor": t}


def out_size(n, k, stride, pad, dilation):
    return (n + 2 * pad - dilation * (k - 1) - 1) // stride + 1


def conv_args(
    cin,
    h,
    w,
    cout,
    kh,
    kw,
    stride=1,
    pad=(0, 0),
    dilation=1,
    groups=1,
    bias=True,
    op="conv",
    clamp=None,
    epilogue=None,
    data=None,
):
    """`data` replaces the initializer of the named tensors (input, weight,
    bias, residual)."""
    ho = out_size(h, kh, stride, pad[0], dilation)
    wo = out_size(w, kw, stride, pad[1], dilation)
    s = float(f"{1 / (cin // groups * kh * kw) ** 0.5:.4g}")
    data = {
        "input": {"rand_uniform_float": [-1, 1]},
        "weight": {"rand_uniform_float": [-s, s]},
        "bias": {"rand_uniform_float": [-1, 1]},
        "residual": {"rand_uniform_float": [-1, 1]},
        **(data or {}),
    }
    args = [
        tensor("input", [1, cin, h, w], data["input"]),
        tensor("weight", [cout, cin // groups, kh, kw], data["weight"], "constant"),
        tensor("bias", [cout], data["bias"], "constant") if bias else None,
        {"int_list": [stride, stride]},
        {"int_list": list(pad)},
        {"int_list": [dilation, dilation]},
        {"bool": False},
        {"int_list": [0, 0]},
        {"int": groups},
    ]
    if op == "clamp":
        args += [{"float": clamp[0]}, {"float": clamp[1]}]
    elif op == "epilogue":
        residual_op, alpha, activation = epilogue
        args += [
            tensor("residual", [1, cout, ho, wo], data["residual"]),
            None,
            {"string": residual_op},
            {"float": alpha},
            {"string": activation},
            {"float": 0.0},
            {"float": 0.0},
        ]
    args.append(tensor("output", [1, cout, ho, wo], {"fill": 0}))
    return args


def k_total(c):
    return c["kh"] * c["kw"] * -(-(c["cin"] // c.get("groups", 1)) // 4) * 4


def split_k_ok(K, k_step, slices):
    steps = -(-K // k_step)
    q = -(-steps // slices)
    return q >= 2 and steps - (slices - 1) * q >= 2


def gather_groups(c):
    """(x, y) work groups of the im2col gather of one row tile."""
    ho = out_size(c["h"], c["kh"], c.get("stride", 1), c.get("pad", (0, 0))[0], 1)
    wo = out_size(c["w"], c["kw"], c.get("stride", 1), c.get("pad", (0, 0))[1], 1)
    K = k_total(c)
    oh_tile = min(ho, max(1, SCRATCH_BUDGET // (wo * K * 2)))
    return -(-K // 4 // 16), -(-(wo * oh_tile) // 4)


def expected_skip(variant, c):
    """The refusal phrase of an im2col_coopmat row that must SKIP, or None."""
    if c.get("groups", 1) != 1:
        return SKIP_PHRASE["groups"]
    if c.get("dilation", 1) != 1:
        return SKIP_PHRASE["dilation"]
    tokens = variant.removeprefix("conv2d_pw_coopmat_").removesuffix("_half").split("_")
    k_step = K_STEP[tokens[0]]
    K = k_total(c)
    if K // 4 > MAX_3D_LAYERS:
        return SKIP_PHRASE["limit"]
    if gather_groups(c)[1] > MAX_GROUPS_Y:
        return SKIP_PHRASE["gather"]
    if "alc" in tokens and K % k_step:
        return SKIP_PHRASE["alc"]
    slices = next((int(t[2:]) for t in tokens if t.startswith("sk")), 1)
    if slices > 1 and not split_k_ok(K, k_step, slices):
        return SKIP_PHRASE["sk"]
    return None


def conv(cin, h, w, cout, kh, kw, **kwargs):
    return {"cin": cin, "h": h, "w": w, "cout": cout, "kh": kh, "kw": kw, **kwargs}


# Spec scenarios of change conv2d-gemm-coopmat, plus K not a multiple of 16
# (Cin 20: K 180; Cin 4: K 36) and K steps too few for split-K (K 48).
CORRECTNESS = [
    ("3x3 256 14x14", conv(256, 14, 14, 256, 3, 3, pad=(1, 1))),
    ("3x3 512 7x7", conv(512, 7, 7, 512, 3, 3, pad=(1, 1))),
    ("3x3 s2 128 28x28", conv(128, 28, 28, 128, 3, 3, stride=2, pad=(1, 1))),
    ("1x7 192 17x17", conv(192, 17, 17, 224, 1, 7, pad=(0, 3))),
    ("7x1 192 17x17", conv(192, 17, 17, 192, 7, 1, pad=(3, 0))),
    ("1x3 384 8x8", conv(384, 8, 8, 256, 1, 3, pad=(0, 1))),
    ("3x1 384 8x8", conv(384, 8, 8, 256, 3, 1, pad=(1, 0))),
    ("3x3 Cin30", conv(30, 14, 14, 64, 3, 3, pad=(1, 1))),
    ("3x3 Cin20 K180", conv(20, 14, 14, 48, 3, 3, pad=(1, 1))),
    ("1x3 Cin16 K48", conv(16, 14, 14, 64, 1, 3, pad=(0, 1))),
    ("3x3 Cin4 K36", conv(4, 14, 14, 32, 3, 3, pad=(1, 1))),
    ("3x3 N48 nobias", conv(192, 14, 14, 48, 3, 3, pad=(1, 1), bias=False)),
    (
        "3x3 clamp",
        conv(64, 14, 14, 64, 3, 3, pad=(1, 1), op="clamp", clamp=(0.0, 0.5)),
    ),
    (
        "3x3 residual add relu",
        conv(
            128,
            14,
            14,
            96,
            3,
            3,
            pad=(1, 1),
            bias=False,
            op="epilogue",
            epilogue=("add", 1.0, "relu"),
        ),
    ),
    # K = 576, W_out = 128: 113 output rows per 16 MiB scratch tile.
    ("3x3 two row tiles", conv(64, 128, 128, 40, 3, 3, pad=(1, 1))),
    ("3x3 dilation 2", conv(64, 14, 14, 64, 3, 3, pad=(2, 2), dilation=2)),
    ("3x3 groups 2", conv(64, 14, 14, 64, 3, 3, pad=(1, 1), groups=2)),
    # K = 65664: 16416 texel layers of the texture3d gather.
    ("3x3 K above 3D limit", conv(7296, 4, 4, 16, 3, 3, pad=(1, 1))),
    # One 512 x 1024 tile (K = 12): 131072 gather work groups in y.
    ("1x3 gather grid limit", conv(4, 512, 1024, 4, 1, 3, pad=(0, 1))),
]


def center_tap_weight(cout, cin, value, channels, taps=9):
    """3x3 (or 1x3, taps 3) weights that are VALUE on the center tap of the
    first `channels` input channels and 0 elsewhere: with padding 1 every
    output pixel takes exactly those products, so the 1x1 cancellations carry
    over."""
    return {
        "repeat": [
            value if k == taps // 2 and ci < channels else 0
            for _ in range(cout)
            for ci in range(cin)
            for k in range(taps)
        ]
    }


def ordered_input(cin, pixels):
    """Per pixel, in channel order: 3C/8 ones, C/4 values of 2^-8, 3C/8 minus
    ones. The exact sum with unit weights is C/1024; an fp16 running sum stays
    at 3C/8 while the small terms are added (a K step of 32 adds 2^-3, below
    half the fp16 spacing at 768) and ends at 0."""
    run = 3 * cin // 8
    per_c = [1.0] * run + [2.0**-8] * (cin // 4) + [-1.0] * run
    return {"repeat": [v for v in per_c for _ in range(pixels)]}


# Precision regressions of the fp32-accumulation pointwise variants, adapted to
# a 3x3 (ordered cancellation: 1x3) padding-1 convolution on a 2x3 plane
# (center tap only). An fp16
# rounding before the bias, the residual or the end of the K loop makes them
# fail; fp32 accumulation must pass. Swept over the fp32 _i2c variants only.
PRECISION = [
    # 65 * 65 = 4225 rounds to 4224 in fp16; 4225 - 4224 = 1.
    (
        "fp32 precision bias cancellation",
        conv(
            64,
            2,
            3,
            16,
            3,
            3,
            pad=(1, 1),
            data={
                "input": {"fill": 65},
                "weight": center_tap_weight(16, 64, 65, 1),
                "bias": {"fill": -4224},
            },
        ),
    ),
    # The same cancellation against the fused residual.
    (
        "fp32 precision residual cancellation",
        conv(
            64,
            2,
            3,
            16,
            3,
            3,
            pad=(1, 1),
            bias=False,
            op="epilogue",
            epilogue=("add", 1.0, "none"),
            data={
                "input": {"fill": 65},
                "weight": center_tap_weight(16, 64, 65, 1),
                "residual": {"fill": -4224},
            },
        ),
    ),
    # Alpha 1 + 2^-12 rounds to 1 in fp16: 64 * 64 - alpha * 4096 = -1 exactly.
    (
        "fp32 precision residual alpha",
        conv(
            64,
            2,
            3,
            16,
            3,
            3,
            pad=(1, 1),
            bias=False,
            op="epilogue",
            epilogue=("add", 1.000244140625, "none"),
            data={
                "input": {"fill": 64},
                "weight": center_tap_weight(16, 64, 64, 1),
                "residual": {"fill": -4096},
            },
        ),
    ),
    # Ordered cancellation over 2048 channels, 1x3 center tap (K = 6144, K / 4
    # within every device's 3D image limit): exact result 2, fp16 gives 0.
    (
        "fp32 precision ordered cancellation",
        conv(
            2048,
            2,
            3,
            16,
            1,
            3,
            pad=(0, 1),
            bias=False,
            data={
                "input": ordered_input(2048, 6),
                "weight": center_tap_weight(16, 2048, 1, 2048, taps=3),
            },
        ),
    ),
]


def fp32_variants(variants):
    return [v for v in variants if "_f32" in v]


UNKNOWN = [
    "conv2d_pw_coopmat_xe2_t128x128_f16_alc_half",
    "conv2d_pw_coopmat_xe2_t999x999_f16_i2c_half",
]


def case_kwargs(c):
    return {
        k: v for k, v in c.items() if k not in ("cin", "h", "w", "cout", "kh", "kw")
    }


def cases(label, c, baselines, variants, timing):
    args = conv_args(
        c["cin"], c["h"], c["w"], c["cout"], c["kh"], c["kw"], **case_kwargs(c)
    )
    base = {
        "operator": OPERATORS[c.get("op", "conv")],
        **timing,
        "validation_method": TOLERANCE,
    }
    name = f"i2c {label}"
    out = [
        {
            "name": f"{name} baseline",
            **base,
            "route_sweep": baselines,
            "args_sweep": [args],
        }
    ]
    if variants:
        out.append(
            {
                "name": f"{name} coopmat",
                **base,
                "route": {
                    "name": "im2col_coopmat",
                    "options": {"variant": "v_VARIANT"},
                },
                "sweep_variables": {"v_VARIANT": variants},
                "args_sweep": [args],
            }
        )
    return out


def correctness(variants):
    out = []
    for label, c in CORRECTNESS:
        if k_total(c) // 4 > MAX_3D_LAYERS:
            # Refusal check only: im2col cannot pack these weights into a 2D
            # image, and auto (direct) exceeds the fp16 tolerance at this K.
            out += cases(label, c, [], variants, VALIDATION_TIMING)[1:]
            continue
        # im2col runs neither groups nor a gather beyond the work group limit.
        baselines = (
            ["auto"]
            if c.get("groups", 1) != 1 or gather_groups(c)[1] > MAX_GROUPS_Y
            else ["auto", "im2col"]
        )
        out += cases(label, c, baselines, variants, VALIDATION_TIMING)
    label, c = CORRECTNESS[0]
    out += cases(f"{label} unknown", c, [], UNKNOWN, VALIDATION_TIMING)[1:]
    for label, c in PRECISION:
        out += cases(label, c, [], fp32_variants(variants), VALIDATION_TIMING)[1:]
    return out


def expected_skips(variants):
    out = {}
    for label, c in CORRECTNESS:
        skips = {v: p for v in variants if (p := expected_skip(v, c))}
        if skips:
            out[f"i2c {label} coopmat"] = skips
    for label, c in PRECISION:
        skips = {v: p for v in fp32_variants(variants) if (p := expected_skip(v, c))}
        if skips:
            out[f"i2c {label} coopmat"] = skips
    out[f"i2c {CORRECTNESS[0][0]} unknown coopmat"] = {
        UNKNOWN[0]: SKIP_PHRASE["not_i2c"],
        UNKNOWN[1]: SKIP_PHRASE["unknown"],
    }
    return out


# The surveyed layers of change conv2d-gemm-coopmat design Context, as the
# models run them (padding 1 for 3x3, (0, 3) for 1x7).
PERFORMANCE = [
    ("3x3 512 7x7", conv(512, 7, 7, 512, 3, 3, pad=(1, 1))),
    ("3x3 256 14x14", conv(256, 14, 14, 256, 3, 3, pad=(1, 1))),
    ("3x3 128 28x28", conv(128, 28, 28, 128, 3, 3, pad=(1, 1))),
    ("3x3 64 56x56", conv(64, 56, 56, 64, 3, 3, pad=(1, 1))),
    ("1x7 192-224 17x17", conv(192, 17, 17, 224, 1, 7, pad=(0, 3))),
    ("3x3 448-384 5x5", conv(448, 5, 5, 384, 3, 3, pad=(1, 1))),
    ("3x3 s2 512 14x14", conv(512, 14, 14, 512, 3, 3, stride=2, pad=(1, 1))),
    ("3x3 192-48 14x14", conv(192, 14, 14, 48, 3, 3, pad=(1, 1))),
]
# The best variants of the B580 probe (OUT/b580/probe.md) and their split-K
# forms.
PERFORMANCE_VARIANTS = [
    "conv2d_pw_coopmat_xe2_t128x128_f16_alc_i2c_half",
    "conv2d_pw_coopmat_xe2_t128x128_f16_alc_sk2_i2c_half",
    "conv2d_pw_coopmat_xe2_t128x128_f16_alc_sk4_i2c_half",
    "conv2d_pw_coopmat_xe2_t32x64_f16_g4x2_alc_i2c_half",
    "conv2d_pw_coopmat_xe2_t32x64_f16_g4x2_alc_sk2_i2c_half",
    "conv2d_pw_coopmat_xe2_t32x64_f16_g4x2_alc_sk4_i2c_half",
]


def performance(variants):
    measured = [v for v in variants if v in PERFORMANCE_VARIANTS or "_xe2_" not in v]
    out = []
    for label, c in PERFORMANCE:
        out += cases(
            label, c, ["auto", "direct", "im2col"], measured, PERFORMANCE_TIMING
        )
    return out


FILES = {
    "im2col_coopmat_correctness.json": correctness,
    "im2col_coopmat_expected_skips.json": expected_skips,
    "im2col_coopmat_performance.json": performance,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="fail if a file is out of date"
    )
    args = parser.parse_args()
    variants = i2c_variants()
    stale = []
    for filename, build in FILES.items():
        path = HERE / filename
        text = format_value(build(variants)) + "\n"
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(filename)
        else:
            path.write_text(text)
    print(f"{len(variants)} _i2c variants")
    if stale:
        sys.exit("out of date: " + ", ".join(stale))


if __name__ == "__main__":
    main()
