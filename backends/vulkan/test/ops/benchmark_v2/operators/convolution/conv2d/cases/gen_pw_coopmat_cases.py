# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Writes the pw_coopmat case files beside this script from the variant list of
conv2d_pw_coopmat.yaml, so the cases stay in step with the shipped variants.

Each shape gets two definitions: the tiled baselines (routes auto and
pw_tiled_4x8) and route pw_coopmat swept over every variant with
sweep_variables, the variants of each family in yaml order. A device runs the
variants it offers and reports the others as SKIP. Inputs are uniform in
[-1, 1], weights in [-1/sqrt(K), 1/sqrt(K)] (with weights in [-1, 1] the FP16
tiled baseline itself misses the tolerance at K = 768), bias and residual in
[-1, 1]; FP16 tolerance atol 0.1, rtol 0.01.

Usage: python3 gen_pw_coopmat_cases.py [--check]
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
from format_json import format_value  # noqa: E402

YAML = (
    HERE.parents[6] / "runtime/graph/ops/glsl/convolution/conv2d/conv2d_pw_coopmat.yaml"
)
FAMILIES = ("xe2", "amd", "nvidia")
TOLERANCE = {"function": "is_close", "atol": 0.1, "rtol": 0.01}


def variants_by_family():
    shader = yaml.safe_load(YAML.read_text())["conv2d_pw_coopmat"]
    names = {family: [] for family in FAMILIES}
    for variant in shader["shader_variants"]:
        name = variant["NAME"]
        family = name.split("_")[3]
        if family not in names:
            raise ValueError(f"{name}: unknown family {family}")
        names[family].append(name)
    return names


def tensor(name, sizes, data, storage="texture3d"):
    t = {"name": name, "sizes": sizes, "dtype": "float16", "storage": storage}
    if storage == "texture3d":
        t["layout"] = "channels"
    t["data"] = data
    return {"tensor": t}


def conv_args(
    M, N, K, plane, bias=True, op="conv", clamp=None, epilogue=None, data=None
):
    """Argument list of aten.convolution.default (op "conv"),
    et_vk.conv_with_clamp.default ("clamp") or et_vk.conv2d.with_epilogue
    ("epilogue") for an FP16 1x1 convolution with an H x W = plane output.
    `data` replaces the initializer of the named tensors."""
    H, W = plane
    assert H * W == M
    s = float(f"{1 / K ** 0.5:.4g}")
    data = {
        "input": {"rand_uniform_float": [-1, 1]},
        "weight": {"rand_uniform_float": [-s, s]},
        "bias": {"rand_uniform_float": [-1, 1]},
        "residual": {"rand_uniform_float": [-1, 1]},
        **(data or {}),
    }
    args = [
        tensor("input", [1, K, H, W], data["input"]),
        tensor("weight", [N, K, 1, 1], data["weight"], "constant"),
        tensor("bias", [N], data["bias"], "constant") if bias else None,
        {"int_list": [1, 1]},
        {"int_list": [0, 0]},
        {"int_list": [1, 1]},
        {"bool": False},
        {"int_list": [0, 0]},
        {"int": 1},
    ]
    if op == "clamp":
        args += [{"float": clamp[0]}, {"float": clamp[1]}]
    elif op == "epilogue":
        residual_op, alpha, scale, activation = epilogue
        args += [
            tensor("residual", [1, N, H, W], data["residual"]),
            (
                tensor(
                    "residual_scale",
                    [1, N, 1, 1],
                    {"rand_uniform_float": [0.5, 1.5]},
                    "constant",
                )
                if scale
                else None
            ),
            {"string": residual_op},
            {"float": alpha},
            {"string": activation},
            {"float": 0.0},
            {"float": 0.0},
        ]
    args.append(tensor("output", [1, N, H, W], {"fill": 0}))
    return args


OPERATORS = {
    "conv": "aten.convolution.default",
    "clamp": "et_vk.conv_with_clamp.default",
    "epilogue": "et_vk.conv2d.with_epilogue",
}


def shape_cases(label, M, N, K, plane, variants, timing, baselines=True, **kwargs):
    op = kwargs.get("op", "conv")
    base = {
        "operator": OPERATORS[op],
        **timing,
        "validation_method": TOLERANCE,
    }
    args = conv_args(M, N, K, plane, **kwargs)
    name = f"pw_coopmat {label} ({M},{N},{K})"
    baseline = {
        "name": f"{name} baseline",
        **base,
        "route_sweep": ["auto", "pw_tiled_4x8"],
        "args_sweep": [args],
    }
    return [baseline] * baselines + [
        {
            "name": f"{name} coopmat",
            **base,
            "route": {"name": "pw_coopmat", "options": {"variant": "v_VARIANT"}},
            "sweep_variables": {"v_VARIANT": variants},
            "args_sweep": [args],
        },
    ]


VALIDATION_TIMING = {"target_ms": 5, "warmup_runs": 1, "benchmark_runs": 1}


def cancellation_input(K, pixels):
    """Per pixel, in K order: 3K/8 ones, K/4 values of 2^-7, 3K/8 minus ones.
    The exact sum with unit weights is K/512; an fp16 running sum stays at 3K/8
    while the 2^-7 terms are added, below half its fp16 spacing, and ends at 0."""
    run = 3 * K // 8
    per_k = [1.0] * run + [2.0**-7] * (K // 4) + [-1.0] * run
    return {"repeat": [v for v in per_k for _ in range(pixels)]}


# Precision regressions (OUT/b580/review/precision.json of the review of
# 727c0f4da7). An fp16 rounding before the bias, the residual or the end of the
# K loop makes them fail; fp32 accumulation must pass every one. Products of
# fp16 operands are exact in fp32.
PRECISION = [
    # 65 * 65 = 4225 rounds to 4224 in fp16; 4225 - 4224 = 1.
    (
        "precision bias cancellation",
        6,
        1,
        1,
        (2, 3),
        {
            "data": {
                "input": {"fill": 65},
                "weight": {"fill": 65},
                "bias": {"fill": -4224},
            }
        },
    ),
    # The same cancellation against the fused residual: 4225 - 4224 = 1.
    (
        "precision residual cancellation",
        6,
        1,
        1,
        (2, 3),
        {
            "bias": False,
            "op": "epilogue",
            "epilogue": ("add", 1.0, False, "none"),
            "data": {
                "input": {"fill": 65},
                "weight": {"fill": 65},
                "residual": {"fill": -4224},
            },
        },
    ),
    # The residual alpha 1 + 2^-12 rounds to 1 in fp16: 64 * 64 - alpha * 4096
    # is -1 with the exact alpha and 0 with the rounded one.
    (
        "precision residual alpha",
        6,
        1,
        1,
        (2, 3),
        {
            "bias": False,
            "op": "epilogue",
            "epilogue": ("add", 1.000244140625, False, "none"),
            "data": {
                "input": {"fill": 64},
                "weight": {"fill": 64},
                "residual": {"fill": -4096},
            },
        },
    ),
    # Bounded operands, ordered cancellation over K = 2048: exact result 4.
    (
        "precision ordered cancellation",
        6,
        1,
        2048,
        (2, 3),
        {
            "bias": False,
            "data": {"input": cancellation_input(2048, 6), "weight": {"fill": 1}},
        },
    ),
]


def fp32_variants(variants):
    return [v for v in variants if "_f32_" in v]


def correctness(variants):
    cases = []
    for label, M, N, K, plane, kwargs in [
        ("aligned", 256, 128, 64, (16, 16), {}),
        ("7x7", 49, 96, 576, (7, 7), {}),
        ("14x14", 196, 64, 384, (14, 14), {}),
        ("N=24", 196, 24, 144, (14, 14), {}),
        ("K=40", 196, 120, 40, (14, 14), {}),
        ("no bias", 196, 96, 576, (14, 14), {"bias": False}),
        ("clamp", 49, 160, 960, (7, 7), {"op": "clamp", "clamp": [0.0, 6.0]}),
        (
            "residual add",
            196,
            32,
            192,
            (14, 14),
            {"op": "epilogue", "epilogue": ("add", 1.0, False, "none")},
        ),
        (
            "residual sub scale relu",
            49,
            24,
            40,
            (7, 7),
            {"op": "epilogue", "epilogue": ("sub", 0.5, True, "relu")},
        ),
    ]:
        cases += shape_cases(
            label, M, N, K, plane, variants, VALIDATION_TIMING, **kwargs
        )
    for label, M, N, K, plane, kwargs in PRECISION:
        cases += shape_cases(
            label,
            M,
            N,
            K,
            plane,
            fp32_variants(variants),
            VALIDATION_TIMING,
            baselines=False,
            **kwargs,
        )
    # 1025 x 1024 output pixels: 65600 tiles of 16 pixels, more than the 65535
    # work groups many devices allow in y.
    cases += shape_cases(
        "dispatch limit",
        1025 * 1024,
        8,
        8,
        (1025, 1024),
        variants,
        VALIDATION_TIMING,
        baselines=False,
    )
    return cases


def fp16_accumulation_limits(variants):
    """The precision regressions for the routes that round partial sums to fp16:
    the fp16-accumulation variants and both tiled shaders. They are expected to
    FAIL (design D4, spec "Accumulation precision"); kept out of the correctness
    collection so that it passes."""
    cases = []
    for label, M, N, K, plane, kwargs in PRECISION:
        cases += shape_cases(
            label.replace("precision", "fp16 limit"),
            M,
            N,
            K,
            plane,
            [v for v in variants if "_f16_" in v],
            VALIDATION_TIMING,
            **kwargs,
        )
    return cases


PERFORMANCE_TIMING = {"target_ms": 50, "warmup_runs": 3, "benchmark_runs": 5}


def plane_of(M):
    side = round(M**0.5)
    assert side * side == M
    return (side, side)


def performance(label, shapes):
    def build(variants):
        cases = []
        for M, N, K in shapes:
            cases += shape_cases(
                label, M, N, K, plane_of(M), variants, PERFORMANCE_TIMING
            )
        return cases

    return build


def upstream_fp16_shapes():
    """The FP16 1x1 shapes of pw_performance.json, in file order."""
    shapes = []
    for case in json.loads((HERE / "pw_performance.json").read_text()):
        for args in case["args_sweep"]:
            x, w, out = args[0]["tensor"], args[1]["tensor"], args[-1]["tensor"]
            if x["dtype"] != "float16":
                continue
            assert w["sizes"][2:] == [1, 1] and args[3]["int_list"] == [1, 1]
            M = out["sizes"][2] * out["sizes"][3]
            shapes.append((M, out["sizes"][1], x["sizes"][1]))
    return shapes


# (M, N, K) = (H_out * W_out, C_out, C_in), batch 1.
# MobileNetV2 and V3 Small FP16 trace shapes.
MOBILENET = [(49, 160, 960), (196, 64, 384), (196, 96, 576), (49, 96, 576)]
# The stride-1 1x1 layers of ResNet-50 at 224 x 224 input.
RESNET50 = [
    (3136, 64, 64),
    (3136, 256, 64),
    (3136, 64, 256),
    (3136, 128, 256),
    (784, 512, 128),
    (784, 128, 512),
    (784, 256, 512),
    (196, 1024, 256),
    (196, 256, 1024),
    (196, 512, 1024),
    (49, 2048, 512),
    (49, 512, 2048),
]

FILES = {
    "pw_coopmat_correctness.json": correctness,
    "pw_coopmat_fp16_accumulation_limits.json": fp16_accumulation_limits,
    "pw_coopmat_mobilenet.json": performance("mobilenet", MOBILENET),
    "pw_coopmat_resnet50.json": performance("resnet50", RESNET50),
    "pw_coopmat_large.json": performance("large", upstream_fp16_shapes()),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="fail if a file is out of date"
    )
    args = parser.parse_args()
    by_family = variants_by_family()
    variants = [name for family in FAMILIES for name in by_family[family]]
    stale = []
    for filename, build in FILES.items():
        path = HERE / filename
        text = format_value(build(variants)) + "\n"
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(filename)
        else:
            path.write_text(text)
    for family in FAMILIES:
        print(f"{family}: {len(by_family[family])} variants")
    if stale:
        sys.exit("out of date: " + ", ".join(stale))


if __name__ == "__main__":
    main()
