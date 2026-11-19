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
# Only the convolution's output is compared: a storage-image input variant
# (_img) binds the input read-write, which would make the input an output too.
TOLERANCE = {"function": "is_close", "atol": 0.1, "rtol": 0.01, "outputs": ["output"]}


def variants_by_family():
    shader = yaml.safe_load(YAML.read_text())["conv2d_pw_coopmat"]
    names = {family: [] for family in FAMILIES}
    for variant in shader["shader_variants"]:
        name = variant["NAME"]
        if name.endswith(("_diag_once_half", "_dlprobe_half", "_dlctl_half")):
            continue
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


K_STEP = {"xe2": 16, "amd": 32, "nvidia": 16}
# Phrases of the construction-time refusals in Conv2dPW.cpp, per variant kind.
SKIP_PHRASE = {
    "al": "not a multiple of the variant's",
    "alc": "not a multiple of the variant's K step",
    "sk": "fewer than two K steps",
    "epi0": "has a bias, activation, residual or scale",
}


def variant_kind(name):
    """Family, N tile, K step, option suffixes and K slices of a variant name
    conv2d_pw_coopmat_<family>_t<M>x<N>_<acc><options>_half."""
    tokens = name.removeprefix("conv2d_pw_coopmat_").removesuffix("_half").split("_")
    family, tile = tokens[0], tokens[1]
    options = set(tokens[3:])
    slices = next((int(t[2:]) for t in options if t.startswith("sk")), 1)
    return {
        "family": family,
        "tile_n": int(tile.split("x")[1]),
        "k_step": K_STEP[family],
        "options": options,
        "slices": slices,
    }


def split_k_ok(K, k_step, slices):
    """Every slice of ceil(steps / slices) K steps, the last one shorter, holds
    at least two K steps (design D6)."""
    steps = -(-K // k_step)
    q = -(-steps // slices)
    return q >= 2 and steps - (slices - 1) * q >= 2


def expected_skip(name, N, K, bias, op):
    """The refusal phrase when the variant kind must report the convolution as
    unsupported at graph construction, else None: aligned-only variants on
    channels that are not multiples of their K step and N tile, clamped-read
    aligned-only variants (_alc) on a K that is not a multiple of their K step, split-K
    variants on a K too short for their slices, _epi0 on a convolution with a
    bias or a fused epilogue."""
    kind = variant_kind(name)
    if "al" in kind["options"] and (K % kind["k_step"] or N % kind["tile_n"]):
        return SKIP_PHRASE["al"]
    if "alc" in kind["options"] and K % kind["k_step"]:
        return SKIP_PHRASE["alc"]
    if kind["slices"] > 1 and not split_k_ok(K, kind["k_step"], kind["slices"]):
        return SKIP_PHRASE["sk"]
    if "epi0" in kind["options"] and (bias or op != "conv"):
        return SKIP_PHRASE["epi0"]
    return None


def expected_skips(build):
    """{case name: {variant: phrase}} of the variant rows of a case file that
    must SKIP."""

    def skips(variants):
        out = {}
        for case in build(variants):
            names = case.get("sweep_variables", {}).get("v_VARIANT", [])
            if not names:
                continue
            args = case["args_sweep"][0]
            N, K = args[1]["tensor"]["sizes"][0], args[0]["tensor"]["sizes"][1]
            bias = args[2] is not None
            op = {v: k for k, v in OPERATORS.items()}[case["operator"]]
            row = {}
            for v in names:
                if phrase := expected_skip(v, N, K, bias, op):
                    row[v] = phrase
            if row:
                out[case["name"]] = row
        return out

    return skips


def first_channel_weight(N, K, value):
    """Weights [N, K, 1, 1] that are VALUE on input channel 0 and 0 elsewhere,
    so an aligned K keeps the single-product cancellations of PRECISION."""
    return {"repeat": [value if k == 0 else 0 for _ in range(N) for k in range(K)]}


# PRECISION with channel counts every aligned-only variant accepts (N = 64,
# K = 32 or 2048).
PRECISION_ALIGNED = [
    (
        "aligned precision bias cancellation",
        6,
        64,
        32,
        (2, 3),
        {
            "data": {
                "input": {"fill": 65},
                "weight": first_channel_weight(64, 32, 65),
                "bias": {"fill": -4224},
            }
        },
    ),
    (
        "aligned precision residual cancellation",
        6,
        64,
        32,
        (2, 3),
        {
            "bias": False,
            "op": "epilogue",
            "epilogue": ("add", 1.0, False, "none"),
            "data": {
                "input": {"fill": 65},
                "weight": first_channel_weight(64, 32, 65),
                "residual": {"fill": -4224},
            },
        },
    ),
    (
        "aligned precision residual alpha",
        6,
        64,
        32,
        (2, 3),
        {
            "bias": False,
            "op": "epilogue",
            "epilogue": ("add", 1.000244140625, False, "none"),
            "data": {
                "input": {"fill": 64},
                "weight": first_channel_weight(64, 32, 64),
                "residual": {"fill": -4096},
            },
        },
    ),
    (
        "aligned precision ordered cancellation",
        6,
        64,
        2048,
        (2, 3),
        {
            "bias": False,
            "data": {"input": cancellation_input(2048, 6), "weight": {"fill": 1}},
        },
    ),
]

# Smallest K that every split-K variant splits: K = 225 gives 8 K steps of 32
# (AMD, two per slice at four slices) and 15 of 16; not a multiple of either.
SPLIT_K = 225

# PRECISION with K = SPLIT_K; the ordered cancellation (K = 2048) of PRECISION
# already splits.
PRECISION_SPLIT = [
    (
        label.replace("aligned precision", "split precision"),
        M,
        N,
        SPLIT_K,
        plane,
        dict(
            kwargs,
            data=dict(
                kwargs["data"],
                weight=first_channel_weight(
                    N, SPLIT_K, kwargs["data"]["weight"]["repeat"][0]
                ),
            ),
        ),
    )
    for label, M, N, K, plane, kwargs in PRECISION_ALIGNED
    if K == 32
]

# Split-K: K not a multiple of the K step or of four slices (K = 257 is 9 AMD
# K steps, which four slices cannot split; K = 321 is 11), and the fused
# residual epilogues with a K every split-K variant splits.
SPLIT = [
    ("K=257", 196, 64, 257, (14, 14), {}),
    ("K=321", 196, 64, 321, (14, 14), {}),
    (
        "split residual add",
        196,
        32,
        SPLIT_K,
        (14, 14),
        {"op": "epilogue", "epilogue": ("add", 1.0, False, "none")},
    ),
    (
        "split residual sub scale relu",
        49,
        24,
        SPLIT_K,
        (7, 7),
        {"op": "epilogue", "epilogue": ("sub", 0.5, True, "relu")},
    ),
]

# Shapes every aligned-only variant accepts: pixel counts one short of and one
# over a multiple of every M tile (63 = 64 - 1, 65 = 64 + 1), a fused clamp and
# a fused residual.
ALIGNED = [
    ("aligned 63px", 63, 64, 64, (7, 9), {}),
    ("aligned 65px", 65, 128, 96, (5, 13), {}),
    ("aligned 65px no bias", 65, 128, 96, (5, 13), {"bias": False}),
    ("aligned clamp", 196, 64, 128, (14, 14), {"op": "clamp", "clamp": [0.0, 6.0]}),
    (
        "aligned residual",
        49,
        64,
        64,
        (7, 7),
        {"op": "epilogue", "epilogue": ("add", 1.0, False, "none")},
    ),
]


# Clamped-read aligned-only variants (_alc) only: output channel counts that are
# not a multiple of the N tile or of four, the MobileNetV2 shapes on the
# 128-channel tile, and split-K cases with K a multiple of the K step 16 (K =
# 288 is 18 steps, four slices 5, 5, 5, 3; K = 224 is 14, four slices 4, 4, 4, 2).
SPLIT_K_ALC = 224
ALC = [
    ("alc N=30", 196, 30, 144, (14, 14), {}),
    ("alc mnv2", 49, 160, 960, (7, 7), {}),
    ("alc mnv2", 196, 64, 384, (14, 14), {}),
    ("alc mnv2", 196, 96, 576, (14, 14), {}),
    ("alc K=288", 196, 64, 288, (14, 14), {}),
    ("alc N=30 63px", 63, 30, SPLIT_K_ALC, (7, 9), {}),
    (
        "alc split residual add",
        196,
        32,
        SPLIT_K_ALC,
        (14, 14),
        {"op": "epilogue", "epilogue": ("add", 1.0, False, "none")},
    ),
    (
        "alc split residual sub scale relu",
        49,
        24,
        SPLIT_K_ALC,
        (7, 7),
        {"op": "epilogue", "epilogue": ("sub", 0.5, True, "relu")},
    ),
]


def alc_variants(variants):
    return [v for v in variants if "_alc" in v]


def fp32_variants(variants):
    return [v for v in variants if "_f32_" in v]


def correctness(variants):
    cases = []
    for label, M, N, K, plane, kwargs in (
        [
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
        ]
        + ALIGNED
        + SPLIT
    ):
        cases += shape_cases(
            label, M, N, K, plane, variants, VALIDATION_TIMING, **kwargs
        )
    for label, M, N, K, plane, kwargs in ALC:
        cases += shape_cases(
            label,
            M,
            N,
            K,
            plane,
            alc_variants(variants),
            VALIDATION_TIMING,
            baselines=False,
            **kwargs,
        )
    for label, M, N, K, plane, kwargs in (
        PRECISION + PRECISION_ALIGNED + PRECISION_SPLIT
    ):
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
    # The same plane with a K that every split-K variant can split.
    cases += shape_cases(
        "split dispatch limit",
        1025 * 1024,
        8,
        SPLIT_K,
        (1025, 1024),
        [v for v in variants if "_sk" in v],
        VALIDATION_TIMING,
        baselines=False,
    )
    # The same plane for the clamped-read variants, with a K they accept.
    cases += shape_cases(
        "alc dispatch limit",
        1025 * 1024,
        8,
        SPLIT_K_ALC,
        (1025, 1024),
        alc_variants(variants),
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
    for label, M, N, K, plane, kwargs in (
        PRECISION + PRECISION_ALIGNED + PRECISION_SPLIT
    ):
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

# Parent variant per device (OUT/GATES.md of conv2d-pw-coopmat-k-latency) and
# its staged-once diagnostic.
PARENTS = [
    "conv2d_pw_coopmat_xe2_t128x128_f16_half",
    "conv2d_pw_coopmat_xe2_t128x128_f32_half",
    "conv2d_pw_coopmat_amd_t16x32_f32_half",
    "conv2d_pw_coopmat_amd_t16x32_f16_half",
]
DIAG = [p.replace("_half", "_diag_once_half") for p in PARENTS]
KSWEEP_K = (32, 64, 128, 256, 576, 960)
KSWEEP_PAIRS = ((49, 96), (49, 160), (196, 64), (196, 96))


def diag_cases(label, M, N, K, plane, timing, **kwargs):
    """The diagnostic variants on one shape, validated against the staged-once
    reference (conv2d_pw_diag_once)."""
    return {
        "name": f"pw_coopmat {label} ({M},{N},{K}) diag",
        "operator": OPERATORS["conv"],
        **timing,
        "validation_method": TOLERANCE,
        "reference": "conv2d_pw_diag_once",
        "route": {"name": "pw_coopmat", "options": {"variant": "v_VARIANT"}},
        "sweep_variables": {"v_VARIANT": DIAG},
        "args_sweep": [conv_args(M, N, K, plane, **kwargs)],
    }


def ksweep(variants):
    """K sweep of design D3 at the four MobileNet (M, N) pairs: the tiled
    baselines, the parents and their diagnostics."""
    cases = []
    for M, N in KSWEEP_PAIRS:
        for K in KSWEEP_K:
            plane = plane_of(M)
            cases += shape_cases("ksweep", M, N, K, plane, PARENTS, PERFORMANCE_TIMING)
            cases.append(diag_cases("ksweep", M, N, K, plane, PERFORMANCE_TIMING))
    return cases


def diag_checks(variants):
    """The diagnostics' output depends on K (K = 64 and 576, ceil(K / k_step)
    times the first step) and on the input (fill 1 and 0.5): a variant that
    passes both members of a pair produced different outputs."""
    cases = []
    for K in (64, 576):
        cases.append(diag_cases("diag K", 196, 96, K, (14, 14), VALIDATION_TIMING))
    for fill in (1.0, 0.5):
        cases.append(
            diag_cases(
                f"diag input {fill}",
                196,
                96,
                576,
                (14, 14),
                VALIDATION_TIMING,
                data={"input": {"fill": fill}},
            )
        )
    return cases


GUARD = 7.0


def guard_case(
    variant,
    tile_m,
    tile_n,
    threads,
    label,
    M,
    N,
    K,
    plane,
    image_input=False,
    no_epilogue=False,
):
    """A raw shader case of an aligned-only variant whose output texture is
    larger than the convolution's output (out_sizes): GUARD rows below the plane
    and one 4-channel texel past the last output texel, filled with GUARD. The
    reference conv2d_pw_packed_guard expects GUARD there, so any store outside
    the output fails validation."""
    H, W = plane
    assert H * W == M
    s = float(f"{1 / K ** 0.5:.4g}")
    tex = lambda name, sizes, data, **kw: {  # noqa: E731
        "texture3d": {
            "name": name,
            "sizes": sizes,
            "dtype": "float16",
            "layout": "channels",
            **kw,
            "data": data,
        }
    }
    return {
        "name": f"pw_coopmat guard {label} ({M},{N},{K}) {variant}",
        "shader": variant,
        "reference": "conv2d_pw_packed_guard",
        "validation_method": {k: v for k, v in TOLERANCE.items() if k != "outputs"},
        "gwg_size": [-(-N // tile_n) * threads, -(-M // tile_m), 1],
        "lwg_size": [threads, 1, 1],
        **VALIDATION_TIMING,
        "bindings": [
            tex(
                "out",
                [1, -(-N // 4) * 4 + 4, H + 16, W],
                {"fill": GUARD},
                access="write",
            ),
            # A storage-image input must be in VK_IMAGE_LAYOUT_GENERAL, which
            # read-write access gives it.
            tex(
                "in",
                [1, K, H, W],
                {"rand_uniform_float": [-1, 1]},
                **({"access": "read_write"} if image_input else {}),
            ),
            {
                "buffer": {
                    "name": "w",
                    "sizes": [-(-K // 4), -(-N // 4) * 16],
                    "dtype": "float16",
                    "data": {"rand_uniform_float": [-s, s]},
                }
            },
            {
                "texture2d": {
                    "name": "bias",
                    "sizes": [1, N],
                    "dtype": "float16",
                    # An _epi0 shader adds no bias.
                    "data": (
                        {"fill": 0} if no_epilogue else {"rand_uniform_float": [-1, 1]}
                    ),
                }
            },
            # The unused residual binding: the input, or for a storage-image
            # input (which cannot also be bound sampled) a separate texture.
            (
                tex("res", [1, N, H, W], {"fill": 0})
                if image_input
                else {"texture3d": {"ref": "in"}}
            ),
            {"texture2d": {"ref": "bias"}},
            {"meta_ubo": "in", "field": "sizes"},
            {"uniform": [{"dtype": "int32", "data": [W, H, N, 1]}]},
        ],
        "specialization_constants": [
            {"dtype": "int32", "value": 1},
            {"dtype": "bool", "value": False},
            {"dtype": "int32", "value": 0},
            {"dtype": "bool", "value": False},
            {"dtype": "bool", "value": False},
            {"dtype": "int32", "value": 0},
        ],
        "push_constants": [
            {"dtype": "int32", "data": [1, 1, 0, 0]},
            {"dtype": "float32", "data": [0, 0, 1, 0, 0, 1]},
        ],
        "outputs": ["out"],
    }


# Clamped-read variants: N not a multiple of the N tile (24) and not of four
# (30, whose last texel holds channels 28 .. 31), with partial pixel tiles.
GUARD_CLAMP_N = [
    ("alc N=24", 63, 24, 64, (7, 9), {}),
    ("alc N=30", 65, 30, 96, (5, 13), {}),
]


def guard(family):
    """The guard cases of one family's aligned-only variants. Raw shader cases
    do not check the device, so each device runs the files of the families it
    offers."""

    def build(variants):
        return guard_cases(family, variants)

    return build


def guard_cases(family, variants):
    shader = yaml.safe_load(YAML.read_text())["conv2d_pw_coopmat"]
    defaults = shader["parameter_names_with_default_values"]
    cases = []
    for v in shader["shader_variants"]:
        p = dict(defaults, **v)
        if (
            not p.get("ALIGNED")
            or p["K_SLICES"] > 1
            or p["NAME"] not in variants
            or not p["NAME"].startswith(f"conv2d_pw_coopmat_{family}_")
        ):
            continue
        threads = p["SG_GRID_X"] * p["SG_GRID_Y"] * p["SUBGROUP_SIZE"]
        shapes = ALIGNED[:2] + (GUARD_CLAMP_N if p.get("CLAMP_N") else [])
        for label, M, N, K, plane, _ in shapes:
            cases.append(
                guard_case(
                    p["NAME"],
                    p["WG_TILE_M"],
                    p["WG_TILE_N"],
                    threads,
                    label,
                    M,
                    N,
                    K,
                    plane,
                    bool(p.get("IMG_INPUT")),
                    bool(p.get("NO_EPILOGUE")),
                )
            )
    return cases


# Direct weight load probe (task 4.8): per family the probe (_dlprobe) and its
# aligned staged control (_dlctl, the same options without DIRECT_B), raw, on
# one aligned shape, with the same input, bias and logical weights.
DLPROBE = {
    "xe2": "conv2d_pw_coopmat_xe2_t128x128_f16",
    "amd": "conv2d_pw_coopmat_amd_t16x32_f16",
    "nvidia": "conv2d_pw_coopmat_nvidia_t16x32_f16",
}
DLPROBE_SHAPE = (196, 128, 576, (14, 14))


def dlprobe_weight(n, k):
    """Logical weight (n, k) of the direct-load probe cases, a multiple of
    1/2048 below 1/16 in magnitude (exact in fp16); PointwisePackedReference.cpp
    conv2d_pw_logical computes the same."""
    return ((37 * n + 101 * k + 13) % 251 - 125) / 2048


def packed_weights(N, K):
    """dlprobe_weight in the 4OC x 4IC layout of pack_fp_linear_weight, [K4, N4 * 16]."""
    K4, N4 = -(-K // 4), -(-N // 4)
    out = [0.0] * (K4 * N4 * 16)
    for n in range(N):
        for k in range(K):
            out[(((k // 4) * N4 + n // 4) * 4 + k % 4) * 4 + n % 4] = dlprobe_weight(
                n, k
            )
    return out


def fragment_weights(N, K):
    """dlprobe_weight in 16 x 16 fragments, [K, N] (K and N multiples of 16)."""
    out = [0.0] * (K * N)
    for n in range(N):
        for k in range(K):
            out[((k // 16) * (N // 16) + n // 16) * 256 + (k % 16) * 16 + n % 16] = (
                dlprobe_weight(n, k)
            )
    return out


def raw_pw_case(shader, weight, weight_data, threads, tile_m, tile_n, M, N, K, plane):
    H, W = plane
    tex = lambda name, sizes, data, **kw: {  # noqa: E731
        "texture3d": {
            "name": name,
            "sizes": sizes,
            "dtype": "float16",
            "layout": "channels",
            **kw,
            "data": data,
        }
    }
    return {
        "name": f"pw_coopmat dlprobe ({M},{N},{K}) {shader}",
        "shader": shader,
        "reference": "conv2d_pw_logical",
        "validation_method": {k: v for k, v in TOLERANCE.items() if k != "outputs"},
        "gwg_size": [-(-N // tile_n) * threads, -(-M // tile_m), 1],
        "lwg_size": [threads, 1, 1],
        **PERFORMANCE_TIMING,
        "bindings": [
            tex("out", [1, N, H, W], {"fill": 0}, access="write"),
            tex("in", [1, K, H, W], {"rand_uniform_float": [-1, 1], "seed": 4801}),
            {
                "buffer": {
                    "name": "w",
                    "sizes": weight,
                    "dtype": "float16",
                    "data": weight_data,
                }
            },
            {
                "texture2d": {
                    "name": "bias",
                    "sizes": [1, N],
                    "dtype": "float16",
                    "data": {"rand_uniform_float": [-1, 1], "seed": 4802},
                }
            },
            {"texture3d": {"ref": "in"}},
            {"texture2d": {"ref": "bias"}},
            {"meta_ubo": "in", "field": "sizes"},
            {"meta_ubo": "out", "field": "sizes"},
        ],
        "specialization_constants": [
            {"dtype": "int32", "value": 1},
            {"dtype": "bool", "value": False},
            {"dtype": "int32", "value": 0},
            {"dtype": "bool", "value": False},
            {"dtype": "bool", "value": False},
            {"dtype": "int32", "value": 0},
        ],
        "push_constants": [
            {"dtype": "int32", "data": [1, 1, 0, 0]},
            {"dtype": "float32", "data": [0, 0, 1, 0, 0, 1]},
        ],
        "outputs": ["out"],
    }


def dlprobe(family):
    """The aligned staged control (_dlctl) with the logical weights in the
    pack_fp_linear_weight layout ([K4, N4 * 16]) and the _dlprobe shader with
    them in 16 x 16 fragments ([K, N]); both checked by conv2d_pw_logical."""

    def build(variants):
        shader = yaml.safe_load(YAML.read_text())["conv2d_pw_coopmat"]
        defaults = shader["parameter_names_with_default_values"]
        by = {v["NAME"]: dict(defaults, **v) for v in shader["shader_variants"]}
        M, N, K, plane = DLPROBE_SHAPE
        cases = []
        for name, weight, data in (
            (
                DLPROBE[family] + "_dlctl_half",
                [-(-K // 4), -(-N // 4) * 16],
                packed_weights(N, K),
            ),
            (DLPROBE[family] + "_dlprobe_half", [K, N], fragment_weights(N, K)),
        ):
            p = by[name]
            threads = p["SG_GRID_X"] * p["SG_GRID_Y"] * p["SUBGROUP_SIZE"]
            cases.append(
                raw_pw_case(
                    name,
                    weight,
                    data,
                    threads,
                    p["WG_TILE_M"],
                    p["WG_TILE_N"],
                    M,
                    N,
                    K,
                    plane,
                )
            )
        return cases

    return build


FILES = {
    "pw_coopmat_correctness.json": correctness,
    "pw_coopmat_fp16_accumulation_limits.json": fp16_accumulation_limits,
    "pw_coopmat_mobilenet.json": performance("mobilenet", MOBILENET),
    "pw_coopmat_resnet50.json": performance("resnet50", RESNET50),
    "pw_coopmat_large.json": performance("large", upstream_fp16_shapes()),
    "pw_coopmat_ksweep.json": ksweep,
    **{f"pw_coopmat_guard_{f}.json": guard(f) for f in FAMILIES},
    **{f"pw_coopmat_dlprobe_{f}.json": dlprobe(f) for f in FAMILIES},
    "pw_coopmat_diag_checks.json": diag_checks,
}
FILES["pw_coopmat_expected_skips.json"] = expected_skips(correctness)


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
