#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Writes coopmat_control_performance.json, the matrix-instruction control of
change conv2d-coopmat-evidence-hardening (design D5): twelve fixed layers of
model_sweep_performance.json (four 1x1, six k x k with dilation 1, two of them
at 112 px or more, two dilated), each with its baseline definition (route
auto) and one definition of every unsplit short-list parent of its form
(pw_coopmat for 1x1, direct_coopmat otherwise) and its _simt twin, with
allow_diagnostic set. Run with --check to fail if the file is out of date.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
from format_json import format_value  # noqa: E402

LAYERS = [
    "1x1 s1x1 p0x0 d1x1 64-256 56x56",
    "1x1 s1x1 p0x0 d1x1 128-512 28x28",
    "1x1 s1x1 p0x0 d1x1 1024-256 14x14",
    "1x1 s1x1 p0x0 d1x1 512-2048 7x7",
    "3x3 s1x1 p1x1 d1x1 64-64 224x224",
    "3x3 s1x1 p1x1 d1x1 32-64 147x147",
    "3x3 s1x1 p1x1 d1x1 256-256 14x14",
    "3x3 s1x1 p1x1 d1x1 512-512 7x7",
    "7x1 s1x1 p3x0 d1x1 192-192 17x17",
    "3x3 s2x2 p1x1 d1x1 256-256 28x28",
    "3x3 s1x1 p2x2 d2x2 256-256 28x28",
    "3x3 s1x1 p12x12 d12x12 2048-256 28x28",
]
PARENTS = {
    "pw_coopmat": [
        "xe2_t128x128_f16_alc",
        "xe2_t32x64_f16_g4x2_alc",
        "amd_t16x32_f16_alc",
        "amd_t128x128_f32_alc",
        "nvidia_t16x32_f16_alc",
        "nvidia_t128x128_f32_alc",
    ],
}
PARENTS["direct_coopmat"] = [p + "_dg" for p in PARENTS["pw_coopmat"]]
OUTPUT = HERE / "coopmat_control_performance.json"


def full(arm):
    return f"conv2d_pw_coopmat_{arm}_half"


def build():
    defs = {
        d["name"]: d
        for d in json.loads((HERE / "model_sweep_performance.json").read_text())
    }
    cases = []
    for label in LAYERS:
        base = defs[f"sweep {label} baseline"]
        route = "pw_coopmat" if label.startswith("1x1") else "direct_coopmat"
        kind = "coopmat" if route == "pw_coopmat" else "direct"
        src = defs[f"sweep {label} {kind}"]
        assert src["route"]["name"] == route, label
        cases.append({**copy.deepcopy(base), "name": f"control {label} baseline"})
        coopmat = copy.deepcopy(src)
        coopmat["name"] = f"control {label} {kind}"
        coopmat["route"] = {
            "name": route,
            "options": {"variant": "v_VARIANT", "allow_diagnostic": True},
        }
        coopmat["sweep_variables"] = {
            "v_VARIANT": [
                full(a) for p in PARENTS[route] for a in (p, p + "_simt")
            ]
        }
        cases.append(coopmat)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = format_value(build()) + "\n"
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != text:
            sys.exit(f"out of date: {OUTPUT.name}")
    else:
        OUTPUT.write_text(text)
    print(f"{OUTPUT.name}: {len(LAYERS)} layers")


if __name__ == "__main__":
    main()
