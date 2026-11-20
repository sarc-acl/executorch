#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FP16 sweep cases of the narrow-N forms (change conv2d-all-used-kernels,
part C): every configuration of model_sweep_configs.json with at most 64
output channels, with the definitions of model_sweep_performance.json (same
names, arguments and timing): route auto, and on pw_coopmat (1x1, stride 1,
padding 0) and direct_coopmat (every configuration) the 64x32 forms for at
most 32 output channels, the 64x64 forms for 33 to 64.

    gen_fp16_narrow_cases.py [--check]   # fp16_narrow_performance.json
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
from format_json import format_value  # noqa: E402

FAMILIES = {"xe2": "_g2x4", "amd": "", "nvidia": ""}
PW = [
    f"conv2d_pw_coopmat_{f}_t64x32_f16{g}_alc{s}_half"
    for f, g in FAMILIES.items()
    for s in ("", "_sk2", "_sk4")
]
DG = [v.replace("_half", "_dg_half") for v in PW]
# 33 to 64 output channels: the 64x64 forms (Xe2 on a 4x2 subgroup grid; NVIDIA).
PW64 = [
    f"conv2d_pw_coopmat_{f}_t64x64_f16{g}_alc{s}_half"
    for f, g in (("xe2", "_g4x2"), ("nvidia", ""))
    for s in ("", "_sk2") + (("_sk4",) if f == "xe2" else ())
]
DG64 = [v.replace("_half", "_dg_half") for v in PW64]


def files():
    configs = json.loads((HERE / "model_sweep_configs.json").read_text())
    narrow = {f"sweep {c['label']}": c["cout"] for c in configs if c["cout"] <= 64}
    out = []
    for d in json.loads((HERE / "model_sweep_performance.json").read_text()):
        shape = d["name"].rsplit(" ", 1)[0]
        if shape not in narrow:
            continue
        if "route_sweep" in d:
            out.append(d)
            continue
        route = d["route"]["name"]
        pw = route == "pw_coopmat"
        if route not in ("pw_coopmat", "direct_coopmat"):
            continue
        x = json.loads(json.dumps(d))
        wide = narrow[shape] > 32
        x["sweep_variables"]["v_VARIANT"] = (
            (PW64 if wide else PW) if pw else (DG64 if wide else DG)
        )
        x["name"] = f"{shape} {'narrow' if pw else 'narrow_dg'}"
        out.append(x)
    return {HERE / "fp16_narrow_performance.json": out}


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
