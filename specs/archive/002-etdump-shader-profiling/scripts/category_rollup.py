#!/usr/bin/env python3
"""Add category rollups to an already-produced raw JSON (Research Decision 4
/ User Story 3) -- no re-profiling needed, matches spec's Independent Test
for US3. Categorizes each Aggregated Kernel Entry by matching its (K, N)
shape against 001's per-model op catalog (results/shapes.json).

Usage:
    python category_rollup.py \
        --raw-json results/raw/llama-3.2-1b_4w.json \
        --shapes-json ../001-minipc-baseline-benchmarks/results/shapes.json
"""
import argparse
import json

CATEGORY_BY_OP = {
    "wq": "attention projection",
    "wk": "attention projection",
    "wv": "attention projection",
    "wo": "attention projection",
    "w1_gate": "feed-forward",
    "w3_up": "feed-forward",
    "w2_down": "feed-forward",
    "lm_head": "output/vocab projection",
}


def build_shape_to_category(model_ops):
    """(K, N) -> category, from 001's shapes.json ops table for this model."""
    mapping = {}
    for op_name, kn in model_ops.items():
        mapping[(kn["k"], kn["n"])] = CATEGORY_BY_OP.get(op_name, "other")
    return mapping


def categorize(shape, shape_to_category, kernel_name):
    if shape is None:
        return "non-shader overhead"
    key = (shape["k"], shape["n"])
    if key in shape_to_category:
        return shape_to_category[key]
    if "sdpa" in kernel_name.lower():
        return "attention (sdpa)"
    return "other"


def rollup(raw_json_path, shapes_json_path):
    d = json.load(open(raw_json_path))
    model = d["config"]["model"]
    shapes = json.load(open(shapes_json_path))
    shape_to_category = build_shape_to_category(shapes[model]["ops"])

    for phase_name, phase in d["phases"].items():
        if phase["status"] != "ok":
            continue
        totals = {}
        for entry in phase["aggregated"]:
            cat = categorize(entry["shape"], shape_to_category, entry["kernel_name"])
            entry["category"] = cat
            totals[cat] = totals.get(cat, 0.0) + entry["total_time_us"]

        phase_us = phase["phase_wall_clock_us_profiled"]
        category_rollup = [
            {
                "category": cat,
                "total_time_us": round(total, 3),
                "pct_of_phase": round(total / phase_us, 4) if phase_us else None,
            }
            for cat, total in sorted(totals.items(), key=lambda kv: -kv[1])
        ]
        unattributed_pct = (
            round(1.0 - phase["attributed_pct"], 4)
            if phase["attributed_pct"] is not None
            else None
        )
        if unattributed_pct and unattributed_pct > 0.0001:
            category_rollup.append(
                {
                    "category": "unattributed",
                    "total_time_us": round(phase_us * unattributed_pct, 3),
                    "pct_of_phase": unattributed_pct,
                }
            )
        phase["category_rollup"] = category_rollup

        total_pct = sum(c["pct_of_phase"] for c in category_rollup if c["pct_of_phase"])
        if abs(total_pct - 1.0) > 0.02:
            print(
                f"WARNING: {model}/{phase_name} category percentages sum to {total_pct:.4f}, not ~1.0"
            )

    json.dump(d, open(raw_json_path, "w"), indent=2)
    print(f"updated {raw_json_path}")
    for phase_name, phase in d["phases"].items():
        if phase["status"] != "ok":
            continue
        print(f"  {phase_name}:")
        for c in phase["category_rollup"]:
            print(
                f"    {c['category']:28s} {c['pct_of_phase']*100:5.1f}%  ({c['total_time_us']:.0f}us)"
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-json", required=True)
    p.add_argument("--shapes-json", required=True)
    args = p.parse_args()
    rollup(args.raw_json, args.shapes_json)


if __name__ == "__main__":
    main()
