#!/usr/bin/env python3
"""Score and shortlist the legal 8da4w tile/subgroup candidates.

Reuses 022's analytical formula shape (occupancy_proxy / register_penalty), but with
8da4w's own lds_bytes/accumulators_per_sg (from tile_constraints.py, research.md
Decision 1/2) and a register-pressure baseline K taken from the shipped config's own
accumulators_per_sg (8, per dbuf_reconfirmation.json's geometry) rather than 022's
4w-calibrated K=8/weight=0.15 constants reused blindly -- here K coincides at 8 for this
shader too, but the penalty weight is kept at 022's 0.15 since no 8da4w-specific
recalibration data (beyond the single shipped-geometry point) exists to justify a
different constant; this is documented as a limitation, not silently assumed identical.
"""
import argparse
import json

MAX_SHARED_MEM = 65536
MAX_WG_INVOCATIONS = 1024
REGISTER_PENALTY_K = 8  # accumulators_per_sg of the shipped 128x64/K32/2x2/s64 config
REGISTER_PENALTY_WEIGHT = 0.15  # reused from 022 -- see module docstring
BUDGET_PCT = 0.15
BUDGET_HARD_CAP = 30
MIN_WG_SIZE = 128


def score(candidate):
    occupancy_proxy = min(
        MAX_SHARED_MEM / candidate["lds_bytes"],
        MAX_WG_INVOCATIONS / candidate["wg_size"],
    )
    register_penalty = (
        1
        + max(0, candidate["accumulators_per_sg"] - REGISTER_PENALTY_K)
        * REGISTER_PENALTY_WEIGHT
    )
    return occupancy_proxy / register_penalty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-out", required=True)
    ap.add_argument("--fourw-winner-token", default="tsweep_t128x64k16g22s32")
    args = ap.parse_args()

    configs = json.load(open(args.configs))
    total_valid_universe = len(configs)
    budget_cap = min(round(BUDGET_PCT * total_valid_universe), BUDGET_HARD_CAP)

    scored = []
    for c in configs:
        s = score(c)
        scored.append({**c, "score": s})

    scored.sort(key=lambda c: c["score"], reverse=True)
    for i, c in enumerate(scored, start=1):
        c["rank"] = i

    shipped_token = "tsweep_t128x64k32g22s64"
    anchors = {shipped_token: "anchor:shipped-config"}

    # 4w's winner is only a legal anchor if it happens to satisfy this shader's own
    # constraints (it doesn't -- it implies subgroup_size=32 -- research.md Decision 1).
    fourw_token = args.fourw_winner_token
    fourw_present = any(c["token"] == fourw_token for c in scored)
    excluded_anchors = []
    if not fourw_present:
        excluded_anchors.append(
            {
                "token": fourw_token,
                "reason": "illegal for 8da4w: 4w's winning geometry uses subgroup_size=32, "
                "which crashes the Xclipse PAL compiler for this shader's int8 WMMA "
                "(research.md Decision 1) -- not enumerated as a candidate at all.",
            }
        )

    # top-rank shortlist, respecting the budget cap; anchors force-included regardless of rank.
    non_anchor_budget = budget_cap - len(anchors)
    shortlist_count = 0
    for c in scored:
        if c["token"] in anchors:
            c["shortlisted"] = True
            c["shortlist_reason"] = anchors[c["token"]]
            continue
        if c["wg_size"] < MIN_WG_SIZE:
            c["shortlisted"] = False
            c["shortlist_reason"] = (
                "excluded: below 022's WG_SIZE>=128 minimum-parallelism floor"
            )
            continue
        if shortlist_count < non_anchor_budget:
            c["shortlisted"] = True
            c["shortlist_reason"] = "top-rank"
            shortlist_count += 1
        else:
            c["shortlisted"] = False
            c["shortlist_reason"] = "excluded"

    with open(args.out, "w") as f:
        json.dump(
            {"shortlist": scored, "excluded_anchors": excluded_anchors}, f, indent=2
        )

    budget = {
        "total_valid_universe": total_valid_universe,
        "budget_cap": budget_cap,
        "configs_measured_on_hardware": 0,
        "total_device_seconds": 0.0,
        "estimated_exhaustive_device_seconds": None,
        "budget_exceeded": False,
    }
    with open(args.budget_out, "w") as f:
        json.dump(budget, f, indent=2)

    n_shortlisted = sum(1 for c in scored if c["shortlisted"])
    print(f"total_valid_universe={total_valid_universe}")
    print(f"budget_cap={budget_cap}")
    print(f"shortlisted={n_shortlisted}")
    print(f"excluded_anchors={excluded_anchors}")


if __name__ == "__main__":
    main()
