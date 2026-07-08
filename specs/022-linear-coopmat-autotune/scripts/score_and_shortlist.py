#!/usr/bin/env python3
"""Score all valid tile-geometry candidates with the analytical cost model
from research.md Decision 2, rank them, and shortlist the top ~24-32 plus
the two known anchors. Zero device interaction -- see contracts/
autotune-report-schema.md #2 for the output contract."""

import argparse
import json
from pathlib import Path

LDS_LIMIT_BYTES = 65536
WG_INVOCATION_LIMIT = 1024
REGISTER_PENALTY_THRESHOLD = 8
REGISTER_PENALTY_SLOPE = 0.15

# Minimum-parallelism floor (research.md Decision 2): every one of the 10
# real configurations ever measured on this hardware uses WG_SIZE >= 128; a
# single-subgroup workgroup (WG_SIZE < 128) can't use this shader family's
# cross-subgroup double-buffered prefetch/compute overlap. Candidates below
# this floor stay in the full ranking (auditable) but are never top-ranked.
MIN_WG_SIZE_FOR_TOP_RANK = 128

ANCHOR_DBUF1_TOKEN = "tsweep_t128x128k16g42s32"
ANCHOR_WINNER_TOKEN = "tsweep_t128x64k16g22s32"
SPECIAL_ANCHOR_REASONS = {
    ANCHOR_DBUF1_TOKEN: "anchor:dbuf1",
    ANCHOR_WINNER_TOKEN: "anchor:sweep-winner",
}

DEFAULT_SHORTLIST_TOP_N = 28  # midpoint of the ~24-32 target range


def load_known_measurement_anchors(known_measurements_path: str) -> tuple[dict, dict]:
    """Per research.md Decision 3 (revised after the T009 calibration
    finding): every previously-measured, compiling config is force-included
    in the shortlist regardless of analytical rank -- we already have real
    ground truth for these, an unfitted heuristic should never override it.
    A known compile failure is explicitly excluded (re-attempting it wastes
    budget on an already-known outcome), not silently omitted."""
    known = json.loads(Path(known_measurements_path).read_text())
    anchors = {}
    excluded = {}
    for k in known:
        token = k["candidate_token"]
        if k["compile_status"] != "compiles":
            excluded[token] = "known_compile_failure"
            continue
        anchors[token] = SPECIAL_ANCHOR_REASONS.get(token, "anchor:known-measurement")
    return anchors, excluded


def score_candidate(cfg: dict) -> tuple[float, float, float]:
    occupancy_proxy = min(
        LDS_LIMIT_BYTES / cfg["lds_bytes"],
        WG_INVOCATION_LIMIT / cfg["wg_size"],
    )
    register_penalty = 1 + max(0, cfg["accumulators_per_sg"] - REGISTER_PENALTY_THRESHOLD) * REGISTER_PENALTY_SLOPE
    score = occupancy_proxy / register_penalty
    return occupancy_proxy, register_penalty, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--known-measurements", default=None)
    parser.add_argument("--top-n", type=int, default=DEFAULT_SHORTLIST_TOP_N)
    args = parser.parse_args()

    configs = json.loads(Path(args.configs).read_text())

    anchors, excluded_known = {}, {}
    if args.known_measurements:
        anchors, excluded_known = load_known_measurement_anchors(args.known_measurements)

    scored = []
    for cfg in configs:
        occupancy_proxy, register_penalty, score = score_candidate(cfg)
        scored.append(
            {
                "candidate_token": cfg["token"],
                "wg_size": cfg["wg_size"],
                "occupancy_proxy": round(occupancy_proxy, 4),
                "register_penalty": round(register_penalty, 4),
                "score": round(score, 4),
            }
        )

    scored.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(scored, start=1):
        r["rank"] = i

    top_rank_eligible = [r for r in scored if r["wg_size"] >= MIN_WG_SIZE_FOR_TOP_RANK]
    top_n_tokens = {r["candidate_token"] for r in top_rank_eligible[: args.top_n]}

    for r in scored:
        token = r["candidate_token"]
        if token in excluded_known:
            r["shortlisted"] = False
            r["shortlist_reason"] = excluded_known[token]
        elif token in anchors:
            r["shortlisted"] = True
            r["shortlist_reason"] = anchors[token]
        elif token in top_n_tokens:
            r["shortlisted"] = True
            r["shortlist_reason"] = "top-rank"
        else:
            r["shortlisted"] = False
            r["shortlist_reason"] = "excluded"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scored, indent=2))

    n_shortlisted = sum(1 for r in scored if r["shortlisted"])
    print(f"Wrote {len(scored)} ranked entries to {out_path}")
    print(f"Shortlisted: {n_shortlisted}")
    missing_anchors = [t for t in anchors if t not in {r['candidate_token'] for r in scored}]
    if missing_anchors:
        print(f"WARNING: anchor token(s) not found in configs.json: {missing_anchors}")


if __name__ == "__main__":
    main()
