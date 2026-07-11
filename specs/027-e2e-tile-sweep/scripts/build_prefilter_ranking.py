"""Merge specs/025 (subgroup=64) and specs/026 (subgroup=32) correctness-passing
candidates into one combined, ranked pre-filter list (research.md Decision 1).

Note (correcting an earlier assumption): specs/025's round2_results.json avg_gflops is a
FLOP-weighted average across all three model sizes' wq+w1_gate shapes (1b/3b/8b), not an
8B-only number as this feature's plan.md/data-model.md initially assumed. shape_family is
therefore recorded as "8B" only as the primary e2e validation target (largest FLOPs,
already validated this session) -- the report must state this is not a strict per-shape
match, unlike what Decision 2 originally claimed. See sweep-report.md Scope Note.
"""

import json

SPEC025_ROUND2 = "specs/025-8da4w-parameter-sweep/results/round2_results.json"
SPEC026_ROUND3 = "specs/026-8da4w-subgroup32-sweep/results/round3_results.json"
OUT = "specs/027-e2e-tile-sweep/results/prefilter_ranking.json"
MODEL_8B = "llama3_1_8b_8da4w_buffer_ctx3072.pte"

candidates = []

r025 = json.load(open(SPEC025_ROUND2))
for e in r025:
    candidates.append(
        {
            "token": e["token"],
            "source_feature": "025",
            "subgroup_size": 64,
            "microbenchmark_gflops": e["avg_gflops"],
            "correctness_all_shapes_pass": True,  # round2 only contains correctness-passing candidates
            "shape_family": "8B",
            "model_used": MODEL_8B,
        }
    )

r026 = json.load(open(SPEC026_ROUND3))
for e in r026:
    if e.get("mean_gflops") is None:
        continue  # eliminated at correctness gate, per spec FR-004
    candidates.append(
        {
            "token": e["candidate_token"],
            "source_feature": "026",
            "subgroup_size": 32,
            "microbenchmark_gflops": e["mean_gflops"],
            "correctness_all_shapes_pass": True,
            "shape_family": "8B",
            "model_used": MODEL_8B,
        }
    )

candidates.sort(key=lambda c: c["microbenchmark_gflops"], reverse=True)
for i, c in enumerate(candidates, start=1):
    c["microbenchmark_rank"] = i
    c["shortlisted"] = i <= 8

json.dump(candidates, open(OUT, "w"), indent=2)
print(
    f"{len(candidates)} candidates, {sum(c['shortlisted'] for c in candidates)} shortlisted"
)
for c in candidates[:10]:
    print(
        c["microbenchmark_rank"],
        c["token"],
        c["microbenchmark_gflops"],
        c["source_feature"],
    )
