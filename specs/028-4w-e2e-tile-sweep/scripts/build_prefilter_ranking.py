#!/usr/bin/env python3
"""Build prefilter_ranking.json from specs/022's round2/round3 results.

Reads specs/022-linear-coopmat-autotune/results/round2_results.json (8
correctness-passing candidates) and cross-references round3_results.json to
set microbenchmark_confirmed on the one matching token. Sorts by
mean_gflops descending to assign microbenchmark_rank. Sets shape_family and
model_used per research.md Decision 1/2. Marks all 8 shortlisted: true.
"""
import json
import pathlib
import sys

SPECS_DIR = pathlib.Path(__file__).resolve().parents[2]
ROUND2 = SPECS_DIR / "022-linear-coopmat-autotune" / "results" / "round2_results.json"
ROUND3 = SPECS_DIR / "022-linear-coopmat-autotune" / "results" / "round3_results.json"
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "prefilter_ranking.json"

MODEL_USED = "llama3_1_8b_4w_buffer_ctx3072.pte"
SHAPE_FAMILY = "8B"


def main():
    round2 = json.loads(ROUND2.read_text())
    round3 = json.loads(ROUND3.read_text())

    confirmed_tokens = {r["candidate_token"] for r in round3}

    passing = [r for r in round2 if r.get("correctness_status") == "pass"]
    if len(passing) != 8:
        print(
            f"WARNING: expected exactly 8 correctness-passing round2 candidates, found {len(passing)}",
            file=sys.stderr,
        )

    passing_sorted = sorted(passing, key=lambda r: r["mean_gflops"], reverse=True)

    candidates = []
    for rank, r in enumerate(passing_sorted, start=1):
        token = r["candidate_token"]
        candidates.append(
            {
                "token": token,
                "source_feature": "022",
                "microbenchmark_gflops": r["mean_gflops"],
                "microbenchmark_rank": rank,
                "microbenchmark_confirmed": token in confirmed_tokens,
                "correctness_all_shapes_pass": True,
                "shape_family": SHAPE_FAMILY,
                "model_used": MODEL_USED,
                "shortlisted": True,
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(candidates, indent=2) + "\n")
    print(f"Wrote {len(candidates)} candidates to {OUT}")
    assert (
        len(candidates) == 8
    ), "Contract requires exactly 8 shortlisted candidates (Decision 1)"
    assert all(c["correctness_all_shapes_pass"] for c in candidates)
    assert all(c["shortlisted"] for c in candidates)


if __name__ == "__main__":
    main()
