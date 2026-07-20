#!/usr/bin/env python3
"""3-run confirmation pass for every candidate with escalated: true, plus
BASELINE_TOKEN (always confirmed). Reuses run_e2e_screen.py's device-driving
primitives (fresh runs, not reusing the screening run's single data point --
research.md Decision 3).

Re-verifies driver hash + clock pin fresh before this round (Constitution
Principles VII/VIII) and records them per-run.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from run_e2e_screen import (  # noqa: E402
    BASELINE_TOKEN,
    clocks_pinned,
    driver_hash,
    run_one,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", required=True)
    ap.add_argument("--escalation", required=True)
    ap.add_argument("--model-stage", required=True)
    ap.add_argument("--model-used", default="llama3_1_8b_4w_buffer_ctx3072.pte")
    ap.add_argument("--board", default="sj1-dmckee-d01/0000088f8e579c33")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary-out", required=True)
    args = ap.parse_args()

    if args.raw_dir:
        os.makedirs(args.raw_dir, exist_ok=True)

    print(f"Pre-round driver hash: {driver_hash()}", file=sys.stderr)
    print(f"Pre-round clocks_pinned: {clocks_pinned()}", file=sys.stderr)

    escalation = json.loads(open(args.escalation).read())
    tokens_to_confirm = [BASELINE_TOKEN] + [
        e["candidate_token"]
        for e in escalation
        if e["model_stage"] == args.model_stage and e["escalated"]
    ]

    # Load any pre-existing confirm_results.json to append/resume rather than
    # clobber (device time is expensive; a crash mid-round shouldn't lose
    # already-completed runs).
    all_results = []
    if os.path.exists(args.out):
        all_results = json.loads(open(args.out).read())

    def already_have(token, run_index):
        return any(
            r["candidate_token"] == token
            and r["model_stage"] == args.model_stage
            and r["run_index"] == run_index
            for r in all_results
        )

    def persist():
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2)

    for token in tokens_to_confirm:
        for run_index in range(1, args.runs + 1):
            if already_have(token, run_index):
                print(
                    f"[{token}] run {run_index} already recorded, skipping",
                    file=sys.stderr,
                )
                continue
            do_coh = run_index == 1  # coherence-check once per token in this round
            r = run_one(
                token,
                args.model_used,
                args.model_stage,
                args.board,
                do_coh,
                args.raw_dir,
                stage="confirm",
                run_index=run_index,
            )
            all_results.append(r)
            persist()

    # Summary: mean/stddev/cov per confirmed token, improvement_pct vs baseline.
    import statistics

    by_token = {}
    for r in all_results:
        if r["model_stage"] != args.model_stage:
            continue
        by_token.setdefault(r["candidate_token"], []).append(r["prefill_tok_s"])

    baseline_vals = by_token.get(BASELINE_TOKEN, [])
    baseline_mean = statistics.mean(baseline_vals) if baseline_vals else None

    summary = []
    for token in tokens_to_confirm:
        vals = by_token.get(token, [])
        if not vals:
            continue
        mean = statistics.mean(vals)
        stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
        cov = (stdev / mean) if mean else 0.0
        improvement_pct = (
            round(100.0 * (mean - baseline_mean) / baseline_mean, 2)
            if baseline_mean and token != BASELINE_TOKEN
            else 0.0
        )
        summary.append(
            {
                "candidate_token": token,
                "model_stage": args.model_stage,
                "model_used": args.model_used,
                "mean_prefill_tok_s": round(mean, 3),
                "stddev_prefill_tok_s": round(stdev, 3),
                "cov": round(cov, 4),
                "baseline_mean_prefill_tok_s": (
                    round(baseline_mean, 3) if baseline_mean else None
                ),
                "improvement_pct": improvement_pct,
            }
        )

    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {len(all_results)} raw confirm runs to {args.out}")
    print(f"Wrote {len(summary)} confirmation summaries to {args.summary_out}")


if __name__ == "__main__":
    main()
