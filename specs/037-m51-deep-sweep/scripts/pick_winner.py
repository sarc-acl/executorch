"""Phase 2: cross-size validation of specs/037's Phase-1 tsweep finalists.

The Phase-1 sweep (specs/036-portable-device-sweep/scripts/sweep.py) screens
and ranks finalists on the 1B pte (cheap). Rank flips between 1B and 8B are
documented (specs/036's protocol.md #5) and ACTIVE-STATUS treats 8B as the
priority size, so this script re-confirms the top-N finalists -- plus the
shipped default, unconditionally, as the thing a new winner has to beat --
on 8B (and 3B for the record), and picks ONE global winner token per shader:
whichever wins at 8B. Falls back to the shipped default if no finalist beats
it there. A "no improvement" tile outcome is normal and expected (specs/036's
own dq8ca sweep found -0.44% at best) -- the point of this script is to
*check*, not to force a new winner into existence.

Reuses measure_android.Session unmodified aside from its new `extra_env`
param (added alongside this script) so the 8B validation runs can set
ET_VK_EXECUTE_NODE_THRESHOLD=32 -- the sgpu watchdog workaround
run_maxclock_ab.py applies to every 8B run of a long prefill.
"""

import argparse
import json
import sys
import time
from pathlib import Path

SWEEP_SCRIPTS = (
    Path(__file__).resolve().parent.parent.parent
    / "036-portable-device-sweep"
    / "scripts"
)
sys.path.insert(0, str(SWEEP_SCRIPTS))
import measure_android as ma  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

SHIPPED_DEFAULT = {
    "q4gsw": "tsweep_t128x128k16g22s32",
    "dq8ca": "tsweep_t64x32k32g12s64",
}
PTE_SCHEME_NAME = {"q4gsw": "4w", "dq8ca": "8da4w"}
MODEL_PTE_PREFIX = {"1b": "llama3_2_1b", "3b": "llama3_2_3b", "8b": "llama3_1_8b"}
NODE_THRESHOLD_MODELS = {"8b"}  # watchdog workaround only needed at this size


def pte_path(model, shader):
    name = f"{MODEL_PTE_PREFIX[model]}_{PTE_SCHEME_NAME[shader]}_buffer_ctx3072.pte"
    return f"{ma.DEVICE_DIR}/{name}"


def load_phase1_finalists(shader, slug_suffix, top_n):
    sweep_results = SWEEP_SCRIPTS.parent / "results"
    matches = sorted(sweep_results.glob(f"sweep_summary_*{slug_suffix}_{shader}.json"))
    if not matches:
        raise FileNotFoundError(
            f"no Phase-1 sweep summary for shader={shader} suffix={slug_suffix} "
            f"in {sweep_results} -- run Phase 1 first."
        )
    summary = json.loads(matches[-1].read_text())
    tokens = [f["token"] for f in summary.get("finalists", [])[:top_n]]
    return tokens, summary


def validate_on_model(shader, model, tokens, quirks, reps=5):
    extra_env = (
        {"ET_VK_EXECUTE_NODE_THRESHOLD": "32"} if model in NODE_THRESHOLD_MODELS else {}
    )
    out_jsonl = RESULTS_DIR / f"validate_{shader}_{model}.jsonl"
    s = ma.Session(
        shader,
        pte=pte_path(model, shader),
        quirks=quirks,
        extra_env=extra_env,
        out_jsonl=out_jsonl,
    )
    base, cov = s.baseline()
    print(f"[{shader}/{model}] baseline {base:.1f} tok/s (noise cov {cov*100:.2f}%)")
    results = {}
    for tok in tokens:
        g = s.gate(tok)
        if g.status != "pass":
            print(f"  {tok}\tGATE:{g.status} -- excluded")
            continue
        m = s.confirm(tok, reps=reps)
        flag = " MISCOMPUTE" if m.correctness_flag else ""
        print(f"  {tok}\t{m.median:.1f}{flag}")
        if not m.correctness_flag:
            results[tok] = m.median
    return base, results


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shader", required=True, choices=("q4gsw", "dq8ca"))
    ap.add_argument(
        "--slug-suffix",
        default="maxclk",
        help="must match the --slug-suffix the Phase-1 sweep.py run used",
    )
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument(
        "--quirk", action="append", default=[], help="e.g. no_int8_wmma_sg32"
    )
    ap.add_argument(
        "--cooldown-s",
        type=int,
        default=120,
        help="pause before validating, so Phase 2's fresh baseline isn't taken "
        "right at the tail of Phase 1's sustained max-clock sweep session "
        "(Phase-1 sweeps have shown noise_floor_cov ~4-5%% and heavy "
        "remeasure_pending counts at max clock vs ~1%% at pinned 509 -- a "
        "thermal-throttle signature, not a bug in the sweep itself)",
    )
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.cooldown_s > 0:
        print(f"cooling down {args.cooldown_s}s before Phase 2 baseline...")
        time.sleep(args.cooldown_s)

    finalist_tokens, phase1_summary = load_phase1_finalists(
        args.shader, args.slug_suffix, args.top_n
    )
    default_tok = SHIPPED_DEFAULT[args.shader]
    candidates = list(dict.fromkeys(finalist_tokens + [default_tok]))
    print(
        f"[{args.shader}] candidates (top-{args.top_n} + shipped default): {candidates}"
    )
    remeasure_pending = set(phase1_summary.get("remeasure_pending", []))
    flagged = [t for t in candidates if t in remeasure_pending]
    if flagged:
        print(
            f"  NOTE: {flagged} were measured during a Phase-1 drifted window "
            f"(remeasure_pending) -- their Phase-1 rank may be unreliable; "
            f"Phase 2's fresh confirm below is the actual arbiter."
        )

    per_model = {}
    for model in ("8b", "3b"):
        base, results = validate_on_model(
            args.shader, model, candidates, args.quirk, reps=args.reps
        )
        per_model[model] = {"baseline_tok_s": base, "results": results}

    eightb = per_model["8b"]["results"]
    default_val = eightb.get(default_tok)
    if eightb:
        best_tok = max(eightb, key=eightb.get)
        best_val = eightb[best_tok]
        if default_val is not None and best_val <= default_val:
            winner_tok, winner_val, source = (
                default_tok,
                default_val,
                "shipped_default_wins_at_8b",
            )
        else:
            winner_tok, winner_val, source = best_tok, best_val, "validated_8b_winner"
    else:
        winner_tok, winner_val, source = (
            default_tok,
            None,
            "shipped_default_fallback_no_8b_data",
        )

    out = {
        "shader": args.shader,
        "winner_token": winner_tok,
        "winner_8b_tok_s": winner_val,
        "shipped_default_token": default_tok,
        "shipped_default_8b_tok_s": default_val,
        "source": source,
        "candidates": candidates,
        "per_model": per_model,
        "phase1_summary_winner_token": (phase1_summary.get("winner") or {}).get(
            "token"
        ),
    }
    out_path = (
        Path(args.out) if args.out else RESULTS_DIR / f"winner_{args.shader}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"winner[{args.shader}]: {winner_tok} ({source})")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
