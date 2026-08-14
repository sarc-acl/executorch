#!/usr/bin/env python3
"""specs/037 Phase 4: turn final_matrix.jsonl into the deliverable -- a 6-row
markdown table of prefill tok/s speedup, T-tiled baseline vs winner-variant
coopmat, for {1B,3B,8B} x {4w,8da4w}."""

import argparse
import json
import statistics
from pathlib import Path

MODELS = ["1b", "3b", "8b"]
SCHEMES = ["4w", "8da4w"]
MODEL_LABEL = {"1b": "1B", "3b": "3B", "8b": "8B"}
SCHEME_LABEL = {"4w": "4w", "8da4w": "8da4w"}


def load_rows(results_path):
    rows = []
    for line in Path(results_path).read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def median_prefill(rows, model, scheme, config):
    vals = [
        r["prefill_tok_s"]
        for r in rows
        if r.get("ok")
        and r["model"] == model
        and r["scheme"] == scheme
        and r["config"] == config
    ]
    return statistics.median(vals) if vals else None, len(vals)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--winner-q4gsw")
    ap.add_argument("--winner-dq8ca")
    ap.add_argument(
        "--winner-json-dir",
        type=Path,
        help="specs/037 results/ dir with winner_{q4gsw,dq8ca}.json",
    )
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    winner_tok = {}
    if args.winner_json_dir:
        for shader in ("q4gsw", "dq8ca"):
            p = args.winner_json_dir / f"winner_{shader}.json"
            if p.exists():
                winner_tok[shader] = json.loads(p.read_text())["winner_token"]
    if args.winner_q4gsw:
        winner_tok["q4gsw"] = args.winner_q4gsw
    if args.winner_dq8ca:
        winner_tok["dq8ca"] = args.winner_dq8ca
    scheme_to_shader = {"4w": "q4gsw", "8da4w": "dq8ca"}

    rows = load_rows(args.results)

    lines = [
        "# specs/037 M51 deep sweep: prefill speedup, coopmat over T-tiled baseline",
        "",
        "| Model | Scheme | Baseline (T-tiled) tok/s | Coopmat tok/s | Speedup | Winner token |",
        "|---|---|---|---|---|---|",
    ]
    any_missing = False
    for model in MODELS:
        for scheme in SCHEMES:
            base, n_base = median_prefill(rows, model, scheme, "baseline")
            cm, n_cm = median_prefill(rows, model, scheme, "coopmat")
            tok = winner_tok.get(scheme_to_shader[scheme], "?")
            if base is None or cm is None:
                any_missing = True
                lines.append(
                    f"| {MODEL_LABEL[model]} | {SCHEME_LABEL[scheme]} | "
                    f"{'MISSING' if base is None else f'{base:.1f} (n={n_base})'} | "
                    f"{'MISSING' if cm is None else f'{cm:.1f} (n={n_cm})'} | -- | {tok} |"
                )
                continue
            speedup = cm / base
            lines.append(
                f"| {MODEL_LABEL[model]} | {SCHEME_LABEL[scheme]} | "
                f"{base:.1f} (n={n_base}) | {cm:.1f} (n={n_cm}) | {speedup:.3f}x | {tok} |"
            )
    if any_missing:
        lines.append("")
        lines.append(
            "**Some cells are MISSING** -- the matrix did not finish or a combo failed every rep. Re-run final_matrix.py with the same --results path to resume/retry only the missing combos."
        )

    out_text = "\n".join(lines) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(out_text)
        print(f"-> {args.out}")
    print(out_text)


if __name__ == "__main__":
    main()
