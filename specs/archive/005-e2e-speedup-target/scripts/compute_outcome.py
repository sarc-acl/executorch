#!/usr/bin/env python3
"""Formalize this workstream's prefill speedup target from 001's baseline,
and compare a future re-measurement against it (data-model.md's Speedup
Target / Re-Measurement / Outcome entities). See research.md Decision 3:
no real re-measurement exists yet, so --selftest proves the comparison
logic correct on synthetic data before it is ever trusted on real data.
"""
import argparse
import json
import os

TARGET_MULTIPLIER = 2.0
CONFIGS = [
    ("llama-3.1-8b", "4w"),
    ("llama-3.1-8b", "8da4w"),
    ("llama-3.2-3b", "4w"),
    ("llama-3.2-3b", "8da4w"),
    ("llama-3.2-1b", "4w"),
    ("llama-3.2-1b", "8da4w"),
]


def generate_target(baseline_dir, out_path):
    configs = []
    for model, scheme in CONFIGS:
        src = os.path.join(baseline_dir, f"{model}_{scheme}.json")
        d = json.load(open(src))
        e2e = d["e2e"]
        baseline_prefill_tps = e2e["prefill_tokens_per_sec"]
        baseline_decode_tps = e2e["decode_tokens_per_sec"]
        baseline_combined_seconds = (
            e2e["prefill_tokens"] / baseline_prefill_tps
            + e2e["decode_tokens"] / baseline_decode_tps
        )
        configs.append(
            {
                "model": model,
                "scheme": scheme,
                "baseline_prefill_tokens_per_sec": baseline_prefill_tps,
                "baseline_decode_tokens_per_sec": baseline_decode_tps,
                "baseline_combined_seconds": round(baseline_combined_seconds, 4),
                "baseline_prefill_stdev": e2e["variance"][
                    "prefill_tokens_per_sec_stdev"
                ],
                "target_multiplier": TARGET_MULTIPLIER,
                "target_prefill_tokens_per_sec": round(
                    baseline_prefill_tps * TARGET_MULTIPLIER, 3
                ),
                "baseline_source": src,
            }
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"configs": configs}, f, indent=2)
    print(f"wrote {out_path}")
    for c in configs:
        print(
            f"  {c['model']}/{c['scheme']}: baseline={c['baseline_prefill_tokens_per_sec']:.1f} "
            f"tok/s -> target={c['target_prefill_tokens_per_sec']:.1f} tok/s"
        )


def load_targets(target_path):
    d = json.load(open(target_path))
    return {(c["model"], c["scheme"]): c for c in d["configs"]}


def load_after_dir(after_dir):
    """--after-dir loading (US2/T007): match <model>_<scheme>.json files to
    their target entry by (model, scheme), same naming convention as 001."""
    after = {}
    for model, scheme in CONFIGS:
        path = os.path.join(after_dir, f"{model}_{scheme}.json")
        if os.path.exists(path):
            after[(model, scheme)] = json.load(open(path))
    return after


def compute_verdict(target, after_record):
    """Core verdict engine (T004): given one Speedup Target entry + one
    Re-Measurement record, return an Outcome record. Combined e2e change is
    computed and reported but never used to decide `verdict` (FR-001/FR-006).
    """
    is_synthetic = after_record.get("is_synthetic", False)

    # Methodology-comparability check (US2/T008, FR-008).
    e2e = after_record["e2e"]
    comparable = after_record.get("methodology_comparable", True)
    note = after_record.get("methodology_note")
    if e2e.get("prefill_tokens") != 2048 or e2e.get("decode_tokens") != 1024:
        comparable = False
        note = note or (
            f"workload size mismatch: prefill_tokens={e2e.get('prefill_tokens')}, "
            f"decode_tokens={e2e.get('decode_tokens')} (expected 2048/1024)"
        )

    if not comparable:
        return {
            "model": target["model"],
            "scheme": target["scheme"],
            "observed_multiplier": None,
            "verdict": "not_comparable",
            "combined_e2e_change_pct": None,
            "methodology_note": note,
            "is_synthetic": is_synthetic,
        }

    baseline_tps = target["baseline_prefill_tokens_per_sec"]
    after_tps = e2e["prefill_tokens_per_sec"]
    observed_multiplier = after_tps / baseline_tps

    # Noise band: the baseline's own measured stdev, expressed as a
    # multiplier delta (research.md Decision 4) -- not an arbitrary new
    # tolerance.
    noise_band = target["baseline_prefill_stdev"] / baseline_tps

    if observed_multiplier < 1.0:
        verdict = "regressed"
    elif observed_multiplier < TARGET_MULTIPLIER - noise_band:
        verdict = "missed"
    elif observed_multiplier <= TARGET_MULTIPLIER + noise_band:
        verdict = "met"
    else:
        verdict = "exceeded"

    # Combined e2e change: prefill + decode wall-clock time, weighted by the
    # fixed 2048/1024 workload -- tracked only, never used for `verdict`
    # (FR-001/FR-006). Computed from real numbers on both sides, not assumed.
    after_combined_seconds = (
        e2e["prefill_tokens"] / e2e["prefill_tokens_per_sec"]
        + e2e["decode_tokens"] / e2e["decode_tokens_per_sec"]
    )
    baseline_combined_seconds = target["baseline_combined_seconds"]
    combined_e2e_change_pct = round(
        (baseline_combined_seconds - after_combined_seconds)
        / baseline_combined_seconds
        * 100,
        2,
    )

    return {
        "model": target["model"],
        "scheme": target["scheme"],
        "observed_multiplier": round(observed_multiplier, 4),
        "verdict": verdict,
        "combined_e2e_change_pct": combined_e2e_change_pct,
        "methodology_note": None,
        "is_synthetic": is_synthetic,
    }


def render_report(outcomes, out_path, synthetic_heading=False):
    lines = []
    if synthetic_heading:
        lines.append("# SYNTHETIC SELF-TEST DATA — NOT A REAL MEASUREMENT")
        lines.append("")
        lines.append(
            "Every entry below is constructed synthetic data (research.md "
            "Decision 3), proving the verdict engine's logic before it is "
            "ever pointed at a real re-measurement. Do not read any number "
            "here as a real optimization result."
        )
    else:
        lines.append("# End-to-End Speedup Outcome Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| Model | Scheme | Verdict | Observed multiplier | Combined e2e change | Synthetic |"
    )
    lines.append("|---|---|---|---:|---:|---|")
    for o in outcomes:
        mult = (
            f"{o['observed_multiplier']:.2f}x"
            if o["observed_multiplier"] is not None
            else "n/a"
        )
        e2e_chg = (
            f"{o['combined_e2e_change_pct']:+.1f}%"
            if o["combined_e2e_change_pct"] is not None
            else "n/a"
        )
        lines.append(
            f"| {o['model']} | {o['scheme']} | **{o['verdict']}** | {mult} | {e2e_chg} | {o['is_synthetic']} |"
        )
    lines.append("")
    lines.append("## Detail")
    lines.append("")
    for o in outcomes:
        lines.append(f"### {o['model']} / {o['scheme']}")
        lines.append("")
        lines.append(f"- verdict: **{o['verdict']}**")
        if o["verdict"] == "not_comparable":
            lines.append(f"- not directly comparable: {o['methodology_note']}")
        else:
            lines.append(
                f"- observed prefill multiplier: {o['observed_multiplier']:.4f}x"
            )
            lines.append(
                f"- combined e2e change (tracked, not a pass/fail bar): "
                f"{o['combined_e2e_change_pct']:+.1f}%"
                if o["combined_e2e_change_pct"] is not None
                else "- combined e2e change: n/a"
            )
        lines.append("")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {out_path}")


SELFTEST_SCENARIOS = [
    # (name, prefill_multiplier, comparable, methodology_note)
    # Decode is left at the baseline's own decode tok/s in every scenario --
    # this workstream has no identified decode fix yet (FR-006), so a
    # realistic near-term "after" only ever changes prefill.
    ("met", 2.0, True, None),
    ("exceeded", 2.6, True, None),
    ("missed", 1.4, True, None),
    ("regressed", 0.8, True, None),
    (
        "not_comparable",
        2.0,
        False,
        "prefill workload was 1024 tokens, not the required 2048",
    ),
]


def run_selftest(target_path, out_dir):
    targets = load_targets(target_path)
    outcomes = []
    scenario_files = {}
    for i, (model, scheme) in enumerate(CONFIGS):
        target = targets[(model, scheme)]
        scenario_name, multiplier, comparable, note = SELFTEST_SCENARIOS[
            i % len(SELFTEST_SCENARIOS)
        ]
        baseline_prefill_tps = target["baseline_prefill_tokens_per_sec"]
        after_record = {
            "model": model,
            "scheme": scheme,
            "e2e": {
                "prefill_tokens_per_sec": round(baseline_prefill_tps * multiplier, 3),
                "decode_tokens_per_sec": target["baseline_decode_tokens_per_sec"],
                "prefill_tokens": 2048 if comparable else 1024,
                "decode_tokens": 1024,
            },
            "methodology_comparable": comparable,
            "methodology_note": note,
            "is_synthetic": True,
        }
        scenario_files[f"{model}_{scheme}"] = {
            "scenario": scenario_name,
            "after_record": after_record,
        }
        outcomes.append(compute_verdict(target, after_record))

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "synthetic_after_records.json"), "w") as f:
        json.dump(scenario_files, f, indent=2)
    render_report(
        outcomes,
        os.path.join(out_dir, "selftest-outcome-report.md"),
        synthetic_heading=True,
    )

    print("\nself-test scenario -> verdict check:")
    all_ok = True
    for i, (model, scheme) in enumerate(CONFIGS):
        expected = SELFTEST_SCENARIOS[i % len(SELFTEST_SCENARIOS)][0]
        actual = outcomes[i]["verdict"]
        ok = expected == actual
        all_ok &= ok
        print(
            f"  {model}/{scheme}: expected={expected} actual={actual} {'OK' if ok else 'MISMATCH'}"
        )
    print(
        "ALL SCENARIOS MATCHED"
        if all_ok
        else "SOME SCENARIOS DID NOT MATCH -- bug in verdict engine"
    )


def run_real(target_path, after_dir, out_path):
    targets = load_targets(target_path)
    after = load_after_dir(after_dir)
    outcomes = []
    for key, target in targets.items():
        if key not in after:
            print(f"WARNING: no re-measurement found for {key[0]}/{key[1]}, skipping")
            continue
        outcomes.append(compute_verdict(target, after[key]))
    render_report(outcomes, out_path, synthetic_heading=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generate-target", action="store_true")
    p.add_argument("--baseline-dir")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--target")
    p.add_argument("--after-dir")
    p.add_argument("--out")
    p.add_argument("--out-dir")
    args = p.parse_args()

    if args.generate_target:
        generate_target(args.baseline_dir, args.out)
    elif args.selftest:
        run_selftest(args.target, args.out_dir)
    else:
        run_real(args.target, args.after_dir, args.out)


if __name__ == "__main__":
    main()
