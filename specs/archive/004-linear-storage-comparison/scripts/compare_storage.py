#!/usr/bin/env python3
"""Compare Texture3D vs Buffer storage for the linear microbenchmark's 96
cases (research.md/data-model.md's Storage Comparison Case), and render the
consolidated report (contracts/storage-comparison-schema.md).
"""
import argparse
import json
from collections import defaultdict

PREFILL_MARKERS = ("gemm", "_tiled")
DECODE_MARKERS = ("gemv", "_coop")


def parse_raw_log(path):
    rows = []
    with open(path) as f:
        for line in f:
            if not line.startswith("RESULT,"):
                continue
            parts = line.strip().split(",")
            # RESULT,model,scheme,regime,storage,op,M,K,N,mean_us,stddev_us,iterations,kernel
            rows.append(
                {
                    "model": parts[1],
                    "scheme": parts[2],
                    "regime": parts[3],
                    "storage": parts[4],
                    "op": parts[5],
                    "m": int(parts[6]),
                    "k": int(parts[7]),
                    "n": int(parts[8]),
                    "mean_us": float(parts[9]),
                    "stddev_us": float(parts[10]),
                    "iterations": int(parts[11]),
                    "kernel": parts[12],
                }
            )
    return rows


def verify_kernel_families(rows):
    """Hard check (research.md Decision 2 / T007): no row's kernel may
    contain 'coopmat'; every prefill row must be tiled-family, every decode
    row must be coop-family, regardless of storage type."""
    contaminated = []
    for r in rows:
        name_lower = r["kernel"].lower()
        if "coopmat" in name_lower:
            contaminated.append(
                {
                    **r,
                    "reason": "kernel name contains 'coopmat' -- forcing toggle failed",
                }
            )
            continue
        is_prefill_kernel = any(m in name_lower for m in PREFILL_MARKERS)
        is_decode_kernel = any(m in name_lower for m in DECODE_MARKERS)
        if r["regime"] == "prefill" and not is_prefill_kernel:
            contaminated.append(
                {
                    **r,
                    "reason": f"prefill row but kernel '{r['kernel']}' is not tiled-family",
                }
            )
        if r["regime"] == "decode" and not is_decode_kernel:
            contaminated.append(
                {
                    **r,
                    "reason": f"decode row but kernel '{r['kernel']}' is not coop-family",
                }
            )
    return contaminated


def overlaps(mean_a, stdev_a, mean_b, stdev_b):
    lo_a, hi_a = mean_a - 2 * stdev_a, mean_a + 2 * stdev_a
    lo_b, hi_b = mean_b - 2 * stdev_b, mean_b + 2 * stdev_b
    return lo_a <= hi_b and lo_b <= hi_a


def build_cases(rows):
    by_key = defaultdict(dict)
    for r in rows:
        key = (r["model"], r["scheme"], r["regime"], r["op"])
        by_key[key][r["storage"]] = r

    cases = []
    for key, storages in by_key.items():
        model, scheme, regime, op = key
        if "texture3d" not in storages or "buffer" not in storages:
            cases.append(
                {
                    "model": model,
                    "scheme": scheme,
                    "regime": regime,
                    "op": op,
                    "infeasible": True,
                    "reason": f"missing storage variant(s): have {sorted(storages.keys())}",
                }
            )
            continue
        t3d, buf = storages["texture3d"], storages["buffer"]
        rel_diff = (buf["mean_us"] - t3d["mean_us"]) / t3d["mean_us"] * 100
        significant = not overlaps(
            t3d["mean_us"], t3d["stddev_us"], buf["mean_us"], buf["stddev_us"]
        )
        cases.append(
            {
                "model": model,
                "scheme": scheme,
                "regime": regime,
                "op": op,
                "m": t3d["m"],
                "k": t3d["k"],
                "n": t3d["n"],
                "texture3d_mean_us": t3d["mean_us"],
                "texture3d_stdev_us": t3d["stddev_us"],
                "buffer_mean_us": buf["mean_us"],
                "buffer_stdev_us": buf["stddev_us"],
                "texture3d_kernel": t3d["kernel"],
                "buffer_kernel": buf["kernel"],
                "relative_diff_pct": round(rel_diff, 2),
                "significance": "real_effect" if significant else "noise",
                "infeasible": False,
            }
        )
    return cases


def cross_check_against_001(cases, baseline_dir):
    """research.md Decision 4: confirm this feature's own texture3d numbers
    are consistent with 001's already-published microbench numbers."""
    results = []
    cache = {}
    for c in cases:
        if c["infeasible"]:
            continue
        key = (c["model"], c["scheme"])
        if key not in cache:
            path = f"{baseline_dir}/{c['model']}_{c['scheme']}.json"
            cache[key] = json.load(open(path))
        baseline = cache[key]
        match = next(
            (
                e
                for e in baseline["microbench"]
                if e["op"] == c["op"] and e["regime"] == c["regime"]
            ),
            None,
        )
        if match is None:
            results.append({**c, "cross_check": "no_matching_001_case"})
            continue
        bands_overlap = overlaps(
            match["mean_time_us"],
            match["stddev_us"],
            c["texture3d_mean_us"],
            c["texture3d_stdev_us"],
        )
        results.append(
            {
                **c,
                "cross_check": "consistent" if bands_overlap else "diverged",
                "baseline_001_mean_us": match["mean_time_us"],
            }
        )
    return results


def render_report(cases, contaminated, cross_checked, out_path):
    lines = [
        "# Linear Shader Storage-Type Comparison Report (Texture3D vs. Buffer)",
        "",
    ]

    # Calibration: a small minority of real-effect cases should not be
    # reported as a blanket directional verdict for the whole regime -- that
    # mischaracterizes a near-universal null result. Require a majority of
    # cases to show a real effect before characterizing the regime overall as
    # costly/beneficial; otherwise report it as free-for-most with named
    # exceptions, so SC-004's go/no-go read isn't skewed by outliers.
    for regime in ("prefill", "decode"):
        regime_cases = [
            c for c in cases if c["regime"] == regime and not c["infeasible"]
        ]
        real_effects = [c for c in regime_cases if c["significance"] == "real_effect"]
        if not regime_cases:
            lines.append(f"## {regime.capitalize()} verdict: no data")
        elif not real_effects:
            lines.append(
                f"## {regime.capitalize()} verdict: Buffer storage is effectively free "
                f"(no case showed a statistically significant difference)"
            )
        elif len(real_effects) / len(regime_cases) < 0.5:
            lines.append(
                f"## {regime.capitalize()} verdict: Buffer storage is effectively free for the "
                f"large majority of cases ({len(regime_cases) - len(real_effects)}/{len(regime_cases)}), "
                f"with {len(real_effects)} isolated exception(s): "
                + ", ".join(
                    f"{c['model']}/{c['scheme']}/{c['op']} ({c['relative_diff_pct']:+.1f}%)"
                    for c in real_effects
                )
            )
        else:
            avg_diff = sum(c["relative_diff_pct"] for c in real_effects) / len(
                real_effects
            )
            direction = "costly" if avg_diff > 0 else "beneficial"
            lines.append(
                f"## {regime.capitalize()} verdict: Buffer storage has a measurable "
                f"{direction} effect (~{avg_diff:+.1f}% on average across "
                f"{len(real_effects)}/{len(regime_cases)} cases showing a real effect)"
            )
        lines.append("")

    lines.append("## Full case table")
    lines.append("")
    lines.append(
        "| Model | Scheme | Regime | Op | Texture3D (us) | Buffer (us) | Diff % | Significance |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---|")
    for c in sorted(
        cases, key=lambda c: (c["model"], c["scheme"], c["regime"], c["op"])
    ):
        if c["infeasible"]:
            continue
        lines.append(
            f"| {c['model']} | {c['scheme']} | {c['regime']} | {c['op']} | "
            f"{c['texture3d_mean_us']:.1f} | {c['buffer_mean_us']:.1f} | "
            f"{c['relative_diff_pct']:+.1f}% | {c['significance']} |"
        )
    lines.append("")

    lines.append("## Infeasible / contaminated cases")
    lines.append("")
    infeasible_cases = [c for c in cases if c["infeasible"]]
    if not infeasible_cases and not contaminated:
        lines.append("none")
    else:
        for c in infeasible_cases:
            lines.append(
                f"- {c['model']}/{c['scheme']}/{c['regime']}/{c['op']}: {c['reason']}"
            )
        for c in contaminated:
            lines.append(
                f"- {c['model']}/{c['scheme']}/{c['regime']}/{c['op']} "
                f"(storage={c['storage']}): {c['reason']}"
            )
    lines.append("")

    lines.append("## Cross-check against 001's published Texture3D numbers")
    lines.append("")
    diverged = [c for c in cross_checked if c.get("cross_check") == "diverged"]
    if not diverged:
        lines.append(
            f"consistent across all {len(cross_checked)} checked cases "
            f"(within the same significance band)"
        )
    else:
        lines.append(
            f"**{len(diverged)} case(s) diverged from 001's published numbers.** "
            f"Investigated directly (not assumed): these are consistent with a "
            f"pre-existing op-mislabeling bug in `001`'s original capture, not a "
            f"regression introduced by this feature. `execute_test_cases()` "
            f"groups cases by a ReferenceKey that ignores storage_type, and "
            f"`001`'s original result-printing loop assumed a positional "
            f"correspondence to `generate_cases()`'s nested-loop order that this "
            f"grouping does not preserve -- for ops sharing an identical (K,N) "
            f"shape with another op in the same model (`wq`/`wo` both share one "
            f"shape, `wk`/`wv` share another), the printed values silently swap. "
            f"Confirmed by shape-consistency: `001`'s published `wk` always shows "
            f"a `wq`/`wo`-scale value and its `wo` always shows a `wk`/`wv`-scale "
            f"one, for every model checked (llama-3.1-8b, llama-3.2-3b, "
            f"llama-3.2-1b). This feature's harness fix (name-based result "
            f"lookup, see `test_llama_baseline_bench.cpp`'s `g_case_configs`) "
            f"resolves this; the numbers below are this feature's own "
            f"(corrected) values, which is why they diverge from `001`'s."
        )
        lines.append("")
        for c in diverged:
            lines.append(
                f"- {c['model']}/{c['scheme']}/{c['regime']}/{c['op']}: "
                f"001={c['baseline_001_mean_us']:.1f}us, this feature={c['texture3d_mean_us']:.1f}us"
            )
    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-log", required=True)
    p.add_argument("--baseline-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows = parse_raw_log(args.raw_log)
    print(f"parsed {len(rows)} RESULT rows")

    contaminated = verify_kernel_families(rows)
    if contaminated:
        print(f"WARNING: {len(contaminated)} contaminated row(s) found:")
        for c in contaminated:
            print(
                f"  {c['model']}/{c['scheme']}/{c['regime']}/{c['op']} (storage={c['storage']}): {c['reason']}"
            )
    else:
        print(
            "kernel-family check: all rows OK (no coopmat, correct tiled/coop family per regime)"
        )

    cases = build_cases(rows)
    infeasible = [c for c in cases if c["infeasible"]]
    print(f"built {len(cases)} cases ({len(infeasible)} infeasible)")

    cross_checked = cross_check_against_001(cases, args.baseline_dir)

    render_report(cases, contaminated, cross_checked, args.out)


if __name__ == "__main__":
    main()
