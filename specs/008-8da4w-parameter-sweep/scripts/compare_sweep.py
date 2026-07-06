#!/usr/bin/env python3
"""Render the sweep report (contracts/sweep-report-schema.md) from the
sweep-phase + full-catalog SWEEP_RESULT raw log, 007's shipped-config
report (Tiled/WMMA columns), for specs/008-8da4w-parameter-sweep.
"""
import argparse
import re
from collections import defaultdict

MODELS = ["llama-3.1-8b", "llama-3.2-3b", "llama-3.2-1b"]
FULL_CATALOG_OPS = ["wq", "wk", "wv", "wo", "w1_gate", "w3_up", "w2_down"]
REPRESENTATIVE_OPS = ["wq", "w1_gate"]

# research.md Decision 4: configs 1, 8, 9, 11 were initially found broken
# by a real A/B-staging thread-count bug (verified via a derived formula
# and config 8's tile-mismatch map), then FIXED (multi-slot-per-thread
# staging loop, mirroring the existing INT4 B-staging pattern) and
# re-tested -- all 11 original candidates now pass correctness. No configs
# excluded on correctness grounds anymore; the winner-selection logic
# below naturally deprioritizes any candidate that's merely slow.
ACTIVE_CANDIDATES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
NEGATIVE_TEST_CONFIG = 12


def parse_sweep_log(path):
    rows = []
    with open(path) as f:
        for line in f:
            if not line.startswith("SWEEP_RESULT,"):
                continue
            parts = line.rstrip("\n").split(",")
            # SWEEP_RESULT,config_id,model,op,m,k,n,outcome,mean_us,stdev_us,iterations,kernel_name,failure_detail
            rows.append(
                {
                    "config_id": int(parts[1]),
                    "model": parts[2],
                    "op": parts[3],
                    "m": parts[4],
                    "k": parts[5],
                    "n": parts[6],
                    "outcome": parts[7],
                    "mean_us": float(parts[8]) if parts[8] else None,
                    "stdev_us": float(parts[9]) if parts[9] else None,
                    "iterations": int(parts[10]) if parts[10] else None,
                    "kernel": parts[11] if len(parts) > 11 else "",
                    "detail": parts[12] if len(parts) > 12 else "",
                }
            )
    return rows


def parse_wmma_report(path):
    """Parse 007's Full case table for the 8da4w scheme's Tiled/WMMA
    (mean, stdev) per (model, op)."""
    out = {}
    row_re = re.compile(
        r"\|\s*([\w.\-]+)\s*\|\s*8da4w\s*\|\s*(\w+)\s*\|\s*"
        r"([\d.]+)\s*±\s*([\d.]+)\s*\|\s*([\d.]+)\s*±\s*([\d.]+)\s*\|"
    )
    with open(path) as f:
        for line in f:
            m = row_re.search(line)
            if not m:
                continue
            model, op, tiled_mean, tiled_std, wmma_mean, wmma_std = m.groups()
            out[(model, op)] = {
                "tiled_mean": float(tiled_mean),
                "tiled_std": float(tiled_std),
                "shipped_mean": float(wmma_mean),
                "shipped_std": float(wmma_std),
            }
    return out


def non_overlapping(mean_a, std_a, mean_b, std_b):
    lo_a, hi_a = mean_a - 2 * std_a, mean_a + 2 * std_a
    lo_b, hi_b = mean_b - 2 * std_b, mean_b + 2 * std_b
    return not (lo_a <= hi_b and lo_b <= hi_a)


def significance(mean_a, std_a, mean_b, std_b):
    return "real_effect" if non_overlapping(mean_a, std_a, mean_b, std_b) else "noise"


def dedupe_latest(rows):
    """Some configs (1, 8, 9, 11) were re-run after the A-staging
    multi-slot fix; the raw log keeps both the stale pre-fix and the
    corrected post-fix rows for history. Keep only the last (most recent)
    row per (config_id, model, op, m) -- the `m` distinguishes a
    correctness-scale row from its performance-scale companion."""
    latest = {}
    order = []
    for r in rows:
        key = (r["config_id"], r["model"], r["op"], r["m"])
        if key not in latest:
            order.append(key)
        latest[key] = r
    return [latest[k] for k in order]


def render_sweep_phase_table(by_config):
    lines = [
        f"## Sweep-Phase Summary ({len(ACTIVE_CANDIDATES)} active candidates x "
        f"{len(MODELS) * len(REPRESENTATIVE_OPS)} shapes = "
        f"{len(ACTIVE_CANDIDATES) * len(MODELS) * len(REPRESENTATIVE_OPS)} rows)\n",
        "| Config | Model | Op | Outcome | Mean (us) | Stdev (us) |",
        "|---|---|---|---|---:|---:|",
    ]
    for cfg in ACTIVE_CANDIDATES:
        for model in MODELS:
            for op in REPRESENTATIVE_OPS:
                match = [
                    r for r in by_config[cfg] if r["model"] == model and r["op"] == op
                ]
                if not match:
                    lines.append(f"| {cfg} | {model} | {op} | **MISSING** | | |")
                    continue
                r = match[0]
                if r["outcome"] == "measured":
                    lines.append(
                        f"| {cfg} | {model} | {op} | measured | {r['mean_us']:.1f} | {r['stdev_us']:.1f} |"
                    )
                else:
                    lines.append(
                        f"| {cfg} | {model} | {op} | {r['outcome']} | -- | -- |"
                    )
    return lines


def render_negative_test(rows):
    lines = ["\n## Negative Test (config 12)\n"]
    neg = [r for r in rows if r["config_id"] == NEGATIVE_TEST_CONFIG]
    if neg:
        r = neg[0]
        status = (
            "PASS (correctly caught)"
            if r["outcome"] == "correctness_failure"
            else "**CRITICAL: did not fail as expected**"
        )
        lines.append(
            f"- config 12 (`WG_TILE_K=64`, mathematically incompatible with `group_size=32`): "
            f"outcome=`{r['outcome']}` -- {status}"
        )
    else:
        lines.append("- **MISSING**: config 12's row not found in the raw log.")
    return lines


def find_winner(by_config, shipped):
    """Best average speedup vs. shipped across representative shapes."""
    winner = None
    best_avg_ratio = None
    for cfg in ACTIVE_CANDIDATES:
        ratios = []
        for model in MODELS:
            for op in REPRESENTATIVE_OPS:
                match = [
                    r
                    for r in by_config[cfg]
                    if r["model"] == model
                    and r["op"] == op
                    and r["outcome"] == "measured"
                ]
                ship = shipped.get((model, op))
                if match and ship:
                    ratios.append(ship["shipped_mean"] / match[0]["mean_us"])
        if ratios:
            avg_ratio = sum(ratios) / len(ratios)
            if best_avg_ratio is None or avg_ratio > best_avg_ratio:
                best_avg_ratio = avg_ratio
                winner = cfg
    return winner, best_avg_ratio


def render_full_catalog_table(rows, winner, shipped):
    full_catalog_rows = [
        r for r in rows if r["config_id"] == winner and r["op"] in FULL_CATALOG_OPS
    ]
    fc_lines = []
    for model in MODELS:
        for op in FULL_CATALOG_OPS:
            match = [
                r for r in full_catalog_rows if r["model"] == model and r["op"] == op
            ]
            ship = shipped.get((model, op))
            if not match or match[0]["outcome"] != "measured" or not ship:
                fc_lines.append(f"| {model} | {op} | MISSING/unmeasured | | | | |")
                continue
            r = match[0]
            mean_us = r["mean_us"]
            stdev_us = r["stdev_us"]
            speedup_shipped = (
                (ship["shipped_mean"] - mean_us) / ship["shipped_mean"] * 100
            )
            speedup_tiled = (ship["tiled_mean"] - mean_us) / ship["tiled_mean"] * 100
            sig_tiled = significance(
                ship["tiled_mean"], ship["tiled_std"], mean_us, stdev_us
            )
            fc_lines.append(
                f"| {model} | {op} | {r['mean_us']:.1f}±{r['stdev_us']:.1f} | "
                f"{speedup_shipped:+.1f}% | {speedup_tiled:+.1f}% | {sig_tiled} |"
            )
    return full_catalog_rows, fc_lines


def compute_beats_tiled(full_catalog_rows, shipped):
    def row_for(m, o):
        matches = [r for r in full_catalog_rows if r["model"] == m and r["op"] == o]
        return matches[0] if matches else None

    beats = False
    for m in MODELS:
        for o in FULL_CATALOG_OPS:
            r = row_for(m, o)
            if not r or r["outcome"] != "measured":
                continue
            sig = significance(
                shipped[(m, o)]["tiled_mean"],
                shipped[(m, o)]["tiled_std"],
                r["mean_us"],
                r["stdev_us"],
            )
            if sig == "real_effect" and shipped[(m, o)]["tiled_mean"] > r["mean_us"]:
                beats = True
    return beats


def render_shader_bug_section(rows):
    lines = [
        "\n## Shader Bug Found and Fixed (configs 1, 8, 9, 11)\n",
        "Configs 1, 8, 9, 11 initially failed correctness -- root-caused (not "
        "guessed) via a derived formula: the A/B-staging LDS thread map assumed "
        "one thread handles one staging slot, requiring "
        "`(WG_TILE_M/4)*(WG_TILE_K/4)` threads for A and "
        "`(WG_TILE_K/4)*(WG_TILE_N/4)` for B, but the workgroup only ever has "
        "`WG_SIZE = 4*SUBGROUP_SIZE` threads for this curated set's "
        "`SG_GRID=2x2`. When the required count exceeds `WG_SIZE`, part of the "
        "LDS staging buffer is silently left unwritten (confirmed via config 8's "
        "per-16x16-tile mismatch map: exactly the rows past the covered range were "
        "wrong). **Fixed** in the test-owned shader "
        "(`dq8ca_q4gsw_coopmat_sweep.glsl`) by generalizing the A-staging path to "
        "the same multi-slot-per-thread loop the B/INT4-weight path already used "
        "(that path was never broken, since it already looped when oversubscribed) "
        "-- all 11 original candidates now pass correctness. Configs 9 and 11 "
        "remain far from competitive on raw speed (10-40x slower than the winner) "
        "due to the serialized per-thread staging overhead once oversubscribed, "
        "but are no longer *wrong* -- reported below for completeness, not "
        "excluded.\n",
    ]
    slow_rows = [
        r for r in rows if r["config_id"] in (9, 11) and r["outcome"] == "measured"
    ]
    for r in slow_rows:
        lines.append(
            f"- config {r['config_id']}, {r['model']}/{r['op']}: "
            f"{r['mean_us']:.1f}±{r['stdev_us']:.1f}us (correct, not competitive)"
        )
    return lines


def render_failure_log(rows):
    lines = ["\n## Failure Log\n"]
    failures = [r for r in rows if r["outcome"] != "measured"]
    if not failures:
        lines.append("(none)\n")
    else:
        for r in failures:
            tag = (
                " [DELIBERATE NEGATIVE TEST]"
                if r["config_id"] == NEGATIVE_TEST_CONFIG
                else ""
            )
            lines.append(
                f"- config {r['config_id']}, {r['model']}/{r['op']} ({r['m']}x{r['k']}x{r['n']}): "
                f"`{r['outcome']}`{tag} -- {r['detail']}"
            )
    return lines


def render_winner_section(winner, best_avg_ratio, beats_tiled, fc_lines):
    lines = [
        "\n## Optimal Configuration Recommendation\n",
        f"**Winner: config {winner}** (avg {(best_avg_ratio - 1) * 100:+.1f}% vs shipped across "
        f"the {len(REPRESENTATIVE_OPS) * len(MODELS)} representative sweep-phase shapes), "
        f"validated against the full {len(FULL_CATALOG_OPS)}-op x {len(MODELS)}-model catalog below.\n",
    ]
    if beats_tiled:
        lines.append(
            "Config beats the tiled baseline with real-effect significance on at least one case.\n"
        )
    else:
        lines.append(
            "**No configuration in this sweep outperforms the tiled baseline with real-effect "
            "significance across the full catalog** (FR-007) -- config's advantage is entirely "
            "relative to the shipped (Xclipse-tuned) WMMA configuration, not the tiled path.\n"
        )
    lines.append(f"\n## Full-Catalog Validation (config {winner}, 21 cases)\n")
    lines.append(
        "| Model | Op | Mean±Stdev (us) | vs Shipped | vs Tiled | Significance (vs tiled) |"
    )
    lines.append("|---|---|---:|---:|---:|---|")
    lines.extend(fc_lines)
    return lines


def render(rows, shipped, out_path):
    rows = dedupe_latest(rows)
    by_config = defaultdict(list)
    for r in rows:
        by_config[r["config_id"]].append(r)

    winner, best_avg_ratio = find_winner(by_config, shipped)
    full_catalog_rows, fc_lines = render_full_catalog_table(rows, winner, shipped)
    beats_tiled = compute_beats_tiled(full_catalog_rows, shipped)

    lines = ["# 8da4w Coopmat Tile/Subgroup Parameter Sweep Report\n"]
    lines += render_sweep_phase_table(by_config)
    lines += render_negative_test(rows)
    lines += render_winner_section(winner, best_avg_ratio, beats_tiled, fc_lines)
    lines += render_failure_log(rows)
    lines += render_shader_bug_section(rows)

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-raw-log", required=True)
    p.add_argument(
        "--shipped-report", required=True, help="007's wmma-improvement-report.md"
    )
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows = parse_sweep_log(args.sweep_raw_log)
    shipped = parse_wmma_report(args.shipped_report)
    render(rows, shipped, args.out)


if __name__ == "__main__":
    main()
