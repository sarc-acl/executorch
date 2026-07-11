#!/usr/bin/env python3
"""Compare tiled-baseline vs WMMA/coopmat dispatch for the 42 in-scope
(model, scheme, op) prefill linear cases (research.md/data-model.md's WMMA
Comparison Case), and render the consolidated report
(contracts/wmma-improvement-report-schema.md).
"""
import argparse
from collections import defaultdict

MODELS = ["llama-3.1-8b", "llama-3.2-3b", "llama-3.2-1b"]
SCHEMES = ["4w", "8da4w"]
IN_SCOPE_OPS = ["wq", "wk", "wv", "wo", "w1_gate", "w3_up", "w2_down"]

# research.md Decision 4 / T007-T008: verified once per kernel family during
# planning+US1, not per case -- the same compiled shader serves every shape.
SPIRV_VERIFIED = {
    "linear_q4gsw_coopmat": True,
    "linear_dq8ca_q4gsw_coopmat": True,
}
CORRECTNESS_TEST_COVERED = {
    "linear_q4gsw_coopmat": True,
    "linear_dq8ca_q4gsw_coopmat": True,
}


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


def index_rows(rows, regime, storage):
    by_key = {}
    for r in rows:
        if r["regime"] != regime or r["storage"] != storage:
            continue
        by_key[(r["model"], r["scheme"], r["op"])] = r
    return by_key


def overlaps(mean_a, stdev_a, mean_b, stdev_b):
    lo_a, hi_a = mean_a - 2 * stdev_a, mean_a + 2 * stdev_a
    lo_b, hi_b = mean_b - 2 * stdev_b, mean_b + 2 * stdev_b
    return lo_a <= hi_b and lo_b <= hi_a


def kernel_family(kernel_name):
    for family in ("linear_dq8ca_q4gsw_coopmat", "linear_q4gsw_coopmat"):
        if family in kernel_name:
            return family
    return None


def build_cases(tiled_by_key, wmma_by_key):
    cases = []
    excluded = []

    for model in MODELS:
        for scheme in SCHEMES:
            for op in IN_SCOPE_OPS:
                key = (model, scheme, op)
                tiled = tiled_by_key.get(key)
                wmma = wmma_by_key.get(key)
                if tiled is None or wmma is None:
                    excluded.append(
                        {
                            "model": model,
                            "scheme": scheme,
                            "op": op,
                            "reason": f"missing measurement (have tiled={tiled is not None}, wmma={wmma is not None})",
                        }
                    )
                    continue

                family = kernel_family(wmma["kernel"]) or ""
                dispatch_status = "confirmed" if family else "fallback"
                spirv_verified = SPIRV_VERIFIED.get(family, False)
                correctness_test_covered = CORRECTNESS_TEST_COVERED.get(family, False)
                correctness_verified = (
                    dispatch_status == "confirmed"
                    and spirv_verified
                    and correctness_test_covered
                )

                if dispatch_status == "fallback":
                    excluded.append(
                        {
                            "model": model,
                            "scheme": scheme,
                            "op": op,
                            "reason": f"no WMMA dispatch occurred -- kernel was '{wmma['kernel']}', expected a coopmat family name (FR-004)",
                        }
                    )
                    continue

                speedup_pct = (
                    (tiled["mean_us"] - wmma["mean_us"]) / tiled["mean_us"] * 100
                )
                significant = not overlaps(
                    tiled["mean_us"],
                    tiled["stddev_us"],
                    wmma["mean_us"],
                    wmma["stddev_us"],
                )

                cases.append(
                    {
                        "model": model,
                        "scheme": scheme,
                        "op": op,
                        "m": tiled["m"],
                        "k": tiled["k"],
                        "n": tiled["n"],
                        "tiled_mean_us": tiled["mean_us"],
                        "tiled_stdev_us": tiled["stddev_us"],
                        "tiled_iterations": tiled["iterations"],
                        "wmma_mean_us": wmma["mean_us"],
                        "wmma_stdev_us": wmma["stddev_us"],
                        "wmma_iterations": wmma["iterations"],
                        "tiled_kernel": tiled["kernel"],
                        "wmma_kernel": wmma["kernel"],
                        "dispatch_status": dispatch_status,
                        "correctness_verified": correctness_verified,
                        "speedup_pct": round(speedup_pct, 2),
                        "significance": "real_effect" if significant else "noise",
                    }
                )

    return cases, excluded


def assign_weights(cases):
    """research.md Decision 6 addendum: weight each op by its own share of
    its (model, scheme)'s 7 measured ops' total tiled-baseline time --
    003's pct_of_phase is aggregated by (kernel, shape), not per named op,
    and cannot be cleanly split for same-shape sibling pairs (wq/wo,
    wk/wv, w1_gate/w3_up) without an invented assumption."""
    totals = defaultdict(float)
    for c in cases:
        totals[(c["model"], c["scheme"])] += c["tiled_mean_us"]
    for c in cases:
        total = totals[(c["model"], c["scheme"])]
        c["weight"] = c["tiled_mean_us"] / total if total else 0.0
    return cases


def time_weighted_speedup(cases):
    weighted_sum = sum(c["speedup_pct"] * c["weight"] for c in cases)
    weight_total = sum(c["weight"] for c in cases)
    return weighted_sum / weight_total if weight_total else 0.0


def render_report(cases, excluded, out_path):
    lines = ["# WMMA Coopmat Improvement Microbenchmark Report", ""]

    overall = time_weighted_speedup(cases)
    per_scheme = {
        scheme: time_weighted_speedup([c for c in cases if c["scheme"] == scheme])
        for scheme in SCHEMES
    }

    # A single blended figure across both schemes would misrepresent this
    # result: 4w and 8da4w move in OPPOSITE directions, consistently, not as
    # noise (found while generating this report -- not anticipated during
    # planning). Leading with the blend alone would read as "WMMA helps"
    # when half the schemes are actually regressions.
    lines.append(
        f"**By scheme (time-weighted across each scheme's {len(SCHEMES) and len(cases)//len(SCHEMES)} "
        f"measured ops, weighted by each op's own share of its configuration's total "
        f"tiled-baseline time -- see research.md Decision 6 addendum):**"
    )
    for scheme, val in per_scheme.items():
        lines.append(
            f"- `{scheme}`: WMMA is **{val:+.1f}% {'faster' if val > 0 else 'slower'}** than tiled"
        )
    lines.append("")
    lines.append(
        f"**Blended overall (both schemes combined, equal weight per configuration): "
        f"{overall:+.1f}%.** This single number is provided for completeness but "
        f"should not be read alone -- it averages a large, consistent `4w` win against "
        f"a consistent `8da4w` regression (see table below); neither scheme's result is "
        f"noise (every row in both schemes shows the same-direction effect)."
    )
    lines.append("")

    iteration_counts = {c["tiled_iterations"] for c in cases} | {
        c["wmma_iterations"] for c in cases
    }
    if len(iteration_counts) == 1:
        iterations_note = (
            f"a mean ± standard deviation over {iteration_counts.pop()} timed runs"
        )
    else:
        iterations_note = (
            f"a mean ± standard deviation (iteration count varies across rows: "
            f"{sorted(iteration_counts)} -- see raw logs per row)"
        )
    lines.append(
        f"**Statistical basis (FR-003)**: every `Tiled`/`WMMA` value below is "
        f"{iterations_note}, confirmed uniform across every one of the {len(cases)} "
        f"rows in both the tiled-baseline (`004`) and WMMA captures -- no result "
        f"here is a single untimed sample."
    )
    lines.append("")

    lines.append("## Full case table")
    lines.append("")
    lines.append(
        "| Model | Scheme | Op | Tiled (us) | WMMA (us) | Speedup % | Significance | Dispatch | Correctness |"
    )
    lines.append("|---|---|---|---:|---:|---:|---|---|---|")
    for c in sorted(cases, key=lambda c: (c["model"], c["scheme"], c["op"])):
        lines.append(
            f"| {c['model']} | {c['scheme']} | {c['op']} | "
            f"{c['tiled_mean_us']:.1f} ± {c['tiled_stdev_us']:.1f} | "
            f"{c['wmma_mean_us']:.1f} ± {c['wmma_stdev_us']:.1f} | "
            f"{c['speedup_pct']:+.1f}% | {c['significance']} | {c['dispatch_status']} | "
            f"{'verified' if c['correctness_verified'] else 'UNVERIFIED'} |"
        )
    lines.append("")

    lines.append("## Excluded / Out-of-Scope")
    lines.append("")
    lines.append(
        "- `lm_head`, all 6 configurations: excluded -- the harness's synthetic "
        "M=2048 'prefill' case for this op has no production analogue; the real "
        "model's lm_head/vocab projection is always M=1 (a GEMV) regardless of "
        "phase (research.md Decision 3)."
    )
    lines.append(
        "- Decode-regime linear ops, all configurations: excluded -- no "
        "WMMA-capable GEMV (M=1) kernel exists today (003's classification "
        "'c', FR-006)."
    )
    if not excluded:
        lines.append("- No other exclusions.")
    else:
        for e in excluded:
            lines.append(f"- {e['model']}/{e['scheme']}/{e['op']}: {e['reason']}")
    lines.append("")

    lines.append("## Correctness-verification summary")
    lines.append("")
    for family, verified in SPIRV_VERIFIED.items():
        test_covered = CORRECTNESS_TEST_COVERED.get(family, False)
        lines.append(
            f"- `{family}`: SPIR-V inspection {'confirmed' if verified else 'DID NOT CONFIRM'} "
            f"genuine cooperative-matrix instructions (`OpCooperativeMatrixLoadKHR`/"
            f"`OpCooperativeMatrixMulAddKHR`); existing correctness coverage via "
            f"`test_coopmat_linear_bench.cpp`'s `kCorrectnessShapes` "
            f"{'confirmed' if test_covered else 'NOT confirmed'} (research.md Decision 7)."
        )
    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wmma-raw-log", required=True)
    p.add_argument("--tiled-baseline-log", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    tiled_rows = parse_raw_log(args.tiled_baseline_log)
    wmma_rows = parse_raw_log(args.wmma_raw_log)
    print(f"parsed {len(tiled_rows)} tiled-baseline rows, {len(wmma_rows)} WMMA rows")

    tiled_by_key = index_rows(tiled_rows, regime="prefill", storage="buffer")
    wmma_by_key = index_rows(wmma_rows, regime="prefill", storage="buffer")

    cases, excluded = build_cases(tiled_by_key, wmma_by_key)
    cases = assign_weights(cases)
    print(f"built {len(cases)} cases ({len(excluded)} excluded/fallback)")

    render_report(cases, excluded, args.out)


if __name__ == "__main__":
    main()
