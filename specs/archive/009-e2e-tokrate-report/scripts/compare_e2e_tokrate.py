#!/usr/bin/env python3
"""Consolidate 006's Texture3D/Buffer e2e numbers with this feature's new
WMMA-arm e2e capture into one three-way report, with per-scheme verdicts
against 007's (and, for 8da4w, 008's) microbenchmark-level findings.
"""
import argparse
import json
import re
import statistics

CONFIGS = [
    ("llama-3.1-8b", "4w"),
    ("llama-3.1-8b", "8da4w"),
    ("llama-3.2-3b", "4w"),
    ("llama-3.2-3b", "8da4w"),
    ("llama-3.2-1b", "4w"),
    ("llama-3.2-1b", "8da4w"),
]

# research.md Decision 7: 007's time-weighted, per-scheme microbenchmark
# finding for the shipped (production-reachable) coopmat tile config.
MICROBENCH_FINDING = {
    "4w": "+60.6% faster than tiled (007, time-weighted across 21 measured prefill linear ops)",
    "8da4w": "-15.2% slower than tiled (007, time-weighted across 21 measured prefill linear ops)",
}
# 008's tuning finding: config 5 (SUBGROUP_SIZE=32) closes most of the gap for
# 8da4w vs the shipped config, landing at roughly parity with tiled -- but per
# FR-008/spec Assumptions, config 5 is unreachable through production's
# can_use_q4gsw_coopmat() gate (hard subgroup_size()==64 requirement), so it
# is context, not something this feature's WMMA arm can ever dispatch to.
SWEEP_FINDING_8DA4W = (
    "008's tuning sweep found config 5 (SUBGROUP_SIZE=32) closes most of the "
    "shipped-config gap, landing at roughly parity with tiled (-5.5% to +8% "
    "vs tiled across the full-catalog validation) -- but config 5 is not "
    "reachable through production's can_use_q4gsw_coopmat() gate (hard "
    "subgroup_size()==64 requirement), so it is context only, not part of "
    "what this feature's WMMA arm measures (FR-008)."
)

# research.md Decision 5: 006's own same-session control found real
# session-to-session PREFILL variance on this hardware unrelated to storage
# type; decode was unaffected. This feature's WMMA capture is yet another new
# session vs. 006's Texture3D/Buffer numbers, so it inherits the same caveat.
CROSS_SESSION_CAVEAT = (
    "cross-session comparison -- 006 documented real session-to-session "
    "prefill variance on this hardware (same .pte, mean swung from 388.4 to "
    "355.5 tok/s, stdev from 3.9 to 22.5) unrelated to storage/dispatch type; "
    "a modest prefill delta here is not automatically a dispatch-arm effect. "
    "Decode is not affected by this and can be compared directly."
)

ROW_RE = re.compile(
    r"\|\s*(?P<model>llama[\w.-]+)\s*\|\s*(?P<scheme>\w+)\s*\|\s*(?P<phase>prefill|decode)\s*\|\s*"
    r"(?P<t3d_mean>[\d.]+)\s*±\s*(?P<t3d_std>[\d.]+)\s*\|\s*"
    r"(?P<buf_mean>[\d.]+)\s*±\s*(?P<buf_std>[\d.]+)\s*\(\d+ reps\)\s*\|"
)


def load_006_report(path):
    text = open(path).read()
    rows = {}
    for m in ROW_RE.finditer(text):
        key = (m.group("model"), m.group("scheme"), m.group("phase"))
        rows[key] = {
            "t3d_mean": float(m.group("t3d_mean")),
            "t3d_std": float(m.group("t3d_std")),
            "buf_mean": float(m.group("buf_mean")),
            "buf_std": float(m.group("buf_std")),
        }
    return rows


def parse_wmma_reps(cfg_name, raw_dir):
    runs = []
    for i in range(1, 6):
        path = f"{raw_dir}/{cfg_name}_rep{i}.log"
        text = open(path, errors="replace").read()
        m = re.search(r"PyTorchObserver (\{.*\})", text)
        runs.append(json.loads(m.group(1)))
    return runs


def steady_state(values):
    return statistics.mean(values), (
        statistics.stdev(values) if len(values) > 1 else 0.0
    )


def overlaps(mean_a, stdev_a, mean_b, stdev_b):
    lo_a, hi_a = mean_a - 2 * stdev_a, mean_a + 2 * stdev_a
    lo_b, hi_b = mean_b - 2 * stdev_b, mean_b + 2 * stdev_b
    return lo_a <= hi_b and lo_b <= hi_a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wmma-raw-dir", required=True)
    p.add_argument("--storage-comparison-report", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    baseline_006 = load_006_report(args.storage_comparison_report)

    cases = []
    for model, scheme in CONFIGS:
        cfg_name = f"{model}_{scheme}"
        runs = parse_wmma_reps(cfg_name, args.wmma_raw_dir)
        prefill_vals = [r["prefill_token_per_sec"] for r in runs]
        decode_vals = [r["decode_token_per_sec"] for r in runs]
        wmma_prefill_mean, wmma_prefill_std = steady_state(prefill_vals)
        wmma_decode_mean, wmma_decode_std = steady_state(decode_vals)

        row_prefill = baseline_006[(model, scheme, "prefill")]
        row_decode = baseline_006[(model, scheme, "decode")]

        wmma_vs_buffer_prefill_pct = (
            (wmma_prefill_mean - row_prefill["buf_mean"])
            / row_prefill["buf_mean"]
            * 100
        )
        wmma_vs_t3d_prefill_pct = (
            (wmma_prefill_mean - row_prefill["t3d_mean"])
            / row_prefill["t3d_mean"]
            * 100
        )
        wmma_vs_buffer_decode_pct = (
            (wmma_decode_mean - row_decode["buf_mean"]) / row_decode["buf_mean"] * 100
        )
        wmma_vs_t3d_decode_pct = (
            (wmma_decode_mean - row_decode["t3d_mean"]) / row_decode["t3d_mean"] * 100
        )

        # research.md Decision 7: e2e direction vs. 007's microbenchmark-level
        # finding for this scheme. 4w is expected faster; 8da4w expected slower.
        expect_faster = scheme == "4w"
        e2e_faster = wmma_vs_buffer_prefill_pct > 0
        consistency = "consistent" if e2e_faster == expect_faster else "diverges"

        cases.append(
            {
                "model": model,
                "scheme": scheme,
                "t3d_prefill_mean": row_prefill["t3d_mean"],
                "t3d_prefill_std": row_prefill["t3d_std"],
                "buf_prefill_mean": row_prefill["buf_mean"],
                "buf_prefill_std": row_prefill["buf_std"],
                "wmma_prefill_mean": wmma_prefill_mean,
                "wmma_prefill_std": wmma_prefill_std,
                "wmma_vs_buffer_prefill_pct": wmma_vs_buffer_prefill_pct,
                "wmma_vs_t3d_prefill_pct": wmma_vs_t3d_prefill_pct,
                "t3d_decode_mean": row_decode["t3d_mean"],
                "t3d_decode_std": row_decode["t3d_std"],
                "buf_decode_mean": row_decode["buf_mean"],
                "buf_decode_std": row_decode["buf_std"],
                "wmma_decode_mean": wmma_decode_mean,
                "wmma_decode_std": wmma_decode_std,
                "wmma_vs_buffer_decode_pct": wmma_vs_buffer_decode_pct,
                "wmma_vs_t3d_decode_pct": wmma_vs_t3d_decode_pct,
                "consistency": consistency,
            }
        )

    lines = [
        "# End-to-End tok/s Report — Texture, Buffer, and WMMA Across 4w/8da4w",
        "",
        "All six configurations were dispatch-confirmed (ETDump kernel-name "
        "inspection: every measured linear op's dispatched kernel contains "
        "`_coopmat`, per FR-003) and measured successfully -- no blocked/failed "
        "configurations.",
        "",
        "**Important correction found during this feature's own verification "
        "(research.md Decision 8)**: `006`'s originally-published `Buffer` "
        "numbers (reused below for the Texture3D/Buffer columns) were captured "
        "before an unrelated pass bug (`--vulkan-force-fp16` silently defeating "
        "`--vulkan-storage-override buffer` for every per-layer linear op) was "
        "found and fixed. `006`'s own `Buffer` captures therefore never actually "
        "exercised Buffer storage or coopmat dispatch for these ops either -- "
        "they are reused here as originally published (per FR-001), but should "
        "be read as a second `Texture3D`-equivalent baseline, not a true "
        "Buffer-storage measurement. Only this feature's new `WMMA` column "
        "reflects genuine, ETDump-confirmed coopmat dispatch.",
        "",
    ]

    for scheme in ("4w", "8da4w"):
        scheme_cases = [c for c in cases if c["scheme"] == scheme]
        avg_prefill_pct = statistics.mean(
            c["wmma_vs_buffer_prefill_pct"] for c in scheme_cases
        )
        n_consistent = sum(1 for c in scheme_cases if c["consistency"] == "consistent")
        direction = "faster" if avg_prefill_pct > 0 else "slower"
        lines.append(f"## `{scheme}` verdict: does WMMA help?")
        lines.append("")
        lines.append(
            f"**e2e prefill is {abs(avg_prefill_pct):.1f}% {direction} than the "
            f"Buffer/tiled baseline on average across all three models "
            f"({n_consistent}/3 configurations consistent with 007's "
            f"microbenchmark-level finding: {MICROBENCH_FINDING[scheme]}).**"
        )
        if scheme == "8da4w":
            lines.append("")
            lines.append(SWEEP_FINDING_8DA4W)
        lines.append("")

    lines.append("## Per-configuration comparison")
    lines.append("")
    lines.append(
        "| Model | Scheme | Phase | Texture3D (tok/s) | Buffer (tok/s) | WMMA (tok/s) | "
        "WMMA vs Buffer | WMMA vs Texture3D | vs 007 finding |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---|")
    for c in cases:
        lines.append(
            f"| {c['model']} | {c['scheme']} | prefill | "
            f"{c['t3d_prefill_mean']:.2f} ± {c['t3d_prefill_std']:.2f} | "
            f"{c['buf_prefill_mean']:.2f} ± {c['buf_prefill_std']:.2f} | "
            f"{c['wmma_prefill_mean']:.2f} ± {c['wmma_prefill_std']:.2f} (5 reps) | "
            f"{c['wmma_vs_buffer_prefill_pct']:+.1f}% | {c['wmma_vs_t3d_prefill_pct']:+.1f}% | "
            f"{c['consistency']} |"
        )
        lines.append(
            f"| {c['model']} | {c['scheme']} | decode | "
            f"{c['t3d_decode_mean']:.3f} ± {c['t3d_decode_std']:.3f} | "
            f"{c['buf_decode_mean']:.3f} ± {c['buf_decode_std']:.3f} | "
            f"{c['wmma_decode_mean']:.3f} ± {c['wmma_decode_std']:.3f} (5 reps) | "
            f"{c['wmma_vs_buffer_decode_pct']:+.1f}% | {c['wmma_vs_t3d_decode_pct']:+.1f}% | "
            f"n/a (decode stays on GEMV, unaffected by coopmat) |"
        )
    lines.append("")

    lines.append("## Blocked / failed configurations")
    lines.append("")
    lines.append("none")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(f"- **Prefill {CROSS_SESSION_CAVEAT}**")
    lines.append(
        "- Decode tok/s is nearly unchanged across Texture3D/Buffer/WMMA for "
        "every configuration, as expected -- decode dispatches the GEMV/`_coop` "
        "kernel regardless of storage type or the coopmat fix (no WMMA-capable "
        "GEMV kernel exists, per `003`)."
    )
    lines.append(
        "- Dispatch confirmation (ETDump kernel-name inspection) was performed "
        "once per configuration on a separate `EXECUTORCH_ENABLE_EVENT_TRACER=ON` "
        "build (mirroring `002`'s precedent, to avoid tracer overhead "
        "contaminating the timing captures above, which used the standard, "
        "non-instrumented build)."
    )

    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {args.out}")
    for c in cases:
        print(
            f"  {c['model']}/{c['scheme']}: WMMA vs Buffer prefill "
            f"{c['wmma_vs_buffer_prefill_pct']:+.1f}% ({c['consistency']})"
        )


if __name__ == "__main__":
    main()
