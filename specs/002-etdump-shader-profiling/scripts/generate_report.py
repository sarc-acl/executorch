#!/usr/bin/env python3
"""Generate results/profiling-report.md from the six raw JSON files, per
contracts/profiling-report-schema.md's "Rendered summary" section."""
import argparse
import glob
import json
import os

MODEL_ORDER = ["llama-3.1-8b", "llama-3.2-3b", "llama-3.2-1b"]
SCHEME_ORDER = ["4w", "8da4w"]


def load_all(raw_dir):
    configs = {}
    for path in glob.glob(os.path.join(raw_dir, "*.json")):
        if path.endswith("_prefill_raw.json") or path.endswith("_decode_raw.json"):
            continue  # per-invocation companion files, not the per-config record
        d = json.load(open(path))
        cfg = d["config"]
        configs[(cfg["model"], cfg["scheme"])] = (d, os.path.basename(path))
    return configs


def render_phase(phase_name, phase):
    lines = [f"#### {phase_name.capitalize()}"]
    lines.append("")
    lines.append("| Category | % of phase | Total time (us) |")
    lines.append("|---|---:|---:|")
    for c in phase["category_rollup"]:
        lines.append(
            f"| {c['category']} | {c['pct_of_phase']*100:.1f}% | {c['total_time_us']:.0f} |"
        )
    lines.append("")
    lines.append(
        f"Top kernels by time (of {len(phase['aggregated'])} distinct kernel+shape entries):"
    )
    lines.append("")
    lines.append("| Kernel | Shape (M,K,N) | Count | Total time (us) | % of phase |")
    lines.append("|---|---|---:|---:|---:|")
    for a in phase["aggregated"][:6]:
        shape = (
            f"({a['shape']['m']},{a['shape']['k']},{a['shape']['n']})"
            if a["shape"]
            else "n/a"
        )
        lines.append(
            f"| `{a['kernel_name']}` | {shape} | {a['invocation_count']} | "
            f"{a['total_time_us']:.0f} | {a['pct_of_phase']*100:.1f}% |"
        )
    lines.append("")
    profiled = phase["phase_wall_clock_us_profiled"]
    baseline = phase["phase_wall_clock_us_baseline"]
    lines.append(
        f"Reconciliation: **{phase['attributed_pct']*100:.1f}%** of this phase's "
        f"profiled wall-clock ({profiled:.0f}us) is attributed to named kernels above "
        f"(the rest is framework/dispatch overhead not captured as a distinct event). "
        f"For comparison, the un-profiled `001` baseline measured this phase at "
        f"{baseline:.0f}us{' (' + str(round((profiled-baseline)/baseline*100,1)) + '% vs. profiled)' if baseline else ''}."
    )
    lines.append("")
    lines.append(
        f"Raw per-invocation data: [`raw/{os.path.basename(phase['raw_invocations_path'])}`](raw/{os.path.basename(phase['raw_invocations_path'])})"
    )
    lines.append("")
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    configs = load_all(args.raw_dir)

    lines = []
    lines.append("# ETDump E2E Shader Profiling Report")
    lines.append("")
    lines.append(
        "Companion to [`001-minipc-baseline-benchmarks/results/baseline-report.md`]"
        "(../../001-minipc-baseline-benchmarks/results/baseline-report.md) -- same "
        "device, same six configurations, same `tiled_baseline` dispatch path. This "
        "report breaks down *where* each phase's time goes: per-kernel time/shape/count, "
        "rolled up into categories."
    )
    lines.append("")
    lines.append(
        "**Device**: `rocky-ryzen` -- AMD Radeon 780M (RADV PHOENIX), RDNA3 mobile integrated GPU."
    )
    lines.append(
        "**Dispatch path**: `tiled_baseline` for every row below (`ET_VK_FORCE_TILED_LINEAR=1`)."
    )
    lines.append(
        "**Capture configuration**: prefill at the fixed 2048 tokens (matching `001`); "
        "decode over a short representative window (7-8 steps) rather than the full "
        "1024-step decode `001` used for throughput -- per-step shader/shape "
        "composition doesn't vary with decode position on this architecture, so a "
        "short window is sufficient for attribution (see `research.md` Decision 5)."
    )
    lines.append("")

    for model in MODEL_ORDER:
        model_configs = [(model, s) for s in SCHEME_ORDER if (model, s) in configs]
        if not model_configs:
            continue
        lines.append(f"## {model}")
        lines.append("")
        for key in model_configs:
            d, fname = configs[key]
            scheme = key[1]
            lines.append(f"### {scheme}")
            lines.append("")
            for phase_name in ("prefill", "decode"):
                phase = d["phases"][phase_name]
                if phase["status"] != "ok":
                    lines.append(
                        f"**{phase_name}**: {phase['status']} -- {phase['failure_reason']}"
                    )
                    lines.append("")
                    continue
                lines.extend(render_phase(phase_name, phase))
            lines.append(f"Full data: [`raw/{fname}`](raw/{fname})")
            lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## Cross-model observations")
    lines.append("")
    lines.append(
        "- **Feed-forward dominates every configuration** (~40-54% of prefill, "
        "~33-51% of decode), followed by attention/SDPA compute (~22-34%), then "
        "attention projection (~9-15%). This is consistent across all three model "
        "sizes and both quantization schemes -- the WMMA/coopmat workstream's "
        "highest-leverage target is the feed-forward linears (`w1_gate`/`w3_up`/"
        "`w2_down`), not attention projection or the output head."
    )
    lines.append(
        "- **`lm_head` is a rounding error during prefill (~0.05%) but ~5-11% of "
        "decode** -- consistent with lm_head only being computed for the last "
        "prompt position during prefill but every step during decode."
    )
    lines.append(
        "- **Non-shader overhead is larger in prefill (~10-21%) than decode "
        "(~2-5%)** -- plausibly one-time weight-prepack/cast costs that don't "
        "repeat across decode steps within the same run."
    )
    lines.append(
        "- **Attribution is consistently high** (99.0-99.7% prefill, 88.3-97.1% "
        "decode) across all six configurations -- the parsing approach (Vulkan-"
        "embedded per-dispatch JSON, no ETRecord) captures nearly all phase time."
    )
    lines.append(
        "- **Profiled decode consistently measures ~22-28% *faster* than the "
        "`001` baseline scaled to the same step count** (and profiled prefill "
        '~4-12% faster too) -- the opposite of what "profiling overhead" would '
        "predict. The likely explanation, not measurement error: `001` found a "
        "reproducible warm-up effect where the first few runs after GPU idle "
        "measure faster than the thermally-settled steady-state used for its "
        "reported baseline numbers (see `001`'s `baseline-report.md` "
        "Observations). This feature's short decode window (7-8 steps) doesn't "
        "run long enough to reach that thermally-throttled steady state, so it "
        "isn't directly comparable to `001`'s sustained-throughput numbers -- "
        "treat this report's phase timings as valid for *attribution* (where "
        "does time go, proportionally) rather than as a corrected throughput "
        "measurement."
    )
    lines.append("")

    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
