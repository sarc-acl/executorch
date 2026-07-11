#!/usr/bin/env python3
"""Classify every shader in one of 002's per-config profiling JSONs by
WMMA/coopmat candidacy (spec.md's classifications a/b/c/d/uncertain), and
roll the classified shaders across all six configs up into a ranked
candidates report. See research.md for the reasoning behind each rule.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

PREFILL_MARKERS = ("gemm", "_tiled")
DECODE_MARKERS = ("gemv", "_coop")
LINEAR_CATEGORIES = {"attention projection", "feed-forward", "output/vocab projection"}

QUANTIZED_LINEAR_CPP = "backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp"
SDPA_CPP = "backends/vulkan/runtime/graph/ops/impl/SDPA.cpp"

PREFILL_LINEAR_REASONS = [
    f"output tensor is rank-3 ([1,M,K]); can_use_q4gsw_coopmat() rejects "
    f"dim_of(output) > 2 ({QUANTIZED_LINEAR_CPP}:192-194)",
    f"output tensor storage is TEXTURE_3D; can_use_q4gsw_coopmat() requires "
    f"Buffer storage ({QUANTIZED_LINEAR_CPP}:196-197)",
]
DECODE_LINEAR_REASONS = [
    f"no WMMA-capable GEMV (M=1) kernel exists; is_gemv_case routes to the "
    f"tiled/coop kernel choice before can_use_q4gsw_coopmat() is ever "
    f"called, and the existing coopmat shaders are tiled multi-row designs "
    f"not applicable at M=1 ({QUANTIZED_LINEAR_CPP})",
]
SDPA_REASONS = [
    f"no WMMA implementation exists for SDPA; add_sdpa_compute_attn_weights_node/"
    f"add_sdpa_compute_out_node only ever select _tiled or _coop kernel names "
    f"({SDPA_CPP}); the generic add_matmul_coopmat_node/coopmat_mm.glsl path "
    f"exists but is not called anywhere in SDPA.cpp",
]


def coopmat_shader_name(kernel_name):
    if "dq8ca" in kernel_name.lower():
        return "linear_dq8ca_q4gsw_coopmat (linear_dq8ca_qw_coopmat.glsl)"
    return "q4gsw_linear_coopmat (linear_qw_coopmat.glsl)"


def classify_entry(category, kernel_name):
    """Return (classification, blocking_reasons, existing_or_prospective_shader)."""
    name_lower = kernel_name.lower()
    if category == "non-shader overhead":
        return "d", [], None
    if category == "attention (sdpa)":
        return "c", SDPA_REASONS, "none exists (see reason)"
    if category in LINEAR_CATEGORIES:
        is_prefill = any(m in name_lower for m in PREFILL_MARKERS)
        is_decode = any(m in name_lower for m in DECODE_MARKERS)
        shader = coopmat_shader_name(kernel_name)
        if is_prefill and not is_decode:
            return "b", PREFILL_LINEAR_REASONS, shader
        if is_decode:
            # Also covers the lm_head-in-prefill case: lm_head dispatches via
            # the GEMV/coop kernel even inside the "prefill" phase block,
            # since only the last prompt position's logits are needed.
            return "c", DECODE_LINEAR_REASONS, shader
        return (
            "uncertain",
            [f"kernel name '{kernel_name}' matched neither prefill nor decode markers"],
            None,
        )
    return "uncertain", [f"unrecognized category '{category}'"], None


def classify_config(profiling_json_path):
    d = json.load(open(profiling_json_path))
    out = {"config": d["config"], "phases": {}}
    for phase_name, phase in d["phases"].items():
        classifications = []
        if phase.get("status") == "ok":
            for e in phase["aggregated"]:
                cls, reasons, shader = classify_entry(e["category"], e["kernel_name"])
                classifications.append(
                    {
                        "kernel_name": e["kernel_name"],
                        "shape": e["shape"],
                        "category": e["category"],
                        "classification": cls,
                        "blocking_reasons": reasons,
                        "existing_or_prospective_shader": shader,
                        "total_time_us": e["total_time_us"],
                        "pct_of_phase": e["pct_of_phase"],
                    }
                )
        out["phases"][phase_name] = {"classifications": classifications}
    return out


GROUP_DEFS = [
    {
        "key": ("linear", "b"),
        "group_name": "Prefill linear GEMM (attention projection + feed-forward + "
        "output projection) -- blocked by rank-3 output + TEXTURE_3D storage",
        "classification": "b",
    },
    {
        "key": ("linear", "c"),
        "group_name": "Decode linear GEMV (attention projection + feed-forward + "
        "output projection) -- no WMMA-capable GEMV kernel exists",
        "classification": "c",
    },
    {
        "key": ("sdpa", "c"),
        "group_name": "SDPA (prefill + decode) -- no WMMA implementation exists",
        "classification": "c",
    },
]


def group_key_for(category, classification):
    family = "sdpa" if category == "attention (sdpa)" else "linear"
    return (family, classification)


def build_groups(all_configs):
    groups = {
        g["key"]: {
            **g,
            "blocking_reasons": None,
            "existing_or_prospective_shaders": set(),
            "member_rows": [],
            "total_time_us_summed": 0.0,
        }
        for g in GROUP_DEFS
    }
    for cfg in all_configs:
        model, scheme = cfg["config"]["model"], cfg["config"]["scheme"]
        for phase_name, phase in cfg["phases"].items():
            for row in phase["classifications"]:
                if row["classification"] not in ("b", "c"):
                    continue
                key = group_key_for(row["category"], row["classification"])
                if key not in groups:
                    continue
                g = groups[key]
                g["blocking_reasons"] = row["blocking_reasons"]
                # A group spans both 4w and 8da4w configs, each with its own
                # coopmat shader -- collect all of them, don't let the last
                # config processed silently overwrite the others.
                g["existing_or_prospective_shaders"].add(
                    row["existing_or_prospective_shader"]
                )
                g["member_rows"].append(
                    {
                        "model": model,
                        "scheme": scheme,
                        "phase": phase_name,
                        "kernel_name": row["kernel_name"],
                        "shape": row["shape"],
                        "total_time_us": row["total_time_us"],
                        "pct_of_phase": row["pct_of_phase"],
                    }
                )
                g["total_time_us_summed"] += row["total_time_us"]
    return [g for g in groups.values() if g["member_rows"]]


def render_report(groups):
    lines = ["# WMMA-Optimizable Shader Candidates Report", ""]
    lines.append(
        "Built entirely from already-classified data across the six "
        "`001`/`002` baseline configurations -- no new profiling. See "
        "[`research.md`](../research.md) for why each group is classified "
        "the way it is, and each config's own "
        "`results/classifications/<model>_<scheme>.json` for full detail."
    )
    lines.append("")
    lines.append(
        '**No `classification: "a"` (WMMA already in effect) entries exist '
        "in this data** -- every capture was taken under `tiled_baseline`. "
        'Nothing below should be read as "already using WMMA in production."'
    )
    lines.append("")

    for cls, section_title in (
        ("b", "Existing implementation blocked"),
        ("c", "No WMMA implementation exists"),
    ):
        section_groups = sorted(
            [g for g in groups if g["classification"] == cls],
            key=lambda g: -g["total_time_us_summed"],
        )
        lines.append(f"## {section_title}")
        lines.append("")
        if not section_groups:
            lines.append("(none)")
            lines.append("")
            continue
        for g in section_groups:
            lines.append(f"### {g['group_name']}")
            lines.append("")
            shaders = ", ".join(sorted(g["existing_or_prospective_shaders"]))
            lines.append(f"**Existing/prospective shader(s)**: {shaders}")
            lines.append("")
            lines.append("**Blocking reason(s)**:")
            for r in g["blocking_reasons"]:
                lines.append(f"- {r}")
            lines.append("")
            total_ms = g["total_time_us_summed"] / 1000.0
            lines.append(
                f"**Total time across all six configurations**: {total_ms:.2f} ms ({g['total_time_us_summed']:.0f} us)"
            )
            lines.append("")
            lines.append("| Model | Scheme | Phase | Total time (us) | % of phase |")
            lines.append("|---|---|---|---:|---:|")
            per_config = defaultdict(lambda: {"total_time_us": 0.0, "pct_of_phase": []})
            for m in g["member_rows"]:
                k = (m["model"], m["scheme"], m["phase"])
                per_config[k]["total_time_us"] += m["total_time_us"]
                per_config[k]["pct_of_phase"].append(m["pct_of_phase"])
            for (model, scheme, phase), agg in sorted(
                per_config.items(), key=lambda kv: -kv[1]["total_time_us"]
            ):
                pct = sum(p for p in agg["pct_of_phase"] if p is not None)
                lines.append(
                    f"| {model} | {scheme} | {phase} | {agg['total_time_us']:.0f} | {pct*100:.1f}% |"
                )
            lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model")
    p.add_argument("--scheme")
    p.add_argument("--profiling-json")
    p.add_argument("--out")
    p.add_argument("--generate-report", action="store_true")
    p.add_argument("--classifications-dir")
    args = p.parse_args()

    if args.generate_report:
        configs = []
        for path in sorted(glob.glob(os.path.join(args.classifications_dir, "*.json"))):
            configs.append(json.load(open(path)))
        groups = build_groups(configs)
        report = render_report(groups)
        with open(args.out, "w") as f:
            f.write(report)
        print(f"wrote {args.out}")
        return

    result = classify_config(args.profiling_json)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.out}")
    for phase_name, phase in result["phases"].items():
        counts = defaultdict(int)
        for row in phase["classifications"]:
            counts[row["classification"]] += 1
        print(f"  {phase_name}: {dict(counts)}")


if __name__ == "__main__":
    main()
