#!/usr/bin/env python3
"""Parse a (prefill, decode-window) pair of ETDump captures for one
model/scheme configuration into the raw + aggregated breakdown described in
specs/002-etdump-shader-profiling/contracts/profiling-report-schema.md.

Shape attribution does not use ETRecord (see research.md Decision 2): the
Vulkan delegate embeds each dispatch's operator name, kernel name, and
per-arg tensor sizes directly into the ETDump event's `name` field as JSON,
whenever the event tracer is enabled. This script parses that JSON directly.

Usage:
    python parse_etdump.py \
        --model llama-3.2-1b --scheme 4w \
        --prefill-etdump results/etdumps/llama-3.2-1b_4w_prefill.etdump \
        --prefill-stats-log results/etdumps/llama-3.2-1b_4w_prefill.log \
        --decode-etdump results/etdumps/llama-3.2-1b_4w_decode.etdump \
        --decode-stats-log results/etdumps/llama-3.2-1b_4w_decode.log \
        --decode-window-steps 8 \
        --baseline-json ../001-minipc-baseline-benchmarks/results/raw/llama-3.2-1b_4w.json \
        --out results/raw/llama-3.2-1b_4w.json \
        --raw-out-dir results/raw
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from executorch.devtools import Inspector
from executorch.devtools.inspector import TimeScale

# Kernel-name substrings treated as matrix multiplications for shape
# attribution (FR-003). Everything else gets shape=None ("not applicable"),
# even though many of them also carry tensor sizes in their embedded JSON.
MATMUL_HINTS = ("linear", "gemm", "gemv", "sdpa", "bmm")

# IMPORTANT (discovered empirically, not assumed): for dynamic-shape exports,
# the tensor "sizes" embedded in the event JSON reflect each Vulkan tensor's
# STATIC allocation bound (sized for the largest M this graph was compiled
# for), not the actual active M for that specific dispatch. K (input's last
# dim) and N (output's last dim) are safe to read directly -- those are the
# model's fixed feature/hidden dimensions, never dynamic. M is not safe to
# read this way; it must come from which dispatch path was actually chosen.
# The dispatch code (backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp)
# picks between the tiled/"_gemm_" kernel (real M, e.g. 2048 for our fixed
# prefill) and the "_gemv_"/"_coop_" kernel (always M=1, GEMV) based on
# `is_gemv_case` -- so the kernel name itself is what tells us M, per event.
DECODE_MARKERS = ("gemv", "_coop_")
PREFILL_MARKERS = ("gemm", "_tiled")


def try_parse_event_json(name):
    try:
        obj = json.loads(name)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict) or "kernel_name" not in obj:
        return None
    return obj


def derive_shape(obj, prefill_m):
    op = obj.get("operator")
    if not op:
        return None
    kernel_name = obj.get("kernel_name", "")
    name_lower = kernel_name.lower()
    if not any(h in name_lower for h in MATMUL_HINTS):
        return None
    args = op.get("args", [])
    tensor_args = [
        a for a in args if a.get("type") in ("TENSOR", "TENSORREF") and "sizes" in a
    ]
    if len(tensor_args) < 2:
        return None
    first, last = tensor_args[0], tensor_args[-1]
    if len(first["sizes"]) < 2 or len(last["sizes"]) < 1:
        return None
    k, n = first["sizes"][-1], last["sizes"][-1]

    # sdpa's tiled/coop kernel names follow the same _tiled/_coop convention as
    # linear (confirmed empirically: sdpa_compute_*_tiled_* at prefill,
    # sdpa_compute_*_coop_* at decode), so this same marker check handles it
    # too. Anything else (bmm, or an unrecognized dispatch variant) falls
    # through to "don't guess M" -- FR-003 allows "not applicable" for
    # anything not confidently a clean matmul.
    if any(m in name_lower for m in DECODE_MARKERS):
        m = 1
    elif any(m in name_lower for m in PREFILL_MARKERS):
        m = prefill_m
    else:
        return None
    return {"m": m, "k": k, "n": n}


def classify_block(events):
    """A single llama_main invocation's .etdump contains one EventBlock per
    distinct RunSignature -- and a decode-window run's file ALSO contains the
    one prefill call needed to seed the KV-cache before decoding, in its own
    block. A prefill block is NOT purely tiled/gemm dispatches, though: the
    lm_head/vocab projection is commonly only computed for the last prompt
    position even during prefill, so one GEMV-shaped dispatch can appear
    inside an otherwise-prefill block (confirmed empirically). So classify by
    presence of a tiled/gemm marker first (prefill, even if gemv also
    appears), then by presence of a gemv/coop marker alone (decode). Blocks
    with neither (e.g. Method::init/load, or the single-event summary block)
    classify as None and are excluded from both phases."""
    has_prefill_marker = False
    has_decode_marker = False
    for e in events:
        obj = try_parse_event_json(e.name)
        if obj is None:
            continue
        name_lower = obj.get("kernel_name", "").lower()
        if any(m in name_lower for m in PREFILL_MARKERS):
            has_prefill_marker = True
        elif any(m in name_lower for m in DECODE_MARKERS):
            has_decode_marker = True
    if has_prefill_marker:
        return "prefill"
    if has_decode_marker:
        return "decode"
    return None


def parse_phase_wall_clock_us(stats_log_path, phase):
    """Extract phase wall-clock time (us) from a llama_main run's
    PyTorchObserver stdout log. phase is 'prefill' or 'decode'."""
    text = Path(stats_log_path).read_text()
    m = re.search(r"PyTorchObserver (\{.*\})", text)
    if not m:
        raise ValueError(f"No PyTorchObserver line found in {stats_log_path}")
    stats = json.loads(m.group(1))
    if phase == "prefill":
        us = (stats["prompt_eval_end_ms"] - stats["inference_start_ms"]) * 1000.0
    else:
        us = (stats["inference_end_ms"] - stats["prompt_eval_end_ms"]) * 1000.0
    return us, stats


def parse_phase(etdump_path, phase_wall_clock_us, target_phase, prefill_m):
    insp = Inspector(etdump_path=str(etdump_path), target_time_scale=TimeScale.US)
    raw_invocations = []
    for eb in insp.event_blocks:
        if classify_block(eb.events) != target_phase:
            continue
        for e in eb.events:
            obj = try_parse_event_json(e.name)
            if obj is None:
                continue
            kernel_name = obj["kernel_name"]
            shape = derive_shape(obj, prefill_m)
            operator_name = (obj.get("operator") or {}).get("name")
            times = list(e.perf_data.raw) if e.perf_data else []
            for t in times:
                raw_invocations.append(
                    {
                        "kernel_name": kernel_name,
                        "operator_name": operator_name,
                        "shape": shape,
                        "time_us": t,
                        "event_block": eb.name,
                    }
                )

    groups = defaultdict(list)
    for inv in raw_invocations:
        key = (inv["kernel_name"], json.dumps(inv["shape"], sort_keys=True))
        groups[key].append(inv["time_us"])

    aggregated = []
    total_attributed = 0.0
    for (kernel_name, shape_key), times in groups.items():
        shape = json.loads(shape_key)
        total = sum(times)
        total_attributed += total
        aggregated.append(
            {
                "kernel_name": kernel_name,
                "shape": shape,
                "total_time_us": round(total, 3),
                "invocation_count": len(times),
                "pct_of_phase": (
                    round(total / phase_wall_clock_us, 4)
                    if phase_wall_clock_us
                    else None
                ),
            }
        )
    aggregated.sort(key=lambda a: -a["total_time_us"])
    attributed_pct = (
        round(total_attributed / phase_wall_clock_us, 4)
        if phase_wall_clock_us
        else None
    )
    return raw_invocations, aggregated, attributed_pct


def build_phase_record(
    etdump_path,
    stats_log_path,
    phase,
    decode_window_steps,
    baseline_stats,
    raw_out_path,
    prefill_m,
):
    phase_wall_clock_us, _run_stats = parse_phase_wall_clock_us(stats_log_path, phase)
    raw_invocations, aggregated, attributed_pct = parse_phase(
        etdump_path, phase_wall_clock_us, phase, prefill_m
    )

    baseline_us = None
    if baseline_stats is not None:
        e2e = baseline_stats["e2e"]
        if phase == "prefill":
            tps, toks = e2e.get("prefill_tokens_per_sec"), e2e.get("prefill_tokens")
        else:
            # 001's decode_tokens is the full 1024-step baseline length, but we
            # only profiled a short window (Research Decision 5) -- scale the
            # baseline to the same step count so this is an apples-to-apples
            # "did profiling add overhead" comparison, not prefill-vs-1024-steps.
            tps, toks = e2e.get("decode_tokens_per_sec"), decode_window_steps
        if tps and toks:
            baseline_us = round(toks / tps * 1e6, 3)

    Path(raw_out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(raw_out_path, "w") as f:
        json.dump(raw_invocations, f, indent=2)

    return {
        "status": "ok",
        "failure_reason": None,
        "etdump_path": str(etdump_path),
        "phase_wall_clock_us_profiled": round(phase_wall_clock_us, 3),
        "phase_wall_clock_us_baseline": baseline_us,
        "attributed_pct": attributed_pct,
        "decode_window_steps": decode_window_steps if phase == "decode" else None,
        "aggregated": aggregated,
        "category_rollup": [],
        "raw_invocations_path": str(raw_out_path),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--scheme", required=True)
    p.add_argument("--device", default="rocky-ryzen")
    p.add_argument("--dispatch-path", default="tiled_baseline")
    p.add_argument("--prefill-etdump", required=True)
    p.add_argument("--prefill-stats-log", required=True)
    p.add_argument("--decode-etdump", required=True)
    p.add_argument("--decode-stats-log", required=True)
    p.add_argument("--decode-window-steps", type=int, required=True)
    p.add_argument("--prefill-tokens", type=int, default=2048)
    p.add_argument("--baseline-json", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--raw-out-dir", required=True)
    args = p.parse_args()

    baseline_stats = None
    if args.baseline_json and Path(args.baseline_json).exists():
        baseline_stats = json.load(open(args.baseline_json))

    raw_dir = Path(args.raw_out_dir)
    prefill_record = build_phase_record(
        args.prefill_etdump,
        args.prefill_stats_log,
        "prefill",
        None,
        baseline_stats,
        raw_dir / f"{args.model}_{args.scheme}_prefill_raw.json",
        args.prefill_tokens,
    )
    decode_record = build_phase_record(
        args.decode_etdump,
        args.decode_stats_log,
        "decode",
        args.decode_window_steps,
        baseline_stats,
        raw_dir / f"{args.model}_{args.scheme}_decode_raw.json",
        args.prefill_tokens,
    )

    result = {
        "config": {
            "model": args.model,
            "scheme": args.scheme,
            "device": args.device,
            "dispatch_path": args.dispatch_path,
        },
        "phases": {"prefill": prefill_record, "decode": decode_record},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.out}")
    print(
        f"  prefill: attributed_pct={prefill_record['attributed_pct']}, "
        f"profiled={prefill_record['phase_wall_clock_us_profiled']}us, "
        f"baseline={prefill_record['phase_wall_clock_us_baseline']}us"
    )
    print(
        f"  decode:  attributed_pct={decode_record['attributed_pct']}, "
        f"profiled={decode_record['phase_wall_clock_us_profiled']}us, "
        f"baseline={decode_record['phase_wall_clock_us_baseline']}us"
    )


if __name__ == "__main__":
    main()
