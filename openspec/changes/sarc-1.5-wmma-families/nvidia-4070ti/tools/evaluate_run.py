"""Recompute the fixed timed-run criteria from runner and NVML raw logs."""

import hashlib
import json
import math
from pathlib import Path
import re
import statistics


def evaluate(log_path, samples_path, returncode, clock_floor=None):
    text = Path(log_path).read_text(errors="replace")
    stats = None
    for line in text.splitlines():
        if "PyTorchObserver" in line:
            stats = json.loads(line[line.index("{"):])
    samples = [json.loads(line) for line in Path(samples_path).read_text().splitlines()] if Path(samples_path).exists() else []
    reasons = []
    if returncode != 0:
        reasons.append("runner_rc")
    if stats is None:
        stats = {}
        reasons.append("missing_stats")
    rate = stats.get("prefill_token_per_sec", 0)
    if not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        reasons.append("invalid_rate")
    if stats.get("prompt_tokens") != 2048:
        reasons.append("prompt_tokens")
    if stats.get("generated_tokens") != 0:
        reasons.append("generated_tokens")
    start, end = stats.get("inference_start_ms", 0), stats.get("prompt_eval_end_ms", 0)
    if end <= start or start <= 0:
        reasons.append("prefill_window")
    window = [s for s in samples if start * 1000 <= s["epoch_us"] <= end * 1000]
    clock = statistics.median(s["clock_mhz"] for s in window) if window else 0
    if len(window) < 5:
        reasons.append("fewer_than_5_clock_samples")
    if clock <= 0:
        reasons.append("clock_nonpositive")
    if clock_floor is not None and clock < clock_floor:
        reasons.append("clock_below_floor")
    thermal = sum(bool(s["event_reasons"] & 0x60) for s in samples)
    foreign = sum(any(c["own"] is not True for c in s["clients"]) for s in samples)
    if thermal:
        reasons.append("thermal_throttle")
    if foreign:
        reasons.append("foreign_gpu_client")
    # The observer always samples once before launch and once after exit.
    if len(samples) < 2:
        reasons.append("missing_boundary_observations")
    generated = "\n".join(line for line in text.splitlines()
                           if "PyTorchObserver" not in line and not re.match(r"^[IWE] |^\[sarc_dev\]", line))
    return {"tok_s": rate, "prompt_tokens": stats.get("prompt_tokens"),
            "generated_tokens": stats.get("generated_tokens"), "prefill_ms": end - start,
            "clock_samples": len(window), "clock_median_mhz": clock,
            "clock_floor_mhz": clock_floor, "sample_count": len(samples),
            "thermal_samples": thermal, "foreign_samples": foreign,
            "valid": int(not reasons), "reason": ";".join(reasons),
            "text_sha256": hashlib.sha256(generated.encode()).hexdigest()}
