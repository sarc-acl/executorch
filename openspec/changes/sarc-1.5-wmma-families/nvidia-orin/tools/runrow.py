#!/usr/bin/env python3
"""runrow.py <log> <clk samples> <expected prompt tokens> <clkmin MHz> <rc> <foreign GPU processes> <tag>

What one llama_main run measured, from its own files: the PyTorchObserver stats of the log and the clock samples
(epoch_us clock_MHz busy% power_W temp_C throttle) that fall inside the measured prefill window. Prints
  tok_s,prompt_tokens,generated_tokens,prefill_ms,clk_n,clk_med_mhz,clk_min_mhz,busy_med,power_med_w,temp_max,valid,reason
e2e5.sh writes these fields into runs.csv; gate_check.py recomputes them from the same files for every timed run
it counts. A timed run (tag prefill) is valid only with rc 0, a positive finite prefill rate, the expected prompt
tokens, 0 generated tokens, no foreign GPU process, at least 5 clock samples in the window (thresholds.txt), a
median clock of at least clkmin, and no thermal throttle reason: the sixth field of every sample in the window
is the state of the device's thermal cooling devices (common.sh sampler: "0" = none of them throttling, else
"<type>:<state>+..."); a sample without that field is no record, and a timed run without the record is invalid
(reason no_throttle_record), never assumed clean. For the next-token runs clock and throttle state are recorded,
not judged."""
import json, math, statistics as st, sys
FIELDS = ["tok_s", "prompt_tokens", "generated_tokens", "prefill_ms", "clk_n", "clk_med_mhz", "clk_min_mhz", "busy_med", "power_med_w", "temp_max", "valid", "reason"]

def evaluate(log, clk, want, clkmin, rc, oth, tag):
    obs = None
    try:
        with open(log, errors="replace") as f:
            for line in f:
                i = line.find("PyTorchObserver")
                if i >= 0:
                    try: obs = json.loads(line[line.index("{", i):])
                    except ValueError: pass
    except OSError: pass
    tok = pt = gt = ms = ""; rows = []; thr = []
    if obs:
        tok = obs.get("prefill_token_per_sec", ""); pt = obs.get("prompt_tokens", ""); gt = obs.get("generated_tokens", "")
        # prefill window: from inference start to prompt-eval end when available, else the execution window
        a = obs.get("inference_start_ms", obs.get("model_execution_start_ms")); b = obs.get("prompt_eval_end_ms", obs.get("model_execution_end_ms"))
        if a and b:
            ms = b - a
            try:
                with open(clk) as f:
                    for l in f:
                        x = l.split()
                        if len(x) in (5, 6) and a * 1000 <= int(x[0]) <= b * 1000:
                            rows.append([float(v) for v in x[:5]]); thr.append(x[5] if len(x) == 6 else None)
            except (OSError, ValueError): rows = []; thr = []
    n = len(rows)
    med = lambda k: round(st.median(r[k] for r in rows), 1) if rows else ""
    cm = med(1); cmin = round(min(r[1] for r in rows), 1) if rows else ""
    reason = []
    if str(rc) != "0": reason.append("rc")
    try: ok = math.isfinite(float(tok)) and float(tok) > 0
    except (TypeError, ValueError): ok = False
    if not ok: reason.append("no_tok_s")
    if str(pt) != str(want): reason.append("prompt_tokens")
    if tag == "prefill" and str(gt) != "0": reason.append("generated_tokens")
    if oth: reason.append("other_gpu_process")
    if tag == "prefill":
        if n < 5: reason.append("clock_unsampled")
        elif cm < float(clkmin): reason.append("clock_low")
        if n and any(t is None or t == "?" for t in thr): reason.append("no_throttle_record")
        elif any(t != "0" for t in thr): reason.append("thermal_throttle")
    vals = [tok, pt, gt, ms, n, cm, cmin, med(2), med(3), round(max(r[4] for r in rows)) if rows else "", 0 if reason else 1, "+".join(reason)]
    return dict(zip(FIELDS, (str(v) for v in vals)))

if __name__ == "__main__":
    r = evaluate(*sys.argv[1:8]); print(",".join(r[k] for k in FIELDS))
