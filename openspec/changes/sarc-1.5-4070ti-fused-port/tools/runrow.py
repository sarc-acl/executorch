#!/usr/bin/env python3
"""runrow.py <log> <clk samples> <expected prompt tokens> <clkmin MHz> <rc> <foreign GPU processes> <tag>

What one llama_main run measured, from its own files: the PyTorchObserver stats of the log and the clock samples
(epoch_us clock_MHz busy% power_W temp_C [sw_thermal hw_thermal hw_slowdown reasons_mask]) that fall inside the
measured prefill window. Prints
  tok_s,prompt_tokens,generated_tokens,prefill_ms,clk_n,clk_med_mhz,clk_min_mhz,busy_med,power_med_w,temp_max,valid,reason,
  thr_n,thr_thermal_n,thr_hw_slowdown_n,thr_masks
The last four come from EVERY sample of the run, not only the window: samples that carry the driver's throttle
reasons, those with a thermal reason active (sw_thermal_slowdown or hw_thermal_slowdown), those with hw_slowdown,
and the distinct raw masks seen (joined with ;).
e2e5.sh writes these fields into runs.csv; gate_check.py recomputes them from the same files for every timed run
it counts. A timed run (tag prefill) is valid only with rc 0, a positive finite prefill rate, the expected prompt
tokens, 0 generated tokens, no foreign GPU process, at least 2 clock samples in the window, a median clock of
at least clkmin, throttle reasons sampled (thr_n >= 2, else `throttle_unsampled`) and no thermal reason in any
sample (`thermal_throttle`). A power-cap reason does not invalidate a run (it is in thr_masks). For the next-token runs the clock is recorded, not judged."""
import json, math, statistics as st, sys
FIELDS = ["tok_s", "prompt_tokens", "generated_tokens", "prefill_ms", "clk_n", "clk_med_mhz", "clk_min_mhz", "busy_med", "power_med_w", "temp_max", "valid", "reason",
          "thr_n", "thr_thermal_n", "thr_hw_slowdown_n", "thr_masks"]

def evaluate(log, clk, want, clkmin, rc, oth, tag, require_thr=True):
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
    try:
        with open(clk) as f: thr = [x for x in (l.split() for l in f) if len(x) == 9]
    except OSError: pass
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
                        if len(x) in (5, 9) and a * 1000 <= int(x[0]) <= b * 1000: rows.append([float(v) for v in x[:5]])
            except (OSError, ValueError): rows = []
    n = len(rows)
    med = lambda k: round(st.median(r[k] for r in rows), 1) if rows else ""
    cm = med(1); cmin = round(min(r[1] for r in rows), 1) if rows else ""
    reason = []; tht = sum(1 for x in thr if x[5] != "0" or x[6] != "0"); thh = sum(1 for x in thr if x[7] != "0")
    if str(rc) != "0": reason.append("rc")
    try: ok = math.isfinite(float(tok)) and float(tok) > 0
    except (TypeError, ValueError): ok = False
    if not ok: reason.append("no_tok_s")
    if str(pt) != str(want): reason.append("prompt_tokens")
    if tag == "prefill" and str(gt) != "0": reason.append("generated_tokens")
    if oth: reason.append("other_gpu_process")
    if tag == "prefill":
        if n < 2: reason.append("clock_unsampled")
        elif cm < float(clkmin): reason.append("clock_low")
        if tht: reason.append("thermal_throttle")
        elif len(thr) < 2 and require_thr: reason.append("throttle_unsampled")
    vals = [tok, pt, gt, ms, n, cm, cmin, med(2), med(3), round(max(r[4] for r in rows)) if rows else "", 0 if reason else 1, "+".join(reason),
            len(thr), tht, thh, ";".join(sorted({x[8] for x in thr}))]
    return dict(zip(FIELDS, (str(v) for v in vals)))

if __name__ == "__main__":
    r = evaluate(*sys.argv[1:8]); print(",".join(r[k] for k in FIELDS))
