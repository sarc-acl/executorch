#!/usr/bin/env python3
"""row.py <kind> <log> <clk> <want tokens> <clkmin> <rc> <others> <tag> <busymax> <start_us> <end_us>

Judge one run of the cross-runtime session and print one CSV fragment:
  tok_s,prompt_tokens,prefill_ms,clk_n,clk_med_mhz,clk_min_mhz,throttled_n,power_avg_w,temp_max,valid,reason,fbusy_pct,samples

kind:
  et  ExecuTorch llama_main: the PyTorchObserver line; window = inference_start_ms .. prompt_eval_end_ms.
  lc  llama.cpp llama-completion on the real prompt: the `prompt eval time` line. Its log prefix
      (min.sec.ms.us since process start) plus <start_us> gives the end of the window; the window is the
      reported prompt-evaluation time before it.
  lb  llama.cpp llama-bench (JSON on stdout, synthetic tokens, repetitions in one process, its own warm-up).
      tok_s = median of samples_ts. There is no per-repetition window, so the clock is taken over the whole
      process and reported but not judged; thermal reasons and foreign engine time are judged over the process.

The clock file is the device sampler's (the tuning campaign's sampler.py): lines
  epoch_us act_freq_MHz throttle_status energy_uJ pkg_temp_mC reasons
  D epoch_us total_cycles foreign_cycles n_clients top_client
Validity rules are the tuning campaign's (e2e5.sh): rc 0, expected prompt tokens, no foreign GPU workload, at
least 5 clock samples in the window, median clock >= clkmin, no thermal throttle reason, foreign engine time
<= busymax. tag `check` (next-token text runs) is judged on rc and token count only.
"""
import json
import os
import re
import statistics as st
import sys

kind, log, clk, want, clkmin, rc, oth, tag, busymax, start_us, end_us = sys.argv[1:12]
start_us, end_us = int(start_us), int(end_us)
txt = open(log, errors="replace").read()
tok = pt = ms = ""
a = b = None  # window, epoch microseconds
samples = ""

if kind == "et":
    obs = None
    for line in txt.splitlines():
        i = line.find("PyTorchObserver")
        if i >= 0:
            try:
                obs = json.loads(line[line.index("{", i):])
            except ValueError:
                pass
    if obs:
        tok = obs.get("prefill_token_per_sec", "")
        pt = obs.get("prompt_tokens", "")
        s = obs.get("inference_start_ms", obs.get("model_execution_start_ms"))
        e = obs.get("prompt_eval_end_ms", obs.get("model_execution_end_ms"))
        if s and e:
            ms = e - s
            a, b = s * 1000, e * 1000
elif kind == "lc":
    # anchored on "prompt eval time": a bare "eval time" also matches the decode line
    m = re.search(r"^(\d+)\.(\d+)\.(\d+)\.(\d+) .*prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens"
                  r".*?([\d.]+)\s*tokens per second", txt, re.M)
    if m:
        ms = float(m.group(5))
        pt = int(m.group(6))
        tok = float(m.group(7))
        rel = ((int(m.group(1)) * 60 + int(m.group(2))) * 1000 + int(m.group(3))) * 1000 + int(m.group(4))
        b = start_us + rel
        a = b - int(ms * 1000)
elif kind == "lb":
    try:
        j = json.loads(txt[txt.index("["):txt.rindex("]") + 1])[0]
        ts = j.get("samples_ts") or []
        if ts:
            tok = round(st.median(ts), 2)
            samples = "/".join(f"{x:.1f}" for x in ts)
            pt = j.get("n_prompt", "")
            ms = round(pt / tok * 1000, 2) if tok else ""
            a, b = start_us, end_us
    except (ValueError, IndexError, KeyError):
        pass

rows, drm, amd_power = [], [], []
fb = ""
if a is not None:
    for line in open(clk, errors="replace"):
        f = line.split()
        if len(f) == 6 and f[0] == "D":
            drm.append([int(x) for x in f[1:4]])
        elif len(f) == 6 and a <= int(f[0]) <= b:
            rows.append([int(x) for x in f[:5]] + [f[5]])
        elif len(f) == 5 and a <= int(f[0]) <= b:
            # AMD sampler: epoch_us sclk_Hz busy_pct power_uW temp_mC -> clock MHz, no status, no energy counter
            rows.append([int(f[0]), int(f[1]) / 1e6, 0, None, int(f[4]), ""])
            amd_power.append(int(f[3]) / 1e6)
    lo = [d for d in drm if d[0] <= a]
    hi = [d for d in drm if d[0] >= b]
    if kind == "lb" and drm:
        lo, hi = lo or drm[:1], hi or drm[-1:]
    if lo and hi and hi[0][1] > lo[-1][1]:
        fb = round((hi[0][2] - lo[-1][2]) / (hi[0][1] - lo[-1][1]) * 100, 2)

n = len(rows)
cm = round(st.median(r[1] for r in rows), 1) if rows else ""
cmin = min(r[1] for r in rows) if rows else ""
if amd_power:
    pw = round(st.median(amd_power), 1)
else:
    pw = round((rows[-1][3] - rows[0][3]) / (rows[-1][0] - rows[0][0]), 1) if n > 1 else ""
thr = sum(1 for r in rows if r[2])
thermal = sum(1 for r in rows if re.search(r"thermal|prochot|ratl", r[5]))
tmax = round(max(r[4] for r in rows) / 1000) if rows else ""

reason = []
if rc != "0":
    reason.append("rc")
if tok == "":
    reason.append("no_tok_s")
if str(pt) != want:
    reason.append("prompt_tokens")
if oth:
    reason.append("other_gpu_process")
if tag == "prefill":
    if n < int(os.environ.get("MIN_CLK_SAMPLES", "5")):
        reason.append("clock_unsampled")
    elif kind != "lb" and cm < float(clkmin):
        reason.append("clock_low")
    elif thermal:
        reason.append("thermal_throttled")
    if busymax and fb == "":
        reason.append("busy_unsampled")
    elif busymax and fb > float(busymax):
        reason.append("foreign_busy")
print(",".join(str(x) for x in [tok, pt, ms, n, cm, cmin, thr, pw, tmax, 0 if reason else 1, "+".join(reason), fb, samples]))
