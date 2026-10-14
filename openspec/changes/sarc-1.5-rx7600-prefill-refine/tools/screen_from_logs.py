#!/usr/bin/env python3
"""screen_from_logs.py <linear-screen dir> <out.csv>: builds the screen CSV (the columns of linear_screen.sh) from the per-job logs when the
JSON has no kernel timing for a variant (the microbench classifies a kernel as coopmat only if its name contains "coopmat"; round-2
variants named before that was known are recorded as 'tiled' with no kernel time). kernel_median_us = the dispatch time printed in the
log for the job's own kernel (the line of a sarc_dev_* shader, not the quantize kernel); for a job that did not run its kernel on a shape
the line shows another kernel and dispatched is 0 when the line's workgroup count does not match the job's tile (only checked when a
tile is given in the file name: t<M>x<N>)."""
import csv, glob, os, re, sys, time
d, out = sys.argv[1], sys.argv[2]
rows = []
for f in sorted(glob.glob(os.path.join(d, "r*-*.log"))):
    m = re.match(r"r(\d+)-(.+)\.log$", os.path.basename(f))
    rnd, cand = m.group(1), m.group(2)
    tm = re.search(r"_t(\d+)x(\d+)k(\d+)", cand)
    txt = open(f).read().split("Executing 1 test cases for LlamaMicrobench")[1:]
    for blk in txt:
        res = re.search(r"^RESULT,linear,([^,]+),8da4w,prefill,[^,]+,(\d+),(\d+),.*?,(\w+),texture3d,(\d+),", blk, re.M)
        ks = [l for l in blk.splitlines() if re.match(r"(sarc_|quantize)", l) and not l.startswith("quantize")]
        if not res or not ks: continue
        model, K, N = res.group(1), int(res.group(2)), int(res.group(3))
        op = re.search(r"prefill_(\w+?)_linear", blk).group(1)
        mm = re.match(r"(\S+)\s+\((\d+),(\d+),(\d+)\)\s+\((\d+),(\d+),(\d+)\)\s+([0-9.]+) μs", ks[-1])
        gx, gy, wg, us = int(mm.group(2)), int(mm.group(3)), int(mm.group(5)), float(mm.group(8))
        disp = 1
        if tm:
            T_M, T_N = int(tm.group(1)), int(tm.group(2)); disp = int(gy == 2048 // T_M and gx == (N // T_N) * wg)
        rows.append([rnd, cand, model, op, 2048, N, K, mm.group(1), disp, us, "", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(os.path.getmtime(f)))])
w = csv.writer(open(out, "w", newline="")); w.writerow("round,cand,model,op,M,N,K,kernel,dispatched,kernel_median_us,kernel_cov,utc".split(","))
w.writerows(rows); print(len(rows), "rows")
