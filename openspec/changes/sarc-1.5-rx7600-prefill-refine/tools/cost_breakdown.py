#!/usr/bin/env python3
"""cost_breakdown.py <microbench> <kernel base> <out csv>: where the wall time of one 4w sweep configuration goes.
Runs the linear prefill sweep (texture3d, correctness off) in several modes under the gpu-lab lock, stamping every
output line, and writes one row per (mode, case): the wall time of the case (graph build, weight generation and
upload, shader module and pipeline creation, warm-up and timed runs) and the kernel time the microbench reports.
Modes: all 12 shapes with 3+5 runs (the full measurement), with 0+1 runs (setup only), no case at all (process
start, Vulkan device, exit), and candidate screening modes."""
import csv, fcntl, os, subprocess, sys, time
bench, kernel, out = sys.argv[1:4]
MODES = [("none", ["--op=none"]), ("full_12x(3+5)", []), ("setup_12x(0+1)", ["--runs=0,1"]),
         ("w1w3_3x(3+5)", ["--op=w1_w3"]), ("w1w3_3x(1+3)", ["--op=w1_w3", "--runs=1,3"]),
         ("w1w3_3x(1+2)", ["--op=w1_w3", "--runs=1,2"]), ("w1w3_3x(0+1)", ["--op=w1_w3", "--runs=0,1"]),
         ("wqwo_3x(1+2)", ["--op=wq_wo", "--runs=1,2"])]
fd = os.open(os.path.expanduser("~/.cache/gpu-lab/lock-00000000-c400-0000-0000-000000000000"), os.O_WRONLY | os.O_APPEND)
w = csv.writer(open(out, "w")); w.writerow("mode,case,case_wall_s,kernel_us,total_wall_s".split(","))
for name, extra in MODES:
    fcntl.flock(fd, fcntl.LOCK_EX)
    t0 = time.time()
    p = subprocess.Popen([bench, "--linear", "--regime=prefill", "--scheme=4w", "--storage=texture3d", "--skip-correctness"] + extra,
                         env=dict(os.environ, ETVK_DEVICE_INDEX="0", ET_VK_SARC_780M_Q4=kernel), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    last = t0; rows = []
    for line in p.stdout:
        now = time.time()
        if line.startswith("RESULT,linear"):
            f = line.strip().split(","); rows.append((f[-1].split("_linear_")[0], now - last, f[8])); last = now
    p.wait(); total = time.time() - t0
    fcntl.flock(fd, fcntl.LOCK_UN)
    for c, dt, us in rows: w.writerow([name, c, f"{dt:.2f}", us, f"{total:.2f}"])
    w.writerow([name, "(process start and exit)", f"{total - sum(r[1] for r in rows):.2f}", "", f"{total:.2f}"])
    print(name, f"total {total:.1f} s, cases {len(rows)}", flush=True)
