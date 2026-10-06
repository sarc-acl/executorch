#!/usr/bin/env python3
"""sweep_space.py <artifact dir> <manifest.csv> <out.csv> [--only fam,fam] [--limit N] [--first N] [--correctness]
                  [--linear-args "<microbench args>"] [--tmax 55]
Resumable kernel-level sweep over plan_space.py's manifest, with <artifact dir>/bin/microbench-<batch> (one
binary per batch). Run it detached (setsid nohup); it appends one row per (configuration, shape) to <out.csv> and
skips configurations already there, so it can be killed and restarted at any time.

  linear (4w, 8da4w)  one process per token: --linear --regime=prefill --storage=texture3d (the model path), all
                      twelve real shapes; kernel median per shape. --correctness keeps the microbench's own
                      numeric check on (column "ok"); without it the run is timing only. --linear-args adds
                      microbench options, e.g. the screening mode "--op=wq_wo --runs=1,2" (one shape per model, one
                      warm-up and two timed runs); use one <out.csv> per mode. --first N takes the first N manifest rows.
  SDPA (qk, av)       one qk token and one av token per process pair: first --sdpa-correctness-only
                      --sdpa-tier=extended (8 cases up to S = 2048, both head dims: dispatched kernel names,
                      mismatch counts, pairing), then --sdpa (op mean per model at S = 2048).
A token is selected by exact kernel base name (ET_VK_SARC_780M_*); a shape it does not fit keeps the table
kernel and is recorded with dispatched=0.

One GPU job at a time under the gpu-lab lock. Before each run it waits while <artifact dir>/PAUSE exists, between
06:40 and 07:40 local (the host's daily job), while another GPU process runs, while the coordinator's HOLD exists
(hold.sh; this one is not skipped by --ignore-pause), and until the GPU is at most --tmax C (at most 120 s). Three failed processes in a row stop the sweep (a lost context means a GPU reset)."""
import csv, fcntl, json, os, re, subprocess, sys, tempfile, time

a = sys.argv[1:]; art, manifest, out = a[:3]
opt = lambda k, d: a[a.index(k) + 1] if k in a else d
only = set(opt("--only", "4w,8da4w,qk,av").split(",")); limit = int(opt("--limit", "0")); tmax = int(opt("--tmax", "55"))
batches = set(opt("--batches", "").split(",")) - {""}
nopause = "--ignore-pause" in a   # a short job that itself holds PAUSE against the long sweep
corr = "--correctness" in a; largs = opt("--linear-args", "").split(); first = int(opt("--first", "0"))
LOCK = "00000000-c400-0000-0000-000000000000"
ENV = {"4w": "ET_VK_SARC_780M_Q4", "8da4w": "ET_VK_SARC_780M_DQ", "qk": "ET_VK_SARC_780M_QK", "av": "ET_VK_SARC_780M_AV"}
hw = next(os.path.join("/sys/class/hwmon", h) for h in os.listdir("/sys/class/hwmon")
          if open(f"/sys/class/hwmon/{h}/name").read().strip() == "amdgpu")
rd = lambda p: int(open(os.path.join(hw, p)).read())
fd = os.open(os.path.expanduser(f"~/.cache/gpu-lab/lock-{LOCK}"), os.O_WRONLY | os.O_APPEND)
COLS = ("family,stage,batch,token,model,op,M,N,K,kernel,dispatched,us,cov,ok,detail,rc,temp_pre,temp_post,"
        "clk_post_mhz,wall_s,utc").split(",")
done = set()
if os.path.exists(out): done = {(r["family"], r["token"]) for r in csv.DictReader(open(out))}
else: csv.writer(open(out, "w")).writerow(COLS)
rows = [r for r in csv.DictReader(open(manifest)) if r["family"] in only and (not batches or r["batch"] in batches)]
if first: rows = rows[:first]
OTHERS = "llama-server|ollama|llama_main|test_llama_micr|vllm"

def wait_ok():
    while True:
        lt = time.localtime(); hm = lt.tm_hour * 60 + lt.tm_min
        if (os.path.exists(os.path.join(art, "PAUSE")) and not nopause) or 6 * 60 + 40 <= hm < 7 * 60 + 40: time.sleep(30); continue
        if subprocess.run(["pgrep", "-x", OTHERS], capture_output=True).stdout.strip(): time.sleep(30); continue
        return

def run(bench, args, env):
    t0 = time.time()
    try:
        p = subprocess.run([bench] + args, env=dict(os.environ, ETVK_DEVICE_INDEX="0", **env), capture_output=True,
                           text=True, timeout=600)
        return p.returncode, p.stdout + p.stderr, time.time() - t0
    except subprocess.TimeoutExpired:
        return 124, "", time.time() - t0

# Coordinator hold (hold.sh): one configuration is one unit; HOLD is looked at again once both locks are held.
HOLD_SH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "hold.sh")
HV = dict(l.split("=", 1) for l in subprocess.run([HOLD_SH, "vars"], capture_output=True, text=True).stdout.split())
busy_fd = os.open(HV["BUSY"], os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)

def locked(fn, what):
    while True:
        wait_ok(); subprocess.run([HOLD_SH, "wait", f"sweep_space.py {os.path.basename(out)}: {what}"])
        fcntl.flock(busy_fd, fcntl.LOCK_SH); fcntl.flock(fd, fcntl.LOCK_EX)
        if not os.path.exists(HV["HOLD"]): break
        fcntl.flock(fd, fcntl.LOCK_UN); fcntl.flock(busy_fd, fcntl.LOCK_UN)
    try:
        t0 = time.time()
        while rd("temp1_input") > tmax * 1000 and time.time() - t0 < 120: time.sleep(5)
        tp = rd("temp1_input") // 1000
        res = fn()
        return res, [tp, rd("temp1_input") // 1000, rd("freq1_input") // 10**6]
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN); fcntl.flock(busy_fd, fcntl.LOCK_UN)

def emit(lines):
    with open(out, "a") as f: csv.writer(f).writerows(lines)

fails = n = 0
def account(rc):
    global fails, n
    fails = fails + 1 if rc != 0 else 0; n += 1
    if rc != 0: time.sleep(30)
    if fails >= 3: print("three failed runs in a row, stopping"); sys.exit(77)
    if limit and n >= limit: print("LIMIT_DONE"); sys.exit(0)

def linear(r):
    bench = os.path.join(art, "bin", f"microbench-{r['batch']}"); fam = r["family"]
    def go():
        with tempfile.TemporaryDirectory() as td:
            js = os.path.join(td, "o.json")
            rc, log, wall = run(bench, ["--linear", "--regime=prefill", f"--scheme={fam}", "--storage=texture3d",
                                        f"--json-out={js}"] + largs + ([] if corr else ["--skip-correctness"]), {ENV[fam]: r["kernel_base"]})
            try: cases = [c for c in json.load(open(js))["cases"] if c.get("suite", "linear") == "linear"]
            except Exception: cases = []
            return rc, cases, wall
    (rc, cases, wall), temps = locked(go, f"{fam} {r['token']}")
    head = [fam, r["stage"], r["batch"], r["token"]]; tail = [rc] + temps + [f"{wall:.1f}", time.strftime("%FT%TZ", time.gmtime())]
    emit([head + [c["model"], c["op"], c["M"], c["N"], c["K"], c["kernel"], int(c["kernel"].startswith(r["kernel_base"] + "_")),
                  c.get("kernel_median_us", ""), c.get("kernel_cov", ""), c.get("correctness", ""), c.get("detail", "")[:80]] + tail
          for c in cases] or [head + [""] * 6 + [0, "", "", "", "no output"] + tail])
    print(f"{fam} {r['token']} rc={rc} cases={len(cases)} wall={wall:.0f}s T={temps[0]}", flush=True)
    account(0 if cases else rc or 1)   # rc is 1 whenever a texture3d case runs a coopmat kernel, as on the table kernels

def sdpa(q, v):
    r0 = q or v; bench = os.path.join(art, "bin", f"microbench-{r0['batch']}")
    env = {ENV[x["family"]]: x["kernel_base"] for x in (q, v) if x}
    def go():
        rc1, log, w1 = run(bench, ["--sdpa-correctness-only", "--sdpa-tier=extended"], env)
        with tempfile.TemporaryDirectory() as td:
            js = os.path.join(td, "o.json"); rc2, _, w2 = run(bench, ["--sdpa", f"--json-out={js}"], env)
            try: cases = json.load(open(js))["cases"]
            except Exception: cases = []
        return rc1, rc2, log, cases, w1 + w2
    (rc1, rc2, log, cases, wall), temps = locked(go, f"sdpa qk={q and q['token']} av={v and v['token']}")
    kern = re.findall(r"\[sdpa-kernels\] (\S+) qk=(\S+) softmax=(\S+) av=(\S+) no_mask_fill=\S+ pairing=(\S+)", log)
    res = dict(re.findall(r"\[sdpa-correctness\] (\S+) S=.*?mismatches=(\S+ \S+)", log))
    tail = [f"{rc1}/{rc2}"] + temps + [f"{wall:.1f}", time.strftime("%FT%TZ", time.gmtime())]
    lines = []
    for x, col in ((q, 1), (v, 3)):
        if not x: continue
        hit = [k for k in kern if k[col + 0].startswith(x["kernel_base"] + "_")] if col == 1 else \
              [k for k in kern if k[3].startswith(x["kernel_base"] + "_")]
        bad = [f"{k[0]}:{res.get(k[0], '?')}" for k in hit if "PASSED" not in res.get(k[0], "") or k[4] != "ok"]
        ok = "FAIL" if bad else ("PASS" if hit else "NOT_DISPATCHED")
        detail = f"cases={len(hit)}/{len(kern)} " + " ".join(bad)[:120]
        for c in cases:
            if c["op"] == x["family"] and c["variant"] == "coopmat" and c["M"] == 2048:
                lines.append([x["family"], x["stage"], x["batch"], x["token"], c["model"], c["op"], c["M"], c["N"], c["K"],
                              x["kernel_base"], len(hit), c["op_mean_us"], c["op_stdev_us"] / c["op_mean_us"] if c["op_mean_us"] else "",
                              ok, detail] + tail)
        if not any(l[3] == x["token"] and l[0] == x["family"] for l in lines):
            lines.append([x["family"], x["stage"], x["batch"], x["token"]] + [""] * 6 + [len(hit), "", "", ok, detail] + tail)
    emit(lines)
    print(f"sdpa qk={q and q['token']} av={v and v['token']} rc={rc1}/{rc2} wall={wall:.0f}s T={temps[0]}", flush=True)
    account(rc1 or rc2)

for b in sorted({r["batch"] for r in rows}):
    if not os.path.exists(os.path.join(art, "bin", f"microbench-{b}")): print(f"{b}: no binary, skipped"); continue
    br = [r for r in rows if r["batch"] == b and (r["family"], r["token"]) not in done]
    for r in br:
        if r["family"] in ("4w", "8da4w"): linear(r)
    qs = [r for r in br if r["family"] == "qk"]; vs = [r for r in br if r["family"] == "av"]
    for i in range(max(len(qs), len(vs))):
        sdpa(qs[i] if i < len(qs) else None, vs[i] if i < len(vs) else None)
print("SWEEP_DONE")
