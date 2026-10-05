#!/usr/bin/env python3
"""sweep_run.py <name> <build tag> <space> <mode> <reps> <checked.csv> [ref=<label>:<token> ...]

Resumable kernel-level measurement of sweep configurations (tools/sweep.py) on b70-0, one process per
(configuration, mode, repeat), each under the gpu-lab lock and the foreign-process guard (tools/gl.sh). Run it
detached. Appends one row per (configuration, shape) to .artifacts/raw/<name>/results.csv and skips every
(configuration, mode, repeat) already there, so a stopped run continues where it ended. Nothing is overwritten:
logs of an attempt that left no row (interrupted) are moved to raw/<name>/superseded/interrupted-<utc>/ first.

  space  4w | 8da4w   ET_VK_SARC_{Q4GSW,DQ8CA}_VARIANT=xs<id>, test_llama_microbench --linear --regime=prefill
                      --storage=texture3d --skip-correctness (the model path)
         qk | av      ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=xe2-xs<id>, test_llama_microbench --sdpa
  mode   cheap        the screening mode, timing only: linear = the 1B shapes (one per shape class: wq/wo,
                      wk/wv, w1/w3, w2); SDPA = the 1B and 3B head configurations (head_dim 64 and 128)
         full         timing of all three models
         corr         correctness only: linear --correctness-only (a separate process from the timing on
                      purpose: on this device the 8da4w correctness gate fails for any tile the rank-3
                      M = 128 cases cannot take, and a failed gate would skip the timing); SDPA
                      --sdpa-correctness-only --sdpa-tier=extended. One row per configuration, shape `corr`
                      (SDPA: `corr64` / `corr128` by head_dim), with correct = passed/total of the cases that
                      dispatched the configuration's own kernel (0/0 = no case exercised it)
  ref=<label>:<token> extra arms measured like a configuration: a variant token (linear) or a profile name
                      (SDPA); an empty token is the device's table choice. `base` (no environment) is always
                      measured first and again after every 50 configurations, as a drift monitor.

Row: id,mode,rep,model,shape,M,N,K,us,cov,dispatched,correct,rc,temp_c,utc. `us` is the kernel median (linear)
or the mean per layer (SDPA); dispatched = 1 if the configuration's own kernel ran that shape (linear; a tile
that does not fit a shape falls back to the table kernel; the SDPA timing output does not name its kernels, so
SDPA dispatch is known from the corr rows only). rc 124 = timeout. A foreign GPU process or a busy lock (gl.sh 76 / 75)
ends the run with SWEEP_ABORTED; three timeouts in a row end it with SWEEP_STOPPED.
"""
import csv, datetime, json, os, pathlib, re, shutil, signal, subprocess, sys

TOOLS = pathlib.Path(__file__).resolve().parent; ART = pathlib.Path(os.environ.get("XE2_ARTIFACTS", TOOLS.parents[4] / ".artifacts"))
name, tag, space, mode, reps, cfgfile = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6]
refs = [a[4:].split(":", 1) for a in sys.argv[7:] if a.startswith("ref=")]
O = ART / "raw" / name; (O / "logs").mkdir(parents=True, exist_ok=True); RES = O / "results.csv"
BIN = ART / "build" / tag / "tests/test_llama_microbench"; LINEAR = space in ("4w", "8da4w")
COLS = ["id", "mode", "rep", "model", "shape", "M", "N", "K", "us", "cov", "dispatched", "correct", "rc", "temp_c", "utc"]
HW = next(pathlib.Path("/sys/bus/pci/devices/0000:01:00.0/hwmon").glob("hwmon*")) / "temp2_input"
utc = lambda: datetime.datetime.now(datetime.timezone.utc).strftime("%FT%TZ")
if not RES.exists(): RES.write_text(",".join(COLS) + "\n")
done = {(r["id"], r["mode"], r["rep"]) for r in csv.DictReader(open(RES))}
with open(O / "env.txt", "a") as f:
    f.write(f"{utc()} start: build {tag} {subprocess.run(['sha256sum', str(BIN)], capture_output=True, text=True).stdout.split()[0]} space {space} mode {mode} reps {reps} cfg {cfgfile} refs {refs} done {len(done)}\n")

child = None
def on_signal(sig, _):                        # a stopped runner takes its GPU job with it
    if child and child.poll() is None: os.killpg(child.pid, signal.SIGTERM)
    with open(O / "env.txt", "a") as f: f.write(f"{utc()} SWEEP_INTERRUPTED signal {sig}\n")
    sys.exit(130)
signal.signal(signal.SIGTERM, on_signal); signal.signal(signal.SIGINT, on_signal)

def run(cmd, env, log, timeout):
    global child
    with open(log, "w") as f:
        child = subprocess.Popen([str(TOOLS / "gl.sh"), "timeout", str(timeout), str(BIN)] + cmd, env=dict(os.environ, **env),
                                 stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        return child.wait()

def load(js): return json.load(open(js))["cases"] if js.exists() and js.stat().st_size else []

def measure(cid, tok, rep):
    """One (configuration, mode, repeat): returns (rows, rc)."""
    stem = O / "logs" / f"{cid}-{mode}-r{rep}"; old = list((O / "logs").glob(f"{cid}-{mode}-r{rep}.*"))
    if old:                                   # an attempt that left no row: keep it, out of the way
        d = O / "superseded" / f"interrupted-{utc()}"; d.mkdir(parents=True, exist_ok=True)
        for p in old: shutil.move(str(p), str(d / p.name))
    js = pathlib.Path(f"{stem}.json"); rows = []; base = {"id": cid, "mode": mode, "rep": rep}
    if LINEAR:
        env = {} if tok is None else {("ET_VK_SARC_Q4GSW_VARIANT" if space == "4w" else "ET_VK_SARC_DQ8CA_VARIANT"): tok}
        cmd = ["--linear", "--regime=prefill", f"--scheme={space}", "--storage=texture3d", f"--json-out={js}"]
        mine = lambda c: not tok or c["kernel"].endswith(f"{tok}_texture3d_texture2d_half")
        if mode == "corr":
            rc = run(cmd + ["--correctness-only"], env, f"{stem}.log", 240)
            cc = [c for c in load(js) if c["suite"] == "correctness" and mine(c)]
            if rc not in (75, 76, 124): rows.append(dict(base, model="-", shape="corr", correct=f'{sum(c["correctness"] == "PASSED" for c in cc)}/{len(cc)}'))
        else:
            rc = run(cmd + ["--skip-correctness"] + (["--model=llama-3.2-1b"] if mode == "cheap" else []), env, f"{stem}.log", 240 if mode == "cheap" else 600)
            for c in load(js):
                if c["suite"] == "linear":
                    rows.append(dict(base, model=c["model"], shape=c["op"], M=c["M"], N=c["N"], K=c["K"], us=c["kernel_median_us"], cov=c["kernel_cov"],
                                     dispatched=int(mine(c)), correct="-"))
    else:
        env = {} if tok is None else {"ET_VK_SARC_UNVERIFIED": "1", "ET_VK_SARC_DEV_PROFILE": tok or "xe2-sdpa0"}
        if mode == "corr":
            rc = run(["--sdpa-correctness-only", "--sdpa-tier=extended"], env, f"{stem}.log", 300)
            txt = open(f"{stem}.log", errors="replace").read(); own = re.sub(r"^xe2-", "", tok or "-")
            ks = re.findall(r"\[sdpa-kernels\] (\S+) qk=(\S+) softmax=\S+ av=(\S+)", txt)
            res = dict(re.findall(r"\[sdpa-correctness\] (\S+) S=\d+ .*? (PASSED|FAILED)", txt))
            dim = dict(re.findall(r"\[sdpa-correctness\] (\S+) S=\d+ input_pos=\d+ D=(\d+)", txt)); corr = {"64": [0, 0], "128": [0, 0]}
            for case, qk, av in ks:
                if re.search(rf"_{own}(nf)?_", qk if space == "qk" else av) and dim.get(case) in corr:
                    corr[dim[case]][0] += res.get(case) == "PASSED"; corr[dim[case]][1] += 1
            if rc not in (75, 76, 124) and ks:
                rows += [dict(base, model="-", shape=f"corr{d}", dispatched=int(t > 0), correct=f"{p}/{t}") for d, (p, t) in corr.items()]
        else:
            rc = run(["--sdpa", f"--json-out={js}"] + (["--model=llama-3.2"] if mode == "cheap" else []), env, f"{stem}.log", 300)
            for c in load(js):
                if c["suite"] == "sdpa" and c["regime"] == "prefill" and c["variant"] == ("tiled" if tok is None else "coopmat") and c["op"] != "total":
                    rows.append(dict(base, model=c["model"], shape=c["op"], M=c["M"], N=c["N"], K=c["K"], us=c["op_mean_us"], cov="", dispatched="", correct="-"))
    if not rows: rows = [dict(base, model="-", shape="-")]
    t = int(HW.read_text()) // 1000; now = utc()
    for r in rows: r.update(rc=rc, temp_c=t, utc=now)
    return rows, rc

def record(rows):
    with open(RES, "a", newline="") as f:
        csv.DictWriter(f, COLS, restval="").writerows(rows)

def stop(word, why):
    with open(O / "env.txt", "a") as f: f.write(f"{utc()} {word} {why}\n")
    print(word, why); sys.exit(76 if word == "SWEEP_ABORTED" else 1)

cfgs = [c for c in csv.DictReader(open(cfgfile)) if c.get("run", "1") == "1"]
arms = [("base", None)] + [(f"ref-{l}", t) for l, t in refs]
for c in cfgs:
    arms.append((c["id"], f'xs{c["id"]}' if LINEAR else f'xe2-xs{c["id"]}'))
nbase = len({r for r in done if r[0] == "base" and r[1] == mode}); timeouts = 0; since = 0
for rep in range(1, reps + 1):
    for cid, tok in arms:
        if cid == "base" and rep > 1: continue
        if mode != "corr" and (since >= 50 or (cid == "base" and nbase == 0)):   # drift monitor
            rows, rc = measure("base", None, f"b{nbase + 1}")
            if rc in (75, 76): stop("SWEEP_ABORTED", f"rc={rc} at base")       # nothing recorded: measured again on resume
            nbase += 1; record(rows); since = 0
        if cid == "base" or (cid, mode, str(rep)) in done: continue
        rows, rc = measure(cid, tok, rep)
        if rc in (75, 76): stop("SWEEP_ABORTED", f"rc={rc} at {cid}")
        record(rows); since += 1
        print(f"{cid} {mode} r{rep} rc={rc} temp={rows[0]['temp_c']} " + " ".join(f"{r['shape']}={r.get('us', '')}" for r in rows[:4]), flush=True)
        timeouts = timeouts + 1 if rc == 124 else 0
        if timeouts >= 3: stop("SWEEP_STOPPED", f"three timeouts in a row, last {cid}")
with open(O / "env.txt", "a") as f: f.write(f"{utc()} SWEEP_DONE\n")
print("SWEEP_DONE")
