#!/usr/bin/env python3
"""sampler.py <out file> [period_s=0.01] [own top pid]: sample card b70-0 (0000:01:00.0) until killed.
From sarc-1.5-b580-fused-port/tools/sampler.py (cea76c634); the card, and a D line also without foreign clients.

One line per sample (every period):
  epoch_us act_freq_MHz throttle_status card_energy_uJ pkg_temp_mC throttle_reasons
and, every fifth sample (50 ms at the default period), one line for the use of the card by other clients:
  D epoch_us total_cycles foreign_cycles n_foreign_clients top_client
foreign_cycles = sum of drm-cycles-{rcs,ccs,bcs,vcs,vecs} over the DRM clients of the card that are not
descendants of <own top pid> (xe fdinfo; counted once per drm-client-id), total_cycles = drm-total-cycles-rcs.
The share of the card other clients used between two D lines is d(foreign_cycles) / d(total_cycles).
foreign_cycles accumulates the increments seen per client from the start of the sampler, so a client that
appears or closes does not make it jump. The client list is rescanned every 5 s, in the same
side thread, so the clock samples keep their period.
Reads sysfs and procfs only; no counter or debug facility of the card is enabled.
A 2048-token prefill of the 1B model lasts about 0.11 s on this card: about 11 clock samples, 2 D lines.
With no foreign client total_cycles stays 0 and foreign_cycles does not move: e2e5.sh reads that as 0 %."""
import glob, os, sys, threading, time
PDEV = "0000:01:00.0"
P = "/sys/bus/pci/devices/" + PDEV; F = P + "/tile0/gt0/freq0"; H = glob.glob(P + "/hwmon/hwmon*")[0]
files = [open(x) for x in (F + "/act_freq", F + "/throttle/status", H + "/energy1_input", H + "/temp2_input", F + "/throttle/reasons")]
period = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
top = int(sys.argv[3]) if len(sys.argv) > 3 else 0
ENG = ("rcs", "ccs", "bcs", "vcs", "vecs")


def mine(pid):
    while pid > 1:
        if pid == top or pid == os.getpid():
            return True
        try:
            pid = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            return False
    return False


def parse(t):
    kv = dict(l.split(":\t", 1) for l in t.splitlines() if ":\t" in l)
    if kv.get("drm-pdev") != PDEV:
        return None
    return kv["drm-client-id"], sum(int(kv.get(f"drm-cycles-{e}", 0)) for e in ENG), int(kv.get("drm-total-cycles-rcs", 0))


def scan():
    """client id -> (fdinfo path, pid) for the foreign clients"""
    c = {}
    for f in glob.glob("/proc/[0-9]*/fdinfo/*"):
        try:
            x = parse(open(f).read())
        except OSError:
            continue
        pid = int(f.split("/")[2])
        if x and x[0] not in c and not mine(pid):
            c[x[0]] = (f, pid)
    return c


def desktop(out):
    clients, last, acc, total, n = {}, {}, 0, 0, 0
    while True:
        if n % 100 == 0:
            clients = scan()
        best = (0, 0)
        for cid, (f, pid) in list(clients.items()):
            try:
                x = parse(open(f).read())
            except OSError:
                x = None
            if not x or x[0] != cid:
                last.pop(cid, None); del clients[cid]; continue
            d = x[1] - last.get(cid, x[1]); last[cid] = x[1]; total = max(total, x[2]); acc += d
            if d > best[0]:
                best = (d, pid)
        out.write(f"D {time.time_ns() // 1000} {total} {acc} {len(clients)} {best[1]}\n")
        n += 1
        time.sleep(5 * period)


with open(sys.argv[1], "w", buffering=1) as out:
    threading.Thread(target=desktop, args=(out,), daemon=True).start()
    while True:
        v = []
        for f in files:
            f.seek(0); v.append(f.read().strip().replace(" ", "+") or "-")
        out.write(f"{time.time_ns() // 1000} {' '.join(v)}\n")
        time.sleep(period)
