#!/usr/bin/env python3
"""desktop_share.py <out.jsonl> <own top pid> [period_s=2]: until killed, one JSON line per period with the
share of the Arc B580's engine time used by DRM clients that are not descendants of <own top pid>
(xe fdinfo: d(sum of drm-cycles-*) / d(drm-total-cycles-rcs)), the GT clock, throttle status and package
temperature. Reads procfs and sysfs only. Method of the tuning campaign's tools/sampler.py."""
import glob, json, os, sys, time
PDEV = "0000:03:00.0"
P = "/sys/bus/pci/devices/" + PDEV; F = P + "/tile0/gt0/freq0"; H = glob.glob(P + "/hwmon/hwmon*")[0]
top = int(sys.argv[2]); period = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
ENG = ("rcs", "ccs", "bcs", "vcs", "vecs")


def mine(pid):
    while pid > 1:
        if pid in (top, os.getpid()):
            return True
        try:
            pid = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            return False
    return False


def scan():
    """client id -> (cycles, total, pid, mine) over every DRM client of the card, once per client id"""
    c = {}
    for f in glob.glob("/proc/[0-9]*/fdinfo/*"):
        try:
            kv = dict(l.split(":\t", 1) for l in open(f).read().splitlines() if ":\t" in l)
        except OSError:
            continue
        if kv.get("drm-pdev") != PDEV or kv["drm-client-id"] in c:
            continue
        pid = int(f.split("/")[2])
        c[kv["drm-client-id"]] = (sum(int(kv.get(f"drm-cycles-{e}", 0)) for e in ENG), int(kv.get("drm-total-cycles-rcs", 0)), pid, mine(pid))
    return c


def rd(p):
    try:
        return open(p).read().strip()
    except OSError:
        return ""


prev = scan()
with open(sys.argv[1], "a", buffering=1) as out:
    while True:
        time.sleep(period)
        cur = scan()
        dt = max((v[1] for v in cur.values()), default=0) - max((v[1] for v in prev.values()), default=0)
        foreign = own = 0; best = (0, 0)
        for cid, (cyc, _, pid, m) in cur.items():
            d = cyc - prev[cid][0] if cid in prev else 0
            if m:
                own += d
            else:
                foreign += d
                if d > best[0]:
                    best = (d, pid)
        prev = cur
        topname = rd(f"/proc/{best[1]}/comm") if best[1] else ""
        out.write(json.dumps({"utc": time.strftime("%FT%TZ", time.gmtime()), "foreign_pct": round(100 * foreign / dt, 2) if dt > 0 else None,
                              "own_pct": round(100 * own / dt, 2) if dt > 0 else None, "foreign_clients": sum(1 for v in cur.values() if not v[3]),
                              "top_foreign": topname, "act_freq_mhz": rd(F + "/act_freq"), "throttle": rd(F + "/throttle/status"),
                              "throttle_reasons": rd(F + "/throttle/reasons"), "pkg_temp_mc": rd(H + "/temp2_input")}) + "\n")
