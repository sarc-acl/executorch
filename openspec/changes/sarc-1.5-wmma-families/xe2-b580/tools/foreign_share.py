#!/usr/bin/env python3
"""foreign_share.py <own top pid> [window_s=60]: print the share (percent) of the B580's engine time that DRM
clients outside <own top pid> used over the window. Same accounting as sampler.py / e2e5.sh fbusy_pct:
d(sum of foreign drm-cycles-*) / d(drm-total-cycles-rcs). Reads procfs only."""
import glob, sys, time
PDEV = "0000:03:00.0"
ENG = ("rcs", "ccs", "bcs", "vcs", "vecs")
top = int(sys.argv[1]); window = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0


def mine(pid):
    while pid > 1:
        if pid == top:
            return True
        try:
            pid = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            return False
    return False


def sample():
    clients, total = {}, 0
    for f in glob.glob("/proc/[0-9]*/fdinfo/*"):
        try:
            kv = dict(l.split(":\t", 1) for l in open(f).read().splitlines() if ":\t" in l)
        except OSError:
            continue
        if kv.get("drm-pdev") != PDEV:
            continue
        total = max(total, int(kv.get("drm-total-cycles-rcs", 0)))
        if kv["drm-client-id"] not in clients and not mine(int(f.split("/")[2])):
            clients[kv["drm-client-id"]] = sum(int(kv.get(f"drm-cycles-{e}", 0)) for e in ENG)
    return clients, total


(a, ta) = sample(); time.sleep(window); (b, tb) = sample()
busy = sum(v - a[k] for k, v in b.items() if k in a and v > a[k])
assert tb > ta, "total cycles did not advance"
print(f"{100 * busy / (tb - ta):.3f}")
