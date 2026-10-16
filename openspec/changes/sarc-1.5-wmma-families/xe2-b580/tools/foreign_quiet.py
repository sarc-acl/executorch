#!/usr/bin/env python3
"""foreign_quiet.py <own top pid> [window_s=3]: exit 0 if the B580's foreign DRM clients used no engine cycles
over the window, else 1 and print the busiest client. Same client accounting as sampler.py; reads procfs only."""
import glob, sys, time
PDEV = "0000:03:00.0"
ENG = ("rcs", "ccs", "bcs", "vcs", "vecs")
top = int(sys.argv[1]); window = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0


def mine(pid):
    while pid > 1:
        if pid == top:
            return True
        try:
            pid = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            return False
    return False


def cycles():
    c = {}
    for f in glob.glob("/proc/[0-9]*/fdinfo/*"):
        try:
            kv = dict(l.split(":\t", 1) for l in open(f).read().splitlines() if ":\t" in l)
        except OSError:
            continue
        if kv.get("drm-pdev") != PDEV or kv["drm-client-id"] in c:
            continue
        pid = int(f.split("/")[2])
        if not mine(pid):
            c[kv["drm-client-id"]] = (pid, sum(int(kv.get(f"drm-cycles-{e}", 0)) for e in ENG))
    return c


a = cycles(); time.sleep(window); b = cycles()
busy = sorted(((v[1] - a[k][1], v[0]) for k, v in b.items() if k in a and v[1] > a[k][1]), reverse=True)
if busy:
    print(f"foreign cycles {sum(d for d, _ in busy)} top pid {busy[0][1]}")
sys.exit(1 if busy else 0)
