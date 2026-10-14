#!/usr/bin/python3
"""sampler.py <out> [period_s=0.01]: one line per sample until SIGTERM, no child processes:
epoch_us sclk_hz busy_pct power_uW temp_max_mC indep_throttle_status(hex) avg_gfxclk_mhz
from the amdgpu hwmon of card1 and gpu_metrics v1.3 (indep_throttle_status at offset 112)."""
import os, signal, struct, sys, time

out, period = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
H, C = "/sys/class/hwmon/hwmon1", "/sys/class/drm/card1/device"
running = True
def stop(*_):
    global running
    running = False
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)

def rd(p):
    with open(p, "rb") as f:
        return f.read()
with open(out, "w", buffering=1) as o:
    while running:
        try:
            t = time.time_ns() // 1000
            sclk = int(rd(f"{H}/freq1_input"))
            busy = int(rd(f"{C}/gpu_busy_percent"))
            pw = int(rd(f"{H}/power1_average"))
            temp = max(int(rd(f"{H}/temp{i}_input")) for i in (1, 2, 3))
            m = rd(f"{C}/gpu_metrics")
            thr = struct.unpack_from("<Q", m, 112)[0] if len(m) >= 120 else -1
            gfx = struct.unpack_from("<H", m, 40)[0] if len(m) >= 42 else -1
            o.write(f"{t} {sclk} {busy} {pw} {temp} {thr:x} {gfx}\n")
        except (OSError, ValueError):
            pass
        time.sleep(period)
