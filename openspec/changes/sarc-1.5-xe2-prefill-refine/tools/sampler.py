#!/usr/bin/env python3
"""sampler.py <out file> [period_s=0.005]: sample b70-0 until killed, one line per sample:
epoch_us act_freq_MHz throttle_status card_energy_uJ pkg_temp_mC throttle_reasons
(sysfs reads only: tile0/gt0/freq0/{act_freq,throttle/status,throttle/reasons}, hwmon energy1_input and
temp2_input). A 2048-token prefill of the 1B model lasts about 0.17 s on this card, so the 780M's 0.1 s shell
sampler would see one or two samples; this one sees about 30."""
import glob, sys, time
P = "/sys/bus/pci/devices/0000:01:00.0"; F = P + "/tile0/gt0/freq0"; H = glob.glob(P + "/hwmon/hwmon*")[0]
files = [open(x) for x in (F + "/act_freq", F + "/throttle/status", H + "/energy1_input", H + "/temp2_input", F + "/throttle/reasons")]
period = float(sys.argv[2]) if len(sys.argv) > 2 else 0.005
with open(sys.argv[1], "w", buffering=1) as out:
    while True:
        v = []
        for f in files:
            f.seek(0); v.append(f.read().strip().replace(" ", "+") or "-")
        out.write(f"{time.time_ns() // 1000} {' '.join(v)}\n")
        time.sleep(period)
