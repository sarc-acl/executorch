#!/usr/bin/env python3
"""roof_fast.py <igpu-roofline copy> <results dir>: device side. Runs igpu-roofline's `fast` plan on this host's
GPU from a private copy of the roofline tree that already exists on the device
(~/.cache/igpu-roofline/fleet-quick-20260925: package, runner binaries, SPIR-V; copied by roof.sh, the original is
not touched). It is that tree's campaign.py without the clock pinning: nothing is written to devfreq, EMC,
jetson_clocks or nvpmodel, the clock is whatever the governor gives (recorded by roof.sh). Run it through gl.sh
(gpu-lab lock, foreign-process watch)."""
import json, os, pathlib, sys, time, traceback
ROOT = pathlib.Path(sys.argv[1]).resolve(); out = pathlib.Path(sys.argv[2]).resolve(); out.mkdir(parents=True, exist_ok=True)
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
from igpu_roofline import paths
from igpu_roofline.device import LocalDevice
from igpu_roofline.session import Session
from igpu_roofline.stages import run_plan
paths.use_target("host")
os.environ["IGPU_ROOFLINE_UUID"] = "b49259c9868c5b7cb6f165a2bf4b63be"
os.environ["IGPU_ROOFLINE_STAGE"] = str(ROOT / "stage")
meta = {"plan": "fast", "clock_mode": "automatic, nothing pinned", "started": time.strftime("%FT%TZ", time.gmtime()), "status": "running"}
s = None; t0 = time.monotonic()
try:
    s = Session(LocalDevice("orin-naughty"), out, plan="fast")
    print("START fast", flush=True); run_plan(s, "fast"); meta["status"] = "finished"
except BaseException as e:
    meta["status"] = "failed"; meta["error"] = repr(e); traceback.print_exc()
finally:
    if s: s.close()
    meta["elapsed_s"] = round(time.monotonic() - t0, 1); meta["ended"] = time.strftime("%FT%TZ", time.gmtime())
    (out / "roof-fast-metadata.json").write_text(json.dumps(meta, indent=1)); print("END", meta["status"], flush=True)
sys.exit(0 if meta["status"] == "finished" else 1)
