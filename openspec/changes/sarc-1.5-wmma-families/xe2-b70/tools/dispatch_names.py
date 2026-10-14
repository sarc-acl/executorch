#!/usr/bin/env python3
"""Extract actual dispatch names from exact-schema flatc ETDump JSON."""
import collections
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
with (root / "dispatch.csv").open("w", newline="") as output:
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["model", "scheme", "arm", "kernel", "operator", "M", "N", "K", "events", "evidence"])
    for model in ("1b", "3b", "8b"):
        for scheme in ("4w", "8da4w"):
            for arm in ("parent", "cand"):
                path = root / f"{model}-{scheme}-{arm}.json"
                data = json.loads(path.read_text())
                dispatches = collections.Counter()
                for run in data["run_data"]:
                    for event in run.get("events", []):
                        profile = event.get("profile_event") or {}
                        name = profile.get("name", "")
                        if not name.startswith("{"):
                            continue
                        item = json.loads(name)
                        kernel = item.get("kernel_name")
                        if not kernel:
                            continue
                        operator = item.get("operator") or {}
                        shapes = [arg["sizes"] for arg in operator.get("args", []) if arg.get("type") == "TENSOR"]
                        mnk = ("", "", "")
                        if len(shapes) >= 2 and any(word in operator.get("name", "") for word in ("linear", "q4gsw", "dq8ca")):
                            inp, out = shapes[0], shapes[-1]
                            mnk = (inp[-2] if len(inp) >= 2 else 1, out[-1], inp[-1])
                        dispatches[(kernel, operator.get("name", ""), *mnk)] += 1
                if not dispatches:
                    raise ValueError(f"No dispatched shaders: {path}")
                for key, count in sorted(dispatches.items(), key=lambda pair: str(pair[0])):
                    writer.writerow([model, scheme, arm, *key, count, path.name])
print("DISPATCH_NAMES_PASS 6 cells x 2 arms")
