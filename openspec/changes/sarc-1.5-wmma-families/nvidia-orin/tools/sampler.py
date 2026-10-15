#!/usr/bin/env python3
"""Read sensors and process ancestry in one process, without sampler children."""

import argparse
import json
import os
from pathlib import Path
import signal
import time


def owned(pid, owner_pid, tag):
    original = Path("/proc") / str(pid)
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == owner_pid:
            return True
        seen.add(pid)
        process = Path("/proc") / str(pid)
        try:
            if ("SARC_CAMPAIGN_TAG=" + tag).encode() in (process / "environ").read_bytes().split(b"\0"):
                return True
        except OSError:
            pass
        try:
            fields = (process / "stat").read_text().rsplit(") ", 1)[1].split()
            pid = int(fields[1])
        except (OSError, ValueError, IndexError):
            return not original.exists()
    return False


def foreign_processes(owner_pid, tag):
    foreign = []
    names = {"llama_main", "test_llama_micr", "logits_dump", "llama-server",
             "ollama", "vllm", "roofline", "roofline_sustai", "inspect"}
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            name = (process / "comm").read_text().strip()
            command = (process / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if (name in names or "Runner.Worker" in command or "comfyui" in command.lower()) and not owned(int(process.name), owner_pid, tag):
            foreign.append({"pid": int(process.name), "comm": name})
    return foreign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths_json", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--owner-pid", required=True, type=int)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    paths = json.loads(args.paths_json.read_text())
    if Path(paths["gone_marker"]).exists():
        raise RuntimeError("GPU_GONE marker exists; recovery requires a person")
    running = True

    def stop(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with args.output_jsonl.open("x", buffering=1) as output:
        while running:
            start = time.monotonic()
            row = {"epoch_us": time.time_ns() // 1000, "sampler_pid": os.getpid()}
            try:
                values = {name: int(Path(paths[name]).read_text())
                          for name in ("clock", "load", "temperature", "voltage", "current")}
                row.update(clock_mhz=values["clock"] / 1e6, busy_percent=values["load"] / 10,
                           temperature_c=values["temperature"] / 1000,
                           power_w=values["voltage"] * values["current"] / 1e6,
                           cooling_states={item["type"] + ":" + item["path"]: int(Path(item["path"]).read_text())
                                           for item in paths["cooling"]},
                           foreign=foreign_processes(args.owner_pid, args.tag))
                if not row["cooling_states"]:
                    raise ValueError("no thermal cooling-state records")
                if values["clock"] <= 0:
                    raise ValueError("GPU clock sensor unavailable")
            except (OSError, ValueError) as error:
                row["sensor_error"] = str(error)
                output.write(json.dumps(row) + "\n")
                Path(paths["gone_marker"]).write_text(json.dumps(row) + "\n")
                raise
            row["sample_latency_ms"] = (time.monotonic() - start) * 1000
            output.write(json.dumps(row) + "\n")
            time.sleep(max(0, 0.1 - (time.monotonic() - start)))


if __name__ == "__main__":
    main()
