"""Portable device fingerprint for sweep provenance.

Replaces specs/028's M5-specific provenance (adb driver md5, /sys/kernel/gpu
clock pins) with a vulkaninfo-derived block that works on any Vulkan stack.
Selection: first non-CPU physical device (llvmpipe/SwiftShader are skipped);
override with --device-index when a box has several real GPUs.

perf_level is best-effort (amdgpu sysfs); "unknown" is fine and never fatal --
it exists so results record whether clocks were forced, not to control them.
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from glob import glob
from pathlib import Path

from tile_constraints import DeviceLimits

_FIELDS = {
    "apiVersion": "api_version",
    "driverVersion": "driver_version",
    "deviceName": "device_name",
    "deviceType": "device_type",
    "driverID": "driver_id",
    "driverInfo": "driver_info",
    "subgroupSize": "subgroup_size_default",
    "minSubgroupSize": "min_subgroup_size",
    "maxSubgroupSize": "max_subgroup_size",
    "maxComputeSharedMemorySize": "max_compute_shared_memory_size",
    "maxComputeWorkGroupInvocations": "max_compute_work_group_invocations",
}
_INT_FIELDS = {
    "subgroup_size_default",
    "min_subgroup_size",
    "max_subgroup_size",
    "max_compute_shared_memory_size",
    "max_compute_work_group_invocations",
}
_LINE_RE = re.compile(r"^\s*(\w+)\s*=\s*(.+?)\s*$")


def _parse_vulkaninfo(text):
    """Sequential state machine: fields stream per device; a new apiVersion
    after a deviceName starts the next device's block."""
    devices, pending = [], {}
    for line in text.splitlines():
        m = _LINE_RE.match(line)
        if not m or m.group(1) not in _FIELDS:
            continue
        key, val = _FIELDS[m.group(1)], m.group(2)
        if key in _INT_FIELDS:
            try:
                val = int(val)
            except ValueError:
                continue
        if key == "api_version" and "device_name" in pending:
            devices.append(pending)
            pending = {}
        pending.setdefault(key, val)
    if pending.get("device_name"):
        devices.append(pending)
    return devices


def _perf_level():
    for p in sorted(
        glob("/sys/class/drm/card*/device/power_dpm_force_performance_level")
    ):
        try:
            return Path(p).read_text().strip()
        except OSError:
            continue
    return "unknown"


def _git_sha():
    try:
        return (
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(Path(__file__).resolve().parent),
                    "rev-parse",
                    "--short",
                    "HEAD",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def fingerprint(device_index=None):
    text = subprocess.run(
        ["vulkaninfo"], capture_output=True, text=True, timeout=120
    ).stdout
    devices = _parse_vulkaninfo(text)
    if not devices:
        raise RuntimeError("vulkaninfo produced no parsable devices")
    real = [
        d
        for d in devices
        if "CPU" not in d.get("device_type", "")
        and "llvmpipe" not in d.get("device_name", "").lower()
    ]
    pool = real or devices
    fp = dict(
        pool[device_index or 0] if device_index is None else devices[device_index]
    )
    fp["os"] = " ".join(
        subprocess.run(["uname", "-srm"], capture_output=True, text=True).stdout.split()
    )
    fp["perf_level"] = _perf_level()
    fp["git_sha"] = _git_sha()
    fp["captured_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return fp


def limits_from_fingerprint(fp, quirks=()):
    lo, hi = fp["min_subgroup_size"], fp["max_subgroup_size"]
    sizes = tuple(s for s in (8, 16, 32, 64, 128) if lo <= s <= hi)
    return DeviceLimits(
        max_shared_mem_bytes=fp["max_compute_shared_memory_size"],
        max_wg_invocations=fp["max_compute_work_group_invocations"],
        subgroup_sizes=sizes,
        quirks=frozenset(quirks),
    )


_STOPWORDS = {"graphics", "inc", "corporation", "technologies", "series"}


def device_slug(fp):
    words = re.sub(r"[^a-z0-9 ]", " ", fp["device_name"].lower()).split()
    words = [w for w in words if w not in _STOPWORDS]
    return "-".join(words)[:40].rstrip("-") or "unknown-device"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="raw vulkaninfo device index; default = first non-CPU device",
    )
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    fp = fingerprint(args.device_index)
    if args.json:
        json.dump(fp, sys.stdout, indent=2)
        print()
    else:
        for k, v in fp.items():
            print(f"{k}: {v}")
        print(f"slug: {device_slug(fp)}")
        print(f"limits: {limits_from_fingerprint(fp)}")
